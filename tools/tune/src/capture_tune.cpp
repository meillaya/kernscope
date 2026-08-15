//===- capture_tune.cpp - capture->tune replay-based tuning (todo 53) -------===//
//
// kernscope tools/tune todo 53. See capture_tune.h for the contract.
//
// REUSE MODEL (todo-53 MUST NOT modify tools/capture or libs/*): the pipeline
// is composed from the SAME public building blocks the todo-48 replay harness
// (tools/capture/src/replay.cpp) is built from — the shared plan builder
// BuildPlan (tools/capture/src/replay_model.cpp, compiled here UNCHANGED),
// the libs/gpuruntime runtime/module/kernel/launch APIs, libs/bench
// (reps, L2 flush, median) and libs/db (records keyed arch /
// kernel-name-hash / capture-bundle-hash). The only difference from
// run_replay_bench: the launch config per candidate is the SEARCH
// VARIABLE — the kernel binary, args and buffers are byte-identical to the
// capture (launch-config-only tuning; the db keys prove it).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/capture_tune.h"

// The todo-48 shared replay plan builder (tools/capture, consumed unchanged —
// replay_model.{h,cpp} is compiled into this library by tools/tune's
// CMakeLists so the capture track's plan semantics are reused 1:1).
#include "replay_model.h"

#include "kernscope/arch.h"
#include "kernscope/bench.h"
#include "kernscope/capture.h"
#include "kernscope/gpu.h"
#include "kernscope/version.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kernscope::tune {
namespace {

namespace fs = std::filesystem;

using kernscope::db::json::Value;

std::string IsoTimestampUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return buf;
}

std::string IsoDateUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900,
                tm.tm_mon + 1, tm.tm_mday);
  return buf;
}

bool ReadFileBytes(const std::string &path, std::vector<std::uint8_t> &out,
                   std::string *error) {
  // Only regular files are readable as byte sources (a device like /dev/zero
  // never reaches EOF — refuse it up front instead of hanging).
  std::error_code ec;
  const fs::file_status st = fs::status(path, ec);
  if (ec || !fs::is_regular_file(st)) {
    *error = "not a regular file: " + path;
    return false;
  }
  const std::uintmax_t size = fs::file_size(path, ec);
  if (ec) {
    *error = "cannot stat " + path;
    return false;
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  out.resize(static_cast<std::size_t>(size));
  in.read(reinterpret_cast<char *>(out.data()),
          static_cast<std::streamsize>(out.size()));
  if (!in) {
    *error = "read failed: " + path;
    return false;
  }
  return true;
}

//--- temp kernel-binary file (the module API loads by path) ----------------//

class TempModuleFile {
public:
  TempModuleFile(std::span<const std::uint8_t> bytes, const std::string &suffix,
                 std::string *error) {
    std::error_code ec;
    const fs::path dir = fs::temp_directory_path(ec);
    if (ec) {
      *error = "temp dir unavailable: " + ec.message();
      return;
    }
    for (int attempt = 0; attempt < 8; ++attempt) {
      path_ = dir / ("kernscope-capture-tune-module-" + std::to_string(attempt) +
                     suffix);
      std::ofstream out(path_, std::ios::binary | std::ios::trunc);
      if (!out) {
        continue;
      }
      out.write(reinterpret_cast<const char *>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
      out.close();
      if (out) {
        return;
      }
    }
    path_.clear();
    *error = "cannot write temp kernel binary";
  }
  ~TempModuleFile() {
    if (!path_.empty()) {
      std::error_code ec;
      fs::remove(path_, ec);
    }
  }
  const fs::path &path() const { return path_; }
  bool ok() const { return !path_.empty(); }

private:
  fs::path path_;
};

//--- launch-config pre-screen (todo-53; documented in capture_tune.h) -------//
// Rejects a candidate BEFORE any replay with a reason string. Limits are the
// documented HIP launch ceilings (per-block 1024 threads, grid x 2^31-1,
// y/z 65535) + the arch's smem-per-block budget via libs/arch.
bool PreScreenConfig(std::uint32_t grid[3], const std::uint32_t block[3],
                     std::uint32_t shared, const ArchInfo &arch,
                     std::string *reason) {
  const std::uint64_t threads =
      static_cast<std::uint64_t>(block[0]) * block[1] * block[2];
  if (threads > kMaxBlockThreads) {
    *reason = "block " + std::to_string(threads) + " exceeds the launch limit " +
              std::to_string(kMaxBlockThreads) +
              " threads per block (HIP maxThreadsPerBlock ceiling)";
    return false;
  }
  if (block[0] == 0 || block[1] == 0 || block[2] == 0) {
    *reason = "block dimension is zero";
    return false;
  }
  if (grid[0] > kMaxGridDimX || grid[1] > kMaxGridDimYZ ||
      grid[2] > kMaxGridDimYZ || grid[0] == 0 || grid[1] == 0 || grid[2] == 0) {
    *reason = "grid (" + std::to_string(grid[0]) + "," +
              std::to_string(grid[1]) + "," + std::to_string(grid[2]) +
              ") exceeds the launch limits (x <= 2^31-1, y/z <= 65535)";
    return false;
  }
  if (arch.smem_per_block_max_bytes > 0 &&
      shared > static_cast<std::uint32_t>(arch.smem_per_block_max_bytes)) {
    *reason = "shared memory " + std::to_string(shared) + " bytes exceeds the " +
              std::string(arch.name) + " per-block budget " +
              std::to_string(arch.smem_per_block_max_bytes) + " bytes";
    return false;
  }
  return true;
}

//--- bundle identity hash (db key part) -------------------------------------//
// v0 text bundles: sha256 of the bundle FILE bytes. v1 *.kcap/ directories:
// sha256 of the manifest.json envelope (the canonical bundle identity).
std::string BundleIdentityHash(const std::string &bundle_path,
                               std::string *error) {
  std::error_code ec;
  if (fs::is_directory(bundle_path, ec)) {
    const std::string manifest = bundle_path + "/manifest.json";
    std::vector<std::uint8_t> bytes;
    if (!ReadFileBytes(manifest, bytes, error)) {
      return "";
    }
    return sha256_hex(bytes);
  }
  std::vector<std::uint8_t> bytes;
  if (!ReadFileBytes(bundle_path, bytes, error)) {
    return "";
  }
  return sha256_hex(bytes);
}

std::string KernelNameHash(const std::string &kernel_name) {
  return sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(kernel_name.data()),
      kernel_name.size()));
}

//--- JSON helpers ------------------------------------------------------------//

Value ConfigToJson(const CaptureTuneCandidate &c) {
  return Value(std::vector<Value::Member>{
      {"grid", Value(std::vector<Value>{Value(static_cast<double>(c.grid[0])),
                                        Value(static_cast<double>(c.grid[1])),
                                        Value(static_cast<double>(c.grid[2]))})},
      {"block", Value(std::vector<Value>{Value(static_cast<double>(c.block[0])),
                                         Value(static_cast<double>(c.block[1])),
                                         Value(static_cast<double>(c.block[2]))})},
      {"shared_mem_bytes", Value(static_cast<double>(c.shared_mem_bytes))},
      {"is_baseline", Value(c.is_baseline)},
  });
}

Value TimingsToJson(double median, double p20, double p80) {
  return Value(std::vector<Value::Member>{
      {"median", Value(median)}, {"p20", Value(p20)}, {"p80", Value(p80)}});
}

//--- candidate enumeration ---------------------------------------------------//
// Candidate #1 is ALWAYS the captured baseline config (explored first — the
// acceptance assertion; guarantees best.speedup >= 1.0). Then the block
// grid: candidate_blocks (default the arch-valid subset of {128, 256, 512,
// 1024} UNION the captured block) at the CAPTURED grid — the grid of a
// captured kernel is workload-semantic (one block per row for layernorm),
// only the within-block partition is searchable without touching the binary.
// Shared-mem carveout: only varied when the captured launch itself used
// dynamic shared memory (captured shared_mem_bytes > 0); candidates then
// carry shared in {captured, 16 KiB, 32 KiB, 48 KiB} within the arch budget.
// --inject-config appends one extra candidate (the acceptance's broken
// config: block exceeds the arch max -> pre-screen rejected, never replayed).

struct CandidateSeed {
  std::uint32_t grid[3];
  std::uint32_t block[3];
  std::uint32_t shared;
  bool is_baseline;
};

std::vector<CandidateSeed> BuildCandidates(const LaunchRecord &lr,
                                           const ArchInfo &arch,
                                           const std::vector<int> &blocks_opt,
                                           const std::string &inject_config,
                                           std::string *error) {
  std::vector<CandidateSeed> seeds;
  std::uint32_t grid[3] = {lr.grid_x, lr.grid_y, lr.grid_z};
  std::uint32_t base_block[3] = {lr.block_x, lr.block_y, lr.block_z};

  seeds.push_back(CandidateSeed{{lr.grid_x, lr.grid_y, lr.grid_z},
                                {lr.block_x, lr.block_y, lr.block_z},
                                lr.shared_mem_bytes, /*is_baseline=*/true});

  std::vector<int> blocks = blocks_opt.empty()
                                ? std::vector<int>{128, 256, 512, 1024}
                                : blocks_opt;
  std::vector<std::uint32_t> seen;
  seen.push_back(lr.block_x);
  for (const int b : blocks) {
    if (b < 1) {
      continue;
    }
    const std::uint32_t ub = static_cast<std::uint32_t>(b);
    if (std::find(seen.begin(), seen.end(), ub) != seen.end()) {
      continue;
    }
    // The captured block is the workload-semantic baseline; only 1-D block
    // sizes within the launch limits are explored.
    if (ub > kMaxBlockThreads ||
        (b > 1 && (base_block[1] > 1 || base_block[2] > 1))) {
      continue;
    }
    seen.push_back(ub);
    seeds.push_back(CandidateSeed{{grid[0], grid[1], grid[2]},
                                  {ub, 1, 1},
                                  lr.shared_mem_bytes, /*is_baseline=*/false});
  }

  // shared-mem carveout: only when the captured launch used dynamic smem
  if (lr.shared_mem_bytes > 0 && arch.smem_per_block_max_bytes > 0) {
    const std::uint32_t cap =
        static_cast<std::uint32_t>(arch.smem_per_block_max_bytes);
    for (const std::uint32_t s :
         {std::uint32_t(16 * 1024), std::uint32_t(32 * 1024),
          std::uint32_t(48 * 1024), std::uint32_t(cap)}) {
      if (s <= lr.shared_mem_bytes) {
        continue;
      }
      if (std::find_if(seeds.begin(), seeds.end(), [&](const CandidateSeed &c) {
            return c.block[0] == lr.block_x && c.block[1] == lr.block_y &&
                   c.block[2] == lr.block_z && c.shared == s;
          }) != seeds.end()) {
        continue;
      }
      seeds.push_back(CandidateSeed{{grid[0], grid[1], grid[2]},
                                    {lr.block_x, lr.block_y, lr.block_z}, s,
                                    /*is_baseline=*/false});
    }
  }

  if (!inject_config.empty()) {
    CandidateSeed inj{{1, 1, 1}, {1, 1, 1}, 0, false};
    unsigned gx;
    unsigned gy;
    unsigned gz;
    unsigned bx;
    unsigned by;
    unsigned bz;
    unsigned shared = 0;
    int parsed = 0;
    if (std::sscanf(inject_config.c_str(), "%u,%u,%u:%u,%u,%u:%u", &gx, &gy,
                    &gz, &bx, &by, &bz, &shared) == 7) {
      parsed = 7;
    } else if (std::sscanf(inject_config.c_str(), "%u,%u,%u:%u,%u,%u", &gx,
                           &gy, &gz, &bx, &by, &bz) == 6) {
      parsed = 6;
    }
    if (parsed == 0) {
      *error = "bad --inject-config '" + inject_config +
               "' (expected gx,gy,gz:bx,by,bz[:shared])";
      return {};
    }
    inj.grid[0] = gx;
    inj.grid[1] = gy;
    inj.grid[2] = gz;
    inj.block[0] = bx;
    inj.block[1] = by;
    inj.block[2] = bz;
    inj.shared = shared;
    seeds.push_back(inj);
  }
  return seeds;
}

//--- the replay session (mirrors tools/capture SetupReplay, todo 48) --------//

struct TuneCtx {
  Plan plan;
  std::unique_ptr<GpuRuntime> rt;
  std::unique_ptr<Module> mod;
  std::unique_ptr<Kernel> kernel;
  std::unique_ptr<TempModuleFile> temp_module;
  std::string device_name;
  std::vector<DevicePtr> devs;
  std::vector<KernelArg> marshalled;
  std::vector<std::uint8_t> slot_data;
  std::vector<std::vector<std::uint8_t>> replay_bytes;
  std::vector<std::vector<std::uint8_t>> ref_bytes;
  std::vector<std::string> ref_sources;
  std::vector<std::string> ref_sha;
};

bool SetupReplay(const CaptureTuneOptions &opts, TuneCtx &ctx,
                 std::string *error) {
  PlanOptions popts;
  popts.vendor = opts.vendor;
  popts.arch = opts.arch;
  popts.blob_dir = opts.blob_dir;
  popts.kernel = opts.kernel;
  if (!BuildPlan(opts.bundle_path, popts, ctx.plan, error)) {
    return false;
  }
  if (ctx.plan.kernel_binary.empty()) {
    *error = "bundle has no kernel binary blob for '" + ctx.plan.kernel_name +
             "' - nothing to launch";
    return false;
  }

  auto rt = create_runtime(Backend::kAuto);
  if (!rt.ok()) {
    *error = "create_runtime: " + rt.message;
    return false;
  }
  if (rt.value->backend_name() == "compile-only") {
    *error = "no live GPU backend (compile-only build): capture->tune needs a "
             "real device";
    return false;
  }
  ctx.rt = std::move(rt.value);
  auto devs = ctx.rt->list_devices();
  if (!devs.ok() || devs.value.empty()) {
    *error = "no GPU devices: " + devs.message;
    return false;
  }
  ctx.device_name = devs.value.front().name;

  const std::string suffix = ctx.plan.vendor == "amd" ? ".hsaco" : ".cubin";
  ctx.temp_module =
      std::make_unique<TempModuleFile>(ctx.plan.kernel_binary, suffix, error);
  if (!ctx.temp_module->ok()) {
    return false;
  }
  auto mod = Module::load(*ctx.rt, ctx.temp_module->path().string());
  if (!mod.ok()) {
    *error = "module load (" + ctx.temp_module->path().string() + "): " +
             mod.message;
    return false;
  }
  auto kernel = Kernel::create(*ctx.rt, *mod.value,
                               StripKernelDescriptor(ctx.plan.kernel_name));
  if (!kernel.ok()) {
    *error = "kernel '" + StripKernelDescriptor(ctx.plan.kernel_name) +
             "' not found in module: " + kernel.message;
    return false;
  }
  ctx.mod = std::move(mod.value);
  ctx.kernel = std::move(kernel.value);

  // device buffers (fresh allocations = the remap destination)
  ctx.devs.reserve(ctx.plan.buffers.size());
  for (const DevBuffer &b : ctx.plan.buffers) {
    auto m = ctx.rt->device_malloc(b.size == 0 ? 1 : b.size);
    if (!m.ok()) {
      *error = "device_malloc(" + std::to_string(b.size) + ") for " + b.id +
               ": " + m.message;
      return false;
    }
    ctx.devs.push_back(m.value);
  }

  // remap table: ORIGINAL VA -> NEW VA (every plan buffer with a VA)
  std::unordered_map<std::uint64_t, std::uint64_t> remap;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].va != 0) {
      remap[ctx.plan.buffers[i].va] = ctx.devs[i].value;
    }
  }

  // input upload; v1 embedded-pointer patch tables applied before upload
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    DevBuffer &b = ctx.plan.buffers[i];
    if (b.kind == BufKind::kOutput || b.bytes.empty()) {
      continue;
    }
    if (!b.patches.empty()) {
      apply_remap(b.bytes, b.patches, remap);
    }
    GpuStatus s = ctx.rt->memcpy_htod(ctx.devs[i], b.bytes.data(), b.bytes.size());
    if (!s.ok()) {
      *error = "upload " + b.id + ": " + s.message;
      return false;
    }
  }

  // argument marshalling: pointer slots carry the NEW device VA
  ctx.slot_data.assign(static_cast<std::size_t>(ctx.plan.args.size()) * 8, 0);
  ctx.marshalled.reserve(ctx.plan.args.size());
  for (std::size_t i = 0; i < ctx.plan.args.size(); ++i) {
    const ArgPlan &a = ctx.plan.args[i];
    std::uint64_t *cell =
        reinterpret_cast<std::uint64_t *>(ctx.slot_data.data() + (i * 8));
    if (a.is_pointer) {
      if (a.buffer < 0 || static_cast<std::size_t>(a.buffer) >= ctx.devs.size()) {
        *error = "arg " + std::to_string(i) + " resolves to no buffer";
        return false;
      }
      *cell = ctx.devs[static_cast<std::size_t>(a.buffer)].value;
      ctx.marshalled.push_back(KernelArg{cell, sizeof(std::uint64_t)});
    } else {
      std::memcpy(cell, &a.scalar, std::min<std::size_t>(a.size, 8));
      ctx.marshalled.push_back(KernelArg{cell, a.size});
    }
  }

  // captured output references: v1 bundle snapshots first; --ref files (v0)
  // override per output in buffer order
  ctx.ref_bytes.resize(ctx.plan.buffers.size());
  ctx.ref_sources.resize(ctx.plan.buffers.size());
  ctx.ref_sha.resize(ctx.plan.buffers.size());
  ctx.replay_bytes.resize(ctx.plan.buffers.size());
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    if (!ctx.plan.output_refs.empty() && i < ctx.plan.output_refs.size() &&
        !ctx.plan.output_refs[i].bytes.empty()) {
      ctx.ref_bytes[i] = ctx.plan.output_refs[i].bytes;
      ctx.ref_sha[i] = ctx.plan.output_refs[i].sha256;
      ctx.ref_sources[i] = "bundle " + ctx.plan.output_refs[i].source;
    }
  }
  std::size_t file_idx = 0;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    if (file_idx >= opts.ref_files.size()) {
      break;
    }
    std::vector<std::uint8_t> bytes;
    if (!ReadFileBytes(opts.ref_files[file_idx], bytes, error)) {
      return false;
    }
    ctx.ref_bytes[i] = std::move(bytes);
    ctx.ref_sha[i] = sha256_hex(ctx.ref_bytes[i]);
    ctx.ref_sources[i] = "ref file " + opts.ref_files[file_idx];
    ++file_idx;
  }
  // a v0 bundle without --ref files and without v1 snapshots has NO reference
  bool have_ref = false;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind == BufKind::kOutput && !ctx.ref_sha[i].empty()) {
      have_ref = true;
    }
  }
  if (!have_ref) {
    *error = "no reference output for validation (v0 bundles: pass --ref "
             "<captured-output>)";
    return false;
  }
  return true;
}

//--- one candidate replay: launch + sync + copy back -------------------------//
// Returns launch_failed=false + validation_pass when the launch ran; the
// caller records the bitwise verdict.
bool ReplayOnce(TuneCtx &ctx, const CandidateSeed &seed,
                CaptureTuneCandidate &cand, std::string *error) {
  LaunchConfig cfg;
  cfg.grid_x = seed.grid[0];
  cfg.grid_y = seed.grid[1];
  cfg.grid_z = seed.grid[2];
  cfg.block_x = seed.block[0];
  cfg.block_y = seed.block[1];
  cfg.block_z = seed.block[2];
  cfg.shared_mem_bytes = seed.shared;

  GpuStatus s = ctx.kernel->launch(cfg, ctx.marshalled);
  if (!s.ok()) {
    *error = "launch: " + s.message;
    return false;
  }
  s = ctx.rt->synchronize();
  if (!s.ok()) {
    *error = "synchronize: " + s.message;
    return false;
  }

  bool pass = true;
  std::string replay_hex;
  std::string ref_hex;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    ctx.replay_bytes[i].resize(ctx.plan.buffers[i].size);
    GpuStatus c =
        ctx.rt->memcpy_dtoh(ctx.replay_bytes[i].data(), ctx.devs[i],
                            ctx.replay_bytes[i].size());
    if (!c.ok()) {
      *error = "copyback " + ctx.plan.buffers[i].id + ": " + c.message;
      return false;
    }
    const bool eq = ctx.replay_bytes[i] == ctx.ref_bytes[i];
    pass = pass && eq;
    replay_hex = sha256_hex(ctx.replay_bytes[i]);
    ref_hex = ctx.ref_sha[i];
  }
  cand.validation_pass = pass;
  cand.replay_sha = replay_hex;
  cand.ref_sha = ref_hex;
  return true;
}

//--- evidence JSON ------------------------------------------------------------//

Value BuildEvidence(const CaptureTuneResult &r) {
  std::vector<Value> cands;
  for (const auto &c : r.candidates) {
    std::vector<Value::Member> m{
        {"config", ConfigToJson(c)},
        {"status", Value(c.status)},
        {"validation", Value(std::vector<Value::Member>{
                            {"pass", Value(c.validation_pass)},
                            {"ref_sha", Value(c.ref_sha)},
                            {"replay_sha", Value(c.replay_sha)}})},
    };
    if (!c.reason.empty()) {
      m.push_back({"reason", Value(c.reason)});
    }
    if (c.benchmarked) {
      m.push_back({"timings", TimingsToJson(c.median_ms, c.p20_ms, c.p80_ms)});
      m.push_back({"speedup", Value(c.speedup)});
    }
    cands.push_back(Value(std::move(m)));
  }
  Value baseline = Value(std::vector<Value::Member>{});
  Value best = Value(std::vector<Value::Member>{});
  if (!r.candidates.empty()) {
    const auto &b = r.candidates.front();
    baseline = Value(std::vector<Value::Member>{
        {"config", ConfigToJson(b)},
        {"timings", TimingsToJson(b.median_ms, b.p20_ms, b.p80_ms)},
        {"validation", Value(std::vector<Value::Member>{
                            {"pass", Value(b.validation_pass)},
                            {"ref_sha", Value(b.ref_sha)},
                            {"replay_sha", Value(b.replay_sha)}})},
    });
  }
  if (r.has_best) {
    best = Value(std::vector<Value::Member>{
        {"config", r.best_config},
        {"timings",
         TimingsToJson(r.best_median_ms, r.best_p20_ms, r.best_p80_ms)},
    });
  }
  return Value(std::vector<Value::Member>{
      {"status", Value(r.status == CaptureTuneStatus::kOk ? "ok"
                                                          : "no_viable_config")},
      {"mode", Value("capture_tune")},
      {"message", Value(r.message)},
      {"bundle", Value(r.candidates.empty() ? std::string()
                                            : r.db_path.empty()
                                                  ? std::string()
                                                  : std::string(""))},
      {"kernel_name", Value(r.kernel_name)},
      {"vendor", Value(r.vendor)},
      {"arch", Value(r.arch)},
      {"device_name", Value(r.device_name)},
      {"kernel_name_hash", Value(r.kernel_name_hash)},
      {"bundle_hash", Value(r.bundle_hash)},
      {"db_key", Value(std::vector<Value::Member>{
                     {"gpu_arch", Value(r.arch)},
                     {"kernel_family", Value(r.kernel_name_hash)},
                     {"shape_hash", Value(r.bundle_hash)}})},
      {"baseline", std::move(baseline)},
      {"best", std::move(best)},
      {"speedup", Value(r.speedup)},
      {"candidates", Value(std::move(cands))},
      {"summary", Value(std::vector<Value::Member>{
                       {"candidates_total", Value(static_cast<double>(r.candidates_total))},
                       {"pre_screen_rejected", Value(static_cast<double>(r.pre_screen_rejected))},
                       {"launch_failed", Value(static_cast<double>(r.launch_failed))},
                       {"validation_failed", Value(static_cast<double>(r.validation_failed))},
                       {"bench_failed", Value(static_cast<double>(r.bench_failed))},
                       {"validated", Value(static_cast<double>(r.validated))},
                   })},
  });
}

} // namespace

//===--- public API ------------------------------------------------------------===//

const char *capture_tune_status_name(CaptureTuneStatus status) noexcept {
  switch (status) {
  case CaptureTuneStatus::kOk:
    return "ok";
  case CaptureTuneStatus::kNoViableConfig:
    return "no_viable_config";
  case CaptureTuneStatus::kNoLiveGpu:
    return "no_live_gpu";
  case CaptureTuneStatus::kUsageError:
    return "usage_error";
  case CaptureTuneStatus::kInternalError:
    return "internal_error";
  }
  return "unknown";
}

CaptureTuneResult tune_from_capture(const CaptureTuneOptions &opts) {
  CaptureTuneResult r;
  if (opts.bundle_path.empty()) {
    r.status = CaptureTuneStatus::kUsageError;
    r.message = "--from-capture requires a bundle path";
    r.evidence_json = BuildEvidence(r).dump();
    return r;
  }

  // --- plan + runtime setup -------------------------------------------------
  TuneCtx ctx;
  std::string error;
  if (!SetupReplay(opts, ctx, &error)) {
    const bool no_gpu = error.find("no live GPU") != std::string::npos ||
                        error.find("create_runtime") != std::string::npos;
    r.status = no_gpu ? CaptureTuneStatus::kNoLiveGpu
                      : CaptureTuneStatus::kUsageError;
    r.message = error;
    r.evidence_json = BuildEvidence(r).dump();
    return r;
  }

  r.kernel_name = ctx.plan.kernel_name;
  r.vendor = ctx.plan.vendor;
  r.arch = ctx.plan.arch;
  r.device_name = ctx.device_name;
  r.kernel_name_hash = "kernel=" + KernelNameHash(ctx.plan.kernel_name);
  std::string bhash = BundleIdentityHash(opts.bundle_path, &error);
  if (bhash.empty()) {
    r.status = CaptureTuneStatus::kUsageError;
    r.message = "bundle identity hash: " + error;
    r.evidence_json = BuildEvidence(r).dump();
    return r;
  }
  r.bundle_hash = "bundle=" + bhash;

  // --- candidate enumeration (baseline ALWAYS first) ------------------------
  ArchInfo arch{};
  // Unknown arch key: run without the smem budget check (limits that do
  // not need the table still apply) — has_arch guards the throwing lookup.
  if (kernscope::has_arch(ctx.plan.arch)) {
    arch = get_arch(ctx.plan.arch);
  }
  std::vector<CandidateSeed> seeds =
      BuildCandidates(ctx.plan.launch_rec, arch, opts.candidate_blocks,
                      opts.inject_config, &error);
  if (seeds.empty()) {
    r.status = CaptureTuneStatus::kUsageError;
    r.message = error.empty() ? "no candidates to explore" : error;
    r.evidence_json = BuildEvidence(r).dump();
    return r;
  }

  r.candidates_total = static_cast<int>(seeds.size());
  r.candidates.reserve(seeds.size());

  // --- candidate loop --------------------------------------------------------
  for (std::size_t si = 0; si < seeds.size(); ++si) {
    const CandidateSeed &seed = seeds[si];
    CaptureTuneCandidate cand;
    std::copy(std::begin(seed.grid), std::end(seed.grid), cand.grid);
    std::copy(std::begin(seed.block), std::end(seed.block), cand.block);
    cand.shared_mem_bytes = seed.shared;
    cand.is_baseline = seed.is_baseline;

    // pre-screen (never replayed when rejected)
    std::string reason;
    if (!PreScreenConfig(cand.grid, cand.block, cand.shared_mem_bytes, arch,
                         &reason)) {
      cand.status = "pre_screen_rejected";
      cand.reason = reason;
      ++r.pre_screen_rejected;
      r.candidates.push_back(std::move(cand));
      continue;
    }

    // replay + bitwise validation
    std::string lerr;
    if (!ReplayOnce(ctx, seed, cand, &lerr)) {
      cand.status = "launch_failed";
      cand.reason = lerr;
      ++r.launch_failed;
      r.candidates.push_back(std::move(cand));
      continue;
    }
    if (!cand.validation_pass) {
      cand.status = "validation_failed";
      ++r.validation_failed;
      r.candidates.push_back(std::move(cand));
      continue;
    }

    // bench (libs/bench: N reps, L2 flush, median — todo-22 methodology)
    bench::Options bopts;
    bopts.l2_flush = opts.l2_flush;
    bopts.min_samples = std::max(1, opts.reps);
    bopts.min_time_ms = 0.0;
    bopts.max_noise = 1.0;
    bopts.timeout_ms = 0.0;
    bopts.rep_ms = 0.0;
    LaunchConfig cfg;
    cfg.grid_x = seed.grid[0];
    cfg.grid_y = seed.grid[1];
    cfg.grid_z = seed.grid[2];
    cfg.block_x = seed.block[0];
    cfg.block_y = seed.block[1];
    cfg.block_z = seed.block[2];
    cfg.shared_mem_bytes = seed.shared;
    const bench::Result br = bench::run(
        *ctx.rt, ctx.device_name,
        [&]() { return ctx.kernel->launch(cfg, ctx.marshalled); }, bopts);
    if (br.error != GpuError::kOk) {
      cand.status = "bench_failed";
      cand.reason = br.error_message;
      ++r.bench_failed;
      r.candidates.push_back(std::move(cand));
      continue;
    }
    cand.status = "validated";
    cand.benchmarked = true;
    cand.median_ms = br.median_ms;
    cand.p20_ms = br.p20_ms;
    cand.p80_ms = br.p80_ms;
    ++r.validated;
    r.candidates.push_back(std::move(cand));
  }

  // --- selection --------------------------------------------------------------
  // baseline median = candidate #1 (the captured config, always explored
  // first); best = min median (ties: p20, then p80, then enumeration order).
  for (const auto &c : r.candidates) {
    if (c.is_baseline && c.benchmarked) {
      r.baseline_median_ms = c.median_ms;
    }
  }
  const CaptureTuneCandidate *best = nullptr;
  for (const auto &c : r.candidates) {
    if (!c.benchmarked) {
      continue;
    }
    if (!best || c.median_ms < best->median_ms ||
        (c.median_ms == best->median_ms && c.p20_ms < best->p20_ms) ||
        (c.median_ms == best->median_ms && c.p20_ms == best->p20_ms &&
         c.p80_ms < best->p80_ms)) {
      best = &c;
    }
  }
  for (auto &c : r.candidates) {
    if (c.benchmarked && r.baseline_median_ms > 0.0) {
      c.speedup = r.baseline_median_ms / c.median_ms;
    }
  }
  if (!best) {
    r.status = CaptureTuneStatus::kNoViableConfig;
    r.message =
        "no valid config: every candidate was pre-screen rejected, failed to "
        "launch, failed bitwise validation, or failed benchmarking";
    r.evidence_json = BuildEvidence(r).dump();
    return r;
  }
  r.has_best = true;
  r.best_config = ConfigToJson(*best);
  r.best_median_ms = best->median_ms;
  r.best_p20_ms = best->p20_ms;
  r.best_p80_ms = best->p80_ms;
  r.speedup = r.baseline_median_ms > 0.0 ? r.baseline_median_ms / best->median_ms
                                         : 1.0;
  r.status = CaptureTuneStatus::kOk;
  r.message = "best config selected (" + std::string(best->status) + ")";

  // --- libs/db persistence (kOk only — the no-viable-config negative asserts
  // NO db write) --------------------------------------------------------------
  if (!opts.db_path.empty()) {
    db::Database merged;
    std::vector<db::LoadWarning> warnings;
    if (fs::exists(opts.db_path)) {
      (void)merged.load(opts.db_path, warnings);
    }
    for (const auto &c : r.candidates) {
      if (!c.benchmarked) {
        continue;
      }
      Value config = Value(std::vector<Value::Member>{
          {"grid", Value(std::vector<Value>{
                       Value(static_cast<double>(c.grid[0])),
                       Value(static_cast<double>(c.grid[1])),
                       Value(static_cast<double>(c.grid[2]))})},
          {"block", Value(std::vector<Value>{
                        Value(static_cast<double>(c.block[0])),
                        Value(static_cast<double>(c.block[1])),
                        Value(static_cast<double>(c.block[2]))})},
          {"shared_mem_bytes", Value(static_cast<double>(c.shared_mem_bytes))},
          {"is_baseline", Value(c.is_baseline)},
          {"kernel", Value(r.kernel_name)},
          {"bundle", Value(opts.bundle_path)},
          {"device_name", Value(r.device_name)},
          {"reps", Value(static_cast<double>(opts.reps))},
          {"l2_flush", Value(opts.l2_flush)},
          {"validation", Value("pass")},
          {"date", Value(IsoDateUtc())},
          {"measured", Value(true)},
      });
      db::Record rec;
      rec.gpu_arch = r.arch;
      rec.kernel_family = r.kernel_name_hash;
      rec.shape_hash = r.bundle_hash;
      rec.config = std::move(config);
      rec.timings = {c.median_ms, c.p20_ms, c.p80_ms};
      rec.provenance.timestamp = IsoTimestampUtc();
      rec.provenance.tool_version = "kernscope-tune " KERNSCOPE_VERSION_STRING;
      rec.provenance.mode = "measured";
      merged.put(std::move(rec));
    }
    const fs::path db_path(opts.db_path);
    const fs::path dir = db_path.parent_path();
    if (!dir.empty() && !fs::exists(dir)) {
      std::error_code ec;
      fs::create_directories(dir, ec);
    }
    if (!merged.save(opts.db_path)) {
      r.message = "best config selected but cannot write db file " +
                  opts.db_path;
      r.status = CaptureTuneStatus::kInternalError;
    } else {
      r.db_path = opts.db_path;
    }
  }

  // --- profile-out (todo-49 launch-record contract, per candidate) ----------
  if (!opts.profile_out.empty()) {
    std::ofstream out(opts.profile_out, std::ios::trunc);
    for (const auto &c : r.candidates) {
      out << "{\"kind\":\"launch\",\"kernel\":\"" << r.kernel_name
          << "\",\"grid\":[" << c.grid[0] << "," << c.grid[1] << "," << c.grid[2]
          << "],\"block\":[" << c.block[0] << "," << c.block[1] << ","
          << c.block[2] << "],\"status\":\"" << c.status << "\",\"median_ms\":"
          << c.median_ms << "}\n";
    }
  }

  r.evidence_json = BuildEvidence(r).dump();
  return r;
}

} // namespace kernscope::tune
