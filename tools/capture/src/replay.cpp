//===- replay.cpp - live replay + validation + timing hooks (todo 48) ------===//
//
// kernscope capture/replay track. See replay.h for the contract.
//
// Execution model (all local, never uploads anywhere):
//
//   1. BuildPlan (replay_model.h) — shared with emit-repro.
//   2. Runtime: create_runtime(kAuto) (libs/gpuruntime) — a live HIP/CUDA
//      backend on this machine; compile-only -> clear error (exit 2).
//   3. RESTORE: the kernel binary is written to a temp file (the abstraction
//      loads modules by path), loaded, and the target kernel is resolved by
//      its descriptor-less symbol name. Buffers are allocated fresh; the
//      remap table maps every ORIGINAL buffer VA -> NEW VA; v1 embedded-
//      pointer patch tables are applied through libs/capture-core's
//      apply_remap BEFORE the upload; pointer ARG slots are rewritten
//      through the same table (docs/report/kernscope-technical-report.pdf). Original-VA
//      restore is not available on HIP (HSA assigns VAs) and is documented
//      as the CUDA-VMM extension point in docs/report/kernscope-technical-report.pdf.
//   4. LAUNCH + synchronize; every launch emits a timestamped
//      LaunchTimingRecord (the todo-49 profiling hook contract).
//   5. VALIDATE every output buffer: bitwise (sha256 + byte compare) by
//      default, tolerance mode per tolerance.h. References: v1 output
//      snapshots from the bundle, or --ref files (v0) in output order.
//   6. bench: the same setup, one validation pass (unless disabled), then N
//      libs/bench timed reps (L2 flush, median) + a libs/db record with
//      measured provenance (mirroring the todo-42 tools/bench style).
//
//===----------------------------------------------------------------------===//

#include "replay.h"
#include "replay_model.h"

#include "kernscope/bench.h"
#include "kernscope/capture.h"
#include "kernscope/db.h"
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
#include <sstream>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kernscope {
namespace {

namespace fs = std::filesystem;

using std::chrono::steady_clock;

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
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "cannot open " + path;
    return false;
  }
  out.assign((std::istreambuf_iterator<char>(in)),
             std::istreambuf_iterator<char>());
  if (in.bad()) {
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
      path_ = dir / ("kernscope-replay-module-" + std::to_string(attempt) +
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

//--- the replay session (shared by replay + bench) --------------------------//

struct ReplayCtx {
  Plan plan;
  std::unique_ptr<GpuRuntime> rt;
  std::unique_ptr<Module> mod;
  std::unique_ptr<Kernel> kernel;
  std::unique_ptr<TempModuleFile> temp_module;
  std::string device_name;
  std::string gcn_arch_full;
  std::vector<DevicePtr> devs;       // per plan buffer
  std::vector<KernelArg> marshalled; // per arg slot
  std::vector<std::uint8_t> slot_data;
  std::vector<std::vector<std::uint8_t>> replay_bytes; // outputs, per buffer
  std::vector<std::vector<std::uint8_t>> ref_bytes;    // outputs, per buffer
  std::vector<std::string> ref_sources;
  std::vector<std::string> ref_sha;
};

/// Builds the session: plan + runtime + module + kernel + buffers + args.
bool SetupReplay(const CaptureBundleOptions &opts, ReplayCtx &ctx,
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
             "' - nothing to launch (the capture mechanism did not dump the "
             "code object)";
    return false;
  }

  auto rt = create_runtime(Backend::kAuto);
  if (!rt.ok()) {
    *error = "create_runtime: " + rt.message;
    return false;
  }
  if (rt.value->backend_name() == "compile-only") {
    *error = "no live GPU backend (compile-only build): replay needs a real "
             "device";
    return false;
  }
  ctx.rt = std::move(rt.value);
  auto devs = ctx.rt->list_devices();
  if (!devs.ok() || devs.value.empty()) {
    *error = "no GPU devices: " + devs.message;
    return false;
  }
  ctx.device_name = devs.value.front().name;
  auto props = ctx.rt->device_properties(ctx.device_name);
  if (props.ok()) {
    ctx.gcn_arch_full = props.value.gcn_arch_full;
  }

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

  // input upload; v1 embedded-pointer patch tables applied before upload so
  // deep pointers land already re-pointed on device
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    DevBuffer &b = ctx.plan.buffers[i];
    if (b.kind == BufKind::kOutput || b.bytes.empty()) {
      continue;
    }
    if (!b.patches.empty()) {
      apply_remap(b.bytes, b.patches, remap);
    }
    GpuStatus s = ctx.rt->memcpy_htod(ctx.devs[i], b.bytes.data(),
                                      b.bytes.size());
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

  // captured output references: v1 bundle snapshots first; --ref files
  // (v0) override per output in buffer order
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
  return true;
}

bool ApplyRefFiles(ReplayCtx &ctx, const std::vector<std::string> &ref_files,
                   std::string *error) {
  std::size_t file_idx = 0;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    if (file_idx >= ref_files.size()) {
      break;
    }
    std::vector<std::uint8_t> bytes;
    if (!ReadFileBytes(ref_files[file_idx], bytes, error)) {
      return false;
    }
    ctx.ref_bytes[i] = std::move(bytes);
    ctx.ref_sha[i] = sha256_hex(ctx.ref_bytes[i]);
    ctx.ref_sources[i] = "ref file " + ref_files[file_idx];
    ++file_idx;
  }
  if (file_idx < ref_files.size()) {
    std::fprintf(stderr,
                 "[kernscope-capture] warning: %zu extra --ref file(s); the "
                 "bundle has %zu output buffer(s)\n",
                 ref_files.size() - file_idx,
                 static_cast<std::size_t>(std::count_if(
                     ctx.plan.buffers.begin(), ctx.plan.buffers.end(),
                     [](const DevBuffer &b) { return b.kind == BufKind::kOutput; })));
  }
  return true;
}

/// One launch + synchronize, with the launch timing record appended to
/// `result.launches`; outputs are copied back into ctx.replay_bytes.
GpuStatus LaunchOnce(ReplayCtx &ctx, ReplayResult *result) {
  const LaunchRecord &lr = ctx.plan.launch_rec;
  LaunchConfig cfg;
  cfg.grid_x = lr.grid_x;
  cfg.grid_y = lr.grid_y;
  cfg.grid_z = lr.grid_z;
  cfg.block_x = lr.block_x;
  cfg.block_y = lr.block_y;
  cfg.block_z = lr.block_z;
  cfg.shared_mem_bytes = lr.shared_mem_bytes;

  const std::int64_t t0 = std::chrono::duration_cast<std::chrono::microseconds>(
                              steady_clock::now().time_since_epoch())
                              .count();
  GpuStatus s = ctx.kernel->launch(cfg, ctx.marshalled);
  if (!s.ok()) {
    return s;
  }
  s = ctx.rt->synchronize();
  if (!s.ok()) {
    return s;
  }
  const std::int64_t t1 = std::chrono::duration_cast<std::chrono::microseconds>(
                              steady_clock::now().time_since_epoch())
                              .count();

  LaunchTimingRecord rec;
  rec.kernel_name = ctx.plan.kernel_name;
  rec.seq = static_cast<std::uint64_t>(result->launches.size() + 1);
  rec.launched_at_us = t0;
  rec.completed_at_us = t1;
  rec.duration_us = static_cast<double>(t1 - t0);
  result->launches.push_back(std::move(rec));

  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    ctx.replay_bytes[i].resize(ctx.plan.buffers[i].size);
    GpuStatus c = ctx.rt->memcpy_dtoh(ctx.replay_bytes[i].data(), ctx.devs[i],
                                      ctx.replay_bytes[i].size());
    if (!c.ok()) {
      return c;
    }
  }
  return s;
}

void ValidateOutputs(ReplayCtx &ctx, bool tolerance, ReplayResult &result) {
  result.validation_pass = true;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    OutputComparison cmp;
    cmp.buffer_id = ctx.plan.buffers[i].id;
    cmp.ref_sha = ctx.ref_sha[i];
    cmp.replay_sha = sha256_hex(ctx.replay_bytes[i]);
    cmp.equal = ctx.replay_bytes[i] == ctx.ref_bytes[i];
    if (tolerance) {
      const ToleranceVerdict v =
          tolerance_verdict(ctx.replay_bytes[i], ctx.ref_bytes[i]);
      cmp.tolerance = v.metrics;
      cmp.tolerance_pass = v.pass;
    }
    cmp.ref_source = ctx.ref_sources[i].empty() ? "NONE" : ctx.ref_sources[i];
    if (ctx.ref_sources[i].empty()) {
      result.has_reference = false;
      result.validation_pass = false;
    } else if (tolerance) {
      result.validation_pass = result.validation_pass && cmp.tolerance_pass;
    } else {
      result.validation_pass = result.validation_pass && cmp.equal;
    }
    result.outputs.push_back(std::move(cmp));
  }
}

//--- profile hook output (todo-49 contract) ---------------------------------//

bool WriteProfileOut(const std::string &path,
                     const std::vector<LaunchTimingRecord> &launches,
                     std::string *error) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    *error = "cannot write " + path;
    return false;
  }
  for (const LaunchTimingRecord &l : launches) {
    out << "{\"kind\":\"launch\",\"kernel\":\"" << l.kernel_name
        << "\",\"seq\":" << l.seq << ",\"launched_at_us\":" << l.launched_at_us
        << ",\"completed_at_us\":" << l.completed_at_us
        << ",\"duration_us\":" << l.duration_us << "}\n";
  }
  return out.good();
}

bool DumpOutputs(ReplayCtx &ctx, const std::string &dir, std::string *error) {
  std::error_code ec;
  fs::create_directories(dir, ec);
  if (ec) {
    *error = "cannot create dump dir " + dir + ": " + ec.message();
    return false;
  }
  int out_idx = 0;
  for (std::size_t i = 0; i < ctx.plan.buffers.size(); ++i) {
    if (ctx.plan.buffers[i].kind != BufKind::kOutput) {
      continue;
    }
    const fs::path p =
        fs::path(dir) / ("output-" + std::to_string(out_idx) + ".bin");
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    if (!f) {
      *error = "cannot write " + p.string();
      return false;
    }
    f.write(reinterpret_cast<const char *>(ctx.replay_bytes[i].data()),
            static_cast<std::streamsize>(ctx.replay_bytes[i].size()));
    if (!f) {
      *error = "write failed: " + p.string();
      return false;
    }
    std::printf("[replay] dumped %s (%zu bytes)\n", p.c_str(),
                ctx.replay_bytes[i].size());
    ++out_idx;
  }
  return true;
}

} // namespace

//===--- entry points -----------------------------------------------------------===//

bool run_replay(const ReplayOptions &opts, ReplayResult &result,
                std::string *error) {
  result.tolerance = opts.tolerance;
  ReplayCtx ctx;
  if (!SetupReplay(opts, ctx, error)) {
    return false;
  }
  if (!ApplyRefFiles(ctx, opts.ref_files, error)) {
    return false;
  }
  result.kernel_name = ctx.plan.kernel_name;
  result.vendor = ctx.plan.vendor;
  result.arch = ctx.plan.arch;
  result.device_name = ctx.device_name;

  GpuStatus s = LaunchOnce(ctx, &result);
  if (!s.ok()) {
    *error = "launch: " + s.message;
    return false;
  }
  ValidateOutputs(ctx, opts.tolerance, result);

  if (!opts.dump_dir.empty()) {
    if (!DumpOutputs(ctx, opts.dump_dir, error)) {
      return false;
    }
  }
  if (!opts.profile_out.empty()) {
    if (!WriteProfileOut(opts.profile_out, result.launches, error)) {
      return false;
    }
  }
  return true;
}

bool run_replay_bench(const BenchOptions &opts, BenchResult &result,
                      std::string *error) {
  ReplayCtx ctx;
  if (!SetupReplay(opts, ctx, error)) {
    return false;
  }
  if (!ApplyRefFiles(ctx, opts.ref_files, error)) {
    return false;
  }

  ReplayResult vres;
  vres.tolerance = false;
  vres.kernel_name = ctx.plan.kernel_name;
  vres.vendor = ctx.plan.vendor;
  vres.arch = ctx.plan.arch;
  vres.device_name = ctx.device_name;
  if (opts.validate) {
    GpuStatus s = LaunchOnce(ctx, &vres);
    if (!s.ok()) {
      *error = "validation launch: " + s.message;
      return false;
    }
    ValidateOutputs(ctx, false, vres);
    result.validation_pass = vres.validation_pass;
    if (!result.validation_pass) {
      *error = "validation FAILED (refusing to benchmark an invalid replay)";
      return false;
    }
    std::printf("[bench] validation PASS (bitwise) before timing\n");
  }

  // timing: N reps, L2 flush, median (libs/bench harness)
  bench::Options bopts;
  bopts.l2_flush = opts.l2_flush;
  bopts.min_samples = std::max(1, opts.reps);
  bopts.min_time_ms = 0.0; // fixed sample count, not a time budget
  bopts.max_noise = 1.0;   // any noise converges; sample count decides
  bopts.timeout_ms = 0.0;  // no hard wall-clock cap for a fixed sample count
  // One sample per convergence-check batch (rep_ms=0 -> n_repeat=1), so the
  // run stops at exactly min_samples timed reps instead of overshooting to
  // a full 100 ms budget batch (~9100 reps at 11 us/call).
  bopts.rep_ms = 0.0;

  LaunchConfig cfg;
  cfg.grid_x = ctx.plan.launch_rec.grid_x;
  cfg.grid_y = ctx.plan.launch_rec.grid_y;
  cfg.grid_z = ctx.plan.launch_rec.grid_z;
  cfg.block_x = ctx.plan.launch_rec.block_x;
  cfg.block_y = ctx.plan.launch_rec.block_y;
  cfg.block_z = ctx.plan.launch_rec.block_z;
  cfg.shared_mem_bytes = ctx.plan.launch_rec.shared_mem_bytes;

  const bench::Result r = bench::run(
      *ctx.rt, ctx.device_name,
      [&]() { return ctx.kernel->launch(cfg, ctx.marshalled); }, bopts);
  if (r.error != GpuError::kOk) {
    *error = "bench: " + r.error_message;
    return false;
  }

  result.median_ms = r.median_ms;
  result.p20_ms = r.p20_ms;
  result.p80_ms = r.p80_ms;
  result.n_reps = r.n_reps;
  result.gpu_arch = ctx.plan.arch;

  // db record (measured provenance, tools/bench todo-42 style)
  using kernscope::db::json::Value;
  Value config = Value(std::vector<Value::Member>{
      {"bundle", Value(ctx.plan.bundle_path)},
      {"kernel", Value(ctx.plan.kernel_name)},
      {"device_name", Value(ctx.device_name)},
      {"gcn_arch_full", Value(ctx.gcn_arch_full)},
      {"reps", Value(static_cast<double>(opts.reps))},
      {"l2_flush", Value(opts.l2_flush)},
      {"validation", Value("pass")},
      {"date", Value(IsoDateUtc())},
      {"measured", Value(true)},
  });
  db::Record rec;
  rec.gpu_arch = ctx.plan.arch;
  rec.kernel_family = "capture_replay";
  rec.shape_hash = "kernel=" + sha256_hex(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(ctx.plan.kernel_name.data()),
      ctx.plan.kernel_name.size()));
  rec.config = std::move(config);
  rec.timings = {r.median_ms, r.p20_ms, r.p80_ms};
  rec.provenance.timestamp = IsoTimestampUtc();
  rec.provenance.tool_version = "kernscope-capture " KERNSCOPE_VERSION_STRING;
  rec.provenance.mode = "measured";
  result.shape_hash = rec.shape_hash;

  const fs::path db_path(opts.db_path);
  const fs::path dir = db_path.parent_path();
  if (!dir.empty() && !fs::exists(dir)) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
      *error = "cannot create " + dir.string() + ": " + ec.message();
      return false;
    }
  }
  db::Database merged;
  std::vector<db::LoadWarning> warnings;
  if (fs::exists(db_path)) {
    (void)merged.load(db_path, warnings);
  }
  merged.put(rec);
  if (!merged.save(db_path)) {
    *error = "cannot write db file " + opts.db_path;
    return false;
  }

  // aggregate profiling record for the bench (todo-49 contract)
  if (!opts.profile_out.empty()) {
    std::ofstream out(opts.profile_out, std::ios::trunc);
    if (!out) {
      *error = "cannot write " + opts.profile_out;
      return false;
    }
    for (const LaunchTimingRecord &l : vres.launches) {
      out << "{\"kind\":\"launch\",\"kernel\":\"" << l.kernel_name
          << "\",\"seq\":" << l.seq << ",\"launched_at_us\":"
          << l.launched_at_us << ",\"completed_at_us\":"
          << l.completed_at_us << ",\"duration_us\":" << l.duration_us
          << "}\n";
    }
    out << "{\"kind\":\"bench\",\"kernel\":\"" << ctx.plan.kernel_name
        << "\",\"reps\":" << r.n_reps << ",\"median_ms\":" << r.median_ms
        << ",\"p20_ms\":" << r.p20_ms << ",\"p80_ms\":" << r.p80_ms << "}\n";
    if (!out) {
      *error = "write failed: " + opts.profile_out;
      return false;
    }
  }
  return true;
}

} // namespace kernscope
