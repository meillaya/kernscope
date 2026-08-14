//===- driver_pipeline_impl.h - tuning driver pipeline (template) ----------===//
//
// kernscope tools/tune todo 28. The family-generic pipeline core. Instantiated
// once per kernel family in the per-family adapter TUs (driver_matmul /
// driver_elem / driver_flash) so that no TU includes more than one generator
// header — the generators each declare their own ConfigVerdict/LaunchGeometry
// in kernscope::tune, so they must never be mixed in one translation unit.
//
// Pipeline (per config): cache-hit check -> GPU-free pre-screen
// (generator validate_config + libs/occupancy; libs/roofline family position
// recorded in the evidence) -> compile (hipcc --genco / nvcc -c) -> golden
// gate (run_hip_gate.py, discard on non-pass:true) -> libs/bench -> select
// best by median -> libs/db JSONL records + Triton-style tunecache JSON.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_DRIVER_PIPELINE_IMPL_H
#define KERNSCOPE_TUNE_DRIVER_PIPELINE_IMPL_H

#include "kernscope/tune/driver.h"

#include "kernscope/arch.h"
#include "kernscope/bench.h"
#include "kernscope/db.h"
#include "kernscope/gpu.h"
#include "kernscope/occupancy.h"
#include "kernscope/roofline.h"
#include "kernscope/version.h"

#include "driver_util.h"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

// Toolchain paths injected by tools/tune/CMakeLists.txt (absolute at
// configure time; empty when the piece was not found -> the driver records an
// environmental failure instead of a broken compile).
#ifndef KERNSCOPE_TUNE_NVCC
#define KERNSCOPE_TUNE_NVCC ""
#endif
#ifndef KERNSCOPE_TUNE_HIPCC
#define KERNSCOPE_TUNE_HIPCC ""
#endif
#ifndef KERNSCOPE_TUNE_HIP_INCLUDE
#define KERNSCOPE_TUNE_HIP_INCLUDE ""
#endif
#ifndef KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE
#define KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE ""
#endif
#ifndef KERNSCOPE_TUNE_REPO_ROOT
#define KERNSCOPE_TUNE_REPO_ROOT ""
#endif
#ifndef KERNSCOPE_TUNE_RUN_HIP_GATE
#define KERNSCOPE_TUNE_RUN_HIP_GATE ""
#endif
#ifndef KERNSCOPE_TUNE_PYTHON
#define KERNSCOPE_TUNE_PYTHON "python3"
#endif
#ifndef KERNSCOPE_TUNE_RUNNER
#define KERNSCOPE_TUNE_RUNNER ""
#endif

namespace kernscope::tune {
namespace driver_detail {

// Toolchain paths resolved once per TU (private to the driver lib).
inline const std::string kNvcc = KERNSCOPE_TUNE_NVCC;
inline const std::string kHipcc = KERNSCOPE_TUNE_HIPCC;
inline const std::string kHipInclude = KERNSCOPE_TUNE_HIP_INCLUDE;
inline const std::string kTraitsInclude = KERNSCOPE_TUNE_KERNEL_TRAITS_INCLUDE;
inline const std::string kRepoRoot = KERNSCOPE_TUNE_REPO_ROOT;
inline const std::string kRunHipGate = KERNSCOPE_TUNE_RUN_HIP_GATE;
inline const std::string kPython = KERNSCOPE_TUNE_PYTHON;
inline const std::string kRunner = KERNSCOPE_TUNE_RUNNER;

// ---------------------------------------------------------------------------
// Tunecache (Triton-style disk cache, brief §1 check_disk_cache).
// ---------------------------------------------------------------------------

struct CachedOutcome {
  std::string status;    // "prescreen_rejected" | "compile_failed" |
                         // "golden_gate_failed" | "bench_failed" |
                         // "benchmarked" | "compiled"
  std::string reason;
  bool has_timings = false;
  double median = 0.0;
  double p20 = 0.0;
  double p80 = 0.0;
  // Prescreen evidence, cached so a replayed entry still carries the
  // occupancy estimate + golden-gate launch geometry in the evidence
  // (todo 29: the report surfaces occupancy/occupancy.limiter in every
  // output mode, cache-hit runs included; the repro (--emit-repro) reads
  // the winning variant's gate geometry from the cache).
  double occupancy = 0.0;
  std::string limiter;
  std::string gate_shape;   // "64,64,64" (empty when never gated)
  std::string gate_grid;    // "gx" or "gx x gy" (run_hip_gate --grid)
  std::string gate_block;   // "bx x by" (run_hip_gate --block)
  std::string gate_symbol;  // kernel symbol the gate resolved
  // The min_occupancy floor this outcome was produced under. The pre-screen
  // (and therefore the outcome) depends on the floor, so a cached entry is
  // only valid for the SAME floor: a "prescreen_rejected" recorded at
  // floor 1.0 must not replay for floor 0.0 (the config may be viable
  // there), and a "benchmarked" recorded at floor 0.0 must not replay for
  // floor 1.0. Entries written before this field existed load as 0.0 and
  // are therefore valid only for the default floor (graceful compatibility).
  double min_occupancy = 0.0;
};

class Tunecache {
public:
  bool load(const std::string &path, std::string *warning) {
    entries_.clear();
    if (path.empty())
      return true;
    const std::string text = read_file(path);
    if (text.empty()) {
      if (!std::filesystem::exists(path))
        return true;  // no cache yet: fine
      if (warning)
        *warning = "tunecache " + path + " is empty; starting fresh";
      return true;
    }
    std::string err;
    const auto parsed = db::json::Value::parse(text, err);
    if (!parsed) {
      if (warning)
        *warning = "tunecache parse failed (" + err + "); starting fresh";
      return false;
    }
    const db::json::Value *entries = parsed->find("entries");
    if (!entries || entries->kind() != db::json::Kind::Object)
      return true;
    for (const auto &[key, entry] : entries->as_object()) {
      const db::json::Value *configs = entry.find("configs");
      if (!configs || configs->kind() != db::json::Kind::Array)
        continue;
      Entry e;
      for (const db::json::Value &cfg : configs->as_array()) {
        const db::json::Value *c = cfg.find("config");
        const db::json::Value *o = cfg.find("outcome");
        if (!c || !o)
          continue;
        CachedOutcome out;
        if (const db::json::Value *s = o->find("status"))
          out.status = s->as_string();
        if (const db::json::Value *r = o->find("reason"))
          out.reason = r->as_string();
        // Pre-floor-field entries (old cache format) have no min_occupancy:
        // they load as 0.0 and stay valid only for the default floor.
        if (const db::json::Value *m = o->find("min_occupancy"))
          out.min_occupancy = m->as_number();
        if (const db::json::Value *t = o->find("timings")) {
          out.has_timings = true;
          if (const db::json::Value *m = t->find("median"))
            out.median = m->as_number();
          if (const db::json::Value *p = t->find("p20"))
            out.p20 = p->as_number();
          if (const db::json::Value *p = t->find("p80"))
            out.p80 = p->as_number();
        }
        if (const db::json::Value *m = o->find("occupancy"))
          out.occupancy = m->as_number();
        if (const db::json::Value *l = o->find("limiter"))
          out.limiter = l->as_string();
        if (const db::json::Value *g = o->find("gate")) {
          if (const db::json::Value *s = g->find("shape"))
            out.gate_shape = s->as_string();
          if (const db::json::Value *s = g->find("grid"))
            out.gate_grid = s->as_string();
          if (const db::json::Value *s = g->find("block"))
            out.gate_block = s->as_string();
          if (const db::json::Value *s = g->find("symbol"))
            out.gate_symbol = s->as_string();
        }
        const std::string canonical = c->dump();  // canonical -> key
        e[canonical] = {*c, std::move(out)};
      }
      entries_[key] = std::move(e);
    }
    return true;
  }

  // Lookup a cached outcome for (cache_key, canonical) produced under
  // `floor`. Returns false when absent OR the stored floor differs (an entry
  // recorded under a different min_occupancy must not satisfy this run — the
  // pre-screen outcome depends on the floor).
  bool get(const std::string &cache_key, const std::string &canonical,
           double floor, CachedOutcome *out) const {
    const auto it = entries_.find(cache_key);
    if (it == entries_.end())
      return false;
    const auto jt = it->second.find(canonical);
    if (jt == it->second.end())
      return false;
    if (jt->second.outcome.min_occupancy != floor)
      return false;
    if (out)
      *out = jt->second.outcome;
    return true;
  }

  void set(const std::string &cache_key, const db::json::Value &config,
           const std::string &canonical, CachedOutcome outcome) {
    entries_[cache_key][canonical] = {config, std::move(outcome)};
  }

  bool save(const std::string &path, const std::string &gpu_arch,
            const std::string &kernel_family, const std::string &shape_hash,
            const std::vector<std::int64_t> &shape, std::string *error) const {
    if (path.empty())
      return true;
    std::vector<db::json::Value::Member> entry_members;
    for (const auto &[key, entry] : entries_) {
      std::vector<db::json::Value> configs;
      for (const auto &[canonical, rec] : entry) {
        (void)canonical;
        std::vector<db::json::Value::Member> out_members{
            {"status", db::json::Value(rec.outcome.status)},
            {"min_occupancy", db::json::Value(rec.outcome.min_occupancy)}};
        if (!rec.outcome.reason.empty())
          out_members.emplace_back("reason", db::json::Value(rec.outcome.reason));
        if (rec.outcome.has_timings)
          out_members.emplace_back(
              "timings", db::json::object(
                             {{"median", db::json::Value(rec.outcome.median)},
                              {"p20", db::json::Value(rec.outcome.p20)},
                              {"p80", db::json::Value(rec.outcome.p80)}}));
        if (rec.outcome.occupancy > 0.0)
          out_members.emplace_back("occupancy",
                                   db::json::Value(rec.outcome.occupancy));
        if (!rec.outcome.limiter.empty())
          out_members.emplace_back("limiter",
                                   db::json::Value(rec.outcome.limiter));
        if (!rec.outcome.gate_shape.empty()) {
          std::vector<db::json::Value::Member> gate_members{
              {"shape", db::json::Value(rec.outcome.gate_shape)},
              {"grid", db::json::Value(rec.outcome.gate_grid)},
              {"block", db::json::Value(rec.outcome.gate_block)},
              {"symbol", db::json::Value(rec.outcome.gate_symbol)}};
          out_members.emplace_back("gate",
                                   db::json::Value(std::move(gate_members)));
        }
        configs.push_back(db::json::object(
            {{"config", rec.config},
             {"outcome", db::json::Value(std::move(out_members))}}));
      }
      std::vector<db::json::Value> shape_arr;
      for (const std::int64_t d : shape)
        shape_arr.emplace_back(static_cast<double>(d));
      entry_members.emplace_back(
          key, db::json::object(
                   {{"key",
                     db::json::object({{"gpu_arch", db::json::Value(gpu_arch)},
                                       {"kernel_family",
                                        db::json::Value(kernel_family)},
                                       {"shape_hash",
                                        db::json::Value(shape_hash)},
                                       {"shape", db::json::Value(shape_arr)}})},
                    {"configs", db::json::Value(std::move(configs))}}));
    }
    const db::json::Value root = db::json::object(
        {{"schema_version", db::json::Value(1)},
         {"entries", db::json::Value(std::move(entry_members))}});
    const std::string text = root.dump() + "\n";
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty())
      std::filesystem::create_directories(dir, ec);
    if (!write_file(path, text)) {
      if (error)
        *error = "cannot write tunecache " + path;
      return false;
    }
    return true;
  }

private:
  struct ConfigRecord {
    db::json::Value config;
    CachedOutcome outcome;
  };
  using Entry = std::map<std::string, ConfigRecord>;
  std::map<std::string, Entry> entries_;  // cache_key -> canonical -> record
};

// ---------------------------------------------------------------------------
// One processed variant (the evidence row + selection input).
// ---------------------------------------------------------------------------

struct VariantRecord {
  int index = 0;
  db::json::Value config;
  std::string canonical;   // canonical config dump — the cache key within entry
  std::string status;      // see CachedOutcome.status
  std::string reason;
  bool cached = false;     // replayed from the tunecache
  bool prescreened = false;
  double occupancy = 0.0;
  std::string limiter;
  int blocks_per_sm = 0;
  // Golden-gate launch geometry (set in the gate step; empty when the gate
  // never ran — cache replays restore them from the tunecache).
  std::string gate_shape;
  std::string gate_grid;
  std::string gate_block;
  std::string gate_symbol;
  bool has_timings = false;
  double median = 0.0;
  double p20 = 0.0;
  double p80 = 0.0;
  bool best = false;
};

// Dumps the golden input tensors for one (family, shape) to raw row-major
// .bin buffers under <outdir> (deterministic — the golden seed is fixed, so
// these are exactly the buffers the gate would have produced at that shape).
// The benchmark uploads THESE tune-shape buffers; the gate's own workdir
// holds gate-shape bins and is never reused for benchmarking.
const char kDumpBenchBinsPy[] = R"PY(import sys, os
sys.path.insert(0, sys.argv[1])
from kernscope_golden import inputs as ki
shape = tuple(int(x) for x in sys.argv[3].split(','))
arrays = ki.generate_inputs(sys.argv[2], shape)
os.makedirs(sys.argv[4], exist_ok=True)
for name, arr in arrays.items():
    with open(os.path.join(sys.argv[4], name + '.bin'), 'wb') as fh:
        fh.write(arr.tobytes())
)PY";

// ---------------------------------------------------------------------------
// The pipeline.
// ---------------------------------------------------------------------------

template <typename Traits>
struct Tuner {
  using Config = typename Traits::Config;
  using Problem = typename Traits::Problem;

  static DriverResult run(const DriverOptions &opts, const DriverHooks *hooks) {
    DriverResult out;
    out.status = DriverStatus::kUsageError;

    // --- option sanity ------------------------------------------------------
    if (!kernscope::has_arch(opts.arch)) {
      out.message = "unknown arch '" + opts.arch + "' (libs/arch DB); use a "
                    "concrete key (gfx1101, sm_80, ...) — 'auto' is resolved "
                    "to gfx1101 by the CLI";
      return out;
    }
    const ArchInfo &arch = kernscope::get_arch(opts.arch);
    if (opts.grid_preset != "auto" && opts.grid_preset != "full") {
      out.message = "unknown grid preset '" + opts.grid_preset +
                    "' (auto|full)";
      return out;
    }
    if (opts.shape.size() < Traits::kShapeArity) {
      out.message = std::string("shape for family '") + Traits::kFamily +
                    "' needs >= " + std::to_string(Traits::kShapeArity) +
                    " dims (got " + std::to_string(opts.shape.size()) + ")";
      return out;
    }
    for (const std::int64_t d : opts.shape) {
      if (d <= 0) {
        out.message = "shape dims must be positive";
        return out;
      }
    }

    const Problem prob = Traits::make_problem(opts);
    out.shape_hash = make_shape_hash(Traits::kFamily, opts.shape);
    out.cache_key = make_cache_key(opts.arch, Traits::kFamily, out.shape_hash);
    out.variants_total = 0;

    // --- workdir ------------------------------------------------------------
    std::string workdir = opts.workdir;
    if (workdir.empty())
      workdir = driver_detail::default_workdir(Traits::kFamily, out.shape_hash);
    std::error_code ec;
    const std::string abs_workdir =
        std::filesystem::absolute(workdir, ec).string();
    if (ec || !driver_detail::ensure_dir(abs_workdir)) {
      out.message = "cannot create workdir " + abs_workdir;
      out.status = DriverStatus::kInternalError;
      return out;
    }

    // --- tunecache (resume) -------------------------------------------------
    Tunecache cache;
    std::string cache_warning;
    if (opts.use_cache)
      cache.load(opts.cache_path, &cache_warning);
    else
      cache_warning = "cache read skipped (--no-cache); fresh results are "
                      "written";

    // --- family-level roofline position (recorded in the evidence) ----------
    db::json::Value roofline_entry;
    {
      const roofline::Shape rshape = Traits::roofline_shape(opts);
      const std::int64_t flops =
          roofline::compute_flops(Traits::kRooflineFamily, rshape);
      const std::int64_t bytes =
          roofline::compute_bytes_moved(Traits::kRooflineFamily, rshape);
      roofline::RooflinePosition pos;
      std::string rerr;
      // The lite templates are scalar (no tensor-core instructions), so the
      // honest compute roof is the FP32 vector peak — measured on gfx1101
      // (projected == false, todo 22).
      const roofline::ErrorCode rc = roofline::roofline_position(
          Traits::kRooflineFamily, rshape, bytes, flops, opts.arch, &pos,
          &rerr, roofline::PeakKind::kFp32);
      if (rc == roofline::ErrorCode::kNone) {
        // position = the workload's fraction of the compute roof
        // (attained_flops / compute ceiling — min(1, AI/ridge)); bound is
        // the stable "compute"|"memory"|"latency" class (todo 29 contract:
        // roofline.position / roofline.bound surface in every output mode).
        const double position =
            pos.compute_bound_ceiling > 0.0
                ? pos.attained_flops / pos.compute_bound_ceiling
                : 0.0;
        roofline_entry = db::json::object(
            {{"ok", db::json::Value(true)},
             {"family", db::json::Value(roofline::kernel_family_name(
                            Traits::kRooflineFamily))},
             {"flops", db::json::Value(static_cast<double>(flops))},
             {"bytes_moved", db::json::Value(static_cast<double>(bytes))},
             {"arithmetic_intensity",
              db::json::Value(pos.arithmetic_intensity)},
             {"ridge_point", db::json::Value(pos.ridge_point)},
             {"bound_class",
              db::json::Value(roofline::bound_class_name(pos.bound_class))},
             {"bound",
              db::json::Value(roofline::bound_class_name(pos.bound_class))},
             {"position", db::json::Value(position)},
             {"compute_bound_ceiling",
              db::json::Value(pos.compute_bound_ceiling)},
             {"memory_bound_ceiling",
              db::json::Value(pos.memory_bound_ceiling)},
             {"attained_flops", db::json::Value(pos.attained_flops)}});
      } else {
        roofline_entry = db::json::object(
            {{"ok", db::json::Value(false)},
             {"family", db::json::Value(roofline::kernel_family_name(
                            Traits::kRooflineFamily))},
             {"flops", db::json::Value(static_cast<double>(flops))},
             {"bytes_moved", db::json::Value(static_cast<double>(bytes))},
             {"error", db::json::Value(rerr)}});
      }
    }

    // --- environmental gate for the live phases -----------------------------
    // NVIDIA runtimes are owner-gated: sm_XX is compile-only by design.
    const bool want_live = opts.live_bench && arch.vendor == Vendor::kAmd;
    if (want_live && kRunner.empty()) {
      out.message =
          "live golden/bench phase needs the kernscope_templates_hip_runner "
          "(not built at configure time; `cmake --build build` first)";
      out.status = DriverStatus::kNoLiveGpu;
      out.evidence_json =
          BuildEvidence(out, opts, abs_workdir, roofline_entry, {},
                        cache_warning, 0, false)
              .dump();
      return out;
    }

    const std::vector<Config> grid = Traits::enumerate(opts);
    out.variants_total = static_cast<int>(grid.size());
    std::vector<VariantRecord> variants;
    variants.reserve(grid.size());
    std::optional<BenchSession> session;  // created lazily on first bench

    // The correctness gate validates the kernel at `gate_shape` (defaults to
    // the tune shape; matmul overrides to the golden-validated 64^3), while
    // the benchmark measures the requested `shape`. The gate's geometry is
    // therefore built from the gate shape and the bench's from the tune shape.
    const std::vector<std::int64_t> gate_shape = Traits::gate_shape(opts);
    const Problem gate_prob = Traits::make_problem_for(gate_shape, opts.arch);

    // --- per-config pipeline -------------------------------------------------
    for (std::size_t i = 0; i < grid.size(); ++i) {
      const Config &cfg = grid[i];
      VariantRecord v;
      v.index = static_cast<int>(i);
      v.config = Traits::config_json(cfg);
      v.canonical = v.config.dump();

      // 0. cache hit -> skip everything (resume; brief §1 check_disk_cache).
      CachedOutcome hit;
      if (cache.get(out.cache_key, v.canonical, opts.min_occupancy, &hit)) {
        v.cached = true;
        v.status = hit.status;
        v.reason = hit.reason;
        v.has_timings = hit.has_timings;
        v.median = hit.median;
        v.p20 = hit.p20;
        v.p80 = hit.p80;
        v.occupancy = hit.occupancy;
        v.limiter = hit.limiter;
        v.gate_shape = hit.gate_shape;
        v.gate_grid = hit.gate_grid;
        v.gate_block = hit.gate_block;
        v.gate_symbol = hit.gate_symbol;
        ++out.cached;
        variants.push_back(std::move(v));
        continue;
      }

      // 1. GPU-free pre-screen: generator validate_config (libs/arch budgets)
      //    then libs/occupancy (smem/block/warp limits + min-occupancy floor).
      //    `auto` (not the family's ConfigVerdict name): the generators each
      //    define their own ConfigVerdict, so the type must stay dependent.
      const auto verdict = Traits::validate(cfg, prob);
      if (!verdict.valid) {
        v.status = "prescreen_rejected";
        v.reason = "generator pre-screen: " + verdict.reason;
        ++out.prescreen_rejected;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }
      const int block = Traits::block_size(cfg, prob);
      const int regs = Traits::estimated_regs(cfg, prob);
      const std::int64_t smem = Traits::smem_bytes(cfg, prob);
      const occupancy::OccupancyResult occ = occupancy::compute_occupancy(
          arch, regs, static_cast<int>(smem), block);
      v.prescreened = true;
      v.occupancy = occ.occupancy;
      v.limiter = occupancy::limiter_name(occ.limiter);
      v.blocks_per_sm = occ.blocks_per_sm;
      if (occ.blocks_per_sm <= 0) {
        v.status = "prescreen_rejected";
        v.reason = "occupancy pre-screen: 0 resident blocks (smem/block/warp "
                   "limits)";
        ++out.prescreen_rejected;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }
      if (opts.min_occupancy > 0.0 && occ.occupancy < opts.min_occupancy) {
        v.status = "prescreen_rejected";
        v.reason = "occupancy pre-screen: occupancy " +
                   std::to_string(occ.occupancy) + " < min_occupancy " +
                   std::to_string(opts.min_occupancy) + " (limiter=" +
                   std::string(v.limiter) + ")";
        ++out.prescreen_rejected;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }

      // 2. compile (generate source -> nvcc/hipcc per target).
      const std::string src = abs_workdir + "/v" + std::to_string(i) + ".hip";
      {
        std::string err;
        if (!Traits::generate(cfg, prob, src, &err)) {
          v.status = "compile_failed";
          v.reason = "generate: " + err;
          ++out.compile_failed;
          CachePut(cache, out.cache_key, v, opts.min_occupancy);
          variants.push_back(std::move(v));
          continue;
        }
      }
      if (hooks && hooks->rewrite_source) {
        std::string source = driver_detail::read_file(src);
        const std::string repl = hooks->rewrite_source(opts, v.config, source);
        if (!repl.empty())
          (void)driver_detail::write_file(src, repl);
      }
      const std::string obj =
          abs_workdir + "/v" + std::to_string(i) +
          (arch.vendor == Vendor::kAmd ? ".hsaco" : ".o");
      const int compile_rc = CompileKernel(arch, src, obj, abs_workdir);
      ++out.compiled;
      if (compile_rc != 0) {
        v.status = "compile_failed";
        v.reason =
            "compile failed (rc=" + std::to_string(compile_rc) + "): " +
            driver_detail::log_tail(
                driver_detail::read_file(abs_workdir + "/cmd.log"));
        ++out.compile_failed;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }

      if (!want_live) {
        // NVIDIA / compile-only: the gate + bench need a live HIP device.
        v.status = "compiled";
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }

      // 3. correctness gate (run_hip_gate.py; discard on non-pass:true).
      //    `auto`: the family's LaunchGeometry name is likewise per-generator.
      const auto gate_geo = Traits::geometry(cfg, gate_prob);
      const std::string grid_str =
          std::to_string(gate_geo.grid_x) +
          (gate_geo.grid_y > 1 ? "x" + std::to_string(gate_geo.grid_y) : "");
      const std::string block_str =
          std::to_string(gate_geo.block_x) + "x" +
          std::to_string(gate_geo.block_y);
      v.gate_shape = Traits::shape_string(gate_shape);
      v.gate_grid = grid_str;
      v.gate_block = block_str;
      v.gate_symbol = Traits::symbol(cfg);
      const int gate_rc =
          RunGate(abs_workdir, obj, Traits::symbol(cfg), grid_str, block_str,
                  Traits::shape_string(gate_shape));
      if (gate_rc != 0) {
        v.status = "golden_gate_failed";
        v.reason =
            "golden gate failed (rc=" + std::to_string(gate_rc) +
            ", verdict not pass:true): " +
            driver_detail::log_tail(
                driver_detail::read_file(abs_workdir + "/cmd.log"));
        ++out.golden_failed;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }

      // 4. benchmark (libs/bench; only AFTER the gate passed — the
      //    correctness-gated discipline). Launch geometry, scalars and output
      //    size come from the TUNE shape.
      if (!session)
        session.emplace();
      if (!InitSession(*session, abs_workdir, Traits::shape_string(opts.shape),
                       out)) {
        v.status = "bench_failed";
        v.reason = "benchmark session: " + out.message;
        ++out.bench_failed;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }
      const auto geo = Traits::geometry(cfg, prob);
      const kernscope::bench::Result br = BenchKernel(
          *session, abs_workdir + "/v" + std::to_string(i) + ".hsaco",
          Traits::symbol(cfg), geo, Traits::scalar_args(opts.shape),
          Traits::out_bytes(opts.shape));
      if (br.error != kernscope::GpuError::kOk) {
        v.status = "bench_failed";
        v.reason = "benchmark failed: " + br.error_message;
        ++out.bench_failed;
        CachePut(cache, out.cache_key, v, opts.min_occupancy);
        variants.push_back(std::move(v));
        continue;
      }
      v.status = "benchmarked";
      v.has_timings = true;
      v.median = br.median_ms;
      v.p20 = br.p20_ms;
      v.p80 = br.p80_ms;
      ++out.benchmarked;
      CachePut(cache, out.cache_key, v, opts.min_occupancy);
      variants.push_back(std::move(v));
    }

    // --- 5. select best by median (ties: p20, then p80, then order) --------
    const int best_idx = SelectBest(variants);
    if (best_idx >= 0) {
      const VariantRecord &best = variants[static_cast<std::size_t>(best_idx)];
      out.has_best = true;
      out.best_config = best.config;
      out.best_median_ms = best.median;
      out.best_p20_ms = best.p20;
      out.best_p80_ms = best.p80;
      variants[static_cast<std::size_t>(best_idx)].best = true;
    }
    for (const VariantRecord &v : variants)
      if (v.has_timings)
        out.measured_medians.push_back(v.median);

    // --- persist libs/db records + tunecache ---------------------------------
    int db_records = 0;
    bool db_ok = true;
    if (!opts.db_path.empty()) {
      db_records = static_cast<int>(std::count_if(
          variants.begin(), variants.end(),
          [](const VariantRecord &v) { return v.has_timings; }));
      db_ok = WriteDbRecords(opts.db_path, out, opts, arch, variants);
    }
    std::string cache_err;
    const bool cache_ok =
        cache.save(opts.cache_path, opts.arch, Traits::kFamily,
                   out.shape_hash, opts.shape, &cache_err);

    // --- status + message ----------------------------------------------------
    if (out.has_best) {
      out.status = DriverStatus::kOk;
      std::string m = "best config median=" +
                      std::to_string(out.best_median_ms) + " ms; variants=" +
                      std::to_string(out.variants_total) + " (prescreen " +
                      std::to_string(out.prescreen_rejected) + ", golden-fail " +
                      std::to_string(out.golden_failed) + ", benchmarked " +
                      std::to_string(out.benchmarked) + ", cached " +
                      std::to_string(out.cached) + ", compiled " +
                      std::to_string(out.compiled) + ")";
      if (!opts.cache_path.empty())
        m += "; cache: " + opts.cache_path;
      out.message = std::move(m);
    } else if (arch.vendor == Vendor::kNvidia) {
      out.status = DriverStatus::kNoLiveGpu;
      out.message = std::string("arch ") + std::string(arch.name) +
                    ": compile-only gates passed (" +
                    std::to_string(out.compiled) +
                    " compiled); benchmarking requires a live HIP device "
                    "(NVIDIA runs are owner-gated)";
    } else if (!opts.live_bench) {
      out.status = DriverStatus::kNoLiveGpu;
      out.message =
          "no viable config: benchmark phase disabled (live_bench=false)";
    } else if (out.prescreen_rejected == out.variants_total &&
               out.variants_total > 0) {
      out.status = DriverStatus::kNoViableConfig;
      const std::string first_reason =
          variants.empty() ? std::string() : variants.front().reason;
      out.message = "no viable config: all " +
                    std::to_string(out.variants_total) +
                    " variants were rejected by the GPU-free pre-screen; "
                    "first reason: " + first_reason;
    } else if (out.compiled > 0) {
      out.status = DriverStatus::kNoViableConfig;
      out.message = "no viable config: " + std::to_string(out.compiled) +
                    " variant(s) compiled but none passed the golden gate "
                    "(correctness-gated discipline)";
    } else {
      out.status = DriverStatus::kNoViableConfig;
      out.message = "no viable config: 0 viable variants";
    }

    // --- evidence ------------------------------------------------------------
    out.evidence_json =
        BuildEvidence(out, opts, abs_workdir, roofline_entry, variants,
                      cache_warning, db_records, db_ok)
            .dump();
    (void)cache_ok;
    (void)cache_err;
    return out;
  }

  // ---------------------------------------------------------------------------
  // Helpers.
  // ---------------------------------------------------------------------------

  // Cache an outcome with the run's min_occupancy floor stamped in, so a
  // later run under a different floor misses it (see CachedOutcome).
  static void CachePut(Tunecache &cache, const std::string &key,
                       const VariantRecord &v, double floor) {
    CachedOutcome o = CachedOutcomeFrom(v);
    o.min_occupancy = floor;
    cache.set(key, v.config, v.canonical, std::move(o));
  }

  static CachedOutcome CachedOutcomeFrom(const VariantRecord &v) {
    CachedOutcome c;
    c.status = v.status;
    c.reason = v.reason;
    c.has_timings = v.has_timings;
    c.median = v.median;
    c.p20 = v.p20;
    c.p80 = v.p80;
    c.occupancy = v.occupancy;
    c.limiter = v.limiter;
    c.gate_shape = v.gate_shape;
    c.gate_grid = v.gate_grid;
    c.gate_block = v.gate_block;
    c.gate_symbol = v.gate_symbol;
    return c;
  }

  static int SelectBest(const std::vector<VariantRecord> &variants) {
    int best = -1;
    for (std::size_t i = 0; i < variants.size(); ++i) {
      const VariantRecord &v = variants[i];
      if (!v.has_timings)
        continue;
      if (best < 0) {
        best = static_cast<int>(i);
        continue;
      }
      const VariantRecord &b = variants[static_cast<std::size_t>(best)];
      if (v.median < b.median ||
          (v.median == b.median && v.p20 < b.p20) ||
          (v.median == b.median && v.p20 == b.p20 && v.p80 < b.p80) ||
          (v.median == b.median && v.p20 == b.p20 && v.p80 == b.p80 &&
           i < static_cast<std::size_t>(best)))
        best = static_cast<int>(i);
    }
    return best;
  }

  // hipcc --genco for AMD, nvcc compile-only for NVIDIA.
  static int CompileKernel(const ArchInfo &arch, const std::string &src,
                           const std::string &out, const std::string &dir) {
    std::string cmd;
    if (arch.vendor == Vendor::kAmd) {
      if (kHipcc.empty() || kHipInclude.empty())
        return 127;
      cmd = kHipcc + " --genco --offload-arch=" + std::string(arch.name) +
            " -gline-tables-only -D__HIP_PLATFORM_AMD__=1 -I" + kHipInclude +
            " -I" + kTraitsInclude + " " + src + " -o " + out;
    } else {
      if (kNvcc.empty())
        return 127;
      cmd = kNvcc + " -arch=" + std::string(arch.name) +
            " -lineinfo -x cu -std=c++20 -I" + kTraitsInclude + " -c " + src +
            " -o " + out;
    }
    return driver_detail::run_command(dir, cmd);
  }

  static int RunGate(const std::string &workdir, const std::string &hsaco,
                     const std::string &symbol, const std::string &grid,
                     const std::string &block,
                     const std::string &gate_shape_csv) {
    std::string cmd = kPython + " " + kRunHipGate + " --root " + kRepoRoot +
                      " --workdir " + workdir + "/gate" + " --hsaco " + hsaco +
                      " --runner " + kRunner + " --symbol " + symbol +
                      " --family " + Traits::kGoldenFamily + " --shape " +
                      gate_shape_csv + " --grid " + grid + " --block " + block;
    if (Traits::kCausal)
      cmd += " --causal";
    return driver_detail::run_command(workdir, cmd);
  }

  // One shared live-runtime session: the golden input bins are uploaded once
  // (they are deterministic across configs — fixed golden seed — so every
  // benchmarked variant measures the SAME data).
  struct BenchSession {
    std::unique_ptr<kernscope::GpuRuntime> rt;
    std::string device;
    std::vector<kernscope::DevicePtr> inputs;
    kernscope::DevicePtr out;
    std::size_t out_bytes = 0;
    std::vector<std::int32_t> scalars;
    std::vector<std::uint64_t> ptrs;
    std::vector<kernscope::KernelArg> args;
    bool ready = false;
  };

  static bool InitSession(BenchSession &s, const std::string &workdir,
                          const std::string &shape_csv, DriverResult &out) {
    if (s.ready)
      return true;
    auto rt = kernscope::create_runtime(kernscope::Backend::kAuto);
    if (!rt.ok() || rt.value->backend_name() != "hip") {
      out.message = "benchmark needs a live HIP device: " + rt.message;
      return false;
    }
    s.rt = std::move(rt.value);
    auto devs = s.rt->list_devices();
    if (!devs.ok() || devs.value.empty()) {
      out.message = "no HIP devices: " + devs.message;
      return false;
    }
    s.device = devs.value.front().name;

    // Dump the tune-shape golden input bins (once per run) into
    // <workdir>/bench, then upload them.
    const std::string bench_dir = workdir + "/bench";
    const std::string script = workdir + "/dump_bench_bins.py";
    if (!driver_detail::write_file(script, kDumpBenchBinsPy)) {
      out.message = "cannot write bin-dump script " + script;
      return false;
    }
    std::string cmd = kPython + " " + script + " " + kRepoRoot +
                      "/tests/golden" + " " + Traits::kGoldenFamily + " " +
                      shape_csv + " " + bench_dir;
    const int rc = driver_detail::run_command(workdir, cmd);
    if (rc != 0) {
      out.message = "bench bin dump failed (rc=" + std::to_string(rc) + "): " +
                    driver_detail::log_tail(
                        driver_detail::read_file(workdir + "/cmd.log"));
      return false;
    }
    for (const std::string &bin : Traits::bench_bins()) {
      const std::string path = bench_dir + "/" + bin;
      const std::string bytes = driver_detail::read_file(path);
      if (bytes.empty()) {
        out.message = "bench input bin missing: " + path;
        return false;
      }
      auto d = s.rt->device_malloc(bytes.size());
      if (!d.ok()) {
        out.message = "input upload alloc: " + d.message;
        return false;
      }
      GpuStatus st = s.rt->memcpy_htod(d.value, bytes.data(), bytes.size());
      if (!st.ok()) {
        out.message = "input upload: " + st.message;
        (void)s.rt->device_free(d.value);
        return false;
      }
      s.inputs.push_back(d.value);
    }
    s.ready = true;
    return true;
  }

  static kernscope::bench::Result BenchKernel(
      BenchSession &s, const std::string &hsaco, const std::string &symbol,
      const LaunchGeometry &geo, const std::vector<std::int32_t> &scalars,
      std::size_t out_bytes) {
    auto mod = kernscope::Module::load(*s.rt, hsaco);
    if (!mod.ok())
      return BenchError("Module::load(" + hsaco + "): " + mod.message);
    auto kern = kernscope::Kernel::create(*s.rt, *mod.value, symbol);
    if (!kern.ok())
      return BenchError("Kernel::create(" + symbol + "): " + kern.message);

    // Output buffer: reallocated when the size changes, reused across reps
    // (Triton's reset-to-zero semantics are not needed — every rep rewrites
    // the same region and timing is unaffected by its contents).
    if (s.out.value == 0 || s.out_bytes != out_bytes) {
      if (s.out.value != 0)
        (void)s.rt->device_free(s.out);
      auto d = s.rt->device_malloc(out_bytes);
      if (!d.ok())
        return BenchError("device_malloc(out): " + d.message);
      s.out = d.value;
      s.out_bytes = out_bytes;
    }

    // Driver-API marshalling: input device pointers by value, then the output
    // pointer, then the int32 scalars (kernel parameter contract).
    std::vector<std::uint64_t> ptrs;
    for (const kernscope::DevicePtr &p : s.inputs)
      ptrs.push_back(p.value);
    ptrs.push_back(s.out.value);
    std::vector<kernscope::KernelArg> args;
    for (const std::uint64_t &p : ptrs)
      args.push_back(kernscope::KernelArg{&p, sizeof(p)});
    for (const std::int32_t &sc : scalars)
      args.push_back(kernscope::KernelArg{&sc, sizeof(sc)});

    kernscope::LaunchConfig cfg;
    cfg.grid_x = static_cast<std::uint32_t>(geo.grid_x);
    cfg.grid_y = static_cast<std::uint32_t>(geo.grid_y);
    cfg.block_x = static_cast<std::uint32_t>(geo.block_x);
    cfg.block_y = static_cast<std::uint32_t>(geo.block_y);

    kernscope::bench::Options bopts;  // todo-22 defaults: 25/100 ms budgets,
                                      // L2 flush, adaptive trials
    return kernscope::bench::run(*s.rt, s.device,
                                 [&]() { return kern.value->launch(cfg, args); },
                                 bopts);
  }

  static kernscope::bench::Result BenchError(const std::string &msg) {
    kernscope::bench::Result r;
    r.error = kernscope::GpuError::kOther;
    r.error_message = msg;
    return r;
  }

  static bool WriteDbRecords(const std::string &path, const DriverResult &out,
                             const DriverOptions &opts, const ArchInfo &arch,
                             const std::vector<VariantRecord> &variants) {
    (void)arch;
    kernscope::db::Database db;
    std::vector<kernscope::db::LoadWarning> warnings;
    if (std::filesystem::exists(path) && !db.load(path, warnings))
      return false;
    std::error_code ec;
    const std::filesystem::path dir = std::filesystem::path(path).parent_path();
    if (!dir.empty())
      std::filesystem::create_directories(dir, ec);
    for (const VariantRecord &v : variants) {
      if (!v.has_timings)
        continue;
      kernscope::db::Record rec;
      rec.gpu_arch = opts.arch;
      rec.kernel_family = Traits::kFamily;
      rec.shape_hash = out.shape_hash;
      rec.config = v.config;
      rec.timings = {v.median, v.p20, v.p80};
      rec.provenance.timestamp = driver_detail::iso_timestamp_utc();
      rec.provenance.tool_version =
          "kernscope-tune " KERNSCOPE_VERSION_STRING;
      rec.provenance.mode = "measured";
      db.put(std::move(rec));
    }
    return db.save(path);
  }

  static db::json::Value BuildEvidence(
      const DriverResult &out, const DriverOptions &opts,
      const std::string &workdir, const db::json::Value &roofline_entry,
      const std::vector<VariantRecord> &variants,
      const std::string &cache_warning, int db_records, bool db_ok) {
    // The naive baseline: the FIRST enumerated variant that was measured
    // (enumeration order is the search-space order — the first viable
    // config, e.g. matmul's smallest tile / no pipelining / single warp).
    // Speedup is computed against it before the variants are re-ordered for
    // the report (measured-first, median-ascending).
    double baseline_median = 0.0;
    db::json::Value baseline_config;  // null when no variant measured
    for (const VariantRecord &v : variants) {
      if (v.has_timings) {
        baseline_median = v.median;
        baseline_config = v.config;
        break;
      }
    }

    // Report order: measured variants first sorted by median (ties p20,
    // p80), then non-measured variants in enumeration order. The check
    // contract: "variants sorted by median ascending".
    std::vector<const VariantRecord *> ordered;
    ordered.reserve(variants.size());
    for (const VariantRecord &v : variants)
      ordered.push_back(&v);
    std::stable_sort(
        ordered.begin(), ordered.end(),
        [](const VariantRecord *a, const VariantRecord *b) {
          if (a->has_timings != b->has_timings)
            return a->has_timings;
          if (a->has_timings && a->median != b->median)
            return a->median < b->median;
          if (a->has_timings && a->p20 != b->p20)
            return a->p20 < b->p20;
          if (a->has_timings && a->p80 != b->p80)
            return a->p80 < b->p80;
          return a->index < b->index;
        });

    std::vector<db::json::Value> var_arr;
    for (const VariantRecord *vp : ordered) {
      const VariantRecord &v = *vp;
      std::vector<db::json::Value::Member> m{
          {"index", db::json::Value(v.index)},
          {"config", v.config},
          {"status", db::json::Value(v.status)},
          {"cached", db::json::Value(v.cached)},
          {"best", db::json::Value(v.best)},
      };
      if (!v.reason.empty())
        m.emplace_back("reason", db::json::Value(v.reason));
      if (v.prescreened || v.occupancy > 0.0) {
        m.emplace_back("occupancy", db::json::Value(v.occupancy));
        if (!v.limiter.empty())
          m.emplace_back("limiter", db::json::Value(v.limiter));
        m.emplace_back("blocks_per_sm", db::json::Value(v.blocks_per_sm));
      }
      if (!v.gate_shape.empty()) {
        m.emplace_back(
            "gate", db::json::object(
                        {{"shape", db::json::Value(v.gate_shape)},
                         {"grid", db::json::Value(v.gate_grid)},
                         {"block", db::json::Value(v.gate_block)},
                         {"symbol", db::json::Value(v.gate_symbol)}}));
      }
      if (v.has_timings)
        m.emplace_back(
            "timings", db::json::object(
                           {{"median", db::json::Value(v.median)},
                            {"p20", db::json::Value(v.p20)},
                            {"p80", db::json::Value(v.p80)}}));
      var_arr.push_back(db::json::Value(std::move(m)));
    }

    std::vector<db::json::Value> shape_arr;
    for (const std::int64_t d : opts.shape)
      shape_arr.emplace_back(static_cast<double>(d));
    std::vector<db::json::Value> gate_shape_arr;
    for (const std::int64_t d : Traits::gate_shape(opts))
      gate_shape_arr.emplace_back(static_cast<double>(d));

    std::vector<db::json::Value::Member> members{
        {"schema_version", db::json::Value(1)},
        {"tool_version",
         db::json::Value(std::string("kernscope-tune ") +
                         KERNSCOPE_VERSION_STRING)},
        {"status", db::json::Value(driver_status_name(out.status))},
        {"message", db::json::Value(out.message)},
        {"gpu_arch", db::json::Value(opts.arch)},
        {"kernel_family", db::json::Value(Traits::kFamily)},
        {"shape", db::json::Value(shape_arr)},
        {"gate_shape", db::json::Value(gate_shape_arr)},
        {"shape_hash", db::json::Value(out.shape_hash)},
        {"cache_key", db::json::Value(out.cache_key)},
        {"grid_preset", db::json::Value(opts.grid_preset)},
        {"grid_size", db::json::Value(out.variants_total)},
        {"min_occupancy", db::json::Value(opts.min_occupancy)},
        {"roofline", roofline_entry},
        {"workdir", db::json::Value(workdir)},
        {"summary",
         db::json::object(
             {{"variants_total", db::json::Value(out.variants_total)},
              {"prescreen_rejected",
               db::json::Value(out.prescreen_rejected)},
              {"compile_failed", db::json::Value(out.compile_failed)},
              {"golden_gate_failed", db::json::Value(out.golden_failed)},
              {"bench_failed", db::json::Value(out.bench_failed)},
              {"benchmarked", db::json::Value(out.benchmarked)},
              {"cached", db::json::Value(out.cached)},
              {"compiled", db::json::Value(out.compiled)}})},
        {"variants", db::json::Value(std::move(var_arr))},
    };
    if (out.has_best) {
      // The winning variant's prescreen evidence (occupancy estimate +
      // limiter) and golden-gate geometry — the report surfaces
      // occupancy / occupancy.limiter and the repro (--emit-repro) consumes
      // best.gate.
      double best_occ = 0.0;
      std::string best_limiter;
      std::string best_gate_shape, best_gate_grid, best_gate_block,
          best_gate_symbol;
      for (const VariantRecord &v : variants) {
        if (!v.best)
          continue;
        best_occ = v.occupancy;
        best_limiter = v.limiter;
        best_gate_shape = v.gate_shape;
        best_gate_grid = v.gate_grid;
        best_gate_block = v.gate_block;
        best_gate_symbol = v.gate_symbol;
      }
      members.emplace_back("occupancy", db::json::Value(best_occ));
      if (!best_limiter.empty())
        members.emplace_back("occupancy.limiter",
                             db::json::Value(best_limiter));
      members.emplace_back("speedup",
                           db::json::Value(baseline_median > 0.0
                                               ? baseline_median /
                                                     out.best_median_ms
                                               : 0.0));
      members.emplace_back(
          "baseline",
          db::json::object(
              {{"config", baseline_config},
               {"timings",
                db::json::object(
                    {{"median", db::json::Value(baseline_median)}})}}));
      std::vector<db::json::Value::Member> best_members{
          {"config", out.best_config},
          {"timings",
           db::json::object(
               {{"median", db::json::Value(out.best_median_ms)},
                {"p20", db::json::Value(out.best_p20_ms)},
                {"p80", db::json::Value(out.best_p80_ms)}})}};
      if (!best_gate_shape.empty())
        best_members.emplace_back(
            "gate",
            db::json::object(
                {{"shape", db::json::Value(best_gate_shape)},
                 {"grid", db::json::Value(best_gate_grid)},
                 {"block", db::json::Value(best_gate_block)},
                 {"symbol", db::json::Value(best_gate_symbol)}}));
      members.emplace_back("best", db::json::Value(std::move(best_members)));
    }
    if (!cache_warning.empty())
      members.emplace_back("cache_warning", db::json::Value(cache_warning));
    if (!opts.db_path.empty())
      members.emplace_back(
          "db", db::json::object(
                    {{"path", db::json::Value(opts.db_path)},
                     {"written", db::json::Value(db_ok)},
                     {"records", db::json::Value(db_records)}}));
    if (!opts.cache_path.empty())
      members.emplace_back("cache",
                           db::json::object(
                               {{"path", db::json::Value(opts.cache_path)}}));
    return db::json::Value(std::move(members));
  }
};

}  // namespace driver_detail
}  // namespace kernscope::tune

#endif  // KERNSCOPE_TUNE_DRIVER_PIPELINE_IMPL_H
