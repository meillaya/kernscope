//===- driver.h - correctness-gated tuning driver -------------------------===//
//
// kernscope tools/tune todo 28: the tuning driver — the integration core of
// the tune track. Consumed by the kernscope-tune CLI (todo 28) and the live
// acceptance suite (tests/tune_driver_test.cpp); later consumed by the todo-29
// report/reproducibility work.
//
// PIPELINE (per candidate config, driven here):
//   1. GPU-FREE PRE-SCREEN: reject infeasible configs WITHOUT compiling,
//      layering three independent libs (the cross-tool multiplier the task
//      calls the analyzer/occupancy/roofline synergy):
//        a. the family generator's `validate_config` — search-space
//           membership + arch budgets (register budget, smem budget,
//           divisibility) via libs/arch, with the generator's reason string;
//        b. libs/occupancy — `compute_occupancy(arch, regs_estimate, smem,
//           block)`; a config with 0 resident blocks is infeasible, and an
//           occupancy below `min_occupancy` (an option; 0 = no floor) is
//           rejected as not worth compiling/measuring;
//        c. libs/roofline — the family-level AI position (arithmetic
//           intensity, ridge point, bound class, attainable FLOPs) recorded
//           in the evidence. The lite templates are scalar (no tensor-core
//           instructions), so the roof is the FP32 vector peak — on gfx1101
//           that is the todo-22 MEASURED peak (projected == false).
//      Every rejection is recorded with its reason in the JSON evidence.
//   2. COMPILE: generate the single-source .hip kernel via the family
//      generator, then compile per target — hipcc --genco for gfxXXXX
//      (the gfx1101 live path) or nvcc compile-only for sm_XX (NVIDIA
//      runtimes are owner-gated; the driver never benchmarks on NVIDIA).
//   3. CORRECTNESS GATE: run the kernel through the live runner +
//      kernscope_golden (tests/golden) via run_hip_gate.py; a non-pass:true
//      verdict DISCARDS the variant and records the failure. The driver
//      NEVER benchmarks a variant that did not pass the golden gate
//      (polykernel correctness-gated discipline).
//   4. BENCHMARK: libs/bench (time-based warmup/rep budgets, L2 flush,
//      adaptive trials, median + p20/p80 quantiles — todo-22 methodology).
//   5. SELECT: best config by median (ties: p20, then p80, then
//      enumeration order — Triton min-over-quantile-tuples), then persist:
//        a. a libs/db JSONL record per measured config, keyed
//           (gpu_arch, kernel_family, shape_hash) — query_best by median;
//        b. a Triton-style tunecache JSON (arch + shape + family key,
//           `configs_timings` semantics — docs/research/
//           autotuner-research-brief.md §1): every processed config's
//           outcome is cached (timings for measured, status+reason for
//           rejected/failed, mirroring Triton's inf-cost for infeasible
//           configs), so a re-run with the same cache compiles NOTHING and
//           re-picks the winner without re-benchmarking (§1 check_disk_cache).
//
// RESUME: configs already in the tunecache are skipped up front (before the
// pre-screen) — the second run of the same (arch, family, shape) compiles 0
// new variants (asserted by the acceptance suite).
//
// ARCH SELECTION: `--arch auto` resolves to gfx1101 on this machine (the
// task contract); the driver itself takes a concrete libs/arch key. NVIDIA
// archs are compile-only by design (MUST NOT auto-run on NVIDIA).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_DRIVER_H
#define KERNSCOPE_TUNE_DRIVER_H

#include "kernscope/db.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kernscope::tune {

// --- pipeline outcome status -------------------------------------------------
enum class DriverStatus : std::uint8_t {
  kOk = 0,             // a best config was selected; db + tunecache written
  kNoViableConfig,     // every config was pre-screen rejected / gate-failed;
                       // `message` carries "no viable config" + a JSON reason
  kNoLiveGpu,          // the bench/gate phase needs a live HIP device (not
                       // available, or the arch is NVIDIA = compile-only)
  kUsageError,         // bad options: unknown family/arch/grid, bad shape
  kInternalError,      // IO or unexpected failure
};

// Stable machine-readable status names ("ok" | "no_viable_config" | ...).
const char *driver_status_name(DriverStatus status) noexcept;

// --- options ----------------------------------------------------------------
struct DriverOptions {
  std::string kernel_family;       // "matmul" | "bias_gelu" | "softmax" |
                                   // "layernorm" | "reduction" | "flash_attention"
  std::vector<std::int64_t> shape; // family-dependent: matmul M,N,K; elem
                                   // R,C; flash_attention Nq,Nk,d
  std::string arch = "gfx1101";    // libs/arch key (CLI resolves "auto")
  std::string grid_preset = "auto";// per-family bounded grid ("auto") or the
                                   // full search space ("full")
  std::string db_path;             // libs/db JSONL to write (empty = no db)
  std::string cache_path;          // tunecache JSON (empty = no cache)
  std::string workdir;             // scratch dir (empty = temp dir)
  double min_occupancy = 0.0;      // pre-screen occupancy floor (0 = disabled)
  bool live_bench = true;          // run golden gate + benchmark (needs a
                                   // live HIP device; false = compile-only)

  // Shape the CORRECTNESS GATE validates the kernel at, decoupled from the
  // benchmark shape. The golden contract pins max_rel_err <= 1e-2 for the
  // bf16 rounding contract, and a long-K fp32-accumulate GEMM legitimately
  // drifts past it at K=512 even though the kernel is bit-correct (cosine
  // 1.0); the family adapters therefore default matmul to the todo-25
  // golden-validated M=N=K=64 gate shape while benchmarking the requested
  // shape (the gate's job — reject wrong kernels — is unchanged: the
  // injected bad variant still fails it). Empty = the `shape` itself.
  std::vector<std::int64_t> gate_shape;
};

// --- test hook --------------------------------------------------------------
// Called after a config passes the pre-screen, right before compilation.
// Return a REPLACEMENT source ("" keeps the generated source). Used by
// tests/tune_driver_test.cpp to inject a variant that compiles but fails the
// golden gate (the driver must discard + record it). Not part of the CLI
// path (the CLI passes nullptr).
struct DriverHooks {
  std::function<std::string(const DriverOptions &opts,
                            const db::json::Value &config,
                            const std::string &generated_source)>
      rewrite_source;
};

// --- result -----------------------------------------------------------------
struct DriverResult {
  DriverStatus status = DriverStatus::kInternalError;
  std::string message;             // human summary / the "no viable config"
                                   // reason on failure
  std::string evidence_json;       // full JSON evidence (the CLI --json body)
  std::string cache_key;           // Triton-style key: arch:family:shape_hash
  std::string shape_hash;          // db workload key

  int variants_total = 0;
  int prescreen_rejected = 0;      // rejected by the GPU-free pre-screen
  int compile_failed = 0;          // hipcc/nvcc refused the generated source
  int golden_failed = 0;           // passed compile but failed the golden gate
  int bench_failed = 0;            // passed the gate but the bench errored
  int benchmarked = 0;             // measured (fresh or replayed from cache)
  int cached = 0;                  // skipped because the tunecache had it
  int compiled = 0;                // sources actually compiled this run

  // Selection. has_best == true iff at least one variant has timings.
  bool has_best = false;
  db::json::Value best_config;     // null when !has_best
  double best_median_ms = 0.0;
  double best_p20_ms = 0.0;
  double best_p80_ms = 0.0;
  // Every measured median (fresh + replayed), in enumeration order — the
  // acceptance assertion "best == min of measured" reads this directly.
  std::vector<double> measured_medians;
};

// --- keys -------------------------------------------------------------------
// Deterministic workload hash (FNV-1a 64 over "family:shape"). The format is
// stable and self-describing; libs/db shape_hash is an opaque workload key,
// so any deterministic string works (tools/bench used "size=N").
std::string make_shape_hash(const std::string &kernel_family,
                            const std::vector<std::int64_t> &shape);

// Triton-style cache key: "gpu_arch:kernel_family:shape_hash" (brief §1:
// re-tune when the keyed args change).
std::string make_cache_key(const std::string &arch,
                           const std::string &kernel_family,
                           const std::string &shape_hash);

// --- entry point ------------------------------------------------------------
// Run the full pipeline for one (family, shape, arch) over the family's
// config grid. Returns the result + JSON evidence. Never throws (all
// subprocess/IO/bench failures are captured in the result).
DriverResult tune(const DriverOptions &opts, const DriverHooks *hooks = nullptr);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_DRIVER_H
