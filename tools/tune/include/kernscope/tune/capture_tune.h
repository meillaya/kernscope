//===- capture_tune.h - capture->tune replay-based launch-config tuning -----===//
//
// kernscope tools/tune todo 53 (plan "integration: capture->tune pipeline,
// replay-based launch-config tuning", Metis H-3). The DISTINCT tuning mode
// for CAPTURED kernels: a capture bundle carries the kernel binary + args +
// buffers but NO source, so the template tuner (todo 28, generate -> compile
// -> golden gate) cannot consume it. Instead the launch configuration is the
// only searchable degree of freedom:
//
//   kernscope-tune --from-capture <bundle> [--vendor amd] [--arch gfx1101]
//       [--blob-dir <dir>] [--kernel <name>] [--ref <file>...]
//       [--candidates <blocks-csv>] [--inject-config <g,b[,shared]>]
//       [--reps <n>] [--no-flush] [--db <path>] [--profile-out <path>]
//       [--json|--text]
//
// PIPELINE (todo 53; each candidate replayed through the todo-48 replay
// harness machinery — the shared plan builder BuildPlan (tools/capture/src/
// replay_model.{h,cpp}, consumed unchanged) + the libs/gpuruntime runtime +
// libs/bench + libs/db — the harness's OWN building blocks; the captured
// kernel BINARY is never modified, only the LaunchConfig passed to the
// driver launch):
//
//   1. BUILD THE REPLAY PLAN: BuildPlan(bundle, vendor/arch/blob-dir/kernel)
//      -> the target launch (grid/block/shared), the device-buffer plan
//      (inputs with captured content, outputs with their validation
//      references), the kernel binary (hsaco/cubin) + the arg marshalling
//      table. v1 *.kcap/ bundles carry their own refs; v0 text bundles need
//      --ref files in output-buffer order.
//   2. RESTORE + LAUNCH: the binary is written to a temp file, loaded via
//      the gpuruntime module API, the kernel resolved by its descriptor-less
//      symbol, buffers allocated fresh, the remap table applied (pointer arg
//      slots carry the NEW device VAs — docs/capture-mapping.md §4.4).
//   3. CANDIDATES: the captured baseline config is ALWAYS candidate #1
//      (explored first — this guarantees best.speedup >= 1.0), followed by
//      the launch-config grid: block sizes from the arch-valid set at the
//      CAPTURED grid (the grid of a captured kernel is workload-semantic;
//      only the within-block partition is searchable without touching the
//      binary), plus --inject-config for the acceptance's broken-config
//      injection. Every candidate is pre-screened against the documented
//      launch limits BEFORE any replay: per-block threads <= 1024 (HIP
//      maxThreadsPerBlock ceiling, RDNA3/CUDA), grid x <= 2^31-1,
//      grid y/z <= 65535, shared <= the arch's smem-per-block budget
//      (libs/arch). A rejection is recorded with its reason.
//   4. REPLAY + VALIDATE: each surviving candidate is launched with its
//      LaunchConfig, the outputs copied back and compared BITWISE (sha256 +
//      byte compare) against the captured reference. A validation failure
//      DISCARDS the candidate and records it (never benchmarked).
//   5. BENCHMARK: a validated candidate is timed via libs/bench (N reps,
//      L2 flush between reps, median + p20/p80 — todo-22 methodology).
//   6. SELECT: best config by median (ties: p20, then p80, then enumeration
//      order), then persist libs/db records keyed (arch, kernel-name-hash,
//      capture-bundle-hash) — kernel_family = "kernel=<sha256(kernel name)>",
//      shape_hash = "bundle=<sha256(bundle identity)>" (v0: the bundle
//      file bytes; v1: manifest.json) — one record per measured candidate
//      (query_best re-picks by median, Triton configs_timings pattern) and
//      the JSON evidence report (baseline vs best speedup, ALL candidates
//      with status + reason).
//
// BOUNDARY VS TEMPLATE TUNING (documented, todo-53 MUST NOT): this mode
// only ever varies the LAUNCH CONFIG (grid/block/shared) of the captured
// kernel binary — the hsaco/cubin bytes, the arguments and the buffers are
// byte-identical to the capture (kernel_name_hash/bundle_hash in the db key
// prove it). Template tuning (todo 28) instead regenerates SOURCE with
// baked-in tile/warp/smem structure and recompiles; the two modes cannot
// consume each other's inputs. Shared-mem carveout is only explored when
// the captured launch itself used dynamic shared memory (shared_mem_bytes >
// 0); a captured kernel with static smem (e.g. layernorm_fused) keeps its
// captured shared value in every candidate.
//
// Exit codes (CLI): 0 = best config selected (db written); 1 = no viable
// config (every candidate pre-screen rejected / validation-failed / bench
// failed — NO db write, asserted by the QA negative); 2 = usage/environment
// error (bad bundle, no live GPU, bad options).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_CAPTURE_TUNE_H
#define KERNSCOPE_TUNE_CAPTURE_TUNE_H

#include "kernscope/db.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope::tune {

// --- pipeline outcome status -------------------------------------------------
enum class CaptureTuneStatus : std::uint8_t {
  kOk = 0,             // a best config was selected; libs/db records written
  kNoViableConfig,     // every candidate was pre-screen rejected / launch-
                       // failed / validation-failed / bench-failed; NO db
                       // write; `message` carries "no valid config"
  kNoLiveGpu,          // replay needs a live HIP device (not available, or
                       // the arch is NVIDIA = compile-only/absent)
  kUsageError,         // bad options / unloadable bundle / bad ref files
  kInternalError,      // IO or unexpected failure
};

// Stable machine-readable status names ("ok" | "no_viable_config" | ...).
const char *capture_tune_status_name(CaptureTuneStatus status) noexcept;

// --- launch limits (todo-53 pre-screen; documented, arch-independent) -------
// The pre-screen rejects a candidate BEFORE any replay with a reason string
// (the acceptance injects block=2048 -> must be rejected pre-replay).
inline constexpr std::uint32_t kMaxBlockThreads = 1024;   // HIP maxThreadsPerBlock ceiling (RDNA3, matches CUDA 9.0+)
inline constexpr std::uint32_t kMaxGridDimX = 0x7FFFFFFF; // HIP gridDim.x max (INT32_MAX)
inline constexpr std::uint32_t kMaxGridDimYZ = 65535;     // HIP gridDim.y/z max

// --- options ----------------------------------------------------------------
struct CaptureTuneOptions {
  std::string bundle_path;           // v1 *.kcap/ dir OR v0 text bundle
  std::string vendor, arch;          // REQUIRED for v0 (v1: manifest wins)
  std::string blob_dir;              // v0 sidecar blobs (default: bundle's dir)
  std::string kernel;                // "" = auto-select first non-internal launch
  std::vector<std::string> ref_files; // v0 reference outputs, output-buffer
                                      // order (override v1 bundle snapshots)
  // Block sizes explored at the captured grid (default: the arch-valid subset
  // of {128, 256, 512, 1024} UNION the captured block). The captured baseline
  // config is ALWAYS candidate #1 regardless of this set.
  std::vector<int> candidate_blocks;
  // "gx,gy,gz:bx,by,bz[:shared_bytes]" — an EXTRA candidate appended after
  // the block grid (test hook: the acceptance injects a block that exceeds
  // the arch max and asserts the pre-replay rejection).
  std::string inject_config;
  int reps = 10;                     // timed samples per candidate (median)
  bool l2_flush = true;              // L2 flush between reps (libs/bench)
  std::string db_path;               // libs/db JSONL ("" = no db write)
  std::string profile_out;           // launch-record JSONL ("" = none)
};

// --- per-candidate outcome ---------------------------------------------------
// `status`: validated | pre_screen_rejected | launch_failed |
//           validation_failed | bench_failed
// A `validated` candidate carries timings (median/p20/p80 ms); replayed
// candidates carry the bitwise comparison (ref/replay sha256); rejected
// candidates carry `reason`.
struct CaptureTuneCandidate {
  std::uint32_t grid[3] = {1, 1, 1};
  std::uint32_t block[3] = {1, 1, 1};
  std::uint32_t shared_mem_bytes = 0;
  bool is_baseline = false;          // the captured launch config
  std::string status;                // see above
  std::string reason;                // pre-screen / launch / bench failure text
  bool validation_pass = false;
  std::string ref_sha, replay_sha;
  bool benchmarked = false;
  double median_ms = 0.0, p20_ms = 0.0, p80_ms = 0.0;
  double speedup = 0.0;              // vs baseline median (validated only)
};

// --- result -----------------------------------------------------------------
struct CaptureTuneResult {
  CaptureTuneStatus status = CaptureTuneStatus::kInternalError;
  std::string message;               // human summary / "no valid config" text
  std::string evidence_json;         // full JSON evidence (the CLI --json body)
  std::string kernel_name;           // descriptor-stripped symbol used to launch
  std::string vendor, arch, device_name;
  std::string kernel_name_hash;      // "kernel=<sha256 hex>" (db key part)
  std::string bundle_hash;           // "bundle=<sha256 hex>" (db key part)
  std::string db_path;               // written db ("" = none)

  int candidates_total = 0;          // every candidate (incl. pre-screen rejects)
  int pre_screen_rejected = 0;
  int launch_failed = 0;
  int validation_failed = 0;
  int bench_failed = 0;
  int validated = 0;                 // passed validation -> benchmarked

  bool has_best = false;
  db::json::Value best_config;       // null when !has_best
  double best_median_ms = 0.0, best_p20_ms = 0.0, best_p80_ms = 0.0;
  double baseline_median_ms = 0.0;   // candidate #1 (captured config)
  double speedup = 1.0;              // baseline_median / best_median (>= 1.0)
  std::vector<CaptureTuneCandidate> candidates; // enumeration order, baseline first
};

// --- entry point ------------------------------------------------------------
// Runs the capture->tune pipeline for one bundle. Never throws (all
// subprocess/IO/bench failures are captured in the result). Writes libs/db
// records ONLY on success (kOk) — the "no valid config" QA negative asserts
// no db write.
CaptureTuneResult tune_from_capture(const CaptureTuneOptions &opts);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_CAPTURE_TUNE_H
