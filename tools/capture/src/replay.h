//===- replay.h - live replay + validation + timing hooks (todo 48) -------*- C++ -*-===//
//
// kernscope capture/replay track.
//
// The todo-48 replay engine: restores a captured launch from a bundle (v1
// *.kcap/ or v0 text) and EXECUTES it live on the local device through the
// libs/gpuruntime abstraction:
//
//   1. RESTORE (docs/capture-mapping.md §4.4): the snapshot is restored at
//      its ORIGINAL VAs where the platform allows, else remapped to fresh
//      allocations with the embedded-pointer patch table applied. Platform
//      policy: HIP/HSA assigns VAs (no user-facing placement API at the
//      hipMalloc level) -> ALWAYS remap; CUDA VMM (cuMemAddressReserve +
//      cuMemMap) can place allocations at chosen addresses in principle and
//      is the documented extension point for the rental-gated NVIDIA path
//      (the gpuruntime abstraction does not expose VMM, so this engine
//      remaps and documents the policy — see docs/capture.md §replay).
//      Pointer ARG slots go through the same remap table.
//   2. LAUNCH with the captured grid/block/shared and the remapped args.
//   3. VALIDATE the captured outputs: BITWISE by default (byte compare +
//      sha256 provenance), opt-in TOLERANCE mode (cosine/max_rel_err/pcc
//      per the golden contract, tolerance.h). References come from v1
//      output snapshots in the bundle, or --ref files for v0 bundles.
//   4. TIMING HOOKS: every launch produces a timestamped launch record
//      (steady_clock us); `bench` runs the launch as a libs/bench
//      measurement (N reps, L2 flush, median) and writes a libs/db record
//      with measured provenance. The launch-record contract feeds todo 49's
//      profiler — documented in docs/capture.md §profiling-hooks.
//
// MUST NOT auto-upload bundles anywhere: this engine is purely local.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CAPTURE_REPLAY_H
#define KERNSCOPE_CAPTURE_REPLAY_H

#include "tolerance.h"

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope {

//===--- shared options --------------------------------------------------------===//

struct CaptureBundleOptions {
  std::string bundle_path;
  std::string blob_dir;       // v0 sidecar blobs (default: bundle's dir)
  std::string vendor, arch;   // REQUIRED for v0 (v1: manifest wins)
  std::string kernel;         // "" = auto-select the first non-internal launch
};

//===--- replay ----------------------------------------------------------------===//

struct ReplayOptions : CaptureBundleOptions {
  bool tolerance = false;          // opt-in tolerance mode (default: bitwise)
  std::vector<std::string> ref_files; // reference outputs (v0; overrides v1 blobs)
  std::string dump_dir;            // dump replayed outputs ("" = none)
  std::string profile_out;         // JSONL of launch records ("" = none)
};

/// One compared output buffer: replayed bytes vs the captured reference.
struct OutputComparison {
  std::string buffer_id;
  std::string ref_source;   // "bundle blob <id> (<file>)" or "ref file <path>"
  std::string ref_sha;      // sha256 of the reference bytes
  std::string replay_sha;   // sha256 of the replayed bytes
  bool equal = false;       // bitwise equality
  ToleranceMetrics tolerance;
  bool tolerance_pass = false;
};

/// One timestamped launch record — the todo-49 profiling hook contract
/// (docs/capture.md §profiling-hooks): {kernel, seq, launched_at_us,
/// completed_at_us, duration_us} on the steady clock.
struct LaunchTimingRecord {
  std::string kernel_name;
  std::uint64_t seq = 0;
  std::int64_t launched_at_us = 0;
  std::int64_t completed_at_us = 0;
  double duration_us = 0.0;
};

struct ReplayResult {
  bool validation_pass = false; // false = FAIL verdict (exit code 1)
  bool tolerance = false;
  std::string kernel_name;
  std::string vendor, arch;
  std::string device_name;
  bool has_reference = true;    // false = replay ran but nothing to compare
  std::vector<OutputComparison> outputs;
  std::vector<LaunchTimingRecord> launches;
};

/// Runs the replay + validation. Returns false + `error` only when the replay
/// could NOT run (usage/env/load failures -> exit code 2). A replay that ran
/// but failed validation returns true with validation_pass=false (exit 1).
bool run_replay(const ReplayOptions &opts, ReplayResult &result,
                std::string *error);

//===--- bench (replay-as-benchmark) -------------------------------------------===//

struct BenchOptions : CaptureBundleOptions {
  std::vector<std::string> ref_files; // reference outputs (as replay)
  int reps = 10;                // timed samples (median over these)
  bool l2_flush = true;         // L2 flush between reps (libs/bench)
  bool validate = true;         // one validation launch before timing
  std::string db_path;          // JSONL db file (created/merged)
  std::string profile_out;      // JSONL of launch records ("" = none)
};

struct BenchResult {
  bool validation_pass = false;
  double median_ms = 0.0;
  double p20_ms = 0.0;
  double p80_ms = 0.0;
  int n_reps = 0;
  std::string shape_hash;   // db key (workload hash)
  std::string gpu_arch;
};

/// Benchmarks the captured launch (N reps, L2 flush, median) after an
/// optional validation pass, and writes ONE libs/db record with measured
/// provenance (mirroring the tools/bench record style, todo 42). Returns
/// false + `error` on any failure.
bool run_replay_bench(const BenchOptions &opts, BenchResult &result,
                      std::string *error);

} // namespace kernscope

#endif // KERNSCOPE_CAPTURE_REPLAY_H
