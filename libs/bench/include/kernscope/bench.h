//===- bench.h - nvbench/Triton-style benchmarking harness ------*- C++ -*-===//
//
// kernscope benchmarking harness (wave-4 todo 22), implementing the validated
// methodology of docs/report/kernscope-technical-report.pdf (all cited):
//   * TIME-BASED warmup/rep budgets (Triton do_bench convention: warmup 25 ms,
//     rep 100 ms — NOT iteration counts). Per-call cost is estimated over a
//     few L2-flushed calibration runs and n_warmup / n_repeat are derived
//     from the budgets (triton/testing.py L288-L358).
//   * L2 flush between reps: a buffer sized from the device's L2 cache
//     attribute (hipDeviceAttributeL2CacheSize) is zeroed (memset) before
//     every timed rep — the nvbench l2flush pattern (l2flush.cuh L28-L63);
//     Triton's zero-fill buffer is 256 MiB, our fallback when the attr is
//     unavailable (third_party/nvidia/backend/driver.py L393-L407).
//   * HIP/CUDA event timing per rep (event pair around each launch;
//     cudaEventRecord/ElapsedTime per sample, nvbench cuda_timer.cuh).
//   * Adaptive trials: keep sampling until the relative-stdev noise drops
//     below max_noise with at least min_samples (nvbench measure_cold
//     convergence), bounded by min_time / timeout stopping criteria
//     (measure_cold.cuh L100-L117 + state.cuh knobs L155-L234).
//   * Summary = median + p20/p80 quantiles (Triton quantiles=(0.5,0.2,0.8),
//     _summarize_statistics).
//
// Results are consumed by tools/bench (the kernscope-bench CLI) which writes
// libs/db records with provenance (device name, gcnArchName, rocm-smi clocks
// as-measured, tool version, measured=true). The harness itself never locks
// clocks (no root) — clocks are recorded as-measured in provenance.
//
// No-hang contract: a kernel that fails (typed GpuStatus error) or throws
// mid-rep is captured in Result (error status + message) and sampling stops
// immediately — the adaptive loop never spins to its timeout waiting for a
// dead measurement.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_BENCH_H
#define KERNSCOPE_BENCH_H

#include "kernscope/gpu.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace kernscope::bench {

// Timing + convergence knobs (documented above; defaults follow the Triton
// do_bench budgets and nvbench measure_cold defaults).
struct Options {
  // Time budgets (ms), NOT iteration counts (Triton do_bench convention).
  double warmup_ms = 25.0;
  double rep_ms = 100.0;

  // --- adaptive trials (nvbench measure_cold) ---
  // Keep sampling until relative_stdev <= max_noise with n_reps >=
  // min_samples; stop earlier when min_time_ms of timed work is reached and
  // later no later than timeout_ms of wall clock (0 = disabled).
  int min_samples = 10;
  double min_time_ms = 50.0;
  double max_noise = 0.05;   // relative stdev target (5%)
  double timeout_ms = 3000.0;

  // --- L2 flush (nvbench l2flush pattern) ---
  bool l2_flush = true;
  // 0 = size the flush buffer from the device L2 attribute
  // (device_l2_cache_bytes); when that is unavailable too, fall back to
  // Triton's 256 MiB zero-fill buffer.
  std::uint64_t l2_flush_bytes = 0;

  // Per-call cost estimate over N L2-flushed timed runs (Triton's 5-run
  // calibration).
  int calibration_runs = 5;
};

// One benchmark run. `median`/`p20`/`p80` are the Triton summary quantiles
// (ms). On failure `error` is set and every quantile stays 0 — a failing
// measurement NEVER reports zeros as timings.
struct Result {
  double median_ms = 0.0;
  double p20_ms = 0.0;
  double p80_ms = 0.0;
  double mean_ms = 0.0;
  double min_ms = 0.0;
  double relative_stdev = 0.0;  // stdev/mean over reps; 0 when n_reps < 2
  int n_reps = 0;
  int n_warmup = 0;
  std::uint64_t l2_flush_bytes_used = 0;
  GpuError error = GpuError::kOk;
  std::string error_message;  // set iff error != kOk (names the failing op)
  bool converged = false;     // relative_stdev <= max_noise at stop
  bool timed_out = false;     // stopped on the hard wall-clock timeout
};

// One kernel rep: flush L2 (if enabled), record start event, invoke `launch`,
// record stop event. `launch` must enqueue one kernel on the default stream
// (the harness owns event recording + the final synchronize). May be called
// with any runtime that can report event elapsed time (live HIP/CUDA or the
// MockRuntime used by the no-gpu unit suite).
using LaunchFn = std::function<GpuStatus()>;

// --- statistics helpers (Triton _summarize_statistics style) ---------------

// Quantile via linear interpolation over sorted samples (empty -> 0.0).
// q in [0,1]. Samples are copied (callers may pass a by-value vector).
double quantile(std::vector<double> samples, double q);

// median(samples) == quantile(samples, 0.5).
double median(std::vector<double> samples);

// Device L2 cache size in bytes (hipDeviceAttributeL2CacheSize on the HIP
// backend; 0 when the backend cannot report it — mock/compile-only/CUDA-free
// builds). `device_name` is the DeviceInfo::name from list_devices().
std::uint64_t device_l2_cache_bytes(GpuRuntime &rt, const std::string &device_name);

// Run the benchmark. `device_name` selects the device for L2-flush sizing.
// Never throws: launch failures and std::exception from `launch` are captured
// in Result (error status). On error the flush buffer is released.
Result run(GpuRuntime &rt, const std::string &device_name, const LaunchFn &launch,
           const Options &opts = {});

} // namespace kernscope::bench

#endif // KERNSCOPE_BENCH_H
