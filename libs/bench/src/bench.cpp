//===- bench.cpp - nvbench/Triton-style benchmarking harness ------*- C++ -*-===//
//
// Implementation of kernscope::bench (docs/report/kernscope-technical-report.pdf
// §6 methodology, cited in bench.h). The L2-attribute query reaches the live
// HIP device directly (hipDeviceGetAttribute / hipDeviceAttributeL2CacheSize),
// guarded by the same portable #ifdef convention libs/gpuruntime uses; on
// builds without ROCm headers the helper degrades to 0 and the harness falls
// back to Triton's 256 MiB flush buffer.
//
//===----------------------------------------------------------------------===//

#include "kernscope/bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <limits>
#include <numeric>
#include <string>

//===--- backend-gated L2 attribute query -----------------------------------===//
// The ROCm host header is included HERE ONLY (never in bench.h), so the
// GPU-free gtest and CUDA builds never need it on the include path.
#if defined(__HIP_PLATFORM_AMD__) || defined(KERNSCOPE_HIP_BACKEND_BUILD)
#define KERNSCOPE_BENCH_HAVE_HIP 1
#include <hip/hip_runtime_api.h>
#endif

namespace kernscope::bench {

//===--- statistics ----------------------------------------------------------===//

double quantile(std::vector<double> samples, double q) {
  if (samples.empty())
    return 0.0;
  std::sort(samples.begin(), samples.end());
  if (samples.size() == 1 || q <= 0.0)
    return samples.front();
  if (q >= 1.0)
    return samples.back();
  const double pos = q * static_cast<double>(samples.size() - 1);
  const std::size_t lo = static_cast<std::size_t>(pos);
  const std::size_t hi = std::min(lo + 1, samples.size() - 1);
  const double frac = pos - static_cast<double>(lo);
  return (samples[lo] * (1.0 - frac)) + (samples[hi] * frac);
}

double median(std::vector<double> samples) {
  return quantile(std::move(samples), 0.5);
}

namespace {

// Mean and relative standard deviation (stdev/mean) over samples.
struct Stats {
  double mean = 0.0;
  double rel_stdev = 0.0;
};
Stats Summarize(const std::vector<double> &samples) {
  Stats out;
  if (samples.empty())
    return out;
  const double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
  out.mean = sum / static_cast<double>(samples.size());
  if (samples.size() >= 2 && out.mean > 0.0) {
    double sq = 0.0;
    for (const double v : samples) {
      const double d = v - out.mean;
      sq += d * d;
    }
    const double variance = sq / static_cast<double>(samples.size() - 1);
    out.rel_stdev = std::sqrt(variance) / out.mean;
  }
  return out;
}

double NowMs() {
  return static_cast<double>(
             std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::steady_clock::now().time_since_epoch())
                 .count()) /
         1e6;
}

} // namespace

//===--- device L2 cache size ------------------------------------------------===//

std::uint64_t device_l2_cache_bytes(GpuRuntime &rt, const std::string &device_name) {
#ifdef KERNSCOPE_BENCH_HAVE_HIP
  // Only the live HIP backend can answer this query; the mock/compile-only
  // runtimes have no L2 attribute (and the real hipDeviceGetAttribute must
  // never be consulted for a non-HIP runtime).
  if (rt.backend_name() != "hip")
    return 0;
  auto devs = rt.list_devices();
  if (!devs.ok())
    return 0;
  for (std::size_t i = 0; i < devs.value.size(); ++i) {
    if (devs.value[i].name == device_name) {
      int bytes = 0;
      if (hipDeviceGetAttribute(&bytes, hipDeviceAttributeL2CacheSize,
                                static_cast<int>(i)) == hipSuccess &&
          bytes > 0) {
        return static_cast<std::uint64_t>(bytes);
      }
      return 0; // live device, but the attr query failed or reported 0
    }
  }
  return 0; // device name not found in list_devices()
#else
  (void)rt;
  (void)device_name;
  return 0;
#endif
}

//===--- run ----------------------------------------------------------------===//
// Pipeline (all steps cited in bench.h):
//   1. allocate the L2 flush buffer (device attr, else Triton 256 MiB);
//   2. calibration: one warm launch + sync, then `calibration_runs` L2-flushed
//      timed runs -> per-call cost estimate;
//   3. n_warmup = warmup_ms/estimate, n_repeat = rep_ms/estimate (ceil);
//   4. warmup loop (time-budget-derived, Triton convention);
//   5. timed reps: {flush L2; start event; launch; stop event} per rep;
//   6. adaptive trials: keep sampling batches until convergence
//      (rel_stdev <= max_noise and n_reps >= min_samples), min_time, or
//      timeout (nvbench measure_cold).
// A failed launch/throw short-circuits everything: Result carries the typed
// error and the harness returns (no-hang contract).
Result run(GpuRuntime &rt, const std::string &device_name, const LaunchFn &launch,
           const Options &opts) {
  Result out;

  // --- 1. L2 flush buffer ------------------------------------------------
  out.l2_flush_bytes_used = opts.l2_flush_bytes;
  if (out.l2_flush_bytes_used == 0)
    out.l2_flush_bytes_used = device_l2_cache_bytes(rt, device_name);
  if (out.l2_flush_bytes_used == 0)
    out.l2_flush_bytes_used = std::uint64_t{256} << 20; // Triton 256 MiB

  DevicePtr flush_buf{};
  auto release = [&]() {
    if (flush_buf.value != 0)
      (void)rt.device_free(flush_buf);
  };
  if (opts.l2_flush) {
    auto buf = rt.device_malloc(out.l2_flush_bytes_used);
    if (!buf.ok()) {
      out.error = buf.err;
      out.error_message = "L2 flush buffer alloc (" +
                          std::to_string(out.l2_flush_bytes_used) +
                          " bytes): " + buf.message;
      return out;
    }
    flush_buf = buf.value;
  }

  // One timed rep. Returns false on any failure (typed error + message in
  // `err`/`err_msg`). Never throws.
  auto run_rep = [&](double &ms, GpuError &err, std::string &err_msg) -> bool {
    if (opts.l2_flush) {
      GpuStatus s = rt.memset_device(flush_buf, 0, out.l2_flush_bytes_used);
      if (!s.ok()) {
        err = s.err;
        err_msg = "L2 flush memset: " + s.message;
        return false;
      }
    }
    auto e0 = Event::create(rt);
    auto e1 = Event::create(rt);
    if (!e0.ok() || !e1.ok()) {
      err = e0.ok() ? e1.err : e0.err;
      err_msg = (e0.ok() ? e1 : e0).message;
      return false;
    }
    GpuStatus s = e0.value->record();
    if (!s.ok()) {
      err = s.err;
      err_msg = "start event record: " + s.message;
      return false;
    }
    try {
      s = launch();
    } catch (const std::exception &e) {
      err = GpuError::kLaunchFailed;
      err_msg = std::string("kernel launch threw: ") + e.what();
      return false;
    } catch (...) {
      err = GpuError::kLaunchFailed;
      err_msg = "kernel launch threw an unknown exception";
      return false;
    }
    if (!s.ok()) {
      err = s.err;
      err_msg = s.message;
      return false;
    }
    s = e1.value->record();
    if (!s.ok()) {
      err = s.err;
      err_msg = "stop event record: " + s.message;
      return false;
    }
    s = e1.value->synchronize();
    if (!s.ok()) {
      err = s.err;
      err_msg = "stop event synchronize: " + s.message;
      return false;
    }
    auto elapsed = e0.value->elapsed_ms(*e1.value);
    if (!elapsed.ok()) {
      err = elapsed.err;
      err_msg = elapsed.message;
      return false;
    }
    ms = elapsed.value;
    return true;
  };

  // --- 2. calibration (Triton do_bench: fn once + sync, then estimate over
  // a few L2-flushed runs) -------------------------------------------------
  {
    GpuStatus s = launch();
    if (!s.ok()) {
      out.error = s.err;
      out.error_message = "calibration warm launch: " + s.message;
      release();
      return out;
    }
    s = rt.synchronize();
    if (!s.ok()) {
      out.error = s.err;
      out.error_message = "calibration synchronize: " + s.message;
      release();
      return out;
    }
  }
  std::vector<double> calib;
  for (int i = 0; i < opts.calibration_runs; ++i) {
    double ms = 0.0;
    GpuError err = GpuError::kOk;
    std::string err_msg;
    if (!run_rep(ms, err, err_msg)) {
      out.error = err;
      out.error_message = "calibration rep: " + err_msg;
      release();
      return out;
    }
    calib.push_back(ms);
  }
  double estimate_ms = *std::min_element(calib.begin(), calib.end());
  if (estimate_ms <= 0.0)
    estimate_ms = 0.001; // degenerate backend timing; keep the loop finite

  // --- 3. time-budget iteration counts (Triton) --------------------------
  out.n_warmup = static_cast<int>(
      std::max(1.0, std::ceil(opts.warmup_ms / estimate_ms)));
  int n_repeat = static_cast<int>(
      std::max(1.0, std::ceil(opts.rep_ms / estimate_ms)));

  // --- 4. warmup (time-budget-derived) ------------------------------------
  for (int i = 0; i < out.n_warmup; ++i) {
    GpuStatus s = launch();
    if (!s.ok()) {
      out.error = s.err;
      out.error_message = "warmup launch: " + s.message;
      release();
      return out;
    }
  }
  {
    GpuStatus s = rt.synchronize();
    if (!s.ok()) {
      out.error = s.err;
      out.error_message = "warmup synchronize: " + s.message;
      release();
      return out;
    }
  }

  // --- 5+6. timed reps + adaptive trials (nvbench measure_cold) ----------
  const std::uint64_t kMaxSamples = 1 << 20; // hard cap; never spins forever
  std::vector<double> samples;
  samples.reserve(n_repeat);
  const double start_wall = NowMs();
  double timed_ms = 0.0;
  int batch = n_repeat;
  bool stop = false;
  while (!stop) {
    for (int i = 0; i < batch && !stop; ++i) {
      double ms = 0.0;
      GpuError err = GpuError::kOk;
      std::string err_msg;
      if (!run_rep(ms, err, err_msg)) {
        // No-hang contract: a failing rep short-circuits the adaptive loop.
        out.error = err;
        out.error_message = "timed rep: " + err_msg;
        stop = true;
        break;
      }
      samples.push_back(ms);
      timed_ms += ms;
      if (samples.size() >= kMaxSamples) {
        out.error_message = "benchmark hit the max-sample cap (" +
                            std::to_string(kMaxSamples) + ")";
        stop = true;
        break;
      }
    }
    if (stop)
      break;

    const Stats stats = Summarize(samples);
    // nvbench measure_cold convergence: stop only when the relative-stdev
    // noise target is met with at least min_samples over at least min_time_ms
    // of timed work — min_time is a FLOOR for a converged stop, not an
    // early-exit. A never-converging (noisy) benchmark stops on timeout.
    const bool converged = stats.rel_stdev <= opts.max_noise &&
                           samples.size() >= static_cast<std::size_t>(opts.min_samples) &&
                           timed_ms >= opts.min_time_ms;
    const double wall_elapsed = NowMs() - start_wall;
    const bool timed_out = opts.timeout_ms > 0.0 && wall_elapsed >= opts.timeout_ms;

    if (converged || timed_out) {
      out.converged = converged;
      out.timed_out = timed_out;
      break;  // dead-store note: `stop` is not read again before this break
    }
    // Not converged yet: keep sampling in batches of n_repeat (bounded by
    // the remaining timeout so we never overshoot a hard deadline).
    batch = n_repeat;
  }

  // --- summary (Triton _summarize_statistics) ----------------------------
  if (out.error == GpuError::kOk && !samples.empty()) {
    const Stats stats = Summarize(samples);
    out.relative_stdev = stats.rel_stdev;
    out.mean_ms = stats.mean;
    out.min_ms = *std::min_element(samples.begin(), samples.end());
    out.median_ms = median(samples);
    out.p20_ms = quantile(samples, 0.2);
    out.p80_ms = quantile(samples, 0.8);
    out.n_reps = static_cast<int>(samples.size());
  }
  release();
  return out;
}

} // namespace kernscope::bench
