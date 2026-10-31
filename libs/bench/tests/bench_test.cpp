//===- bench_test.cpp - bench harness unit tests (ctest LABELS no-gpu) ----===//
//
// kernscope libs/bench harness suite (todo 22). Runs against the MockRuntime
// and a test-local StubRuntime (controllable event timer) — NO device needed,
// so the suite is labeled no-gpu. Coverage (validated methodology of
// docs/report/kernscope-technical-report.pdf):
//   * quantile helpers hand-computed + p20 <= median <= p80 ordering;
//   * timing structure on the mock: median > 0, quantile ordering, adaptive
//     convergence (nvbench measure_cold);
//   * L2 flush: buffer sized from the device L2 attr (live path) with the
//     Triton 256 MiB fallback on attr-less backends (mock), one memset per
//     timed rep;
//   * adaptive trials: additional sampling until noise convergence, and the
//     hard-timeout path (never spins forever);
//   * failure injection: a kernel that fails/throws mid-rep -> the harness
//     records the typed error and returns immediately (no-hang contract via
//     the adaptive-trials short-circuit).
//
//===----------------------------------------------------------------------===//

#include "kernscope/bench.h"

#include "kernscope/gpu.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kernscope {
namespace {

using kernscope::bench::Options;
using kernscope::bench::Result;
using kernscope::bench::device_l2_cache_bytes;
using kernscope::bench::median;
using kernscope::bench::quantile;
using kernscope::bench::run;

//------------------------------------------------------------------------------
// StubRuntime: a mock-like GpuRuntime whose event timer is test-controlled,
// so adaptive-trials behavior (convergence, timeout) is deterministic.
//------------------------------------------------------------------------------

class StubEvent : public Event {
public:
  explicit StubEvent(std::function<double()> timer) : timer_(std::move(timer)) {}
  GpuStatus record() override { return {}; }
  GpuStatus synchronize() override { return {}; }
  GpuResult<double> elapsed_ms(const Event & /*other*/) const override {
    return GpuResult<double>{timer_()};
  }

private:
  std::function<double()> timer_;
};

class StubRuntime : public GpuRuntime {
public:
  std::string_view backend_name() const noexcept override { return "stub"; }

  GpuResult<std::vector<DeviceInfo>> list_devices() override {
    DeviceInfo dev;
    dev.name = "stub-gpu-0";
    dev.arch = "gfx1101";
    dev.memory_bytes = std::uint64_t{256} << 20;
    dev.vendor = "stub";
    return GpuResult<std::vector<DeviceInfo>>{{dev}};
  }

  GpuResult<DeviceProperties> device_properties(const std::string &name) override {
    if (name != "stub-gpu-0")
      return GpuResult<DeviceProperties>{GpuError::kDeviceNotFound,
                                         "stub: unknown device"};
    DeviceProperties props;
    props.name = name;
    props.gcn_arch = "gfx1101";
    props.gcn_arch_full = "gfx1101:sramecc+:xnack-";
    props.memory_bytes = std::uint64_t{256} << 20;
    props.vendor = "stub";
    return GpuResult<DeviceProperties>{std::move(props)};
  }

  GpuResult<std::unique_ptr<Module>> load_module(const std::string & /*path*/) override {
    return GpuResult<std::unique_ptr<Module>>{
        GpuError::kUnsupportedBackend, "stub: no modules"};
  }
  GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module & /*module*/, const std::string & /*name*/) override {
    return GpuResult<std::unique_ptr<Kernel>>{
        GpuError::kUnsupportedBackend, "stub: no kernels"};
  }

  GpuResult<DevicePtr> device_malloc(std::size_t bytes) override {
    (void)bytes;
    return GpuResult<DevicePtr>{DevicePtr{next_handle_++}};
  }
  GpuStatus device_free(DevicePtr /*ptr*/) override { return {}; }
  GpuStatus memcpy_htod(DevicePtr /*dst*/, const void * /*src*/, std::size_t /*bytes*/) override { return {}; }
  GpuStatus memcpy_dtoh(void * /*dst*/, DevicePtr /*src*/, std::size_t /*bytes*/) override { return {}; }
  GpuStatus memset_device(DevicePtr /*dst*/, std::uint8_t /*value*/, std::size_t /*bytes*/) override { return {}; }

  GpuResult<std::unique_ptr<Event>> create_event() override {
    return GpuResult<std::unique_ptr<Event>>{std::make_unique<StubEvent>(timer)};
  }
  GpuStatus synchronize() override { return {}; }

  // Test hook: per-rep event elapsed time (ms). Counted calls are used by the
  // convergence ramp tests.
  std::function<double()> timer = []() { return 0.5; };
  std::uint64_t next_handle_ = 1;
};

//------------------------------------------------------------------------------
// Quantile helpers (hand-computed; Triton _summarize_statistics style).
//------------------------------------------------------------------------------

TEST(Bench, QuantileMedianHandComputed) {
  const std::vector<double> s{1.0, 2.0, 3.0, 4.0, 5.0};
  // pos = q*(n-1): p20 -> 0.8 -> 1*0.2+2*0.8 = 1.8; median -> 3; p80 -> 4.2.
  EXPECT_DOUBLE_EQ(quantile(s, 0.2), 1.8);
  EXPECT_DOUBLE_EQ(median(s), 3.0);
  EXPECT_DOUBLE_EQ(quantile(s, 0.8), 4.2);
  EXPECT_LE(quantile(s, 0.2), median(s));
  EXPECT_LE(median(s), quantile(s, 0.8));

  EXPECT_DOUBLE_EQ(quantile({}, 0.5), 0.0);  // empty -> 0
  EXPECT_DOUBLE_EQ(quantile(s, 0.0), 1.0);
  EXPECT_DOUBLE_EQ(quantile(s, 1.0), 5.0);

  // Ordering must hold on an even-sized sample too (p20 <= median <= p80).
  const std::vector<double> even{1.0, 2.0, 3.0, 4.0};
  EXPECT_LE(quantile(even, 0.2), median(even));
  EXPECT_LE(median(even), quantile(even, 0.8));
}

//------------------------------------------------------------------------------
// Timing structure + adaptive convergence on the mock (fixed 0.5 ms events).
//------------------------------------------------------------------------------

TEST(Bench, MockTimingStructureAndQuantileOrdering) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok()) << rt.message;

  int launches = 0;
  Options opts;
  opts.min_samples = 10;
  opts.min_time_ms = 0.0;
  opts.max_noise = 0.05;
  // The mock really memsets the flush buffer (byte-exact contract), so keep
  // it small in unit tests; the live suite asserts the real device-attr size.
  opts.l2_flush_bytes = std::uint64_t{1} << 10;

  const Result r = run(*rt.value, "mock-gpu-0", [&]() {
    ++launches;
    return GpuStatus{};
  }, opts);

  ASSERT_EQ(r.error, GpuError::kOk) << r.error_message;
  EXPECT_GT(r.median_ms, 0.0);  // mock events report 0.5 ms -> median > 0
  EXPECT_LE(r.p20_ms, r.median_ms);
  EXPECT_LE(r.median_ms, r.p80_ms);
  EXPECT_GE(r.n_reps, opts.min_samples);
  EXPECT_TRUE(r.converged);      // fixed timing -> zero noise -> converged
  EXPECT_FALSE(r.timed_out);
  EXPECT_DOUBLE_EQ(r.relative_stdev, 0.0);
  // warmup + calibration + timed reps all launched through the harness.
  EXPECT_GE(launches, opts.min_samples + 2);
}

//------------------------------------------------------------------------------
// L2 flush: option sizing, mock 256 MiB fallback, one memset per timed rep.
//------------------------------------------------------------------------------

TEST(Bench, L2FlushBufferSizedFromOptionOneMemsetPerRep) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok()) << rt.message;
  auto &mock = static_cast<MockRuntime &>(*rt.value);

  constexpr std::uint64_t kFlush = std::uint64_t{1} << 20;  // 1 MiB
  Options opts;
  opts.l2_flush_bytes = kFlush;
  opts.min_samples = 5;
  opts.min_time_ms = 0.0;

  const Result r = run(*rt.value, "mock-gpu-0", []() { return GpuStatus{}; }, opts);
  ASSERT_EQ(r.error, GpuError::kOk) << r.error_message;
  EXPECT_EQ(r.l2_flush_bytes_used, kFlush);

  // Calibration reps are L2-flushed too (Triton estimates over flushed runs):
  // exactly calibration_runs + n_reps memset ops, each of the flush size.
  std::size_t memsets = 0;
  for (const auto &op : mock.ops()) {
    if (op.type == MockRuntime::Op::Type::kMemset) {
      ++memsets;
      EXPECT_EQ(op.size, kFlush);
    }
  }
  EXPECT_EQ(memsets, static_cast<std::size_t>(opts.calibration_runs + r.n_reps));
}

TEST(Bench, L2FlushFallbackIsTriton256MiBWhenNoDeviceAttr) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok()) << rt.message;
  // The mock reports no L2 attribute -> device_l2_cache_bytes is 0.
  EXPECT_EQ(device_l2_cache_bytes(*rt.value, "mock-gpu-0"), 0u);

  Options opts;
  opts.min_samples = 3;
  opts.min_time_ms = 0.0;
  // Shrink the rep budget so the 256 MiB fallback flush (real host memset on
  // the mock) is touched only ~10 times; the assertion is on the buffer size.
  opts.rep_ms = 5.0;
  const Result r = run(*rt.value, "mock-gpu-0", []() { return GpuStatus{}; }, opts);
  ASSERT_EQ(r.error, GpuError::kOk) << r.error_message;
  // Triton's 256 MiB zero-fill buffer is the documented fallback
  // (third_party/nvidia/backend/driver.py L393-L407).
  EXPECT_EQ(r.l2_flush_bytes_used, std::uint64_t{256} << 20);
}

//------------------------------------------------------------------------------
// Adaptive trials: convergence via additional sampling + the timeout path.
//------------------------------------------------------------------------------

TEST(Bench, AdaptiveTrialsConvergeViaAdditionalSamples) {
  StubRuntime rt;
  // Clock-ramp noise: early samples are slow and jittery, amplitude decays
  // exponentially, so the sample relative stdev only drops below max_noise
  // once enough samples dilute the ramp (nvbench measure_cold convergence).
  std::uint64_t calls = 0;
  rt.timer = [&calls]() {
    ++calls;
    const double c = static_cast<double>(calls);
    return 1.0 + (0.8 * std::exp(-c / 10.0) * std::sin(c));
  };

  int launches = 0;
  Options opts;
  opts.min_samples = 20;
  opts.min_time_ms = 0.0;
  opts.max_noise = 0.05;
  opts.timeout_ms = 10000.0;

  const Result r = run(rt, "stub-gpu-0", [&]() {
    ++launches;
    return GpuStatus{};
  }, opts);

  ASSERT_EQ(r.error, GpuError::kOk) << r.error_message;
  EXPECT_TRUE(r.converged);
  EXPECT_FALSE(r.timed_out);
  EXPECT_GE(r.n_reps, opts.min_samples);
  EXPECT_LE(r.relative_stdev, opts.max_noise);
  // The first time-budget batch (~rep_ms / estimate) alone did NOT converge;
  // the adaptive loop added samples until the noise threshold was met.
  EXPECT_GT(r.n_reps, 1);
}

TEST(Bench, AdaptiveTrialsRespectTimeout) {
  StubRuntime rt;
  // High-dispersion, never-converging timing (rel stdev ~0.5 >> 0.01): only
  // the hard timeout can stop the loop.
  std::uint64_t calls = 0;
  rt.timer = [&calls]() {
    ++calls;
    const double c = static_cast<double>(calls);
    return 1.0 + (0.9 * std::sin(c));
  };

  Options opts;
  opts.min_samples = 100000;  // unreachable
  opts.min_time_ms = 1e9;      // unreachable -> min_time is only a floor
  opts.max_noise = 0.01;       // unreachable with rel stdev ~0.6
  opts.timeout_ms = 250.0;

  const auto t0 = std::chrono::steady_clock::now();
  const Result r = run(rt, "stub-gpu-0", []() { return GpuStatus{}; }, opts);
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t0).count();

  ASSERT_EQ(r.error, GpuError::kOk) << r.error_message;
  EXPECT_TRUE(r.timed_out);
  EXPECT_FALSE(r.converged);
  EXPECT_GT(r.n_reps, 0);
  // The timeout stopped the loop: the run returned promptly with results, it
  // did not spin (no-hang contract of the adaptive-trials timeout path).
  EXPECT_LT(elapsed_ms, 5000);
  EXPECT_GE(elapsed_ms, 200);  // it really did spend ~timeout_ms wall time
}

//------------------------------------------------------------------------------
// Failure injection: a kernel that fails/throws mid-rep records the typed
// error and the harness returns immediately (never hangs).
//------------------------------------------------------------------------------

TEST(Bench, FailingKernelMidRepRecordsErrorAndStops) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok()) << rt.message;

  int launches = 0;
  // Calibration (1 warm + 5 reps) then warmup consume launches 1..56 with the
  // default budgets (estimate 0.5 ms -> n_warmup 50); fail on launch 60, i.e.
  // the 4th timed rep.
  constexpr int kFailAfter = 59;
  Options opts;
  opts.l2_flush_bytes = std::uint64_t{1} << 10;  // keep mock flushes cheap
  const Result r = run(*rt.value, "mock-gpu-0", [&]() {
    ++launches;
    if (launches > kFailAfter)
      return GpuStatus{GpuError::kLaunchFailed, "injected: kernel threw mid-rep"};
    return GpuStatus{};
  }, opts);

  EXPECT_EQ(r.error, GpuError::kLaunchFailed);
  EXPECT_NE(r.error_message.find("timed rep"), std::string::npos);
  EXPECT_NE(r.error_message.find("injected"), std::string::npos);
  // Short-circuit: sampling stopped at the failing rep, it did NOT keep going
  // to min_samples (default 10) or the default 3 s timeout.
  EXPECT_EQ(launches, kFailAfter + 1);
  EXPECT_EQ(r.n_reps, 0);  // no timings reported for a failed run
}

TEST(Bench, ThrowingKernelIsCapturedAsTypedError) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok()) << rt.message;

  // The launch throws on the first timed rep (after calibration + warmup);
  // the harness must catch it and convert it to a typed launch error.
  int launches = 0;
  Options opts;
  opts.l2_flush_bytes = std::uint64_t{1} << 10;  // keep mock flushes cheap
  const Result r = run(*rt.value, "mock-gpu-0", [&]() -> GpuStatus {
    ++launches;
    if (launches > 56)
      throw std::runtime_error("boom");
    return GpuStatus{};
  }, opts);

  EXPECT_EQ(r.error, GpuError::kLaunchFailed);
  EXPECT_NE(r.error_message.find("threw"), std::string::npos);
  EXPECT_NE(r.error_message.find("boom"), std::string::npos);
}

} // namespace
} // namespace kernscope
