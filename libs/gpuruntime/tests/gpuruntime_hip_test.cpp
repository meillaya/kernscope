//===- gpuruntime_hip_test.cpp - live HIP tests (ctest LABELS gpu-amd) ----===//
//
// kernscope gpuruntime live-device suite (todo 21). Runs ONLY on machines
// with a live AMD GPU (this machine: RX 7800 XT / gfx1101, rocminfo-verified;
// /dev/kfd must be accessible). The kernel module is built at CMake build
// time by `hipcc --genco --offload-arch=gfx1101` (tests/kernels/add_kernel.hip
// -> add_kernel.hsaco) and loaded through Module::load(path) -> hipModuleLoad,
// so the whole module/kernel/launch/memcpy/event path is exercised live.
//
// If the build had no hipcc (KERNSCOPE_NO_LIVE_HIP), the suite SKIPs instead
// of failing - the ctest gpu-amd label only exists on AMD machines anyway.
//
//===----------------------------------------------------------------------===//

#include "kernscope/gpu.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

#ifndef KERNSCOPE_TEST_HSACO
#error "gpuruntime-hip-test requires KERNSCOPE_TEST_HSACO (libs/gpuruntime/CMakeLists.txt)"
#endif

namespace kernscope {
namespace {

// The live suite is built only when hipcc + ROCm headers were found at
// configure time (see libs/gpuruntime/CMakeLists.txt).
#ifndef KERNSCOPE_NO_LIVE_HIP

constexpr std::uint32_t kElems = 1024;

TEST(GpuruntimeHip, DeviceQueryReturnsGfx1101) {
  auto rt = create_runtime(Backend::kAuto);
  ASSERT_TRUE(rt.ok()) << rt.message;
  // Auto-detection must pick the live HIP backend on this machine.
  EXPECT_EQ(rt.value->backend_name(), "hip");

  auto devs = rt.value->list_devices();
  ASSERT_TRUE(devs.ok()) << devs.message;
  ASSERT_FALSE(devs.value.empty());

  const DeviceInfo &dev = devs.value.front();
  EXPECT_EQ(dev.vendor, "amd");
  EXPECT_GT(dev.memory_bytes, std::uint64_t{1024} << 20);

  auto props = rt.value->device_properties(dev.name);
  ASSERT_TRUE(props.ok()) << props.message;
  EXPECT_EQ(props.value.vendor, "amd");
  // The local device is a Radeon RX 7800 XT (gfx1101) - the hsaco below is
  // compiled for exactly this arch.
  EXPECT_EQ(props.value.gcn_arch, "gfx1101");
  EXPECT_GT(props.value.memory_bytes, std::uint64_t{1024} << 20);
}

TEST(GpuruntimeHip, AddKernelLaunchMemcpyRoundtrip) {
  auto rt = create_runtime(Backend::kHip);
  ASSERT_TRUE(rt.ok()) << rt.message;

  auto mod = Module::load(*rt.value, KERNSCOPE_TEST_HSACO);
  ASSERT_TRUE(mod.ok()) << mod.message;
  auto kern = Kernel::create(*rt.value, *mod.value, "kernscope_add");
  ASSERT_TRUE(kern.ok()) << kern.message;

  // Host inputs: deterministic values (input[i] = i * 0.5 + 1).
  std::vector<float> a(kElems);
  std::vector<float> b(kElems);
  std::vector<float> expect(kElems);
  for (std::uint32_t i = 0; i < kElems; ++i) {
    a[i] = (static_cast<float>(i) * 0.5f) + 1.0f;
    b[i] = (static_cast<float>(i) * 0.25f) - 2.0f;
    expect[i] = a[i] + b[i];
  }

  auto a_d = rt.value->device_malloc(kElems * sizeof(float));
  auto b_d = rt.value->device_malloc(kElems * sizeof(float));
  auto out_d = rt.value->device_malloc(kElems * sizeof(float));
  ASSERT_TRUE(a_d.ok()) << a_d.message;
  ASSERT_TRUE(b_d.ok()) << b_d.message;
  ASSERT_TRUE(out_d.ok()) << out_d.message;

  ASSERT_TRUE(rt.value->memcpy_htod(a_d.value, a.data(), kElems * sizeof(float)).ok());
  ASSERT_TRUE(rt.value->memcpy_htod(b_d.value, b.data(), kElems * sizeof(float)).ok());

  // Driver-API convention: device pointers pass BY VALUE (uintptr), n by value.
  const std::uint64_t a_ptr = a_d.value.value;
  const std::uint64_t b_ptr = b_d.value.value;
  const std::uint64_t out_ptr = out_d.value.value;
  const std::int32_t n = static_cast<std::int32_t>(kElems);
  const std::vector<KernelArg> args = {
      KernelArg{&a_ptr, sizeof(a_ptr)},
      KernelArg{&b_ptr, sizeof(b_ptr)},
      KernelArg{&out_ptr, sizeof(out_ptr)},
      KernelArg{&n, sizeof(n)},
  };

  LaunchConfig cfg;
  cfg.grid_x = 4;
  cfg.block_x = 256;
  auto launch = kern.value->launch(cfg, args);
  ASSERT_TRUE(launch.ok()) << launch.message;
  ASSERT_TRUE(rt.value->synchronize().ok());

  std::vector<float> got(kElems, -1.0f);
  ASSERT_TRUE(rt.value->memcpy_dtoh(got.data(), out_d.value, kElems * sizeof(float)).ok());

  // Roundtrip verified against host-side compute (exact float addition).
  for (std::uint32_t i = 0; i < kElems; ++i) {
    EXPECT_FLOAT_EQ(got[i], expect[i]) << "element " << i;
  }

  ASSERT_TRUE(rt.value->device_free(a_d.value).ok());
  ASSERT_TRUE(rt.value->device_free(b_d.value).ok());
  ASSERT_TRUE(rt.value->device_free(out_d.value).ok());
}

TEST(GpuruntimeHip, EventTimingIsPositive) {
  auto rt = create_runtime(Backend::kHip);
  ASSERT_TRUE(rt.ok()) << rt.message;

  auto mod = Module::load(*rt.value, KERNSCOPE_TEST_HSACO);
  ASSERT_TRUE(mod.ok()) << mod.message;
  auto kern = Kernel::create(*rt.value, *mod.value, "kernscope_add");
  ASSERT_TRUE(kern.ok()) << kern.message;

  auto a_d = rt.value->device_malloc(kElems * sizeof(float));
  ASSERT_TRUE(a_d.ok());
  const std::uint64_t a_ptr = a_d.value.value;
  const std::uint64_t out_ptr = a_ptr;
  const std::int32_t n = static_cast<std::int32_t>(kElems);
  const std::vector<KernelArg> args = {
      KernelArg{&a_ptr, sizeof(a_ptr)},
      KernelArg{&a_ptr, sizeof(a_ptr)},
      KernelArg{&out_ptr, sizeof(out_ptr)},
      KernelArg{&n, sizeof(n)},
  };
  LaunchConfig cfg;
  cfg.grid_x = 1;
  cfg.block_x = 256;

  auto e0 = Event::create(*rt.value);
  auto e1 = Event::create(*rt.value);
  ASSERT_TRUE(e0.ok()) << e0.message;
  ASSERT_TRUE(e1.ok()) << e1.message;

  // hipEventRecord/hipEventElapsedTime contract: the elapsed wall-clock time
  // between two recorded events on the same stream must be > 0 once a kernel
  // (or any device work) sits between them (HIP API, hipEventElapsedTime).
  ASSERT_TRUE(e0.value->record().ok());
  ASSERT_TRUE(kern.value->launch(cfg, args).ok());
  ASSERT_TRUE(e1.value->record().ok());
  ASSERT_TRUE(e1.value->synchronize().ok());

  auto elapsed = e0.value->elapsed_ms(*e1.value);
  ASSERT_TRUE(elapsed.ok()) << elapsed.message;
  EXPECT_GT(elapsed.value, 0.0);
}

#else // KERNSCOPE_NO_LIVE_HIP

TEST(GpuruntimeHip, LiveSuiteSkippedWithoutHipcc) {
  GTEST_SKIP() << "hipcc/ROCm headers not found at configure time "
                  "(KERNSCOPE_NO_LIVE_HIP); live HIP suite skipped";
}

#endif // KERNSCOPE_NO_LIVE_HIP

} // namespace
} // namespace kernscope
