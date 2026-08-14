//===- gpuruntime_test.cpp - gpuruntime unit tests (no GPU needed) --------===//
//
// kernscope gpuruntime test suite, ctest LABELS no-gpu (todo 21). Exercises
// the two backends that need no device at all:
//   * MockRuntime  - op-log + BYTE-EXACT buffer retention. Expectations cite
//     the CUDA/HIP API contract docs (Metis M-5.i), e.g. cudaMemcpy copies
//     the exact byte sequence of the source region (CUDA Runtime API
//     documentation, Memory Management > cudaMemcpy) and cuLaunchKernel
//     passes the kernelParams array to the kernel unmodified (CUDA Driver
//     API documentation, cuLaunchKernel). The mock is validated against the
//     documented contract, never against itself (no circular validation).
//   * CompileOnlyRuntime - every call returns a typed GpuError::kNoDevice
//     result (asserted below), so the suite runs on GPU-less machines.
//
// This file must compile without any ROCm/CUDA header on the include path
// (the backend headers live only in src/gpu.cpp) - the include/ directory is
// all the test needs.
//
//===----------------------------------------------------------------------===//

#include "kernscope/gpu.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace kernscope {
namespace {

//===--- factory -------------------------------------------------------------===//

TEST(GpuruntimeFactory, MockAndCompileOnlyAreAlwaysAvailable) {
  auto mock = create_runtime(Backend::kMock);
  ASSERT_TRUE(mock.ok()) << mock.message;
  EXPECT_EQ(mock.value->backend_name(), "mock");

  auto co = create_runtime(Backend::kCompileOnly);
  ASSERT_TRUE(co.ok()) << co.message;
  EXPECT_EQ(co.value->backend_name(), "compile-only");
}

TEST(GpuruntimeFactory, AutoReturnsAnOkRuntimeOnAnyMachine) {
  // kAuto never fails: a live compiled backend wins; otherwise the
  // deterministic compile-only runtime is returned (typed per-call errors).
  auto rt = create_runtime(Backend::kAuto);
  ASSERT_TRUE(rt.ok()) << rt.message;
  EXPECT_FALSE(rt.value->backend_name().empty());
}

TEST(GpuruntimeFactory, UncompiledBackendIsATypedError) {
  // The backend the build did NOT compile must surface as a typed
  // unsupported-backend error, never a crash (asserted symmetrically for
  // whichever backend the build compiled).
  if (kHipBackendCompiled) {
    auto cuda = create_runtime(Backend::kCuda);
    ASSERT_FALSE(cuda.ok());
    EXPECT_EQ(cuda.err, GpuError::kUnsupportedBackend);
  }
  if (kCudaBackendCompiled) {
    auto hip = create_runtime(Backend::kHip);
    ASSERT_FALSE(hip.ok());
    EXPECT_EQ(hip.err, GpuError::kUnsupportedBackend);
  }
}

//===--- MockRuntime: op log + byte-exact buffers -----------------------------===//

TEST(GpuruntimeMock, ByteExactMemcpyRoundtrip) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());
  auto *mock = dynamic_cast<MockRuntime *>(rt.value.get());
  ASSERT_NE(mock, nullptr);

  // Contract: cudaMemcpy copies the exact byte sequence of the source region
  // (CUDA Runtime API, Memory Management > cudaMemcpy). The mock must retain
  // those exact bytes so a later D2H copy returns the identical blob.
  const std::size_t kBytes = 4096;
  auto in = rt.value->device_malloc(kBytes);
  ASSERT_TRUE(in.ok()) << in.message;

  std::vector<std::uint8_t> golden(kBytes);
  for (std::size_t i = 0; i < kBytes; ++i) {
    golden[i] = static_cast<std::uint8_t>((i * 31 + 7) & 0xFF);
  }

  auto h2d = rt.value->memcpy_htod(in.value, golden.data(), kBytes);
  ASSERT_TRUE(h2d.ok()) << h2d.message;

  // The retained buffer must be byte-exact (review-mandated: todo 44's
  // cross-vendor equivalence test reads these bytes).
  auto retained = mock->buffer(in.value);
  ASSERT_EQ(retained.size(), kBytes);
  EXPECT_EQ(std::memcmp(retained.data(), golden.data(), kBytes), 0);

  std::vector<std::uint8_t> out(kBytes, 0);
  auto d2h = rt.value->memcpy_dtoh(out.data(), in.value, kBytes);
  ASSERT_TRUE(d2h.ok()) << d2h.message;
  EXPECT_EQ(std::memcmp(out.data(), golden.data(), kBytes), 0);

  // The memcpy ops carry the byte-exact payload in the log too.
  const auto &ops = mock->ops();
  ASSERT_GE(ops.size(), 2);
  EXPECT_EQ(ops[ops.size() - 2].type, MockRuntime::Op::Type::kMemcpyHtoD);
  ASSERT_EQ(ops[ops.size() - 2].payload.size(), kBytes);
  EXPECT_EQ(std::memcmp(ops[ops.size() - 2].payload.data(), golden.data(), kBytes), 0);
  EXPECT_EQ(ops.back().type, MockRuntime::Op::Type::kMemcpyDtoH);
  ASSERT_EQ(ops.back().payload.size(), kBytes);
  EXPECT_EQ(std::memcmp(ops.back().payload.data(), golden.data(), kBytes), 0);
}

TEST(GpuruntimeMock, MemsetIsByteExact) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());
  auto *mock = dynamic_cast<MockRuntime *>(rt.value.get());
  ASSERT_NE(mock, nullptr);

  auto buf = rt.value->device_malloc(64);
  ASSERT_TRUE(buf.ok());
  auto set = rt.value->memset_device(buf.value, 0xA5, 64);
  ASSERT_TRUE(set.ok()) << set.message;

  // Contract: cudaMemset fills every byte of the region with the given value
  // (CUDA Runtime API, Memory Management > cudaMemset).
  auto bytes = mock->buffer(buf.value);
  ASSERT_EQ(bytes.size(), 64);
  for (std::uint8_t b : bytes) {
    EXPECT_EQ(b, 0xA5);
  }

  auto free_ok = rt.value->device_free(buf.value);
  ASSERT_TRUE(free_ok.ok());
  // Freed handle no longer retains contents.
  EXPECT_TRUE(mock->buffer(buf.value).empty());
}

TEST(GpuruntimeMock, LaunchRecordsByteExactArgs) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());
  auto *mock = dynamic_cast<MockRuntime *>(rt.value.get());
  ASSERT_NE(mock, nullptr);

  auto mod = Module::load(*rt.value, "/mock/add.hsaco");
  ASSERT_TRUE(mod.ok()) << mod.message;
  auto kern = Kernel::create(*rt.value, *mod.value, "kernscope_add");
  ASSERT_TRUE(kern.ok()) << kern.message;

  // Contract: cuLaunchKernel passes the kernelParams array to the kernel
  // unmodified (CUDA Driver API documentation, cuLaunchKernel) - the mock
  // records the argument blob byte-exactly so tests can assert it.
  std::uint64_t a_ptr = 0x1234, b_ptr = 0x5678, out_ptr = 0x9ABC;
  std::int32_t n = 1024;
  std::vector<KernelArg> args = {
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

  const auto &ops = mock->ops();
  // Op-order contract (asserted exactly, see OpOrderContract below).
  ASSERT_GE(ops.size(), 3);
  EXPECT_EQ(ops[0].type, MockRuntime::Op::Type::kModuleLoad);
  EXPECT_EQ(ops[0].detail, "/mock/add.hsaco");
  EXPECT_EQ(ops[1].type, MockRuntime::Op::Type::kKernelCreate);
  EXPECT_EQ(ops[1].detail, "kernscope_add");
  EXPECT_EQ(ops[2].type, MockRuntime::Op::Type::kLaunch);

  const MockRuntime::Op &op = ops[2];
  EXPECT_EQ(op.detail, "kernscope_add");
  EXPECT_EQ(op.config.grid_x, 4);
  EXPECT_EQ(op.config.block_x, 256);
  // Byte-exact arg blob: 3 device pointers as uint64 LE, then the int32 n.
  ASSERT_EQ(op.payload.size(), 3 * sizeof(std::uint64_t) + sizeof(std::int32_t));
  // Byte-exact arg blob: pointers as uint64 LE, then the int32 n.
  std::uint64_t got_a = 0, got_b = 0, got_out = 0;
  std::int32_t got_n = 0;
  std::memcpy(&got_a, op.payload.data(), sizeof(got_a));
  std::memcpy(&got_b, op.payload.data() + 8, sizeof(got_b));
  std::memcpy(&got_out, op.payload.data() + 16, sizeof(got_out));
  std::memcpy(&got_n, op.payload.data() + 24, sizeof(got_n));
  EXPECT_EQ(got_a, a_ptr);
  EXPECT_EQ(got_b, b_ptr);
  EXPECT_EQ(got_out, out_ptr);
  EXPECT_EQ(got_n, n);
}

TEST(GpuruntimeMock, OpOrderContract) {
  // The full op sequence must match the CUDA/HIP API call contract exactly
  // (module load -> kernel create -> alloc -> H2D -> launch -> D2H -> free).
  // QA: an unexpected op order fails this test by citing the expected
  // contract position (see the EXPECT_EQ on type below).
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());
  auto *mock = dynamic_cast<MockRuntime *>(rt.value.get());
  ASSERT_NE(mock, nullptr);

  auto mod = Module::load(*rt.value, "/mock/add.hsaco");
  ASSERT_TRUE(mod.ok());
  auto kern = Kernel::create(*rt.value, *mod.value, "kernscope_add");
  ASSERT_TRUE(kern.ok());

  const std::uint32_t kElems = 8;
  auto in = rt.value->device_malloc(kElems * sizeof(float));
  ASSERT_TRUE(in.ok());
  auto out = rt.value->device_malloc(kElems * sizeof(float));
  ASSERT_TRUE(out.ok());

  std::vector<float> host_in(kElems, 1.5f);
  ASSERT_TRUE(rt.value->memcpy_htod(in.value, host_in.data(), host_in.size() * sizeof(float)).ok());

  std::uint64_t a = in.value.value, b = in.value.value, o = out.value.value;
  std::int32_t n = static_cast<std::int32_t>(kElems);
  std::vector<KernelArg> args = {
      KernelArg{&a, sizeof(a)}, KernelArg{&b, sizeof(b)},
      KernelArg{&o, sizeof(o)}, KernelArg{&n, sizeof(n)},
  };
  LaunchConfig cfg;
  cfg.grid_x = 1;
  cfg.block_x = 256;
  ASSERT_TRUE(kern.value->launch(cfg, args).ok());

  std::vector<float> host_out(kElems, 0.0f);
  ASSERT_TRUE(rt.value->memcpy_dtoh(host_out.data(), out.value, host_out.size() * sizeof(float)).ok());
  ASSERT_TRUE(rt.value->device_free(in.value).ok());

  const auto &ops = mock->ops();
  const MockRuntime::Op::Type expected[] = {
      MockRuntime::Op::Type::kModuleLoad,
      MockRuntime::Op::Type::kKernelCreate,
      MockRuntime::Op::Type::kMalloc,
      MockRuntime::Op::Type::kMalloc,
      MockRuntime::Op::Type::kMemcpyHtoD,
      MockRuntime::Op::Type::kLaunch,
      MockRuntime::Op::Type::kMemcpyDtoH,
      MockRuntime::Op::Type::kFree,
  };
  ASSERT_EQ(ops.size(), 8);
  for (std::size_t i = 0; i < 8; ++i) {
    // Contract position i: an op of the wrong type at this position means the
    // call order violated the API sequence the driver would require.
    EXPECT_EQ(static_cast<int>(ops[i].type), static_cast<int>(expected[i]))
        << "op[" << i << "] violated the API call-order contract (got "
        << static_cast<int>(ops[i].type) << ", want " << static_cast<int>(expected[i]) << ")";
  }
}

TEST(GpuruntimeMock, EventsAreRecordedAndDeterministic) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());
  auto *mock = dynamic_cast<MockRuntime *>(rt.value.get());
  ASSERT_NE(mock, nullptr);

  auto e0 = Event::create(*rt.value);
  auto e1 = Event::create(*rt.value);
  ASSERT_TRUE(e0.ok());
  ASSERT_TRUE(e1.ok());
  ASSERT_TRUE(e0.value->record().ok());
  ASSERT_TRUE(e1.value->record().ok());
  ASSERT_TRUE(e1.value->synchronize().ok());

  auto elapsed = e0.value->elapsed_ms(*e1.value);
  ASSERT_TRUE(elapsed.ok()) << elapsed.message;
  // The >0 timing contract is asserted live (gpu-amd suite); the mock pins a
  // fixed deterministic value so elapsed-based logic is unit-testable.
  EXPECT_EQ(elapsed.value, 0.5);

  const auto &ops = mock->ops();
  ASSERT_EQ(ops.size(), 4);
  EXPECT_EQ(ops[0].type, MockRuntime::Op::Type::kEventRecord);
  EXPECT_EQ(ops[1].type, MockRuntime::Op::Type::kEventRecord);
  EXPECT_EQ(ops[2].type, MockRuntime::Op::Type::kEventSynchronize);
  EXPECT_EQ(ops[3].type, MockRuntime::Op::Type::kEventElapsed);
}

TEST(GpuruntimeMock, DeviceQuery) {
  auto rt = create_runtime(Backend::kMock);
  ASSERT_TRUE(rt.ok());

  auto devs = rt.value->list_devices();
  ASSERT_TRUE(devs.ok()) << devs.message;
  ASSERT_EQ(devs.value.size(), 1);
  EXPECT_EQ(devs.value[0].name, "mock-gpu-0");
  EXPECT_EQ(devs.value[0].vendor, "mock");

  auto props = rt.value->device_properties(devs.value[0].name);
  ASSERT_TRUE(props.ok()) << props.message;
  EXPECT_EQ(props.value.gcn_arch, "gfx1101");

  auto missing = rt.value->device_properties("no-such-gpu");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.err, GpuError::kDeviceNotFound);
}

//===--- CompileOnlyRuntime: typed no-device errors ---------------------------===//

TEST(GpuruntimeCompileOnly, EveryCallReturnsTypedNoDevice) {
  auto rt = create_runtime(Backend::kCompileOnly);
  ASSERT_TRUE(rt.ok());
  EXPECT_EQ(rt.value->backend_name(), "compile-only");

  // Contract: on a machine with no GPU/driver every call returns a typed
  // "no-device" error (never crashes, never succeeds silently) - this is the
  // compile-only mode the unit suite depends on.
  auto devs = rt.value->list_devices();
  ASSERT_FALSE(devs.ok());
  EXPECT_EQ(devs.err, GpuError::kNoDevice);

  auto props = rt.value->device_properties("anything");
  ASSERT_FALSE(props.ok());
  EXPECT_EQ(props.err, GpuError::kNoDevice);

  auto mod = rt.value->load_module("/nope.hsaco");
  ASSERT_FALSE(mod.ok());
  EXPECT_EQ(mod.err, GpuError::kNoDevice);

  auto buf = rt.value->device_malloc(64);
  ASSERT_FALSE(buf.ok());
  EXPECT_EQ(buf.err, GpuError::kNoDevice);

  EXPECT_EQ(rt.value->device_free(DevicePtr{1}).err, GpuError::kNoDevice);
  EXPECT_EQ(rt.value->memcpy_htod(DevicePtr{1}, nullptr, 4).err, GpuError::kNoDevice);
  std::uint8_t sink = 0;
  EXPECT_EQ(rt.value->memcpy_dtoh(&sink, DevicePtr{1}, 1).err, GpuError::kNoDevice);
  EXPECT_EQ(rt.value->memset_device(DevicePtr{1}, 0, 4).err, GpuError::kNoDevice);
  EXPECT_EQ(rt.value->synchronize().err, GpuError::kNoDevice);

  auto ev = rt.value->create_event();
  ASSERT_FALSE(ev.ok());
  EXPECT_EQ(ev.err, GpuError::kNoDevice);
}

} // namespace
} // namespace kernscope
