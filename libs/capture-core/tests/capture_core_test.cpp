//===- capture_core_test.cpp - gtest suite for libs/capture-core -*- C++ -*-===//
//
// (a) a mock-runtime scenario (todo-21 MockRuntime) is driven in lockstep
//     with a CaptureSession exactly like the shim records what it sees at the
//     cuda* boundary, and the launch record (name/grid/block/args) plus the
//     allocation map sizes are asserted EXACT - the arg bytes come from the
//     mock's byte-exact op payload, mirroring what cuFuncGetParamInfo +
//     cuLaunchKernel capture on the driver path;
// (b) fatbin registration records the cubin bytes HASH (the bundle's cubin
//     field source for runtime-API apps; GPGPU-Sim g_mallocPtr_Size pattern,
//     docs/research/capture-replay-research-brief.md §2/§5);
// (c) cuFuncGetParamInfo arg introspection sizes the driver-API arg capture;
// (d) negative QA: an unknown cudaFree is logged "unknown-free" and forwarded
//     without a crash (cudaFree on a pointer cudaMalloc did not return is
//     undefined per the API contract - the shim must never crash on it);
// (e) the v0 bundle writer round-trips deterministically.
//
// All expectations cite the CUDA 12.6 header contracts in comments
// ($CUDA_HOME/include/cuda_runtime_api.h, cuda.h - Metis M-5.i: validate
// against the documented API contract, never against the mock's own
// behavior).
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture.h"

#include "kernscope/gpu.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

using kernscope::CaptureSession;
using kernscope::LaunchVia;

// Every test starts from a fresh session (the shim's singleton is exercised
// by the smoke binary instead).
class CaptureSessionTest : public ::testing::Test {
protected:
  void SetUp() override { session_ = std::make_unique<CaptureSession>(); }

  std::unique_ptr<CaptureSession> session_;
};

//--- (a) mock-runtime scenario -------------------------------------------------

TEST_F(CaptureSessionTest, MockScenarioCapturesLaunchAndExactAllocSizes) {
  // The app behavior runs against the todo-21 MockRuntime; every call is
  // recorded into the session through the SAME mapping the shim applies at
  // the cuda* boundary (cudaMalloc -> record_alloc, cudaMemcpy(H2D) ->
  // record_copy, cudaLaunchKernel -> record_launch).
  auto runtime = kernscope::create_runtime(kernscope::Backend::kMock);
  ASSERT_TRUE(runtime.ok());
  auto &mock = static_cast<kernscope::MockRuntime &>(*runtime.value);

  // cudaMalloc contract (cuda_runtime_api.h: "Allocates size bytes of linear
  // memory on the device"): the returned pointer is the start of a region of
  // EXACTLY `size` bytes -> the allocation map must record ptr->size exactly.
  auto d1 = mock.device_malloc(1024);
  auto d2 = mock.device_malloc(64);
  ASSERT_TRUE(d1.ok() && d2.ok());
  session_->record_alloc(d1.value.value, 1024);
  session_->record_alloc(d2.value.value, 64);

  // cudaMemcpy contract (cuda_runtime_api.h: "copies count bytes from the
  // memory area pointed to by src to the memory area pointed to by dst"): the
  // mock retains the copied blob byte-exactly for assertions.
  std::vector<std::uint8_t> host_blob(512);
  for (std::size_t i = 0; i < host_blob.size(); ++i) {
    host_blob[i] = static_cast<std::uint8_t>(i * 3);
  }
  ASSERT_TRUE(mock.memcpy_htod(d1.value, host_blob.data(), 512).ok());
  kernscope::CopyRecord copy;
  copy.kind = "memcpy";
  copy.dst = d1.value.value;
  copy.src = reinterpret_cast<std::uint64_t>(host_blob.data());
  copy.bytes = 512;
  copy.cuda_kind = 1; // cudaMemcpyKind (12.6): 1 = HostToDevice
  session_->record_copy(copy);

  // cudaLaunchKernel contract (cuda_runtime_api.h): gridDim/blockDim are the
  // launch bounds; args is an array of N pointers to the parameter values,
  // which must reach the kernel byte-exactly.
  const std::uint64_t ptr_arg = d1.value.value;
  const std::uint32_t count_arg = 512;
  const kernscope::KernelArg args[] = {
      {&ptr_arg, sizeof(ptr_arg)}, {&count_arg, sizeof(count_arg)}};
  kernscope::LaunchConfig cfg;
  cfg.grid_x = 2;
  cfg.grid_y = 1;
  cfg.grid_z = 1;
  cfg.block_x = 128;
  cfg.block_y = 1;
  cfg.block_z = 1;
  auto module = mock.load_module("vec_add.cubin");
  ASSERT_TRUE(module.ok());
  auto kernel = mock.create_kernel(*module.value, "vec_add");
  ASSERT_TRUE(kernel.ok());
  ASSERT_TRUE(kernel.value->launch(cfg, args).ok());

  kernscope::LaunchRecord launch;
  launch.kernel_name = "vec_add";
  launch.via = LaunchVia::kRuntimeApi;
  launch.grid_x = cfg.grid_x;
  launch.grid_y = cfg.grid_y;
  launch.grid_z = cfg.grid_z;
  launch.block_x = cfg.block_x;
  launch.block_y = cfg.block_y;
  launch.block_z = cfg.block_z;
  // Byte-exact args: the mock op payload IS the exact parameter bytes that
  // cross the launch boundary (MockRuntime::Op.payload) - the same bytes the
  // driver path obtains via cuFuncGetParamInfo sizing.
  const auto &launch_op = mock.ops().back();
  ASSERT_EQ(launch_op.type, kernscope::MockRuntime::Op::Type::kLaunch);
  std::size_t offset = 0;
  for (const auto &arg : args) {
    kernscope::KernelArgSlot slot;
    slot.size = arg.size;
    slot.bytes.assign(launch_op.payload.begin() + offset,
                      launch_op.payload.begin() + offset + arg.size);
    launch.args.push_back(std::move(slot));
    offset += arg.size;
  }
  session_->record_launch(std::move(launch));

  // --- assertions -------------------------------------------------------------
  // Allocation map sizes EXACT (acceptance): 1024 and 64 bytes, no more.
  ASSERT_EQ(session_->allocation_count(), 2u);
  EXPECT_EQ(session_->allocation_size(d1.value.value), std::optional<std::size_t>(1024));
  EXPECT_EQ(session_->allocation_size(d2.value.value), std::optional<std::size_t>(64));
  EXPECT_EQ(session_->allocation_size(0x9999), std::nullopt);

  // Launch record: name/grid/block/args.
  ASSERT_EQ(session_->launches().size(), 1u);
  const auto &rec = session_->launches()[0];
  EXPECT_EQ(rec.kernel_name, "vec_add");
  EXPECT_EQ(rec.grid_x, 2u);
  EXPECT_EQ(rec.grid_y, 1u);
  EXPECT_EQ(rec.grid_z, 1u);
  EXPECT_EQ(rec.block_x, 128u);
  EXPECT_EQ(rec.block_y, 1u);
  EXPECT_EQ(rec.block_z, 1u);
  ASSERT_EQ(rec.args.size(), 2u);
  EXPECT_EQ(rec.args[0].size, 8u);
  EXPECT_EQ(rec.args[1].size, 4u);
  // Arg bytes byte-exact against the mock's recorded payload (contract above).
  ASSERT_EQ(rec.args[0].bytes.size(), 8u);
  EXPECT_EQ(rec.args[0].bytes, std::vector<std::uint8_t>(launch_op.payload.begin(),
                                                         launch_op.payload.begin() + 8));
  ASSERT_EQ(rec.args[1].bytes.size(), 4u);
  EXPECT_EQ(rec.args[1].bytes, std::vector<std::uint8_t>(launch_op.payload.begin() + 8,
                                                         launch_op.payload.end()));

  // Copy record carries the map-attributed dst size (H2D: dst is device).
  ASSERT_EQ(session_->copies().size(), 1u);
  EXPECT_EQ(session_->copies()[0].bytes, 512u);
  EXPECT_EQ(session_->copies()[0].dst_size, 1024u);

  // Events are sequenced in call order.
  EXPECT_LT(session_->allocs()[0].seq, session_->allocs()[1].seq);
  EXPECT_LT(session_->allocs()[1].seq, session_->copies()[0].seq);
  EXPECT_LT(session_->copies()[0].seq, session_->launches()[0].seq);
}

TEST_F(CaptureSessionTest, FreeErasesMapAndUnknownFreeIsLoggedAndForwarded) {
  // cudaFree contract (cuda_runtime_api.h): "Frees the memory space pointed
  // to by devPtr" - the map must drop the pointer; the size is remembered in
  // the free event for replay. Freeing a pointer cudaMalloc did NOT return is
  // UNDEFINED behavior per the contract, so the session logs the unknown-free
  // QA event (never crashes) and the shim forwards.
  session_->record_alloc(0x1000, 256);
  session_->record_free(0x1000, /*forwarded=*/true);
  EXPECT_EQ(session_->allocation_size(0x1000), std::nullopt);
  EXPECT_EQ(session_->allocation_count(), 0u);
  ASSERT_EQ(session_->allocs().size(), 2u);
  EXPECT_TRUE(session_->allocs()[0].is_alloc);
  EXPECT_EQ(session_->allocs()[0].size, 256u);
  EXPECT_FALSE(session_->allocs()[1].is_alloc);
  EXPECT_EQ(session_->allocs()[1].size, 256u); // remembered size

  session_->record_free(0xDEADBEEF, /*forwarded=*/true);
  ASSERT_EQ(session_->unknown_frees().size(), 1u);
  EXPECT_EQ(session_->unknown_frees()[0].ptr, std::uint64_t{0xDEADBEEF});
  EXPECT_TRUE(session_->unknown_frees()[0].forwarded);
}

//--- (b) fatbin / cubin registration + sha256 ----------------------------------

TEST_F(CaptureSessionTest, FatBinaryRegistrationRecordsCubinBytesHash) {
  // __cudaRegisterFatBinary contract (the runtime's registration entry point,
  // GPGPU-Sim reimplements it in libcuda/cuda_runtime_api.cc): the shim
  // extracts the cubin payload from the fatbin image and the session hashes
  // EXACTLY those bytes - this is the bundle's cubin-field source for
  // runtime-API apps (acceptance: "fatbinary registration recorded with cubin
  // bytes HASH").
  std::vector<std::uint8_t> cubin(256);
  for (std::size_t i = 0; i < cubin.size(); ++i) {
    cubin[i] = static_cast<std::uint8_t>(i * 7 + 1);
  }
  const std::uint64_t reg_id = session_->register_fatbinary(cubin);
  EXPECT_EQ(reg_id, 1u);
  ASSERT_EQ(session_->modules().size(), 1u);
  const auto &mod = session_->modules()[0];
  EXPECT_EQ(mod.kind, kernscope::ModuleRecord::Kind::kFatBinary);
  EXPECT_EQ(mod.sha256, kernscope::sha256_hex(cubin));
  EXPECT_TRUE(mod.kernels.empty());

  // __cudaRegisterFunction associates the kernel name with the registration.
  EXPECT_TRUE(session_->register_function(reg_id, "vec_add"));
  EXPECT_FALSE(session_->register_function(99, "nope")); // unknown reg id
  ASSERT_EQ(session_->modules()[0].kernels.size(), 1u);
  EXPECT_EQ(session_->modules()[0].kernels[0], "vec_add");
}

TEST_F(CaptureSessionTest, Sha256MatchesFips1804Vectors) {
  // FIPS 180-4 published test vectors (the empty-string vector catches the
  // classic padding bug).
  EXPECT_EQ(kernscope::sha256_hex({}),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  const std::string abc = "abc";
  EXPECT_EQ(kernscope::sha256_hex(
                std::span<const std::uint8_t>(
                    reinterpret_cast<const std::uint8_t *>(abc.data()),
                    abc.size())),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

TEST_F(CaptureSessionTest, CubinModuleLoadRecordsBlobHash) {
  // cuModuleLoadData contract (cuda.h): "Loads [a] module with data from
  // memory" - the image is the cubin blob; the session hashes the exact
  // bytes the shim sized from the image.
  std::vector<std::uint8_t> image = {0x1, 0x2, 0x3, 0x4};
  const std::uint64_t mod_id = session_->register_cubin(image);
  EXPECT_EQ(mod_id, 1u);
  ASSERT_EQ(session_->modules().size(), 1u);
  EXPECT_EQ(session_->modules()[0].kind, kernscope::ModuleRecord::Kind::kCubin);
  EXPECT_EQ(session_->modules()[0].sha256, kernscope::sha256_hex(image));
}

//--- (c) cuFuncGetParamInfo arg introspection -----------------------------------

TEST_F(CaptureSessionTest, ParamInfoSizesDriverLaunchArgs) {
  // cuda.h cuFuncGetParamInfo contract (group__CUDA__EXEC): returns the
  // per-parameter offset/size "retrieved directly from the kernel's image" -
  // an interceptor learns the arg LAYOUT this way (brief §5). The session
  // then sizes the arg bytes captured for the matching cuLaunchKernel record.
  constexpr std::uint64_t kCufunction = 0x9000;
  session_->record_param_info(kCufunction, 0, 0, 8);
  session_->record_param_info(kCufunction, 1, 8, 4);
  EXPECT_EQ(session_->param_size(kCufunction, 0), std::optional<std::size_t>(8));
  EXPECT_EQ(session_->param_size(kCufunction, 1), std::optional<std::size_t>(4));
  EXPECT_EQ(session_->param_size(kCufunction, 2), std::nullopt);

  // cuModuleGetFunction contract (cuda.h): resolves a CUfunction for a named
  // kernel - the driver-path launch record gets its kernel name from it.
  session_->register_cufunction(kCufunction, "vec_add");
  EXPECT_EQ(session_->kernel_name_for_cufunction(kCufunction), "vec_add");

  // The shim's CaptureDriverArgs equivalent: size slots from the layout and
  // copy the parameter bytes (kernelParams[i] -> value).
  std::uint64_t ptr_value = 0x1000;
  std::uint32_t count_value = 512;
  kernscope::LaunchRecord rec;
  rec.via = LaunchVia::kDriverApi;
  rec.kernel_name = session_->kernel_name_for_cufunction(kCufunction);
  std::vector<std::uint8_t> ptr_bytes(sizeof(ptr_value));
  std::vector<std::uint8_t> count_bytes(sizeof(count_value));
  std::memcpy(ptr_bytes.data(), &ptr_value, sizeof(ptr_value));
  std::memcpy(count_bytes.data(), &count_value, sizeof(count_value));
  kernscope::KernelArgSlot s0{8, ptr_bytes};
  kernscope::KernelArgSlot s1{4, count_bytes};
  rec.args = {std::move(s0), std::move(s1)};
  session_->record_launch(std::move(rec));

  ASSERT_EQ(session_->launches().size(), 1u);
  EXPECT_EQ(session_->launches()[0].kernel_name, "vec_add");
  ASSERT_EQ(session_->launches()[0].args.size(), 2u);
  EXPECT_EQ(session_->launches()[0].args[0].bytes, ptr_bytes);
  EXPECT_EQ(session_->launches()[0].args[1].bytes, count_bytes);
}

//--- interception table ----------------------------------------------------------

TEST_F(CaptureSessionTest, ApiTableCountsEveryInterposedApi) {
  // Metis C-1: the fixed interception table - every API the shim interposes
  // on has a stable name + a call counter.
  const auto &table = session_->api_table();
  EXPECT_EQ(table.size(), std::size_t(kernscope::ApiKind::kCount));
  std::size_t named = 0;
  for (const auto &entry : table) {
    EXPECT_EQ(entry.name, kernscope::api_name(entry.kind));
    EXPECT_EQ(entry.calls, 0u);
    EXPECT_NE(entry.name[0], '?');
    ++named;
  }
  EXPECT_EQ(named, std::size_t(kernscope::ApiKind::kCount));

  session_->note_api(kernscope::ApiKind::kCudaMalloc);
  session_->note_api(kernscope::ApiKind::kCudaMalloc);
  session_->note_api(kernscope::ApiKind::kCudaFree);
  EXPECT_EQ(session_->api_count(kernscope::ApiKind::kCudaMalloc), 2u);
  EXPECT_EQ(session_->api_count(kernscope::ApiKind::kCudaFree), 1u);
  EXPECT_EQ(session_->api_count(kernscope::ApiKind::kCuLaunchKernel), 0u);
}

//--- streams ---------------------------------------------------------------------

TEST_F(CaptureSessionTest, StreamsGetStableLogicalIds) {
  // cudaStreamCreate contract (cuda_runtime_api.h): "Creates a new
  // asynchronous stream" - each created handle maps to a stable logical id;
  // synchronize records the id (0 = handle never seen).
  EXPECT_EQ(session_->record_stream_create(0x1234), 1u);
  EXPECT_EQ(session_->record_stream_create(0x5678), 2u);
  EXPECT_EQ(session_->record_stream_create(0x1234), 1u); // idempotent
  session_->record_stream_sync(0x1234);
  session_->record_stream_sync(0x9999); // unknown handle: id 0, no crash
  ASSERT_EQ(session_->streams().size(), 5u);
  EXPECT_FALSE(session_->streams()[0].is_sync);
  EXPECT_EQ(session_->streams()[0].id, 1u);
  EXPECT_TRUE(session_->streams()[3].is_sync);
  EXPECT_EQ(session_->streams()[3].id, 1u);
  EXPECT_TRUE(session_->streams()[4].is_sync);
  EXPECT_EQ(session_->streams()[4].id, 0u);
}

//--- (e) v0 bundle writer ----------------------------------------------------------

TEST_F(CaptureSessionTest, BundleWriterV0IsDeterministicAndComplete) {
  session_->note_api(kernscope::ApiKind::kCudaMalloc);
  session_->record_alloc(0x1000, 1024);
  session_->record_free(0xBEEF, true); // unknown free (QA negative)
  kernscope::LaunchRecord launch;
  launch.kernel_name = "vec_add";
  launch.via = LaunchVia::kRuntimeApi;
  launch.grid_x = 2;
  launch.block_x = 128;
  kernscope::KernelArgSlot slot;
  slot.size = 4;
  slot.bytes = {0x00, 0x01, 0x02, 0x03};
  launch.args.push_back(std::move(slot));
  session_->record_launch(std::move(launch));

  std::vector<std::uint8_t> cubin = {0xAA, 0xBB, 0xCC};
  session_->register_fatbinary(cubin);

  auto writer = kernscope::BundleWriter::create_v0();
  ASSERT_NE(writer, nullptr);
  std::ostringstream first, second;
  ASSERT_TRUE(writer->write(*session_, first));
  ASSERT_TRUE(writer->write(*session_, second));
  EXPECT_EQ(first.str(), second.str()); // deterministic

  const std::string out = first.str();
  EXPECT_NE(out.find("kernscope-capture-v0"), std::string::npos);
  EXPECT_NE(out.find("api cudaMalloc calls=1"), std::string::npos);
  EXPECT_NE(out.find("alloc ptr=0x1000 size=1024"), std::string::npos);
  EXPECT_NE(out.find("unknown-free ptr=0xbeef forwarded=true"), std::string::npos);
  EXPECT_NE(out.find("launch kernel=\"vec_add\" via=runtime grid=2x1x1 "
                     "block=128x1x1 shared=0 stream=0x0 graph=0x0 args=1 "
                     "[4:00010203]"),
            std::string::npos);
  EXPECT_NE(out.find("fatbin mod=1 sha256=" + kernscope::sha256_hex(cubin)),
            std::string::npos);
  EXPECT_NE(out.find("end-capture-v0"), std::string::npos);
}

} // namespace
