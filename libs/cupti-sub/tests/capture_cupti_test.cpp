//===- capture_cupti_test.cpp - gtest suite for libs/cupti-sub ----------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Unit suite (todo 44, acceptance `ctest -R capture_cupti`): callback
// dispatch, argument capture and allocation-size learning from CUPTI Memory
// activity records - driven by a SYNTHETIC CUPTI-callback harness
// (tests/synthetic_cupti.h) in the mock style of the todo-43 mock-cudart:
// the harness constructs the exact structs CUPTI would deliver and invokes
// the same dispatcher function CUPTI calls on hardware
// (kernscope::cupti::handle_callback / handle_activity_record). No NVIDIA
// runtime is involved; real-NVIDIA behavior is rental-gated.
//
// All expectations cite the CUDA 12.6 / CUPTI 12.6 header contracts in
// comments (Metis M-5.i: validate against the documented API contract, never
// against the harness's own behavior).
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture_cupti.h"

#include "synthetic_cupti.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

using kernscope::LaunchVia;
using kernscope::cupti::handle_activity_record;
namespace syn = kernscope::cupti::synthetic;

constexpr CUpti_CallbackDomain kRuntime = CUPTI_CB_DOMAIN_RUNTIME_API;
constexpr CUpti_CallbackDomain kDriver = CUPTI_CB_DOMAIN_DRIVER_API;
constexpr CUpti_CallbackDomain kResource = CUPTI_CB_DOMAIN_RESOURCE;

// Every test starts from a fresh session armed through the synthetic-harness
// entry (arm_session: same dispatcher, no real subscription - deterministic
// on this driverless box).
class CuptiCaptureTest : public ::testing::Test {
protected:
  void SetUp() override {
    session_ = std::make_unique<kernscope::CaptureSession>();
    kernscope::cupti::arm_session(session_.get());
  }

  std::unique_ptr<kernscope::CaptureSession> session_;
};

//--- callback dispatch ---------------------------------------------------------

TEST_F(CuptiCaptureTest, CallbackDispatchRecordsMallocFree) {
  // cudaMalloc contract (cuda_runtime_api.h): "Allocates size bytes of linear
  // memory on the device and returns in *devPtr a pointer to the allocated
  // memory" - the size must land in the ptr->size allocation map EXACTLY.
  void *ptr = reinterpret_cast<void *>(0x1234);
  cudaMalloc_v3020_params p;
  p.devPtr = &ptr;
  p.size = 4096;
  auto cb = syn::CudaMalloc(&p, cudaSuccess);
  cb.At(CUPTI_API_EXIT)->Send(kRuntime, static_cast<CUpti_CallbackId>(
                        CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020));
  ASSERT_EQ(session_->api_count(kernscope::ApiKind::kCudaMalloc), 1u);
  ASSERT_EQ(session_->allocation_size(0x1234),
            std::optional<std::size_t>(4096));

  // cudaFree: the map entry is removed and a free record is emitted.
  cudaFree_v3020_params fp;
  fp.devPtr = ptr;
  auto free_cb = syn::CudaFree(&fp, cudaSuccess);
  free_cb.At(CUPTI_API_EXIT)->Send(kRuntime, static_cast<CUpti_CallbackId>(
                             CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020));
  ASSERT_EQ(session_->api_count(kernscope::ApiKind::kCudaFree), 1u);
  ASSERT_FALSE(session_->allocation_size(0x1234).has_value());
  ASSERT_EQ(session_->allocs().size(), 2u);
}

TEST_F(CuptiCaptureTest, CallbackDispatchRecordsMemcpy) {
  // cudaMemcpy contract (cuda_runtime_api.h): "copies count bytes from the
  // memory area pointed to by src to the memory area pointed to by dst" with
  // kind cudaMemcpyHostToDevice (1) for H2D copies.
  std::uint8_t host[64];
  std::memset(host, 0xAB, sizeof(host));
  void *dst = reinterpret_cast<void *>(0x2000);
  cudaMemcpy_v3020_params p;
  p.dst = dst;
  p.src = host;
  p.count = sizeof(host);
  p.kind = cudaMemcpyHostToDevice;
  const CUpti_CallbackId cbid =
      static_cast<CUpti_CallbackId>(CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020);
  auto cb = syn::CudaMemcpy(&p, cudaSuccess);
  cb.At(CUPTI_API_ENTER)->Send(kRuntime, cbid);
  cb.At(CUPTI_API_EXIT)->Send(kRuntime, cbid);

  ASSERT_EQ(session_->api_count(kernscope::ApiKind::kCudaMemcpy), 1u);
  ASSERT_EQ(session_->copies().size(), 1u);
  const kernscope::CopyRecord &c = session_->copies().back();
  ASSERT_EQ(c.bytes, sizeof(host));
  ASSERT_EQ(c.cuda_kind, 1);
  ASSERT_EQ(c.dst, 0x2000u);
  ASSERT_EQ(c.src, reinterpret_cast<std::uint64_t>(host));
}

TEST_F(CuptiCaptureTest, CallbackDispatchRecordsLaunchWithArgs) {
  // The launch capture path (the todo-44 crux): grid/block/shared/stream from
  // cudaLaunchKernel functionParams + kernel name from symbolName + per-arg
  // bytes from the args array sized by the cuFuncGetParamInfo layout facts
  // (brief §5). Contract citations: cudaLaunchKernel (cuda_runtime_api.h):
  // "args is an array of pointers to the kernel's arguments";
  // cuFuncGetParamInfo (cuda.h): per-parameter offset/size from the kernel
  // image.
  constexpr std::uint64_t kCufunc = 0xABCD;

  // The driver-API sequence a real app performs: cuModuleGetFunction maps the
  // CUfunction to its name first, then cuFuncGetParamInfo queries the param
  // layout of that function (cuda.h contracts).
  auto modfn = syn::CuModuleGetFunction(kCufunc,
                                        "kernscope_templates_elementwise_add");
  modfn.At(CUPTI_API_EXIT)->Send(kDriver, static_cast<CUpti_CallbackId>(
                          CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction));

  // Driver-domain layout facts (a synthetic driver querying the image).
  const std::size_t sizes[] = {8, 8, 8, 4}; // ptr, ptr, ptr, int32
  for (int i = 0; i < 4; ++i) {
    auto cb = syn::CuFuncGetParamInfo(kCufunc, i, i * 8, sizes[i]);
    cb.At(CUPTI_API_EXIT)->Send(kDriver, static_cast<CUpti_CallbackId>(
                         CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo));
  }

  // The kernel's four args, byte-identical values at the API boundary.
  std::uint64_t a = 0x3000;
  std::uint64_t b = 0x4000;
  std::uint64_t out = 0x5000;
  std::int32_t n = 1024;
  void *args[] = {&a, &b, &out, &n};

  cudaLaunchKernel_v7000_params p;
  p.func = reinterpret_cast<const void *>(0xFEED);
  p.gridDim = dim3(4, 1, 1);
  p.blockDim = dim3(256, 1, 1);
  p.args = args;
  p.sharedMem = 0;
  p.stream = nullptr;

  const char *symbol = "kernscope_templates_elementwise_add";
  auto cb = syn::CudaLaunchKernel(&p);
  cb.At(CUPTI_API_ENTER, symbol)->Send(
      kRuntime,
      static_cast<CUpti_CallbackId>(CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000));

  ASSERT_EQ(session_->launches().size(), 1u);
  const kernscope::LaunchRecord &l = session_->launches().back();
  ASSERT_EQ(l.kernel_name, symbol);
  ASSERT_EQ(l.via, LaunchVia::kRuntimeApi);
  ASSERT_EQ(l.grid_x, 4u);
  ASSERT_EQ(l.block_x, 256u);
  ASSERT_EQ(l.args.size(), 4u);
  ASSERT_EQ(l.args[0].size, 8u);
  std::uint64_t captured_a = 0;
  std::memcpy(&captured_a, l.args[0].bytes.data(), 8);
  ASSERT_EQ(captured_a, a); // pointer VALUE byte-exact (VA normalized later)
  ASSERT_EQ(l.args[3].size, 4u);
  std::int32_t captured_n = 0;
  std::memcpy(&captured_n, l.args[3].bytes.data(), 4);
  ASSERT_EQ(captured_n, n); // scalar byte-exact, no normalization needed
}

TEST_F(CuptiCaptureTest, CallbackDispatchModuleLoadRecordsCubin) {
  // CUPTI_CBID_RESOURCE_MODULE_LOADED contract (cupti_callbacks.h):
  // CUpti_ModuleResourceData carries cubinSize + pCubin - the code-capture
  // path (brief §5). The session must register the blob hash (provenance).
  const std::vector<std::uint8_t> cubin = {0x7F, 'E', 'L', 'F', 1, 2, 3, 4};
  auto cb = syn::ModuleLoaded(cubin);
  cb.Send(kResource, static_cast<CUpti_CallbackId>(
                         CUPTI_CBID_RESOURCE_MODULE_LOADED));

  ASSERT_EQ(session_->modules().size(), 1u);
  ASSERT_EQ(session_->modules().back().kind,
            kernscope::ModuleRecord::Kind::kCubin);
  ASSERT_EQ(session_->modules().back().sha256,
            kernscope::sha256_hex(cubin));
}

//--- alloc-size learning from Memory activity records ---------------------------

TEST_F(CuptiCaptureTest, MemoryRecordsLearnAllocationSizes) {
  // CUpti_ActivityMemory4 contract (cupti_activity.h): alloc/free memory
  // operations report `address` + `bytes` WITHOUT any allocation hook - the
  // ptr->size allocation-map learner (brief §5 heuristic 2).
  CUpti_ActivityMemory4 alloc{};
  alloc.kind = CUPTI_ACTIVITY_KIND_MEMORY;
  alloc.memoryOperationType = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION;
  alloc.memoryKind = CUPTI_ACTIVITY_MEMORY_KIND_DEVICE;
  alloc.address = 0x6000;
  alloc.bytes = 2048;
  handle_activity_record(reinterpret_cast<const CUpti_Activity *>(&alloc));
  ASSERT_EQ(session_->allocation_size(0x6000),
            std::optional<std::size_t>(2048));
  ASSERT_EQ(session_->allocs().size(), 1u);

  // A non-device kind (pageable host memory) must NOT feed the device map.
  CUpti_ActivityMemory4 host_alloc = alloc;
  host_alloc.memoryKind = CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE;
  host_alloc.address = 0x7000;
  handle_activity_record(reinterpret_cast<const CUpti_Activity *>(&host_alloc));
  ASSERT_FALSE(session_->allocation_size(0x7000).has_value());

  // FREE removes the entry and records the remembered size.
  CUpti_ActivityMemory4 free_rec = alloc;
  free_rec.memoryOperationType = CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE;
  handle_activity_record(reinterpret_cast<const CUpti_Activity *>(&free_rec));
  ASSERT_FALSE(session_->allocation_size(0x6000).has_value());
  ASSERT_EQ(session_->allocs().size(), 2u);
  ASSERT_FALSE(session_->allocs().back().is_alloc);
  ASSERT_EQ(session_->allocs().back().size, 2048u);
}

} // namespace
