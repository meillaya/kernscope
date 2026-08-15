//===- synthetic_cupti.h - synthetic CUPTI-callback test harness ----------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// The synthetic CUPTI harness (todo 44) in the mock style of the todo-43
// mock-cudart: it constructs the EXACT objects CUPTI would deliver to the
// subscription callback (CUpti_CallbackData + the functionParams structs from
// generated_cuda_runtime_api_meta.h / generated_cuda_meta.h /
// cupti_callbacks.h) and invokes the SAME dispatcher function the real CUPTI
// calls on hardware (kernscope::cupti::handle_callback /
// handle_activity_record). The real subscription is never involved - no
// NVIDIA runtime exists on this box (compile+unit only; real-NVIDIA behavior
// is rental-gated).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_SYNTHETIC_CUPTI_H
#define KERNSCOPE_SYNTHETIC_CUPTI_H

#include "kernscope/cupti_capture.h"

#include <cstdint>
#include <cstring>
#include <vector>

namespace kernscope::cupti::synthetic {

/// One synthetic callback delivery: a CUpti_CallbackData whose fields mirror
/// the CUPTI contract (cupti_callbacks.h: functionParams valid only within
/// the callback; symbolName valid on launch callbacks), with the params
/// copied byte-for-byte so pointer fields stay valid.
struct Callback {
  CUpti_CallbackData data{};
  CUpti_ResourceData resource{};
  CUpti_ModuleResourceData resource_module{};
  cudaError_t retval = cudaSuccess;
  std::vector<std::uint8_t> params;

  Callback *At(CUpti_ApiCallbackSite site, const char *symbol = nullptr) {
    data.callbackSite = site;
    data.functionParams = params.data();
    data.functionReturnValue = &retval;
    data.symbolName = symbol;
    data.functionName = "synthetic";
    data.context = nullptr;
    return this;
  }

  void Send(CUpti_CallbackDomain domain, CUpti_CallbackId cbid) {
    if (data.functionParams == nullptr && !params.empty()) {
      data.functionParams = params.data();
    }
    if (data.functionReturnValue == nullptr) {
      data.functionReturnValue = &retval;
    }
    const void *payload =
        static_cast<int>(domain) == CUPTI_CB_DOMAIN_RESOURCE
            ? static_cast<const void *>(&resource)
            : static_cast<const void *>(&data);
    kernscope::cupti::handle_callback(domain, cbid, payload);
  }
};

inline Callback CudaMalloc(cudaMalloc_v3020_params *p, cudaError_t rc) {
  Callback cb;
  cb.params.resize(sizeof(*p));
  std::memcpy(cb.params.data(), p, sizeof(*p));
  cb.retval = rc;
  return cb;
}

inline Callback CudaFree(cudaFree_v3020_params *p, cudaError_t rc) {
  Callback cb;
  cb.params.resize(sizeof(*p));
  std::memcpy(cb.params.data(), p, sizeof(*p));
  cb.retval = rc;
  return cb;
}

inline Callback CudaMemcpy(cudaMemcpy_v3020_params *p, cudaError_t rc) {
  Callback cb;
  cb.params.resize(sizeof(*p));
  std::memcpy(cb.params.data(), p, sizeof(*p));
  cb.retval = rc;
  return cb;
}

inline Callback CudaLaunchKernel(cudaLaunchKernel_v7000_params *p) {
  Callback cb;
  cb.params.resize(sizeof(*p));
  std::memcpy(cb.params.data(), p, sizeof(*p));
  cb.retval = cudaSuccess;
  return cb;
}

inline Callback CudaStreamSynchronize(cudaStreamSynchronize_v3020_params *p,
                                      cudaError_t rc) {
  Callback cb;
  cb.params.resize(sizeof(*p));
  std::memcpy(cb.params.data(), p, sizeof(*p));
  cb.retval = rc;
  return cb;
}

inline Callback CuFuncGetParamInfo(std::uint64_t cufunc, std::size_t param_index,
                                   std::size_t offset, std::size_t size) {
  // The struct sits at the buffer head (functionParams points at it); the
  // out-param VALUES follow it and paramOffset/paramSize point at them
  // (heap-stable across Callback moves) - the driver out-param contract
  // (cuda.h: cuFuncGetParamInfo writes offset/size into caller out-params).
  Callback cb;
  cuFuncGetParamInfo_params p{};
  p.func = reinterpret_cast<CUfunction>(cufunc);
  p.paramIndex = param_index;
  cb.params.resize(sizeof(p) + 2 * sizeof(std::size_t));
  auto *buf = cb.params.data();
  std::memcpy(buf, &p, sizeof(p));
  auto *fixed = reinterpret_cast<cuFuncGetParamInfo_params *>(buf);
  fixed->paramOffset = reinterpret_cast<std::size_t *>(buf + sizeof(p));
  fixed->paramSize =
      reinterpret_cast<std::size_t *>(buf + sizeof(p) + sizeof(std::size_t));
  std::memcpy(buf + sizeof(p), &offset, sizeof(offset));
  std::memcpy(buf + sizeof(p) + sizeof(std::size_t), &size, sizeof(size));
  return cb;
}

inline Callback CuModuleGetFunction(std::uint64_t cufunc, const char *name) {
  // The struct sits at the buffer head (functionParams points at it); hfunc
  // points at the CUfunction out-VALUE stored after it (heap-stable across
  // Callback moves) - the driver out-param contract.
  Callback cb;
  cuModuleGetFunction_params p{};
  p.hmod = nullptr;
  p.name = name;
  cb.params.resize(sizeof(p) + sizeof(cufunc));
  auto *buf = cb.params.data();
  std::memcpy(buf, &p, sizeof(p));
  auto *fixed = reinterpret_cast<cuModuleGetFunction_params *>(buf);
  fixed->hfunc = reinterpret_cast<CUfunction *>(buf + sizeof(p));
  std::memcpy(buf + sizeof(p), &cufunc, sizeof(cufunc));
  cb.retval = cudaSuccess;
  return cb;
}

/// Resource-domain module load (cupti_callbacks.h: the RESOURCE payload is
/// CUpti_ResourceData, whose resourceDescriptor carries CUpti_ModuleResourceData
/// with cubinSize + pCubin).
inline Callback ModuleLoaded(const std::vector<std::uint8_t> &cubin) {
  Callback cb;
  cb.resource.resourceDescriptor = &cb.resource_module;
  cb.resource_module.moduleId = 1;
  cb.resource_module.cubinSize = cubin.size();
  cb.resource_module.pCubin = reinterpret_cast<const char *>(cubin.data());
  cb.data.functionName = "resource";
  return cb;
}

} // namespace kernscope::cupti::synthetic

#endif // KERNSCOPE_SYNTHETIC_CUPTI_H
