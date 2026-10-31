//===- capture_cupti.cpp - CUPTI subscription capture implementation -----*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Implements libkernscope_cupti_sub.so (todo 44): the NVIDIA-PRIMARY capture
// mechanism. See include/kernscope/capture_cupti.h for the mechanism
// narrative, docs/report/kernscope-technical-report.pdf for the CUPTI
// API contracts cited below, and docs/report/kernscope-technical-report.pdf for the field
// correspondence the equivalence test enforces.
//
// Recording contract ("MUST NOT alter application behavior beyond
// observation"): callbacks only READ the functionParams structs CUPTI hands
// us and record into the session; the library NEVER calls CUDA/CUPTI APIs
// from inside callbacks (the CUPTI docs warn the context can change between
// ENTER and EXIT of a runtime call). The only CUPTI entry points used here
// are the subscription/flush ones from the initialization path and the
// activity buffer callbacks - the same Kineto-shaped enable-then-flush
// pattern documented in docs/report/kernscope-technical-report.pdf
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture_cupti.h"

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kernscope::cupti {
namespace {

using kernscope::ApiKind;
using kernscope::CopyRecord;
using kernscope::KernelArgSlot;
using kernscope::LaunchRecord;
using kernscope::LaunchVia;

//===--- state ------------------------------------------------------------------===//
// One capture per process (single-tool scope, documented in the header). The
// session is the caller's session when provided (hermetic tests), else the
// process-wide CaptureSession.

struct State {
  CaptureSession *session = nullptr;
  Mechanism mech = Mechanism::kNone;
  bool live = false;
  std::atomic<bool> flushed{false};
};
State &G() {
  // Leaked on purpose: an injection library must not destroy capture state at
  // exit (same rationale as CaptureSession::instance).
  static State *s = new State();
  return *s;
}

constexpr const char *kLogTag = "[kernscope-cupti-sub] ";

void LogFallback(const char *reason) {
  if (reason != nullptr && reason[0] != '\0') {
    std::fprintf(stderr, "%s%s\n", kLogTag, reason);
  }
  std::fprintf(stderr, "%s%s\n", kLogTag, kFallbackLogLine);
}

//===--- blob output -------------------------------------------------------------===//
// Blob directory (KERNSCOPE_CUPTI_BLOB_DIR, default "."): cubin-<id>.cubin and
// input-buffer-<n>.bin - the same naming family as the todo-45 HIP path so the
// equivalence test maps them field-for-field.

std::string BlobDir() {
  const char *dir = std::getenv(kBlobDirEnv);
  return (dir != nullptr && dir[0] != '\0') ? dir : ".";
}

void WriteBlob(const std::string &name, const void *data, std::size_t size) {
  const std::string path = BlobDir() + "/" + name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (out) {
    out.write(static_cast<const char *>(data),
              static_cast<std::streamsize>(size));
  } else {
    std::fprintf(stderr, "%swarning: cannot write blob %s\n", kLogTag,
                 path.c_str());
  }
}

/// Dumps the host-side source of an H2D copy to input-buffer-<n>.bin (v0
/// scope of the Kerncap address-space closure; the equivalence test
/// hash-matches these against the HIP path's blobs). The host source is valid
/// at the API boundary; best-effort with a sanity cap so a runaway app cannot
/// fill the disk.
std::atomic<std::uint64_t> g_input_seq{0};
void DumpH2DInput(const void *src, std::size_t bytes) {
  constexpr std::size_t kMaxDumpBytes = 512u << 20; // 512 MiB sanity cap
  if (src == nullptr || bytes == 0) {
    return;
  }
  if (bytes > kMaxDumpBytes) {
    std::fprintf(stderr,
                 "%swarning: H2D input dump skipped (%zu bytes > 512 MiB "
                 "cap)\n", kLogTag, bytes);
    return;
  }
  WriteBlob("input-buffer-" + std::to_string(++g_input_seq) + ".bin", src,
            bytes);
}

//===--- param-layout registry ---------------------------------------------------===//
// Launch callbacks expose grid/block/shared/stream + the opaque args pointer
// array, but per-argument SIZES live in the kernel image and are resolved by
// cuFuncGetParamInfo (brief §5) - observed as a DRIVER-domain EXIT callback
// (the out-params are valid at EXIT). Runtime-API launches carry no
// CUfunction, so the registry is keyed by kernel name too (symbolName from
// the launch callback; CUfunction->name from cuModuleGetFunction).

struct LayoutEntry {
  std::size_t offset = 0;
  std::size_t size = 0;
};

std::mutex g_layout_mu;
std::unordered_map<std::uint64_t, std::vector<LayoutEntry>> g_layout_by_func;
std::unordered_map<std::string, std::vector<LayoutEntry>> g_layout_by_name;
std::unordered_map<std::uint64_t, std::string> g_func_names;

void LearnLayout(std::uint64_t cufunction, std::string_view kernel_name,
                 std::size_t param_index, std::size_t offset,
                 std::size_t size) {
  std::lock_guard<std::mutex> lock(g_layout_mu);
  // Key by the CUfunction value (driver-API launches resolve here)...
  if (cufunction != 0) {
    auto &v = g_layout_by_func[cufunction];
    if (v.size() <= param_index) {
      v.resize(param_index + 1);
    }
    v[param_index] = LayoutEntry{offset, size};
  }
  // ...AND by the kernel name when it is known (runtime-API launches carry
  // no CUfunction; symbolName is the lookup key). The join: cuModuleGetFunction
  // mapped the CUfunction to its name (g_func_names) before the app queried
  // the param layout - the real driver-API sequence.
  std::string name(kernel_name);
  if (name.empty() && cufunction != 0) {
    const auto it = g_func_names.find(cufunction);
    if (it != g_func_names.end()) {
      name = it->second;
    }
  }
  if (!name.empty()) {
    auto &v = g_layout_by_name[name];
    if (v.size() <= param_index) {
      v.resize(param_index + 1);
    }
    v[param_index] = LayoutEntry{offset, size};
  }
}

const std::vector<LayoutEntry> *FindLayout(std::uint64_t cufunction,
                                           std::string_view kernel_name) {
  std::lock_guard<std::mutex> lock(g_layout_mu);
  if (cufunction != 0) {
    const auto it = g_layout_by_func.find(cufunction);
    if (it != g_layout_by_func.end()) {
      return &it->second;
    }
  }
  if (!kernel_name.empty()) {
    const auto it = g_layout_by_name.find(std::string(kernel_name));
    if (it != g_layout_by_name.end()) {
      return &it->second;
    }
  }
  return nullptr;
}

/// Copies the launch argument bytes from the opaque args pointer array
/// (kernelParams convention: one pointer per parameter, each pointing at the
/// by-value parameter bytes - cudaLaunchKernel/cuLaunchKernel docs). With a
/// known layout, `size` bytes are copied from each non-null pointer - the
/// byte-exact contract the todo-43 capture-core tests assert on the driver
/// path. Without a layout the launch records args=[] (documented: runtime-API
/// launches carry no per-arg sizes at the boundary; capture.h note).
std::vector<KernelArgSlot> CaptureArgs(void *const *args,
                                       const std::vector<LayoutEntry> &layout) {
  std::vector<KernelArgSlot> slots;
  if (args == nullptr || layout.empty()) {
    return slots;
  }
  slots.reserve(layout.size());
  for (std::size_t i = 0; i < layout.size(); ++i) {
    const LayoutEntry &e = layout[i];
    if (e.size == 0) {
      continue;
    }
    KernelArgSlot slot;
    slot.size = e.size;
    if (args[i] != nullptr) {
      const auto *bytes = static_cast<const std::uint8_t *>(args[i]);
      slot.bytes.assign(bytes, bytes + slot.size);
    }
    slots.push_back(std::move(slot));
  }
  return slots;
}

//===--- runtime-API cbids (CUPTI_RUNTIME_TRACE_CBID_*, cupti_runtime_cbid.h) ----===//

void OnCudaMallocExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaMalloc);
  const auto *p = static_cast<const cudaMalloc_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  // cudaMalloc contract (cuda_runtime_api.h): returns a pointer to `size`
  // bytes of linear device memory. The result is only valid at EXIT and only
  // on success (cudaError_t pointed to by functionReturnValue).
  if (data->functionReturnValue != nullptr &&
      *static_cast<const cudaError_t *>(data->functionReturnValue) ==
          cudaSuccess &&
      p->devPtr != nullptr && *p->devPtr != nullptr) {
    G().session->record_alloc(reinterpret_cast<std::uint64_t>(*p->devPtr),
                              p->size);
  }
}

void OnCudaFreeExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaFree);
  const auto *p = static_cast<const cudaFree_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  // The session looks the pointer up in the allocation map while it is still
  // live and records an unknown-free (QA negative path) if it is not known -
  // it never crashes (cudaFree on a pointer cudaMalloc did not return is
  // undefined per the API contract).
  G().session->record_free(reinterpret_cast<std::uint64_t>(p->devPtr));
}

void OnCudaMemcpyEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  const auto *p = static_cast<const cudaMemcpy_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  // cudaMemcpyKind (12.6): 1 = HostToDevice - the host source is valid at
  // ENTER; dump it so the equivalence test can hash-match the captured input
  // buffers against the HIP path's blobs.
  if (p->kind == cudaMemcpyHostToDevice) {
    DumpH2DInput(p->src, p->count);
  }
}

void OnCudaMemcpyExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaMemcpy);
  const auto *p = static_cast<const cudaMemcpy_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  if (data->functionReturnValue != nullptr &&
      *static_cast<const cudaError_t *>(data->functionReturnValue) !=
          cudaSuccess) {
    return;
  }
  CopyRecord rec;
  rec.kind = "memcpy";
  rec.dst = reinterpret_cast<std::uint64_t>(p->dst);
  rec.src = reinterpret_cast<std::uint64_t>(p->src);
  rec.bytes = p->count;
  rec.cuda_kind = static_cast<int>(p->kind);
  G().session->record_copy(rec);
}

void OnCudaMemcpyAsyncEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  const auto *p =
      static_cast<const cudaMemcpyAsync_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  if (p->kind == cudaMemcpyHostToDevice) {
    DumpH2DInput(p->src, p->count);
  }
}

void OnCudaMemcpyAsyncExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaMemcpyAsync);
  const auto *p =
      static_cast<const cudaMemcpyAsync_v3020_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  if (data->functionReturnValue != nullptr &&
      *static_cast<const cudaError_t *>(data->functionReturnValue) !=
          cudaSuccess) {
    return;
  }
  CopyRecord rec;
  rec.kind = "memcpy-async";
  rec.dst = reinterpret_cast<std::uint64_t>(p->dst);
  rec.src = reinterpret_cast<std::uint64_t>(p->src);
  rec.bytes = p->count;
  rec.cuda_kind = static_cast<int>(p->kind);
  rec.stream = reinterpret_cast<std::uint64_t>(p->stream);
  G().session->record_copy(rec);
}

void OnCudaLaunchKernelEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaLaunchKernel);
  const auto *p =
      static_cast<const cudaLaunchKernel_v7000_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  // CUpti_CallbackData.symbolName is valid on launch callbacks (brief §1):
  // the runtime-resolved kernel name (demangled).
  const std::string name =
      data->symbolName != nullptr ? data->symbolName : "";
  LaunchRecord launch;
  launch.kernel_name = name;
  launch.via = LaunchVia::kRuntimeApi;
  launch.grid_x = p->gridDim.x;
  launch.grid_y = p->gridDim.y;
  launch.grid_z = p->gridDim.z;
  launch.block_x = p->blockDim.x;
  launch.block_y = p->blockDim.y;
  launch.block_z = p->blockDim.z;
  launch.shared_mem_bytes = static_cast<std::uint32_t>(p->sharedMem);
  launch.stream = reinterpret_cast<std::uint64_t>(p->stream);
  // Byte-exact args via the learned param layout (cuFuncGetParamInfo EXIT
  // callbacks); empty when no layout is known (documented).
  if (const auto *layout = FindLayout(0, name)) {
    launch.args = CaptureArgs(p->args, *layout);
  }
  G().session->record_launch(std::move(launch));
}

void OnCudaLaunchKernelExCEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaLaunchKernelExC);
  const auto *p = static_cast<const cudaLaunchKernelExC_v11060_params *>(
      data->functionParams);
  if (p == nullptr || p->config == nullptr) {
    return;
  }
  const cudaLaunchConfig_t &cfg = *p->config;
  const std::string name =
      data->symbolName != nullptr ? data->symbolName : "";
  LaunchRecord launch;
  launch.kernel_name = name;
  launch.via = LaunchVia::kRuntimeApiEx;
  launch.grid_x = cfg.gridDim.x;
  launch.grid_y = cfg.gridDim.y;
  launch.grid_z = cfg.gridDim.z;
  launch.block_x = cfg.blockDim.x;
  launch.block_y = cfg.blockDim.y;
  launch.block_z = cfg.blockDim.z;
  launch.shared_mem_bytes = static_cast<std::uint32_t>(cfg.dynamicSmemBytes);
  launch.stream = reinterpret_cast<std::uint64_t>(cfg.stream);
  if (const auto *layout = FindLayout(0, name)) {
    launch.args = CaptureArgs(p->args, *layout);
  }
  G().session->record_launch(std::move(launch));
}

void OnCudaStreamCreateExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaStreamCreate);
  const auto *p =
      static_cast<const cudaStreamCreate_v3020_params *>(data->functionParams);
  if (p == nullptr || p->pStream == nullptr) {
    return;
  }
  if (data->functionReturnValue != nullptr &&
      *static_cast<const cudaError_t *>(data->functionReturnValue) ==
          cudaSuccess) {
    G().session->record_stream_create(
        reinterpret_cast<std::uint64_t>(*p->pStream));
  }
}

void OnCudaStreamSynchronize(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCudaStreamSynchronize);
  const auto *p = static_cast<const cudaStreamSynchronize_v3020_params *>(
      data->functionParams);
  if (p == nullptr) {
    return;
  }
  G().session->record_stream_sync(reinterpret_cast<std::uint64_t>(p->stream));
}

//===--- driver-API cbids (CUPTI_DRIVER_TRACE_CBID_*, cupti_driver_cbid.h) -------===//

void OnCuLaunchKernelEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCuLaunchKernel);
  const auto *p = static_cast<const cuLaunchKernel_params *>(data->functionParams);
  if (p == nullptr) {
    return;
  }
  std::string name = data->symbolName != nullptr ? data->symbolName : "";
  if (name.empty()) {
    std::lock_guard<std::mutex> lock(g_layout_mu);
    const auto it = g_func_names.find(reinterpret_cast<std::uint64_t>(p->f));
    if (it != g_func_names.end()) {
      name = it->second;
    }
  }
  LaunchRecord launch;
  launch.kernel_name = name;
  launch.via = LaunchVia::kDriverApi;
  launch.grid_x = p->gridDimX;
  launch.grid_y = p->gridDimY;
  launch.grid_z = p->gridDimZ;
  launch.block_x = p->blockDimX;
  launch.block_y = p->blockDimY;
  launch.block_z = p->blockDimZ;
  launch.shared_mem_bytes = p->sharedMemBytes;
  launch.stream = reinterpret_cast<std::uint64_t>(p->hStream);
  if (const auto *layout =
          FindLayout(reinterpret_cast<std::uint64_t>(p->f), name)) {
    launch.args = CaptureArgs(p->kernelParams, *layout);
  }
  G().session->record_launch(std::move(launch));
}

void OnCuLaunchKernelExEnter(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCuLaunchKernelEx);
  const auto *p = static_cast<const cuLaunchKernelEx_params *>(data->functionParams);
  if (p == nullptr || p->config == nullptr) {
    return;
  }
  std::string name = data->symbolName != nullptr ? data->symbolName : "";
  if (name.empty()) {
    std::lock_guard<std::mutex> lock(g_layout_mu);
    const auto it = g_func_names.find(reinterpret_cast<std::uint64_t>(p->f));
    if (it != g_func_names.end()) {
      name = it->second;
    }
  }
  const CUlaunchConfig &cfg = *p->config;
  LaunchRecord launch;
  launch.kernel_name = name;
  launch.via = LaunchVia::kDriverApiEx;
  launch.grid_x = cfg.gridDimX;
  launch.grid_y = cfg.gridDimY;
  launch.grid_z = cfg.gridDimZ;
  launch.block_x = cfg.blockDimX;
  launch.block_y = cfg.blockDimY;
  launch.block_z = cfg.blockDimZ;
  launch.shared_mem_bytes = cfg.sharedMemBytes;
  launch.stream = reinterpret_cast<std::uint64_t>(cfg.hStream);
  if (const auto *layout =
          FindLayout(reinterpret_cast<std::uint64_t>(p->f), name)) {
    launch.args = CaptureArgs(p->kernelParams, *layout);
  }
  G().session->record_launch(std::move(launch));
}

void OnCuFuncGetParamInfoExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  const auto *p =
      static_cast<const cuFuncGetParamInfo_params *>(data->functionParams);
  if (p == nullptr || p->paramOffset == nullptr || p->paramSize == nullptr) {
    return;
  }
  // cuFuncGetParamInfo contract (cuda.h): returns the offset and size of
  // parameter `paramIndex` of `func` - the driver's param-layout resolver.
  // The out-params are valid at EXIT. This is the documented source of the
  // per-argument sizes the launch callbacks merge (brief §5).
  G().session->record_param_info(reinterpret_cast<std::uint64_t>(p->func),
                                 p->paramIndex, *p->paramOffset, *p->paramSize);
  LearnLayout(reinterpret_cast<std::uint64_t>(p->func), {}, p->paramIndex,
              *p->paramOffset, *p->paramSize);
}

void OnCuModuleGetFunctionExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCuModuleGetFunction);
  const auto *p =
      static_cast<const cuModuleGetFunction_params *>(data->functionParams);
  if (p == nullptr || p->hfunc == nullptr || p->name == nullptr) {
    return;
  }
  if (data->functionReturnValue != nullptr &&
      *static_cast<const CUresult *>(data->functionReturnValue) !=
          CUDA_SUCCESS) {
    return;
  }
  const std::string name = p->name;
  G().session->register_cufunction(reinterpret_cast<std::uint64_t>(*p->hfunc),
                                   name);
  {
    std::lock_guard<std::mutex> lock(g_layout_mu);
    g_func_names[reinterpret_cast<std::uint64_t>(*p->hfunc)] = name;
  }
}

void OnCuModuleLoadExit(const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  G().session->note_api(ApiKind::kCuModuleLoad);
  const auto *p = static_cast<const cuModuleLoad_params *>(data->functionParams);
  if (p == nullptr || p->fname == nullptr) {
    return;
  }
  if (data->functionReturnValue != nullptr &&
      *static_cast<const CUresult *>(data->functionReturnValue) !=
          CUDA_SUCCESS) {
    return;
  }
  // cuModuleLoad file path: hash the file ourselves ("" sha = unreadable).
  std::ifstream fh(p->fname, std::ios::binary);
  if (fh) {
    std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(fh)),
                                    std::istreambuf_iterator<char>());
    if (!bytes.empty()) {
      const std::uint64_t mod = G().session->register_cubin(bytes);
      WriteBlob("cubin-" + std::to_string(mod) + ".cubin", bytes.data(),
                bytes.size());
    }
  } else {
    (void)G().session->register_module_file(p->fname);
  }
}

//===--- RESOURCE domain ----------------------------------------------------------===//
// CUPTI_CBID_RESOURCE_MODULE_LOADED: the callback payload for the RESOURCE
// domain is CUpti_ResourceData, whose resourceDescriptor carries
// CUpti_ModuleResourceData (cubin bytes + size, cupti_callbacks.h) - the
// code-capture path (brief §5).

void OnModuleLoaded(const void *raw) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  if (raw == nullptr) {
    return;
  }
  const auto *res = static_cast<const CUpti_ResourceData *>(raw);
  if (res->resourceDescriptor == nullptr) {
    return;
  }
  const auto *mod =
      static_cast<const CUpti_ModuleResourceData *>(res->resourceDescriptor);
  if (mod->pCubin == nullptr || mod->cubinSize == 0) {
    return;
  }
  const std::uint64_t reg = G().session->register_cubin(
      std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(mod->pCubin),
          mod->cubinSize));
  WriteBlob("cubin-" + std::to_string(reg) + ".cubin", mod->pCubin,
            mod->cubinSize);
}

//===--- the dispatcher -----------------------------------------------------------===//

void DispatchRuntime(CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  switch (static_cast<int>(cbid)) {
  case CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCudaMallocExit(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCudaFreeExit(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCudaMemcpyEnter(data);
    } else {
      OnCudaMemcpyExit(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCudaMemcpyAsyncEnter(data);
    } else {
      OnCudaMemcpyAsyncExit(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCudaLaunchKernelEnter(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCudaLaunchKernelExCEnter(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCudaStreamCreateExit(data);
    }
    break;
  case CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020:
    OnCudaStreamSynchronize(data);
    break;
  default:
    break;
  }
}

void DispatchDriver(CUpti_CallbackId cbid, const CUpti_CallbackData *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  switch (static_cast<int>(cbid)) {
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCuLaunchKernelEnter(data);
    }
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
    if (data->callbackSite == CUPTI_API_ENTER) {
      OnCuLaunchKernelExEnter(data);
    }
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCuFuncGetParamInfoExit(data);
    }
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCuModuleGetFunctionExit(data);
    }
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData:
    // cuModuleLoadData(Ex) deliver the cubin image to the driver at EXIT; the
    // bytes are captured by the RESOURCE module-loaded callback instead (it
    // carries cubinSize, which the driver-API boundary does not expose for
    // in-memory images). Here we only note the interception-table entry.
    G().session->note_api(ApiKind::kCuModuleLoadData);
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx:
    G().session->note_api(ApiKind::kCuModuleLoadDataEx);
    break;
  case CUPTI_DRIVER_TRACE_CBID_cuModuleLoad:
    if (data->callbackSite == CUPTI_API_EXIT) {
      OnCuModuleLoadExit(data);
    }
    break;
  default:
    break;
  }
}

void DispatchResource(CUpti_CallbackId cbid, const void *data) {
  if (G().session == nullptr) return;  // never crash a host app: callbacks may fire before session assignment
  switch (static_cast<int>(cbid)) {
  case CUPTI_CBID_RESOURCE_MODULE_LOADED:
    OnModuleLoaded(data);
    break;
  default:
    break;
  }
}

} // namespace

//===--- public API ---------------------------------------------------------------===//

CaptureSession &session() {
  if (G().session != nullptr) {
    return *G().session;
  }
  return CaptureSession::instance();
}

bool initialized() { return G().live; }

Mechanism mechanism() { return G().mech; }

void arm_session(CaptureSession *session) {
  G().session = session != nullptr ? session : &CaptureSession::instance();
}

void handle_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                     const void *data) {
  if (data == nullptr || G().session == nullptr) {
    // Not recording into a session (no try_initialize yet): the synthetic
    // harness/tests always initialize first; on real hardware CUPTI cannot
    // deliver callbacks before our subscription exists either.
    return;
  }
  // The RESOURCE domain delivers CUpti_ResourceData through the callback
  // payload slot; the RUNTIME/DRIVER domains deliver CUpti_CallbackData
  // (cupti_callbacks.h).
  if (static_cast<int>(domain) == CUPTI_CB_DOMAIN_RESOURCE) {
    DispatchResource(cbid, data);
    return;
  }
  const auto *cbdata = static_cast<const CUpti_CallbackData *>(data);
  switch (static_cast<int>(domain)) {
  case CUPTI_CB_DOMAIN_RUNTIME_API:
    DispatchRuntime(cbid, cbdata);
    break;
  case CUPTI_CB_DOMAIN_DRIVER_API:
    DispatchDriver(cbid, cbdata);
    break;
  default:
    break;
  }
}

void handle_activity_record(const CUpti_Activity *record) {
  if (record == nullptr || G().session == nullptr) {
    return;
  }
  if (record->kind != CUPTI_ACTIVITY_KIND_MEMORY) {
    return; // the capture only consumes Memory records; todo 49's profiler
            // consumes the other kinds through this same library
  }
  const auto *mem = reinterpret_cast<const CUpti_ActivityMemory4 *>(record);
  // The allocation map is the device-pointer -> size learner (brief §5
  // heuristic 2: "CUPTI CUpti_ActivityMemory4 gives alloc/free with address
  // and bytes without any interception"). Only DEVICE allocations feed the
  // map.
  if (mem->memoryKind != CUPTI_ACTIVITY_MEMORY_KIND_DEVICE) {
    return;
  }
  switch (mem->memoryOperationType) {
  case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION:
    G().session->record_alloc(mem->address,
                              static_cast<std::size_t>(mem->bytes));
    break;
  case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE:
    G().session->record_free(mem->address);
    break;
  default:
    break;
  }
}

//===--- subscription establishment (dlopen; no link dependency on libcupti) ------===//

namespace {

using CuptiSubscribeFn = CUptiResult(CUPTIAPI *)(CUpti_SubscriberHandle *,
                                                 CUpti_CallbackFunc, void *);
using CuptiEnableCallbackFn = CUptiResult(CUPTIAPI *)(
    uint32_t, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId);
using CuptiActivityRegisterCallbacksFn = CUptiResult(CUPTIAPI *)(
    CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc);
using CuptiActivityEnableFn = CUptiResult(CUPTIAPI *)(CUpti_ActivityKind);
using CuptiActivityFlushAllFn = CUptiResult(CUPTIAPI *)(uint32_t);
using CuptiActivityGetNextRecordFn = CUptiResult(CUPTIAPI *)(
    uint8_t *, size_t, CUpti_Activity **);
using CuptiUnsubscribeFn = CUptiResult(CUPTIAPI *)(CUpti_SubscriberHandle);

struct CuptiApi {
  CuptiSubscribeFn subscribe = nullptr;
  CuptiEnableCallbackFn enable_callback = nullptr;
  CuptiActivityRegisterCallbacksFn activity_register_callbacks = nullptr;
  CuptiActivityEnableFn activity_enable = nullptr;
  CuptiActivityFlushAllFn activity_flush_all = nullptr;
  CuptiActivityGetNextRecordFn activity_get_next_record = nullptr;
  CuptiUnsubscribeFn unsubscribe = nullptr;
};

CuptiApi g_cupti{}; // resolved once by try_initialize
void *g_cupti_handle = nullptr;
CUpti_SubscriberHandle g_subscriber = nullptr;

// Activity buffer contract (cupti_activity.h): CUPTI requests buffers from
// the tool (BufferRequested) and hands completed ones back (BufferCompleted);
// records iterate via cuptiActivityGetNextRecord. 8 MiB buffers, the
// sample-scale default (cuptisamples activity_trace_async).
void CUPTIAPI BufferRequested(std::uint8_t **buffer, std::size_t *size,
                              std::size_t *max_num_records) {
  constexpr std::size_t kBufferBytes = 8u << 20;
  *buffer = static_cast<std::uint8_t *>(std::malloc(kBufferBytes));
  if (*buffer == nullptr) {
    *size = 0;
    *max_num_records = 0;
    return;
  }
  *size = kBufferBytes;
  *max_num_records = 0; // CUPTI infers from the buffer size
}

void CUPTIAPI BufferCompleted(CUcontext /*context*/, CUpti_CallbackId /*cbid*/,
                              std::uint8_t *buffer, std::size_t /*size*/,
                              std::size_t valid_size) {
  if (buffer == nullptr || valid_size == 0 ||
      g_cupti.activity_get_next_record == nullptr) {
    std::free(buffer);
    return;
  }
  // 12.6 iteration contract (cupti_activity.h): cuptiActivityGetNextRecord
  // takes the buffer + valid byte count and advances an internal record
  // cursor; stop when it reports CUPTI_ERROR_MAX_LIMIT_REACHED (or any
  // other error - no record arithmetic, no record->size field in 12.6).
  CUpti_Activity *record = nullptr;
  for (;;) {
    const CUptiResult rc =
        g_cupti.activity_get_next_record(buffer, valid_size, &record);
    if (rc != CUPTI_SUCCESS) {
      break;
    }
    handle_activity_record(record);
  }
  std::free(buffer);
}

// The CUPTI callback wrapper: the real CUPTI delivers
// (userdata, domain, cbid, cbdata) with cbdata typed `const void*` (the
// per-domain payload structs: CUpti_CallbackData / CUpti_ResourceData).
void CUPTIAPI CuptiCallback(void * /*userdata*/, CUpti_CallbackDomain domain,
                            CUpti_CallbackId cbid, const void *cbdata) {
  handle_callback(domain, cbid, cbdata);
}

constexpr struct {
  CUpti_CallbackDomain domain;
  CUpti_CallbackId cbid;
} kEnabledCallbacks[] = {
    // RUNTIME domain (cupti_runtime_cbid.h)
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMalloc_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaFree_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API,
     CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API,
     CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000},
    {CUPTI_CB_DOMAIN_RUNTIME_API,
     CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernelExC_v11060},
    {CUPTI_CB_DOMAIN_RUNTIME_API, CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreate_v3020},
    {CUPTI_CB_DOMAIN_RUNTIME_API,
     CUPTI_RUNTIME_TRACE_CBID_cudaStreamSynchronize_v3020},
    // DRIVER domain (cupti_driver_cbid.h)
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuFuncGetParamInfo},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleGetFunction},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadData},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoadDataEx},
    {CUPTI_CB_DOMAIN_DRIVER_API, CUPTI_DRIVER_TRACE_CBID_cuModuleLoad},
    // RESOURCE domain (cupti_callbacks.h)
    {CUPTI_CB_DOMAIN_RESOURCE, CUPTI_CBID_RESOURCE_MODULE_LOADED},
};

const char *CuptiErrorString(CUptiResult rc) {
  // Small local table of the errors the subscription path can hit
  // (cupti_result.h).
  switch (rc) {
  case CUPTI_ERROR_NOT_INITIALIZED: return "CUPTI_ERROR_NOT_INITIALIZED";
  case CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED:
    return "CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED";
  case CUPTI_ERROR_INVALID_PARAMETER: return "CUPTI_ERROR_INVALID_PARAMETER";
  case CUPTI_ERROR_INVALID_CONTEXT: return "CUPTI_ERROR_INVALID_CONTEXT";
  default: return "CUPTI_ERROR";
  }
}

} // namespace

namespace {
bool try_initialize(CaptureSession *session) {
  if (G().live) {
    return true;
  }
  // --- locate + dlopen libcupti.so (NOT a link dependency: the injection
  // contract and the driverless negative path both require graceful
  // degradation when CUPTI is absent). Candidates: CUPTI_HOME/lib/libcupti.so
  // (devenv), then the loader's search path.
  if (g_cupti_handle == nullptr) {
    std::string home_path;
    if (const char *home = std::getenv("CUPTI_HOME");
        home != nullptr && home[0] != '\0') {
      home_path = std::string(home) + "/lib/libcupti.so";
    }
    const char *candidates[] = {home_path.c_str(), "libcupti.so.12",
                                "libcupti.so"};
    for (const char *c : candidates) {
      if (c == nullptr || c[0] == '\0') {
        continue;
      }
      if (void *h = dlopen(c, RTLD_NOW | RTLD_LOCAL)) {
        g_cupti_handle = h;
        break;
      }
    }
    if (g_cupti_handle == nullptr) {
      LogFallback("cannot load libcupti.so (CUPTI_HOME unset or no CUPTI on "
                  "the load path)");
      return false;
    }
    g_cupti.subscribe = reinterpret_cast<CuptiSubscribeFn>(
        dlsym(g_cupti_handle, "cuptiSubscribe"));
    g_cupti.enable_callback = reinterpret_cast<CuptiEnableCallbackFn>(
        dlsym(g_cupti_handle, "cuptiEnableCallback"));
    g_cupti.activity_register_callbacks =
        reinterpret_cast<CuptiActivityRegisterCallbacksFn>(
            dlsym(g_cupti_handle, "cuptiActivityRegisterCallbacks"));
    g_cupti.activity_enable = reinterpret_cast<CuptiActivityEnableFn>(
        dlsym(g_cupti_handle, "cuptiActivityEnable"));
    g_cupti.activity_flush_all = reinterpret_cast<CuptiActivityFlushAllFn>(
        dlsym(g_cupti_handle, "cuptiActivityFlushAll"));
    g_cupti.activity_get_next_record =
        reinterpret_cast<CuptiActivityGetNextRecordFn>(
            dlsym(g_cupti_handle, "cuptiActivityGetNextRecord"));
    g_cupti.unsubscribe = reinterpret_cast<CuptiUnsubscribeFn>(
        dlsym(g_cupti_handle, "cuptiUnsubscribe"));
    if (g_cupti.subscribe == nullptr || g_cupti.enable_callback == nullptr) {
      LogFallback("libcupti.so loaded but cuptiSubscribe/cuptiEnableCallback "
                  "symbols missing");
      return false;
    }
  }

  G().session = session != nullptr ? session : &CaptureSession::instance();

  // --- subscribe (cuptiSubscribe contract, brief §1: callback API) ---------
  CUptiResult rc = g_cupti.subscribe(&g_subscriber, &CuptiCallback, nullptr);
  if (rc == CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED) {
    LogFallback("another CUPTI subscriber is active (single-tool scope; "
                "CUPTI 13.3+ cuptiSubscribe_v2 is the documented migration)");
    return false;
  }
  if (rc != CUPTI_SUCCESS) {
    LogFallback("cuptiSubscribe failed (no NVIDIA driver on this host?)");
    return false;
  }

  // --- enable the capture cbids (cuptiEnableCallback contract) --------------
  for (const auto &e : kEnabledCallbacks) {
    rc = g_cupti.enable_callback(1, g_subscriber, e.domain, e.cbid);
    if (rc != CUPTI_SUCCESS) {
      std::fprintf(stderr,
                   "%swarning: cuptiEnableCallback(domain=%d cbid=%d) "
                   "failed: %s; continuing with the remaining cbids\n",
                   kLogTag, static_cast<int>(e.domain),
                   static_cast<int>(e.cbid), CuptiErrorString(rc));
    }
  }

  // --- activity buffers: Memory records feed the allocation map ------------
  // (kineto-shaped enable-then-flush, profiler-research-brief.md §1.3).
  if (g_cupti.activity_register_callbacks != nullptr &&
      g_cupti.activity_enable != nullptr) {
    (void)g_cupti.activity_register_callbacks(&BufferRequested,
                                              &BufferCompleted);
    (void)g_cupti.activity_enable(CUPTI_ACTIVITY_KIND_MEMORY);
  } else {
    std::fprintf(stderr,
                 "%swarning: activity APIs missing; allocation sizes will "
                 "not be learned from Memory records\n",
                 kLogTag);
  }

  G().live = true;
  G().mech = Mechanism::kCuptiLinked;
  std::atexit([] { flush_bundle(); });
  std::fprintf(stderr,
               "%sinfo: CUPTI subscription live (RUNTIME+DRIVER+RESOURCE "
               "callbacks + MEMORY activity records)\n",
               kLogTag);
  return true;
}
} // namespace

bool try_initialize() { return try_initialize(nullptr); }

bool initialize_from_injection() {
  const bool ok = try_initialize();
  if (ok) {
    G().mech = Mechanism::kCuptiInjected;
  }
  return ok;
}

void flush_bundle() {
  if (!G().live || G().flushed.exchange(true)) {
    return;
  }
  const char *path = std::getenv(kBundleEnv);
  if (path == nullptr || path[0] == '\0') {
    path = std::getenv(kCaptureLogEnv);
  }
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  CaptureSession &s = session();
  if (s.empty()) {
    return; // never litter for processes that made no CUDA calls
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    std::fprintf(stderr, "%swarning: cannot write bundle %s\n", kLogTag, path);
    return;
  }
  kernscope::BundleWriter::create_v0()->write(s, out);
  std::fprintf(stderr, "%sinfo: bundle written to %s\n", kLogTag, path);
  // Forced flush of any remaining activity buffers (session end semantics).
  if (g_cupti.activity_flush_all != nullptr) {
    (void)g_cupti.activity_flush_all(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  }
}

} // namespace kernscope::cupti
