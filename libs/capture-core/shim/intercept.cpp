//===- intercept.cpp - CUDA LD_PRELOAD interception shim ---------*- C++ -*-===//
//
// libkernscope_intercept.so: the todo-43 CUDA interception shim. Preloaded
// via LD_PRELOAD, it interposes the FULL API TABLE (capture.h) and records
// every observable call into libs/capture-core's CaptureSession, then
// forwards to the real implementation through dlsym(RTLD_NEXT). Compiled
// against the CUDA 12.6 headers (SYSTEM include) so every interposed
// signature is checked against the real header declarations at build time.
//
// KNOWN LIMITATIONS OF THE LD_PRELOAD MECHANISM - recorded here per
// docs/research/capture-replay-research-brief.md §2 (pitfalls 1-5; the
// PyTorch/TensorFlow precedent). These are exactly why todo 44's CUPTI
// callback/injection path is the NVIDIA-PRIMARY mechanism and this shim is
// NOT:
//   1. cuGetProcAddress bypass (pitfall 1): apps may fetch driver entry
//      points BY ADDRESS (CUDA Driver API, "Driver Entry Point Access"),
//      which never goes through our PLT-level symbols. CUPTI injection hooks
//      from inside the driver and has no such hole.
//   2. static cudart (pitfall 5): `-cudart static` apps never bind the
//      dynamic symbols this shim interposes on; CUPTI injection still sees
//      every launch.
//   3. driver-vs-runtime split (pitfall 1): we intercept BOTH layers and
//      dedup the runtime launch that the runtime re-dispatches through the
//      driver (thread-local flag around the runtime forward). A runtime that
//      dispatches via an INTERNAL (non-public) driver entry point still shows
//      up once - through our runtime-API record - so records stay truthful.
//   4. symbol-level coverage only: per-thread-default-stream variants
//      (cudaLaunchKernel_ptsz / cuLaunchKernel_ptsz) are not interposed.
//   5. init-order (pitfall 3): this shim's constructor runs at load and
//      touches no CUDA state; session state is created on first use.
//   6. arg introspection is driver-API-only: runtime-API launches
//      (cudaLaunchKernel / cudaLaunchKernelExC) carry an opaque void** args
//      array with NO per-argument sizes at the boundary (sizes live in the
//      kernel image; cuFuncGetParamInfo resolves them for the driver path).
//      Byte-exact runtime-path arg capture is a todo-44 CUPTI functionParams
//      concern; the mock scenario in the unit tests asserts the byte-exact
//      path the driver-API record takes.
//   7. cubin sizing: cuModuleLoadData gives no image length; we compute it
//      from the ELF section headers (best-effort, guarded); unparsable
//      images degrade to a 64-byte prefix hash. todo 44's CUPTI module
//      resource callback provides exact bytes.
//
// "MUST NOT modify application behavior beyond observation": every wrapper
// forwards the ORIGINAL arguments to the real function and returns its exact
// result; recording happens around the forward. When the real symbol is
// absent (no CUDA runtime loaded - the mock smoke scenario), the wrapper
// returns the typed no-device error (cudaErrorNoDevice / CUDA_ERROR_NO_DEVICE)
// so the app sees a first-class error instead of a crash.
//
// Build contract: statically linked against libs/capture-core (no runtime
// dependency - preload ordering is unaffected); the shim's own symbols are
// hidden except the interposed table and kernscope_intercept_flush.
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <dlfcn.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

//===--- dlsym(RTLD_NEXT) resolution (lazy + cached) -------------------------===//
// Pitfall 2 in the brief: "must resolve the real symbol lazily and cache it".
// Each slot caches the real function pointer; a benign benign race between
// threads stores the same value (idempotent). RTLD_NEXT searches objects
// loaded AFTER this shim (the app, libcudart/libcuda), so a preloaded shim
// never resolves to itself.

using RealFn = void (*)();

RealFn ResolveSymbol(const char *name, std::atomic<RealFn> &slot) {
  RealFn p = slot.load(std::memory_order_acquire);
  if (p == nullptr) {
    void *sym = dlsym(RTLD_NEXT, name);
    p = reinterpret_cast<RealFn>(sym);
    slot.store(p, std::memory_order_release);
  }
  return p;
}

#define KERNSCOPE_REAL_SLOT(name)                                              \
  std::atomic<RealFn> g_real_##name{nullptr}
#define KERNSCOPE_REAL(name)                                                   \
  reinterpret_cast<decltype(&name)>(ResolveSymbol(#name, g_real_##name))

// The registration symbols are NOT declared by the CUDA 12.6 public headers
// (see their wrappers below), so their real pointers are resolved with typed
// casts against dlsym names that carry the __ prefix.
#define KERNSCOPE_REAL_TYPED(slot, cpp_type, dlsym_name)                       \
  reinterpret_cast<cpp_type>(ResolveSymbol(dlsym_name, slot))

//===--- interception state --------------------------------------------------===//

kernscope::CaptureSession &Session() {
  return kernscope::CaptureSession::instance();
}

void RecordApi(kernscope::ApiKind kind) { Session().note_api(kind); }

/// Thread-local flag: set while a runtime-API launch forwards, so the
/// driver-API wrapper can skip the duplicate record of the runtime's own
/// re-dispatch through libcuda (pitfall 1, "lower choke point" note above).
thread_local bool g_in_runtime_launch = false;

constexpr std::size_t kMaxArgs = 64;         // driver-path arg introspection cap
constexpr std::size_t kMaxArgBytes = 1 << 20; // per-arg byte cap (sanity)
constexpr std::size_t kMaxCubinBytes = 1 << 30; // cubin size cap (sanity)

//===--- fatbin parsing (GPGPU-Sim pattern, brief §2) ------------------------===//
// __cudaRegisterFatBinary receives a pointer to a __fatBinCHeader (magic
// 0x466243B1) whose `data` points at a __cudaFatCudaBinary (magic 0x466243B2)
// with an entries array of {type, binary, binarySize}; type 1 = cubin. The
// shim concatenates the cubin payloads and hashes them - this is the bundle's
// "cubin field" source for runtime-API apps. Every dereference is
// magic-guarded; an unparsable image degrades to a hash of the raw header
// bytes (documented limitation).

struct FatBinCHeader {
  int magic; // 0x466243B1
  int version;
  const void *data;
  void *(*f)(void *);
};

struct FatCudaBinary {
  int magic; // 0x466243B2
  int version;
  const unsigned long long *cubin;
  const void *entries;
  void *(*getEntry)(const void *);
  void **bundle;
  unsigned int count;
};

struct FatCudaBinaryEntry {
  unsigned int type; // 1 = cubin
  const void *binary;
  unsigned int binarySize;
};

constexpr std::uint32_t kFatBinMagic = 0x466243B1;
constexpr std::uint32_t kFatCudaMagic = 0x466243B2;

std::vector<std::uint8_t> HeaderBytes(const FatBinCHeader *hdr) {
  const auto *raw = reinterpret_cast<const std::uint8_t *>(hdr);
  return std::vector<std::uint8_t>(raw, raw + sizeof(FatBinCHeader));
}

std::vector<std::uint8_t> FatBinCubinBytes(void *fatCubin) {
  if (fatCubin == nullptr) {
    return {};
  }
  const auto *hdr = static_cast<const FatBinCHeader *>(fatCubin);
  if (hdr->magic != kFatBinMagic || hdr->data == nullptr) {
    return HeaderBytes(hdr);
  }
  const auto *fb = static_cast<const FatCudaBinary *>(hdr->data);
  if (fb->magic != kFatCudaMagic || fb->entries == nullptr || fb->count == 0 ||
      fb->count > kMaxArgs) {
    return HeaderBytes(hdr);
  }
  const auto *entry = static_cast<const FatCudaBinaryEntry *>(fb->entries);
  std::vector<std::uint8_t> blob;
  for (unsigned int i = 0; i < fb->count; ++i) {
    if (entry[i].type != 1 || entry[i].binary == nullptr ||
        entry[i].binarySize == 0 || entry[i].binarySize > kMaxCubinBytes) {
      continue;
    }
    const auto *bytes = static_cast<const std::uint8_t *>(entry[i].binary);
    blob.insert(blob.end(), bytes, bytes + entry[i].binarySize);
  }
  if (blob.empty()) {
    return HeaderBytes(hdr);
  }
  return blob;
}

//===--- cubin sizing via the ELF section table (limitation 7) ----------------===//
// cuModuleLoadData has no image length at the boundary; the image IS an ELF
// file, so the file size = max(sh_offset + sh_size) over the section table.
// All fields are read from the caller's image buffer with sanity caps.

std::size_t CubinImageSize(const std::uint8_t *base) {
  if (base == nullptr || base[0] != 0x7F || base[1] != 'E' || base[2] != 'L' ||
      base[3] != 'F') {
    return 0;
  }
  const std::uint8_t elf_class = base[4]; // 1 = ELF32, 2 = ELF64
  if (base[5] != 1) {                     // little-endian only
    return 0;
  }
  std::size_t shoff = 0, shentsize = 0, shnum = 0;
  if (elf_class == 2) { // ELF64: e_shoff@0x28(8), e_shentsize@0x3A(2), e_shnum@0x3C(2)
    std::memcpy(&shoff, base + 0x28, 8);
    std::uint16_t a = 0, b = 0;
    std::memcpy(&a, base + 0x3A, 2);
    std::memcpy(&b, base + 0x3C, 2);
    shentsize = a;
    shnum = b;
  } else if (elf_class == 1) { // ELF32: e_shoff@0x20(4), e_shentsize@0x2E(2), e_shnum@0x30(2)
    std::uint32_t a = 0;
    std::memcpy(&a, base + 0x20, 4);
    shoff = a;
    std::uint16_t c = 0, d = 0;
    std::memcpy(&c, base + 0x2E, 2);
    std::memcpy(&d, base + 0x30, 2);
    shentsize = c;
    shnum = d;
  } else {
    return 0;
  }
  const std::size_t min_shentsize = elf_class == 2 ? 64 : 40;
  if (shentsize < min_shentsize || shnum == 0 || shnum > 4096 ||
      shoff > kMaxCubinBytes) {
    return 0;
  }
  std::size_t size = shoff + shnum * shentsize;
  for (std::size_t i = 0; i < shnum; ++i) {
    const std::uint8_t *sh = base + shoff + i * shentsize;
    std::size_t sh_offset = 0, sh_size = 0;
    if (elf_class == 2) { // sh_offset@0x18(8), sh_size@0x20(8)
      std::memcpy(&sh_offset, sh + 0x18, 8);
      std::memcpy(&sh_size, sh + 0x20, 8);
    } else {
      std::uint32_t a = 0, b = 0;
      std::memcpy(&a, sh + 0x10, 4);
      std::memcpy(&b, sh + 0x14, 4);
      sh_offset = a;
      sh_size = b;
    }
    if (sh_offset <= kMaxCubinBytes && sh_size <= kMaxCubinBytes) {
      const std::size_t end = sh_offset + sh_size;
      if (end > size) {
        size = end;
      }
    }
  }
  return size > kMaxCubinBytes ? 0 : size;
}

std::vector<std::uint8_t> CubinBytes(const void *image) {
  const auto *base = static_cast<const std::uint8_t *>(image);
  if (base == nullptr) {
    return {};
  }
  const std::size_t size = CubinImageSize(base);
  if (size == 0) {
    // Unparsable: degrade to the 64-byte prefix hash (limitation 7).
    return std::vector<std::uint8_t>(base, base + 64);
  }
  return std::vector<std::uint8_t>(base, base + size);
}

//===--- fatbin handle indirection -------------------------------------------===//
// The app receives OUR handle from __cudaRegisterFatBinary and passes it back
// to __cudaRegisterFunction; the real runtime must receive the handle IT
// produced, so the slot carries both (the standard registration-handle
// indirection used by cuda-intercept-style shims). Slots are leaked: a
// registration handle must stay valid for the whole process.

struct FatBinSlot {
  void **real_handle = nullptr;
  std::uint64_t reg_id = 0;
};

FatBinSlot *AsSlot(void **handle) {
  auto *slot = reinterpret_cast<FatBinSlot *>(handle);
  return (slot != nullptr && slot->reg_id != 0) ? slot : nullptr;
}

//===--- real-symbol slots (one per interposed API) ---------------------------===//
// cudaMalloc and the launch pair use typed slots: cuda_runtime.h declares C++
// TEMPLATE overloads (`cudaMalloc<T>`, `cudaLaunchKernel<T>`) alongside the C
// API, so decltype(&name) is ambiguous for those three.
using RealCudaMalloc = cudaError_t (*)(void **, size_t);
using RealCudaLaunchKernel =
    cudaError_t (*)(const void *, dim3, dim3, void **, size_t, cudaStream_t);
using RealCudaLaunchKernelExC =
    cudaError_t (*)(const cudaLaunchConfig_t *, const void *, void **);
std::atomic<RealFn> g_real_cudaMalloc{nullptr};
std::atomic<RealFn> g_real_cudaLaunchKernel{nullptr};
std::atomic<RealFn> g_real_cudaLaunchKernelExC{nullptr};
KERNSCOPE_REAL_SLOT(cudaFree);
KERNSCOPE_REAL_SLOT(cudaMemcpy);
KERNSCOPE_REAL_SLOT(cudaMemcpyAsync);
KERNSCOPE_REAL_SLOT(cudaMemcpy2D);
KERNSCOPE_REAL_SLOT(cudaMemcpy2DAsync);
KERNSCOPE_REAL_SLOT(cudaMemset);
KERNSCOPE_REAL_SLOT(cudaMemsetAsync);
KERNSCOPE_REAL_SLOT(cudaGraphLaunch);
KERNSCOPE_REAL_SLOT(cudaStreamCreate);
KERNSCOPE_REAL_SLOT(cudaStreamSynchronize);
KERNSCOPE_REAL_SLOT(cuModuleLoad);
KERNSCOPE_REAL_SLOT(cuModuleLoadData);
KERNSCOPE_REAL_SLOT(cuModuleLoadDataEx);
KERNSCOPE_REAL_SLOT(cuModuleGetFunction);
KERNSCOPE_REAL_SLOT(cuFuncGetParamInfo);
KERNSCOPE_REAL_SLOT(cuLaunchKernel);
KERNSCOPE_REAL_SLOT(cuLaunchKernelEx);
// Typed slots for the __cudaRegister* pair (not declared in the headers).
using RealFatBinaryRegister = void **(*)(void *);
using RealFunctionRegister =
    void (*)(void **, const char *, char *, const char *, int, uint3 *, uint3 *,
             int *, int);
std::atomic<RealFn> g_real_register_fatbinary{nullptr};
std::atomic<RealFn> g_real_register_function{nullptr};

void RecordCopy(kernscope::CopyRecord rec) { Session().record_copy(rec); }

} // namespace

//===--- exported symbols -----------------------------------------------------===//
// All interposed symbols must be DEFAULT visibility (they replace the real
// ones in the global scope for the app's bindings); everything else in this
// .so is hidden (set via CMake CXX_VISIBILITY_PRESET).

#define KERNSCOPE_INTERCEPT_EXPORT __attribute__((visibility("default")))

//--- runtime API: memory -----------------------------------------------------//

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMalloc(void **devPtr, size_t size) {
  RecordApi(kernscope::ApiKind::kCudaMalloc);
  cudaError_t rc = cudaErrorNoDevice;
  if (RealCudaMalloc real =
          KERNSCOPE_REAL_TYPED(g_real_cudaMalloc, RealCudaMalloc,
                               "cudaMalloc")) {
    rc = real(devPtr, size);
  }
  if (rc == cudaSuccess && devPtr != nullptr && *devPtr != nullptr) {
    // cuda_runtime_api.h cudaMalloc contract: "Allocates size bytes of linear
    // memory on the device" - the map must record EXACTLY `size` for *devPtr.
    Session().record_alloc(reinterpret_cast<std::uint64_t>(*devPtr), size);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t cudaFree(void *devPtr) {
  RecordApi(kernscope::ApiKind::kCudaFree);
  const bool forwarded = KERNSCOPE_REAL(cudaFree) != nullptr;
  const cudaError_t rc =
      forwarded ? KERNSCOPE_REAL(cudaFree)(devPtr) : cudaErrorNoDevice;
  // cuda_runtime_api.h cudaFree contract: frees the memory space pointed to
  // by devPtr; freeing a pointer cudaMalloc did not return is UNDEFINED - the
  // session logs the unknown-free QA event and the shim forwards regardless
  // (never a crash).
  Session().record_free(reinterpret_cast<std::uint64_t>(devPtr), forwarded);
  return rc;
}

//--- runtime API: copies + memsets -------------------------------------------//

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind) {
  RecordApi(kernscope::ApiKind::kCudaMemcpy);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemcpy) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemcpy)(dst, src, count, kind);
  }
  if (rc == cudaSuccess) {
    kernscope::CopyRecord rec;
    rec.kind = "memcpy";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = count;
    rec.cuda_kind = static_cast<int>(kind);
    RecordCopy(rec);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemcpyAsync(void *dst, const void *src, size_t count, cudaMemcpyKind kind,
                cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaMemcpyAsync);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemcpyAsync) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemcpyAsync)(dst, src, count, kind, stream);
  }
  if (rc == cudaSuccess) {
    kernscope::CopyRecord rec;
    rec.kind = "memcpy-async";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = count;
    rec.cuda_kind = static_cast<int>(kind);
    rec.stream = reinterpret_cast<std::uint64_t>(stream);
    RecordCopy(rec);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemcpy2D(void *dst, size_t dpitch, const void *src, size_t spitch,
             size_t width, size_t height, cudaMemcpyKind kind) {
  RecordApi(kernscope::ApiKind::kCudaMemcpy2D);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemcpy2D) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemcpy2D)(dst, dpitch, src, spitch, width, height,
                                      kind);
  }
  if (rc == cudaSuccess) {
    kernscope::CopyRecord rec;
    rec.kind = "memcpy2d";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = (width != 0 && height <= kMaxCubinBytes / width)
                    ? width * height
                    : 0;
    rec.cuda_kind = static_cast<int>(kind);
    RecordCopy(rec);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                  size_t width, size_t height, cudaMemcpyKind kind,
                  cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaMemcpy2DAsync);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemcpy2DAsync) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemcpy2DAsync)(dst, dpitch, src, spitch, width,
                                           height, kind, stream);
  }
  if (rc == cudaSuccess) {
    kernscope::CopyRecord rec;
    rec.kind = "memcpy2d-async";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = (width != 0 && height <= kMaxCubinBytes / width)
                    ? width * height
                    : 0;
    rec.cuda_kind = static_cast<int>(kind);
    rec.stream = reinterpret_cast<std::uint64_t>(stream);
    RecordCopy(rec);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemset(void *devPtr, int value, size_t count) {
  RecordApi(kernscope::ApiKind::kCudaMemset);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemset) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemset)(devPtr, value, count);
  }
  if (rc == cudaSuccess) {
    kernscope::MemsetRecord rec;
    rec.ptr = reinterpret_cast<std::uint64_t>(devPtr);
    rec.value = value;
    rec.bytes = count;
    Session().record_memset(rec);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaMemsetAsync);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaMemsetAsync) != nullptr) {
    rc = KERNSCOPE_REAL(cudaMemsetAsync)(devPtr, value, count, stream);
  }
  if (rc == cudaSuccess) {
    kernscope::MemsetRecord rec;
    rec.ptr = reinterpret_cast<std::uint64_t>(devPtr);
    rec.value = value;
    rec.bytes = count;
    rec.stream = reinterpret_cast<std::uint64_t>(stream);
    Session().record_memset(rec);
  }
  return rc;
}

//--- runtime API: launches ----------------------------------------------------//
// Arg introspection limitation (header limitation 6): the runtime boundary
// exposes an opaque void** args array with no per-arg sizes, so runtime-path
// launch records carry the kernel name (resolved via the host-function
// registry from __cudaRegisterFunction), the full launch config, and NO arg
// bytes. Byte-exact runtime-path args are todo-44 CUPTI functionParams; the
// driver path below DOES capture arg bytes via cuFuncGetParamInfo.

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim, void **args,
                 size_t sharedMem, cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaLaunchKernel);
  kernscope::LaunchRecord rec;
  rec.via = kernscope::LaunchVia::kRuntimeApi;
  rec.kernel_name = Session().kernel_name_for_host_function(
      reinterpret_cast<std::uint64_t>(func));
  rec.grid_x = gridDim.x;
  rec.grid_y = gridDim.y;
  rec.grid_z = gridDim.z;
  rec.block_x = blockDim.x;
  rec.block_y = blockDim.y;
  rec.block_z = blockDim.z;
  rec.shared_mem_bytes = static_cast<std::uint32_t>(sharedMem);
  rec.stream = reinterpret_cast<std::uint64_t>(stream);

  g_in_runtime_launch = true;
  cudaError_t rc = cudaErrorNoDevice;
  if (RealCudaLaunchKernel real = KERNSCOPE_REAL_TYPED(
          g_real_cudaLaunchKernel, RealCudaLaunchKernel, "cudaLaunchKernel")) {
    rc = real(func, gridDim, blockDim, args, sharedMem, stream);
  }
  g_in_runtime_launch = false;
  if (rc == cudaSuccess) {
    Session().record_launch(std::move(rec));
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaLaunchKernelExC(const cudaLaunchConfig_t *config, const void *func,
                    void **args) {
  RecordApi(kernscope::ApiKind::kCudaLaunchKernelExC);
  kernscope::LaunchRecord rec;
  rec.via = kernscope::LaunchVia::kRuntimeApiEx;
  rec.kernel_name = Session().kernel_name_for_host_function(
      reinterpret_cast<std::uint64_t>(func));
  if (config != nullptr) {
    rec.grid_x = config->gridDim.x;
    rec.grid_y = config->gridDim.y;
    rec.grid_z = config->gridDim.z;
    rec.block_x = config->blockDim.x;
    rec.block_y = config->blockDim.y;
    rec.block_z = config->blockDim.z;
    rec.shared_mem_bytes =
        static_cast<std::uint32_t>(config->dynamicSmemBytes);
    rec.stream = reinterpret_cast<std::uint64_t>(config->stream);
  }

  g_in_runtime_launch = true;
  cudaError_t rc = cudaErrorNoDevice;
  if (RealCudaLaunchKernelExC real = KERNSCOPE_REAL_TYPED(
          g_real_cudaLaunchKernelExC, RealCudaLaunchKernelExC,
          "cudaLaunchKernelExC")) {
    rc = real(config, func, args);
  }
  g_in_runtime_launch = false;
  if (rc == cudaSuccess) {
    Session().record_launch(std::move(rec));
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaGraphLaunch);
  kernscope::LaunchRecord rec;
  rec.via = kernscope::LaunchVia::kGraph;
  rec.kernel_name = "<graph>";
  rec.stream = reinterpret_cast<std::uint64_t>(stream);
  rec.graph_handle = reinterpret_cast<std::uint64_t>(graphExec);
  // Per-node kernel names are unreachable at this boundary (the exec handle
  // does not expose the graph); todo 44's CUPTI resource/launch callbacks
  // walk graph nodes. Grid/block likewise - recorded as 1x1x1.
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaGraphLaunch) != nullptr) {
    rc = KERNSCOPE_REAL(cudaGraphLaunch)(graphExec, stream);
  }
  if (rc == cudaSuccess) {
    Session().record_launch(std::move(rec));
  }
  return rc;
}

//--- runtime API: streams -----------------------------------------------------//

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaStreamCreate(cudaStream_t *pStream) {
  RecordApi(kernscope::ApiKind::kCudaStreamCreate);
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaStreamCreate) != nullptr) {
    rc = KERNSCOPE_REAL(cudaStreamCreate)(pStream);
  }
  if (rc == cudaSuccess && pStream != nullptr && *pStream != nullptr) {
    // cuda_runtime_api.h cudaStreamCreate contract: "Creates a new
    // asynchronous stream" - the stream handle gets a stable logical id.
    Session().record_stream_create(reinterpret_cast<std::uint64_t>(*pStream));
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT cudaError_t
cudaStreamSynchronize(cudaStream_t stream) {
  RecordApi(kernscope::ApiKind::kCudaStreamSynchronize);
  Session().record_stream_sync(reinterpret_cast<std::uint64_t>(stream));
  cudaError_t rc = cudaErrorNoDevice;
  if (KERNSCOPE_REAL(cudaStreamSynchronize) != nullptr) {
    rc = KERNSCOPE_REAL(cudaStreamSynchronize)(stream);
  }
  return rc;
}

//--- registration: fat binaries + kernel names --------------------------------//
// These symbols are NOT declared by the CUDA 12.6 public headers (they are
// internal runtime entry points, declared in the toolkit's crt/host_runtime.h
// which nixpkgs does not ship); the signatures follow the documented
// contract used by every generated launch stub and by GPGPU-Sim's
// reimplementation (brief §2).

extern "C" KERNSCOPE_INTERCEPT_EXPORT void **
__cudaRegisterFatBinary(void *fatCubin) {
  RecordApi(kernscope::ApiKind::kRegisterFatBinary);
  auto *slot = new FatBinSlot(); // leaked: registration handles outlive main
  if (RealFatBinaryRegister real = KERNSCOPE_REAL_TYPED(
          g_real_register_fatbinary, RealFatBinaryRegister,
          "__cudaRegisterFatBinary")) {
    slot->real_handle = real(fatCubin);
  }
  const std::vector<std::uint8_t> cubin = FatBinCubinBytes(fatCubin);
  slot->reg_id = Session().register_fatbinary(cubin);
  return reinterpret_cast<void **>(slot);
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT void
__cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                       char *deviceFun, const char *deviceName,
                       int thread_limit, uint3 *tid, uint3 *blockDim,
                       int *dimMask, int unrollFactor) {
  RecordApi(kernscope::ApiKind::kRegisterFunction);
  const char *name = (deviceName != nullptr && deviceName[0] != '\0')
                         ? deviceName
                         : (deviceFun != nullptr ? deviceFun : "");
  FatBinSlot *slot = AsSlot(fatCubinHandle);
  if (slot != nullptr) {
    Session().register_function(slot->reg_id, name);
    Session().register_host_function(
        reinterpret_cast<std::uint64_t>(hostFun), name);
  }
  void **real_handle = (slot != nullptr) ? slot->real_handle : fatCubinHandle;
  if (RealFunctionRegister real = KERNSCOPE_REAL_TYPED(
          g_real_register_function, RealFunctionRegister,
          "__cudaRegisterFunction")) {
    if (real_handle != nullptr) {
      real(real_handle, hostFun, deviceFun, deviceName, thread_limit, tid,
           blockDim, dimMask, unrollFactor);
    }
  }
}

//--- driver API: module loading (cubin capture) -------------------------------//
// cuModuleLoad/cuModuleLoadData(Ex) are the cubin source for driver-API apps
// (brief §5 "Module/function handle capture"); without them the bundle's
// cubin field is unsatisfiable for apps that never go through the runtime
// registration path.

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuModuleLoad(CUmodule *module, const char *fname) {
  RecordApi(kernscope::ApiKind::kCuModuleLoad);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuModuleLoad) != nullptr) {
    rc = KERNSCOPE_REAL(cuModuleLoad)(module, fname);
  }
  if (rc == CUDA_SUCCESS && fname != nullptr) {
    std::ifstream in(fname, std::ios::binary);
    if (in) {
      std::vector<std::uint8_t> bytes((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
      if (bytes.size() <= kMaxCubinBytes) {
        Session().register_cubin(bytes);
      }
    } else {
      Session().register_module_file(fname);
    }
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuModuleLoadData(CUmodule *module, const void *image) {
  RecordApi(kernscope::ApiKind::kCuModuleLoadData);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuModuleLoadData) != nullptr) {
    rc = KERNSCOPE_REAL(cuModuleLoadData)(module, image);
  }
  if (rc == CUDA_SUCCESS && image != nullptr) {
    Session().register_cubin(CubinBytes(image));
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions,
                   CUjit_option *options, void **optionValues) {
  RecordApi(kernscope::ApiKind::kCuModuleLoadDataEx);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuModuleLoadDataEx) != nullptr) {
    rc = KERNSCOPE_REAL(cuModuleLoadDataEx)(module, image, numOptions, options,
                                            optionValues);
  }
  if (rc == CUDA_SUCCESS && image != nullptr) {
    Session().register_cubin(CubinBytes(image));
  }
  return rc;
}

//--- driver API: function identity + arg introspection ------------------------//

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
  RecordApi(kernscope::ApiKind::kCuModuleGetFunction);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuModuleGetFunction) != nullptr) {
    rc = KERNSCOPE_REAL(cuModuleGetFunction)(hfunc, hmod, name);
  }
  if (rc == CUDA_SUCCESS && hfunc != nullptr && *hfunc != nullptr &&
      name != nullptr) {
    Session().register_cufunction(reinterpret_cast<std::uint64_t>(*hfunc),
                                  name);
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuFuncGetParamInfo(CUfunction func, size_t paramIndex, size_t *paramOffset,
                   size_t *paramSize) {
  RecordApi(kernscope::ApiKind::kCuFuncGetParamInfo);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuFuncGetParamInfo) != nullptr) {
    rc = KERNSCOPE_REAL(cuFuncGetParamInfo)(func, paramIndex, paramOffset,
                                            paramSize);
  }
  if (rc == CUDA_SUCCESS && paramOffset != nullptr && paramSize != nullptr) {
    // cuda.h cuFuncGetParamInfo contract (group__CUDA__EXEC): returns the
    // per-parameter offset/size "retrieved directly from the kernel's image".
    // These facts size the arg capture in cuLaunchKernel below.
    Session().record_param_info(reinterpret_cast<std::uint64_t>(func),
                                paramIndex, *paramOffset, *paramSize);
  }
  return rc;
}

//--- driver API: launches (the lower choke point, with arg introspection) -----//

namespace {

kernscope::LaunchRecord DriverLaunchRecord(CUfunction f, unsigned int gx,
                                           unsigned int gy, unsigned int gz,
                                           unsigned int bx, unsigned int by,
                                           unsigned int bz,
                                           unsigned int shared,
                                           CUstream hStream) {
  kernscope::LaunchRecord rec;
  rec.via = kernscope::LaunchVia::kDriverApi;
  rec.kernel_name = Session().kernel_name_for_cufunction(
      reinterpret_cast<std::uint64_t>(f));
  rec.grid_x = gx;
  rec.grid_y = gy;
  rec.grid_z = gz;
  rec.block_x = bx;
  rec.block_y = by;
  rec.block_z = bz;
  rec.shared_mem_bytes = shared;
  rec.stream = reinterpret_cast<std::uint64_t>(hStream);
  return rec;
}

void CaptureDriverArgs(std::uint64_t cufunction, void **kernelParams,
                       kernscope::LaunchRecord &rec) {
  if (kernelParams == nullptr) {
    return;
  }
  // kernelParams[i] is a pointer to the i-th parameter's host-side value; its
  // size comes from the cuFuncGetParamInfo layout learned above (brief §5:
  // "cuFuncGetParamInfo(func, paramIndex, &paramOffset, &paramSize)").
  for (std::size_t i = 0; i < kMaxArgs; ++i) {
    auto size = Session().param_size(cufunction, i);
    if (!size.has_value() || *size > kMaxArgBytes) {
      break; // layout known only up to here; never read past it
    }
    kernscope::KernelArgSlot slot;
    slot.size = *size;
    if (*size > 0 && kernelParams[i] != nullptr) {
      const auto *bytes = static_cast<const std::uint8_t *>(kernelParams[i]);
      slot.bytes.assign(bytes, bytes + *size);
    }
    rec.args.push_back(std::move(slot));
  }
}

} // namespace

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY,
               unsigned int gridDimZ, unsigned int blockDimX,
               unsigned int blockDimY, unsigned int blockDimZ,
               unsigned int sharedMemBytes, CUstream hStream, void **kernelParams,
               void **extra) {
  RecordApi(kernscope::ApiKind::kCuLaunchKernel);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuLaunchKernel) != nullptr) {
    rc = KERNSCOPE_REAL(cuLaunchKernel)(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream, kernelParams, extra);
  }
  // g_in_runtime_launch: the runtime re-dispatched our runtime-API forward
  // through the driver - the runtime-API record already exists, skip (dedup).
  if (rc == CUDA_SUCCESS && !g_in_runtime_launch) {
    kernscope::LaunchRecord rec = DriverLaunchRecord(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ,
        sharedMemBytes, hStream);
    CaptureDriverArgs(reinterpret_cast<std::uint64_t>(f), kernelParams, rec);
    Session().record_launch(std::move(rec));
  }
  return rc;
}

extern "C" KERNSCOPE_INTERCEPT_EXPORT CUresult
cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f, void **kernelParams,
                 void **extra) {
  RecordApi(kernscope::ApiKind::kCuLaunchKernelEx);
  CUresult rc = CUDA_ERROR_NO_DEVICE;
  if (KERNSCOPE_REAL(cuLaunchKernelEx) != nullptr) {
    rc = KERNSCOPE_REAL(cuLaunchKernelEx)(config, f, kernelParams, extra);
  }
  if (rc == CUDA_SUCCESS && !g_in_runtime_launch && config != nullptr) {
    kernscope::LaunchRecord rec;
    rec.via = kernscope::LaunchVia::kDriverApiEx;
    rec.kernel_name = Session().kernel_name_for_cufunction(
        reinterpret_cast<std::uint64_t>(f));
    rec.grid_x = config->gridDimX;
    rec.grid_y = config->gridDimY;
    rec.grid_z = config->gridDimZ;
    rec.block_x = config->blockDimX;
    rec.block_y = config->blockDimY;
    rec.block_z = config->blockDimZ;
    rec.shared_mem_bytes = config->sharedMemBytes;
    rec.stream = reinterpret_cast<std::uint64_t>(config->hStream);
    CaptureDriverArgs(reinterpret_cast<std::uint64_t>(f), kernelParams, rec);
    Session().record_launch(std::move(rec));
  }
  return rc;
}

//--- flush entry point + atexit fallback ---------------------------------------//

extern "C" KERNSCOPE_INTERCEPT_EXPORT int kernscope_intercept_flush(
    const char *path) {
  if (path == nullptr || path[0] == '\0') {
    return -1;
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    return -1;
  }
  return kernscope::BundleWriter::create_v0()->write(
             kernscope::CaptureSession::instance(), out)
             ? 0
             : -1;
}

namespace {

void AtExitFlush() {
  const char *path = std::getenv("KERNSCOPE_CAPTURE_LOG");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  kernscope::CaptureSession &session = kernscope::CaptureSession::instance();
  if (session.empty()) {
    return; // never litter the cwd for processes that made no CUDA calls
  }
  (void)kernscope_intercept_flush(path);
}

struct ShimAtExitRegistrar {
  ShimAtExitRegistrar() { std::atexit(AtExitFlush); }
};
ShimAtExitRegistrar g_shim_atexit;

} // namespace
