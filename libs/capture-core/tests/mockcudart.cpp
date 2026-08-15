//===- mockcudart.cpp - fake CUDA runtime shared library ---------*- C++ -*-===//
//
// The "mock provides the symbols" half of the todo-43 LD_PRELOAD acceptance:
// a SHARED library exporting the same cuda*/cu* symbols libkernscope_intercept.so
// interposes on, with a deterministic fake device arena. The smoke executable
// LINKS this library, so its cuda* references are UNDEFINED in the executable
// and resolved by the dynamic linker - which is exactly what makes the
// LD_PRELOAD interposition work (an executable's own DEFINITIONS bind
// internally and cannot be preloaded over; symbols it imports from a shared
// library can).
//
// Symbol resolution with the shim preloaded:
//   exe call -> shim (LD_PRELOAD searched first) -> dlsym(RTLD_NEXT) -> this
//   library's fake -> deterministic fake-device behavior.
//
//===----------------------------------------------------------------------===//

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

extern "C" {

//===--- fake device ---------------------------------------------------------===//
// Deterministic: allocation addresses step 0x10000; a 1 MiB arena holds
// byte-exact contents (mirrors MockRuntime's byte-exactness contract).

constexpr std::uintptr_t kFakeBase = 0x10000;
constexpr std::uintptr_t kFakeStride = 0x10000;
constexpr std::size_t kArenaBytes = 1u << 20;
static std::uint8_t g_arena[kArenaBytes];
static std::vector<std::pair<void *, std::size_t>> g_allocs;

static std::uintptr_t NextFreePointer(std::size_t size) {
  for (std::uintptr_t p = kFakeBase; p < kFakeBase + kArenaBytes;
       p += kFakeStride) {
    bool free = true;
    for (const auto &a : g_allocs) {
      if (reinterpret_cast<std::uintptr_t>(a.first) == p) {
        free = false;
        break;
      }
    }
    if (free) {
      g_allocs.push_back({reinterpret_cast<void *>(p), size});
      return p;
    }
  }
  return 0;
}

static std::size_t FakeSizeOf(void *ptr) {
  for (const auto &a : g_allocs) {
    if (a.first == ptr) {
      return a.second;
    }
  }
  return 0;
}

static void *ArenaAt(void *ptr) {
  return g_arena + (reinterpret_cast<std::uintptr_t>(ptr) - kFakeBase);
}

// Kernel identity registry: host stub <-> fake CUfunction.
static std::unordered_map<const void *, CUfunction> g_host_functions;
static std::uintptr_t g_next_function = 0x9000;

// Synthetic fatbin (the layout the shim parses, GPGPU-Sim §2).
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

static std::vector<std::uint8_t> g_fat_cubin(256, 0x5A);
static FatCudaBinaryEntry g_fat_entry{1, g_fat_cubin.data(),
                                      static_cast<unsigned int>(g_fat_cubin.size())};
static FatCudaBinary g_fat_binary{0x466243B2, 1, nullptr, &g_fat_entry, nullptr,
                                  nullptr, 1};
static FatBinCHeader g_fat_header{0x466243B1, 1, &g_fat_binary, nullptr};

// Synthetic cubin: a minimal little-endian ELF64 (ehdr + 2 section headers +
// a .text payload) so the shim can compute the image length from the section
// table (cuModuleLoadData has no length at the boundary).
static std::vector<std::uint8_t> g_fake_cubin = [] {
  const std::size_t kEhdrSize = 64;
  const std::size_t kShdrSize = 64;
  const std::size_t kTextSize = 16;
  std::vector<std::uint8_t> elf(kEhdrSize + 2 * kShdrSize + kTextSize, 0);
  auto put16 = [&](std::size_t off, std::uint16_t v) {
    elf[off] = static_cast<std::uint8_t>(v);
    elf[off + 1] = static_cast<std::uint8_t>(v >> 8);
  };
  auto put32 = [&](std::size_t off, std::uint32_t v) {
    for (int i = 0; i < 4; ++i) {
      elf[off + i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
  };
  auto put64 = [&](std::size_t off, std::uint64_t v) {
    for (int i = 0; i < 8; ++i) {
      elf[off + i] = static_cast<std::uint8_t>(v >> (8 * i));
    }
  };
  elf[0] = 0x7F;
  elf[1] = 'E';
  elf[2] = 'L';
  elf[3] = 'F';
  elf[4] = 2; // ELF64
  elf[5] = 1; // little-endian
  elf[6] = 1; // ELF version
  put16(0x34, 1);       // e_type = ET_REL
  put64(0x28, kEhdrSize); // e_shoff
  put16(0x3A, kShdrSize); // e_shentsize
  put16(0x3C, 2);         // e_shnum
  const std::size_t text_off = kEhdrSize + 2 * kShdrSize;
  const std::size_t sh = kEhdrSize + kShdrSize;
  put32(sh + 0x00, 1);         // sh_type = SHT_PROGBITS
  put64(sh + 0x18, text_off);  // sh_offset
  put64(sh + 0x20, kTextSize); // sh_size
  for (std::size_t i = 0; i < kTextSize; ++i) {
    elf[text_off + i] = static_cast<std::uint8_t>(0xC0 + i);
  }
  return elf;
}();

//===--- fake runtime API definitions (the interposed symbols) ----------------===//

cudaError_t cudaMalloc(void **devPtr, size_t size) {
  const std::uintptr_t p = NextFreePointer(size);
  *devPtr = reinterpret_cast<void *>(p);
  return p == 0 ? cudaErrorMemoryAllocation : cudaSuccess;
}

cudaError_t cudaFree(void *devPtr) {
  for (auto it = g_allocs.begin(); it != g_allocs.end(); ++it) {
    if (it->first == devPtr) {
      g_allocs.erase(it);
      return cudaSuccess;
    }
  }
  // Unknown pointer (the QA negative scenario): forward succeeds, no crash.
  return cudaSuccess;
}

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,
                       cudaMemcpyKind kind) {
  const std::size_t dst_off = FakeSizeOf(dst);
  const std::size_t src_off = FakeSizeOf(const_cast<void *>(src));
  if (kind == cudaMemcpyHostToDevice && dst_off != 0 && count <= dst_off) {
    std::memcpy(ArenaAt(dst), src, count);
  } else if (kind == cudaMemcpyDeviceToHost && src_off != 0 &&
             count <= src_off) {
    std::memcpy(dst, ArenaAt(const_cast<void *>(src)), count);
  } else if (kind == cudaMemcpyDeviceToDevice && dst_off != 0 && src_off != 0 &&
             count <= dst_off && count <= src_off) {
    std::memcpy(ArenaAt(dst), ArenaAt(const_cast<void *>(src)), count);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpy(dst, src, count, kind);
}

// The 2D fakes below copy DIRECTLY (no cudaMemcpy delegation): on real
// hardware cudaMemcpy2D dispatches to the DRIVER (cuMemcpy2D), not to
// cudaMemcpy, so a delegating mock would fabricate an intermediate record
// that never exists on a real stack.
cudaError_t cudaMemcpy2D(void *dst, size_t dpitch, const void *src,
                         size_t spitch, size_t width, size_t height,
                         cudaMemcpyKind kind) {
  (void)dpitch;
  (void)spitch;
  const std::size_t count = width * height;
  const std::size_t dst_off = FakeSizeOf(dst);
  const std::size_t src_off = FakeSizeOf(const_cast<void *>(src));
  if (kind == cudaMemcpyHostToDevice && dst_off != 0 && count <= dst_off) {
    std::memcpy(ArenaAt(dst), src, count);
  } else if (kind == cudaMemcpyDeviceToHost && src_off != 0 &&
             count <= src_off) {
    std::memcpy(dst, ArenaAt(const_cast<void *>(src)), count);
  } else {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

cudaError_t cudaMemcpy2DAsync(void *dst, size_t dpitch, const void *src,
                              size_t spitch, size_t width, size_t height,
                              cudaMemcpyKind kind, cudaStream_t stream) {
  (void)stream;
  return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
  const std::size_t off = FakeSizeOf(devPtr);
  if (off == 0 || count > off) {
    return cudaErrorInvalidValue;
  }
  std::memset(ArenaAt(devPtr), value, count);
  return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count,
                            cudaStream_t stream) {
  (void)stream;
  return cudaMemset(devPtr, value, count);
}

cudaError_t cudaLaunchKernel(const void *func, dim3 gridDim, dim3 blockDim,
                             void **args, size_t sharedMem,
                             cudaStream_t stream) {
  // Mirrors the real runtime's re-dispatch through the driver API: forwards
  // to cuLaunchKernel, exercising the shim's TLS dedup on the driver path.
  const auto it = g_host_functions.find(func);
  CUfunction f = it != g_host_functions.end() ? it->second : nullptr;
  const CUresult r =
      cuLaunchKernel(f, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y,
                     blockDim.z, static_cast<unsigned>(sharedMem), stream, args,
                     nullptr);
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
}

cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *config,
                                const void *func, void **args) {
  const auto it = g_host_functions.find(func);
  CUfunction f = it != g_host_functions.end() ? it->second : nullptr;
  const CUresult r =
      cuLaunchKernel(f, config->gridDim.x, config->gridDim.y, config->gridDim.z,
                     config->blockDim.x, config->blockDim.y, config->blockDim.z,
                     static_cast<unsigned>(config->dynamicSmemBytes),
                     config->stream, args, nullptr);
  return r == CUDA_SUCCESS ? cudaSuccess : cudaErrorLaunchFailure;
}

cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
  (void)graphExec;
  (void)stream;
  return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *pStream) {
  static std::uintptr_t next = 0x5000;
  *pStream = reinterpret_cast<cudaStream_t>(next);
  next += 0x100;
  return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
  (void)stream;
  return cudaSuccess;
}

void **__cudaRegisterFatBinary(void *fatCubin) {
  (void)fatCubin;
  static void *sink; // the fake handle the fake __cudaRegisterFunction accepts
  return &sink;
}

void __cudaRegisterFunction(void **fatCubinHandle, const char *hostFun,
                            char *deviceFun, const char *deviceName,
                            int thread_limit, uint3 *tid, uint3 *blockDim,
                            int *dimMask, int unrollFactor) {
  (void)fatCubinHandle;
  (void)deviceFun;
  (void)deviceName;
  (void)thread_limit;
  (void)tid;
  (void)blockDim;
  (void)dimMask;
  (void)unrollFactor;
  auto *f = reinterpret_cast<CUfunction>(g_next_function);
  g_next_function += 0x100;
  g_host_functions[hostFun] = f;
}

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
  (void)fname;
  *module = reinterpret_cast<CUmodule>(0x7000);
  return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image) {
  (void)image;
  *module = reinterpret_cast<CUmodule>(0x7001);
  return CUDA_SUCCESS;
}

CUresult cuModuleLoadDataEx(CUmodule *module, const void *image,
                            unsigned int numOptions, CUjit_option *options,
                            void **optionValues) {
  (void)image;
  (void)numOptions;
  (void)options;
  (void)optionValues;
  *module = reinterpret_cast<CUmodule>(0x7002);
  return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod,
                             const char *name) {
  (void)hmod;
  (void)name;
  *hfunc = reinterpret_cast<CUfunction>(0x9100);
  return CUDA_SUCCESS;
}

CUresult cuFuncGetParamInfo(CUfunction func, size_t paramIndex,
                            size_t *paramOffset, size_t *paramSize) {
  (void)func;
  // Fixed 2-param layout for the fake kernel: ptr(8) @ 0, count(4) @ 8.
  if (paramIndex == 0) {
    *paramOffset = 0;
    *paramSize = 8;
    return CUDA_SUCCESS;
  }
  if (paramIndex == 1) {
    *paramOffset = 8;
    *paramSize = 4;
    return CUDA_SUCCESS;
  }
  return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX,
                        unsigned int gridDimY, unsigned int gridDimZ,
                        unsigned int blockDimX, unsigned int blockDimY,
                        unsigned int blockDimZ, unsigned int sharedMemBytes,
                        CUstream hStream, void **kernelParams, void **extra) {
  (void)f;
  (void)gridDimX;
  (void)gridDimY;
  (void)gridDimZ;
  (void)blockDimX;
  (void)blockDimY;
  (void)blockDimZ;
  (void)sharedMemBytes;
  (void)hStream;
  (void)kernelParams;
  (void)extra;
  return CUDA_SUCCESS;
}

CUresult cuLaunchKernelEx(const CUlaunchConfig *config, CUfunction f,
                          void **kernelParams, void **extra) {
  (void)config;
  (void)f;
  (void)kernelParams;
  (void)extra;
  return CUDA_SUCCESS;
}

//===--- scenario fixtures (the synthetic images the shim parses) -------------===//

const void *mockcudart_fatbin() { return &g_fat_header; }
const void *mockcudart_fatbin_cubin() { return g_fat_cubin.data(); }
std::size_t mockcudart_fatbin_cubin_size() { return g_fat_cubin.size(); }
const void *mockcudart_elf_cubin() { return g_fake_cubin.data(); }
std::size_t mockcudart_elf_cubin_size() { return g_fake_cubin.size(); }

} // extern "C"
