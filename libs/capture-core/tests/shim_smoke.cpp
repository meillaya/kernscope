//===- shim_smoke.cpp - LD_PRELOAD smoke (mock-linked binary) ------*- C++ -*-===//
//
// The todo-43 LD_PRELOAD acceptance binary. It is MOCK-LINKED: its cuda*/cu*
// references are UNDEFINED in the executable and satisfied at load time from
// libkernscope_mockcudart.so (the fake runtime with a deterministic fake
// device arena) - an executable's own DEFINITIONS would bind internally and
// could not be preloaded over. With the shim preloaded
//
//   LD_PRELOAD=$PWD/build/libkernscope_intercept.so ./capture-shim-smoke [log]
//
// every call lands in the shim first (LD_PRELOAD is searched before the
// executable's dependencies), is recorded, and is forwarded via
// dlsym(RTLD_NEXT) into the mock. The binary then flushes the shim's capture
// log and SELF-CHECKS that the log contains the expected records (launches
// incl. name/grid/block, allocation map sizes, fatbin + cubin hashes,
// unknown-free). Exit 0 == interception verified end-to-end with no NVIDIA
// runtime present (the todo-21 mock provides the symbols).
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// The shim's flush entry point (resolved from libkernscope_intercept.so,
// which the binary links).
extern "C" int kernscope_intercept_flush(const char *path);

// The registration entry points are not declared by the CUDA 12.6 public
// headers (toolkit-internal); declared here with the documented contract
// (same declarations the shim's wrappers are built against).
extern "C" void **__cudaRegisterFatBinary(void *fatCubin);
extern "C" void __cudaRegisterFunction(void **fatCubinHandle,
                                       const char *hostFun, char *deviceFun,
                                       const char *deviceName, int thread_limit,
                                       uint3 *tid, uint3 *blockDim,
                                       int *dimMask, int unrollFactor);

// Scenario fixtures from the mock runtime library.
extern "C" const void *mockcudart_fatbin();
extern "C" const void *mockcudart_fatbin_cubin();
extern "C" std::size_t mockcudart_fatbin_cubin_size();
extern "C" const void *mockcudart_elf_cubin();
extern "C" std::size_t mockcudart_elf_cubin_size();

// Stable address shared by __cudaRegisterFunction and cudaLaunchKernel (the
// runtime-API launch resolves the kernel name by host-function pointer; the
// mock registers hostFun -> fake CUfunction under this address).
static const char kHostStubName[] = "host_vec_add_stub";

namespace {

bool LogContains(const std::string &log, const std::string &needle) {
  return log.find(needle) != std::string::npos;
}

std::string ReadFile(const char *path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return {};
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

} // namespace

int main(int argc, char **argv) {
  const char *log_path = (argc > 1) ? argv[1] : "kernscope-capture.log";

  // 1. runtime registration path (fatbin + function).
  void **fatbin_handle = __cudaRegisterFatBinary(
      const_cast<void *>(mockcudart_fatbin()));
  __cudaRegisterFunction(fatbin_handle,
                         const_cast<char *>(kHostStubName),
                         const_cast<char *>("vec_add"), "vec_add", -1, nullptr,
                         nullptr, nullptr, 0);

  // 2. driver module path (cubin + function + param introspection).
  CUmodule mod = nullptr;
  if (cuModuleLoadData(&mod, mockcudart_elf_cubin()) != CUDA_SUCCESS) {
    std::fprintf(stderr, "smoke: cuModuleLoadData failed\n");
    return 1;
  }
  CUfunction fn = nullptr;
  if (cuModuleGetFunction(&fn, mod, "vec_add") != CUDA_SUCCESS) {
    std::fprintf(stderr, "smoke: cuModuleGetFunction failed\n");
    return 1;
  }
  size_t off0 = 0, sz0 = 0, off1 = 0, sz1 = 0;
  if (cuFuncGetParamInfo(fn, 0, &off0, &sz0) != CUDA_SUCCESS ||
      cuFuncGetParamInfo(fn, 1, &off1, &sz1) != CUDA_SUCCESS) {
    std::fprintf(stderr, "smoke: cuFuncGetParamInfo failed\n");
    return 1;
  }

  // 3. allocation + copies + memsets.
  void *dptr = nullptr;
  if (cudaMalloc(&dptr, 1024) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaMalloc failed\n");
    return 1;
  }
  std::uint8_t host[512];
  for (std::size_t i = 0; i < sizeof(host); ++i) {
    host[i] = static_cast<std::uint8_t>(i);
  }
  if (cudaMemcpy(dptr, host, 256, cudaMemcpyHostToDevice) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaMemcpy failed\n");
    return 1;
  }
  if (cudaMemset(dptr, 0xAB, 64) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaMemset failed\n");
    return 1;
  }
  if (cudaMemcpy2D(dptr, 256, host, 256, 128, 4, cudaMemcpyHostToDevice) !=
      cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaMemcpy2D failed\n");
    return 1;
  }

  // 4. launches: runtime API (re-dispatch -> driver, deduped), ExC, direct
  //    driver API, graph; streams.
  std::uint64_t ptr_arg = reinterpret_cast<std::uint64_t>(dptr);
  std::uint32_t count_arg = 512;
  void *args[] = {&ptr_arg, &count_arg};
  dim3 grid(2, 1, 1), block(128, 1, 1);
  if (cudaLaunchKernel(reinterpret_cast<const void *>(kHostStubName), grid,
                       block, args, 0, nullptr) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaLaunchKernel failed\n");
    return 1;
  }
  cudaLaunchConfig_t config{};
  config.gridDim = grid;
  config.blockDim = block;
  config.dynamicSmemBytes = 0;
  config.stream = nullptr;
  if (cudaLaunchKernelExC(&config,
                          reinterpret_cast<const void *>(kHostStubName),
                          args) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaLaunchKernelExC failed\n");
    return 1;
  }
  cudaStream_t stream = nullptr;
  if (cudaStreamCreate(&stream) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaStreamCreate failed\n");
    return 1;
  }
  if (cuLaunchKernel(fn, 2, 1, 1, 128, 1, 1, 0, stream, args, nullptr) !=
      CUDA_SUCCESS) {
    std::fprintf(stderr, "smoke: cuLaunchKernel failed\n");
    return 1;
  }
  if (cudaStreamSynchronize(stream) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaStreamSynchronize failed\n");
    return 1;
  }
  if (cudaGraphLaunch(reinterpret_cast<cudaGraphExec_t>(0x9876), stream) !=
      cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaGraphLaunch failed\n");
    return 1;
  }
  if (cudaMemsetAsync(dptr, 0x11, 32, stream) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaMemsetAsync failed\n");
    return 1;
  }

  // 5. QA negative: unknown free must be logged and forwarded, no crash.
  if (cudaFree(reinterpret_cast<void *>(0x42424242)) != cudaSuccess) {
    std::fprintf(stderr, "smoke: unknown cudaFree failed\n");
    return 1;
  }
  if (cudaFree(dptr) != cudaSuccess) {
    std::fprintf(stderr, "smoke: cudaFree failed\n");
    return 1;
  }

  // 6. flush + self-check against the shim's capture log.
  if (kernscope_intercept_flush(log_path) != 0) {
    std::fprintf(stderr, "smoke: kernscope_intercept_flush(%s) failed\n",
                 log_path);
    return 1;
  }
  const std::string log = ReadFile(log_path);
  int failures = 0;
  auto check = [&](const std::string &needle) {
    if (!LogContains(log, needle)) {
      std::fprintf(stderr, "smoke: MISSING record: %s\n", needle.c_str());
      ++failures;
    }
  };

  const std::vector<std::uint8_t> fat_cubin(
      static_cast<const std::uint8_t *>(mockcudart_fatbin_cubin()),
      static_cast<const std::uint8_t *>(mockcudart_fatbin_cubin()) +
          mockcudart_fatbin_cubin_size());
  const std::vector<std::uint8_t> elf_cubin(
      static_cast<const std::uint8_t *>(mockcudart_elf_cubin()),
      static_cast<const std::uint8_t *>(mockcudart_elf_cubin()) +
          mockcudart_elf_cubin_size());

  check("kernscope-capture-v0");
  check("end-capture-v0");
  check("api cudaMalloc calls=1");
  check("api __cudaRegisterFatBinary calls=1");
  check("api cuFuncGetParamInfo calls=2");
  check("alloc ptr=0x10000 size=1024"); // allocation map EXACT size
  check("free ptr=0x10000 size=1024");
  // Fatbin registration with the cubin bytes hash (exact) + kernel name.
  check("fatbin mod=1 sha256=" + kernscope::sha256_hex(fat_cubin));
  check("fatbin mod=1 sha256=" + kernscope::sha256_hex(fat_cubin) +
        " kernels=\"vec_add\"");
  // Cubin blob hash (exact; the shim sized the ELF image from its section
  // table since cuModuleLoadData carries no length).
  check("cubin mod=2 sha256=" + kernscope::sha256_hex(elf_cubin));
  // Launch records: runtime (name/grid/block; arg bytes unreachable at the
  // runtime boundary - documented limitation), ExC, direct driver (arg
  // bytes via cuFuncGetParamInfo), graph.
  check("launch kernel=\"vec_add\" via=runtime grid=2x1x1 block=128x1x1 "
        "shared=0 stream=0x0 graph=0x0 args=0");
  check("launch kernel=\"vec_add\" via=runtime-ex grid=2x1x1 block=128x1x1 "
        "shared=0 stream=0x0 graph=0x0 args=0");
  check("launch kernel=\"vec_add\" via=driver grid=2x1x1 block=128x1x1 "
        "shared=0 stream=0x5000 graph=0x0 args=2");
  check("[8:0000010000000000] [4:00020000]"); // driver-path arg bytes (LE)
  check("launch kernel=\"<graph>\" via=graph");
  check("stream create handle=0x5000 id=1");
  check("stream sync handle=0x5000 id=1");
  // Allocation-map attribution at the copy boundary: H2D (cudaMemcpyKind 1)
  // sizes the dst from the map -> dst-size=1024.
  check("copy kind=memcpy dst=0x10000 src=0x");
  check("bytes=256 cuda-kind=1 dst-size=1024 src-size=0");
  check("copy kind=memcpy2d dst=0x10000 src=0x");
  check("memset ptr=0x10000 value=171 bytes=64 ptr-size=1024");
  // QA negative: unknown-free logged + forwarded.
  check("unknown-free ptr=0x42424242 forwarded=true");
  // Dedup: the runtime launch's internal driver re-dispatch must NOT create a
  // second driver record (TLS flag) - exactly ONE via=driver line.
  std::size_t driver_count = 0;
  for (std::size_t pos = 0;
       (pos = log.find("via=driver ", pos)) != std::string::npos; pos += 1) {
    ++driver_count;
  }
  if (driver_count != 1) {
    std::fprintf(stderr,
                 "smoke: expected exactly 1 via=driver record, got %zu\n",
                 driver_count);
    ++failures;
  }

  if (failures > 0) {
    std::fprintf(stderr,
                 "smoke: %d capture-log assertions failed\n"
                 "hint: run with LD_PRELOAD=$PWD/build/libkernscope_intercept.so\n",
                 failures);
    return 1;
  }
  std::printf("smoke: PASS - capture log %s verified (%zu bytes)\n", log_path,
              log.size());
  return 0;
}
