//===- equiv_hip_app.hip - equivalence test: HIP-side program (gfx1101) ------*- HIP -*-===//
//
// kernscope shared-core library tree.
//
// The HIP side of the todo-44 dual-mechanism equivalence test
// (ctest -R capture_equiv -L gpu-amd): the SAME program the CUDA side
// replays synthetically (see equiv_cuda_side.cpp), compiled for gfx1101 and
// run LIVE under libkernscope_capture_hip.so (todo 45). It embeds the
// todo-24 portable kernel source and mirrors tools/hip_template_runner.cpp
// EXACTLY, except that the copies use hipMemcpy (the todo-45 tool wraps
// hipMemcpy/hipMemcpyAsync, NOT hipMemcpyHtoD/hipMemcpyDtoH):
//
//   hipMalloc a (2048) -> hipMemcpy H2D a -> hipMalloc b (2048) ->
//   hipMemcpy H2D b -> hipMalloc out (2048) -> hipLaunchKernel
//   (grid 4, block 256, args [a, b, out, int32 n]) -> hipDeviceSynchronize ->
//   hipMemcpy D2H out -> write out.bin -> hipFree a, b, out
//
// CLI: equiv_hip_app <a.bin> <b.bin> <out.bin> <n>
//
//===----------------------------------------------------------------------===//

#include "../../../kernels/templates/reference/elementwise_add.hip"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <hip/hip_runtime.h>

namespace {

std::vector<char> ReadFile(const char *path) {
  std::ifstream fh(path, std::ios::binary);
  if (!fh) {
    return {};
  }
  return std::vector<char>((std::istreambuf_iterator<char>(fh)),
                           std::istreambuf_iterator<char>());
}

bool WriteFile(const char *path, const std::vector<char> &data) {
  std::ofstream fh(path, std::ios::binary | std::ios::trunc);
  if (!fh) {
    return false;
  }
  fh.write(data.data(), static_cast<std::streamsize>(data.size()));
  return static_cast<bool>(fh);
}

int Fail(const std::string &what, hipError_t rc) {
  std::fprintf(stderr, "equiv_hip_app: %s (hipError %d: %s)\n", what.c_str(),
               static_cast<int>(rc), hipGetErrorString(rc));
  return 1;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: equiv_hip_app <a.bin> <b.bin> <out.bin> <n>\n");
    return 1;
  }
  const std::vector<char> a = ReadFile(argv[1]);
  const std::vector<char> b = ReadFile(argv[2]);
  const std::int32_t n = static_cast<std::int32_t>(std::strtol(argv[4], nullptr, 10));
  if (a.empty() || b.empty() || a.size() != b.size()) {
    return Fail("bad input files", hipErrorInvalidValue);
  }

  void *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
  hipError_t rc = hipMalloc(&d_a, a.size());
  if (rc != hipSuccess) {
    return Fail("hipMalloc(a)", rc);
  }
  rc = hipMemcpy(d_a, a.data(), a.size(), hipMemcpyHostToDevice);
  if (rc != hipSuccess) {
    return Fail("hipMemcpy H2D a", rc);
  }
  rc = hipMalloc(&d_b, b.size());
  if (rc != hipSuccess) {
    return Fail("hipMalloc(b)", rc);
  }
  rc = hipMemcpy(d_b, b.data(), b.size(), hipMemcpyHostToDevice);
  if (rc != hipSuccess) {
    return Fail("hipMemcpy H2D b", rc);
  }
  rc = hipMalloc(&d_out, a.size());
  if (rc != hipSuccess) {
    return Fail("hipMalloc(out)", rc);
  }

  // Driver-API convention: device pointers pass BY VALUE (uintptr).
  std::uint64_t da = reinterpret_cast<std::uint64_t>(d_a);
  std::uint64_t db = reinterpret_cast<std::uint64_t>(d_b);
  std::uint64_t dout = reinterpret_cast<std::uint64_t>(d_out);
  void *args[] = {&da, &db, &dout, const_cast<std::int32_t *>(&n)};
  rc = hipLaunchKernel(reinterpret_cast<const void *>(
                           &kernscope_templates_elementwise_add),
                       dim3(4, 1, 1), dim3(256, 1, 1), args, 0, nullptr);
  if (rc != hipSuccess) {
    return Fail("hipLaunchKernel", rc);
  }
  rc = hipDeviceSynchronize();
  if (rc != hipSuccess) {
    return Fail("hipDeviceSynchronize", rc);
  }

  std::vector<char> out(a.size());
  rc = hipMemcpy(out.data(), d_out, out.size(), hipMemcpyDeviceToHost);
  if (rc != hipSuccess) {
    return Fail("hipMemcpy D2H out", rc);
  }
  if (!WriteFile(argv[3], out)) {
    return Fail("cannot write output", hipErrorUnknown);
  }
  std::printf("equiv_hip_app: wrote %zu bytes -> %s\n", out.size(), argv[3]);

  (void)hipFree(d_a);
  (void)hipFree(d_b);
  (void)hipFree(d_out);
  return 0;
}
