//===- add_kernel.hip - trivial add kernel for the live HIP test ----------===//
//
// kernscope gpuruntime live-test kernel (todo 21). Compiled at BUILD time by
// `hipcc --genco --offload-arch=gfx1101` into tests/kernels/add_kernel.hsaco
// (see libs/gpuruntime/CMakeLists.txt), then loaded by the gpu-amd gtest
// through Module::load(path) -> hipModuleLoad and launched through
// Kernel::create + launch (hipModuleGetFunction + hipLaunchKernel).
//
// Kernels need the builtin launch-identifiers (blockIdx/blockDim/threadIdx)
// from hip_runtime.h. hipcc --genco does not inject the ROCm include dirs
// itself, so CMake passes -I<clr>/include (see libs/gpuruntime/CMakeLists.txt).
#include <hip/hip_runtime.h>

extern "C" __global__ void kernscope_add(const float *a, const float *b,
                                         float *out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = a[i] + b[i];
  }
}
