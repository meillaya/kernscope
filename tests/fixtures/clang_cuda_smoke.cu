// clang_cuda_smoke.cu — minimal CUDA kernel for the clang-21 <-> CUDA-12.6
// header gate (scripts/smoke_toolchain.sh stage a).
//
// Exercises exactly the constructs that break when clang's CUDA support
// version does not match the installed toolkit headers:
//   * <cuda_runtime.h> include chain (must resolve from --cuda-path)
//   * __syncthreads() (the CUDART intrinsic + llvm.nvvm.barrier0 lowering)
//   * <<< >>> launch syntax with a launch-bounds-worthy config
//
// Compile (inside devenv shell; single-root $CUDA_HOME from todo 2):
//   clang++ -x cuda --no-cuda-version-check --cuda-path=$CUDA_HOME \
//           -emit-llvm -c clang_cuda_smoke.cu -o /tmp/clang_cuda_smoke.bc
//
// NOTE: the --no-cuda-version-check flag is MANDATORY: clang 21.1.8's
// supported CUDA version is older than the 12.6 headers in the devenv, so
// the version gate would otherwise reject the toolkit (see the negative QA
// scenario in .omo/evidence/task-5-gpu-kernel-suite-negative.txt).
//
// This file is never run (no local NVIDIA GPU); -emit-llvm only.

#include <cuda_runtime.h>

// clang 21 lowers the <<< >>> launch stub via cudaConfigureCall (legacy
// path) when no CUDA >= 9 SDK version is visible on the host target. The
// declaration is nvcc-frontend-internal and absent from the nix CUDA
// package set, so the smoke source declares it (self-contained decl, see
// tests/fixtures/README.md "clang/CUDA header fallback").
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void clang_cuda_smoke_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i * 2;
    }
    __syncthreads();
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1024 * sizeof(int));
    clang_cuda_smoke_kernel<<<4, 256, 0, 0>>>(d_out, 1024);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
