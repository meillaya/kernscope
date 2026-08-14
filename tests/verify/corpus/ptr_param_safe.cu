// ptr_param_safe.cu — kernscope-verify pointer-discipline corpus fixture
// (todo 34), the SAFE half for kernel-pointer parameters:
//
//   param_guarded_kernel    `int* out` + `int n`: the canonical `if (i < n)`
//                           guarded access. The pointer VALUE arithmetic is
//                           bounded by the launch dims, and the allocation
//                           size is a runtime argument — class 5 makes no
//                           static escape claim for kernel params (runtime
//                           allocation) -> SAFE. Class 7: param pointees are
//                           host-initialized -> SAFE.
//
// Compile-time contract (tools/verify/tests/fixtures/README.md): clang 21
// lowers <<< >>> via cudaConfigureCall unless a CUDA >= 9 SDK is visible,
// so the source declares the nvcc-frontend-internal cudaConfigureCall
// (extern "C") and uses the full four-argument launch form. Compiled by the
// verifier's clang ingestion (-O1 host+device passes, then LLVM link);
// never executed.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void param_guarded_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i;
    }
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);
    param_guarded_kernel<<<32, 256, 0, 0>>>(d_out, 1024);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
