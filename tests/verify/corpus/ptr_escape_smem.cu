// ptr_escape_smem.cu — kernscope-verify pointer-arithmetic corpus fixture
// (todo 34), check class 5 on SHARED memory. Two kernels:
//
//   smem_escape_kernel      `volatile __shared__ int s[64]`; a pointer
//                           arithmetic expression escapes the 256-byte
//                           allocation (s + 1000) -> PTR_OOB (the pointer
//                           VALUE check applies to shared allocations too).
//   smem_sub_same_kernel    subtraction of two pointers rooted in the SAME
//                           shared allocation (s + tid) - (s + 1) -> SAFE
//                           (same-allocation subtraction is legal C).
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

__device__ int g_out[64];

__global__ void smem_escape_kernel()
{
    volatile __shared__ int s[64];  // 256 bytes
    volatile int* p = s + 1000;              // pointer value escapes: 4000 > 256
    *p = 1;
}

__global__ void smem_sub_same_kernel()
{
    volatile __shared__ int s[64];
    volatile int* p = s + threadIdx.x;
    volatile int* q = s + 1;
    long d = p - q;                 // same allocation: legal subtraction
    g_out[threadIdx.x] = (int)d;
}

int main()
{
    smem_escape_kernel<<<1, 64, 0, 0>>>();
    smem_sub_same_kernel<<<1, 64, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
