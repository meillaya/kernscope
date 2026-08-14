// ptr_oob.cu — kernscope-verify pointer-arithmetic corpus fixture (todo 34),
// check class 5 (illegal pointer arithmetic). Three kernels, all PTR_OOB:
//
//   ptr_oob_escape_kernel   int* p = g_buf + 20000; *p = 7;  — the pointer
//                           VALUE escapes g_buf's 65536-byte allocation
//                           (constant GEP index) -> PTR_OOB.
//   ptr_oob_range_kernel    g_buf[threadIdx.x * 100] — the GEP index range
//                           [0, 25600) elements * 4 bytes = [0, 102400)
//                           exceeds the allocation -> PTR_OOB (range proof
//                           via the propagated launch dims / SCEV).
//   ptr_sub_unrelated_kernel  long d = p - q with p = g_buf + i and
//                           q = g_other + i: subtracting pointers rooted in
//                           DIFFERENT allocations -> PTR_OOB (unrelated
//                           pointer subtraction, UB).
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

__device__ int g_buf[16384];   // 65536 bytes
__device__ int g_other[1024];  // 4096 bytes

__global__ void ptr_oob_escape_kernel()
{
    // Constant-index escape: the pointer value g_buf + 20000 sits 13696
    // bytes past the allocation end (16384 elements * 4 bytes).
    int* p = g_buf + 20000;
    *p = 7;
}

__global__ void ptr_oob_range_kernel()
{
    // threadIdx.x in [0, 256): index range [0, 25600) elements -> byte span
    // [0, 102400) > 65536 -> the pointer arithmetic is provably escaping.
    g_buf[threadIdx.x * 100] = threadIdx.x;
}

__global__ void ptr_sub_unrelated_kernel()
{
    // p and q are rooted in different __device__ allocations; their
    // difference is undefined behavior (unrelated pointer subtraction).
    // The difference feeds a STORE so instcombine cannot fold the
    // subtraction into a pointer comparison (verified against clang-21 -O1:
    // `if (d != 0)` alone becomes `icmp eq ptr` and the sub vanishes).
    int* p = g_buf + threadIdx.x;
    int* q = g_other + blockIdx.x;
    long d = p - q;
    g_buf[0] = (int)d;
}

int main()
{
    ptr_oob_escape_kernel<<<1, 1, 0, 0>>>();
    ptr_oob_range_kernel<<<1, 256, 0, 0>>>();
    ptr_sub_unrelated_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
