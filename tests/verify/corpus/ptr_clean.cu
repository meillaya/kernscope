// ptr_clean.cu — kernscope-verify pointer-discipline corpus fixture (todo
// 34), CLEAN half of every pair. All three check classes (5 illegal pointer
// arithmetic, 7 uninitialized reads, 8 invalid address-space casts) must
// emit SAFE for every kernel here.
//
//   ptr_clean_kernel    <<<32, 256>>>: every GEP index is bounded by the
//                       propagated launch dims and every pointer stays
//                       inside its statically sized allocation; smem is
//                       written before it is read (write s[t], then read
//                       s[t]); no addrspacecast appears -> SAFE for classes
//                       5, 7, 8.
//
// Compile-time contract (tools/verify/tests/fixtures/README.md "clang/CUDA
// header gate"): clang 21 lowers <<< >>> via cudaConfigureCall unless a
// CUDA >= 9 SDK is visible, so the source declares the nvcc-frontend-internal
// cudaConfigureCall (extern "C") and uses the full four-argument launch form.
// Compiled by the verifier's clang ingestion (-O1 host+device passes, then
// LLVM link); never executed.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__device__ int g_buf[16384];

__global__ void ptr_clean_kernel()
{
    // volatile: at -O1 the smem round-trip is otherwise DCE'd away (the
    // documented ingestion gotcha — checks/oob/README.md); the kernel must
    // keep a real read-after-write for check class 7.
    volatile __shared__ int s[256];
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    g_buf[i] = i;        // i in [0, 8224) < 16384 elements -> SAFE (class 5)
    s[threadIdx.x] = i;  // write before read -> SAFE (class 7)
    const int v = s[threadIdx.x];
    g_buf[i] = v;        // pointer re-based from g_buf, index bounded -> SAFE
}

int main()
{
    // Tier (a): constant dims at the launch site -> exact dim ranges.
    ptr_clean_kernel<<<32, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
