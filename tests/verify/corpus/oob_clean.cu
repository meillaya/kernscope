// oob_clean.cu — kernscope-verify global-OOB corpus fixture (todo 31), CLEAN
// half of the pair.  Both kernels access STATICALLY SIZED device globals with
// index expressions that ScalarEvolution + the propagated launch dims can
// bound, so check class 1 (global out-of-bounds) must emit SAFE.
//
//   oob_clean_kernel   <<<32, 256>>>: i = blockIdx.x*blockDim.x + threadIdx.x
//                      -> SCEV range [0, 8224) elements, allocation 16384
//                      elements -> SAFE (proof names the SCEV range).
//   oob_param_kernel   <<<32, 256>>>, out is a runtime kernel parameter with
//                      the canonical `if (i < n)` guard: the INDEX is proven
//                      bounded by the launch dims, but the allocation size is
//                      a runtime argument — the verdict stays SAFE on index
//                      grounds with an explicit runtime-allocation note
//                      (may-analysis scope: static allocation-size proof
//                      requires host-side cudaMalloc tracking, not in scope).
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

// Statically sized allocation: the check can prove bounds against 16384
// elements (65536 bytes) without any runtime information.
__device__ int g_buf[16384];

__global__ void oob_clean_kernel()
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    // i in [0, 8224) via SCEV (lattice: [0,32)*[256,257) + [0,256)) — well
    // inside g_buf[16384] -> SAFE.
    g_buf[i] = i;
}

__global__ void oob_param_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    // Index bounded by the launch dims; allocation size is the runtime `n`.
    // The guard is the runtime check that makes this safe at runtime.
    if (i < n) {
        out[i] = i;
    }
}

int main()
{
    // Tier (a): constant dims at the launch site -> exact dim ranges.
    oob_clean_kernel<<<32, 256, 0, 0>>>();
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);
    oob_param_kernel<<<32, 256, 0, 0>>>(d_out, 1024);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
