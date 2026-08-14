// smem_overflow.cu — kernscope-verify shared-memory bounds corpus fixture
// (todo 31), BUGGY half of the pair.  One kernel accesses past the declared
// __shared__ size (MAY_OOB), the sibling kernel stays within a wide buffer
// (SAFE) — the same thread count, different declared sizes, so the verdicts
// differ purely on the smem bound.
//
//   smem_overflow_kernel   <<<1, 256>>>: s[tid] with s declared [64 x i32]
//                          (256 bytes) -> byte span [0, 1024) vs 256 bytes
//                          -> MAY_OOB (smem).
//   smem_safe_wide_kernel  <<<1, 256>>>: w[tid] with w declared [512 x i32]
//                          (2048 bytes) -> byte span [0, 1024) vs 2048 bytes
//                          -> SAFE.
//
// See tools/verify/tests/fixtures/README.md for the cudaConfigureCall /
// 4-arg launch contract (same as oob_clean.cu).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void smem_overflow_kernel()
{
    // 256 threads index a 64-element buffer: elements 64..255 are past the
    // declared size -> MAY_OOB.  volatile: keeps the accesses alive at -O1
    // (LLVM does not model cross-thread visibility; see smem_clean.cu).
    volatile __shared__ int s[64];
    s[threadIdx.x] = threadIdx.x;
}

__global__ void smem_safe_wide_kernel()
{
    // Same thread count, but the buffer covers all 256 threads -> SAFE.
    volatile __shared__ int w[512];
    w[threadIdx.x] = threadIdx.x;
}

int main()
{
    smem_overflow_kernel<<<1, 256, 0, 0>>>();
    smem_safe_wide_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
