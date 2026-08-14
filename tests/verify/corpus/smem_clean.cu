// smem_clean.cu — kernscope-verify shared-memory bounds corpus fixture
// (todo 31), CLEAN half of the pair.  All accesses to the statically sized
// __shared__ array are within the declared size, so check class 2
// (shared-memory bounds) must emit SAFE.
//
//   smem_clean_kernel  <<<1, 256>>>: s[tid], s declared [256 x i32]
//                      (1024 bytes); threadIdx.x in [0, 256) -> byte span
//                      [0, 1024) == allocation size -> SAFE (exact-fit
//                      boundary case: the exclusive upper bound is the last
//                      accessed byte + 1, so 1024 <= 1024 is in-bounds).
//
// See tools/verify/tests/fixtures/README.md for the cudaConfigureCall /
// 4-arg launch contract (same as oob_clean.cu).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void smem_clean_kernel()
{
    // volatile: at -O1 the optimizer does not model cross-thread visibility,
    // so a non-volatile shared round-trip would be dead-code-eliminated and
    // the fixture would contain no accesses (the verifier must see them).
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    s[t] = t;
    // Read back: also in-bounds.
    int v = s[t];
    (void)v;
}

int main()
{
    smem_clean_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
