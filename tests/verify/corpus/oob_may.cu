// oob_may.cu — kernscope-verify global-OOB corpus fixture (todo 31), BUGGY
// half of the pair.  Both kernels access a STATICALLY SIZED device global
// with index expressions that exceed (or cannot be proven within) the
// allocation, so check class 1 (global out-of-bounds) must emit MAY_OOB with
// the SCEV-derived range and the allocation bound in the diagnostic.
//
//   oob_may_kernel          <<<1, 256>>>: i = threadIdx.x * 1024 -> SCEV
//                           range [0, 262144) elements vs g_buf[1024] ->
//                           MAY_OOB.
//   oob_data_kernel         <<<1, 256>>>: g_buf[in[0]] — DATA-DEPENDENT
//                           index (a runtime load): neither the lattice nor
//                           SCEV can bound it -> MAY_OOB (may-analysis; the
//                           check deliberately does NOT claim completeness
//                           for data-dependent indices — soundness over
//                           false safety).
//
// See tools/verify/tests/fixtures/README.md for the cudaConfigureCall /
// 4-arg launch contract (same as oob_clean.cu).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__device__ int g_buf[1024];

__global__ void oob_may_kernel()
{
    // 256 threads stride by 1024 elements: worst-case element 255*1024 =
    // 261120 is far beyond the 1024-element allocation -> MAY_OOB.
    const int i = threadIdx.x * 1024;
    g_buf[i] = i;
}

__global__ void oob_data_kernel(const int* in)
{
    // Data-dependent index: in[0] is a runtime value, unprovable statically.
    // May-analysis: cannot prove in-bounds -> MAY_OOB, never SAFE.
    g_buf[in[0]] = 1;
}

int main()
{
    oob_may_kernel<<<1, 256, 0, 0>>>();
    int* d_in = nullptr;
    cudaMalloc(&d_in, 1 << 20);
    oob_data_kernel<<<1, 256, 0, 0>>>(d_in);
    cudaDeviceSynchronize();
    cudaFree(d_in);
    return 0;
}
