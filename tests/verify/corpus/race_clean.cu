// race_clean.cu — kernscope-verify race corpus fixture (todo 33), CLEAN
// side: the barrier-separated reduction and the SCEV-disjoint pair.
//
//   race_clean_kernel  <<<1, 256>>>: write s[t] (phase 0) then
//   __syncthreads() then reads s[i] (phase 1, the reduction loop).  The
//   barrier splits the kernel into two barrier phases, so the write and the
//   reads are in DIFFERENT phases -> SAFE (the canonical
//   reduction-with-barrier pattern).
//
//   race_clean_disjoint_kernel  <<<1, 256>>>: write s[t] and read s[t+256]
//   on a [512]-element array with NO barrier.  Both indices are
//   thread-dependent, but ScalarEvolution proves the byte spans DISJOINT
//   ([0,1024) vs [1024,2048)) -> no address may be shared -> SAFE (the
//   sound case where the may-alias test can actually prove disjointness).
//
//   race_clean_uniform_kernel  <<<2, 64>>>: write s[t] under
//   `if (blockIdx.x == 0)` and read s[t] under the else branch, no barrier.
//   The branch condition is BLOCK-UNIFORM: every thread of a block takes the
//   same side, so the write and the read are on mutually exclusive
//   execution paths -> SAFE (uniform-branch exclusivity; the classic
//   false-positive trap this module guards against).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void race_clean_kernel(int* out)
{
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    s[t] = t;            // phase 0 write
    __syncthreads();     // barrier: phase 0 -> phase 1
    int sum = 0;
    for (int i = 0; i < 256; ++i) {
        sum += s[i];     // phase 1 reads
    }
    if (t == 0) {
        out[0] = sum;
    }
}

__global__ void race_clean_disjoint_kernel()
{
    volatile __shared__ int s[512];
    const int t = threadIdx.x;
    s[t] = t;            // bytes [0, 1024)
    int v = s[t + 256];  // bytes [1024, 2048) — SCEV-disjoint -> SAFE
    (void)v;
}

__global__ void race_clean_uniform_kernel()
{
    volatile __shared__ int s[64];
    const int t = threadIdx.x;
    if (blockIdx.x == 0) {
        s[t] = t;        // uniform branch: every thread of block 0 writes
    } else {
        int v = s[t];    // every thread of block 1 reads — exclusive paths
        (void)v;
    }
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);
    race_clean_kernel<<<1, 256, 0, 0>>>(d_out);
    race_clean_disjoint_kernel<<<1, 256, 0, 0>>>();
    race_clean_uniform_kernel<<<2, 64, 0, 0>>>();
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
