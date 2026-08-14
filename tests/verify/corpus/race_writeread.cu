// race_writeread.cu — kernscope-verify race corpus fixture (todo 33),
// write-read half of the race-prone pairs.
//
//   race_writeread_kernel  <<<1, 256>>>: write s[t] then read s[255 - t]
//   with NO barrier between them.  Both indices are thread-id-dependent and
//   their SCEV byte spans overlap across the whole array (thread 0 writes
//   s[0] while thread 255 reads s[0]), so check class 4 must emit MAY_RACE
//   with BOTH source sites in the diagnostic JSON.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void race_writeread_kernel()
{
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    s[t] = t;            // site 1: write s[t]
    int v = s[255 - t];  // site 2: read s[255-t] — different threads can
                         // touch the same address within one phase
    (void)v;
}

int main()
{
    race_writeread_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
