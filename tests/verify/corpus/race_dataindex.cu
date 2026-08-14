// race_dataindex.cu — kernscope-verify race corpus fixture (todo 33),
// QA-negative half: DATA-DEPENDENT index race candidate.
//
//   race_data_kernel  <<<1, 256>>>, idx is a global pointer parameter: the
//   shared-memory index `j = idx[t]` is loaded from memory, so the analysis
//   cannot determine which addresses different threads touch (GPUVerify's
//   adversarial reads: a thread may observe a different value than another).
//   The write s[j] and the read s[j] sit in the same barrier phase with no
//   barrier between them, so the pair is a genuine race candidate — but its
//   addresses are UNKNOWABLE, not merely overlapping.
//
//   Expected verdict: UNKNOWN_RACE (soundness naming — NOT SAFE, because a
//   race cannot be excluded, and NOT MAY_RACE, because the analysis cannot
//   even bound the addresses).  Asserted as the QA negative in race_test.cpp
//   (with evidence) and in manifest.race.json.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void race_data_kernel(const int* idx)
{
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    const int j = idx[t];   // data-dependent index (global-memory load)
    s[j] = t;               // site 1: write s[j] — address unknowable
    int v = s[j];           // site 2: read s[j] — same phase, no barrier
    (void)v;
}

int main()
{
    int* d_idx = nullptr;
    cudaMalloc(&d_idx, 1 << 20);
    race_data_kernel<<<1, 256, 0, 0>>>(d_idx);
    cudaDeviceSynchronize();
    cudaFree(d_idx);
    return 0;
}
