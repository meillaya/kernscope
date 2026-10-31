// race_writewrite.cu — kernscope-verify race corpus fixture (todo 33),
// write-write half of the race-prone pairs.
//
//   race_writewrite_kernel  <<<1, 256>>>: two writes to s[threadIdx.x]
//   with NO barrier between them.  Both indices are thread-id-dependent and
//   may denote the same address (SCEV spans both cover the whole array), so
//   check class 4 (race-prone shared-memory writes) must emit MAY_RACE with
//   BOTH source sites in the diagnostic JSON (the acceptance contract).
//
// Runtime semantics note (soundness scope, docs/report/kernscope-technical-report.pdf): the
// two stores are per-thread sequential writes to the thread's OWN element,
// but the range-based may-alias analysis cannot prove that distinct threads
// obtain distinct addresses (no thread-variable modeling), so the pair is
// reported MAY_RACE — the plan-mandated over-approximation, listed as a
// documented false-positive class.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void race_writewrite_kernel()
{
    // volatile: at -O1 the optimizer does not model cross-thread
    // visibility, so non-volatile shared round-trips would be
    // dead-code-eliminated and the fixture would contain no accesses.
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    s[t] = t;        // site 1: write s[t], thread-dependent index
    s[t] = 2 * t;    // site 2: write s[t], thread-dependent index
}

int main()
{
    race_writewrite_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
