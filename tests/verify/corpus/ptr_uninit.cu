// ptr_uninit.cu — kernscope-verify uninitialized-read corpus fixture (todo
// 34), check class 7 (uninitialized reads). Two BAD kernels + one CLEAN:
//
//   uninit_read_kernel        reads __shared__ s[threadIdx.x] BEFORE any
//                             write in the kernel -> MAY_UNINIT_READ (with
//                             evidence path). Shared memory is
//                             uninitialized at kernel entry (kernel-boundary
//                             assumption: no cross-kernel initialization
//                             analysis).
//   uninit_alloca_kernel      reads a volatile local BEFORE writing it ->
//                             MAY_UNINIT_READ (allocas start uninitialized).
//   uninit_phase_clean_kernel writes s[threadIdx.x], hits __syncthreads()
//                             (phase boundary: any-thread writes before the
//                             barrier become readable by all threads after
//                             it), then reads s[255 - threadIdx.x] -> SAFE.
//
// `volatile` is required on the smem/local accesses: at -O1 the verifier's
// ingestion DCEs non-volatile read-after-write round-trips (the documented
// checks/oob/README.md gotcha).
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

__device__ int g_buf[16384];

__global__ void uninit_read_kernel()
{
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    const int v = s[t];  // read before any write -> MAY_UNINIT_READ
    s[t] = v + 1;
    g_buf[t] = s[t];
}

__global__ void uninit_alloca_kernel()
{
    volatile int a;        // never written before the read
    const int v = a;       // MAY_UNINIT_READ (alloca starts uninitialized)
    g_buf[threadIdx.x] = v;
}

__global__ void uninit_phase_clean_kernel()
{
    volatile __shared__ int s[256];
    const int t = threadIdx.x;
    s[t] = t;              // every thread writes its own element
    __syncthreads();       // phase boundary: writes become visible to all
    const int v = s[255 - t];  // other threads wrote it before the barrier
    g_buf[t] = v;              // -> SAFE
}

int main()
{
    uninit_read_kernel<<<1, 256, 0, 0>>>();
    uninit_alloca_kernel<<<1, 256, 0, 0>>>();
    uninit_phase_clean_kernel<<<1, 256, 0, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
