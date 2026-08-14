// barrier_divergent.cu — kernscope-verify barrier corpus fixture (todo 32).
//
// Class 6 acceptance case: __syncthreads() inside a thread-id-dependent
// conditional. Per CUDA C Programming Guide B.6 the conditional must
// evaluate identically across the whole thread block, otherwise behavior
// is undefined -> expected verdict DIVERGENT_BARRIER (error) with a source
// location. (A class-3 BARRIER_MISMATCH at the join is also true: threads
// reach it having executed 1 vs 0 barriers.)
//
// Compile-time contract (tests/fixtures/README.md "clang/CUDA header
// gate"): clang 21 lowers <<< >>> via cudaConfigureCall unless a CUDA >= 9
// SDK is visible, so the source declares the nvcc-frontend-internal
// cudaConfigureCall (extern "C") and uses the full four-argument launch
// form. Compiled by the verifier's clang ingestion (-O1 host+device
// passes, then LLVM link); never executed.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void divergent_kernel(int* out)
{
    if (threadIdx.x == 0) {
        __syncthreads();
    }
    out[threadIdx.x] = 1;
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);

    divergent_kernel<<<1, 64, 0, 0>>>(d_out);

    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
