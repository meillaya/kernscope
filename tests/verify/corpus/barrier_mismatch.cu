// barrier_mismatch.cu — kernscope-verify barrier corpus fixture (todo 32).
//
// Class 3 acceptance case: unequal bar.sync counts on the two paths of a
// divergent branch before a common join. Threads taking the then-path
// execute 3 barriers before the join barrier; the else-path threads execute
// 1 -> expected verdict BARRIER_MISMATCH (warning) at the join point. (The
// two then-path barriers are also class-6 divergent-barrier errors: they
// sit inside the thread-dependent region.)

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void mismatch_kernel(int* out)
{
    __syncthreads();

    if (threadIdx.x < 32) {
        __syncthreads();
        __syncthreads();
    }

    __syncthreads();

    out[threadIdx.x] = 1;
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);

    mismatch_kernel<<<1, 64, 0, 0>>>(d_out);

    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
