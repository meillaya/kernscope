// barrier_uniform_loop.cu — kernscope-verify barrier corpus fixture
// (todo 32).
//
// QA negative / regression guard: a barrier under a UNIFORM loop-carried
// condition (j == 2, j a plain loop counter) must stay SAFE — the classic
// over-reporting trap. The loop trip count is uniform, so every thread of
// the block evaluates j == 2 identically and reaches the same number of
// barriers; the post-loop __syncthreads() is likewise executed by all
// threads with the same count. Expected verdict: SAFE (no class-3 warning,
// no class-6 error) — asserted by the ctest AND the QA-negative evidence.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void uniform_loop_kernel(int* out)
{
    for (int j = 0; j < 4; ++j) {
        if (j == 2) {
            __syncthreads();
        }
    }

    __syncthreads();

    out[threadIdx.x] = 1;
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);

    uniform_loop_kernel<<<1, 64, 0, 0>>>(d_out);

    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
