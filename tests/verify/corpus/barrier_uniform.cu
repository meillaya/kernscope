// barrier_uniform.cu — kernscope-verify barrier corpus fixture (todo 32).
//
// Class 6/3 negative case: barriers under UNIFORM conditions must stay SAFE
// (the classic false-positive trap). The blockIdx.x == 0 condition is
// uniform across every thread of the block (llvm.nvvm.read.ptx.sreg.ctaid.x
// is block-uniform), and the leading __syncthreads() is unconditional.
// Expected verdict: SAFE (no class-3 warning, no class-6 error).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void uniform_kernel(int* out)
{
    __syncthreads();

    if (blockIdx.x == 0) {
        __syncthreads();
    }

    out[threadIdx.x] = 1;
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);

    uniform_kernel<<<2, 64, 0, 0>>>(d_out);

    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
