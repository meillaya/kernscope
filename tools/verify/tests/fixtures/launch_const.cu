// launch_const.cu — kernscope-verify core ingestion fixture (todo 30).
//
// Three kernels exercising the launch-config tiers (research brief §4.6):
//   exact_kernel   <<<32, 256, 0, 0>>>  -> tier (a) EXACT: threadIdx.x in
//                                          [0, 256), blockIdx.x in [0, 32)
//   bound_kernel   __launch_bounds__(256), launched with RUNTIME dims
//                                          -> tier (b) sound upper bound 256
//   symbolic_kernel launched with runtime grid  -> tier (c) symbolic dims
//
// Compile-time contract (tests/fixtures/README.md "clang/CUDA header
// gate"): clang 21 lowers <<< >>> via cudaConfigureCall unless a CUDA >= 9
// SDK is visible, so the source declares the nvcc-frontend-internal
// cudaConfigureCall (extern "C") and uses the full four-argument launch
// form. All launches are 4-arg: <<<g, b, 0, 0>>>.
//
// Compiled by the verifier's clang ingestion (-O1 host+device passes, then
// LLVM link); never executed.

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void exact_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i;
    }
}

__global__ void __launch_bounds__(256) bound_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i * 2;
    }
}

__global__ void symbolic_kernel(int* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = i * 3;
    }
}

int main()
{
    int* d_out = nullptr;
    cudaMalloc(&d_out, 1 << 20);

    // Tier (a): constant dims at the launch site.
    exact_kernel<<<32, 256, 0, 0>>>(d_out, 1024);

    // Tier (b): runtime dims at the launch site; the kernel's
    // __launch_bounds__(256) still gives a sound upper bound on blockDim.
    const int bx = d_out[0];
    bound_kernel<<<bx, bx, 0, 0>>>(d_out, 1024);

    // Tier (c): runtime grid -> symbolic dims.
    const int gx = d_out[0] + 1;
    symbolic_kernel<<<gx, 128, 0, 0>>>(d_out, 1024);

    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
