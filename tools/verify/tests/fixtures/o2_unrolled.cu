// o2_unrolled.cu — kernscope-verify -O2 ingestion fixture (todo 30).
//
// Acceptance case 3: an -O2 ingestion either reconstructs index bounds via
// SCEV or emits a degraded verdict; BOTH paths must be covered. This file
// carries one kernel per path (both launched with constant dims):
//
//   reconstruct_kernel  index = threadIdx-derived base + small unrolled
//                       offsets: at -O2 the 4-iteration loop fully unrolls
//                       into constant offsets; the range lattice bounds
//                       every access (reconstruction succeeds).
//   opaque_kernel       index = in[0] (an opaque runtime load): neither the
//                       lattice nor ScalarEvolution can bound it; the -O2
//                       degraded verdict (SCEV reconstruction failed) fires.
//
// See tests/fixtures/README.md for the cudaConfigureCall/4-arg launch
// contract (same as launch_const.cu).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

__global__ void reconstruct_kernel(int* out)
{
    const int base = threadIdx.x * 4;
    for (int j = 0; j < 4; ++j) {
        out[base + j] = j;
    }
}

__global__ void opaque_kernel(const int* in, int* out)
{
    const int base = in[0];
    out[base] = 1;
}

int main()
{
    int* d = nullptr;
    cudaMalloc(&d, 1 << 20);
    reconstruct_kernel<<<1, 64, 0, 0>>>(d);
    opaque_kernel<<<1, 64, 0, 0>>>(d, d);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
