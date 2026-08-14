// smem_dynamic.cu — kernscope-verify dynamic shared-memory corpus fixture
// (todo 31).  `extern __shared__` (dynamic smem): the buffer is ZERO-SIZED in
// the IR ([0 x i32]); its real size comes from the launch-site smem argument
// (<<<grid, block, smemBytes, stream>>>).  The check must resolve the
// per-kernel launch smem bytes and bound the accesses against THAT.
//
//   smem_dyn_safe_kernel  <<<1, 64, 256, 0>>>   (256 bytes = 64 ints):
//                          dyn_s[tid], tid in [0, 64) -> byte span [0, 256)
//                          == 256 bytes -> SAFE.
//   smem_dyn_oob_kernel   <<<1, 256, 256, 0>>>  (256 bytes = 64 ints):
//                          dyn_s[tid], tid in [0, 256) -> byte span
//                          [0, 1024) vs 256 bytes -> MAY_OOB.
//
// Both kernels reference the SAME extern shared global; the launch-site smem
// argument per kernel is what the bound is derived from.
//
// See tools/verify/tests/fixtures/README.md for the cudaConfigureCall /
// 4-arg launch contract (same as oob_clean.cu).

#include <cuda_runtime.h>

extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         size_t sharedMem,
                                         cudaStream_t stream);

// Dynamic shared memory: size comes from the launch configuration only.
extern __shared__ volatile int dyn_s[];

__global__ void smem_dyn_safe_kernel()
{
    dyn_s[threadIdx.x] = threadIdx.x;
}

__global__ void smem_dyn_oob_kernel()
{
    dyn_s[threadIdx.x] = threadIdx.x;
}

int main()
{
    // 256 bytes of dynamic smem = 64 ints.
    smem_dyn_safe_kernel<<<1, 64, 256, 0>>>();
    smem_dyn_oob_kernel<<<1, 256, 256, 0>>>();
    cudaDeviceSynchronize();
    return 0;
}
