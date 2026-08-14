// hip_smoke.cpp — trivial HIP kernel for the gfx1101 compile+run gate
// (scripts/smoke_toolchain.sh stage c).
//
// Compiled with `hipcc --offload-arch=gfx1101` and RUN on the local GPU
// (RX 7800 XT = gfx1101, RDNA3). Exits 0 only if the kernel launches and
// every element of the output buffer equals threadIdx.x * 2.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void hip_smoke_kernel(int* out)
{
    out[threadIdx.x] = threadIdx.x * 2;
}

int main()
{
    int* d_out = nullptr;
    if (hipMalloc(&d_out, 32 * sizeof(int)) != hipSuccess) {
        std::fprintf(stderr, "[FAIL] hip_smoke: hipMalloc\n");
        return 1;
    }
    hipLaunchKernelGGL(hip_smoke_kernel, dim3(1), dim3(32), 0, 0, d_out);
    if (hipGetLastError() != hipSuccess) {
        std::fprintf(stderr, "[FAIL] hip_smoke: launch\n");
        return 1;
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        std::fprintf(stderr, "[FAIL] hip_smoke: sync\n");
        return 1;
    }
    int h[32];
    if (hipMemcpy(h, d_out, sizeof(h), hipMemcpyDeviceToHost) != hipSuccess) {
        std::fprintf(stderr, "[FAIL] hip_smoke: memcpy\n");
        return 1;
    }
    for (int i = 0; i < 32; ++i) {
        if (h[i] != i * 2) {
            std::fprintf(stderr, "[FAIL] hip_smoke: out[%d]=%d, want %d\n",
                         i, h[i], i * 2);
            return 1;
        }
    }
    hipFree(d_out);
    std::printf("[PASS] hip_smoke: gfx1101 kernel produced correct output\n");
    return 0;
}
