// hip_ingest.hip — kernscope-verify HIP ingestion fixture (todo 30).
//
// Compile-gated HIP path (acceptance: HIP gtest skips when the ROCm
// toolchain is unusable). Uses the same 4-arg launch contract as the CUDA
// fixtures; hipcc lowers <<< >>> to __hipPushCallConfiguration (packed
// i64 grid/block) + hipLaunchKernel with a DIRECT kernel reference (no
// __device_stub__), both of which the launch-config resolver handles.
// Never executed (compile-only ingestion).

#include <hip/hip_runtime.h>

__global__ void hip_ingest_kernel(float* out, int n)
{
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        out[i] = static_cast<float>(i);
    }
}

int main()
{
    float* d = nullptr;
    hipMalloc(&d, 4096);
    hip_ingest_kernel<<<2, 64, 0, 0>>>(d, 1024);
    hipDeviceSynchronize();
    hipFree(d);
    return 0;
}
