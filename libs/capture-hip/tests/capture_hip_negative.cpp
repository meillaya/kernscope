// capture_hip_negative.cpp — todo-45 QA negative fixture.
//
// Allocates a buffer through an API the capture tool deliberately does NOT
// hook (hipMallocPitch), copies host data into it, launches a kernel that
// reads it, and frees it. The capture must NOT crash and must mark the
// pitch-allocated buffer "size-unknown" (the ptr->size map has no entry) in
// the bundle + emit a warning; the app itself must behave normally.

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void neg_kernel(const float *in, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = in[i] * 2.0f;
  }
}

int main() {
  constexpr int kCols = 64;
  constexpr int kRows = 8;
  std::vector<float> h(kCols * kRows, 1.0f);
  std::vector<float> hout(kCols * kRows, 0.0f);

  float *dpitch = nullptr;
  size_t pitch = 0;
  hipError_t err = hipMallocPitch(&dpitch, &pitch, kCols * sizeof(float),
                                  kRows); // UNSUPPORTED API (not hooked)
  if (err != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: hipMallocPitch: %s\n",
                 hipGetErrorString(err));
    return 1;
  }
  float *dout = nullptr;
  if ((err = hipMalloc(&dout, hout.size() * sizeof(float))) != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: hipMalloc: %s\n", hipGetErrorString(err));
    return 1;
  }
  // Copy into the pitch buffer: dst is NOT in the allocation map -> the
  // bundle must mark dst-size=0 (size-unknown) + the capture logs a warning.
  if ((err = hipMemcpy(dpitch, h.data(), h.size() * sizeof(float),
                       hipMemcpyHostToDevice)) != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: hipMemcpy: %s\n", hipGetErrorString(err));
    return 1;
  }
  neg_kernel<<<1, kCols * kRows>>>(dpitch, dout, kCols * kRows);
  if ((err = hipGetLastError()) != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: launch: %s\n", hipGetErrorString(err));
    return 1;
  }
  if ((err = hipDeviceSynchronize()) != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: sync: %s\n", hipGetErrorString(err));
    return 1;
  }
  if ((err = hipMemcpy(hout.data(), dout, hout.size() * sizeof(float),
                       hipMemcpyDeviceToHost)) != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: D2H: %s\n", hipGetErrorString(err));
    return 1;
  }
  bool ok = true;
  for (float v : hout) {
    ok = ok && (v == 2.0f);
  }
  // hipFree is [[nodiscard]]: check + report so a failing free is not dropped.
  hipError_t ferr = hipFree(dout);
  if (ferr != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: hipFree(dout): %s\n",
                 hipGetErrorString(ferr));
    ok = false;
  }
  ferr = hipFree(dpitch);
  if (ferr != hipSuccess) {
    std::fprintf(stderr, "[FAIL] neg: hipFree(dpitch): %s\n",
                 hipGetErrorString(ferr));
    ok = false;
  }
  std::fprintf(stderr, "[INFO] neg: output %s\n", ok ? "correct" : "WRONG");
  return ok ? 0 : 1;
}
