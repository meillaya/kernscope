// Showcase kernel: row-per-thread softmax (todo 42, Wave 7) — FRESH
// REWRITE (the /home/mei/projects/tensors/cuda/kernels softmax is a
// SEMANTIC reference only).
//
// The deliberate NO-TRANSFORM showcase: the row length `n` is a RUNTIME
// parameter, so the column loop's trip count is not SCEV-provable and the
// pass pipeline declines at every step (the report records
// "no-transform": true HONESTLY — the QA negative case of todo 42):
//
//   * kernscope-tile      — trip count not a compile-time constant ->
//     decline (kNoConstantTrip).
//   * kernscope-promote   — the {row*n,+,1} window has a non-constant trip
//     count -> the [s, s+T) extent is unbounded -> decline.
//   * kernscope-vectorize — one load per loop iteration, no consecutive
//     chains -> untouched.
//   * kernscope-coalesce  — one thread per row: the per-lane stride is
//     row*n (n runtime -> Mul(Unknown(%n), Unknown(%tid))) -> the affine
//     coefficient is not a compile-time constant -> unknown-pattern
//     (HONEST — a runtime-strided access is never claimed coalesced).
//
// Portable CUDA/HIP: same source feeds clang (pass pipeline + PTX) and
// hipcc --genco (live gfx1101 bench).

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y, int n) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  // Max pass.
  float m = -3.402823466e+38f;
  for (int c = 0; c < n; ++c) {
    const float v = x[row * n + c];
    m = fmaxf(m, v);
  }
  // Sum-of-exp pass.
  float s = 0.0f;
  for (int c = 0; c < n; ++c) {
    s += expf(x[row * n + c] - m);
  }
  // Normalize pass.
  for (int c = 0; c < n; ++c) {
    y[row * n + c] = expf(x[row * n + c] - m) / s;
  }
}
