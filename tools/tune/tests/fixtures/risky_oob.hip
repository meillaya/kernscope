// risky_oob.hip — the todo-54 e2e's INJECTED risky variant source.
//
// A deliberately unsafe kernel: a statically-sized __device__ buffer written
// PAST its bounds with an index the todo-35 verifier can bound exactly
// (affine in threadIdx/blockIdx with a constant launch tier), so the OOB
// check MUST emit MAY_OOB (an error-class verdict) deterministically. The
// tune's --verify gate discards this variant BEFORE compilation and records
// the verdict id — the acceptance asserts the injected risky config is
// gated (never compiled).
//
// Self-contained by design (no kernscope includes): the gate harness wraps
// this kernel as-is with a constant-dimension launch.

__device__ float g_ks_risky_buf[256];

extern "C" __global__ void kernscope_tune_risky_oob() {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  // i in [0, grid*block) — the harness launches this kernel with the
  // duplicated grid config's geometry, so i reaches far past g_buf[256]:
  // index range [256, grid*block) vs the 256-element allocation.
  g_ks_risky_buf[i + 256] = 1.0f;
}
