// Showcase kernel: thread-per-row layernorm (todo 42, Wave 7) — FRESH
// REWRITE (the /home/mei/projects/tensors/cuda/kernels layernorm is a
// SEMANTIC reference only).
//
// One row per thread (64 rows, 64 columns; the row selector is threadIdx.x,
// the row slab is &LN_IN[row*64]). The feature loop walks the slab in
// strides of 16 with 16 CONSECUTIVE loads per iteration — the exact shape
// kernscope-vectorize merges into <4 x float> loads:
//
//   * the chain head offset SCEV is {row*64,+,16} — min trailing zeros 4
//     (16 elements per iteration) -> 16-byte provable head alignment on
//     top of the base's __align__(16) -> <4 x float> legal (vector_width
//     1 -> 4 in after.ll, the todo-42 report delta).
//   * the four loads in[i..i+3] differ by exactly the element size (SCEV
//     folds) and no intervening memory op may alias them -> the AA gate
//     passes.
//
// Warp-level honesty (reported by kernscope-coalesce): one row per thread
// means consecutive threads read rows 64 elements apart -> per-lane stride
// 256 B -> 32 sectors per warp for 128 B of useful data (12.5% efficiency)
// -> classified uncoalesced. This is the real tradeoff of the
// thread-per-row variant (block-per-row is coalesced but its tid-dependent
// heads defeat IR-level alignment proofs); the before/after report records
// both dimensions honestly.
//
// Portable CUDA/HIP: same source feeds clang (pass pipeline + PTX) and
// hipcc --genco (live gfx1101 bench).

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#define LN_COLS 64
#define LN_ROWS 64

__attribute__((aligned(16))) __device__ float LN_IN[LN_COLS * LN_ROWS];
__device__ float LN_OUT[LN_COLS * LN_ROWS];

__global__ void layernorm_kernel() {
  const int row = threadIdx.x;  // one row per thread (rows <= threads)
  const float* __restrict__ in = &LN_IN[row * LN_COLS];
  float* __restrict__ out = &LN_OUT[row * LN_COLS];

  // Pass 1: sum — stride-16 loop, 16 consecutive loads per iteration.
  float sum = 0.0f;
  for (int i = 0; i < LN_COLS; i += 16) {
    sum += in[i] + in[i + 1] + in[i + 2] + in[i + 3];
    sum += in[i + 4] + in[i + 5] + in[i + 6] + in[i + 7];
    sum += in[i + 8] + in[i + 9] + in[i + 10] + in[i + 11];
    sum += in[i + 12] + in[i + 13] + in[i + 14] + in[i + 15];
  }
  const float mean = sum / LN_COLS;

  // Pass 2: sum of squares.
  float sq = 0.0f;
  for (int i = 0; i < LN_COLS; i += 16) {
    const float d0 = in[i] - mean;
    const float d1 = in[i + 1] - mean;
    const float d2 = in[i + 2] - mean;
    const float d3 = in[i + 3] - mean;
    const float d4 = in[i + 4] - mean;
    const float d5 = in[i + 5] - mean;
    const float d6 = in[i + 6] - mean;
    const float d7 = in[i + 7] - mean;
    const float d8 = in[i + 8] - mean;
    const float d9 = in[i + 9] - mean;
    const float d10 = in[i + 10] - mean;
    const float d11 = in[i + 11] - mean;
    const float d12 = in[i + 12] - mean;
    const float d13 = in[i + 13] - mean;
    const float d14 = in[i + 14] - mean;
    const float d15 = in[i + 15] - mean;
    sq += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5 +
          d6 * d6 + d7 * d7 + d8 * d8 + d9 * d9 + d10 * d10 + d11 * d11 +
          d12 * d12 + d13 * d13 + d14 * d14 + d15 * d15;
  }
  const float inv = 1.0f / (1e-5f + sq / LN_COLS);

  // Pass 3: normalize — consecutive loads + stores.
  for (int i = 0; i < LN_COLS; i += 16) {
    out[i] = (in[i] - mean) * inv;
    out[i + 1] = (in[i + 1] - mean) * inv;
    out[i + 2] = (in[i + 2] - mean) * inv;
    out[i + 3] = (in[i + 3] - mean) * inv;
    out[i + 4] = (in[i + 4] - mean) * inv;
    out[i + 5] = (in[i + 5] - mean) * inv;
    out[i + 6] = (in[i + 6] - mean) * inv;
    out[i + 7] = (in[i + 7] - mean) * inv;
    out[i + 8] = (in[i + 8] - mean) * inv;
    out[i + 9] = (in[i + 9] - mean) * inv;
    out[i + 10] = (in[i + 10] - mean) * inv;
    out[i + 11] = (in[i + 11] - mean) * inv;
    out[i + 12] = (in[i + 12] - mean) * inv;
    out[i + 13] = (in[i + 13] - mean) * inv;
    out[i + 14] = (in[i + 14] - mean) * inv;
    out[i + 15] = (in[i + 15] - mean) * inv;
  }
}
