//===- matmul_skeleton.hip - portable tiled-matmul template ----------------===//
//
// kernscope kernels/templates reference template (todo 24). Tiled matmul
// skeleton: C[M,N] = A[M,K] @ B[K,N] with a 16x16 shared-memory tile,
// fp32 smem accumulation, bf16 in/out (todo 23 rounding contract). Same
// single-source portable convention as elementwise_add.hip; compiled under
// nvcc -arch=sm_80/sm_90 (compile-only, GPU-less) and hipcc
// --offload-arch=gfx1101 (compile gate; the tuned matmul variants of todos
// 25/27 build on this skeleton and add pipelining/swizzle/vectorization).
//
// Launch geometry: grid (ceil(M/16), ceil(N/16)), block (16, 16); thread
// (ty, tx) computes tile row = blockIdx.y*16+ty, tile col =
// blockIdx.x*16+tx. Bounds-checked so non-multiple M/N/K are safe.
//
//===----------------------------------------------------------------------===//

#include "kernscope/kernel_traits.h"

namespace {
constexpr int kTile = 16; // per-side tile size (smem staging square)
} // namespace

extern "C" __global__ void kernscope_templates_matmul_skeleton(
    const ks_bf16 *a, const ks_bf16 *b, ks_bf16 *out, int m, int n, int k) {
  // fp32 smem staging with +1 column padding (bank-conflict-avoidance
  // skeleton; layout tuning is todo 25's search space).
  __shared__ float a_s[kTile][kTile + 1];
  __shared__ float b_s[kTile][kTile + 1];

  const int row = blockIdx.y * kTile + threadIdx.y;
  const int col = blockIdx.x * kTile + threadIdx.x;

  float acc = 0.0f;
  for (int k0 = 0; k0 < k; k0 += kTile) {
    // Coalesced global loads: threadIdx.x walks the K dimension of A and
    // the N dimension of B (columns of both tiles are contiguous).
    const int a_k = k0 + threadIdx.x;
    const int b_k = k0 + threadIdx.y;
    a_s[threadIdx.y][threadIdx.x] =
        (row < m && a_k < k) ? ks_bf16_to_float(a[row * k + a_k]) : 0.0f;
    b_s[threadIdx.y][threadIdx.x] =
        (b_k < k && col < n) ? ks_bf16_to_float(b[b_k * n + col]) : 0.0f;
    __syncthreads();

    for (int kk = 0; kk < kTile; ++kk) {
      acc += a_s[threadIdx.y][kk] * b_s[kk][threadIdx.x];
    }
    __syncthreads(); // guard the next tile's smem write-back
  }

  if (row < m && col < n) {
    out[row * n + col] = ks_float_to_bf16(acc);
  }
}
