// Showcase kernel: tiled sgemm (todo 42, Wave 7) — FRESH REWRITE (the
// /home/mei/projects/tensors/cuda/kernels gemm is a SEMANTIC reference
// only; nothing is copied).
//
// One output element per thread, block 16x16, grid 4x4 covering the 64x64
// C tile; the k-loop runs 0..99 with LITERAL bounds (compile-time constants
// — SCEV-provable, the shape the kernscope passes consume; a parameter
// bound would be declined by every pass, see softmax/kernel.cu for that
// honest no-transform case).
//
// Pass-pipeline story (what the artifacts must show):
//   * kernscope-tile   — k-loop 0..99, tile 32 (pinned via fn attribute in
//     the artifact pipeline, same mechanism as the tile fixtures): 3 full
//     tiles + a 4-iteration remainder loop.
//   * kernscope-promote — the a-row window {i*100,+,1} of the remainder
//     loop (constant start, constant trip 4) is promoted into a cooperative
//     block load into addrspace(3) + llvm.nvvm.barrier0; the b-access
//     b[k*64+j] is {j,+,64}-strided in k -> v1 unit-stride decline (honest
//     partial promotion, reflected in the report).
//   * kernscope-vectorize — v1 rewrites global chains only; after promote
//     the reads are smem -> untouched (the pipeline stays clean).
//   * kernscope-coalesce — b-loads/stores: j = tid.x -> per-lane stride 1
//     element -> contiguous-128B (fully); a-loads: row i depends on tid.y
//     (warp-varying, blockDim.x = 16 -> half-warp rows) -> the per-lane
//     affine is unresolvable in tid.x -> unknown-pattern (HONEST — never
//     claimed coalesced; documented v1 limitation of the classifier).
//
// Portable CUDA/HIP: hipcc --genco consumes the same source (the pass
// pipeline + PTX use clang with a staged CUDA root; the live gfx1101 bench
// uses the hipcc-compiled hsaco).

#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#ifndef TILE_KS
#define TILE_KS 32
#endif
#define M_SHOW 64
#define N_SHOW 64
#define K_SHOW 100

__global__ void gemm_kernel(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c) {
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  float acc = 0.0f;
  for (int k = 0; k < K_SHOW; ++k) {
    acc += a[i * K_SHOW + k] * b[k * N_SHOW + j];
  }
  c[i * N_SHOW + j] = acc;
}
