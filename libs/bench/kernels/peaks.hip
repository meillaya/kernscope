//===- peaks.hip - gfx1101 roofline peak microbenchmarks ----------*- HIP -*-===//
//
// kernscope roofline peak microbenches (wave-4 todo 22), compiled at BUILD
// time by `hipcc --genco --offload-arch=gfx1101` into peaks.hsaco and loaded
// by kernscope-bench / the bench live suite through Module::load (same
// convention as libs/gpuruntime/tests/kernels/add_kernel.hip). Every kernel is
// a DRAM/LDS/ALU microbenchmark whose byte/FLOP count is defined in
// tools/bench/main.cpp and the live gtest — the kernel itself only moves data
// or issues FMAs, it never computes a peak.
//
// Kernels (extern "C" so the .hsaco symbol names are unmangled):
//   kernscope_peak_memcpy  D2D copy, grid-stride, 16 B/thread/iter
//                           (DRAM write+read bandwidth peak)
//   kernscope_peak_read    read-only accumulation, grid-stride 16 B/iter
//                           (DRAM read bandwidth peak)
//   kernscope_peak_fma     register-resident fp32 FFMA loops, 8 independent
//                           chains + 1 varying operand (fp32 FLOP peak)
//   kernscope_peak_lds     LDS read+write loop (shared-memory bandwidth)
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime.h>
#include <cstdint>

//--- D2D copy: 16 bytes (float4) moved per lane per iteration ---------------//
// Bytes moved by the CALLER counts read+write: 2 * size.
extern "C" __global__ void kernscope_peak_memcpy(const float4 *__restrict__ src,
                                                 float4 *__restrict__ dst,
                                                 std::int64_t n4) {
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  for (std::int64_t i =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    dst[i] = src[i];
  }
}

//--- read-only: accumulate; never-true conditional keeps loads alive --------//
// Bytes moved by the CALLER counts 1 * size (reads only; the one-lane
// conditional write is never executed).
extern "C" __global__ void kernscope_peak_read(const float4 *__restrict__ src,
                                               float *__restrict__ out,
                                               std::int64_t n4) {
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * blockDim.x;
  float4 acc{0.0f, 0.0f, 0.0f, 0.0f};
  for (std::int64_t i =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n4; i += stride) {
    const float4 v = src[i];
    acc.x += v.x;
    acc.y += v.y;
    acc.z += v.z;
    acc.w += v.w;
  }
  const float sum = acc.x + acc.y + acc.z + acc.w;
  if (sum == 1234567.5f) {
    out[0] = sum; // never true; the loads above are provably live
  }
}

//--- fp32 FFMA peak: register-resident, no memory traffic -------------------//
// FLOPs by the CALLER: iters * (kFmaChains + 1) * 2 (8 independent chains plus
// the varying-operand update; each v_fma_f32 = 2 FLOPs). Independent chains
// give the RDNA3 dual-issue scheduler room; `k` changes every iteration so
// the FMAs can never be hoisted/constant-folded.
extern "C" __global__ void kernscope_peak_fma(float *__restrict__ out,
                                              int iters) {
  constexpr int kChains = 8;
  const int tid = threadIdx.x;
  float a[kChains];
  for (int c = 0; c < kChains; ++c) {
    a[c] = 1.0001f + static_cast<float>(tid) * 0.25f + static_cast<float>(c);
  }
  float k = 1.0f;
  for (int j = 0; j < iters; ++j) {
    k = fmaf(k, a[0], a[1]); // varies every iteration: keeps the chains live
#pragma unroll
    for (int c = 0; c < kChains; ++c) {
      a[c] = fmaf(a[c], k, k); // 8 independent v_fma_f32
    }
  }
  float sum = 0.0f;
  for (int c = 0; c < kChains; ++c) {
    sum += a[c];
  }
  if (sum == 1234567.5f) {
    out[0] = sum;
  }
}

//--- LDS bandwidth: per-lane read + write every iteration --------------------//
// Bytes by the CALLER: blocks * blockDim * iters * 8 (4 B read + 4 B write per
// lane per iter). Consecutive-lane addresses are bank-conflict-free; read and
// write hit different halves of the buffer.
extern "C" __global__ void kernscope_peak_lds(float *__restrict__ out,
                                              int iters) {
  __shared__ float smem[1024];
  const int tid = threadIdx.x;
  smem[tid] = static_cast<float>(tid) * 0.25f + 1.0f;
  __syncthreads();
  float acc = 0.0f;
  for (int j = 0; j < iters; ++j) {
    const int r = (tid + (j << 2)) & 1023;
    const int w = (r + 512) & 1023;
    acc += smem[r]; // LDS read
    smem[w] = acc;  // LDS write (bank-disjoint from the read)
  }
  __syncthreads();
  if (acc == 1234567.5f) {
    out[0] = acc;
  }
}
