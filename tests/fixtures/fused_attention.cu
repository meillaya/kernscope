// fused_attention.cu — golden fixture for the kernscope static analyzer.
//
// Deliberately engineered fused-attention kernel exhibiting every warning /
// suggestion class from the owner's analyzer example (docs/report/kernscope-technical-report.pdf, Section A):
//
//   (a) accumulator array   : per-thread float acc[64] drives the register
//                             pressure into the 97-112 range (~112 regs/thread
//                             at --maxrregcount=112; see manifest.json)
//   (b) divergent branch    : `if ((lane & 3) == (j & 3))` inside the inner
//                             j-loop — a threadIdx-derived predicate that is
//                             warp-divergent for every j (class b)
//   (c) bank conflicts      : sK / sV shared tiles carry a 68-float row stride
//                             (HEAD_DIM + 4 pad); the column-major accesses
//                             sK[tid][j] / sV[tid][k] walk a stride of 68
//                             floats == 4 (mod 32) over the 32-bank layout ->
//                             8 distinct banks -> exactly 4-way conflicts
//   (d) uncoalesced loads   : the Q tile is staged via stride-2 global reads
//                             (addresses 2*i / 2*i+1) — each warp access
//                             spans 256 B for 128 B of useful data ->
//                             coalescing efficiency 0.5
//
// Barriers: __syncthreads() after tile staging and after the score phase.
// Instruction mix: FMA chains (fmaf), ld.shared / ld.global, st.shared /
// st.global, ex2.approx.f32 (softmax via __expf), setp/bra (divergence),
// bar.sync — so instruction-mix analysis has content.
//
// The tile working set (66 KiB) exceeds the 48 KiB static shared-memory
// limit, so the tiles live in DYNAMIC shared memory (extern __shared__),
// exactly like real fused-attention kernels; main() documents the runtime
// opt-in (cudaFuncAttributeMaxDynamicSharedMemorySize). This fixture is a
// compile-only golden (no local NVIDIA GPU), so the launch is never run.
//
// Compile (ptxas runs GPU-less):
//   nvcc -arch=sm_86 -lineinfo --maxrregcount=112 -ptx \
//        tests/fixtures/fused_attention.cu -o tests/fixtures/fused_attention.ptx
// Register gate (97-112, review-mandated) verified via `-Xptxas -v`.

#include <cuda_runtime.h>

#define BLOCK_SIZE 64
#define HEAD_DIM   64
#define SEQ_TILE   64
// Shared tiles are padded to 68 floats per row: 68 % 32 == 4, so a
// column-major (stride-68) access collides 4-way on the 32-bank layout.
#define SMEM_STRIDE (HEAD_DIM + 4)

// Dynamic shared-memory layout (floats):
//   sQ      [SEQ_TILE][HEAD_DIM]    = 64*64 = 4096 floats  (16 KiB)
//   sK      [SEQ_TILE][SMEM_STRIDE] = 64*68 = 4352 floats  (17 KiB, padded)
//   sV      [HEAD_DIM][SMEM_STRIDE] = 64*68 = 4352 floats  (17 KiB, padded)
//   sScore  [SEQ_TILE][HEAD_DIM]    = 64*64 = 4096 floats  (16 KiB)
// total 16896 floats = 67584 B = 66 KiB (opt-in via cudaFuncSetAttribute)
#define SMEM_FLOATS (SEQ_TILE * HEAD_DIM + SEQ_TILE * SMEM_STRIDE + \
                     HEAD_DIM * SMEM_STRIDE + SEQ_TILE * HEAD_DIM)
#define SMEM_BYTES (SMEM_FLOATS * sizeof(float))

__global__ void fused_attention(const float* __restrict__ Q,
                                const float* __restrict__ K,
                                const float* __restrict__ V,
                                float* __restrict__ O,
                                int seq_len)
{
    extern __shared__ float smem[];
    float* const sQ     = smem;                              // 4096 floats
    float* const sK     = sQ + SEQ_TILE * HEAD_DIM;          // 4352 floats
    float* const sV     = sK + SEQ_TILE * SMEM_STRIDE;       // 4352 floats
    float* const sScore = sV + HEAD_DIM * SMEM_STRIDE;       // 4096 floats

    const int tid  = threadIdx.x;   // 0..63; one output column per thread
    const int lane = tid & 31;

    // ---- stage Q / K / V tiles ---------------------------------------------
    // Q: stride-2 gather (flat addresses 2*i and 2*i+1) — each warp access
    //    spans 256 B for 128 B of useful data -> coalescing efficiency 0.5
    //    (class d).
    for (int i = tid; i < (SEQ_TILE * HEAD_DIM) / 2; i += BLOCK_SIZE) {
        sQ[2 * i]     = Q[2 * i];
        sQ[2 * i + 1] = Q[2 * i + 1];
    }
    // K / V: contiguous cooperative loads (coalesced; deliberate contrast).
    for (int i = tid; i < SEQ_TILE * HEAD_DIM; i += BLOCK_SIZE) {
        sK[(i / HEAD_DIM) * SMEM_STRIDE + (i % HEAD_DIM)] = K[i];  // pad cols unused
        sV[(i % HEAD_DIM) * SMEM_STRIDE + (i / HEAD_DIM)] = V[i];  // transposed: sV[col][k]
    }
    __syncthreads();

    // ---- scores: sScore[r][tid] = sum_j Q[r][j] * K[tid][j] -----------------
    // 8-wide unrolled dot product -> 8 independent accumulators (ILP + regs).
    for (int r = 0; r < SEQ_TILE; ++r) {
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
        float s4 = 0.f, s5 = 0.f, s6 = 0.f, s7 = 0.f;
        for (int j = 0; j < HEAD_DIM; j += 8) {
            const float qv0 = sQ[r * HEAD_DIM + j + 0];    // broadcast: same word
            const float qv1 = sQ[r * HEAD_DIM + j + 1];
            const float qv2 = sQ[r * HEAD_DIM + j + 2];
            const float qv3 = sQ[r * HEAD_DIM + j + 3];
            const float qv4 = sQ[r * HEAD_DIM + j + 4];
            const float qv5 = sQ[r * HEAD_DIM + j + 5];
            const float qv6 = sQ[r * HEAD_DIM + j + 6];
            const float qv7 = sQ[r * HEAD_DIM + j + 7];
            const float kv0 = sK[tid * SMEM_STRIDE + j + 0]; // col-major stride 68
            const float kv1 = sK[tid * SMEM_STRIDE + j + 1]; // -> 4-way (class c)
            const float kv2 = sK[tid * SMEM_STRIDE + j + 2];
            const float kv3 = sK[tid * SMEM_STRIDE + j + 3];
            const float kv4 = sK[tid * SMEM_STRIDE + j + 4];
            const float kv5 = sK[tid * SMEM_STRIDE + j + 5];
            const float kv6 = sK[tid * SMEM_STRIDE + j + 6];
            const float kv7 = sK[tid * SMEM_STRIDE + j + 7];
            float qk0 = qv0 * kv0;
            float qk1 = qv1 * kv1;
            float qk2 = qv2 * kv2;
            float qk3 = qv3 * kv3;
            float qk4 = qv4 * kv4;
            float qk5 = qv5 * kv5;
            float qk6 = qv6 * kv6;
            float qk7 = qv7 * kv7;
            // divergent branch inside the inner loop (class b): the predicate
            // (lane & 7) == (j & 7) depends on threadIdx and splits every warp.
            if ((lane & 7) == (j & 7)) {
                qk0 *= 0.5f;
                qk1 *= 0.5f;
                qk2 *= 0.5f;
                qk3 *= 0.5f;
                qk4 *= 0.5f;
                qk5 *= 0.5f;
                qk6 *= 0.5f;
                qk7 *= 0.5f;
            }
            s0 += qk0;
            s1 += qk1;
            s2 += qk2;
            s3 += qk3;
            s4 += qk4;
            s5 += qk5;
            s6 += qk6;
            s7 += qk7;
        }
        sScore[r * HEAD_DIM + tid] =
            __expf(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7)));
    }
    __syncthreads();

    // ---- V multiply: O[r][tid] = sum_k sScore[r][k] * V[tid][k] -------------
    // 64-float accumulator array per thread (class a: register pressure).
    // The r-loop is fully unrolled so the accumulator array is register-
    // resident (a rolled r-loop would demote it to local memory).
    float acc[HEAD_DIM];
#pragma unroll
    for (int r = 0; r < SEQ_TILE; ++r) {
        acc[r] = 0.f;
    }
#pragma unroll
    for (int r = 0; r < SEQ_TILE; ++r) {
#pragma unroll 4
        for (int k = 0; k < HEAD_DIM; k += 4) {
            const float p0 = sScore[r * HEAD_DIM + k + 0];  // broadcast
            const float p1 = sScore[r * HEAD_DIM + k + 1];
            const float p2 = sScore[r * HEAD_DIM + k + 2];
            const float p3 = sScore[r * HEAD_DIM + k + 3];
            const float v0 = sV[tid * SMEM_STRIDE + k + 0]; // col-major stride 68
            const float v1 = sV[tid * SMEM_STRIDE + k + 1]; // -> 4-way (class c)
            const float v2 = sV[tid * SMEM_STRIDE + k + 2];
            const float v3 = sV[tid * SMEM_STRIDE + k + 3];
            acc[r] = fmaf(p0, v0, acc[r]);
            acc[r] = fmaf(p1, v1, acc[r]);
            acc[r] = fmaf(p2, v2, acc[r]);
            acc[r] = fmaf(p3, v3, acc[r]);
        }
    }

    // ---- write outputs: column `tid` of O (coalesced: stride-1 across warp) -
    for (int r = 0; r < SEQ_TILE; ++r) {
        O[r * HEAD_DIM + tid] = acc[r];
    }
}

// Host driver documenting the intended launch config + 66 KiB dynamic smem
// opt-in. Compile-only golden: never executed on this machine (no local
// NVIDIA GPU). nvcc -ptx emits device PTX only; this host code is not part
// of the committed .ptx artifact.
int main()
{
    cudaFuncSetAttribute(fused_attention,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SMEM_BYTES);
    // placeholder launch config (documentation only):
    // fused_attention<<<gridDim, BLOCK_SIZE, SMEM_BYTES>>>(Q, K, V, O, seq_len);
    return 0;
}
