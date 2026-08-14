//===- flash_attention_gen.cpp - FlashAttention-lite template generator ----===//
//
// Implementation of kernscope::tune::flash_attention_gen (todo 27). Emits
// single-source portable .hip kernels implementing online-softmax
// FlashAttention (Dao et al. NeurIPS 2022) over a documented search space,
// validates every candidate PRE-COMPILE (search-space membership, block
// divisibility, vector-width tiling, smem budget via libs/arch) and exposes
// the 4N²d FLOPs model cross-checked against libs/roofline.
//
// Lite scope: forward only — no dropout, no backward, no paged KV, no
// FlashAttention-2 parity claim (documented in the header + tools/tune/README).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/flash_attention_gen.h"

#include "kernscope/arch.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>

namespace kernscope::tune {

namespace {

// kThreadsX = d / vec_width: columns per thread group. Both arch families cap
// a block at 1024 threads, so threads_y is clamped to keep threads_x *
// threads_y <= 1024.
int threads_x(const FlashAttentionConfig &cfg, const FlashAttentionProblem &prob) noexcept {
  return static_cast<int>(prob.d / cfg.vec_width);
}

int threads_y(const FlashAttentionConfig &cfg, const FlashAttentionProblem &prob) noexcept {
  const int tx = threads_x(cfg, prob);
  const int max_y = tx > 0 ? 1024 / tx : 1;
  const int ty = cfg.tile_m < max_y ? cfg.tile_m : max_y;
  return ty > 0 ? ty : 1;
}

int rows_per_thread(const FlashAttentionConfig &cfg, const FlashAttentionProblem &prob) noexcept {
  const int ty = threads_y(cfg, prob);
  return (cfg.tile_m + ty - 1) / ty;
}

int d_pitch(const FlashAttentionConfig &cfg, const FlashAttentionProblem &prob) noexcept {
  // Padding layout pads to d+1; XOR swizzle uses row stride d (d % 4 == 0 is
  // guaranteed by validate_config, so (col ^ (row & 3)) stays in the row).
  return cfg.smem_layout == SmemLayout::kPadding ? static_cast<int>(prob.d) + 1
                                                 : static_cast<int>(prob.d);
}

} // namespace

const char *smem_layout_name(SmemLayout layout) noexcept {
  switch (layout) {
  case SmemLayout::kPadding:
    return "padding";
  case SmemLayout::kXorSwizzle:
    return "xor_swizzle";
  }
  return "unknown";
}

std::int64_t smem_bytes(const FlashAttentionConfig &cfg,
                        const FlashAttentionProblem &prob) noexcept {
  const int pitch = d_pitch(cfg, prob);
  // Q tile + stages-double-buffered K/V tiles + S tile (+1 pad) + m/l arrays,
  // all fp32 (4 bytes/element). See the header for the documented formula.
  const std::int64_t elements = static_cast<std::int64_t>(cfg.tile_m) * pitch +
                                2 * static_cast<std::int64_t>(cfg.stages) *
                                    cfg.tile_n * pitch +
                                static_cast<std::int64_t>(cfg.tile_m) * (cfg.tile_n + 1) +
                                2 * static_cast<std::int64_t>(cfg.tile_m);
  return 4 * elements;
}

ConfigVerdict validate_config(const FlashAttentionConfig &cfg,
                              const FlashAttentionProblem &prob) noexcept {
  // --- search-space membership ----------------------------------------------
  if (cfg.tile_m != 32 && cfg.tile_m != 64)
    return {false, "tile_m (Br) must be 32 or 64 (search space)"};
  if (cfg.tile_n != 32 && cfg.tile_n != 64)
    return {false, "tile_n (Bc) must be 32 or 64 (search space)"};
  if (cfg.stages != 1 && cfg.stages != 2)
    return {false, "stages must be 1 or 2 (search space)"};
  if (cfg.vec_width != 1 && cfg.vec_width != 2 && cfg.vec_width != 4)
    return {false, "vec_width must be 1, 2 or 4 (search space)"};

  // --- problem sanity --------------------------------------------------------
  if (prob.nq <= 0 || prob.nk <= 0 || prob.d <= 0)
    return {false, "problem dims (nq, nk, d) must be positive"};

  // --- block divisibility (the lite kernel is unguarded by design) ----------
  if (prob.nq % cfg.tile_m != 0)
    return {false, "nq (" + std::to_string(prob.nq) +
                       ") not divisible by tile_m (Br = " +
                       std::to_string(cfg.tile_m) + "): lite kernel is unguarded"};
  if (prob.nk % cfg.tile_n != 0)
    return {false, "nk (" + std::to_string(prob.nk) +
                       ") not divisible by tile_n (Bc = " +
                       std::to_string(cfg.tile_n) + "): lite kernel is unguarded"};

  // --- vector-width tiling: threads_x = d/vec must tile Bc exactly ----------
  if (prob.d % cfg.vec_width != 0)
    return {false, "head dim d (" + std::to_string(prob.d) +
                       ") not divisible by vec_width (" +
                       std::to_string(cfg.vec_width) + ")"};
  const int tx = threads_x(cfg, prob);
  if (cfg.tile_n % tx != 0)
    return {false, "tile_n (Bc = " + std::to_string(cfg.tile_n) +
                       ") not divisible by threads_x = d/vec_width (" +
                       std::to_string(tx) + "): S-row column mapping must tile Bc"};

  // --- XOR swizzle constraint (permutation must stay inside a row) ----------
  if (cfg.smem_layout == SmemLayout::kXorSwizzle && prob.d % 4 != 0)
    return {false, "XOR swizzle requires d % 4 == 0 (got d = " +
                       std::to_string(prob.d) + ")"};

  // --- shared-memory budget via libs/arch ------------------------------------
  const ArchInfo *arch = nullptr;
  try {
    arch = &get_arch(prob.arch);
  } catch (const std::out_of_range &) {
    return {false, "unknown arch '" + prob.arch + "' (libs/arch DB)"};
  }
  if (arch->smem_per_block_max_bytes <= 0)
    return {false, "arch '" + prob.arch +
                       "' has no modeled per-block smem budget"};
  const std::int64_t bytes = smem_bytes(cfg, prob);
  if (bytes > arch->smem_per_block_max_bytes)
    return {false,
            "smem budget exceeded: " + std::to_string(bytes) +
                " bytes needed > " +
                std::to_string(arch->smem_per_block_max_bytes) +
                " available per block on " + std::string(arch->name)};
  return {true, ""};
}

LaunchGeometry launch_geometry(const FlashAttentionConfig &cfg,
                               const FlashAttentionProblem &prob) noexcept {
  LaunchGeometry g;
  g.grid_x = cfg.tile_m > 0 ? static_cast<int>(prob.nq / cfg.tile_m) : 1;
  g.grid_y = 1;
  g.block_x = threads_x(cfg, prob);
  g.block_y = threads_y(cfg, prob);
  return g;
}

std::string kernel_symbol(const FlashAttentionConfig &cfg) {
  std::string sym = "kernscope_tune_flash_attention_br";
  sym += std::to_string(cfg.tile_m);
  sym += "_bc" + std::to_string(cfg.tile_n);
  sym += "_s" + std::to_string(cfg.stages);
  sym += "_v" + std::to_string(cfg.vec_width);
  sym += cfg.smem_layout == SmemLayout::kPadding ? "_pad" : "_swz";
  if (cfg.causal)
    sym += "_causal";
  return sym;
}

std::string generate_source(const FlashAttentionConfig &cfg,
                            const FlashAttentionProblem &prob,
                            std::string *error) {
  const ConfigVerdict verdict = validate_config(cfg, prob);
  if (!verdict.valid) {
    if (error)
      *error = "invalid FlashAttention config: " + verdict.reason;
    return "";
  }

  const int tx = threads_x(cfg, prob);
  const int ty = threads_y(cfg, prob);
  const int rpt = rows_per_thread(cfg, prob);
  const int pitch = d_pitch(cfg, prob);
  const bool swizzle = cfg.smem_layout == SmemLayout::kXorSwizzle;
  const std::string symbol = kernel_symbol(cfg);

  char scale_buf[32];
  std::snprintf(scale_buf, sizeof(scale_buf), "%.9g",
                1.0f / std::sqrt(static_cast<float>(prob.d)));

  std::ostringstream out;
  out << "//===- " << symbol
      << ".hip - FlashAttention-lite template (GENERATED) -===//\n"
         "//\n"
         "// GENERATED by kernscope::tune flash-attention generator (todo 27) "
         "— do not edit.\n"
         "// Config: Br="
      << cfg.tile_m << " Bc=" << cfg.tile_n << " stages=" << cfg.stages
      << " vec_width=" << cfg.vec_width << " layout=" << smem_layout_name(cfg.smem_layout)
      << " causal=" << (cfg.causal ? "true" : "false") << ".\n"
         "//\n"
         "// LITE SCOPE: forward pass only — no dropout, no backward, no paged "
         "KV, no\n"
         "// FlashAttention-2 parity claim (tools/tune/README.md). Algorithm: "
         "online-\n"
         "// softmax FlashAttention (Dao et al., NeurIPS 2022): running max, "
         "rescale and\n"
         "// accumulate across the K/V tiles, so O accumulates correctly in one "
         "pass over Q.\n"
         "// Single-source portable convention (kernels/templates, todo 24): "
         "compiles under\n"
         "// nvcc -x cu (sm_80/sm_90) and hipcc (gfx1101). bf16 in, fp32 "
         "accumulate, bf16 out;\n"
         "// 1/sqrt(d) scaling matches the golden attention reference "
         "(tests/golden refs.attention).\n"
         "//\n"
         "//===----------------------------------------------------------------------===//\n"
         "\n"
         "#include \"kernscope/kernel_traits.h\"\n"
         "\n"
         "namespace {\n"
         "constexpr int kBr = "
      << cfg.tile_m << ";            // tile_m: query rows per block\n"
         "constexpr int kBc = "
      << cfg.tile_n << ";            // tile_n: KV rows per tile\n"
         "constexpr int kStages = "
      << cfg.stages << ";     // software-pipelined K/V buffers\n"
         "constexpr int kVec = "
      << cfg.vec_width << ";        // columns per thread (threads_x = d / kVec)\n"
         "constexpr int kDPitch = "
      << pitch << ";    // row stride for Q/K/V tiles (d"
      << (swizzle ? ", XOR swizzle" : " + 1, padding")
      << ")\n"
         "constexpr int kSPitch = "
      << cfg.tile_n + 1 << ";    // S tile row stride (+1 pad, layout-independent)\n"
         "constexpr int kThreadsX = "
      << tx << ";  // d / kVec\n"
         "constexpr int kThreadsY = "
      << ty << ";  // min(Br, 1024/kThreadsX)\n"
         "constexpr int kRowsPerThread = "
      << rpt << ";  // ceil(Br / kThreadsY)\n"
         "constexpr float kScale = "
      << scale_buf << ";  // 1/sqrt(d)\n"
         "constexpr bool kCausal = "
      << (cfg.causal ? "true" : "false") << ";\n"
         "} // namespace\n"
         "\n";

  if (swizzle) {
    out << "// XOR swizzle: column ^= (row & 3) permutes banks across rows while\n"
           "// staying inside the row (d % 4 == 0 guaranteed by validate_config).\n"
           "inline __device__ int tile_index(int row, int col) {\n"
           "  return row * kDPitch + (col ^ (row & 3));\n"
           "}\n";
  } else {
    out << "// Padding layout: row stride d+1 (bank-conflict-avoidance pad).\n"
           "inline __device__ int tile_index(int row, int col) {\n"
           "  return row * kDPitch + col;\n"
           "}\n";
  }

  out << "\nextern \"C\" __global__ void " << symbol
      << "(const ks_bf16 *q, const ks_bf16 *k, const ks_bf16 *v, ks_bf16 *out,\n"
         "                             int nq, int nk, int d) {\n"
         "  __shared__ float q_s[kBr * kDPitch];\n"
         "  __shared__ float k_s[kStages * kBc * kDPitch];\n"
         "  __shared__ float v_s[kStages * kBc * kDPitch];\n"
         "  __shared__ float s_s[kBr * kSPitch];\n"
         "  __shared__ float m_s[kBr];\n"
         "  __shared__ float l_s[kBr];\n"
         "\n"
         "  const int tx = threadIdx.x;\n"
         "  const int ty = threadIdx.y;\n"
         "  const int tid = ty * kThreadsX + tx;\n"
         "  const int nthreads = kThreadsX * kThreadsY;\n"
         "  const int q0 = blockIdx.x * kBr;\n"
         "\n"
         "  // Cooperative load of the Br x d Q tile (resident for the whole "
         "block).\n"
         "  for (int e = tid; e < kBr * d; e += nthreads) {\n"
         "    const int r = e / d;\n"
         "    const int c = e % d;\n"
         "    q_s[tile_index(r, c)] = ks_bf16_to_float(q[(q0 + r) * d + c]);\n"
         "  }\n"
         "\n"
         "  // Per-thread O accumulator in registers; m/l row state lives in "
         "shared\n"
         "  // memory so every column thread of a row shares the running max/"
         "sum.\n"
         "  float acc[kRowsPerThread][kVec];\n"
         "  for (int rr = 0; rr < kRowsPerThread; ++rr)\n"
         "    for (int vv = 0; vv < kVec; ++vv) acc[rr][vv] = 0.0f;\n"
         "\n"
         "  if (tid == 0) {\n"
         "    for (int r = 0; r < kBr; ++r) {\n"
         "      m_s[r] = -INFINITY;  // first tile: alpha = exp(-inf - m_new) = 0\n"
         "      l_s[r] = 0.0f;\n"
         "    }\n"
         "  }\n"
         "  __syncthreads();\n"
         "\n"
         "  const int nblocks = nk / kBc;\n"
         "  for (int it = 0; it < nblocks; ++it) {\n"
         "    const int cur = it % kStages;\n"
         "    const int nxt = (it + 1) % kStages;\n"
         "\n"
         "    // Cooperative load of the Bc x d K/V tiles for block `it` into "
         "buffer `cur`.\n"
         "    for (int e = tid; e < kBc * d; e += nthreads) {\n"
         "      const int j = e / d;\n"
         "      const int c = e % d;\n"
         "      const int gj = it * kBc + j;\n"
         "      k_s[cur * kBc * kDPitch + tile_index(j, c)] = "
         "ks_bf16_to_float(k[gj * d + c]);\n"
         "      v_s[cur * kBc * kDPitch + tile_index(j, c)] = "
         "ks_bf16_to_float(v[gj * d + c]);\n"
         "    }\n"
         "    __syncthreads();\n"
         "\n"
         "    // S = (Q K^T) / sqrt(d): thread (tx, ty) covers the S rows it "
         "owns,\n"
         "    // columns j == tx (mod kThreadsX).\n"
         "    for (int rr = 0; rr < kRowsPerThread; ++rr) {\n"
         "      const int r = ty + rr * kThreadsY;\n"
         "      if (r >= kBr) continue;\n"
         "      const int gi = q0 + r;\n"
         "      for (int j = tx; j < kBc; j += kThreadsX) {\n"
         "        const int gj = it * kBc + j;\n"
         "        float s = 0.0f;\n"
         "        for (int kk = 0; kk < d; ++kk)\n"
         "          s += q_s[tile_index(r, kk)] * k_s[cur * kBc * kDPitch + "
         "tile_index(j, kk)];\n"
         "        s *= kScale;\n"
         "        if (kCausal && gj > gi) s = -INFINITY;  // causal mask\n"
         "        s_s[r * kSPitch + j] = s;\n"
         "      }\n"
         "    }\n"
         "    __syncthreads();\n"
         "\n"
         "    // Online-softmax update (running max + rescale + accumulate) and\n"
         "    // O accumulation into the register accumulator.\n"
         "    for (int rr = 0; rr < kRowsPerThread; ++rr) {\n"
         "      const int r = ty + rr * kThreadsY;\n"
         "      if (r >= kBr) continue;\n"
         "      const int gi = q0 + r;\n"
         "      // Causal: a K/V block entirely in the future of row gi (it*kBc\n"
         "      // > gi) contributes NOTHING to that row — every score is masked,\n"
         "      // so m_new would collapse to -inf and expf(m_old - m_new) to NaN.\n"
         "      // Skipping the update keeps m/l/O untouched (same as the golden\n"
         "      // reference, where exp(-inf) == 0 in the softmax row).\n"
         "      if (kCausal && it * kBc > gi) continue;\n"
         "      const float m_old = m_s[r];\n"
         "      const float l_old = l_s[r];\n"
         "      float m_new = -INFINITY;\n"
         "      for (int j = 0; j < kBc; ++j) m_new = fmaxf(m_new, "
         "s_s[r * kSPitch + j]);\n"
         "      const float alpha = expf(m_old - m_new);\n"
         "      float l_new = 0.0f;\n"
         "      for (int j = 0; j < kBc; ++j) l_new += expf(s_s[r * kSPitch + j] "
         "- m_new);\n"
         "      const float l_final = alpha * l_old + l_new;\n"
         "      for (int vv = 0; vv < kVec; ++vv) acc[rr][vv] *= alpha;\n"
         "      const int c0 = tx * kVec;\n"
         "      for (int vv = 0; vv < kVec; ++vv) {\n"
         "        float sumv = 0.0f;\n"
         "        for (int j = 0; j < kBc; ++j) {\n"
         "          const float p = expf(s_s[r * kSPitch + j] - m_new);\n"
         "          sumv += p * v_s[cur * kBc * kDPitch + tile_index(j, c0 + vv)];\n"
         "        }\n"
         "        acc[rr][vv] += sumv;\n"
         "      }\n"
         "      // Single-writer row-state update (tx == 0 owns row r's m/l).\n"
         "      if (tx == 0) {\n"
         "        m_s[r] = m_new;\n"
         "        l_s[r] = l_final;\n"
         "      }\n"
         "    }\n"
         "    __syncthreads();\n"
         "\n"
         "    // Prefetch the next K/V block into buffer `nxt` (software "
         "pipelining;\n"
         "    // synchronous loads for lite — the buffer alternation is what the\n"
         "    // stages dimension searches).\n"
         "    if (it + 1 < nblocks) {\n"
         "      const int nit = it + 1;\n"
         "      for (int e = tid; e < kBc * d; e += nthreads) {\n"
         "        const int j = e / d;\n"
         "        const int c = e % d;\n"
         "        const int gj = nit * kBc + j;\n"
         "        k_s[nxt * kBc * kDPitch + tile_index(j, c)] = "
         "ks_bf16_to_float(k[gj * d + c]);\n"
         "        v_s[nxt * kBc * kDPitch + tile_index(j, c)] = "
         "ks_bf16_to_float(v[gj * d + c]);\n"
         "      }\n"
         "    }\n"
         "    __syncthreads();\n"
         "  }\n"
         "\n"
         "  // Epilogue: O = acc / l, rounded to bf16 exactly once (todo-23 "
         "contract).\n"
         "  for (int rr = 0; rr < kRowsPerThread; ++rr) {\n"
         "    const int r = ty + rr * kThreadsY;\n"
         "    if (r >= kBr) continue;\n"
         "    const int gi = q0 + r;\n"
         "    const float l_final = l_s[r];\n"
         "    for (int vv = 0; vv < kVec; ++vv) {\n"
         "      const int c = tx * kVec + vv;\n"
         "      out[gi * d + c] = ks_float_to_bf16(acc[rr][vv] / l_final);\n"
         "    }\n"
         "  }\n"
         "}\n";

  return out.str();
}

bool generate_to_file(const FlashAttentionConfig &cfg,
                      const FlashAttentionProblem &prob,
                      const std::string &path, std::string *error) {
  const std::string src = generate_source(cfg, prob, error);
  if (src.empty())
    return false;
  std::ofstream fh(path, std::ios::binary | std::ios::trunc);
  if (!fh) {
    if (error)
      *error = "cannot open " + path + " for writing";
    return false;
  }
  fh << src;
  if (!fh) {
    if (error)
      *error = "write failed for " + path;
    return false;
  }
  return true;
}

} // namespace kernscope::tune
