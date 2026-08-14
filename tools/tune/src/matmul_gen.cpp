//===- matmul_gen.cpp - tiled-matmul template generator -------------------===//
//
// Implementation of kernscope::tune::matmul_gen (todo 25). Emits single-source
// portable .hip kernels implementing shared-memory tiled GEMM
// C[M,N] = A[M,K] @ B[K,N] (bf16 in, fp32 accumulate, bf16 out; todo-23
// rounding contract) over a documented search space, validates every candidate
// PRE-COMPILE (search-space membership, warp/block divisibility, vector-width
// tiling, register budget, smem budget via libs/arch) and refuses to emit
// source for an invalid config.
//
// The emitted kernel is bounds-checked (zero-fill for the OOB M/N/K tail), so
// the problem's M/N/K carry NO divisibility constraint — they are runtime
// kernel arguments (matmul family golden contract, run_hip_gate.py).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/matmul_gen.h"

#include "kernscope/arch.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <utility>

namespace kernscope::tune {

namespace {

// Per-thread register overhead estimate beyond the acc fragment (loop
// induction, smem addresses, bf16 convert temporaries, epilogue): the task
// contract pins "fragments per thread vs max_regs_per_thread" as the rule.
constexpr int kRegisterOverhead = 32;

// Per-thread column fragment = BN / threads_x (threads_x = 32·warps_n).
int cols_per_thread(const MatmulConfig &cfg,
                    const WarpDecomposition &wd) noexcept {
  return cfg.tile_n / (32 * wd.warps_n);
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

std::int64_t smem_bytes(const MatmulConfig &cfg) noexcept {
  const int pad = cfg.smem_layout == SmemLayout::kPadding ? 1 : 0;
  // Per-stage A tile BM×(BK+pad) + B tile BK×(BN+pad), fp32 (4 bytes).
  return static_cast<std::int64_t>(cfg.stages) *
         (static_cast<std::int64_t>(cfg.tile_m) * (cfg.tile_k + pad) +
          static_cast<std::int64_t>(cfg.tile_k) * (cfg.tile_n + pad)) *
         4;
}

WarpDecomposition warp_decomposition(const MatmulConfig &cfg) noexcept {
  // (warps_m, warps_n) factor pair that tiles the output tile:
  // threads_x = 32·warps_n must divide BN and warps_m must divide BM.
  // Prefer the pair whose warp tile (BM/warps_m) × (BN/warps_n) is most
  // square (smallest |BM/warps_m − BN/warps_n|); ties keep the first.
  WarpDecomposition best{0, 0};
  int best_score = -1;
  for (int wm = 1; wm <= cfg.warps; ++wm) {
    if (cfg.warps % wm != 0)
      continue;
    const int wn = cfg.warps / wm;
    const int tx = 32 * wn;
    if (cfg.tile_m % wm != 0 || cfg.tile_n % tx != 0)
      continue;
    const int score = (cfg.tile_m / wm) > (cfg.tile_n / wn)
                          ? (cfg.tile_m / wm) - (cfg.tile_n / wn)
                          : (cfg.tile_n / wn) - (cfg.tile_m / wm);
    if (best_score < 0 || score < best_score) {
      best_score = score;
      best = WarpDecomposition{wm, wn};
    }
  }
  return best;
}

ConfigVerdict validate_config(const MatmulConfig &cfg,
                              const MatmulProblem &prob) noexcept {
  // --- search-space membership ----------------------------------------------
  if (cfg.tile_m != 64 && cfg.tile_m != 128)
    return {false, "tile_m (BM) must be 64 or 128 (search space)"};
  if (cfg.tile_n != 64 && cfg.tile_n != 128)
    return {false, "tile_n (BN) must be 64 or 128 (search space)"};
  if (cfg.tile_k != 8 && cfg.tile_k != 16 && cfg.tile_k != 32 &&
      cfg.tile_k != 64)
    return {false, "tile_k (BK) must be 8, 16, 32 or 64 (search space)"};
  if (cfg.warps != 1 && cfg.warps != 2 && cfg.warps != 4 && cfg.warps != 8)
    return {false, "warps must be 1, 2, 4 or 8 (search space)"};
  if (cfg.stages < 1 || cfg.stages > 4)
    return {false, "stages must be 1, 2, 3 or 4 (search space)"};
  if (cfg.vec_width != 1 && cfg.vec_width != 2 && cfg.vec_width != 4)
    return {false, "vec_width must be 1, 2 or 4 (search space)"};
  if (cfg.unroll != 1 && cfg.unroll != 2 && cfg.unroll != 4)
    return {false, "unroll must be 1, 2 or 4 (search space)"};

  // --- problem sanity --------------------------------------------------------
  if (prob.m <= 0 || prob.n <= 0 || prob.k <= 0)
    return {false, "problem dims (m, n, k) must be positive"};

  // --- warp/block divisibility ----------------------------------------------
  const WarpDecomposition wd = warp_decomposition(cfg);
  if (wd.warps_m == 0)
    return {false,
            "no valid warp decomposition: BM=" + std::to_string(cfg.tile_m) +
                " BN=" + std::to_string(cfg.tile_n) + " warps=" +
                std::to_string(cfg.warps) +
                " (need BM % warps_m == 0 and BN % (32·warps_n) == 0)"};

  // --- threads/block cap (1024 on both arch families; also the arch model) --
  const int nthreads = 32 * cfg.warps;
  if (nthreads > 1024)
    return {false, "threads/block = " + std::to_string(nthreads) +
                       " exceeds the 1024-thread block cap"};
  const ArchInfo *arch = nullptr;
  try {
    arch = &get_arch(prob.arch);
  } catch (const std::out_of_range &) {
    return {false, "unknown arch '" + prob.arch + "' (libs/arch DB)"};
  }
  if (nthreads > arch->max_threads_per_sm)
    return {false,
            "threads/block = " + std::to_string(nthreads) +
                " exceeds " + std::string(arch->name) + "'s " +
                std::to_string(arch->max_threads_per_sm) +
                " max_threads_per_sm"};

  // --- vector-width tiling: kVec must divide the per-thread column fragment -
  const int cpt = cols_per_thread(cfg, wd);
  if (cfg.vec_width > 0 && cpt % cfg.vec_width != 0)
    return {false,
            "vector width " + std::to_string(cfg.vec_width) +
                " must divide the per-thread column fragment cpt = BN/(32·"
                "warps_n) = " +
                std::to_string(cpt)};

  // --- XOR swizzle constraint (permutation must stay inside a row) ----------
  if (cfg.smem_layout == SmemLayout::kXorSwizzle && cfg.tile_k % 4 != 0)
    return {false, "XOR swizzle requires BK % 4 == 0 (got BK = " +
                       std::to_string(cfg.tile_k) + ")"};

  // --- register budget (fragments per thread vs arch max) -------------------
  // acc fragment = rpt·cpt = BM·BN/(32·warps) fp32 registers per thread.
  const int frag_regs = cfg.tile_m * cfg.tile_n / (32 * cfg.warps);
  const int est = frag_regs + kRegisterOverhead;
  if (arch->max_regs_per_thread > 0 && est > arch->max_regs_per_thread)
    return {false,
            "register budget exceeded: " + std::to_string(frag_regs) +
                " acc fragment regs/thread (+" +
                std::to_string(kRegisterOverhead) + " overhead) = " +
                std::to_string(est) + " > " +
                std::to_string(arch->max_regs_per_thread) +
                " max_regs_per_thread on " + std::string(arch->name)};

  // --- shared-memory budget via libs/arch ------------------------------------
  if (arch->smem_per_block_max_bytes <= 0)
    return {false, "arch '" + prob.arch +
                       "' has no modeled per-block smem budget"};
  const std::int64_t bytes = smem_bytes(cfg);
  if (bytes > arch->smem_per_block_max_bytes)
    return {false,
            "smem budget exceeded: " + std::to_string(bytes) +
                " bytes needed > " +
                std::to_string(arch->smem_per_block_max_bytes) +
                " available per block on " + std::string(arch->name)};
  return {true, ""};
}

LaunchGeometry launch_geometry(const MatmulConfig &cfg,
                               const MatmulProblem &prob) noexcept {
  LaunchGeometry g;
  const WarpDecomposition wd = warp_decomposition(cfg);
  g.grid_x = cfg.tile_n > 0 ? static_cast<int>((prob.n + cfg.tile_n - 1) / cfg.tile_n) : 1;
  g.grid_y = cfg.tile_m > 0 ? static_cast<int>((prob.m + cfg.tile_m - 1) / cfg.tile_m) : 1;
  g.block_x = 32 * (wd.warps_n > 0 ? wd.warps_n : 1);
  g.block_y = wd.warps_m > 0 ? wd.warps_m : 1;
  return g;
}

std::string kernel_symbol(const MatmulConfig &cfg) {
  (void)cfg;  // fixed entry symbol the runner resolves (task contract)
  return "kernscope_tune_matmul";
}

std::string generate_source(const MatmulConfig &cfg,
                            const MatmulProblem &prob,
                            std::string *error) {
  const ConfigVerdict verdict = validate_config(cfg, prob);
  if (!verdict.valid) {
    if (error)
      *error = "invalid matmul config: " + verdict.reason;
    return "";
  }

  const WarpDecomposition wd = warp_decomposition(cfg);
  const bool swizzle = cfg.smem_layout == SmemLayout::kXorSwizzle;
  const int pad = swizzle ? 0 : 1;
  const std::string symbol = kernel_symbol(cfg);

  std::ostringstream out;
  out << "//===- " << symbol
      << ".hip - tiled matmul template (GENERATED) -===//\n"
         "//\n"
         "// GENERATED by kernscope::tune matmul generator (todo 25) — do not "
         "edit.\n"
         "// Config: BM="
      << cfg.tile_m << " BN=" << cfg.tile_n << " BK=" << cfg.tile_k
      << " warps=" << cfg.warps << " (" << wd.warps_m << "x" << wd.warps_n
      << ") stages=" << cfg.stages << " vec_width=" << cfg.vec_width
      << " layout=" << smem_layout_name(cfg.smem_layout)
      << " unroll=" << cfg.unroll << ".\n"
         "//\n"
         "// Shared-memory tiled GEMM, C[M,N] = A[M,K] @ B[K,N]: A tile BM×BK "
         "and B tile\n"
         "// BK×BN staged in fp32 shared memory with a "
      << cfg.stages
      << "-deep software pipeline; the K dimension walks in BK-wide chunks;\n"
         "// per-thread fp32 fragment acc[rpt][cpt]; bounds-checked for "
         "non-multiple M/N/K\n"
         "// (OOB loads zero-fill, OOB stores are skipped). bf16 in, fp32 "
         "accumulate, bf16 out\n"
         "// (todo-23 rounding contract: the bf16 round happens exactly once, "
         "in the epilogue).\n"
         "// Single-source portable convention (kernels/templates, todo 24): "
         "compiles under\n"
         "// nvcc -x cu (sm_80/sm_90) and hipcc (gfx1101).\n"
         "//\n"
         "//===----------------------------------------------------------------------===//\n"
         "\n"
         "#include \"kernscope/kernel_traits.h\"\n"
         "\n"
         "namespace {\n"
         "constexpr int kBM = "
      << cfg.tile_m << ";          // tile_m: output rows per block\n"
         "constexpr int kBN = "
      << cfg.tile_n << ";          // tile_n: output cols per block\n"
         "constexpr int kBK = "
      << cfg.tile_k << ";          // tile_k: inner (K) tile size\n"
         "constexpr int kStages = "
      << cfg.stages << ";     // software-pipelined smem tile buffers\n"
         "constexpr int kVec = "
      << cfg.vec_width << ";        // consecutive columns per thread\n"
         "constexpr int kUnroll = "
      << cfg.unroll << ";       // K-loop unroll factor\n"
         "constexpr int kPad = "
      << pad << ";            // smem row pad (0 XOR swizzle, 1 padding)\n"
         "constexpr int kAPitch = kBK + kPad;  // A tile row stride\n"
         "constexpr int kBPitch = kBN + kPad;  // B tile row stride\n"
         "constexpr int kWarpsM = "
      << wd.warps_m << ";      // block y-dimension warps\n"
         "constexpr int kWarpsN = "
      << wd.warps_n << ";      // block x-dimension warps\n"
         "constexpr int kThreadsX = 32 * kWarpsN;  // threads covering BN\n"
         "constexpr int kThreadsY = kWarpsM;       // threads covering BM\n"
         "constexpr int kRpt = kBM / kThreadsY;    // fragment rows per thread\n"
         "constexpr int kCpt = kBN / kThreadsX;    // fragment cols per thread\n"
         "} // namespace\n"
         "\n";

  if (swizzle) {
    out << "// XOR swizzle: column ^= (row & 3) permutes banks across rows while\n"
           "// staying inside the row (BK % 4 == 0 guaranteed by validate_config;\n"
           "// simplified CuTe Swizzle<3,3,3> idiom, same as todo 27).\n"
           "inline __device__ int a_index(int row, int col) {\n"
           "  return row * kAPitch + (col ^ (row & 3));\n"
           "}\n"
           "inline __device__ int b_index(int row, int col) {\n"
           "  return row * kBPitch + (col ^ (row & 3));\n"
           "}\n";
  } else {
    out << "// Padding layout: row stride +1 (bank-conflict-avoidance pad).\n"
           "inline __device__ int a_index(int row, int col) {\n"
           "  return row * kAPitch + col;\n"
           "}\n"
           "inline __device__ int b_index(int row, int col) {\n"
           "  return row * kBPitch + col;\n"
           "}\n";
  }

  out << "\nextern \"C\" __global__ void " << symbol
      << "(const ks_bf16 *a, const ks_bf16 *b, ks_bf16 *out, int m, int n,\n"
         "                             int k) {\n"
         "  __shared__ float a_s[kStages][kBM * kAPitch];\n"
         "  __shared__ float b_s[kStages][kBK * kBPitch];\n"
         "\n"
         "  const int tx = threadIdx.x;\n"
         "  const int ty = threadIdx.y;\n"
         "  const int tid = ty * kThreadsX + tx;\n"
         "  const int nthreads = kThreadsX * kThreadsY;\n"
         "  const int row0 = blockIdx.y * kBM;\n"
         "  const int col0 = blockIdx.x * kBN;\n"
         "\n"
         "  float acc[kRpt][kCpt];\n"
         "  #pragma unroll\n"
         "  for (int rr = 0; rr < kRpt; ++rr)\n"
         "    #pragma unroll\n"
         "    for (int cc = 0; cc < kCpt; ++cc) acc[rr][cc] = 0.0f;\n"
         "\n"
         "  const int nk_tiles = (k + kBK - 1) / kBK;\n"
         "  for (int it = 0; it < nk_tiles; ++it) {\n"
         "    const int cur = it % kStages;\n"
         "    const int nxt = (it + 1) % kStages;\n"
         "    const int k0 = it * kBK;\n"
         "\n"
         "    // Cooperative load of the BM x BK A tile into buffer `cur`: a\n"
         "    // flattened walk (tid -> consecutive K columns) keeps the global\n"
         "    // loads coalesced; OOB rows/cols (non-multiple M/K) zero-fill.\n"
         "    for (int e = tid; e < kBM * kBK; e += nthreads) {\n"
         "      const int r = e / kBK;\n"
         "      const int c = e % kBK;\n"
         "      const int gr = row0 + r;\n"
         "      const int gk = k0 + c;\n"
         "      const float v = (gr < m && gk < k) ? ks_bf16_to_float(a[gr * k + gk]) : 0.0f;\n"
         "      a_s[cur][a_index(r, c)] = v;\n"
         "    }\n"
         "    // Cooperative load of the BK x BN B tile into buffer `cur`.\n"
         "    for (int e = tid; e < kBK * kBN; e += nthreads) {\n"
         "      const int r = e / kBN;\n"
         "      const int c = e % kBN;\n"
         "      const int gk = k0 + r;\n"
         "      const int gc = col0 + c;\n"
         "      const float v = (gk < k && gc < n) ? ks_bf16_to_float(b[gk * n + gc]) : 0.0f;\n"
         "      b_s[cur][b_index(r, c)] = v;\n"
         "    }\n"
         "    __syncthreads();\n"
         "\n"
         "    // Compute on buffer `cur`: acc += A[rows] @ B[kk][cols]. Each\n"
         "    // thread owns rpt consecutive rows (ty) x cpt consecutive cols\n"
         "    // (tx); kVec consecutive B columns are grouped into bv[kVec] per\n"
         "    // kk (the vec_width search dimension).\n";

  if (cfg.unroll > 1) {
    out << "    #pragma unroll " << cfg.unroll << "\n";
  }
  out << "    for (int kk = 0; kk < kBK; ++kk) {\n"
         "      #pragma unroll\n"
         "      for (int rr = 0; rr < kRpt; ++rr) {\n"
         "        const float av = a_s[cur][a_index(ty * kRpt + rr, kk)];\n";

  if (cfg.vec_width > 1) {
    out << "        for (int cc = 0; cc < kCpt; cc += kVec) {\n"
           "          float bv[kVec];\n"
           "          #pragma unroll\n"
           "          for (int u = 0; u < kVec; ++u)\n"
           "            bv[u] = b_s[cur][b_index(kk, tx * kCpt + cc + u)];\n"
           "          #pragma unroll\n"
           "          for (int u = 0; u < kVec; ++u)\n"
           "            acc[rr][cc + u] += av * bv[u];\n"
           "        }\n";
  } else {
    out << "        for (int cc = 0; cc < kCpt; ++cc)\n"
           "          acc[rr][cc] += av * b_s[cur][b_index(kk, tx * kCpt + cc)];\n";
  }

  out << "      }\n"
         "    }\n"
         "    __syncthreads();\n"
         "\n"
         "    // Prefetch the next K chunk into buffer `nxt` (software pipeline;\n"
         "    // synchronous loads for lite — the buffer alternation is what the\n"
         "    // stages dimension searches, mirroring todo 27).\n"
         "    if (it + 1 < nk_tiles) {\n"
         "      const int k1 = (it + 1) * kBK;\n"
         "      for (int e = tid; e < kBM * kBK; e += nthreads) {\n"
         "        const int r = e / kBK;\n"
         "        const int c = e % kBK;\n"
         "        const int gr = row0 + r;\n"
         "        const int gk = k1 + c;\n"
         "        const float v = (gr < m && gk < k) ? ks_bf16_to_float(a[gr * k + gk]) : 0.0f;\n"
         "        a_s[nxt][a_index(r, c)] = v;\n"
         "      }\n"
         "      for (int e = tid; e < kBK * kBN; e += nthreads) {\n"
         "        const int r = e / kBN;\n"
         "        const int c = e % kBN;\n"
         "        const int gk = k1 + r;\n"
         "        const int gc = col0 + c;\n"
         "        const float v = (gk < k && gc < n) ? ks_bf16_to_float(b[gk * n + gc]) : 0.0f;\n"
         "        b_s[nxt][b_index(r, c)] = v;\n"
         "      }\n"
         "    }\n"
         "    __syncthreads();\n"
         "  }\n"
         "\n"
         "  // Epilogue: acc -> C, rounded to bf16 exactly once (todo-23 "
         "contract);\n"
         "  // OOB rows/cols (non-multiple M/N) are skipped, not written.\n"
         "  for (int rr = 0; rr < kRpt; ++rr) {\n"
         "    const int gr = row0 + ty * kRpt + rr;\n"
         "    if (gr >= m) continue;\n"
         "    for (int cc = 0; cc < kCpt; ++cc) {\n"
         "      const int gc = col0 + tx * kCpt + cc;\n"
         "      if (gc < n) out[gr * n + gc] = ks_float_to_bf16(acc[rr][cc]);\n"
         "    }\n"
         "  }\n"
         "}\n";

  return out.str();
}

bool generate_to_file(const MatmulConfig &cfg, const MatmulProblem &prob,
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
