//===- matmul_gen.h - tiled-matmul template generator ---------------------===//
//
// kernscope tools/tune todo 25: the matmul template generator. Consumed by
// the tune track (todo 28 tuning driver) and the gate/live tests
// (tests/matmul_gen_test.cpp). Emits a single-source portable .hip kernel
// (kernels/templates convention: includes only kernscope/kernel_traits.h,
// compiles under nvcc -x cu for sm_80/sm_90 AND hipcc for gfx1101).
//
// GENERATION-ONLY (no hand-written per-config sources): a config is baked
// into the emitted source as constexpr constants + a `#pragma unroll` /
// smem-layout / vector-group structure selected by the generator; the same
// C++ template in matmul_gen.cpp produces every search-space variant.
//
// ALGORITHM: classic shared-memory tiled GEMM, C[M,N] = A[M,K] @ B[K,N],
// with the skeleton shape of kernels/templates/reference/matmul_skeleton.hip
// generalized to arbitrary BM/BN/BK tiles:
//   * A tile BM×BK, B tile BK×BN staged in shared memory (fp32);
//   * stages-deep software pipeline: multi-buffered tiles; the K dimension
//     walks in BK-wide chunks with zero-fill for the OOB tail (non-multiple
//     M/N/K are bounds-checked, never padded to a divisible shape);
//   * per-thread fragment acc[rpt][cpt] in registers, fp32 accumulate,
//     bf16 in / bf16 out (todo-23 rounding contract: round to bf16 exactly
//     once in the epilogue);
//   * `extern "C" __global__ void kernscope_tune_matmul(...)` entry symbol
//     (the kernscope_templates_hip_runner resolves kernels by this name).
//
// SEARCH SPACE (what the tuner, todo 28, searches; validate_config rejects
// every infeasible candidate PRE-COMPILE with a reason string):
//   * BM, BN (output tile)   : 64 | 128
//   * BK (inner tile)        : 8 | 16 | 32 | 64
//   * warps (per block)      : 1 | 2 | 4 | 8   -- split into (warps_m,
//     warps_n) = (threads_y, threads_x/32) derived from BM/BN divided by the
//     warp tile (see warp_decomposition): threads_x = 32·warps_n cover BN,
//     threads_y = warps_m cover BM.
//   * stages (pipeline)      : 1 | 2 | 3 | 4  -- multi-buffered smem tiles.
//     The buffers alternate per K-chunk (software pipeline); loads are
//     synchronous in the lite template (no cp.async intrinsics — the
//     *structure* the stages dimension searches is the multi-buffer staging,
//     mirroring todo 27's lite scope).
//   * vec_width              : 1 | 2 | 4  -- consecutive columns per thread
//     (the fragment cc-loop loads kVec consecutive b_s columns into a
//     register group bv[kVec] per kk, which the compiler vectorizes).
//   * smem_layout            : padding (BK+1 row stride) | XOR swizzle
//     (row stride BK, column ^= (row & 3) — the simplified CuTe
//     Swizzle<3,3,3> idiom already used by todo 27).
//   * unroll (K-loop)        : 1 | 2 | 4  -- `#pragma unroll kUnroll` on
//     the in-tile BK accumulation loop.
//
// VALIDITY RULES (libs/arch, all rejected PRE-COMPILE with a reason string):
//   * search-space membership (dimension choices above);
//   * warp/block divisibility: the (warps_m, warps_n) factorization must
//     tile the output tile (BM % warps_m == 0 and BN % (32·warps_n) == 0);
//   * vector width: cpt % vec_width == 0, where cpt = BN/(32·warps_n) is the
//     per-thread column fragment (the kVec group must divide the fragment);
//   * threads/block: 32·warps <= 1024 AND <= arch->max_threads_per_sm;
//   * register budget: acc fragment regs/thread (BM·BN/(32·warps)) plus a
//     32-register overhead estimate must not exceed arch->max_regs_per_thread
//     (a 128×128 tile at warps=1 would need 512+ acc regs -> rejected);
//   * shared-memory budget via libs/arch: the tiles below must fit
//     arch->smem_per_block_max_bytes:
//
//         4 · stages · ( BM·(BK+pad) + BK·(BN+pad) )   bytes, pad ∈ {0,1}
//
//     (fp32 smem, per-stage A + B tiles, padded/swizzled row stride).
//     A config whose tiles exceed the budget is REJECTED before any source
//     is generated — the failure mode is a reason string, never a broken
//     compile.
//
// Kernel parameter contract (kernels/templates/README.md, matmul family):
// (const ks_bf16* a, const ks_bf16* b, ks_bf16* out, int m, int n, int k)
// — two input device pointers, one output pointer, three int32 scalars,
// passed by value per the driver-API convention (tools/hip_template_runner;
// tests/run_hip_gate.py --family matmul --shape M,N,K).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_MATMUL_GEN_H
#define KERNSCOPE_TUNE_MATMUL_GEN_H

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace kernscope::tune {

// --- smem layout for the A/B tiles (search-space dimension) ----------------
enum class SmemLayout : std::uint8_t {
  kPadding,    // row stride BK+1 (bank-conflict-avoidance pad idiom)
  kXorSwizzle  // column ^= (row & 3), row stride BK (CuTe Swizzle<3,3,3>-style)
};

// Stable layout names ("padding" | "xor_swizzle"), for diagnostics.
const char *smem_layout_name(SmemLayout layout) noexcept;

// --- one candidate configuration (the tuner's search point) ----------------
struct MatmulConfig {
  int tile_m = 64;           // BM: output rows per block
  int tile_n = 64;           // BN: output cols per block
  int tile_k = 16;           // BK: inner (K) tile size
  int warps = 1;             // threads/block = 32·warps (1 | 2 | 4 | 8)
  int stages = 1;            // software-pipelined smem tile buffers (1-4)
  int vec_width = 1;         // consecutive columns per thread (1 | 2 | 4)
  SmemLayout smem_layout = SmemLayout::kPadding;
  int unroll = 1;            // K-loop unroll factor (1 | 2 | 4)

  bool operator==(const MatmulConfig &) const = default;
};

// --- the problem the generated kernel is validated against -----------------
// M/N/K are RUNTIME kernel arguments in the emitted source (the kernel is
// bounds-checked, so no divisibility constraint on the problem); the problem
// only selects the arch budget + dtype size for the pre-compile checks.
struct MatmulProblem {
  std::int64_t m = 0;   // rows of A / C
  std::int64_t n = 0;   // cols of B / C
  std::int64_t k = 0;   // inner dimension
  std::string arch;     // libs/arch key ("gfx1101", "sm_80", ...)
  int dtype_bytes = 2;  // bytes/element (bf16)
};

// --- pre-compile validity verdict ------------------------------------------
struct ConfigVerdict {
  bool valid = true;
  std::string reason;  // empty when valid; the reject reason otherwise
};

// Validate one candidate pre-compile (search-space membership, warp/block
// divisibility, vector-width tiling, register budget, smem budget via
// libs/arch). NEVER compiles anything: an invalid config is rejected here
// with a reason string before any source is generated.
ConfigVerdict validate_config(const MatmulConfig &cfg,
                              const MatmulProblem &prob) noexcept;

// Exact shared-memory footprint (bytes) of a config, matching the
// allocations the generated kernel declares (documented formula above).
std::int64_t smem_bytes(const MatmulConfig &cfg) noexcept;

// The 2D warp split of a config: threads_x = 32·warps_n, threads_y =
// warps_m, warps_m · warps_n == warps. Returns {0,0} when no factor pair
// tiles the output tile (validate_config rejects those pre-compile).
struct WarpDecomposition {
  int warps_m = 0;
  int warps_n = 0;
};
WarpDecomposition warp_decomposition(const MatmulConfig &cfg) noexcept;

// Launch geometry the generated kernel expects: grid covers ceil(N/BN)
// column-blocks in x and ceil(M/BM) row-blocks in y; block is
// (32·warps_n, warps_m). Non-multiple M/N are handled by bounds checks.
struct LaunchGeometry {
  int grid_x = 1;
  int grid_y = 1;
  int block_x = 1;
  int block_y = 1;
};
LaunchGeometry launch_geometry(const MatmulConfig &cfg,
                               const MatmulProblem &prob) noexcept;

// Stable kernel symbol: the fixed entry name "kernscope_tune_matmul" the
// runner resolves (task contract; every emitted variant exports this symbol,
// so live-gate + compile gates pass it to --kernel / --symbol).
std::string kernel_symbol(const MatmulConfig &cfg);

// --- search space ----------------------------------------------------------
inline constexpr std::array<int, 2> kTileMChoices{64, 128};
inline constexpr std::array<int, 2> kTileNChoices{64, 128};
inline constexpr std::array<int, 4> kTileKChoices{8, 16, 32, 64};
inline constexpr std::array<int, 4> kWarpChoices{1, 2, 4, 8};
inline constexpr std::array<int, 4> kStageChoices{1, 2, 3, 4};
inline constexpr std::array<int, 3> kVecWidthChoices{1, 2, 4};
inline constexpr std::array<int, 3> kUnrollChoices{1, 2, 4};
inline constexpr std::array<SmemLayout, 2> kSmemLayoutChoices{
    SmemLayout::kPadding, SmemLayout::kXorSwizzle};

// --- source generation -----------------------------------------------------
// Generate the single-source .hip kernel for a config+problem. Returns empty
// with *error set on invalid configs (call validate_config first; this is a
// safety net, not the primary reject path).
std::string generate_source(const MatmulConfig &cfg,
                            const MatmulProblem &prob,
                            std::string *error);

// Convenience: generate + write to `path`. Returns false with *error set on
// failure (invalid config or IO error).
bool generate_to_file(const MatmulConfig &cfg, const MatmulProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_MATMUL_GEN_H
