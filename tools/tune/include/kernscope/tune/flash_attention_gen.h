//===- flash_attention_gen.h - FlashAttention-lite template generator -----===//
//
// kernscope tools/tune todo 27: the FlashAttention-lite template generator.
// Consumed by the tune track (todo 28 tuning driver) and the live/gate tests
// (tests/flash_gen_test.cpp). Emits a single-source portable .hip kernel
// (kernels/templates convention: includes only kernscope/kernel_traits.h,
// compiles under nvcc -x cu for sm_80/sm_90 AND hipcc for gfx1101).
//
// LITE SCOPE (documented, enforced by review): FORWARD PASS ONLY. No dropout,
// no backward pass, no paged KV, no multi-head batching inside the template
// (heads are flattened into the query/KV sequence dimension by the caller),
// no tensor-core instructions (WMMA/MFMA are out of scope for the lite
// template; the tile/matmul structure below is deliberately scalar so it is
// correct on both vendors' vector units). This generator does NOT claim
// FlashAttention-2 parity or the IO-aware kernel-fusion asymptotics of the
// paper beyond the shared-memory tiling + online softmax structure.
//
// ALGORITHM (the FlashAttention online softmax, Dao et al., "FlashAttention:
// Fast and Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022;
// brief docs/research/autotuner-research-brief.md §7 L135): for each Br-row
// query tile the block owns, the K/V sequence is walked in Bc-row tiles with
// running max m and running sum l:
//
//     S     = (Q Kᵀ) · (1/√d)            (fp32, scaled like the golden ref)
//     m_new = max(m_old, rowmax S)
//     α     = exp(m_old − m_new)
//     l     = α·l + rowsum(exp(S − m_new))
//     O     = α·O + softmax(S − m_new)·V
//
// so O accumulates correctly across K/V tiles in ONE pass over Q (no two-pass
// rescale, no materialized N×N attention matrix). The O tiles are bf16-in /
// fp32-accumulate / bf16-out per the todo-23 rounding contract; the kernel
// applies the same 1/√d scaling as the golden attention reference.
//
// SEARCH SPACE (what the tuner, todo 28, searches; the generator validates
// every candidate pre-compile):
//   * Br (tile_m)      : 32 | 64      query rows per block
//   * Bc (tile_n)      : 32 | 64      KV rows per tile
//   * stages           : 1 | 2        software-pipelined K/V buffers
//   * vec_width        : 1 | 2 | 4    columns per thread (threads_x = d/vec)
//   * smem_layout      : padding | xor_swizzle  Q/K/V tile row layout
//   * causal           : bool         causal mask baked into the variant
//
// VALIDITY RULES (libs/arch, all rejected PRE-COMPILE with a reason string):
//   * search-space membership (above);
//   * block divisibility: nq % Br == 0 and nk % Bc == 0 (the lite kernel is
//     unguarded by design; a non-divisible shape is rejected, not padded);
//   * vector width: d % vec_width == 0 and Bc % (d/vec_width) == 0 (the
//     thread->column mapping must tile Bc exactly);
//   * XOR swizzle additionally requires d % 4 == 0 (the (col ^ (row & 3))
//     permutation must stay inside a row);
//   * shared-memory budget: the tiles below must fit the arch's
//     per-block shared memory (ArchInfo::smem_per_block_max_bytes via
//     libs/arch). Budget formula (bytes):
//
//         (Br·dPitch + 2·stages·Bc·dPitch + Br·(Bc+1) + 2·Br) · 4
//
//     with dPitch = d+1 (padding) or d (XOR swizzle): Q tile + K/V tiles
//     (stages double-buffered) + the S scores tile (+1 pad) + the m/l row
//     arrays. The m/l arrays are shared-memory row scalars (Br each, fp32);
//     the O accumulator lives in registers per thread.
//
// Kernel parameter contract (kernels/templates/README.md): (const ks_bf16*
// q, const ks_bf16* k, const ks_bf16* v, ks_bf16* out, int nq, int nk, int d)
// — three input device pointers, one output pointer, three int32 scalars, all
// passed by value per the driver-API convention (tools/hip_template_runner).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_FLASH_ATTENTION_GEN_H
#define KERNSCOPE_TUNE_FLASH_ATTENTION_GEN_H

#include "kernscope/roofline.h"

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace kernscope::tune {

// --- smem layout for the Q/K/V tiles (search-space dimension) --------------
enum class SmemLayout : std::uint8_t {
  kPadding,    // row stride d+1 (bank-conflict-avoidance pad idiom)
  kXorSwizzle  // column ^= (row & 3), row stride d (CuTe-style XOR swizzle)
};

// Stable layout names ("padding" | "xor_swizzle"), for kernel symbols and
// diagnostics.
const char *smem_layout_name(SmemLayout layout) noexcept;

// --- one candidate configuration (the tuner's search point) ----------------
struct FlashAttentionConfig {
  int tile_m = 64;           // Br: query rows per block
  int tile_n = 64;           // Bc: KV rows per K/V tile
  int stages = 1;            // software-pipelined K/V buffers (1 | 2)
  int vec_width = 1;         // columns per thread (1 | 2 | 4)
  SmemLayout smem_layout = SmemLayout::kPadding;
  bool causal = false;       // causal mask (see refs.attention(causal=...))

  bool operator==(const FlashAttentionConfig &) const = default;
};

// --- the problem the generated kernel is validated against -----------------
struct FlashAttentionProblem {
  std::int64_t nq = 0;   // number of query rows (sequence length, flattened heads)
  std::int64_t nk = 0;   // number of key/value rows (== nq for self-attention)
  std::int64_t d = 0;    // head dimension
  std::string arch;      // libs/arch key ("gfx1101", "sm_80", ...) for the budget
  int dtype_bytes = 2;   // bytes/element (bf16)
};

// --- pre-compile validity verdict ------------------------------------------
struct ConfigVerdict {
  bool valid = true;
  std::string reason;  // empty when valid; the reject reason otherwise
};

// Validate one candidate pre-compile (search-space membership, block
// divisibility, vector-width tiling, XOR-swizzle constraint, smem budget via
// libs/arch). NEVER compiles anything: an invalid config is rejected here
// with a reason string before any source is generated.
ConfigVerdict validate_config(const FlashAttentionConfig &cfg,
                              const FlashAttentionProblem &prob) noexcept;

// Exact shared-memory footprint (bytes) of a config over a problem, matching
// the allocations the generated kernel declares (documented formula above).
std::int64_t smem_bytes(const FlashAttentionConfig &cfg,
                        const FlashAttentionProblem &prob) noexcept;

// Launch geometry the generated kernel expects: grid covers nq/Br query
// blocks in x; block is (threads_x, threads_y) with threads_x = d/vec and
// threads_y = min(Br, 1024/threads_x) (block <= 1024 threads on both archs).
struct LaunchGeometry {
  int grid_x = 1;
  int grid_y = 1;
  int block_x = 1;
  int block_y = 1;
};
LaunchGeometry launch_geometry(const FlashAttentionConfig &cfg,
                               const FlashAttentionProblem &prob) noexcept;

// Stable kernel symbol for a config, e.g.
// "kernscope_tune_flash_attention_br64_bc64_s1_v1_pad_causal".
std::string kernel_symbol(const FlashAttentionConfig &cfg);

// --- search space ----------------------------------------------------------
inline constexpr std::array<int, 2> kTileChoices{32, 64};
inline constexpr std::array<int, 2> kStageChoices{1, 2};
inline constexpr std::array<int, 3> kVecWidthChoices{1, 2, 4};
inline constexpr std::array<SmemLayout, 2> kSmemLayoutChoices{
    SmemLayout::kPadding, SmemLayout::kXorSwizzle};

// --- FLOPs model (cross-checked against libs/roofline) ---------------------
// FlashAttention FLOPs = 4·N²·d per head (S=QKᵀ and O=PV, each 2·N²·d;
// brief §7 L135, Dao et al. NeurIPS 2022 Theorem 2). Matches
// kernscope::roofline::compute_flops(kFlashAttention, {n, d}) exactly.
inline std::int64_t flash_attention_flops(std::int64_t n, std::int64_t d) noexcept {
  return 4 * n * n * d;
}

// --- source generation -----------------------------------------------------
// Generate the single-source .hip kernel for a config+problem. Returns empty
// with *error set on invalid configs (call validate_config first; this is a
// safety net, not the primary reject path).
std::string generate_source(const FlashAttentionConfig &cfg,
                            const FlashAttentionProblem &prob,
                            std::string *error);

// Convenience: generate + write to `path`. Returns false with *error set on
// failure (invalid config or IO error).
bool generate_to_file(const FlashAttentionConfig &cfg,
                      const FlashAttentionProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_FLASH_ATTENTION_GEN_H
