//===- roofline.h - Roofline engine (AI models + bound classification) -----===//
//
// kernscope roofline engine (wave-2 todo 12). Implements the roofline model of
// Williams, Waterman & Patterson, *Roofline: An Insightful Visual Performance
// Model for Floating-Point Programs and Multicore Architectures*, CACM 52(4),
// Apr 2009, DOI 10.1145/1498765.1498785 (cited in
// docs/research/autotuner-research-brief.md §7):
//     attainable performance = min(peak FLOPS, arithmetic intensity × peak BW)
//     ridge point = peak FLOPS / peak BW   (where the two roofs meet)
//     compute-bound iff arithmetic intensity reaches the ridge point.
//
// Per-family arithmetic-intensity models (brief §7 L132-135 "Arithmetic
// intensity for your kernels"):
//   GEMM (M×K · K×N)      FLOPs = 2MNK; DRAM bytes = s·(MK + KN + MN) with
//                         s = bytes/elem (tile-traffic assumption: each matrix
//                         read once, C written once — achievable with good
//                         tiling); square fp16: AI = N/3 (e.g. N=2048: 682.67).
//   fused bias+GELU       elementwise: 14 FLOPs/elem (1 bias add + 13-op
//                         tanh-approximation GELU) over 3s bytes/elem
//                         (read x, read bias, write y) -> memory-bound.
//   softmax               ~5 FLOPs/elem (max, sub, exp, sum, div; brief L134)
//                         over 2s bytes/elem (read once + write once) -> AI
//                         ≈ 5/(2s) ≈ 0.6 F/B (fp32) — deeply memory-bound.
//   layernorm             11 FLOPs/elem over 2s bytes/elem (read + write).
//   reduction             1 FLOP/elem over s bytes/elem (read once; output is
//                         O(1) so its write is amortized out) — memory-bound.
//   FlashAttention        FLOPs = 4N²d per head (two N×N matmuls, S = QKᵀ and
//                         O = PV, each 2N²d; brief §7 L135, Dao et al. NeurIPS
//                         2022 Theorem 2). Default DRAM traffic = standard-
//                         attention s·(4Nd + 4N²); a tiled FlashAttention's
//                         true Θ(N²d²/M) traffic (M = SRAM size) must be passed
//                         as explicit bytes_moved for an accurate position —
//                         the brief: "the roofline point should be computed
//                         from actual HBM traffic of your tiled implementation".
//
// Machine peaks (brief §7 L127-130 "Machine peaks (verified from vendor
// docs)"): A100/sm_80 (FP32 19.5 TF, FP16/BF16 TC 312 TF dense, HBM2e 2039
// GB/s 80GB SXM), RTX 4090/sm_89 (FP32 82.6 TF, 1008 GB/s GDDR6X), MI250X/
// gfx90a (FP16/BF16 matrix 383 TF, 3.28 TB/s HBM2e — 3.2-3.28 range caveat in
// brief §9), MI300X/gfx942 (BF16 matrix 1307.4 TF dense, FP32 vector 163.4 TF,
// 5.3 TB/s HBM3). gfx1101 (RX 7800 XT) is filled with MEASURED values +
// provenance by todo 22 (libs/bench / kernscope-bench on this machine,
// 2026-08-14): fp32 16.32 TF, DRAM 591 GB/s, LDS 9.7 TB/s — querying gfx1101
// returns projected == false with the measured peaks, never a fabricated
// datasheet number (plan todo 12 MUST NOT; Metis M-6). Peaks for archs outside
// this table (e.g. sm_90) are an explicit error, not silent zeros.
//
// Bound classification (documented thresholds below; unit-tested for all three
// classes):
//   1. latency-bound: DRAM time-to-move = bytes_moved / bandwidth <
//      kDramLatencyNs — the kernel is too small to saturate the memory roof
//      (Little's law: peak bandwidth B needs B×latency bytes in flight).
//   2. compute-bound: AI >= ridge_point (polykernel oracle convention,
//      /home/mei/projects/polykernel/lib/Analysis/Roofline.cpp L51-53).
//   3. memory-bound: otherwise (attainable = AI × bandwidth < compute roof).
//
// Cross-validation oracle: polykernel Roofline.cpp + tests/roofline_test.cpp
// (AI = 2MNK / s(MK+KN+MN), ridge = peak_flops/peak_bw, compute iff AI >=
// ridge — values read for validation only, no code copied).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ROOFLINE_H
#define KERNSCOPE_ROOFLINE_H

#include <cstdint>
#include <string_view>

namespace kernscope::roofline {

// The six kernel families of the C2 autotuner (plan Scope; six families in
// tools/tune: matmul, fused bias+GELU, softmax, layernorm, reduction,
// FlashAttention-lite).
enum class KernelFamily : std::uint8_t {
  kGemm,
  kFusedBiasGelu,
  kSoftmax,
  kLayerNorm,
  kReduction,
  kFlashAttention,
};

// Stable machine-readable family names ("gemm", "bias_gelu", "softmax",
// "layernorm", "reduction", "flash_attention").
const char *kernel_family_name(KernelFamily family) noexcept;

// One problem shape. Which fields are meaningful depends on the family:
//   kGemm             m, n, k           (M×K · K×N)
//   kFusedBiasGelu / kSoftmax / kLayerNorm / kReduction
//                     n                 (number of elements)
//   kFlashAttention   n, d              (sequence length, head dim)
// dtype_bytes is bytes per element (fp32 = 4, fp16/bf16 = 2) and drives the
// tile-traffic byte counts below.
struct Shape {
  std::int64_t m = 0;
  std::int64_t n = 0;
  std::int64_t k = 0;
  std::int64_t d = 0;
  int dtype_bytes = 4;
};

// --- Per-family FLOP/element constants (documented; brief §7 L132-135) ------
// bias+GELU: 1 bias add + tanh-approx GELU 0.5x(1+tanh(√(2/π)(x+0.044715x³))):
//   x³ = 2 mul, 0.044715·x³ = 1 mul, x+· = 1 add, √(2/π)· = 1 mul, tanh(u) =
//   2u(1 mul)+e^(2u)(1)+1(1 add)+div(1) ≈ 5, 1+tanh = 1 add, 0.5x = 1 mul,
//   product = 1 mul -> 13, + bias add = 14.
inline constexpr std::int64_t kGeluFlopsPerElement = 14;
// softmax: max, sub, exp, sum, div (brief §7 L134 counts exactly these five).
inline constexpr std::int64_t kSoftmaxFlopsPerElement = 5;
// layernorm: mean (sum+div), variance (sub+square+sum+div+eps), normalize
// (sub, fused mul), scale+shift (mul, add) = 1+1+1+1+1+1+1+1+1+1+1.
inline constexpr std::int64_t kLayerNormFlopsPerElement = 11;
// reduction: one add per element (log-tree levels amortize to O(1)/elem).
inline constexpr std::int64_t kReductionFlopsPerElement = 1;

// --- Bound-classification threshold (documented; see header comment) --------
// DRAM round-trip latency used by Little's-law latency bound. Cited:
// docs/research/gpu-static-analyzer-research.md §8.1 (A100 global/DRAM =
// 566 cycles ≈ 0.40 µs at ~1.4 GHz; H800 656 cycles).
inline constexpr double kDramLatencyNs = 400.0;

// FLOPs for `family` over `shape` (formulas above; brief §7 L133-135).
// Returns 0 for empty shapes. The GEMM/attention results fit in int64 for any
// practical shape (2·2048³ = 1.7e10, 4·(2^16)²·256 ≈ 1.7e14).
std::int64_t compute_flops(KernelFamily family, const Shape &shape) noexcept;

// Minimum DRAM bytes moved for `family` over `shape` under the tile-traffic
// assumptions documented above (brief §7 L133-135). Returns 0 for empty
// shapes. FlashAttention returns the standard-attention traffic s·(4Nd + 4N²);
// pass an explicit bytes_moved for a tiled implementation's real Θ(N²d²/M).
std::int64_t compute_bytes_moved(KernelFamily family, const Shape &shape) noexcept;

// arithmetic_intensity = compute_flops / compute_bytes_moved (FLOP per byte of
// DRAM traffic — the roofline paper's "operational intensity"). 0 when no
// bytes are moved.
double arithmetic_intensity(KernelFamily family, const Shape &shape) noexcept;

// Error codes returned by the query functions. On any error the output
// struct is left UNTOUCHED — a failure NEVER reports zeros as if they were
// peaks or a position.
enum class ErrorCode : std::uint8_t {
  kNone = 0,
  kUnknownArch,     // unknown arch name (message contains "unknown arch")
  kProjectedPeaks,  // arch peaks are pending-measurement placeholders (gfx1101
                    // was filled with measured peaks by todo 22)
  kNoPeaksTabulated,// known GPU with no row in the todo-12 peaks table
  kNoPeakForDtype,  // no tabulated compute peak for the requested PeakKind
  kInvalidInput,    // non-positive dims / bytes_moved / flops
};

// ---------------------------------------------------------------------------
// Machine peaks (brief §7 L127-130 + measured gfx1101 row from todo 22).
// ---------------------------------------------------------------------------

// One row of the machine-peaks table. All peak values are VERIFIED from the
// brief's vendor-doc citations (see `provenance`); nothing is guessed.
// projected == true means the row is a pending-measurement PLACEHOLDER
// (peaks are 0 and MUST NOT be treated as real numbers). gfx1101's row was
// filled with MEASURED values + provenance by todo 22 and is projected ==
// false (see kPeaks in roofline.cpp).
struct MachinePeak {
  std::string_view arch;               // arch DB key (sm_80, gfx90a, ...)
  std::string_view gpu;                // commercial GPU name
  std::string_view matrix_dtype;       // matrix-peak dtype label ("fp16/bf16"); "" = none
  double matrix_peak_flops = 0.0;      // FP16/BF16 tensor/matrix peak, FLOP/s; 0 = none tabulated
  double fp32_peak_flops = 0.0;        // FP32 vector peak, FLOP/s; 0 = none tabulated
  double bandwidth_bytes_per_s = 0.0;  // peak DRAM bandwidth, bytes/s
  bool projected = false;              // true = pending-measurement placeholder
  std::string_view provenance;         // citation (brief §7 L...) or measurement record
};

// Query the machine-peaks table. Returns ErrorCode::kNone and fills `out` on
// success — gfx1101 now returns its MEASURED peaks (projected == false).
// Errors:
//   kUnknownArch      arch name is not in the libs/arch DB (message contains
//                     "unknown arch");
//   kNoPeaksTabulated arch is a known GPU but has no peaks row in the
//                     todo-12 table (e.g. sm_90);
//   kProjectedPeaks   (never returned here; see roofline_position).
ErrorCode machine_peak(std::string_view arch, MachinePeak *out,
                       std::string *error_message = nullptr);

// ---------------------------------------------------------------------------
// Roofline position + bound classification.
// ---------------------------------------------------------------------------

// Which compute roof to use for the ridge point: FP32 vector peak for
// elementwise kernels, FP16/BF16 tensor/matrix peak for GEMM/attention.
enum class PeakKind : std::uint8_t { kFp32, kMatrix };

// The three bound classes. "latency" is a third class beyond the classic two:
// kernels too small to saturate the memory roof (time-to-move < DRAM latency).
enum class BoundClass : std::uint8_t { kCompute, kMemory, kLatency };

// Stable bound names: "compute" | "memory" | "latency".
const char *bound_class_name(BoundClass bound) noexcept;

// ---------------------------------------------------------------------------
// Roofline position.
//   arithmetic_intensity     = flops / bytes_moved (FLOP per byte)
//   ridge_point              = compute peak / bandwidth (FLOP per byte)
//   compute_bound_ceiling    = compute peak (FLOP/s)
//   memory_bound_ceiling     = AI × bandwidth (FLOP/s)
//   attained_flops           = min(compute, memory) ceiling (FLOP/s) — the
//                              paper's attainable-performance definition; a
//                              latency-bound kernel sits BELOW both roofs, so
//                              this is an upper bound, not the measured time.
struct RooflinePosition {
  double arithmetic_intensity = 0.0;
  double ridge_point = 0.0;
  double compute_bound_ceiling = 0.0;
  double memory_bound_ceiling = 0.0;
  double attained_flops = 0.0;
  BoundClass bound_class = BoundClass::kMemory;
};

// Compute the roofline position for (family, shape, bytes_moved, flops) on
// `arch`. `bytes_moved`/`flops` are the caller's values (measured or derived
// via compute_bytes_moved/compute_flops) so a tiled FlashAttention can pass
// its real HBM traffic. Returns ErrorCode::kNone and fills `out` on success.
// Errors (output struct untouched):
//   kInvalidInput      bytes_moved <= 0 || flops <= 0
//   kUnknownArch       arch not in the libs/arch DB ("unknown arch" in message)
//   kNoPeaksTabulated  known GPU with no row in the todo-12 peaks table
//   kProjectedPeaks    arch peaks are pending-measurement placeholders —
//                      position cannot be computed without measured peaks
//                      (guarded, never fabricated; gfx1101 is now measured)
//   kNoPeakForDtype    the requested PeakKind has no tabulated peak for `arch`
//                      (e.g. PeakKind::kFp32 on gfx90a, which the brief
//                      tabulates only for matrix ops)
ErrorCode roofline_position(KernelFamily family, const Shape &shape,
                            std::int64_t bytes_moved, std::int64_t flops,
                            std::string_view arch, RooflinePosition *out,
                            std::string *error_message = nullptr,
                            PeakKind peak_kind = PeakKind::kFp32);

} // namespace kernscope::roofline

#endif // KERNSCOPE_ROOFLINE_H
