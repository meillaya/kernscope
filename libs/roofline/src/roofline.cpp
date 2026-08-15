//===- roofline.cpp - Roofline engine (AI models + bound classification) ----===//
//
// Implementation of kernscope::roofline. The AI formulas, machine-peaks table
// and classification thresholds are pinned against
// docs/research/autotuner-research-brief.md §7 (Williams/Waterman/Patterson
// CACM 2009, DOI 10.1145/1498765.1498785); see roofline.h for the derivations
// and citations. The gfx1101 row is filled with MEASURED values + provenance
// by todo 22 (libs/bench / kernscope-bench, run on this machine's RX 7800 XT
// 2026-08-14) — projected == false, real peaks, never a fabricated number.
//
//===----------------------------------------------------------------------===//

#include "kernscope/roofline.h"

#include "kernscope/arch.h"

#include <algorithm>
#include <string>

namespace kernscope::roofline {

const char *kernel_family_name(KernelFamily family) noexcept {
  switch (family) {
  case KernelFamily::kGemm:
    return "gemm";
  case KernelFamily::kFusedBiasGelu:
    return "bias_gelu";
  case KernelFamily::kSoftmax:
    return "softmax";
  case KernelFamily::kLayerNorm:
    return "layernorm";
  case KernelFamily::kReduction:
    return "reduction";
  case KernelFamily::kFlashAttention:
    return "flash_attention";
  }
  return "unknown";
}

// FLOPs per family (brief §7 L133-135). Elementwise families multiply the
// documented per-element constants (roofline.h) by the element count.
std::int64_t compute_flops(KernelFamily family, const Shape &shape) noexcept {
  switch (family) {
  case KernelFamily::kGemm:
    return 2 * shape.m * shape.n * shape.k;
  case KernelFamily::kFusedBiasGelu:
    return shape.n * kGeluFlopsPerElement;
  case KernelFamily::kSoftmax:
    return shape.n * kSoftmaxFlopsPerElement;
  case KernelFamily::kLayerNorm:
    return shape.n * kLayerNormFlopsPerElement;
  case KernelFamily::kReduction:
    return shape.n * kReductionFlopsPerElement;
  case KernelFamily::kFlashAttention:
    return 4 * shape.n * shape.n * shape.d;
  }
  return 0;
}

// Minimum DRAM bytes under the tile-traffic assumptions (brief §7 L133-135):
// GEMM: s·(MK + KN + MN), each matrix read once + C written once (achievable
// with good tiling). Elementwise: s·(read + write) per element; bias+GELU also
// reads the bias vector (3s/elem). Reduction: s·N read once, output O(1).
// FlashAttention: standard-attention s·(4Nd + 4N²) — a tiled implementation's
// real Θ(N²d²/M) traffic must be passed as explicit bytes_moved instead.
std::int64_t compute_bytes_moved(KernelFamily family,
                                 const Shape &shape) noexcept {
  const std::int64_t s = shape.dtype_bytes;
  switch (family) {
  case KernelFamily::kGemm:
    return s * (shape.m * shape.k + shape.k * shape.n + shape.m * shape.n);
  case KernelFamily::kFusedBiasGelu:
    return 3 * s * shape.n;
  case KernelFamily::kSoftmax:
  case KernelFamily::kLayerNorm:
    return 2 * s * shape.n;
  case KernelFamily::kReduction:
    return s * shape.n;
  case KernelFamily::kFlashAttention:
    return s * (4 * shape.n * shape.d + 4 * shape.n * shape.n);
  }
  return 0;
}

double arithmetic_intensity(KernelFamily family, const Shape &shape) noexcept {
  const std::int64_t flops = compute_flops(family, shape);
  const std::int64_t bytes = compute_bytes_moved(family, shape);
  if (bytes <= 0)
    return 0.0;
  return static_cast<double>(flops) / static_cast<double>(bytes);
}

// ---------------------------------------------------------------------------
// Machine peaks (brief §7 L127-130 "Machine peaks (verified from vendor docs)")
// ---------------------------------------------------------------------------

namespace {

// One entry per table row. `provenance` names the brief section + line and the
// vendor-doc citation it comes from; nothing here is guessed. The gfx1101 row
// carries MEASURED values + provenance from todo 22 (libs/bench / the
// kernscope-bench CLI run on THIS machine's RX 7800 XT 2026-08-14; details in
// reports/peaks/gfx1101-bench.jsonl) — projected == false, never fabricated.
constexpr MachinePeak kPeaks[] = {
    {"sm_80", "A100 (SXM 80GB)", "fp16/bf16",
     /*matrix=*/312.0e12, /*fp32=*/19.5e12, /*bw=*/2039.0e9, /*projected=*/false,
     "brief §7 L127: FP16/BF16 TC 312 TF dense; FP32 19.5 TF; HBM2e 2039 GB/s "
     "(80GB SXM) [Ampere whitepaper, A100 datasheet]"},
    {"sm_89", "RTX 4090 (Ada AD102)", "fp16/bf16",
     /*matrix=*/165.2e12, /*fp32=*/82.6e12, /*bw=*/1008.0e9, /*projected=*/false,
     "brief §7 L128: FP32 82.6 TF; FP16 TC ≈ 2× FP32 dense = 165.2 TF (brief "
     "note, not a datasheet peak); GDDR6X 1008 GB/s [Ada whitepaper]"},
    {"gfx90a", "MI250X (CDNA2)", "fp16/bf16",
     /*matrix=*/383.0e12, /*fp32=*/0.0, /*bw=*/3.28e12, /*projected=*/false,
     "brief §7 L129: FP16/BF16 matrix 383 TF @1.7 GHz; HBM2e ≈3.2-3.28 TB/s "
     "per OAM (3.28 used; 3.2-vs-3.28 caveat in brief §9) [AMD ebook]"},
    {"gfx942", "MI300X (CDNA3)", "fp16/bf16",
     /*matrix=*/1307.4e12, /*fp32=*/163.4e12, /*bw=*/5.3e12, /*projected=*/false,
     "brief §7 L130: FP16/BF16 matrix 1307.4 TF dense; FP32 vector 163.4 TF; "
     "HBM3 5.3 TB/s [MI300X datasheet, MI300 arch docs]"},
    {"gfx1101", "RX 7800 XT (RDNA3)", "",
     /*matrix=*/0.0, /*fp32=*/16.32e12, /*bw=*/591.0e9, /*projected=*/false,
     "MEASURED 2026-08-14 on this machine by kernscope-bench (libs/bench, todo "
     "22): fp32 16.32 TF (register-resident FFMA loop); DRAM 591 GB/s "
     "(read-only 512 MiB rep; D2D copy 476 GB/s); LDS 9.7 TB/s. Clocks "
     "as-measured via rocm-smi (sclk/mclk; never locked, no root). Provenance: "
     "reports/peaks/gfx1101-bench.jsonl (device name, gcnArchName, clocks, "
     "tool version, measured=true)"},
};

} // namespace

ErrorCode machine_peak(std::string_view arch, MachinePeak *out,
                       std::string *error_message) {
  for (const MachinePeak &peak : kPeaks) {
    if (peak.arch == arch) {
      if (out != nullptr)
        *out = peak;
      return ErrorCode::kNone;
    }
  }
  // Distinguish a genuinely-unknown arch name (message MUST contain "unknown
  // arch", asserted by the negative-path QA) from a known GPU that simply has
  // no row in the todo-12 table (e.g. sm_90, which libs/arch knows).
  if (!kernscope::has_arch(arch)) {
    if (error_message != nullptr)
      *error_message =
          "kernscope::roofline: unknown arch '" + std::string(arch) + "'";
    return ErrorCode::kUnknownArch;
  }
  if (error_message != nullptr)
    *error_message =
        "kernscope::roofline: no machine peaks tabulated for arch '" +
        std::string(arch) +
        "' (todo-12 table covers sm_80, sm_89, gfx90a, gfx942, gfx1101)";
  return ErrorCode::kNoPeaksTabulated;
}

const char *bound_class_name(BoundClass bound) noexcept {
  switch (bound) {
  case BoundClass::kCompute:
    return "compute";
  case BoundClass::kMemory:
    return "memory";
  case BoundClass::kLatency:
    return "latency";
  }
  return "unknown";
}

// The problem shape must carry positive values for the dims its family uses
// (GEMM: m/n/k; elementwise + reduction: n; attention: n + d) plus a positive
// element size — roofline_position rejects anything else up front.
namespace {
bool shape_valid(KernelFamily family, const Shape &shape) {
  if (shape.dtype_bytes <= 0)
    return false;
  switch (family) {
  case KernelFamily::kGemm:
    return shape.m > 0 && shape.n > 0 && shape.k > 0;
  case KernelFamily::kFusedBiasGelu:
  case KernelFamily::kSoftmax:
  case KernelFamily::kLayerNorm:
  case KernelFamily::kReduction:
    return shape.n > 0;
  case KernelFamily::kFlashAttention:
    return shape.n > 0 && shape.d > 0;
  }
  return false;
}
}  // namespace

ErrorCode roofline_position(KernelFamily family, const Shape &shape,
                            std::int64_t bytes_moved, std::int64_t flops,
                            std::string_view arch, RooflinePosition *out,
                            std::string *error_message, PeakKind peak_kind) {
  if (!shape_valid(family, shape) || bytes_moved <= 0 || flops <= 0) {
    if (error_message != nullptr)
      *error_message =
          "kernscope::roofline: shape (" + std::to_string(shape.m) + "," +
          std::to_string(shape.n) + "," + std::to_string(shape.k) + "," +
          std::to_string(shape.d) + ", " + std::to_string(shape.dtype_bytes) +
          ") invalid for family '" + kernel_family_name(family) +
          "' or bytes_moved/flops not positive (got " +
          std::to_string(bytes_moved) + ", " + std::to_string(flops) + ")";
    return ErrorCode::kInvalidInput;
  }

  MachinePeak peak;
  const ErrorCode lookup = machine_peak(arch, &peak, error_message);
  if (lookup != ErrorCode::kNone)
    return lookup;

  // Placeholder guard (todo-22 contract): gfx1101 has no measured peaks yet,
  // so a roofline position cannot be computed — return an explicit error, and
  // NEVER let a fabricated ridge/ceiling fall out. The output struct stays
  // untouched (no zeros masquerading as a position).
  if (peak.projected) {
    if (error_message != nullptr)
      *error_message =
          "kernscope::roofline: arch '" + std::string(arch) +
          "' peaks are pending-measurement placeholders (projected=true; todo "
          "22 fills measured values + provenance) — roofline position cannot "
          "be computed without measured peaks";
    return ErrorCode::kProjectedPeaks;
  }

  // Select the compute roof for the requested dtype class.
  double compute_peak = 0.0;
  switch (peak_kind) {
  case PeakKind::kFp32:
    if (peak.fp32_peak_flops > 0.0) {
      compute_peak = peak.fp32_peak_flops;
    } else {
      if (error_message != nullptr)
        *error_message =
            "kernscope::roofline: no FP32 peak tabulated for arch '" +
            std::string(arch) +
            "' (the brief tabulates only the " +
            std::string(peak.matrix_dtype) + " matrix peak)";
      return ErrorCode::kNoPeakForDtype;
    }
    break;
  case PeakKind::kMatrix:
    if (peak.matrix_peak_flops > 0.0) {
      compute_peak = peak.matrix_peak_flops;
    } else {
      if (error_message != nullptr)
        *error_message =
            "kernscope::roofline: no matrix peak tabulated for arch '" +
            std::string(arch) + "'";
      return ErrorCode::kNoPeakForDtype;
    }
    break;
  }

  const double ai =
      static_cast<double>(flops) / static_cast<double>(bytes_moved);
  const double ridge = compute_peak / peak.bandwidth_bytes_per_s;
  const double memory_ceiling = ai * peak.bandwidth_bytes_per_s;
  const double compute_ceiling = compute_peak;

  // Bound classification (documented thresholds; see roofline.h):
  //   1. Little's law: time-to-move < DRAM latency -> too small to saturate
  //      the memory roof -> latency-bound.
  //   2. AI >= ridge -> compute-bound (polykernel oracle convention).
  //   3. otherwise memory-bound.
  BoundClass bound = BoundClass::kMemory;
  const double time_to_move_ns =
      static_cast<double>(bytes_moved) * 1e9 /
      peak.bandwidth_bytes_per_s;
  if (time_to_move_ns < kDramLatencyNs)
    bound = BoundClass::kLatency;
  else if (ai >= ridge)
    bound = BoundClass::kCompute;

  if (out != nullptr) {
    out->arithmetic_intensity = ai;
    out->ridge_point = ridge;
    out->compute_bound_ceiling = compute_ceiling;
    out->memory_bound_ceiling = memory_ceiling;
    out->attained_flops = std::min(compute_ceiling, memory_ceiling);
    out->bound_class = bound;
  }
  return ErrorCode::kNone;
}

} // namespace kernscope::roofline
