//===- roofline_test.cpp - Roofline engine unit tests ----------------------===//
//
// Every expected value is hand-computed here and pinned against
// docs/report/kernscope-technical-report.pdf (the Williams/Waterman/
// Patterson CACM 2009 roofline model, DOI 10.1145/1498765.1498785) and
// cross-checked against the polykernel oracle
// (/home/mei/projects/polykernel/lib/Analysis/tests/roofline_test.cpp —
// values read for validation only, no code copied).
//
//===----------------------------------------------------------------------===//

#include "kernscope/roofline.h"

#include "kernscope/arch.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using kernscope::roofline::BoundClass;
using kernscope::roofline::ErrorCode;
using kernscope::roofline::KernelFamily;
using kernscope::roofline::MachinePeak;
using kernscope::roofline::PeakKind;
using kernscope::roofline::RooflinePosition;
using kernscope::roofline::Shape;
using kernscope::roofline::arithmetic_intensity;
using kernscope::roofline::bound_class_name;
using kernscope::roofline::compute_bytes_moved;
using kernscope::roofline::compute_flops;
using kernscope::roofline::kernel_family_name;
using kernscope::roofline::machine_peak;
using kernscope::roofline::roofline_position;

// ---------------------------------------------------------------------------
// Per-family arithmetic-intensity formulas (brief §7 L132-135).
// ---------------------------------------------------------------------------

// THE pinned acceptance case: GEMM AI(2048^3, fp16). Hand computation from
// the brief's formula (L133: FLOPs = 2MNK; minimum DRAM bytes = s·(MK+KN+MN),
// s = 2 for fp16):
//   FLOPs = 2·2048³ = 2·8589934592 = 17179869184
//   bytes = 2·(2048² + 2048² + 2048²) = 2·3·4194304 = 25165824
//   AI    = 17179869184 / 25165824 = 2048/3 = 682.6666666666667
// NOTE: the brief's "≈683 F/B" example (N=4096) counts FLOPs = MNK (its
// "N/(3s)" form); we count 2MNK exactly per the task spec and the polykernel
// oracle (roofline_test.cpp L23-32: AI = 2·4096³/(3·4096²·2) = 1365.333).
TEST(roofline, AiFormulasGemm2048CubeFp16) {
  const Shape s{.m = 2048, .n = 2048, .k = 2048, .dtype_bytes = 2};
  EXPECT_EQ(compute_flops(KernelFamily::kGemm, s), 17179869184LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kGemm, s), 25165824LL);
  // Hand value: 17179869184 / 25165824 = 2048/3 exactly (within 1e-9).
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kGemm, s), 2048.0 / 3.0,
              1e-9);
}

// Hand-computed AI for every family. All byte counts follow the tile-traffic
// assumptions (brief §7 L133-135); elementwise FLOP counts are the documented
// per-element constants in roofline.h (softmax = 5 from brief L134).
TEST(roofline, AiFormulasAllFamiliesHandComputed) {
  // GEMM 2048³ fp16 (see the pinned case above).
  const Shape gemm{.m = 2048, .n = 2048, .k = 2048, .dtype_bytes = 2};
  // bias+GELU, n=1024, fp32: 1024·14 FLOPs over 3·4·1024 bytes -> 7/6.
  const Shape gelu{.n = 1024, .dtype_bytes = 4};
  // softmax, n=1024, fp32: 1024·5 over 2·4·1024 bytes -> 0.625 (brief L134).
  const Shape softmax{.n = 1024, .dtype_bytes = 4};
  // layernorm, n=1024, fp32: 1024·11 over 2·4·1024 bytes -> 11/8.
  const Shape layernorm{.n = 1024, .dtype_bytes = 4};
  // reduction, n=1024, fp32: 1024 over 4·1024 bytes -> 1/4.
  const Shape reduction{.n = 1024, .dtype_bytes = 4};
  // FlashAttention, n=128, d=64, fp16: 4·128²·64 FLOPs over
  // 2·(4·128·64 + 4·128²) bytes (standard-attention traffic) -> 64/3.
  const Shape attention{.n = 128, .d = 64, .dtype_bytes = 2};

  EXPECT_EQ(compute_flops(KernelFamily::kGemm, gemm), 17179869184LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kGemm, gemm), 25165824LL);

  EXPECT_EQ(compute_flops(KernelFamily::kFusedBiasGelu, gelu), 14336LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kFusedBiasGelu, gelu), 12288LL);
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kFusedBiasGelu, gelu),
              7.0 / 6.0, 1e-12);

  EXPECT_EQ(compute_flops(KernelFamily::kSoftmax, softmax), 5120LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kSoftmax, softmax), 8192LL);
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kSoftmax, softmax), 0.625,
              1e-12);

  EXPECT_EQ(compute_flops(KernelFamily::kLayerNorm, layernorm), 11264LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kLayerNorm, layernorm), 8192LL);
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kLayerNorm, layernorm),
              11.0 / 8.0, 1e-12);

  EXPECT_EQ(compute_flops(KernelFamily::kReduction, reduction), 1024LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kReduction, reduction), 4096LL);
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kReduction, reduction), 0.25,
              1e-12);

  EXPECT_EQ(compute_flops(KernelFamily::kFlashAttention, attention),
            4194304LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kFlashAttention, attention),
            196608LL);
  EXPECT_NEAR(arithmetic_intensity(KernelFamily::kFlashAttention, attention),
              64.0 / 3.0, 1e-9);
}

// Empty shapes must yield 0, never NaN/Inf.
TEST(roofline, AiFormulasEmptyShapeIsZero) {
  const Shape empty{.n = 0, .dtype_bytes = 2};
  EXPECT_EQ(compute_flops(KernelFamily::kSoftmax, empty), 0LL);
  EXPECT_EQ(compute_bytes_moved(KernelFamily::kSoftmax, empty), 0LL);
  EXPECT_EQ(arithmetic_intensity(KernelFamily::kSoftmax, empty), 0.0);
}

TEST(roofline, FamilyNames) {
  EXPECT_STREQ(kernel_family_name(KernelFamily::kGemm), "gemm");
  EXPECT_STREQ(kernel_family_name(KernelFamily::kFusedBiasGelu), "bias_gelu");
  EXPECT_STREQ(kernel_family_name(KernelFamily::kSoftmax), "softmax");
  EXPECT_STREQ(kernel_family_name(KernelFamily::kLayerNorm), "layernorm");
  EXPECT_STREQ(kernel_family_name(KernelFamily::kReduction), "reduction");
  EXPECT_STREQ(kernel_family_name(KernelFamily::kFlashAttention),
               "flash_attention");
  EXPECT_STREQ(bound_class_name(BoundClass::kCompute), "compute");
  EXPECT_STREQ(bound_class_name(BoundClass::kMemory), "memory");
  EXPECT_STREQ(bound_class_name(BoundClass::kLatency), "latency");
}

// ---------------------------------------------------------------------------
// Machine peaks (brief §7 L127-130).
// ---------------------------------------------------------------------------

// Every table value pinned against the brief's verified vendor-doc numbers.
TEST(roofline, MachinePeaksBriefTablePins) {
  MachinePeak peak;
  std::string error;

  // A100: FP32 19.5 TF, FP16/BF16 TC 312 TF dense, HBM2e 2039 GB/s (80GB SXM)
  // [brief §7 L127].
  ASSERT_EQ(machine_peak("sm_80", &peak, &error), ErrorCode::kNone);
  EXPECT_FALSE(peak.projected);
  EXPECT_DOUBLE_EQ(peak.fp32_peak_flops, 19.5e12);
  EXPECT_DOUBLE_EQ(peak.matrix_peak_flops, 312.0e12);
  EXPECT_DOUBLE_EQ(peak.bandwidth_bytes_per_s, 2039.0e9);

  // RTX 4090: FP32 82.6 TF, GDDR6X 1008 GB/s [brief §7 L128]. The FP16 TC peak
  // is the brief's own "≈ 2× FP32 dense" note (165.2 TF), provenance-labeled.
  ASSERT_EQ(machine_peak("sm_89", &peak, &error), ErrorCode::kNone);
  EXPECT_FALSE(peak.projected);
  EXPECT_DOUBLE_EQ(peak.fp32_peak_flops, 82.6e12);
  EXPECT_DOUBLE_EQ(peak.matrix_peak_flops, 165.2e12);
  EXPECT_DOUBLE_EQ(peak.bandwidth_bytes_per_s, 1008.0e9);

  // MI250X: FP16/BF16 matrix 383 TF, ≈3.2-3.28 TB/s HBM2e (3.28 used; no FP32
  // vector peak in the brief) [brief §7 L129].
  ASSERT_EQ(machine_peak("gfx90a", &peak, &error), ErrorCode::kNone);
  EXPECT_FALSE(peak.projected);
  EXPECT_DOUBLE_EQ(peak.matrix_peak_flops, 383.0e12);
  EXPECT_DOUBLE_EQ(peak.bandwidth_bytes_per_s, 3.28e12);
  EXPECT_DOUBLE_EQ(peak.fp32_peak_flops, 0.0);

  // MI300X: BF16 matrix 1307.4 TF dense, FP32 vector 163.4 TF, HBM3 5.3 TB/s
  // [brief §7 L130].
  ASSERT_EQ(machine_peak("gfx942", &peak, &error), ErrorCode::kNone);
  EXPECT_FALSE(peak.projected);
  EXPECT_DOUBLE_EQ(peak.matrix_peak_flops, 1307.4e12);
  EXPECT_DOUBLE_EQ(peak.fp32_peak_flops, 163.4e12);
  EXPECT_DOUBLE_EQ(peak.bandwidth_bytes_per_s, 5.3e12);
}

// gfx1101 (RX 7800 XT) is filled with MEASURED peaks + provenance by todo 22
// (libs/bench / kernscope-bench on this machine, 2026-08-14): the row is
// projected == false with the measured fp32 + DRAM values — never a fabricated
// datasheet number (plan todo 12 MUST NOT / Metis M-6).
TEST(roofline, Gfx1101QueryReturnsMeasuredPeaks) {
  MachinePeak peak;
  std::string error;
  ASSERT_EQ(machine_peak("gfx1101", &peak, &error), ErrorCode::kNone);
  EXPECT_FALSE(peak.projected);
  EXPECT_DOUBLE_EQ(peak.fp32_peak_flops, 16.32e12);
  EXPECT_DOUBLE_EQ(peak.bandwidth_bytes_per_s, 591.0e9);
  EXPECT_DOUBLE_EQ(peak.matrix_peak_flops, 0.0);
  EXPECT_NE(peak.provenance.find("measured"), peak.provenance.npos);
  EXPECT_NE(peak.provenance.find("kernscope-bench"), peak.provenance.npos);
}

// ---------------------------------------------------------------------------
// Bound classification — all three classes, hand-computed on A100.
// ---------------------------------------------------------------------------

// Compute-bound: GEMM 2048³ fp16 on A100 (PeakKind::kMatrix roof).
//   ridge = 312e12 / 2039e9 ≈ 153.02 F/B (brief §7 L133: "A100 ridge ≈
//   312 TF/2.04 TB/s ≈ 153 F/B"); AI = 2048/3 ≈ 682.67 >= ridge -> compute.
TEST(roofline, ClassificationComputeBoundGemmA100) {
  const Shape s{.m = 2048, .n = 2048, .k = 2048, .dtype_bytes = 2};
  const std::int64_t bytes = compute_bytes_moved(KernelFamily::kGemm, s);
  const std::int64_t flops = compute_flops(KernelFamily::kGemm, s);
  RooflinePosition pos;
  std::string error;
  ASSERT_EQ(roofline_position(KernelFamily::kGemm, s, bytes, flops, "sm_80",
                              &pos, &error, PeakKind::kMatrix),
            ErrorCode::kNone);
  EXPECT_NEAR(pos.arithmetic_intensity, 2048.0 / 3.0, 1e-9);
  EXPECT_NEAR(pos.ridge_point, 312.0e12 / 2039.0e9, 1e-6);
  EXPECT_NEAR(pos.ridge_point, 153.0, 0.1);  // the brief's "≈153 F/B"
  EXPECT_EQ(pos.bound_class, BoundClass::kCompute);
  EXPECT_DOUBLE_EQ(pos.compute_bound_ceiling, 312.0e12);
  EXPECT_DOUBLE_EQ(pos.memory_bound_ceiling,
                   (2048.0 / 3.0) * 2039.0e9);
  // Attainable = min(compute, memory) per Williams et al. 2009.
  EXPECT_DOUBLE_EQ(pos.attained_flops, 312.0e12);
}

// Memory-bound: softmax of 2^20 fp32 elements on A100 (fp32 roof).
//   AI = 5/(2·4) = 0.625 (brief §7 L134: "AI ≈ 5/(2s) ≈ 0.6 F/B (fp32)");
//   ridge = 19.5e12/2039e9 ≈ 9.56; bytes = 2·4·2^20 = 8 MiB -> time-to-move
//   ≈ 4.1 µs >> kDramLatencyNs, so not latency-bound; AI < ridge -> memory.
TEST(roofline, ClassificationMemoryBoundSoftmaxA100) {
  const Shape s{.n = 1 << 20, .dtype_bytes = 4};
  const std::int64_t bytes = compute_bytes_moved(KernelFamily::kSoftmax, s);
  const std::int64_t flops = compute_flops(KernelFamily::kSoftmax, s);
  EXPECT_EQ(bytes, 8388608LL);
  RooflinePosition pos;
  std::string error;
  ASSERT_EQ(roofline_position(KernelFamily::kSoftmax, s, bytes, flops, "sm_80",
                              &pos, &error, PeakKind::kFp32),
            ErrorCode::kNone);
  EXPECT_NEAR(pos.arithmetic_intensity, 0.625, 1e-12);
  EXPECT_EQ(pos.bound_class, BoundClass::kMemory);
  EXPECT_DOUBLE_EQ(pos.memory_bound_ceiling, 0.625 * 2039.0e9);
  EXPECT_DOUBLE_EQ(pos.attained_flops, 0.625 * 2039.0e9);  // below fp32 roof
}

// Latency-bound: a tiny softmax (n=64, fp32 = 512 bytes) on A100. Little's
// law (roofline.h): bytes below bandwidth × kDramLatencyNs cannot saturate
// the memory roof — time-to-move = 512e9/2.039e12 ≈ 0.25 ns < 400 ns.
TEST(roofline, ClassificationLatencyBoundTinyKernel) {
  const Shape s{.n = 64, .dtype_bytes = 4};
  const std::int64_t bytes = compute_bytes_moved(KernelFamily::kSoftmax, s);
  const std::int64_t flops = compute_flops(KernelFamily::kSoftmax, s);
  EXPECT_EQ(bytes, 512LL);
  RooflinePosition pos;
  std::string error;
  ASSERT_EQ(roofline_position(KernelFamily::kSoftmax, s, bytes, flops, "sm_80",
                              &pos, &error, PeakKind::kFp32),
            ErrorCode::kNone);
  EXPECT_EQ(pos.bound_class, BoundClass::kLatency);
}

// ---------------------------------------------------------------------------
// Placeholder + negative-path guards.
// ---------------------------------------------------------------------------

// roofline_position on gfx1101 now COMPUTES a position from the todo-22
// MEASURED peaks (projected == false): the GEMM 2048³ fp32 case is
// compute-bound (AI 682.67 >= ridge 16.32e12/591e9 ≈ 27.6).
TEST(roofline, Gfx1101MeasuredPositionComputed) {
  const Shape s{.m = 2048, .n = 2048, .k = 2048, .dtype_bytes = 2};
  RooflinePosition pos;
  std::string error;
  EXPECT_EQ(roofline_position(KernelFamily::kGemm, s, 25165824LL,
                              17179869184LL, "gfx1101", &pos, &error,
                              PeakKind::kFp32),
            ErrorCode::kNone);
  EXPECT_NEAR(pos.arithmetic_intensity, 2048.0 / 3.0, 1e-9);
  EXPECT_NEAR(pos.ridge_point, 16.32e12 / 591.0e9, 1e-6);
  EXPECT_DOUBLE_EQ(pos.compute_bound_ceiling, 16.32e12);
  EXPECT_DOUBLE_EQ(pos.memory_bound_ceiling, (2048.0 / 3.0) * 591.0e9);
  EXPECT_EQ(pos.bound_class, BoundClass::kCompute);
  EXPECT_DOUBLE_EQ(pos.attained_flops, 16.32e12);  // min(compute, memory)
}

// gfx1101 has no matrix peak tabulated (todo-22 measured the FP32 vector +
// DRAM peaks only): PeakKind::kMatrix must fail with kNoPeakForDtype, not
// report a fabricated matrix number.
TEST(roofline, Gfx1101NoMatrixPeakIsTypedError) {
  const Shape s{.m = 2048, .n = 2048, .k = 2048, .dtype_bytes = 2};
  RooflinePosition pos;
  std::string error;
  EXPECT_EQ(roofline_position(KernelFamily::kGemm, s, 25165824LL,
                              17179869184LL, "gfx1101", &pos, &error,
                              PeakKind::kMatrix),
            ErrorCode::kNoPeakForDtype);
  EXPECT_NE(error.find("matrix"), std::string::npos);
}

// Unknown arch -> kUnknownArch + a message containing "unknown arch" (asserted
// by the negative-path QA), NOT zeros.
TEST(roofline, UnknownArchReturnsErrorCodeAndMessage) {
  const Shape s{.n = 1024, .dtype_bytes = 4};

  MachinePeak peak;
  std::string error;
  EXPECT_EQ(machine_peak("sm_77", &peak, &error), ErrorCode::kUnknownArch);
  EXPECT_NE(error.find("unknown arch"), std::string::npos);

  RooflinePosition pos;
  pos.arithmetic_intensity = -1.0;  // sentinel: must stay untouched
  error.clear();
  EXPECT_EQ(roofline_position(KernelFamily::kSoftmax, s, 8192LL, 5120LL,
                              "sm_77", &pos, &error, PeakKind::kFp32),
            ErrorCode::kUnknownArch);
  EXPECT_NE(error.find("unknown arch"), std::string::npos);
  EXPECT_NE(error.find("sm_77"), std::string::npos);
  EXPECT_DOUBLE_EQ(pos.arithmetic_intensity, -1.0);
}

// A known GPU with no row in the todo-12 table (sm_90) is a distinct error —
// still never zeros.
TEST(roofline, KnownGpuWithoutPeaksRowIsExplicitError) {
  EXPECT_TRUE(kernscope::has_arch("sm_90"));
  MachinePeak peak;
  std::string error;
  EXPECT_EQ(machine_peak("sm_90", &peak, &error),
            ErrorCode::kNoPeaksTabulated);
  EXPECT_NE(error.find("no machine peaks tabulated"), std::string::npos);
}

// PeakKind::kFp32 on gfx90a: the brief tabulates MI250X only for matrix ops.
TEST(roofline, NoFp32RoofOnGfx90a) {
  const Shape s{.n = 1024, .dtype_bytes = 4};
  RooflinePosition pos;
  std::string error;
  EXPECT_EQ(roofline_position(KernelFamily::kSoftmax, s, 8192LL, 5120LL,
                              "gfx90a", &pos, &error, PeakKind::kFp32),
            ErrorCode::kNoPeakForDtype);
  EXPECT_NE(error.find("FP32"), std::string::npos);
}

// Invalid inputs are rejected up front (negative-path contract).
TEST(roofline, InvalidInputRejected) {
  const Shape s{.n = 1024, .dtype_bytes = 4};
  RooflinePosition pos;
  std::string error;
  EXPECT_EQ(roofline_position(KernelFamily::kSoftmax, s, 0LL, 5120LL, "sm_80",
                              &pos, &error, PeakKind::kFp32),
            ErrorCode::kInvalidInput);
  EXPECT_EQ(roofline_position(KernelFamily::kSoftmax, s, 8192LL, -1LL, "sm_80",
                              &pos, &error, PeakKind::kFp32),
            ErrorCode::kInvalidInput);
  const Shape bad{.n = 0, .dtype_bytes = 4};  // no elements
  EXPECT_EQ(roofline_position(KernelFamily::kSoftmax, bad, 8192LL, 5120LL,
                              "sm_80", &pos, &error, PeakKind::kFp32),
            ErrorCode::kInvalidInput);
}

} // namespace
