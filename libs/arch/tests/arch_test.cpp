//===- arch_test.cpp - architecture-model DB unit tests -------------------===//
//
// Every assertion is pinned against docs/research/gpu-static-analyzer-research.md
// (cited as "brief §X LNN") or docs/research/rdna3-constants.md (the
// ISA-resolved RDNA3 VGPR file). Hand-computed occupancy cases are
// cross-checked against the polykernel oracle
// (/home/mei/projects/polykernel/lib/Analysis/tests/occupancy_test.cpp —
// values read for validation only, no code copied).
//
//===----------------------------------------------------------------------===//

#include "kernscope/arch.h"

#include <gtest/gtest.h>

#include <string>

namespace {

using kernscope::OccupancyPrediction;
using kernscope::Vendor;
using kernscope::allocated_regs_per_thread;
using kernscope::get_arch;
using kernscope::has_arch;
using kernscope::has_tensor_dtype;
using kernscope::max_resident_warps;
using kernscope::predict_occupancy;
using kernscope::waves_per_simd_from_vgprs;

// ---------------------------------------------------------------------------
// NVIDIA table (brief §1.2 table at L74-84; shared constants at L70-72:
// warp size 32; 64K registers/SM; 255 registers max per thread)
// ---------------------------------------------------------------------------

TEST(arch, NvidiaSharedConstants) {
  // brief §1.2 L70-71: warp size 32; 64K 32-bit registers/SM; 255 regs max.
  for (const char *n : {"sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100",
                        "sm_120"}) {
    const auto &a = get_arch(n);
    EXPECT_EQ(a.vendor, Vendor::kNvidia) << n;
    EXPECT_EQ(a.warp_size, 32) << n << " (brief §1.2 L70)";
    EXPECT_EQ(a.regs_per_sm, 65536) << n << " (brief §1.2 L70)";
    EXPECT_EQ(a.max_regs_per_thread, 255) << n << " (brief §1.2 L71)";
    // brief §1.2 L91-99: per-thread granule 8 => 256 registers per warp.
    EXPECT_EQ(a.reg_granule, 8) << n << " (brief §1.2 L95-96)";
    EXPECT_EQ(a.simds_per_cu, 1) << n;
  }
}

TEST(arch, NvidiaPerArchTable) {
  // Row-by-row pins of brief §1.2 L74-84 (max threads/SM, max warps/SM,
  // max blocks/SM, smem/SM KB, smem/block opt-in KB).
  struct Row {
    const char *name;
    int threads, warps, blocks, smem_kb, smem_block_kb;
  };
  constexpr Row kRows[] = {
      {"sm_75", 1024, 32, 16, 64, 64},      // L76
      {"sm_80", 2048, 64, 32, 164, 163},    // L77
      {"sm_86", 1536, 48, 16, 100, 99},     // L78
      {"sm_89", 1536, 48, 24, 100, 99},     // L80
      {"sm_90", 2048, 64, 32, 228, 227},    // L81
      {"sm_100", 2048, 64, 32, 228, 227},   // L82
      {"sm_120", 1536, 48, 24, 100, 99},    // L84
  };
  for (const Row &r : kRows) {
    const auto &a = get_arch(r.name);
    EXPECT_EQ(a.max_threads_per_sm, r.threads) << r.name << " (brief §1.2 L74)";
    EXPECT_EQ(a.max_warps_per_sm, r.warps) << r.name << " (brief §1.2 L74)";
    EXPECT_EQ(a.max_blocks_per_sm, r.blocks)
        << r.name << " (brief §1.2 L86-88)";
    EXPECT_EQ(a.smem_per_sm_bytes, r.smem_kb * 1024)
        << r.name << " (brief §1.2 L74; KB=KiB per guide)";
    EXPECT_EQ(a.smem_per_block_max_bytes, r.smem_block_kb * 1024)
        << r.name << " (brief §1.2 L74)";
  }
}

TEST(arch, NvidiaTensorDtypes) {
  // brief §5.1 L343-344 (CPG Table 33): 7.5: f16/s8/s4; 8.0 adds f64/tf32/
  // bf16; 8.9 adds FP8; 10.0 adds FP6/FP4; 12.x: tf32/bf16/f16/fp8/fp6/fp4/int8.
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_75"), "f16"));
  EXPECT_FALSE(has_tensor_dtype(get_arch("sm_75"), "bf16"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_80"), "tf32"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_80"), "bf16"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_80"), "f64"));
  EXPECT_FALSE(has_tensor_dtype(get_arch("sm_86"), "fp8"));  // FP8 needs sm_89
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_89"), "fp8"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_90"), "fp8"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_100"), "fp6"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_100"), "fp4"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_120"), "fp8"));
  EXPECT_TRUE(has_tensor_dtype(get_arch("sm_120"), "int8"));
  EXPECT_FALSE(has_tensor_dtype(get_arch("sm_120"), "f64"));  // brief L344 list
}

// ---------------------------------------------------------------------------
// AMD CDNA (brief §1.3 L106-131)
// ---------------------------------------------------------------------------

TEST(arch, CdnaConstants) {
  // brief §1.3: CU = 4 SIMDs L107; wavefront 64 L107; VGPR file 512/SIMD with
  // granule 8 L108-112; SGPR granule 16 L117-121; LDS 64 KB/CU L123; max
  // 8 waves/SIMD L122; up to 256 VGPRs per wave L108-110.
  for (const char *n : {"gfx90a", "gfx942"}) {
    const auto &a = get_arch(n);
    EXPECT_EQ(a.vendor, Vendor::kAmd) << n;
    EXPECT_EQ(a.warp_size, 64) << n << " (brief §1.3 L107)";
    EXPECT_EQ(a.simds_per_cu, 4) << n << " (brief §1.3 L107)";
    EXPECT_EQ(a.regs_per_sm, 512) << n << " (brief §1.3 L108-112)";
    EXPECT_EQ(a.reg_granule, 8) << n << " (brief §1.3 L108)";
    EXPECT_EQ(a.sgpr_granule, 16) << n << " (brief §1.3 L117)";
    EXPECT_EQ(a.smem_per_sm_bytes, 64 * 1024) << n << " (brief §1.3 L123)";
    EXPECT_EQ(a.max_warps_per_sm, 8) << n << " (brief §1.3 L122)";
    EXPECT_EQ(a.max_regs_per_thread, 256) << n << " (brief §1.3 L109-110)";
    EXPECT_EQ(max_resident_warps(a), 32) << n << " (brief §1.3 L122: 8x4)";
  }
}

TEST(arch, CdnaOccShThresholdTable) {
  // brief §1.3 L113-116: ROCm/Triton occ.sh equivalent of floor(512/ceil8):
  // <=64->8, 65-72->7, 73-80->6, 81-96->5, 97-128->4, 129-168->3, 169-256->2,
  // >256->1 wave/SIMD.
  const auto &gfx90a = get_arch("gfx90a");
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 64), 8);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 72), 7);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 80), 6);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 96), 5);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 128), 4);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 168), 3);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 256), 2);
  EXPECT_EQ(waves_per_simd_from_vgprs(gfx90a, 257), 1);
}

// ---------------------------------------------------------------------------
// AMD RDNA3 (brief §1.3 L133-144 + docs/research/rdna3-constants.md)
// ---------------------------------------------------------------------------

TEST(arch, Rdna3Constants) {
  // brief §1.3 L136: native wave32; L142-143: 64 KB LDS per work-group;
  // L140-141: LLVM max 16 waves/EU; L134-135: CU = 2 SIMD32s.
  // VGPR file 1536/SIMD + granule 24 wave32 / 12 wave64:
  // docs/research/rdna3-constants.md (RDNA3 ISA §3.3.2.1; LLVM
  // AMDGPUBaseInfo.cpp L1295-1344 @llvmorg-21.1.8; driver
  // regsPerMultiprocessor=196608; empirical probe on RX 7800 XT).
  const auto &a = get_arch("gfx1101");
  EXPECT_EQ(a.vendor, Vendor::kAmd);
  EXPECT_EQ(a.warp_size, 32) << " (brief §1.3 L136)";
  EXPECT_EQ(a.simds_per_cu, 2) << " (brief §1.3 L134-135)";
  EXPECT_EQ(a.smem_per_sm_bytes, 64 * 1024) << " (brief §1.3 L142-143)";
  EXPECT_EQ(a.smem_per_block_max_bytes, 64 * 1024) << " (brief §1.3 L143)";
  EXPECT_EQ(a.max_warps_per_sm, 16) << " (brief §1.3 L140-141)";
  EXPECT_EQ(a.max_regs_per_thread, 256) << " (brief §1.3 L137)";
  // THE pinned resolution: 1536 VGPRs/SIMD, granule 24 (wave32) / 12 (wave64).
  EXPECT_EQ(a.regs_per_sm, 1536)
      << "gfx1101 VGPR file per SIMD is 1536 (wave32 view), NOT 768: "
         "docs/research/rdna3-constants.md (RDNA3 ISA §3.3.2.1; LLVM "
         "AMDGPUBaseInfo.cpp L1336-1344; driver regsPerMultiprocessor = "
         "1536x2x64)";
  EXPECT_EQ(a.reg_granule, 24)
      << "gfx1101 VGPR granule must be 24 (wave32) / 12 (wave64), never 16: "
         "docs/research/rdna3-constants.md (RDNA3 ISA §3.3.2.1; granule-16 "
         "refuted empirically on RX 7800 XT: occupancy steps observed at "
         "align-12 positions 108/120/132/168/204/228)";
  EXPECT_EQ(a.vgpr_granule_wave64, 12)
      << "gfx1101 wave64 granule is 12: docs/research/rdna3-constants.md";
  EXPECT_EQ(a.sgpr_granule, 16) << " (brief §1.3 L117-121)";
}

TEST(arch, Gfx1101MeasuredPeaksFilledByTodo22) {
  // todo 22 (libs/bench / kernscope-bench) measured the gfx1101 roofline
  // peaks on THIS machine (RX 7800 XT, 2026-08-14) and filled the placeholders
  // with provenance (reports/peaks/gfx1101-bench.jsonl). The floors match the
  // todo-22 acceptance (bw > 300 GB/s, fp32 > 10 TFLOPS) and the source tag
  // carries the measured-<date>-<arch> provenance form.
  const auto &a = get_arch("gfx1101");
  EXPECT_TRUE(a.measured);
  EXPECT_GT(a.peak_bw_gbps, 300.0);
  EXPECT_GT(a.peak_fp32_tflops, 10.0);
  EXPECT_GT(a.peak_lds_gbps, 1000.0);
  EXPECT_NE(std::string(a.peak_source).find("measured-2026-08-14-gfx1101"),
            std::string::npos);
}

// Negative golden: tampering the RDNA3 granule to 16 must NOT reproduce the
// pinned model — the prediction diverges, and the pinned-value assertions
// above fail with a message citing docs/research/rdna3-constants.md.
TEST(arch, Rdna3Granule16TamperDiverges) {
  const auto &pinned = get_arch("gfx1101");
  kernscope::ArchInfo tampered = pinned;
  tampered.reg_granule = 16;  // the "don't hardcode 16" trap [brief §10.4 L551-553]
  tampered.vgpr_granule_wave64 = 8;

  // 97 VGPRs/lane wave32: pinned granule 24 -> ceil to 120 -> 12 waves/SIMD;
  // tampered granule 16 -> ceil to 112 -> 13 waves/SIMD. Must diverge.
  const auto pinned_occ = predict_occupancy(pinned, 32, 97, 0);
  const auto tampered_occ = predict_occupancy(tampered, 32, 97, 0);
  EXPECT_NE(pinned_occ.blocks_per_cu, tampered_occ.blocks_per_cu)
      << "tampering gfx1101 VGPR granule to 16 must change occupancy: "
         "docs/research/rdna3-constants.md pins 24 (wave32)/12 (wave64) from "
         "RDNA3 ISA §3.3.2.1 (brief §10.4 L551-553: \"don't hardcode 16\")";
  EXPECT_NE(pinned_occ.blocks_per_cu, 26) << "pinned model: 24 blocks at 97 "
      "VGPRs (granule 24); 26 would imply the refuted granule-16 model "
      "(docs/research/rdna3-constants.md evidence (e))";
  EXPECT_EQ(pinned_occ.blocks_per_cu, 24)
      << "gfx1101 97 VGPRs x 32 threads -> 24 blocks/CU with pinned granule "
         "24 (floor(1536/align24(97)) x 2 SIMDs); see "
         "docs/research/rdna3-constants.md";
}

// ---------------------------------------------------------------------------
// Occupancy prediction (brief §1.1 L33-45 min-wins rule) — hand-computed
// cases cross-validated against the polykernel oracle
// /home/mei/projects/polykernel/lib/Analysis/tests/occupancy_test.cpp
// (values read oracle-only; no code copied).
// ---------------------------------------------------------------------------

TEST(arch, HandComputedRegisterLimitedSm90) {
  // Oracle case 1 (occupancy_test.cpp L22-35): 128 regs, 32 KB smem,
  // 256 threads on sm_90 -> blocks=2 (regs), 16 warps, 25%.
  const auto o = predict_occupancy(get_arch("sm_90"), 256, 128, 32 * 1024);
  EXPECT_EQ(o.blocks_per_cu, 2);        // 65536/(128*256) [brief §1.2 L101-102]
  EXPECT_EQ(o.active_warps, 16);
  EXPECT_EQ(o.max_warps, 64);           // brief §1.2 L81
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 25.0);
  EXPECT_STREQ(o.limiter, "registers");
}

TEST(arch, HandComputedRegisterLimitedSm80) {
  // Oracle case 2 (occupancy_test.cpp L40-45): same shape on sm_80
  // (smem/SM 167936 B = 164 KB per brief §1.2 L77).
  const auto o = predict_occupancy(get_arch("sm_80"), 256, 128, 32 * 1024);
  EXPECT_EQ(o.blocks_per_cu, 2);
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 25.0);
  EXPECT_STREQ(o.limiter, "registers");
  EXPECT_EQ(get_arch("sm_80").smem_per_sm_bytes, 167936);  // 164 KB, brief L77
}

TEST(arch, HandComputedSmemLimitedSm90) {
  // Oracle case 3 (occupancy_test.cpp L52-58): 32 regs, 100 KB smem,
  // 256 threads -> 2 blocks (smem), 25%.
  const auto o = predict_occupancy(get_arch("sm_90"), 256, 32, 100 * 1024);
  EXPECT_EQ(o.blocks_per_cu, 2);        // 233472/102400 [brief §1.2 L81]
  EXPECT_EQ(o.active_warps, 16);
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 25.0);
  EXPECT_STREQ(o.limiter, "smem");
  EXPECT_EQ(get_arch("sm_90").smem_per_sm_bytes, 233472);  // 228 KB, brief L81
}

TEST(arch, HandComputedWarpLimitedSm90) {
  // Oracle case 4 (occupancy_test.cpp L65-71): 16 regs, 0 smem, 1024 threads
  // -> 2 blocks (warps), 100%.
  const auto o = predict_occupancy(get_arch("sm_90"), 1024, 16, 0);
  EXPECT_EQ(o.blocks_per_cu, 2);        // 64 warps / 32 warps-per-block
  EXPECT_EQ(o.active_warps, 64);
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 100.0);
  EXPECT_STREQ(o.limiter, "warps");
}

TEST(arch, HandComputedBlockLimitedSm90) {
  // Oracle case 5 (occupancy_test.cpp L77-83): 16 regs, 0 smem, 32 threads
  // -> 32 blocks (blocks cap), 50%.
  const auto o = predict_occupancy(get_arch("sm_90"), 32, 16, 0);
  EXPECT_EQ(o.blocks_per_cu, 32);       // max blocks/SM, brief §1.2 L86-88
  EXPECT_EQ(o.active_warps, 32);
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 50.0);
  EXPECT_STREQ(o.limiter, "blocks");
}

TEST(arch, MatmulKernelSm90) {
  // Oracle case 6 (occupancy_test.cpp L90-95): 31 regs, 1024 B, 256 threads
  // -> 8 blocks (regs tie-break), 100%.
  const auto o = predict_occupancy(get_arch("sm_90"), 256, 31, 1024);
  EXPECT_EQ(o.blocks_per_cu, 8);        // 65536/(align8(31)*256) = 8
  EXPECT_EQ(o.active_warps, 64);
  EXPECT_DOUBLE_EQ(o.occupancy_pct, 100.0);
  EXPECT_STREQ(o.limiter, "registers");
}

TEST(arch, OccupancyHelpers) {
  EXPECT_EQ(max_resident_warps(get_arch("sm_86")), 48);  // brief §1.2 L78
  EXPECT_EQ(max_resident_warps(get_arch("sm_100")), 64);  // brief §1.2 L82
  EXPECT_EQ(max_resident_warps(get_arch("gfx1101")), 32);  // 16 x 2 SIMDs
  // Granule rounding: brief §1.2 L95-96 (256/warp), §1.3 L108-112 (CDNA 8).
  EXPECT_EQ(allocated_regs_per_thread(get_arch("sm_90"), 31), 32);
  EXPECT_EQ(allocated_regs_per_thread(get_arch("gfx90a"), 65), 72);
  EXPECT_EQ(allocated_regs_per_thread(get_arch("gfx1101"), 97), 120);
}

TEST(arch, LookupAndNames) {
  EXPECT_TRUE(has_arch("sm_80"));
  EXPECT_TRUE(has_arch("gfx1101"));
  EXPECT_FALSE(has_arch("sm_77"));
  EXPECT_FALSE(has_arch("gfx1102"));  // not in DB (no 1.5xVGPRs, rdna3-constants.md caveat)
  EXPECT_THROW(get_arch("sm_77"), std::out_of_range);
  EXPECT_EQ(get_arch("sm_86").name, "sm_86");
  EXPECT_EQ(get_arch("gfx942").name, "gfx942");
}

// ---------------------------------------------------------------------------
// RDNA3 occupancy sanity (brief §1.3 L139-141 formula; granule per
// rdna3-constants.md): waves = floor(1536 / ceil24(vgpr)) clamped to 16.
// ---------------------------------------------------------------------------

TEST(arch, Rdna3WavesPerSimd) {
  const auto &a = get_arch("gfx1101");
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 24), 16);    // 1536/24, clamped
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 48), 16);    // 1536/48 = 32 -> cap 16
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 97), 12);    // ceil24 -> 120: 12.8
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 160), 9);    // ceil24 -> 168: 9.1
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 200), 7);    // ceil24 -> 216: 7.1
  EXPECT_EQ(waves_per_simd_from_vgprs(a, 256), 5);    // ceil24 -> 264: 5.8
}

}  // namespace
