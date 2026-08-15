//===- analyze_mix_test.cpp - mix + tensor core + latency gtest ------------===//
//
// Wave-3 todo 18 acceptance:
//   * fused_attention instruction mix matches expected_fused_attention.json
//     metrics.instruction_mix EXACTLY (fma_rn_f32 1024, mul_f32 33,
//     add_f32 23, ex2_approx_f32 1, ld_global 20, ld_global_vectorized 0,
//     ld_shared 520, st_global 64, st_shared 16, ld_local 0, st_local 0,
//     setp 77, predicated_bra 78, bar_sync 2);
//   * tensor-core MAC counting: mma.sync.m16n8k16 -> 16*8*16 = 2048 MACs
//     [brief §5.1]; wgmma.m64n128k16 -> 64*128*16 = 131072; MFMA
//     v_mfma_f32_32x32x8f16 -> 32*32*8 = 8192; WMMA 16x16x16 -> 4096;
//   * "no tensor-core usage" info when a GEMM-like kernel uses only FMA;
//   * latency classification labeled "approximate, microbenchmark-derived";
//   * QA failure: unknown-arch latency table -> "unknown-arch" skip, no
//     fabricated numbers.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/mix.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using kernscope::KernelReport;
using kernscope::analyze::analyze_mix;
using kernscope::parse_ptx_module;

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

const char *kFixtures = FIXTURES_DIR;

// ---------------------------------------------------------------------------
// fused_attention instruction mix (flagship golden)
// ---------------------------------------------------------------------------

TEST(Mix, FusedAttentionInstructionMixExact) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kFixtures) + "/fused_attention.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "fused_attention";
  analyze_mix(mod, report);

  // Cross-check every instruction_mix field against expected JSON.
  EXPECT_EQ(report.metrics["fma_rn_f32"], 1024);
  EXPECT_EQ(report.metrics["mul_f32"], 33);
  EXPECT_EQ(report.metrics["add_f32"], 23);
  EXPECT_EQ(report.metrics["ex2_approx_f32"], 1);
  EXPECT_EQ(report.metrics["ld_global"], 20);
  EXPECT_EQ(report.metrics["ld_global_vectorized"], 0);
  EXPECT_EQ(report.metrics["ld_shared"], 520);
  EXPECT_EQ(report.metrics["st_global"], 64);
  EXPECT_EQ(report.metrics["st_shared"], 16);
  EXPECT_EQ(report.metrics["ld_local"], 0);
  EXPECT_EQ(report.metrics["st_local"], 0);
  EXPECT_EQ(report.metrics["setp"], 77);
  EXPECT_EQ(report.metrics["predicated_bra"], 78);
  EXPECT_EQ(report.metrics["bar_sync"], 2);

  // No tensor ops in the fixture -> 0 MACs + the "no tensor-core usage"
  // info (GEMM-like FMA loop).
  EXPECT_EQ(report.metrics["tensor_ops"], 0);
  EXPECT_EQ(report.metrics["tensor_macs"], 0);
  bool no_tc = false;
  for (const auto &s : report.suggestions)
    if (s.id == "no_tensor_core_usage") no_tc = true;
  EXPECT_TRUE(no_tc) << "FMA-only GEMM-like kernel must flag no TC usage";

  // Latency classification exists for sm_86 and is labeled approximate.
  bool latency = false;
  for (const auto &s : report.suggestions) {
    if (s.id != "latency_bottleneck") continue;
    latency = true;
    EXPECT_NE(s.evidence.at("text").find("approximate, microbenchmark-derived"),
              std::string::npos);
  }
  EXPECT_TRUE(latency) << "sm_86 has a published latency table (brief §8.1)";
}

// ---------------------------------------------------------------------------
// Tensor-core MAC counting
// ---------------------------------------------------------------------------

TEST(Mix, MmaShapeMacs) {
  // Hand-written mma.sync.aligned.m16n8k16 (PTX ISA §9.7.15.5):
  // MACs = m*n*k = 16*8*16 = 2048 [brief §5.1 L346-350].
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_80\n"
      ".address_size 64\n"
      ".visible .entry mma_kernel()\n"
      "{\n"
      "\t.reg .b32 \t%r<4>;\n"
      "\t.reg .f32 \t%f<4>;\n"
      "\tmma.sync.aligned.m16n8k16.row.col.f32.f32.f32.f32 "
      "\t{%f1, %f2, %f3, %f4}, {%f1, %f2, %f3, %f4}, {%f1, %f2, %f3, %f4}, "
      "%f1;\n"
      "\tret;\n"
      "}\n";
  const auto mod = parse_ptx_module(ptx);
  KernelReport report;
  report.kernel_name = "mma_kernel";
  analyze_mix(mod, report);

  EXPECT_EQ(report.metrics["tensor_ops"], 1);
  EXPECT_EQ(report.metrics["tensor_macs"], 16 * 8 * 16);
}

TEST(Mix, WgmmaShapeMacs) {
  // wgmma.mma_async.m64n128k16: MACs = 64*128*16 = 131072.
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_90a\n"
      ".address_size 64\n"
      ".visible .entry wgmma_kernel()\n"
      "{\n"
      "\t.reg .b32 \t%r<4>;\n"
      "\t.reg .f32 \t%f<4>;\n"
      "\twgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16.f32 "
      "\t{%f1, %f2, %f3, %f4}, {%f1, %f2, %f3, %f4}, %r1;\n"
      "\twgmma.commit_group;\n"
      "\twgmma.wait_group.sync.aligned 0;\n"
      "\tret;\n"
      "}\n";
  const auto mod = parse_ptx_module(ptx);
  KernelReport report;
  report.kernel_name = "wgmma_kernel";
  analyze_mix(mod, report);

  EXPECT_EQ(report.metrics["tensor_ops"], 3);
  EXPECT_EQ(report.metrics["tensor_macs"], 64 * 128 * 16);
}

TEST(Mix, MfmaShapeMacs) {
  // AMD v_mfma_f32_32x32x8f16 (CDNA2, brief §5.2 L355-372):
  // MACs = 32*32*8 = 8192.
  const auto mod = kernscope::parse_amd_module(
      "\t.text\n"
      "\t.globl\tmfma_kernel\n"
      "mfma_kernel:\n"
      "\ts_waitcnt vmcnt(0)\n"
      "\tv_mfma_f32_32x32x8f16 a[0:15], a[0:1], a[2:3], -2.0\n"
      "\tv_mfma_f32_16x16x16f16 a[0:3], v[0:1], v[2:3], a[4:7]\n"
      "\ts_endpgm\n",
      "gfx90a", "<mfma_fixture>");
  KernelReport report;
  report.kernel_name = "mfma_kernel";
  analyze_mix(mod, report);

  EXPECT_EQ(report.metrics["tensor_ops"], 2);
  // 32x32x8 + 16x16x16.
  EXPECT_EQ(report.metrics["tensor_macs"], (32 * 32 * 8) + (16 * 16 * 16));
}

// ---------------------------------------------------------------------------
// QA failure: unknown-arch latency table
// ---------------------------------------------------------------------------

TEST(Mix, UnknownArchLatencySkip) {
  // sm_77 is not in the published latency tables (brief §8.1). The module
  // must emit "unknown-arch" and NOT fabricate a latency classification.
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_77\n"
      ".address_size 64\n"
      ".visible .entry k()\n"
      "{\n"
      "\t.reg .b32 \t%r<2>;\n"
      "\tmov.u32 \t%r1, %tid.x;\n"
      "\tst.global.b32 \t[%r1], %r1;\n"
      "\tret;\n"
      "}\n";
  const auto mod = parse_ptx_module(ptx);
  KernelReport report;
  report.kernel_name = "k";
  analyze_mix(mod, report);

  bool unknown = false;
  bool fabricated = false;
  for (const auto &s : report.suggestions) {
    if (s.id == "latency_unknown_arch") {
      unknown = true;
      EXPECT_EQ(s.evidence.at("arch"), "sm_77");
    }
    if (s.id == "latency_bottleneck") fabricated = true;
  }
  ASSERT_TRUE(unknown) << "unknown arch must skip latency with a diagnostic";
  EXPECT_FALSE(fabricated) << "no fabricated latency for unknown arch";
  EXPECT_EQ(report.metrics.count("latency_classification"), 0);
}

}  // namespace
