//===- analyze_coalescing_test.cpp - coalescing sector-model gtest --------===//
//
// Wave-3 todo 15 acceptance:
//   * contiguous unit-stride 4-byte load -> 4 sectors/warp, 100% efficiency
//     (brief §3.1 L218-226: "four 32-byte memory transactions", 100%);
//   * stride-32B pattern -> 32 sectors, 12.5% efficiency (brief §3.1 worst
//     case, "1024 bytes... utilization 12.5%");
//   * fused_attention fixture -> "coalescing" warning with efficiency 0.5
//     (the fixture's Q-tile stride-2 gathers: 8 sectors x 32 B transferred
//     for 128 B used, cross-checked against expected_fused_attention.json
//     evidence bytes_used_per_warp 128 / bytes_transferred_per_warp 256 /
//     efficiency 0.5 / vectorized_ld_global 0);
//   * QA failure: indirect/unresolvable pattern -> severity
//     "unknown-pattern" (never a false "coalesced").
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/coalescing.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using kernscope::KernelReport;
using kernscope::analyze::analyze_coalescing;
using kernscope::parse_ptx_module;

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

const char *kCorpus = TEST_CORPUS_DIR;
const char *kFixtures = FIXTURES_DIR;

// ---------------------------------------------------------------------------
// Golden 1: fully-coalesced unit-stride load -> 4 sectors, 100%
// ---------------------------------------------------------------------------

TEST(Coalesce, CoalescedUnitStrideFourSectors) {
  const auto mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/coalesced.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "coalesced";
  analyze_coalescing(mod, report);

  // Per-lane stride 4B, width 4B: 32 lanes span 128 B = 4 sectors, 100%.
  EXPECT_EQ(report.metrics["global_sectors_per_warp"], 4);
  EXPECT_NEAR(report.metrics["global_coalescing_efficiency"], 1.0, 1e-9);
  EXPECT_EQ(report.metrics["global_bytes_used_per_warp"], 128);
  EXPECT_EQ(report.metrics["global_bytes_transferred_per_warp"], 128);
  bool warned = false;
  for (const auto &w : report.warnings)
    if (w.id == "coalescing") warned = true;
  EXPECT_FALSE(warned) << "fully-coalesced access must not warn";
}

// ---------------------------------------------------------------------------
// Golden 2: stride-32B pattern -> 32 sectors, 12.5%
// ---------------------------------------------------------------------------

TEST(Coalesce, Stride32BThirtyTwoSectors) {
  const auto mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/strided_global.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "strided_global";
  analyze_coalescing(mod, report);

  // Each lane touches a distinct 32B sector -> 32 sectors, efficiency
  // 128/(32*32) = 0.125 (brief §3.1 worst case).
  EXPECT_EQ(report.metrics["global_sectors_per_warp"], 32);
  EXPECT_NEAR(report.metrics["global_coalescing_efficiency"], 0.125, 1e-9);
  bool warned = false;
  for (const auto &w : report.warnings)
    if (w.id == "coalescing") warned = true;
  ASSERT_TRUE(warned) << "strided access must warn";
}

// ---------------------------------------------------------------------------
// fused_attention cross-check (flagship golden)
// ---------------------------------------------------------------------------

TEST(Coalesce, FusedAttentionEfficiency050) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kFixtures) + "/fused_attention.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "fused_attention";
  analyze_coalescing(mod, report);

  bool found = false;
  for (const auto &w : report.warnings) {
    if (w.id != "coalescing") continue;
    found = true;
    // Cross-check against expected_fused_attention.json coalescing evidence.
    EXPECT_DOUBLE_EQ(std::stod(w.evidence.at("bytes_used_per_warp")), 128.0);
    EXPECT_DOUBLE_EQ(
        std::stod(w.evidence.at("bytes_transferred_per_warp")), 256.0);
    EXPECT_NEAR(std::stod(w.evidence.at("efficiency")), 0.5, 1e-9);
    EXPECT_EQ(std::stoi(w.evidence.at("vectorized_ld_global")), 0);
  }
  ASSERT_TRUE(found) << "fused_attention must produce a coalescing warning";
  // The metric mirrors the evidence.
  EXPECT_NEAR(report.metrics["global_coalescing_efficiency"], 0.5, 1e-9);
  EXPECT_LT(report.metrics["global_coalescing_efficiency"], 1.0);
}

// ---------------------------------------------------------------------------
// QA failure: indirect/unresolvable pattern -> "unknown-pattern" severity
// ---------------------------------------------------------------------------

TEST(Coalesce, IndirectPatternUnknownSeverity) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/indirect_global.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "indirect_global";
  analyze_coalescing(mod, report);

  // The load's index comes from a global read -> unresolvable. The module
  // must say so with severity "unknown-pattern", never a false "coalesced"
  // warning and never a fabricated efficiency number.
  bool unknown = false;
  for (const auto &w : report.warnings) {
    if (w.id != "coalescing_unknown") continue;
    unknown = true;
    EXPECT_EQ(w.severity, "unknown-pattern");
  }
  ASSERT_TRUE(unknown) << "unresolvable pattern must be flagged unknown";
  bool false_coalesced = false;
  for (const auto &w : report.warnings)
    if (w.id == "coalescing") false_coalesced = true;
  EXPECT_FALSE(false_coalesced) << "no false 'coalesced' verdict allowed";
  EXPECT_EQ(report.metrics.count("global_coalescing_efficiency"), 0);
}

}  // namespace
