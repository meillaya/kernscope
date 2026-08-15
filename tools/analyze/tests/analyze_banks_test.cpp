//===- analyze_banks_test.cpp - bank-conflict gtest ------------------------===//
//
// Wave-3 todo 16 acceptance:
//   * 4-way fixture (stride 68 words) -> warning id "bank_conflict" with
//     way==4, stride_words 68, stride_mod_32 4, distinct_banks 8
//     (cross-checked against expected_fused_attention.json bank_conflict
//     evidence);
//   * padded fixture (stride 69, +1 pad idiom) -> NO bank_conflict warning
//     (conflict-free);
//   * fused_attention fixture -> bank_conflict with way==4, stride_words 68,
//     stride_mod_32 4, distinct_banks 8;
//   * QA failure: data-dependent smem index -> "unknown" classification,
//     never conflict-free.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/banks.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

namespace {

using kernscope::KernelReport;
using kernscope::analyze::analyze_banks;
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

bool has_warning(const KernelReport &r, const std::string &id) {
  return std::ranges::any_of(r.warnings,
                             [&](const auto &w) { return w.id == id; });
}

// ---------------------------------------------------------------------------
// Golden 1: 68-word stride -> 4-way bank conflict
// ---------------------------------------------------------------------------

TEST(Banks, Stride68WordsFourWay) {
  const auto mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/bank_conflict.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "bank_conflict";
  analyze_banks(mod, report);

  ASSERT_TRUE(has_warning(report, "bank_conflict")) << "4-way conflict missing";
  for (const auto &w : report.warnings) {
    if (w.id != "bank_conflict") continue;
    EXPECT_EQ(w.evidence.at("way"), "4");
    EXPECT_EQ(w.evidence.at("stride_words"), "68");
    EXPECT_EQ(w.evidence.at("stride_mod_32"), "4");
    EXPECT_EQ(w.evidence.at("distinct_banks"), "8");
    EXPECT_EQ(w.evidence.at("banks_per_sm"), "32");
  }
  // Metrics mirror the evidence.
  EXPECT_EQ(report.metrics["shared_bank_way"], 4);
  EXPECT_EQ(report.metrics["shared_stride_words"], 68);
  EXPECT_EQ(report.metrics["shared_stride_mod_32"], 4);
  EXPECT_EQ(report.metrics["shared_distinct_banks"], 8);
}

// ---------------------------------------------------------------------------
// Golden 2: padded (+1) stride -> conflict-free
// ---------------------------------------------------------------------------

TEST(Banks, PaddedStride69ConflictFree) {
  const auto mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/bank_padded.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "bank_padded";
  analyze_banks(mod, report);

  // 69 % 32 = 5, gcd(5, 32) = 1 -> 32 distinct banks, way 1.
  EXPECT_FALSE(has_warning(report, "bank_conflict")) << "padded stride warned";
  EXPECT_EQ(report.metrics["shared_bank_way"], 1);
  EXPECT_EQ(report.metrics["shared_distinct_banks"], 32);
}

// ---------------------------------------------------------------------------
// fused_attention cross-check (flagship golden)
// ---------------------------------------------------------------------------

TEST(Banks, FusedAttentionWay4CrossCheck) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kFixtures) + "/fused_attention.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "fused_attention";
  analyze_banks(mod, report);

  ASSERT_TRUE(has_warning(report, "bank_conflict"));
  for (const auto &w : report.warnings) {
    if (w.id != "bank_conflict") continue;
    // Cross-check against expected_fused_attention.json bank_conflict:
    // stride_words 68, stride_mod_32 4, distinct_banks 8, way 4.
    EXPECT_EQ(w.evidence.at("way"), "4");
    EXPECT_EQ(w.evidence.at("stride_words"), "68");
    EXPECT_EQ(w.evidence.at("stride_mod_32"), "4");
    EXPECT_EQ(w.evidence.at("distinct_banks"), "8");
    EXPECT_EQ(w.evidence.at("banks_per_sm"), "32");
  }
}

// ---------------------------------------------------------------------------
// QA failure: data-dependent smem index -> "unknown", never conflict-free
// ---------------------------------------------------------------------------

TEST(Banks, DataDependentIndexUnknown) {
  const auto mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/bank_unknown.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "bank_unknown";
  analyze_banks(mod, report);

  EXPECT_FALSE(has_warning(report, "bank_conflict"))
      << "data-dependent index must never look conflict-free (or N-way)";
  bool unknown = false;
  for (const auto &w : report.warnings) {
    if (w.id == "bank_conflict_unknown") {
      unknown = true;
      EXPECT_EQ(w.severity, "unknown-pattern");
    }
  }
  ASSERT_TRUE(unknown) << "data-dependent smem index must be flagged unknown";
  EXPECT_EQ(report.metrics.count("shared_bank_way"), 0)
      << "no fabricated bank count for an unknown pattern";
}

}  // namespace
