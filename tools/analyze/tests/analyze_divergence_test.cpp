//===- analyze_divergence_test.cpp - divergence + barrier gtest ------------===//
//
// Wave-3 todo 17 acceptance:
//   * divergent-branch-in-loop fixture -> warning id "divergent_inner_loop"
//     with predicate_derived_from threadIdx.x and loop_depth 2;
//   * uniform-control fixture -> ZERO divergence warnings;
//   * barrier-in-branch fixture -> "divergent_barrier" warning (CUDA UB);
//   * fused_attention -> divergent_inner_loop (predicate from threadIdx.x,
//     loop_depth 2, setp_count 77, predicated_branch_count 78, matching
//     expected_fused_attention.json);
//   * QA failure: data-dependent branch -> "possibly-divergent"
//     (three-valued), never claimed convergent.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/divergence.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

namespace {

using kernscope::KernelReport;
using kernscope::analyze::analyze_divergence;
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
// Golden 1: divergent branch inside a nested loop
// ---------------------------------------------------------------------------

TEST(Div, DivergentBranchInLoopWarning) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/divergent_inner_loop.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "divergent_inner_loop";
  analyze_divergence(mod, report);

  ASSERT_TRUE(has_warning(report, "divergent_inner_loop"));
  for (const auto &w : report.warnings) {
    if (w.id != "divergent_inner_loop") continue;
    EXPECT_EQ(w.evidence.at("predicate_derived_from"), "threadIdx.x");
    // Nested loop: outer + inner back edges dominate the branch block.
    EXPECT_EQ(w.evidence.at("loop_depth"), "2");
  }
}

// ---------------------------------------------------------------------------
// Golden 2: uniform control -> zero divergence warnings
// ---------------------------------------------------------------------------

TEST(Div, UniformControlNoDivergenceWarnings) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/uniform_control.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "uniform_control";
  analyze_divergence(mod, report);

  for (const auto &w : report.warnings) {
    EXPECT_NE(w.id, "divergent_inner_loop");
    EXPECT_NE(w.id, "possibly_divergent_branch");
    EXPECT_NE(w.id, "divergent_barrier");
  }
}

// ---------------------------------------------------------------------------
// Golden 3: barrier inside a divergent region -> divergent barrier
// ---------------------------------------------------------------------------

TEST(Div, BarrierInDivergentRegionWarning) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/barrier_in_branch.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "barrier_in_branch";
  analyze_divergence(mod, report);

  ASSERT_TRUE(has_warning(report, "divergent_barrier"));
}

// ---------------------------------------------------------------------------
// fused_attention cross-check (flagship golden)
// ---------------------------------------------------------------------------

TEST(Div, FusedAttentionDivergentInnerLoop) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kFixtures) + "/fused_attention.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "fused_attention";
  analyze_divergence(mod, report);

  ASSERT_TRUE(has_warning(report, "divergent_inner_loop"));
  for (const auto &w : report.warnings) {
    if (w.id != "divergent_inner_loop") continue;
    // expected_fused_attention.json divergent_inner_loop evidence:
    // predicate_derived_from threadIdx.x, loop_depth 2, setp_count 77,
    // predicated_branch_count 78.
    EXPECT_EQ(w.evidence.at("predicate_derived_from"), "threadIdx.x");
    EXPECT_EQ(w.evidence.at("loop_depth"), "2");
    EXPECT_EQ(std::stoi(w.evidence.at("setp_count")), 77);
    EXPECT_EQ(std::stoi(w.evidence.at("predicated_branch_count")), 78);
  }
  // Barrier inventory: exactly 2 bar.sync (expected JSON bar_sync 2).
  EXPECT_EQ(report.metrics["bar_sync_count"], 2);
  // The two top-level barriers are NOT divergent (no false positive).
  EXPECT_FALSE(has_warning(report, "divergent_barrier"));
}

// ---------------------------------------------------------------------------
// QA failure: data-dependent branch -> possibly-divergent (three-valued)
// ---------------------------------------------------------------------------

TEST(Div, DataDependentBranchPossiblyDivergent) {
  const auto mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/data_dependent_branch.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  KernelReport report;
  report.kernel_name = "data_dependent_branch";
  analyze_divergence(mod, report);

  ASSERT_TRUE(has_warning(report, "possibly_divergent_branch"));
  for (const auto &w : report.warnings) {
    if (w.id != "possibly_divergent_branch") continue;
    EXPECT_EQ(w.evidence.at("verdict"), "possibly-divergent");
  }
  // Never claimed convergent for a data-dependent condition.
  EXPECT_FALSE(has_warning(report, "divergent_inner_loop"));
}

}  // namespace
