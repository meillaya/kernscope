//===- analyze_regs_test.cpp - register pressure + spill gtest ------------===//
//
// Wave-3 todo 13 acceptance:
//   * ptxas -v parsing supports BOTH published formats (cubin one-liner +
//     NVCC doc two-part form) [brief §6.1 L397-409];
//   * static proxy counts declared .reg registers + .local frame bytes;
//   * spill-positive fixture (tests/analyze/corpus/spill_positive.ptx, an
//     nvcc --maxrregcount=16 build of a huge dynamically-indexed accumulator
//     kernel) -> findings include a "spill" warning with a byte count;
//   * spill-negative fixture -> no spill warning;
//   * fused_attention cross-check: ptxas -v "Used 102 registers" parses to
//     regs_per_thread 102, spill 0/0/0 (matches expected_fused_attention.json
//     metrics.registers_per_thread==102, spill_stores==0, spill_loads==0,
//     local_stack_bytes==0);
//   * QA failure: a kernel body with ZERO .reg declarations yields the
//     graceful "cannot estimate register pressure" diagnostic, never a
//     crash and never a fabricated register count.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/regs.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using kernscope::PtxModule;
using kernscope::analyze::PtxasVReport;
using kernscope::analyze::RegPressureEstimate;
using kernscope::analyze::analyze_regs;
using kernscope::analyze::estimate_reg_pressure;
using kernscope::analyze::parse_ptxas_v;
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
// ptxas -v parsing (both formats)
// ---------------------------------------------------------------------------

TEST(Regs, PtxasVCubinOneLiner) {
  // The committed fused_attention manifest's exact ptxas -v line
  // (tests/fixtures/manifest.json ptxas_v_report).
  const PtxasVReport v = parse_ptxas_v(
      "ptxas info    : Used 102 registers, used 1 barriers, 388 bytes cmem[0]");
  ASSERT_TRUE(v.ok);
  EXPECT_EQ(v.regs_per_thread, 102);
  EXPECT_EQ(v.stack_frame_bytes, 0);
  EXPECT_EQ(v.spill_stores_bytes, 0);
  EXPECT_EQ(v.spill_loads_bytes, 0);
  EXPECT_EQ(v.cmem_bytes, 388);
}

TEST(Regs, PtxasVNvccDocTwoPartForm) {
  // The NVCC §8.4 documented format: stack/spill line + register line
  // [brief §6.1 L399-405].
  const PtxasVReport v = parse_ptxas_v(
      "ptxas info    : Compiling entry function 'acos_main' for 'sm_80'\n"
      "ptxas info    : Function properties for acos_main\n"
      "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
      "ptxas info    : Used 6 registers, 1536 bytes smem, 32 bytes cmem[0]\n");
  ASSERT_TRUE(v.ok);
  EXPECT_EQ(v.regs_per_thread, 6);
  EXPECT_EQ(v.smem_bytes, 1536);
  EXPECT_EQ(v.cmem_bytes, 32);
  EXPECT_EQ(v.spill_stores_bytes, 0);
  EXPECT_EQ(v.spill_loads_bytes, 0);
}

TEST(Regs, PtxasVWithSpills) {
  const PtxasVReport v = parse_ptxas_v(
      "ptxas info    : Function properties for foo\n"
      "    648 bytes stack frame, 1348 bytes spill stores, "
      "1348 bytes spill loads\n"
      "ptxas info    : Used 24 registers, used 0 barriers\n");
  ASSERT_TRUE(v.ok);
  EXPECT_EQ(v.regs_per_thread, 24);
  EXPECT_EQ(v.stack_frame_bytes, 648);
  EXPECT_EQ(v.spill_stores_bytes, 1348);
  EXPECT_EQ(v.spill_loads_bytes, 1348);
}

TEST(Regs, PtxasVGarbageIsNotOk) {
  EXPECT_FALSE(parse_ptxas_v("this is not a ptxas log\n").ok);
  EXPECT_FALSE(parse_ptxas_v("").ok);
}

// ---------------------------------------------------------------------------
// Static proxy + spill detection on the committed fixtures
// ---------------------------------------------------------------------------

TEST(Regs, SpillPositiveFixtureHasSpillWarning) {
  // nvcc --maxrregcount=16 build of a huge dynamically-indexed accumulator:
  // PTX carries a 512-byte .local frame and ld.local/st.local traffic.
  const PtxModule mod =
      parse_ptx_module(read_file(std::string(kCorpus) + "/spill_positive.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  const RegPressureEstimate est = estimate_reg_pressure(mod.kernels().front());
  ASSERT_TRUE(est.can_estimate);
  EXPECT_GT(est.stack_frame_bytes, 0);
  EXPECT_GT(est.spill_stores, 0);
  EXPECT_GT(est.spill_loads, 0);

  kernscope::KernelReport report;
  report.kernel_name = "spill_positive";
  analyze_regs(mod, report);

  bool found_spill = false;
  for (const auto &w : report.warnings) {
    if (w.id == "spill") {
      found_spill = true;
      // Byte count evidence is present and machine-readable.
      EXPECT_FALSE(w.evidence.at("spill_bytes").empty());
      EXPECT_EQ(w.evidence.at("spill_stores"), std::to_string(est.spill_stores));
      EXPECT_EQ(w.evidence.at("spill_loads"), std::to_string(est.spill_loads));
    }
  }
  ASSERT_TRUE(found_spill) << "spill warning missing on spill-positive fixture";
  EXPECT_EQ(report.metrics["spill_stores"], est.spill_stores);
  EXPECT_EQ(report.metrics["spill_loads"], est.spill_loads);
  EXPECT_EQ(report.metrics["local_stack_bytes"], est.stack_frame_bytes);
  EXPECT_EQ(report.metrics["spill_bytes"], est.stack_frame_bytes);
}

TEST(Regs, SpillNegativeFixtureHasNoSpillWarning) {
  // hoge kernel (no arrays, no .local): must NOT produce a spill warning.
  const PtxModule mod = parse_ptx_module(
      read_file(std::string(kCorpus) + "/spill_negative.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  kernscope::KernelReport report;
  report.kernel_name = "hoge";
  analyze_regs(mod, report);

  for (const auto &w : report.warnings) {
    EXPECT_NE(w.id, "spill") << "spill warning on spill-negative fixture";
  }
  EXPECT_EQ(report.metrics["spill_stores"], 0);
  EXPECT_EQ(report.metrics["spill_loads"], 0);
}

// ---------------------------------------------------------------------------
// fused_attention cross-check (flagship golden)
// ---------------------------------------------------------------------------

TEST(Regs, FusedAttentionMatchesExpectedJson) {
  // ptxas -v (manifest.json) -> regs_per_thread 102; the expected JSON pins
  // metrics.registers_per_thread==102, spill_stores==0, spill_loads==0,
  // local_stack_bytes==0.
  const PtxModule mod = parse_ptx_module(
      read_file(std::string(kFixtures) + "/fused_attention.ptx"));
  ASSERT_FALSE(mod.kernels().empty());

  kernscope::KernelReport report;
  report.kernel_name = "fused_attention";
  analyze_regs(mod,
               "ptxas info    : Used 102 registers, used 1 barriers, "
               "388 bytes cmem[0]",
               report);

  EXPECT_EQ(report.metrics["registers_per_thread"], 102);
  EXPECT_EQ(report.metrics["spill_stores"], 0);
  EXPECT_EQ(report.metrics["spill_loads"], 0);
  EXPECT_EQ(report.metrics["local_stack_bytes"], 0);
  EXPECT_EQ(report.metrics["spill_bytes"], 0);
  bool no_spill = true;
  for (const auto &w : report.warnings)
    if (w.id == "spill") no_spill = false;
  EXPECT_TRUE(no_spill);
}

// ---------------------------------------------------------------------------
// QA failure: zero .reg declarations -> graceful diagnostic
// ---------------------------------------------------------------------------

TEST(Regs, ZeroRegDeclarationsGracefulDiagnostic) {
  // A body declaring no registers at all: the module must say so instead of
  // inventing a number.
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_86\n"
      ".address_size 64\n"
      ".visible .entry empty()\n"
      "{\n"
      "\tld.param.u64 \t%rd1, [empty_param_0];\n"
      "\tret;\n"
      "}\n";
  const PtxModule mod = parse_ptx_module(ptx);
  ASSERT_FALSE(mod.kernels().empty());
  ASSERT_TRUE(mod.kernels().front().variables.empty())
      << "fixture must declare zero registers";

  kernscope::KernelReport report;
  report.kernel_name = "empty";
  EXPECT_NO_THROW(analyze_regs(mod, report));

  bool diagnostic = false;
  for (const auto &w : report.warnings) {
    if (w.id == "cannot_estimate_registers") {
      diagnostic = true;
      EXPECT_NE(w.evidence.at("text").find("cannot estimate register pressure"),
                std::string::npos);
    }
  }
  ASSERT_TRUE(diagnostic) << "graceful diagnostic missing on zero-.reg kernel";
  EXPECT_EQ(report.metrics.count("registers_per_thread"), 0);
}

}  // namespace
