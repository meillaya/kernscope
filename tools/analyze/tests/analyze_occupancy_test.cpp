//===- analyze_occupancy_test.cpp - occupancy + limiter gtest --------------===//
//
// Wave-3 todo 14 acceptance:
//   * fused_attention @ pinned (sm_86, 64-thread) -> occupancy==0.375 (1e-9),
//     limiter=="registers" [tests/fixtures/manifest.json + expected JSON];
//   * brief worked example (sm_90, 128 regs, 256 threads) -> occupancy==0.25
//     [brief §1.2 L101-102], fed as synthetic resource values;
//   * all occupancy figures are labeled THEORETICAL (static tool, never
//     "achieved");
//   * QA failure: block size exceeding the arch max threads -> limiter
//     "threads" + a warning, never a crash.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/occupancy.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using kernscope::KernelReport;
using kernscope::analyze::OccupancyInputs;
using kernscope::analyze::analyze_occupancy;
using kernscope::get_arch;
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
// fused_attention golden (flagship)
// ---------------------------------------------------------------------------

TEST(Occ, FusedAttentionGoldenOccupancy0375) {
  const std::string ptx =
      read_file(std::string(kFixtures) + "/fused_attention.ptx");
  const auto mod = parse_ptx_module(ptx);
  ASSERT_FALSE(mod.kernels().empty());

  // Registers come from todo 13 (ptxas -v, manifest.json); block size from
  // the fixture manifest (64). The static .shared bytes are 0 (extern
  // dynamic smem[]), so the register limiter binds at 9 blocks x 2 warps /
  // 48 max warps = 0.375, limiter "registers".
  KernelReport report;
  report.kernel_name = "fused_attention";
  report.metrics["registers_per_thread"] = 102;  // manifest.json
  analyze_occupancy(mod, OccupancyInputs{64, 0, ""}, report);

  EXPECT_NEAR(report.metrics["occupancy"], 0.375, 1e-9);
  EXPECT_EQ(report.limiter, "registers");
  EXPECT_EQ(report.metrics["blocks_per_sm"], 9);
  EXPECT_EQ(report.metrics["warps_resident_per_sm"], 18);
  EXPECT_EQ(report.metrics["max_warps_per_sm"], 48);
  EXPECT_EQ(report.metrics["max_blocks_per_sm"], 16);
  // Theoretical labeling is mandatory (never "achieved").
  EXPECT_EQ(report.metrics["occupancy_is_theoretical"], 1.0);
  // Manifest: 102 regs -> 104 (granule-8) x 32 = 3328 regs/warp.
  EXPECT_EQ(report.metrics["registers_allocated_per_warp"], 3328);
  // Register limiter wins over threads (24) / warps (24) headroom.
  EXPECT_LE(report.metrics["headroom_blocks_by_registers"], 9.0);
  EXPECT_EQ(report.metrics["headroom_blocks_by_threads"], 24);
  EXPECT_EQ(report.metrics["headroom_blocks_by_warps"], 24);
}

// ---------------------------------------------------------------------------
// Brief worked example (sm_90, 128 regs, 256 threads -> 0.25)
// ---------------------------------------------------------------------------

TEST(Occ, BriefWorkedExampleSm90_025) {
  // brief §1.2 L101-102: 128 regs x 256 threads = 32768 regs/block ->
  // floor(65536/32768) = 2 blocks -> 16 warps / 64 = 0.25. Fed synthetically
  // through the resource-level entry.
  KernelReport report;
  report.kernel_name = "brief_example";
  analyze_occupancy(get_arch("sm_90"), /*regs_per_thread=*/128, /*smem_bytes_per_block=*/0,
                    /*block_size=*/256, report);

  EXPECT_NEAR(report.metrics["occupancy"], 0.25, 1e-9);
  EXPECT_EQ(report.limiter, "registers");
  EXPECT_EQ(report.metrics["blocks_per_sm"], 2);
  EXPECT_EQ(report.metrics["warps_resident_per_sm"], 16);
  EXPECT_EQ(report.metrics["max_warps_per_sm"], 64);
  EXPECT_EQ(report.metrics["occupancy_is_theoretical"], 1.0);
}

TEST(Occ, SmemLimiterBindsWithStaticShared) {
  // When static .shared declarations carry a size, the smem limiter binds:
  // sm_86 (100 KB smem/SM), 16 regs, 32 KB per block -> floor(102400/32768)
  // = 3 blocks; occupancy 3x2/48 = 0.125, limiter "shared".
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_86\n"
      ".address_size 64\n"
      ".visible .entry s()\n"
      "{\n"
      "\t.shared .align 4 .b32 sdata[8192];\n"  // 32 KB
      "\t.reg .b32 %r<4>;\n"
      "\tmov.u32 \t%r1, %tid.x;\n"
      "\tst.shared.b32 \t[sdata], %r1;\n"
      "\tret;\n"
      "}\n";
  const auto mod = parse_ptx_module(ptx);
  KernelReport report;
  report.kernel_name = "s";
  report.metrics["registers_per_thread"] = 16;
  analyze_occupancy(mod, OccupancyInputs{64, 0, ""}, report);

  EXPECT_EQ(report.metrics["shared_memory_bytes"], 32 * 1024);
  EXPECT_EQ(report.limiter, "shared");
  EXPECT_EQ(report.metrics["blocks_per_sm"], 3);
  EXPECT_NEAR(report.metrics["occupancy"], 3.0 * 2.0 / 48.0, 1e-9);
}

// ---------------------------------------------------------------------------
// QA failure: block size exceeding arch max threads
// ---------------------------------------------------------------------------

TEST(Occ, OversizedBlockLimiterThreadsWarning) {
  // sm_86 max threads/SM = 1536; block 2048 -> floor(1536/2048) = 0 blocks ->
  // limiter "threads", theoretical occupancy 0, and a warning. Never a crash.
  KernelReport report;
  report.kernel_name = "oversized";
  analyze_occupancy(get_arch("sm_86"), /*regs_per_thread=*/32, /*smem_bytes_per_block=*/0,
                    /*block_size=*/2048, report);

  EXPECT_EQ(report.limiter, "threads");
  EXPECT_EQ(report.metrics["occupancy"], 0.0);
  bool warned = false;
  for (const auto &w : report.warnings)
    if (w.id == "block_size_exceeds_arch_max") warned = true;
  ASSERT_TRUE(warned) << "oversized-block warning missing";
  EXPECT_EQ(report.metrics["occupancy_is_theoretical"], 1.0);
}

// ---------------------------------------------------------------------------
// Block size unknown -> graceful diagnostic
// ---------------------------------------------------------------------------

TEST(Occ, UnknownBlockSizeGracefulDiagnostic) {
  const std::string ptx =
      ".version 8.5\n"
      ".target sm_86\n"
      ".address_size 64\n"
      ".visible .entry k()\n"
      "{\n"
      "\t.reg .b32 \t%r<2>;\n"
      "\tret;\n"
      "}\n";
  const auto mod = parse_ptx_module(ptx);
  KernelReport report;
  report.kernel_name = "k";
  report.metrics["registers_per_thread"] = 16;
  analyze_occupancy(mod, report);

  bool diagnostic = false;
  for (const auto &w : report.warnings)
    if (w.id == "block_size_unknown") diagnostic = true;
  ASSERT_TRUE(diagnostic) << "block_size_unknown diagnostic missing";
}

}  // namespace
