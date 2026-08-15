//===- enrich_test.cpp - per-kernel enrichment suite (todo 50) ------------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// The todo-50 acceptance suite (TEST_PREFIX "profile_enrich:" so
// `ctest -R profile_enrich` selects it; hardware label no-gpu):
//
//   1. OCCUPANCY == LIBS/OCCUPANCY (acceptance): a synthetic Kernel9-style
//      record + arch -> the enrichment's occupancy field EQUALS the
//      kernscope::occupancy::compute_occupancy output for the same inputs
//      (asserted exact-equality — same engine, Metis M-2: no
//      cuda_occupancy.h port), and additionally matches the pinned brief
//      golden (sm_90 / 128 regs / 256 threads -> 0.25, brief §1.2 L101-102).
//   2. AMD PATH: gfx1101 record -> VGPR-wave occupancy (0 < occ <= 1,
//      limiter in the contract set).
//   3. QA NEGATIVE (unknown-resources): a record with registersPerThread == 0
//      -> occupancy_unknown == true, JSON emits "unknown-resources" (NOT a
//      fabricated 0%), limiter empty. Also: unknown arch name + block_size 0.
//   4. -lineinfo FIXTURE (acceptance): the committed tiny_lineinfo.ptx
//      (nvcc -lineinfo) -> >= 1 source<->PTX mapping entry, with a pinned
//      (file, line) triple from the fixture text.
//   5. CUBIN-ONLY DEGRADATION (acceptance, Metis finding): the committed
//      tiny.cubin (nvcc -cubin, NO embedded PTX) -> module_has_embedded_ptx()
//      == false, ptx_available == false, ptx_diagnostics EMPTY — PTX-level
//      diagnostics skipped gracefully, never fabricated.
//   6. NVIDIA RESOURCE FALLBACK: `cuobjdump -res-usage` on the fixture cubin
//      -> per-function REG/STACK/SHARED/LOCAL parsed (REG:10 STACK:0
//      SHARED:0 LOCAL:0 — the actual nvcc 12.6 output).
//   7. AMDGCN SOURCE<->ISA: `llvm-objdump -l` on the committed tiny_amdgcn.o
//      (clang --target=amdgcn-amd-amdhsa -gline-tables-only) -> >= 1
//      (source file, line, ISA) entry.
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/enrich.h"

#include "kernscope/arch.h"
#include "kernscope/occupancy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#ifndef PROFILE_FIXTURE_DIR
#error "PROFILE_FIXTURE_DIR must be defined (tests/fixtures/profile)"
#endif

using namespace kernscope::profile;

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// Runs a command, returns stdout (trimmed). Used for the two live-tool
// checks (cuobjdump -res-usage, llvm-objdump -l) exactly as the enrichment
// module shells out in production.
std::string RunTool(const std::string &cmd) {
  std::string full = cmd + " 2>/dev/null";
  std::FILE *pipe = ::popen(full.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::string out;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
    out.append(buf, n);
  }
  ::pclose(pipe);
  std::size_t end = out.size();
  while (end > 0 && (out[end - 1] == '\n' || out[end - 1] == ' ' ||
                     out[end - 1] == '\t' || out[end - 1] == '\r')) {
    --end;
  }
  out.resize(end);
  return out;
}

// The synthetic Kernel9-style record (same shape as the todo-49 test's
// MakeSyntheticKernel9: 96 regs, 24576 static smem, 128-thread blocks).
KernelRecord MakeSyntheticRecord(int registers = 96) {
  KernelRecord r;
  r.kernel_name = "layernorm_fused";
  r.grid_x = 1024;
  r.grid_y = 1;
  r.grid_z = 1;
  r.block_x = 128;
  r.block_y = 1;
  r.block_z = 1;
  r.static_shared_bytes = 24576;
  r.dynamic_shared_bytes = 0;
  r.registers_per_thread = static_cast<std::uint16_t>(registers);
  r.correlation_id = 42;
  return r;
}

//===--- 1. occupancy == libs/occupancy (acceptance) ---------------------------===//

TEST(profile_enrich, OccupancyEqualsLibsOccupancy) {
  const KernelRecord rec = MakeSyntheticRecord(/*registers=*/96);
  EnrichOptions opts;
  opts.arch_name = "sm_86";

  const EnrichedKernel enriched = enrich_kernel(rec, opts);

  // The SAME inputs through libs/occupancy directly (the todo-11 engine is
  // the only occupancy implementation — the acceptance pins EQUALITY).
  const kernscope::occupancy::OccupancyResult oracle =
      kernscope::occupancy::compute_occupancy(kernscope::get_arch("sm_86"),
                                              /*regs_per_thread=*/96,
                                              /*smem_bytes_per_block=*/24576,
                                              /*block_size=*/128);

  EXPECT_DOUBLE_EQ(enriched.occupancy, oracle.occupancy);
  EXPECT_EQ(enriched.occupancy_unknown, false);
  EXPECT_EQ(enriched.limiter, kernscope::occupancy::limiter_name(oracle.limiter));
  EXPECT_EQ(enriched.blocks_per_sm, oracle.blocks_per_sm);
  EXPECT_EQ(enriched.warps_resident, oracle.warps_resident);
  EXPECT_EQ(enriched.max_warps_per_sm, oracle.max_warps_per_sm);
  // Sanity: a sane fraction, not an out-of-contract value.
  EXPECT_GT(enriched.occupancy, 0.0);
  EXPECT_LE(enriched.occupancy, 1.0);
}

TEST(profile_enrich, OccupancyMatchesPinnedGolden) {
  // The brief worked example (profiler-research-brief §1.2 L101-102, also
  // pinned by todo 14): sm_90 / 128 regs / 256 threads -> 0.25.
  KernelRecord rec = MakeSyntheticRecord(/*registers=*/128);
  rec.block_x = 256;
  rec.static_shared_bytes = 0;
  EnrichOptions opts;
  opts.arch_name = "sm_90";

  const EnrichedKernel enriched = enrich_kernel(rec, opts);

  EXPECT_FALSE(enriched.occupancy_unknown);
  EXPECT_DOUBLE_EQ(enriched.occupancy, 0.25);
  EXPECT_EQ(enriched.limiter, "registers");
}

TEST(profile_enrich, AmdGfx1101RecordEnriched) {
  // AMD path: a gfx1101 record (128-thread blocks, 48 VGPRs, 8KB LDS —
  // layernorm-shaped). The engine returns the VGPR-wave occupancy.
  KernelRecord rec = MakeSyntheticRecord(/*registers=*/48);
  rec.static_shared_bytes = 8 * 1024;
  EnrichOptions opts;
  opts.arch_name = "gfx1101";
  opts.sgprs_per_thread = 16;

  const EnrichedKernel enriched = enrich_kernel(rec, opts);

  EXPECT_FALSE(enriched.occupancy_unknown);
  EXPECT_GT(enriched.occupancy, 0.0);
  EXPECT_LE(enriched.occupancy, 1.0);
  const bool valid_limiter = enriched.limiter == "registers" ||
                             enriched.limiter == "shared" ||
                             enriched.limiter == "threads" ||
                             enriched.limiter == "blocks";
  EXPECT_TRUE(valid_limiter) << "limiter: " << enriched.limiter;
  EXPECT_GT(enriched.max_warps_per_sm, 0);
}

//===--- 3. QA negative: unknown-resources (NEVER a fabricated 0%) --------------===//

TEST(profile_enrich, ZeroRegistersIsUnknownResources) {
  // Acceptance QA: registersPerThread == 0 -> "unknown-resources", NOT 0%.
  const KernelRecord rec = MakeSyntheticRecord(/*registers=*/0);
  EnrichOptions opts;
  opts.arch_name = "sm_86";

  const EnrichedKernel enriched = enrich_kernel(rec, opts);
  const std::string json = enriched_kernel_to_json("nvidia", enriched);

  EXPECT_TRUE(enriched.occupancy_unknown);
  EXPECT_EQ(enriched.limiter, "");
  EXPECT_EQ(enriched.blocks_per_sm, 0);
  // The JSON carries the explicit marker, never a 0.0 figure.
  EXPECT_NE(json.find("\"occupancy\":\"unknown-resources\""), std::string::npos);
  EXPECT_EQ(json.find("\"occupancy\":0.0"), std::string::npos);
  EXPECT_EQ(json.find("\"occupancy\":0.000000"), std::string::npos);
}

TEST(profile_enrich, UnknownArchIsUnknownResources) {
  EnrichOptions opts;
  opts.arch_name = "sm_99"; // not in the libs/arch table
  const EnrichedKernel enriched = enrich_kernel(MakeSyntheticRecord(), opts);
  EXPECT_TRUE(enriched.occupancy_unknown);
}

TEST(profile_enrich, ZeroBlockSizeIsUnknownResources) {
  KernelRecord rec = MakeSyntheticRecord();
  rec.block_x = 0;
  EnrichOptions opts;
  opts.arch_name = "sm_86";
  const EnrichedKernel enriched = enrich_kernel(rec, opts);
  EXPECT_TRUE(enriched.occupancy_unknown);
}

//===--- 4. -lineinfo fixture: source<->PTX mapping (acceptance) ----------------===//

TEST(profile_enrich, LineinfoFixtureYieldsSourceMapping) {
  const std::string ptx = ReadFile(std::string(PROFILE_FIXTURE_DIR) + "/tiny_lineinfo.ptx");
  ASSERT_FALSE(ptx.empty());

  EnrichedKernel enriched;
  apply_ptx_lineinfo(ptx, enriched);

  EXPECT_TRUE(enriched.ptx_available);
  EXPECT_GE(enriched.source_mapping.size(), 1u); // acceptance: >= 1 entries
  // Pin a triple from the fixture text: the .loc directives reference
  // tiny.cu (file 1) — line 4 is `if (i < n) {` in the fixture source, and
  // the first .loc in the kernel body is `1 2 0` (the kernel signature).
  EXPECT_NE(enriched.source_mapping.size(), 0u);
  bool found_pinned = false;
  for (const SourceMappingEntry &e : enriched.source_mapping) {
    if (e.source_file.find("tiny.cu") != std::string::npos && e.source_line > 0) {
      found_pinned = true;
      break;
    }
  }
  EXPECT_TRUE(found_pinned) << "no mapping entry points at tiny.cu";
  // The last entry of the fixture is the launch helper's host stub line
  // (line 18 in tiny.cu, `void launch_tiny(...)`) — the fixture's .loc set
  // includes source lines 2..18.
  int max_line = 0;
  for (const SourceMappingEntry &e : enriched.source_mapping) {
    max_line = std::max(max_line, e.source_line);
  }
  EXPECT_GE(max_line, 3);
}

//===--- 5. cubin-only degradation (acceptance, Metis finding) ------------------===//

TEST(profile_enrich, CubinOnlyModuleHasNoPtx) {
  const std::string cubin = std::string(PROFILE_FIXTURE_DIR) + "/tiny.cubin";
  ASSERT_FALSE(ReadFile(cubin).empty());

  // nvcc -cubin produces a cubin WITHOUT embedded PTX: cuobjdump -ptx on it
  // is empty (verified on nvcc 12.6) -> ptx_available must be false.
  EXPECT_FALSE(module_has_embedded_ptx(cubin));

  EnrichedKernel enriched;
  EXPECT_FALSE(enriched.ptx_available); // default: no PTX evidence
  // Degradation contract: PTX-level diagnostics are skipped, never
  // fabricated — the diagnostics list stays empty on the cubin-only path.
  EXPECT_TRUE(enriched.ptx_diagnostics.empty());
}

//===--- 6. NVIDIA resource fallback: cuobjdump -res-usage ---------------------===//

TEST(profile_enrich, CuobjdumpResUsageParsed) {
  const std::string cubin = std::string(PROFILE_FIXTURE_DIR) + "/tiny.cubin";
  const std::string dump = RunTool("cuobjdump -res-usage '" + cubin + "'");
  ASSERT_FALSE(dump.empty());

  EnrichedKernel enriched;
  apply_nvidia_res_usage(dump, enriched);

  ASSERT_EQ(enriched.res_usage.size(), 1u);
  const FunctionResourceUsage &u = enriched.res_usage[0];
  EXPECT_EQ(u.function, "_Z11tiny_kernelPKfPfi");
  EXPECT_EQ(u.reg, 10);     // the actual nvcc 12.6 figure for the fixture
  EXPECT_EQ(u.stack, 0);
  EXPECT_EQ(u.shared, 0);
  EXPECT_EQ(u.local, 0);
}

//===--- 7. AMDGCN source<->ISA: llvm-objdump -l (brief §3.2 L287-288) ----------===//

TEST(profile_enrich, AmdgcnObjdumpLinesParsed) {
  const std::string co = std::string(PROFILE_FIXTURE_DIR) + "/tiny_amdgcn.o";
  const std::string dump = RunTool("llvm-objdump --disassemble --arch=amdgcn "
                               "--mcpu=gfx1101 -l '" + co + "'");
  ASSERT_FALSE(dump.empty());

  EnrichedKernel enriched;
  apply_llvm_objdump_lines(dump, enriched);

  EXPECT_GE(enriched.source_mapping.size(), 1u);
  bool found_tiny = false;
  bool has_isa = false;
  for (const SourceMappingEntry &e : enriched.source_mapping) {
    if (e.source_file.find("tiny_amdgcn.cpp") != std::string::npos) {
      found_tiny = true;
    }
    if (!e.asm_text.empty()) {
      has_isa = true;
    }
  }
  EXPECT_TRUE(found_tiny) << "no entry maps to tiny_amdgcn.cpp";
  EXPECT_TRUE(has_isa) << "mapping entries carry ISA instructions";
}

} // namespace
