//===- profile_diag_test.cpp - ranking + diagnostics suite (todo 51) --------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// The todo-51 acceptance suite (TEST_PREFIX "profile_diag:" so
// `ctest -R profile_diag` selects it; hardware label no-gpu):
//
//   1. HOT RANKING (acceptance): a synthetic trace of 3 kernels (one
//      register-bloated, one layernorm-family, one small/healthy) ->
//      RANKING ORDER EXACT (total_ns desc) with exact total/median/launch
//      counts and % of wall time asserted numerically.
//   2. REGISTER-BLOATED DIAGNOSTIC (acceptance): the bloated kernel (168
//      regs/thread, block 256, sm_80) -> limiter == "occupancy" + hint rule
//      id "register_bloat" whose evidence carries the regs_per_thread field.
//   3. MEMORY-BOUND via libs/roofline (plan spec): the layernorm-family
//      kernel -> limiter "memory" + memory_bound hint with the
//      record-derived bytes_moved_estimate + arithmetic_intensity evidence.
//   4. ANALYZER MERGE (acceptance, the profiler<->analyzer cross-tool
//      multiplier): the committed profile_bank_conflict.ptx fixture (68-word
//      stride -> 4-way bank conflict) provided as "extracted PTX" ->
//      diagnose_kernel INVOKES the real kernscope-analyze CLI and merges its
//      finding: hint id "bank_conflict" with evidence
//      analyze_finding_id == "bank_conflict" + way == 4.
//   5. QA NEGATIVE (insufficient-evidence): kernel with NO PTX + unknown
//      family + unknown occupancy -> diagnostics degrade to exactly the
//      "insufficient_evidence" rule (asserted), NO fabricated hints.
//   6. NO FABRICATION (QA-adjacent): a healthy kernel (occupancy 1.0, no
//      family, no PTX) -> NO hints at all, limiter "".
//   7. JSON EMISSION (plan spec: "JSON fields for every diagnostic (rule id,
//      evidence refs)"): the rendered JSON parses and carries the rule ids +
//      numeric evidence (Metis: numeric JSON assertions only).
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/diagnostics.h"

#include "kernscope/db.h"
#include "kernscope/profile/enrich.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef PROFILE_FIXTURE_DIR
#error "PROFILE_FIXTURE_DIR must be defined (tests/fixtures/profile)"
#endif
#ifndef KERNSCOPE_ANALYZE_CLI
#error "KERNSCOPE_ANALYZE_CLI must be defined (build/bin/kernscope-analyze)"
#endif

using namespace kernscope::profile;

namespace {

std::string ReadFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

// A minimal EnrichedKernel record (the trace's building block).
EnrichedKernel MakeKernel(const char *name, std::uint64_t duration_ns,
                          std::int64_t grid_x, std::int64_t block_x,
                          std::uint16_t regs = 0) {
  EnrichedKernel ek;
  ek.record.kernel_name = name;
  ek.record.duration_ns = duration_ns;
  ek.record.start_ns = 0;
  ek.record.end_ns = duration_ns;
  ek.record.grid_x = grid_x;
  ek.record.grid_y = 1;
  ek.record.grid_z = 1;
  ek.record.block_x = block_x;
  ek.record.block_y = 1;
  ek.record.block_z = 1;
  ek.record.registers_per_thread = regs;
  return ek;
}

// Enrich through the real todo-50 layer (libs/occupancy behind it).
EnrichedKernel Enrich(EnrichedKernel ek, const char *arch) {
  EnrichOptions opts;
  opts.arch_name = arch;
  return enrich_kernel(ek.record, opts);
}

const EnrichedKernel *FindKernel(const std::vector<EnrichedKernel> &kernels,
                                 const char *name) {
  for (const EnrichedKernel &k : kernels) {
    if (k.record.kernel_name == name) {
      return &k;
    }
  }
  return nullptr;
}

bool HasHint(const KernelDiagnostics &d, const char *id) {
  for (const kernscope::Finding &f : d.hints) {
    if (f.id == id) {
      return true;
    }
  }
  return false;
}

const kernscope::Finding *FindHint(const KernelDiagnostics &d,
                                   const char *id) {
  for (const kernscope::Finding &f : d.hints) {
    if (f.id == id) {
      return &f;
    }
  }
  return nullptr;
}

// Parse a machine-readable numeric evidence value back to double (the
// libs/report convention: evidence values are numeric strings).
double EvidenceNumber(const kernscope::Finding &f, const char *key) {
  const auto it = f.evidence.find(key);
  if (it == f.evidence.end()) {
    return std::nan("");
  }
  char *end = nullptr;
  const double d = std::strtod(it->second.c_str(), &end);
  return end != it->second.c_str() ? d : std::nan("");
}

// The synthetic 3-kernel trace (acceptance fixture): one register-bloated
// kernel (168 regs/thread), one layernorm-family kernel, one small kernel.
std::vector<EnrichedKernel> MakeSyntheticTrace() {
  std::vector<EnrichedKernel> out;
  // bloated: 168 regs/thread, block 256 -> register-file-limited, occupancy
  // well below 1 on sm_80 (the brief's register-bloat case).
  out.push_back(Enrich(MakeKernel("bloated_kernel", 300000, 1024, 256, 168),
                       "sm_80"));
  out.push_back(Enrich(MakeKernel("bloated_kernel", 300000, 1024, 256, 168),
                       "sm_80"));
  out.push_back(Enrich(MakeKernel("bloated_kernel", 300000, 1024, 256, 168),
                       "sm_80"));
  // layernorm family: 32 regs/thread -> full occupancy on sm_80.
  out.push_back(Enrich(MakeKernel("layernorm_fused", 200000, 1024, 256, 32),
                       "sm_80"));
  out.push_back(Enrich(MakeKernel("layernorm_fused", 200000, 1024, 256, 32),
                       "sm_80"));
  // small/healthy: 32 regs/thread, one launch.
  out.push_back(Enrich(MakeKernel("tiny_coalesced", 100000, 1024, 256, 32),
                       "sm_80"));
  return out;
}

// Pointer-safe object lookup (db::json::Value has non-explicit constructors;
// same wrapper rationale as the analyze golden test).
const kernscope::db::json::Value *Child(const kernscope::db::json::Value *obj,
                                        const char *key) {
  return (obj != nullptr) ? obj->find(key) : nullptr;
}

} // namespace

//------------------------------------------------------------------------------
// 1. Hot-kernel ranking (acceptance: RANKING ORDER EXACT)
//------------------------------------------------------------------------------

TEST(ProfileDiag, HotRankingOrderExact) {
  const std::vector<EnrichedKernel> trace = MakeSyntheticTrace();
  const std::vector<HotKernelEntry> ranking = rank_hot_kernels(trace, 10);

  ASSERT_EQ(ranking.size(), 3u);

  // Order: bloated (900000 ns) > layernorm_fused (400000) > tiny (100000).
  EXPECT_EQ(ranking[0].kernel_name, "bloated_kernel");
  EXPECT_EQ(ranking[1].kernel_name, "layernorm_fused");
  EXPECT_EQ(ranking[2].kernel_name, "tiny_coalesced");

  EXPECT_EQ(ranking[0].launches, 3u);
  EXPECT_EQ(ranking[0].total_ns, 900000u);
  EXPECT_EQ(ranking[0].median_ns, 300000u);
  EXPECT_EQ(ranking[1].launches, 2u);
  EXPECT_EQ(ranking[1].total_ns, 400000u);
  EXPECT_EQ(ranking[1].median_ns, 200000u);
  EXPECT_EQ(ranking[2].launches, 1u);
  EXPECT_EQ(ranking[2].total_ns, 100000u);
  EXPECT_EQ(ranking[2].median_ns, 100000u);

  // % of wall time (wall = 1,400,000 ns).
  EXPECT_NEAR(ranking[0].pct_wall_time, 64.2857142857, 1e-9);
  EXPECT_NEAR(ranking[1].pct_wall_time, 28.5714285714, 1e-9);
  EXPECT_NEAR(ranking[2].pct_wall_time, 7.1428571428, 1e-9);

  // top_n truncation is deterministic too.
  const std::vector<HotKernelEntry> top1 = rank_hot_kernels(trace, 1);
  ASSERT_EQ(top1.size(), 1u);
  EXPECT_EQ(top1[0].kernel_name, "bloated_kernel");
}

//------------------------------------------------------------------------------
// 2. Register-bloated kernel: limiter "occupancy" + register_bloat rule
//------------------------------------------------------------------------------

TEST(ProfileDiag, RegisterBloatedGetsOccupancyLimiterAndRegisterBloat) {
  const EnrichedKernel *bloated = FindKernel(MakeSyntheticTrace(),
                                             "bloated_kernel");
  ASSERT_NE(bloated, nullptr);
  // The enrichment must have found a real, low occupancy (register-file
  // limited): the register_bloat rule needs this evidence.
  EXPECT_FALSE(bloated->occupancy_unknown);
  EXPECT_LT(bloated->occupancy, 0.5);
  EXPECT_EQ(bloated->limiter, "registers");

  DiagnosticsOptions opts;
  opts.arch_name = "sm_80";
  opts.analyzer_cli = ""; // no PTX in this fixture: the merge must not run
  const KernelDiagnostics diag = diagnose_kernel(*bloated, "", opts);

  EXPECT_EQ(diag.kernel_name, "bloated_kernel");
  EXPECT_EQ(diag.limiter, "occupancy"); // acceptance: "Likely limiter: occupancy"

  // The occupancy-limiter hint always accompanies the classification.
  ASSERT_TRUE(HasHint(diag, "occupancy_limiter"));

  // acceptance: hint rule id "register_bloat" with evidence regs_per_thread.
  const kernscope::Finding *bloat = FindHint(diag, "register_bloat");
  ASSERT_NE(bloat, nullptr);
  EXPECT_EQ(bloat->severity, "warning");
  EXPECT_EQ(EvidenceNumber(*bloat, "regs_per_thread"), 168.0);
  EXPECT_NEAR(EvidenceNumber(*bloat, "occupancy"), bloated->occupancy, 1e-9);
  EXPECT_EQ(bloat->evidence.at("limiter"), "registers");
  // Every hint carries evidence (the acceptance's MUST NOT: hints without
  // evidence fields).
  ASSERT_FALSE(bloat->evidence.empty());
}

//------------------------------------------------------------------------------
// 3. Memory-bound hint via libs/roofline for a KNOWN family
//------------------------------------------------------------------------------

TEST(ProfileDiag, MemoryBoundHintForKnownFamily) {
  const EnrichedKernel *layernorm = FindKernel(MakeSyntheticTrace(),
                                               "layernorm_fused");
  ASSERT_NE(layernorm, nullptr);

  DiagnosticsOptions opts;
  opts.arch_name = "sm_80";
  opts.analyzer_cli = "";
  const KernelDiagnostics diag = diagnose_kernel(*layernorm, "", opts);

  // Full occupancy -> not occupancy-limited; the roofline rule classifies
  // the layernorm family as memory-bound (AI 11/(2*4) = 1.375 < ridge).
  EXPECT_EQ(diag.limiter, "memory");

  const kernscope::Finding *mem = FindHint(diag, "memory_bound");
  ASSERT_NE(mem, nullptr);
  EXPECT_EQ(mem->evidence.at("family"), "layernorm");
  EXPECT_EQ(mem->evidence.at("bound_class"), "memory");
  // n = 1024 blocks x 256 threads = 262144 elements; layernorm moves
  // 2 * n * 4 bytes (read + write, fp32 default).
  EXPECT_EQ(EvidenceNumber(*mem, "bytes_moved_estimate"), 2097152.0);
  EXPECT_NEAR(EvidenceNumber(*mem, "arithmetic_intensity"), 1.375, 1e-9);
  EXPECT_GT(EvidenceNumber(*mem, "ridge_point"), 0.0);
  ASSERT_FALSE(mem->evidence.empty());
}

//------------------------------------------------------------------------------
// 4. Analyzer merge: real kernscope-analyze invocation on extracted PTX
//------------------------------------------------------------------------------

TEST(ProfileDiag, AnalyzerMergeBankConflictFinding) {
  // The committed fixture: PTX with a 68-word shared stride -> 4-way bank
  // conflict, provided as the "extracted PTX" of a profiled kernel.
  const std::string ptx = ReadFile(std::string(PROFILE_FIXTURE_DIR) +
                                   "/profile_bank_conflict.ptx");
  ASSERT_FALSE(ptx.empty());

  EnrichedKernel ek = MakeKernel("profile_bank_conflict", 150000, 1024, 256,
                                 32);
  ek = Enrich(ek, "sm_86");
  ek.ptx_available = true;

  DiagnosticsOptions opts;
  opts.arch_name = "sm_86";
  opts.analyzer_cli = KERNSCOPE_ANALYZE_CLI;
  const KernelDiagnostics diag = diagnose_kernel(ek, ptx, opts);

  // The merged analyzer finding: id "bank_conflict" + the evidence refs
  // (acceptance: hint includes the analyze finding id "bank_conflict").
  const kernscope::Finding *bank = FindHint(diag, "bank_conflict");
  ASSERT_NE(bank, nullptr)
      << "kernscope-analyze must have been invoked on the fixture PTX";
  EXPECT_EQ(bank->evidence.at("analyze_finding_id"), "bank_conflict");
  EXPECT_EQ(bank->evidence.at("source"), "kernscope-analyze");
  EXPECT_EQ(EvidenceNumber(*bank, "way"), 4.0);
  ASSERT_FALSE(bank->evidence.empty());
}

//------------------------------------------------------------------------------
// 5. QA negative: no PTX + unknown family -> insufficient-evidence only
//------------------------------------------------------------------------------

TEST(ProfileDiag, InsufficientEvidenceDegradation) {
  // No resource evidence (regs 0 -> occupancy unknown), no PTX, no known
  // kernel family: every evidence axis is missing.
  EnrichedKernel ek = MakeKernel("custom_proprietary_kernel", 50000, 100, 256,
                                 0);
  ek.occupancy_unknown = true;
  ek.occupancy = -1.0;
  ek.limiter.clear();
  ek.ptx_available = false;

  DiagnosticsOptions opts;
  opts.arch_name = "sm_80";
  opts.analyzer_cli = "";
  const KernelDiagnostics diag = diagnose_kernel(ek, "", opts);

  EXPECT_EQ(diag.limiter, "insufficient-evidence");
  // Exactly the degradation rule — NO fabricated hints.
  ASSERT_EQ(diag.hints.size(), 1u);
  EXPECT_EQ(diag.hints[0].id, "insufficient_evidence");
  EXPECT_EQ(diag.hints[0].evidence.at("ptx_available"), "false");
  EXPECT_EQ(diag.hints[0].evidence.at("occupancy_known"), "false");
  EXPECT_EQ(diag.hints[0].evidence.at("family"), "unknown");
  EXPECT_NE(diag.hints[0].evidence.at("reason").find("no-ptx"),
            std::string::npos);
  ASSERT_FALSE(diag.hints[0].evidence.empty());
}

//------------------------------------------------------------------------------
// 6. No fabrication: a healthy kernel gets NO hints
//------------------------------------------------------------------------------

TEST(ProfileDiag, HealthyKernelEmitsNoHints) {
  const EnrichedKernel *tiny = FindKernel(MakeSyntheticTrace(),
                                          "tiny_coalesced");
  ASSERT_NE(tiny, nullptr);
  EXPECT_FALSE(tiny->occupancy_unknown);
  EXPECT_NEAR(tiny->occupancy, 1.0, 1e-9);

  DiagnosticsOptions opts;
  opts.arch_name = "sm_80";
  opts.analyzer_cli = "";
  const KernelDiagnostics diag = diagnose_kernel(*tiny, "", opts);

  EXPECT_TRUE(diag.limiter.empty());
  EXPECT_TRUE(diag.hints.empty())
      << "a healthy kernel must not get fabricated hints";
}

//------------------------------------------------------------------------------
// 7. JSON emission: rule ids + evidence refs for every diagnostic
//------------------------------------------------------------------------------

TEST(ProfileDiag, JsonEmissionCarriesRuleIdsAndEvidence) {
  const std::vector<EnrichedKernel> trace = MakeSyntheticTrace();
  const std::vector<HotKernelEntry> ranking = rank_hot_kernels(trace, 10);

  DiagnosticsOptions opts;
  opts.arch_name = "sm_80";
  opts.analyzer_cli = "";

  std::vector<KernelDiagnostics> diags;
  diags.push_back(diagnose_kernel(*FindKernel(trace, "bloated_kernel"), "",
                                  opts));
  EnrichedKernel negative = MakeKernel("custom_kernel", 50000, 100, 256, 0);
  negative.occupancy_unknown = true;
  negative.occupancy = -1.0;
  negative.ptx_available = false;
  diags.push_back(diagnose_kernel(negative, "", opts));

  const std::string json = diagnostics_to_json(ranking, diags);

  // Machine-readable: parse it back and assert numerically (Metis).
  std::string err;
  const std::optional<kernscope::db::json::Value> parsed =
      kernscope::db::json::Value::parse(json, err);
  ASSERT_TRUE(parsed.has_value()) << err;

  const kernscope::db::json::Value *ranking_arr =
      Child(&*parsed, "ranking");
  ASSERT_NE(ranking_arr, nullptr);
  ASSERT_EQ(ranking_arr->kind(), kernscope::db::json::Kind::Array);
  ASSERT_EQ(ranking_arr->as_array().size(), 3u);
  const kernscope::db::json::Value *first = &ranking_arr->as_array()[0];
  EXPECT_EQ(Child(first, "kernel_name")->as_string(), "bloated_kernel");
  // The JSON renders numbers with %.10g (libs/report convention): assert the
  // round-trip within that rendering precision.
  EXPECT_NEAR(Child(first, "pct_wall_time")->as_number(), 64.2857142857,
              1e-6);
  EXPECT_EQ(Child(first, "total_ns")->as_number(), 900000.0);

  const kernscope::db::json::Value *kernels_arr = Child(&*parsed, "kernels");
  ASSERT_NE(kernels_arr, nullptr);
  ASSERT_EQ(kernels_arr->as_array().size(), 2u);
  const kernscope::db::json::Value *bloated = &kernels_arr->as_array()[0];
  EXPECT_EQ(Child(bloated, "limiter")->as_string(), "occupancy");
  const kernscope::db::json::Value *hints = Child(bloated, "hints");
  ASSERT_NE(hints, nullptr);
  bool saw_register_bloat = false;
  for (const kernscope::db::json::Value &h : hints->as_array()) {
    if (Child(&h, "id")->as_string() == "register_bloat") {
      saw_register_bloat = true;
      const kernscope::db::json::Value *evidence = Child(&h, "evidence");
      ASSERT_NE(evidence, nullptr);
      // evidence refs: regs_per_thread is machine-readable numeric.
      const kernscope::db::json::Value *regs = Child(evidence, "regs_per_thread");
      ASSERT_NE(regs, nullptr);
      char *end = nullptr;
      const double v = std::strtod(regs->as_string().c_str(), &end);
      EXPECT_EQ(v, 168.0);
    }
  }
  EXPECT_TRUE(saw_register_bloat);
  const kernscope::db::json::Value *neg = &kernels_arr->as_array()[1];
  EXPECT_EQ(Child(neg, "limiter")->as_string(), "insufficient-evidence");
}
