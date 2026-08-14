//===- analyze_suggest.cpp - optimization suggestion rules engine ---------===//
//
// Wave-3 todo 19. See suggest.h for the design rationale and
// docs/analyzer-rules.md for the rule table.
//
// Each rule:
//   * has an explicit TRIGGER (a finding id + conditions on its evidence or
//     on report.limiter / report.metrics);
//   * fires ONLY when the trigger is present — a suggestion with no
//     triggering finding would be generic advice, which is banned;
//   * carries the triggering finding id in evidence["triggered_by"];
//   * derives machine-readable evidence from the report (never from the
//     golden JSON — the expected fixture is test data only).
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/suggest.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace kernscope::analyze {
namespace {

// ---------------------------------------------------------------------------
// Documented heuristic constants (see docs/analyzer-rules.md for the full
// derivation narrative).
// ---------------------------------------------------------------------------

// "many FMA/accumulator regs" floor for the accumulator-pattern rule. A
// register-limited kernel with fewer FMAs than this has no real accumulator
// loop to fragment.
constexpr long long kMinAccumulatorFma = 16;

// Fixed per-thread working-set registers that are NOT part of the FMA
// accumulator state, subtracted from registers_per_thread to estimate the
// register-resident accumulator fragment: ~8 base/pointer registers,
// ~8 loop counter + bound registers, ~8 predicate/temp registers, ~14
// address/shuffle/scratch registers = 38 (documented static approximation;
// ptxas reuses registers, so this is a heuristic, not a cycle-exact claim).
constexpr int kAccumulatorOverheadRegs = 38;

// Shared-memory word width (bytes) for accumulator/stride arithmetic.
constexpr int kWordBytes = 4;

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

const Finding *find_warning(const KernelReport &report,
                            const std::string &id) {
  for (const Finding &w : report.warnings)
    if (w.id == id) return &w;
  return nullptr;
}

std::string evidence_str(const Finding &f, const std::string &key,
                         const std::string &fallback = "") {
  const auto it = f.evidence.find(key);
  return it != f.evidence.end() ? it->second : fallback;
}

double evidence_double(const Finding &f, const std::string &key,
                       double fallback = 0.0) {
  const std::string v = evidence_str(f, key);
  char *end = nullptr;
  const double d = std::strtod(v.c_str(), &end);
  return end != v.c_str() ? d : fallback;
}

long long evidence_int(const Finding &f, const std::string &key,
                       long long fallback = 0) {
  const std::string v = evidence_str(f, key);
  if (v.empty()) return fallback;
  char *end = nullptr;
  const double d = std::strtod(v.c_str(), &end);
  return end != v.c_str() ? static_cast<long long>(std::llround(d)) : fallback;
}

double metric_double(const KernelReport &report, const std::string &key,
                     double fallback = 0.0) {
  const auto it = report.metrics.find(key);
  return it != report.metrics.end() ? it->second : fallback;
}

long long metric_int(const KernelReport &report, const std::string &key,
                     long long fallback = 0) {
  const auto it = report.metrics.find(key);
  if (it == report.metrics.end()) return fallback;
  return static_cast<long long>(std::llround(it->second));
}

// ---------------------------------------------------------------------------
// RULE 1 — tile_k_32 (bank_conflict, way >= 2)
// ---------------------------------------------------------------------------
//
// Trigger: warning "bank_conflict" whose evidence "way" >= 2.
// Suggestion: pad the shared stride / tile to a conflict-free layout, tiling
// the K dimension by 32 (the owner's "tile K dimension by 32").
//
// Evidence derivation (docs/analyzer-rules.md §tile_k_32):
//   current_tile_k     = stride_words - (stride_words % 32). The bank-conflict
//                        stride is (K_tile + pad) words; the padding is the
//                        stride's residue mod 32 (68 words -> 4 pad -> the
//                        clean K tile is 64). A conflict-free K tile must be
//                        a 32-word multiple, so this residue IS the padding.
//   suggested_tile_k   = 32 (the owner's convention).
//   shared_bytes_current  = metrics["shared_memory_bytes"] (static .shared
//                        size; 0 for an extern-dynamic tile whose size lives
//                        only in the launch config).
//   shared_bytes_suggested = shared_bytes_current * 32 / current_tile_k
//                        (shared footprint scales with the K tile length).
void rule_tile_k_32(const KernelReport &report, std::vector<Finding> &out) {
  const Finding *bank = find_warning(report, "bank_conflict");
  if (bank == nullptr) return;
  if (evidence_int(*bank, "way", 0) < 2) return;

  const long long stride_words = evidence_int(*bank, "stride_words", 0);
  const long long current_tile_k =
      stride_words > 0 ? stride_words - (stride_words % 32) : 0;
  if (current_tile_k < 8) return;  // unresolvable stride: keep the rule quiet

  const long long suggested_tile_k = 32;
  const double shared_current =
      metric_double(report, "shared_memory_bytes", 0.0);
  const double shared_suggested =
      std::round(shared_current * static_cast<double>(suggested_tile_k) /
                 static_cast<double>(current_tile_k));

  Finding f;
  f.id = "tile_k_32";
  f.severity = "suggestion";
  f.evidence["text"] =
      "pad shared stride / tile to conflict-free layout (tile K dimension by "
      "32 when K-stride causes the conflict)";
  f.evidence["triggered_by"] = "bank_conflict";
  f.evidence["current_tile_k"] = std::to_string(current_tile_k);
  f.evidence["suggested_tile_k"] = std::to_string(suggested_tile_k);
  f.evidence["shared_bytes_current"] = std::to_string(
      static_cast<long long>(std::llround(shared_current)));
  f.evidence["shared_bytes_suggested"] =
      std::to_string(static_cast<long long>(std::llround(shared_suggested)));
  out.push_back(std::move(f));
}

// ---------------------------------------------------------------------------
// RULE 2 — split_accumulator (register limiter + accumulator pattern)
// ---------------------------------------------------------------------------
//
// Trigger: report.limiter == "registers" AND the instruction mix shows an
// FMA accumulator loop (metrics["fma_rn_f32"] >= kMinAccumulatorFma). The
// register limiter winning means the kernel's register footprint (usually a
// register-resident accumulator array) caps occupancy.
//
// Evidence derivation (docs/analyzer-rules.md §split_accumulator):
//   registers_per_thread        = metrics["registers_per_thread"].
//   accumulator_elements        = registers_per_thread - 38 (fixed
//                                 non-accumulator working set, see the
//                                 constant comment).
//   accumulator_bytes           = accumulator_elements * 4 (f32).
//   suggested_fragment_elements = accumulator_elements / 4 (split the
//                                 accumulator into 4 fragments).
void rule_split_accumulator(const KernelReport &report,
                            std::vector<Finding> &out) {
  if (report.limiter != "registers") return;
  if (metric_int(report, "fma_rn_f32", 0) < kMinAccumulatorFma) return;

  const long long regs = metric_int(report, "registers_per_thread", 0);
  if (regs <= 0) return;
  const long long accumulator_elements =
      std::max<long long>(8, regs - kAccumulatorOverheadRegs);
  const long long accumulator_bytes = accumulator_elements * kWordBytes;
  const long long suggested_fragment_elements =
      std::max<long long>(4, accumulator_elements / 4);

  Finding f;
  f.id = "split_accumulator";
  f.severity = "suggestion";
  f.evidence["text"] =
      "split accumulator fragment / lower unroll / maxrregcount";
  f.evidence["triggered_by"] = "registers_limiter";
  f.evidence["accumulator_elements"] = std::to_string(accumulator_elements);
  f.evidence["accumulator_bytes"] = std::to_string(accumulator_bytes);
  f.evidence["suggested_fragment_elements"] =
      std::to_string(suggested_fragment_elements);
  f.evidence["registers_per_thread"] = std::to_string(regs);
  out.push_back(std::move(f));
}

// ---------------------------------------------------------------------------
// CFG loop-depth helper for RULE 3 (hoist_predicate)
// ---------------------------------------------------------------------------
//
// A compact dominator/loop-depth pass (same formulation as the divergence
// module, kept local so the suggestion engine stays self-contained): a back
// edge b->s exists when s dominates b; a block's loop depth is the number of
// back-edge headers that dominate it.

struct SuggestCfg {
  std::size_t n = 0;
  std::vector<std::vector<std::size_t>> preds;
  std::vector<std::vector<std::size_t>> succs;
  std::vector<std::set<std::size_t>> doms;
};

void suggest_compute_dominators(SuggestCfg &cfg) {
  std::set<std::size_t> universe;
  for (std::size_t i = 0; i < cfg.n; ++i) universe.insert(i);
  cfg.doms.assign(cfg.n, {});
  for (std::size_t i = 1; i < cfg.n; ++i) cfg.doms[i] = universe;
  cfg.doms[0] = {0};
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 1; i < cfg.n; ++i) {
      if (cfg.preds[i].empty()) continue;
      std::set<std::size_t> newdoms = universe;
      for (std::size_t p : cfg.preds[i]) {
        std::set<std::size_t> inter;
        for (std::size_t d : newdoms)
          if (cfg.doms[p].count(d)) inter.insert(d);
        newdoms = std::move(inter);
      }
      newdoms.insert(i);
      if (newdoms != cfg.doms[i]) {
        cfg.doms[i] = std::move(newdoms);
        changed = true;
      }
    }
  }
}

std::vector<int> suggest_loop_depths(const PtxKernel &kernel) {
  SuggestCfg cfg;
  cfg.n = kernel.cfg.size();
  cfg.preds.resize(cfg.n);
  cfg.succs.resize(cfg.n);
  std::map<std::string, std::size_t> label_to_idx;
  for (std::size_t i = 0; i < kernel.cfg.size(); ++i)
    label_to_idx[kernel.cfg[i].label] = i;

  auto add_edge = [&cfg](std::size_t from, std::size_t to) {
    if (std::find(cfg.succs[from].begin(), cfg.succs[from].end(), to) ==
        cfg.succs[from].end()) {
      cfg.succs[from].push_back(to);
      cfg.preds[to].push_back(from);
    }
  };
  for (std::size_t i = 0; i < kernel.cfg.size(); ++i) {
    for (const PtxInstruction &insn : kernel.cfg[i].instructions) {
      if (!insn.is_branch) continue;
      if (!insn.branch_target.empty()) {
        auto it = label_to_idx.find(insn.branch_target);
        if (it != label_to_idx.end()) add_edge(i, it->second);
      }
      if (insn.has_predicate && i + 1 < kernel.cfg.size())
        add_edge(i, i + 1);
    }
  }
  for (std::size_t i = 0; i < kernel.cfg.size(); ++i) {
    if (cfg.succs[i].empty() && i + 1 < kernel.cfg.size()) {
      bool terminated = false;
      for (const PtxInstruction &insn : kernel.cfg[i].instructions)
        if (insn.cls == InstructionClass::kReturn ||
            insn.cls == InstructionClass::kExit)
          terminated = true;
      if (!terminated) add_edge(i, i + 1);
    }
  }
  suggest_compute_dominators(cfg);

  std::vector<bool> is_header(cfg.n, false);
  for (std::size_t b = 0; b < cfg.n; ++b)
    for (std::size_t s : cfg.succs[b])
      if (cfg.doms[b].count(s)) is_header[s] = true;
  std::vector<int> depth(cfg.n, 0);
  for (std::size_t i = 0; i < cfg.n; ++i)
    for (std::size_t h = 0; h < cfg.n; ++h)
      if (is_header[h] && cfg.doms[i].count(h)) depth[i]++;
  return depth;
}

// Number of predicated branch sites inside loops of depth >= min_depth, from
// the parsed module (a self-contained re-scan; the divergence module does
// not expose per-loop branch-site counts in the report).
int count_predicated_branches_in_loops(const PtxModule &module,
                                       int min_depth) {
  int count = 0;
  for (const PtxKernel &kernel : module.kernels()) {
    if (!kernel.entry_point) continue;
    const std::vector<int> depth = suggest_loop_depths(kernel);
    for (std::size_t bi = 0; bi < kernel.cfg.size(); ++bi) {
      if (depth[bi] < min_depth) continue;
      for (const PtxInstruction &insn : kernel.cfg[bi].instructions)
        if (insn.is_branch && insn.has_predicate) ++count;
    }
    break;  // first entry kernel
  }
  return count;
}

// ---------------------------------------------------------------------------
// RULE 3 — hoist_predicate (divergent branch inside inner loop)
// ---------------------------------------------------------------------------
//
// Trigger: warning "divergent_inner_loop". The divergent predicate is often
// loop-invariant (derived only from threadIdx); moving it out of the loop
// collapses the per-iteration branch sites into one.
//
// Evidence derivation (docs/analyzer-rules.md §hoist_predicate):
//   predicate_depends_on   = "lane" when the warning's predicate_derived_from
//                            is threadIdx.x (the warp-divergent component of
//                            tid); "tid" otherwise.
//   loop_invariant_in      = the enclosing loop descriptor, from the
//                            warning's loop_depth (the module has no source
//                            names; the depth is the stable handle).
//   current_branch_sites   = predicated branch sites inside the divergent
//                            loop (module re-scan, depth >= warning depth).
//   suggested_branch_sites = 1 (a single branch site after hoisting).
void rule_hoist_predicate(const PtxModule &module, const KernelReport &report,
                          std::vector<Finding> &out) {
  const Finding *div = find_warning(report, "divergent_inner_loop");
  if (div == nullptr) return;

  const std::string derived = evidence_str(*div, "predicate_derived_from");
  const std::string depends_on = derived == "threadIdx.x" ? "lane" : "tid";
  const long long loop_depth = evidence_int(*div, "loop_depth", 1);
  const int current_sites =
      count_predicated_branches_in_loops(module, static_cast<int>(loop_depth));

  Finding f;
  f.id = "hoist_predicate";
  f.severity = "suggestion";
  f.evidence["text"] = "move predicate outside loop / restructure to predication";
  f.evidence["triggered_by"] = "divergent_inner_loop";
  f.evidence["predicate_depends_on"] = depends_on;
  f.evidence["loop_invariant_in"] =
      "inner loop at depth " + std::to_string(loop_depth);
  f.evidence["current_branch_sites"] = std::to_string(current_sites);
  f.evidence["suggested_branch_sites"] = "1";
  out.push_back(std::move(f));
}

// ---------------------------------------------------------------------------
// RULE 4 — vectorize_coalescing (coalescing < 1.0, no bank-conflict path)
// ---------------------------------------------------------------------------
//
// Trigger: warning "coalescing" with evidence efficiency < 1.0, AND no
// "bank_conflict" warning in the same report.
//
// Priority policy (docs/analyzer-rules.md §vectorize_coalescing): when a
// kernel ALSO has shared-memory bank conflicts, the smem-layout fix
// (tile_k_32) already addresses the same staging data path, and adding a
// generic vectorize suggestion would be redundant noise — so the rule stays
// silent for kernels whose memory problem is dominated by the shared-memory
// path. It fires for kernels whose ONLY memory issue is scalarized global
// loads (contiguous-partial, not 128B-coalesced).
void rule_vectorize_coalescing(const KernelReport &report,
                               std::vector<Finding> &out) {
  const Finding *co = find_warning(report, "coalescing");
  if (co == nullptr) return;
  if (find_warning(report, "bank_conflict") != nullptr) return;
  const double efficiency = evidence_double(*co, "efficiency", 1.0);
  if (efficiency >= 1.0 - 1e-12) return;

  const long long bytes_transferred =
      evidence_int(*co, "bytes_transferred_per_warp", 0);
  const long long vectorized = evidence_int(*co, "vectorized_ld_global", 0);

  Finding f;
  f.id = "vectorize_coalescing";
  f.severity = "suggestion";
  f.evidence["text"] = "vectorize to 128B contiguous per warp";
  f.evidence["triggered_by"] = "coalescing";
  f.evidence["efficiency"] =
      std::to_string(static_cast<double>(std::floor(efficiency * 1000.0) /
                                         1000.0));
  f.evidence["current_bytes_per_warp"] =
      std::to_string(bytes_transferred > 0 ? bytes_transferred : 0);
  f.evidence["target_bytes_per_warp"] = "128";
  f.evidence["vectorized_ld_global"] = std::to_string(vectorized);
  out.push_back(std::move(f));
}

}  // namespace

void analyze_suggest(const PtxModule &module, KernelReport &report) {
  // The suggestion engine is the sole owner of the suggestions section: drop
  // informational (severity "info") findings from the analysis modules (e.g.
  // the instruction-mix notes) so the report carries exactly the
  // trigger-backed, actionable suggestions (suggest.h ownership contract).
  std::vector<Finding> actionable;
  for (Finding &f : report.suggestions)
    if (f.severity == "suggestion") actionable.push_back(std::move(f));
  report.suggestions = std::move(actionable);

  // Rules evaluate in the owner's suggestion order. Each rule returns early
  // (or pushes nothing) when its trigger is absent — no generic advice.
  rule_tile_k_32(report, report.suggestions);
  rule_split_accumulator(report, report.suggestions);
  rule_hoist_predicate(module, report, report.suggestions);
  rule_vectorize_coalescing(report, report.suggestions);
}

}  // namespace kernscope::analyze
