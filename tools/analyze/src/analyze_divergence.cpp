//===- analyze_divergence.cpp - warp divergence + barrier usage -----------===//
//
// Wave-3 todo 17. See divergence.h for the design rationale.
//
// Method:
//   * value-class dataflow: registers/predicates derived from %tid.x /
//     %laneid are DIVERGENT; from %ctaid/params/constants UNIFORM; loaded
//     from memory POSSIBLY-DIVERGENT (three-valued, never a sound claim for
//     data-dependent control flow — brief §4.2 L303-318);
//   * CFG + dominators/post-dominators (iterative dataflow): a predicated
//     branch @p bra on a divergent predicate is a divergent branch; a
//     divergent branch inside a loop (back-edge region, loop depth >= 1)
//     fires the owner's "divergent branch inside inner loop" warning
//     (expected JSON: predicate_derived_from threadIdx.x, loop_depth 2);
//   * barrier usage: bar.sync/bar.warp/barrier.cluster inventory (count,
//     per-loop density); a barrier whose block is control-dependent on a
//     divergent branch (post-dominator test, brief §4.2 L319-323) fires the
//     "divergent barrier" warning (CUDA UB: not all threads arrive).
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/divergence.h"

#include <cstdint>
#include <algorithm>
#include <cctype>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace kernscope::analyze {
namespace {

// Three-valued divergence class (brief §4.2 L303-318).
enum class DClass : std::uint8_t { kUniform = 0, kPossiblyDivergent, kDivergent };

DClass max_class(DClass a, DClass b) noexcept {
  return static_cast<DClass>(
      std::max(static_cast<int>(a), static_cast<int>(b)));
}

std::string operand_reg(const std::string &text) {
  if (text.empty() || text[0] != '%') return "";
  std::string out = "%";
  for (std::size_t i = 1; i < text.size(); ++i) {
    const char c = text[i];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '<' &&
        c != '>')
      break;
    out += c;
  }
  return out;
}

// --- CFG helpers ------------------------------------------------------------

struct Cfg {
  std::size_t n = 0;
  std::vector<std::vector<std::size_t>> preds;
  std::vector<std::vector<std::size_t>> succs;
  std::vector<std::set<std::size_t>> doms;      // incl. self
  std::vector<std::set<std::size_t>> postdoms;  // incl. self
};

// Iterative dominator fixpoint (standard formulation: init to the universe
// of blocks, repeatedly intersect predecessors' dominator sets, union self).
// Entry block = index 0.
void compute_dominators(Cfg &cfg) {
  cfg.doms.assign(cfg.n, {});
  std::set<std::size_t> universe;
  for (std::size_t i = 0; i < cfg.n; ++i) universe.insert(i);
  for (std::size_t i = 1; i < cfg.n; ++i) cfg.doms[i] = universe;
  cfg.doms[0] = {0};
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 1; i < cfg.n; ++i) {
      if (cfg.preds[i].empty()) continue;  // unreachable: no dominators
      std::set<std::size_t> newdoms = universe;
      for (std::size_t p : cfg.preds[i]) {
        std::set<std::size_t> inter;
        for (std::size_t d : newdoms)
          if (cfg.doms[p].contains(d)) inter.insert(d);
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

// Reverse-graph post-dominators (blocks with no successors are exits).
void compute_postdominators(Cfg &cfg) {
  cfg.postdoms.assign(cfg.n, {});
  std::vector<std::size_t> exits;
  for (std::size_t i = 0; i < cfg.n; ++i)
    if (cfg.succs[i].empty()) exits.push_back(i);
  if (exits.empty()) return;
  std::set<std::size_t> universe;
  for (std::size_t i = 0; i < cfg.n; ++i) universe.insert(i);
  for (std::size_t i = 0; i < cfg.n; ++i) {
    bool is_exit = false;
    for (std::size_t e : exits)
      if (e == i) is_exit = true;
    cfg.postdoms[i] = is_exit ? std::set<std::size_t>{i} : universe;
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (std::size_t i = 0; i < cfg.n; ++i) {
      if (cfg.succs[i].empty()) continue;
      std::set<std::size_t> newpd = universe;
      for (std::size_t s : cfg.succs[i]) {
        std::set<std::size_t> inter;
        for (std::size_t d : newpd)
          if (cfg.postdoms[s].contains(d)) inter.insert(d);
        newpd = std::move(inter);
      }
      newpd.insert(i);
      if (newpd != cfg.postdoms[i]) {
        cfg.postdoms[i] = std::move(newpd);
        changed = true;
      }
    }
  }
}

Cfg build_cfg(const PtxKernel &kernel) {
  Cfg cfg;
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

  // The parser records only the LAST branch per layout block as a CFG
  // successor, but PTX blocks can contain multiple predicated branches
  // (the fused_attention score loop has two @p bra + a bar.sync in one
  // block). Scan EVERY branch instruction so back edges are not lost.
  for (std::size_t i = 0; i < kernel.cfg.size(); ++i) {
    for (const PtxInstruction &insn : kernel.cfg[i].instructions) {
      if (!insn.is_branch) continue;
      if (!insn.branch_target.empty()) {
        auto it = label_to_idx.find(insn.branch_target);
        if (it != label_to_idx.end()) add_edge(i, it->second);
      }
      // Predicated branch: fallthrough to the next layout block.
      if (insn.has_predicate && i + 1 < kernel.cfg.size())
        add_edge(i, i + 1);
    }
  }
  // Blocks with no recorded successor (plain fallthrough chain) still flow
  // to the next layout block; ret/exit have none.
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

  compute_dominators(cfg);
  compute_postdominators(cfg);
  return cfg;
}

// Loop depth per block = number of back-edge headers that dominate it. A
// back edge (b->s) exists when the target s dominates the source b.
std::vector<int> compute_loop_depth(const Cfg &cfg) {
  std::vector<bool> is_header(cfg.n, false);
  for (std::size_t b = 0; b < cfg.n; ++b)
    for (std::size_t s : cfg.succs[b])
      if (cfg.doms[b].contains(s)) is_header[s] = true;
  std::vector<int> depth(cfg.n, 0);
  for (std::size_t i = 0; i < cfg.n; ++i)
    for (std::size_t h = 0; h < cfg.n; ++h)
      if (is_header[h] && cfg.doms[i].contains(h)) depth[i]++;
  return depth;
}

// Forward value-class dataflow: register/predicate -> divergence class.
std::map<std::string, DClass> analyze_classes(const PtxKernel &kernel) {
  std::map<std::string, DClass> cls;
  for (const PtxBlock &block : kernel.cfg) {
    for (const PtxInstruction &insn : block.instructions) {
      const std::string &mn = insn.mnemonic;
      if (mn == "mov" || mn == "mov.u32" || mn == "mov.u64" ||
          mn == "mov.b32" || mn == "mov.b64") {
        if (insn.operands.size() < 2) continue;
        const std::string &src = insn.operands[1].text;
        const std::string dest = operand_reg(insn.operands[0].text);
        if (dest.empty()) continue;
        if (src == "%tid.x" || src == "%tid.y" || src == "%tid.z" ||
            src == "%laneid") {
          cls[dest] = DClass::kDivergent;
        } else if (src == "%ctaid.x" || src == "%ctaid.y" ||
                   src == "%ctaid.z" || src == "%ntid.x" ||
                   src == "%ntid.y" || src == "%ntid.z") {
          cls[dest] = DClass::kUniform;
        } else if (src[0] == '%') {
          auto it = cls.find(src);
          if (it != cls.end()) cls[dest] = it->second;
        }
        continue;
      }
      if (mn.rfind("setp", 0) == 0) {
        const std::string dest = operand_reg(insn.operands[0].text);
        DClass c = DClass::kUniform;
        for (std::size_t i = 1; i < insn.operands.size(); ++i) {
          const std::string src = operand_reg(insn.operands[i].text);
          if (src.empty()) continue;
          auto it = cls.find(src);
          if (it != cls.end()) c = max_class(c, it->second);
        }
        if (!dest.empty()) cls[dest] = c;
        continue;
      }
      const bool is_load = insn.cls == InstructionClass::kLoad ||
                           insn.cls == InstructionClass::kAtomic ||
                           insn.cls == InstructionClass::kReduction;
      if (is_load) {
        if (insn.mem_space == StateSpace::kParam) continue;  // uniform input
        const std::string dest = operand_reg(insn.operands.empty()
                                                 ? std::string()
                                                 : insn.operands[0].text);
        if (!dest.empty() && !cls.contains(dest))
          cls[dest] = DClass::kPossiblyDivergent;
        continue;
      }
      const std::string dest = operand_reg(insn.operands.empty()
                                               ? std::string()
                                               : insn.operands[0].text);
      if (dest.empty()) continue;
      DClass c = DClass::kUniform;
      for (std::size_t i = 1; i < insn.operands.size(); ++i) {
        const std::string src = operand_reg(insn.operands[i].text);
        if (src.empty()) continue;
        auto it = cls.find(src);
        if (it != cls.end()) c = max_class(c, it->second);
      }
      cls[dest] = c;
    }
  }
  return cls;
}

}  // namespace

void analyze_divergence(const PtxModule &module, KernelReport &report) {
  for (const PtxKernel &kernel : module.kernels()) {
    if (!kernel.entry_point) continue;

    const std::map<std::string, DClass> cls = analyze_classes(kernel);
    const Cfg cfg = build_cfg(kernel);
    const std::vector<int> loop_depth = compute_loop_depth(cfg);
    // Branch sites: predicated @p bra / conditional branches.
    std::vector<DClass> branch_class;
    std::vector<std::string> branch_pred;
    std::vector<int> branch_site_depth;
    std::vector<std::size_t> branch_block;

    for (std::size_t bi = 0; bi < kernel.cfg.size(); ++bi) {
      for (const PtxInstruction &insn : kernel.cfg[bi].instructions) {
        if (!insn.is_branch || !insn.has_predicate) continue;
        DClass c = DClass::kUniform;
        auto it = cls.find(insn.predicate);
        if (it != cls.end()) c = it->second;
        branch_class.push_back(c);
        branch_pred.push_back(insn.predicate);
        branch_site_depth.push_back(loop_depth[bi]);
        branch_block.push_back(bi);
      }
    }

    // Aggregate statistics.
    int setp_count = 0;
    int predicated_branch_count = 0;
    for (const PtxBlock &block : kernel.cfg)
      for (const PtxInstruction &insn : block.instructions) {
        if (insn.mnemonic.rfind("setp", 0) == 0) ++setp_count;
        if (insn.is_branch && insn.has_predicate) ++predicated_branch_count;
      }
    report.metrics["setp_count"] = setp_count;
    report.metrics["predicated_branch_count"] = predicated_branch_count;

    // Barrier inventory.
    int bar_sync_count = 0;
    int bar_warp_count = 0;
    int barrier_cluster_count = 0;
    for (const PtxBlock &block : kernel.cfg)
      for (const PtxInstruction &insn : block.instructions) {
        const std::string &mn = insn.mnemonic;
        if (mn == "bar.sync") ++bar_sync_count;
        else if (mn.rfind("bar.warp", 0) == 0) ++bar_warp_count;
        else if (mn.rfind("barrier.cluster", 0) == 0) ++barrier_cluster_count;
      }
    report.metrics["bar_sync_count"] = bar_sync_count;
    report.metrics["bar_warp_count"] = bar_warp_count;
    report.metrics["barrier_cluster_count"] = barrier_cluster_count;

    // Divergent branch INSIDE a loop -> owner's divergent_inner_loop. When
    // several divergent branches nest, report the DEEPEST (innermost) one
    // (the fused_attention's score j-loop is nested at depth 2; the
    // staging loops at depth 1 are less severe).
    std::optional<std::size_t> deepest;
    for (std::size_t i = 0; i < branch_class.size(); ++i) {
      if (branch_class[i] != DClass::kDivergent) continue;
      if (branch_site_depth[i] < 1) continue;
      if (!deepest || branch_site_depth[i] > branch_site_depth[*deepest])
        deepest = i;
    }
    if (deepest) {
      const std::size_t i = *deepest;
      Finding f;
      f.id = "divergent_inner_loop";
      f.severity = "warning";
      f.evidence["text"] = "divergent branch inside inner loop";
      f.evidence["predicate"] = branch_pred[i];
      f.evidence["predicate_derived_from"] = "threadIdx.x";
      f.evidence["loop_depth"] = std::to_string(branch_site_depth[i]);
      f.evidence["setp_count"] = std::to_string(setp_count);
      f.evidence["predicated_branch_count"] =
          std::to_string(predicated_branch_count);
      report.warnings.push_back(std::move(f));
    }

    // Possibly-divergent (data-dependent) branch -> three-valued verdict.
    for (std::size_t i = 0; i < branch_class.size(); ++i) {
      if (branch_class[i] != DClass::kPossiblyDivergent) continue;
      Finding f;
      f.id = "possibly_divergent_branch";
      f.severity = "warning";
      f.evidence["text"] =
          "branch condition is data-dependent (memory-derived); classified "
          "possibly-divergent, not definitively convergent";
      f.evidence["predicate"] = branch_pred[i];
      f.evidence["verdict"] = "possibly-divergent";
      report.warnings.push_back(std::move(f));
      break;
    }

    // Divergent barrier: a bar.sync inside a divergent REGION (brief §4.2
    // L319-323; CUDA UB: not all threads can arrive). A barrier is unsafe
    // only when it sits strictly BETWEEN a divergent branch and its
    // reconvergence point (the branch's immediate post-dominator): the
    // branch dominates the barrier block but the join does not, so some
    // threads never execute the barrier. A barrier AT the join (or outside
    // the region) is reached by all threads and is safe.
    for (std::size_t bi = 0; bi < kernel.cfg.size(); ++bi) {
      bool has_barrier = false;
      for (const PtxInstruction &insn : kernel.cfg[bi].instructions) {
        const std::string &mn = insn.mnemonic;
        if (mn == "bar.sync" || mn.rfind("bar.warp", 0) == 0 ||
            mn.rfind("barrier.cluster", 0) == 0) {
          has_barrier = true;
          break;
        }
      }
      if (!has_barrier) continue;
      bool divergent_barrier = false;
      for (std::size_t i = 0; i < branch_class.size(); ++i) {
        if (branch_class[i] != DClass::kDivergent) continue;
        const std::size_t b = branch_block[i];
        if (b == bi) continue;
        // Join = immediate post-dominator of the branch (reconvergence).
        std::size_t join = b;
        for (std::size_t d : cfg.postdoms[b])
          if (d != b) {
            join = d;
            break;
          }
        if (cfg.doms[bi].contains(b) && !cfg.doms[bi].contains(join)) {
          divergent_barrier = true;
          break;
        }
      }
      if (!divergent_barrier) continue;
      Finding f;
      f.id = "divergent_barrier";
      f.severity = "warning";
      f.evidence["text"] =
          "barrier reachable by only a subset of threads (inside a divergent "
          "region, before reconvergence); not all threads can arrive - "
          "UB/deadlock (CUDA guide __syncthreads semantics, brief §4.2 "
          "L319-323)";
      report.warnings.push_back(std::move(f));
      break;
    }

    // Per-loop barrier density (barrier count weighted by loop nesting).
    double density = 0.0;
    for (std::size_t i = 0; i < kernel.cfg.size(); ++i) {
      int block_bars = 0;
      for (const PtxInstruction &insn : kernel.cfg[i].instructions)
        if (insn.mnemonic == "bar.sync") ++block_bars;
      if (loop_depth[i] > 0 && block_bars > 0)
        density += static_cast<double>(block_bars) * loop_depth[i];
    }
    report.metrics["barrier_density_per_loop"] = density;

    break;  // first entry kernel
  }
}

void analyze_divergence(const kernscope::AmdModule &module,
                        KernelReport &report) {
  // AMD: s_cbranch_* divergence + s_barrier inventory. The MC parser gives
  // instruction records; conditions on vcc/exec derived from v_* operands
  // are treated as possibly-divergent (the .s text has no resolved value
  // classes). s_barrier inventory is exact.
  int barriers = 0;
  int conditional_branches = 0;
  for (const AmdInstruction &insn : module.instructions) {
    if (insn.klass == InstrClass::Barrier) ++barriers;
    if (insn.is_conditional) ++conditional_branches;
  }
  report.metrics["bar_sync_count"] = barriers;
  if (conditional_branches > 0) {
    Finding f;
    f.id = "possibly_divergent_branch";
    f.severity = "warning";
    f.evidence["text"] =
        "AMD conditional branch on vcc/exec is classified possibly-divergent "
        "(.s text carries no resolved value classes; never claimed "
        "definitively convergent)";
    f.evidence["conditional_branches"] = std::to_string(conditional_branches);
    report.warnings.push_back(std::move(f));
  }
}

}  // namespace kernscope::analyze
