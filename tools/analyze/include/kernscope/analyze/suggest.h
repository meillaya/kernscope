//===- suggest.h - Optimization suggestion rules engine -------------------===//
//
// Wave-3 todo 19: the analyzer's suggestion engine. A small rules engine
// that maps the finding set already emitted by todos 13-18 (report.warnings,
// report.limiter, report.metrics) to ACTIONABLE optimization suggestions,
// one suggestion per triggering finding — never generic advice.
//
// Rule table (documented in docs/report/kernscope-technical-report.pdf, one row per rule):
//
//   | trigger (finding + condition)                 | suggestion id       |
//   |-----------------------------------------------|---------------------|
//   | warning bank_conflict, way >= 2               | tile_k_32           |
//   | limiter == "registers" + FMA-heavy mix        | split_accumulator   |
//   | warning divergent_inner_loop                  | hoist_predicate     |
//   | warning coalescing, efficiency < 1.0,         | vectorize_coalescing|
//   |   and NO bank_conflict warning                |                     |
//
// Every suggestion carries:
//   * evidence["text"]            - the one-line human summary (text renderer)
//   * evidence["triggered_by"]    - the triggering finding id (never emitted
//                                   without one — a suggestion with no trigger
//                                   would be generic advice, which is banned)
//   * machine-readable evidence fields per rule (see the header docs below
//     and docs/report/kernscope-technical-report.pdf for the exact derivation formulas).
//
// Ownership contract: the suggestion engine is the SOLE owner of the
// report.suggestions section. It runs LAST (after all six analysis modules)
// and REBUILDS the suggestions list from scratch: informational findings
// tagged severity "info" by the instruction-mix module (e.g.
// no_tensor_core_usage, latency_bottleneck) are not actionable suggestions
// and are dropped here, so the report's suggestions section carries exactly
// the rule-derived, trigger-backed suggestions (the fused_attention golden
// therefore shows the owner's three suggestions and nothing else).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_SUGGEST_H
#define KERNSCOPE_ANALYZE_SUGGEST_H

#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Run the rules engine: inspect the report's warnings / limiter / metrics
/// (and the parsed module where a rule needs CFG detail) and append the
/// triggered suggestions to report.suggestions. Rebuilds the suggestions
/// section first (drops informational non-suggestion findings), so this
/// function MUST run after every analysis module that writes suggestions.
void analyze_suggest(const PtxModule &module, KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_SUGGEST_H
