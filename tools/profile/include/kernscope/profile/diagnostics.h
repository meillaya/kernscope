//===- diagnostics.h - hot-kernel ranking + compiler-aware diagnostics (todo 51) -*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// The todo-51 layer on top of the todo-50 EnrichedKernel model: (1) HOT-KERNEL
// RANKING — total/median time per kernel across the trace, top-N hot list
// with % of wall time; (2) a DIAGNOSTICS RULES ENGINE emitting libs/report
// Findings ("why is this slow?", Nsight Compute SOL/rules style per
// docs/report/kernscope-technical-report.pdf):
//
//   * occupancy-limited -> limiter "occupancy" + a hint naming the binding
//     resource (register bloat from unrolled accumulators -> "consider split
//     reduction or lower unroll factor" per the owner's layernorm_fused
//     example, docs/report/kernscope-technical-report.pdf Section B); low occupancy +
//     high regs -> maxrregcount/launch_bounds suggestion;
//   * memory-traffic estimate via libs/roofline (todo 12) arithmetic
//     intensity for KNOWN families -> memory-bound hint (the bytes-moved
//     estimate is DERIVED FROM THE RECORD's observed work-item count, never
//     fabricated);
//   * for kernels with PTX available: INVOKE the kernscope-analyze CLI
//     (todo 20) on the extracted PTX and MERGE its findings (bank conflicts,
//     divergence, coalescing) as compiler hints — the profiler<->analyzer
//     cross-tool multiplier;
//   * degradation: a kernel with NO PTX + unknown family + unknown occupancy
//     -> the "insufficient-evidence" rule (asserted in the QA negative);
//     NEVER a hint without evidence fields; NEVER fabricated hints.
//
// MUST NOT require counters/PerfWorks — records + static analysis only.
// MUST NOT modify tools/analyze — the analyzer is INVOKED as a subprocess.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_DIAGNOSTICS_H
#define KERNSCOPE_PROFILE_DIAGNOSTICS_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "kernscope/profile/enrich.h"
#include "kernscope/report.h"

namespace kernscope::profile {

//===--- hot-kernel ranking ---------------------------------------------------===//
//
// One ranking entry: the aggregated profile of one kernel name across the
// whole trace. total_ns = sum of every launch's duration_ns; median_ns = the
// median launch duration; pct_wall_time = total_ns / (sum of ALL kernels'
// totals) * 100 — "wall time" is the sum of kernel execution times observed
// in the trace (documented; deterministic for overlapping launches).
struct HotKernelEntry {
  std::string kernel_name;
  std::size_t launches = 0;      // number of times the kernel ran
  std::uint64_t total_ns = 0;    // sum of durations across launches
  std::uint64_t median_ns = 0;   // median launch duration
  double pct_wall_time = 0.0;    // 0..100, share of total kernel time
};

/// Rank kernels by hotness. Deterministic order (acceptance: exact order
/// asserted): total_ns descending, then median_ns descending, then kernel
/// name ascending. top_n <= 0 or >= kernel count returns every kernel.
std::vector<HotKernelEntry> rank_hot_kernels(
    const std::vector<EnrichedKernel> &kernels, int top_n);

//===--- "why is this slow?" diagnostics --------------------------------------===//
//
// Per-kernel bound classification + compiler hints. `limiter` is the
// SOL-style "likely limiter" (Nsight Compute methodology, brief §6.2):
//
//   "occupancy"            occupancy known and < 1.0 (a resource caps it);
//   "memory"/"latency"/    libs/roofline bound class for a KNOWN family with
//   "compute"              a record-derived shape (brief §6.2.4);
//   "insufficient-evidence"no evidence axis could classify the bound (the
//                          degradation rule — QA negative);
//   ""                     healthy: occupancy 1.0, nothing else to say.
//
// Every hint is a libs/report Finding: stable rule id + severity +
// machine-readable evidence. A hint NEVER fires without evidence fields
// (asserted).
struct KernelDiagnostics {
  std::string kernel_name;
  std::string limiter; // "occupancy"|"memory"|"latency"|"compute"|
                       // "insufficient-evidence"|""
  std::vector<kernscope::Finding> hints;
};

/// Options feeding the rules engine (pure host-side; no GPU required).
struct DiagnosticsOptions {
  std::string arch_name = "sm_80"; // roofline peaks + analyzer --arch key
  std::string analyzer_cli = "kernscope-analyze"; // path to the todo-20 CLI
  int top_n = 10;                  // ranking list size (rank_hot_kernels)
  int register_bloat_threshold = 128; // regs/thread bloat rule (brief §6.3)
};

/// Diagnose ONE enriched kernel. `ptx_text` is the EXTRACTED PTX for kernels
/// with ptx_available == true (the caller extracts it; the caller passes ""
/// when no PTX is available — the analyzer merge then cannot run and the
/// kernel may degrade to "insufficient-evidence"). Never throws: analyzer
/// invocation/parse failures degrade gracefully (the analyzer findings are
/// simply not merged; nothing is fabricated).
KernelDiagnostics diagnose_kernel(const EnrichedKernel &kernel,
                                  const std::string &ptx_text,
                                  const DiagnosticsOptions &options);

/// Render the full diagnostics model as one JSON document:
/// {"schema_version", "report_type", "wall_time_ns", "ranking":[...],
///  "kernels":[{kernel_name, limiter, hints:[{id, severity, evidence}]}]}.
/// Every hint carries its rule id + evidence refs (Metis: numeric JSON
/// assertions only).
std::string diagnostics_to_json(const std::vector<HotKernelEntry> &ranking,
                                const std::vector<KernelDiagnostics> &kernels);

} // namespace kernscope::profile

#endif // KERNSCOPE_PROFILE_DIAGNOSTICS_H
