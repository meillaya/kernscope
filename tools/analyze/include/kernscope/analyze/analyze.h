//===- analyze.h - Full-kernel analysis orchestrator ----------------------===//
//
// Wave-3 todo 19: `analyze_ptx_module` runs all six analysis modules (todos
// 13-18) plus the suggestion engine (todo 19) in dependency order and
// produces the COMPLETE KernelReport: kernel name, metrics (occupancy,
// registers_per_thread, instruction mix, ...), main limiter, warnings, and
// suggestions.
//
// Order (each module consumes only what its predecessors already wrote, so
// the sequence is the dependency order):
//   1. analyze_regs       (todo 13)  registers_per_thread + spill metrics
//   2. analyze_occupancy  (todo 14)  occupancy + limiter, from the register
//                                    figure of step 1 + the block size
//   3. analyze_coalescing (todo 15)  coalescing warnings + efficiency
//   4. analyze_banks      (todo 16)  bank-conflict warnings + way
//   5. analyze_divergence (todo 17)  divergence warnings + barrier inventory
//   6. analyze_mix        (todo 18)  instruction mix + tensor + latency
//   7. analyze_suggest    (todo 19)  suggestions, from the full finding set
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_ANALYZE_H
#define KERNSCOPE_ANALYZE_ANALYZE_H

#include <string>

#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Launch/config inputs that are NOT statically present in the PTX text
/// (the analyzer is a static tool; block size, dynamic smem opt-in and the
/// ptxas -v register count come from the launch manifest / build log).
struct AnalyzeInputs {
  std::string kernel_name;       ///< report name override (mangled PTX names
                                 ///< are not user-facing)
  int block_size = 0;            ///< manifest block size (0 = unknown)
  int dynamic_smem_bytes = 0;    ///< extern .shared smem[] opt-in size (0 =
                                 ///< unknown; recorded as a metric only, per
                                 ///< the occupancy golden reconciliation —
                                 ///< the theoretical occupancy for
                                 ///< extern-dynamic kernels is pinned at the
                                 ///< register-limited combo)
  std::string ptxas_v;           ///< ptxas -v resource text when available
                                 ///< (exact register/spill figures)
  std::string arch_override;     ///< use instead of module.target when set
};

/// Run every analysis module + the suggestion engine on a parsed PTX module
/// with default inputs (block size 0: occupancy emits its graceful
/// block_size_unknown diagnostic; provide AnalyzeInputs for a computable
/// occupancy).
KernelReport analyze_ptx_module(const PtxModule &module);

/// Overload with launch-config inputs (manifest-driven runs).
KernelReport analyze_ptx_module(const PtxModule &module,
                                const AnalyzeInputs &inputs);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_ANALYZE_H
