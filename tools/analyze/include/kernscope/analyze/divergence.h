//===- divergence.h - Warp divergence + barrier usage analysis -------------===//
//
// Wave-3 todo 17: analyzer module #5+#6. Branch-target/post-dominator
// analysis on the PTX CFG with thread-id-dependent condition detection:
//   * setp on tid-derived values -> divergent branch; uniform -> convergent;
//     data-dependent -> "possibly-divergent" (three-valued, never a sound
//     claim for data-dependent control flow);
//   * divergent branches INSIDE loops (back-edge detection) -> the owner's
//     "divergent branch inside inner loop" warning;
//   * barrier usage: bar.sync/bar.warp/barrier.cluster inventory, count,
//     uniformity of reachability (barrier inside a divergent region ->
//     "divergent barrier" warning per CUDA UB), barrier density per loop.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_DIVERGENCE_H
#define KERNSCOPE_ANALYZE_DIVERGENCE_H

#include "kernscope/amd.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Analyze warp divergence + barrier usage from a parsed PTX module.
void analyze_divergence(const PtxModule &module, KernelReport &report);

/// AMD overload: s_cbranch_* divergence + s_barrier inventory.
void analyze_divergence(const kernscope::AmdModule &module,
                        KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_DIVERGENCE_H
