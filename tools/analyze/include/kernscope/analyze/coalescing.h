//===- coalescing.h - Global memory coalescing analysis --------------------===//
//
// Wave-3 todo 15: analyzer module #3. Symbolic per-warp address-pattern
// analysis from the PTX index arithmetic feeding ld.global/st.global:
//   * classify the per-lane address pattern (thread-id-dependent affine);
//   * contiguous-128B / contiguous-partial / strided / random;
//   * sector counts per warp on the 32B-sector / 128B-line model
//     [brief §3.1 L216-238];
//   * coalescing efficiency % = useful_bytes / (sectors x 32);
//   * warning id "coalescing" when efficiency < threshold (pinned in code,
//     cited to the brief).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_COALESCING_H
#define KERNSCOPE_ANALYZE_COALESCING_H

#include "kernscope/amd.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Analyze global-memory coalescing from a parsed PTX module.
void analyze_coalescing(const PtxModule &module, KernelReport &report);

/// AMD overload: global_load_*/buffer_load_*/flat_load_* sector analysis.
void analyze_coalescing(const kernscope::AmdModule &module,
                        KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_COALESCING_H
