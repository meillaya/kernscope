//===- occupancy.h - Occupancy limits + main limiter analysis -------------===//
//
// Wave-3 todo 14: analyzer module #2. Consumes libs/occupancy
// (kernscope::occupancy::compute_occupancy) with the kernel resources:
//   * regs   - from todo 13 (ptxas -v when available, else static proxy),
//              read from report.metrics["registers_per_thread"];
//   * smem   - parsed .shared declarations + the dynamic smem param
//              (fused_attention uses extern .shared smem[] dynamic);
//   * block  - from .reqntid or the fixture manifest.
// Emits the THEORETICAL occupancy (never "achieved" - this is a static
// tool), the MAIN LIMITER (registers|shared|threads|blocks) and per-limiter
// headroom numbers.
//
// Goldens (see analyze_occupancy_test.cpp): fused_attention @ sm_86/64-thread
// -> occupancy 0.375, limiter "registers"; the brief worked example
// (sm_90, 128 regs, 256 threads) -> occupancy 0.25 [brief §1.2 L101-102].
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_OCCUPANCY_H
#define KERNSCOPE_ANALYZE_OCCUPANCY_H

#include "kernscope/arch.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Occupancy analysis inputs that are NOT statically derivable from the PTX
/// text alone (block size from the launch manifest, dynamic smem opt-in).
struct OccupancyInputs {
  int block_size = 0;          ///< .reqntid / manifest block size (0 = unknown)
  int dynamic_smem_bytes = 0;  ///< extern .shared smem[] dynamic size
  std::string arch_override;   ///< use instead of module.target when set
};

/// Analyze theoretical occupancy + main limiter from a parsed PTX module.
void analyze_occupancy(const PtxModule &module, KernelReport &report);

/// Overload with launch-config inputs not present in the PTX text.
void analyze_occupancy(const PtxModule &module, const OccupancyInputs &inputs,
                       KernelReport &report);

/// Direct resource-level entry (tests feed synthetic values, e.g. the brief
/// worked example sm_90/128 regs/256 threads -> 0.25).
void analyze_occupancy(const ArchInfo &arch, int regs_per_thread,
                       int smem_bytes_per_block, int block_size,
                       KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_OCCUPANCY_H
