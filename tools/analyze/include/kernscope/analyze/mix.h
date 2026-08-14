//===- mix.h - Instruction mix + tensor core + latency bottleneck ----------===//
//
// Wave-3 todo 18: analyzer modules #7+#8+#10.
//   * instruction-mix histogram by class (FP32/FP16/INT/LSU/SMEM/TENSOR/
//     CTRL/other) from the parser taxonomy;
//   * tensor-core detection + MAC counting (mma/wgmma/wmma/tcgen05 shapes ->
//     MACs = m*n*k [brief §5.1 L346-350]; MFMA + RDNA3 WMMA on AMD
//     [brief §5.2]) + "no tensor-core usage" info when GEMM-like loops use
//     only FMA;
//   * approximate latency bottleneck: static critical-path estimate per
//     basic-loop body using the published latency tables [brief §8.1] with
//     issue-slot reasoning -> ALU-bound/LSU-bound/latency-bound, dominant
//     class named; every estimate labeled "approximate,
//     microbenchmark-derived" (never cycle-exact).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_MIX_H
#define KERNSCOPE_ANALYZE_MIX_H

#include "kernscope/amd.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Analyze instruction mix + tensor-core usage + latency bottleneck from a
/// parsed PTX module.
void analyze_mix(const PtxModule &module, KernelReport &report);

/// AMD overload: MFMA/WMMA MAC counting + class histogram + latency.
void analyze_mix(const kernscope::AmdModule &module, KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_MIX_H
