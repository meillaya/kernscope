//===- banks.h - Shared-memory bank conflict analysis ----------------------===//
//
// Wave-3 todo 16: analyzer module #4. Detects shared-memory access stride
// patterns from ld.shared/st.shared index affine in the thread id and
// computes the bank footprint on the 32-bank x 4B model [brief §2.1
// L158-184] (AMD: LDS model incl. wave32 for RDNA3 [brief §2.2 L186-210]).
// Classifies conflict-free / N-way / broadcast and emits the owner's
// "shared memory stride likely causes N-way bank conflicts" warning with
// stride evidence + a padding suggestion (+1 pad or XOR swizzle).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_BANKS_H
#define KERNSCOPE_ANALYZE_BANKS_H

#include "kernscope/amd.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Analyze shared-memory bank conflicts from a parsed PTX module.
void analyze_banks(const PtxModule &module, KernelReport &report);

/// AMD overload: ds_read_*/ds_write_* on the LDS bank model (wave32 RDNA3).
void analyze_banks(const kernscope::AmdModule &module, KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_BANKS_H
