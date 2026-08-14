// kernscope::pass — kernscope-rle: GPU barrier-aware redundant load
// elimination (todo 39, Wave 7).
//
// Eliminates reloads of invariant locations:
//
//   %a = load T, ptr %p              %a = load T, ptr %p
//   ...                              ...
//   %b = load T, ptr %p        ->    (uses of %b rewritten to %a,
//   ...                               the load erased)
//
// For every candidate load L2 (non-volatile, non-atomic) the pass walks
// the block's MemorySSA ACCESS LIST backward in program order (the
// use-def chain alone never reaches earlier loads — sibling uses hang off
// the same def):
//
//   * MemoryUse   — a plain must-alias load (AA isMustAlias on
//     MemoryLocation::get of both loads) is the redundancy proof; any
//     other use is transparent.
//   * MemoryDef   — if the def's instruction does not may-modify the
//     location (AA getModRefInfo), it is transparent: continue. Otherwise
//     STOP (a may-alias write between the two loads).
//   * Barrier     — a call whose callee carries the `convergent`
//     attribute (llvm.nvvm.barrier0, llvm.ptx.bar.sync,
//     llvm.amdgcn.s.barrier, __syncthreads, ...). Calls with unknown
//     callees (indirect calls, inline asm) are treated as barriers
//     conservatively. To continue PAST a barrier the location must be
//     GLOBALLY INVARIANT (no MemoryDef anywhere in the function
//     may-modifies it, per AA).
//   * MemoryPhi   (head of a block's list) — continue only if the
//     location is globally invariant AND every incoming edge walks to
//     the same must-alias load (a single load dominating all paths to
//     L2).
//   * Volatile/atomic accesses — STOP the walk outright (their ordering
//     semantics are not location-value semantics).
//
// Reaching a plain must-alias load L1 makes L2 redundant. All candidates
// are collected BEFORE any mutation (MemorySSA is stale after the first
// rewrite and must never be re-queried); replacements then erasures are
// applied afterwards.
//
// THE TWO SOUNDNESS RULES (the GPU-specific value of this pass)
// --------------------------------------------------------------
//
// 1. WITHIN a phase (no barrier between the loads): plain AA + MemorySSA
//    suffices and is sound for concurrent threads — on NVPTX/AMDGPU,
//    writes by other threads only become visible at barriers
//    (__syncthreads / bar.sync / s.barrier), so a may-alias-def-free
//    chain between the two loads proves the value is unchanged.
//
// 2. ACROSS a barrier: elimination is sound ONLY for GLOBALLY INVARIANT
//    locations — locations no MemoryDef in the whole function
//    may-modifies: read-only/constant objects, never-escaping allocas,
//    etc. Other threads may write any shared location between barriers,
//    so only a location no thread could write may be reloaded across one.
//
// Never eliminated: volatile loads, atomic loads, and loads whose walk
// crosses a volatile/atomic access — those ops are MemoryDefs and
// additionally block the walk explicitly (their ordering semantics are
// not location-value semantics).

#pragma once

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

/// kernscope-rle: barrier-aware redundant load elimination (see above).
class RedundantLoadEliminationPass
    : public llvm::PassInfoMixin<RedundantLoadEliminationPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace pass
}  // namespace kernscope
