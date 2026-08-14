//===- pad_pass.h - shared-memory padding pass (todo 38) ------------------===//
//
// kernscope-pad: transforms addrspace(3) allocations (and internal
// addrspace(3) globals, the clang-lowered form of `__shared__` arrays) whose
// static stride pattern provably causes shared-memory bank conflicts into
// padded layouts using the classic +1 column pad idiom, rewriting every GEP
// access consistently.
//
// Bank model: 32 banks x 4-byte words; bank(word) = (word/4) mod 32; a stride
// is conflict-prone when the per-lane footprint over one warp lands more than
// one lane on the same bank (way > 1). This is the SAME formula the static
// analyzer's todo-16 bank module implements
// (tools/analyze/src/analyze_banks.cpp, compute_banks), cross-checked
// against the research brief §2.1; the smem budget guard reuses libs/arch
// (kernscope/arch.h) constants instead of hardcoding limits.
//
// Applicability gate (all must hold or the pass declines, leaving IR
// unchanged):
//   1. the allocation is an array (possibly nested) in addrspace(3);
//   2. every GEP index on the allocation is statically affine in the lane
//      variable (tid.x / laneid / workitem.id.x) or a constant — a
//      data-dependent index means the conflict is not provable;
//   3. some NON-innermost stride has a provable conflict (the +1 pad on the
//      innermost dimension only rescues row/column strides; an innermost
//      stride conflict would be untouched by the pad and is declined);
//   4. the padded size stays within the arch's smem-per-block budget
//      (libs/arch `smem_per_block_max_bytes`, falling back to
//      `smem_per_sm_bytes` when the arch does not model the opt-in cap).
//
// AMD LDS variant: RDNA3 (gfx1101) is the wave32 model — 32 lanes over the
// same 32-bank x DWORD LDS, so the identical footprint math applies. CDNA
// (gfx90a/gfx942) LDS is wave64; the pass declines with a note rather than
// mis-model 64 lanes (the analyzer's todo-16 AMD path has the same wave32
// scope).
//
// Registration: REGISTER_PASS_CALLBACK (register_pass.h contract, todo 36).
// The pass is a MODULE pass (allocations can be globals shared across
// kernels; the arch resolution reads module flags), so the parsing callback
// and the pipeline-start EP add it straight to the ModulePassManager — the
// createModuleToFunctionPassAdaptor wrapper is not needed. The peephole EP is
// intentionally not registered: re-running allocation rewrites after every
// InstCombine would be both useless and unsafe (module-scope mutation).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PASS_PASSES_PAD_PAD_PASS_H
#define KERNSCOPE_PASS_PASSES_PAD_PAD_PASS_H

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

class SharedMemoryPadPass final
    : public llvm::PassInfoMixin<SharedMemoryPadPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &MAM);
};

}  // namespace pass
}  // namespace kernscope

#endif  // KERNSCOPE_PASS_PASSES_PAD_PAD_PASS_H
