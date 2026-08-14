// kernscope::pass — kernscope-addrsimp: SCEV-based address simplification
// (todo 39, Wave 7).
//
// Three rewrites, framed for GPU address computation:
//
// 1. mul X, 2^k -> shl X, k, when the constant is SCEV-proven to be a
//    power of two (v1: a ConstantInt whose SCEV folds to a SCEVConstant
//    with a power-of-two APInt). nsw/nuw are preserved — for k >= 1 a mul
//    by 2^k and a shl by k overflow identically, so the flags stay
//    truthful. k == 0 (mul X, 1) erases the mul: X is the same value and
//    the flags are trivially satisfied.
//
// 2. GEP folding via llvm::simplifyInstruction (InstSimplify): zero-offset
//    GEPs, folding of constant-indexed chains, etc. — any GEP that
//    simplifies to a different value is replaced.
//
// 3. Loop-invariant address-computation hoisting: GEP / integer-address
//    chains inside a loop whose SCEV is loop-invariant
//    (SE.isLoopInvariant) and whose operands are all defined outside the
//    loop are hoisted to the preheader. v1 whitelist: non-inbounds GEPs
//    and integer arith without nsw/nuw/exact — hoisting an inbounds GEP
//    would need an in-bounds proof and a flag-carrying op could move
//    poison to an earlier point; loads/stores/calls are never hoisted.
//    Nested loops are processed children-first so every instruction lands
//    in the innermost preheader whose operands permit it.
//
// WHAT UPSTREAM ALREADY COVERS (this pass wraps/extends — no
// duplication): InstCombine's Mul->Shl (instcombine rewrites
// mul-by-power-of-2 into shl when the flags permit) and LICM's generic
// hoisting are the scalar single-thread baselines. This pass re-derives
// both under the SCEV + barrier-phase GPU framing the kernel suite needs:
// address computations are the dominant uniform-per-phase cost on
// NVPTX/AMDGPU (uniform datapath), and the mul->shl rewrite feeds the
// register-pressure / instruction-count reports (todo 42).

#pragma once

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

/// kernscope-addrsimp: SCEV-based address simplification (see above).
class AddressSimplificationPass
    : public llvm::PassInfoMixin<AddressSimplificationPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace pass
}  // namespace kernscope
