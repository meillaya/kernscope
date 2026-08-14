// kernscope::pass — global-to-shared memory promotion (todo 40, Wave 7).
//
// kernscope-promote detects tiled/loop-wide addrspace(1) (global) read
// patterns that are reused by every thread of the block and rewrites them
// into a cooperative block load into a fresh addrspace(3) (shared) tile,
// followed by a barrier, with the original reads redirected to the tile.
// This is the "scattered global loads -> cooperative block load -> shared
// tile -> vectorized compute" story of docs/research/owner-examples.md.
// There is no upstream LLVM-IR-level equivalent (the IREE
// GPUPromoteMatmulOperands / MLIR gpu MemoryPromotion precedents are
// MLIR-only — research brief §4.7, §5.1).
//
// Soundness rules (the plan's MUST-NOTs):
//
//   (1) The tile covers EVERY element any rewritten load can read: each
//       candidate load's SCEV element offset (relative to its underlying
//       object) must be a compile-time constant c in [0, N), or an
//       {s,+,1}-AddRec in a loop whose constant trip count T bounds it to
//       [s, s+T) in [0, N). The tile extent N = max reachable element.
//       The cooperative loop then loads global[tid], global[tid+bdx], ...
//       for every element of [0, N) — exactly once each, so after the
//       barrier every smem read observes the loaded value.
//
//   (2) AA guard (the cross-iteration-write MUST-NOT): no memory op in the
//       function may-modify the tile range [base, base + N*elemSize)
//       (AAResults getModRefInfo over every instruction, incl. calls). The
//       cooperative load captures the global state once, at function
//       entry; any store the kernel performs to the same range could be
//       observed by a later rewritten read, which would be a miscompile.
//       Declines, IR untouched.
//
//   (3) Budget guard: N*elemSize must fit the arch's per-block smem budget
//       (libs/arch kernscope::get_arch/has_arch — never hardcoded; arch
//       from the "kernscope.promote.arch" fn attribute, else the
//       nvvmarch/amdgpu-arch module flags, else decline).
//
//   (4) Uniform execution: the cooperative load + barrier are inserted at
//       function entry (every thread passes them before any divergence),
//       and the rewritten region must itself be reached by the whole
//       block — the standard CUDA __syncthreads() contract, documented,
//       not verified (verification is the C3 verifier's job).
//
// v1 scope (documented limits): unit-stride contiguous tiles only
// ({s,+,1} element offsets, constant s); strided (k>1) and nested-AddRec
// (2D tiled) patterns are declined. Entry terminator must be an
// unconditional branch. Scalar int/float elements of 8/16/32/64 bits.
//
// Barrier intrinsic by target triple: llvm.nvvm.barrier0 (NVPTX) /
// llvm.amdgcn.s.barrier (AMDGPU). The cooperative loop's stride is the
// blockDim.x intrinsic of the backend (llvm.nvvm.read.ptx.sreg.ntid.x /
// llvm.amdgcn.workgroup.size.x).
//
// Report hook (todo 42): opt -kernscope-promote-report prints
// "kernscope-promote: <fn>: promoted N loads to shared (M-element tile)".
// PromoteStats (below) is the same contract a unit test could assert on.

#pragma once

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

/// Per-function rewrite statistics, returned by promoteFunction() (and used
/// by the todo-42 report hook).
struct PromoteStats {
  unsigned loadsPromoted = 0;   // rewritten reads
  unsigned tileElements = 0;    // elements of the emitted shared tile
  unsigned groupsPromoted = 0;  // distinct (object, element-type) groups
};

/// The pass (new-PM shape; registered in src/promote.cpp).
class GlobalToSharedPromotePass
    : public llvm::PassInfoMixin<GlobalToSharedPromotePass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace pass
}  // namespace kernscope
