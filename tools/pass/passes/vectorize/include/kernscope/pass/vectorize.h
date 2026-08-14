// kernscope::pass — vectorized load/store rewrite (todo 37, Wave 7).
//
// VectorizeLoadStorePass rewrites chains of scalar global-memory loads and
// stores in GPU kernels into vector loads/stores (width 2 or 4) when
// contiguity and alignment are provable:
//
//   * contiguity  — consecutive chain addresses differ by exactly the
//     element size, proven with SCEV: the difference of the two pointer
//     SCEVs folds to a compile-time constant (works inside loops, where
//     the base is an induction expression: a[16*i+j] vs a[16*i+j+1]).
//   * alignment   — the chain head must be aligned to the full vector
//     byte width. Proven as min(known alignment of the underlying object,
//     2^minTrailingZeros(SCEV offset of the head)). For a strided loop
//     with stride 16*e (e = element bytes) the offset {0,+,16} has
//     trailing zeros 4 -> 16-byte alignment, exactly the ld.global.v4 /
//     st.global.v4 requirement on NVPTX.
//   * address space — v1 rewrites addrspace(1) (NVPTX global) only;
//     addrspace(3) (shared) vectorization is flagged future work.
//   * aliasing   — a chain is only rewritten when no intervening memory
//     instruction may alias ANY of the merged locations (AAResults-based,
//     mirroring LoadStoreVectorizer's intervening-op discipline). Chains
//     are built per underlying object (getUnderlyingObject), so accesses
//     to distinct objects are never merged even if adjacent.
//
// NVPTX IR in -> NVPTX IR out; the pass is semantics-preserving
// (conservative: anything not provable stays scalar).
//
// Report hook for the todo-42 coalescing-analysis report: run opt with
// -kernscope-vectorize-report to print per-function rewrite counts to
// stderr ("kernscope-vectorize: <fn>: vectorized N loads, M stores
// (K chains)"); alternatively call vectorizeFunction() directly (the
// VectorizeStats struct it returns is the same contract the gtest
// vectors, see passes/vectorize/tests/vectorize_rewrite_test.cpp).

#pragma once

#include <llvm/ADT/ArrayRef.h>
#include <llvm/IR/PassManager.h>

namespace llvm {
class AAResults;
class Function;
class ScalarEvolution;
}  // namespace llvm

namespace kernscope {
namespace pass {

/// Rewrite counters for one function (or one run, when summed).
struct VectorizeStats {
  unsigned loadsVectorized = 0;   ///< scalar loads replaced by vector loads
  unsigned storesVectorized = 0;  ///< scalar stores replaced by vector stores
  unsigned chainsRewritten = 0;   ///< distinct vector ops emitted
};

/// The todo-37 pass: run through opt -passes=kernscope-vectorize or via the
/// PassBuilder extension points (clang -fpass-plugin).
class VectorizeLoadStorePass
    : public llvm::PassInfoMixin<VectorizeLoadStorePass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

/// The transform core, exposed for the unit tests (todo-37 acceptance:
/// rewrite counts asserted on 3 fixtures) and the todo-42 report. Applies
/// the rewrite to F, returns the counts. Requires SE and AA for F already
/// computed (the pass wrapper fetches them from the FunctionAnalysisManager;
/// the tests wire the analyses via PassBuilder + crossRegisterProxies).
VectorizeStats vectorizeFunction(llvm::Function &F, llvm::ScalarEvolution &SE,
                                 llvm::AAResults &AA);

}  // namespace pass
}  // namespace kernscope
