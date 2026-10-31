// kernscope::pass — coalescing analysis (todo 42, Wave 7).
//
// IR-level consumer of the todo-15 coalescing classification concept. The
// SAME concept is consumed at PTX level by the static analyzer
// (tools/analyze/src/analyze_coalescing.cpp, Wave-3 todo 15): per-lane
// affine address `base + coeff*lane + const` under the 32B-sector /
// 128B-line model of docs/report/kernscope-technical-report.pdf
// There is NO shared libs/ implementation (verified 2026-08-15: libs/ has
// no coalescing classifier — the analyzer's bank/coalescing math lives in
// tools/analyze/src/analyze_banks.cpp / analyze_coalescing.cpp, same
// plan-vs-actuality note as todo 38's bank-math note); the IR-level
// classifier therefore lives in THIS module, and the IR-vs-PTX split is
// documented honestly in src/coalesce.cpp:
//
//   * IR level (this pass): classification from SCEV on llvm::Value
//     pointers — the lane (threadIdx.x / workitem.id.x) enters the address
//     as the value of the backend tid intrinsic, whose SCEVUnknown term in
//     the pointer SCEV carries the per-lane byte coefficient.
//   * PTX level (todo 15): classification from PTX index arithmetic
//     (mov/add/shl/mul chains on %tid.x) — a different, textual resolver.
//   * Both emit the SAME pattern vocabulary (contiguous-128B |
//     contiguous-partial | strided | unknown-pattern) and the SAME 32B
//     sector model, so reports are comparable across levels.
//
// Classification (per memory op, addrspace(1) global load/store):
//   consecutive-thread stride 1 element   -> fully coalesced
//   uniform/broadcast (no lane term)      -> fully coalesced
//   stride k>1 elements / partial sectors -> partially / uncoalesced
//   unresolvable (non-affine SCEV)        -> unknown (NEVER claimed
//                                            coalesced — same honesty rule
//                                            as the analyzer's
//                                            "coalescing_unknown").
//
// The pass is ANALYSIS ONLY (no transforms): run() returns
// PreservedAnalyses::all(). Report hook (todo-42 report): opt
// -kernscope-coalesce-report prints per-function counts to stderr:
// "kernscope-coalesce: <fn>: N global accesses (F fully, P partial,
// U uncoalesced, K unknown); widest access W B; vectorized V".

#pragma once

#include <llvm/IR/PassManager.h>

#include <string>

namespace llvm {
class Function;
class ScalarEvolution;
class Type;
class Value;
}  // namespace llvm

namespace kernscope {
namespace pass {

/// One global memory op's coalescing class. `pattern` uses the analyzer's
/// vocabulary (tools/analyze/src/analyze_coalescing.cpp: classify_pattern)
/// so IR-level and PTX-level reports name classes identically; `cls` is the
/// todo-42 wording (fully | partial | uncoalesced | unknown).
struct CoalescingClass {
  std::string pattern;            // contiguous-128B | contiguous-partial |
                                  // strided | unknown-pattern
  std::string cls;                // fully | partial | uncoalesced | unknown
  double efficiency = 0.0;        // useful bytes / transferred bytes
  int sectors = 0;                // distinct 32B sectors touched by a warp
  int lane_stride_elements = 0;   // per-lane stride in elements (0 = uniform)
};

/// Per-function classification counts — the todo-42 report contract (the
/// report.json delta fields are derived from these + the sibling passes'
/// VectorizeStats/PromoteStats).
struct CoalesceStats {
  unsigned total = 0;         // global (addrspace 1) load/store ops
  unsigned fully = 0;         // contiguous-128B
  unsigned partial = 0;       // contiguous-partial
  unsigned uncoalesced = 0;   // strided
  unsigned unknown = 0;       // unknown-pattern
  unsigned vectorized = 0;    // accesses wider than one element
  unsigned maxWidthBytes = 0; // widest access byte width
};

/// The todo-42 analysis pass (new-PM shape; registered in src/coalesce.cpp
/// as `kernscope-coalesce`).
class CoalescePass : public llvm::PassInfoMixin<CoalescePass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

/// NVPTX generic-address-space canonicalization (src/as1.cpp, registered as
/// `kernscope-as1`). Restores direct AS1/AS3 accesses from clang-21's
/// generic (flat, p0) lowering so the committed v1 passes (AS1/AS3-scoped)
/// can consume real clang IR. Runs ONLY via the explicit pipeline name —
/// deliberately not on the default-pipeline EPs.
class As1CanonPass : public llvm::PassInfoMixin<As1CanonPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

/// Classify ONE global memory access. `Ptr` is the memory pointer, WidthBytes
/// its access width, SE the function's ScalarEvolution, LaneValue the
/// function's threadIdx.x / workitem.id.x intrinsic call (nullptr when the
/// function has none — then every access is uniform w.r.t. lanes, i.e. the
/// analyzer's coeff==0 case). Exposed for the unit test
/// (passes/coalesce/tests/coalesce_classify_test.cpp) which asserts the
/// stride-1/element fully-coalesced and stride-k strided classes on the
/// committed fixtures.
CoalescingClass classifyGlobalAccess(llvm::Value *Ptr, unsigned WidthBytes,
                                     llvm::ScalarEvolution &SE,
                                     llvm::Value *LaneValue);

/// Analyze one function: classify every global load/store, aggregate into
/// CoalesceStats. Exposed for the unit test + the pass wrapper.
CoalesceStats analyzeFunction(llvm::Function &F, llvm::ScalarEvolution &SE);

}  // namespace pass
}  // namespace kernscope
