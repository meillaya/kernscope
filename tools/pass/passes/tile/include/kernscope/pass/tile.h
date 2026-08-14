// kernscope::pass — loop tiling (todo 41, Wave 7).
//
// kernscope-tile tiles perfectly-nested GPU loops by arch-derived tile
// sizes (libs/arch smem budget + register budget estimates) so that
// todo-40's global-to-shared promotion (kernscope-promote) can consume
// constant-trip-count, unit-stride {s,+,1} windows (docs/research/
// gpu-memory-pass-research.md §4.7 — IREE's GPUTile/GPUTensorTile are the
// production precedent; this is the LLVM-IR-level tiler the §4.9 table
// marks "must write"). The autotuner (docs/research/autotuner-research-
// brief.md) later SEARCHES tile sizes; this pass APPLIES one arch-derived
// size once, deterministically.
//
// Transform: for an innermost loop with constant trip count N and a
// canonical {0,+,1} IV, split the iteration space into full tiles of TS
// iterations plus a remainder loop (SCEV-driven bound handling):
//
//   for (i = 0; i < N; ++i) body(i);
//     ==>
//   for (t = 0; t < N/TS; ++t)          // tile loop (trip N/TS, constant)
//     for (i = t*TS; i < t*TS+TS; ++i)  // full-tile loop (trip TS, constant)
//       body(i);
//   for (r = 0; r < N%TS; ++r)          // remainder loop (trip N%TS,
//     body(Tfull*TS + r);               //  constant) — only when N%TS != 0
//
// The inner full-tile loop has a CONSTANT trip count TS and its IV is an
// {s,+,1}-AddRec — but its start is the (non-constant) tile offset, so
// kernscope-promote's v1 (constant-start unit-stride tiles) declines the
// full-tile loads. The REMAINDER loop, however, has a constant start
// Tfull*TS and a constant trip count N%TS: its loads are exactly the
// {s,+,1} window promote v1 consumes. That is the tiled-fixture design
// rule the pipeline test relies on (tile -> promote -> vectorize).
//
// Non-IV loop-carried phis (e.g. reduction accumulators) are carried
// across tiles by a phi in the tile-loop header fed through an LCSSA phi
// in the tile latch; the remainder loop continues the carried value, and
// the loop exit phis are retargeted to the final carrier. This is what
// makes tiling a reduction loop correct (a naive per-tile reset would
// miscompile).
//
// Eligibility gates (all must hold or the pass declines, IR unchanged):
//   1. innermost loop in simplified form (single preheader/backedge/
//      exit-block/latch — LoopSimplify); single IV with {0,+,1} shape
//      starting at a non-negative constant;
//   2. constant trip count N = SE.getSmallConstantTripCount(L) > 0
//      (SCEV-driven bounds: non-divisible N yields the remainder loop);
//   3. N >= tile size TS (trip count < tile => tiling is a no-op with a
//      negative remainder => decline);
//   4. dependence gate (the plan's MUST-NOT): LoopAccessAnalysis must
//      prove the loop memory-safe (no cross-iteration dependencies —
//      LAI canVectorizeMemory, conservative: any unknown/indirect/
//      loop-carried dependence declines); additionally no convergent
//      (barrier) and no volatile/atomic memory ops in the body (cloning
//      the remainder body must not duplicate side-effecting semantics);
//   5. arch-derived tile size: TS = largest power-of-two in {32, 64, 128,
//      256, 512, 1024} with TS*elemBytes <= smem budget AND TS <=
//      max_regs_per_thread (a whole tile promotable to smem within the
//      per-block budget, and addressable within the per-thread register
//      file — libs/arch kernscope::get_arch/has_arch, never hardcoded).
//      elemBytes = largest scalar element size among the loop's memory
//      accesses (default 4). Tests may pin TS with the
//      "kernscope.tile.size" function attribute; the arch comes from
//      "kernscope.tile.arch" (else the nvvmarch/amdgpu-arch module
//      flags, else decline).
//
// The pass is a FUNCTION pass (per-loop transform) run through the module
// adaptor for the textual pipeline, exactly like kernscope-promote; it
// does NOT register the peephole EP (re-running a loop restructure after
// every InstCombine would be useless), only pipeline-start + textual.
//
// Report hook (todo 42): opt -kernscope-tile-report prints
// "kernscope-tile: <fn>: tiled loop <hdr> (N -> TS tiles of <TS> + R)".
// TileStats (below) is the same contract a unit test could assert on.

#pragma once

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

/// Per-function rewrite statistics, returned by tileFunction() (and used by
/// the todo-42 report hook).
struct TileStats {
  unsigned loopsTiled = 0;    // loops restructured (incl. remainder)
  unsigned fullTiles = 0;     // sum of N/TS over tiled loops
  unsigned remainderIters = 0;  // sum of N%TS over tiled loops
};

/// The pass (new-PM shape; registered in src/tile.cpp).
class LoopTilingPass : public llvm::PassInfoMixin<LoopTilingPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace pass
}  // namespace kernscope
