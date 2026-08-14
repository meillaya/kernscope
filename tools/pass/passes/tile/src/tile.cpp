// kernscope::pass — loop tiling (todo 41, Wave 7).
//
// Skeleton stage: the pass is registered (textual name kernscope-tile)
// and runs cleanly on any function, changing nothing. The analysis +
// rewrite (tiled nest + remainder, decline cases, pipeline interplay)
// land with the following increment.
//
// Design: tools/pass/passes/tile/include/kernscope/pass/tile.h.

#include <llvm/Analysis/LoopInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/raw_ostream.h>

#include <kernscope/pass/register_pass.h>
#include <kernscope/pass/tile.h>

using namespace llvm;

namespace kernscope {
namespace pass {

TileStats tileFunction(Function &F, LoopInfo &LI) {
  TileStats Stats;
  (void)F;
  (void)LI;
  return Stats;
}

PreservedAnalyses LoopTilingPass::run(Function &F,
                                      FunctionAnalysisManager &FAM) {
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  TileStats Stats = tileFunction(F, LI);
  if (Stats.loopsTiled == 0) return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseTileName(StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
  if (Name == "kernscope-tile") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::LoopTilingPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h). Textual name +
// pipeline-start EP; NO peephole EP (a loop restructure re-run after every
// InstCombine would be useless — documented in tile.h).
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseTileName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::LoopTilingPass()));
      });
});
