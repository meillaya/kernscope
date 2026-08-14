// kernscope::pass — kernscope-rle (todo 39). SKELETON: registered pass,
// body lands in the feature commit. Design notes:
// include/kernscope/pass/rle.h.

#include <llvm/IR/Function.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <kernscope/pass/register_pass.h>
#include <kernscope/pass/rle.h>

using namespace llvm;

namespace kernscope {
namespace pass {

PreservedAnalyses RedundantLoadEliminationPass::run(Function &F,
                                                    FunctionAnalysisManager &FAM) {
  // SKELETON: no-op; the MemorySSA + AA barrier-aware walk lands here.
  (void)F;
  (void)FAM;
  return PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseRLEName(StringRef Name, ModulePassManager &MPM,
                  ArrayRef<PassBuilder::PipelineElement>) {
  if (Name == "kernscope-rle") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::RedundantLoadEliminationPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h).
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseRLEName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::RedundantLoadEliminationPass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::RedundantLoadEliminationPass());
      });
});
