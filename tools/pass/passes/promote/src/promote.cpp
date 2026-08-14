// kernscope::pass — kernscope-promote skeleton (todo 40, Wave 7).
//
// Skeleton stage: the pass is registered (textual name + EP callbacks) and
// runs, but performs no rewrite yet. Design notes:
// include/kernscope/pass/promote.h. The full analysis (SCEV-grouped
// loads -> cooperative block load -> addrspace(3) tile + barrier -> reads
// redirected) lands in the next increment of this file.

#include <llvm/IR/Function.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <kernscope/pass/promote.h>
#include <kernscope/pass/register_pass.h>

using namespace llvm;

namespace kernscope {
namespace pass {

PreservedAnalyses GlobalToSharedPromotePass::run(Function &F,
                                                 FunctionAnalysisManager &) {
  // Skeleton: no rewrite yet.
  (void)F;
  return PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parsePromoteName(StringRef Name, ModulePassManager &MPM,
                      ArrayRef<PassBuilder::PipelineElement>) {
  if (Name == "kernscope-promote") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::GlobalToSharedPromotePass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h). Function-scoped
// transform: textual name + pipeline-start EP + peephole EP (the same set
// as the todo-39 rle module).
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parsePromoteName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::GlobalToSharedPromotePass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::GlobalToSharedPromotePass());
      });
});
