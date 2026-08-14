// kernscope::pass — kernscope-addrsimp (todo 39). SKELETON: registered
// pass, body lands in the feature commit. Design notes:
// include/kernscope/pass/addrsimp.h.

#include <llvm/IR/Function.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <kernscope/pass/addrsimp.h>
#include <kernscope/pass/register_pass.h>

using namespace llvm;

namespace kernscope {
namespace pass {

PreservedAnalyses AddressSimplificationPass::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {
  // SKELETON: no-op; the SCEV mul->shl / GEP-fold / hoisting body lands
  // here.
  (void)F;
  (void)FAM;
  return PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseAddrSimpName(StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement>) {
  if (Name == "kernscope-addrsimp") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::AddressSimplificationPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h).
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseAddrSimpName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::AddressSimplificationPass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::AddressSimplificationPass());
      });
});
