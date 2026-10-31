// kernscope::pass — HelloPass implementation + self-registration (todo 36).
//
// This translation unit is the reference model for todos 37-41: it defines
// the pass class (hello_pass.h) and registers it with the REGISTER_PASS_
// CALLBACK macro at file scope — NO shared file is edited; the plugin entry
// (src/plugin.cpp) finds the module purely through the global registry.
//
// The registration lambda wires the pass THREE ways, demonstrating the
// full contract (docs/report/kernscope-technical-report.pdf):
//
//   1. registerPipelineParsingCallback  — textual name "kernscope-hello"
//      for `opt -passes=kernscope-hello` (the Bye example pattern).
//   2. registerPipelineStartEPCallback  — runs once at module pipeline
//      start (opt -passes=default<O2> AND clang -fpass-plugin).
//   3. registerPeepholeEPCallback       — function-level, inserted after
//      each InstCombine (clang -fpass-plugin at -O1/O2).
//
// HelloPass is a FUNCTION pass, so the two module-level slots adapt it with
// createModuleToFunctionPassAdaptor (the standard idiom — e.g. the Bye
// example runs `function(bye)` under a module manager the same way).

#include <llvm/IR/Function.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <kernscope/pass/hello_pass.h>
#include <kernscope/pass/register_pass.h>

namespace kernscope {
namespace pass {

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
llvm::PreservedAnalyses HelloPass::run(llvm::Function &F,
                                       llvm::FunctionAnalysisManager & /*unused*/) {
  F.addFnAttr("kernscope.hello");
  return llvm::PreservedAnalyses::none();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseHelloPass(llvm::StringRef Name, llvm::ModulePassManager &MPM,
                    llvm::ArrayRef<llvm::PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-hello") {
    MPM.addPass(
        llvm::createModuleToFunctionPassAdaptor(kernscope::pass::HelloPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (the todo-36 contract — see register_pass.h). The
// callback receives the PassBuilder of the process that loaded the plugin
// (opt or clang) and wires the EP callbacks + the textual name.
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseHelloPass);

  // Pipeline-start EP: whole-module view at the very start of the pipeline
  // (the recommended EP for GPU-memory passes, research brief §1.3).
  PB.registerPipelineStartEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
            kernscope::pass::HelloPass()));
      });

  // Peephole EP: function-level, after each InstCombine.
  PB.registerPeepholeEPCallback(
      [](llvm::FunctionPassManager &FPM, llvm::OptimizationLevel) {
        FPM.addPass(kernscope::pass::HelloPass());
      });
});
