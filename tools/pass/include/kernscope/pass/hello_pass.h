// kernscope::pass — HelloPass (todo 36 smoke pass).
//
// The "hello world" of the pass track: a trivial function pass that stamps
// every function it runs on with the marker attribute "kernscope.hello".
// It exists to prove BOTH plugin load paths end-to-end before todos 37-41
// land real transforms:
//
//   * `opt -load-pass-plugin=libkernscope-pass.so -passes=kernscope-hello`
//     (textual name registered via registerPipelineParsingCallback), and
//   * `clang -fpass-plugin=libkernscope-pass.so` (the pass auto-runs via the
//     PassBuilder extension-point callbacks — pipeline-start + peephole —
//     wired in hello_pass.cpp's REGISTER_PASS_CALLBACK lambda).
//
// The attribute is chosen to be inert to every optimization (custom
// function attribute, value = empty), so the marker survives the whole
// pipeline and FileCheck can assert it in the emitted IR. The pass is
// intentionally idempotent: running it via the peephole EP after the
// pipeline-start EP just re-stamps the same attribute.

#pragma once

#include <llvm/IR/PassManager.h>

namespace kernscope {
namespace pass {

/// New-PM pass (PassInfoMixin CRTP, per llvm/docs/WritingAnLLVMNewPMPass.
/// rst): adds the "kernscope.hello" marker attribute to the function it
/// runs on. Returns PreservedAnalyses::none() — attribute changes invalidate
/// function analyses conservatively.
class HelloPass : public llvm::PassInfoMixin<HelloPass> {
 public:
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM);
};

}  // namespace pass
}  // namespace kernscope
