// kernscope::pass — PASS MODULE REGISTRY API (todo 36, Wave-7 foundation).
// The contract todos 37-41 implement against, mirroring the verify track's
// REGISTER_CHECK pattern (tools/verify/include/kernscope/verify/
// check_plugin.h) so each pass module registers itself from its OWN
// translation unit and no shared file ever needs editing again.
//
// A pass module (tools/pass/passes/<name>/) is a self-contained plugin:
//
//   1. Define its pass class(es) in its own include/ + src/ files, deriving
//      from llvm::PassInfoMixin<PassT> with a
//      run(IRUnitT&, AnalysisManagerT&) returning llvm::PreservedAnalyses
//      (the new-PM shape documented in llvm/docs/
//      WritingAnLLVMNewPMPass.rst).
//   2. Register them with the REGISTER_PASS_CALLBACK macro at file scope in
//      ONE of the module's .cpp files. The lambda receives the llvm::
//      PassBuilder and wires the module's passes in — the canonical shapes:
//
//        // (a) textual name for `opt -passes=kernscope-my-pass`:
//        PB.registerPipelineParsingCallback(
//            [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
//               llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
//              if (Name == "kernscope-my-pass") {
//                MPM.addPass(
//                    llvm::createModuleToFunctionPassAdaptor(MyPass()));
//                return true;
//              }
//              return false;
//            });
//
//        // (b) PassBuilder extension-point (EP) callbacks — these are what
//        //     make `clang -fpass-plugin` and `opt -passes=default<O2>`
//        //     run the pass automatically. The todo-36 scaffold wires
//        //     pipeline-start + peephole:
//        PB.registerPipelineStartEPCallback(
//            [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
//              MPM.addPass(
//                  llvm::createModuleToFunctionPassAdaptor(MyPass()));
//            });
//        PB.registerPeepholeEPCallback(
//            [](llvm::FunctionPassManager &FPM, llvm::OptimizationLevel) {
//              FPM.addPass(MyPass());
//            });
//
//      (Full EP list + ABI-stability caveats: docs/research/
//      gpu-memory-pass-research.md §1.3 — plugins must be rebuilt per LLVM
//      release, same compiler + RTTI/exception settings as the LLVM they
//      load into.)
//   3. That's it. The plugin entry (src/plugin.cpp, llvmGetPassPluginInfo)
//      iterates the global registry at load time and invokes every
//      registered callback with the PassBuilder — no shared-file edits,
//      which is why tools/pass/passes/{vectorize,pad,rle,promote,tile} are
//      pre-declared as stub subdirectories: todos 37-41 only fill their own
//      directory.

#pragma once

#include <functional>
#include <utility>
#include <vector>

namespace llvm {
class PassBuilder;
}  // namespace llvm

namespace kernscope {
namespace pass {

/// A pass-module registration callback: wires the module's passes into the
/// PassBuilder (textual names via registerPipelineParsingCallback, and/or EP
/// callbacks such as registerPipelineStartEPCallback / registerPeepholeEP
/// Callback). Invoked once per plugin load, in registration order.
using PassRegistrationCallback = std::function<void(llvm::PassBuilder&)>;

/// Global registry of registered pass-module callbacks (singleton; lazily
/// constructed, no static-init-order hazards). The plugin entry iterates
/// callbacks() when the plugin is loaded by opt/clang.
class PassRegistry {
 public:
  static PassRegistry& instance();

  /// Appends one callback (called by REGISTER_PASS_CALLBACK).
  void registerCallback(PassRegistrationCallback cb);

  /// All registered callbacks, in registration order.
  const std::vector<PassRegistrationCallback>& callbacks() const;

 private:
  PassRegistry() = default;
  std::vector<PassRegistrationCallback> callbacks_;
};

/// Registrar type used by the macros below (public so they work from any TU;
/// do not construct manually).
class PassRegistrar {
 public:
  explicit PassRegistrar(PassRegistrationCallback cb);
};

}  // namespace pass
}  // namespace kernscope

/// Self-registration: defines a static registrar that appends one callback
/// to the global registry. Place at file scope in the pass module's .cpp:
///
///   REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
///     PB.registerPipelineParsingCallback(/* ... */);
///   });
///
/// No shared-file edits needed — this is the registry contract todos 37-41
/// rely on.
#define REGISTER_PASS_CALLBACK(CallbackLambda)                                \
  /* NOLINTNEXTLINE(misc-use-anonymous-namespace) - macro-expanded registry   \
     entry (each expansion must register at namespace scope) */               \
  static ::kernscope::pass::PassRegistrar kernscope_pass_register_##__COUNTER__(\
      CallbackLambda)
