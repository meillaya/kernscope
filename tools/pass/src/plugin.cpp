// kernscope-pass — plugin entry point (todo 36, Wave-7 foundation).
//
// The entire out-of-tree plugin contract is one exported C symbol,
// `llvmGetPassPluginInfo`, returning a llvm::PassPluginLibraryInfo
// (llvm/include/llvm/Passes/PassPlugin.h). opt and clang dlopen the shared
// library and call its RegisterPassBuilderCallbacks member with their own
// llvm::PassBuilder; the callback simply iterates the self-registered
// pass-module callbacks (kernscope::pass::PassRegistry, filled by
// REGISTER_PASS_CALLBACK from each module's own TU).
//
// This file is GENERIC: it never names a specific pass. Todos 37-41 wire
// their transforms through the registry in their own subdirectory, so this
// entry point (and tools/pass/CMakeLists.txt) never needs editing again.
//
// ABI note (research brief §1.2): the APIVersion field only protects the
// struct layout; a plugin must be compiled against the SAME LLVM version
// (+ compatible RTTI/EH settings) as the tool loading it — loading a plugin
// built against another LLVM fails with a diagnostic, never a segfault
// (QA-negative fixture in .omo/evidence/task-36-gpu-kernel-suite-negative.
// txt).

#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <kernscope/pass/register_pass.h>

using namespace llvm;

namespace {

void registerPassBuilderCallbacks(PassBuilder &PB) {
  // Every self-registered pass module wires its passes (textual names +
  // EP callbacks) into this PassBuilder instance, in registration order.
  // See tools/pass/README.md for the module contract.
  for (const auto &callback :
       kernscope::pass::PassRegistry::instance().callbacks()) {
    callback(PB);
  }
}

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "kernscope-pass", LLVM_VERSION_STRING,
          registerPassBuilderCallbacks};
}
