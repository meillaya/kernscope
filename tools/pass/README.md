# kernscope-pass — the out-of-tree LLVM 21 pass-plugin (todo 36, Wave 7)

Loadable via **both** driver paths:
`opt -load-pass-plugin=...` and `clang -fpass-plugin=...`.

## Layout

```
tools/pass/
  CMakeLists.txt              plugin target + lit wiring + module subdirs
  include/kernscope/pass/
    register_pass.h           THE self-registration contract (below)
    hello_pass.h              HelloPass smoke pass
  src/
    plugin.cpp                llvmGetPassPluginInfo — the generic entry
    register_pass.cpp         the global registry singleton
    hello_pass.cpp            HelloPass + its REGISTER_PASS_CALLBACK
  passes/                     todos 37-41 live here, one dir per module
    {vectorize,pad,rle,promote,tile}/CMakeLists.txt   (stubs)
tests/pass/                   lit suite (check-kernscope)
  lit.cfg.py / lit.site.cfg.py.in
  hello_pass.test             opt -load-pass-plugin smoke
  clang_fpass_plugin.test     clang -fpass-plugin .cu smoke
```

## The self-registration contract (what todos 37-41 implement)

A pass module is a subdirectory of `tools/pass/passes/<name>/` that:

1. Defines its pass class(es) deriving `llvm::PassInfoMixin<PassT>` with
   `run(IRUnitT&, AnalysisManagerT&) -> llvm::PreservedAnalyses`
   (new-PM shape; in-tree references: `llvm/examples/Bye`,
   `llvm/examples/IRTransforms`, `llvm/lib/Transforms/Utils/HelloWorld.cpp`).
2. Registers them with `REGISTER_PASS_CALLBACK(lambda)` at file scope in
   one of its own `.cpp` files (declared in
   `tools/pass/include/kernscope/pass/register_pass.h`). The lambda
   receives the `llvm::PassBuilder` of the process that loaded the plugin
   and wires the module in:

   ```cpp
   REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
     // textual name for `opt -passes=kernscope-<name>` (Bye pattern):
     PB.registerPipelineParsingCallback(
         [](llvm::StringRef Name, llvm::ModulePassManager &MPM,
            llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
           if (Name == "kernscope-<name>") {
             MPM.addPass(llvm::createModuleToFunctionPassAdaptor(MyPass()));
             return true;
           }
           return false;
         });
     // EP callbacks — what makes `clang -fpass-plugin` auto-run the pass.
     // Full EP list: docs/research/gpu-memory-pass-research.md §1.3.
     PB.registerPipelineStartEPCallback(
         [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
           MPM.addPass(llvm::createModuleToFunctionPassAdaptor(MyPass()));
         });
     PB.registerPeepholeEPCallback(
         [](llvm::FunctionPassManager &FPM, llvm::OptimizationLevel) {
           FPM.addPass(MyPass());
         });
   });
   ```

3. Declares its sources in its own `passes/<name>/CMakeLists.txt` (the
   stub is already `add_subdirectory`'d from `tools/pass/CMakeLists.txt`)
   and adds its own lit tests under `tests/pass/`.

**No shared-file edits are ever needed again**: `src/plugin.cpp` stays
generic (it iterates the registry), and `tools/pass/CMakeLists.txt` already
declares all five module subdirectories. That is the todo-36 isolation
contract — the verify track's `REGISTER_CHECK` pattern applied to passes.

## Plugin ABI (read before touching the entry)

- The entire contract is one exported symbol, `llvmGetPassPluginInfo`
  (`llvm::PassPluginLibraryInfo`), defined in `src/plugin.cpp`.
- `APIVersion` only protects the struct layout — the C++ API is NOT
  ABI-stable: **rebuild the plugin for each LLVM release** with the same
  compiler/RTTI/EH settings as the LLVM it loads into (store llvm-21.1.8:
  `LLVM_ENABLE_RTTI=TRUE`, `LLVM_ENABLE_PLUGINS=ON` — no special flags).
  Loading a plugin built against a different LLVM fails with a graceful
  diagnostic, never a segfault (see the todo-36 negative fixture).
- The store `llvm-21.1.8` has the NVPTX target registered, so clang
  `--cuda-gpu-arch=sm_XX` device compiles work out of the box.

## Lit wiring

The suite runs inside the single `check-kernscope` target
(`test/CMakeLists.txt`), site cfg configured in `tools/pass/CMakeLists.txt`
(mirrors `tools/verify/`). The `LLVM_EXTERNAL_LIT` pin to the nix
`.lit-wrapped` python script is handled by `scripts/configure.sh` (tt-flow
README:44-50 pattern) — do not bypass it.

## Building

```sh
devenv shell
bash scripts/configure.sh     # sets LLVM_DIR/MLIR_DIR/LLVM_EXTERNAL_LIT
cmake --build build           # builds build/lib/kernscope-pass.so
cmake --build build --target check-kernscope   # runs all lit suites
```
