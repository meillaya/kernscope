# Writing LLVM/MLIR Passes for GPU Memory-Access Optimization — Research Brief

**Date:** 2026-08-13 · **Scope:** LLVM/Clang/MLIR (llvm-project `main` @ [`2078da4`](https://github.com/llvm/llvm-project/commit/2078da43e25a4623cab2d0d60decddf709aaea28), post-LLVM-23-branch → 24.0.0git; findings cross-checked against release tag `llvmorg-22.1.8`), Triton @ [`4fa6ce6`](https://github.com/triton-lang/triton/commit/4fa6ce6f86d398cc8e582f869d22096e481ca85c), IREE @ [`b9a8b77`](https://github.com/iree-org/iree/commit/b9a8b77dc5c115f871eba1f61b514b6518a08d8b).
**Release context (verified via GitHub releases API, 2026-08-13):** latest stable branch is **LLVM 22** (`llvmorg-22.1.8`); **LLVM 23** is in release-candidate phase (`llvmorg-23.1.0-rc3`). Everything cited below was verified identical on `llvmorg-22.1.8` where it matters (HelloWorld location, Bye example, plugin API).

Permalink convention used below: `llvm-project@2078da4` unless noted.

---

## 1. LLVM New Pass Manager & out-of-tree pass plugins (LLVM 20/21/22+)

### 1.1 Pass shape
New-PM passes are CRTP classes deriving from `PassInfoMixin<PassT>` with a `run(IRUnitT &, AnalysisManagerT &)` returning `PreservedAnalyses` — documented in [`llvm/docs/WritingAnLLVMNewPMPass.rst`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/WritingAnLLVMNewPMPass.rst) (lines 12–31). The in-tree "hello world" reference is now [`llvm/lib/Transforms/Utils/HelloWorld.cpp`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Utils/HelloWorld.cpp#L14-L18) (the old `llvm/lib/Transforms/HelloNew/` was removed; `HelloWorld` moved to Utils — commit [`f0205446`](https://github.com/llvm/llvm-project/commit/f020544601), "Move HelloWorld to Utils", 2021; verified `HelloNew` is 404 on `llvmorg-22.1.8` too).

### 1.2 Plugin ABI: `PassPluginLibraryInfo`
The entire plugin contract is a single C struct + one exported symbol, defined in [`llvm/include/llvm/Passes/PassPlugin.h#L25-L54`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Passes/PassPlugin.h#L25-L54):

```c
#define LLVM_PLUGIN_API_VERSION 1
struct PassPluginLibraryInfo {
  uint32_t APIVersion;
  const char *PluginName;
  const char *PluginVersion;
  void (*RegisterPassBuilderCallbacks)(PassBuilder &);
};
```

Loading resolves the symbol `llvmGetPassPluginInfo` via `sys::DynamicLibrary` ([`llvm/lib/Passes/PassPlugin.cpp#L27-L39`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/PassPlugin.cpp#L27-L39)).

**ABI-stability caveats (verified from source/docs):**
- The header comment is explicit: the driver checks `APIVersion` against `LLVM_PLUGIN_API_VERSION` and **"a mismatch is an error. The supported version will be incremented for ABI-breaking changes to the `PassPluginLibraryInfo` struct"** ([PassPlugin.h#L25-L33](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Passes/PassPlugin.h#L25-L33)). That only protects the *struct layout* — it does **not** make the C++ API (`PassBuilder`, `PassManager`, analysis classes) ABI-stable. A plugin must be compiled against the *same* LLVM version (and compatible compiler/flags, e.g. `-fno-rtti` if LLVM was built without RTTI — llvm-tutor's CMake enforces this, see §6.3). In practice: **rebuild the plugin per LLVM release; expect source-level churn** (e.g. pass-registration APIs changed between 13→14→…; `HelloNew` removal; `IntrinsicsNVPTX.td` → `IntrinsicsNVVM.td` rename, §2.4).
- Plugins require an LLVM built with plugin support: `LLVM_ENABLE_PLUGINS` (default ON for shared/dylib builds, otherwise follows PIC) — [`llvm/CMakeLists.txt#L1059-L1069`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/CMakeLists.txt#L1059-L1069). lit gates plugin tests on the `plugins` feature ([`llvm/test/lit.cfg.py#L395-L396`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/test/lit.cfg.py#L395-L396)).
- New PM only: `-fpass-plugin` help text says "only with new pass manager" ([Options.td#L3932-L3936](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L3932-L3936)). The legacy PM still exists **only for the codegen pipeline** (WritingAnLLVMNewPMPass.rst warning, lines 12–16) — relevant because the NVPTX/AMDGPU backends run their memory passes there (§4.1).
- Windows note: clang's plugin test marks `UNSUPPORTED: target={{.*windows.*}}` and AIX `XFAIL` ([`clang/test/CodeGen/codegen-plugins.c`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/test/CodeGen/codegen-plugins.c#L5-L8)) — plugin loading on Windows is unreliable.
- LTO: `registerPipelineStartEPCallback` "does not apply to 'backend' compiles (LTO and ThinLTO link-time pipelines)" ([PassBuilder.h#L489-L494](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Passes/PassBuilder.h#L489-L494)); use the dedicated Full-LTO callbacks (§1.3) if you must run under LTO.

### 1.3 PassBuilder extension-point callbacks (complete current list)
From [`llvm/include/llvm/Passes/PassBuilder.h#L417-L575`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Passes/PassBuilder.h#L417-L575):

| Callback | Insertion point | Line |
|---|---|---|
| `registerPeepholeEPCallback` | after each InstCombine | L417 |
| `registerLateLoopOptimizationsEPCallback` | late loop opts, before loop deletion (LoopPassManager) | L431 |
| `registerLoopOptimizerEndEPCallback` | end of loop optimizer | L442 |
| `registerScalarOptimizerLateEPCallback` | after main scalar opts, before cleanup | L451 |
| `registerCGSCCOptimizerLateEPCallback` | end of CGSCC passes | L462 |
| `registerVectorizerStartEPCallback` | before vectorizer / target-specific opts | L473 |
| `registerVectorizerEndEPCallback` | after vectorizer | L486 |
| `registerPipelineStartEPCallback` | once at pipeline start (not LTO backends) | L494 |
| `registerPipelineEarlySimplificationEPCallback` | right after basic simplification | L504 |
| `registerOptimizerEarlyEPCallback` | before function optimization pipeline | L513 |
| `registerOptimizerLastEPCallback` | very end of function optimization pipeline | L523 |
| `registerFullLinkTimeOptimizationEarlyEPCallback` / `...LastEPCallback` | start/end of full-LTO pipeline | L535/L542 |
| `registerParseAACallback` | register custom AA in `AAManager` | L549 |
| `registerAnalysisRegistrationCallback` (×5: CGSCC/Function/Loop/Module/MachineFunction AMs) | register custom analyses | L557-L575 |

Plus `registerPipelineParsingCallback` for textually-named passes runnable via `opt -passes=...` (used by the Bye example, [`llvm/examples/Bye/Bye.cpp#L57-L75`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/examples/Bye/Bye.cpp#L57-L75)).

**Recommended EPs for GPU-memory passes:** `registerPipelineStartEPCallback` (whole-module view; needed to see `addrspace(3)` globals + kernels together) or `registerOptimizerLastEPCallback`; for interop with backend passes, note the backend re-runs its own IR passes later (§4.1).

### 1.4 Tooling entry points
- **opt:** `-load-pass-plugin=<so>` — [`llvm/tools/opt/optdriver.cpp#L285-L287`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/tools/opt/optdriver.cpp#L285-L287); docs: [`llvm/docs/NewPassManager.rst#L183-L186`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/NewPassManager.rst#L183-L186).
- **clang:** `-fpass-plugin=<so>` → forwarded to cc1 ([`clang/lib/Driver/ToolChains/Clang.cpp#L7603-L7606`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/Driver/ToolChains/Clang.cpp#L7603-L7606)), loaded in [`clang/lib/CodeGen/BackendUtil.cpp#L981-L990`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/CodeGen/BackendUtil.cpp#L981-L990) (`PassPlugin::Load` + `registerPassBuilderCallbacks`).
- **CMake:** `add_llvm_pass_plugin(<name> sources...)` for in-tree-style builds; static linking via `LLVM_${NAME}_LINK_INTO_TOOLS=ON` + `HANDLE_EXTENSION` in `llvm/Support/Extension.def` — [WritingAnLLVMNewPMPass.rst#L236-L289](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/WritingAnLLVMNewPMPass.rst#L236-L289).

### 1.5 Key APIs for memory passes (all present in current tree)
- `IRBuilder` — [`llvm/include/llvm/IR/IRBuilder.h`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IRBuilder.h)
- `ScalarEvolution` — [`llvm/include/llvm/Analysis/ScalarEvolution.h`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Analysis/ScalarEvolution.h) (loop trip counts / affine address strides; request via `FunctionAnalysisManager::getResult<ScalarEvolutionAnalysis>`)
- `AAResults` — [`llvm/include/llvm/Analysis/AliasAnalysis.h#L318`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Analysis/AliasAnalysis.h#L318); LoadStoreVectorizer groups by `getUnderlyingObject()` ([LoadStoreVectorizer.cpp#L46](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp#L46)); NVPTX ships its own AA (`NVPTXAliasAnalysis.cpp` in `llvm/lib/Target/NVPTX/`)
- `DominatorTree` — [`llvm/include/llvm/IR/Dominators.h`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/Dominators.h) (used e.g. by the IRTransforms example plugin, [`llvm/examples/IRTransforms/SimplifyCFG.cpp#L383`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/examples/IRTransforms/SimplifyCFG.cpp#L383))
- Also useful: `Loads.h` (`isSafeToLoadUnconditionally`), `ValueTracking.h`, `MemoryLocation` (AliasAnalysis.h).

---

## 2. GPU address spaces in LLVM IR (VERIFIED tables)

### 2.1 NVPTX
Canonical table — [`llvm/docs/NVPTXUsage.rst#L95-L117`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/NVPTXUsage.rst#L95-L117) and the C++ enum [`llvm/include/llvm/Support/NVPTXAddrSpace.h`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Support/NVPTXAddrSpace.h#L21-L31):

| AS | Memory space |
|----|--------------|
| 0 | Generic (default) |
| 1 | Global |
| 2 | Internal use |
| 3 | **Shared** |
| 4 | Constant |
| 5 | Local (per-thread private) |
| 6 | Tensor (TENSOR, newer; not in the docs table, in the header enum) |
| 7 | Shared Cluster (sm_90+ DSMEM) |
| 101 | ENTRY_PARAM (kernel params) |

Your guess (0/1/3/4/5) is correct; note **2 = internal**, **7 = shared::cluster**, **6 = tensor** additions.

### 2.2 AMDGPU
Canonical table — [`llvm/docs/AMDGPUUsage.rst#L836-L866`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/AMDGPUUsage.rst#L836-L866) ("AMDGPU Address Spaces") and [`llvm/include/llvm/Support/AMDGPUAddrSpace.h`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/Support/AMDGPUAddrSpace.h#L28-L46):

| AS | Name | HSA segment | Hardware | NULL |
|----|------|-------------|----------|------|
| 0 | Generic | flat | flat | 0x0 |
| 1 | Global | global | global | 0x0 |
| 2 | Region | N/A | GDS | n/a (not implemented for AMDHSA) |
| 3 | **Local** | group | **LDS** | 0xFFFFFFFF |
| 4 | **Constant** | constant | same as global | 0x0 |
| 5 | **Private** | private | scratch | 0xFFFFFFFF |
| 6 | Constant 32-bit | TODO | | 0x0 |
| 7 | Buffer fat pointer (160-bit, not used by backend) | | | |
| 8 | Buffer resource (128-bit V#) | | | |

**Your guess verified:** 0=flat, 1=global, 3=local/LDS, 4=constant, 5=private — plus 2=GDS, 6=const32, 7/8 buffer pseudospaces. Note LDS pointers are **32-bit** (`p3:32:32` in the amdgcn data layout, [`clang/lib/Basic/Targets/AMDGPU.cpp#L33-L37`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/Basic/Targets/AMDGPU.cpp#L33-L37)) and NULL in AS3/AS5 is `0xFFFFFFFF` — matters for any pass that reasons about nullability/poison across `addrspacecast`.

### 2.3 `addrspacecast` semantics
[`llvm/docs/LangRef.rst#L12608-L12656`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/LangRef.rst#L12608-L12656): may be a no-op or a complex value change depending on target/AS pair; if the result is dereferenceable, both pointers alias the same location; no side effects; no capture; unsupported casts yield **poison**; reversibility guaranteed for integral pointers when result is dereferenceable. This is the hook your "address computation simplification" pass exploits (see InferAddressSpaces, §4.3).

### 2.4 Memory-relevant intrinsics
File note: the NVPTX intrinsic definitions live in **`llvm/include/llvm/IR/IntrinsicsNVVM.td`** (renamed from `IntrinsicsNVPTX.td`; verified the rename predates LLVM 20 — `IntrinsicsNVPTX.td` is 404 on `llvmorg-20.1.0`/`21.1.0`/`22.1.8`).

- **Barriers:** `llvm.nvvm.barrier0` → `llvm.nvvm.barrier.cta.sync.aligned.all(0)`; full barrier family [`IntrinsicsNVVM.td#L131-L136`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsNVVM.td#L131-L136), [`#L1292-L1308`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsNVVM.td#L1292-L1308). Any pass that moves loads/stores across threads must respect these.
- **`ldmatrix` (shared→register for MMA):** [`IntrinsicsNVVM.td#L320-L396`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsNVVM.td#L320-L396) (`llvm.nvvm.ldmatrix.sync.aligned...`).
- **`cp.async` (async global→shared):** `int_nvvm_cp_async_ca_shared_global_{4,8,16}`, `int_nvvm_cp_async_cg_shared_global_16`, `cp_async_commit_group`, `cp_async_wait_group` — [`IntrinsicsNVVM.td#L1354-L1386`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsNVVM.td#L1354-L1386). **These are the natural emission target for a global→shared promotion pass on sm_80+.**
- **AMDGPU:** buffer ops `llvm.amdgcn.raw.buffer.load/store*` ([`IntrinsicsAMDGPU.td#L1372-L1507`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L1372-L1507)), `raw_buffer_load_lds` (global→LDS direct, [#L1893](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L1893)), `global_load_lds` ([#L2690-L2703](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L2690-L2703)), `ds_bpermute` ([#L2652](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/include/llvm/IR/IntrinsicsAMDGPU.td#L2652)). Plain LDS access is just load/store in `addrspace(3)`.

### 2.5 CUDA `__shared__` → `addrspace(3)` global (verified path)
1. Language-level AS enum: `LangAS::cuda_device/cuda_constant/cuda_shared` — [`clang/include/clang/Basic/AddressSpaces.h#L25-L45`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Basic/AddressSpaces.h#L25-L45).
2. CodeGen picks the LangAS from attributes: `GetGlobalVarAddressSpace` — [`clang/lib/CodeGen/CodeGenModule.cpp#L5535-L5571`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/CodeGen/CodeGenModule.cpp#L5535-L5571) (`CUDASharedAttr → LangAS::cuda_shared`, etc.).
3. Target maps LangAS → numeric AS: NVPTX map [`clang/lib/Basic/Targets/NVPTX.h#L27-L56`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/Basic/Targets/NVPTX.h#L27-L56) (`cuda_device→1, cuda_constant→4, cuda_shared→3`); AMDGPU map [`clang/lib/Basic/Targets/AMDGPU.cpp#L39-L66`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/Basic/Targets/AMDGPU.cpp#L39-L66) (same CUDA mapping onto AMDGPUAS). Generic accessor: `TargetInfo::getTargetAddressSpace` ([`clang/include/clang/Basic/TargetInfo.h#L1650-L1655`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Basic/TargetInfo.h#L1650-L1655)).
4. Resulting IR, including **function-local statics becoming internal `addrspace(3)` globals** and implicit `addrspacecast` to generic at use sites: [`clang/test/CodeGenCUDA/address-spaces.cu`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/test/CodeGenCUDA/address-spaces.cu#L1-L45) — the single best reference test for this feature (runs for both `nvptx` and `amdgcn` triples).

---

## 3. Compiling CUDA/HIP to LLVM IR and PTX with clang (verified flags)

### 3.1 CUDA driver flags (clang)
All from [`clang/include/clang/Driver/Options.td`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td):

| Flag | Meaning | Line |
|---|---|---|
| `-x cuda` | treat input as CUDA (else inferred from `.cu`) | (language option) |
| `--cuda-gpu-arch=sm_XX` | alias of `--offload-arch=` | [L1279-L1280](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1279-L1280), [L1153-L1157](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1153-L1157) |
| `--cuda-path=<dir>` | CUDA toolkit location | [L1300-L1301](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1300-L1301) |
| `--cuda-device-only` / `--offload-device-only` | compile device side only (→ device IR/PTX, no host stubs) | [L1286](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1286), [L1171](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1171) |
| `--cuda-host-only` / `--offload-host-only` | host side only | [L1288](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1288) |
| `--no-cuda-version-check` | don't error on newer/older toolkit | [L1296-L1298](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L1296-L1298) |
| `-nocudainc` | alias of `-no-offload-inc` (skip CUDA headers/libdevice includes) | [L5856](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L5856) |
| `-nocudalib` | alias of `-no-offload-lib` (skip CUDA runtime/libs) | [L5868](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/include/clang/Driver/Options.td#L5868) |
| `-fgpu-rdc` | relocatable device code (separate device compilation; needed for cross-TU device globals) | (Options.td `fgpu_rdc`) |

Device triple is `nvptx64-nvidia-cuda` (or `nvptx-` for 32-bit), constructed in [`clang/lib/Driver/Driver.cpp#L136`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/lib/Driver/Driver.cpp#L136). Official usage doc: [`llvm/docs/CompileCudaWithLLVM.rst`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/CompileCudaWithLLVM.rst) (invocation examples lines 44–80; notes the CUDA 7.0–12.1 support window, lines 24–26).

**Recipes for the deliverables:**
```bash
# 1) CUDA C++ -> device LLVM IR (no toolkit needed for pure-IR kernels):
clang++ -x cuda --cuda-device-only --offload-arch=sm_80 \
        -nocudainc -nocudalib --no-cuda-version-check \
        -O2 -S -emit-llvm kernel.cu -o kernel.ll
# (with a real toolkit: drop -nocudainc/-nocudalib, add --cuda-path=/usr/local/cuda)

# cc1-level equivalent (what tests use, cf. address-spaces.cu):
clang -cc1 -emit-llvm -fcuda-is-device -triple nvptx64-nvidia-cuda kernel.cu -o -

# 2) IR -> PTX:
llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 kernel.ll -o kernel.ptx
#    (NVPTXUsage.rst shows `llc -mcpu=sm_20 kernel.ll -o kernel.ptx`, lines 2165-2176)

# 3) Full libdevice flow (NVPTXUsage.rst lines 2554-2563):
llvm-link kernel.bc libdevice.compute_XX.bc -o linked.bc
opt -internalize -internalize-public-api-list=kernel -nvvm-reflect -O3 linked.bc -o opt.bc
llc -mcpu=sm_80 opt.bc -o kernel.ptx
```

### 3.2 HIP equivalents
From [`clang/docs/HIPSupport.rst`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/docs/HIPSupport.rst#L56-L92):
```bash
clang++ -x hip --offload-arch=gfx906 sample.cpp -c -o sample.o   # or name file .hip
clang++ --offload-arch=native -xhip sample.cpp -o sample
```
- HIP reuses the CUDA offloading driver machinery (`--offload-arch`, `--offload-device-only`, `-nogpulib/-nogpuinc` are the generic spellings); `--hip-path=<path>` and `--hip-device-lib-path=<path>` locate the HIP/ROCm device libs ([HIPSupport.rst#L105-L135](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/docs/HIPSupport.rst#L105-L135)).
- Device IR → AMDGCN: `llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx906 kernel.ll -o kernel.s` (same pattern; AMDGPUUsage.rst is the reference).
- `__shared__`/`__constant__` map identically on AMDGPU (§2.5) — `address-spaces.cu` runs both triples.

---

## 4. Existing passes & tools doing similar transforms

### 4.1 What the GPU backends ALREADY run (legacy PM, codegen pipeline) — don't reinvent
[`llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L300-L400):
- `addEarlyCSEOrGVNPass` (GVN at `-O3`, else EarlyCSE) — [L300-L305](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L300-L305)
- `addAddressSpaceInferencePasses`: SROA → NVPTXLowerAlloca → **InferAddressSpaces** → NVPTXAtomicLower — [L307-L316](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L307-L316) (comment: "TODO: Consider running InferAddressSpaces during opt, earlier in the compilation flow" — exactly the gap a mid-end plugin can fill)
- `addStraightLineScalarOptimizationPasses`: SeparateConstOffsetFromGEP → SpeculativeExecution → StraightLineStrengthReduce → GVN/EarlyCSE → NaryReassociate → EarlyCSE — [L318-L333](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L318-L333) (this is upstream's "address computation simplification" stack)
- `addIRPasses`: NVPTXImageOptimizer, **GenericToNVVM** (moves const-qualified globals to `.const`/`.global` sections, [NVPTXGenericToNVVM.cpp#L1-L12](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXGenericToNVVM.cpp#L1-L12)), NVPTXLowerArgs, then LSR, GVN/EarlyCSE again, **LoadStoreVectorizer** (gated by `-disable-nvptx-load-store-vectorizer`, [L50](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L50) and [L396-L397](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L396-L397)), SROA, NVPTXTagInvariantLoads — [L335-L400](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetMachine.cpp#L335-L400)
- NVPTXAllocaHoisting hoists non-entry `alloca`s to entry (local memory layout) — [NVPTXAllocaHoisting.cpp#L1-L11](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXAllocaHoisting.cpp#L1-L11)

AMDGPU similarly runs **LoadStoreVectorizer** ([`AMDGPUTargetMachine.cpp#L1383`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L1383)) and **InferAddressSpaces** in its new-PM codegen pipeline ([L819, L827, L883, L931](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp#L819)).

### 4.2 LoadStoreVectorizer — status: ALIVE, GPU-motivated, legacy-PM only
[`llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp#L1-L60`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Vectorize/LoadStoreVectorizer.cpp#L1-L60): merges loads/stores of sequential addresses into vector loads/stores; "motivated by the microarchitectural quirks of nVidia and AMD GPUs" (free element extraction ⇒ always beneficial); runs **late**, after other vectorization; O(n²) chain-building per BB; groups by `getUnderlyingObject()` + element size; respects intervening may-alias ops. This is upstream's "vectorized load/store rewrite" — **reusable as-is in the backend pipeline; if you need it mid-end/new-PM you must port it (it is a legacy `FunctionPass`).**

### 4.3 InferAddressSpaces — the existing "address computation simplification"
[`llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp#L1-L35`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Scalar/InferAddressSpaces.cpp#L1-L35): propagates specific address spaces through generic-pointer expressions so `ld.shared` (≈10% faster than generic `ld` on K40c) can be emitted; explicitly motivated by CUDA/clang's conservative `addrspacecast`-to-generic pattern. New-PM pass (`InferAddressSpacesPass`), parameterized by flat AS — **directly reusable**; your plugin can run it earlier than the backend does.

### 4.4 Redundant load elimination
- **EarlyCSE** (`llvm/lib/Transforms/Scalar/EarlyCSE.cpp`) and **GVN** (`llvm/lib/Transforms/Scalar/GVN.cpp`) are the workhorses; NVPTX already runs both in codegen (§4.1). For memory ops they use `MemoryLocation`/AA — works across `addrspacecast` only as well as your AA lets it.
- **NewGVN** ([`llvm/lib/Transforms/Scalar/NewGVN.cpp#L1-L30`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Scalar/NewGVN.cpp#L1-L30)): sparse predicated GVN (Gargi, "A Sparse Algorithm for Predicated Global Value Numbering"), "mostly-complete" redundant-expression detection — stronger than GVN on some patterns but still treated as experimental (not in default pipelines). Available as `-passes=newgvn` for experiments.
- NVPTXTagInvariantLoads (in `llvm/lib/Target/NVPTX/`) is a GPU-specific redundant/uniform-load pass worth reading.

### 4.5 LoopVectorize on GPU targets
- LoopVectorize is part of the standard middle-end pipeline ([`llvm/lib/Passes/PassBuilderPipelines.cpp#L1301-L1302`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/PassBuilderPipelines.cpp#L1301-L1302), gated by `PTO.LoopVectorization` / `-fvectorize` / cl::opt `vectorize-loops` default true, [`LoopVectorize.cpp#L384-L385`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Vectorize/LoopVectorize.cpp#L384-L385)), so it *does* run on NVPTX device IR at -O2/-O3.
- NVPTX TTI deliberately keeps it enabled but neuters register heuristics: `getNumberOfRegisters` returns 1 — "just enough to enable the vectorizers but disables heuristics based on the number of registers" ([`NVPTXTargetTransformInfo.h#L75-L80`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetTransformInfo.h#L75-L80)); vector chain legality via `isLegalToVectorizeLoadChain` (alignment ≥ chain size, [L64-L72](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Target/NVPTX/NVPTXTargetTransformInfo.h#L64-L72)).
- Practical consequence: LV on NVPTX mostly helps indirectly (exposing ld.v4 opportunities consumed by LoadStoreVectorizer); NVPTX has no true vector registers and much vector IR is scalarized. **For GPU memory tiling/vectorization, loop-level transforms are better done explicitly (your pass, or MLIR tiling §5) than by relying on LoopVectorize.**

### 4.6 Triton (reference designs; Triton @ `4fa6ce6`)
Pass registry: [`include/triton/Dialect/TritonGPU/Transforms/Passes.td`](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td):
- **`tritongpu-coalesce`** ([L254-L260](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L254-L260)): "analyses loads/stores with type `tensor<tt.ptr<>>` and replaces the layouts of these operations with coalesced layouts, i.e. cache friendly access patterns" — implementation [`lib/Dialect/TritonGPU/Transforms/Coalesce.cpp`](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp#L25-L40) (picks vector size up to 128-bit per thread via AxisInfo analysis). This is the cleanest open-source **coalescing analysis** design: it's *layout assignment*, i.e. choosing per-thread data contiguity rather than rewriting pointers.
- **`tritongpu-pipeline`** ([L6-L12](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L6-L12)): software pipelining, converts loads to async loads + multi-buffering; implementation in [`lib/Dialect/TritonGPU/Transforms/Pipeliner/`](https://github.com/triton-lang/triton/tree/4fa6ce6f86d398cc8e582f869d22096e481ca85c/lib/Dialect/TritonGPU/Transforms/Pipeliner) (`SoftwarePipeliner.cpp`, `ScheduleLoops.cpp`, `PipelineExpander.cpp`, `WGMMAPipeline.cpp`, `TMAStoresPipeline.cpp`).
- **`tritongpu-coalesce-async-copy`** ([L357-L362](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L357-L362)), **`tritongpu-prefetch`** ([L198-L204](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L198-L204)), **`tritongpu-reduce-data-duplication`** ([L326-L330](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L326-L330): converts register→dotOperand conversions through **shared** to enable reuse), **`tritongpu-remove-layout-conversions`** ([L269-L275](https://github.com/triton-lang/triton/blob/4fa6ce6f86d398cc8e582f869d22096e481ca85c/include/triton/Dialect/TritonGPU/Transforms/Passes.td#L269-L275)).
- Design paper: Tillet, Kung, Cox, *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations*, MAPL@PLDI 2019, [DOI 10.1145/3315508.3329973](https://doi.org/10.1145/3315508.3329973) ([PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf)) — describes the tile-level passes incl. shared-memory allocation/synchronization.
- **Reusability:** Triton's passes are tightly coupled to TritonGPU layouts/encodings — treat as *design references*, not libraries.

### 4.7 IREE (MLIR-based; IREE @ `b9a8b77`) — closest production analog of your deliverables
[`compiler/src/iree/compiler/Codegen/Common/GPU/`](https://github.com/iree-org/iree/tree/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/Common/GPU) contains near-1:1 counterparts of your target transforms:
- **Shared-memory promotion:** [`GPUPromoteMatmulOperands.cpp`](https://github.com/iree-org/iree/blob/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/Common/GPU/GPUPromoteMatmulOperands.cpp#L33-L45) (promotes operands to workgroup memory via `linalg.copy` + derived thread config) + [`GPUPromotionAnalysis.cpp/.h`](https://github.com/iree-org/iree/blob/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/Common/GPU/GPUPromotionAnalysis.h#L17-L28) (backward analysis deciding promotion kind) + `GPUTensorAlloc.cpp`, `GPUVectorAlloc.cpp`.
- **Bank-conflict mitigation:** [`GPUReduceBankConflicts.cpp`](https://github.com/iree-org/iree/blob/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/Common/GPU/GPUReduceBankConflicts.cpp#L38-L45) — "Pad out the inner dimension of the `memref.alloc` op in order to reduce the chances to have bank conflicts" (plus swizzle-based strategies); see also `GPUApplyPaddingLevel.cpp`, `GPUPadOperands.cpp`, `GPUPadConvs.cpp`.
- **Tiling/distribution:** `GPUTile.cpp`, `GPUTensorTile.cpp`, `GPUApplyTilingLevel.cpp`, `GPUDistribute*.cpp`, `LLVMGPUTileAndDistribute.cpp` (in `Codegen/LLVMGPU/`).
- **Pipelining/multi-buffering:** `GPUPipelining.cpp`, `GPUMultiBuffering.cpp`.
- **Coalescing:** `GPUConvertToCoalescedDMA.cpp`, `AMDGPULowerCoalescedDMAToGatherLDS.cpp`, `LLVMGPUGroupGlobalLoads.cpp`.
- **Shared-mem packing/reuse:** [`LLVMGPUPackSharedMemoryAlloc.cpp`](https://github.com/iree-org/iree/blob/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/LLVMGPU/LLVMGPUPackSharedMemoryAlloc.cpp) (packs `nvgpu` shared allocs), `GPUReuseSharedMemoryAllocs.cpp`.
- **Transform dialect integration:** IREE ships its own transform-dialect interpreter pass ([`Common/TransformDialectInterpreter.cpp#L23`](https://github.com/iree-org/iree/blob/b9a8b77dc5c115f871eba1f61b514b6518a08d8b/compiler/src/iree/compiler/Codegen/Common/TransformDialectInterpreter.cpp#L23)) and custom transform ops in `Common/TransformExtensions/` — the model for "user-specified tiling strategies".
- **Reusability:** IREE passes are real, maintained code you can study and (license-permitting, Apache-2.0 WITH LLVM-exception) lift patterns from; they are not drop-in libraries (depend on IREE's codegen dialects/attrs).

### 4.8 TVM
- Classic schedule primitives: `te.schedule` `storage_align` (padding to avoid bank conflicts), `set_scope("shared")`, `double_buffer`, `compute_at` — see TVM docs/tutorial and e.g. [this discussion of storage_align + double_buffer for shared-memory bank conflicts](https://discuss.tvm.apache.org/t/problem-with-storage-align-and-double-buffer/12601).
- Bank-conflict pass in-tree: [`src/s_tir/transform/inject_permuted_layout.cc`](https://github.com/apache/tvm/blob/bb750012/src/s_tir/transform/inject_permuted_layout.cc) — "injects permuted layout for shared memory buffers to avoid bank conflicts" (XOR-swizzled 8×8/8×4 layouts for ldmatrix-style access).
- Modern GPU scheduling: DLight rule-based GPU scheduling (`dl.gpu.Matmul/Reduction/...`) — [TVM docs](https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/dlight_gpu_scheduling.html); shared-memory-related TIR passes (`MergeSharedMemoryAllocations`, `ManifestSharedMemoryLocalStage`, `ThreadSync`) — [s_tir.transform reference](https://tvm.apache.org/docs/reference/api/python/s_tir/transform.html).
- **Reusability:** reference designs only (different IR/scheduler model).

### 4.9 Reusable vs write-your-own (bottom line)

| Transform | Reusable today | Must write |
|---|---|---|
| Vectorized load/store rewrite | `LoadStoreVectorizer` (backend, legacy PM) | new-PM/mid-end version if you need it before codegen |
| Redundant load elimination | EarlyCSE/GVN (run `-passes=earlycse,gvn` or via plugin); NewGVN experimental | cross-thread/uniform-value awareness (GPU-specific) |
| Address computation simplification | `InferAddressSpacesPass`, SeparateConstOffsetFromGEP, NaryReassociate, SLSR (all upstream; backend runs them late) | earlier/more aggressive application; addrspace-aware GEP combining |
| Global→shared promotion | MLIR: `gpu` MemoryPromotion utilities (§5); IREE GPUPromoteMatmulOperands | LLVM-IR-level promotion pass (no upstream equivalent) |
| Loop tiling | MLIR linalg/affine tiling + transform dialect (§5) | LLVM-IR tiling (LoopVectorize is not a tiler for this purpose) |
| Shared-memory padding (bank conflicts) | IREE GPUReduceBankConflicts (pattern), TVM inject_permuted_layout (pattern) | the pass itself for LLVM IR/MLIR in your pipeline |
| Coalescing analysis | Triton `tritongpu-coalesce` (design reference) | everything for LLVM IR (nothing upstream) |

---

## 5. MLIR vs LLVM IR — where to build what

### 5.1 MLIR GPU memory model
- `gpu.address_space` attribute: `global` / `workgroup` / `private`, usable as `memref` `memorySpace`; "the `gpu.module`/`gpu.func` compilation pipeline will lower such memory space usages to the correct address spaces on target platforms" — [`mlir/docs/Dialects/GPU.md#L27-L42`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/docs/Dialects/GPU.md#L27-L42).
- Upstream **memory promotion utilities** already exist: [`mlir/lib/Dialect/GPU/Transforms/MemoryPromotion.cpp`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/lib/Dialect/GPU/Transforms/MemoryPromotion.cpp#L1-L50) — emits loop nests copying between memory levels, mapping innermost loops to thread dims "in reverse order to enable access coalescing in the innermost loop" ([L29-L31](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/lib/Dialect/GPU/Transforms/MemoryPromotion.cpp#L29-L31)).
- GPU-dialect transform passes: kernel outlining, loop→GPU mapping (`gpu-map-parallel-loops` "greedily maps loops to GPU hardware dimensions"), barrier elimination, memref decomposition, module→binary, NVVM/ROCDL/SPIRV target attach — [`mlir/include/mlir/Dialect/GPU/Transforms/Passes.td#L14-L262`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/include/mlir/Dialect/GPU/Transforms/Passes.td#L14-L262).
- Lowering to NVVM maps `workgroup→addrspace(3)`, `global→addrspace(1)`, private→0 (alloca): [`mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp#L440-L460`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/lib/Conversion/GPUToNVVM/LowerGpuOpsToNVVMOps.cpp#L440-L460).

### 5.2 Transform dialect for tiling
- Overview: [`mlir/docs/Dialects/Transform.md`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/docs/Dialects/Transform.md#L1-L60) — transform IR orchestrates fine-grain payload transforms (find loops by property → tile → unroll), "not intended as a replacement for the pass infrastructure".
- Tutorial with a worked **tiling example**: [`mlir/docs/Tutorials/transform/Ch2.md`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/mlir/docs/Tutorials/transform/Ch2.md#L299-L339) (`transform.structured.tile_using_forall` + fusion); chapters Ch0–Ch4 + ChH in the same directory.
- IREE's usage (§4.7) shows the production pattern: ship an interpreter pass + project-specific transform ops.

### 5.3 Decision guide
| Concern | LLVM IR (new-PM plugin) | MLIR |
|---|---|---|
| Input you already have | CUDA/HIP via clang `-emit-llvm` is trivial; ecosystem (lit, update_test_checks) mature | Need import (clang MLIR codegen is limited; polyhedral/linalg frontends) |
| Loop tiling | Manual (no first-class loop-nest op; SCEV-based) | **First-class**: linalg/affine/scf tiling, transform dialect |
| Shared-memory promotion | You model allocas/globals + addrspace(3) yourself | **First-class**: memref memory spaces, gpu.address_space, MemoryPromotion utilities |
| Vectorized ld/st rewrite | **Strong**: LoadStoreVectorizer patterns, TTI legality | vector dialect transfer_read/write + vectorization passes |
| Redundant load elim | **Strong**: EarlyCSE/GVN/NewGVN + AA | memref dataflow opts are thinner |
| Address-space inference | InferAddressSpaces exists | you keep memory spaces explicit (that's the point) |
| Coalescing analysis | Nothing upstream | Triton/IREE designs available to imitate |
| PTX emission | `llc -mtriple=nvptx64...` directly | gpu.module → NVVM attach target → `gpu-to-llvm`/`gpu-module-to-binary`, or translate to LLVM IR then llc |
| ABI/plugin stability | Pass-plugin ABI caveats (§1.2) | In-tree APIs churn faster, but you build against your pinned LLVM |

**Recommendation (evidence-based):** build the *load/store-level* transforms (vectorized ld/st, redundant-load elim, address simplification, coalescing analysis) as **LLVM IR passes** — the IR is already in the right shape from clang, the analyses (AA/SCEV/DomTree) exist, and you can A/B against the backend's own passes. Build *tiling + promotion* in **MLIR** if you control the frontend path (linalg-on-tensors → gpu → NVVM), using the transform dialect for strategy specification; otherwise implement promotion at the LLVM-IR level (allocate `addrspace(3)` global/alloca, rewrite uses, insert barriers — mirroring what clang emits for `__shared__`, §2.5).

---

## 6. Testing: lit/FileCheck for out-of-tree pass plugins

### 6.1 Conventions (in-tree reference)
- [`llvm/docs/TestingGuide.rst`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/TestingGuide.rst#L226-L330): `RUN:` lines, FileCheck usage, extra inputs via `Inputs/` dirs + `split-file`.
- **Plugin-loading test patterns** (copy these idioms):
  - clang side: [`clang/test/CodeGen/codegen-plugins.c`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/test/CodeGen/codegen-plugins.c#L1-L8) — `RUN: %clang_cc1 -S < %s -fpass-plugin=%llvmshlibdir/Bye%pluginext ...` with `REQUIRES: plugins, llvm-examples`.
  - opt side: pipeline-dump tests use `%loadbye` / `%llvmcheckext` substitutions wired in [`llvm/test/lit.cfg.py#L395-L432`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/test/lit.cfg.py#L395-L432) (`%loadnewpmbye` expands to `-load-pass-plugin=%llvmshlibdir/Bye%pluginext`); see [`llvm/test/Other/new-pm-defaults.ll#L10-L15`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/test/Other/new-pm-defaults.ll#L10-L15). (Note: the historical `llvm/test/Other/new-pass-manager/` plugin-test directory is gone in LLVM 20–23; the above are the current idioms. Unit tests live in `llvm/unittests/Passes/Plugins/`.)
  - Canonical RUN line for your own plugin:
    ```llvm
    ; RUN: opt -load-pass-plugin=%llvmshlibdir/MyPlugin%pluginext -passes=my-pass -S %s | FileCheck %s
    ; REQUIRES: plugins
    ```

### 6.2 `update_test_checks.py`
[`llvm/utils/update_test_checks.py`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/utils/update_test_checks.py#L1-L33): auto-generates FileCheck lines from `opt` output; supports `--tool-binary`/`--tool` override; documented 9-step workflow (baseline → patch → regenerate → diff). For plugin passes, point `--tool-binary` at an `opt` that can load your plugin and keep the `-load-pass-plugin=...` in the RUN line (the script reuses the RUN line's tool invocation). Siblings: `update_analyze_test_checks.py`, `update_cc_test_checks.py` (same directory).

### 6.3 Out-of-tree wiring (the standard model: llvm-tutor)
[banach-space/llvm-tutor](https://github.com/banach-space/llvm-tutor) is the canonical out-of-tree tutorial (actively maintained; requires LLVM ≥ 22 per its [CMakeLists](https://github.com/banach-space/llvm-tutor/blob/main/HelloWorld/CMakeLists.txt)):
- Standalone CMake project consuming `LLVMConfig.cmake` (`find_package(LLVM CONFIG)`), matching `-fno-rtti` to `LLVM_ENABLE_RTTI`.
- `test/lit.cfg.py` + `lit.site.cfg.py.in` configured from CMake (`add_lit_testsuite`), with `%opt`/plugin substitutions pointing at your build — see [llvm-tutor test/lit.cfg.py](https://github.com/banach-space/llvm-tutor/blob/main/test/lit.cfg.py).
- In-tree example plugins to copy: [`llvm/examples/Bye/Bye.cpp`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/examples/Bye/Bye.cpp) and [`llvm/examples/IRTransforms/SimplifyCFG.cpp`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/examples/IRTransforms/SimplifyCFG.cpp#L395-L412).

---

## 7. Report artifacts: before/after IR, PTX, dumps

- **Per-pass IR dumps (new PM):** `-print-before-all` / `-print-after-all` (and `-print-before=<pass>`, `-print-after=<pass>`) — defined in [`llvm/lib/IR/PrintPasses.cpp#L31-L39`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/IR/PrintPasses.cpp#L31-L39), consumed by `PrintIRInstrumentation` ([`llvm/lib/Passes/StandardInstrumentations.cpp#L977-L983`](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/StandardInstrumentations.cpp#L977-L983)). Works with plugins: `opt -load-pass-plugin=My.so -passes=my-pass -print-after-all in.ll`.
- **Dump to files:** `-ir-dump-directory=<dir>` ([StandardInstrumentations.cpp#L131-L135](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/StandardInstrumentations.cpp#L131-L135)) writes one file per pass — ideal for generating report artifacts. Pass numbering for precise dumps: `-print-pass-numbers` + `-print-before-pass-number=N` / `-print-after-pass-number=N` ([L118-L129](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/StandardInstrumentations.cpp#L118-L129)).
- **Change-only diffs:** `-print-changed` (+ `-print-before-changed`, dot-CFG rendering options) — [StandardInstrumentations.cpp#L60-L100](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Passes/StandardInstrumentations.cpp#L60-L100).
- **Before/after diffing:** emit `opt -S` before/after files and use `llvm-diff` ([CommandGuide](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/docs/CommandGuide/llvm-diff.rst)) for semantic IR diff, or plain `diff -u` for reports.
- **PTX artifacts:** `llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_XX before.ll / after.ll` (§3.1); compare PTX side-by-side. For full clang-driven flow add `-save-temps` to keep intermediate outputs. Legacy codegen pipeline dumps also support `-print-after-all` via `llc` (codegen instrumentation).
- **Benchmarks (practical notes):** measure with CUDA events or Nsight Compute; the metrics that directly validate your transforms are bank-conflict counters (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared*`), coalescing/sector counters (`lts__t_sectors_srcunit_tex_op_*`, `l1tex__t_sectors_pipe_lsu_mem_global_op_*`), and shared vs global load instruction mix in the PTX (`ld.shared` vs `ld.global`, `ld.global.v4`). (Nsight Compute metric names per NVIDIA Nsight Compute documentation; verify against your toolkit version.)

---

## 8. Key papers & docs (verified)

1. **Triton:** P. Tillet, H. T. Kung, D. Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL @ PLDI 2019. [DOI 10.1145/3315508.3329973](https://dl.acm.org/doi/10.1145/3315508.3329973), [PDF](https://www.eecs.harvard.edu/~htk/publication/2019-mapl-tillet-kung-cox.pdf).
2. **MLIR:** C. Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation", IEEE/ACM CGO 2021. [DOI 10.1109/CGO51591.2021.9370308](https://doi.org/10.1109/CGO51591.2021.9370308); extended version "MLIR: A Compiler Infrastructure for the End of Moore's Law", [arXiv:2002.11054](https://arxiv.org/abs/2002.11054).
3. **GVN algorithm (NewGVN):** K. Gargi, "A Sparse Algorithm for Predicated Global Value Numbering" (cited in [NewGVN.cpp#L16-L18](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/llvm/lib/Transforms/Scalar/NewGVN.cpp#L16-L18)).
4. **GPU occupancy/ILP & shared-memory motivation:** V. Volkov, "Better performance at lower occupancy", GTC 2010 (classic justification for ILP-oriented memory transforms and bank-conflict avoidance).
5. **NVIDIA references:** CUDA C++ Programming Guide — "Shared Memory" / bank-conflict sections; CUDA C++ Best Practices Guide — "Coalesced Memory Access" (docs.nvidia.com/cuda/). PTX ISA reference for `cp.async`, `ldmatrix`, `ld.shared` semantics.
6. **LLVM docs used above:** NVPTXUsage.rst, AMDGPUUsage.rst, LangRef.rst (addrspacecast), WritingAnLLVMNewPMPass.rst, NewPassManager.rst, TestingGuide.rst, CompileCudaWithLLVM.rst; clang HIPSupport.rst.

## 9. Caveats / things that changed recently (watch out)

- `IntrinsicsNVPTX.td` → **`IntrinsicsNVVM.td`** (already in LLVM 20; older blogs/tutorials show the old name).
- `llvm/lib/Transforms/HelloNew/` removed; HelloWorld now in `Transforms/Utils`; plugin examples are `llvm/examples/Bye` and `llvm/examples/IRTransforms`.
- `llvm/test/Other/new-pass-manager/` plugin tests removed (LLVM ≤19 era); use `codegen-plugins.c` / `new-pm-defaults.ll` idioms + `llvm/unittests/Passes/Plugins`.
- `ClangCommandLineReference.rst` is now **generated** from `clang/include/clang/Driver/ClangOptionDocs.td` at docs-build time ([clang/docs/CMakeLists.txt#L135](https://github.com/llvm/llvm-project/blob/2078da43e25a4623cab2d0d60decddf709aaea28/clang/docs/CMakeLists.txt#L135)) — cite `Options.td` instead.
- NVPTX added AS 6 (tensor) and AS 7 (shared cluster) beyond the classic 0/1/3/4/5 table — relevant for sm_90+ promotion work.
- NVPTXImageOptimizer is now about image/`istypep` intrinsics, not constant-memory folding (don't trust older descriptions).
