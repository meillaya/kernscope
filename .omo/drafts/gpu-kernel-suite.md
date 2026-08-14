---
slug: gpu-kernel-suite
status: approved-review-passed
intent: clear
review_required: true
plan_path: .omo/plans/gpu-kernel-suite.md
plan_sha256: b9fc0e61a7aef274b45ec9f39148010fc9dae1d7edbd926dc1bffca43ad136dd
review_round_id: round-20260813-r4
round_status: approved
pending-action: DELIVERED - execution only via user-started worker (/start-work); planner stops here
review:
  summary: DUAL high-accuracy review PASSED in round 4 (both lanes APPROVE, zero outstanding issues). Rounds 1-3 were CHANGES_REQUESTED and every cited issue was fixed and re-verified.
  round-4-results:
    momus:
      verdict: APPROVE
      session: ses_001f6e176ffeHPIqOR79kiENmB
      launch_id: launch-momus-20260813-r4
      receipt: momus-review-r4@entropyos-nix (descriptor no-follow read, SHA-256 verified == plan_sha256)
      verified: three round-3 edits quoted+confirmed; F1 numbering 1-57; F2 Depends==Blocked-by all 57; F3a forward transpose PASS; F3b reverse transpose PASS (wildcard semantics documented); F3c todo-Blocks-lines==matrix PASS; F4 acyclic; F1-F4 self-consistent; grounding artifacts present
    independent:
      verdict: APPROVE
      session: ses_001f6a78cffenx2tnLUYu7jSsG
      launch_id: launch-oracle-20260813-r4
      receipt: oracle lane pid 3342060 @ entropyos-nix, concluded 2026-08-14T02:22:11Z (descriptor no-follow read, SHA-256 verified == plan_sha256)
      verified: three round-3 fixes quoted; full mechanical audit clean - Blocked-by<->Depends exact all 57, F1 transpose PASS, Blocks lines<->matrix exact, direct-dependents coverage complete, zero phantom Blocks entries, parallelize symmetric+independent, Kahn topo all 57 nodes (131 edges, acyclic), all reachable from 1, 57 unique sink, 57 acceptance/57 QA/57 Commit lines
    final-live-validation: sha256sum re-run pre-handoff == b9fc0e61a7aef274b45ec9f39148010fc9dae1d7edbd926dc1bffca43ad136dd (exact match, CHAIN-OK) - both approvals valid
---
approach: ONE umbrella project (working name "kernscope") in this repo - shared core libs (arch-model DB, PTX/AMDGPU parsers, occupancy, roofline, bench/golden harnesses, report engine, CUDA/HIP runtime abstraction) + six tool modules (analyze, tune, verify, capture, pass, profile) forming a profile->capture->analyze->transform/generate->verify->benchmark loop; devenv-latest env (llvmPackages_21 + cudaPackages_12_6 + rocmPackages, polykernel-validated on this machine); DUAL-PATH hardware strategy (CUDA first-class + HIP per module; local AMD runtime loop; owner-gated NVIDIA rentals); 4-layer test convention (lit/FileCheck + gtest/ctest + NumPy golden + compile checks); dependency-ordered waves, each module independently shippable
review:
  round-1-results:
    momus: CHANGES_REQUESTED (3 HIGH / 4 MEDIUM / 6 LOW, 0 CRITICAL) - session ses_00239d969ffe12PhsXJtNdO6N0; verified solid: grammar, scope fidelity (10/6/8/7/7 mapping), occupancy math, sibling refs, brief citations
    independent: CHANGES_REQUESTED (3 HIGH / 5 MEDIUM / 4 LOW, 0 CRITICAL) - session ses_002397d4dffeeNHG1VG2uIDT17; independently re-derived occupancy golden, verified matrix vs todos, machine facts
    fixes-applied: single-root CUDA_HOME + CUPTI_HOME + CUDA_PATH (todo 2); todo 44 deps += 45,24 + equivalence moved to field-for-field-under-mapping semantics; sm_100/tcgen05 fixtures hand-written parse-only (todo 9); header guard -> vendored-file detection (todo 4 + Scope + Success criteria); circularity breaker rewritten (Verification strategy + todo 21 byte-exact mock buffers); owner-examples.md materialized in todo 1 + all citations re-pointed; F1-F4 given agent-executed checklists incl. F3 defect-seeding; disk preflight df>=40G + nixpkgs-rev reuse note (todo 2); todo 5 stop-the-line rule replaces deviation fallback; helper scripts assigned to owning todos (2/5/7/19/20/22/29/52/55); matrix reconciled (rows 9/11/12/21/22/24/43/44/45) + wave prose fixed; commit count 54->57; todo 28 fp16->bf16; todo 32 owns classes 3+6 + typo fixed; todo 53 baseline-config pinned; pass-brief citations corrected to §4.x; polykernel python/ ref corrected
  round-2-results:
    momus: CHANGES_REQUESTED (1 MEDIUM / 2 LOW) - session ses_0021bb08bffe7VyaGyrjcFGzCV; verified ALL round-1 fixes present; independently recomputed golden math + full matrix transpose audit
    independent: CHANGES_REQUESTED (3 MEDIUM / 3 LOW) - session ses_0021b428bffeB6GaAlQoIBxfEq; verified ALL 12 round-1 fixes substantively landed; empirically tested guard regex + F3 executability
    fixes-applied: matrix row 20 Blocks += 51 (both reviewers' MEDIUM); todo 9 body sm_100 nvcc wording fixed; todo 3 now owns .clang-tidy + -Wall -Wextra -Werror (F2 prerequisite); todo 9/12/21 forward Blocks lines aligned with matrix (+11/+42/+43); pass-brief citations corrected (todo 37 -> §4.2+§4.5, 38 -> §4.6+§4.8, 39 -> §4.4 + MemorySSA-as-LLVM-API note, 40 -> §4.7 GPUPromoteMatmulOperands); byte-diff golden wording -> field-for-field assertion discipline; evidence location unified (.omo/evidence/ IS committed)
  round-3-results:
    momus: CHANGES_REQUESTED (1 MEDIUM) - session ses_002088c61ffecb5rd1ggQlzuuR; verified ALL round-2/round-3 fixes present + regression-free (numbering, Depends==Blocked-by all 57, transpose, acyclic, citations spot-checked vs brief headings); sole issue: todo 20 Blocks line missing 51; stated "once todo 20's Blocks line is aligned, this plan is ready to APPROVE"
    independent: CHANGES_REQUESTED (1 MEDIUM / 2 LOW) - session ses_002083593ffeBmtB4HyHzc0Xee; verified all six round-2 fixes (M1-M3, L4-L6) genuinely fixed with quotes; mechanical audit all-57: Blocked-by<->Depends exact, F1 transpose PASS, acyclic, clang-tidy empirically available in nix store; new findings: todo 20 Blocks line missing 51 (same MEDIUM), phantom Blocks entry 7->48, phantom Blocks entry 9->35
    fixes-applied: todo 20 Blocks line -> "51,53,54,55"; phantom 7->48 removed from matrix row 7 + todo 7 Blocks line; phantom 9->35 removed from matrix row 9 + todo 9 Blocks line
  momus:
    status: in_flight
    workspace_root: /home/mei/projects/llvm.h
    runtime_home: /home/mei
    target: .omo/plans/gpu-kernel-suite.md
    round_id: round-20260813-r4
    plan_sha256: b9fc0e61a7aef274b45ec9f39148010fc9dae1d7edbd926dc1bffca43ad136dd
    launch_id: launch-momus-20260813-r4
    session: ses_001f6e176ffeHPIqOR79kiENmB (bg_8e55d1ea)
    result: null
  independent:
    status: in_flight
    workspace_root: /home/mei/projects/llvm.h
    runtime_home: /home/mei
    target: .omo/plans/gpu-kernel-suite.md
    round_id: round-20260813-r4
    plan_sha256: b9fc0e61a7aef274b45ec9f39148010fc9dae1d7edbd926dc1bffca43ad136dd
    launch_id: launch-oracle-20260813-r4
    session: ses_001f6a78cffenx2tnLUYu7jSsG (bg_e7e8c771)
    result: null
---

# Draft: gpu-kernel-suite

## Components (topology ledger)
<!-- Lock the SHAPE before depth. One row per top-level component that can succeed or fail independently. -->
<!-- id | outcome (one line) | status: active|deferred | evidence path -->
- C0 | Monorepo foundation: devenv.sh (latest) environments, build system, shared C++ infra, CI, test harness | active | env probe + devenv lane done (findings below); polykernel devenv.nix:1-112 as proven template
- C1 | PTX/AMDGPU static analyzer: assembly in -> occupancy/coalescing/bank-conflict/divergence/spill/latency report out | active | gpu-static-analyzer-research.md; polykernel overlap = PARTIAL (ptxas -v only)
- C2 | Autotuning kernel generator: template variants -> compile -> benchmark -> best config per arch (matmul, bias+GELU, softmax, layernorm, reduction, flash-attn-lite) | active | autotuner lane brief (tool-output file); polykernel overlap = PARTIAL (bounded grid, own kernels only)
- C3 | GPU kernel verifier: eBPF-style static safety checks on GPU IR before backend lowering | active | gpu-kernel-verifier-research-brief.md; overlap NONE; closest prior art GPUVerify (unmaintained) + HGRD 2026
- C4 | Kernel capture & replay: intercept launches, snapshot state, emit standalone C++ reproducers | active | capture_replay_research_brief.md; overlap NONE; prior art Mneme/Kerncap/R3 (differentiate: dual-vendor + tune/analyze integration)
- C5 | LLVM/MLIR GPU memory optimization passes + lit/FileCheck tests + before/after IR/PTX | active | gpu-memory-pass-research.md; overlap NONE
- C6 | Tiny CUDA/ROCm profiler with compiler-aware "why is this slow" diagnostics | active | profiler-research-brief.md; overlap NONE (polykernel compile-time only)
- OPEN-DECISION | Merge C1-C6 into ONE umbrella project vs keep separate | RESOLVED->recommend MERGE (see Decisions) | user delegated: "If it's possible to merge them into one concrete project idea say so"

## Open assumptions (announced defaults)
<!-- Record any default you adopt instead of asking, so the user can veto it at the gate. -->
<!-- assumption | adopted default | rationale | reversible? -->
- Project naming | working name "kernscope" for the umbrella suite (repo dir stays llvm.h unless renamed) | matches sibling naming style (PolyKernel/TensixFlow/tensorforge); identity appears in resume bullets | yes (rename = grep+mv)
- Build system | CMake + Ninja superbuild, one env | all GPU siblings (polykernel/tt-flow/tiny-rasterizer) use CMake+Ninja; tensors/Bazel is the lone exception | yes
- C++ standard | C++20 | tensors badge = C++20; polykernel/tt-flow build fine on it; clang 22 available | yes
- LLVM version | llvmPackages_21 (21.1.8, nixpkgs default) | polykernel VALIDATED MLIR-21 out-of-tree build on this exact machine; 22 available if wanted | yes
- CUDA version | cudaPackages_12_6 | polykernel-validated on this machine; CUPTI Checkpoint API present since 11.5; Kernel9 records documented; avoids CUDA-13 API-removal churn | yes
- ROCm | rocmPackages (ROCm 7.2.3) via nixpkgs binaries; gfx1101 target | polykernel runs kernels on this exact card via rocmPackages.clr; /dev/kfd present | yes
- Code reuse from polykernel | fresh implementations; polykernel as design reference + cross-validation oracle only | clean ownership/scope per repo; avoids cross-repo coupling | yes
- Research briefs | move the 5 repo-root *.md briefs + autotuner brief (extract from tool-output file) into docs/research/ in foundation wave | keep repo root clean; briefs are executor evidence | yes
- nixpkgs config location | devenv.yaml-native (nixpkgs.cuda_support/rocm_support/allow_unfree) if no flake needed, else flake.nix per polykernel note | polykernel verified devenv module system has no nixpkgs.config option; yaml options exist since 1.7/2.0.7 | yes
- Cloud NVIDIA validation | owner-gated rentals (RunPod/Modal/PrimeIntellect pattern from polykernel/tensors), never auto-triggered, never on my initiative | established user pattern; cost is owner's | n/a (process)

## Findings (cited - path:lines)
### Environment probe (bg_0c4e3154, verified exact command outputs)
- OS: CachyOS (Arch-like rolling), kernel 7.1.8-1-cachyos, x86_64, host entropyos-nix
- Nix stack PRESENT: Determinate Nix 3.21.7 (nix 2.34.8), devenv 2.2.1+8f297ea at /home/mei/.nix-profile/bin/devenv
- GPU: AMD Navi 32 [Radeon RX 7700 XT / 7800 XT] (RDNA3, gfx1101) - ONLY GPU. amdgpu module loaded; /dev/kfd exists (render group). NO NVIDIA GPU, nvidia-smi NOT PRESENT, nvcc NOT PRESENT, no /opt/cuda
- ROCm userspace NOT installed (rocm-smi/rocminfo/hipcc NOT PRESENT; pacman only opencl-mesa) - but kernel-side ready (/dev/kfd), provisionable via nixpkgs rocmPackages
- Compilers: gcc/g++ 16.1.1, clang/clang++ 22.1.8 (/usr/bin), cmake 4.1.2, node v24.16.0, python 3.13.13; hipcc NOT PRESENT
- Repo /home/mei/projects/llvm.h: empty placeholder (.codegraph symlink + .omo only), NOT a git repo - needs git init
- Disk: /home and /nix share /dev/nvme1n1p2: 462G total, 112G free, 76% used - budget ROCm/LLVM closures carefully
- Sibling mini-projects each = own git repo + devenv.nix/devenv.yaml (pattern: atom-bridge, enten, flow-qs, forgestack, hiraeth, inference-ops, polykernel, tensors, tt-flow...). No "project 1" dir exists
- GPU-relevant siblings needing overlap check: polykernel (kernels/tuning/benchmarks/tools), tt-flow (kernels/benchmarks/reports), tensors (cuda/ subdir, Bazel), kernels/kernel-v1..v6
- devenv docs (context7 /cachix/devenv): v1.10+ monorepo imports in devenv.yaml (/ git-root paths, shared inputs); v1.7+ CUDA support via nixpkgs.config.cudaSupport + cudaCapabilities + allowUnfree. NOTE: installed devenv is 2.2.1 - verify 2.x config surface via librarian lane bg_21bd096b
### devenv/nixpkgs research (bg_21bd096b, librarian, SHA-pinned citations; nixpkgs master @0db4e962 2026-08-13)
- Latest stable devenv = v2.2.2 (released 2026-08-13); local install 2.2.1. Lineage: 2.2 (2026-07-28 attach/up, --from out-of-tree envs), 2.1 (require_version, shell option), 2.0 (TUI, native Rust process manager replacing process-compose), 1.10 (monorepo devenv.yaml imports), 1.7 (CUDA support)
- devenv.yaml options (https://devenv.sh/reference/yaml-options/): inputs (default github:cachix/devenv-nixpkgs/rolling), imports (relative | /git-root | input names; local yaml imports since 1.10), nixpkgs.allow_unfree (1.7), nixpkgs.cuda_support + cuda_capabilities (1.7), nixpkgs.rocm_support (2.0.7), nixpkgs.per_platform (1.7), require_version (2.1), profile (1.11), reload (2.0)
- Files: devenv.nix (module), devenv.yaml, devenv.lock (per devenv.yaml dir; commit all), devenv.local.nix/yaml (uncommitted). Tasks DAG (exec, before/after, execIfModified, namespaces, devenv:enterShell/enterTest); processes are tasks
- Monorepo pattern (https://devenv.sh/guides/monorepo/): root devenv.yaml + shared/devenv.nix + per-project devenv.yaml `imports: [/shared]`; config.git.root; require_version ">=2.2" pins contributors
- nixpkgs CUDA: cudaPackages_12_6 (12.6.3), 12_8, 12_9 (12.9.1), 13_0..13_3; cudaPackages_12=12_9; cudaPackages_13=13_2; default cudaPackages=12_x. Packages split: cuda_nvcc, cuda_cudart, cuda_nvrtc, libcublas, cuda_cuobjdump, cudnn, cutlass. Unfree (allowUnfreePredicate = _cuda.lib.allowUnfreeCudaPredicate). Multi-GB closures; driver only needed at RUNTIME (compile/ptxas fine without NVIDIA GPU); cuda_compat for forward compat. Manual: https://nixos.org/manual/nixpkgs/unstable/#sec-cuda
- nixpkgs ROCm: rocmPackages = pkgs/development/rocm-modules, ROCm 7.2.3 (LLVM-22-based fork); hipcc MIT linux-only; clr provides hipcc+HIP runtime and sets HIP_PATH/ROCM_PATH; rocminfo, rocm-smi, amdsmi available; binaries Hydra-cached (release-attrPaths.json); non-NixOS needs host amdgpu + /dev/kfd (PRESENT here); known ROCM_PATH workaround (nixpkgs issue 388548); source builds brutal (avoid; use binaries)
- nixpkgs LLVM: llvmPackages_18 (18.1.8) .. 22 (22.1.8), 23 (23.1.0-rc1), 24.0.0-git; DEFAULT llvmPackages_21 (21.1.8). libllvm dev output has lib/cmake/llvm; mlir attr per version (lib/cmake/mlir in mlir.dev); FileCheck in ${llvm}/bin; pkgs.lit = python lit; also standalone pkgs.filecheck
- languages.cplusplus module is thin (clang-tools, cmake, clang, ccls LSP) - real toolchains come via packages list
- DOC LAG notes: 1.7 blog shows old yaml syntax (nixpkgs.config.x86_64-linux.cudaSupport) vs current nixpkgs.cuda_support; cudaPackages_13 alias lags 13_3
### Autotuner prior art (bg_3287938a; full brief: /home/mei/.local/share/opencode/tool-output/tool_ffd31250a0016mwdsgYKo46vPL)
- Triton autotuner (triton-lang/triton@4fa6ce6f): Config(kwargs,num_warps=4,num_stages=3,num_ctas,maxnreg,pre_hook); Autotuner(keys, cache, prune_configs_by{early_config_prune,perf_model,top_k}); do_bench(warmup=25ms,rep=100ms time-based, L2 flush each iter, CUDA events, quantiles 0.5/0.2/0.8, select by median); disk cache sha256 -> <fn>.autotune.json
- TVM@27c2e019: AutoScheduler/Ansor REMOVED from main (cite OSDI'20 paper); MetaSchedule under src/s_tir/meta_schedule: evolutionary_search.cc (population/mutation/crossover scored by cost model), XGBoost+MLP cost models, JSONDatabase (workload JSON + JSONL records keyed by workload hash), round-robin task scheduler
- Halide Adams2019: beam_size=32 default, learned cost model, no live benchmark in scheduler (CostModel.h L19-56)
- CUTLASS@dcf215af: profiler enumerates Instance Library manifest, instantiation levels, --warmup-iterations=10 default, CUDA-graph capture timing w/ cudaEventRecordExternal, sleep between periods for thermals; dispatch_policy.hpp kernel schedules (KernelTmaWarpSpecialized{Cooperative,Pingpong}); CuTe Swizzle<B,M,S> (swizzle.hpp L41-55); sgemm_sm80.cu uses Swizzle<3,3,3>; efficient_gemm.md hierarchical tiling + pipelining
- nvbench (rapidsai/nvbench, ARCHIVED 2021): L2 flush via cudaDevAttrL2CacheSize buffer + cudaMemsetAsync; measure_cold adaptive trials (min_time/min_samples/max_noise/timeout); CUDA event timing
- Roofline: Williams/Waterman/Patterson CACM 2009 DOI 10.1145/1498765.1498785; peaks: A100 FP32 19.5TF/BF16-TC 312TF dense/HBM2e 2039GB/s (80GB SXM); RTX4090 FP32 82.6TF/1008GB/s; MI250X FP16 383TF/3.28TB/s; MI300X BF16 1307.4TF/HBM3 5.3TB/s; FlashAttention FLOPs 4N^2d
### Verifier prior art (bg_5b7c9b4f; brief: gpu-kernel-verifier-research-brief.md)
- eBPF verifier: Documentation/bpf/verifier.rst (no checker.rst); verifier.c anchors (do_check, check_mem_access, adjust_ptr_min_max_vals, sanitize_ptr_alu, check_map_access, regsafe/states_equal pruning, precision backtracking); BPF_COMPLEXITY_LIMIT_JMP_SEQ=8192 on current master; tnum tristate; 512B stack, 11 regs
- GPUVerify: OOPSLA'12 DOI 10.1145/2384616.2384625, TOPLAS'15 DOI 10.1145/2743017, CAV'14; mc-imperial/GPUVerify last push 2022-07 (effectively unmaintained); pipeline CUDA/OpenCL -> clang/LLVM 6.0 -> C# kernel analyzer -> Boogie; checks barrier divergence + data races
- GPU race-detection lit: HiRace arXiv 2401.04701 (tool table); Faial/FaialAA DOI 10.1145/3689797 (static DRF via memory access protocols); HGRD arXiv 2604.02106 (2026: host-code-guided static race detection over MLIR + CP-SAT - closest modern prior art); SESA SC'14; GKLEE PPoPP'12; GRace PPoPP'11; GMRace TPDS'14; CURD ISSTA'18; ESBMC-GPU JSCP'17 DOI 10.1016/j.scico.2017.09.005 (races+OOB+ptr safety via context-bounded BMC); GPU-PRAMChecker/GKRace NOT verifiable (flagged)
- SPIR-V validator spirv-val: source/val/*.cpp 53 category files; one-sided errors; universal limits
- LLVM reusable infra: DivergenceAnalysis.rst + ConvergenceRegions.rst (GPU targets); NVPTXUsage.rst address spaces; LangRef addrspacecast; IKOS (NASA abstract interpretation for LLVM), PhASAR, SVF, SeaHorn/Crab as building blocks
- CUDA semantics: __syncthreads UB if subset of block reaches it; Independent Thread Scheduling (Volta+); launch bounds known at launch site -> threadIdx/blockIdx ranges for SCEV bounds
### Capture/replay prior art (bg_41d3ba5b; brief: capture_replay_research_brief.md)
- Mneme (Olympus-HPC/Mneme, LLNL): record+replay CUDA/HIP kernels as standalone executables; Apache-2.0 w/ LLVM exceptions; ACTIVE (pushed Aug 2026); paper Parasyris et al. SC'23 "Scalable Tuning of GPU Applications via Kernel Record and Replay" - THE "similar recent work"
- Kerncap arXiv 2605.03208 (May 2026): automated kernel extraction/isolation for AMD GPUs; HSA-level interception; address-space closure snapshots; self-contained reproducer projects; code = AMDResearch/intellikit (MIT, active)
- Record-Remix-Replay (R3) arXiv 2604.11109 (2026, LLNL): preload lib intercepts launches+memops, records LLVM IR+launch config+args+device state, replay server, evolutionary optimization on Proteus JIT
- CUPTI: injection via CUDA_INJECTION64_PATH + InitializeInjection; callback API (CUPTI_API_ENTER/EXIT, CUpti_CallbackData.functionParams = launch args); CUPTI Checkpoint API (device state save/restore); ActivityKernel records (grid/block/stream/correlation, NO args); cuptiSubscribe_v2 multi-subscriber (CUDA 13.3+)
- GPGPU-Sim@a4ce3fea: libcuda/cuda_runtime_api.cc full-replacement interception; g_mallocPtr_Size map learns buffer sizes from cudaMalloc; BSD-3-style; checkpoint.md
- NVBit: NVlabs/NVBit, NVIDIA EULA (research-only), active; Nsight Compute replay modes: kernel/application/range
- ROCm: roctracer/rocprofiler/rocprofiler-sdk consolidated into ROCm/rocm-systems (2026); ROCdbgapi MIT active; Kernel Tuner HIP ext arXiv 2407.11488; SIP arXiv 2403.16863; Tally arXiv 2410.07381
- CUDA Graph stream capture: captures launches/memcpys/events, NOT memory contents; cudaMallocAsync-only allocations during capture
### LLVM/MLIR pass research (bg_487d7878; brief: gpu-memory-pass-research.md)
- LLVM state: stable 22.1.8, 23.1.0-rc3; main@2078da43 (23-dev): HelloNew DELETED (sample plugin now llvm/lib/Transforms/Utils/HelloWorld.cpp), canonical plugin examples llvm/examples/Bye + llvm/examples/IRTransforms, plugin lit tests existed <=22 at llvm/test/Other/new-pass-manager; ClangCommandLineReference generated from ClangOptionDocs.td; -fpass-plugin + PassBuilder EP callbacks; plugin ABI NOT stable across versions
- Address spaces VERIFIED: NVPTX 0=generic,1=global,2=internal,3=shared,4=const,5=local,7=shared-cluster(sm_90+); AMDGPU 0=flat,1=global,2=region/GDS,3=local/LDS,4=const,5=private/scratch,6=const32,7/8=buffer
- clang CUDA flow: -x cuda --cuda-device-only --offload-arch=sm_XX -S -emit-llvm (-nocudalib -nocudainc for no-toolkit); HIP: -x hip --offload-arch=gfxXXXX; IR->PTX: llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_XX; __shared__ -> addrspace(3) via TargetInfo::getGlobalVarAddressSpace
- Existing passes: LoadStoreVectorizer (legacy PM; used by AMDGPU), LoopVectorize NVPTX status, NewGVN/EarlyCSE for RLE; IREE PromoteOperandsToSharedMemory (MLIR); Triton passes Pipelining/Coalesce; TVM storage_align for padding
- Testing: lit + FileCheck, update_test_checks.py --opt-binary, llvm-tutor as out-of-tree template; opt -print-after-all via StandardInstrumentations; llvm-diff for IR diffs
### Profiler research (bg_f2746784; brief: profiler-research-brief.md)
- CUPTI CUDA 13 headers (vendored terryye/cuda_GEMM@07ffeb56): CUpti_ActivityKernel10 (L2909-3212) fields start/end/queued/submitted/completed, deviceId/contextId/streamId/correlationId, gridXYZ/blockXYZ, registersPerThread, staticSharedMemory/dynamicSharedMemory, localMemoryPerThread/Total, name/demangledName, launchType, partitionedGlobalCache*; NO occupancy field -> compute via header-only cuda_occupancy.h (cudaOccMaxActiveBlocksPerMultiprocessor) exactly as Nsight Compute does launch__* metrics without replay
- KERNEL kind=3 SERIALIZES kernels; CONCURRENT_KERNEL kind=10 low-overhead (use this); CUPTI_API_VERSION 130001
- CUDA 13 REMOVED: nvprof, NVVP, CUPTI Event/Metric APIs, PC-sampling activity, SASS metrics; Profiling API deprecated -> cupti_range_profiler.h + PM Sampling (Turing+)
- cuFuncGetAttributes: CU_FUNC_ATTRIBUTE_NUM_REGS/SHARED_SIZE_BYTES/LOCAL_SIZE_BYTES/MAX_THREADS_PER_BLOCK...; cuOccupancyMaxActiveBlocksPerMultiprocessor/MaxPotentialBlockSize
- Source<->asm: nvcc -lineinfo (+ newer -src-in-ptx), nvdisasm -g/-gi/-gp (.nv_debug_line_sass -> PTX correlation), cuobjdump -res-usage (REG/STACK/SHARED/LOCAL per function) + --dump-sass/--dump-ptx/--dump-elf
- rocprofiler-sdk@3a924a0d: rocprofiler_kernel_dispatch_info_t (grid/workgroup/group_segment_size=LDS/private_segment_size=scratch); buffer_tracing kernel dispatch records (start/end/correlation_id); code-object tracing service gives sgpr_count/arch_vgpr_count/accum_vgpr_count + kernel_address
- Omniperf renamed rocprofiler-compute; omnitrace -> rocprofiler-trace; kineto@f10dcd9a pattern: enable CONCURRENT_KERNEL+RUNTIME+DRIVER+OVERHEAD+EXTERNAL_CORRELATION, FlushAll(FORCED)
- Nsight Compute: sections (SpeedOfLight, Occupancy, WarpStateStats...), Python rules engine (NvRules); SOL interpretation: both compute+memory <~60-80% -> latency bound -> check occupancy/warp stalls; roofline docs
- cuptisamples@794cac63 activity_trace_async = canonical minimal tracer (RegisterCallbacks L2109, FlushAll L2145)
### Sibling overlap survey (direct reads, this session)
- polykernel (/home/mei/projects/polykernel): out-of-tree MLIR-21 compiler (11 passes, lit suite) + portable CUDA/HIP kernel templates (nvcc sm_80/sm_90 + hipcc gfx1101, RUNS on local RX 7800 XT, golden-validated) + GPU-free analysis (lib/Analysis: PtxasParser, Occupancy sm_80/sm_90, Roofline, AmdIsaAnalyzer VGPR/SGPR/LDS/spills gfx1101/gfx942, KernelReport) + correctness-gated autotuner (lib/Autotune: ConfigSpace, TuningCache, AmdTuningDb, Benchmark; JSON cache keyed (GPU,op,shape)) + runtime (DeviceDetect, KernelCache, HipRuntime) + Cerebras-style dataflow sim + Modal deploy + HTML dashboards. README carries resume bullets. Cloud NVIDIA numbers via owner-gated rentals (RunPod/Modal)
- polykernel devenv convention (devenv.nix:1-112): flake.nix sets allowUnfree/cudaSupport/rocmSupport/cudaCapabilities (devenv module system has NO nixpkgs.config option - verified by them on v1.11.2/v2.1.2); packages = llvmPackages_21.{mlir,tblgen,llvm,clang} + lit + cmake + ninja + cudaPackages_12_6.{cuda_nvcc,cuda_cudart,cuda_nvrtc,libcublas,cuda_cuobjdump} + rocmPackages.{clr,rocminfo,rocm-smi} + gtest + python3(numpy,ml-dtypes,pytest,modal); CC/CXX=llvmPackages_21.clang; CUDA_HOME=makeSearchPathOutput dev; LD_LIBRARY_PATH includes /run/opengl-driver
- tt-flow (TensixFlow): MLIR compiler for Tenstorrent-style spatial accelerators (NOT GPUs) - no tool overlap; shares conventions (nix develop LLVM 20.1.8, lit/FileCheck, gtest/ctest, self-contained HTML reports, python golden via uv)
- tensors (tensorforge): C++20/CUDA 12.6 tensor+autograd runtime, Bazel 9.1.1, CUDA kernels (gemm naive/tiled/optimized, softmax, layernorm, im2col), benchmarks vs cuBLAS + nsys/ncu profiling docs; NVIDIA numbers from rented H100 (PrimeIntellect/lambdalabs). Potential source of benchmark kernels / capture targets; different build system (Bazel)
- OVERLAP VERDICT: (2 analyzer) PARTIAL - polykernel parses ptxas -v summaries only, no real PTX parsing/dataflow (coalescing/banks/divergence/tensor-core/latency all NEW); (3 autotuner) PARTIAL - polykernel tunes own emitted kernels on bounded grid, NOT standalone Triton-style generator over 6 kernel families; (4 verifier) NONE; (5 capture/replay) NONE; (6 LLVM passes) NONE (polykernel passes are op-level fusion/layout); (7 profiler) NONE (polykernel is compile-time only)
- User's established patterns: resume-bullet READMEs, golden correctness harnesses (NumPy bf16, cosine>=0.999), GPU-free analysis where hardware absent, owner-gated cloud GPU rentals, self-contained HTML reports, JSON caches, docs/ dirs, lit+gtest+ctest layers

## Decisions (with rationale)
- MERGE RECOMMENDATION (the user-delegated call): MERGE the six tools into ONE umbrella project. Rationale:
  1. Shared substrate is large and load-bearing: per-arch constants DB (NVIDIA sm_75..sm_120 + AMD gfx90a/gfx942/gfx1101), PTX + AMDGPU-asm parsers, occupancy engine, roofline engine, bench harness, report engine (JSON+text+self-contained HTML), NumPy golden harness, CUDA/HIP runtime abstraction - each used by 3-6 of the tools.
  2. The tools form one optimization LOOP, not six islands: profile(C6) -> capture(C4) -> analyze(C1) -> transform(C5)/generate(C2) -> verify(C3) -> benchmark(C2 harness) -> report. Recent literature validates exactly this loop as the valuable artifact (Mneme SC'23, R3 2026, Kerncap 2026).
  3. Cross-tool multipliers only exist if merged: analyzer pre-screens autotuner variants GPU-free before compile (polykernel proved GPU-free analysis); profiler "why slow" hints invoke the analyzer on the kernel's PTX; verifier gates autotuner output (polykernel already does correctness-gating; verifier adds safety); capture feeds real app kernels to the tuner.
  4. User's repo convention (one project = one repo + devenv) is satisfied by ONE repo with a single devenv env (monorepo imports available if per-module envs wanted later).
  5. Resume story is stronger unified: "end-to-end GPU kernel analysis/verification/tuning suite" vs six fragments.
  Shape: libs/ (arch-model, parsers, occupancy, roofline, bench, report, golden, gpu-runtime) + tools/ (analyze, tune, verify, capture, pass, profile) + docs/research (the briefs) + tests + reports. Dependency waves keep each module independently shippable.
- Separate-projects fallback (if user vetoes merge): six sibling repos each with own devenv; shared core duplicated or published as a 7th "core" repo - strictly more work, same scope.
- HARDWARE STRATEGY (user-confirmed): dual-path with local AMD runtime loop. CUDA first-class per spec; HIP/ROCm path per module; local runtime validation on gfx1101 (ROCm 7.2.3 binaries from nixpkgs); NVIDIA numbers via owner-gated rentals only (never auto-triggered). CUPTI-dependent modules (capture, profiler) develop+unit-test against headers/records locally; their NVIDIA runtime paths validate on rentals; their ROCm counterparts (rocprofiler-sdk) validate locally.
- TEST STRATEGY (user-confirmed): 4-layer convention - lit/FileCheck (IR/pass/analyzer/verifier artifacts), gtest/ctest (unit), NumPy bf16 golden harness cosine>=0.999 (kernel correctness), generated-code compile checks; implementation+test = one todo; all QA agent-executed, zero human intervention.

## Scope IN
- All six tools with their FULL user-specified feature lists (analyzer: 10 analyses + bonus suggestion engine; autotuner: 6 kernel families + full search space + roofline + occupancy estimator + JSON DB + comparison reports + reproducibility scripts; verifier: all 8 check classes + CUDA/HIP C++ -> LLVM IR/MLIR -> verifier -> backend pipeline; capture/replay: LD_PRELOAD CUDA interception + all capture fields + standalone C++ reproducer + replay validation + timing hooks + HIP stretch; LLVM passes: all 7 pass types + lit/FileCheck + before/after IR + PTX + benchmarks; profiler: all 7 features incl. source-to-PTX mapping and compiler hints)
- Shared core libs + foundation (git init, devenv latest, CMake, CI, docs/research move, README with resume bullets per user convention)
- CUDA-first tooling surfaces (nvcc/ptxas/PTX/CUPTI) with HIP/ROCm paths per module (stretch where user marked stretch)
- Local AMD gfx1101 runtime validation + owner-gated NVIDIA rental validation (per hardware-strategy answer)

## Scope OUT (Must NOT have)
- No Cerebras/dataflow-fabric simulation (polykernel already has it; not requested here)
- No PyTorch/ONNX/StableHLO frontends or op zoos (not requested)
- No serving/deployment endpoints (Modal/FastAPI etc. - polykernel territory; not in these six specs)
- No LLM-driven kernel generation/optimization (Sakana-style; not requested)
- No claims of beating cuBLAS/vLLM/production stacks (machinery demonstration, per polykernel's honest framing)
- No NVBit/Nsight-PerfWorks-dependent features (EULA/closed APIs) - CUPTI Activity/Callback/Checkpoint + rocprofiler-sdk only
- No source builds of LLVM/ROCm (nixpkgs binaries only; disk budget 112G)
- No auto-triggered cloud spend; rentals stay owner-gated

## Open questions
- RESOLVED: merge vs separate -> MERGE recommended (see Decisions); presented at the gate
- RESOLVED: no "project 1" exists in ~/projects or session history; numbering is external
- RESOLVED: overlap check done (direct reads) - see Sibling overlap survey
- RESOLVED (user answered): hardware/runtime strategy = DUAL-PATH with local AMD runtime loop: CUDA first-class per spec (nvcc/PTX/CUPTI tooling) + HIP/ROCm path per module; runtime validation local on RX 7800 XT via nixpkgs ROCm; NVIDIA numbers via owner-gated rentals (polykernel pattern)
- RESOLVED (user answered): test strategy = 4-layer sibling convention: lit/FileCheck + gtest/ctest + NumPy bf16 golden (cosine>=0.999) + generated-code compile checks; tests WITH each todo (impl+test = one todo)

## Approval gate
status: approved-plan-complete
<!-- When exploration is exhausted and unknowns are answered, set status: awaiting-approval. -->
<!-- That durable record is the loop guard: on a later turn read it and resume at the gate instead of re-running exploration. -->
- APPROVED by user ("approve") after brief; both interview forks resolved (dual-path AMD runtime loop; 4-layer test convention).
- Plan WRITTEN: .omo/plans/gpu-kernel-suite.md - 57 todos across waves 0-10 + F1-F4; Metis gap analysis (bg_34e13437, 16m27s) folded: 6 CRITICAL (capture dual-mechanism CUPTI-primary, verifier host+device ingestion, fixture materialization, devenv packages cuda_cupti/cuda_nvdisasm/rocprofiler-sdk + cuda_capabilities, AMD occupancy oracle + RDNA3 1536-VGPR resolution, HIP capture todo), 6 HIGH, 8 MEDIUM, 6 LOW all incorporated; validated non-issues recorded (nixpkgs llvm-21.1.8: NVPTX targets + LLVM_ENABLE_PLUGINS ON).
- pending-action: DELIVERY QUESTION - start work now vs dual high-accuracy review first (CLEAR, review_required=false -> owner chooses).
- Next workflow action on "start": user runs /start-work (execution belongs to the worker session, never this planner). On "review": run dual review (momus + independent oracle) against .omo/plans/gpu-kernel-suite.md per full-workflow.md intake contract.
