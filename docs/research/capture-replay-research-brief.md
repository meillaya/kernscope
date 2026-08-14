# Kernel Capture-and-Replay for CUDA/HIP — Research Brief
Date: 2026-08-13 · Scope: papers, GitHub repos, NVIDIA/AMD docs · All links verified this session

## Executive summary

The exact system you describe — intercept launches, snapshot device memory, emit standalone
reproducers, replay+validate — has been built **twice in the last 14 months**, once per vendor:

| System | Vendor target | Paper | Code | License | Status |
|---|---|---|---|---|---|
| **Mneme** (+ R³ on top) | CUDA + HIP | SC'23; arXiv:2604.11109 (2026) | github.com/Olympus-HPC/Mneme | Apache-2.0 w/ LLVM exceptions | Active (Aug 2026), LLNL |
| **Kerncap** (in IntelliKit) | AMD (HIP + Triton) | arXiv:2605.03208 (May 2026) | github.com/AMDResearch/intellikit | MIT | Active (Jul 2026), AMD Research |

On NVIDIA, the building blocks are mature and official (CUPTI Callback/Activity/Checkpoint
APIs, CUDA_INJECTION64_PATH, Nsight Compute kernel replay), but **no open tool emits editable
standalone C++ reproducers for captured CUDA kernels** — that specific combination is the open
niche (Kerncap's authors make the same claim for their AMD-native positioning).

---

## 1. CUPTI (CUDA Profiling Tools Interface)

**Availability.** CUPTI ships inside every CUDA Toolkit at
`/usr/local/cuda-<ver>/extras/CUPTI/` (dynamic lib on all platforms; static on Linux x86_64
and arm64 SBSA). It is *not* added to `LD_LIBRARY_PATH` automatically.
→ https://docs.nvidia.com/cupti/index.html (v2026.2.1, updated Jun 2026)

**Callback API (synchronous enter/exit interception).**
- `cuptiSubscribe(&handle, callback, userdata)` then `cuptiEnableCallback(enable, sub, domain, cbid)`.
  Domains include `CUPTI_CB_DOMAIN_RUNTIME_API` and `CUPTI_CB_DOMAIN_DRIVER_API`; cbids are per-API
  (e.g. `CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000` in `cupti_runtime_cbid.h`,
  `CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel` in `cupti_driver_cbid.h`).
- Each callback receives `CUpti_CallbackData` with `callbackSite` = `CUPTI_API_ENTER` /
  `CUPTI_API_EXIT`, `functionName`, **`functionParams` (pointer to the actual API arguments —
  for cudaLaunchKernel this includes grid/block dims, args array, stream)**,
  `functionReturnValue` (valid only at EXIT), **`symbolName` (kernel name, valid for launch
  callbacks)**, `context`, and correlation data. NVIDIA documents the param structs in
  `generated_cuda_runtime_api_meta.h` / `generated_cuda_meta.h`.
  → https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html
  → https://docs.nvidia.com/cupti/api/structCUpti__CallbackData.html
- This is the sanctioned way to see launch arguments at runtime (activity records do NOT carry
  argument values — see §5).

**Activity API (asynchronous records).**
- `CUPTI_ACTIVITY_KIND_KERNEL` / `CONCURRENT_KERNEL` records (`CUpti_ActivityKernel12` current)
  carry: `deviceId`, `contextId`, **`streamId`**, **`gridX/Y/Z`, `blockX/Y/Z`**, static/dynamic
  shared memory, registers, `correlationId`, `queued/submitted/start/end` timestamps, demangled
  name. → https://docs.nvidia.com/cupti/api/structCUpti__ActivityKernel12.html
- `correlationId` joins the runtime/driver API record (`CUPTI_ACTIVITY_KIND_RUNTIME` /
  `_DRIVER`) to the GPU kernel/memcpy/memset records it produced — the standard join key for a
  capture tool. → https://docs.nvidia.com/cupti/main/main.html (§2.3)
- Allocation tracking without interception: memory activity records (`CUpti_ActivityMemory4`)
  report `memoryOperationType` (alloc/free), `memoryKind`, **`address` and `bytes`**, deviceId,
  streamId. → https://docs.nvidia.com/cupti/api/structCUpti__ActivityMemory4.html
- Graph observability: resource callbacks `CUPTI_CBID_RESOURCE_GRAPH_CREATED`,
  `GRAPHNODE_CREATED`, `GRAPHEXEC_CREATED`, … plus `CUpti_ActivityGraphTrace` records;
  `CUpti_ModuleResourceData`/`CUpti_ActivityModule` expose module (cubin) load events.
  → callback API page above.

**Checkpoint API (device-state snapshot/restore — the replay primitive).**
`cuptiCheckpointSave` / `cuptiCheckpointRestore` / `cuptiCheckpointFree`
(→ https://docs.nvidia.com/cupti/api/group__CUPTI__CHECKPOINT__API.html).
Introduced in **CUDA 11.5**: *"A new CUPTI Checkpoint API is introduced, enabling automatic
saving and restoring of device state, and facilitating development of kernel replay tools.
This is helpful for User Replay mode of the CUPTI Profiling API, but is not limited to use
with CUPTI."* → https://docs.nvidia.com/cupti/release-notes/release-notes.html (§1.1.29.7).
Requires Turing+ (CC ≥ 7.5). This is NVIDIA's official answer to "snapshot device memory state".

**Injection mechanism (no app changes, no LD_PRELOAD needed).**
Set `CUDA_INJECTION64_PATH=<tool.so>`; during CUDA init the driver loads the library and calls
its exported `InitializeInjection()` (there is also `InitializeInjectionNvtx`/`2` for NVTX via
`NVTX_INJECTION64_PATH`). This is how Compute Sanitizer and NVIDIA's own profilers attach, and
it is the mechanism the CUPTI `injection` samples use.
→ NVIDIA forum (official staff): https://forums.developer.nvidia.com/t/segment-fault-when-using-cupti-profiling-injection-with-nvidia-triton-and-tensorrt/203927/4
→ Worked sample: https://github.com/eunomia-bpf/cupti-tutorial/blob/master/cupti_trace_injection/README.txt
  (and profiling_injection/ in the same repo, which wires Kernel-Replay-mode metric collection)

**Limitations (verified):**
- Legacy single-subscriber rule: `cuptiSubscribe` returns
  `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` if another CUPTI client is active **or the
  app runs under Nsight Systems/Compute, cuda-gdb, Compute Sanitizer**. Multiple concurrent
  subscribers only via `cuptiSubscribe_v2` on **CUDA 13.3+ / driver r610+**.
  → callback API page; https://docs.nvidia.com/cupti/main/main.html (§2.2)
- Hardware-counter profiling (Profiler/Range-Profiler/PC-Sampling/Checkpoint/SASS-metrics)
  needs CC ≥ 7.5, and on CC-devtools (confidential compute) mode several of these APIs error out;
  MIG *shared* compute instances can't collect metrics; vGPU needs `enable_profiling`; WSL2
  supports tracing (Turing+) but profiling only on Win11/WSL2 with extra setup.
  → https://docs.nvidia.com/cupti/special-configurations/special-configurations.html
  → release notes (CC-devtools restrictions)
- GPU perf counters require elevated permissions unless non-admin profiling is enabled
  (`NVreg_RestrictProfilingToAdminUsers`; see https://developer.nvidia.com/ERR_NVGPUCTRPERM) —
  flagged in the Nsight Compute Profiling Guide.
- The old Profiling API (cupti_profiler_target) is **deprecated as of CUDA 13.0** in favor of
  the Range Profiling API → CUPTI index page.
- Activity records never contain kernel argument values; only the Callback API sees those.

**Framework precedent (CUPTI is the industry interception layer):**
- PyTorch Kineto (BSD-3, Meta; active): `cuptiSubscribe` in
  https://github.com/pytorch/kineto/blob/main/libkineto/src/CuptiCallbackApi.cpp
- TF/XLA profiler (Apache-2.0): CUPTI wrapper incl. CUDA 13.2+ V2 multi-subscriber support:
  https://github.com/tensorflow/tensorflow/blob/master/third_party/xla/xla/backends/profiler/gpu/cupti_tracer.cc
- cupti-python (official Python bindings, Linux only): https://docs.nvidia.com/cupti-python/

---

## 2. LD_PRELOAD interception of libcudart/libcuda — prior art & pitfalls

**GPGPU-Sim (the classic full-replacement approach).**
GPGPU-Sim does not `dlsym(RTLD_NEXT)` — it ships a **complete reimplementation of the CUDA
runtime API** (`libcuda/cuda_runtime_api.cc`, derived from NVIDIA's CUDA 1.1 header) and makes
apps bind to its `libcudart.so` via environment setup (LD_LIBRARY_PATH override), then
functionally simulates PTX.
- `cudaLaunchKernel` → https://github.com/gpgpu-sim/gpgpu-sim_distribution/blob/a4ce3feac901c97a4b4601f679e43cf3589c79de/libcuda/cuda_runtime_api.cc#L2928-L2935
- launch arg walk (`entry->get_param_config(i)` gives per-arg size/offset from PTX parsing) →
  same file #L2037-L2062
- **allocation-size tracking**: `g_mallocPtr_Size[(unsigned long long)*devPtr] = size;` inside
  its `cudaMalloc` → same file #L1065-L1089 (this is the canonical "learn sizes by intercepting
  cudaMalloc" heuristic).
- It also implements GPU **checkpoint/restore** (global/local/shared mem, registers, SIMT stack)
  for fast-forward simulation → `checkpoint.md` in repo root.
- License: BSD-3-clause-style (UBC) with an explicit note that `libcuda/cuda_runtime_api.cc`
  is derived from NVIDIA's CUDA Toolset → COPYRIGHT file. 1.7k★; last push Feb 2025 —
  **semi-active research software** (Accel-Sim framework builds on it).

**NVBit (SASS-level binary instrumentation, no source/recompile).**
NVlabs/NVBit (344★, pushed Apr 2026 — **maintained**). Dynamic instrumentation of GPU SASS:
tools see each `__global__`/`__device__` function at module load and can inject device-side
instrumentation calls before/after any SASS instruction. Tools are `.so`s built against
`libnvbit.a` (see e.g. Accel-Sim's tracer Makefile:
https://github.com/accel-sim/accel-sim-framework/blob/dev/util/tracer_nvbit/tracer_tool/Makefile).
MICRO'19 paper: https://github.com/NVlabs/NVBit/releases/download/v1.0/MICRO_19_NVBit.pdf
Supports CC 3.5–12.1, CUDA ≥ 12.0, driver ≤ 575.xx (README).
**License caveat:** *"NVBit is covered by the same End User License Agreement as the NVIDIA CUDA
Toolkit… a research prototype… provided as-is"* — i.e. **not open source**; commercial use
requires NVIDIA research licensing (README).
Relevance: instruction-accurate tracing / memory-reference tracing (Accel-Sim trace-driven
simulation), not argument/buffer capture.

**cuda-api-intercept style repos.**
- `XpuOS/cuda-intercept` (Apache-2.0, pushed Aug 2025, small): toolkit for intercepting CUDA
  **driver** APIs — generate wrappers, inject code before/after or replace driver functions;
  extracted from XSched (OSDI'25: https://www.usenix.org/conference/osdi25/presentation/shen-weihang).
  https://github.com/XpuOS/cuda-intercept
- `argyrosan/cuda_interception` (2023, no license, dormant) — minor.
- `facebookexperimental/CUTracer` (MIT): NVBit-based CUDA tracing tool from Meta —
  https://github.com/facebookexperimental/CUTracer
- Tally (arXiv:2410.07381) is a recent production-style example of a transparent virtualization
  layer intercepting kernel-launch APIs + PTX rewriting for scheduling.

**PyTorch/TensorFlow.** Both deliberately avoid LD_PRELOAD and use CUPTI (§1 links) — a strong
signal that CUPTI/injection is the robust path on NVIDIA; LD_PRELOAD of libcudart is mostly a
simulator/VM trick (GPGPU-Sim, rCUDA-style remoting).

**Practical pitfalls (consolidated from the above):**
1. **Driver-vs-runtime split**: intercepting only `libcudart.so` misses apps/libraries that call
   `libcuda.so` directly (cuBLAS/cuDNN internals, Triton, PyTorch custom ops increasingly use the
   driver API). Runtime calls funnel into the driver, so intercepting `libcuda.so`
   (`cuLaunchKernel`) is the lower choke point — but you then lose runtime-level sugar and must
   handle `cuGetProcAddress`-based dynamic dispatch (apps can fetch driver entry points by
   address, bypassing your PLT hooks; see CUDA Driver API docs §"Driver Entry Point Access").
2. **dlsym(RTLD_NEXT) fragility**: must resolve the *real* symbol lazily and cache it; watch for
   versioned symbols (`cuLaunchKernel` vs `cuLaunchKernelEx`, `cudaLaunchKernel` vs
   `cudaLaunchKernelExC`, graph launches) and for the runtime itself being dlopen'd later.
3. **Init-order races**: if any other library initializes CUDA before your preload constructor
   runs, tables/handles are already live. AMD documents exactly this failure mode for its
   intercept-table API (ROCm-aware MPI example) —
   https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/intercept_table.html
4. **TLS/context state**: current context is per-thread (primary-context sharing, per-thread
   default stream); a capture layer must track `cuCtxSetCurrent`/runtime migration, and must not
   call CUDA APIs from arbitrary threads during ENTER callbacks (CUPTI warns context can change
   between enter and exit of a runtime call — CUpti_CallbackData docs).
5. **Symbol interposition vs static linking**: apps statically linking cudart (`-cudart static`)
   cannot be LD_PRELOAD-intercepted at all — injection (`CUDA_INJECTION64_PATH`) or CUPTI still
   work because they hook from inside the driver.

---

## 3. Recent (2023–2026) kernel isolation/replay work — the "similar recent work"

**Mneme — the CUDA/HIP record-replay engine (LLNL).**
- Repo: https://github.com/Olympus-HPC/Mneme (Apache-2.0 + LLVM exceptions, `LLNL-CODE-2000766`,
  pushed Aug 2026 — **actively maintained**; 24★).
- Paper: Parasyris, Georgakoudis, Rangel, Laguna, Doerfert — *"Scalable Tuning of (OpenMP) GPU
  Applications via Kernel Record and Replay"*, SC '23 (bibtex in README).
- Architecture (3 phases, per docs/concepts/overview.md and mneme-phases.md):
  1. **Instrument (build time)**: a Clang/LLVM pass (built on **Proteus**, LLNL's JIT layer —
     https://github.com/Olympus-HPC/Proteus, Apache-2.0, active) embeds each kernel's LLVM IR +
     device globals into the binary and **redirects all kernel launches through the Proteus
     runtime** (transparent to the app).
  2. **Record (runtime)**: `mneme record` preloads a library that intercepts launches and
     allocations. The CUDA preload overrides `cudaMalloc`/`cudaMallocManaged`
     (https://github.com/Olympus-HPC/Mneme/blob/4442a493f7cdb3844d3c16c43ffabc10d30828df/src/MnemeRecordPreloadCUDA.cpp#L40-L46)
     and runs a **virtual page-based device-memory manager** that tracks size/address/lifetime of
     every allocation — "precisely identify which regions of device memory may be accessed by a
     kernel". Before each launch it saves a **prologue** snapshot, after it an **epilogue**
     snapshot (device heap + globals + kernel-arg memory). Artifacts: JSON metadata, `.bc` LLVM
     IR, `.mneme` memory blobs.
  3. **Replay/Tune**: reconstructs the device VA space, re-executes the kernel standalone,
     verifies against the epilogue using **`mneme::annotate()` tolerance policies** (dtype,
     threshold, norm); tunes block/grid + compiler passes via Optuna; JIT-recompiles variants
     through Proteus.
- Documented limitations (docs/limitations.md): no managed memory; `__device__` globals that
  hold addresses of other globals unsupported; CUDA RDC untested.

**Record-Remix-Replay (R³) — arXiv:2604.11109 (Apr 2026, LLNL).**
https://arxiv.org/abs/2604.11109 — LLM evolutionary search + Bayesian opt over source, compiler
passes, and launch config, powered by a record-replay engine built on Mneme/Proteus. Key
architecture points: recording via a **preload library intercepting kernel launches and device
memory ops**, storing per replay-unit: kernel LLVM IR, launch config, kernel arguments,
pre-execution device state (prologue), post-execution state (epilogue). Contributions:
**persistent replay server** (state stays resident across thousands of evaluations), in-memory
LLVM IR transformation, bitwise-exact correctness by default with opt-in tolerance predicates,
CUDA **and** HIP support. Result: 28.4% compute-time reduction on QUDA; ~10× faster than
evolutionary search without replay. Replay engine contributions are in Mneme; R³ itself
"planned to be open-sourced".

**Kerncap — arXiv:2605.03208 (May 2026, AMD) — the closest complete system.**
https://arxiv.org/abs/2605.03208 · Code: https://github.com/AMDResearch/intellikit
(MIT, 31★, pushed Jul 2026, docs at rocm.docs.amd.com/projects/intellikit — **actively
maintained**; part of AMD's agent-first "IntelliKit" toolbox with MCP servers/skills).
Architecture (§4 of paper):
- **Interception**: `libkerncap.so` via LD_PRELOAD registers through **rocprofiler-sdk**
  (`rocprofiler_configure` + `rocprofiler_at_intercept_table_registration`), receives the live
  `HsaApiTable`, and hooks: queue creation (→ `hsa_amd_queue_intercept_create` per-packet
  callback), memory alloc/VMEM APIs (ptr→size map), symbol lookups (handle→mangled name), code
  object readers (captures HSACO blobs). For post-execution capture it **replaces the dispatch
  completion signal**, blocks on it, snapshots, then releases the app.
- **Memory**: *address-space closure* — snapshot **every tracked allocation at its original
  virtual address** (`hsa_memory_copy` in 64 MiB chunks; `memory_regions.json` + per-region
  `.bin`), so embedded `T**` pointer graphs replay for free — no DWARF, no pointer chasing, no
  per-argument size inference. Handles `__constant__`/module variables (Kokkos case) via symbol
  enumeration. Snapshots from 152 MB to 30 GB (vLLM MoE weights).
- **Reproducers**: self-contained project with `capture/` (dispatch.json, kernarg.bin,
  kernel.hsaco, memory regions), editable source + headers, a **Clang VFS overlay** so the exact
  original compile command rebuilds the kernel, and a Makefile with
  `run / recompile / run-variant / validate-variant`. Triton kernels get tuning-pinned Python
  reproducers (autotuner config bound into the artifact).
- **Validation**: byte-exact memory diff (HIP), numpy.allclose (Triton), smoke test.
- Result: 13.6× faster edit-recompile-validate loop on llama.cpp.
- Limitations (§7): single kernel at a time, single GPU, no host-side state, snapshot race if
  app frees buffers before copy, autotuner configs are hardware-specific.

**NVIDIA's own kernel replay (Nsight Compute).**
Three replay modes (`--replay-mode`): **Kernel** — per-kernel transparent re-execution; first
pass saves *all GPU memory the kernel can access*, the written subset is restored before each
subsequent pass (save target chosen device→host fallback); **Application** — rerun the whole
deterministic app per pass (no save/restore; supports host-interdependent kernels; matching via
`--app-replay-match`, default kernel-name+grid); **Range** — capture/replay a whole range of API
calls + kernels with host/device memory save-restore (CUPTI `cu(da)ProfilerStart/Stop` or NVTX
markers define ranges). Cache-flush/clock-control options per replay iteration.
→ https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html (Kernel/Application/Range
Replay sections) and https://docs.nvidia.com/nsight-compute/NsightCompute/index.html (Replay
Mode UI/CLI). The CUPTI Checkpoint API (§1) is the third-party interface to this machinery.

**Adjacent 2023–2026 work** (context for the tuning-loop motivation):
- KernelBench & GEAK (LLM kernel generation) — consumers of isolated validated replay harnesses
  (cited by Kerncap §6).
- SIP (arXiv:2403.16863): SASS-schedule autotuning by mutating cubins intercepted from Triton.
- KPerfIR (arXiv:2505.21661): in-compiler (MLIR/Triton) instrumentation + trace replay.
- Kernel Tuner + HIP extension (arXiv:2407.11488): standalone kernel benchmarking/autotuning
  (synthetic inputs, no capture) — github.com/benvanwerkhoven/kernel_tuner.
- TORCH_COMPILE_DEBUG emits standalone `output_code.py` Triton reproducers (shapes/dtypes only,
  no runtime tensor data) — noted by Kerncap §6 as de-facto reproducer prior art.
- Process-level RR (rr, CRIU) does NOT capture GPU state — Kerncap §6.

---

## 4. CUDA Graph stream capture as an alternative capture mechanism

Semantics (CUDA C++ Programming Guide §"Stream Capture", 12.6 text quoted):
- `cudaStreamBeginCapture(stream, mode)` puts the stream in capture mode: *"work launched into
  the stream is not enqueued for execution. It is instead appended to an internal graph"*;
  `cudaStreamEndCapture` returns the `cudaGraph_t`. Works on any stream except legacy NULL
  stream; capture modes Global / ThreadLocal / Relaxed; `cudaStreamIsCapturing` queries state;
  `cudaStreamBeginCaptureToGraph` captures into an existing graph.
- Cross-stream deps via `cudaEventRecord`/`cudaStreamWaitEvent` extend the capture graph
  (fork-join; all streams must rejoin the origin stream before EndCapture).
- **Prohibited/unhandled**: synchronizing or querying captured streams/events; any use of the
  legacy stream while any stream in the context is capturing; **synchronous APIs like
  `cudaMemcpy()` are invalid during capture** (they enqueue to the legacy stream) — CUDA prefers
  errors over silently dropping dependencies.
  → https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html#stream-capture
- **What it captures**: the DAG of operations (kernel launches with grid/block/shared/stream,
  memcpys/memsets, host fns, events, child graphs) — **NOT memory contents**. Buffer pointers
  are baked in as addresses; data must be separately snapshotted. Allocations during capture must
  be stream-ordered (`cudaMallocAsync`). HIP mirrors this exactly, incl. the sync-API ban
  (hipMalloc/hipMemcpy/hipFree forbidden during capture; use *Async variants):
  → https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- Verdict for your system: stream capture is a great *launch-topology* recorder (and CUPTI
  exposes graph/node callbacks + `CUpti_ActivityGraphTrace`), but it cannot replace state
  snapshotting — you still need §5.

---

## 5. State snapshot problems: buffer sizes, deep pointers, module handles

**Why sizes are hard.** `cuLaunchKernel(f, …, void** kernelParams, void** extra)`: *"If f has N
parameters, then kernelParams needs to be an array of N pointers. Each of kernelParams[0]…
must point to a region of memory from which the actual kernel parameter will be copied. The
number of kernel parameters and their offsets and sizes do not need to be specified as that
information is retrieved directly from the kernel's image."*
→ https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__EXEC.html
So at the API boundary you see N opaque pointers-to-values; a pointer argument is just an 8-byte
value with **no size**. Two official escape hatches:
- **`cuFuncGetParamInfo(func, paramIndex, &paramOffset, &paramSize)`** — driver API that returns
  per-parameter offset/size from the kernel image (same page). This lets an interceptor learn
  argument *layout* (which slots are pointers vs scalars still needs type info or heuristics).
- CUPTI callback `symbolName` + `functionParams` at ENTER (§1) to pair layout with values.

**Heuristics used by prior tools (all verified above):**
1. **cudaMalloc tracking** — GPGPU-Sim's `g_mallocPtr_Size` map (§2 permalink); Mneme's preload
   allocator override + virtual page manager; Kerncap's hooks of
   `hsa_amd_memory_pool_allocate`/VMEM APIs → ptr→size map.
2. **Allocation activity records** — CUPTI `CUpti_ActivityMemory4` gives alloc/free with address
   and bytes without any interception (§1 link).
3. **Whole-address-space closure** — Kerncap snapshots *all* tracked allocations at original VAs;
   embedded pointer graphs (`T**`, vLLM MoE indirection) then work with zero type info. This is
   the strongest known solution to the deep-pointer problem (cost: snapshot size — 30 GB worst
   case in their eval; mitigated by chunked streaming).
4. **Dynamic access-based save** — Nsight Compute kernel replay saves all memory the kernel can
   access and restores the written subset (hardware-assisted determination, closed source).
5. **CUPTI Checkpoint API** — automatic save/restore of device state (CUDA 11.5+; per Kerncap,
   restore works via explicit per-allocation save/restore hooks).
6. **User annotations** — Mneme `mneme::annotate(ptr, policy)` (dtype/threshold/norm) for
   verification semantics; R³ relaxed predicates via source annotations; Kerncap tolerance mode
   for Triton. Annotations don't fix sizes but fix *validation* semantics.

**Struct/by-value arguments with embedded pointers:** no prior open CUDA tool marshals these
recursively; VA-faithful replay (Kerncap) and full-heap snapshot (Mneme) sidestep it. Mneme's
documented failure case: `__device__ int* g_ptr = &g_value;` (global holding global's address) —
relocation breaks replay (docs/limitations.md).

**Module/function handle capture (cubin-loaded kernels):**
- Kerncap hooks code-object load (`hsa_code_object_reader_create_from_memory`,
  `hsa_executable_load_agent_code_object`) and lazily builds kernel_object→symbol→executable→
  HSACO-blob chains; kernarg segment size queried via
  `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE` (paper §4.1–4.2).
- CUDA equivalents: intercept `cuModuleLoad/cuModuleLoadData(Ex)` to copy the cubin/PTX bytes;
  CUPTI resource callback `CUPTI_CBID_RESOURCE_MODULE_LOADED` + `CUpti_ModuleResourceData`
  (module data passed into resource callbacks) and `CUpti_ActivityModule`/`_Function` records;
  `cuFuncGetParamInfo` then resolves the param layout per CUfunction. For runtime-API kernels,
  capture the fatbinary via `__cudaRegisterFatBinary` interception (GPGPU-Sim reimplements this
  registration in its libcuda shim).

---

## 6. Standalone reproducer generation — prior art

| Tool | Emits | Editable source? | Language |
|---|---|---|---|
| **Kerncap** | full project: capture dir + `kernel_variant.cpp` + deps + `vfs.yaml` + Makefile (`run/recompile/run-variant/validate-variant`) | yes (Clang VFS overlay keeps original compile flags) | HIP, Triton |
| **Mneme** | replayable unit executed by Mneme runtime (JSON + `.bc` IR + `.mneme` blobs); not an emitted .cpp | n/a (IR-level, JIT via Proteus) | CUDA, HIP |
| **R³** | replay server + JSON deployment spec (transformations/flags/launch) | n/a | CUDA, HIP |
| **TORCH_COMPILE_DEBUG** | `output_code.py` (Inductor/Triton source; shapes/dtypes only) | yes | Triton |
| **Nsight Compute** | `.ncu-rep` (internal replay for counters; no source artifact) | no | CUDA |

**Gap:** for CUDA, **no open tool today emits a compilable standalone C++ file that replays a
captured kernel with its real inputs** — GPGPU-Sim simulates in-process, Mneme replays through
its own runtime, Nsight keeps state internal. Kerncap does exactly this but AMD-only (HSA/HSACO,
raw-HSA replay binary bypassing HIP). Your CUDA-side system's differentiator would be:
CUPTI/injection-based capture (no recompile, unlike Mneme's LLVM pass) + emitted C++ reproducer
(like Kerncap) + cuFuncGetParamInfo-driven arg layout + Checkpoint-API or malloc-map snapshots.

---

## 7. HIP/ROCm equivalents

- **roctracer** (MIT; repo now **[DEPRECATED] → moved into ROCm/rocm-systems monorepo**,
  May 2026): API tracing for HIP/HSA/rocTX — callback + activity-record style, the historical
  analog of CUPTI Activity/Callback. https://github.com/ROCm/roctracer
- **rocprofiler / rocprofiler-sdk** (MIT; also folded into ROCm/rocm-systems, active):
  rocprofiler-sdk is the modern API — **official runtime intercept tables**: tool exports
  `rocprofiler_configure`; runtimes hand their dispatch tables to rocprofiler via
  rocprofiler-register; tools call `rocprofiler_at_intercept_table_registration(cb,
  ROCPROFILER_HSA_TABLE | ROCPROFILER_HIP_RUNTIME_TABLE | ROCPROFILER_MARKER_CORE_TABLE, …)`
  and wrap function pointers ("chaining"). Docs warn about early-init races (call it inside
  `rocprofiler_configure`). Plus `hsa_amd_queue_intercept_create/register` for per-AQL-packet
  dispatch callbacks (what Kerncap uses), buffer/callback tracing services, and rocTX tracing.
  → https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/intercept_table.html
  → https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/api-reference/rocprofiler-sdk_api/modules/intercept_table.html
  → sample: samples/intercept_table in the rocprofiler-sdk source.
  `rocprofv3` is the CLI front end (Kerncap uses `rocprofv3 --kernel-trace` for ranking).
- **rocTX**: annotation/marker API; traced by rocprofiler-sdk (`ROCPROFILER_MARKER_CORE_TABLE`).
  The standalone ROCm/rocTX repo no longer resolves (404) — it lives in the rocm-systems monorepo
  consolidation (verified Aug 2026).
- **ROCdbgapi** (MIT, pushed Aug 2026 — **maintained**): the AMD Debugger API — dispatch
  interception, breakpoints, single-stepping support for rocgdb; the "ROCm debugger agent" you
  asked about. It's for interactive debugging, not capture-replay (same distinction Kerncap
  draws for cuda-gdb). https://github.com/ROCm/ROCdbgapi
- **hipGraph stream capture**: full parity with CUDA semantics incl. async-only memory ops
  during capture → https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html
- **hipLaunchKernel interception**: hipModuleLaunchKernel sits behind the HIP runtime's dispatch
  table — the supported hook point is the intercept-table mechanism above (LD_PRELOAD of
  libamdhip64.so with dlsym chaining also works in practice — that's Kerncap's transport — but
  AMD's blessed path is rocprofiler-sdk registration).
- **Kerncap/IntelliKit** (MIT, AMD Research, active) is the working end-to-end exemplar on ROCm
  (§3/§6) — including HSA-level kernarg decoding via `amd_comgr` code-object metadata
  (offset/size/value_kind per kernarg slot), which is AMD's answer to "no type info at runtime".

---

## Maintenance & license ledger (verified Aug 2026)

| Project | License | Last activity | Verdict |
|---|---|---|---|
| CUPTI (NVIDIA) | CUDA Toolkit EULA (proprietary) | ships w/ toolkit, docs Jun 2026 | active, supported |
| Nsight Compute replay | proprietary | current | active |
| GPGPU-Sim | BSD-3-style (UBC) + NVIDIA-derived-file notice | Feb 2025 | semi-active research code |
| NVBit | CUDA Toolkit EULA (research artifact, **not OSS**) | Apr 2026 (v1.7.x) | maintained, restrictive license |
| Accel-Sim framework | (uses NVBit + GPGPU-Sim) | ongoing | research |
| Mneme | Apache-2.0 + LLVM exceptions | Aug 2026 | **active** |
| Proteus | Apache-2.0 + LLVM exceptions | Aug 2026 | **active** |
| Kerncap/IntelliKit | MIT | Jul 2026 | **active** (AMD Research) |
| Kineto (PyTorch) | BSD-3 | Aug 2026 | active |
| TF/XLA CUPTI tracer | Apache-2.0 | current | active |
| XpuOS/cuda-intercept | Apache-2.0 | Aug 2025 | young/small (OSDI'25 lineage) |
| CUTracer (Meta) | MIT | — | niche |
| roctracer / rocprofiler(-sdk) | MIT | repos deprecated → ROCm/rocm-systems (active monorepo, Aug 2026) | active under new home |
| ROCdbgapi | MIT | Aug 2026 | active |
| eunomia-bpf/cupti-tutorial | tutorial repo | 2025–2026 | reference examples |

## Build recommendations (synthesis)

1. **Interception**: prefer CUPTI Callback API via `CUDA_INJECTION64_PATH` injection (no
   LD_PRELOAD fragility, works with `-cudart static`, sees both runtime+driver domains, gives
   `functionParams`+`symbolName` at ENTER/EXIT). Fall back to LD_PRELOAD+dlsym only for APIs
   CUPTI doesn't expose. Handle `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` gracefully;
   adopt `cuptiSubscribe_v2` where CUDA ≥ 13.3.
2. **Launch metadata**: Activity `CUpti_ActivityKernel12` for grid/block/stream/timestamps,
   joined by correlationId to the callback-captured arguments.
3. **Buffer sizes**: maintain an allocation map from CUPTI memory activity records (or
   intercepted cudaMalloc/cudaMallocAsync + cuMemMap family); use `cuFuncGetParamInfo` for param
   layout; snapshot the allocation *closure* at original VAs (Kerncap's trick) rather than
   chasing per-argument sizes; consider CUPTI Checkpoint API as the vendor-supported shortcut
   (CC ≥ 7.5, CUDA ≥ 11.5).
4. **Code capture**: intercept `cuModuleLoadData`/fatbinary registration to store cubins; map
   CUfunction→cubin+symbol; emit reproducers that reload via `cuModuleLoadData` so you never
   need the original build (Kerncap's VFS-overlay approach is the alternative when source
   editability matters).
5. **Reproducer**: emit standalone C++ (driver-API based) + binary blobs + JSON manifest;
   validate bitwise by default with opt-in tolerances (Mneme/R³ policy).
6. **Timing hooks**: CUPTI timestamps (queued/submitted/start/end) + CUDA events around replay;
   for counters, CUPTI Range Profiling API (Profiling API is deprecated in CUDA 13.0).
7. **HIP stretch**: reuse the same architecture on rocprofiler-sdk intercept tables +
   `hsa_amd_queue_intercept` (Kerncap is MIT — you can literally study/borrow its HSA layer),
   amd_comgr for kernarg layout, hipGraph capture mirroring CUDA.
