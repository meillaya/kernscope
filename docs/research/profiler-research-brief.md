# Tiny CUDA/ROCm Profiler with Compiler-Aware Diagnostics — Research Brief

Date: 2026-08-13. All CUPTI/CUDA struct fields below were verified by reading the actual
CUDA 13 headers (vendored copy of the CUDA 13.0 toolkit headers at
`terryye/cuda_GEMM@07ffeb5`, which matches the CUPTI 13.x docs) and the rocprofiler-sdk
headers at `ROCm/rocprofiler-sdk@3a924a0`.

---

## 1. CUPTI Activity API

### 1.1 KERNEL vs CONCURRENT_KERNEL

Both kinds use the *same* record struct; they differ only in collection behavior. From
`cupti_activity.h`
([permalink](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_activity.h#L208-L263)):

```c
/**
 * A kernel executing on the GPU. This activity kind may significantly change
 * the overall performance characteristics of the application because all
 * kernel executions are serialized on the GPU. Other activity kind for kernel
 * CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL doesn't break kernel concurrency.
 * The corresponding activity record structure is \ref CUpti_ActivityKernel10.
 */
CUPTI_ACTIVITY_KIND_KERNEL   = 3,
...
/**
 * A kernel executing on the GPU. This activity kind doesn't break
 * kernel concurrency. The corresponding activity record structure
 * is \ref CUpti_ActivityKernel10.
 */
CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL = 10,
```

**Verdict for a tiny profiler: always enable `CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL`.**
`KERNEL` serializes all launches (a behavior change, not just overhead). This is also what
PyTorch Kineto does ([CuptiActivityApi.cpp#L349](https://github.com/pytorch/kineto/blob/f10dcd9a601d0e761064614d6992a003745def7e/libkineto/src/CuptiActivityApi.cpp#L336-L380)).

### 1.2 The kernel record: CUpti_ActivityKernel9 (CUDA 12.x) / CUpti_ActivityKernel10 (CUDA 13.x)

CUDA 12.x reports `CUpti_ActivityKernel9`; CUDA 13.x reports `CUpti_ActivityKernel10`
(the old struct moved to `cupti_activity_deprecated.h`). Kernel10 adds only
`isDeviceLaunched` (+padding) at the tail; all fields relevant to us are identical.
Verified field list from
[cupti_activity.h#L2909-L3212](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_activity.h#L2909-L3212):

| Field | Type | Notes |
|---|---|---|
| `kind` | `CUpti_ActivityKind` | KERNEL or CONCURRENT_KERNEL |
| `cacheConfig` | union (requested:4/executed:4) | `CUfunc_cache` values |
| `sharedMemoryConfig` | `uint8_t` | `CUsharedconfig` value (bank mode) |
| `registersPerThread` | `uint16_t` | **registers used per thread** |
| `partitionedGlobalCacheRequested/Executed` | enum | |
| `start`, `end` | `uint64_t` ns | GPU timestamps; 0/0 = not collected |
| `completed` | `uint64_t` ns | incl. child kernels (CDP) |
| `deviceId`, `contextId`, `streamId` | `uint32_t` | |
| `gridX`, `gridY`, `gridZ` | `int32_t` | grid dims (blocks) |
| `blockX`, `blockY`, `blockZ` | `int32_t` | block dims (threads) |
| `staticSharedMemory` | `int32_t` bytes | compile-time smem |
| `dynamicSharedMemory` | `int32_t` bytes | launch-time smem |
| `localMemoryPerThread` | `uint32_t` bytes | **register spill indicator** |
| `localMemoryTotal` | `uint32_t` | deprecated in CUDA 11.8 |
| `localMemoryTotal_v2` | `uint64_t` | replacement |
| `correlationId` | `uint32_t` | joins to DRIVER/RUNTIME API record |
| `gridId` | `int64_t` | |
| `name` | `const char*` | mangled kernel name |
| `queued`, `submitted` | `uint64_t` ns | **off by default** — need `cuptiActivityEnableLatencyTimestamps()` |
| `launchType` | `uint8_t` | regular vs cooperative (`CUpti_ActivityLaunchType`) |
| `isSharedMemoryCarveoutRequested`, `sharedMemoryCarveoutRequested` | `uint8_t` | carveout % |
| `sharedMemoryExecuted` | `uint32_t` | smem size actually set by driver |
| `graphNodeId`, `graphId` | `uint64_t`/`uint32_t` | CUDA Graphs provenance |
| `shmemLimitConfig` | `CUpti_FuncShmemLimitConfig` | DEFAULT vs OPTIN (>48KB dynamic smem) |
| `pAccessPolicyWindow` | ptr | L2 persistence window |
| `channelID`, `channelType` | | HW channel (Green Contexts) |
| `clusterX/Y/Z`, `clusterSchedulingPolicy` | `uint32_t` | CC 9.0+ thread block clusters |
| `maxPotentialClusterSize`, `maxActiveClusters` | `uint32_t` | |
| `isDeviceLaunched` | `uint8_t` | **new in Kernel10 (CUDA 13)** |

**Occupancy: there is NO occupancy field (achieved or theoretical) in the kernel activity
record.** Verified by full read of the struct. Achieved occupancy requires HW counters
(PerfWorks/PM sampling). The standard trick — used by Nsight Compute for its `launch__*`
metrics — is to compute *theoretical* occupancy offline from `registersPerThread`,
`staticSharedMemory + dynamicSharedMemory`, block size, and device limits via the CUDA
Occupancy Calculator (see §2.3). The Nsight Compute Profiling Guide states:
"`launch__*` metrics are collected per kernel launch, and do not require an additional
replay pass. They are available as part of the kernel launch parameters ... or are computed
using the CUDA Occupancy Calculator"
([ProfilingGuide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

### 1.3 Enable / flush flow

Canonical flow (header decls in
[cupti_activity.h#L7424-L8055](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_activity.h#L7424-L7747),
semantics from the [CUPTI Usage docs](https://docs.nvidia.com/cupti/main/main.html)):

1. `cuptiSubscribe(&subscriber)` (optional for activity-only tracing, required for callbacks).
2. `cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted)` — CUPTI asks you for
   buffers (you allocate, e.g. 8 MB) and later hands back completed buffers.
3. `cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)` (+ `DRIVER`/`RUNTIME` for
   launch-side API records, `MEMCPY`/`MEMSET` for bytes-moved accounting).
4. Buffers are delivered automatically when full (heuristically; a CUPTI worker thread does
   the delivery to limit app-thread perturbation).
5. On demand: `cuptiActivityFlushAll(0)` returns buffers whose records are all complete;
   `cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED)` also returns buffers with
   incomplete records — use forced flush at session end. `cuptiActivityFlushPeriod`
   (CUDA 11.1+) controls the worker wake period.
6. Iterate records inside a buffer with `cuptiActivityGetNextRecord(buffer, size, &record)`;
   every record starts with `CUpti_ActivityKind kind` so you dispatch on it.
7. Join GPU kernel records to host launch sites via `correlationId`: the DRIVER/RUNTIME
   record (`CUpti_ActivityAPI`: `kind, start, end, processId, threadId, correlationId,
   returnValue`, cbid) carries the identical correlation ID.

Reference implementation (~a few hundred lines): CUPTI sample `activity_trace_async` +
`helper_cupti_activity.h`
([cuptisamples@794cac6, L2109](https://github.com/mmcloughlin/cuptisamples/blob/794cac63e7fc3bbc508d700f8609cdd0e5d6d155/samples/common/helper_cupti_activity.h#L2109-L2145)).
Note: the official NVIDIA/cupti GitHub repo no longer exists; `mmcloughlin/cuptisamples`
mirrors the shipped samples.

### 1.4 Overhead characteristics

- Activity tracing is asynchronous and buffered; CUPTI offloads buffer delivery to a worker
  thread ("To minimize the overhead, CUPTI wakes up the worker thread based on certain
  heuristics" — [CUPTI docs](https://docs.nvidia.com/cupti/main/main.html)).
- Behavioral overhead: `CUPTI_ACTIVITY_KIND_KERNEL` serializes kernels (header comment above);
  `CONCURRENT_KERNEL` does not.
- Tunables: `CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE`, `..._DEVICE_BUFFER_POOL_LIMIT`,
  `..._ZEROED_OUT_ACTIVITY_BUFFER`, `..._MEM_ALLOCATION_TYPE_HOST_PINNED`
  ([cupti_activity.h#L7037-L7198](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_activity.h#L7037-L7234)).
  Since CUDA 12.0 Update 1 CUPTI allocates the profiling/device buffers itself
  (header note at L7028).
- `queued`/`submitted` latency timestamps are off by default (`cuptiActivityEnableLatencyTimestamps`)
  — enabling them adds cost; keep off unless diagnosing launch latency.

### 1.5 CUDA 12 → 13 changes that matter

From the [CUPTI 13.0 release notes](https://docs.nvidia.com/cupti/13.0.0/release-notes/release-notes.html)
and [CUDA 13.0 toolkit release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/):

- Kernel record bumped `CUpti_ActivityKernel9` → `CUpti_ActivityKernel10`.
- **Dropped**: CUPTI Event API (`cupti_events.h`), Metric API (`cupti_metrics.h`),
  PC-Sampling Activity API (use `cupti_pcsampling.h`), SASS-level metrics
  (`cupti_sass_metrics.h`). Calls return `CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED` (=38,
  [cupti_result.h#L275](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_result.h#L275)).
  The activity kinds tied to them (`CUPTI_ACTIVITY_KIND_EVENT`, `_METRIC`,
  `_INSTANTANEOUS_*`) are marked unsupported in the CUDA 13 header.
- **Deprecated**: CUPTI Profiling API (`cupti_profiler_target.h`) and PerfWorks Metric API
  (`nvperf_host.h`) → replacements: **CUPTI Range Profiling API** (`cupti_range_profiler.h`)
  and Profiler Host API (`cupti_profiler_host.h`); **PM Sampling API**
  (`cupti_pmsampling.h`, Turing+) samples perf monitors at fixed intervals.
- **New**: multi-subscriber tracing (V2 APIs: `cuptiSubscribe_v2`, `cuptiActivityEnable_v2`
  with field selection, `cuptiActivityRegisterCallbacks_v2`) — beta in 13.0, production later.
- **nvprof and NVIDIA Visual Profiler were removed in CUDA 13.0.**
- `CUPTI_API_VERSION` is now `130001`
  ([cupti_version.h#L126](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cupti_version.h#L121-L126));
  it is distinct from `CUDA_VERSION`.

---

## 2. Driver/Runtime attributes & occupancy APIs

### 2.1 cuFuncGetAttribute / cudaFuncGetAttributes

Driver API enum `CUfunction_attribute`
([cuda.h#L933-L1065](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cuda.h#L933-L1065)):

| Enum | Val | Meaning |
|---|---|---|
| `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK` | 0 | max threads/block for this function |
| `CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES` | 1 | static smem bytes |
| `CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES` | 2 | constant memory bytes |
| `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES` | 3 | local (spill) bytes/thread |
| `CU_FUNC_ATTRIBUTE_NUM_REGS` | 4 | **registers per thread** |
| `CU_FUNC_ATTRIBUTE_PTX_VERSION` | 5 | PTX version (major*10+minor) |
| `CU_FUNC_ATTRIBUTE_BINARY_VERSION` | 6 | SASS arch version |
| `CU_FUNC_ATTRIBUTE_CACHE_MODE_CA` | 7 | |
| `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` | 8 | opt-in limit |
| `CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT` | 9 | |
| `CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET` | 10 | |
| `CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH/HEIGHT/DEPTH` | 11-13 | |
| `CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED` | 14 | |

Entry points: `cuFuncGetAttribute(int*, CUfunction_attribute, CUfunction)`
([cuda.h#L17190](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cuda.h#L17190)),
`cuFuncSetAttribute` (L17257), runtime `cudaFuncGetAttributes(cudaFuncAttributes*, func)` with
struct `cudaFuncAttributes { sharedSizeBytes, constSizeBytes, localSizeBytes,
maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA,
maxDynamicSharedSizeBytes, preferredShmemCarveout }`
([driver_types.h#L1661](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/driver_types.h#L1661-L1701)).

These are *static* (per-function, pre-launch) views; CUPTI's kernel record gives the same
data *per launch* (registers/smem can differ per launch when the driver adjusts them).

### 2.2 Occupancy APIs (driver)

All declared in `cuda.h`
([L21626-L21891](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cuda.h#L21626-L21891)):

- `cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction, int blockSize, size_t dynamicSMemSize)`
- `cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(..., unsigned flags)`
- `cuOccupancyMaxPotentialBlockSize(int *minGridSize, int *blockSize, CUfunction, CUoccupancyB2DSize blockSizeToDynamicSMemSize, size_t dynamicSMemSize, int blockSizeLimit)`
- `cuOccupancyMaxPotentialBlockSizeWithFlags(...)`
- `cuOccupancyAvailableDynamicSMemPerBlock(...)` (CUDA 11+)
- `cuOccupancyMaxPotentialClusterSize` / `cuOccupancyMaxActiveClusters` (CC 9.0+)

Runtime mirrors: `cudaOccupancyMaxActiveBlocksPerMultiprocessor(WithFlags)`,
`cudaOccupancyMaxPotentialBlockSize(WithFlags)`.

### 2.3 The header-only occupancy calculator (what a tiny profiler should embed)

`cuda_occupancy.h` ships a **header-only, runtime-free occupancy model**
([cuda_occupancy.h#L144-L150, L401-L440](https://github.com/terryye/cuda_GEMM/blob/07ffeb56ad4ce8c501542852f6d8a2c169b90a83/headers/cuda13/cuda_occupancy.h#L144-L150)):

```c
cudaOccError cudaOccMaxActiveBlocksPerMultiprocessor(
    cudaOccResult *result, const cudaOccDeviceProp *properties,
    const cudaOccFuncAttributes *attributes, const cudaOccDeviceState *state,
    int blockSize, size_t dynamicSmemSize);

struct cudaOccFuncAttributes {
    int maxThreadsPerBlock;
    int numRegs;             // from CUPTI registersPerThread
    size_t sharedSizeBytes;  // from CUPTI staticSharedMemory
    cudaOccPartitionedGCConfig partitionedGCConfig;
    cudaOccFuncShmemConfig shmemLimitConfig;  // maps to CUPTI shmemLimitConfig
    size_t maxDynamicSharedSizeBytes;
    int numBlockBarriers;
};
```

plus `cudaOccMaxPotentialOccupancyBlockSize` / `...VariableSMem` (launch configurator) and
`cudaOccAvailableDynamicSMemPerBlock`. `cudaOccDeviceProp` needs `computeMajor/Minor,
maxThreadsPerBlock/PerMultiprocessor, regsPerBlock/PerMultiprocessor, warpSize,
sharedMemPerBlock/PerMultiprocessor/PerBlockOptin, numSms, reservedSharedMemPerBlock`.
Feed it the CUPTI record fields and you get theoretical occupancy + **which resource is the
limiter** (regs vs smem vs threads vs barriers) — this is exactly how Nsight Compute derives
its no-replay `launch__occupancy_*` metrics. This is the core of the "occupancy estimate"
feature; no GPU counters needed.

---

## 3. Source ↔ assembly correlation

### 3.1 NVIDIA

- **Compile**: `nvcc -lineinfo` (`--generate-line-info`): "Generate line-number information
  for device code" — explicitly the profiling-oriented flag; does not disable optimizations
  ([nvcc docs §4.2.3.11](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html),
  [CUDA Programming Guide / nvcc section](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/nvcc.html)).
  Newer: `-src-in-ptx` keeps original source inside PTX (requires `-lineinfo`).
- **Where it lands**: line tables in the cubin ELF (`.debug_line`) and a SASS→PTX mapping in
  the `.nv_debug_line_sass` section.
- **nvdisasm** ([CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)):
  - `-g` / `--print-line-info`: annotate SASS with source file/line from `.debug_line`
    (output shows `//## File "...", line N` per instruction block).
  - `-gi`: same + inlining info ("line 17 inlined at line 23").
  - `-gp` / `--print-line-info-ptx`: annotate SASS with **PTX** line info from
    `.nv_debug_line_sass` — this is the SASS↔PTX hop.
  - register **liveness** display (per-instruction GPR/predicate assigned/used/live table +
    total register count) — perfect for "register bloat" diagnostics.
  - `-cfg`/`-bbcfg` (DOT control-flow graphs), `-json` (machine-readable SASS: opcode,
    operands, per-function start/length).
- **cuobjdump** (same doc): `-sass`/`--dump-sass`, `-ptx`/`--dump-ptx`, `-elf`/`--dump-elf`,
  `-lelf`/`-xelf` (list/extract cubins from fatbins & host binaries), `-lptx`/`-xptx`, and
  crucially `-res-usage`/`--dump-resource-usage`:
  `Function foo: REG:24 STACK:8 SHARED:0 LOCAL:0 CONSTANT[0]:472 ...` — static per-kernel
  register/smem/local usage straight from the ELF, no execution needed.
- **Nsight Compute Source view**: requires `-lineinfo`; maps CUDA-C ↔ PTX ↔ SASS and
  aggregates per-instruction metrics up to source lines
  ([Nsight "CUDA Source View" doc](https://docs.nvidia.com/gameworks/content/developertools/desktop/analysis_source_code_correlation.htm));
  the Nsight Compute Customization Guide describes `SourceMetrics` entries in section files
  ([CustomizationGuide](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html)).

**Tiny-profiler recipe**: build with `-lineinfo`; at report time run
`cuobjdump -res-usage` (static regs/smem/local per kernel) and `nvdisasm -g -json`
(SASS + source lines), then attribute hot SASS ranges (from PC sampling, if added later) or
just show annotated SASS next to the CUPTI record.

### 3.2 AMD

- Compile with `hipcc -g` / `-gline-tables-only`; line tables land in the code object ELF.
- Disassemble with ROCm LLVM: `llvm-objdump --disassemble --arch=amdgcn --mcpu=gfx90a <.hsaco/.co>`
  (use `/opt/rocm/llvm/bin/llvm-objdump`; `--mcpu` must match the kernel's target; for fat
  bundles first `clang-offload-bundler --unbundle`). Verified usage in
  [ROCm/rocBLAS#1295](https://github.com/ROCm/rocBLAS/issues/1295) and
  [ROCm/LLVM-AMDGPU-Assembler-Extra](https://github.com/ROCm/LLVM-AMDGPU-Assembler-Extra);
  `llvm-objdump -l/--line-numbers` prints source lines when present
  ([llvm-objdump docs](https://llvm.org/docs/CommandGuide/llvm-objdump.html)).
- Runtime correlation hook: rocprofiler-sdk's code-object service reports each kernel symbol's
  `kernel_address` + entry offset (§4), so PC-sample addresses can be mapped to disassembled
  ISA lines.

---

## 4. ROCm equivalents

### 4.1 rocprofiler-sdk ("rocprofiler v2")

[Docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html); repo
`ROCm/rocprofiler-sdk` (ROCm ≤7.0; ROCm 7.1+ moved to `ROCm/rocm-systems`,
`projects/rocprofiler-sdk`). Counters/SQTT go through the AQLprofile companion library.
CLI: `rocprofv3`; output formats include `rocpd` (SQLite).

Verified record structures (SHA `3a924a0`):

**Kernel dispatch trace record**
([buffer_tracing.h#L324-L339](https://github.com/ROCm/rocprofiler-sdk/blob/3a924a0d36a3b4be70ce3ef6e81f7e302e30258e/source/include/rocprofiler-sdk/buffer_tracing.h#L322-L339)):

```c
typedef struct rocprofiler_buffer_tracing_kernel_dispatch_record_t {
    uint64_t                                size;
    rocprofiler_buffer_tracing_kind_t       kind;   // ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH
    rocprofiler_kernel_dispatch_operation_t operation; // COMPLETE only
    rocprofiler_async_correlation_id_t      correlation_id;
    rocprofiler_thread_id_t                 thread_id;
    rocprofiler_timestamp_t                 start_timestamp; // ns
    rocprofiler_timestamp_t                 end_timestamp;   // ns
    rocprofiler_kernel_dispatch_info_t      dispatch_info;
} rocprofiler_buffer_tracing_kernel_dispatch_record_t;
```

with
([fwd.h#L792-L810](https://github.com/ROCm/rocprofiler-sdk/blob/3a924a0d36a3b4be70ce3ef6e81f7e302e30258e/source/include/rocprofiler-sdk/fwd.h#L789-L810)):

```c
typedef struct rocprofiler_kernel_dispatch_info_t {
    uint64_t               size;
    rocprofiler_agent_id_t agent_id;
    rocprofiler_queue_id_t queue_id;
    rocprofiler_kernel_id_t kernel_id;
    rocprofiler_dispatch_id_t dispatch_id;
    uint32_t               private_segment_size; // scratch/spill per work-item
    uint32_t               group_segment_size;   // LDS per work-group (static+dynamic)
    rocprofiler_dim3_t     workgroup_size;       // runtime block dims
    rocprofiler_dim3_t     grid_size;            // runtime grid dims (total threads)
    uint8_t                reserved_padding[56];
} rocprofiler_kernel_dispatch_info_t;
```

**Registers (VGPR/SGPR) are NOT in the dispatch record** — they come from the code-object
kernel-symbol callback
([callback_tracing.h#L204-L217](https://github.com/ROCm/rocprofiler-sdk/blob/3a924a0d36a3b4be70ce3ef6e81f7e302e30258e/source/include/rocprofiler-sdk/callback_tracing.h#L204-L217)):

```c
typedef struct rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t {
    uint64_t   size;
    uint64_t   kernel_id;          // joins to dispatch records' kernel_id
    uint64_t   code_object_id;
    const char* kernel_name;
    uint64_t   kernel_object;
    uint32_t   kernarg_segment_size, kernarg_segment_alignment;
    uint32_t   group_segment_size;    // static LDS
    uint32_t   private_segment_size;  // static scratch
    uint32_t   sgpr_count;            // scalar GPRs
    uint32_t   arch_vgpr_count;       // architectural VGPRs
    uint32_t   accum_vgpr_count;      // accumulator VGPRs (AGPRs, CDNA matrix cores)
    int64_t    kernel_code_entry_byte_offset;
    rocprofiler_address_t kernel_address; // for PC→ISA mapping
} ...;
```

So the AMD analogue of CUPTI's `registersPerThread` is `arch_vgpr_count` (+`accum_vgpr_count`
on CDNA), joined by `kernel_id`; the analogue of `staticSharedMemory`/`dynamicSharedMemory`
is the symbol's static `group_segment_size` plus the dispatch-time `group_segment_size`
(which is the runtime total, "AKA: LDS size").

Other services: HIP/HSA API callback tracing, dispatch/attaching **counter collection**
(`rocprofiler_counter_record_t` with per-instance values, via AQLprofile), **PC sampling**
(`pc_sampling.h`, stochastic + host-trap records, CDNA support), **SPM** (streaming
performance monitor), **thread trace**, external correlation ids, and **roctx** markers
(`rocprofiler-sdk-rotx/roctx.h`: `roctxMarkA`, `roctxRangePushA`, ... — the NVTX analogue).

Legacy `rocprof` (v1, `ROCm/rocprofiler`) reports per-kernel
`DispatchNs/BeginNs/EndNs/CompleteNs` timestamps and `pmc:` counter groups with derived
metrics from `metrics.xml`
([rocprof how-to](https://rocmdocs.amd.com/projects/rocprofiler/en/latest/how-to/using-rocprof.html)).

### 4.2 Omniperf / Omnitrace as architectural references

- **Omniperf** (now `rocprofiler-compute`): kernel-level profiler built on rocprof counters.
  Architecture = 3 components: profiling (raw counters via app replay + micro-benchmarks for
  hierarchical roofline), analysis (metrics engine), Grafana/CLI analyzer
  ([High Level Design](https://rocm.github.io/rocprofiler-compute/high_level_design.html)).
  Computes System Speed-of-Light, HW-block SOL (VALU/SALU/MFMA/LDS/TCP/TCC utilization %),
  memory chart, roofline, baseline comparison
  ([intro](https://rocm.github.io/rocprofiler-compute/introduction.html),
  [performance model](https://rocm.github.io/rocprofiler-compute/performance_model.html)).
  This is the AMD mirror of Nsight Compute's sections; a tiny profiler can copy the
  *SOL percentage-of-peak* framing.
- **Omnitrace**: tracing + sampling tool (dyninst binary instrumentation, call-stack
  sampling, process sampling, causal profiling; HIP/HSA/Kokkos/OMPT; Perfetto output)
  ([README](https://github.com/ROCm/omnitrace)). Reference for trace presentation, not for
  kernel internals.

---

## 5. Prior lightweight profilers to model

| Project | What it does | Reusable idea |
|---|---|---|
| CUPTI sample `activity_trace_async` ([cuptisamples](https://github.com/mmcloughlin/cuptisamples/blob/794cac63e7fc3bbc508d700f8609cdd0e5d6d155/samples/activity_trace_async/activity_trace_async.cu)) | minimal CUPTI tracer | buffer-request/complete callbacks, record dispatch loop |
| PyTorch **Kineto** ([CuptiActivityApi.cpp](https://github.com/pytorch/kineto/blob/f10dcd9a601d0e761064614d6992a003745def7e/libkineto/src/CuptiActivityApi.cpp#L336-L380)) | CUPTI-based library profiler | enables CONCURRENT_KERNEL + RUNTIME + DRIVER + OVERHEAD + EXTERNAL_CORRELATION; forced flush at stop; correlation-id join; Chrome trace output |
| **NVBench** ([README](https://github.com/NVIDIA/nvbench)) | kernel benchmarking | cold (L2-flushed) vs batch timing; reports bytes/s and **%-of-peak bandwidth**; clock locking (Volta+) |
| `NERSC/timemory` (368★) | modular perf analysis toolkit | metric composition + hierarchical output |
| `JamesTheZ/CudaProf`, `anshumang/cupti_profiler`, `neoblizz/cupti-plus-plus`, `cwpearson/cupti`, `nlesc-recruit/nvmetrics` | tiny CUPTI profilers on GitHub | prior art for minimal record→table pipelines |
| **nvprof** (removed CUDA 13.0) | timeline + events/metrics | its model: *event* = raw HW counter, *metric* = formula over events (`--query-metrics`); deprecated since Volta was last fully supported arch ([Profiler Users Guide](https://docs.nvidia.com/cuda/profiler-users-guide/)) |
| **Nsight Compute** | kernel profiler + rules | sections/rules architecture (§6) |

**Nsight Compute sections** (from the
[Profiling Guide §2.2.2](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)):
`SpeedOfLight` (compute vs memory %-of-peak throughput), `Occupancy` (theoretical vs
achieved warps; "large discrepancies ... indicate highly imbalanced workloads"),
`SchedulerStats` (theoretical/active/eligible/issued warps; skipped issue slots = poor
latency hiding), `WarpStateStats` (stall reasons per issued instruction),
`ComputeWorkloadAnalysis` (IPC + pipeline utilization), `LaunchStats`,
`MemoryWorkloadAnalysis`. Section sets: `basic` (default, low overhead, mostly no-replay
metrics) vs `full`. Sections are `.section` protobuf-ish text files; rules are Python files
implementing `get_identifier`/`apply` via the NvRules API
([Customization Guide §1.2-1.3](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html)).

**CUPTI vs PerfWorks — what's reproducible without PerfWorks/nvperf:**

- Reproducible with plain CUPTI Activity API (no counters): kernel timeline, durations,
  grid/block/cluster, regs/thread, static+dynamic smem, local memory (spills), launch
  latency (queued/submitted), memcpy/memset bytes, API correlation, graph provenance,
  **theoretical occupancy + occupancy limiter** (via `cuda_occupancy.h`), and static
  resource usage from the ELF (`cuobjdump -res-usage`). This is exactly Nsight's
  `launch__*` no-replay tier.
- Requires PerfWorks-class access (Nsight Compute / CUPTI Range Profiling API /
  PM Sampling): all achieved HW counters — SOL compute/memory %, achieved occupancy,
  warp stall reasons, cache hit rates, DRAM bytes, instruction counts. In CUDA 13 the
  old Event/Metric APIs are gone; the sanctioned paths are `cupti_range_profiler.h`
  (range profiling) and `cupti_pmsampling.h` (PM sampling), both layered over PerfWorks
  ([CUPTI 13.0 release notes](https://docs.nvidia.com/cupti/13.0.0/release-notes/release-notes.html)).
  Design the tiny profiler so diagnostics degrade gracefully: rules that need counters
  are skipped when only Activity data is present.

---

## 6. Diagnostic methodology: "why is this slow?"

### 6.1 Inputs available per kernel (no HW counters)

`duration = end - start`; launch latency = `start - queued` / `submitted - queued`;
`gridX*gridY*gridZ` blocks, `blockX*blockY*blockZ` threads; `registersPerThread`;
`staticSharedMemory + dynamicSharedMemory`; `localMemoryPerThread`; `sharedMemoryConfig`;
carveout; theoretical occupancy + limiter (computed, §2.3); kernel name → source via
`-lineinfo`/nvdisasm; static `REG/STACK/SHARED/LOCAL` via `cuobjdump -res-usage`.
Bytes moved: from user annotation or from MEMCPY records (kernel-internal bytes need
counters — treat as optional input).

### 6.2 Limiter decision tree (implementable today)

1. **Launch-path problem**: `start - queued` large relative to duration → launch/buffer
   latency, not kernel math (only visible when latency timestamps enabled).
2. **Occupancy limiter analysis** (the compiler-aware core): run `cudaOccMaxActiveBlocksPerMultiprocessor`
   with the record's regs/smem/block; the calculator tells you the binding resource:
   - register-limited → "occupancy capped at X% by N regs/thread"
   - smem-limited → "capped by (static+dynamic) smem per block"
   - thread/block-limited → block too small (warps/SM underfilled)
   Cross-check with Nsight's guidance: "low occupancy always reduces the ability to hide
   latencies" ([Profiling Guide, Occupancy section](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).
3. **Spill detection**: `localMemoryPerThread > 0` (or `LOCAL>0` in `-res-usage`) ⇒
   register spills to local memory ⇒ per-thread latency/bandwidth tax. Classic compiler
   cause: unrolled accumulators / aggressive inlining bloating live ranges.
4. **Roofline classification** (when FLOPs and bytes are known): arithmetic intensity
   `AI = FLOPs / bytes`; compare to machine balance point `peakFLOPs / peakBW`.
   `AI < balance` → memory/traffic-bound; else compute-bound; if achieved fraction of both
   peaks is low → latency-bound (needs occupancy/stall evidence). This is the classic
   Roofline model (Williams, Waterman, Patterson, *CACM* 52(4), 2009) as integrated into
   Nsight Compute ([NVIDIA roofline blog](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/),
   [Profiling Guide ch. 8 Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html));
   Nsight's roofline sections (`SpeedOfLight_*RooflineChart.section`) compute FLOPs from
   SASS instruction counts (`smsp__sass_thread_inst_executed_op_*`) and traffic from
   `dram__bytes.sum` — counter-derived, so a counter-less tiny profiler should accept
   FLOPs/bytes as user annotations or estimate from instruction mix via nvdisasm.
5. **Nsight Compute rules engine** as the model for phrasing: rules are Python attached to
   sections, emitting warnings + speedup estimates + focus metrics + source markers
   ([Customization Guide, Rule System](https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html);
   shipped rules live in the install `sections/` dir and are readable). The SOL heuristic
   pattern to imitate: compute %-of-peak for compute and memory; whichever is higher is the
   "likely limiter"; both low → latency → drill into occupancy/stalls.

### 6.3 Compiler-hint phrasings (codegen-aware output examples)

- Register bloat: "Kernel uses 128 regs/thread → occupancy limited to 25% (register file).
  Likely cause: unrolled accumulators keeping N live values (see nvdisasm liveness table).
  Try `__launch_bounds__(BLOCK, MIN_BLOCKS)`, `--maxrregcount`, or restructuring the unroll
  to reuse accumulators." (regs/thread from CUPTI record; liveness from `nvdisasm`.)
- Spills: "localMemoryPerThread = 64 B → register spills; expect local-memory traffic.
  Reduce live range pressure or raise the register budget if occupancy allows."
- Smem: "dynamic smem 96 KB/block with `shmemLimitConfig=OPTIN` → 1 block/SM; consider
  carveout (`CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT`) or tiling smaller."
- Tail effects: grid blocks < `numSms * blocksPerSM` → "grid underfills the GPU; duration
  dominated by partial waves" (pure arithmetic from record fields).
- Bank conflicts hint: `sharedMemoryConfig` (bank mode) + smem-heavy kernel → suggest
  checking padding (conflict confirmation needs counters/PC sampling — say so).

---

## 7. Version-change ledger (verified)

| Change | Version | Evidence |
|---|---|---|
| `cuptiActivityFlushPeriod` added | CUDA 11.1 | CUPTI usage docs |
| `localMemoryTotal` deprecated → `localMemoryTotal_v2` | CUDA 11.8 | header comment, cupti_activity.h L3049 |
| CUPTI device-buffer allocation change | CUDA 12.0 U1 | header note L7028 |
| Cluster fields (`clusterX/Y/Z`, `maxActiveClusters`, ...) | CUDA 12.x (CC 9.0 support) | CUpti_ActivityKernel9 |
| `CUpti_ActivityKernel10` (+`isDeviceLaunched`); Event/Metric/PC-sampling-activity APIs dropped; Profiling+PerfWorks APIs deprecated; Range Profiler + PM Sampling + multi-subscriber V2 | CUDA 13.0 | CUPTI 13.0 release notes; CUDA 13.0 toolkit release notes |
| nvprof + Visual Profiler removed | CUDA 13.0 | CUDA 13.0 toolkit release notes |

## 8. Caveats / uncertainties

- The CUDA 13 headers cited here come from a vendored mirror (`terryye/cuda_GEMM`); field
  names were cross-checked against the CUPTI 13.x docs where published. The official
  NVIDIA/cupti GitHub repo is gone — samples are mirrored at `mmcloughlin/cuptisamples`.
- CUPTI kernel records have **no achieved-occupancy or stall fields**; anything "achieved"
  needs Range Profiling/PM Sampling (PerfWorks-based) — plan the feature matrix accordingly.
- rocprofiler-sdk: `ROCm/rocprofiler-sdk` is the ROCm ≤7.0 home; newer development is in
  `ROCm/rocm-systems` (per the docs index). Record layout verified against the former at
  SHA `3a924a0` (branch `amd-staging_deprecated`).
- Omniperf roofline requires its micro-benchmark pass (MI200+); not free at runtime.
