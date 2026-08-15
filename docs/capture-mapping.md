# Capture Field Mapping — CUDA (CUPTI) ↔ HIP (rocprofiler-sdk)

Status: todo 44 deliverable. Author: capture track (todo 44). Maintained for
the equivalence test (`ctest -R capture_equiv -L gpu-amd`), todo 48 (replay)
and todo 49 (profiler hooks).

This document defines the FIELD-FOR-FIELD correspondence between a bundle
produced by the **CUDA mechanism** (todo 44 `libkernscope_cupti_sub.so`,
CUPTI callback/injection) and a bundle produced by the **HIP mechanism**
(todo 45 `libkernscope_capture_hip.so`, rocprofiler-sdk intercept tables) for
the SAME program — the todo-24 portable kernel run on gfx1101 (HIP live) and
its CUDA twin (CUPTI-on-synthetic, "shim-on-mock"). Both mechanisms record
into the SAME `libs/capture-core` session model (never modified) and serialize
through the SAME v0 `BundleWriter`, so the bundle line grammar is identical;
the mapping below is about the VALUES.

## 1. The equivalence contract

Two bundles are EQUIVALENT under this mapping iff, after applying the
normalization rules of §4:

| # | Field (v0 line token) | CUDA side (CUPTI) | HIP side (rocprofiler-sdk) | Comparison |
|---|---|---|---|---|
| M1 | `launch kernel` | `CUpti_CallbackData.symbolName` (runtime-resolved, demangled) | code-object kernel symbol (mangled), `.kd` suffix stripped | EQUAL after §4.1 kernel-name normalization (base name + `_Z` demangling) |
| M2 | `launch grid` | `cudaLaunchKernel_params.gridDim` / `CUlaunchConfig.gridDim*` (blocks) | HSA dispatch `grid_size / workgroup_size` (blocks, converted from work-items) | EQUAL exactly |
| M3 | `launch block` | `blockDim` | `workgroup_size` | EQUAL exactly |
| M4 | `launch shared` | `sharedMem` / `dynamicSmemBytes` | pending-launch `sharedMemBytes` | EQUAL exactly |
| M5 | `launch stream` | `stream` handle value | `stream` handle value | EQUAL after §4.2 handle normalization (default stream → 0) |
| M6 | `launch args=<n>` count | layout-derived param count (cuFuncGetParamInfo facts) | AMDGPU-metadata user-arg count (hidden args skipped) | EQUAL exactly |
| M7 | per-arg `[size:bytes]` scalar slots | `size` from cuFuncGetParamInfo; bytes copied from `kernelParams[i]` at ENTER | `size` from AMDGPU metadata; bytes from `kernel_params[i]` | size EQUAL; bytes BYTE-IDENTICAL |
| M8 | per-arg pointer slots | same, pointer VALUE (8 B) | same, pointer VALUE (8 B) | EQUAL after §4.3 VA normalization (both → allocation ordinal) |
| M9 | `alloc size=` | `CUpti_ActivityMemory4.bytes` (DEVICE kind) | wrapped `hipMalloc` size | EQUAL exactly (sorted multiset) |
| M10 | `copy bytes=` + kind | `cudaMemcpy(count, kind)` at ENTER/EXIT | wrapped `hipMemcpy(sizeBytes, kind)` | EQUAL exactly (H2D=1, D2H=2, D2D=3) |
| M11 | input-buffer blobs (`input-buffer-<n>.bin`) | H2D memcpy host source dump | H2D memcpy host source dump | **BYTE-IDENTICAL** (sha256 equal) after §4.4 ordering |
| M12 | kernel-binary blob | `cubin-<n>.cubin` (module-load callback) | `code-object-<n>.hsaco` | presence + PER-VENDOR sha256 recorded; **NEVER cross-vendor byte equality** (cubin ≠ hsaco by construction) |
| M13 | module record `sha256` | cubin hash | hsaco hash | each matches its own vendor blob on disk (bijection per side); hashes recorded |
| M14 | `stream` records | `cudaStreamCreate/Synchronize` | `hipStreamSynchronize`/device sync | same logical id assignment (1-based, per handle) |

Explicitly NOT compared: `api` call COUNTS (the HIP runtime re-dispatches
sync→async internally; the todo-45 wrappers dedup, but counts are a
mechanism artifact, not a capture fact), `ptr` values before normalization,
timestamps (absent in v0), and the kernel-binary BYTES across vendors.

## 2. Why scalar args and input blobs are byte-identical

Both mechanisms copy the argument bytes from the HOST side of the launch
boundary (kernelParams / kernel_params pointer array — the CUDA driver-API
convention, brief §5), sized by the kernel-image layout facts
(cuFuncGetParamInfo on CUDA, AMDGPU metadata on HIP). For a scalar `int32 n`
the copied bytes ARE the program's value on both sides — no VA involved, so
no normalization, byte-identical after M7. Input buffers are the host
sources of H2D copies — the app's actual input bytes on both sides (M11).

## 3. Why pointer args are normalized, not compared

Device VAs are platform-assigned: CUPTI reports the allocation address the
CUDA driver chose; HIP reports the HSA VA. The same logical buffer (e.g.
"the a input") has different raw addresses on the two sides. §4.3 replaces
every device pointer with the ordinal of the allocation that OWNS it (by
value match against the alloc records, in allocation order) — after which
"arg0 points at allocation #1" is comparable.

## 4. Normalization rules (applied by the equivalence test)

### 4.1 Kernel-name normalization
- Strip the HIP `.kd` kernel-descriptor suffix.
- Strip the `_Z...` Itanium mangling prefix on both sides (CUDA
  `symbolName` is demangled; HIP code-object names are mangled): the base
  name is `kernscope_templates_elementwise_add` for the todo-24 kernel.

### 4.2 Stream-handle normalization
- Both mechanisms record the default stream as handle value 0 (nullptr /
  hipStreamLegacy 0). Non-default streams map to their logical id (1-based)
  — equal per side.

### 4.3 VA normalization (device pointers)
- Build the allocation ordinals: the ordinal of an allocation is its position
  in the program's ALLOCATION ORDER (the alloc record sequence, 1-based).
  Both mechanisms record allocations in the same program order (a, b, out
  for the todo-24 kernel), so the same logical buffer receives the SAME
  ordinal on both sides regardless of the raw VAs - pointer-VALUE sorting
  would be meaningless cross-vendor (the VAs are platform-assigned).
- Replace every device pointer VALUE in launch pointer-arg slots, copy
  dst/src (the DEVICE side only: H2D → dst, D2H → src, D2D → both) and
  alloc/free records with `va#N`.
- The ordinal assignment is per-side; after substitution, compare the
  positions (both sides must reference allocation #1 for the same logical
  buffer). Host-side pointer VALUES (copy sources for H2D, destinations for
  D2H) are process-local and NOT compared.

### 4.4 Input-blob ordering
- Both sides number `input-buffer-<n>.bin` in H2D-copy call order; the
  equivalence test compares by the same call index (the k-th H2D copy's blob
  on one side == the k-th on the other).

## 5. Mechanism facts (provenance for the mapping)

| | CUDA (todo 44) | HIP (todo 45) |
|---|---|---|
| transport | CUDA_INJECTION64_PATH (`InitializeInjection`) or direct link; fallback LD_PRELOAD (todo 43 shim) | LD_PRELOAD of the tool .so (rocprofiler_configure) |
| allocation sizes | CUPTI Memory activity records (`CUpti_ActivityMemory4`, no hooks) | wrapped hipMalloc/hipFree hooks |
| arg layout | cuFuncGetParamInfo driver callbacks | AMDGPU metadata (parse_amdgpu_metadata) |
| kernel binary | RESOURCE module-loaded callback (cubin) | code-object callback (hsaco) |
| args source | kernelParams pointer array at ENTER | kernel_params pointer array at launch wrap |
| single-tool scope | yes (documented; CUPTI 12.6 single-subscriber) | n/a (SDK supports concurrent tools) |

## 6. Degradation & documented scope (MUST-NOT guards)

- No CUPTI-subscriber EXCLUSIVITY is assumed beyond the documented
  single-tool scope: `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED` is
  handled gracefully (fallback log, no crash — QA negative).
- The equivalence test never asserts cross-vendor kernel-binary byte
  equality (M12/M13: presence + per-vendor hash only).
- Runtime-API launches without known layout record `args=[]` (sizes
  unavailable at the boundary; capture.h note) — the equivalence test always
  drives the layout facts (cuFuncGetParamInfo) like a real driver would.
- CUDA 13.x migration surface (documented for todo 49): `cuptiSubscribe_v2`
  (13.3+) for multi-subscriber; `CUpti_ActivityKernel9→10` for profiler
  records — none of it affects the M1–M14 capture mapping.
