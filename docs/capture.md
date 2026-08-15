# Capture & Replay (C4) — docs/capture.md

Status: todo 48 deliverable (replay engine + validation + timing hooks + CLI
+ e2e). Author: capture track. Pairs with `docs/capture-format.md` (bundle
format, todo 46) and `docs/capture-mapping.md` (CUDA↔HIP field mapping,
todo 44).

## 1. Scope

The capture/replay track records what a GPU kernel launch *observably did*
and lets you (a) emit a STANDALONE reproducer project (`emit-repro`,
todo 47), (b) replay the captured launch live and validate its outputs
(`replay`, todo 48), (c) benchmark the captured launch with a validated
methodology (`bench`, todo 48) and store the result in the libs/db JSONL
database, and (d) leave timestamped launch records for the profiler track
(`--profile-out`, the todo-49 hook contract, §6).

Nothing is ever uploaded anywhere: record/replay/bench are purely local
(MUST NOT auto-upload bundles).

## 2. Capture mechanisms (comparison)

All three mechanisms record into the SAME `libs/capture-core` session model
and serialize through the same `BundleWriter`; a bundle written by any
mechanism parses identically (the equivalence rules are
docs/capture-mapping.md).

| | CUPTI (NVIDIA primary) | LD_PRELOAD shim (fallback) | hip-sdk (AMD) |
|---|---|---|---|
| library | `libs/cupti-sub` (`libkernscope_cupti_sub.so`, todo 44) | `libs/capture-core/shim` (`libkernscope_intercept.so`, todo 43) | `libs/capture-hip` (`libkernscope_capture_hip.so`, todo 45) |
| transport | CUPTI callback/injection (`CUDA_INJECTION64_PATH` / direct link; dlopen'd `libcupti.so`) | `LD_PRELOAD` PLT interposition of cuda*/cu* | `LD_PRELOAD` + rocprofiler-sdk intercept table registration |
| allocation sizes | CUPTI Memory4 activity records (no hooks) | interposed cudaMalloc/cudaFree | wrapped hipMalloc/hipFree |
| arg layout | cuFuncGetParamInfo driver callbacks | same (driver path) | AMDGPU metadata parse |
| kernel binary | RESOURCE module-load callback (cubin) | fatbin/cuModuleLoadData bytes | code-object callback (hsaco) |
| runtime validation | **rental-gated** (scripts/rental-validate.sh) | **rental-gated**; locally validated on the MOCK runtime | **live on gfx1101** (local) |
| known holes | single-subscriber CUPTI (handled) | driver-vs-runtime split, static cudart | HSA VA assignment (see §4) |

Vendor auto-detect for `kernscope-capture record`: `--vendor` wins; else
`$KERNSCOPE_VENDOR`; else the first capture `.so` next to the CLI (hip
preferred — this machine's local loop), else an error naming both.

## 3. The bundle

Two formats, one reader (`BundleReader`, libs/capture-core):

* **v1** `*.kcap/` directory (todo 46): JSON manifest + events + blobs with
  mandatory sha256 provenance, ORIGINAL VAs and embedded-pointer patch
  tables (`BufferSnapshot`/`PatchEntry`/`apply_remap`).
* **v0** text bundle (todos 43–45 output): line-oriented; sidecar blobs
  (`input-buffer-<n>.bin`, `code-object-<id>.hsaco`) in a blob dir; **no**
  output blobs and **no** per-blob hashes — replay validation for v0
  bundles uses `--ref` files supplied by the caller.

## 4. The replay engine

`kernscope-capture replay <bundle>` (tools/capture/src/replay.cpp):

1. **Model**: the shared plan builder (`replay_model`, identical for
   `emit-repro`) resolves buffers (alloc sizes, H2D input blobs, D2H output
   markers, v1 input/output snapshots), the kernel binary and the arg
   marshalling plan. Size-unknown buffers become warned synthetic inputs.
2. **Restore** (docs/capture-mapping.md §4.4): the kernel binary is loaded
   (temp file -> module API) and the target kernel resolved by its
   descriptor-less symbol name. Buffers are allocated FRESH; a remap table
   maps every ORIGINAL buffer VA -> NEW VA; v1 embedded-pointer patch
   tables are applied with `apply_remap` (libs/capture-core) BEFORE the
   upload; pointer ARG slots are rewritten through the same remap table.
   *Original-VA policy*: restore at original VAs **where the platform
   allows**. HIP/HSA assigns VAs (no user-facing placement API at the
   hipMalloc level) → ALWAYS remap on AMD. CUDA's VMM
   (`cuMemAddressReserve` + `cuMemMap`) can place allocations at chosen
   addresses in principle — that is the documented extension point for the
   rental-gated NVIDIA path (the `libs/gpuruntime` abstraction does not
   expose VMM; the engine remaps and documents the policy here).
3. **Launch** with the captured grid/block/shared and remapped args;
   every launch emits a timestamped launch record (§6).
4. **Validate** every output buffer: **bitwise** by default (byte compare
   + sha256 of both sides, hash diff reported), **tolerance** opt-in with
   `--mode tolerance` (cosine / max_rel_err / pcc per the golden contract,
   §5). References come from v1 output snapshots in the bundle, or `--ref`
   files (v0) in output-buffer order.

Exit codes: 0 = validation PASS; 1 = replay ran but validation FAILED (the
QA-negative verdict); 2 = replay could not run (environment/usage/load).

## 5. Tolerance mode (golden contract)

`--mode tolerance` compares every output as **bf16 chunks upcast to fp32**
(2-byte little-endian elements; odd trailing bytes ignored) — the todo-23
rounding contract — and computes the three metrics with the formulas of
`tests/golden/kernscope_golden/metrics.py` (the repo's golden reference
implementation; `tools/capture/src/tolerance.cpp` is a faithful C++ port,
cross-checked for equality by the e2e against the Python original):

* cosine ≥ 0.999, max_rel_err ≤ 1e-2 (eps 1e-6 in the denom), pcc ≥ 0.99 —
  all three, inclusive; PASS iff all hold.

Use tolerance when the captured launch is numerically-but-not-bitwise
deterministic (e.g. cross-toolchain bf16 noise); the e2e asserts BOTH sides
on a bf16-noise fixture: bitwise FAIL (exit 1, hash diff) and tolerance
PASS, with the C++ verdict equal to the golden Python verdict.

## 6. Timing hooks — the todo-49 profiling contract

Two hook points:

* **replay** emits one `LaunchTimingRecord` per launch (steady clock, µs):
  `{"kind":"launch","kernel":"<name>","seq":1,"launched_at_us":<t0>,
  "completed_at_us":<t1>,"duration_us":<t1-t0>}` — written as JSONL to
  `--profile-out` (todo 49's tracer records carry the same semantic fields
  from rocprofiler-sdk/CUPTI: kernel name, start/end timestamps, duration;
  the replay hook makes a replay session's timings directly consumable for
  cross-validation against a profiler trace).
* **bench** writes the validation launch record(s) + one aggregate
  `{"kind":"bench","kernel":"...","reps":N,"median_ms":...,"p20_ms":...,
  "p80_ms":...}` record.

Consumers (todo 49) read `--profile-out` JSONL lines keyed by `kind`.

## 7. Replay-as-benchmark (`bench`)

`kernscope-capture bench <bundle>`: one bitwise validation launch (refused
to time an invalid replay; `--no-validate` skips), then the captured launch
is timed with the `libs/bench` methodology — N reps with **L2 flush**
between reps, median + p20/p80 (event-timed). `--reps N` (default 10) is a
fixed sample count (the harness is configured with `rep_ms=0` so each
convergence batch is exactly one rep; warmup stays time-budgeted). The
result is written as ONE libs/db JSONL record, keyed
(gpu_arch, kernel_family=`capture_replay`, shape_hash=`kernel=<sha256 of
the kernel name>`), with measured provenance (timestamp, tool version,
`mode:"measured"`) and config carrying bundle/kernel/device/gcn/reps/
L2-flush/validation — mirroring the todo-42 `tools/bench` record style.

`kernscope-capture db-query <db.jsonl> [--arch A] [--family F] [--shape S]`
loads the JSONL through libs/db and prints every matching record
(exit 0 = found, 1 = none, 2 = unreadable).

## 8. CLI reference

```
kernscope-capture record <app> [app-args...] [--vendor auto|amd|nvidia]
    [--bundle <path>] [--blob-dir <dir>] [--env K=V]
kernscope-capture emit-repro <bundle> [--out <dir>] [--blob-dir <dir>]
    [--vendor amd|nvidia] [--arch <arch>] [--kernel <name>] [--force]
kernscope-capture replay <bundle> [--vendor amd|nvidia] [--arch <arch>]
    [--blob-dir <dir>] [--kernel <name>] [--mode bitwise|tolerance]
    [--ref <file>]... [--dump-dir <dir>] [--profile-out <path>]
kernscope-capture bench <bundle> [--vendor amd|nvidia] [--arch <arch>]
    [--blob-dir <dir>] [--kernel <name>] [--ref <file>]... [--reps <n>]
    [--no-flush] [--no-validate] [--db <path>] [--profile-out <path>]
kernscope-capture db-query <db-file> [--arch <a>] [--family <f>] [--shape <s>]
```

v0 bundles require `--vendor`/`--arch` (no envelope); v1 reads the
manifest. `--ref` files are required for v0 replay/bench validation (the
captured output), in output-buffer order.

## 9. e2e + QA (gpu-amd, `ctest -R capture_e2e -L gpu-amd`)

* `capture_e2e` (HIP live on gfx1101): record layernorm_fused (auto-detect)
  → replay bitwise PASS (output hash == captured) → bf16-noise fixture:
  bitwise FAIL + tolerance PASS, cross-checked against the golden Python
  metrics → bench → db record EXISTS (queried via `db-query`, median
  match asserted) → profile JSONL launch + bench records.
* `capture_e2e_cuda_mock` (no GPU): mock-cudart app under the LD_PRELOAD
  shim → v0 bundle → emit-repro → nvcc compile-only gate (runtime on
  NVIDIA is rental-gated via scripts/rental-validate.sh).
* `capture_e2e_negative` (QA): corrupted input blob → replay exit 1,
  validation FAIL verdict, hash diff (ref != replay), no crash.
* `capture_db_query` (no-gpu): the db-query CLI contract (found/not-found/
  unreadable).
