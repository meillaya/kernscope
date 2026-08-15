# kernscope-tune — correctness-gated autotuner

`kernscope-tune` is the C2 autotuner CLI (tools/tune, plan todos 25-29): it
searches a kernel family's config grid for the fastest **correct** kernel on
a given GPU, and reports the winner with roofline + occupancy evidence, a
variant-comparison report, and a reproducibility bundle.

```
kernscope-tune --kernel <family> --shape <M,N,K|R,C|Nq,Nk,d>
               --arch <auto|sm_XX|gfxXXXX>
               [--grid <auto|full>] [--json|--html|--text]
               [--db <path>] [--cache <path>] [--workdir <dir>]
               [--min-occupancy <f>] [--no-bench] [--no-cache|--force]
               [--emit-repro <dir>] [--version] [--help]
```

Families: `matmul | bias_gelu | softmax | layernorm | reduction |
flash_attention`. `--arch auto` resolves to **gfx1101** (this machine's
RX 7800 XT). NVIDIA archs (`sm_XX`) are **compile-only** — benchmarking on
NVIDIA is owner-gated and never auto-triggered.

## Search space

Each family owns a template generator that validates every candidate
**pre-compile** (search-space membership + `libs/arch` budgets: registers,
shared memory, warp/block divisibility, vector-width tiling) and rejects
infeasible configs with a reason string — the failure mode is a reason, never
a broken compile.

| Family | Dimensions (auto grid) |
|---|---|
| matmul | (BM,BN) ∈ {(64,64),(128,128)} × BK ∈ {16,32} × warps ∈ {1,2,4} × stages ∈ {1,2} — 24 configs (vec=1, padding, unroll=1 fixed); `--grid full` expands to the full 4608-combo space (BK 8-64, stages 1-4, vec 1-4, XOR swizzle, unroll 1-4) |
| bias_gelu / softmax / layernorm / reduction | block ∈ {128,256,512,1024} × vec_width ∈ {1,2,4,8}; softmax/layernorm add a mode (onepass/twopass), reduction (strided/tree) |
| flash_attention | Br ∈ {32,64} × Bc ∈ {32,64} × stages ∈ {1,2} × vec ∈ {1,2,4} × smem layout (padding/XOR) × causal on/off (validated subset) |

## Pipeline

One candidate config moves through five stages; the first three are
GPU-free or compile-only:

1. **Cache hit** — a config already in the tunecache is replayed (timings +
   outcome + occupancy + gate geometry) without compiling anything.
2. **GPU-free pre-screen** — the generator's `validate_config` (libs/arch
   budgets) then `libs/occupancy::compute_occupancy` (registers/smem/block
   limits; a config with 0 resident blocks is rejected, and an occupancy
   below `--min-occupancy` is rejected as not worth measuring).
3. **Compile** — single-source `.hip` via the family generator, then
   `hipcc --genco` (gfxXXXX) or `nvcc -c` (sm_XX).
4. **Correctness gate** — the kernel runs on the live HIP device through
   the `kernscope_templates_hip_runner` + `kernscope_golden` reference
   (`run_hip_gate.py`); a verdict that is not `pass:true` (cosine ≥ 0.999 ∧
   max_rel_err ≤ 1e-2 ∧ pcc ≥ 0.99, todo-23 contract) **discards** the
   variant. The driver never benchmarks a wrong kernel. The gate validates
   at the family's golden shape (matmul: M=N=K=64; the bf16 contract drifts
   on long-K fp32 accumulate), decoupled from the benchmark shape.
5. **Benchmark + select** — `libs/bench` (warmup/rep time budgets, L2 flush,
   adaptive trials, median + p20/p80). The best config wins by median, ties
   broken by p20, then p80, then enumeration order. Every measured config is
   persisted to a `libs/db` JSONL record (keyed
   `gpu_arch:family:shape_hash`, `query_best`-able) and the tunecache.

## Cache semantics

The tunecache is a Triton-style disk cache (one JSON file, default
`reports/tune/<family>-<shape_hash>.jsonl.tunecache.json`), keyed
`gpu_arch:kernel_family:shape_hash`:

- a config already cached is **skipped up front** — a re-run of the same
  (arch, family, shape) compiles 0 new variants and re-picks the winner
  without re-benchmarking;
- every outcome is cached (timings for measured; status + reason for
  rejected / gate-failed / compile-failed), mirroring Triton's inf-cost for
  infeasible configs;
- each entry is stamped with the **min-occupancy floor** it was produced
  under: the pre-screen outcome depends on the floor, so an entry recorded
  at floor 1.0 never replays for floor 0.0 and vice versa;
- `--no-cache` (alias `--force`) **skips the cache read** — a fresh session
  always re-runs the whole pipeline (the clean acceptance/repro story) —
  while still writing the cache with this run's entries.

## Roofline + occupancy methodology

The roofline position and the occupancy estimate are computed by the driver
(`libs/roofline` todo 12, `libs/occupancy` todo 11) and surfaced in **every**
output mode as machine-readable JSON fields:

| Field | Meaning |
|---|---|
| `roofline.position` | the workload's fraction of the compute roof — `attained_flops / compute_bound_ceiling` = min(1, AI/ridge); in (0,1] |
| `roofline.bound` | stable bound class: `compute` \| `memory` \| `latency` |
| `roofline.arithmetic_intensity` / `ridge_point` / `attained_flops` | full position (FLOP/B, FLOP/s) |
| `occupancy` | the **winning config's** theoretical occupancy (fraction 0..1) |
| `occupancy.limiter` | the winning config's main occupancy limiter (registers/smem/blocks/threads/warps) |

The lite templates are scalar (no tensor-core instructions), so the compute
roof is the **FP32 vector peak — measured on gfx1101** (todo 22,
`projected == false`); an arch without measured peaks reports
`roofline.ok == false` and no position. Per-variant occupancy/limiter also
live in each `variants[i]` entry.

## Output modes

- `--json` — the full evidence document: best config + timings, `roofline.*`,
  `occupancy` / `occupancy.limiter`, `baseline` (the first enumerated
  measured config — the naive baseline) + `speedup` vs it, the `variants`
  array (**measured first, sorted by median ascending**), pipeline summary
  counts, cache key.
- `--text` (default) — the owner's example format
  (docs/research/owner-examples.md), adapted to tune fields:
  `Kernel:` / `Time:` / `Best config:` / `Occupancy:` / `Main limiter:` /
  `Roofline position:` / `Speedup vs naive baseline:`.
- `--html` — a **self-contained single-file** variant-comparison report:
  every measured variant ranked by median with p20/p80 and speedup vs the
  naive baseline, the libs/report model JSON embedded in the standard
  `<script id="report-data" type="application/json">` payload (byte-exact
  extractable, schema-validated) plus the full evidence in
  `<script id="tune-evidence">`; inline CSS/JS, zero network, opens from
  `file://`.

Exit codes: 0 = best config selected; 1 = no viable config (every variant
pre-screen-rejected or gate-failed — a search failure, not a crash);
2 = usage/environment error; 3 = unknown kernel family.

## Reproducibility

```
kernscope-tune --kernel matmul --shape 512,512,512 --arch gfx1101 \
               --emit-repro /tmp/tune_repro
bash /tmp/tune_repro/run.sh
```

`--emit-repro <dir>` writes a standalone bundle into `<dir>`:

- **run.sh** — regenerates the winning config's source via the family
  generator CLI (`kernscope-tune-gen` / `kernscope-tune-elem-gen`), compiles
  it with the exact `hipcc --genco --offload-arch=<arch>` command the tune
  used, and re-runs the golden gate (`run_hip_gate.py`, verdict must be
  `pass:true`) at the winning config's gate shape/grid/block/symbol (read
  from `best.gate` in the evidence). Toolchain paths are pinned to the
  configure-time resolution; the **nixpkgs revision from `devenv.lock`** is
  embedded and cross-checked against `KERNSCOPE_EXPECT_NIXPKGS_REV`
  (warn-only).
- **README.md** — how to run it (`devenv shell`, then `bash run.sh`), the
  environment pin, and the winning config.

The repro needs a live HIP device (the golden gate runs the kernel); it
works for the families with a generator CLI (matmul + the four elementwise
families); `flash_attention` has no generator CLI and is rejected with a
clear message.

## Tests

- `tune_report` (ctest, LABELS no-gpu): `tests/tune/check_report.py` runs
  the matmul 512³ tune and asserts the numeric report contract (best config
  keys, roofline position/bound, occupancy, ≥ 20 variants sorted by median,
  speedup ≥ 1.0).
- `tune_driver` (no-gpu) / `tune_driver_live` (gpu-amd): the todo-28 driver
  acceptance (infeasible-at-floor-1.0 negative; live 24-grid tune, cache-hit
  re-run compiles 0 new variants).

## Capture->tune mode (`--from-capture`, todo 53)

The **distinct tuning mode for CAPTURED kernels** (Metis H-3): a capture
bundle carries the kernel binary + arguments + buffers but **no source**, so
the template pipeline above (generate -> compile -> golden gate) cannot
consume it. The launch configuration is the only searchable degree of
freedom:

```
kernscope-tune --from-capture <bundle> [--vendor amd] [--arch gfx1101]
               [--blob-dir <dir>] [--kernel <name>] [--ref <file>...]
               [--candidates <blocks-csv>]
               [--inject-config <gx,gy,gz:bx,by,bz[:shared]>]
               [--reps <n>] [--no-flush] [--db <path>]
               [--profile-out <path>] [--json|--text]
```

Pipeline (each candidate replayed through the todo-48 replay machinery via
its public APIs — `BuildPlan` from tools/capture + libs/gpuruntime +
libs/bench + libs/db; the replay_model.cpp plan builder is compiled into the
capture-tune library **unchanged**):

1. **Plan** — `BuildPlan(bundle, vendor/arch/blob-dir/kernel)` restores the
   target launch, the device-buffer plan (inputs with captured content,
   outputs with their validation references), the kernel binary and the arg
   marshalling table. v1 `*.kcap/` bundles carry their own output refs;
   v0 text bundles need `--ref <captured-output>` (output-buffer order).
2. **Candidates** — the captured baseline config is **always candidate #1**
   (explored first — guarantees `best.speedup >= 1.0`), then block sizes
   from the arch-valid set (`--candidates`, default {128,256,512,1024}) at
   the **captured grid** (a captured kernel's grid is workload-semantic;
   only the within-block partition is searchable without touching the
   binary). Shared-mem carveout is explored only when the captured launch
   used dynamic shared memory. `--inject-config` appends a test candidate
   (the acceptance injects block 2048 -> must be rejected pre-replay).
3. **Pre-screen** — every candidate is validated against the documented
   launch limits BEFORE any replay: per-block threads <= 1024 (HIP
   maxThreadsPerBlock ceiling), grid x <= 2^31-1, grid y/z <= 65535, shared
   <= the arch smem-per-block budget. Rejections carry a reason.
4. **Replay + validate** — each surviving candidate is launched with its
   (grid, block, shared) and the outputs are compared **bitwise** (sha256 +
   byte compare) against the captured reference; a validation failure
   DISCARDS the candidate and records it (never benchmarked).
5. **Bench** — a validated candidate is timed via libs/bench (N reps, L2
   flush, median + p20/p80).
6. **Select + persist** — best config by median (ties: p20, p80, order),
   then libs/db records keyed **(arch, kernel-name-hash,
   capture-bundle-hash)** — `kernel_family = "kernel=<sha256(kernel name)>"`,
   `shape_hash = "bundle=<sha256(bundle identity)>"` (v0: the bundle file;
   v1: manifest.json) — one record per measured candidate plus the JSON
   evidence report (baseline vs best speedup, ALL candidates with status +
   reason).

**Boundary vs template tuning (documented):** this mode only ever varies the
LAUNCH CONFIG of the captured kernel binary — the hsaco/cubin bytes, the
arguments and the buffers are byte-identical to the capture (the db key
proves it). Template tuning instead regenerates SOURCE with baked-in
tile/warp/smem structure and recompiles; the two modes cannot consume each
other's inputs.

Exit codes: 0 = best config selected (db written); 1 = no valid config
(every candidate pre-screen rejected / launch-failed / validation-failed /
bench-failed — **no db write**, asserted); 2 = usage/environment error.

Tests:

- `integ_capture_tune` (ctest, LABELS gpu-amd): live acceptance on gfx1101
  over the committed layernorm_fused bundle (`tests/fixtures/capture/
  layernorm-live/`) — >= 3 candidates explored, baseline first, validation
  PASS for every explored config, `best.speedup >= 1.0`, db record
  queryable via the libs/db CLI (`kernscope-capture db-query`) under the
  exact (arch, kernel-name-hash, capture-bundle-hash) key, the injected
  block-2048 config REJECTED PRE-REPLAY with a reason, and the QA negative
  (garbage reference -> exit 1, "no valid config", NO db write).
- `capture_tune_usage` (no-gpu): the --from-capture flag fails cleanly
  (exit 2) without a bundle / with a broken bundle.
