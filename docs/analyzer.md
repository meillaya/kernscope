# kernscope-analyze — static GPU-kernel analyzer

`kernscope-analyze` is the kernscope static analyzer CLI: PTX or AMDGPU ISA
text in, one structured kernel report out (register pressure, occupancy
limits + main limiter, memory coalescing, bank conflicts, warp divergence,
barrier usage, instruction mix, tensor-core/MFMA/WMMA usage, spill
detection, latency bottlenecks — **ten analyses** — plus a suggestion
engine). It is the Wave-3 deliverable of the analyzer module
(`tools/analyze`), built on `libs/ptxparse`, `libs/amdparse`, `libs/arch`,
`libs/occupancy`, `libs/report` and the six analysis modules
(`tools/analyze/src/analyze_{regs,occupancy,coalescing,banks,divergence,mix,suggest,all}.cpp`).

## CLI reference

```
usage: kernscope-analyze <file.ptx|file.s|file.isa> [options]

options:
  --arch <sm_XX|gfxXXXX>  architecture (REQUIRED for AMDGPU input; optional
                          for PTX, defaults to the .target directive;
                          known: sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120
                          gfx90a gfx942 gfx1101)
  --block <N>             launch block size (occupancy input; absent ->
                          occupancy emits its block_size_unknown diagnostic)
  --json | --text | --html
                          output format (default: --text; --json is
                          schema-validated; --html is self-contained with an
                          embedded JSON payload)
  --source <file.cu|file.hip>
                          CUDA/HIP source file; its stem names the kernel in
                          the report (PTX entry names are mangled and not
                          user-facing)
  --version               print kernscope-analyze <version> and exit
  --help                  print the full help and exit
```

`--help` prints the same reference the plan's acceptance checks against
(all inputs and outputs enumerated).

### Input language auto-detection

| Extension | Language | Notes |
|-----------|----------|-------|
| `.ptx` | PTX | `libs/ptxparse` (`parse_ptx_module`) |
| `.s` / `.isa` | AMDGPU ISA text | `libs/amdparse` (`parse_amd_module`, LLVM 21 MC); `--arch` required |
| other | content sniff | `.version` + `.target` directives present → PTX, else AMDGPU |

### Launch-config discovery (manifest.json)

Launch-config inputs that are **not statically present in the PTX text**
(block size, the exact `ptxas -v` register figure) come from the launch
manifest — the committed fixture convention
(`tests/fixtures/README.md`). When the input's directory contains
`manifest.json`, its `arch` / `block_size` / `ptxas_v_report` fields seed
the analysis; explicit `--arch` / `--block` flags override the manifest.

Without the `ptxas -v` register figure the analyzer falls back to its
static declared-register proxy, which is an **upper bound** (ptxas reuses
registers — `analyze_regs.h`). This is why the golden fused_attention
report (102 registers → occupancy 0.375, limiter `registers`) requires the
manifest's `ptxas -v` line, exactly as the todo-19 golden tests drive it.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | report produced, clean (no warnings) |
| 1 | report produced; warnings present (a successful analysis run) |
| 2 | error: parse failure (names line:col), unknown arch, unreadable file, render schema violation |
| 3 | usage error (bad flags, missing input, missing `--arch` for AMDGPU input) |

## The ten analyses

Every analysis names its module (`tools/analyze/src/`) and the
`docs/research/gpu-static-analyzer-research.md` ("the brief") section it
implements. All emitted quantities are machine-readable metrics (flat
numeric map in the report model) or findings with stable ids.

### 1. Register pressure — `analyze_regs.cpp` (brief §6.1, L386-409)

Precedence: (1) `ptxas -v` register figure when available (both published
formats, §6.1 L397-409); (2) static proxy = sum of declared `.reg`
register counts + `.local` frame bytes (upper bound — ptxas reuses
registers; documented, never a fabricated exact claim). AMD path: counts
`scratch_load`/`scratch_store` traffic (the `next_free_vgpr` figure lives
in code-object notes, not `.s` text — §6.2).

Metrics: `registers_per_thread`, `registers_allocated_per_warp`,
`local_stack_bytes`. Findings: `spill` (warning), `cannot_estimate_registers`.

### 2. Occupancy limits + main limiter — `analyze_occupancy.cpp` (brief §1)

Theoretical occupancy via `libs/occupancy` + `libs/arch`, min-wins rule
(§1.1 L31-64): `blocks = min(limit_threads, limit_warps, limit_smem,
limit_regs, limit_blocks)` with per-architecture constants (§1.2 NVIDIA
L66-102, §1.3 AMD L104-144; RDNA3 VGPR values per
`docs/research/rdna3-constants.md`). Register limiter: NVIDIA
`65536 / (granule-allocated regs × block threads)` (§1.2 L101-102); AMD
waves-per-SIMD × SIMDs (§1.3 L112, L139). Main limiter is the binding
resource, with per-limiter headroom metrics. Block size comes from the
launch manifest / `--block`; without it a `block_size_unknown` diagnostic
is emitted. **Golden:** fused_attention @ sm_86 / 64 threads / 102 regs →
`occupancy = 9 blocks × 2 warps / 48 = 0.375`, limiter `registers`
(pinned in `tests/fixtures/manifest.json`; the kernel's dynamic smem is
recorded as a metric only, per the golden reconciliation note in
`analyze_all.cpp`).

Metrics: `occupancy`, `blocks_per_sm`, `warps_resident_per_sm`,
`max_warps_per_sm`, `max_blocks_per_sm`, `shared_memory_bytes`,
`headroom_blocks_by_{threads,warps,smem,registers,blocks_cap}`,
`occupancy_is_theoretical`.

### 3. Global memory coalescing — `analyze_coalescing.cpp` (brief §3.1)

Global ld/st classified into 128B-sector patterns; coalescing efficiency =
used bytes ÷ transferred bytes per warp (§3.1: full sector utilization
required; a `coalescing` warning fires when efficiency < 1.0 with the
owner's "global loads not 128B coalesced" class). Vectorized loads counted
separately. AMD overload maps global/buffer/flat accesses (§3.2).

Metrics: `global_coalescing_efficiency`, `global_sectors_per_warp`,
`global_bytes_used_per_warp`, `global_bytes_transferred_per_warp`,
`vectorized_ld_global`. Findings: `coalescing`, `coalescing_unknown`.
**Golden:** fused_attention tile staging → efficiency 0.5.

### 4. Shared-memory bank conflicts — `analyze_banks.cpp` (brief §2.1)

Shared ld/st stride analysis: bank = (word ÷ 4) mod 32 (§2.1: 32 banks ×
4 B, CPG Table 31); `way = lanes / distinct_banks`, conflict when way ≥ 2.
Padding/XOR-swizzle evidence on the finding (§2.1 L173-179). AMD overload
uses the LDS taxonomy (§2.2).

Metrics: `shared_stride_words`, `shared_stride_mod_32`,
`shared_distinct_banks`, `shared_bank_way`. Findings: `bank_conflict`,
`bank_conflict_unknown`. **Golden:** fused_attention 68-word column
stride → banks {0,4,...,28}, way 4.

### 5. Warp divergence — `analyze_divergence.cpp` (brief §4)

Branch-target/post-dominator analysis on the per-kernel CFG with
thread-id-dependence detection on `setp` predicates (§4.2 L303-329):
tid-derived → divergent, uniform → convergent, data-dependent →
three-valued "possibly divergent" (never a sound claim for data-dependent
control flow). A divergent branch **inside a loop** (back-edge detection)
fires the owner's `divergent_inner_loop` warning. AMD overload classifies
`s_cbranch_*` conditions (§4.2, AMDGPU branch forms).

Findings: `divergent_inner_loop`, `possibly_divergent_branch`. **Golden:**
fused_attention score j-loop (`lane % 8 == j % 8`, depth 2).

### 6. Barrier usage — `analyze_divergence.cpp` (brief §4.1)

`bar.sync` / `bar.warp` / `barrier.cluster` inventory + count (parser
derived statistics), barrier density per loop, and a `divergent_barrier`
warning when a barrier is reachable under a divergent branch (CUDA UB).
AMD: `s_barrier` inventory.

Metrics: `bar_sync_count`, `bar_warp_count`, `barrier_cluster_count`,
`barrier_density_per_loop`. Findings: `divergent_barrier`. **Golden:**
fused_attention `bar_sync_count = 2`.

### 7. Instruction mix — `analyze_mix.cpp` (brief §5.1/§8.1)

Class histogram from the parser taxonomy (FP32/FP16/INT/LSU/SMEM/TENSOR/
CTRL buckets) with per-class counts and percentages, plus the explicit
fma/mul/add/ld/st/setp counters (brief §5.1 L343-350). AMD overload
buckets VALU/SALU/LDS/global/buffer/flat/branch/barrier/wait/endpgm.

Metrics: `mix_count_<bucket>`, `mix_pct_<bucket>`, `fma_rn_f32`,
`mul_f32`, `add_f32`, `ld_shared`, `st_shared`, `ld_global`, `st_global`,
`ld_local`, `st_local`, `setp`, `predicated_bra`, `bar_sync`.
**Golden:** fused_attention `fma_rn_f32 = 1024` (FMA-heavy GEMM body).

### 8. Tensor-core / MFMA / WMMA usage — `analyze_mix.cpp` (brief §5)

`mma.sync` / `wmma` / `wgmma` / `tcgen05` families on PTX (§5.1 L333-351)
and `v_mfma_*` (CDNA2/3) / `v_wmma_*` (RDNA3) on AMD (§5.2 L352-383).
MAC counting from m×n×k shapes (brief §5.1 L346-350). A GEMM-like
(FMA-heavy) kernel with zero tensor ops gets the `no_tensor_core_usage`
info finding.

Metrics: `tensor_ops`, `tensor_macs`. **Golden:** fused_attention is
FMA-only → `tensor_ops = 0` + info finding; `mfma_fixture.s` →
`tensor_ops = 4`, `tensor_macs = 24576`.

### 9. Spill detection — `analyze_regs.cpp` (brief §6)

PTX: `ld.local`/`st.local` traffic + `.local` frame bytes (§6.1 L386-409)
→ `spill_bytes`; AMD: `scratch_load`/`scratch_store` ops (scratch = spill
class, brief §6.2). Any spill signal fires the `spill` warning with
machine-readable evidence.

Metrics: `spill_bytes`, `spill_stores`, `spill_loads`, `local_stack_bytes`.
**Golden:** fused_attention 0 spills (ptxas: 0 spill stores/loads).

### 10. Latency bottlenecks — `analyze_mix.cpp` (brief §8)

Approximate static critical-path estimate per basic-loop body from the
published microbenchmark latency tables (§8.1 L458-493, Volta→Hopper) with
issue-slot reasoning (§8.2): ALU-bound / LSU-bound / latency-bound,
dominant class named — explicitly "approximate, microbenchmark-derived",
never cycle-exact. Unknown arch → `latency_unknown_arch` info finding, no
fabricated numbers.

Metrics: `latency_classification`, `latency_estimated_cycles`. Findings:
`latency_bottleneck` (info, with `classification` + `dominant_class`
evidence), `latency_unknown_arch`. **Golden:** fused_attention
`latency_classification = 1` (bound classification present).

## Suggestion engine — `analyze_suggest.cpp`

A rules engine (rule table in the module header and `docs/analyzer-rules.md`)
mapping the finding set already emitted by the ten analyses to actionable
suggestions — one suggestion per triggering finding, never generic advice.
Every suggestion carries `triggered_by` evidence. Rules: `bank_conflict`
way ≥ 2 → `tile_k_32`; `divergent_inner_loop` → `hoist_predicate`;
`coalescing` without `bank_conflict` → `vectorize_coalescing`; register
limiter + FMA-heavy accumulator pattern → `split_accumulator`. The engine
is the **sole owner** of `report.suggestions` and rebuilds the section from
scratch (informational findings are dropped there). **Golden:**
fused_attention → exactly the owner's three suggestions
(`tile_k_32`, `split_accumulator`, `hoist_predicate`).

## AMDGPU path

For `.s`/`.isa` input the CLI runs the five analyses with `AmdModule`
overloads — registers (scratch), coalescing, banks, divergence, mix —
and the two PTX-specific analyses emit **documented "not applicable"
findings** instead of crashing:

| Analysis | AMD path |
|----------|----------|
| register pressure (scratch proxy) | runs |
| occupancy limits + limiter | `occupancy_not_applicable` info finding — register-limited occupancy needs ptxas-level allocation, absent from `.s` text (limiter = `not-modeled`) |
| coalescing / bank conflicts / divergence / barriers / instruction mix / tensor-core / spills | run (AMD overloads) |
| latency bottlenecks | runs on AMD mnemonic classes (mix module) |
| suggestion engine | `suggestions_not_applicable` info finding — rules consume the PTX CFG + occupancy limiter |

## Rendering

All formats come from `libs/report` and **validate the model first**
(SchemaViolation → exit 2, never a silent invalid report):

- `--text` — the owner's Section A format
  (`docs/research/owner-examples.md`): `Kernel:` / `Occupancy:` /
  `Main limiter:` / `Warnings:` / `Suggested:` with indented bullets.
- `--json` — schema-validated against `docs/schema/report.schema.json`
  (draft 2020-12): flat numeric metrics, findings with stable ids.
- `--html` — self-contained single file (inline CSS/JS, embedded JSON
  payload in `<script id="report-data">`, zero network requests).

## Golden example

```
$ kernscope-analyze tests/fixtures/fused_attention.ptx --arch sm_86 --block 64 --text
Kernel: fused_attention
Occupancy: 37.5%
Main limiter: registers
Warnings:
  - global loads not 128B coalesced (brief §3.1 threshold: 100% sector utilization required)
  - shared memory stride likely causes 4-way bank conflicts
  - divergent branch inside inner loop
Suggested:
  - pad shared stride / tile to conflict-free layout (tile K dimension by 32 when K-stride causes the conflict)
  - split accumulator fragment / lower unroll / maxrregcount
  - move predicate outside loop / restructure to predication
```

The same run with `--json` emits the full numeric report; the golden
numbers (occupancy 0.375, limiter `registers`, bank-conflict way 4,
coalescing efficiency 0.5, `divergent_inner_loop`, three warnings, three
suggestions) are pinned in `tests/fixtures/expected_fused_attention.json`
+ `manifest.json` and asserted by `tools/analyze/tests/analyze_golden_test.cpp`,
the lit suite (`test/analyze/golden.test`) and
`tools/analyze/tests/check_ten_analyses.py` (all ten analyses enumerated,
one asserted field each).
