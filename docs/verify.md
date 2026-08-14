# kernscope-verify — the static GPU-kernel verifier CLI

`kernscope-verify` is the user-facing verifier: CUDA/HIP source in, a
per-check-class verdict document out, and — when every verdict is SAFE —
the verified IR handed off for backend codegen. It is the Wave-6
deliverable of the verify track (`tools/verify`), built on the todo-30
core pipeline (ingestion → launch-config tiers → diagnostic engine →
check-plugin registry) and the four check modules (todos 31–34). The
per-check soundness scope lives in `docs/verify-soundness.md` (class 4)
and the per-module `tools/verify/checks/*/README.md` files; this document
is the umbrella view: the eight check classes, the eBPF-verifier design
analogy, the soundness scope, and CLI usage.

## The eight check classes

`diagnostic.h` reserves stable class ids 1–8 (`kernscope::verify::CheckClass`),
mapped by the registry to the check plugins:

| id | class | plugin | verdict vocabulary |
|----|-------|--------|--------------------|
| 1 | global out-of-bounds | `checks/oob` (`global-oob`) | `SAFE` / `MAY_OOB` / `SYMBOLIC_DIMS` |
| 2 | shared-memory bounds | `checks/oob` (same plugin) | `SAFE` / `MAY_OOB` / `SYMBOLIC_DIMS` |
| 3 | barrier correctness | `checks/barrier` | `SAFE` / `BARRIER_MISMATCH` |
| 4 | race-prone shared writes | `checks/race` | `SAFE` / `MAY_RACE` / `UNKNOWN_RACE` |
| 5 | illegal pointer arithmetic | `checks/ptr` | `SAFE` / `PTR_OOB` |
| 6 | divergent barriers | `checks/barrier` (same plugin) | `SAFE` / `DIVERGENT_BARRIER` |
| 7 | uninitialized reads | `checks/ptr` (same plugin) | `SAFE` / `MAY_UNINIT_READ` |
| 8 | invalid address-space casts | `checks/ptr` (same plugin) | `SAFE` / `INVALID_ADDRSPACE_CAST` |

Engine-degraded verdicts use ids ≥ 100 (`EngineClass`): `SYMBOLIC_DIMS`
(100 — launch dims are runtime values, nothing proven), `SCEV_RECONSTRUCTION_FAILED`
(101 — `-O2` ingestion), `BUDGET_EXCEEDED` (102 — the per-function
explored-state cap, the eBPF complexity-limit analog).

**Verdict semantics.** `SAFE` is the only approval-class verdict. Every
other verdict — including the degraded engine verdicts — means *no bound
was proven*, so the CLI treats it as error-class: exit code 1 and NO
approved IR. This is deliberate: a degraded verdict is not a proof of
safety, and the approved-IR handoff must never carry an unproven module
(soundness over false safety).

## eBPF verifier design analogy (research brief §1)

The Linux eBPF verifier is the UX model (docs/research/
gpu-kernel-verifier-research-brief.md §1):

- **Reject with reason.** `bpf()` rejects a program with one concrete
  line-level message ("invalid indirect read from stack off -8+0 size 8",
  "R2 !read_ok") — never a bare "failed". kernscope-verify mirrors this:
  every non-SAFE verdict carries `Diagnostic`s with `SourceLoc`
  (DebugLoc-backed `file:line:col`), the enclosing kernel, the offending
  instruction, a human message and an optional evidence path.
- **Type/range tracking.** eBPF tracks register types (`PTR_TO_STACK`,
  `PTR_TO_MAP_VALUE`, …) plus signed/unsigned min/max ranges and refines
  them at branch points. The core's `queryRange` + SCEV + the launch-dim
  range lattice is the GPU analog: `threadIdx.x` gets `[0, blockDim.x)`
  from the launch config tier, and index expressions are bounded against
  allocation sizes.
- **Complexity caps, not hangs.** `BPF_COMPLEXITY_LIMIT_INSNS` /
  `_STATES` produce "BPF program is too large" instead of hanging; the
  per-function explored-state budget (default 65536, `--budget N`) emits
  the `BUDGET_EXCEEDED` degraded verdict the same way.
- **Degraded verdicts.** eBPF refuses to load when it cannot prove
  safety; the verifier reports `SYMBOLIC_DIMS` + a runtime-check
  suggestion when launch dims are runtime values instead of claiming
  safety it cannot prove.

## Launch-config tiers and ingestion

Default ingestion merges host+device into ONE LLVM module (two clang
passes + in-process link; see `tools/verify/README.md`), so the launch
site is visible and dims resolve to a tier:

- **(a) Exact** — constant dims at the launch site (`<<<32, 256>>>`):
  exact ranges for `threadIdx/blockIdx`.
- **(b) UpperBound** — `__launch_bounds__`/`nvvm.maxntid`: sound upper
  bound on `blockDim`.
- **(c) Symbolic** — runtime dims: degraded `SYMBOLIC_DIMS` verdict with
  a runtime-check suggestion (e.g. `assert(blockDim.x <= 256)`).

`--device-only` (secondary, degraded) ingests only the device pass: dims
are lost unless the kernel carries `__launch_bounds__`.

## CLI reference

```
usage: kernscope-verify <src.cu|src.hip> [--device-only] [--json]
      [--out <base>] [--list-checks] [--budget N] [--opt O1|O2]
      [--arch sm_80|gfx1101] [--version]
```

- `--device-only` — degraded ingestion (device IR only; dims become
  symbolic).
- `--json` — full verdict document on stdout (machine-readable; the lit
  corpus gate parses it).
- `--out <base>` — approved-IR base name; `<base>.verified.ll` is written
  only when every verdict is SAFE. Default: `<src>` stem in the current
  directory.
- `--list-checks` — enumerate the registered check plugins (from the
  todo-30 `CheckRegistry::enumerate()`), then exit 0.
- `--budget N` / `--opt` / `--arch` — core pipeline knobs (see
  `tools/verify/README.md`).

**Exit codes.** `0` = approved (every verdict SAFE, `.verified.ll`
emitted); `1` = at least one error-class verdict (no IR emitted); `2` =
ingestion failure (clang diagnostics forwarded verbatim, never a crash);
`3` = usage error.

**The verdict document** (`--json`) contains the source, device_only
flag, the `SAFE | WARNINGS | ERRORS` summary, the registered-checks
enumeration, per-kernel verdicts (kernel label, launch tier + origin,
per-class verdict with severity, `error_class` flag and the full
diagnostics with source locations), and the `approved_ir` path (or
`null`).

### Approved-IR handoff

On `SAFE` (or warnings-only), the verified module is projected onto its
GPU code — kernels + reachable device helpers; host functions (main,
launch stubs, host helpers) are dropped, and `llvm.global_ctors`/
`llvm.used` entries pointing at them are rewritten — and written to
`<out>.verified.ll`. The projection matters for the handoff: the proofs
apply to device code, and the AMDGPU backend rejects the host half's
`addrspace(0)` allocas. The lit e2e (`tests/verify/e2e_approved_ir.test`)
compiles a SAFE kernel's `.verified.ll` with `llc -mtriple=nvptx64-nvidia-cuda`,
runs an `amdgcn-amd-amdhsa` smoke through the HIP ingestion path (CUDA
IR carries NVPTX intrinsics that the AMDGPU backend cannot select), and
asserts that error kernels exit 1 and emit NO `.verified.ll`.

## Soundness scope

All eight checks are **MAY analyses** over the LLVM IR at `-O1`
(verify-early, before loop destruction — research brief §5.2): they
report what they cannot rule out. In scope: global/shared memory
accesses with SCEV/launch-dim-bounded indices (classes 1–2); barrier
counts + divergent-conditional barriers (3, 6); intra-kernel
shared-memory access pairs with barrier phase partitioning (4 — full
scope in `docs/verify-soundness.md`); pointer arithmetic, uninitialized
reads and address-space casts (5, 7, 8). Out of scope, documented per
module: data-dependent indices (reported `MAY_OOB`/`UNKNOWN_RACE`, never
`SAFE`), runtime-parameter allocations (index-bounded `SAFE` with an
explicit note), atomics, host-side allocation tracking. When a proof is
impossible the verdict degrades (`SYMBOLIC_DIMS`, `SCEV_RECONSTRUCTION_FAILED`,
`BUDGET_EXCEEDED`) — the verifier never asserts safety it cannot prove.

## Corpus gate (anti-self-report)

`tests/verify/corpus/` holds 19 committed fixtures; the four per-check
manifests (`manifest.{oob,barrier,race,ptr}.json`) drive the per-check
ctests (`verify_bounds/verify_barrier/verify_race/verify_ptr`), and
`tests/verify/manifest.json` is the consolidated 34-entry view (all
eight classes). The `kernscope-verify` lit suite (part of
`check-kernscope`) runs the CLI `--json` over EVERY corpus kernel and
diffs the emitted verdicts against the manifest read **live** from disk —
corrupting an expected verdict fails the suite naming the kernel. The
CLI never sees the manifest: the gate validates the whole pipeline
against committed expectations rather than against itself.
