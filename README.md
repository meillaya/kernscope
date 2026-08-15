# kernscope

## What is kernscope

kernscope is a unified C++ toolchain for GPU kernel engineering: a static
analyzer, an autotuning kernel generator, a static verifier, a capture-and-
replay system, LLVM optimization passes, and a profiler, all in one repo so
the tools share infrastructure and validate each other. It speaks both
vendor stacks: NVIDIA CUDA (including PTX, NVIDIA's virtual instruction set,
and CUPTI, NVIDIA's profiling interface) and AMD HIP (AMD's CUDA-compatible
API, traced with rocprofiler-sdk).

> Six CLIs, one shared library stack, the loop closes on this machine.

## Why: today's GPU work is a pile of disconnected tools

Writing a fast, correct GPU kernel usually means juggling tools that never
talk to each other. The vendor profiler tells you a kernel is hot, but not
why. The disassembler shows you assembly, and you interpret it by eye. You
rewrite variants by hand and benchmark them ad hoc. Your tests check whole
programs, so a bad change surfaces late, and the fix reaches yet another
tool.

Each step gets reinvented per project. Static analysis, transformation,
verification, and benchmarking live in separate repos with separate
conventions, so findings rarely agree and evidence rarely survives.

kernscope replaces that pile with one closed loop. Each stage consumes the
previous stage's real output, and the tools validate each other instead of
self-reporting:

profile → capture → analyze → transform/generate → verify → benchmark

## Core concepts

### The loop

```
profile ──► capture ──► analyze ──► transform/generate ──► verify ──► benchmark
 (C6)        (C4)        (C1)          (C2 tune / C5 pass)   (C3)      (libs/bench)
   ▲                                                                    │
   └──────────────────── delta reports feed back ───────────────────────┘
```

Walk through each stage:

- **profile (C6, kernscope-profile)**: trace a running binary, time every
  kernel launch, rank the hot kernels, and explain "why is this slow" with
  compiler-aware hints. Uses CUPTI on NVIDIA, rocprofiler-sdk live on AMD.
- **capture (C4, kernscope-capture)**: record the hot kernel's launches,
  inputs, and virtual addresses into a bundle that replays bitwise, and emit
  a standalone C++ reproducer.
- **analyze (C1, kernscope-analyze)**: static analysis over the *captured*
  kernel's actual ISA text, PTX or AMDGPU assembly. Ten checks: register
  pressure, occupancy (the fraction of the GPU's thread slots a kernel can
  fill), memory coalescing, bank conflicts (threads in a warp hitting the
  same shared-memory bank, which serializes them), warp divergence (threads
  in a warp taking different branches), barrier usage, instruction mix,
  tensor-core use (WMMA on NVIDIA, MFMA on AMD), spill detection, and
  latency bottlenecks.
- **transform / generate (C2 kernscope-tune, C5 kernscope-pass)**: tune
  generates template variants of GEMM, softmax, layernorm, and similar
  kernels; pass rewrites LLVM IR (LLVM's intermediate representation) with
  optimizations such as global-to-shared promotion and loop tiling.
- **verify (C3, kernscope-verify)**: compile CUDA/HIP down to LLVM IR and
  check 8 error classes (out-of-bounds, shared-memory bounds, barriers,
  races, illegal pointer arithmetic, divergent barriers, uninitialized reads,
  address-space casts). It gates candidates before they compile.
- **benchmark (libs/bench)**: correctness-gated timing with an L2 flush and
  median plus p20/p80 quantiles, positioned on the roofline (the ceiling set
  by the slower of compute or memory throughput).

The loop is closed by design: the analyzer runs on the kernel capture from
stage 2, the verifier gates what the tuner generates, the profiler's hints
reference analyzer finding ids, and benchmark deltas feed back into the next
iteration. `scripts/demo.sh` runs all of it in one command on this machine
and emits one aggregated cross-tool report at `reports/demo/report.json`.

### Dual-path hardware strategy

Every NVIDIA-runtime capability has two implementations with different
venues, so the whole suite is testable on this machine with zero rentals:

- **CUDA is compile-only locally.** `nvcc` checks kernels for sm_80 and
  sm_90 (NVIDIA compute-capability codes for Ampere and Hopper) against
  headers the nix store provides. Nothing NVIDIA runs on this machine.
- **HIP runs live.** `hipcc` compiles for gfx1101, the AMD architecture code
  for the RDNA 3 RX 7800 XT on this desk, and the kernels actually run and
  benchmark.
- **NVIDIA runtime is owner-gated.** Live NVIDIA behavior, such as CUPTI
  capture and replay, runs only on rented A100/H100-class hosts through
  `scripts/rental-validate.sh`, and only by the owner.

### The four-layer test convention

1. **lit/FileCheck**: LLVM's test driver and pattern matcher verify the LLVM
   passes and the parsers (`check-kernscope`).
2. **gtest/ctest**: C++ unit and integration tests, run through CMake's
   CTest and tagged with hardware labels.
3. **NumPy bf16 golden**: generated kernels must match reference outputs in
   bfloat16 (bf16, a 16-bit float format) under strict tolerances.
4. **Generated-code compile checks**: every template variant must compile
   with `nvcc` (compile-only) and `hipcc` before it is considered.

Each layer fails fast and cheaply, so a regression stops at the cheapest
test that can see it.

### Honesty guards

- Reports carry `schema_version: "1.0"` and per-module `provenance.mode`:
  `measured` (live gfx1101 runs) or `projected` (static analysis and
  compile-only artifacts). `tests/report/check_dashboard.py` asserts the
  flags; projected numbers are never presented as measured.
- No vendored NVIDIA headers are committed. The nix store resolves
  `#include` names at build time, and ci.sh guards both filenames and
  content.
- `reports/rentals/` does not exist yet. NVIDIA runtime numbers appear there
  only when the owner produces them, and the README will not claim them
  meanwhile.

## Quick start

```bash
# enter the pinned toolchain (LLVM/MLIR-21, CUDA headers, ROCm, python)
devenv shell

# configure + build + full CI gate:
#   configure (ci.sh stage 1 calls scripts/configure.sh),
#   ctest -L "no-gpu|gpu-amd", lit (check-kernscope),
#   golden harness, compile checks, fixture size + NVIDIA-header guards
bash scripts/ci.sh
```

Then close the loop in one command:

```bash
# one-command end-to-end demo on the local gfx1101:
#   profile -> capture -> analyze -> tune -> verify
# requires the build from ci.sh and an AMD GPU (/dev/kfd)
# writes reports/demo/report.json (aggregated, cross-tool)
bash scripts/demo.sh
```

NVIDIA-runtime validation is owner-gated only, never auto-triggered:

```bash
bash scripts/rental-validate.sh --dry-run   # print the exact command sequence
bash scripts/rental-validate.sh             # exec (owner, on the rental host)
```

## Reproducing everything

```bash
devenv shell                                   # enter the pinned toolchain
bash scripts/reproduce-all.sh                  # lock check -> ci.sh -> demo.sh -> dashboard
```

`scripts/reproduce-all.sh` is the master reproducibility entry point.
Details and the owner-executed NVIDIA rental runbook live in
`docs/reproducibility.md`. Architecture notes (the loop, the dependency
graph, the dual-path hardware strategy, the CUDA-13 migration surface) live
in `docs/architecture.md`.

## The six tools

| Module | Description |
|---|---|
| `libs/` | Shared C++ infrastructure used across all tools (GPU architecture DB, PTX/AMDGPU parsers, occupancy/roofline engines, report engine, benchmark + correctness harnesses) |
| `tools/analyze` | **C1 static analyzer**: PTX/AMDGPU in; register pressure, occupancy limits + limiter, coalescing, bank conflicts, warp divergence, barrier usage, instruction mix, tensor-core/MFMA/WMMA usage, spill detection, latency bottlenecks |
| `tools/tune` | **C2 autotuner**: template-based CUDA/HIP kernel generator, correctness-gated benchmarking, JSON config cache keyed by GPU arch, roofline + occupancy estimates, variant-comparison reports |
| `tools/verify` | **C3 static verifier**: CUDA/HIP → clang → LLVM IR → 8 check classes (OOB, smem bounds, barriers, races, illegal pointer arithmetic, divergent barriers, uninitialized reads, address-space casts), eBPF-style diagnostics, approved-IR handoff |
| `tools/capture` | **C4 capture/replay**: CUDA via CUPTI callback/injection + LD_PRELOAD shim, HIP via rocprofiler-sdk intercept; VA-faithful capture, standalone C++ reproducer emission, replay correctness validation |
| `tools/pass` | **C5 LLVM passes**: out-of-tree LLVM-21 plugin: global-to-shared promotion, loop tiling, vectorized load/store rewrite, smem padding, redundant-load elimination, coalescing analysis (lit/FileCheck) |
| `tools/profile` | **C6 profiler**: kernel launch tracing, per-kernel timing, occupancy/register/smem estimates, hot-kernel ranking, compiler-aware "why slow" diagnostics (CUPTI on NVIDIA, rocprofiler-sdk live on AMD) |
| `tests/` | gtest (ctest, hardware labels `no-gpu`/`gpu-amd`/`gpu-nvidia-rental`), lit/FileCheck suite, NumPy golden harness, committed fixtures (≤ 64MB) |
| `docs/` | research briefs (docs/research/) + per-tool docs |
| `scripts/` | `ci.sh` CI gate, `configure.sh`, `rental-validate.sh` (owner-gated NVIDIA runs) |
| `reports/` | committed benchmark/profiling evidence, incl. `reports/rentals/` from owner-gated NVIDIA rental runs |

Two shorthand terms: *smem* means shared memory, and eBPF-style diagnostics
are concise verdict lines in the style of the Linux eBPF (extended Berkeley
Packet Filter) verifier.

### What you type, what you get

| CLI | You type | You get |
|---|---|---|
| `kernscope-analyze` | `kernscope-analyze fused_attention.ptx` | static findings: occupancy %, the register count that limits it, bank conflicts, divergence, plus suggestions |
| `kernscope-tune` | `kernscope-tune --kernel matmul --shape 64,64,64 --arch gfx1101 --json` | correctness-gated benchmark across launch configs, best config, JSON report |
| `kernscope-verify` | `kernscope-verify kernel.cu --json` | LLVM IR-level verdicts across the 8 check classes |
| `kernscope-capture` | `kernscope-capture record ./my_app input.bin output.bin --bundle app.bundle` | a bundle that replays bitwise (`kernscope-capture replay app.bundle`) |
| `kernscope-pass-report` | `kernscope-pass-report --report-pipeline layernorm --build-dir build` | analyzer-measured before/after deltas for the LLVM pass pipeline (vector width, vectorized loads) |
| `kernscope-profile` | `kernscope-profile build/tests/fixtures/layernorm_fused --json` | per-kernel timings, hot-kernel ranking, occupancy/register/smem estimates, "why slow" hints |

## Repository layout

- `libs/`: shared infrastructure: GPU architecture DB, PTX/AMDGPU parsers,
  occupancy + roofline engines, golden + bench harnesses, report engine,
  runtime wrapper, CUPTI subscription. No tool depends downward on another
  tool.
- `tools/`: the six tools (`analyze`, `tune`, `verify`, `capture`, `pass`,
  `profile`) plus `bench`, `dashboard`, and `ptxdump`.
- `kernels/`: template sources the tuner and the compile-check gates use.
- `tests/`: gtest/ctest suite, lit/FileCheck tests, golden harness,
  committed fixtures (≤ 64MB total, conventions in
  `tests/fixtures/README.md`).
- `reports/`: committed evidence: `demo/`, `passes/`, `peaks/`, `tune/`,
  `dashboard.html`/`.json`. `reports/rentals/` is reserved for
  owner-produced NVIDIA numbers.
- `scripts/`: `ci.sh`, `configure.sh`, `demo.sh`, `reproduce-all.sh`,
  `rental-validate.sh`, `compile_checks.sh`, `smoke_toolchain.sh`.
- `docs/`: architecture, reproducibility, per-tool references, research
  briefs, JSON schemas.
- `THIRD-PARTY.md`: vendor license notes.

## Documentation

- `docs/architecture.md`: the loop, the library dependency graph, the
  dual-path hardware strategy, the CUDA-13 migration surface.
- `docs/reproducibility.md`: the master `reproduce-all.sh`, toolchain pins,
  clean-checkout CI, the owner-executed NVIDIA rental runbook.
- Per tool:
  - C1 analyzer: `docs/analyzer.md`, `docs/analyzer-rules.md`
  - C2 tuner: `docs/tuning.md`
  - C3 verifier: `docs/verify.md`, `docs/verify-soundness.md`
  - C4 capture: `docs/capture.md`, `docs/capture-format.md`,
    `docs/capture-mapping.md`
  - C5 passes: `tools/pass/README.md`
  - C6 profiler: `docs/profile.md`
- `docs/research/`: cited research briefs (analyzer, verifier, autotuner,
  profiler, memory passes, capture/replay, RDNA3 constants, owner examples).
- `docs/schema/`: `report.schema.json`, `capture-bundle.schema.json`.
- `reports/research-report/`: 18-page research report (report.tex + compiled
  report.pdf) with TikZ diagrams and 70+ references, covering the six-tool
  loop, dual-path hardware strategy, and honesty/provenance design.

## Testing & CI

`scripts/ci.sh` runs the verification pipeline in order and exits non-zero
on the first failing stage:

1. configure + build (`scripts/configure.sh`, then `cmake --build build`)
2. `ctest -L "no-gpu|gpu-amd"` (hardware-labeled suite), with a
   non-decreasing test-count guard against `scripts/ci-baseline.txt`
3. lit suite: `cmake --build build --target check-kernscope`
4. golden harness: `pytest tests/golden` (NumPy bf16 comparisons)
5. generated-code compile checks: `scripts/compile_checks.sh` (nvcc
   sm_80/sm_90 compile-only + hipcc gfx1101 compile + live golden run)
6. committed-fixture size assertion (`tests/fixtures` ≤ 64MB total)
7. NVIDIA-header guards (vendored filename + content)

Test labels say which hardware a test needs:

| Label | Meaning |
|---|---|
| `no-gpu` | always runs, no hardware required |
| `gpu-amd` | needs the local AMD GPU (`/dev/kfd`) |
| `gpu-nvidia-rental` | owner-executed NVIDIA rental runs, never auto-triggered |

The gate must pass from a clean checkout with no `build/` directory; the
procedure is in `docs/reproducibility.md`.

## Honesty & provenance

- The README and dashboard report local gfx1101 numbers only.
- Every report carries `schema_version: "1.0"` and per-module
  `provenance.mode` = `measured` (live gfx1101 runs) | `projected` (static
  analysis / IR+PTX compile-only artifacts).
  `tests/report/check_dashboard.py` asserts the honesty flags; projected
  numbers are never presented as measured.
- No NVIDIA runtime claims without `reports/rentals/`. That directory does
  not exist yet; rental numbers appear there only when the owner produces
  them via the runbook, at which point this README may reference them.
- No vendored NVIDIA headers are committed (ci.sh filename + content
  guards); the nix store resolves `#include <cuda_occupancy.h>`-style names
  at build time.
- Rental runbook: `scripts/rental-validate.sh` (with `--dry-run` first) and
  `docs/reproducibility.md`.
