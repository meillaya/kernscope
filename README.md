# kernscope

A unified C++ GPU-kernel tooling suite: a PTX/AMDGPU static analyzer,
a CUDA/HIP autotuning kernel generator, a static GPU kernel verifier,
a CUDA/HIP kernel capture-and-replay system, MLIR/LLVM GPU optimization
passes, and a tiny CUDA/ROCm profiler — all in one repo so the tools can
share infrastructure and validate each other.

## Modules

| Module | Description |
|---|---|
| `libs/` | Shared C++ infrastructure used across all tools (GPU architecture DB, PTX/AMDGPU parsers, occupancy/roofline engines, report engine, benchmark + correctness harnesses) |
| `tools/analyze` | **C1 static analyzer** — PTX/AMDGPU in; register pressure, occupancy limits + limiter, coalescing, bank conflicts, warp divergence, barrier usage, instruction mix, tensor-core/MFMA/WMMA usage, spill detection, latency bottlenecks |
| `tools/tune` | **C2 autotuner** — template-based CUDA/HIP kernel generator, correctness-gated benchmarking, JSON config cache keyed by GPU arch, roofline + occupancy estimates, variant-comparison reports |
| `tools/verify` | **C3 static verifier** — CUDA/HIP → clang → LLVM IR → 8 check classes (OOB, smem bounds, barriers, races, illegal pointer arithmetic, divergent barriers, uninitialized reads, address-space casts), eBPF-style diagnostics, approved-IR handoff |
| `tools/capture` | **C4 capture/replay** — CUDA via CUPTI callback/injection + LD_PRELOAD shim, HIP via rocprofiler-sdk intercept; VA-faithful capture, standalone C++ reproducer emission, replay correctness validation |
| `tools/pass` | **C5 LLVM passes** — out-of-tree LLVM-21 plugin: global-to-shared promotion, loop tiling, vectorized load/store rewrite, smem padding, redundant-load elimination, coalescing analysis (lit/FileCheck) |
| `tools/profile` | **C6 profiler** — kernel launch tracing, per-kernel timing, occupancy/register/smem estimates, hot-kernel ranking, compiler-aware "why slow" diagnostics (CUPTI on NVIDIA, rocprofiler-sdk live on AMD) |
| `tests/` | gtest (ctest, hardware labels `no-gpu`/`gpu-amd`/`gpu-nvidia-rental`), lit/FileCheck suite, NumPy golden harness, committed fixtures (≤ 64MB) |
| `docs/` | research briefs (docs/research/) + per-tool docs |
| `scripts/` | `ci.sh` CI gate, `configure.sh`, `rental-validate.sh` (owner-gated NVIDIA runs) |
| `reports/` | committed benchmark/profiling evidence, incl. `reports/rentals/` from owner-gated NVIDIA rental runs |

## Quick start

```bash
# enter the toolchain (LLVM/MLIR-21, CUDA headers, ROCm, python)
devenv shell

# configure + build + full CI gate:
#   configure, ctest -L "no-gpu|gpu-amd", lit (check-kernscope),
#   golden harness, compile checks, fixture size + NVIDIA-header guards
bash scripts/ci.sh
```

NVIDIA-runtime validation is owner-gated only — never auto-triggered:

```bash
bash scripts/rental-validate.sh --dry-run   # print the exact command sequence
bash scripts/rental-validate.sh             # exec (owner, on the rental host)
```

## Resume bullets

Every bullet maps to a committed artifact — the "(numbers from …)" pointers
are real files under `reports/` or `tests/`. All measurements are LOCAL
gfx1101 (AMD Radeon RX 7800 XT) runs unless stated otherwise; nothing here
claims rental/H100 results (they do not exist yet — `reports/rentals/` gets
them only when the owner produces them). See
`docs/research/owner-examples.md` for the canonical per-project bullets this
section adapts.

1. **Umbrella — one toolchain, one loop.** *"Built kernscope, a unified C++
   GPU-kernel tooling suite — static analyzer, autotuning kernel generator,
   static verifier, capture/replay system, LLVM optimization passes, and a
   profiler — sharing one infrastructure library so each tool validates the
   others; the five-stage loop (profile → capture → analyze → tune → verify)
   closes end-to-end with one command on local AMD hardware"* — the
   one-command loop (`scripts/demo.sh`) ran in 152 s under its 900 s budget
   and produced one aggregated cross-tool report with the profiler hint
   referencing analyzer finding ids (numbers from
   `reports/demo/report.json`).

2. **C1 static analyzer.** *"Developed a C++ PTX/AMDGPU static analyzer
   modeling register pressure, occupancy, memory coalescing, bank conflicts,
   and warp divergence, producing optimization reports for generated
   CUDA/ROCm kernels"* — 10 analysis classes over 24 committed PTX/AMDGPU
   corpus files; on
   the captured layernorm_fused kernel's ISA it emitted 216 ALU-instruction
   metrics, 1 warning (possibly-divergent branch) and 3 info findings
   (numbers from `reports/demo/stage-3-analyze.json`,
   `docs/analyzer.md`).

3. **C2 autotuning kernel generator.** *"Built a C++ CUDA/HIP autotuning
   kernel generator for GEMM, softmax, layernorm, and fused transformer ops,
   exploring tile sizes, warp mappings, vector widths, and shared-memory
   layouts to select architecture-specific kernels"* — GPU-free pre-screen
   rejects infeasible configs before any compile; a launch-config search
   over the captured kernel validated 4 candidates bitwise and picked the
   best grid/block (numbers from `reports/demo/stage-4-capture-tune.json`,
   `reports/tune/`, `docs/tuning.md`).

4. **C3 static verifier.** *"Designed a C++ static verifier for GPU kernels
   analyzing LLVM-IR memory accesses, synchronization, address spaces, and
   barrier safety before backend lowering"* — 8 check classes with
   eBPF-style reject-with-reason diagnostics, a committed bug+clean corpus
   (every check class has both a bug and a clean example), and verified-IR
   handoff; the same gate rejects an injected out-of-bounds variant inside
   the tuning loop (numbers from `tests/verify/`,
   `reports/demo/stage-5-verify-tune.json`, `docs/verify.md`).

5. **C4 capture/replay.** *"Built a CUDA/HIP kernel capture-and-replay
   system intercepting runtime launches (CUPTI + LD_PRELOAD shim on NVIDIA,
   rocprofiler-sdk on AMD), snapshotting device inputs, and emitting
   standalone C++ reproducers for isolated benchmarking, debugging, and
   compiler optimization"* — the layernorm_fused bundle replayed
   bitwise-EQUAL with VA-faithful capture on gfx1101 (numbers from
   `reports/demo/capture/replay.txt`, `docs/capture.md`).

6. **C5 LLVM passes.** *"Implemented LLVM GPU optimization passes for
   shared-memory promotion, vectorized memory access, loop tiling, and
   redundant load elimination, with FileCheck tests and CUDA benchmark
   validation"* — an out-of-tree LLVM-21 plugin whose layernorm
   before/after delta (vector width 1 → 4, 0 → 12 vectorized global loads)
   is measured by the analyzer, not self-reported (numbers from
   `reports/passes/layernorm/`, `reports/demo/stage-5-pass-delta.json`).

7. **C6 profiler.** *"Created a C++ CUDA/ROCm profiler correlating kernel
   runtime, register pressure, shared-memory usage, occupancy, and emitted
   PTX/AMDGPU assembly to produce compiler-aware optimization
   diagnostics"* — rocprofiler-sdk live tracing on gfx1101, hot-kernel
   ranking, and a memory-bound hint whose evidence cites the roofline ridge
   point and cross-references analyzer findings (numbers from
   `reports/demo/stage-1-profile.json`, `reports/peaks/gfx1101-bench.jsonl`,
   `docs/profile.md`).

## Reproducing everything

```bash
devenv shell                                   # enter the pinned toolchain
bash scripts/reproduce-all.sh                  # lock check -> ci.sh -> demo.sh -> dashboard
```

`scripts/reproduce-all.sh` is the master reproducibility entry point —
details and the owner-executed NVIDIA rental runbook live in
`docs/reproducibility.md`. Architecture notes (the loop, the dependency
graph, the dual-path hardware strategy, the CUDA-13 migration surface) live
in `docs/architecture.md`.
