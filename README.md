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

(Placeholder — see `docs/research/owner-examples.md` for the canonical
per-project resume bullets.)
