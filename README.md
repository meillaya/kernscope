# kernscope

A unified C++ GPU-kernel tooling suite: a PTX/AMDGPU static analyzer,
a CUDA/HIP autotuning kernel generator, a static GPU kernel verifier,
a CUDA/HIP kernel capture-and-replay system, MLIR/LLVM GPU optimization
passes, and a tiny CUDA/ROCm profiler — all in one repo so the tools can
share infrastructure and validate each other.

## Modules

| Module | Description |
|---|---|
| `libs/` | Shared C++ infrastructure used across all tools |
| `tools/` | The six tools (analyzer, autotuner, verifier, capture/replay, LLVM passes, profiler) |
| `tests/` |  |
| `docs/` |  |
| `scripts/` |  |
| `reports/` |  |

## Resume bullets

(Placeholder — see `docs/research/owner-examples.md` for the canonical
per-project resume bullets.)
