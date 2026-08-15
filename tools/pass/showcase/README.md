# tools/pass/showcase/ — todo-42 showcase kernels + before/after artifact pipeline

Three portable CUDA/HIP kernels (one fresh rewrite each; the
`/home/mei/projects/tensors/cuda/kernels` sources were used as SEMANTIC
reference only — nothing copied), each with a deliberate pass-pipeline
story:

| kernel | dir | pipeline story | report delta |
|---|---|---|---|
| gemm | `gemm/kernel.cu` | literal 64x64x100 tiled sgemm: `kernscope-tile` (tile 32) -> `kernscope-promote` (a-row window {i*100,+,1} -> addrspace(3) cooperative load + barrier0; b-access {j,+,64} honestly declined: v1 unit-stride only) | `smem_promoted: true` |
| layernorm | `layernorm/kernel.cu` | block-per-row, 4x unrolled feature loop over a 16B-aligned `__device__` global: `kernscope-vectorize` merges the 4 consecutive loads into one `<4 x float>` load | `vector_width: 1 -> 4` |
| softmax | `softmax/kernel.cu` | row-per-thread with RUNTIME row length: every pass declines (no SCEV-provable trip, non-unit-stride window, no chains) | `no-transform: true` (the honest QA-negative case) |

Each source compiles TWO ways (the portable pattern in the file headers):

1. **clang** (NVPTX IR + PTX, compile-only — no NVIDIA runtime anywhere):
   clang needs a staged CUDA root (the lit `_stage_cuda_root` idiom:
   `include/` from `$CUDA_HOME` + `bin/`/`nvvm/` from the nvcc package +
   the curand shim), or `-nocudainc` with the `#else` self-contained decl
   branch. `scripts/run_pipeline.sh` stages the root under `build/` the
   same way `tests/pass/lit.cfg.py` does.
2. **hipcc `--genco --offload-arch=gfx1101`** (live hsaco for the AMD
   analog bench on this machine's RX 7800 XT).

## The artifact pipeline (deliverable 4 of todo 42)

`scripts/run_pipeline.sh <kernel>` runs, per showcase:

```
kernel.cu --clang sm_80--> before.ll
  (sed-annotate: kernscope.tile.arch/size + kernscope.promote.arch fn
   attributes — the fixture mechanism; the annotation is a documented part
   of the pipeline, not a hidden edit)
before.ll --opt (kernscope-tile,kernscope-promote,kernscope-vectorize,
             kernscope-pad,kernscope-coalesce)--> after.ll
before.ll/after.ll --llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80-->
  before.ptx / after.ptx                      (compile-only, exit-0 gate)
llvm-diff before.ll after.ll -> summary      (committed)
kernscope-coalesce -report on before+after   (classification deltas)
```

Artifacts land in `reports/passes/<kernel>/` (`before.ll`, `after.ll`,
`before.ptx`, `after.ptx`, `llvm-diff.txt`, `report.json`).

## The live AMD-analog bench (Metis L-6 venue rule)

NVIDIA timings are rental-gated — locally the PTX is COMPILE-ONLY. The
measured numbers come from the AMD analog: `bench/bench_driver.cpp`
(committed) links the todo-22 harness (`libs/bench` + `libs/db` +
`libs/gpuruntime`, header/source reuse — `libs/*` is not modified) and
benchmarks each kernel's hsaco on gfx1101 through the same measured
provenance path as `tools/bench` (device query + rocm-smi clocks
as-measured + `mode: "measured"`), appending records to
`reports/passes/bench/gfx1101-showcase-bench.jsonl` (the
`reports/peaks/gfx1101-bench.jsonl` provenance format, mirrored).
