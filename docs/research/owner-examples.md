# Owner Examples: Outputs & Resume Bullets

Single source of truth for the owner's example outputs and resume bullets,
referenced by the kernscope work plan.

---

## Section A — fused_attention analyzer example

Example output of the static analyzer (Project 2) on a fused attention kernel:

```
Kernel: fused_attention.ptx
Occupancy: 37.5%
Main limiter: 112 registers/thread
Warnings:
  - shared memory stride likely causes 4-way bank conflicts
  - divergent branch inside inner loop
  - global loads not 128B coalesced
Suggested:
  - tile K dimension by 32
  - split accumulator fragment
  - move predicate outside loop
```

---

## Section B — layernorm_fused profiler example

Example output of the profiler (Project 7) on a fused layer-norm kernel:

```
Kernel: layernorm_fused
Time: 41.2 us
Registers/thread: 96
Shared memory/block: 24 KB
Likely limiter: occupancy
Compiler hint:
  accumulator array likely expanded into registers;
  consider split reduction or lower unroll factor.
```

---

## Section C — per-project resume bullets

### Project 2 — PTX/AMDGPU static analyzer
Static analyzer that models register pressure, occupancy, memory coalescing,
bank conflicts, and warp divergence, producing optimization reports for
generated CUDA/ROCm kernels.

**Resume bullet:**
"Developed a C++ PTX/AMDGPU static analyzer that models register pressure, occupancy, memory coalescing, bank conflicts, and warp divergence, producing optimization reports for generated CUDA/ROCm kernels."

### Project 3 — autotuning kernel generator
C++ CUDA/HIP autotuning kernel generator for GEMM, softmax, layernorm, and
fused transformer ops, exploring tile sizes, warp mappings, vector widths,
and shared-memory layouts to select architecture-specific kernels.

**Resume bullet:**
"Built a C++ CUDA/HIP autotuning kernel generator for GEMM, softmax, layernorm, and fused transformer ops, exploring tile sizes, warp mappings, vector widths, and shared-memory layouts to select architecture-specific kernels."

### Project 4 — GPU kernel verifier
Static verifier for GPU kernels that analyzes LLVM/MLIR-level memory accesses,
synchronization, address spaces, and barrier safety before lowering to
CUDA/ROCm backends.

**Resume bullet:**
"Designed a C++ static verifier for GPU kernels that analyzes LLVM/MLIR-level memory accesses, synchronization, address spaces, and barrier safety before lowering to CUDA/ROCm backends."

### Project 5 — kernel capture and replay
CUDA/HIP kernel capture-and-replay system that intercepts runtime launches,
snapshots device inputs, and emits standalone C++ reproducers for isolated
benchmarking, debugging, and compiler optimization.

**Resume bullet:**
"Built a CUDA/HIP kernel capture-and-replay system that intercepts runtime launches, snapshots device inputs, and emits standalone C++ reproducers for isolated benchmarking, debugging, and compiler optimization."

### Project 6 — LLVM pass for GPU memory optimization
MLIR/LLVM GPU optimization passes for shared-memory promotion, vectorized
memory access, loop tiling, and redundant load elimination, with FileCheck
tests and CUDA benchmark validation.

**Resume bullet:**
"Implemented MLIR/LLVM GPU optimization passes for shared-memory promotion, vectorized memory access, loop tiling, and redundant load elimination, with FileCheck tests and CUDA benchmark validation."

### Project 7 — tiny CUDA/ROCm profiler
C++ CUDA/ROCm profiler that correlates kernel runtime, register pressure,
shared-memory usage, occupancy, and emitted PTX/AMDGPU assembly to produce
compiler-aware optimization diagnostics.

**Resume bullet:**
"Created a C++ CUDA/ROCm profiler that correlates kernel runtime, register pressure, shared-memory usage, occupancy, and emitted PTX/AMDGPU assembly to produce compiler-aware optimization diagnostics."
