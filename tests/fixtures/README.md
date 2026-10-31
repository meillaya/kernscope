# tests/fixtures — golden fixtures & toolchain smoke gates

Committed, ≤ 64 MB total (asserted by `scripts/ci.sh`). Owned by foundation
todo 5. Everything here is produced with GPU-less tooling except the two
stages that explicitly require the local AMD GPU (hipcc run, rocminfo).

## Contents

| Path | What it is |
|---|---|
| `fused_attention.cu` | fused-attention-style CUDA kernel engineered to exhibit every owner example class (docs/report/kernscope-technical-report.pdf) |
| `fused_attention.ptx` | committed PTX: `nvcc -arch=sm_86 -lineinfo --maxrregcount=112 -ptx` (ptxas runs GPU-less) |
| `expected_fused_attention.json` | numeric expected analyzer report (occupancy 0.375, limiter, warnings, suggestions) |
| `manifest.json` | pins arch/block/occupancy/limiter/registers/warnings/suggestions (`schema_version: 1`) |
| `validate_manifest.py` | asserts the manifest pins (exit 0 / non-zero) |
| `clang_cuda_smoke.cu` | minimal `__syncthreads` + `<<< >>>` kernel for the clang↔CUDA header gate |
| `hip_smoke.cpp` | trivial HIP kernel, compiled `--offload-arch=gfx1101` and RUN on the local GPU |
| `layernorm_fused/` | HIP fused layer-norm app (kernel+host, CMakeLists) + NumPy golden inputs/outputs + `check_layernorm.py` |
| `README.md` | this file |

Entry points: `python3 tests/fixtures/validate_manifest.py`,
`python3 tests/fixtures/layernorm_fused/check_layernorm.py`,
`bash scripts/smoke_toolchain.sh` (all inside `devenv shell`).

## fused_attention: the engineered classes

| # | Class | Mechanism |
|---|---|---|
| a | ~112 regs/thread (accumulator) | per-thread `float acc[64]` with the V-multiply r-loop fully unrolled so the array is register-resident |
| b | divergent branch inside inner loop | `if ((lane & 7) == (j & 7))` in the score j-loop — threadIdx-derived, warp-divergent for every j |
| c | 4-way bank conflict | `sK[tid][j]` / `sV[tid][k]` column-major accesses with row stride 68 floats (HEAD_DIM+4 pad); 68 % 32 == 4 → 8 distinct banks → 4-way |
| d | non-128B-coalesced global loads | Q tile staged via stride-2 gathers; each warp access spans 256 B for 128 B used → efficiency 0.5 |

Plus: two `__syncthreads()` barriers, and a mix of fma / ld.shared /
ld.global / st.shared / st.global / ex2.approx / setp / `@p`-branches for the
instruction-mix analysis (counts in `expected_fused_attention.json`).

The tile working set (66 KiB) exceeds the 48 KiB static shared-memory limit,
so tiles live in **dynamic shared memory** (`extern __shared__`, opt-in via
`cudaFuncSetAttribute`, as real fused-attention kernels do). The fixture is a
compile-only golden — never executed locally (no NVIDIA GPU).

### Occupancy math (review-verified, sm_86 / 64-thread blocks)

Per docs/report/kernscope-technical-report.pdf: sm_86 has 65,536
registers/SM, 48 max warps/SM, 16 max blocks/SM; register allocation uses a
256-register-per-warp granule: `per_warp = ceil(regs × 32 / 256) × 256`.

```
regs/thread   per-warp alloc   per-block (2 warps)   blocks/SM   warps resident   occupancy
 96            12×256=3072          6144              10           20             20/48=0.4167
 97–104        13×256=3328          6656               9           18             18/48=0.375
105–112        14×256=3584          7168               9           18             18/48=0.375
113            15×256=3840          7680               8           16             16/48=0.333
```

Any compiled register count in **[97, 112]** therefore yields EXACTLY
`9 blocks × 2 warps / 48 max warps = 0.375` occupancy with limiter
"registers" — never deviating. The STOP-THE-LINE rule: the golden is only
accepted when `ptxas -v` reports a count in 97–112; the current golden is
**102 registers, 0 spill stores, 0 spill loads** (see `manifest.json`).

## clang/CUDA header gate

Stage (a) of `scripts/smoke_toolchain.sh` compiles `clang_cuda_smoke.cu`
(headers + `__syncthreads` + `<<< >>>`) with the unwrapped clang 21.1.8:

```
clang -x cuda --no-cuda-version-check --cuda-path=<synthetic root> -nocudalib \
      --cuda-host-only  -emit-llvm -c clang_cuda_smoke.cu   # launch syntax
clang -x cuda --no-cuda-version-check --cuda-path=<synthetic root> -nocudalib \
      --cuda-device-only -emit-llvm -c clang_cuda_smoke.cu  # __syncthreads →
                                                            # llvm.nvvm.barrier*
```

`--no-cuda-version-check` is MANDATORY: clang 21.1.8's CUDA support tops out
below 12.6, so without it the version gate rejects the toolkit (see the
negative QA evidence in .omo/evidence/task-5-gpu-kernel-suite-negative.txt).

The devenv's nix CUDA packages split the toolkit, so the gate stages a
**synthetic single-root CUDA layout** (in its temp dir): `include/` = cudart
headers + the nvcc package's `crt/` + a minimal **curand shim**,
`bin/` + `nvvm/libdevice` from the nvcc package. Two reasons this is
necessary: (1) clang 21's `CudaInstallationDetector` requires `include/`,
`bin/`, and `nvvm/libdevice` to exist; (2) its
`__clang_cuda_runtime_wrapper.h` force-includes `curand_mtgp32_kernel.h`,
which the devenv's package set (no `cuda_curand`) does not ship. The shim is
a test-owned stub generated by the gate script — NO NVIDIA header is vendored
(guarded by ci.sh).

Two other nix-toolchain quirks, both documented here because they are
deterministic:

- The nix **clang wrapper** bakes host-only flags (`-mtls-dialect=gnu2`,
  `-fzero-call-used-regs=…`, hardening) into every invocation; nvptx64
  rejects them, so the gate uses the **unwrapped** clang binary (derived
  from the wrapper's own hardcoded path).
- clang 21 lowers `<<< >>>` through `cudaConfigureCall` when it does not see
  a CUDA ≥ 9 SDK version on the host target; the declaration is
  nvcc-frontend-internal and absent from the nix packages, so
  `clang_cuda_smoke.cu` declares it (`extern "C"`) and uses the full
  four-argument launch form `<<<4, 256, 0, 0>>>`.

### Fallback branch (if clang↔CUDA-12.6 headers ever break)

If the header gate cannot be made green (e.g. the nix package split changes
again), the corpus switches to self-contained device code that depends on
neither the CUDA headers nor libdevice:

- compile the smoke kernel with `-nocudainc` (no CUDA include paths);
- replace `__syncthreads()` with the intrinsic directly:
  `asm("bar.sync 0");` (or `call void @llvm.nvvm.barrier0()` / the newer
  `llvm.nvvm.barrier.cta.sync.aligned.all(i32 0)` in IR);
- replace `<<< >>>` with an explicit `cudaConfigureCall`/launch-style host
  call or drop the launch site in favor of a bare kernel definition;
- declare `threadIdx`/`blockIdx`/`blockDim` as
  `extern __device__ unsigned threadIdx;` etc. when compiling without the
  clang runtime wrapper.

That branch is only exercised if stage (a) regresses; it is not currently
active.

## layernorm_fused

HIP fused layer-norm (load tile → mean → variance → normalize, bias/scale),
block-per-row with a shared-memory tree reduction (`ds_read`/`ds_write`
traffic for the AMD path). Golden convention (polykernel,
/home/mei/projects/polykernel/README.md:116-120): **bf16
round-to-nearest-even inputs, fp32 accumulate, bf16 output**; thresholds
cosine ≥ 0.999, max rel err ≤ 1e-2, PCC ≥ 0.99. The bf16↔fp32 conversions in
`layernorm.hip` are bit-exact RNE, verified against `ml_dtypes.bfloat16`.

- `golden_inputs.npz` / `golden_outputs.npz` — committed goldens
  (`generate_golden.py` regenerates; `input.bin` is a scratch artifact the
  checker rebuilds and does NOT commit).
- `check_layernorm.py` — builds (hipcc `--offload-arch=gfx1101`), runs the
  app on the local GPU, compares against the golden; exit 0 on pass.
- `CMakeLists.txt` — standalone CMake target (not wired into the root
  superbuild, which belongs to foundation todo 3):
  `cmake -B build -S tests/fixtures/layernorm_fused && cmake --build build`.

## Expected report model (todo 7 shape)

`expected_fused_attention.json` is numeric-only (Metis C-3: no prose
assertions). Shape: `kernel` identity, `metrics` map (incl. occupancy,
limiter, instruction mix), `warnings[]` / `suggestions[]` each with `id`,
`severity`, `evidence`. `manifest.json` pins the review-mandated combo;
`validate_manifest.py` asserts it.
