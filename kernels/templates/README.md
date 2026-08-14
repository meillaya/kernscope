# kernels/templates — portable CUDA/HIP kernel template convention (todo 24)

The single-source portable kernel convention that todos 25-27 (tune track)
instantiate and todo 44 (capture circularity breaker) consumes. One template
source compiles under **nvcc for sm_80 and sm_90** (compile-only, GPU-less)
AND under **hipcc for gfx1101** (compile AND run on this machine), with
correctness validated by the [golden harness](../../tests/golden/).

```
kernels/templates/
├── CMakeLists.txt          kernscope_add_cuda_kernel / kernscope_add_hip_kernel
├── include/kernscope/
│   └── kernel_traits.h     the ONLY platform-dependent part (ks_bf16 + guards)
├── reference/
│   ├── elementwise_add.hip  out[i] = a[i] + b[i]  (bf16, fp32 add)
│   └── matmul_skeleton.hip  tiled C = A@B, 16x16 smem tile, fp32 accumulate
├── tools/
│   └── hip_template_runner.cpp  generic live runner (gpuruntime path)
└── tests/
    └── run_hip_gate.py      live gate: gen → run → golden check
```

## The single-source convention

A template file includes **only** `kernscope/kernel_traits.h`, which selects
the platform with the polykernel-proven guard:

```cpp
#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
using ks_bf16 = hip_bfloat16;          // ROCm 7.2 type name
#else
#include <cuda_runtime.h>
#include <cuda_bf16.h>
using ks_bf16 = __nv_bfloat16;         // CUDA 12.6 type name
#endif
```

plus the device-side conversion helpers `ks_bf16_to_float` / `ks_float_to_bf16`
(fp32 is the accumulate type; the output is rounded to bf16 exactly once —
the todo-23 rounding contract). Kernels are `extern "C" __global__` with the
builtin launch identifiers (`blockIdx`/`blockDim`/`threadIdx`), identical on
both platforms.

**Kernel parameter contract** (used by the runner and the tune track):

```
(void) kernel(const T* input..., T* output, int32 scalar...)
```

device pointers are passed BY VALUE (uintptr) per the driver-API convention
proven in libs/gpuruntime (todo 21); scalars are int32 values.

### Toolchain facts pinned on this machine

| Fact | Detail |
|---|---|
| nvcc 12.6 | targets `sm_80`/`sm_90` **only** (NOT sm_100); `-x cu` is required because the sources carry the `.hip` extension |
| hipcc 7.2 | **no longer defines `__HIP_PLATFORM_AMD__` implicitly** (verified via `-dM` on host + `--genco` device passes) — the CMake function passes `-D__HIP_PLATFORM_AMD__=1` explicitly |
| HIP bf16 | ROCm 7.2 names it `hip_bfloat16` (struct with RNE float ctor / `operator float()`); CUDA names it `__nv_bfloat16` |

## CMake functions

### `kernscope_add_cuda_kernel(name SOURCES <file.hip>...)`

nvcc **compile-only** gates, one ctest test per arch (`LABELS no-gpu`):

```
nvcc -arch=sm_80 -lineinfo -x cu -std=c++20 -I<include> -c <src> -o <obj>
nvcc -arch=sm_90 -lineinfo -x cu -std=c++20 -I<include> -c <src> -o <obj>
```

`-lineinfo` is the profiler line-info flag (no device needed). Gates:
`kernscope.templates.<name>.cuda.sm_80` / `...sm_90`.

### `kernscope_add_hip_kernel(name SOURCES <file.hip>... [RUN_* args])`

hipcc gate (ctest `kernscope.templates.<name>.hip.genco`, `LABELS no-gpu`):

```
hipcc --genco --offload-arch=gfx1101 -gline-tables-only -D__HIP_PLATFORM_AMD__=1 \
      -I<clr>/include -I<include> <src> -o <name>.hsaco
```

With the run arguments the function ALSO registers the LIVE gate
(`kernscope.templates.<name>.hip.live`, `LABELS gpu-amd`):

```cmake
kernscope_add_hip_kernel(elementwise_add
  SOURCES reference/elementwise_add.hip
  RUN_SYMBOL kernscope_templates_elementwise_add   # symbol inside the hsaco
  GOLDEN_FAMILY reduction                          # golden family (below)
  GOLDEN_SHAPE "2,1024"                            # golden --shape
  GRID "4"                                         # W or WxH
  BLOCK "256")                                     # X or XxY
```

The live gate builds the hsaco + `kernscope_templates_hip_runner` (linked
against `kernscope::gpuruntime`, the todo-21 module/kernel/launch/memcpy
path), then `tests/run_hip_gate.py`:

1. `cd tests/golden && python3 -m kernscope_golden gen --kernel <family> --shape <shape> --out ref.npz` (canonical CLI invocation — cwd MUST be `tests/golden`; never set PYTHONPATH),
2. dumps the npz input tensors to raw row-major `.bin` buffers,
3. runs the kernel via `hip_template_runner` with `--grid/--block/--in/--out/--out-size/--int` (runner CLI below),
4. `check`s the output `.bin` and **requires verdict `pass:true`** (cosine ≥ 0.999 ∧ max_rel_err ≤ 1e-2 ∧ pcc ≥ 0.99 — todo-23 thresholds).

A missing hsaco/runner or any non-zero step FAILS the gate loudly — there is
no silent-skip path.

### `hip_template_runner` CLI

```
hip_template_runner --hsaco <f> --kernel <sym> --grid <W[xH]> --block <X[xY]> \
    [--in <raw.bin>]... --out <out.bin> --out-size <bytes> [--int <i32>]...
```

Kernel args are marshalled as: `--in` device pointers (in order), then the
`--out` device pointer, then the `--int` scalars (in order) — matching the
kernel parameter contract above. Buffers are opaque bytes (bf16 = 2
bytes/element). Exit 0 on success; typed errors otherwise.

## Golden mapping

The golden harness exposes six families (matmul, bias_gelu, softmax,
layernorm, reduction, attention); there is no plain-add family. The
elementwise-add template therefore validates against **`reduction` with shape
`2,N`**: the reference `y[j] = sum_i x[i,j]` (axis-0 sum of a 2-row tensor)
is exactly `x[0] + x[1]` — the elementwise add of two vectors, in fp32, so
the kernel output is bit-identical to the reference. The matmul skeleton /
tuned matmul variants validate against `matmul` with shape `M,N,K` (run gate
wired in `run_hip_gate.py`, used from todo 25 on).

## How todos 25-27 instantiate this

- **Todo 25 (matmul generator)**: emit `.hip` sources that include
  `kernscope/kernel_traits.h` and declare `kernscope_templates_*` kernels
  with the parameter contract above; register each generated variant with
  `kernscope_add_cuda_kernel` (sm_80/sm_90 compile gates) and
  `kernscope_add_hip_kernel(... RUN_SYMBOL ... GOLDEN_FAMILY matmul ...)`
  for the live gfx1101 golden gate.
- **Todos 26/27 (elementwise/reduction families + attention)**: same
  registration pattern; `GOLDEN_FAMILY` selects the family, `GOLDEN_SHAPE`
  the validation shape, `GRID`/`BLOCK` the launch geometry.
- **Todo 44 (capture equivalence)**: the same hsaco/kernel built by
  `kernscope_add_hip_kernel` is captured on the HIP path; the CUDA-side
  bundle uses the identical source compiled with `kernscope_add_cuda_kernel`.

## Reference templates

| Template | Kernel symbol | Semantics | Gates |
|---|---|---|---|
| `reference/elementwise_add.hip` | `kernscope_templates_elementwise_add` | `out = a + b` (bf16, fp32 add) | nvcc sm_80/sm_90 + hipcc genco + **live run/golden** |
| `reference/matmul_skeleton.hip` | `kernscope_templates_matmul_skeleton` | tiled `C = A@B`, 16×16 tile, bounds-checked | nvcc sm_80/sm_90 + hipcc genco (live validation lands with todos 25/27) |

The skeleton deliberately has **no** WMMA/MFMA/cp.async variants — those are
later tracks (todos 25-27); an unsupported intrinsic on gfx1101 must FAIL the
compile gate (the gate names the arch in its command line:
`--offload-arch=gfx1101`), never skip silently.
