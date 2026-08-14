# kernscope-verify — core pipeline (todo 30)

CUDA/HIP → clang → LLVM IR → kernel launch-config tiers + eBPF-style
diagnostics + check-plugin registry. The foundation todos 31-34 (check
modules) and 35 (CLI + corpus) build on.

## Layout

| Path | Contents |
|---|---|
| `include/kernscope/verify/` | public API: `ingestion.h`, `launch_config.h` (tiers), `diagnostic.h` (+ `budget.h`), `verifier_state.h` (VerifierState + VerifierContext), `check_plugin.h` (the check-plugin contract) |
| `src/` | implementation (`ingestion.cpp`, `launch_config.cpp`, `verifier_state.cpp`, `budget.cpp`, `diagnostic.cpp`, `check_plugin.cpp`, `verify_core_main.cpp`) |
| `checks/{oob,barrier,race,ptr}/` | STUB check-module subdirectories (todos 31-34 fill only these) |
| `tests/` | `verify_core_test.cpp` + fixtures (`launch_const.cu`, `o2_unrolled.cu`, `hip_ingest.hip`, `bad_syntax.cu`) |

## Ingestion modes

**Default (host+device in ONE module).** The verifier needs the launch
configuration, which lives on the HOST side
(`__cudaPushCallConfiguration` / `cudaLaunchKernel` — research brief §4.6),
so `--cuda-device-only` is NEVER the default. Two clang passes run
(`--cuda-host-only` + `--cuda-device-only`, `-O1` pinned for verify-early
before loop destruction — brief §5.2) and the IR is merged with LLVM's
Linker into one module:

```
clang -x cuda --no-cuda-version-check --cuda-path=<staged root> -nocudalib \
      -gline-tables-only -O1 --cuda-host-only -emit-llvm -S \
      --cuda-gpu-arch=sm_80 <src> -o host.ll
clang -x cuda ... --cuda-device-only ... -o dev.ll
(host.ll + dev.ll linked in-process)
```

Two passes are required on this toolchain: clang 21's single `-emit-llvm`
runs fatbinary on the device bitcode and dies with "fatbinary elf
mismatch" (documented in tests/fixtures/README.md + todo-5 evidence). The
CUDA root is staged synthetically exactly like `scripts/smoke_toolchain.sh`
(include symlink farm + nvcc crt/bin/nvvm + test-owned curand shim). The
HIP path drives the UNWRAPPED clang 21 with `--hip-path=<ROCm root>`:
hipcc's bundled hipClang emits 1-arg lifetime intrinsics that LLVM 21's
parser rejects.

**`--device-only` (secondary, degraded).** Only the device pass runs;
launch dims are lost, so kernels without `nvvm.maxntid` degrade to
`LaunchTier::Symbolic` and the engine emits the "symbolic dims" degraded
verdict with a runtime-check suggestion. `__launch_bounds__` kernels still
resolve to sound upper bounds in this mode.

## Launch-config tiers (research brief §4.6)

- **(a) Exact** — constant dims at the launch site (`<<<32, 256, 0, 0>>>`):
  exact ranges propagate to `threadIdx.x in [0, 256)` and
  `blockIdx.x in [0, 32)`.
- **(b) UpperBound** — `__launch_bounds__` / `nvvm.maxntid`: sound upper
  bound on blockDim (e.g. `threadIdx.x in [0, 256)` even when the launch
  dims are runtime values).
- **(c) Symbolic** — runtime dims: degraded verdict + runtime-check
  suggestion.

`--dump-state` prints per-kernel module, tier, origin, exact dim ranges,
propagated ranges and budget consumption (the todo-30 acceptance gate).

## eBPF-style diagnostics + budget

`Diagnostic`: severity (info/warning/error), class id + name, source
location (DebugLoc + function + instruction), message, optional path.
The per-function explored-state budget (default 65536, `--budget N`)
mirrors the eBPF verifier's complexity limits: hitting the cap emits a
"budget exceeded" degraded verdict instead of hanging.

## Check plugins (todos 31-34)

The contract lives in `include/kernscope/verify/check_plugin.h`
(Doxygen'd): derive `CheckPlugin`, set `id`/`name`/`check_class_ids`,
implement `run(VerifierContext&)`, register with `REGISTER_CHECK(Class)`.
No shared file is edited — that is why the four check subdirectories are
pre-declared stubs.

## Harness (todo 30; full CLI is todo 35)

```
kernscope-verify-core <src.cu|src.hip> [--device-only] [--dump-state]
    [--budget N] [--opt O1|O2] [--arch sm_80|gfx1101]
```

Exit codes: 0 = ingested + analyzed (no Error findings), 1 = Error
findings, 2 = ingestion failure with the clang diagnostics forwarded
verbatim (never a crash).
