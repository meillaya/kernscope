# Profiler (C6) — docs/profile.md

Status: todo 52 deliverable (kernscope-profile CLI + layernorm_fused golden
e2e). Author: profile track. Pairs with `docs/research/owner-examples.md`
(Section B — the profiler example SHAPE this CLI's text output mirrors) and
`docs/research/profiler-research-brief.md` (the methodology).

## 1. Scope

`kernscope-profile` is the Wave-9 closer: ONE command runs the whole
todo-49/50/51 stack — trace the target under the kernel tracer, enrich every
kernel record, rank the hot kernels, diagnose "why is this slow?" — and
emits one report through libs/report.

```
trace (49) -> enrich (50) -> rank + diagnose (51) -> report (libs/report)
```

| stage | what happens | who |
|---|---|---|
| trace | the target runs under LD_PRELOAD of the rocprofiler-sdk tracer tool (`libkernscope_profile_hip.so`); every kernel dispatch + HIP API call emits one JSONL line | tools/profile/src/amd_tracer.cpp (todo 49) |
| enrich | each kernel record carries occupancy + limiter (computed by libs/occupancy, todo 11), the code-object vgpr/sgpr/LDS facts and honest degradation markers | todo 50 |
| rank | hot-kernel ranking: total/median per kernel name, % of wall time | rank_hot_kernels (todo 51) |
| diagnose | SOL-style rules engine: occupancy limiter hints, roofline memory-traffic classification for known families, the insufficient-evidence degradation; every hint carries evidence | diagnose_kernel (todo 51) |
| report | `--json` = the profile report model; `--text` = the owner's Section B shape (Kernel/Time/Registers/Shared/Limiter/Hint) with LOCALLY-MEASURED values; `--html` = libs/report's self-contained page (embedded JSON payload) | libs/report reuse (todo 7) |

## 2. Usage

```
kernscope-profile <binary|dir> [--vendor auto|hip|cuda] [--json|--text|--html]
                 [--] [app args...]
```

The target is a binary path OR a directory containing the binary (the
committed layernorm_fused fixture build dir resolves `layernorm_app`
inside). `--` separates app arguments from CLI arguments.

For the layernorm_fused fixture the CLI seeds the app's input in its scratch
dir (tests/fixtures/README.md contract: M=1024 x N=128, bf16) — the app's
default-args contract is `input.bin`/`output.bin` in the CWD.

## 3. Exit codes

| code | meaning |
|---|---|
| 0 | report produced — INCLUDING "no kernels traced": a target that runs fine but traces zero kernels is a VALID report (the QA negative), never an error |
| 2 | error: missing/unrunnable target, tracer not built, CUDA runtime unavailable locally, render schema violation |
| 3 | usage error |

## 4. Vendors

| vendor | local behavior |
|---|---|
| hip | LIVE: LD_PRELOAD of the rocprofiler-sdk tool; the golden e2e profiles the todo-5 layernorm_fused fixture on gfx1101 |
| cuda | compile+unit only locally; real-NVIDIA tracing is OWNER-GATED via `bash scripts/rental-validate.sh` (never auto-triggered) |
| auto | detects hip/cuda from `ldd` on the target; defaults to hip (this host is AMD; the tracer is inert for non-GPU binaries — exactly the no-kernels negative path) |

## 5. Report shape

The `--text` output mirrors the owner's profiler example
(docs/research/owner-examples.md Section B) per hot kernel:

```
Kernel: _Z15layernorm_fusedPKtS0_S0_Ptfi.kd
Time: 9.48 us
Registers/thread: 24
Shared memory/block: 1 KB
Likely limiter: memory
Compiler hint:
  likely memory/traffic-bound (arithmetic intensity below the ridge point; ...)
```

All values are measured locally (or computed by libs/occupancy / libs/roofline
from the local records) — no vendor numbers are ever claimed.

The `--json` report model carries the enriched per-kernel trace table
(`kernels[]`), the hot `ranking[]`, the owner-style `owner_example` showcase
and the `no_kernels_traced` flag; numeric fields are machine-readable. The
golden asserts RANGES only — never absolute durations
(tests/fixtures/expected_layernorm_profile.json + check_profile.py).

The `--html` output is libs/report's self-contained page for the showcase
kernel (the first kernel whose name contains "layernorm", falling back to the
hottest); the embedded JSON payload parses (tests/report/check_html.py
contract).

## 6. Testing

| test | label | what it asserts |
|---|---|---|
| `profile_cli:*` (5 gtest cases) | no-gpu | JSONL round-trip, no-kernels negative, numeric report JSON, owner text shape, embedded-JSON HTML |
| `profile_cli_amd` | gpu-amd | the e2e GOLDEN: `kernscope-profile build/tests/fixtures/layernorm_fused --json \| python3 tests/fixtures/check_profile.py` exit 0 + --text shape + --html parse |
| `profile_cli_no_kernels` | gpu-amd | the QA negative: kernel-free binary -> "no kernels traced" + exit 0 |
| `profile-cli-version` | no-gpu | CLI version banner |
