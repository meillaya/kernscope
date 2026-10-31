# Third-Party Attribution

kernscope is an original implementation, but its design was informed by the
following prior art. No code from any of these projects is copied; they are
studied as design references only.

## Studied Prior Art

| Project | License | Role |
|---|---|---|
| [Kerncap](https://github.com/AMDResearch/intellikit) / intellikit | MIT | Design reference for kernel capture and replay (runtime interception, input snapshotting, reproducer emission). |
| [Mneme](https://github.com/Olympus-HPC/Mneme) | Apache-2.0 with LLVM exceptions | Design reference for LLVM-level kernel replay/verification infrastructure. |
| [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim) | BSD-3-style with NVIDIA-derived notice | Design reference for occupancy, register-pressure, and memory-behavior modeling in the static analyzer. |
| [GPUVerify](https://github.com/mc-imperial/GPUVerify) | Imperial College (academic) | Unmaintained; design reference only for GPU kernel verification (memory access and barrier safety). |

## Oracle-Only Reference

The local sibling project at `/home/mei/projects/polykernel` (no public
remote) is used as an **oracle-only** design reference: it may be consulted to
compare expected analyzer/verifier behavior, but no code is copied from it
into kernscope.

## capture-hip (todo 45) additions

| Project | License | Role |
|---|---|---|
| [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) samples (intercept_table / code_object_tracing, ROCm 7.2.3) | MIT | API-usage reference for `rocprofiler_configure` + `rocprofiler_at_intercept_table_registration` (HipDispatchTable wrapping) and the callback-tracing service registration pattern. Pattern studied; implementation is original. |
| Kerncap / intellikit (again) | MIT | The HSA-layer intercept-table pattern (wrap runtime dispatch-table entries for alloc hooks + launches) is the direct design precedent for libs/capture-hip §1. Pattern studied only; no code copied. |

`hsa_amd_queue_intercept_create` (per-AQL-packet interception) was considered
and DOCUMENTED as the alternative kernel-dispatch mechanism (see
docs/report/kernscope-technical-report.pdf); libs/capture-hip uses the
rocprofiler-sdk kernel-dispatch callback-tracing service instead, which
exposes the same grid/workgroup data with less plumbing. roctracer is NOT
used (deprecated).

## Report Tooling

| Project | License | Role |
|---|---|---|
| [tikz-flora](https://github.com/wn-mitch/tikz-flora) | MIT | Watercolour margin flora for LuaLaTeX. Vendored, unmodified, in `docs/report/`; see `docs/report/tikzflora-LICENSE`. |
