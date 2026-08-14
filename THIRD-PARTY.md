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
