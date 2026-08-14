# Third-Party Attribution

kernscope is an original implementation, but its design was informed by the
following prior art. No code from any of these projects is copied; they are
studied as design references only.

## Studied Prior Art

| Project | License | Role |
|---|---|---|
| [Kerncap](https://github.com/snarky-puppy/kerncap) / intellikit | MIT | Design reference for kernel capture and replay (runtime interception, input snapshotting, reproducer emission). |
| [Mneme](https://github.com/llvm/Mneme) | Apache-2.0 with LLVM exceptions | Design reference for LLVM-level kernel replay/verification infrastructure. |
| [GPGPU-Sim](https://github.com/gpgpu-sim/gpgpu-sim) | BSD-3-style with NVIDIA-derived notice | Design reference for occupancy, register-pressure, and memory-behavior modeling in the static analyzer. |
| [GPUVerify](https://github.com/mcguirewood/gpuverify) | Imperial College (academic) | Unmaintained; design reference only for GPU kernel verification (memory access and barrier safety). |

## Oracle-Only Reference

[`/home/mei/projects/polykernel`](https://github.com/mei/polykernel) is used as
an **oracle-only** design reference: it may be consulted to compare expected
analyzer/verifier behavior, but no code is copied from it into kernscope.
