# tests/ptxparse — PTX parser corpus

Parse-only corpus for `libs/ptxparse` (wave-2 todo 9). Every fixture's kernel
statistics are asserted EXACTLY by `libs/ptxparse/tests/ptxparse_test.cpp`
against `corpus/manifest.json`.

## Fixture provenance

Hand-written (parse-only, never assembled):

* `minimal_kernel.ptx` — trivial kernel (mov/bar.sync/ret).
* `divergent_branch.ptx` — predicated `@%p1 bra` on a `%tid.x`-derived value
  with an explicit fallthrough label (4 blocks / 4 CFG edges).
* `strided_smem.ptx` — shared-memory stride pattern (stride 68 words).
* `tcgen05_sm100.ptx` — HAND-WRITTEN per PTX ISA 8.6+ syntax. nvcc 12.6 cannot
  target sm_100 (Blackwell requires CUDA >= 12.8), so this fixture is
  parse-only corpus: never assembled, never ptxas-validated
  (FIXTURE GENERATION RULE, plan todo 9). Manifest entry carries
  `"parse_only": true`.
* `truncated_kernel.ptx` — negative fixture: a truncated kernel body used by
  the malformed-input gtest and the `ptxparse-ptxdump-negative` ctest.

nvcc-generated (regenerable with the pinned nvcc 12.6 in `devenv shell`):

```bash
# mma_sm80.ptx — mma.sync.aligned.m16n8k16 (sm_80)
nvcc -arch=sm_80 -ptx -o mma_sm80.ptx mma_sm80.cu

# wgmma_sm90.ptx — wgmma.mma_async.m64n64k16 + commit/wait (sm_90a)
nvcc -arch=sm_90a -ptx -o wgmma_sm90.ptx wgmma_sm90.cu
```

The tiny `.cu` generator sources are not committed (their only meaningful
content is the inline asm, visible verbatim in the emitted PTX); the emitted
fixtures are the parse target and are stable.
