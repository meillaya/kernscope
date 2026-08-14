# tests/amdparse/corpus — AMDGPU ISA text corpus for libs/amdparse

Committed fixtures parsed by the AMDGPU ISA parser (`libs/amdparse`, todo 10)
and asserted by `libs/amdparse/tests/amdparse_test.cpp` (gtest, LABELS no-gpu).

| File | Target | How generated | Purpose |
|---|---|---|---|
| `layernorm_gfx1101.s` | gfx1101 (RDNA3) | `hipcc --offload-arch=gfx1101 -S -O3 --cuda-device-only tests/fixtures/layernorm_fused/layernorm.hip` (ROCm 7.2.3 / clang 22) | real hipcc output: ds_* + global_load_* traffic, s_cbranch CFG |
| `wmma_fixture.s` | gfx1101 (RDNA3) | hand-authored | all six `v_wmma_*_16x16x16_*` variants (brief §5) |
| `mfma_fixture.s` | gfx90a (CDNA2) | hand-authored | `v_mfma_*` shapes incl. `v_mfma_f32_32x32x8f16` (brief §5) |

Hand-authored fixtures use the exact LLVM-MC encoding syntax (cross-checked
against `llvm/test/MC/AMDGPU/gfx11_asm_wmma.s` and `mai-gfx90a.s`).

## Regenerating the hipcc corpus

```bash
cd <repo root>
devenv shell -- hipcc --offload-arch=gfx1101 -S -O3 --cuda-device-only \
    tests/fixtures/layernorm_fused/layernorm.hip \
    -o tests/amdparse/corpus/layernorm_gfx1101.s
```

(hipcc resolves through clr in the devenv; the generated `.s` is the ISA
*text* — the parser never reads SASS/code objects.)

## Parse sanity (llvm-mc equivalence)

Each corpus file must assemble under LLVM 21's own MC (the parser drives the
same engine):

```bash
devenv shell -- llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1101 -filetype=null \
    tests/amdparse/corpus/layernorm_gfx1101.s   # and per-fixture cpu
```

Note: the `.amdgcn_target` directive inside hipcc output validates against
the TargetID computed from the subtarget; the parser (like llvm-mc above)
uses the bare `amdgcn-amd-amdhsa` triple with the CPU passed separately.
