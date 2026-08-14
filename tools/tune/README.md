# tools/tune — the C2 autotuner (todos 25-28)

Directory owned jointly by the parallel tune-track workers:

| Section | Todo | Owner |
|---|---|---|
| FlashAttention-lite template generator | 27 (this commit) | `kernscope::tune::flash_attention_gen` |
| matmul template generator | 25 | parallel worker (appends below) |
| elementwise/reduction template generator | 26 | parallel worker (appends below) |
| tuning driver (GPU-free pre-screen → compile → golden gate → bench → select) | 28 | later wave |

Shared-file convention: every target is prefixed with its owning generator
(`kernscope_flash_gen` / `kernscope_matmul_gen` / ...) so appends to the
shared `CMakeLists.txt` can never collide.

## FlashAttention-lite template generator (todo 27)

`kernscope_flash_gen` (header `include/kernscope/tune/flash_attention_gen.h`,
impl `src/flash_attention_gen.cpp`) emits single-source portable HIP kernels
that implement **forward** FlashAttention with the **online softmax** running
max / rescale / accumulate algorithm (Dao et al., "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness", NeurIPS 2022) — O tiles
accumulate correctly across the K/V tiles in ONE pass over Q, with no
materialized N×N attention matrix and no two-pass rescale.

### Lite scope (documented, review-enforced)

- Forward pass only. **No** backward pass, **no** dropout, **no** paged KV,
  **no** tensor-core instructions (WMMA/MFMA are out of scope — the tile
  matmuls are scalar fp32 so the template is correct on both vendors).
- No FlashAttention-2 parity or IO-aware-kernel-asymptotics claim beyond the
  shared-memory tiling + online-softmax structure.
- Causal masking IS supported (a `causal` config flag, baked into the variant)
  with the same semantics as the golden reference
  (`refs.attention(..., causal=True)` in tests/golden): row i attends only to
  columns j ≤ i; masked scores are −inf before the softmax.
- The generated kernel applies the same 1/√d scaling as the golden attention
  reference (cosine/PCC are scale-invariant but max_rel_err is not, so the
  scaling must match exactly — todo-23 note).

### Search space (validated pre-compile, `validate_config`)

| Dimension | Choices |
|---|---|
| Br (tile_m, query rows/block) | 32, 64 |
| Bc (tile_n, KV rows/tile) | 32, 64 |
| stages (software-pipelined K/V buffers) | 1, 2 |
| vec_width (columns per thread, threads_x = d/vec) | 1, 2, 4 |
| smem layout (Q/K/V tile rows) | padding (d+1) \| XOR swizzle (col ^ (row & 3)) |
| causal | on / off |

Validity rules (all rejected **pre-compile** with a reason string): search-space
membership; block divisibility (nq % Br == 0, nk % Bc == 0 — the lite kernel
is unguarded by design); vector-width tiling (d % vec == 0, Bc % (d/vec) == 0);
XOR swizzle requires d % 4 == 0; and the shared-memory budget
`(Br·dPitch + 2·stages·Bc·dPitch + Br·(Bc+1) + 2·Br) · 4` bytes must fit the
arch's per-block smem (`ArchInfo::smem_per_block_max_bytes` via libs/arch).
A config whose tiles exceed the budget is REJECTED before any source is
generated — the failure mode is a reason string, never a broken compile.

### Tests

- `flash_gen_test.cpp` (no-gpu, `ctest -R tune_flash_gen`): nvcc sm_80/sm_90
  compile-only gates, hipcc gfx1101 `--genco` compile-only gate, the
  over-budget rejection (reason asserted), the roofline AI cross-check
  (FLOPs == 4N²d, equal to `libs/roofline::compute_flops(kFlashAttention,
  {n,d})`), and full search-space validity over the run shape.
- `flash_gen_test.cpp` (gpu-amd, `ctest -R tune_flash_gen_live`): the
  non-causal AND causal variants run on gfx1101 through the
  `kernscope_templates_hip_runner` + `kernscope_golden` attention reference
  and must earn verdict pass:true (cosine ≥ 0.999 ∧ max_rel_err ≤ 1e-2 ∧
  pcc ≥ 0.99) on N=64, d=32.

Golden extension (sanctioned): `refs.attention` gained a `causal` parameter
and the golden CLI gained `--causal` (tests/golden); `run_hip_gate.py` gained
the `attention` family + `--causal` flag (kernels/templates). Both are
backward-compatible additions.
