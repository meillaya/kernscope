# tools/tune — the C2 autotuner (todos 25-28)

Directory owned jointly by the parallel tune-track workers:

| Section | Todo | Owner |
|---|---|---|
| FlashAttention-lite template generator | 27 (this commit) | `kernscope::tune::flash_attention_gen` |
| matmul template generator | 25 (this commit) | `kernscope::tune::matmul_gen` |
| elementwise/reduction template generator | 26 | parallel worker (appends below) |
| tuning driver (GPU-free pre-screen → compile → golden gate → bench → select) | 28 | later wave |

Shared-file convention: every target is prefixed with its owning generator
(`kernscope_flash_gen` / `kernscope_matmul_gen` / ...) so appends to the
shared `CMakeLists.txt` can never collide.

## matmul template generator (todo 25)

`kernscope_matmul_gen` (header `include/kernscope/tune/matmul_gen.h`, impl
`src/matmul_gen.cpp`, CLI `src/matmul_gen_cli.cpp`) emits single-source
portable HIP kernels for shared-memory tiled GEMM **C[M,N] = A[M,K] @ B[K,N]**
(bf16 in, fp32 accumulate, bf16 out — todo-23 rounding contract) over the
FULL search space the tuner (todo 28) searches. Generation-only: a config is
baked into the emitted source as constexpr constants + the pipeline /
smem-layout / vector-group / unroll structure; no per-config source is ever
hand-written.

### Search space (validated pre-compile, `validate_config`)

| Dimension | Choices |
|---|---|
| BM, BN (output tile) | 64, 128 |
| BK (inner tile) | 8, 16, 32, 64 |
| warps (per block) | 1, 2, 4, 8 — split into (warps_m, warps_n) = (threads_y, threads_x/32) with threads_x = 32·warps_n covering BN, threads_y = warps_m covering BM |
| stages (pipeline) | 1, 2, 3, 4 — multi-buffered smem tiles; synchronous loads in the lite template (the buffer alternation is the searched structure, mirroring todo 27) |
| vec_width (cols/thread) | 1, 2, 4 — kVec consecutive B columns grouped into bv[kVec] per kk |
| smem layout | padding (BK+1) \| XOR swizzle (col ^ (row & 3), simplified CuTe Swizzle<3,3,3>) |
| unroll (K-loop) | 1, 2, 4 — `#pragma unroll kUnroll` |

Validity rules (all rejected **pre-compile** with a reason string): search-space
membership; warp/block divisibility (the (warps_m, warps_n) factorization must
tile the output tile: BM % warps_m == 0 and BN % (32·warps_n) == 0); vector
width (cpt = BN/(32·warps_n) % vec_width == 0); threads/block cap (32·warps
<= 1024 and the arch max); register budget (BM·BN/(32·warps) fragment regs +
32 overhead vs the arch's `max_regs_per_thread`); and the smem budget
`4 · stages · (BM·(BK+pad) + BK·(BN+pad))` bytes vs the arch's
`smem_per_block_max_bytes` via libs/arch. A config whose tiles exceed a budget
is REJECTED before any source is generated — the failure mode is a reason
string, never a broken compile.

The emitted kernel is bounds-checked (OOB M/N/K loads zero-fill, OOB stores
are skipped), so the problem shape carries no divisibility constraint; M/N/K
are runtime kernel arguments. Entry symbol: `kernscope_tune_matmul` (the
runner resolves by this name; matmul family contract).

### Tests

- `matmul_gen_test.cpp` (no-gpu, `ctest -R tune_matmul_gen`): nvcc sm_80/sm_90
  compile-only gates, hipcc gfx1101 `--genco` compile-only gate, the bounded
  grid enumeration (exactly 16 configs), the invalid-config rejection counts
  (register budget, smem over-budget with the budget + arch named, vector-width
  misdivision) and a full 4608-combo search-space enumeration sanity check.
- `matmul_gen_test.cpp` (gpu-amd, `ctest -R tune_matmul_gen_live`): the live
  config (BM=BN=64, BK=16, warps=2, stages=1, vec=1, padding) runs on gfx1101
  through the `kernscope_templates_hip_runner` + `kernscope_golden` matmul
  reference and must earn verdict pass:true (cosine ≥ 0.999 ∧ max_rel_err ≤
  1e-2 ∧ pcc ≥ 0.99) on M=N=K=64.
- CLI (`kernscope-tune-gen`): `--arch --bm --bn --bk --warps --stages --vec
  --layout padded|xor --unroll --out <file>`; exits non-zero with the validity
  reason on rejected configs.

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
