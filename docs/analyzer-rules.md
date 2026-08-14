# kernscope analyzer — suggestion rules table

Wave-3 todo 19. The suggestion engine (`tools/analyze/src/analyze_suggest.cpp`)
maps the finding set produced by the six analysis modules (todos 13-18) to
**actionable optimization suggestions**. It is a strict rules engine:

* every suggestion fires ONLY from a triggering finding (warning id, limiter,
  or a metrics condition) — **a suggestion with no triggering finding is
  generic advice and is never emitted**;
* every suggestion carries the triggering finding id in
  `evidence["triggered_by"]`;
* the engine runs LAST and owns `report.suggestions` (informational
  `severity="info"` findings from the instruction-mix module are dropped, so
  the suggestions section carries exactly the trigger-backed suggestions).

## Rule table

| Trigger (finding id + conditions) | Suggestion id | Suggestion text (evidence["text"]) | Evidence fields |
|---|---|---|---|
| warning `bank_conflict`, `evidence["way"] >= 2` | `tile_k_32` | pad shared stride / tile to conflict-free layout (tile K dimension by 32 when K-stride causes the conflict) | `current_tile_k`, `suggested_tile_k`, `shared_bytes_current`, `shared_bytes_suggested` |
| `limiter == "registers"` AND `metrics["fma_rn_f32"] >= 16` (accumulator pattern) | `split_accumulator` | split accumulator fragment / lower unroll / maxrregcount | `accumulator_elements`, `accumulator_bytes`, `suggested_fragment_elements`, `registers_per_thread` |
| warning `divergent_inner_loop` | `hoist_predicate` | move predicate outside loop / restructure to predication | `predicate_depends_on`, `loop_invariant_in`, `current_branch_sites`, `suggested_branch_sites` |
| warning `coalescing`, `evidence["efficiency"] < 1.0`, and NO `bank_conflict` warning | `vectorize_coalescing` | vectorize to 128B contiguous per warp | `efficiency`, `current_bytes_per_warp`, `target_bytes_per_warp`, `vectorized_ld_global` |

Every suggestion also carries `evidence["text"]` (the one-line human summary
the text renderer prints) and `evidence["triggered_by"]` (the triggering
finding id). Evidence values are strings per
[`docs/schema/report.schema.json`](../schema/report.schema.json); the table
below lists the *machine-readable* numeric fields.

## Rule details

### `tile_k_32` — shared-memory bank conflicts

**Trigger**: the `bank_conflict` warning (todo 16, way >= 2) — i.e. the
shared access stride collides ≥ 2 lanes per bank on the 32-bank × 4B model
(research brief §2.1).

**Derivation**:
* `current_tile_k = stride_words - (stride_words % 32)`. The conflict comes
  from a row stride of `(K_tile + pad)` words; the padding is exactly the
  stride's residue mod 32 (fused_attention: 68 words → 68 % 32 = 4 pad →
  the clean K tile is 64). A conflict-free tile must be a multiple of 32
  words, so the residue *is* the padding.
* `suggested_tile_k = 32` (the owner's "tile K dimension by 32" convention).
* `shared_bytes_current = metrics["shared_memory_bytes"]` (the static
  `.shared` size; 0 for an extern-dynamic tile whose size lives only in the
  launch config — the analyzer is a static tool and cannot see it).
* `shared_bytes_suggested = shared_bytes_current * 32 / current_tile_k` (the
  shared footprint scales with the K tile length).

**Evidence**: `current_tile_k`, `suggested_tile_k`, `shared_bytes_current`,
`shared_bytes_suggested`, `triggered_by = "bank_conflict"`.

### `split_accumulator` — register limiter + accumulator pattern

**Trigger**: the occupancy analysis (todo 14) reported
`limiter == "registers"` AND the instruction mix (todo 18) shows an FMA
accumulator loop (`metrics["fma_rn_f32"] >= 16`). A register-resident
accumulator array (often an unrolled `acc[N]`) is what pins occupancy below
the threads/block limit.

**Derivation** (documented heuristic, `kAccumulatorOverheadRegs = 38`):
* `registers_per_thread = metrics["registers_per_thread"]` (ptxas -v when
  available, else the static proxy).
* `accumulator_elements = registers_per_thread - 38`, clamped to ≥ 8. The 38
  is the fixed per-thread working set that is NOT accumulator state: ~8
  base/pointer regs, ~8 loop counter+bound regs, ~8 predicate/temp regs,
  ~14 address/shuffle/scratch regs. (Static approximation: ptxas reuses
  registers, so this is a heuristic, not a cycle-exact claim. It reproduces
  the fused_attention fixture: 102 regs → 64 accumulator floats = `acc[64]`.)
* `accumulator_bytes = accumulator_elements * 4` (f32 elements).
* `suggested_fragment_elements = accumulator_elements / 4`, clamped to ≥ 4
  (split the accumulator into 4 fragments / lower the unroll factor).

**Evidence**: `accumulator_elements`, `accumulator_bytes`,
`suggested_fragment_elements`, `registers_per_thread`,
`triggered_by = "registers_limiter"`.

### `hoist_predicate` — divergent branch inside an inner loop

**Trigger**: the `divergent_inner_loop` warning (todo 17). The divergent
predicate is frequently loop-invariant (derived only from `threadIdx`);
executing the branch every iteration serializes the warp on every trip.

**Derivation**:
* `predicate_depends_on = "lane"` when the warning's
  `predicate_derived_from` is `threadIdx.x` (the warp-divergent component of
  tid), else `"tid"`.
* `loop_invariant_in`: the enclosing loop descriptor from the warning's
  `loop_depth` (the module carries no source loop names; depth is the stable
  handle).
* `current_branch_sites`: predicated branch sites inside the divergent loop
  (a self-contained module re-scan, counting predicated `bra` at loop depth ≥
  the warning's depth).
* `suggested_branch_sites = 1`: hoisting the loop-invariant predicate leaves
  a single branch site before the loop.

**Evidence**: `predicate_depends_on`, `loop_invariant_in`,
`current_branch_sites`, `suggested_branch_sites`,
`triggered_by = "divergent_inner_loop"`.

### `vectorize_coalescing` — uncoalesced global loads (general rule)

**Trigger**: the `coalescing` warning (todo 15) with
`evidence["efficiency"] < 1.0` AND **no** `bank_conflict` warning in the same
report.

**Priority policy**: when a kernel also has shared-memory bank conflicts, the
smem-layout fix (`tile_k_32`) already addresses the same staging data path;
adding a generic vectorize suggestion would be redundant noise. The rule
therefore stays silent for kernels whose memory problem is dominated by the
shared-memory path and fires for kernels whose ONLY memory issue is
scalarized global loads (contiguous-partial, not 128B-coalesced).

**Derivation**: `efficiency` from the warning evidence;
`current_bytes_per_warp` from the warning's `bytes_transferred_per_warp`;
`target_bytes_per_warp = 128` (one contiguous 128B warp span, brief §3.1);
`vectorized_ld_global` from the warning evidence.

**Evidence**: `efficiency`, `current_bytes_per_warp`,
`target_bytes_per_warp`, `vectorized_ld_global`,
`triggered_by = "coalescing"`.

## Evaluation order

Rules evaluate in the owner's suggestion order; each rule returns early when
its trigger is absent, so the report never receives a suggestion without a
trigger:

1. `tile_k_32`
2. `split_accumulator`
3. `hoist_predicate`
4. `vectorize_coalescing`

For the fused_attention golden (occupancy 0.375, limiter registers, 3
warnings, 3 suggestions) the engine emits exactly the owner's three
suggestions — `vectorize_coalescing` stays quiet because the report carries a
`bank_conflict` warning (priority policy above).
