# kernscope-verify soundness scope — race-prone shared-memory writes (todo 33)

Check class 4 (id `race`, plugin `tools/verify/checks/race/race_check.cpp`)
is a **MAY analysis**: it reports race *candidates* it cannot rule out; it
does **NOT** prove data-race freedom. This document states exactly what is
in scope, what is reported, and the known false-positive and false-negative
classes. It is the todo-33 owner's contract for the check's behavior.

## Verdicts

| Verdict | Meaning |
|---|---|
| `SAFE` | No same-phase race candidate: for every pair of shared-memory accesses (≥1 write) with a thread-dependent index, either a barrier separates them (different barrier phases), the paths are mutually exclusive under a block-uniform branch, or ScalarEvolution proves the accessed byte spans disjoint. |
| `MAY_RACE` | At least one same-phase pair (≥1 write, ≥1 thread-dependent index) whose byte spans overlap or cannot be bounded. Warning with both source sites + the missing-barrier path. |
| `UNKNOWN_RACE` | At least one same-phase pair whose index is **data-dependent** (derived from a memory load, an unknown call, or a kernel argument): the analysis cannot determine which addresses different threads touch. Soundness naming — NOT `SAFE` (a race cannot be excluded) and NOT `MAY_RACE` (addresses are unknowable, not merely overlapping). |

`MAY_RACE` dominates `UNKNOWN_RACE` in the per-kernel verdict; `SAFE` only
when neither exists. The verdicts are embedded in the diagnostic
`verdict_json:` payloads (module-owned vocabulary; the core has no verdict
enum for check classes).

## In-scope analysis (what is proven)

Intra-kernel only (per `(plugin, kernel)` invocation), on shared memory
(addrspace 3, including clang 21's `addrspacecast`-to-generic lowering) —
loads/stores only; atomics (`atomicrmw`, atomic load/store) are excluded.

1. **Barrier-delimited phase partitioning** (GPUVerify's barrier-interval
   reasoning, TOPLAS'15 §2.1/§3; SDV semantics, OOPSLA'12 §3). Barrier
   calls (`llvm.nvvm.barrier0*`, `llvm.nvvm.barrier.cta.*`,
   `llvm.amdgcn.s.barrier`) split each block into segments; connected
   components of the segment graph over CFG edges are the *barrier phases*.
   Accesses in different components have a barrier on every path between
   them ⇒ proven race-free. Same component ⇒ MAY-be-simultaneous
   (fully-async interleaving within a phase — sound under lockstep and
   Volta independent-thread-scheduling hardware, research brief §2.3).
2. **Thread separation** (plan scope: thread-id-dependent indices). An
   index expression whose SSA use chain reaches a thread-id intrinsic
   (`tid.x/y/z`, `laneid`, `workitem.id.x/y/z`) is `thread-dependent`.
   Pairs where both indices are uniform are broadcasts and are skipped.
   Pairs on mutually exclusive paths of a **block-uniform** conditional
   branch (condition free of thread-id leaves; region test via DT/PDT,
   mirroring the barrier module's divergent-region logic) are proven
   non-simultaneous and skipped.
3. **Address separation via ScalarEvolution** (research brief §4.1). Per
   GEP level, `VerifierContext::queryRange` (launch-dim-aware range
   lattice first, SCEV second) yields the unsigned index range; the access
   byte span is the linearized offset range (variable indices contribute
   `[min, max+1) * stride`; constant indices — e.g. the leading `i64 0` —
   contribute fixed offsets). Disjoint spans prove the addresses can never
   coincide ⇒ `SAFE`.

## False-positive classes (reported `MAY_RACE` though possibly benign)

The check deliberately over-approximates; these are documented, not bugs:

- **Identical thread-indexed expressions**: `s[t] = a; s[t] = b;` or
  `s[t] = v; v2 = s[t];` — per-thread sequential accesses that may be
  thread-local at runtime, but the range-based may-alias test cannot model
  thread variables, so same-phase same-span pairs are reported (the
  plan-mandated corpus contract: same-index write-write ⇒ `MAY_RACE`).
- **Disjoint index arithmetic SCEV cannot prove**: e.g. `s[t]` vs
  `s[255-t]` have overlapping SCEV spans `[0, 1024)` though the runtime
  instances are per-thread mirrored — reported (genuinely racy in this
  example; the point is SCEV ranges are element-set over-approximations).
- **Unbounded thread indices**: an index whose range the lattice/SCEV
  cannot bound is treated as overlapping everything (conservative).
- **Loop-carried phases**: a barrier *inside* a loop does not distinguish
  iterations — a write before the loop's barrier in iteration *k* and the
  same site in iteration *k+1* fall in the same phase component and are
  paired (across-iteration accesses are in fact time-separated by the
  barrier). Cross-iteration drift of threads (Volta ITS) is not modeled.
- **Unknown ranges with uniform partners**: a thread-dependent access with
  unbounded span paired with a uniform-address access is reported.

## False-negative classes (missed races)

- **Data-dependent indices** are reported `UNKNOWN_RACE`, never `SAFE` —
  but `UNKNOWN_RACE` is a naming, not a race proof: a data-dependent index
  may or may not race (e.g. a deliberately coordinated global-flag pattern).
- **Uniform-address value races**: `s[0] = tid;` — the address is uniform
  (broadcast), only the *value* varies per thread; value-based analysis is
  out of v1 scope (the plan scopes class 4 to thread-id-dependent indices).
- **Lone data-dependent stores**: a single store `s[j] = ...` with
  data-dependent `j` and no same-phase partner access can self-race (two
  threads loading the same `j`), but is not reported (would flag every
  gather/scatter pattern; documented trade-off).
- **Divergent barriers** (class-6 UB) are still treated as phase
  separators; races around a barrier executed by only some threads may be
  missed (the code is undefined anyway; class 6 reports it).
- **`__syncwarp` / `__threadfence_block()` / atomics** are not modeled as
  synchronization: warp-scoped or fence-based coordination may make a
  reported candidate benign at runtime (over-report, not under) — and
  atomics, which are intentional concurrent synchronization, are excluded
  from the access set entirely.
- **Inter-block races**: shared memory is per-block, so only intra-block
  races are in scope (global-memory inter-block races are not class 4).

## Future upgrade path (NOT implemented)

- **HGRD** (2026, MLIR + host-code constraints + CP-SAT; research brief
  §3.1) solves precise conflict constraints over access pairs — dominance
  of `__syncthreads`/`__syncwarp` in the CFG to split barrier intervals,
  host-derived parameter constraints, CP-SAT solving, zero-false-alarm
  claims. Its constraint model is the natural precision upgrade for the
  SCEV may-alias step (replacing span overlap with satisfiable index
  constraints) and for host-launch-parameter-aware phase reasoning.
- **Faial / FaialAA** memory-access protocols (behavioral types over array
  index expressions, SMT-checked; research brief §3.1) are the reference
  design for a sound+partially-complete DRF mode with symbolic loop bounds
  — the candidate follow-on if a proof mode (vs may mode) is wanted.
- GPUVerify's **two-thread reduction with equality/adversarial abstraction**
  (TOPLAS'15 App. A) is the reference for turning the pairwise index test
  into a sound DRF proof over distinct thread variables.
