# GPU Kernel Verifier — Research Brief
### "eBPF verifier, but for GPU kernels": prior art, techniques, and IR tradeoffs
Date: 2026-08-13 · Scope: kernel docs, papers (DOI/arXiv), GitHub repos
Method note: every tool claim below is marked **[tool]** (repo/docs evidence — "tool exists and does X") or **[paper]** (publication claim). Items I could not verify are flagged explicitly in §3.6 and §7.

---

## 0. Executive summary

- The Linux eBPF verifier is the best engineering template for a pre-execution kernel verifier: typed registers + range tracking (signed/unsigned min/max + bitwise tnum), path-sensitive symbolic execution with state pruning via subsumption, precision backtracking, and hard complexity caps. Its architecture is documented in `Documentation/bpf/verifier.rst` and implemented across `kernel/bpf/verifier.c` + recently-split files (`states.c`, `liveness.c`, `backtrack.c`, `cfg.c`).
- GPUVerify (Imperial College) is the canonical academic GPU kernel verifier: proves intra/inter-group data-race freedom and barrier-divergence freedom by reducing the kernel to a *sequential* two-thread predicated program checked with Boogie+Z3. Papers: OOPSLA'12, CAV'14, TOPLAS'15. Repo last pushed 2022-07-28; pinned to LLVM 6.0; effectively unmaintained. Its docs confirm the pipeline is source → Clang/LLVM bitcode → **Bugle** → **Boogie** (not "GIL", not CBMC — see §2.3).
- The GPU race-detection literature is rich but mostly dynamic; the strongest *static* lines are PUG (SMT, FSE'10), SESA (SC'14), Faial/FaialAA (memory access protocols, sound + partially complete, 2022–2024), and HGRD (2026, MLIR + host-code constraints + CP-SAT). PTX-level formal verification exists (Princeton ILA model, ICCAD'18). SPIR-V has a mature structural validator (spirv-val).
- For your checks on LLVM IR: ScalarEvolution/ConstantRange/LazyValueInfo + alias analysis for index bounds; NVPTX address-space mapping (0 generic/1 global/3 shared/4 const/5 local/7 shared::cluster) with addrspacecast legality constrained to generic↔non-generic; LLVM's UniformityAnalysis (POPL'21 abstract interpretation) + `convergent` attribute for barrier correctness; may/must-initialized dataflow (prior art: IKOS, Clang SA, UniSan) for uninitialized reads; launch-site dim3 propagation (`__cudaPushCallConfiguration`/`cudaLaunchKernel`) + `__launch_bounds__` (→ `.maxntid`) for bounds.
- IR choice: LLVM IR has the deepest existing analysis infrastructure and is what GPUVerify/LLOV/PhASAR/SVF/IKOS all use; MLIR preserves launch config and loop structure structurally (gpu dialect, sparse dataflow framework incl. IntegerRangeAnalysis) and is where the newest static race detector (HGRD) works; PTX is the stable, hardware-faithful layer (explicit `bar.sync`, state spaces) but loses types. Recommendation in §5.4.

---

## 1. LINUX eBPF VERIFIER — architecture and techniques

All source links below are permalinks at linux master SHA `3aa1dcaa4f6f5ae08936491e08bd456f331f2d40`.

### 1.0 Documentation location (correction)
There is **no `checker.rst`** in `Documentation/bpf/`; the design doc is [`Documentation/bpf/verifier.rst`](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/Documentation/bpf/verifier.rst) (verified by listing the directory). It states the two-step design: (1) DAG check to disallow loops and validate CFG (detecting unreachable instructions), (2) path-sensitive simulation of every instruction observing register/stack state. It also says "See details of eBPF verifier in kernel/bpf/verifier.c".

### 1.1 Two-stage architecture
- **Stage 1 — CFG/DAG check**: `bpf_check_cfg()` ([kernel/bpf/cfg.c#L562](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/cfg.c#L562)), invoked from `bpf_check()` ([verifier.c#L19887](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L19887)). Subprogram boundaries validated by `check_subprogs()` ([verifier.c#L2840](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L2840)).
- **Stage 2 — path-sensitive simulation**: `do_check()` main loop ([verifier.c#L17340](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L17340)), entered via `do_check_common()`/`do_check_main()` ([#L18354](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L18354), [#L18579](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L18579)); top-level entry `bpf_check()` ([#L19771](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L19771)).
- Note for your design: in current master the monolith has been **modularized** — pruning/equivalence lives in `kernel/bpf/states.c`, register liveness in `liveness.c`, precision backtracking in `backtrack.c`, CFG in `cfg.c`, tristate arithmetic in `tnum.c`. This split itself is evidence of how the design decomposes into reusable analyses.

### 1.2 Register type tracking
- Register state is `struct bpf_reg_state` ([include/linux/bpf_verifier.h#L38](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/linux/bpf_verifier.h#L38)) — one struct unifying scalars and pointers; comment: "Ordering of fields matters. See states_equal()".
- Pointer types are an explicit enum `enum bpf_reg_type` ([include/linux/bpf.h#L1033](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/linux/bpf.h#L1033)): `NOT_INIT` (never written → unreadable), `SCALAR_VALUE`, `PTR_TO_CTX`, `CONST_PTR_TO_MAP`, `PTR_TO_MAP_VALUE`, `PTR_TO_MAP_KEY`, `PTR_TO_STACK`, `PTR_TO_PACKET*`, `PTR_TO_SOCKET`, `PTR_TO_BTF_ID`, …
- verifier.rst documents the typing rules: `R2=R1` copies the type; `R2=R1+R1` on two pointers yields `SCALAR_VALUE` ("addition of two valid pointers makes invalid pointer"); in secure mode *any* pointer arithmetic is rejected to prevent kernel-address leaks. Loads/stores allowed only for valid pointer types, "bounds and alignment checked".
- **GPU analogy**: this is exactly the "pointer has an address space + provenance + known object" discipline you need for global/shared/const/local — an addrspace-tagged pointer type lattice where arithmetic rules differ per type.

### 1.3 Value bounds / ranges (the core of bounds checking)
verifier.rst ("Register value tracking") describes the abstract state per register:
- signed min/max and unsigned min/max (`smin/smax/umin/umax`), plus
- **tnum** — tristate bitwise knowledge (mask+value), implemented in [kernel/bpf/tnum.c](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/tnum.c). Doc example: byte load ⇒ `(0x0; 0xff)`; OR 0x40 ⇒ `(0x40; 0xbf)`; +1 ⇒ `(0x0; 0x1ff)` (carries).
- Pointers = base type + **fixed offset** (known constant) + **variable offset** (bounded scalar); alignment is tracked through the tnum (doc's `4n+2 + NET_IP_ALIGN` example).
- **Branch refinement**: a comparison splits ranges per branch (e.g. `>8` ⇒ umin=9 on true side, umax=8 on false side); signed/unsigned info is combined. Implemented in `check_cond_jmp_op()` ([verifier.c#L15945](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L15945)) with `is_branch_taken()` ([#L15449](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L15449)) detecting always/never-taken branches.
- ALU transfer functions: `adjust_ptr_min_max_vals()` ([#L13664](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L13664)) for ptr±scalar, `adjust_scalar_min_max_vals()` ([#L14591](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L14591)) for scalar ops (mul/shift propagate tnum+bounds), dispatched from `check_alu_op()` ([#L14917](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L14917)).
- **Pointer ids**: copies of a packet/map pointer share an `id`; bounds-checking one copy grants a safe range `r=` to all copies with the same id (verifier.rst "Direct packet access": `R3=pkt(id=0,off=0,r=14)` ⇒ bytes [R3, R3+14) are accessible). This id-based range transfer is directly reusable for "prove `base + idx` is inside the buffer once, use it everywhere".

### 1.4 Pointer-arithmetic / address-space rules
- `sanitize_ptr_alu()` ([verifier.c#L13451](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L13451)) enforces which pointer types may participate in ALU at all (e.g. `CONST_PTR_TO_MAP`, `PTR_TO_PACKET_END`: arithmetic forbidden per verifier.rst).
- Stack-pointer arithmetic restricted by `check_stack_access_for_ptr_arithmetic()` ([#L13603](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L13603)).
- Overflow guard (verifier.rst): adding an unknown >16-bit scalar to a packet pointer invalidates later range checks ("r3 += rX may overflow and become less than original skb->data").
- **GPU analogy**: your "illegal pointer arithmetic" check = same structure: per-address-space arithmetic policy (e.g. shared-memory pointers only + bounded offsets; const pointers read-only; no global↔shared arithmetic), plus overflow-aware range invalidation.

### 1.5 Memory access checks (maps/stack/packet/helper args)
- Central dispatcher `check_mem_access()` ([verifier.c#L6078](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L6078)).
- Map bounds: `check_map_access()` ([#L4543](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L4543)) — access must be within `off < reg->off + reg->var_off.value` vs the map value size; verifier.rst shows rejecting `map_value_or_null` without NULL check and misaligned accesses ("misaligned access off 4 size 8").
- Stack: fixed/variable-offset read/write checks `check_stack_write_fixed_off/var_off`, `check_stack_read*` ([#L3452–L4008](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L3452-L4008)), bounds window `[-MAX_BPF_STACK, 0)` with `MAX_BPF_STACK 512` ([include/linux/filter.h#L100](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/linux/filter.h#L100)); `check_stack_access_within_bounds()` ([#L5986](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L5986)). Stack is 128 8-byte slots (`STACK_SLOTS`, [bpf_verifier.h#L253](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/linux/bpf_verifier.h#L253)).
- Helper-argument checking: `check_func_arg()` ([#L8211](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L8211)) + `check_helper_mem_access()` ([#L6741](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L6741)) — each helper declares what memory types/sizes each argument may be; verifier.rst: "The verifier will guarantee that the function is called with valid arguments."

### 1.6 Uninitialized-read handling (eBPF precedent)
- Registers: `NOT_INIT` is unreadable — verifier.rst example rejects `R0=R2` at entry with "R2 !read_ok"; exit requires R0 written ("R0 !read_ok" otherwise).
- Stack: "The verifier will allow eBPF program to read data from stack only after it wrote into it"; indirect reads via helper calls are size-checked ("invalid indirect read from stack off -8+0 size 8").
- Spilled registers tracked per slot (pointer spill/fill). This is a per-slot may-initialized lattice — the minimal version of what you need for GPU local/shared memory.

### 1.7 Precision tracking (managing range blow-up)
- `mark_chain_precision()` ([verifier.c#L3294](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L3294)) + [kernel/bpf/backtrack.c](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/backtrack.c): only registers/stack slots that actually matter for a later check get exact tracking; everything else stays cheaply abstract. This is the key trick that keeps a path-sensitive verifier fast.

### 1.8 State pruning (the anti-path-explosion mechanism)
- verifier.rst "Pruning": for each branch, compare against previously-seen states at the same instruction; if an old state **subsumes** the new one (registers, stack, spilled regs all "safe"), prune. Docs name `regsafe()` and `states_equal()`.
- Implementation today: `regsafe()` ([states.c#L507](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/states.c#L507)), scalar subsumption `range_within()` ([states.c#L302](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/states.c#L302)), id-consistency `check_ids()` ([states.c#L319](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/states.c#L319)), whole-state `states_equal()` ([states.c#L979](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/states.c#L979)), checkpoint lookup `bpf_is_state_visited()` ([states.c#L1238](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/states.c#L1238)).
- **Liveness** makes pruning effective ([kernel/bpf/liveness.c](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/liveness.c)): registers/stack slots never used later are dropped from cached states (verifier.rst "Register liveness tracking" gives the r0/r1 checkpoint example).

### 1.9 Path-explosion limits (hard caps)
- `BPF_COMPLEXITY_LIMIT_INSNS 1000000 /* yes. 1M insns */` — total instructions processed across all paths ([include/linux/bpf.h#L2386](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/linux/bpf.h#L2386)); enforced in the `do_check()` loop with "BPF program is too large. Processed %d insn" ([verifier.c#L17367-L17370](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L17367-L17370)).
- `BPF_COMPLEXITY_LIMIT_JMP_SEQ 8192` (verification-stack depth) and `BPF_COMPLEXITY_LIMIT_STATES 64` (states cached per instruction) ([verifier.c#L196-L197](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L196-L197)); stack-depth checks at [#L1768](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L1768) and [#L2288](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/kernel/bpf/verifier.c#L2288). (Stable across v5.15→master; the classic cBPF program-size cap is `BPF_MAXINSNS 4096`, [include/uapi/linux/bpf_common.h#L49-L54](https://github.com/torvalds/linux/blob/3aa1dcaa4f6f5ae08936491e08bd456f331f2d40/include/uapi/linux/bpf_common.h#L49-L54).)
- **Takeaway**: a GPU verifier needs the same triple — subsumption pruning, liveness-based state minimization, and hard budgets (per-instruction state count, total work) with graceful "inconclusive" fallback.

---

## 2. GPUVerify (Imperial College)

Repo: [mc-imperial/GPUVerify](https://github.com/mc-imperial/GPUVerify) at SHA `49219770aad01231edd0d8e0fa3ed036006cf32a`.

### 2.1 Properties checked
**[tool]** README: "GPUVerify is a static analyser for verifying race- and divergence-freedom of GPU kernels written in OpenCL and CUDA." Verified output includes: "no data races within work groups / no data races between work groups / no barrier divergence / no assertion failures".
**[paper]** OOPSLA'12/TOPLAS'15 define and check:
- **Intra-group data race**: two threads of the same group access the same global-or-shared location, ≥1 write, no intervening barrier.
- **Inter-group data race**: threads of different groups conflict on global memory (added in later versions; TOPLAS'15 evaluates it).
- **Barrier divergence**: threads in a group reach different barriers / different barrier counts (undefined per CUDA: "execution is likely to hang or produce unintended side effects" — quoted in TOPLAS'15 §2.2).
- **Functional correctness**: only via user `__assert` annotations (tool checks "no assertion failures"); GPUVerify has **no contract/pre-postcondition system** — it is a safety-property verifier, not a deductive verifier.
**[tool]** Known unsoundness/limits are documented in [limitations.rst](https://github.com/mc-imperial/GPUVerify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/Documentation/limitations.rst): **no out-of-bounds checking by default**; assumes input arrays are distinct (warns without `__restrict__`); pointer model can cause false negatives when enumerating pointer values; atomics assumed not to overflow. (Note: OOB checking exists only via user assertions — relevant gap your verifier would fill.)

### 2.2 Method (the part worth stealing)
**[paper]** (OOPSLA'12 §3–4; TOPLAS'15; CAV'14; AVoCS'14 tutorial):
- **SDV semantics** ("synchronous, delayed visibility"): group execution is synchronous (for precise divergence checking); writes to shared/global become visible to other threads only at barriers (delayed visibility); all accesses are logged (for race detection). Defined over KPL, a core kernel language.
- **Two-thread reduction**: races and barrier divergence are pairwise properties ⇒ verify an arbitrary pair of threads with other threads' effects abstracted (soundness proof in TOPLAS'15 App. A). "Adversarial abstraction" (reads return arbitrary values) vs "equality abstraction" (shared state modeled concretely).
- **Barrier intervals**: race checking is per barrier-to-barrier interval; access sets reset at barriers.
- **Predicated execution** for control flow: every statement guarded by an "enabled" predicate; a barrier check is just `assert en1 == en2` (this is the divergence check).
- **Race encoding**: per-thread, per-array read/write index sets (LOG_RD/LOG_WR instrumentation); nondeterministic index choice keeps formulas small; CAV'14 adds "watchdog race checking" (nondeterministically choose barrier intervals to check) to cut invariant blow-up.
- **Invariant inference**: candidate invariants with existential Boole constants, discharged by Boogie's **Houdini** algorithm (**[tool]** documented in [advanced_features.rst](https://github.com/mc-imperial/GPUVerify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/Documentation/advanced_features.rst), incl. `access lower bound` candidates).

### 2.3 Pipeline (corrections to the task's premise)
**[tool]** Evidence from the repo:
- Frontend: Clang/LLVM — pinned to **LLVM 6.0**, builds NVPTX target + libclc ([developer_guide.rst](https://github.com/mc-imperial/GPUVerify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/Documentation/developer_guide.rst)).
- **Bugle** is "the component of GPUVerify that translates LLVM bitcode into Boogie" ([basic_usage.rst#L548](https://github.com/mc-imperial/GPUVerify/blob/49219770aad01231edd0d8e0fa3ed036006cf32a/Documentation/basic_usage.rst#L548)); intermediate artifacts are `.bc → .gbpl → .bpl/.cbpl`; backend is **Boogie** (+ Z3), components `GPUVerifyLib / GPUVerifyVCGen / GPUVerifyCruncher / GPUVerifyBoogieDriver` (repo root).
- ⚠️ **No "GIL" intermediate language and no CBMC backend anywhere in the repo/docs.** The task's "CUDA/OpenCL → GIL → Boogie/CBMC" description does not match GPUVerify; the actual chain is **source → Clang/LLVM bitcode → Bugle → Boogie → Z3**. (GIL exists elsewhere in the verification world but is not part of GPUVerify.)
- Thread/block modeling **[paper]**: divergence checked under lockstep assumption (whole group = one subgroup, sound for finer granularity); races checked under fully-async interleaving between barriers (sound for lockstep hardware) — i.e., deliberately worst-case assumptions in both directions.

### 2.4 Publications (citations)
1. **[paper]** A. Betts, N. Chong, A. F. Donaldson, S. Qadeer, P. Thomson. *GPUVerify: a verifier for GPU kernels.* OOPSLA 2012, pp. 113–132. DOI: [10.1145/2384616.2384625](https://doi.org/10.1145/2384616.2384625). (163 kernels; 49/71 evaluation kernels fully automatic.)
2. **[paper]** A. Betts, N. Chong, A. F. Donaldson, J. Ketema, S. Qadeer, P. Thomson, J. Wickerson. *The Design and Implementation of a Verification Technique for GPU Kernels.* ACM TOPLAS 37(3), Article 10, May 2015. DOI: [10.1145/2743017](https://doi.org/10.1145/2743017). (Full two-thread-reduction soundness proof; inter-group races.)
3. **[paper]** P. Deligiannis, A. F. Donaldson, J. Ketema, D. Liew, S. Qadeer. *Engineering a Static Verification Tool for GPU Kernels.* CAV 2014. ([PDF](https://www.doc.ic.ac.uk/~afd/papers/2014/CAV.pdf)) — 2.5 years of engineering lessons: barrier-interval optimization, watchdog race checking, invariant generation pragmatics.
4. **[paper]** A. F. Donaldson. *The GPUVerify Method: a Tutorial Overview.* AVoCS 2014. ([PDF](http://www.doc.ic.ac.uk/~afd/papers/2014/AVoCS.pdf))

### 2.5 Maintenance status
**[tool]** GitHub API: last push **2022-07-28**, not archived, 82 stars. Pinned to LLVM 6.0 (2018). **[paper]** HiRace's 2024 survey table lists GPUVerify as "Not Maintained, static analyzer, limited applicability" (arXiv:2401.04701 Table I), and HGRD (arXiv:2604.02106) reports ~35 false positives (~79% of its reports) on their benchmark because GPUVerify ignores host-code constraints (asserts, parameter relations, launch dims). Verdict: research-grade, unmaintained, but still the methodological reference point.

---

## 3. OTHER GPU VERIFICATION / RACE DETECTION

### 3.1 Static / symbolic verifiers (closest to your project)
| Tool | What it does | Evidence |
|---|---|---|
| **PUG** (Utah) | **[paper]** First comprehensive SMT symbolic verifier for CUDA C: data races, barrier mismatches/deadlocks, bank conflicts, wrong results; barrier-interval decomposition; loop normalization/overapprox/invariants; two-thread reduction. FSE 2010: G. Li, G. Gopalakrishnan. *Scalable SMT-based verification of GPU kernel functions.* DOI: [10.1145/1882291.1882320](https://doi.org/10.1145/1882291.1882320); also PPoPP'10 WOLFGANG workshop slides ([PDF](https://collections.lib.utah.edu/dl_files/48/ff/48ff55caf6b6d5d8716d3838081526c238a94daa.pdf)). **[tool-status]** unmaintained (per HiRace Table I). |
| **SESA** | **[paper]** Symbolic execution + static analysis for CUDA C++; scales to Parboil/Lonestar. SC 2014: P. Li, G. Li, G. Gopalakrishnan. *Practical symbolic race checking of GPU programs.* DOI: [10.1109/SC.2014.20](https://doi.org/10.1109/SC.2014.20). |
| **GKLEE** | **[paper]** Concolic verification + test generation for GPUs (races, deadlocks, bank conflicts, warp divergence). PPoPP 2012, pp. 215–224 (G. Li, P. Li, G. Sawaya, G. Gopalakrishnan, I. Ghosh, S. P. Rajan). Extended by DSGEN (concolic shape generation, [NSF PAR](https://par.nsf.gov/servlets/purl/10267624)). **[tool-status]** unmaintained, CUDA-as-of-2016 (HiRace Table I). |
| **Faial / FaialAA** (Rice, Cogumbreiro et al.) | **[paper]** Static DRF via **Memory Access Protocols** (behavioral types over array index expressions; SMT checking; no loop invariants needed; symbolic loop bounds). FaialAA adds approximation analysis + True-Positive theorem for CIDI kernels (59.5% of 2,770 kernels), interprocedural analysis, benign-race detection. DOI: [10.1145/3689797](https://doi.org/10.1145/3689797); theory: [PLACES'22](https://cogumbreiro.github.io/assets/provable-races-places22.pdf); dataset study via [NSF PAR 10574724](https://par.nsf.gov/biblio/10574724) — finding: of 191 DRF kernels, **98% need a specific thread-configuration assumption and 27% need user assertions** to be analyzable (direct argument for ingesting launch config!). **[tool]** setup repo: [cogumbreiro/setup-faial](https://github.com/cogumbreiro/setup-faial). |
| **HGRD** (2026) | **[paper]** *Towards an Accurate GPU Data Race Detector.* arXiv: [2604.02106](https://arxiv.org/abs/2604.02106). Static; works on **MLIR** of kernel *and host* code; builds conflict constraints over access pairs, uses **dominance/post-dominance of `__syncthreads`/`__syncwarp` in the MLIR CFG** to split barrier intervals, adds host-derived parameter constraints (asserts, relations, allocation sizes), solves with OR-Tools **CP-SAT**; detects inter-block, intra-block and **intra-warp** races and acquire-release synchronization. Claims zero false alarms where GPUVerify/FaialAA raise dozens. Most directly relevant 2026 prior art for your pipeline. |
| **ESBMC-GPU** (Cordeiro et al.) | **[paper]** SMT-based context-bounded model checker for CUDA: data races, **array out-of-bounds**, pointer safety, overflow, div-by-zero, assertions; CUDA Operational Model (COM) models launches/threads; MPOR + two-thread analysis prune interleavings. SAC 2016 ([PDF](https://ssvlab.github.io/lucasccordeiro/papers/sac2016.pdf)); J. SCP 2017 DOI: [10.1016/j.scico.2017.09.005](https://doi.org/10.1016/j.scico.2017.09.005); extended version CCPE ([PDF](https://ssvlab.github.io/lucasccordeiro/papers/cppe2017.pdf)). **[tool]** Built on open-source [ESBMC](https://github.com/esbmc/esbmc). Bounded (loop unwinding/context bounds), not a proof system. |
| **CIVL** | **[paper-claim]** Concurrency Intermediate Verification Language framework; compared as a CUDA verifier in ESBMC-GPU evaluations (assertions, deadlocks, pointer deref, OOB, div-by-zero). See ESBMC-GPU papers above for the claim. |
| **GPU ILA (PTX-level)** (Princeton) | **[paper]** *A Formal Instruction-Level GPU Model for Scalable Verification.* ICCAD 2018: Y. Xing, B.-Y. Huang, A. Gupta, S. Malik. DOI: [10.1145/3240765.3240771](https://doi.org/10.1145/3240765.3240771). Formal model of NVIDIA GPU **at PTX level** (Instruction-Level Abstraction, ILA TODAES DOI [10.1145/3282444](https://doi.org/10.1145/3282444)), incl. all sync primitives; data-race checking via BMC+SMT with two-thread reduction; argues PTX level avoids pointer-analysis pain and API churn. The closest existing thing to a "PTX verifier". |
| **LLOV** | **[paper/tool]** *A Fast Static Data-Race Checker for OpenMP Programs* on **LLVM IR** (polyhedral-style region analysis). ACM TACO 17(4), 2020. DOI: [10.1145/3418597](https://doi.org/10.1145/3418597). Not GPU-specific but demonstrates LLVM-IR-level static race-freedom proofs for SPMD-ish parallel regions. |

### 3.2 Dynamic race detectors (context; not your architecture but define the bar)
**[paper]** GRace (PPoPP'11, DOI [10.1145/1941553.1941574](https://doi.org/10.1145/1941553.1941574)); GMRace (IEEE TPDS 25(1), 2014, DOI [10.1109/TPDS.2013.44](https://doi.org/10.1109/TPDS.2013.44), static pre-filter + dynamic check); LDetector/LD (ACM TACO 2017, [PDF](https://people.cs.rutgers.edu/zz124/assets/pdf/taco17.pdf), value-based, no access monitoring); CURD (PLDI'18, pp. 390–403, DOI [10.1145/3296979.3192368](https://doi.org/10.1145/3296979.3192368)); BARRACUDA (PLDI'17, binary-level, DOI [10.1145/3062341.3062342](https://doi.org/10.1145/3062341.3062342)); ScoRD (ISCA'20, scoped-sync races, DOI [10.1109/ISCA45697.2020.00088](https://doi.org/10.1109/ISCA45697.2020.00088)); iGUARD (SOSP'21, DOI [10.1145/3477132.3483545](https://doi.org/10.1145/3477132.3483545), **[tool]** repo [csl-iisc/iGUARD-SOSP21](https://github.com/csl-iisc/iGUARD-SOSP21), NVBit SASS instrumentation); HiRace (arXiv [2401.04701](https://arxiv.org/abs/2401.04701), source-level FSM-based, checks shared **and global** memory). HiRace's Table I is the best 2024 status map of the field (most tools unmaintained).

### 3.3 NVIDIA Compute Sanitizer (dynamic, vendor — the practical baseline)
**[tool]** [Compute Sanitizer docs](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html): **memcheck** (OOB + misaligned accesses), **racecheck** (shared-memory hazards only — "Currently, this tool only supports detecting accesses to on-chip shared memory"), **initcheck** (uninitialized accesses to **global** memory), **synccheck** (invalid usage of synchronization primitives). No global-memory race detection (HiRace's complaint). Your static verifier's niche: pre-execution coverage of exactly these four classes, incl. global races.

### 3.4 SPIR-V validator (spirv-val) — check categories
**[tool]** [KhronosGroup/SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) at SHA `bb9d0b1b97bd3548128f87d180194e912df4d5df`:
- [README §Validator](https://github.com/KhronosGroup/SPIRV-Tools/blob/bb9d0b1b97bd3548128f87d180194e912df4d5df/README.md#L88-L111): checks the SPIR-V spec's validation rules; **one-sided error** (only errors when an implemented rule is violated); **explicitly incomplete**; checks Universal Limits (spec §2.17), parameterizable for more capable consumers; operates on the binary; library entry `spvValidate`.
- Check categories ≈ one file each in [`source/val/`](https://github.com/KhronosGroup/SPIRV-Tools/tree/bb9d0b1b97bd3548128f87d180194e912df4d5df/source/val) (53 files): structural/CFG (`validate_cfg.cpp`, `validate_function.cpp`, `validate_instruction.cpp`), IDs/types (`validate_id.cpp`, `validate_type.cpp`, `validate_constants.cpp`), memory (`validate_memory.cpp`, `validate_memory_semantics.cpp`, `validate_scopes.cpp`), **barriers** (`validate_barriers.cpp`), atomics (`validate_atomics.cpp`), execution-model limits (`validate_execution_limitations.cpp`, `validate_non_uniform.cpp`, `validate_group.cpp`), plus capability/decoration/layout/interface checks and client-API-specific rules (Vulkan/OpenCL envs).
- **Lesson**: spirv-val is *structural/semantic well-formedness*, not memory-safety or race analysis — it's the "instruction-set legality" layer of your pipeline (analogous to eBPF's DAG/CFG stage), not the bounds/race layer.

### 3.5 PTX-level verification status
**[tool]** NVIDIA's `ptxas` validates PTX at compile time (proprietary; error reporting only), and the driver API exposes `nvPTXCompiler`. No open-source general PTX verifier exists beyond:
- **[paper]** Princeton's ILA PTX model + race checker (ICCAD'18, §3.1) — formal semantics + BMC, the only published PTX-level verification work found.
- **[tool]** Compute Sanitizer's PTX/SASS-level dynamic checks (§3.3); NVBit for dynamic binary instrumentation.

### 3.6 ⚠️ Names from the task I could NOT verify
- **"GPU-PRAMChecker"**: multiple targeted searches (incl. "PRAMChecker" + GPU race checker) returned **no publication or repo**. Possibly a misremembering; the closest real tools are GRace/GMRace (PRAM-ish bulk-synchronous reasoning) and GKLEE. Treat as unverified.
- **"GKRace"**: no publication found; only an unrelated GitHub username. Possibly confusion with **GRace** (PPoPP'11), **GMRace** (TPDS'14), or **GKLEE** (PPoPP'12). Treat as unverified.
- **"GIL" in GPUVerify**: not present (§2.3).

---

## 4. TECHNIQUES PER TARGET CHECK ON LLVM IR

LLVM permalinks below are at llvm-project SHA `f7f96294e59d289fc0a07a25cacab7d3451ae785`.

### 4.1 Out-of-bounds global / shared memory access (index bounds)
- **ScalarEvolution (SCEV)**: models loop induction as add-recurrences `{start,+,step}` with wrap flags (NSW/NUW) — see [ScalarEvolution.h#L81-L98](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/include/llvm/Analysis/ScalarEvolution.h#L81-L98). For `A[blockIdx.x*blockDim.x + threadIdx.x]` inside a loop, SCEV gives you the index as an affine AddRec; combine with known launch dims (§4.6) to prove `0 ≤ idx < N` or emit a check. Implementation: [llvm/lib/Analysis/ScalarEvolution.cpp](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/lib/Analysis/ScalarEvolution.cpp).
- **ConstantRange / LazyValueInfo / ValueTracking**: LLVM's generic range machinery (`ConstantRange`, [LazyValueInfo.cpp](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/lib/Analysis/LazyValueInfo.cpp), [ValueTracking.cpp](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/lib/Analysis/ValueTracking.cpp) for implied-condition propagation) is the eBPF-style min/max refinement, already path-sensitive via branch conditions.
- **Alias analysis** to know *which* object a pointer indexes: [BasicAliasAnalysis.cpp](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/lib/Analysis/BasicAliasAnalysis.cpp) (+ MemorySSA for per-access mod/ref). GPUVerify's restrict/aliasing assumption (§2.1) shows aliasing is a real precision cliff — BasicAA + `!noalias`/restrict metadata is your mitigation.
- **Abstract interpretation route**: IKOS (**[tool]** [NASA-SW-VnV/ikos](https://github.com/NASA-SW-VnV/ikos): LLVM-based, translates to an AR abstract domain, proves absence of runtime errors incl. **buffer overflow** — README demonstrates proving/rejecting array bounds) is the existence proof that AI-style bounds checking works on LLVM IR; its checks are CPU-oriented but reusable. For GPU-specific index algebra, Faial's MAPs (symbolic index protocols) and HGRD's CP-SAT constraints are the published designs.
- **Practical recipe**: for each load/store in addrspace 1/3, compute `base + offset` range (SCEV+LVI), intersect with object size (from allocation site / kernel parameter metadata / launch dims), emit: proven-safe / needs-runtime-check / rejected. This is precisely eBPF's `check_map_access()` pattern with SCEV instead of tnum.

### 4.2 CUDA address spaces in LLVM IR + addrspacecast validity
**[tool]** NVPTX address-space mapping ([NVPTXUsage.md#L109-L126](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/NVPTXUsage.md#L109-L126)):

| AS | Memory space | AS | Memory space |
|----|--------------|----|--------------|
| 0 | Generic | 4 | Constant |
| 1 | Global | 5 | Local |
| 2 | Internal use | 7 | Shared::cluster |
| 3 | Shared | | |

(matches the task's 0/1/3/4/5 list; 7 = Hopper cluster shared). Key rules from the same doc:
- "since address space 0 is the generic space, it is illegal to have global variables in address space 0" ([#L138-L140](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/NVPTXUsage.md#L138-L140)).
- Conversions: "Intrinsics are provided which can be used to convert pointers between the **generic and non-generic** address spaces" ([#L123-L126](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/NVPTXUsage.md#L123-L126), [#L3955-L3960](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/NVPTXUsage.md#L3955-L3960)) — i.e., direct shared↔global casts must route through generic; a verifier should reject/flag `addrspacecast` between two non-generic spaces on NVPTX.
- `addrspacecast` IR semantics ([LangRef.md#L13207-L13245](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/LangRef.md#L13207-L13245)): "can be a no-op cast or a complex value modification, **depending on the target and the address space pair**"; must be side-effect-free and non-capturing; same-space conversions must be `bitcast`. So legality is target policy — your verifier's "invalid address-space cast" check = per-target legality table (generic↔{global,shared,const,local} OK; non-generic↔non-generic not representable on NVPTX) + provenance check that a cast result is only dereferenced when the pointer actually points into that space (NVPTX provides `llvm.nvvm.ptr.gen.to.*` / `*.*.to.gen` intrinsics; generic-address validity is runtime-only unless you track provenance — exactly an eBPF-style pointer-type lattice).
- MLIR equivalent **[tool]**: `gpu.address_space` = {global, workgroup, private, constant} as memref memorySpace ([GPU.md#L30-L45](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/docs/Dialects/GPU.md#L30-L45)).

### 4.3 Barrier correctness / divergent `__syncthreads`
**Semantics evidence:**
- **[tool]** CUDA C++ Programming Guide (2026 structure), Synchronization Primitives: "`__syncthreads*()` wait until **all non-exited threads** in the thread block **simultaneously reach the same** `__syncthreads*()` program or exit" and provide memory ordering ("strongly happens before any participating thread is unblocked") — [cpp-language-extensions §5.4.4.1](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html). Hence a barrier inside a divergent branch: non-participating threads never arrive ⇒ waiting threads block forever / UB.
- **[tool]** PTX ISA: `bar.cta.sync` = "arrive, wait for all threads in CTA to also arrive" (or a specified thread count `bar.cta.sync n, CNT`); barrier instructions carry `.cta`/`.cluster` scope — [PTX ISA, Parallel Synchronization](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html).
- **[tool]** Independent Thread Scheduling (Volta+), CUDA 11.8 guide §K.6.2: pre-Volta warps had **one shared program counter + active mask**; Volta "maintains execution state per thread, including a program counter and call stack, and can yield execution at a per-thread granularity" ([archive link](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-c-programming-guide/index.html#independent-thread-scheduling)). **Nuance for your verifier**: ITS makes intra-warp progress independent (locks/mutexes within a warp become viable), but `__syncthreads` still requires *all non-exited block threads* to arrive — ITS does **not** legalize a barrier reachable by only some threads; it only changes what can happen *between* barriers (no more relying on lockstep for correctness). GPUVerify's conservative model (lockstep for divergence, async for races) remains sound under ITS.
**Analysis machinery on LLVM IR:**
- **[tool]** LLVM's `convergent` function/call attribute ([LangRef.md#L2182-L2186](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/LangRef.md#L2182-L2186)) marks ops whose result depends on the participating thread set — `__syncthreads`/barriers are convergent; the optimizer is forbidden to sink/hoist/merge them across divergent control.
- **[tool]** [ConvergenceAndUniformity.rst](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/ConvergenceAndUniformity.rst): LLVM's static convergence/uniformity analysis, extending **[paper]** J. Rosemann, S. Moll, S. Hack. *An Abstract Interpretation for SPMD Divergence on Reducible Control Flow Graphs.* PACMPL (POPL) 5, Article 31, 2021. DOI: [10.1145/3434312](https://doi.org/10.1145/3434312); implemented in [UniformityAnalysis.cpp](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/lib/Analysis/UniformityAnalysis.cpp). This gives you, per barrier call site, whether the controlling conditions are uniform ⇒ **divergent-barrier detection as: barrier is reachable under a non-uniform condition ⇒ reject** (GPUVerify's `assert en1 == en2` is the two-thread version of the same predicate).
- Also: LLVM `convergencectrl` operand bundles ([LangRef.md#L3405](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/llvm/docs/LangRef.md#L3405)) for controlled-convergence ops (convergent loops) — relevant for `__syncwarp`-style patterns.

### 4.4 Race-prone shared-memory writes (static)
Published designs to port:
- GPUVerify's read/write-set encoding per barrier interval + two-thread reduction (§2.2) — **[paper]** soundness proven in TOPLAS'15.
- HGRD's CFG criterion **[paper]**: an access pair can only race if no `__syncthreads` **dominates the later and post-dominates the earlier** access (arXiv:2604.02106 §III) — cheap dominance prefilter on LLVM IR/MLIR before expensive index reasoning.
- Faial MAPs **[paper]**: behavioral types over index expressions, DRF via SMT; partial completeness for CIDI kernels (DOI 10.1145/3689797).
- For LLVM IR plumbing you'd build on: MemorySSA (per-access def/use), alias results, and the convergence analysis above (races only matter between concurrently-executing threads; uniform writes to the same address by all threads are benign — a check GPUVerify/HGRD both exploit).

### 4.5 Uninitialized memory reads (static)
- **Lattice**: per-allocation-site may/must-initialized dataflow (kill on store, gen on alloc-with-init, meet at joins; eBPF's NOT_INIT stack-slot tracking is the degenerate version, §1.6). On LLVM IR, run it over MemorySSA def-use chains; treat loads from `addrspace(3)` (shared, no implicit init) and `addrspace(5)` (local) as the high-risk classes; global memory is zero-init only if the runtime guarantees it (CUDA device globals are zero-initialized at program start — but kernel-to-kernel state persists).
- **Prior art**:
  - **[tool]** IKOS checks runtime errors incl. uninitialized reads on LLVM IR via abstract interpretation ([ikos README](https://github.com/NASA-SW-VnV/ikos)).
  - **[paper]** *Detecting Uninitialized Variables in C++ with the Clang Static Analyzer* (Acta Cybernetica, [PDF](https://cyber.bibl.u-szeged.hu/index.php/actcybern/article/view/4100/4014)) — shipped as `optin.cplusplus.UninitializedObject` since Clang 9; symbolic-execution based with heuristics for intentional non-init.
  - **[paper]** *Static Detection of Uninitialized Stack Variables in Binary Code* (RUB, 2019, [PDF](https://informatik.rub.de/veroeffentlichungenbkp/syssec/veroeffentlichungen/2019/pdfs/2019_Static_Detection_of_Uninitialized_Stack_Variables_in_Binary_Code.pdf)) — flow/context/field-sensitive "safe zones" + under-constrained symbolic execution; the safe-zone idea ports to IR.
  - LLVM-based hardeners UniSan (kernel info leaks) and SafeInit (init-at-allocation) are cited in that paper's related work as LLVM dataflow approaches.
  - **[tool]** Dynamic ground truth: MSan (CGO 2015, bit-precise shadow + origin tracking; [clang docs](https://clang.llvm.org/docs/MemorySanitizer.html)) and Compute Sanitizer **initcheck** (global-memory uninit, §3.3) — useful as oracles for your static analysis.

### 4.6 Mapping kernel launch bounds into bounds checks
- **Where the config lives**: Clang lowers `kernel<<<grid,block>>>(...)` to `__cudaPushCallConfiguration(grid, block, smem, stream)` + `cudaLaunchKernel(...)` (**[tool]** [SemaCUDA.cpp#L1187](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/clang/lib/Sema/SemaCUDA.cpp#L1187) picks the runtime fn; lowering verified by [clang/test/CodeGenCUDA/kernel-call.cu](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/clang/test/CodeGenCUDA/kernel-call.cu)). So at the launch site in LLVM IR, grid/block dims are call arguments — **interprocedurally propagate them** (SCCP/constant ranges) to the kernel's `llvm.nvvm.read.ptx.sreg.*` intrinsics, giving `0 ≤ threadIdx.x < blockDim.x`, `0 ≤ blockIdx.x < gridDim.x` as axioms for §4.1's index proofs. This is exactly HGRD's "host code constrains kernel parameters" insight **[paper]** (arXiv:2604.02106) and ESBMC-GPU's COM launch modeling **[paper]**.
- **Static bounds from attributes**: `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` compiles to `.maxntid`/`.minnctapersm` (**[tool]** [CUDA guide, launch bounds](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)); in LLVM IR this is the `nvvm.maxntid` kernel attribute (NVPTXUsage.md "Running the Kernel"/kernel attributes) — a sound *upper* bound on blockDim usable even without the launch site.
- **Why it matters**: Faial's study found 98% of DRF kernels need the thread-configuration assumption to verify **[paper/NSF PAR]**; GPUVerify's false positives on `reduce`/`pitchLinear` came from ignoring grid dims and allocation sizes **[paper]**. Design implication: verifier API should accept (or resolve) launch config per kernel, with three tiers: (a) compile-time constants, (b) symbolic dims with host-side constraints (HGRD-style), (c) fully symbolic ⇒ emit runtime checks instead of rejecting.

---

## 5. IR CHOICE: LLVM IR vs MLIR vs PTX

### 5.1 What exists at each level
| Dimension | LLVM IR | MLIR (gpu/affine/scf dialects) | PTX text |
|---|---|---|---|
| Types | Full typed SSA (first-class types, !dbg) | Full types + memref layout maps | Mostly erased (b32/b64 regs; typed ld/st variants) |
| Address spaces | NVPTX AS 0/1/3/4/5/7 on every pointer type | `gpu.address_space` memref attribute (global/workgroup/private/constant) | Explicit state spaces on every mem op (`.global/.shared/.const/.local`) |
| Loop structure | Present pre-optimization; fragile after -O2 (unswitch/unroll/vectorize) | `affine`/`scf` preserve structured loops + bounds syntactically | Gone (branches/labels; loops implicit) |
| Launch config | Only at host-side call site (`__cudaPushCallConfiguration`) + `nvvm.maxntid` attr | `gpu.launch`/`gpu.launch_func` carry grid/block operands structurally ([GPU.md](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/docs/Dialects/GPU.md#L186-L192)) | Grid/block are runtime launch args; `.maxntid` directive only |
| Barriers | `llvm.nvvm.barrier0` / convergent calls | `gpu.barrier` op with scope attr (**[tool]** [GPUOps.td#L1427](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/include/mlir/Dialect/GPU/IR/GPUOps.td#L1427)) | Explicit `bar.sync/barrier` with CTA/cluster scope + thread counts |
| Analysis infra | SCEV, BasicAA, MemorySSA, LazyValueInfo, ValueTracking, UniformityAnalysis, DominatorTree — decades mature | Sparse/dense dataflow framework (**[tool]** [SparseAnalysis.h](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/include/mlir/Analysis/DataFlow/SparseAnalysis.h#L34): lattices + `SparseForwardDataFlowAnalysis`; ready-made analyses incl. [`IntegerRangeAnalysis`](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/include/mlir/Analysis/DataFlow/IntegerRangeAnalysis.h#L68), divisibility, liveness, dead-code) — newer, less battle-tested | None open-source (must build; ICCAD'18 ILA is the only formal base) |
| Frontends | clang CUDA/HIP (`--cuda-device-only -emit-llvm`), nvcc via llvm-lto; SPIR-V translators | Polygeist/CIR paths emerging; fewer mature CUDA frontends | nvcc/ptxas output; stable target across GPU languages |
| Who verifies there | GPUVerify (bitcode→Boogie), LLOV, IKOS, PhASAR (**[tool]** [TACAS'19](https://doi.org/10.1007/978-3-030-17465-1_22), LLVM-based IFDS/IDE), SVF (**[tool]** [CC'16](https://yuleisui.github.io/publications/cc16.pdf), value-flow/pointer infra) | HGRD (2026 race detection) | Princeton ILA race checker (ICCAD'18); ptxas syntax validation |

### 5.2 LLVM IR — strengths/risks
- **Strengths**: best-in-class existing analyses (§4.1, §4.3); address spaces are first-class; convergent/uniformity machinery is *exactly* your barrier check; proven frontend for CUDA/HIP; every serious LLVM-based verifier (GPUVerify, IKOS, PhASAR, SVF, LLOV) chose it. **[tool]** LLVM's design as an analysis substrate: **[paper]** C. Lattner, V. Adve. *LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation.* CGO 2004. DOI: [10.1109/CGO.2004.1281665](https://doi.org/10.1109/CGO.2004.1281665).
- **Risks**: (a) verify **early** — at `-O0`/`-O1` before loop destruction, or accept SCEV-reconstructable loops; GPUVerify compiles with its own pipeline for this reason; (b) CUDA semantics survive only as intrinsics/attributes (fine: `llvm.nvvm.*` are stable); (c) host+device split requires linking launch sites to kernels yourself (§4.6).

### 5.3 MLIR — strengths/risks
- **Strengths**: launch config and barriers are *structural* (gpu.launch/gpu.barrier ops), address spaces as attributes, structured control flow survives, and the dataflow framework gives you range/liveness lattices to assemble checks declaratively; HGRD (2026) validates this choice end-to-end for race detection. **[paper]** MLIR itself: C. Lattner et al. *MLIR: A Compiler Infrastructure for the End of the Specialization Era.* arXiv: [1903.10267](https://arxiv.org/abs/1903.10267).
- **Risks**: you re-host or lower from LLVM IR (extra pipeline step); fewer off-the-shelf safety analyses; the gpu dialect is explicitly "more likely to change than others" (**[tool]** [GPU.md](https://github.com/llvm/llvm-project/blob/f7f96294e59d289fc0a07a25cacab7d3451ae785/mlir/docs/Dialects/GPU.md) header note); CUDA frontend story is less mature than clang→LLVM IR.

### 5.4 PTX — strengths/risks + recommendation
- **Strengths**: what actually executes (no frontend-lowering bugs escape); explicit state spaces and `bar.sync`; stable spec; ICCAD'18 argues PTX-level modeling avoids pointer-analysis pain and API churn **[paper]**.
- **Risks**: types/source info gone (poor diagnostics); no open analysis infra; must model the full ISA; ptxas may still transform further (SASS) — verification gap remains.
- **Recommendation (evidence-weighted)**: primary verifier on **LLVM IR at low optimization level** (max existing infrastructure; precedent: GPUVerify/IKOS/PhASAR/LLOV), with (a) an eBPF-style typed-register + range + pruning engine (§1) for the checks that need path sensitivity, (b) host-side launch-config propagation (§4.6), and (c) **PTX (or SPIR-V, with spirv-val as the well-formedness gate) as a secondary audit layer** for what survived lowering. Use MLIR if your input is already MLIR (e.g., DSL/ML-framework kernels) or if you want HGRD-style structural barrier reasoning — its dataflow framework is the cheapest way to get range/liveness analyses if you're not reusing LLVM's.

---

## 6. REFERENCES

### Papers (DOI/arXiv)
1. Betts, Chong, Donaldson, Qadeer, Thomson. *GPUVerify: a verifier for GPU kernels.* OOPSLA 2012. DOI: 10.1145/2384616.2384625
2. Betts, Chong, Donaldson, Ketema, Qadeer, Thomson, Wickerson. *The Design and Implementation of a Verification Technique for GPU Kernels.* TOPLAS 37(3), 2015. DOI: 10.1145/2743017
3. Deligiannis, Donaldson, Ketema, Liew, Qadeer. *Engineering a Static Verification Tool for GPU Kernels.* CAV 2014. doc.ic.ac.uk/~afd/papers/2014/CAV.pdf
4. Donaldson. *The GPUVerify Method: a Tutorial Overview.* AVoCS 2014. doc.ic.ac.uk/~afd/papers/2014/AVoCS.pdf
5. Li, Gopalakrishnan. *Scalable SMT-based verification of GPU kernel functions (PUG).* FSE 2010. DOI: 10.1145/1882291.1882320
6. Li, Li, Gopalakrishnan. *Practical symbolic race checking of GPU programs (SESA).* SC 2014. DOI: 10.1109/SC.2014.20
7. Li, Li, Sawaya, Gopalakrishnan, Ghosh, Rajan. *GKLEE: Concolic Verification and Test Generation for GPUs.* PPoPP 2012, 215–224
8. Zheng, Ravi, Qin, Agrawal. *GRace: A Low-overhead Mechanism for Detecting Data Races in GPU Programs.* PPoPP 2011. DOI: 10.1145/1941553.1941574
9. Zheng, Ravi, Qin, Agrawal. *GMRace.* IEEE TPDS 25(1), 2014. DOI: 10.1109/TPDS.2013.44
10. Li, Ding, Hu, Soyata. *LDetector (LD).* ACM TACO 2017. people.cs.rutgers.edu/zz124/assets/pdf/taco17.pdf
11. Peng, Grover, Devietti. *CURD: a dynamic CUDA race detector.* PLDI 2018, 390–403. DOI: 10.1145/3296979.3192368
12. Eizenberg, Peng, Pigli, Mansky, Devietti. *BARRACUDA.* PLDI 2017. DOI: 10.1145/3062341.3062342
13. (ScoRD authors). *ScoRD: hardware race detection for scoped sync.* ISCA 2020. DOI: 10.1109/ISCA45697.2020.00088
14. Kamath, Basu. *iGUARD: In-GPU Advanced Race Detection.* SOSP 2021. DOI: 10.1145/3477132.3483545
15. Jacobson, Burtscher, Gopalakrishnan. *HiRace.* arXiv:2401.04701 (2024)
16. Liew, Barbosa, Cogumbreiro et al. *Sound and Partially-Complete Static Analysis of Data-Races in GPU Programs (FaialAA).* ACM 2024. DOI: 10.1145/3689797; PLACES'22: cogumbreiro.github.io/assets/provable-races-places22.pdf
17. (HGRD authors). *Towards an Accurate GPU Data Race Detector.* arXiv:2604.02106 (2026)
18. Pereira, Albuquerque, da Silva, Marques, Monteiro, Ferreira, Cordeiro. *SMT-Based Context-Bounded Model Checking for CUDA Programs (ESBMC-GPU).* Concurrency & Computation: Practice & Experience (2017); JSCP 2017. DOI: 10.1016/j.scico.2017.09.005; SAC 2016: ssvlab.github.io/lucasccordeiro/papers/sac2016.pdf
19. Xing, Huang, Gupta, Malik. *A Formal Instruction-Level GPU Model for Scalable Verification.* ICCAD 2018. DOI: 10.1145/3240765.3240771; ILA: DOI: 10.1145/3282444
20. Rosemann, Moll, Hack. *An Abstract Interpretation for SPMD Divergence on Reducible Control Flow Graphs.* POPL 2021 (PACMPL 5). DOI: 10.1145/3434312
21. Schubert, Hermann, Bodden. *PhASAR: An Inter-procedural Static Analysis Framework for C/C++.* TACAS 2019. DOI: 10.1007/978-3-030-17465-1_22
22. Sui, Xue. *SVF: Interprocedural Static Value-Flow Analysis in LLVM.* CC 2016. yuleisui.github.io/publications/cc16.pdf
23. (LLOV authors). *LLOV: A Fast Static Data-Race Checker for OpenMP Programs.* ACM TACO 17(4), 2020. DOI: 10.1145/3418597
24. Lattner, Adve. *LLVM: A Compilation Framework for Lifelong Program Analysis & Transformation.* CGO 2004. DOI: 10.1109/CGO.2004.1281665
25. Lattner et al. *MLIR: A Compiler Infrastructure for the End of the Specialization Era.* arXiv:1903.10267
26. Serebryany, Stepanov. *MemorySanitizer: fast detector of uninitialized memory use in C++.* CGO 2015. clang.llvm.org/docs/MemorySanitizer.html
27. (Szeged authors). *Detecting Uninitialized Variables in C++ with the Clang Static Analyzer.* Acta Cybernetica. cyber.bibl.u-szeged.hu/index.php/actcybern/article/view/4100/4014
28. (RUB authors). *Static Detection of Uninitialized Stack Variables in Binary Code.* 2019. informatik.rub.de (SysSec)

### Repos (with SHA permalinks used above)
- torvalds/linux @ 3aa1dcaa4f6f (verifier.c, states.c, cfg.c, liveness.c, backtrack.c, tnum.c, bpf_verifier.h, bpf.h, filter.h, Documentation/bpf/verifier.rst)
- mc-imperial/GPUVerify @ 49219770aa (README, Documentation/*)
- KhronosGroup/SPIRV-Tools @ bb9d0b1b97 (README, source/val/)
- llvm/llvm-project @ f7f96294e59d (NVPTXUsage.md, LangRef.md, ConvergenceAndUniformity.rst, ScalarEvolution.h, UniformityAnalysis.cpp, mlir GPU dialect + DataFlow headers, clang SemaCUDA.cpp)
- NASA-SW-VnV/ikos; SVF-tools/SVF; secure-software-engineering/phasar; cogumbreiro/setup-faial; csl-iisc/iGUARD-SOSP21; esbmc/esbmc

### Official docs
- Linux: Documentation/bpf/verifier.rst (no checker.rst exists)
- CUDA C++ Programming Guide (2026 multi-page): cpp-language-extensions (syncthreads semantics, __launch_bounds__), writing-cuda-kernels; CUDA 11.8 archive: Independent Thread Scheduling (K.6.2)
- PTX ISA: bar.sync/barrier, state spaces
- Compute Sanitizer: memcheck/racecheck/initcheck/synccheck
- SPIR-V Specification (Universal Limits §2.17)

---

## 7. VERIFICATION NOTES (tool-exists vs paper-claims; corrections)
1. **checker.rst does not exist** — the eBPF design doc is `Documentation/bpf/verifier.rst` (verified via repo listing).
2. **GPUVerify pipeline**: source → Clang/LLVM bitcode → Bugle → Boogie (+Z3). **"GIL" and "CBMC" are not part of GPUVerify** (verified: no mentions in repo/docs). Functional-correctness contracts are not supported; only assertions.
3. **GPU-PRAMChecker / GKRace**: could not be verified as real tools/papers after targeted searches; flagged unverified (§3.6).
4. eBPF complexity limits cited are current-master values (1M processed insns; 8192 stack; 64 states/instruction) — stable since at least v5.15 (verified both tags).
5. HiRace's maintenance table and HGRD's false-positive numbers are **paper claims** about other tools, not independently re-verified here.
6. ESBMC-GPU's CCPE DOI appears truncated in the PDF ("10.1002/cpe"); the JSCP DOI (10.1016/j.scico.2017.09.005) is verified.
7. All GitHub line-number citations were taken from clones/raw fetches at the stated SHAs on 2026-08-13; master drift will move them — permalinks pin the exact commit.
