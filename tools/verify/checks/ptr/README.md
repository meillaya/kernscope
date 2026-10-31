# kernscope-verify checks/ptr — pointer discipline (todo 34)
# ==========================================================
#
# Check classes 5 (illegal pointer arithmetic), 7 (uninitialized reads) and
# 8 (invalid address-space casts) of the eight-class verifier, implemented
# as ONE CheckPlugin (id "ptr", check_class_ids {5, 7, 8}) in
# `ptr_check.cpp` (+ shared helpers in `ptr_common.{h,cpp}`), registered via
# REGISTER_CHECK (no shared-file edits — the todo-30 plugin contract).
#
# Class 5 — PTR_OOB (illegal pointer arithmetic, Warning)
# -------------------------------------------------------
# Every GEP in the kernel (including ConstantExpr GEPs clang 21 folds into
# operands — `g_buf + 20000` becomes an inline GEP expression inside the
# store) is resolved to its allocation root and its byte span is computed
# from the index expressions via VerifierContext::queryRange (the core's
# range lattice first, ScalarEvolution second — research brief §4.1):
#   * a pointer VALUE whose max offset exceeds the statically sized
#     allocation is PTR_OOB (the eBPF sanitize_ptr_alu typed-pointer
#     discipline adapted: pointer arithmetic must stay within bounds);
#   * an UNBOUNDED (data-dependent) index is PTR_OOB — may-analysis: no
#     in-bounds claim (soundness over false safety);
#   * subtraction of pointers rooted in DIFFERENT known allocations
#     (ptrtoint pairs under `sub`) is PTR_OOB (unrelated-pointer
#     subtraction, UB); same-allocation subtraction stays SAFE;
#   * kernel-parameter allocations (runtime size) get a PARAM_BOUNDED
#     note — index bounded, no static escape claim;
#   * one-past-end pointers are legal C, but constant-index GEPs whose
#     element width is unrecoverable are conservatively flagged (max
#     offset computed as span.upper - 1). Documented over-approximation.
#
# Class 8 — INVALID_ADDRSPACE_CAST (Error, citing the verified table)
# -------------------------------------------------------------------
# Legality follows the VERIFIED address-space tables of
# docs/report/kernscope-technical-report.pdf (NVPTX: generic(0) <->
# {global(1), shared(3), constant(4), local(5)} only) and §2.2 (AMDGPU:
# generic(0) <-> {global(1), local(3), constant(4), private(5)},
# flat-mediated). A DIRECT addrspacecast between two non-generic spaces
# (e.g. shared(3) -> global(1)) has no hardware instruction and yields
# poison per LLVM LangRef §addrspacecast — an Error citing the table.
# Generic-mediated chains (shared -> generic -> global) emit two LEGAL
# casts and are never flagged (the QA negative). Note the fixture
# technique: well-formed CUDA cannot express a direct non-generic pair, so
# the corpus uses `__attribute__((address_space(1)))` casts, which clang 21
# lowers to the direct ConstantExpr addrspacecast; the legal chain needs an
# empty `asm volatile` barrier between the casts, otherwise instcombine
# folds 3->0->1 into the direct 3->1 form (cast-pair elimination).
#
# Class 7 — MAY_UNINIT_READ (uninitialized reads, Warning, with path)
# -------------------------------------------------------------------
# Must/may-initialized lattice dataflow over the kernel body
# (GPUVerify-style must-init, per-allocation granularity):
#   * initial states: kernel params (and pointees) + __device__/__constant__
#     globals INITIALIZED (host-side setup before launch); allocas and
#     __shared__ globals UNINITIALIZED;
#   * per object, two bits flow: must_written (every reaching path wrote
#     it) and may_written (some path wrote it); joins: must = AND, may =
#     OR; transfer is instruction-order sensitive within a block;
#   * a load is flagged iff must_written is false at that point, with an
#     evidence path;
#   * __syncthreads() is a PHASE BOUNDARY: at a barrier every smem object
#     written by ANY path before it (may_written) becomes readable by ALL
#     threads after it (must_written := may_written, addrspace(3) only).
#
# KERNEL-BOUNDARY ASSUMPTION (documented in code + this file): the analysis
# NEVER reasons across kernel launches — a kernel's shared memory is
# assumed uninitialized at entry regardless of prior kernels (CUDA gives no
# such guarantee without host-side cudaMemset tracking, out of scope).
#
# Soundness scope (may-analysis — no completeness claim)
# ------------------------------------------------------
#   * Per-allocation granularity: writing ANY element marks the whole
#     object written (span precision is future work); cross-thread index
#     patterns within one phase are the race module's (class 4) domain.
#   * Only the kernel body is scanned; reads/arith through unresolvable
#     bases (inttoptr, unknown calls) are not claimed on.
#   * The -O1 DCE gotcha (checks/oob/README.md): smem/local round-trips in
#     fixtures need `volatile` to survive ingestion.
#
# Corpus + manifest (committed fixtures)
# --------------------------------------
#   tests/verify/corpus/ptr_clean.cu        class 5/7/8 SAFE pair
#   tests/verify/corpus/ptr_oob.cu          class 5 PTR_OOB (escape, range,
#                                           unrelated sub)
#   tests/verify/corpus/ptr_escape_smem.cu  class 5 PTR_OOB (smem) + SAFE
#   tests/verify/corpus/ptr_param_safe.cu   class 5/7 SAFE (param pointees)
#   tests/verify/corpus/ptr_cast_bad.cu     class 8 INVALID_ADDRSPACE_CAST
#                                           + SAFE legal chain (QA negative)
#   tests/verify/corpus/ptr_uninit.cu       class 7 MAY_UNINIT_READ (smem +
#                                           alloca) + SAFE phase-clean
#   tests/verify/corpus/manifest.ptr.json   expected verdicts (fragment;
#                                           the consolidated manifest.json
#                                           is todo 35's file)
#
# Acceptance: `ctest -R verify_ptr` (verify-ptr-test asserts the manifest
# entries against the emitted verdicts + the QA negatives: the legal
# generic-mediated cast stays unflagged; the phase-clean smem read-after-
# barrier stays SAFE).
