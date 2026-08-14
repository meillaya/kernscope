# kernscope-verify checks/oob — global OOB + shared-memory bounds (todo 31)
# ========================================================================
#
# Check classes 1 (global out-of-bounds) and 2 (shared-memory bounds) of the
# eight-class verifier, implemented as ONE CheckPlugin (id "global-oob",
# check_class_ids {1, 2}) in `memory_bounds_check.cpp`, registered via
# REGISTER_CHECK (no shared-file edits — the todo-30 plugin contract).
#
# What the checks prove
# ---------------------
# For every load/store in a kernel, the plugin:
#   1. resolves the allocation root (device global / __shared__ array /
#      extern __shared__ / kernel pointer parameter),
#   2. computes the accessed byte span from the GEP index expressions using
#      VerifierContext::queryRange — the core's range lattice first (exact
#      threadIdx/blockIdx ranges propagated from the launch config, tier (a)),
#      ScalarEvolution getUnsignedRange second (research brief §4.1),
#   3. compares the span's exclusive upper bound (last accessed byte + 1)
#      against the allocation size:
#        - __device__ globals / __shared__ arrays: static size from the
#          global's type,
#        - extern __shared__ (dynamic smem): size from the CONSTANT
#          launch-site smem argument (cudaConfigureCall arg 2 /
#          __cudaPushCallConfiguration arg 2 / cudaLaunchKernel arg 4),
#   4. emits per-access findings + one per-kernel verdict diagnostic whose
#      message embeds a machine-parseable `verdict_json: {...}` payload
#      (this module's verdict vocabulary — the core has no verdict enum for
#      check classes):
#        SAFE         — every access proven within its (known-size)
#                       allocation, or index-bounded against a runtime
#                       allocation (see scope below),
#        MAY_OOB      — at least one access possibly out of bounds, or with
#                       an unprovable index,
#        SYMBOLIC_DIMS — launch config unknown (e.g. --device-only): no
#                       bound provable — degraded verdict, NEVER SAFE.
#
# Soundness scope (may-analysis — the check does NOT claim completeness)
# ----------------------------------------------------------------------
#   * Data-dependent indices (not boundable by the lattice or SCEV) are
#     reported MAY_OOB.  No completeness claim: a data-dependent index may
#     or may not be out of bounds at runtime; the check refuses to say SAFE
#     (soundness over false safety).
#   * Kernel pointer parameters have a RUNTIME allocation size.  The check
#     proves the INDEX is bounded (the class-1 contract: index expressions
#     bounded via SCEV + propagated launch dims) and emits an info note; a
#     static allocation-size proof would require host-side cudaMalloc size
#     tracking, which is NOT in scope.
#   * Dynamic smem with a RUNTIME smem launch argument: the bound is
#     unknowable statically -> MAY_OOB (no in-bounds claim).
#   * addrspace(0) generic pointers, addrspace(4) __constant__ globals and
#     local memory (addrspace(5) allocas) are not analyzed (v1 scope).
#   * Negative GEP indices wrap into a huge unsigned span and are
#     conservatively flagged MAY_OOB.
#
# Corpus + manifest (committed fixtures)
# --------------------------------------
#   tests/verify/corpus/oob_clean.cu       class 1 SAFE pair
#   tests/verify/corpus/oob_may.cu         class 1 MAY_OOB pair
#   tests/verify/corpus/smem_clean.cu      class 2 SAFE
#   tests/verify/corpus/smem_overflow.cu   class 2 MAY_OOB pair
#   tests/verify/corpus/smem_dynamic.cu    class 2 dynamic smem pair
#   tests/verify/corpus/manifest.oob.json  expected verdicts (fragment; the
#                                          consolidated manifest.json is
#                                          todo 35's file)
#
# Acceptance: `ctest -R verify_bounds` (verify-bounds-test asserts the
# manifest entries against the emitted verdicts + the device-only
# degradation).
