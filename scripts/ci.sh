#!/usr/bin/env bash
# ci.sh — kernscope CI gate (local-only; no external CI service is wired).
#
# Runs the verification pipeline in order and exits non-zero on the FIRST
# failing stage. Stages whose artifacts have not landed yet (parallel
# foundation todos / later waves) print [SKIP] and stay green; the moment the
# artifact exists the stage becomes a hard gate.
#
# Stage order (wave-0 foundation, todo 4):
#   a. configure + build                     (scripts/configure.sh, todo 3)
#   b. ctest -L "no-gpu|gpu-amd"             (hardware-label suite: always-run
#                                             tests + local AMD RX 7800 XT)
#   c. cmake --build build --target check-kernscope   (lit/FileCheck suite)
#   d. pytest tests/golden -q                (NumPy bf16 golden harness,
#                                             todo 23 / wave 4)
#   e. generated-code compile checks         (hook placeholder — activates at
#                                             todo 24 when the script lands)
#   f. committed-fixture size assertion      (tests/fixtures <= 64MB total)
#   g. NVIDIA-header VENDORED-FILE guard     (filename + content guards)

set -uo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of invocation dir

pass_count=0
skip_count=0

log() { printf '[%s] %s\n' "$1" "$2"; }
pass() { log "PASS" "$1"; pass_count=$((pass_count + 1)); }
skip() { log "SKIP" "$1"; skip_count=$((skip_count + 1)); }
fail() { log "FAIL" "$1"; exit 1; }

echo "== kernscope CI gate =="

# --- (a) configure + build -------------------------------------------------
# CI_SKIP_CONFIGURE=1 is a documented bootstrap escape hatch (owner-set
# explicitly): it skips the configure+build stage while scripts/configure.sh
# is mid-flight (its nix-store LLVM_DIR resolution is being fixed by a
# parallel worker). The stage becomes a hard gate the moment configure.sh
# works; the escape exists only to keep the rest of the gate reachable in
# the meantime.
if [ "${CI_SKIP_CONFIGURE:-0}" = "1" ]; then
    skip "configure+build — CI_SKIP_CONFIGURE=1 (owner bootstrap escape hatch)"
else
    if [ -f scripts/configure.sh ]; then
        bash scripts/configure.sh || fail "configure (scripts/configure.sh)"
        pass "configure (scripts/configure.sh)"
    else
        skip "configure — scripts/configure.sh lands with todo 3 (parallel worker)"
    fi

    if [ -f build/CMakeCache.txt ]; then
        cmake --build build || fail "build (cmake --build build)"
        pass "build (cmake --build build)"
    else
        skip "build — build/ not configured yet (todo 3)"
    fi
fi

# --- (b) ctest hardware-label suite ----------------------------------------
if [ -f build/CTestTestfile.cmake ]; then
    ctest --test-dir build -L "no-gpu|gpu-amd" --output-on-failure \
        || fail 'ctest -L "no-gpu|gpu-amd"'
    pass 'ctest -L "no-gpu|gpu-amd"'
else
    skip 'ctest -L "no-gpu|gpu-amd" — no CTest suite yet (todo 3)'
fi

# --- (b2) ctest test-count guard --------------------------------------------
# Hardening (todo-12 follow-up): the no-gpu|gpu-amd suite count must never
# regress. Todo 10 (f4bcd31) replaced add_subdirectory(libs/roofline) with
# libs/amdparse, silently dropping 14 registered tests while the remaining
# tests still passed — a pure green/red ctest gate cannot see such a
# registration regression, only a count assertion can.
# Mechanism: `ctest -L "no-gpu|gpu-amd" -N` (dry-run) ends with "Total
# Tests: N"; we require N >= the committed floor in scripts/ci-baseline.txt
# (a single integer). The check is non-decreasing (>=), not exact-match, so
# it never goes stale: later waves add tests and simply raise the floor in
# scripts/ci-baseline.txt when they land; until they do, extra tests are
# always green. If the suite ever drops below the floor, CI fails loudly.
if [ -f build/CTestTestfile.cmake ] && [ -f scripts/ci-baseline.txt ]; then
    baseline=$(tr -d '[:space:]' < scripts/ci-baseline.txt)
    count_line=$(cd build && ctest -L "no-gpu|gpu-amd" -N 2>/dev/null | tail -1)
    actual=$(printf '%s' "$count_line" | grep -oE '[0-9]+' | tail -1)
    if [ -z "$actual" ]; then
        fail "ctest count guard — could not parse ctest -N output: '$count_line'"
    fi
    if [ "$actual" -ge "$baseline" ]; then
        pass "ctest count guard: ${actual} tests >= baseline ${baseline} (scripts/ci-baseline.txt)"
    else
        fail "ctest count guard — ${actual} tests < baseline ${baseline}; suite regressed (raise scripts/ci-baseline.txt only when tests are ADDED)"
    fi
else
    skip "ctest count guard — needs build/CTestTestfile.cmake + scripts/ci-baseline.txt"
fi

# --- (c) lit suite (check-kernscope) ---------------------------------------
if [ -f build/CTestTestfile.cmake ]; then
    cmake --build build --target check-kernscope \
        || fail "check-kernscope (lit suite)"
    pass "check-kernscope (lit suite)"
else
    skip "check-kernscope — lit suite lands with todo 3"
fi

# --- (d) golden harness (pytest) -------------------------------------------
if [ -d tests/golden ]; then
    pytest tests/golden -q || fail "pytest tests/golden -q"
    pass "pytest tests/golden -q"
else
    skip "pytest tests/golden — golden harness lands in wave 4 (todo 23)"
fi

# --- (e) generated-code compile checks (hook placeholder) -------------------
# This gate is a placeholder until todo 24: once scripts/compile_checks.sh
# exists (nvcc sm_80/sm_90 compile-only + hipcc gfx1101 compile-and-run of
# generated/emitted kernels), the hook below activates and becomes a hard
# gate. Until then it is a no-op so CI stays green pre-wave-4.
if [ -f scripts/compile_checks.sh ]; then
    bash scripts/compile_checks.sh || fail "generated-code compile checks (scripts/compile_checks.sh)"
    pass "generated-code compile checks (scripts/compile_checks.sh)"
else
    skip "generated-code compile checks — hook placeholder; activates when scripts/compile_checks.sh lands (todo 24)"
fi

# --- (f) committed-fixture size assertion ----------------------------------
if [ -d tests/fixtures ]; then
    size_mb=$(du -sm tests/fixtures 2>/dev/null | cut -f1 || echo 0)
    size_mb=${size_mb:-0}
    if [ "$size_mb" -le 64 ]; then
        pass "fixture size: ${size_mb}MB <= 64MB (tests/fixtures)"
    else
        fail "fixture size: ${size_mb}MB > 64MB (committed fixtures must total <= 64MB)"
    fi
else
    skip "fixture size — tests/fixtures lands in wave 1 (todo 5)"
fi

# --- (g) NVIDIA-header VENDORED-FILE guard ----------------------------------
# Filename guard: no committed VENDORED NVIDIA/CUPTI/NVTX header FILES.
# This targets committed .h FILES only — `#include <cuda_occupancy.h>` style
# NAME references resolved from the nix store remain legal (that include is
# used by the todo-11 oracle tests).
hits=$(git ls-files | grep -E '(^|/)(cuda_occupancy|cupti_[a-z_]*|cuda[a-z_]*|nvtx[a-z_]*)\.h$' || true)
if [ -z "$hits" ]; then
    pass "NVIDIA-header filename guard (no vendored CUDA/CUPTI/NVTX .h files tracked)"
else
    fail "NVIDIA-header filename guard — vendored headers tracked: $(printf '%s ' $hits)"
fi

# Content guard: no tracked file carries a vendor NVIDIA copyright marker
# outside the documented exceptions (docs/research/, THIRD-PARTY.md,
# docs/capture-mapping.md). ':!.omo' excludes the plan/tracking machinery
# (its todo-4 definition quotes this guard's regex) and ':!scripts/ci.sh'
# excludes this guard script itself, whose source necessarily contains the
# pattern literal (this command). Both are self-reference false positives,
# NOT vendored header content; keep both exclusions while the pattern is
# quoted in this file.
hits=$(git grep -I -l "Copyright.*NVIDIA" -- ':!docs/research' ':!THIRD-PARTY.md' ':!docs/capture-mapping.md' ':!.omo' ':!scripts/ci.sh' || true)
if [ -z "$hits" ]; then
    pass "NVIDIA-content guard (no vendor NVIDIA copyright markers outside documented exceptions)"
else
    fail "NVIDIA-content guard — matches: $(printf '%s ' $hits)"
fi

# --- summary ----------------------------------------------------------------
echo
log "PASS" "kernscope CI gate green (${pass_count} pass, ${skip_count} skip, 0 fail)"
exit 0
