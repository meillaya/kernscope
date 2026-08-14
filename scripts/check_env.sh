#!/usr/bin/env bash
# kernscope toolchain environment gate.
# Prints one [PASS]/[FAIL] line per toolchain assertion; exits non-zero if any
# check fails. Run INSIDE `devenv shell` (or with the env materialized).
#
# NOTE: output is captured via command substitution instead of `cmd | grep -q`
# pipes: with `set -o pipefail`, grep -q exits on the first match, SIGPIPEs the
# producer (e.g. rocminfo exits 141), and the pipeline reports FAIL even on a
# genuine match. Capturing first makes the gate deterministic.
set -uo pipefail

fail=0
pass() { printf '[PASS] %s\n' "$*"; }
failout() { printf '[FAIL] %s\n' "$*"; fail=1; }

# --- 1. nvcc is CUDA 12.6 ---
nvcc_out=$(nvcc --version 2>&1)
if printf '%s\n' "$nvcc_out" | grep -q "12\.6"; then
  pass "nvcc is 12.6 ($(printf '%s\n' "$nvcc_out" | grep release | sed 's/^ *//'))"
else
  failout "nvcc is NOT 12.6: $(printf '%s\n' "$nvcc_out" | head -2 | tr '\n' ' ')"
fi

# --- 2. hipcc works ---
if hipcc --version >/dev/null 2>&1; then
  pass "hipcc works ($(hipcc --version 2>&1 | head -1))"
else
  failout "hipcc failed: $(hipcc --version 2>&1 | head -3 | tr '\n' ' ')"
fi

# --- 3. rocminfo lists gfx1101 (local RX 7800 XT) ---
rocminfo_out=$(rocminfo 2>&1)
if printf '%s\n' "$rocminfo_out" | grep -qi "gfx1101"; then
  pass "rocminfo enumerates gfx1101"
else
  failout "rocminfo does NOT list gfx1101 (see polykernel docs/hip_backend.md: HSA_OVERRIDE_GFX_VERSION=11.0.1 fallback only; check /dev/kfd + video/render groups + amdgpu driver). rocminfo output tail: $(printf '%s\n' "$rocminfo_out" | tail -3 | tr '\n' ' ')"
fi

# --- 4. clang is LLVM 21 ---
clang_out=$(clang --version 2>&1)
if printf '%s\n' "$clang_out" | head -1 | grep -q "21\."; then
  pass "clang is LLVM 21 ($(printf '%s\n' "$clang_out" | head -1))"
else
  failout "clang is NOT 21.x: $(printf '%s\n' "$clang_out" | head -1)"
fi

# --- 5. CUDA_HOME is a SINGLE path (no colon) with headers ---
if [ -z "${CUDA_HOME:-}" ]; then
  failout "CUDA_HOME is unset"
elif case "$CUDA_HOME" in *:*) true;; *) false;; esac; then
  failout "CUDA_HOME contains a colon (must be a single root): $CUDA_HOME"
elif [ ! -d "$CUDA_HOME/include" ]; then
  failout "CUDA_HOME/include missing: $CUDA_HOME"
elif [ ! -f "$CUDA_HOME/include/cuda_runtime.h" ]; then
  failout "CUDA_HOME/include/cuda_runtime.h missing: $CUDA_HOME"
else
  pass "CUDA_HOME single root with cuda_runtime.h ($CUDA_HOME)"
fi

# --- 6. CUPTI_HOME has cupti_activity.h ---
if [ -n "${CUPTI_HOME:-}" ] && [ -f "$CUPTI_HOME/include/cupti_activity.h" ]; then
  pass "CUPTI_HOME/include/cupti_activity.h present ($CUPTI_HOME)"
else
  failout "CUPTI_HOME/include/cupti_activity.h missing: ${CUPTI_HOME:-<unset>}"
fi

# --- 7. df preflight: /nix has >= 40G available ---
avail_gb=$(df --output=avail -BG /nix 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -ge 40 ]; then
  pass "df preflight: /nix has ${avail_gb}G available (>= 40G required)"
else
  failout "df preflight: /nix has ${avail_gb:-?}G available (< 40G required)"
fi

printf '\n'
if [ "$fail" -eq 0 ]; then
  printf 'ENV GATE VERDICT: PASS\n'
else
  printf 'ENV GATE VERDICT: FAIL (%s check(s) failed)\n' "$fail"
fi
exit "$fail"
