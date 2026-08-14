#!/usr/bin/env bash
# smoke_toolchain.sh — [PASS]/[FAIL] gate for the kernscope toolchain.
#
# Stage set (todo 5, golden fixture materialization). Every stage prints
# `[PASS]` or `[FAIL] <reason>`; the script exits non-zero if ANY stage fails.
#
#   a. clang-21 <-> CUDA-12.6 header gate: clang++ -x cuda
#      --no-cuda-version-check --cuda-path=$CUDA_HOME -emit-llvm on
#      tests/fixtures/clang_cuda_smoke.cu (uses __syncthreads + <<< >>>).
#      Fallback branch if this ever breaks is documented in
#      tests/fixtures/README.md ("clang/CUDA header fallback").
#   b. nvcc/ptxas GPU-less compile of fused_attention.cu; asserts the
#      ptxas -v register count is within [97, 112] (the 0.375 occupancy pin).
#   c. hipcc --offload-arch=gfx1101 compile+run of tests/fixtures/hip_smoke.cpp
#      on the local GPU (expects correct output).
#   d. rocminfo reports gfx1101 present.
#
# Run inside `devenv shell` (single-root CUDA_HOME set by todo 2).
# Usage: bash scripts/smoke_toolchain.sh

set -uo pipefail

cd "$(dirname "$0")/.."   # repo root

PASS=0
FAIL=0
fails=()

pass() { printf '[PASS] %s\n' "$1"; PASS=$((PASS + 1)); }
fail() { printf '[FAIL] %s\n' "$1"; FAIL=$((FAIL + 1)); fails+=("$1"); }

TMPDIR="${TMPDIR:-/tmp}"
WORK="$(mktemp -d "${TMPDIR}/smoke_toolchain.XXXXXX" 2>/dev/null || echo "${TMPDIR}/smoke_toolchain")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK"

echo "== kernscope toolchain smoke gates (todo 5) =="

# --- (a) clang-21 <-> CUDA-12.6 headers -------------------------------------
# The nix CUDA packages split the toolkit (cudart lacks bin/, nvvm/, and the
# nvcc-internal crt/ headers; cuda_curand is absent entirely). clang 21's
# CudaInstallationDetector requires include/ + bin/ + nvvm/libdevice, and its
# __clang_cuda_runtime_wrapper.h force-includes curand_mtgp32_kernel.h. The
# gate therefore stages a synthetic single-root CUDA layout in $WORK and a
# minimal non-NVIDIA curand stub (no NVIDIA header is vendored), and uses the
# UNWRAPPED clang 21 binary (the nix clang wrapper bakes in host-only flags
# such as -mtls-dialect=gnu2 that nvptx64 rejects). `-emit-llvm` on both
# passes: --cuda-host-only (parses the <<< >>> launch syntax) and
# --cuda-device-only (lowers __syncthreads to llvm.nvvm.barrier*). See
# tests/fixtures/README.md "clang/CUDA header gate" + fallback branches.
if [ -z "${CUDA_HOME:-}" ]; then
    fail "a. clang/CUDA headers: CUDA_HOME unset (run inside \`devenv shell\`)"
else
    CLANG_WRAPPER="$(readlink -f "$(command -v clang++)" 2>/dev/null || true)"
    CLANG_BIN=""
    if [ -n "${CLANG_WRAPPER}" ]; then
        CLANG_BIN="$(grep -oE '/nix/store/[a-z0-9]+-clang-[0-9.]+/bin/clang' \
            "${CLANG_WRAPPER}" | head -1)"
    fi
    NVCC_BIN="$(command -v nvcc 2>/dev/null || true)"
    NVCC_PKG=""
    [ -n "${NVCC_BIN}" ] && NVCC_PKG="$(dirname "$(dirname \
        "$(readlink -f "${NVCC_BIN}")")")"

    SYNTH="${WORK}/cuda_root"
    rm -rf "${SYNTH}"
    mkdir -p "${SYNTH}/include"
    if [ -d "${CUDA_HOME}/include" ]; then
        for f in "${CUDA_HOME}"/include/*; do
            ln -s "$f" "${SYNTH}/include/$(basename "$f")" 2>/dev/null || true
        done
    fi
    if [ -n "${NVCC_PKG}" ]; then
        ln -sf "${NVCC_PKG}/include/crt" "${SYNTH}/include/crt" 2>/dev/null || true
        ln -s "${NVCC_PKG}/bin" "${SYNTH}/bin" 2>/dev/null || true
        ln -s "${NVCC_PKG}/nvvm" "${SYNTH}/nvvm" 2>/dev/null || true
    fi
    # curand shim: clang's runtime wrapper force-includes this name; the
    # devenv ships no cuda_curand. Test-owned stub, NOT an NVIDIA header.
    printf '%s\n' \
        "// kernscope smoke-test shim (NOT an NVIDIA header)." \
        "// clang's __clang_cuda_runtime_wrapper.h force-includes this name;" \
        "// the devenv CUDA package set omits cuda_curand. Never used." \
        "#ifndef __KERNSCOPE_CURAND_SHIM__" \
        "#define __KERNSCOPE_CURAND_SHIM__" \
        "#endif" > "${SYNTH}/include/curand_mtgp32_kernel.h"

    if [ -z "${CLANG_BIN}" ] || [ ! -x "${CLANG_BIN}" ]; then
        fail "a. clang/CUDA headers: unwrapped clang not found via ${CLANG_WRAPPER}"
    elif [ -z "${NVCC_PKG}" ] || [ ! -f "${SYNTH}/nvvm/libdevice/libdevice.10.bc" ]; then
        fail "a. clang/CUDA headers: cannot stage synthetic CUDA root (nvcc pkg: ${NVCC_PKG})"
    else
        A_OK=1
        A_LOG="${WORK}/clang_cuda_smoke.log"
        : > "${A_LOG}"
        # (a.1) host pass: headers + <<< >>> launch syntax
        if ! "${CLANG_BIN}" -x cuda --no-cuda-version-check \
             --cuda-path="${SYNTH}" -nocudalib --cuda-host-only -emit-llvm \
             -c tests/fixtures/clang_cuda_smoke.cu -o "${WORK}/host.bc" \
             >>"${A_LOG}" 2>&1; then
            A_OK=0
            fail "a. clang/CUDA headers: host-only -emit-llvm failed (${A_LOG})"
        fi
        # (a.2) device pass: headers + __syncthreads -> llvm.nvvm.barrier*
        if [ "${A_OK}" = 1 ] && ! "${CLANG_BIN}" -x cuda \
             --no-cuda-version-check --cuda-path="${SYNTH}" -nocudalib \
             --cuda-device-only -emit-llvm -c tests/fixtures/clang_cuda_smoke.cu \
             -o "${WORK}/dev.bc" >>"${A_LOG}" 2>&1; then
            A_OK=0
            fail "a. clang/CUDA headers: device-only -emit-llvm failed (${A_LOG})"
        fi
        # (a.3) __syncthreads must lower to an llvm.nvvm.barrier intrinsic
        if [ "${A_OK}" = 1 ]; then
            if command -v llvm-dis >/dev/null 2>&1; then
                DEV_IR="$(llvm-dis "${WORK}/dev.bc" -o - 2>/dev/null)"
                if ! printf '%s' "${DEV_IR}" | grep -q "llvm.nvvm.barrier"; then
                    A_OK=0
                    fail "a. clang/CUDA headers: no llvm.nvvm.barrier intrinsic in device IR"
                fi
            fi
        fi
        if [ "${A_OK}" = 1 ]; then
            pass "a. clang-21 <-> CUDA-12.6 headers: host+device -emit-llvm, __syncthreads -> llvm.nvvm.barrier*"
        fi
    fi
fi

# --- (b) nvcc/ptxas GPU-less fused_attention compile + register gate ---------
if [ ! -f tests/fixtures/fused_attention.cu ]; then
    fail "b. nvcc/ptxas: tests/fixtures/fused_attention.cu missing"
elif ! nvcc -arch=sm_86 -lineinfo --maxrregcount=112 -Xptxas -v \
     -cubin tests/fixtures/fused_attention.cu \
     -o "${WORK}/fused_attention.cubin" 2>"${WORK}/ptxas.log"; then
    fail "b. nvcc/ptxas: GPU-less cubin compile failed (see ${WORK}/ptxas.log)"
else
    regs="$(sed -n 's/.*Used \([0-9][0-9]*\) registers.*/\1/p' "${WORK}/ptxas.log" | tail -1)"
    if [ -z "${regs}" ]; then
        fail "b. nvcc/ptxas: could not parse ptxas -v register count"
    elif [ "${regs}" -ge 97 ] && [ "${regs}" -le 112 ]; then
        pass "b. nvcc/ptxas: GPU-less compile, ptxas -v reports ${regs} regs (97-112 pin)"
    else
        fail "b. nvcc/ptxas: ptxas -v reports ${regs} regs, outside [97,112] — golden occupancy would deviate from 0.375"
    fi
    if [ -f tests/fixtures/fused_attention.ptx ]; then
        pass "b. nvcc/ptxas: committed fused_attention.ptx present"
    else
        fail "b. nvcc/ptxas: tests/fixtures/fused_attention.ptx missing"
    fi
fi

# --- (c) hipcc gfx1101 compile+run ------------------------------------------
if hipcc --offload-arch=gfx1101 -std=c++17 -O3 tests/fixtures/hip_smoke.cpp \
    -o "${WORK}/hip_smoke" 2>"${WORK}/hipcc.log"; then
    if "${WORK}/hip_smoke" >"${WORK}/hip_smoke.out" 2>&1; then
        pass "c. hipcc gfx1101 compile+run: $(cat "${WORK}/hip_smoke.out")"
    else
        fail "c. hipcc gfx1101 run failed: $(cat "${WORK}/hip_smoke.out")"
    fi
else
    fail "c. hipcc --offload-arch=gfx1101 compile failed (see ${WORK}/hipcc.log)"
fi

# --- (d) rocminfo gfx1101 ----------------------------------------------------
if command -v rocminfo >/dev/null 2>&1; then
    # Capture first: under `set -o pipefail`, `grep -q` closing its stdin
    # early SIGPIPEs rocminfo and turns a match into exit 141.
    ROC_OUT="$(rocminfo 2>/dev/null)"
    if printf '%s' "${ROC_OUT}" | grep -qi "gfx1101"; then
        pass "d. rocminfo reports gfx1101 (RDNA3) present"
    else
        fail "d. rocminfo: gfx1101 not found in rocminfo output"
    fi
else
    fail "d. rocminfo: not found on PATH (run inside \`devenv shell\`)"
fi

# --- summary -----------------------------------------------------------------
echo "== smoke summary: ${PASS} passed, ${FAIL} failed =="
if [ "${FAIL}" -ne 0 ]; then
    printf 'Failed stages:\n'
    for f in "${fails[@]}"; do printf '  - %s\n' "$f"; done
    exit 1
fi
exit 0
