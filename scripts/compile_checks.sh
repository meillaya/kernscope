#!/usr/bin/env bash
# compile_checks.sh — generated-code compile checks (todo 24).
#
# Runs the kernels/templates portable CUDA/HIP gate suite: nvcc -arch=sm_80
# and -arch=sm_90 COMPILE-ONLY gates for the reference templates (GPU-less,
# LABELS no-gpu), the hipcc --genco gfx1101 compile gates, and the LIVE HIP
# gate (elementwise_add runs on this machine and must earn a kernscope_golden
# verdict pass:true, LABELS gpu-amd).
#
# The gates are ctest tests, so they also run in stage (b); this stage runs
# them explicitly so a generated-code regression fails HERE naming the exact
# gate, independently of the general suite. The hook in scripts/ci.sh becomes
# a hard gate the moment this script exists.
set -uo pipefail

cd "$(dirname "$0")/.."   # repo root, regardless of invocation dir

if [ ! -f build/CTestTestfile.cmake ]; then
    echo "[compile_checks] no build/CTestTestfile.cmake — run scripts/configure.sh first" >&2
    exit 1
fi

ctest --test-dir build -R "kernscope\.templates" --output-on-failure
