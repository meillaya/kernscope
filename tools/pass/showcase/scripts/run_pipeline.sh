#!/usr/bin/env bash
# run_pipeline.sh — todo-42 before/after artifact pipeline for one showcase
# kernel (tools/pass/showcase/<kernel>/kernel.cu).
#
#   usage: run_pipeline.sh <gemm|layernorm|softmax> [--build-dir DIR]
#
# Produces, per showcase, under reports/passes/<kernel>/:
#   before.ll / after.ll   (clang sm_80 device IR; after = the full
#                           kernscope pipeline on the annotated before)
#   before.ptx / after.ptx (llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80,
#                           COMPILE-ONLY — no NVIDIA runtime is claimed;
#                           the measured numbers are the gfx1101 analog)
#   llvm-diff.txt          (llvm-diff summary of before vs after)
#   pass_reports.txt       (per-pass report/decline lines from opt stderr)
#   report.json            (deltas: vector_width, smem_bytes,
#                           coalescing_class, transform status)
#
# The kernel sources are portable CUDA/HIP (see the showcase README); this
# script compiles the CUDA path with the REAL unwrapped clang-21 (the nix
# wrapper injects host-only flags the nvptx64 target rejects — the lit.cfg
# unwrap idiom) against a STAGED CUDA root (lit.cfg _stage_cuda_root idiom:
# include/ from $CUDA_HOME + bin/ + nvvm/ from the nvcc package + the
# curand shim).
#
# The annotation step (sed on before.ll) pins the arch + tile size as
# function attributes — the same mechanism the committed pass fixtures
# (tests/pass/Inputs/tile/pipeline.ll) use; it is a documented part of the
# pipeline, not a hidden edit.

set -euo pipefail

KERNEL="${1:?usage: run_pipeline.sh <gemm|layernorm|softmax>}"
BUILD_DIR="${2:-build}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"

SRC="$ROOT/$KERNEL/kernel.cu"
[ -f "$SRC" ] || { echo "error: no showcase kernel at $SRC" >&2; exit 2; }

OUT_DIR="$REPO/reports/passes/$KERNEL"
mkdir -p "$OUT_DIR"

CLANG_WRAPPER="$(command -v clang)"
CLANG="$CLANG_WRAPPER"
if grep -q '/nix/store/' "$CLANG_WRAPPER" 2>/dev/null; then
  CLANG="$(grep -o '/nix/store/[^ ]*bin/clang' "$CLANG_WRAPPER" | head -1)"
fi
[ -n "$CLANG" ] || { echo "error: unwrapped clang not found" >&2; exit 2; }

# --- stage the synthetic CUDA root (lit.cfg _stage_cuda_root idiom) --------
CUDA_ROOT="${CUDA_HOME:-}"
if [ -z "$CUDA_ROOT" ] || [ ! -d "$CUDA_ROOT/include" ]; then
  echo "error: CUDA_HOME not set or missing include/ (run inside devenv shell)" >&2
  exit 2
fi
STAGE="$BUILD_DIR/showcase_cuda_root"
rm -rf "$STAGE"
mkdir -p "$STAGE/include"
for f in "$CUDA_ROOT"/include/*; do
  ln -s "$f" "$STAGE/include/" 2>/dev/null || true
done
NVCC_PKG="$(dirname "$(dirname "$(command -v nvcc)")")"
[ -d "$NVCC_PKG/bin" ] && ln -s "$NVCC_PKG/bin" "$STAGE/bin"
[ -d "$NVCC_PKG/nvvm" ] && ln -s "$NVCC_PKG/nvvm" "$STAGE/nvvm"
[ -d "$NVCC_PKG/include/crt" ] && ln -s "$NVCC_PKG/include/crt" "$STAGE/include/crt"
printf '#ifndef __KERNSCOPE_CURAND_SHIM__\n#define __KERNSCOPE_CURAND_SHIM__\n#endif\n' > "$STAGE/include/curand_mtgp32_kernel.h"

OPT="$(command -v opt)"
LLC="$(command -v llc)"
LLVM_DIFF="$(command -v llvm-diff)"
PLUGIN="$BUILD_DIR/lib/kernscope-pass.so"
[ -f "$PLUGIN" ] || { echo "error: plugin not built: $PLUGIN" >&2; exit 2; }

# kernscope-as1 first: clang 21 lowers ALL NVPTX device accesses through
# the generic (flat) address space (addrspacecast AS1/AS3 -> p0 constant
# expressions + p0 kernel params); the committed v1 passes are
# direct-AS-scoped, so the pipeline canonicalizes real clang IR into the
# fixture form first (documented in tools/pass/passes/coalesce/src/as1.cpp).
PIPELINE="kernscope-as1,kernscope-tile,kernscope-promote,kernscope-vectorize,kernscope-pad,kernscope-coalesce"

# --- 1. clang -> before.ll ---------------------------------------------------
"$CLANG" -x cuda --cuda-device-only -nocudalib --no-cuda-version-check \
  --cuda-path="$STAGE" --cuda-gpu-arch=sm_80 -O1 -S -emit-llvm \
  "$SRC" -o "$OUT_DIR/before.ll" 2>"$OUT_DIR/clang.log"

# --- 2. annotate arch/tile attrs (fixture mechanism) -------------------------
# kernscope-tile/promote resolve the arch from fn attributes before module
# flags (tile.cpp:97, promote.cpp:97); tile 32 mirrors the committed
# fixtures so promote's {s,+,1} remainder windows are deterministic.
sed -i 's/^define \(.*\)@\([a-zA-Z0-9_]*\)(\(.*\))$/define \1@\2(\3) "kernscope.tile.arch"="sm_80" "kernscope.tile.size"="32" "kernscope.promote.arch"="sm_80"/' \
  "$OUT_DIR/before.ll"

# --- 3. opt pipeline -> after.ll (pass reports captured) ---------------------
# The canonicalized intermediate (canon.ll) is kept so the coalescing
# report can be produced separately on the canonical before (the pipeline
# run's own report lines mix pre/post-transform state).
TMP="$BUILD_DIR/showcase_tmp/$KERNEL"
mkdir -p "$TMP"
"$OPT" -load-pass-plugin="$PLUGIN" \
  -passes="kernscope-as1,verify" \
  -S "$OUT_DIR/before.ll" -o "$TMP/canon.ll" 2>"$TMP/as1.log"
"$OPT" -load-pass-plugin="$PLUGIN" \
  -passes="$PIPELINE,verify" \
  -kernscope-tile-report -kernscope-promote-report \
  -kernscope-vectorize-report -kernscope-coalesce-report \
  -S "$TMP/canon.ll" -o "$OUT_DIR/after.ll" 2>"$OUT_DIR/pass_reports.txt"
# Coalescing classification on the canonical before (the honest "before"
# counts) and the after — the report deltas.
"$OPT" -load-pass-plugin="$PLUGIN" -passes=kernscope-coalesce \
  -kernscope-coalesce-report -S "$TMP/canon.ll" -o /dev/null \
  2>"$OUT_DIR/coalesce_before.txt"
"$OPT" -load-pass-plugin="$PLUGIN" -passes=kernscope-coalesce \
  -kernscope-coalesce-report -S "$OUT_DIR/after.ll" -o /dev/null \
  2>"$OUT_DIR/coalesce_after.txt"

# --- 4. llc PTX (compile-only) ------------------------------------------------
set +e
"$LLC" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$OUT_DIR/before.ll" \
  -o "$OUT_DIR/before.ptx" 2>"$OUT_DIR/llc_before.log"
PTX_BEFORE_EXIT=$?
"$LLC" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$OUT_DIR/after.ll" \
  -o "$OUT_DIR/after.ptx" 2>"$OUT_DIR/llc_after.log"
PTX_AFTER_EXIT=$?
"$LLVM_DIFF" "$OUT_DIR/before.ll" "$OUT_DIR/after.ll" > "$OUT_DIR/llvm-diff.txt" 2>&1
LLVM_DIFF_EXIT=$?
set -e

# --- 5. report.json ------------------------------------------------------------
python3 "$ROOT/scripts/make_report.py" \
  --kernel "$KERNEL" \
  --before "$OUT_DIR/before.ll" \
  --after "$OUT_DIR/after.ll" \
  --reports "$OUT_DIR/pass_reports.txt" \
  --coalesce-before "$OUT_DIR/coalesce_before.txt" \
  --coalesce-after "$OUT_DIR/coalesce_after.txt" \
  --ptx-before-exit "$PTX_BEFORE_EXIT" \
  --ptx-after-exit "$PTX_AFTER_EXIT" \
  --llvm-diff-exit "$LLVM_DIFF_EXIT" \
  --diff "$OUT_DIR/llvm-diff.txt" \
  --out "$OUT_DIR/report.json"

echo "== pipeline OK: $KERNEL"
echo "   before.ll/after.ll: $(wc -l < "$OUT_DIR/before.ll")/$(wc -l < "$OUT_DIR/after.ll") lines"
echo "   ptx exits: before=$PTX_BEFORE_EXIT after=$PTX_AFTER_EXIT"
echo "   report: $OUT_DIR/report.json"
