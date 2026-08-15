#!/usr/bin/env bash
# run_analyze_pipeline.sh — todo-54 pass->analyzer artifact pipeline for one
# showcase kernel. Re-runs the todo-42 pipeline (clang -> before.ll, the
# kernscope-pass plugin pipeline -> after.ll, llc -> before.ptx/after.ptx)
# into a CALLER-OWNED workdir, then NORMALIZES the llc PTX for
# kernscope-analyze (the ptxparse module does not accept llc's
# `.param .u64 .ptr .align N` spelling — the `.ptr .align N` qualifiers are
# dropped, keeping the parameter list otherwise byte-identical).
#
#   usage: run_analyze_pipeline.sh <gemm|layernorm|softmax>
#                                 <build-dir> <workdir> <repo-root>
#
# Produces under <workdir>/:
#   before.ll / after.ll   (clang sm_80 device IR; after = the full
#                           kernscope pipeline on the annotated before)
#   before.ptx / after.ptx (llc -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80,
#                           COMPILE-ONLY — no NVIDIA runtime is claimed)
#   before_norm.ptx / after_norm.ptx (analyzer-normalized param lists)
#   pass_reports.txt       (per-pass report/decline lines from opt stderr)
#   coalesce_before.txt / coalesce_after.txt (per-access class counts)
#
# The kernel sources are the todo-42 showcase kernels (tools/pass/showcase,
# read-only); the pipeline, the plugin and the staging idiom mirror
# tools/pass/showcase/scripts/run_pipeline.sh exactly (the CUDA root is
# staged synthetically: include symlink farm + nvcc bin/nvvm/crt + the
# curand shim).

set -euo pipefail

KERNEL="${1:?usage: run_analyze_pipeline.sh <kernel> <build-dir> <workdir> <repo-root>}"
BUILD_DIR="${2:?}"
WORKDIR="${3:?}"
REPO="${4:?}"

SRC="$REPO/tools/pass/showcase/$KERNEL/kernel.cu"
[ -f "$SRC" ] || { echo "error: no showcase kernel at $SRC" >&2; exit 2; }
mkdir -p "$WORKDIR"

CLANG_WRAPPER="$(command -v clang)"
CLANG="$CLANG_WRAPPER"
if grep -q '/nix/store/' "$CLANG_WRAPPER" 2>/dev/null; then
  CLANG="$(grep -o '/nix/store/[^ ]*bin/clang' "$CLANG_WRAPPER" | head -1)"
fi
[ -n "$CLANG" ] || { echo "error: unwrapped clang not found" >&2; exit 2; }

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
PLUGIN="$BUILD_DIR/lib/kernscope-pass.so"
[ -f "$PLUGIN" ] || { echo "error: plugin not built: $PLUGIN" >&2; exit 2; }
[ -n "$LLC" ] || { echo "error: llc not found" >&2; exit 2; }

PIPELINE="kernscope-as1,kernscope-tile,kernscope-promote,kernscope-vectorize,kernscope-pad,kernscope-coalesce"

# --- 1. clang -> before.ll ---------------------------------------------------
"$CLANG" -x cuda --cuda-device-only -nocudalib --no-cuda-version-check \
  --cuda-path="$STAGE" --cuda-gpu-arch=sm_80 -O1 -S -emit-llvm \
  "$SRC" -o "$WORKDIR/before.ll" 2>"$WORKDIR/clang.log"

# --- 2. annotate arch/tile attrs (the todo-42 fixture mechanism) -------------
sed -i 's/^define \(.*\)@\([a-zA-Z0-9_]*\)(\(.*\))$/define \1@\2(\3) "kernscope.tile.arch"="sm_80" "kernscope.tile.size"="32" "kernscope.promote.arch"="sm_80"/' \
  "$WORKDIR/before.ll"

# --- 3. opt pipeline -> after.ll + coalesce class reports --------------------
TMP="$WORKDIR/canon"
mkdir -p "$TMP"
"$OPT" -load-pass-plugin="$PLUGIN" \
  -passes="kernscope-as1,verify" \
  -S "$WORKDIR/before.ll" -o "$TMP/canon.ll" 2>"$TMP/as1.log"
"$OPT" -load-pass-plugin="$PLUGIN" \
  -passes="$PIPELINE,verify" \
  -kernscope-tile-report -kernscope-promote-report \
  -kernscope-vectorize-report -kernscope-coalesce-report \
  -S "$TMP/canon.ll" -o "$WORKDIR/after.ll" 2>"$WORKDIR/pass_reports.txt"
"$OPT" -load-pass-plugin="$PLUGIN" -passes=kernscope-coalesce \
  -kernscope-coalesce-report -S "$TMP/canon.ll" -o /dev/null \
  2>"$WORKDIR/coalesce_before.txt"
"$OPT" -load-pass-plugin="$PLUGIN" -passes=kernscope-coalesce \
  -kernscope-coalesce-report -S "$WORKDIR/after.ll" -o /dev/null \
  2>"$WORKDIR/coalesce_after.txt"

# --- 4. llc PTX (compile-only) + analyzer normalization ----------------------
"$LLC" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$WORKDIR/before.ll" \
  -o "$WORKDIR/before.ptx" 2>"$WORKDIR/llc_before.log"
"$LLC" -mtriple=nvptx64-nvidia-cuda -mcpu=sm_80 "$WORKDIR/after.ll" \
  -o "$WORKDIR/after.ptx" 2>"$WORKDIR/llc_after.log"
sed -e 's/\.param \(\.u64\|\.b64\) \.ptr \.align [0-9]* /\.param \1 /g' \
  "$WORKDIR/before.ptx" > "$WORKDIR/before_norm.ptx"
sed -e 's/\.param \(\.u64\|\.b64\) \.ptr \.align [0-9]* /\.param \1 /g' \
  "$WORKDIR/after.ptx" > "$WORKDIR/after_norm.ptx"

echo "== analyze-pipeline OK: $KERNEL"
echo "   before.ll/after.ll: $(wc -l < "$WORKDIR/before.ll")/$(wc -l < "$WORKDIR/after.ll") lines"
echo "   ptx: $WORKDIR/before_norm.ptx + $WORKDIR/after_norm.ptx"
