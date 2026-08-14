#!/usr/bin/env bash
# kernscope configure script.
#
# Configures the out-of-tree CMake build against LLVM 21 (+ optional MLIR 21)
# from the nix store. If devenv.yaml exists (todo 2 owns it) the whole
# configure runs inside `devenv shell`; otherwise the nix-store cmake dirs are
# resolved with `nix eval` against the same nixpkgs rev polykernel pins - the
# proven LLVM/MLIR-21 toolchain on this machine
# (/home/mei/projects/polykernel/README.md:74-90).
#
# LLVM_EXTERNAL_LIT pin (CRITICAL, proven in /home/mei/projects/tt-flow/
# README.md:44-50): nix's bin/lit is a bash wrapper that breaks LLVM's
# add_lit_target (AddLLVM.cmake spawns "${Python3_EXECUTABLE} <lit>", and the
# wrapper's re-exec loses the interpreter context). The real python lit sits
# next to it as `.lit-wrapped`; we point LLVM_EXTERNAL_LIT at that file.
#
# Environment overrides (all optional): LLVM_DIR, MLIR_DIR, LLVM_EXTERNAL_LIT,
# BUILD_DIR, GENERATOR. Extra args are forwarded to cmake.
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

BUILD_DIR="${BUILD_DIR:-build}"
GENERATOR="${GENERATOR:-Ninja}"

# --- Step 1: prefer running inside the devenv shell --------------------------
# Probe first: the todo-2 worker owns devenv.yaml/devenv.nix and the env may
# be mid-materialization (or temporarily fail to evaluate). If the shell
# cannot be readied within 90s, fall through to the nix-store resolution
# below instead of hard-failing the whole configure.
if [[ -f devenv.yaml ]] && command -v devenv >/dev/null 2>&1 \
  && [[ -z "${INSIDE_DEVENV:-}" ]]; then
  if timeout 90 devenv shell -- true >/dev/null 2>&1; then
    echo "[kernscope/configure] entering devenv shell"
    exec env INSIDE_DEVENV=1 devenv shell -- bash "$0" "$@"
  else
    echo "[kernscope/configure] devenv shell not ready; using nix-store paths"
  fi
fi

# --- Step 2: resolve LLVM-21 / MLIR-21 nix-store cmake dirs ------------------
# nix eval against polykernel's pinned nixpkgs input (the proven 21.1.8 pin on
# this machine). If the resolved path is not yet materialized in the store,
# fetch it from the binary cache first (one-time; the store dedups against the
# devenv the parallel foundation worker materializes).
nixpkgs_attr() {
  nix eval --raw --impure --expr \
    "(builtins.getFlake \"/home/mei/projects/polykernel\").inputs.nixpkgs.legacyPackages.x86_64-linux.$1.outPath" \
    2>/dev/null
}

ensure_materialized() {
  local attr="$1"
  local out_path
  out_path="$(nixpkgs_attr "$attr")"
  if [[ ! -e "${out_path}" ]]; then
    echo "[kernscope/configure] ${attr} not in store; fetching from binary cache..."
    nix build --no-link --impure --expr \
      "(builtins.getFlake \"/home/mei/projects/polykernel\").inputs.nixpkgs.legacyPackages.x86_64-linux.${attr}" >/dev/null
  fi
  printf '%s' "${out_path}"
}

if [[ -z "${LLVM_DIR:-}" ]]; then
  LLVM_DEV="$(ensure_materialized llvmPackages_21.llvm)"
  LLVM_DIR="${LLVM_DEV}/lib/cmake/llvm"
fi
if [[ -z "${MLIR_DIR:-}" ]]; then
  MLIR_DEV="$(ensure_materialized llvmPackages_21.mlir)"
  MLIR_DIR="${MLIR_DEV}/lib/cmake/mlir"
fi
export LLVM_DIR MLIR_DIR
echo "[kernscope/configure] LLVM_DIR=${LLVM_DIR}"
echo "[kernscope/configure] MLIR_DIR=${MLIR_DIR}"
[[ -f "${LLVM_DIR}/LLVMConfig.cmake" ]] \
  || { echo "[kernscope/configure] FATAL: ${LLVM_DIR}/LLVMConfig.cmake missing" >&2; exit 1; }

# --- Step 3: pin LLVM_EXTERNAL_LIT to the real python lit --------------------
# Precedence: explicit env var > `.lit-wrapped` next to the `lit` on PATH
# (devenv shell case) > nix-eval'd python3Packages.lit output.
if [[ -z "${LLVM_EXTERNAL_LIT:-}" ]]; then
  if command -v lit >/dev/null 2>&1; then
    lit_dir="$(dirname -- "$(command -v lit)")"
    if [[ -f "${lit_dir}/.lit-wrapped" ]]; then
      LLVM_EXTERNAL_LIT="${lit_dir}/.lit-wrapped"
    fi
  fi
  if [[ -z "${LLVM_EXTERNAL_LIT:-}" ]]; then
    LLVM_EXTERNAL_LIT="$(ensure_materialized python3Packages.lit)/bin/.lit-wrapped"
  fi
fi
export LLVM_EXTERNAL_LIT
echo "[kernscope/configure] LLVM_EXTERNAL_LIT=${LLVM_EXTERNAL_LIT}"
[[ -f "${LLVM_EXTERNAL_LIT}" ]] \
  || { echo "[kernscope/configure] FATAL: ${LLVM_EXTERNAL_LIT} missing (nix bin/lit is a bash wrapper; use .lit-wrapped)" >&2; exit 1; }

# --- Step 4: cmake configure --------------------------------------------------
if ! command -v ninja >/dev/null 2>&1; then
  echo "[kernscope/configure] ninja not found; falling back to Unix Makefiles"
  GENERATOR="Unix Makefiles"
fi

cmake -B "${BUILD_DIR}" -G "${GENERATOR}" \
  -DLLVM_DIR="${LLVM_DIR}" \
  -DMLIR_DIR="${MLIR_DIR}" \
  -DLLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT}" \
  "$@"
