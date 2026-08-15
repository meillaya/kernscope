#!/usr/bin/env bash
# build_and_run.sh — compile the todo-42 showcase kernels to gfx1101 hsacos
# and run the LIVE bench through the todo-22 harness, appending libs/db
# records (measured provenance) to reports/passes/bench/.
#
#   usage: build_and_run.sh [--db <path>] [--hsaco-dir <dir>]
#
# Links the todo-22 harness by SOURCE (libs/{gpuruntime,bench,db,arch} —
# REUSED via headers/sources, never modified; the libs CMake targets are
# not available to this ad-hoc hipcc build). Exit codes: 0 = measured +
# records written; 1 = measured but a bench failed; 2 = environment error.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO/build}"
HSACO_DIR="${HSACO_DIR:-$BUILD_DIR/showcase_hsaco}"
DB_PATH="${DB_PATH:-$REPO/reports/passes/bench/gfx1101-showcase-bench.jsonl}"

mkdir -p "$HSACO_DIR" "$(dirname "$DB_PATH")"

for k in gemm layernorm softmax; do
  hipcc --genco --offload-arch=gfx1101 -O2 \
    "$ROOT/$k/kernel.cu" -o "$HSACO_DIR/$k.hsaco"
done
echo "== hsacos built: $(ls "$HSACO_DIR"/*.hsaco | tr '\n' ' ')"

hipcc -std=c++20 -O2 -D__HIP_PLATFORM_AMD__=1 \
  -I "$REPO/libs/gpuruntime/include" \
  -I "$REPO/libs/bench/include" \
  -I "$REPO/libs/db/include" \
  -I "$REPO/libs/arch/include" \
  "$ROOT/bench/bench_driver.cpp" \
  "$REPO/libs/gpuruntime/src/gpu.cpp" \
  "$REPO/libs/bench/src/bench.cpp" \
  "$REPO/libs/db/src/db.cpp" \
  -o "$BUILD_DIR/showcase-bench"

echo "== running live bench on gfx1101 (todo-22 harness)..."
"$BUILD_DIR/showcase-bench" --hsaco-dir "$HSACO_DIR" --db "$DB_PATH"
