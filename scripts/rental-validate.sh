#!/usr/bin/env bash
# rental-validate.sh — OWNER-GATED NVIDIA rental validation for kernscope.
#
# The local box is AMD-only (/dev/kfd, RX 7800 XT). Everything that needs an
# NVIDIA runtime (CUPTI tracing, CUDA capture injection, CUDA kernel
# execution) is verified on a rented NVIDIA host — and ONLY the repo owner
# may initiate that run. This script is never auto-triggered by CI or by any
# other script (ci.sh does NOT call it; see the CI-refusal guard below).
#
#   bash scripts/rental-validate.sh --dry-run   # print the exact owner
#                                               # command sequence, exit 0
#   bash scripts/rental-validate.sh             # exec: run the
#                                               # gpu-nvidia-rental ctest
#                                               # label on this host
#
# CUDA-13 toolkit-version detection
# --------------------------------
# Per docs/research/profiler-research-brief.md §1.5 (CUDA 12 -> 13 changes):
# CUDA 13 dropped the CUPTI Event API (cupti_events.h), Metric API
# (cupti_metrics.h), PC-sampling activity API, SASS metrics, and removed
# nvprof + NVIDIA Visual Profiler. Activity kinds CUPTI_ACTIVITY_KIND_EVENT /
# _METRIC / _INSTANTANEOUS_* are unsupported, kernel records moved to
# CUpti_ActivityKernel10, CUPTI_API_VERSION is now 130001, and the sanctioned
# paths are the CUPTI Range-Profiler / PM-sampling / multi-subscriber V2
# APIs. This script therefore degrades GRACEFULLY on newer hosts: it detects
# the toolkit major version from `nvcc --version`, prints a notice that
# Event/Metric/nvprof paths are unavailable, and proceeds with the
# modern-API expectations — it never hard-fails on a CUDA-13 host.

set -uo pipefail

cd "$(dirname "$0")/.."   # repo root

LABEL="gpu-nvidia-rental"
EVIDENCE_DIR="reports/rentals"

# --- dry-run: print the exact owner-executed command sequence ---------------
if [ "${1:-}" = "--dry-run" ]; then
    cat <<DRYRUN
[kernscope] rental-validate.sh --dry-run
Owner-executed command sequence for an NVIDIA rental validation run
(owner-initiated only — never auto-triggered by CI):

  STEP 1 — CUDA toolkit version check (on the rental host)
      ssh <rental-host> nvcc --version
      # CUDA 13 note: Event/Metric/PC-sampling APIs + nvprof removed
      # (docs/research/profiler-research-brief.md §1.5); expect
      # CUpti_ActivityKernel10 + CUPTI_API_VERSION 130001 on 13.x.

  STEP 2 — CUPTI version detect (on the rental host)
      ssh <rental-host> bash -c 'grep -m1 "CUPTI_API_VERSION" "\$(find / -path \"*/extras/CUPTI/include/cupti_version.h\" 2>/dev/null | head -1)"'

  STEP 3 — run the NVIDIA-runtime ctest label
      cd build && ctest -L gpu-nvidia-rental --output-on-failure

  STEP 4 — evidence lands under reports/rentals/ (owner commits it after
      inspecting the results; e.g. reports/rentals/<timestamp>-summary.txt)

Run the real thing with:  bash scripts/rental-validate.sh
DRYRUN
    exit 0
fi

# --- exec mode ---------------------------------------------------------------
echo "[kernscope] rental-validate.sh exec mode — OWNER-GATED NVIDIA validation"

# Belt-and-suspenders: never auto-triggered. If any CI service ever invokes
# this script, refuse outright.
if [ "${CI:-}" = "true" ]; then
    echo "[FAIL] refusing to run under CI — rental validation is owner-initiated only" >&2
    exit 1
fi

if [ ! -f build/CTestTestfile.cmake ]; then
    echo "[FAIL] build/ not configured yet — run scripts/ci.sh (or configure first)" >&2
    exit 1
fi

# STEP 1 — CUDA toolkit version check (documents CUDA-13 degradation, §1.5)
cuda_major=""
if command -v nvcc >/dev/null 2>&1; then
    nvcc --version
    cuda_major=$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -1)
    if [ "${cuda_major:-0}" -ge 13 ]; then
        echo "[WARN] CUDA ${cuda_major} toolkit: Event/Metric/PC-sampling APIs and nvprof are REMOVED (§1.5); using CUpti_ActivityKernel10 / CUPTI_API_VERSION 130001 expectations."
    else
        echo "[INFO] CUDA ${cuda_major:-?} toolkit — legacy Event/Metric/nvprof paths still available for comparison."
    fi
else
    echo "[WARN] nvcc not found on this host — this script must run ON the rental host (or via ssh); continuing with ctest below."
fi

# STEP 2 — CUPTI version detect
cupti_ver="not detected"
for cand in \
    "${CUDA_HOME:-}/extras/CUPTI/include/cupti_version.h" \
    /usr/local/cuda/extras/CUPTI/include/cupti_version.h \
    /opt/cuda/extras/CUPTI/include/cupti_version.h; do
    if [ -f "$cand" ]; then
        cupti_ver=$(grep -m1 -o 'CUPTI_API_VERSION[^0-9]*[0-9]*' "$cand" || true)
        break
    fi
done
echo "[INFO] CUPTI version detect: ${cupti_ver}"

# STEP 3 — run the gpu-nvidia-rental ctest label
mkdir -p "$EVIDENCE_DIR"
stamp=$(date +%Y%m%d-%H%M%S)
if ctest --test-dir build -L gpu-nvidia-rental --output-on-failure; then
    verdict="PASS"
    rc=0
else
    verdict="FAIL"
    rc=$?
fi

# STEP 4 — evidence lands under reports/rentals/
summary="$EVIDENCE_DIR/${stamp}-summary.txt"
{
    echo "# kernscope NVIDIA-rental validation — ${stamp}"
    echo "label: ${LABEL}   verdict: ${verdict}   ctest exit: ${rc}"
    echo "cuda_major: ${cuda_major:-n/a}   cupti: ${cupti_ver}"
    echo "command: ctest --test-dir build -L gpu-nvidia-rental --output-on-failure"
} > "$summary"
echo "[INFO] evidence written to ${summary} (owner commits it after inspection)"

if [ "$verdict" = "PASS" ]; then
    echo "[PASS] gpu-nvidia-rental validation green"
    exit 0
fi
echo "[FAIL] gpu-nvidia-rental validation failed (see ctest output above)" >&2
exit 1
