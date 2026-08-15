#!/usr/bin/env bash
# reproduce-all.sh — todo-57: the MASTER reproducibility entry point.
#
#   bash scripts/reproduce-all.sh
#
# Chains the whole plan into ONE command, pinned by devenv.lock:
#
#   1. lock check      devenv.lock must exist and be tracked (pinned
#                      toolchain); the QA negative deletes it and this
#                      script must FAIL FAST with the exact message
#                      "run devenv update to regenerate lock".
#   2. ci.sh           scripts/ci.sh — configure + build + ctest + lit +
#                      golden harness + compile checks + fixture-size +
#                      NVIDIA-header guards (the full gate).
#   3. demo.sh         scripts/demo.sh — the five-stage end-to-end loop
#                      (profile -> capture -> analyze -> tune -> verify)
#                      on gfx1101, producing reports/demo/report.json.
#   4. dashboard       build/bin/kernscope-dashboard -> reports/dashboard
#                      (.html + .json), aggregating all committed module
#                      reports; SOURCE_DATE_EPOCH defaults to the HEAD
#                      commit time so regeneration is byte-identical.
#
# Exit codes:
#   0  all four stages green
#   1  lock check failed (devenv.lock missing / untracked) or a stage
#      failed
#   2  usage/environment error (missing devenv, build artifacts)
#
# MUST NOT: modify the six tools or libs/*, require network, auto-run on
# NVIDIA, or claim rental results (dashboard honesty discipline: local
# gfx1101 numbers only; rental numbers land under reports/rentals/ only
# when the owner produces them).

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

log() { printf '\n== [reproduce-all] %s\n' "$*"; }

# --- 1. lock check -----------------------------------------------------------
# The devenv.lock pin is the reproducibility contract: the toolchain
# (LLVM/MLIR-21, CUDA 12.6, ROCm 7.2.3) must be the exact locked build.
# Deleting the lock breaks that contract — fail fast, never regenerate
# silently.
if [ ! -f devenv.lock ]; then
    echo "[reproduce-all] FAIL: devenv.lock is missing — run devenv update to regenerate the lock (the pinned toolchain contract is broken)" >&2
    exit 1
fi
if [ -z "$(git ls-files devenv.lock 2>/dev/null)" ]; then
    echo "[reproduce-all] FAIL: devenv.lock exists but is NOT tracked — run devenv update to regenerate the lock and commit it (reproducibility requires the committed pin)" >&2
    exit 1
fi
log "lock check OK — devenv.lock tracked and present (require_version honored by devenv itself)"
log "toolchain check — devenv + toolchain must be on PATH"
command -v devenv >/dev/null 2>&1 || {
    echo "[reproduce-all] devenv not found on PATH — run \`devenv shell\` first (or \`devenv shell -- bash scripts/reproduce-all.sh\`)" >&2
    exit 2
}
for bin in nvcc hipcc llvm-objdump; do
    command -v "$bin" >/dev/null 2>&1 || {
        echo "[reproduce-all] $bin not found on PATH — this script runs inside \`devenv shell\` (toolchain pinned by devenv.lock)" >&2
        exit 2
    }
done

# --- 2. ci.sh ----------------------------------------------------------------
log "stage 1/3 — scripts/ci.sh (configure + build + ctest + lit + golden + compile checks + guards)"
bash scripts/ci.sh

# --- 3. demo.sh --------------------------------------------------------------
log "stage 2/3 — scripts/demo.sh (five-stage loop on gfx1101 -> reports/demo/report.json)"
bash scripts/demo.sh

# --- 4. dashboard ------------------------------------------------------------
log "stage 3/3 — kernscope-dashboard -> reports/dashboard.{html,json}"
DASH="${DASH:-build/bin/kernscope-dashboard}"
if [ ! -x "$DASH" ]; then
    echo "[reproduce-all] missing dashboard binary: $DASH (ci.sh builds it; is this repo configured?)" >&2
    exit 2
fi
# Byte-reproducible "generated" timestamp: default to the HEAD commit time
# (reproducible-builds convention honored by kernscope-dashboard), so two
# runs on the same commit produce identical dashboards.
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}" \
    "$DASH" --out reports/dashboard.html
python3 - "$ROOT/reports/dashboard.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d.get("report_type") == "kernscope_dashboard", d.get("report_type")
secs = [m.get("module") for m in d.get("modules", []) if m.get("available")]
print(f"[reproduce-all]   dashboard ok: {len(secs)} module section(s) with data: {', '.join(secs)}")
EOF

echo
echo "== reproduce-all OK: lock check + ci.sh + demo.sh + dashboard all green"
echo "   evidence: reports/demo/report.json, reports/dashboard.html"
