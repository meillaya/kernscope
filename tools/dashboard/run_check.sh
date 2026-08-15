#!/usr/bin/env bash
# ctest runner for kernscope-dashboard (todo 56): generate + check in one
# test so the dashboard never passes without its validator.
#   happy:    all six module sections available; check_dashboard.py asserts
#             sections + provenance.measured + zero external assets.
#   negative: --missing verify,profile -> "no data" panels + available:false
#             JSON flags, page still valid (never a broken page).
set -euo pipefail
GEN="$1"; CHECK="$2"; OUT="$3"; MODE="$4"
rm -rf "$OUT"
mkdir -p "$OUT"
if [[ "$MODE" == "negative" ]]; then
  "$GEN" --out "$OUT/dashboard.html" --missing verify,profile
  python3 "$CHECK" "$OUT/dashboard.html" --negative
else
  "$GEN" --out "$OUT/dashboard.html"
  python3 "$CHECK" "$OUT/dashboard.html"
fi
echo "[kernscope-dashboard] $MODE check PASS"
