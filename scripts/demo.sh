#!/usr/bin/env bash
# demo.sh — todo-55: the ONE-COMMAND end-to-end loop demo (gpu-kernel-suite).
#
#   bash scripts/demo.sh
#
# Runs the full kernscope loop on this machine (gfx1101, no network, no
# rentals) and emits ONE aggregated, self-contained report:
#
#   reports/demo/report.json  — the aggregated model (libs/report
#                               conventions: schema_version 1.0,
#                               machine-readable evidence refs)
#   reports/demo/report.html  — self-contained page with the embedded JSON
#                               payload (zero network requests)
#
# The five stages (each stage CLI INVOKED, never modified):
#   1. profile      kernscope-profile build/tests/fixtures/layernorm_fused
#                   -> hot kernel + compiler-aware diagnostics (todo 52)
#   2. capture      kernscope-capture record + replay (bitwise validation)
#                   of the hot kernel -> bundle (todo 45/48)
#   3. analyze      kernscope-analyze on the CAPTURED kernel's AMDGPU ISA
#                   (disassembled from the code-object blob) -> static
#                   findings (todo 20)
#   4. capture_tune kernscope-tune --from-capture: launch-config search on
#                   the bundle, bitwise-validated (todo 53)
#   5. verify_tune  kernscope-tune --verify (verifier-gated template tune,
#                   todo 54 happy family bias_gelu + injected risky
#                   variant) AND kernscope-pass-report --report-pipeline
#                   layernorm (pass->analyzer delta, todo 54)
#
# The aggregated report links all five stages via evidence refs (paths that
# exist on disk — asserted by tests/integration/check_loop.py) and carries
# the cross-tool diagnostic: profiler hints reference the analyzer finding
# ids from the stage-3 analysis of the captured kernel's ISA.
#
# GPU gate: requires an AMD GPU (/dev/kfd). KERNSCOPE_NO_GPU=1 simulates a
# masked device for the QA negative: the demo ABORTS EARLY with a clear
# "gpu-amd required" message and writes a well-formed all-skipped report
# (graceful; the negative contract is asserted by
# tests/integration/check_loop.py --allow-skipped).
#
# Exit codes:
#   0  full loop + aggregated report + check_loop assertions green
#   1  gpu-amd required (KERNSCOPE_NO_GPU=1 or no /dev/kfd) — the QA
#      negative; or a stage failed / budget exceeded
#   2  usage/environment error (missing build artifacts)
#
# MUST NOT: modify the five stage tools (they are invoked via their CLIs),
# modify libs/* (reused via the CLIs' outputs + conventions), require
# network or rentals, auto-run on NVIDIA.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build"
DEMO_DIR="$ROOT/reports/demo"
RUNTIME_BUDGET=900  # seconds — the asserted < 15 min budget

PROFILE="$BUILD/bin/kernscope-profile"
CAPTURE="$BUILD/kernscope-capture"
ANALYZE="$BUILD/bin/kernscope-analyze"
TUNE="$BUILD/tools/tune/kernscope-tune"
PASS_REPORT="$BUILD/tools/tune/kernscope-pass-report"
AGGREGATE=(python3 "$ROOT/scripts/demo_aggregate.py")
CHECK_LOOP=(python3 "$ROOT/tests/integration/check_loop.py")

LAYERNORM_DIR="$ROOT/tests/fixtures/layernorm_fused"
APP_DIR="$BUILD/tests/fixtures/layernorm_fused"
APP="$APP_DIR/layernorm_app"
RISKY="$ROOT/tools/tune/tests/fixtures/risky_oob.hip"

log() { printf '\n== [demo] %s\n' "$*"; }

# --- environment sanity ------------------------------------------------------
for tool in "$PROFILE" "$CAPTURE" "$ANALYZE" "$TUNE" "$PASS_REPORT"; do
    if [ ! -x "$tool" ]; then
        echo "[demo] missing tool: $tool (run cmake --build build)" >&2
        exit 2
    fi
done
if [ ! -f "$APP" ]; then
    echo "[demo] missing fixture app: $APP (run cmake --build build)" >&2
    exit 2
fi
if [ ! -f "$RISKY" ]; then
    echo "[demo] missing verify-risky fixture: $RISKY" >&2
    exit 2
fi
command -v llvm-objdump >/dev/null 2>&1 || {
    echo "[demo] llvm-objdump not found (run inside \`devenv shell\`)" >&2
    exit 2
}
python3 -c 'import numpy' 2>/dev/null || {
    echo "[demo] python3 numpy not found (run inside \`devenv shell\`)" >&2
    exit 2
}

# --- GPU gate (QA hook: KERNSCOPE_NO_GPU=1 masks /dev/kfd) -------------------
if [ "${KERNSCOPE_NO_GPU:-0}" = "1" ] || [ ! -e /dev/kfd ]; then
    echo "[demo] gpu-amd required: /dev/kfd not available (KERNSCOPE_NO_GPU=${KERNSCOPE_NO_GPU:-0}) — aborting the loop before any stage" >&2
    "${AGGREGATE[@]}" skipped "$DEMO_DIR" "gpu-amd required: /dev/kfd masked (KERNSCOPE_NO_GPU=${KERNSCOPE_NO_GPU:-0}); stages not run (graceful abort)"
    echo "[demo] wrote all-skipped report to $DEMO_DIR/report.json (graceful negative)" >&2
    exit 1
fi

START_S="$(date +%s)"
mkdir -p "$DEMO_DIR/capture/blobs"
CAPTURE_DIR="$DEMO_DIR/capture"

# === stage 1: profile (todo 52) ==============================================
log "stage 1/5 — kernscope-profile: trace layernorm_fused -> hot kernel + diagnostics"
"$PROFILE" "$APP_DIR" --json > "$DEMO_DIR/stage-1-profile.json"
"$PROFILE" "$APP_DIR" --html > "$DEMO_DIR/stage-1-profile.html"
python3 - "$DEMO_DIR/stage-1-profile.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["report_type"] == "kernscope_profile_report" and not d["no_kernels_traced"]
hints = sum(len(k.get("hints", [])) for k in d["kernels"])
print(f"[demo]   profile ok: {d['kernel_count']} kernels, hot layernorm hint(s): {hints}")
EOF

# === stage 2: capture the hot kernel -> bundle (todo 45/48) ==================
log "stage 2/5 — kernscope-capture: record the hot kernel -> bundle + replay (bitwise)"
python3 - "$LAYERNORM_DIR/golden_inputs.npz" "$CAPTURE_DIR/input.bin" <<'EOF'
import numpy as np, sys
d = np.load(sys.argv[1])
x = d["X"].astype(np.uint16); g = d["gamma"].astype(np.uint16); b = d["beta"].astype(np.uint16)
M, N = x.shape
with open(sys.argv[2], "wb") as f:
    f.write(np.int32(M).tobytes()); f.write(np.int32(N).tobytes())
    f.write(x.tobytes()); f.write(g.tobytes()); f.write(b.tobytes())
print(f"[demo]   input.bin: {M}x{N} uint16 (committed golden)")
EOF
"$CAPTURE" record "$APP" "$CAPTURE_DIR/input.bin" "$CAPTURE_DIR/output.bin" \
    --bundle "$CAPTURE_DIR/bundle.v0" --blob-dir "$CAPTURE_DIR/blobs" \
    > "$DEMO_DIR/stage-2-record.log" 2>&1
grep -q "auto-detect -> amd" "$DEMO_DIR/stage-2-record.log" \
    || { echo "[demo] capture vendor auto-detect did not select amd" >&2; cat "$DEMO_DIR/stage-2-record.log" >&2; exit 1; }
"$CAPTURE" replay "$CAPTURE_DIR/bundle.v0" --vendor amd --arch gfx1101 \
    --blob-dir "$CAPTURE_DIR/blobs" --ref "$CAPTURE_DIR/output.bin" \
    --mode bitwise --dump-dir "$CAPTURE_DIR/dump" \
    > "$CAPTURE_DIR/replay.txt" 2>&1
grep -q "bitwise EQUAL" "$CAPTURE_DIR/replay.txt" \
    || { echo "[demo] replay bitwise validation did not PASS" >&2; cat "$CAPTURE_DIR/replay.txt" >&2; exit 1; }
grep -q "validation: PASS" "$CAPTURE_DIR/replay.txt" \
    || { echo "[demo] replay validation missing" >&2; exit 1; }
echo "[demo]   bundle: $CAPTURE_DIR/bundle.v0 (bitwise replay PASS)"

# === stage 3: analyze the captured kernel's ISA (todo 20) ====================
log "stage 3/5 — kernscope-analyze: static findings on the captured kernel's AMDGPU ISA"
HSA=""
TUNE_WORK=""
PASS_WORK=""
OBJDUMP_TMP="$(mktemp)"
trap 'rm -rf "$TUNE_WORK" "$PASS_WORK" "$OBJDUMP_TMP"' EXIT
for blob in "$CAPTURE_DIR"/blobs/code-object-*.hsaco; do
    # grep a FILE, not the pipe: `grep -q` on a pipe closes it after the
    # first match, SIGPIPEs llvm-objdump, and with `set -o pipefail` the
    # pipeline then reports 141 even though the pattern matched.
    if llvm-objdump -d "$blob" > "$OBJDUMP_TMP" 2>/dev/null \
        && grep -q "layernorm_fused" "$OBJDUMP_TMP"; then
        HSA="$blob"
        break
    fi
done
[ -n "$HSA" ] || { echo "[demo] no layernorm code-object blob captured" >&2; exit 1; }
"${AGGREGATE[@]}" sanitize-isa "$HSA" "$CAPTURE_DIR/layernorm.isa"
"$ANALYZE" "$CAPTURE_DIR/layernorm.isa" --arch gfx1101 --block 128 --json \
    > "$DEMO_DIR/stage-3-analyze.json" || rc=$?
rc="${rc:-0}"
# analyzer exit codes: 0/1 = report produced (1 = warnings present), 2/3 = error
if [ "$rc" -ge 2 ]; then
    echo "[demo] kernscope-analyze failed (exit $rc)" >&2
    exit 1
fi
"$ANALYZE" "$CAPTURE_DIR/layernorm.isa" --arch gfx1101 --block 128 --html \
    > "$DEMO_DIR/stage-3-analyze.html" || true
python3 - "$DEMO_DIR/stage-3-analyze.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
n = len(d.get("warnings", [])) + len(d.get("suggestions", []))
print(f"[demo]   analyzer ok: {n} findings (warnings+suggestions)")
EOF

# === stage 4: capture->tune launch-config search (todo 53) ===================
log "stage 4/5 — kernscope-tune --from-capture: launch-config search on the bundle"
"$TUNE" --from-capture "$CAPTURE_DIR/bundle.v0" --vendor amd --arch gfx1101 \
    --blob-dir "$CAPTURE_DIR/blobs" --ref "$CAPTURE_DIR/output.bin" \
    --json --db "$DEMO_DIR/stage-4-tune.jsonl" \
    > "$DEMO_DIR/stage-4-capture-tune.json"
python3 - "$DEMO_DIR/stage-4-capture-tune.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "ok" and d["mode"] == "capture_tune"
validated = [c for c in d.get("candidates", []) if c.get("status") == "validated"]
assert len(validated) >= 3 and all(c["validation"]["pass"] for c in validated)
best = d.get("best", {}).get("timings", {}).get("median")
print(f"[demo]   capture-tune ok: {len(validated)} bitwise-validated candidates, "
      f"best median {best} ms, speedup {d.get('speedup')}")
EOF

# === stage 5: verifier-gated template tune + pass->analyzer delta (todo 54) ==
log "stage 5/5 — kernscope-tune --verify (verifier-gated) + kernscope-pass-report --report-pipeline"
TUNE_WORK="$(mktemp -d)"
PASS_WORK="$(mktemp -d)"
"$TUNE" --kernel bias_gelu --shape 4096,4096 --arch gfx1101 --verify \
    --inject-risky "$RISKY" --no-cache --json \
    --db "$DEMO_DIR/stage-5-verify-tune.jsonl" --workdir "$TUNE_WORK" \
    > "$DEMO_DIR/stage-5-verify-tune.json"
python3 - "$DEMO_DIR/stage-5-verify-tune.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["status"] == "ok" and d["verify_gate"] is True
g = [v for v in d.get("variants", []) if v.get("status") == "verifier_gated"]
assert g and any("MAY_OOB" in v.get("verdicts", []) for v in g)
print(f"[demo]   verify-tune ok: {d['summary'].get('verifier_gated')} variant(s) "
      f"gated (incl. injected MAY_OOB), {d['summary'].get('compiled')} compiled")
EOF
"$PASS_REPORT" --report-pipeline layernorm --build-dir "$BUILD" \
    --workdir "$PASS_WORK" > "$DEMO_DIR/stage-5-pass-delta.json"
python3 - "$DEMO_DIR/stage-5-pass-delta.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
delta = d.get("delta", {})
assert d.get("report_type") == "kernscope_report" and delta.get("no_transform") is False
assert delta.get("vector_width_delta") == 3 and delta.get("vectorized_ld_global_delta", 0) > 0
print(f"[demo]   pass-delta ok: vector_width 1 -> 4, vectorized loads 0 -> "
      f"{delta.get('vectorized_ld_global_delta')} (analyzer-measured)")
EOF

# === aggregate + assert ======================================================
END_S="$(date +%s)"
RUNTIME=$((END_S - START_S))
log "aggregate — one report linking all five stages"
"${AGGREGATE[@]}" aggregate "$DEMO_DIR" "$RUNTIME"
log "assert — tests/integration/check_loop.py"
"${CHECK_LOOP[@]}" "$DEMO_DIR/report.json"

if [ "$RUNTIME" -ge "$RUNTIME_BUDGET" ]; then
    echo "[demo] FAIL: runtime ${RUNTIME}s >= budget ${RUNTIME_BUDGET}s (< 15 min, asserted)" >&2
    exit 1
fi

echo
echo "== demo OK: loop closed on $(hostname) in ${RUNTIME}s (< ${RUNTIME_BUDGET}s budget)"
echo "   report: $DEMO_DIR/report.json"
echo "   html:   $DEMO_DIR/report.html"
