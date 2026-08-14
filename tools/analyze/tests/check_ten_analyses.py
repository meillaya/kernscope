#!/usr/bin/env python3
#===- check_ten_analyses.py - ten-analysis completeness check -------------===#
#
# Wave-3 todo 20 (owned by this todo): runs the real kernscope-analyze CLI
# (--json) over the committed fused_attention golden and asserts that ALL
# TEN analyses of the analyzer are present in the report, one asserted field
# per analysis (Metis L-2 discipline: numeric JSON assertions only, no prose):
#
#    1. register pressure     -> metrics.registers_per_thread
#    2. occupancy limits      -> metrics.occupancy
#    3. coalescing            -> metrics.global_coalescing_efficiency
#    4. bank conflicts        -> metrics.shared_bank_way
#    5. divergence            -> warning id "divergent_inner_loop"
#    6. barrier usage         -> metrics.bar_sync_count
#    7. instruction mix       -> metrics.fma_rn_f32
#    8. tensor-core/MFMA/WMMA -> metrics.tensor_ops
#    9. spill detection       -> metrics.spill_bytes
#   10. latency bottlenecks   -> metrics.latency_classification
#
# The CLI exits 1 when warnings are present (the golden carries three) —
# that is a SUCCESSFUL analysis run, so return codes 0 and 1 both proceed;
# only 2 (error) and 3 (usage) fail the run.
#
# The script is cwd-independent: it resolves the repo root from its own
# location (tools/analyze/tests/ -> repo root) and locates the CLI via
# argv[1] (ctest passes $<TARGET_FILE:kernscope-analyze>) or the default
# <root>/build/bin/kernscope-analyze.
#
# Usage: python3 tools/analyze/tests/check_ten_analyses.py [path-to-cli]
# Exit 0 on pass, 1 on any failure (the failing analysis is named).
#===-=======================================================================#

import json
import os
import subprocess
import sys

# tools/analyze/tests/check_ten_analyses.py -> repo root (4 dirnames).
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

GOLDEN_PTX = os.path.join(ROOT, "tests", "fixtures", "fused_attention.ptx")

# (analysis name, check kind, field) — kind "metric" asserts the key exists
# in metrics with a finite number; kind "finding" asserts a finding with the
# exact id exists in the warnings array.
TEN_ANALYSES = [
    ("register pressure",        "metric",  "registers_per_thread"),
    ("occupancy limits",         "metric",  "occupancy"),
    ("coalescing",               "metric",  "global_coalescing_efficiency"),
    ("bank conflicts",           "metric",  "shared_bank_way"),
    ("warp divergence",          "finding", "divergent_inner_loop"),
    ("barrier usage",            "metric",  "bar_sync_count"),
    ("instruction mix",          "metric",  "fma_rn_f32"),
    ("tensor-core/MFMA/WMMA",    "metric",  "tensor_ops"),
    ("spill detection",          "metric",  "spill_bytes"),
    ("latency bottlenecks",      "metric",  "latency_classification"),
]


def fail(msg):
    print(f"FAIL: {msg}")
    sys.exit(1)


def main():
    cli = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, "build", "bin", "kernscope-analyze")
    if not os.path.isfile(cli):
        fail(f"kernscope-analyze not found at {cli}")

    proc = subprocess.run(
        [cli, GOLDEN_PTX, "--arch", "sm_86", "--block", "64", "--json"],
        capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        fail(f"kernscope-analyze exited {proc.returncode} (expected 0 or 1: "
             f"1 = warnings present, a successful analysis); stderr: "
             f"{proc.stderr.strip()}")

    try:
        report = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        fail(f"report is not valid JSON: {e}; stdout head: "
             f"{proc.stdout[:200]!r}")

    metrics = report.get("metrics", {})
    warnings = report.get("warnings", [])
    warning_ids = {w.get("id") for w in warnings}

    passed = 0
    for name, kind, field in TEN_ANALYSES:
        if kind == "metric":
            value = metrics.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                print(f"PASS: {name:<24} {field} = {value}")
                passed += 1
            else:
                fail(f"{name}: metric '{field}' missing or not numeric in "
                     f"report metrics")
        else:
            if field in warning_ids:
                print(f"PASS: {name:<24} warning id '{field}' present")
                passed += 1
            else:
                fail(f"{name}: warning id '{field}' missing; got "
                     f"warnings={sorted(warning_ids)}")

    print(f"PASS: all ten analyses asserted ({passed}/10) on "
          f"{os.path.basename(GOLDEN_PTX)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
