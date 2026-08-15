#!/usr/bin/env python3
"""make_report.py — todo-42 per-showcase report.json (deltas + honest
transform status).

Parses the before/after IR + the captured pass reports and emits the
report.json the acceptance criteria assert on:

  * vector_width_before/after      — widest vector load in the .ll
  * smem_bytes_before/after        — addrspace(3) storage totals
  * coalescing_class_before/after  — per-class counts from the
    -kernscope-coalesce-report line (the IR-level classifier, same
    vocabulary as the todo-15 PTX analyzer)
  * transform / smem_promoted / no-transform — the honest verdict:
    smem_promoted = promote report line present (cooperative block load
    into addrspace(3)); vectorized = vectorize report line present;
    no-transform = neither (recorded HONESTLY — the todo-42 QA-negative
    case, asserted in the report JSON and the evidence file)
  * ptx_compile_before/after_exit  — llc exit codes (compile-only)
  * provenance                    — pipeline name, arch, tool version,
    mode "projected" (the static-artifact side; the measured gfx1101
    timings live in the libs/db records with mode "measured")
"""

import argparse
import json
import re
import sys

ELEM_BYTES = {"float": 4, "double": 8, "i32": 4, "i64": 8, "i16": 2, "i8": 1}


def vector_width(ir: str) -> int:
    """Widest vector load width in elements (1 = all scalar)."""
    width = 1
    for m in re.finditer(r"load <(\d+) x", ir):
        width = max(width, int(m.group(1)))
    return width


def smem_bytes(ir: str) -> int:
    """Total addrspace(3) storage bytes (clang emits smem as globals)."""
    total = 0
    for m in re.finditer(
        r"addrspace\(3\) global \[(\d+) x (float|double|i\d+)\]", ir
    ):
        n, ty = int(m.group(1)), m.group(2)
        total += n * ELEM_BYTES.get(ty, 4)
    return total


def coalescing_counts(reports: str) -> dict:
    """Sum the per-function -kernscope-coalesce-report lines."""
    counts = {"fully": 0, "partial": 0, "uncoalesced": 0, "unknown": 0,
              "total": 0}
    for m in re.finditer(
        r"kernscope-coalesce: \S+: (\d+) global accesses \((\d+) fully, "
        r"(\d+) partial, (\d+) uncoalesced, (\d+) unknown\)",
        reports,
    ):
        counts["total"] += int(m.group(1))
        counts["fully"] += int(m.group(2))
        counts["partial"] += int(m.group(3))
        counts["uncoalesced"] += int(m.group(4))
        counts["unknown"] += int(m.group(5))
    return counts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--coalesce-before", required=True)
    ap.add_argument("--coalesce-after", required=True)
    ap.add_argument("--ptx-before-exit", type=int, required=True)
    ap.add_argument("--ptx-after-exit", type=int, required=True)
    ap.add_argument("--llvm-diff-exit", type=int, required=True)
    ap.add_argument("--diff", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    before = open(args.before).read()
    after = open(args.after).read()
    reports = open(args.reports).read()
    coalesce_before = open(args.coalesce_before).read()
    coalesce_after = open(args.coalesce_after).read()

    vw_before, vw_after = vector_width(before), vector_width(after)
    smem_before, smem_after = smem_bytes(before), smem_bytes(after)

    # Honest transform verdicts from the per-pass report lines.
    smem_promoted = bool(
        re.search(r"kernscope-promote: .*promoted", reports)
    )
    vectorized = bool(
        re.search(r"kernscope-vectorize: .*vectorized [1-9]", reports)
    )
    tiled = bool(re.search(r"kernscope-tile: .*tiled", reports))
    if smem_promoted:
        transform = "promote"
    elif vw_after > vw_before or vectorized:
        transform = "vectorize"
    else:
        transform = "no-transform"

    diff_text = open(args.diff).read()
    # llvm-diff: exit 0 = identical IRs, exit 1 = differences (the
    # pass pipeline changed the IR).
    ir_changed = args.llvm_diff_exit == 1 or bool(diff_text.strip())
    report = {
        "kernel": args.kernel,
        "ir_changed": ir_changed,
        "llvm_diff_summary": diff_text.strip().splitlines()[:4],
        "transform": transform,
        "no-transform": transform == "no-transform",
        "smem_promoted": smem_promoted,
        "vectorized": vectorized,
        "tiled": tiled,
        "vector_width_before": vw_before,
        "vector_width_after": vw_after,
        "vector_width_increased": vw_after > vw_before,
        "smem_bytes_before": smem_before,
        "smem_bytes_after": smem_after,
        "smem_bytes_increased": smem_after > smem_before,
        "coalescing_class_before": coalescing_counts(coalesce_before),
        "coalescing_class_after": coalescing_counts(coalesce_after),
        "ptx_compile_before_exit": args.ptx_before_exit,
        "ptx_compile_after_exit": args.ptx_after_exit,
        "ptx_compile_all_exit0": args.ptx_before_exit == 0
        and args.ptx_after_exit == 0,
        "llvm_diff_exit": args.llvm_diff_exit,
        "pipeline": "kernscope-tile,kernscope-promote,kernscope-vectorize,"
                    "kernscope-pad,kernscope-coalesce",
        "arch": "sm_80",
        "provenance": {
            "mode": "projected",
            "tool_version": "kernscope-pass 0.1.0",
            "note": "static IR/PTX artifacts (llc compile-only); the "
                    "measured numbers are the gfx1101 analog in the "
                    "libs/db records (mode: measured)",
        },
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
