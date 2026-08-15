#!/usr/bin/env python3
"""integ_pass_analyze_check.py — todo-54 acceptance for the pass->analyzer
delta report loop: kernscope-pass-report --report-pipeline <kernel> re-runs
the todo-42 pipeline on a showcase kernel, runs kernscope-analyze (todo 20)
on the before/after PTX, and emits a DELTA report (registers estimate, smem
bytes, coalescing class, bank-conflict way, vector width) via libs/report.

Flow (all asserted, exit 0 on pass):
  1. kernscope-pass-report --report-pipeline layernorm: the TRANSFORMED
     showcase (vectorize: 1-wide -> 4-wide loads). The delta JSON must
     carry before/after values for ALL FIVE fields (registers_estimate,
     smem_bytes, coalescing_class, bank_conflict_way, vector_width) and at
     least one IMPROVED metric asserted NUMERICALLY (vector_width 1 -> 4
     and the PTX-level vectorized_ld_global 0 -> 12 are the measured
     deltas).
  2. kernscope-pass-report --report-pipeline gemm: the honest
     NO-TRANSFORM showcase (softmax is analyzer-unparsable: llc emits
     callseq blocks for its expf library calls, outside libs/ptxparse
     scope; gemm is the analyzer-friendly no-transform showcase) —
     its delta records no_transform true instead of fabricating
     improvements; the five fields are still present.

Usage: integ_pass_analyze_check.py <build-dir> <source-dir>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
TOOL = BUILD_DIR / "tools" / "tune" / "kernscope-pass-report"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


FIVE_FIELDS = ["registers_estimate", "smem_bytes", "coalescing_class",
               "bank_conflict_way", "vector_width"]


def run_delta(kernel: str, work: Path) -> tuple[int, dict]:
    r = subprocess.run(
        [str(TOOL), "--report-pipeline", kernel, "--build-dir",
         str(BUILD_DIR), "--workdir", str(work / ("w_" + kernel))],
        capture_output=True, text=True)
    print(r.stdout, end="")
    print(r.stderr, end="")
    if r.returncode != 0 or not r.stdout.strip():
        return r.returncode, {}
    brace = r.stdout.find("{")
    return r.returncode, json.loads(r.stdout[brace:])


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] kernscope-pass-report binary missing: {TOOL} "
              f"(run cmake --build)")
        return 1

    with tempfile.TemporaryDirectory(prefix="integ_pass_analyze_") as td:
        work = Path(td)

        # --- 1. layernorm: the transformed showcase --------------------------
        rc, d = run_delta("layernorm", work)
        check(rc == 0, "layernorm delta report exits 0")
        check(d.get("report_type") == "kernscope_report",
              "delta report uses the libs/report model")
        before, after, delta = d.get("before", {}), d.get("after", {}), \
            d.get("delta", {})
        check(all(f in before and f in after for f in FIVE_FIELDS),
              "all five delta fields present in before/after")
        check(before.get("vector_width") == 1 and
              after.get("vector_width") == 4,
              f"vector_width measured 1 -> 4 "
              f"(got {before.get('vector_width')} -> "
              f"{after.get('vector_width')})")
        check(before.get("vectorized_ld_global") == 0 and
              after.get("vectorized_ld_global") > 0,
              f"PTX-level vectorized loads 0 -> "
              f"{after.get('vectorized_ld_global')} (analyzer measured)")
        improved = delta.get("improved_metrics", [])
        check(isinstance(improved, list) and len(improved) >= 1,
              f"at least one IMPROVED metric (got {improved})")
        check("vector_width" in improved,
              "vector_width in improved_metrics")
        check(before.get("registers_estimate", -1) > 0 and
              after.get("registers_estimate", -1) > 0,
              f"registers estimate present both phases "
              f"({before.get('registers_estimate')} -> "
              f"{after.get('registers_estimate')})")
        check(isinstance(before.get("smem_bytes"), (int, float)) and
              isinstance(after.get("smem_bytes"), (int, float)),
              "smem_bytes present both phases (numbers)")
        check(delta.get("no_transform") is False,
              "layernorm delta is not marked no-transform")

        # --- 2. gemm: the honest no-transform showcase ------------------------
        rc2, d2 = run_delta("gemm", work)
        check(rc2 == 0, "gemm delta report exits 0")
        b2, a2, dd2 = d2.get("before", {}), d2.get("after", {}), \
            d2.get("delta", {})
        check(all(f in b2 and f in a2 for f in FIVE_FIELDS),
              "gemm: all five delta fields present in before/after")
        check(dd2.get("no_transform") is True,
              "gemm records no_transform true HONESTLY (no fabricated "
              "improvements)")

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("integ_pass_analyze: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
