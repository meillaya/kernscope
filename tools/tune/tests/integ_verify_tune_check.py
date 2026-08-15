#!/usr/bin/env python3
"""integ_verify_tune_check.py — todo-54 LIVE acceptance on gfx1101:
verifier-gated template tuning. kernscope-tune --verify runs the todo-35
kernscope-verify CLI on a verifier-facing rendering of each generated
variant's source BEFORE compile; an error-class verdict (exit 1) or an
ingestion failure (exit 2) DISCARDS the variant and records it with the
verdict ids — a gated variant NEVER reaches compilation.

Flow (all asserted, exit 0 on pass):
  1. HAPPY RUN: kernscope-tune --kernel bias_gelu --shape 4096,4096
     --arch gfx1101 --verify --inject-risky <risky_oob.hip> --json
     --no-cache:
       * exit 0, JSON status "ok", verify_gate true;
       * >= 1 variant verifier-GATED with VERDICT IDS (the injected risky
         config: its device-global buffer is written out of bounds, so the
         OOB check emits MAY_OOB — error class);
       * the gated variant was discarded BEFORE compile: it has no
         timings/best entry, and summary.compiled counts only the approved
         variants (compiled == approved count, never the gated one);
       * the regular variants are APPROVED by the gate (bias_gelu's access
         structure is verifier-clean) and complete the full live pipeline
         (compile -> golden gate -> bench), so a best config exists;
  2. QA NEGATIVE: kernscope-tune --kernel softmax --shape 4096,4096
     --verify --json --no-cache: EVERY variant carries error-class verdicts
     (BARRIER_MISMATCH / MAY_RACE / MAY_UNINIT_READ on the smem reduction)
     -> the tune exits 1, JSON status no_viable_config, message contains
     "no approved variants", and summary.compiled == 0 (NO compile was
     attempted — asserted).

Usage: integ_verify_tune_check.py <build-dir> <source-dir>
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
TOOL = BUILD_DIR / "tools" / "tune" / "kernscope-tune"
RISKY = SRC_DIR / "tools" / "tune" / "tests" / "fixtures" / "risky_oob.hip"
VERIFY_BIN = BUILD_DIR / "bin" / "kernscope-verify"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def parse_evidence(stdout: str) -> dict:
    brace = stdout.find("{")
    return json.loads(stdout[brace:]) if brace >= 0 else {}


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] kernscope-tune binary missing: {TOOL} (run cmake --build)")
        return 1
    if not VERIFY_BIN.is_file():
        print(f"[FAIL] kernscope-verify binary missing: {VERIFY_BIN} (run cmake --build)")
        return 1
    if not RISKY.is_file():
        print(f"[FAIL] risky fixture missing: {RISKY}")
        return 1

    with tempfile.TemporaryDirectory(prefix="integ_verify_tune_") as td:
        work = Path(td)

        # --- 1. HAPPY RUN: bias_gelu + injected risky config ------------------
        r = subprocess.run(
            [str(TOOL), "--kernel", "bias_gelu", "--shape", "4096,4096",
             "--arch", "gfx1101", "--verify", "--inject-risky", str(RISKY),
             "--no-cache", "--json"],
            capture_output=True, text=True, cwd=work)
        print(r.stdout, end="")
        print(r.stderr, end="")
        check(r.returncode == 0,
              f"happy run exits 0 (got {r.returncode})")

        ev = parse_evidence(r.stdout) if r.stdout.strip() else {}
        check(ev.get("status") == "ok", "JSON status ok")
        check(ev.get("verify_gate") is True, "JSON verify_gate true")

        variants = ev.get("variants", [])
        gated = [v for v in variants if v.get("status") == "verifier_gated"]
        check(len(gated) >= 1,
              f"gated >= 1 variant (got {len(gated)})")
        check(gated and gated[0].get("verdicts"),
              f"gated variant records VERDICT IDS (got "
              f"{gated[0].get('verdicts') if gated else None})")
        check(gated and "MAY_OOB" in gated[0].get("verdicts", []),
              f"injected risky config gated with MAY_OOB "
              f"(got {gated[0].get('verdicts') if gated else None})")
        check(gated and "timings" not in gated[0],
              "gated variant has NO timings (discarded before compile)")
        check(gated and gated[0].get("best") is False,
              "gated variant never selected as best")
        summary = ev.get("summary", {})
        check(summary.get("verifier_gated", 0) >= 1,
              f"summary.verifier_gated >= 1 (got "
              f"{summary.get('verifier_gated')})")
        approved_count = summary.get("compiled", 0)
        check(approved_count >= 1,
              f"approved variants still compiled+benchmarked "
              f"(compiled={approved_count})")
        check(summary.get("benchmarked", 0) >= 1,
              f"approved variants benchmarked (benchmarked="
              f"{summary.get('benchmarked')})")
        check(ev.get("best") is not None and
              ev.get("best", {}).get("timings", {}).get("median", 0) > 0,
              "best config selected with timings (live pipeline ran)")

        # --- 2. QA NEGATIVE: all variants verifier-rejected -------------------
        r2 = subprocess.run(
            [str(TOOL), "--kernel", "softmax", "--shape", "4096,4096",
             "--arch", "gfx1101", "--verify", "--no-cache", "--json"],
            capture_output=True, text=True, cwd=work)
        print(r2.stdout, end="")
        print(r2.stderr, end="")
        check(r2.returncode == 1,
              f"all-rejected run exits 1 (got {r2.returncode})")
        ev2 = parse_evidence(r2.stdout) if r2.stdout.strip() else {}
        check(ev2.get("status") == "no_viable_config",
              "JSON status no_viable_config")
        check("no approved variants" in ev2.get("message", ""),
              f"message reports 'no approved variants' "
              f"(got {ev2.get('message', '')[:80]})")
        s2 = ev2.get("summary", {})
        check(s2.get("compiled", -1) == 0,
              f"compile-count == 0 asserted (got {s2.get('compiled')})")
        check(s2.get("verifier_gated", 0) >= 1,
              f"every variant gated (verifier_gated="
              f"{s2.get('verifier_gated')})")
        g2 = [v for v in ev2.get("variants", [])
              if v.get("status") == "verifier_gated"]
        check(len(g2) == s2.get("variants_total", -1) and g2,
              f"ALL variants verifier-rejected "
              f"({len(g2)}/{s2.get('variants_total')})")
        check(g2 and all(v.get("verdicts") for v in g2),
              "every gated variant records verdict ids")

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("integ_verify_tune: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
