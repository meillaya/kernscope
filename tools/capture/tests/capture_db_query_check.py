#!/usr/bin/env python3
"""capture_db_query_check.py — todo-48 trivial gate: the libs/db JSONL query
works through the kernscope-capture CLI (`db-query` subcommand).

The full bench->db->query flow is exercised by capture_e2e_check.py on
gpu-amd; this is the GPU-less smoke that pins the CLI contract itself: a
hand-written JSONL record must be loadable and queryable, and a filter with
no match must exit 1 (not crash).

Usage: capture_db_query_check.py <build-dir> <source-dir>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
CLI = BUILD_DIR / "kernscope-capture"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not CLI.is_file():
        print(f"[FAIL] kernscope-capture CLI missing: {CLI} (run cmake --build)")
        return 1

    record = {
        "schema_version": 1,
        "gpu_arch": "gfx1101",
        "kernel_family": "capture_replay",
        "shape_hash": "kernel=abc123",
        "config": {"kernel": "layernorm_fused", "measured": True},
        "timings": {"median": 0.1234, "p20": 0.12, "p80": 0.13},
        "provenance": {"timestamp": "2026-08-14T00:00:00Z",
                       "tool_version": "kernscope-capture 0.1.0",
                       "mode": "measured"},
    }

    with tempfile.TemporaryDirectory(prefix="capture_db_query_") as td:
        work = Path(td)
        db = work / "bench.jsonl"
        db.write_text(json.dumps(record) + "\n")

        q = subprocess.run([str(CLI), "db-query", str(db)],
                           capture_output=True, text=True, cwd=work)
        print(q.stdout, end="")
        check(q.returncode == 0, "db-query exit 0 (record found)")
        check("capture_replay" in q.stdout,
              "db-query prints the capture_replay record")
        check("median_ms=0.1234" in q.stdout, "db-query prints the median")

        f = subprocess.run(
            [str(CLI), "db-query", str(db), "--family", "nope"],
            capture_output=True, text=True, cwd=work)
        print(f.stdout, end="")
        check(f.returncode == 1, "db-query exit 1 on no match (no crash)")

        bad = subprocess.run([str(CLI), "db-query", str(work / "missing.jsonl")],
                             capture_output=True, text=True, cwd=work)
        check(bad.returncode == 2, "db-query exit 2 on unreadable db")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_db_query: kernscope-capture db-query works")
    return 0


if __name__ == "__main__":
    sys.exit(main())
