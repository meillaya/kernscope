#!/usr/bin/env python3
"""capture_tune_usage_check.py — todo-53 trivial no-gpu usage gate.

Asserts the kernscope-tune CLI's --from-capture mode fails CLEANLY (exit 2,
usage error) when the bundle is missing, without touching any GPU. This is
the skeleton gate wired by the first todo-53 commit; the live acceptance
(integ_capture_tune, gpu-amd) exercises the full pipeline.

Usage: capture_tune_usage_check.py <kernscope-tune-binary>
"""

import subprocess
import sys
from pathlib import Path

TOOL = Path(sys.argv[1]).resolve()

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] kernscope-tune binary missing: {TOOL}")
        return 1

    # --from-capture without a bundle -> exit 2 + usage error
    r = subprocess.run([str(TOOL), "--from-capture"],
                       capture_output=True, text=True)
    check(r.returncode == 2, "--from-capture without a bundle exits 2")
    check("requires an argument" in r.stderr,
          "error message names the missing argument")

    # --from-capture with a nonexistent bundle -> exit 2 (no crash, no GPU)
    r = subprocess.run(
        [str(TOOL), "--from-capture", "/nonexistent/bundle.v0",
         "--vendor", "amd", "--arch", "gfx1101", "--json"],
        capture_output=True, text=True)
    check(r.returncode == 2,
          "--from-capture with a nonexistent bundle exits 2 (usage/env error)")
    check("unrecognized bundle" in r.stderr or "cannot open" in r.stderr or
          "no live GPU" in r.stderr,
          f"error message names the failure ({r.stderr.strip()[:80]!r})")

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("capture_tune_usage: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
