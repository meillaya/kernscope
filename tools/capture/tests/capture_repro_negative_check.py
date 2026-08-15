#!/usr/bin/env python3
"""capture_repro_negative_check.py — todo-47 QA negative: a bundle with a
size-unknown buffer must NOT crash the generator; the emitted reproducer
carries a CLEARLY-MARKED synthetic input + warning header and COMPILES.

Fixture: tests/fixtures/capture/negative-size-unknown.bundle (hand-written
v0 text):
  * launch arg #0 points at 0xdead0000 - NOT in the allocation map, and the
    only copy touching it carries dst-size=0 (size-unknown, the v0
    convention the todo-45 capture emits for unhooked allocations);
  * launch arg #1 points at 0x7f0000000000 - a KNOWN allocation with NO
    captured blob (missing H2D content).

The generator must classify both as pointer args, substitute synthetic
inputs, mark the source KERNSCOPE_SYNTHETIC_INPUT + WARNING, and emit a
project that still COMPILES with hipcc (no GPU needed).

Usage: capture_repro_negative_check.py <build-dir> <capture-source-dir>
       <tests/fixtures/capture dir>
"""

import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURES_DIR = Path(sys.argv[3]).resolve()
CLI = BUILD_DIR / "kernscope-capture"
FIXTURE = FIXTURES_DIR / "negative-size-unknown.bundle"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not CLI.is_file():
        print(f"[FAIL] kernscope-capture CLI missing: {CLI}")
        return 1
    if not FIXTURE.is_file():
        print(f"[FAIL] negative fixture missing: {FIXTURE}")
        return 1

    with tempfile.TemporaryDirectory(prefix="capture_repro_neg_") as td:
        work = Path(td)
        repro = work / "neg-reproducer"

        # 1. the generator must survive the size-unknown bundle (no crash)
        emit = subprocess.run(
            [str(CLI), "emit-repro", str(FIXTURE),
             "--vendor", "amd", "--arch", "gfx1101",
             "--out", str(repro), "--force"],
            capture_output=True, text=True, cwd=work)
        print(emit.stdout, end="")
        print(emit.stderr, end="")
        check(emit.returncode == 0, "emit-repro exited 0 on size-unknown "
              "bundle (no crash)")

        main_src = repro / "main.hip"
        check(main_src.is_file(), "reproducer emitted")

        # 2. clearly-marked synthetic input + warning header (asserted)
        if main_src.is_file():
            text = main_src.read_text()
            check("KERNSCOPE_SYNTHETIC_INPUT" in text,
                  "source carries the KERNSCOPE_SYNTHETIC_INPUT marker")
            check("WARNING" in text,
                  "source carries a WARNING header")
            check("size-unknown" in text,
                  "warning names the size-unknown buffer explicitly")
            check("0xdead0000" in text,
                  "synthetic buffer identified by original VA")

        # 3. the synthetic-input reproducer COMPILES (hipcc, no GPU needed)
        build = subprocess.run(
            ["cmake", "-S", str(repro), "-B", str(repro / "build")],
            capture_output=True, text=True, cwd=work)
        check(build.returncode == 0, "reproducer cmake configure exit 0")
        build = subprocess.run(
            ["cmake", "--build", str(repro / "build")],
            capture_output=True, text=True, cwd=work)
        check(build.returncode == 0, "synthetic-input reproducer compiles "
              "(flagged, no crash)")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_repro_negative: size-unknown bundle -> synthetic "
          "input + warning header, compiles, no crash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
