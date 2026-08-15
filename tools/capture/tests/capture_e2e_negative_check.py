#!/usr/bin/env python3
"""capture_e2e_negative_check.py — todo-48 QA negative: replay with a
CORRUPTED input blob must produce a validation FAIL verdict with a hash
diff, exit 1, and never crash.

The committed todo-45/47 live fixture (layernorm-live) is COPIED to a temp
dir (never mutated in place), its first input blob is corrupted (one byte
flipped), and the replay is run against the captured reference output:
the launch still runs (the kernel is intact), the output differs, and the
bitwise validation reports the hash diff -> FAIL -> exit 1. A crash would
surface as a non-1 exit / signal; a missing verdict would fail the string
asserts.

Usage: capture_e2e_negative_check.py <build-dir> <capture-source-dir>
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE = SRC_DIR.parent.parent / "tests" / "fixtures" / "capture" / "layernorm-live"
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
        print(f"[FAIL] kernscope-capture CLI missing: {CLI}")
        return 1
    if not FIXTURE.is_dir():
        print(f"[FAIL] layernorm-live fixture missing: {FIXTURE}")
        return 1

    with tempfile.TemporaryDirectory(prefix="capture_e2e_neg_") as td:
        work = Path(td)
        bundle_dir = work / "layernorm-live"
        shutil.copytree(FIXTURE, bundle_dir)
        blobs = bundle_dir / "blobs"
        inputs = sorted(blobs.glob("input-buffer-*.bin"))
        check(len(inputs) >= 1, "fixture has input blobs")
        victim = inputs[0]
        data = bytearray(victim.read_bytes())
        check(len(data) > 0, "input blob non-empty")
        data[len(data) // 2] ^= 0xFF  # corrupt one byte in the middle
        victim.write_bytes(bytes(data))

        rp = subprocess.run(
            [str(CLI), "replay", str(bundle_dir / "bundle.v0"),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(blobs),
             "--ref", str(bundle_dir / "output.bin"),
             "--mode", "bitwise"],
            capture_output=True, text=True, cwd=work)
        print(rp.stdout, end="")
        print(rp.stderr, end="")
        check(rp.returncode == 1,
              f"replay on corrupted blob exited 1 (FAIL verdict, got "
              f"{rp.returncode})")
        check(rp.returncode not in (-11, -6, 139, 134),
              "no crash (no SIGSEGV/SIGABRT)")
        check("validation: FAIL" in rp.stdout, "validation FAIL reported")
        check("BITWISE DIFF" in rp.stdout, "bitwise diff reported")
        check("ref=" in rp.stdout and "replay=" in rp.stdout,
              "hash diff shown (ref sha != replay sha)")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_e2e_negative: corrupted blob -> FAIL verdict, "
          "hash diff, exit 1, no crash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
