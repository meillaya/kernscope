#!/usr/bin/env python3
"""capture_hip_negative_check.py — todo-45 QA negative: an app that allocates
via an UNSUPPORTED API (hipMallocPitch) must be captured GRACEFULLY:

  * the app runs unmodified and produces correct output (no crash),
  * the bundle marks the pitch buffer "size-unknown" (copy dst-size=0),
  * the capture logs a warning (stderr), exit 0.

Usage: capture_hip_negative_check.py <build-dir>
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
APP = Path(sys.argv[2]).resolve() / "capture_hip_negative_app"
TOOL = BUILD_DIR / "libkernscope_capture_hip.so"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not APP.is_file():
        print(f"[FAIL] negative app missing: {APP}")
        return 1
    if not TOOL.is_file():
        print(f"[FAIL] tool library missing: {TOOL}")
        return 1

    with tempfile.TemporaryDirectory(prefix="capture_hip_neg_") as td:
        work = Path(td)
        bundle = work / "capture.bundle"
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(TOOL)
        env["KERNSCOPE_HIP_BUNDLE"] = str(bundle)
        env["KERNSCOPE_HIP_BLOB_DIR"] = str(work / "blobs")
        (work / "blobs").mkdir()
        run = subprocess.run([str(APP)], env=env, capture_output=True,
                             text=True, cwd=work)
        print(run.stderr, end="")

        # 1. graceful: app exited 0 with correct output
        check(run.returncode == 0, "app exit 0 (graceful, no crash)")
        check("neg: output correct" in run.stderr, "app output correct")

        # 2. warning logged for the size-unknown buffer
        check("size-unknown" in run.stderr.lower() or "unknown" in run.stderr.lower(),
              "capture logs a warning for the size-unknown buffer")

        # 3. bundle marks the buffer size-unknown: the H2D copy into the
        #    pitch buffer has dst-size=0 (the v0 size-unknown convention)
        if not bundle.is_file():
            print("[FAIL] bundle missing")
            return 1
        lines = bundle.read_text().splitlines()
        copies = [l for l in lines if l.startswith("copy ")]
        h2d = [c for c in copies if "cuda-kind=1" in c]
        check(len(h2d) == 1, "one H2D copy record")
        if h2d:
            m = re.search(r"dst-size=(\d+)", h2d[0])
            check(m is not None and int(m.group(1)) == 0,
                  f"H2D copy dst-size=0 (size-unknown mark, got "
                  f"{m.group(1) if m else None})")
            check("bytes=" in h2d[0], "copy byte count still recorded")

        # 4. bundle complete
        check(lines[-1] == "end-capture-v0", "bundle complete")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_hip_negative: size-unknown marked + warning, "
          "no crash")
    return 0


if __name__ == "__main__":
    sys.exit(main())
