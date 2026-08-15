#!/usr/bin/env python3
"""profile_cli_negative_check.py — todo-52 QA negative (no GPU kernels).

A target binary with NO GPU kernels must yield a valid "no kernels traced"
report with EXIT 0 — NOT an error. The target is a plain host-only C++
binary (no hip/cuda linkage, no kernel launches); the CLI still traces it
(the tracer is inert without a GPU runtime), sees zero kernel records and
reports the negative path.

Assertions:
  * kernscope-profile <plain-binary> --json exits 0;
  * the report JSON carries no_kernels_traced == true (and kernel_count == 0);
  * kernscope-profile <plain-binary> --text exits 0 and says exactly
    "no kernels traced".

Usage: profile_cli_negative_check.py <build-dir> <profile-src-dir>
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
CLI = BUILD_DIR / "bin" / "kernscope-profile"

PLAIN_APP_SRC = r"""
// no GPU kernels: plain host code only (todo-52 QA negative target)
#include <cstdio>
int main() {
    std::printf("plain host app: no GPU kernels\\n");
    return 0;
}
"""

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not CLI.is_file():
        print(f"[FAIL] CLI missing: {CLI} (run cmake --build)")
        return 1

    with tempfile.TemporaryDirectory(prefix="profile_cli_neg_") as td:
        work = Path(td)
        src = work / "plain_app.cpp"
        src.write_text(PLAIN_APP_SRC)
        binary = work / "plain_app"
        cc = subprocess.run(
            ["c++", "-std=c++17", str(src), "-o", str(binary)],
            capture_output=True, text=True)
        print(cc.stderr, end="")
        check(cc.returncode == 0, "plain (kernel-free) app compiled")
        check(binary.is_file(), "plain app binary exists")
        if cc.returncode != 0:
            return 1

        # JSON: report says "no kernels traced", exit 0 (NOT an error)
        run = subprocess.run(
            [str(CLI), str(binary), "--json"],
            capture_output=True, text=True)
        print(run.stderr, end="")
        check(run.returncode == 0,
              "kernscope-profile exits 0 on a kernel-free target (not an "
              "error)")
        parsed = None
        if run.stdout:
            try:
                parsed = json.loads(run.stdout)
            except json.JSONDecodeError as e:
                print(f"[FAIL] report JSON does not parse: {e}")
                failures.append("report JSON parses")
        check(parsed is not None, "report JSON parses")
        if parsed is not None:
            check(parsed.get("no_kernels_traced") is True,
                  "no_kernels_traced == true")
            check(parsed.get("kernel_count", -1) == 0,
                  "kernel_count == 0")

        # Text: the exact phrase, exit 0
        text = subprocess.run(
            [str(CLI), str(binary), "--text"],
            capture_output=True, text=True)
        print(text.stderr, end="")
        check(text.returncode == 0, "--text exits 0")
        check("no kernels traced" in text.stdout,
              "--text says \"no kernels traced\"")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] profile_cli_no_kernels: a kernel-free binary profiles as "
          "\"no kernels traced\" with exit 0 (the negative is a valid report, "
          "never an error)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
