#!/usr/bin/env python3
"""profile_cli_amd_check.py — todo-52 e2e GOLDEN (local HIP) on gfx1101.

The Wave-9 closer acceptance, live against the todo-5 layernorm_fused
fixture:

  1. build the fixture app (hipcc --offload-arch=gfx1101) into
     build/tests/fixtures/layernorm_fused/layernorm_app — the CLI target
     (a directory; the CLI resolves layernorm_app inside);
  2. THE ACCEPTANCE COMMAND verbatim:
       kernscope-profile build/tests/fixtures/layernorm_fused --json
         | python3 tests/fixtures/check_profile.py
     must exit 0 — every numeric/range assertion in the golden (kernel name
     contains "layernorm", duration_ns > 0, occupancy in (0,1], limiter in
     {registers,shared,threads,blocks}, >= 1 compiler hint with evidence,
     grid/workgroup == the fixture launch contract, RANGES never absolutes);
  3. --text: the owner's Section B SHAPE (docs/research/owner-examples.md:
     Kernel:/Time:/Registers/thread:/Shared memory/block:/Likely limiter:/
     Compiler hint:) with LOCALLY-MEASURED values (no vendor claims);
  4. --html: libs/report's self-contained page — the embedded JSON payload
     (tests/report/check_html.py extraction contract) PARSES and names the
     layernorm kernel.

Usage: profile_cli_amd_check.py <build-dir> <profile-src-dir>
"""

import html.parser
import json
import shutil
import subprocess
import sys
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
CLI = BUILD_DIR / "bin" / "kernscope-profile"
FIXTURE_DIR = SRC_DIR.parent.parent / "tests" / "fixtures"
LAYERNORM_DIR = FIXTURE_DIR / "layernorm_fused"
APP_DIR = BUILD_DIR / "tests" / "fixtures" / "layernorm_fused"
APP = APP_DIR / "layernorm_app"
CHECK_PROFILE = FIXTURE_DIR / "check_profile.py"

SCRIPT_ID = "report-data"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


class PayloadExtractor(html.parser.HTMLParser):
    """Extract the <script id=report-data> payload (tests/report/check_html.py
    contract — byte-faithful, convert_charrefs=False)."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._depth = 0
        self._capture = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            return
        if self._capture:
            self._depth += 1
            return
        if dict(attrs).get("id") == SCRIPT_ID:
            self._capture = True
            self._depth = 1

    def handle_endtag(self, tag):
        if tag == "script" and self._capture:
            self._depth -= 1
            if self._depth == 0:
                self._capture = False

    def handle_data(self, data):
        if self._capture:
            self.parts.append(data)

    def payload(self):
        return "".join(self.parts) if self.parts else None


def main() -> int:
    if not CLI.is_file():
        print(f"[FAIL] CLI missing: {CLI} (run cmake --build)")
        return 1

    # 1. build the fixture app into the CLI's target directory
    APP_DIR.mkdir(parents=True, exist_ok=True)
    build = subprocess.run(
        ["hipcc", "--offload-arch=gfx1101", "-std=c++17", "-O3",
         str(LAYERNORM_DIR / "layernorm.hip"), "-o", str(APP)],
        capture_output=True, text=True)
    print(build.stderr, end="")
    check(build.returncode == 0, "layernorm_fused app built (hipcc gfx1101)")
    check(APP.is_file(), "app landed at build/tests/fixtures/layernorm_fused/")
    if build.returncode != 0:
        return 1

    # 2. THE ACCEPTANCE COMMAND: --json piped through the golden helper
    run = subprocess.run(
        [str(CLI), str(APP_DIR), "--json"],
        capture_output=True, text=True)
    print(run.stderr, end="")
    check(run.returncode == 0, "kernscope-profile --json exits 0")
    check(run.stdout, "JSON report produced on stdout")
    if run.returncode != 0 or not run.stdout:
        return 1

    golden = subprocess.run(
        [sys.executable, str(CHECK_PROFILE)],
        input=run.stdout, capture_output=True, text=True)
    print(golden.stdout, end="")
    print(golden.stderr, end="")
    check(golden.returncode == 0,
          "check_profile.py: all numeric/range assertions pass "
          "(acceptance command)")

    # 3. --text: owner Section B shape, locally-measured
    text = subprocess.run(
        [str(CLI), str(APP_DIR), "--text"],
        capture_output=True, text=True)
    print(text.stderr, end="")
    check(text.returncode == 0, "kernscope-profile --text exits 0")
    for field in ("Kernel: ", "Time: ", "Registers/thread: ",
                  "Shared memory/block: ", "Likely limiter: ",
                  "Compiler hint:"):
        check(field in text.stdout,
              f"--text carries the owner's Section B field '{field.strip()}'")
    check("layernorm" in text.stdout,
          "--text reports the layernorm kernel (locally-measured)")

    # 4. --html: libs/report page; embedded JSON payload parses
    html = subprocess.run(
        [str(CLI), str(APP_DIR), "--html"],
        capture_output=True, text=True)
    print(html.stderr, end="")
    check(html.returncode == 0, "kernscope-profile --html exits 0")
    parser = PayloadExtractor()
    parser.feed(html.stdout)
    payload = parser.payload()
    check(payload is not None,
          "HTML embeds the <script id=report-data> JSON payload")
    if payload is None:
        return 1
    try:
        data = json.loads(payload)
        payload_ok = True
        name = data.get("kernel", {}).get("name", "")
    except json.JSONDecodeError as e:
        payload_ok = False
        name = ""
        print(f"[FAIL] embedded payload is not valid JSON: {e}")
    check(payload_ok, "embedded JSON payload parses (libs/report html check)")
    check("layernorm" in name,
          f"payload names the layernorm kernel ({name})")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] profile_cli_amd: layernorm_fused profiled live on gfx1101; "
          "golden report matches expected RANGES (never absolutes), --text "
          "mirrors the owner's Section B shape with locally-measured values, "
          "--html embeds a parseable JSON payload")
    return 0


if __name__ == "__main__":
    sys.exit(main())
