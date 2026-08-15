#!/usr/bin/env python3
"""check_loop.py — todo-55 integration helper: asserts the one-command
end-to-end loop demo's aggregated report (reports/demo/report.json).

The demo (scripts/demo.sh) runs the five stage CLIs on this machine
(profile -> capture -> analyze -> capture->tune -> verifier-gated tune +
pass->analyzer delta) and aggregates ONE self-contained report
(report.json + report.html, libs/report conventions: schema_version 1.0,
embedded-JSON HTML payload). This script is the demo's machine-checkable
contract:

Happy path (default):
  1. the aggregated JSON parses, report_type == "kernscope_demo_report",
     schema_version == "1.0";
  2. runtime_seconds < 900 (the asserted < 15 min budget on this machine);
  3. exactly FIVE stage entries, each status == "ok", each with >= 1
     evidence path that EXISTS on disk (the acceptance's "evidence paths
     EXISTING on disk" assertion);
  4. the hot kernel is identified and matches stage-1's profile evidence;
  5. the profiler<->analyzer cross-tool diagnostic: every profiler hint
     carried for the hot kernel references analyzer finding ids
     (analyzer_cross_refs.finding_ids) that all EXIST in the stage-3
     analyzer report, and the stage-3 evidence path exists on disk —
     "profiler hints must reference analyzer finding ids where PTX/ISA
     exists", asserted in the demo JSON;
  6. stage-4 evidence reports capture_tune mode with status ok;
  7. stage-5 evidence carries the verifier gate (verify_gate true) and the
     pass->analyzer delta fields;
  8. report.html exists and its embedded <script id="report-data"> payload
     round-trips byte-exact to report.json (tests/report/check_html.py
     extraction contract).

QA negative (--allow-skipped): the KERNSCOPE_NO_GPU=1 run aborts early and
emits a well-formed report where ALL five stages are status "skipped" with
reason mentioning "gpu-amd required" (graceful degradation, never a crash);
the script then exits 0 too, so the negative contract is itself asserted.

Usage:
    python3 tests/integration/check_loop.py [--allow-skipped] [<report.json>]
    <report.json> defaults to <repo-root>/reports/demo/report.json.
"""

import argparse
import hashlib
import html.parser
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = REPO_ROOT / "reports" / "demo" / "report.json"

RUNTIME_BUDGET_SECONDS = 900.0  # the asserted < 15 min budget
FIVE_STAGE_NAMES = ["profile", "capture", "analyze", "capture_tune",
                    "verify_tune_pass_delta"]
SCRIPT_ID = "report-data"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def exists_on_disk(path_str: str) -> bool:
    """Evidence paths are repo-root-relative; refuse empty paths."""
    if not path_str:
        return False
    p = Path(path_str)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.is_file()


class PayloadExtractor(html.parser.HTMLParser):
    """Extract <script id=report-data> text (check_html.py contract:
    convert_charrefs=False keeps the payload byte-faithful)."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._depth = 0
        self._capture = False
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag != "script":
            return
        if self._capture:
            self._depth += 1
            return
        attrs = dict(attrs)
        if attrs.get("id") == SCRIPT_ID:
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

    def payload(self) -> str:
        return "".join(self.parts)


def extract_payload(html_path: Path) -> str:
    with html_path.open(encoding="utf-8") as f:
        parser = PayloadExtractor()
        parser.feed(f.read())
    return parser.payload()


def assert_happy(report: dict, report_path: Path) -> None:
    check(report.get("report_type") == "kernscope_demo_report",
          "aggregated report_type == kernscope_demo_report")
    check(report.get("schema_version") == "1.0",
          "schema_version == 1.0 (libs/report convention)")

    runtime = report.get("runtime_seconds")
    check(isinstance(runtime, (int, float)) and runtime >= 0,
          f"runtime_seconds present ({runtime})")
    check(isinstance(runtime, (int, float)) and
          runtime < RUNTIME_BUDGET_SECONDS,
          f"demo runtime {runtime}s < {RUNTIME_BUDGET_SECONDS}s "
          f"(< 15 min budget, asserted)")

    stages = report.get("loop", {}).get("stages")
    check(isinstance(stages, list) and len(stages) == 5,
          f"exactly five stage entries (got {len(stages) if stages else 0})")
    stage_names = [s.get("name") for s in (stages or [])]
    check(stage_names == FIVE_STAGE_NAMES,
          f"stage names/order == {FIVE_STAGE_NAMES} "
          f"(got {stage_names})")
    for s in (stages or []):
        check(s.get("status") == "ok",
              f"stage {s.get('stage')} ({s.get('name')}): status ok")
        ev = s.get("evidence", [])
        check(isinstance(ev, list) and len(ev) >= 1,
              f"stage {s.get('stage')} ({s.get('name')}): >= 1 evidence path")
        for e in ev:
            check(exists_on_disk(e),
                  f"stage {s.get('stage')} ({s.get('name')}): evidence "
                  f"EXISTS on disk: {e}")

    # --- hot kernel + profiler<->analyzer cross-reference -------------------
    hot = report.get("loop", {}).get("hot_kernel", "")
    check(isinstance(hot, str) and "layernorm" in hot,
          f"hot kernel identified and is the layernorm kernel "
          f"(got {hot!r})")
    stage1 = (stages or [])[0]
    stage1_evidence = stage1.get("evidence", [])
    prof = None
    if stage1_evidence:
        p = REPO_ROOT / stage1_evidence[0]
        if p.is_file():
            try:
                prof = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                prof = None
    check(prof is not None and prof.get("report_type") ==
          "kernscope_profile_report",
          "stage-1 evidence parses as a kernscope profile report")
    if prof:
        owner = prof.get("owner_example", {})
        check(owner.get("kernel", "") == hot,
              "stage-1 owner_example kernel matches the demo hot kernel")

    cross = report.get("cross_tool", {})
    check(isinstance(cross, dict) and cross.get("hot_kernel") == hot,
          "cross_tool section names the same hot kernel")
    hints = cross.get("profiler_hints", [])
    check(isinstance(hints, list) and len(hints) >= 1,
          f"cross_tool carries >= 1 profiler hint "
          f"(got {len(hints)})")
    findings = cross.get("analyzer_findings", [])
    finding_ids = {f.get("id") for f in findings}
    check(len(finding_ids) >= 1,
          f"cross_tool carries the analyzer findings "
          f"(got {len(finding_ids)} ids)")
    for h in hints:
        refs = h.get("analyzer_cross_refs", {})
        check(refs.get("source") == "kernscope-analyze",
              f"hint {h.get('id')}: cross-ref source == kernscope-analyze")
        ids = refs.get("finding_ids", [])
        check(isinstance(ids, list) and len(ids) >= 1 and
              all(i in finding_ids for i in ids),
              f"hint {h.get('id')}: finding_ids all EXIST in the stage-3 "
              f"analyzer report ({ids})")
        check(exists_on_disk(refs.get("stage_3_evidence", "")),
              f"hint {h.get('id')}: stage-3 evidence path exists on disk: "
              f"{refs.get('stage_3_evidence')}")

    # --- stage-level spot checks (the CLIs' own suites own deep checks) -----
    stage4 = (stages or [])[3]
    s4 = _load_first_evidence(stage4)
    check(s4 is not None and s4.get("status") == "ok" and
          s4.get("mode") == "capture_tune",
          "stage-4 evidence: capture_tune status ok")
    if s4:
        check(s4.get("candidates") and
              s4["candidates"][0].get("config", {}).get("is_baseline") is True,
              "stage-4 evidence: captured baseline explored first")

    stage5 = (stages or [])[4]
    s5a, s5b = _load_all_evidence(stage5)
    check(s5a is not None and s5a.get("verify_gate") is True,
          "stage-5a evidence: verifier gate ran (verify_gate true)")
    check(s5b is not None and s5b.get("report_type") == "kernscope_report" and
          "delta" in s5b,
          "stage-5b evidence: pass->analyzer delta report present")

    # --- self-contained HTML with an embedded, byte-exact payload -----------
    html_path = report_path.with_suffix(".html")
    check(html_path.is_file(), f"report.html exists: {html_path}")
    if html_path.is_file():
        payload = extract_payload(html_path)
        round_trip = False
        try:
            round_trip = json.loads(payload) == report
        except json.JSONDecodeError:
            round_trip = False
        check(round_trip,
              "report.html embedded JSON payload round-trips to report.json "
              "(check_html.py contract)")


def assert_skipped(report: dict) -> None:
    check(report.get("report_type") == "kernscope_demo_report",
          "negative report_type == kernscope_demo_report")
    check(report.get("gpu_required") is False,
          "negative report marks gpu_required false")
    check(report.get("runtime_seconds") == 0,
          "negative report runtime_seconds == 0 (aborted before stages)")
    stages = report.get("loop", {}).get("stages", [])
    check(len(stages) == 5, "negative: all five stage entries present")
    for s in stages:
        check(s.get("status") == "skipped" and
              "gpu-amd required" in s.get("reason", ""),
              f"stage {s.get('stage')} ({s.get('name')}): skipped with "
              f"'gpu-amd required' reason")
        ev = s.get("evidence", [])
        check(ev == [], f"stage {s.get('stage')}: no evidence in skip mode")
    check(report.get("cross_tool", {}).get("profiler_hints") == [],
          "negative: no fabricated cross-tool hints")


def _load_first_evidence(stage: dict):
    ev = stage.get("evidence", [])
    if not ev:
        return None
    p = REPO_ROOT / ev[0]
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _load_all_evidence(stage: dict) -> tuple:
    """Stage 5 has two JSON artifacts (verify-tune, pass-delta)."""
    out = []
    for e in stage.get("evidence", []):
        if not e.endswith(".json"):
            continue
        p = REPO_ROOT / e
        if p.is_file():
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                out.append(None)
        else:
            out.append(None)
    while len(out) < 2:
        out.append(None)
    return out[0], out[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-skipped", action="store_true",
                    help="assert the KERNSCOPE_NO_GPU=1 graceful-skip "
                         "negative contract instead of the happy path")
    ap.add_argument("report", nargs="?", default=str(DEFAULT_REPORT))
    args = ap.parse_args()

    report_path = Path(args.report).resolve()
    if not report_path.is_file():
        print(f"[FAIL] aggregated report not found: {report_path} "
              f"(run bash scripts/demo.sh first)")
        return 1
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"[FAIL] aggregated report is not valid JSON: {e}")
        return 1

    if args.allow_skipped:
        assert_skipped(report)
    else:
        assert_happy(report, report_path)

    if failures:
        print(f"\n{len(failures)} check(s) FAILED")
        return 1
    mode = "skipped-negative" if args.allow_skipped else "happy"
    print(f"\n[PASS] check_loop: all assertions green ({mode} mode)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
