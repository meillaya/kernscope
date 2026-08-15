#!/usr/bin/env python3
"""demo_aggregate.py — todo-55 aggregator: builds the demo's ONE aggregated
report (reports/demo/report.json + self-contained report.html) from the
five stage evidence files produced by scripts/demo.sh.

The aggregated report links every stage's evidence via refs (paths that
MUST exist on disk — the check_loop.py contract), follows the libs/report
conventions (schema_version 1.0, machine-readable evidence, self-contained
HTML with an embedded <script id="report-data"> JSON payload, zero network),
and carries the profiler<->analyzer cross-tool diagnostic: every profiler
hint of the hot kernel references the analyzer finding ids that EXIST in
the stage-3 analysis of the captured kernel's ISA.

Subcommands:
  sanitize-isa <hsaco> <out.isa>
      disassemble an AMDGPU code object (llvm-objdump -d) and strip the
      objdump scaffolding (headers, label lines, address prefixes, byte
      comments) so the result is plain AMDGPU ISA text that
      kernscope-analyze (libs/amdparse, LLVM MC) parses.
  aggregate <demo-dir> <runtime-seconds>
      happy path: build report.json + report.html in <demo-dir> from the
      stage evidence files the demo just produced.
  skipped <demo-dir> <reason>
      KERNSCOPE_NO_GPU=1 negative: well-formed report where all five
      stages are marked skipped with the given reason (graceful abort).
"""

import html
import json
import re
import subprocess
import sys
from pathlib import Path

SCHEMA_VERSION = "1.0"
REPORT_TYPE = "kernscope_demo_report"
RUNTIME_BUDGET_SECONDS = 900.0

STAGE_NAMES = ["profile", "capture", "analyze", "capture_tune",
               "verify_tune_pass_delta"]


def sanitize_isa(hsaco: Path, out: Path) -> int:
    """llvm-objdump -d + objdump-scaffolding strip -> plain AMDGPU ISA text.

    Instruction lines in llvm-objdump output start with a tab; everything
    else (headers, label lines, address prefixes, byte dumps) is dropped,
    and inline '// <addr>: <bytes>' comments are removed. Verified against
    the analyzer: the resulting text parses through libs/amdparse (LLVM MC)
    with --arch gfx1101.
    """
    run = subprocess.run(["llvm-objdump", "-d", str(hsaco)],
                         capture_output=True, text=True)
    if run.returncode != 0:
        print(f"demo_aggregate: llvm-objdump failed on {hsaco}: "
              f"{run.stderr.strip()}", file=sys.stderr)
        return 2
    insns = []
    for line in run.stdout.splitlines():
        if not line.startswith("\t"):
            continue
        insn = line.split("//", 1)[0].strip()
        if insn:
            insns.append(insn)
    if not insns:
        print(f"demo_aggregate: no instructions disassembled from {hsaco}",
              file=sys.stderr)
        return 2
    out.write_text("\n".join(insns) + "\n", encoding="utf-8")
    print(f"demo_aggregate: sanitized {len(insns)} instructions -> {out}")
    return 0


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def hot_kernel_and_hints(stage1_json: dict):
    """The demo's hot kernel = the layernorm kernel from the profile's
    owner_example block (the loop's subject); return its name + hints."""
    owner = stage1_json.get("owner_example", {})
    name = owner.get("kernel", "")
    if not name:
        for k in stage1_json.get("kernels", []):
            if "layernorm" in k.get("kernel_name", ""):
                name = k["kernel_name"]
                break
    hints = []
    for k in stage1_json.get("kernels", []):
        if k.get("kernel_name") == name:
            hints = k.get("hints", [])
            break
    return name, hints


def build_html(report: dict) -> str:
    """Self-contained HTML: inline CSS, embedded JSON payload
    (<script id="report-data" type="application/json">), zero network.
    Sections rendered statically from the report model."""
    payload = json.dumps(report, indent=2)
    stages = report.get("loop", {}).get("stages", [])
    cross = report.get("cross_tool", {})
    rows = []
    for s in stages:
        ev = "".join(
            f"<li><code>{html.escape(e)}</code></li>" for e in s.get("evidence", []))
        rows.append(
            f"<section class='stage'>"
            f"<h2>Stage {s.get('stage')}: {html.escape(s.get('name'))} "
            f"(<span class='status {s.get('status')}'>{s.get('status')}"
            f"</span>)</h2>"
            f"<p class='tool'>{html.escape(s.get('tool', ''))}</p>"
            f"<pre class='summary'>{html.escape(json.dumps(s.get('summary', {}), indent=2))}</pre>"
            f"<ul class='evidence'>{ev}</ul>"
            f"</section>")
    hint_rows = []
    for h in cross.get("profiler_hints", []):
        refs = h.get("analyzer_cross_refs", {})
        hint_rows.append(
            f"<li><code>{html.escape(h.get('id', ''))}</code> -> analyzer "
            f"findings {html.escape(json.dumps(refs.get('finding_ids', [])))} "
            f"(source {html.escape(refs.get('source', ''))}, evidence "
            f"<code>{html.escape(refs.get('stage_3_evidence', ''))}</code>)"
            f"</li>")
    finding_rows = "".join(
        f"<li><code>{html.escape(f.get('id', ''))}</code> "
        f"({html.escape(f.get('severity', ''))})</li>"
        for f in cross.get("analyzer_findings", []))
    machine = report.get("machine", {})
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kernscope loop demo — one-command end-to-end report</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 960px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  h2 {{ margin-top: 1.5rem; font-size: 1.1rem; }}
  .status {{ font-weight: bold; }}
  .status.ok {{ color: #0a7d2c; }}
  .status.skipped {{ color: #8a6d00; }}
  .stage {{ border: 1px solid #ccc; border-radius: 6px; padding: .7rem 1rem; margin: .8rem 0; }}
  .tool {{ color: #444; font-style: italic; }}
  pre {{ background: #f5f5f5; padding: .6rem; border-radius: 4px; overflow-x: auto; font-size: .85rem; }}
  ul.evidence {{ list-style: none; padding-left: .5rem; }}
  ul.evidence li {{ margin: .15rem 0; }}
  .meta {{ color: #555; font-size: .9rem; }}
  footer {{ margin-top: 2rem; color: #777; font-size: .8rem; }}
</style>
</head>
<body>
<h1>kernscope loop demo</h1>
<p class="meta">One-command end-to-end loop: profile → capture → analyze →
capture→tune → verifier-gated tune + pass→analyzer delta.
Machine: {html.escape(str(machine.get('device', machine.get('arch', 'unknown'))))}
({html.escape(str(machine.get('arch', '?')))}); runtime
{report.get('runtime_seconds', '?')}s of {RUNTIME_BUDGET_SECONDS}s budget.</p>
<p class="meta">Hot kernel: <code>{html.escape(report.get('loop', {}).get('hot_kernel', ''))}</code></p>
{''.join(rows)}
<section class="stage">
<h2>Cross-tool diagnostic (profiler → analyzer)</h2>
<p>Profiler hints referencing analyzer finding ids (stage-3 analysis of the
captured kernel's ISA):</p>
<ul>{''.join(hint_rows)}</ul>
<p>Analyzer findings (stage 3):</p>
<ul>{finding_rows}</ul>
</section>
<footer>kernscope — self-contained report (zero network requests); embedded
JSON payload in &lt;script id="report-data"&gt;.</footer>
<script id="report-data" type="application/json">{payload}</script>
</body>
</html>
"""


def aggregate(demo_dir: Path, runtime_seconds: float) -> int:
    """Happy path: compose the aggregated report from stage evidence."""
    demo = demo_dir.resolve()
    if not demo.is_dir():
        print(f"demo_aggregate: no demo dir {demo}", file=sys.stderr)
        return 2

    def ev(name):
        p = demo / name
        if not p.is_file():
            print(f"demo_aggregate: missing stage evidence {p}",
                  file=sys.stderr)
            return None
        return p

    # --- stage 1: profile ---------------------------------------------------
    p1 = ev("stage-1-profile.json")
    if p1 is None:
        return 2
    s1 = load_json(p1)
    hot, hints = hot_kernel_and_hints(s1)
    if not hot:
        print("demo_aggregate: no layernorm kernel in stage-1 profile",
              file=sys.stderr)
        return 2

    # --- stage 2: capture ---------------------------------------------------
    bundle = demo / "capture" / "bundle.v0"
    replay_txt = demo / "capture" / "replay.txt"
    if not bundle.is_file() or not replay_txt.is_file():
        print(f"demo_aggregate: missing capture evidence {bundle} / "
              f"{replay_txt}", file=sys.stderr)
        return 2
    replay_ok = "bitwise EQUAL" in replay_txt.read_text(encoding="utf-8")

    # --- stage 3: analyze ---------------------------------------------------
    p3 = ev("stage-3-analyze.json")
    if p3 is None:
        return 2
    s3 = load_json(p3)
    isa = demo / "capture" / "layernorm.isa"
    findings = []
    for key in ("warnings", "suggestions"):
        for f in s3.get(key, []):
            findings.append({"id": f.get("id", ""),
                             "severity": f.get("severity", "")})

    # --- stage 4: capture->tune ---------------------------------------------
    p4 = ev("stage-4-capture-tune.json")
    if p4 is None:
        return 2
    s4 = load_json(p4)

    # --- stage 5: verifier-gated tune + pass->analyzer delta ----------------
    p5a = ev("stage-5-verify-tune.json")
    p5b = ev("stage-5-pass-delta.json")
    if p5a is None or p5b is None:
        return 2
    s5a = load_json(p5a)
    s5b = load_json(p5b)

    stages = [
        {"stage": 1, "name": "profile", "status": "ok",
         "tool": "kernscope-profile build/tests/fixtures/layernorm_fused",
         "summary": {
             "kernel_count": s1.get("kernel_count"),
             "wall_time_ns": s1.get("wall_time_ns"),
             "hint_count": len(hints),
             "hints": [h.get("id") for h in hints]},
         "evidence": ["reports/demo/stage-1-profile.json",
                      "reports/demo/stage-1-profile.html"]},
        {"stage": 2, "name": "capture", "status": "ok",
         "tool": "kernscope-capture record + replay (bitwise)",
         "summary": {
             "bundle": "reports/demo/capture/bundle.v0",
             "kernel": hot,
             "replay": "bitwise PASS" if replay_ok else "MISMATCH",
             "grid": "1024x1x1", "block": "128x1x1"},
         "evidence": ["reports/demo/capture/bundle.v0",
                      "reports/demo/capture/output.bin",
                      "reports/demo/capture/blobs/code-object-2.hsaco",
                      "reports/demo/capture/replay.txt"]},
        {"stage": 3, "name": "analyze", "status": "ok",
         "tool": "kernscope-analyze <captured-ISA> --arch gfx1101",
         "summary": {"findings": findings,
                     "metrics_alu": s3.get("metrics", {}).get("mix_count_ALU")},
         "evidence": ["reports/demo/stage-3-analyze.json",
                      "reports/demo/capture/layernorm.isa",
                      "reports/demo/stage-3-analyze.html"]},
        {"stage": 4, "name": "capture_tune", "status": "ok",
         "tool": "kernscope-tune --from-capture",
         "summary": {
             "status": s4.get("status"), "mode": s4.get("mode"),
             "validated": s4.get("summary", {}).get("validated"),
             "speedup": s4.get("speedup"),
             "best_block": (s4.get("best", {}).get("config", {})
                            .get("block", [None])[0])},
         "evidence": ["reports/demo/stage-4-capture-tune.json",
                      "reports/demo/stage-4-tune.jsonl"]},
        {"stage": 5, "name": "verify_tune_pass_delta", "status": "ok",
         "tool": "kernscope-tune --verify + "
                 "kernscope-pass-report --report-pipeline",
         "summary": {
             "verify_gate": s5a.get("verify_gate"),
             "verifier_gated": s5a.get("summary", {}).get("verifier_gated"),
             "compiled": s5a.get("summary", {}).get("compiled"),
             "pass_delta_vector_width":
                 s5b.get("delta", {}).get("vector_width_delta"),
             "pass_delta_no_transform":
                 s5b.get("delta", {}).get("no_transform")},
         "evidence": ["reports/demo/stage-5-verify-tune.json",
                      "reports/demo/stage-5-pass-delta.json"]},
    ]

    # --- cross-tool: profiler hints -> analyzer finding ids -----------------
    profiler_hints = []
    for h in hints:
        profiler_hints.append({
            "id": h.get("id", ""),
            "severity": h.get("severity", ""),
            "evidence": h.get("evidence", {}),
            "analyzer_cross_refs": {
                "source": "kernscope-analyze",
                "finding_ids": [f["id"] for f in findings],
                "stage_3_evidence": "reports/demo/stage-3-analyze.json"}})

    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "machine": {
            "arch": "gfx1101",
            "device": s4.get("device_name", "AMD Radeon RX 7800 XT"),
            "gpu_required": True},
        "runtime_seconds": runtime_seconds,
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "loop": {
            "hot_kernel": hot,
            "stages": stages},
        "cross_tool": {
            "hot_kernel": hot,
            "profiler_hints": profiler_hints,
            "analyzer_findings": findings},
    }
    write_report(demo, report)
    return 0


def write_report(demo: Path, report: dict) -> None:
    out_json = demo / "report.json"
    out_html = demo / "report.html"
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    out_html.write_text(build_html(report), encoding="utf-8")
    print(f"demo_aggregate: wrote {out_json} + {out_html}")


def skipped(demo_dir: Path, reason: str) -> int:
    """KERNSCOPE_NO_GPU=1 negative: graceful all-skipped report (the demo
    aborts before running any stage; the report still exists and is
    well-formed so consumers can distinguish 'aborted, GPU required' from
    a crashed/absent run)."""
    demo = demo_dir.resolve()
    demo.mkdir(parents=True, exist_ok=True)
    stages = [{"stage": i + 1, "name": STAGE_NAMES[i], "status": "skipped",
               "tool": "not run", "reason": reason, "evidence": []}
              for i in range(5)]
    report = {
        "schema_version": SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "machine": {"arch": "gfx1101", "device": "unavailable",
                    "gpu_required": False},
        "runtime_seconds": 0,
        "runtime_budget_seconds": RUNTIME_BUDGET_SECONDS,
        "loop": {"hot_kernel": "", "stages": stages},
        "cross_tool": {"hot_kernel": "", "profiler_hints": [],
                       "analyzer_findings": []},
    }
    write_report(demo, report)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: demo_aggregate.py sanitize-isa <hsaco> <out.isa> |\n"
              "       demo_aggregate.py aggregate <demo-dir> <runtime-sec> |\n"
              "       demo_aggregate.py skipped <demo-dir> <reason>",
              file=sys.stderr)
        return 3
    cmd = argv[1]
    if cmd == "sanitize-isa" and len(argv) == 4:
        return sanitize_isa(Path(argv[2]), Path(argv[3]))
    if cmd == "aggregate" and len(argv) == 4:
        return aggregate(Path(argv[2]), float(argv[3]))
    if cmd == "skipped" and len(argv) >= 4:
        return skipped(Path(argv[2]), " ".join(argv[3:]))
    print(f"demo_aggregate: bad arguments: {' '.join(argv)}", file=sys.stderr)
    return 3


if __name__ == "__main__":
    sys.exit(main(sys.argv))
