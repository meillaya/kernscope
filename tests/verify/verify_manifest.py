#!/usr/bin/env python3
"""kernscope-verify manifest diff driver (todo 35).

Runs `kernscope-verify --json` over every corpus kernel named in the
consolidated tests/verify/manifest.json and DIFFS the emitted verdicts
against the manifest's {file, kernel, check, expected_verdict} entries.

Anti-self-report contract: the manifest is read LIVE from disk at test
time and the CLI never sees it — corrupting any expected_verdict makes
this driver fail NAMING the kernel (the QA negative gate). This is a
one-way diff: every manifest entry must be satisfied by the CLI output;
extra CLI verdicts (e.g. the SAFE fallback for classes a plugin covered
without findings) are allowed and expected.

Usage: verify_manifest.py --cli <kernscope-verify> --corpus <dir>
       --manifest <manifest.json>
Exit 0: every manifest entry matched (prints "manifest diff: EMPTY").
Exit 1: at least one mismatch, missing verdict, or failed ingestion
(prints per-kernel FAIL lines).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# CLI exit codes: 0 approved, 1 error-class verdict (still a valid verdict
# document on stdout), 2 ingestion failure (invalid for corpus fixtures).
_ACCEPTABLE_CLI_EXITS = (0, 1)


def run_cli(cli, src, corpus, out_base):
    """Runs the CLI on one corpus file, returns the parsed verdict document."""
    path = os.path.join(corpus, src)
    try:
        p = subprocess.run(
            [cli, path, "--json", "--out", out_base],
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"kernscope-verify timed out on {src}")
    if p.returncode not in _ACCEPTABLE_CLI_EXITS:
        raise RuntimeError(
            f"kernscope-verify exited {p.returncode} on {src} (ingestion "
            f"failed); stderr:\n{p.stderr[:2000]}"
        )
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"kernscope-verify emitted invalid JSON on {src}: {exc}\n"
            f"stdout:\n{p.stdout[:2000]}"
        )


def collect_verdicts(doc):
    """(kernel, check) -> verdict from the CLI's JSON document."""
    out = {}
    for k in doc.get("kernels", []):
        for v in k.get("verdicts", []):
            out[(k["kernel"], v["check"])] = v["verdict"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    with open(args.manifest) as fh:
        manifest = json.load(fh)

    by_file = {}
    for e in manifest:
        by_file.setdefault(e["file"], []).append(e)

    total = len(manifest)
    matched = 0
    failures = 0
    checked_kernels = 0
    out_dir = tempfile.mkdtemp(prefix="kernscope-verify-manifest-")
    try:
        for src in sorted(by_file):
            entries = by_file[src]
            out_base = os.path.join(out_dir, os.path.splitext(src)[0])
            doc = run_cli(args.cli, src, args.corpus, out_base)
            actual = collect_verdicts(doc)
            # Every kernel named in the manifest must exist in the CLI doc.
            emitted_kernels = {k["kernel"] for k in doc.get("kernels", [])}
            for e in entries:
                checked_kernels += 1
                if e["kernel"] not in emitted_kernels:
                    failures += 1
                    print(
                        f"FAIL: {src} {e['kernel']} check {e['check']}: kernel "
                        f"not found in CLI output"
                    )
                    continue
                got = actual.get((e["kernel"], e["check"]))
                if got is None:
                    failures += 1
                    print(
                        f"FAIL: {src} {e['kernel']} check {e['check']}: no verdict "
                        f"emitted (expected {e['expected_verdict']})"
                    )
                elif got == e["expected_verdict"]:
                    matched += 1
                    print(
                        f"diff: {src} {e['kernel']} check {e['check']}: "
                        f"{got} == {e['expected_verdict']} (match)"
                    )
                else:
                    failures += 1
                    print(
                        f"FAIL: {src} {e['kernel']} check {e['check']}: expected "
                        f"{e['expected_verdict']}, got {got}"
                    )
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)

    if failures:
        print(
            f"manifest diff: {failures} MISMATCH(ES) / {total} entries "
            f"({checked_kernels} checked across {len(by_file)} corpus files)"
        )
        return 1
    print(
        f"manifest diff: EMPTY ({matched}/{total} entries matched across "
        f"{len(by_file)} corpus files, {checked_kernels} kernels checked)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
