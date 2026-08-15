#!/usr/bin/env python3
"""check_dashboard.py — todo-56 validator for the kernscope dashboard HTML.

Reuses the todo-7 validation discipline (tests/report/check_html.py): the
embedded JSON payload in <script id="report-data" type="application/json">
is extracted with python3's html.parser (assertion discipline: never
"renders correctly" — the payload must parse and carry the asserted model).

Asserts (happy path):
  1. the embedded payload is valid JSON and parses;
  2. ALL SIX module sections are present in the model
     (analyzer / tune / verify / capture / profile / pass);
  3. each section carries a provenance object with a boolean `measured`
     field (the projected-vs-measured honesty discipline — projected
     numbers are never presented as measured);
  4. each available section's evidence `files` exist on disk;
  5. every available section is rendered in the HTML (id="module-<name>");
  6. the loop diagram marker is present;
  7. the HTML is self-contained: no <script src=, no <link, no <img tags,
     no fetch() calls (zero network / external assets).

QA negative path (--negative): at least one section has available == false,
each such section is rendered with data-available="false" and a "no data"
panel (never a broken page), and the embedded payload still parses.

Usage:
    python3 check_dashboard.py <dashboard.html> [--negative]
Exit codes: 0 = all assertions pass, 1 = assertion failure, 2 = IO/parse
failure.
"""

import html.parser
import json
import os
import sys

SCRIPT_ID = "report-data"
SECTION_ORDER = ["analyzer", "tune", "verify", "capture", "profile", "pass"]


class PayloadExtractor(html.parser.HTMLParser):
    """Collect the text content of the <script id=SCRIPT_ID> element
    byte-faithfully (convert_charrefs=False), same contract as
    check_html.py."""

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self._depth = 0
        self._capture = False
        self.parts = []
        self.external_refs = []

    def handle_starttag(self, tag, attrs):
        if tag != "script" and tag not in ("link", "img"):
            return
        attrs = dict(attrs)
        if tag == "script" and attrs.get("id") == SCRIPT_ID:
            self._capture = True
            self._depth = 1
            return
        if self._capture:
            self._depth += 1
            return
        if tag == "script" and "src" in attrs:
            self.external_refs.append("script src=" + attrs["src"])
        if tag == "link":
            self.external_refs.append("link href=" + attrs.get("href", "?"))
        if tag == "img":
            self.external_refs.append("img src=" + attrs.get("src", "?"))

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


def extract(html_path):
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    parser = PayloadExtractor()
    parser.feed(content)
    return parser.payload(), parser.external_refs, content


def repo_root():
    """Walk up from the CWD to the repo root (marker:
    tests/verify/manifest.json), so evidence-path assertions hold no matter
    the CWD (ctest runs with CWD = build dir)."""
    cur = os.getcwd()
    for cand in [cur, *iter_parents(cur)]:
        if os.path.exists(os.path.join(cand, "tests", "verify", "manifest.json")):
            return cand
    return cur


def iter_parents(path):
    while True:
        parent = os.path.dirname(path)
        if parent == path:
            return
        yield parent
        path = parent


def exists_on_disk(f, root):
    if os.path.isabs(f):
        return os.path.exists(f)
    return os.path.exists(os.path.join(root, f)) or os.path.exists(f)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    negative = "--negative" in argv
    if negative:
        argv.remove("--negative")
    if len(argv) != 1:
        print("usage: check_dashboard.py <dashboard.html> [--negative]")
        return 2

    html_path = argv[0]
    try:
        payload, external_refs, content = extract(html_path)
    except OSError as e:
        print(f"check_dashboard: cannot read {html_path}: {e}")
        return 2
    if payload is None:
        return fail(f"no <script id={SCRIPT_ID!r}> payload found in {html_path}")

    try:
        model = json.loads(payload)
    except json.JSONDecodeError as e:
        return fail(f"embedded payload is not valid JSON: {e}")

    sections = model.get("sections")
    if not isinstance(sections, dict):
        return fail("model has no 'sections' object")

    # --- honesty discipline: six sections, each with provenance.measured ---
    missing_secs = [n for n in SECTION_ORDER if n not in sections]
    if missing_secs:
        return fail("missing module section(s) in embedded model: "
                    + ", ".join(missing_secs))

    for name in SECTION_ORDER:
        sec = sections[name]
        prov = sec.get("provenance")
        if not isinstance(prov, dict) or "measured" not in prov:
            return fail(f"section {name}: no provenance.measured field "
                        f"(honesty discipline)")
        if not isinstance(prov["measured"], bool):
            return fail(f"section {name}: provenance.measured is not a bool")
        if prov.get("mode") not in ("measured", "projected", "no_data"):
            return fail(f"section {name}: bad provenance.mode "
                        f"{prov.get('mode')!r}")
        if sec.get("available") and sec.get("status") == "no_data":
            return fail(f"section {name}: available but status no_data")
        if not sec.get("available") and sec.get("status") != "no_data":
            return fail(f"section {name}: unavailable but status "
                        f"{sec.get('status')!r}")

    # --- evidence files exist on disk (for available sections) ------------
    if not negative:
        root = repo_root()
        for name in SECTION_ORDER:
            sec = sections[name]
            if not sec.get("available"):
                return fail(f"section {name}: unexpectedly unavailable "
                            f"(not in --negative mode)")
            for f in sec.get("files", []):
                if not exists_on_disk(f, root):
                    return fail(f"section {name}: evidence file missing: {f}")

    no_data_secs = [n for n in SECTION_ORDER
                    if not sections[n].get("available")]
    if negative and not no_data_secs:
        return fail("--negative mode: expected >= 1 section with "
                    "available == false")

    # --- HTML rendering checks ----------------------------------------------
    for name in SECTION_ORDER:
        marker = f"id='module-{name}'"
        if marker not in content:
            return fail(f"HTML does not render section {name} "
                        f"(missing {marker})")
    if "loop-diagram" not in content:
        return fail("HTML is missing the loop diagram (loop-diagram marker)")

    for name in no_data_secs:
        marker = f"id='module-{name}' data-available='false'"
        if marker not in content:
            return fail(f"no-data section {name}: missing "
                        f"data-available='false' marker")
        if "no data" not in content:
            return fail(f"no-data section {name}: 'no data' text not rendered")
        if f"id='module-{name}' data-available='true'" in content:
            return fail(f"no-data section {name}: contradictory "
                        f"data-available='true'")

    # --- zero external assets ------------------------------------------------
    if external_refs:
        return fail("external asset reference(s) found: "
                    + ", ".join(external_refs))
    if "fetch(" in content:
        return fail("fetch() call found — dashboard must be zero-network")
    if "<script src=" in content or "<link " in content or "<img " in content:
        return fail("external script/link/img tag found")

    mode = "negative (no-data)" if negative else "happy"
    print(f"check_dashboard: PASS ({mode}) — {len(SECTION_ORDER)} module "
          f"sections, provenance.measured on each, embedded JSON parses, "
          f"zero external assets, loop diagram present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
