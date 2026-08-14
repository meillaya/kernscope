#!/usr/bin/env python3
"""Extract the embedded JSON payload from a self-contained kernscope HTML
report and compare it against the expected input JSON.

The HTML reports produced by libs/report (and reused by todo 56's dashboard)
embed the full report model in a <script id="report-data" type="application/json">
tag with zero network requests; this checker parses the HTML with
python3's html.parser (assertion discipline: never "renders correctly" —
the embedded payload must round-trip exactly).

Usage:
    python3 check_html.py <report.html> [<expected.json>]
The expected JSON defaults to a sibling file with the same basename and a
.json extension (e.g. sample.html -> sample.json), which is what the ctest
gate uses.

Exit codes: 0 = payload extracted and matches expected JSON,
            1 = mismatch, 2 = extraction/IO failure.
"""

import html.parser
import json
import os
import sys

SCRIPT_ID = "report-data"


class PayloadExtractor(html.parser.HTMLParser):
    """Collect the text content of the <script id=SCRIPT_ID> element.

    convert_charrefs=False keeps the payload byte-faithful: inside <script>
    browsers do not decode entities, and JSON escapes with \\uXXXX anyway.
    """

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

    def payload(self):
        return "".join(self.parts) if self.parts else None


def extract_payload(html_path):
    with open(html_path, encoding="utf-8") as f:
        content = f.read()
    parser = PayloadExtractor()
    parser.feed(content)
    return parser.payload()


def first_diff(expected, actual):
    """Return a human-readable description of the first difference."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                return "key %r present in payload but not expected" % key
            if key not in actual:
                return "key %r missing from payload" % key
            d = first_diff(expected[key], actual[key])
            if d:
                return "key %r: %s" % (key, d)
        return None
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return "array length %d != expected %d" % (len(actual), len(expected))
        for i, (e, a) in enumerate(zip(expected, actual)):
            d = first_diff(e, a)
            if d:
                return "array[%d]: %s" % (i, d)
        return None
    if expected != actual:
        return "value %r != expected %r" % (actual, expected)
    return None


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        sys.stderr.write(
            "usage: check_html.py <report.html> [<expected.json>]\n")
        return 2

    html_path = sys.argv[1]
    if len(sys.argv) == 3:
        expected_path = sys.argv[2]
    else:
        expected_path = os.path.splitext(html_path)[0] + ".json"

    try:
        payload = extract_payload(html_path)
    except OSError as e:
        sys.stderr.write("check_html: cannot read %s: %s\n" % (html_path, e))
        return 2
    if payload is None:
        sys.stderr.write(
            "check_html: no <script id=%r> payload found in %s\n"
            % (SCRIPT_ID, html_path))
        return 2

    try:
        actual = json.loads(payload)
    except json.JSONDecodeError as e:
        sys.stderr.write("check_html: embedded payload is not valid JSON: %s\n" % e)
        return 2

    try:
        with open(expected_path, encoding="utf-8") as f:
            expected = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        sys.stderr.write("check_html: cannot load expected %s: %s\n"
                         % (expected_path, e))
        return 2

    diff = first_diff(expected, actual)
    if diff is not None:
        sys.stderr.write("check_html: MISMATCH: %s\n" % diff)
        return 1

    print("check_html: %s embedded payload matches %s"
          % (html_path, expected_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
