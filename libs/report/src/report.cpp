//===- report.cpp - Structured kernel report model + renderers ----*- C++ -*-===//
//
// Implementation of the kernscope report engine (libs/report, wave-1 todo 7):
// one model (KernelReport/Finding), three renderers (text in the owner's
// Section A format, JSON validated against docs/report/kernscope-technical-report.pdf,
// and a self-contained HTML with an embedded JSON payload). Every renderer
// validates the model first and throws SchemaViolation instead of silently
// emitting an invalid report.
//
//===----------------------------------------------------------------------===//

#include "kernscope/report.h"

#include <algorithm>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string_view>

namespace kernscope {

namespace {

constexpr const char *kJsonPayloadTag =
    "<script id=\"report-data\" type=\"application/json\">";

std::string SchemaError(const std::string &message) {
  return "report schema violation: " + message;
}

/// Quote + escape a string as a JSON string literal.
std::string JsonEscape(std::string_view s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        char buf[8];
        std::snprintf(buf, sizeof(buf), "\\u%04x",
                      static_cast<unsigned>(c));
        out += buf;
      } else {
        out += c;
      }
    }
  }
  return out;
}

/// Compact double rendering for JSON (up to 10 significant digits).
/// 0.375 -> "0.375", 112 -> "112", 106954752 -> "106954752".
std::string FmtNumber(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", value);
  return std::string(buf);
}

/// Percent rendering for the text renderer's Occupancy line:
/// 0.375 -> "37.5" (owner example format "Occupancy: 37.5%").
std::string FmtPercent(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g", value * 100.0);
  return std::string(buf);
}

/// The one-line human summary of a finding: evidence["text"] or the id.
const std::string &FindingText(const Finding &finding) {
  const auto it = finding.evidence.find("text");
  if (it != finding.evidence.end() && !it->second.empty())
    return it->second;
  return finding.id;
}

/// Validate one finding; on violation append to `error` and return false.
bool ValidateFinding(const Finding &finding, const char *section,
                     std::string *error) {
  if (finding.id.empty()) {
    *error = SchemaError(std::string("finding in ") + section +
                         " is missing required field 'id'");
    return false;
  }
  if (finding.severity.empty()) {
    *error = SchemaError(std::string("finding [") + finding.id + "] in " +
                         section +
                         " is missing required field 'severity'");
    return false;
  }
  return true;
}

} // namespace

bool validate(const KernelReport &report, std::string *error) {
  if (report.kernel_name.empty()) {
    *error = SchemaError("kernel.name is required (kernel_name is empty)");
    return false;
  }
  if (report.limiter.empty()) {
    *error = SchemaError("limiter is required (empty)");
    return false;
  }
  for (const auto &[key, value] : report.metrics) {
    if (!std::isfinite(value)) {
      *error = SchemaError("metric '" + key +
                           "' is not a finite number (JSON numbers must be "
                           "finite)");
      return false;
    }
  }
  for (const auto &warning : report.warnings) {
    if (!ValidateFinding(warning, "warnings", error))
      return false;
  }
  return std::ranges::all_of(report.suggestions, [&](const auto &suggestion) {
    return ValidateFinding(suggestion, "suggestions", error);
  });
}

std::string to_text(const KernelReport &report) {
  std::string error;
  if (!validate(report, &error))
    throw SchemaViolation(error);

  std::string out;
  out += "Kernel: " + report.kernel_name + "\n";

  const auto occupancy = report.metrics.find("occupancy");
  if (occupancy != report.metrics.end())
    out += "Occupancy: " + FmtPercent(occupancy->second) + "%\n";

  out += "Main limiter: " + report.limiter + "\n";

  out += "Warnings:\n";
  if (report.warnings.empty()) {
    out += "  (none)\n";
  } else {
    for (const auto &warning : report.warnings)
      out += "  - " + FindingText(warning) + "\n";
  }

  out += "Suggested:\n";
  if (report.suggestions.empty()) {
    out += "  (none)\n";
  } else {
    for (const auto &suggestion : report.suggestions)
      out += "  - " + FindingText(suggestion) + "\n";
  }
  return out;
}

std::string to_json(const KernelReport &report) {
  std::string error;
  if (!validate(report, &error))
    throw SchemaViolation(error);

  std::ostringstream out;

  out << "{\n";
  out << "  \"schema_version\": \"1.0\",\n";
  out << "  \"report_type\": \"kernscope_report\",\n";
  out << "  \"kernel\": {\n";
  out << "    \"name\": \"" << JsonEscape(report.kernel_name) << "\"\n";
  out << "  },\n";
  out << "  \"metrics\": {\n";
  bool first_metric = true;
  for (const auto &[key, value] : report.metrics) {
    if (!first_metric)
      out << ",\n";
    first_metric = false;
    out << "    \"" << JsonEscape(key) << "\": " << FmtNumber(value);
  }
  out << "\n  },\n";
  out << "  \"limiter\": \"" << JsonEscape(report.limiter) << "\",\n";

  const auto findings_json = [&out](const char *name,
                                    const std::vector<Finding> &findings) {
    out << "  \"" << name << "\": [\n";
    for (size_t i = 0; i < findings.size(); ++i) {
      const Finding &finding = findings[i];
      out << "    {\n";
      out << "      \"id\": \"" << JsonEscape(finding.id) << "\",\n";
      out << "      \"severity\": \"" << JsonEscape(finding.severity)
          << "\",\n";
      out << "      \"evidence\": {\n";
      bool first_evidence = true;
      for (const auto &[key, value] : finding.evidence) {
        if (!first_evidence)
          out << ",\n";
        first_evidence = false;
        out << "        \"" << JsonEscape(key) << "\": \""
            << JsonEscape(value) << "\"";
      }
      out << "\n      }\n";
      out << "    }";
      if (i + 1 < findings.size())
        out << ",";
      out << "\n";
    }
    out << "  ]";
  };
  findings_json("warnings", report.warnings);
  out << ",\n";
  findings_json("suggestions", report.suggestions);
  out << "\n}\n";
  return out.str();
}

std::string to_html(const KernelReport &report) {
  std::string error;
  if (!validate(report, &error))
    throw SchemaViolation(error);

  // The payload is the exact JSON model. Escape "</" as "<\/" (a valid JSON
  // escape; JSON.parse and json.loads both decode it) so a closing
  // </script> can never appear inside a payload string.
  std::string payload = to_json(report);
  std::string escaped;
  escaped.reserve(payload.size() + 8);
  for (size_t i = 0; i < payload.size(); ++i) {
    if (payload[i] == '<' && i + 1 < payload.size() && payload[i + 1] == '/') {
      escaped += "<\\/";
      ++i;
    } else {
      escaped += payload[i];
    }
  }

  constexpr const char *kHead = R"html(<!DOCTYPE html>
<!--
  kernscope kernel report (libs/report, todo 7).
  STATIC + SELF-CONTAINED: inline CSS + inline JS + the embedded JSON payload
  in a script element carrying id="report-data" type="application/json".
  No external dependencies, no network fetch, no framework, no build step ->
  opens offline from file:// (polykernel/tt-flow self-contained convention).
  Numbers are static analytic estimates, not wall-clock hardware performance.
-->
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kernscope kernel report</title>
<style>
:root {
  --bg: #0d1117; --panel: #161b22; --ink: #e6edf3; --muted: #9198a1;
  --line: #30363d; --accent: #58a6ff; --warn: #d29922; --good: #3fb950;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink);
  font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 20px 28px 14px; border-bottom: 1px solid var(--line); }
header h1 { margin: 0 0 6px; font-size: 20px; letter-spacing: .2px; }
header .meta { margin: 0; color: var(--muted); font-size: 12.5px; }
main { padding: 20px 28px 48px; max-width: 1100px; margin: 0 auto; }
section { margin: 22px 0; }
section > h2 { font-size: 14px; margin: 0 0 10px; padding-bottom: 5px;
  border-bottom: 1px solid var(--line); text-transform: uppercase;
  letter-spacing: .6px; color: var(--muted); }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 10px 12px; }
.card .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .4px; }
.card .v { font-size: 20px; font-weight: 650; margin-top: 3px; font-variant-numeric: tabular-nums; }
.limiter { display: inline-block; background: var(--panel); border: 1px solid var(--accent);
  color: var(--accent); border-radius: 999px; padding: 3px 12px; font-size: 13px; font-weight: 600; }
.finding { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 10px 12px; margin: 8px 0; }
.fhead { display: flex; align-items: baseline; gap: 10px; margin-bottom: 6px; }
.fid { font-weight: 650; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.fsev { font-size: 11px; text-transform: uppercase; letter-spacing: .4px;
  padding: 1px 8px; border-radius: 999px; }
.fsev.warning { background: rgba(210,153,34,.18); color: var(--warn); }
.fsev.suggestion { background: rgba(63,185,80,.16); color: var(--good); }
table.data { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.data td { padding: 3px 8px; border-top: 1px solid var(--panel); }
table.data td:first-child { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  width: 34%; vertical-align: top; }
footer { color: var(--muted); font-size: 11.5px; padding: 8px 28px 28px; }
</style>
</head>
<body>
<header>
  <h1>kernscope kernel report</h1>
  <p class="meta">kernel: <span id="kernel-name">-</span> &middot; report type: <span id="report-type">-</span></p>
</header>
<main>
  <section>
    <h2>Main limiter</h2>
    <span class="limiter" id="limiter">-</span>
  </section>
  <section>
    <h2>Metrics</h2>
    <div class="cards" id="metrics"></div>
  </section>
  <section>
    <h2>Warnings</h2>
    <div id="warnings"></div>
  </section>
  <section>
    <h2>Suggestions</h2>
    <div id="suggestions"></div>
  </section>
</main>
<footer>Static analytic estimate &mdash; not wall-clock hardware performance.
Machine-readable model embedded below (extract with python3 html.parser, see
tests/report/check_html.py).</footer>
)html";

  constexpr const char *kTail = R"html(</script>
<script>
(function () {
  'use strict';
  var raw = document.getElementById('report-data').textContent;
  var data = JSON.parse(raw);
  // Exposed for tooling / debuggers; the payload itself is the source of truth.
  window.__kernscopeReport = data;
  document.title = 'kernscope report: ' + data.kernel.name;
  document.getElementById('kernel-name').textContent = data.kernel.name;
  document.getElementById('report-type').textContent = data.report_type || '';
  document.getElementById('limiter').textContent = data.limiter;

  var metricsEl = document.getElementById('metrics');
  Object.keys(data.metrics).forEach(function (key) {
    var card = document.createElement('div');
    card.className = 'card';
    var k = document.createElement('div');
    k.className = 'k';
    k.textContent = key;
    var v = document.createElement('div');
    v.className = 'v';
    v.textContent = String(data.metrics[key]);
    card.appendChild(k);
    card.appendChild(v);
    metricsEl.appendChild(card);
  });

  function renderFindings(list, container) {
    if (!list.length) {
      var none = document.createElement('div');
      none.className = 'finding';
      none.textContent = '(none)';
      container.appendChild(none);
      return;
    }
    list.forEach(function (f) {
      var item = document.createElement('div');
      item.className = 'finding';
      var head = document.createElement('div');
      head.className = 'fhead';
      var id = document.createElement('span');
      id.className = 'fid';
      id.textContent = f.id;
      var sev = document.createElement('span');
      sev.className = 'fsev ' + String(f.severity).toLowerCase();
      sev.textContent = f.severity;
      head.appendChild(id);
      head.appendChild(sev);
      item.appendChild(head);
      var table = document.createElement('table');
      table.className = 'data';
      Object.keys(f.evidence).forEach(function (key) {
        var tr = document.createElement('tr');
        var k = document.createElement('td');
        k.textContent = key;
        var v = document.createElement('td');
        v.textContent = f.evidence[key];
        tr.appendChild(k);
        tr.appendChild(v);
        table.appendChild(tr);
      });
      item.appendChild(table);
      container.appendChild(item);
    });
  }
  renderFindings(data.warnings, document.getElementById('warnings'));
  renderFindings(data.suggestions, document.getElementById('suggestions'));
})();
</script>
</body>
</html>
)html";

  std::string html;
  html.reserve(std::strlen(kHead) + payload.size() + std::strlen(kTail) +
               128);
  html += kHead;
  // The payload sits directly between the tags with no surrounding
  // whitespace: extraction is byte-exact (to_json's own trailing newline
  // survives, and check_html.py's json.loads tolerates it either way).
  html += kJsonPayloadTag;
  html += escaped;
  html += kTail;
  return html;
}

} // namespace kernscope
