//===- tune_cli.cpp - kernscope-tune CLI -----------------------------------===//
//
// kernscope tools/tune todos 28 + 29. The `kernscope-tune` command-line
// front end for the correctness-gated tuning driver. One invocation tunes
// one (kernel_family, shape, arch) over the family's config grid and reports
// the best config in the requested output mode.
//
//   kernscope-tune --kernel <family> --shape <M,N,K|R,C|Nq,Nk,d>
//                  --arch <auto|sm_XX|gfxXXXX>
//                  [--grid <auto|full>] [--json|--html|--text]
//                  [--db <path>] [--cache <path>] [--workdir <dir>]
//                  [--min-occupancy <f>] [--no-bench] [--no-cache|--force]
//                  [--emit-repro <dir>] [--version] [--help]
//
//   --arch auto  -> gfx1101 (this machine's RX 7800 XT; NVIDIA runtimes are
//                  owner-gated — an explicit sm_XX runs compile-only and
//                  exits 2 because benchmarking needs a live HIP device).
//   --json       the full JSON evidence (best config, roofline.position /
//                  bound, occupancy / occupancy.limiter, baseline + speedup,
//                  variants sorted by median ascending, cache key).
//   --text       the owner's example format (docs/research/
//                  owner-examples.md), adapted to tune fields: Kernel /
//                  Time / Best config / Occupancy / Main limiter / Roofline
//                  position+bound / Speedup vs naive baseline.
//   --html       a self-contained single-file variant-comparison report:
//                  the libs/report model JSON embedded in the standard
//                  <script id="report-data" type="application/json"> payload
//                  (the byte-exact extraction pattern) plus the full tune
//                  evidence in <script id="tune-evidence">, all measured
//                  variants ranked by median with p20/p80 and speedup vs the
//                  naive baseline; zero network, opens from file://.
//   --emit-repro <dir>  writes run.sh + README.md that regenerate the
//                  winning config via the family generator CLI, compile it
//                  with hipcc --genco, and re-run the golden gate
//                  (run_hip_gate.py, verdict pass:true) on the same
//                  gfx1101 machine the tune ran on. The scripts pin the
//                  devenv.lock nixpkgs revision + the exact toolchain paths.
//
// Exit codes: 0 = best config selected; 1 = no viable config (every variant
// pre-screen rejected / gate-failed — a search failure, not a crash);
// 2 = usage/environment error; 3 = unknown kernel family (usage).
//
//===----------------------------------------------------------------------===//

#include "kernscope/report.h"
#include "kernscope/tune/capture_tune.h"
#include "kernscope/tune/driver.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

const char *kFamilies[] = {"matmul",       "bias_gelu", "softmax",
                           "layernorm",    "reduction", "flash_attention"};

// The devenv.lock nixpkgs revision the tune ran under (read from
// devenv.lock at repo root — the --emit-repro scripts reference it as the
// environment pin).
constexpr const char *kNixpkgsRev = "044bfe75bfe4c7bbe043dc17b5e42ea823b84a09";

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --kernel <family> --shape <M,N,K|R,C|Nq,Nk,d>\n"
               "              --arch <auto|sm_XX|gfxXXXX> [--grid <auto|full>]\n"
               "              [--gate-shape <csv>] [--json|--html|--text]\n"
               "              [--db <path>] [--cache <path>]\n"
               "              [--workdir <dir>] [--min-occupancy <f>]\n"
               "              [--no-bench] [--no-cache|--force]\n"
               "              [--emit-repro <dir>] [--version]\n"
               "  family: matmul | bias_gelu | softmax | layernorm |\n"
               "          reduction | flash_attention\n"
               "  --arch auto resolves to gfx1101 on this machine\n"
               "  --no-cache skips the tunecache READ (always re-runs the\n"
               "            pipeline) but still writes the cache\n"
               "  --emit-repro <dir> writes run.sh + README.md that\n"
               "            regenerate + golden-check the winning config\n",
               argv0);
}

bool parse_shape(const std::string &s, std::vector<std::int64_t> *out) {
  out->clear();
  std::size_t pos = 0;
  while (pos <= s.size()) {
    const std::size_t comma = s.find(',', pos);
    const std::string tok = s.substr(
        pos, comma == std::string::npos ? std::string::npos : comma - pos);
    try {
      const long long v = std::stoll(tok);
      if (v <= 0)
        return false;
      out->push_back(static_cast<std::int64_t>(v));
    } catch (...) {
      return false;
    }
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  return !out->empty();
}

bool parse_double(const char *s, double *out) {
  if (!s || !*s)
    return false;
  char *end = nullptr;
  const double v = std::strtod(s, &end);
  if (!end || *end != '\0' || v < 0.0)
    return false;
  *out = v;
  return true;
}

std::string shape_str(const std::vector<std::int64_t> &shape) {
  std::string s;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0)
      s += ",";
    s += std::to_string(shape[i]);
  }
  return s;
}

std::string fmt_num(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g", v);
  return std::string(buf);
}

// Percent like libs/report's FmtPercent: 0.375 -> "37.5".
std::string fmt_pct(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4g", v * 100.0);
  return std::string(buf);
}

std::string html_escape(const std::string &s) {
  std::string out;
  for (const char c : s) {
    switch (c) {
    case '&':
      out += "&amp;";
      break;
    case '<':
      out += "&lt;";
      break;
    case '>':
      out += "&gt;";
      break;
    case '"':
      out += "&quot;";
      break;
    default:
      out += c;
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// --text: the owner's example format (docs/research/owner-examples.md),
// adapted to tune-specific fields.
// ---------------------------------------------------------------------------

void print_text(const std::string &evidence_json, const std::string &family,
                const std::string &arch, const std::string &shape_csv) {
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  const bool have = ev && ev->find("status");
  const std::string kernel_line =
      "Kernel: " + family + " " + shape_csv + " (" + arch + ")\n";
  if (!have) {
    std::printf("%s", kernel_line.c_str());
    std::printf("kernscope-tune: no evidence (%s)\n", err.c_str());
    return;
  }

  const auto *best = ev->find("best");
  if (!best) {
    std::printf("%s", kernel_line.c_str());
    std::printf("kernscope-tune: %s\n",
                ev->find("message") ? ev->find("message")->as_string().c_str()
                                    : "no viable config");
    return;
  }

  const auto *timings = best->find("timings");
  const double median =
      timings && timings->find("median") ? timings->find("median")->as_number() : 0.0;
  const double p20 =
      timings && timings->find("p20") ? timings->find("p20")->as_number() : 0.0;
  const double p80 =
      timings && timings->find("p80") ? timings->find("p80")->as_number() : 0.0;

  std::printf("%s", kernel_line.c_str());
  std::printf("Time: %.4f ms (median; p20 %.4f, p80 %.4f)\n", median, p20, p80);
  std::printf("Best config: %s\n",
              best->find("config") ? best->find("config")->dump().c_str() : "{}");

  double occ = 0.0;
  if (const auto *o = ev->find("occupancy"))
    occ = o->as_number();
  std::printf("Occupancy: %s%%\n", fmt_pct(occ).c_str());
  std::printf("Main limiter: %s\n",
              ev->find("occupancy.limiter")
                  ? ev->find("occupancy.limiter")->as_string().c_str()
                  : "n/a");

  if (const auto *rl = ev->find("roofline")) {
    const bool ok = rl->find("ok") && rl->find("ok")->as_bool();
    if (ok) {
      const double position =
          rl->find("position") ? rl->find("position")->as_number() : 0.0;
      const std::string bound =
          rl->find("bound") ? rl->find("bound")->as_string() : "n/a";
      const double ai =
          rl->find("arithmetic_intensity") ? rl->find("arithmetic_intensity")->as_number() : 0.0;
      const double ridge = rl->find("ridge_point") ? rl->find("ridge_point")->as_number() : 0.0;
      const double flops =
          rl->find("attained_flops") ? rl->find("attained_flops")->as_number() : 0.0;
      std::printf("Roofline position: %s (bound: %s)\n", fmt_num(position).c_str(),
                  bound.c_str());
      std::printf("Roofline: AI %s FLOP/B, ridge %s, attained %s FLOP/s\n",
                  fmt_num(ai).c_str(), fmt_num(ridge).c_str(),
                  fmt_num(flops).c_str());
    }
  }

  if (const auto *speedup = ev->find("speedup")) {
    const double sp = speedup->as_number();
    double base = 0.0;
    if (const auto *b = ev->find("baseline"))
      if (const auto *bt = b->find("timings"))
        if (bt->find("median"))
          base = bt->find("median")->as_number();
    std::printf("Speedup vs naive baseline: %sx (baseline median %.4f ms)\n",
                fmt_num(sp).c_str(), base);
  }
}

// ---------------------------------------------------------------------------
// --html: self-contained variant-comparison report (libs/report model JSON
// in the standard report-data payload + the full evidence + ranked table).
// ---------------------------------------------------------------------------

const kernscope::KernelReport BuildReport(const std::string &evidence_json,
                                          const std::string &family,
                                          const std::string &arch,
                                          const std::string &shape_csv) {
  kernscope::KernelReport report;
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  report.kernel_name = family + " " + shape_csv + " (" + arch + ")";

  const auto *best = ev ? ev->find("best") : nullptr;
  if (!best) {
    report.limiter = "n/a";
    kernscope::Finding status;
    status.id = "status";
    status.severity = "warning";
    status.evidence["text"] =
        ev && ev->find("message") ? ev->find("message")->as_string() : "no viable config";
    report.warnings.push_back(status);
    return report;
  }

  const auto *timings = best->find("timings");
  const auto *rl = ev->find("roofline");
  const double position =
      rl && rl->find("position") ? rl->find("position")->as_number() : 0.0;
  const std::string bound =
      rl && rl->find("bound") ? rl->find("bound")->as_string() : "n/a";
  const double occ = ev->find("occupancy") ? ev->find("occupancy")->as_number() : 0.0;
  const double speedup = ev->find("speedup") ? ev->find("speedup")->as_number() : 0.0;

  report.metrics["time_median_ms"] =
      timings && timings->find("median") ? timings->find("median")->as_number() : 0.0;
  report.metrics["time_p20_ms"] =
      timings && timings->find("p20") ? timings->find("p20")->as_number() : 0.0;
  report.metrics["time_p80_ms"] =
      timings && timings->find("p80") ? timings->find("p80")->as_number() : 0.0;
  report.metrics["occupancy"] = occ;
  report.metrics["roofline_position"] = position;
  report.metrics["roofline_arithmetic_intensity"] =
      rl && rl->find("arithmetic_intensity") ? rl->find("arithmetic_intensity")->as_number() : 0.0;
  report.metrics["roofline_ridge_point"] =
      rl && rl->find("ridge_point") ? rl->find("ridge_point")->as_number() : 0.0;
  report.metrics["roofline_attained_flops"] =
      rl && rl->find("attained_flops") ? rl->find("attained_flops")->as_number() : 0.0;
  report.metrics["speedup_vs_baseline"] = speedup;
  if (const auto *sm = ev->find("summary")) {
    if (sm->find("variants_total"))
      report.metrics["variants_total"] = sm->find("variants_total")->as_number();
    if (sm->find("benchmarked"))
      report.metrics["variants_measured"] = sm->find("benchmarked")->as_number();
    if (sm->find("prescreen_rejected"))
      report.metrics["prescreen_rejected"] = sm->find("prescreen_rejected")->as_number();
    if (sm->find("golden_gate_failed"))
      report.metrics["golden_gate_failed"] = sm->find("golden_gate_failed")->as_number();
    if (sm->find("cached"))
      report.metrics["cached"] = sm->find("cached")->as_number();
    if (sm->find("compiled"))
      report.metrics["compiled"] = sm->find("compiled")->as_number();
  }

  report.limiter =
      ev->find("occupancy.limiter") ? ev->find("occupancy.limiter")->as_string() : "n/a";

  kernscope::Finding bound_finding;
  bound_finding.id = "roofline_bound";
  bound_finding.severity = "warning";
  bound_finding.evidence["bound"] = bound;
  bound_finding.evidence["text"] =
      "workload is " + bound + "-bound (AI " +
      fmt_num(report.metrics["roofline_arithmetic_intensity"]) +
      " FLOP/B vs ridge " +
      fmt_num(report.metrics["roofline_ridge_point"]) + " FLOP/B)";
  report.warnings.push_back(bound_finding);

  kernscope::Finding occ_finding;
  occ_finding.id = "occupancy";
  occ_finding.severity = "suggestion";
  occ_finding.evidence["occupancy"] = fmt_num(occ);
  occ_finding.evidence["text"] =
      occ < 0.5 ? "occupancy below 50%: consider smaller tiles or more "
                  "warps to raise resident blocks"
                : "occupancy is healthy: the winning config fills the SM well";
  report.suggestions.push_back(occ_finding);
  return report;
}

// JSON payloads embedded between script tags: escape "</" as "<\/" (a valid
// JSON escape decoded by both JSON.parse and json.loads) so a closing
// </script> can never appear inside a payload string.
std::string ScriptSafe(const std::string &json) {
  std::string out;
  out.reserve(json.size() + 8);
  for (std::size_t i = 0; i < json.size(); ++i) {
    if (json[i] == '<' && i + 1 < json.size() && json[i + 1] == '/') {
      out += "<\\/";
      ++i;
    } else {
      out += json[i];
    }
  }
  return out;
}

void print_html(const std::string &evidence_json, const std::string &family,
                const std::string &arch, const std::string &shape_csv) {
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  const std::string model_json =
      kernscope::to_json(BuildReport(evidence_json, family, arch, shape_csv));

  std::printf("<!DOCTYPE html>\n");
  std::printf("<!--\n"
              "  kernscope-tune variant-comparison report (todo 29).\n"
              "  STATIC + SELF-CONTAINED: inline CSS + inline JS, the\n"
              "  libs/report model JSON in <script id=\"report-data\"\n"
              "  type=\"application/json\"> (the byte-exact extraction\n"
              "  pattern) + the full tune evidence in <script\n"
              "  id=\"tune-evidence\" type=\"application/json\">. No external\n"
              "  dependencies, no network fetch -> opens offline from file://.\n"
              "-->\n");
  std::printf("<html lang=\"en\"><head><meta charset=\"utf-8\">\n");
  std::printf("<title>kernscope-tune: %s %s (%s)</title>\n",
              html_escape(family).c_str(), html_escape(shape_csv).c_str(),
              html_escape(arch).c_str());
  std::printf("<style>"
              ":root{--bg:#0d1117;--panel:#161b22;--ink:#e6edf3;--muted:#9198a1;"
              "--line:#30363d;--accent:#58a6ff;--warn:#d29922;--good:#3fb950}"
              "*{box-sizing:border-box}body{margin:0;background:var(--bg);"
              "color:var(--ink);font:14px/1.5 ui-sans-serif,system-ui,sans-serif}"
              "header{padding:18px 28px 12px;border-bottom:1px solid var(--line)}"
              "header h1{margin:0 0 4px;font-size:19px}header .meta{margin:0;"
              "color:var(--muted);font-size:12.5px}main{padding:16px 28px 40px;"
              "max-width:1200px;margin:0 auto}section{margin:20px 0}"
              "section>h2{font-size:13px;margin:0 0 10px;padding-bottom:4px;"
              "border-bottom:1px solid var(--line);text-transform:uppercase;"
              "letter-spacing:.6px;color:var(--muted)}"
              ".cards{display:grid;grid-template-columns:repeat(auto-fit,"
              "minmax(150px,1fr));gap:10px}.card{background:var(--panel);"
              "border:1px solid var(--line);border-radius:8px;padding:9px 12px}"
              ".card .k{color:var(--muted);font-size:10.5px;text-transform:"
              "uppercase;letter-spacing:.4px}.card .v{font-size:18px;"
              "font-weight:650;margin-top:2px;font-variant-numeric:tabular-nums}"
              "table.data{border-collapse:collapse;width:100%;font-size:12.5px}"
              "table.data th{text-align:left;color:var(--muted);font-weight:600;"
              "padding:4px 8px;border-bottom:1px solid var(--line)}"
              "table.data td{padding:4px 8px;border-top:1px solid var(--panel);"
              "font-variant-numeric:tabular-nums}"
              "tr.best td{background:rgba(63,185,80,.10);color:var(--good)}"
              "td.cfg{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
              "font-size:11.5px;color:var(--muted)}"
              "span.status{color:var(--muted)}footer{color:var(--muted);"
              "font-size:11.5px;padding:4px 28px 24px}</style></head><body>\n");
  std::printf("<header><h1>kernscope-tune variant report: %s %s (%s)</h1>"
              "<p class=\"meta\">correctness-gated autotune &middot; "
              "GPU-free pre-screen &rarr; compile &rarr; golden gate &rarr; "
              "bench &rarr; select</p></header><main>\n",
              html_escape(family).c_str(), html_escape(shape_csv).c_str(),
              html_escape(arch).c_str());

  const bool has_best = ev && ev->find("best");
  std::printf("<section><h2>Summary</h2><div class=\"cards\">");
  if (has_best) {
    const auto *best = ev->find("best");
    const auto *timings = best->find("timings");
    const double median =
        timings && timings->find("median") ? timings->find("median")->as_number() : 0.0;
    const double occ = ev->find("occupancy") ? ev->find("occupancy")->as_number() : 0.0;
    const double position =
        ev->find("roofline") && ev->find("roofline")->find("position")
            ? ev->find("roofline")->find("position")->as_number()
            : 0.0;
    const std::string bound =
        ev->find("roofline") && ev->find("roofline")->find("bound")
            ? ev->find("roofline")->find("bound")->as_string()
            : "n/a";
    const double speedup =
        ev->find("speedup") ? ev->find("speedup")->as_number() : 0.0;
    std::printf("<div class=\"card\"><div class=\"k\">median time</div>"
                "<div class=\"v\">%.4f ms</div></div>",
                median);
    std::printf("<div class=\"card\"><div class=\"k\">occupancy</div>"
                "<div class=\"v\">%s%%</div></div>",
                fmt_pct(occ).c_str());
    std::printf("<div class=\"card\"><div class=\"k\">roofline position</div>"
                "<div class=\"v\">%s</div></div>",
                fmt_num(position).c_str());
    std::printf("<div class=\"card\"><div class=\"k\">roofline bound</div>"
                "<div class=\"v\">%s</div></div>",
                html_escape(bound).c_str());
    std::printf("<div class=\"card\"><div class=\"k\">speedup vs baseline</div>"
                "<div class=\"v\">%sx</div></div>",
                fmt_num(speedup).c_str());
  } else {
    std::printf("<div class=\"card\"><div class=\"k\">status</div>"
                "<div class=\"v\">no viable config</div></div>");
  }
  std::printf("</div></section>\n");

  // Variant table: the driver emits measured variants first, ranked by
  // median ascending (ties p20/p80), then non-measured variants.
  std::printf("<section><h2>Variants ranked by median</h2>"
              "<table class=\"data\"><tr><th>#</th><th>config</th>"
              "<th>median ms</th><th>p20</th><th>p80</th>"
              "<th>speedup</th><th>occupancy</th><th>status</th></tr>");
  double baseline_median = 0.0;
  if (has_best && ev->find("baseline") && ev->find("baseline")->find("timings"))
    if (ev->find("baseline")->find("timings")->find("median"))
      baseline_median =
          ev->find("baseline")->find("timings")->find("median")->as_number();
  int rank = 0;
  if (ev && ev->find("variants")) {
    for (const auto &v : ev->find("variants")->as_array()) {
      const bool is_best = v.find("best") && v.find("best")->as_bool();
      const bool has_t = v.find("timings") != nullptr;
      std::printf("<tr%s>", is_best ? " class=\"best\"" : "");
      if (has_t) {
        ++rank;
        const double median =
            v.find("timings")->find("median") ? v.find("timings")->find("median")->as_number() : 0.0;
        const double p20 =
            v.find("timings")->find("p20") ? v.find("timings")->find("p20")->as_number() : 0.0;
        const double p80 =
            v.find("timings")->find("p80") ? v.find("timings")->find("p80")->as_number() : 0.0;
        const double sp =
            (baseline_median > 0.0 && median > 0.0) ? baseline_median / median : 0.0;
        const double occ =
            v.find("occupancy") ? v.find("occupancy")->as_number() : 0.0;
        std::printf("<td>%d</td><td class=\"cfg\">%s</td><td>%.4f</td>"
                    "<td>%.4f</td><td>%.4f</td><td>%sx</td><td>%s%%</td><td>%s</td>",
                    rank, html_escape(v.find("config") ? v.find("config")->dump() : "{}").c_str(),
                    median, p20, p80, fmt_num(sp).c_str(), fmt_pct(occ).c_str(),
                    html_escape(v.find("status") ? v.find("status")->as_string() : "").c_str());
      } else {
        std::printf("<td>-</td><td class=\"cfg\">%s</td><td>-</td><td>-</td>"
                    "<td>-</td><td>-</td><td>-</td><td><span class=\"status\">%s%s</span></td>",
                    html_escape(v.find("config") ? v.find("config")->dump() : "{}").c_str(),
                    html_escape(v.find("status") ? v.find("status")->as_string() : "").c_str(),
                    v.find("reason")
                        ? html_escape(": " + v.find("reason")->as_string()).c_str()
                        : "");
      }
      std::printf("</tr>");
    }
  }
  std::printf("</table></section></main>\n");

  // Embedded payloads: the libs/report model (byte-exact extractable) and
  // the full tune evidence (variants, roofline, occupancy, baseline).
  std::printf("<script id=\"report-data\" type=\"application/json\">%s</script>\n",
              ScriptSafe(model_json).c_str());
  std::printf("<script id=\"tune-evidence\" type=\"application/json\">%s</script>\n",
              ScriptSafe(evidence_json).c_str());
  std::printf("<script>(function(){'use strict';"
              "var d=JSON.parse(document.getElementById('report-data').textContent);"
              "window.__kernscopeReport=d;"
              "document.title='kernscope-tune: '+d.kernel.name;})();</script>\n");
  std::printf("</body></html>\n");
}

// ---------------------------------------------------------------------------
// --emit-repro: standalone regeneration + golden-check of the winning config.
// ---------------------------------------------------------------------------

std::string exe_dir(const char *argv0) {
  const std::string s = argv0 ? argv0 : "";
  const std::size_t slash = s.find_last_of('/');
  if (slash == std::string::npos)
    return ".";
  if (slash == 0)
    return "/";
  return s.substr(0, slash);
}

std::string shell_quote(const std::string &s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\' || c == '$' || c == '`')
      out += '\\';
    out += c;
  }
  out += "\"";
  return out;
}

// Generator CLI args for the winning config. Returns false when the family
// has no generator CLI (flash_attention) — the repro cannot regenerate it.
bool gen_cli_args(const std::string &family, const std::string &arch,
                  const kernscope::db::json::Value &config,
                  const std::string &shape_csv, const std::string &out_path,
                  std::string *args, std::string *err) {
  if (family == "matmul") {
    auto num = [&](const char *key, double fallback) {
      return config.find(key) ? static_cast<long long>(config.find(key)->as_number())
                              : static_cast<long long>(fallback);
    };
    *args = "--arch " + arch + " --bm " + std::to_string(num("tile_m", 64)) +
            " --bn " + std::to_string(num("tile_n", 64)) + " --bk " +
            std::to_string(num("tile_k", 16)) + " --warps " +
            std::to_string(num("warps", 1)) + " --stages " +
            std::to_string(num("stages", 1)) + " --vec " +
            std::to_string(num("vec_width", 1)) + " --layout " +
            (config.find("smem_layout")
                 ? config.find("smem_layout")->as_string()
                 : "padding") +
            " --unroll " + std::to_string(num("unroll", 1)) + " --out \"" +
            out_path + "\"";
    return true;
  }
  if (family == "bias_gelu" || family == "softmax" || family == "layernorm" ||
      family == "reduction") {
    auto num = [&](const char *key, double fallback) {
      return config.find(key) ? static_cast<long long>(config.find(key)->as_number())
                              : static_cast<long long>(fallback);
    };
    *args = "--kernel " + family + " --arch " + arch + " --block " +
            std::to_string(num("block", 256)) + " --vec " +
            std::to_string(num("vec_width", 1));
    if (family != "bias_gelu" && config.find("mode") &&
        config.find("mode")->as_string() != "-")
      *args += " --mode " + config.find("mode")->as_string();
    *args += " --shape " + shape_csv + " --out \"" + out_path + "\"";
    return true;
  }
  if (err)
    *err = "family '" + family +
           "' has no generator CLI; --emit-repro supports matmul, bias_gelu, "
           "softmax, layernorm, reduction";
  return false;
}

void emit_repro(const std::string &evidence_json, const std::string &family,
                const std::string &arch, const std::string &shape_csv,
                const std::string &dir, const char *argv0) {
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  const auto *best = ev && ev->find("best") ? ev->find("best") : nullptr;
  if (!best) {
    std::fprintf(stderr, "kernscope-tune: --emit-repro needs a live-tuned "
                         "best config (this run found none)\n");
    std::exit(2);
  }
  const auto *gate = best->find("gate");
  if (!gate) {
    std::fprintf(stderr, "kernscope-tune: --emit-repro needs the winning "
                         "config's golden-gate geometry (tune on a live HIP "
                         "device, e.g. --arch gfx1101)\n");
    std::exit(2);
  }
  const auto *config = best->find("config");
  if (!config) {
    std::fprintf(stderr, "kernscope-tune: --emit-repro: best config missing\n");
    std::exit(2);
  }

  const kernscope::tune::ToolchainPaths tc = kernscope::tune::toolchain_paths();
  // Absolute: the emitted run.sh resolves its own dir, not the CLI's CWD.
  std::error_code abs_ec;
  const std::string gen_bin = std::filesystem::absolute(
      exe_dir(argv0) +
      (family == "matmul" ? "/kernscope-tune-gen" : "/kernscope-tune-elem-gen"),
      abs_ec);
  std::string gen_args;
  if (!gen_cli_args(family, arch, *config, shape_csv, "$DIR/kernel.hip",
                    &gen_args, &err)) {
    std::fprintf(stderr, "kernscope-tune: %s\n", err.c_str());
    std::exit(2);
  }

  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  if (ec) {
    std::fprintf(stderr, "kernscope-tune: cannot create repro dir %s: %s\n",
                 dir.c_str(), ec.message().c_str());
    std::exit(2);
  }

  const std::string gate_shape = gate->find("shape") ? gate->find("shape")->as_string() : "";
  const std::string grid = gate->find("grid") ? gate->find("grid")->as_string() : "";
  const std::string block = gate->find("block") ? gate->find("block")->as_string() : "";
  const std::string symbol = gate->find("symbol") ? gate->find("symbol")->as_string() : "";

  const std::string run_sh = std::string(
      "#!/usr/bin/env bash\n"
      "# kernscope-tune --emit-repro artifact (tools/tune todo 29).\n"
      "# Reproduces the winning config on the machine the tune ran on\n"
      "# (gfx1101 / RX 7800 XT): regenerate the source via the family\n"
      "# generator CLI, compile with hipcc --genco, re-run the golden gate\n"
      "# (run_hip_gate.py; the verdict must be pass:true).\n"
      "#\n"
      "# Environment pin (devenv.lock at the repo root):\n"
      "#   nixpkgs rev: ") +
      kNixpkgsRev +
      "\n"
      "# Toolchain the tune ran with (resolved at configure time):\n"
      "#   hipcc:   " +
      tc.hipcc +
      "\n"
      "#   python:  " +
      tc.python +
      "\n"
      "#   runner:  " +
      tc.runner +
      "\n"
      "#   gate:    " +
      tc.run_hip_gate +
      "\n"
      "#   repo:    " +
      tc.repo_root +
      "\n"
      "# Winning config: " +
      config->dump() +
      "\n"
      "# Gate: shape " +
      gate_shape + ", grid " + grid + ", block " + block + ", symbol " +
      symbol +
      "\n"
      "set -euo pipefail\n"
      "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
      "REPO_ROOT=" +
      shell_quote(tc.repo_root) +
      "\n"
      "RUNNER=" +
      shell_quote(tc.runner) +
      "\n"
      "PYTHON=" +
      shell_quote(tc.python) +
      "\n"
      "HIPCC=" +
      shell_quote(tc.hipcc) +
      "\n"
      "HIP_INCLUDE=" +
      shell_quote(tc.hip_include) +
      "\n"
      "TRAITS_INCLUDE=" +
      shell_quote(tc.traits_include) +
      "\n"
      "GATE=" +
      shell_quote(tc.run_hip_gate) +
      "\n"
      "GEN=" +
      shell_quote(gen_bin) +
      "\n"
      "ARCH=" +
      arch +
      "\n"
      "FAMILY=" +
      family +
      "\n"
      "GATE_SHAPE=" +
      gate_shape +
      "\n"
      "GRID=" +
      grid +
      "\n"
      "BLOCK=" +
      block +
      "\n"
      "SYMBOL=" +
      symbol +
      "\n"
      "NIXPKGS_REV=" +
      kNixpkgsRev +
      "\n"
      "\n"
      "# Soft pin check: the environment reports its nixpkgs rev via\n"
      "# KERNSCOPE_EXPECT_NIXPKGS_REV (devenv.nix sets it); warn (not fail)\n"
      "# when it differs from the rev this repro was generated under.\n"
      "if [ -n \"${KERNSCOPE_EXPECT_NIXPKGS_REV:-}\" ] && \\\n"
      "   [ \"$KERNSCOPE_EXPECT_NIXPKGS_REV\" != \"$NIXPKGS_REV\" ]; then\n"
      "  echo \"warn: nixpkgs $KERNSCOPE_EXPECT_NIXPKGS_REV != pinned $NIXPKGS_REV\" >&2\n"
      "fi\n"
      "for bin in \"$HIPCC\" \"$RUNNER\" \"$PYTHON\" \"$GEN\"; do\n"
      "  if [ ! -x \"$bin\" ]; then\n"
      "    echo \"repro: missing executable: $bin\" >&2\n"
      "    exit 1\n"
      "  fi\n"
      "done\n"
      "if [ ! -f \"$GATE\" ]; then\n"
      "  echo \"repro: missing gate script: $GATE\" >&2\n"
      "  exit 1\n"
      "fi\n"
      "\n"
      "echo \"[repro] generating kernel.hip via $GEN\"\n"
      "\"$GEN\" " +
      gen_args +
      "\n"
      "echo \"[repro] compiling kernel.hsaco with $HIPCC (--genco \"$ARCH\")\"\n"
      "\"$HIPCC\" --genco --offload-arch=\"$ARCH\" -gline-tables-only \\\n"
      "  -D__HIP_PLATFORM_AMD__=1 -I\"$HIP_INCLUDE\" -I\"$TRAITS_INCLUDE\" \\\n"
      "  \"$DIR/kernel.hip\" -o \"$DIR/kernel.hsaco\"\n"
      "echo \"[repro] golden gate at shape $GATE_SHAPE (grid $GRID, block \""
      "\"$BLOCK, symbol $SYMBOL)\"\n"
      "\"$PYTHON\" \"$GATE\" --root \"$REPO_ROOT\" --workdir \"$DIR/gate\" \\\n"
      "  --hsaco \"$DIR/kernel.hsaco\" --runner \"$RUNNER\" --symbol \"$SYMBOL\" \\\n"
      "  --family \"$FAMILY\" --shape \"$GATE_SHAPE\" --grid \"$GRID\" \\\n"
      "  --block \"$BLOCK\"\n"
      "echo \"[repro] PASS: winning config regenerated, compiled, and \""
      "\"golden-validated on $ARCH\"\n";

  const std::string readme = std::string(
      "# kernscope-tune reproducibility bundle\n"
      "\n"
      "Generated by `kernscope-tune --emit-repro <dir>` (tools/tune, todo 29)\n"
      "for the winning config of a live tune on this machine (gfx1101,\n"
      "RX 7800 XT). It proves the selected config is reproducible: the same\n"
      "source, the same compile, the same golden verdict.\n"
      "\n"
      "## Files\n"
      "\n"
      "- `run.sh` — regenerate `kernel.hip` via the family generator CLI,\n"
      "  compile `kernel.hsaco` with `hipcc --genco --offload-arch=gfx1101`,\n"
      "  then re-run the correctness gate (`kernscope_golden` via\n"
      "  `run_hip_gate.py`). Exit 0 only when the verdict is `pass:true`\n"
      "  (cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99, todo 23).\n"
      "\n"
      "## How to run\n"
      "\n"
      "```bash\n"
      "cd <repo-root>\n"
      "devenv shell      # toolchain (hipcc 7.2, python, runner)\n"
      "bash <dir>/run.sh\n"
      "```\n"
      "\n"
      "Expected output ends with:\n"
      "\n"
      "```\n"
      "[repro] PASS: winning config regenerated, compiled, and golden-validated on gfx1101\n"
      "```\n"
      "\n"
      "## Environment pin\n"
      "\n"
      "The tune ran under the devenv shell pinned by `devenv.lock` at the repo\n"
      "root. The nixpkgs revision is:\n"
      "\n"
      "```\n") +
      kNixpkgsRev +
      "\n"
      "```\n"
      "\n"
      "`run.sh` checks `KERNSCOPE_EXPECT_NIXPKGS_REV` (set by `devenv.nix`) and\n"
      "warns (does not fail) when it differs. Toolchain paths are pinned to the\n"
      "exact binaries the tune ran with (see the `run.sh` header).\n"
      "\n"
      "## Winning config\n"
      "\n"
      "```json\n" +
      config->dump() +
      "\n"
      "```\n"
      "\n"
      "Golden gate: shape `" +
      gate_shape + "`, grid `" + grid + "`, block `" + block +
      "`, symbol `" + symbol + "`.\n";

  const std::string run_path = dir + "/run.sh";
  const std::string readme_path = dir + "/README.md";
  FILE *f = std::fopen(run_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "kernscope-tune: cannot write %s\n", run_path.c_str());
    std::exit(2);
  }
  std::fputs(run_sh.c_str(), f);
  std::fclose(f);
  // 0755
  std::filesystem::permissions(run_path, std::filesystem::perms::owner_all |
                                              std::filesystem::perms::group_read |
                                              std::filesystem::perms::group_exec |
                                              std::filesystem::perms::others_read |
                                              std::filesystem::perms::others_exec,
                               ec);
  f = std::fopen(readme_path.c_str(), "w");
  if (!f) {
    std::fprintf(stderr, "kernscope-tune: cannot write %s\n", readme_path.c_str());
    std::exit(2);
  }
  std::fputs(readme.c_str(), f);
  std::fclose(f);
  std::printf("kernscope-tune: repro bundle written to %s (run.sh + README.md)\n",
              dir.c_str());
}

// --- todo 53: capture->tune text report -------------------------------------

void print_capture_text(const std::string &evidence_json) {
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  if (!ev) {
    std::printf("kernscope-tune: no evidence (%s)\n", err.c_str());
    return;
  }
  std::printf("Kernel: %s (%s/%s on %s)\n",
              ev->find("kernel_name") ? ev->find("kernel_name")->as_string().c_str() : "?",
              ev->find("vendor") ? ev->find("vendor")->as_string().c_str() : "?",
              ev->find("arch") ? ev->find("arch")->as_string().c_str() : "?",
              ev->find("device_name") ? ev->find("device_name")->as_string().c_str() : "?");
  const auto *best = ev->find("best");
  if (!best) {
    std::printf("kernscope-tune: %s\n",
                ev->find("message") ? ev->find("message")->as_string().c_str()
                                    : "no valid config");
    return;
  }
  const double median = best->find("timings")
                            ? best->find("timings")->find("median")->as_number()
                            : 0.0;
  std::printf("Best config: %s\n", best->find("config")->dump().c_str());
  std::printf("Time: %.4f ms (median)\n", median);
  if (const auto *sp = ev->find("speedup"))
    std::printf("Speedup vs captured baseline: %sx\n",
                fmt_num(sp->as_number()).c_str());
  if (const auto *base = ev->find("baseline")) {
    if (const auto *bt = base->find("timings"))
      std::printf("Captured baseline median: %.4f ms\n",
                  bt->find("median")->as_number());
  }
  if (const auto *cands = ev->find("candidates")) {
    std::printf("Candidates (%zu):\n", cands->as_array().size());
    for (const auto &c : cands->as_array()) {
      std::printf("  %s %s %s\n",
                  c.find("config") ? c.find("config")->dump().c_str() : "{}",
                  c.find("status") ? c.find("status")->as_string().c_str() : "?",
                  c.find("reason") ? (": " + c.find("reason")->as_string()).c_str() : "");
    }
  }
}

} // namespace

int main(int argc, char **argv) {
  kernscope::tune::DriverOptions opts;
  std::string arch_arg;
  std::string shape_arg;
  bool json = false;
  bool html = false;
  bool text = false;
  std::string repro_dir;

  // --- todo 53: capture->tune mode (--from-capture) ------------------------
  kernscope::tune::CaptureTuneOptions ctopts;
  std::string capture_blocks_arg;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-tune: %s requires an argument\n", name);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--kernel")
      opts.kernel_family = need("--kernel");
    else if (a == "--shape")
      shape_arg = need("--shape");
    else if (a == "--arch")
      arch_arg = need("--arch");
    else if (a == "--grid")
      opts.grid_preset = need("--grid");
    else if (a == "--gate-shape") {
      const std::string gs = need("--gate-shape");
      if (!parse_shape(gs, &opts.gate_shape)) {
        std::fprintf(stderr, "kernscope-tune: invalid --gate-shape '%s'\n",
                     gs.c_str());
        return 2;
      }
    } else if (a == "--db")
      opts.db_path = need("--db");
    else if (a == "--cache")
      opts.cache_path = need("--cache");
    else if (a == "--workdir")
      opts.workdir = need("--workdir");
    else if (a == "--min-occupancy") {
      if (!parse_double(need("--min-occupancy"), &opts.min_occupancy)) {
        std::fprintf(stderr, "kernscope-tune: bad --min-occupancy value\n");
        return 2;
      }
    } else if (a == "--no-bench")
      opts.live_bench = false;
    else if (a == "--no-cache" || a == "--force")
      opts.use_cache = false;
    else if (a == "--json")
      json = true;
    else if (a == "--html")
      html = true;
    else if (a == "--text")
      text = true;
    else if (a == "--emit-repro")
      repro_dir = need("--emit-repro");
    // --- todo 53 capture->tune flags --------------------------------------
    else if (a == "--from-capture")
      ctopts.bundle_path = need("--from-capture");
    else if (a == "--vendor")
      ctopts.vendor = need("--vendor");
    else if (a == "--blob-dir")
      ctopts.blob_dir = need("--blob-dir");
    else if (a == "--candidates")
      capture_blocks_arg = need("--candidates");
    else if (a == "--inject-config")
      ctopts.inject_config = need("--inject-config");
    else if (a == "--ref")
      ctopts.ref_files.push_back(need("--ref"));
    else if (a == "--reps") {
      const int v = std::atoi(need("--reps"));
      if (v <= 0) {
        std::fprintf(stderr, "kernscope-tune: --reps must be > 0\n");
        return 2;
      }
      ctopts.reps = v;
    } else if (a == "--no-flush")
      ctopts.l2_flush = false;
    else if (a == "--profile-out")
      ctopts.profile_out = need("--profile-out");
    else if (a == "--version") {
      std::printf("kernscope-tune 0.2.0\n");
      return 0;
    } else if (a == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "kernscope-tune: unknown option '%s'\n", a.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  // --- todo 53 dispatch: capture->tune mode --------------------------------
  if (!ctopts.bundle_path.empty()) {
    if (arch_arg.empty() || arch_arg == "auto")
      ctopts.arch = "gfx1101";
    else
      ctopts.arch = arch_arg;
    ctopts.kernel = opts.kernel_family; // --kernel names the launch in a bundle
    if (!opts.db_path.empty())
      ctopts.db_path = opts.db_path;
    if (ctopts.db_path.empty()) {
      // default: reports/tune/capture-<bundle-basename>.jsonl
      const std::string base =
          std::filesystem::path(ctopts.bundle_path).filename().string();
      ctopts.db_path = "reports/tune/capture-" + base + ".jsonl";
    }
    if (!capture_blocks_arg.empty()) {
      std::size_t pos = 0;
      while (pos <= capture_blocks_arg.size()) {
        const std::size_t comma = capture_blocks_arg.find(',', pos);
        const std::string tok = capture_blocks_arg.substr(
            pos, comma == std::string::npos ? std::string::npos : comma - pos);
        try {
          const int v = std::stoi(tok);
          if (v > 0)
            ctopts.candidate_blocks.push_back(v);
        } catch (...) {
        }
        if (comma == std::string::npos)
          break;
        pos = comma + 1;
      }
    }
    const int modes = (json ? 1 : 0) + (html ? 1 : 0) + (text ? 1 : 0);
    if (modes > 1) {
      std::fprintf(stderr, "kernscope-tune: --json, --html, --text are exclusive\n");
      return 2;
    }
    const kernscope::tune::CaptureTuneResult r =
        kernscope::tune::tune_from_capture(ctopts);
    if (html) {
      std::fprintf(stderr,
                   "kernscope-tune: --html is not supported for --from-capture "
                   "(use --json or --text)\n");
      return 2;
    }
    if (json) {
      std::printf("%s\n", r.evidence_json.c_str());
    } else {
      print_capture_text(r.evidence_json);
    }
    switch (r.status) {
    case kernscope::tune::CaptureTuneStatus::kOk:
      return 0;
    case kernscope::tune::CaptureTuneStatus::kNoViableConfig:
      return 1;
    default:
      std::fprintf(stderr, "kernscope-tune: %s\n", r.message.c_str());
      return 2;
    }
  }

  // --- argument validation ---------------------------------------------------
  bool known = false;
  for (const char *f : kFamilies)
    known = known || (opts.kernel_family == f);
  if (!known) {
    std::fprintf(stderr,
                 "kernscope-tune: unknown --kernel '%s' (matmul | bias_gelu | "
                 "softmax | layernorm | reduction | flash_attention)\n",
                 opts.kernel_family.c_str());
    return 3;
  }
  if (shape_arg.empty()) {
    std::fprintf(stderr, "kernscope-tune: --shape is required (M,N,K | R,C | "
                         "Nq,Nk,d)\n");
    return 2;
  }
  if (!parse_shape(shape_arg, &opts.shape)) {
    std::fprintf(stderr, "kernscope-tune: invalid --shape '%s'\n",
                 shape_arg.c_str());
    return 2;
  }
  // --arch auto = gfx1101 on this machine (the task contract).
  if (arch_arg.empty() || arch_arg == "auto")
    opts.arch = "gfx1101";
  else
    opts.arch = arch_arg;
  const int modes = (json ? 1 : 0) + (html ? 1 : 0) + (text ? 1 : 0);
  if (modes > 1) {
    std::fprintf(stderr, "kernscope-tune: --json, --html, --text are exclusive\n");
    return 2;
  }

  // --- default output paths --------------------------------------------------
  if (opts.db_path.empty()) {
    opts.db_path = "reports/tune/" + opts.kernel_family + "-" +
                   kernscope::tune::make_shape_hash(opts.kernel_family,
                                                    opts.shape) +
                   ".jsonl";
  }
  if (opts.cache_path.empty())
    opts.cache_path = opts.db_path + ".tunecache.json";

  const kernscope::tune::DriverResult r =
      kernscope::tune::tune(opts, /*hooks=*/nullptr);
  const std::string shape_csv = shape_str(opts.shape);

  if (!repro_dir.empty())
    emit_repro(r.evidence_json, opts.kernel_family, opts.arch, shape_csv,
               repro_dir, argv[0]);

  if (html) {
    print_html(r.evidence_json, opts.kernel_family, opts.arch, shape_csv);
  } else if (json) {
    std::printf("%s\n", r.evidence_json.c_str());
  } else {
    print_text(r.evidence_json, opts.kernel_family, opts.arch, shape_csv);
  }

  switch (r.status) {
  case kernscope::tune::DriverStatus::kOk:
    return 0;
  case kernscope::tune::DriverStatus::kNoViableConfig:
    return 1;
  default:
    return 2;
  }
}
