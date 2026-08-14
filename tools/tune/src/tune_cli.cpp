//===- tune_cli.cpp - kernscope-tune CLI -----------------------------------===//
//
// kernscope tools/tune todo 28: the `kernscope-tune` command-line front end
// for the correctness-gated tuning driver. One invocation tunes one
// (kernel_family, shape, arch) over the family's config grid and reports the
// best config.
//
//   kernscope-tune --kernel <family> --shape <M,N,K|R,C|Nq,Nk,d>
//                  --arch <auto|sm_XX|gfxXXXX>
//                  [--grid <auto|full>] [--json|--html] [--db <path>]
//                  [--cache <path>] [--workdir <dir>] [--min-occupancy <f>]
//                  [--no-bench] [--version] [--help]
//
//   --arch auto  -> gfx1101 (this machine's RX 7800 XT; NVIDIA runtimes are
//                  owner-gated — an explicit sm_XX runs compile-only and
//                  exits 2 because benchmarking needs a live HIP device).
//   --json       print the full JSON evidence (best config, roofline
//                  position, occupancy estimate, variants, cache key).
//   --html       print a minimal self-contained variant-comparison table
//                  (todo 29 owns the full libs/report HTML integration).
//   --db         libs/db JSONL to write (default reports/tune/...jsonl).
//   --cache      Triton-style tunecache JSON (default <db>.tunecache.json);
//                  a config already cached is skipped — re-runs compile 0.
//
// Exit codes: 0 = best config selected; 1 = no viable config (every variant
// pre-screen rejected / gate-failed — a search failure, not a crash);
// 2 = usage/environment error.
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/driver.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char *kFamilies[] = {"matmul",       "bias_gelu", "softmax",
                           "layernorm",    "reduction", "flash_attention"};

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --kernel <family> --shape <M,N,K|R,C|Nq,Nk,d>\n"
               "              --arch <auto|sm_XX|gfxXXXX> [--grid <auto|full>]\n"
               "              [--gate-shape <csv>] [--json|--html]\n"
               "              [--db <path>] [--cache <path>]\n"
               "              [--workdir <dir>] [--min-occupancy <f>]\n"
               "              [--no-bench] [--no-cache|--force] [--version]\n"
               "  family: matmul | bias_gelu | softmax | layernorm |\n"
               "          reduction | flash_attention\n"
               "  --arch auto resolves to gfx1101 on this machine\n"
               "  --no-cache skips the tunecache READ (always re-runs the\n"
               "            pipeline) but still writes the cache\n",
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

// Minimal self-contained HTML variant table (todo 29 owns the full report).
void print_html(const std::string &evidence_json) {
  std::string err;
  const auto ev = kernscope::db::json::Value::parse(evidence_json, err);
  if (!ev) {
    std::printf("<html><body><p>no evidence: %s</p></body></html>\n",
                err.c_str());
    return;
  }
  std::vector<std::pair<double, std::string>> rows;  // (median, row html)
  if (const auto *variants = ev->find("variants")) {
    for (const auto &v : variants->as_array()) {
      const auto *timings = v.find("timings");
      if (!timings)
        continue;
      const double median = timings->find("median")
                                ? timings->find("median")->as_number()
                                : 0.0;
      std::string cfg = v.find("config") ? v.find("config")->dump() : "{}";
      std::string status =
          v.find("status") ? v.find("status")->as_string() : "";
      const bool best = v.find("best") && v.find("best")->as_bool();
      char row[512];
      std::snprintf(row, sizeof(row),
                    "<tr%s><td>%s</td><td>%.4f</td><td>%s</td></tr>",
                    best ? " class=\"best\"" : "", cfg.c_str(), median,
                    status.c_str());
      rows.emplace_back(median, row);
    }
  }
  std::sort(rows.begin(), rows.end());
  std::printf("<!DOCTYPE html><html><head><style>"
              ".best{background:#dfd}td{font-family:monospace;padding:2px 8px}"
              "</style></head><body><h1>kernscope-tune variants "
              "(sorted by median)</h1><table border=1>");
  for (const auto &[median, row] : rows) {
    (void)median;
    std::printf("%s", row.c_str());
  }
  std::printf("</table></body></html>\n");
}

} // namespace

int main(int argc, char **argv) {
  kernscope::tune::DriverOptions opts;
  std::string arch_arg;
  std::string shape_arg;
  bool json = false;
  bool html = false;

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
    }     else if (a == "--no-bench")
      opts.live_bench = false;
    else if (a == "--no-cache" || a == "--force")
      opts.use_cache = false;
    else if (a == "--json")
      json = true;
    else if (a == "--html")
      html = true;
    else if (a == "--version") {
      std::printf("kernscope-tune 0.1.0\n");
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
  if (json && html) {
    std::fprintf(stderr, "kernscope-tune: --json and --html are exclusive\n");
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

  if (html) {
    print_html(r.evidence_json);
  } else if (json) {
    std::printf("%s\n", r.evidence_json.c_str());
  } else {
    std::printf("kernscope-tune: %s\n", r.message.c_str());
    if (r.has_best)
      std::printf("  best config: %s (median %.4f ms, p20 %.4f, p80 %.4f)\n",
                  r.best_config.dump().c_str(), r.best_median_ms, r.best_p20_ms,
                  r.best_p80_ms);
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
