//===- pass_report_cli.cpp - kernscope-pass-report CLI (todo 54) -----------===//
//
// The pass->analyzer delta-report loop (the plan's `kernscope-pass
// --report-pipeline <kernel>`): re-runs the todo-42 pipeline (clang ->
// before.ll, the kernscope-pass plugin pipeline -> after.ll, llc ->
// before.ptx/after.ptx) on ONE showcase kernel, then runs kernscope-analyze
// (todo 20, INVOKED — never modified) on the before/after PTX and emits a
// DELTA report (registers estimate, smem bytes, coalescing class,
// bank-conflict way, vector width) via libs/report.
//
//   usage: kernscope-pass-report --report-pipeline <gemm|layernorm|softmax>
//                                [--build-dir <dir>] [--workdir <dir>]
//                                [--json] [--out <path>] [--help] [--version]
//
//   --report-pipeline <kernel>  the todo-42 showcase kernel to run
//   --workdir                   scratch dir (default: temp dir)
//   --json                      delta report as JSON on stdout (default)
//   --out <path>                also write the JSON to a file
//
// Exit codes: 0 = report emitted; 2 = pipeline/analyze failure; 3 = usage.
//
// The DELTA fields:
//   registers_estimate  kernscope-analyze metrics.registers_per_thread
//   smem_bytes          static .shared array bytes parsed from the PTX
//   coalescing_class    the analyzer's coalescing finding class + the
//                       pass-level per-access class counts
//                       (coalesce_before/after.txt)
//   bank_conflict_way   kernscope-analyze metrics.shared_bank_way
//   vector_width        the vectorize pass report (vectorized loads /
//                       chains); the analyzer's vectorized_ld_global count
//                       is the PTX-level evidence
// A metric is "improved" when the after value is strictly better (vector
// width / vectorized loads up, registers / bank-conflict way / uncoalesced
// accesses down). The no-transform showcase (softmax) records its honest
// no_transform: true instead of fabricating improvements.
//
//===----------------------------------------------------------------------===//

#include "kernscope/db.h"
#include "kernscope/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#ifndef KERNSCOPE_TUNE_REPO_ROOT
#define KERNSCOPE_TUNE_REPO_ROOT ""
#endif

namespace {

using kernscope::db::json::Value;

constexpr const char *kKernels[] = {"gemm", "layernorm", "softmax"};

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --report-pipeline <gemm|layernorm|softmax>\n"
               "      [--build-dir <dir>] [--workdir <dir>] [--json]\n"
               "      [--out <path>] [--help] [--version]\n",
               argv0);
}

std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool write_file(const std::string &path, const std::string &content) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f)
    return false;
  f << content;
  return static_cast<bool>(f);
}

// Runs `cmd` with cwd `dir`, forwarding stdout; returns the exit code.
int run_command(const std::string &dir, const std::string &cmd) {
  const std::string full = "cd " + dir + " && " + cmd;
  const int status = std::system(full.c_str());
  if (status == -1 || !WIFEXITED(status))
    return -1;
  return WEXITSTATUS(status);
}

std::string exe_dir(const char *argv0) {
  const std::string s = argv0 ? argv0 : "";
  const std::size_t slash = s.find_last_of('/');
  if (slash == std::string::npos)
    return ".";
  if (slash == 0)
    return "/";
  return s.substr(0, slash);
}

// Parses a JSON document from a subprocess stdout (the analyze CLI prints
// only the report to stdout).
std::optional<Value> parse_json_stdout(const std::string &out) {
  const std::size_t brace = out.find('{');
  if (brace == std::string::npos)
    return std::nullopt;
  std::string err;
  auto v = Value::parse(out.substr(brace), err);
  if (!v)
    return std::nullopt;
  return v;
}

// --- delta-field extractors --------------------------------------------------

// analyzer metrics.registers_per_thread (nullopt when absent).
std::optional<double> analyze_regs(const Value &doc) {
  const Value *m = doc.find("metrics");
  const Value *r = m ? m->find("registers_per_thread") : nullptr;
  return r ? std::optional<double>(r->as_number()) : std::nullopt;
}

// analyzer metrics.shared_bank_way (nullopt when the kernel has no shared
// memory accesses — the honest n/a).
std::optional<double> analyze_bank_way(const Value &doc) {
  const Value *m = doc.find("metrics");
  const Value *b = m ? m->find("shared_bank_way") : nullptr;
  return b ? std::optional<double>(b->as_number()) : std::nullopt;
}

// analyzer metrics.vectorized_ld_global (the metric key is
// ld_global_vectorized; the coalescing finding evidence carries the same
// value as a string when the metric is absent).
std::optional<double> analyze_vectorized_ld(const Value &doc) {
  const Value *m = doc.find("metrics");
  if (m) {
    const Value *v = m->find("ld_global_vectorized");
    if (v)
      return v->as_number();
  }
  const Value *ws = doc.find("warnings");
  if (ws) {
    for (const Value &w : ws->as_array()) {
      const Value *id = w.find("id");
      const std::string s = id ? id->as_string() : "";
      if (s == "coalescing" || s == "coalescing_unknown") {
        const Value *ev = w.find("evidence");
        const Value *vl = ev ? ev->find("vectorized_ld_global") : nullptr;
        if (vl)
          return std::atof(vl->as_string().c_str());
      }
    }
  }
  return std::nullopt;
}

// The analyzer's coalescing verdict class (finding id coalescing /
// coalescing_unknown; "none" when no global accesses were analyzed).
std::string analyze_coalescing_class(const Value &doc) {
  std::string cls;
  const Value *ws = doc.find("warnings");
  if (ws) {
    for (const Value &w : ws->as_array()) {
      const Value *id = w.find("id");
      const std::string s = id ? id->as_string() : "";
      if (s == "coalescing" || s == "coalescing_unknown")
        cls = s;
    }
  }
  if (cls.empty()) {
    const Value *m = doc.find("metrics");
    if (m && m->find("ld_global"))
      cls = "coalesced";
  }
  return cls.empty() ? "none" : cls;
}

// Static shared-memory bytes from a PTX file (.shared array declarations).
double ptx_smem_bytes(const std::string &ptx) {
  double total = 0.0;
  std::size_t pos = 0;
  while ((pos = ptx.find(".shared", pos)) != std::string::npos) {
    const std::size_t eol = ptx.find('\n', pos);
    const std::string line = ptx.substr(pos, eol - pos);
    // .shared .align 4 .b32 arr[1024];
    std::size_t b = line.find(".b");
    std::size_t o = line.find('[');
    std::size_t c = line.find(']', o);
    if (b != std::string::npos && o != std::string::npos && c != std::string::npos) {
      const int bits = std::atoi(line.substr(b + 2, o - b - 2).c_str());
      const double count = std::atof(line.substr(o + 1, c - o - 1).c_str());
      if (bits > 0 && count > 0)
        total += count * (bits / 8.0);
    }
    pos = eol == std::string::npos ? ptx.size() : eol + 1;
  }
  return total;
}

// Vector width from the vectorize pass report line
// ("kernscope-vectorize: <k>: vectorized N loads, S stores (C chains)");
// 1.0 when the pass did not transform.
double pass_vector_width(const std::string &reports) {
  const std::size_t at = reports.find("vectorized ");
  if (at == std::string::npos)
    return 1.0;
  const std::size_t loads_end = reports.find(" loads", at);
  if (loads_end == std::string::npos)
    return 1.0;
  const double loads =
      std::atof(reports.substr(at + 10, loads_end - at - 10).c_str());
  const std::size_t chains_at = reports.find('(', loads_end);
  const std::size_t chains_end = chains_at == std::string::npos
                                     ? std::string::npos
                                     : reports.find(" chains", chains_at);
  if (chains_at == std::string::npos || chains_end == std::string::npos)
    return 1.0;
  const double chains = std::atof(
      reports.substr(chains_at + 1, chains_end - chains_at - 1).c_str());
  if (loads <= 0.0 || chains <= 0.0)
    return 1.0;
  return loads / chains;
}

// Per-access coalescing class counts from a coalesce pass report line
// ("N global accesses (A fully, B partial, C uncoalesced, D unknown)").
bool parse_coalesce_counts(const std::string &text, double *uncoalesced_out) {
  const std::size_t open = text.find('(');
  if (open == std::string::npos)
    return false;
  const std::size_t close = text.find(')', open);
  if (close == std::string::npos)
    return false;
  std::string inner = text.substr(open + 1, close - open - 1);
  double uncoalesced = 0;
  std::size_t pos = 0;
  int idx = 0;
  while (pos <= inner.size()) {
    const std::size_t comma = inner.find(',', pos);
    const std::string tok =
        inner.substr(pos, comma == std::string::npos ? std::string::npos
                                                     : comma - pos);
    const double n = std::atof(tok.c_str());
    if (idx == 2)
      uncoalesced = n;
    ++idx;
    if (comma == std::string::npos)
      break;
    pos = comma + 1;
  }
  if (idx < 4)
    return false;
  *uncoalesced_out = uncoalesced;
  return true;
}

bool transformed(const std::string &reports) {
  std::istringstream ss(reports);
  std::string line;
  while (std::getline(ss, line)) {
    const bool is_pass = line.rfind("kernscope-tile:", 0) == 0 ||
                         line.rfind("kernscope-promote:", 0) == 0 ||
                         line.rfind("kernscope-vectorize:", 0) == 0 ||
                         line.rfind("kernscope-pad:", 0) == 0;
    if (is_pass && line.find("declining") == std::string::npos &&
        line.find("no ") == std::string::npos)
      return true;
  }
  return false;
}

// --- the delta report (libs/report model + before/after/delta) --------------

Value build_delta(const std::string &kernel, const Value &before_doc,
                  const Value &after_doc, const std::string &pass_reports,
                  const std::string &coalesce_before,
                  const std::string &coalesce_after,
                  const std::string &ptx_before,
                  const std::string &ptx_after) {
  const double regs_b = analyze_regs(before_doc).value_or(-1);
  const double regs_a = analyze_regs(after_doc).value_or(-1);
  const double way_b = analyze_bank_way(before_doc).value_or(-1);
  const double way_a = analyze_bank_way(after_doc).value_or(-1);
  const double vecld_b = analyze_vectorized_ld(before_doc).value_or(-1);
  const double vecld_a = analyze_vectorized_ld(after_doc).value_or(-1);
  const double smem_b = ptx_smem_bytes(ptx_before);
  const double smem_a = ptx_smem_bytes(ptx_after);
  const double vw_before = 1.0;  // no vectorization before the pipeline
  const double vw_after = pass_vector_width(pass_reports);
  double unc_b = -1;
  double unc_a = -1;
  parse_coalesce_counts(coalesce_before, &unc_b);
  parse_coalesce_counts(coalesce_after, &unc_a);

  std::vector<std::string> improved;
  if (regs_a >= 0 && regs_b >= 0 && regs_a < regs_b)
    improved.push_back("registers_estimate");
  if (way_a >= 0 && way_b >= 0 && way_a < way_b)
    improved.push_back("bank_conflict_way");
  if (vecld_a > vecld_b)
    improved.push_back("vectorized_ld_global");
  if (vw_after > vw_before)
    improved.push_back("vector_width");
  if (unc_a >= 0 && unc_b >= 0 && unc_a < unc_b)
    improved.push_back("coalescing");

  auto metric = [](const char *name, double v) {
    return Value::Member{name,
                         v < 0 ? Value() : Value(v)};  // null when n/a
  };
  std::vector<Value::Member> before_members{
      metric("registers_estimate", regs_b),
      {"smem_bytes", Value(smem_b)},
      {"coalescing_class", Value(analyze_coalescing_class(before_doc))},
      metric("bank_conflict_way", way_b),
      {"vector_width", Value(vw_before)},
      metric("vectorized_ld_global", vecld_b),
  };
  std::vector<Value::Member> after_members{
      metric("registers_estimate", regs_a),
      {"smem_bytes", Value(smem_a)},
      {"coalescing_class", Value(analyze_coalescing_class(after_doc))},
      metric("bank_conflict_way", way_a),
      {"vector_width", Value(vw_after)},
      metric("vectorized_ld_global", vecld_a),
  };
  std::vector<Value::Member> delta_members{
      metric("registers_estimate_delta", regs_a >= 0 && regs_b >= 0 ? regs_a - regs_b : -1),
      {"smem_bytes_delta", Value(smem_a - smem_b)},
      metric("bank_conflict_way_delta", way_a >= 0 && way_b >= 0 ? way_a - way_b : -1),
      {"vector_width_delta", Value(vw_after - vw_before)},
      metric("vectorized_ld_global_delta", vecld_a >= 0 && vecld_b >= 0 ? vecld_a - vecld_b : -1),
  };
  if (unc_a >= 0 && unc_b >= 0)
    delta_members.emplace_back("uncoalesced_delta", Value(unc_a - unc_b));
  std::vector<Value> improved_arr;
  improved_arr.reserve(improved.size());
  for (const std::string &i : improved)
    improved_arr.emplace_back(i);
  delta_members.emplace_back("improved_metrics", Value(std::move(improved_arr)));
  delta_members.emplace_back("no_transform", Value(!transformed(pass_reports)));

  return Value(std::vector<Value::Member>{
      {"schema_version", Value("1.0")},
      {"report_type", Value("kernscope_report")},
      {"tool", Value("kernscope-pass-report")},
      {"kernel", Value(std::vector<Value::Member>{
                     {"name", Value(kernel)}, {"arch", Value("sm_80")}})},
      {"before", Value(std::move(before_members))},
      {"after", Value(std::move(after_members))},
      {"delta", Value(std::move(delta_members))},
      {"provenance",
       Value(std::vector<Value::Member>{
           {"mode", Value("projected")},
           {"note",
            Value("static IR/PTX artifacts (llc compile-only); analyzer "
                  "deltas on before/after PTX; measured numbers are the "
                  "gfx1101 analog in the libs/db records")}})},
  });
}

}  // namespace

int main(int argc, char **argv) {
  std::string kernel;
  std::string build_dir;
  std::string workdir;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-pass-report: %s requires an argument\n",
                     name);
        std::exit(3);
      }
      return argv[++i];
    };
    if (a == "--report-pipeline")
      kernel = need("--report-pipeline");
    else if (a == "--build-dir")
      build_dir = need("--build-dir");
    else if (a == "--workdir")
      workdir = need("--workdir");
    else if (a == "--out")
      out_path = need("--out");
    else if (a == "--json") {
      // default output mode; accepted for symmetry
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else if (a == "--version") {
      std::printf("kernscope-pass-report 0.1.0 (todo 54)\n");
      return 0;
    } else {
      std::fprintf(stderr, "kernscope-pass-report: unknown option '%s'\n",
                   a.c_str());
      usage(argv[0]);
      return 3;
    }
  }

  bool known = false;
  for (const char *k : kKernels)
    known = known || (kernel == k);
  if (!known) {
    std::fprintf(stderr,
                 "kernscope-pass-report: --report-pipeline needs one of "
                 "gemm | layernorm | softmax\n");
    return 3;
  }

  const std::string repo_root = KERNSCOPE_TUNE_REPO_ROOT;
  if (repo_root.empty()) {
    std::fprintf(stderr,
                 "kernscope-pass-report: repo root not resolved at "
                 "configure time\n");
    return 2;
  }
  if (build_dir.empty())
    build_dir = repo_root + "/build";
  std::error_code abs_ec;
  build_dir = std::filesystem::absolute(build_dir, abs_ec).string();
  if (workdir.empty()) {
    char tmp[] = "/tmp/kernscope-pass-report-XXXXXX";
    if (!mkdtemp(tmp)) {
      std::fprintf(stderr, "kernscope-pass-report: cannot create workdir\n");
      return 2;
    }
    workdir = tmp;
  } else {
    workdir = std::filesystem::absolute(workdir, abs_ec).string();
    std::error_code ec;
    std::filesystem::create_directories(workdir, ec);
  }

  // 1. the todo-42 pipeline (script; showcase kernels + plugin read-only).
  // The script's own stdout is captured into pipeline.log — the tool's
  // stdout must carry ONLY the delta report JSON.
  const std::string script =
      repo_root + "/tools/tune/tests/run_analyze_pipeline.sh";
  const std::string pipe_cmd = "bash " + script + " " + kernel + " " +
                               build_dir + " " + workdir + " " + repo_root +
                               " > pipeline.log 2>&1";
  const int pipe_rc = run_command(workdir, pipe_cmd);
  if (pipe_rc != 0) {
    std::fprintf(stderr,
                 "kernscope-pass-report: pipeline failed (rc=%d): %s\n",
                 pipe_rc, pipe_cmd.c_str());
    return 2;
  }

  // 2. kernscope-analyze (INVOKED) on the normalized before/after PTX.
  std::error_code ec;
  const std::string analyze_bin =
      (std::filesystem::absolute(exe_dir(argv[0]), ec) / ".." / ".." / "bin" /
       "kernscope-analyze")
          .lexically_normal()
          .string();
  const std::string ptx_before = workdir + "/before_norm.ptx";
  const std::string ptx_after = workdir + "/after_norm.ptx";
  auto run_analyze = [&](const std::string &ptx) -> std::string {
    const std::string cmd = "cd " + workdir + " && " + analyze_bin + " " +
                            ptx + " --arch sm_80 --json 2>&1";
    FILE *p = ::popen(cmd.c_str(), "r");
    if (!p)
      return "";
    std::string o;
    char buf[8192];
    std::size_t r = 0;
    while ((r = std::fread(buf, 1, sizeof(buf), p)) > 0)
      o.append(buf, r);
    (void)::pclose(p);
    return o;
  };
  const std::string before_json = run_analyze(ptx_before);
  const std::string after_json = run_analyze(ptx_after);
  const auto before_doc = parse_json_stdout(before_json);
  const auto after_doc = parse_json_stdout(after_json);
  if (!before_doc || !after_doc) {
    std::fprintf(stderr,
                 "kernscope-pass-report: kernscope-analyze failed on the "
                 "before/after PTX (before ok=%d after ok=%d)\n",
                 before_doc.has_value(), after_doc.has_value());
    return 2;
  }

  // 3. the delta report via libs/report conventions.
  const std::string reports = read_file(workdir + "/pass_reports.txt");
  const std::string cb = read_file(workdir + "/coalesce_before.txt");
  const std::string ca = read_file(workdir + "/coalesce_after.txt");
  const Value delta = build_delta(kernel, *before_doc, *after_doc, reports, cb,
                                  ca, ptx_before, ptx_after);
  const std::string json = delta.dump() + "\n";
  if (!out_path.empty() && !write_file(out_path, json)) {
    std::fprintf(stderr, "kernscope-pass-report: cannot write %s\n",
                 out_path.c_str());
    return 2;
  }
  std::printf("%s", json.c_str());
  return 0;
}
