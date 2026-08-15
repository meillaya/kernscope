//===- main.cpp - kernscope-profile CLI (todo 52) --------------------------===//
//
// Wave-9 todo 52: the full `kernscope-profile` command line — the profiler
// track closer. One command runs the whole todo-49/50/51 stack:
//
//   usage: kernscope-profile <binary|dir> [options]
//
//   1. TRACE   — run the target under the todo-49 tracer (LD_PRELOAD of the
//                rocprofiler-sdk tool for HIP; CUDA runtime tracing is
//                rental-gated, compile+unit only locally);
//   2. ENRICH  — parse the todo-50 enriched JSONL records back into the
//                per-kernel model (occupancy via libs/occupancy, code-object
//                vgpr/sgpr/LDS facts — carried through verbatim, never
//                recomputed or fabricated);
//   3. RANK    — todo-51 hot-kernel ranking (total/median, % of wall time);
//   4. DIAGNOSE— todo-51 rules engine -> libs/report Findings with evidence;
//   5. REPORT  — libs/report reuse: --json = the profile report model,
//                --text = the owner's Section B shape
//                (docs/research/owner-examples.md: Kernel/Time/Registers/
//                Shared/Limiter/Hint) with LOCALLY-MEASURED values, --html =
//                kernscope::to_html() of the hottest kernel (embedded JSON
//                payload, self-contained).
//
// The target argument is a binary path OR a directory (the committed
// fixture build dir — the CLI finds layernorm_app inside — or a capture
// repro-dir). `--` separates app arguments from CLI arguments.
//
// Exit codes (documented in docs/profile.md):
//   0  report produced (INCLUDING the "no kernels traced" negative: a target
//      that runs fine but traces zero kernels is a valid report, not an
//      error)
//   2  error (target missing/unrunnable, tracer missing, vendor unavailable
//      locally, render schema violation)
//   3  usage error
//
// MUST NOT: modify libs/* or the tools/* of other tracks; claim numbers the
// local run did not measure; auto-run on NVIDIA (rental-gated).
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "cli_pipeline.h"
#include "kernscope/profile/diagnostics.h"
#include "kernscope/report.h"
#include "kernscope/version.h"

#ifndef PROFILE_HIP_TOOL
#define PROFILE_HIP_TOOL ""
#endif

#ifndef KERNSCOPE_ANALYZE_CLI
#define KERNSCOPE_ANALYZE_CLI "kernscope-analyze"
#endif

namespace {

enum class ExitCode : int {
  kOk = 0,    // report produced (incl. no-kernels negative)
  kError = 2, // trace/render error
  kUsage = 3, // command-line usage error
};

enum class Format { kText, kJson, kHtml };

struct Options {
  std::string target;
  std::string vendor = "auto"; // auto | hip | cuda
  Format format = Format::kText;
  std::vector<std::string> app_args; // after --
  bool show_help = false;
  bool show_version = false;
};

void print_usage(std::FILE *out) {
  std::fprintf(
      out,
      "usage: kernscope-profile <binary|dir> [options]\n"
      "\n"
      "Tiny CUDA/ROCm profiler: trace a target binary under the kernel\n"
      "tracer, enrich every kernel record (occupancy, registers, shared\n"
      "memory), rank the hot kernels and emit compiler-aware \"why is this\n"
      "slow?\" diagnostics — one report.\n"
      "\n"
      "The target is a binary path or a directory containing the binary\n"
      "(the committed layernorm_fused fixture build dir finds layernorm_app\n"
      "inside). For the layernorm fixture the CLI seeds the app's input in\n"
      "its scratch dir (tests/fixtures/README.md contract).\n"
      "\n"
      "options:\n"
      "  --vendor <auto|hip|cuda>  tracing vendor (default: auto — detected\n"
      "                            from the binary; CUDA runtime tracing is\n"
      "                            rental-gated, compile+unit only locally)\n"
      "  --json | --text | --html  output format (default: --text; --json\n"
      "                            is the machine-readable report, --html is\n"
      "                            libs/report's self-contained page with an\n"
      "                            embedded JSON payload)\n"
      "  --                        pass everything after it to the target\n"
      "                            binary as its arguments\n"
      "  --version                 print kernscope-profile <version> and exit\n"
      "  --help                    print this help and exit\n"
      "\n"
      "exit codes:\n"
      "  0  report produced (INCLUDING \"no kernels traced\" — a target that\n"
      "     runs fine but traces zero kernels is a valid report)\n"
      "  2  error (missing/unrunnable target, tracer not built, vendor\n"
      "     unavailable locally, render schema violation)\n"
      "  3  usage error\n"
      "\n"
      "environment:\n"
      "  KERNSCOPE_PROFILE_HIP_TOOL  override the HIP tracer tool path\n"
      "\n"
      "See docs/profile.md for the full reference.\n");
}

std::optional<ExitCode> parse_args(Options &opt, int argc, char **argv) {
  bool format_set = false;
  bool after_dashdash = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (after_dashdash) {
      opt.app_args.push_back(arg);
      continue;
    }
    if (arg == "--") {
      after_dashdash = true;
      continue;
    }
    if (arg == "--help" || arg == "-h") {
      opt.show_help = true;
      continue;
    }
    if (arg == "--version") {
      opt.show_version = true;
      continue;
    }
    if (arg == "--vendor") {
      if (i + 1 >= argc) {
        std::fprintf(stderr,
                     "kernscope-profile: option '--vendor' requires an "
                     "argument (auto|hip|cuda)\n");
        return ExitCode::kUsage;
      }
      opt.vendor = argv[++i];
      if (opt.vendor != "auto" && opt.vendor != "hip" &&
          opt.vendor != "cuda") {
        std::fprintf(stderr,
                     "kernscope-profile: --vendor must be auto|hip|cuda "
                     "(got '%s')\n",
                     opt.vendor.c_str());
        return ExitCode::kUsage;
      }
      continue;
    }
    if (arg == "--json" || arg == "--text" || arg == "--html") {
      const Format f = arg == "--json" ? Format::kJson
                       : arg == "--text" ? Format::kText
                                         : Format::kHtml;
      if (format_set && f != opt.format) {
        std::fprintf(stderr,
                     "kernscope-profile: conflicting output formats "
                     "(--json/--text/--html are mutually exclusive)\n");
        return ExitCode::kUsage;
      }
      opt.format = f;
      format_set = true;
      continue;
    }
    if (arg.size() >= 2 && arg.front() == '-') {
      std::fprintf(stderr, "kernscope-profile: unknown option '%s'\n",
                   arg.c_str());
      return ExitCode::kUsage;
    }
    if (!opt.target.empty()) {
      std::fprintf(stderr,
                   "kernscope-profile: expected exactly one target\n");
      return ExitCode::kUsage;
    }
    opt.target = arg;
  }
  if (!opt.show_help && !opt.show_version && opt.target.empty()) {
    std::fprintf(stderr, "kernscope-profile: missing target (binary or dir)\n");
    return ExitCode::kUsage;
  }
  return std::nullopt;
}

std::string hip_tool_path() {
  const char *env = std::getenv("KERNSCOPE_PROFILE_HIP_TOOL");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  return PROFILE_HIP_TOOL;
}

int render(const Options &opt, const kernscope::profile::cli::ProfileReport &report) {
  std::string out;
  try {
    switch (opt.format) {
    case Format::kText:
      out = kernscope::profile::cli::report_to_text(report);
      break;
    case Format::kJson:
      out = kernscope::profile::cli::report_to_json(report);
      break;
    case Format::kHtml: {
      // libs/report reuse (todo 7): the showcase kernel's report model, HTML
      // with the embedded JSON payload (parses via tests/report/check_html.py
      // extraction).
      const std::size_t hot_index =
          kernscope::profile::cli::showcase_kernel_index(report);
      const kernscope::KernelReport model =
          kernscope::profile::cli::build_kernel_report_model(report,
                                                             hot_index);
      out = kernscope::to_html(model);
      break;
    }
    }
  } catch (const kernscope::SchemaViolation &e) {
    std::fprintf(stderr, "kernscope-profile: %s\n", e.what());
    return static_cast<int>(ExitCode::kError);
  }
  std::fputs(out.c_str(), stdout);
  return static_cast<int>(ExitCode::kOk);
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  const std::optional<ExitCode> early = parse_args(opt, argc, argv);
  if (early.has_value()) {
    return static_cast<int>(*early);
  }

  if (opt.show_help) {
    print_usage(stdout);
    return static_cast<int>(ExitCode::kOk);
  }
  if (opt.show_version) {
    std::printf("kernscope-profile %s\n", KERNSCOPE_VERSION_STRING);
    return static_cast<int>(ExitCode::kOk);
  }

  kernscope::profile::cli::TraceOptions topts;
  topts.target = opt.target;
  topts.vendor = opt.vendor;
  if (topts.vendor == "auto") {
    std::string error;
    const std::optional<std::string> resolved =
        kernscope::profile::cli::resolve_target(opt.target, &error);
    if (!resolved) {
      std::fprintf(stderr, "kernscope-profile: %s\n", error.c_str());
      return static_cast<int>(ExitCode::kError);
    }
    topts.vendor = kernscope::profile::cli::detect_vendor(*resolved);
  }
  topts.hip_tool = hip_tool_path();
  topts.arch_name = topts.vendor == "cuda" ? "sm_80" : "gfx1101";
  topts.analyzer_cli = KERNSCOPE_ANALYZE_CLI;
  topts.app_args = opt.app_args;

  const kernscope::profile::cli::TraceResult trace =
      kernscope::profile::cli::run_trace(topts);
  if (!trace.ok) {
    std::fprintf(stderr, "kernscope-profile: %s\n", trace.error.c_str());
    return static_cast<int>(ExitCode::kError);
  }

  const kernscope::profile::cli::ProfileReport report =
      kernscope::profile::cli::build_report(trace.kernels, topts);
  if (report.no_kernels_traced) {
    std::fprintf(stderr,
                 "[kernscope-profile] info: %zu API record(s), zero kernel "
                 "records — \"no kernels traced\" (exit 0, not an error)\n",
                 trace.api_records);
  }
  return render(opt, report);
}
