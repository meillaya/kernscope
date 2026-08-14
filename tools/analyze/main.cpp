//===- main.cpp - kernscope-analyze CLI -----------------------------------===//
//
// Wave-3 todo 20: the full `kernscope-analyze` command line.
//
//   usage: kernscope-analyze <file.ptx|file.s|file.isa> --arch <sm_XX|gfxXXXX>
//                            [--block N] [--json|--text|--html]
//                            [--source <file.cu|file.hip>]
//
// Input language is auto-detected (extension first, then content sniff), and
// the appropriate analysis path runs:
//   * PTX        -> libs/ptxparse (parse_ptx_module) + analyze_ptx_module
//                   (all six modules + the suggestion engine, todo 19);
//   * AMDGPU asm -> libs/amdparse (parse_amd_module, LLVM MC) + the five
//                   analysis modules with AmdModule overloads (regs,
//                   coalescing, banks, divergence, mix). The two PTX-specific
//                   analyses (occupancy + the suggestion engine) emit
//                   documented "not applicable to this input" info findings
//                   instead of crashing.
//
// Reports render through libs/report (text = the owner's Section A format,
// json = schema-validated, html = self-contained with embedded JSON).
//
// Exit codes (documented in docs/analyzer.md):
//   0  clean report (no warnings)
//   1  report produced; warnings present
//   2  error: parse failure, unknown arch, unreadable file, render schema
//      violation
//   3  usage error
//
// Launch-config discovery: when the input's directory contains manifest.json
// (the committed fixture convention, tests/fixtures/README.md), its arch /
// block_size / ptxas_v_report fields seed the analysis; explicit --arch /
// --block flags override the manifest. Without the ptxas -v register figure
// the analyzer falls back to its static declared-register proxy, which is an
// UPPER BOUND (ptxas reuses registers) — this is why the golden
// fused_attention report (102 registers, occupancy 0.375, limiter
// "registers") requires the manifest's ptxas line, exactly as the todo-19
// golden tests drive it.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "kernscope/amd.h"
#include "kernscope/analyze/analyze.h"
#include "kernscope/analyze/banks.h"
#include "kernscope/analyze/coalescing.h"
#include "kernscope/analyze/divergence.h"
#include "kernscope/analyze/mix.h"
#include "kernscope/analyze/regs.h"
#include "kernscope/arch.h"
#include "kernscope/db.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"
#include "kernscope/version.h"

namespace {

enum class ExitCode : int {
  kOk = 0,       // report clean (no warnings)
  kWarnings = 1, // report produced; warnings present
  kError = 2,    // parse failure / unknown arch / unreadable file
  kUsage = 3,    // command-line usage error
};

enum class Format { kText, kJson, kHtml };
enum class Language { kPtx, kAmdGpu };

struct Options {
  std::string input;
  std::string arch;    // --arch (required for AMDGPU input)
  int block_size = 0;  // --block (0 = not provided)
  Format format = Format::kText;
  std::string source;  // --source <file.cu|file.hip>
  bool show_help = false;
  bool show_version = false;
};

constexpr const char *kKnownArches =
    "sm_75 sm_80 sm_86 sm_89 sm_90 sm_100 sm_120 gfx90a gfx942 gfx1101";

void print_usage(std::FILE *out) {
  std::fprintf(
      out,
      "usage: kernscope-analyze <file.ptx|file.s|file.isa> [options]\n"
      "\n"
      "Static GPU-kernel analyzer: ten analyses (register pressure, occupancy\n"
      "limits + limiter, coalescing, bank conflicts, warp divergence, barrier\n"
      "usage, instruction mix, tensor-core/MFMA/WMMA usage, spill detection,\n"
      "latency bottlenecks) plus a suggestion engine, one report.\n"
      "\n"
      "Input language is auto-detected: .ptx -> PTX; .s/.isa -> AMDGPU ISA\n"
      "text; otherwise content-sniffed (.version/.target directives -> PTX).\n"
      "\n"
      "options:\n"
      "  --arch <sm_XX|gfxXXXX>  architecture (REQUIRED for AMDGPU input;\n"
      "                          optional for PTX, defaults to the .target\n"
      "                          directive; known: %s)\n"
      "  --block <N>             launch block size (occupancy input; absent\n"
      "                          -> occupancy emits its block_size_unknown\n"
      "                          diagnostic)\n"
      "  --json | --text | --html\n"
      "                          output format (default: --text; --json is\n"
      "                          schema-validated; --html is self-contained\n"
      "                          with an embedded JSON payload)\n"
      "  --source <file.cu|file.hip>\n"
      "                          CUDA/HIP source file; its stem names the\n"
      "                          kernel in the report (PTX entry names are\n"
      "                          mangled and not user-facing)\n"
      "  --version               print kernscope-analyze <version> and exit\n"
      "  --help                  print this help and exit\n"
      "\n"
      "exit codes:\n"
      "  0  clean report (no warnings)\n"
      "  1  report produced; warnings present\n"
      "  2  error (parse failure, unknown arch, unreadable file, render\n"
      "     schema violation)\n"
      "  3  usage error\n"
      "\n"
      "launch-config discovery: when the input's directory contains\n"
      "manifest.json (the committed fixture convention), its arch / block_size\n"
      "/ ptxas_v_report fields seed the analysis; explicit flags override.\n"
      "See docs/analyzer.md for the full reference.\n",
      kKnownArches);
}

// Returns an exit code when argument parsing must stop (usage error), or
// std::nullopt when parsing succeeded and execution should proceed.
std::optional<ExitCode> parse_args(Options &opt, int argc, char **argv) {
  bool format_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      opt.show_help = true;
      continue;
    }
    if (arg == "--version") {
      opt.show_version = true;
      continue;
    }
    if (arg == "--arch" || arg == "--block" || arg == "--source") {
      if (i + 1 >= argc) {
        std::fprintf(stderr,
                     "kernscope-analyze: option '%s' requires an argument\n",
                     std::string(arg).c_str());
        return ExitCode::kUsage;
      }
      const std::string value = argv[++i];
      if (arg == "--arch") {
        opt.arch = value;
      } else if (arg == "--source") {
        opt.source = value;
      } else {
        char *end = nullptr;
        const long n = std::strtol(value.c_str(), &end, 10);
        if (end == value.c_str() || *end != '\0' || n <= 0) {
          std::fprintf(stderr,
                       "kernscope-analyze: --block must be a positive integer "
                       "(got '%s')\n",
                       value.c_str());
          return ExitCode::kUsage;
        }
        opt.block_size = static_cast<int>(n);
      }
      continue;
    }
    if (arg == "--json" || arg == "--text" || arg == "--html") {
      const Format f = arg == "--json" ? Format::kJson
                       : arg == "--text" ? Format::kText
                                         : Format::kHtml;
      if (format_set && f != opt.format) {
        std::fprintf(stderr,
                     "kernscope-analyze: conflicting output formats "
                     "(--json/--text/--html are mutually exclusive)\n");
        return ExitCode::kUsage;
      }
      opt.format = f;
      format_set = true;
      continue;
    }
    if (arg.size() >= 2 && arg.front() == '-') {
      std::fprintf(stderr, "kernscope-analyze: unknown option '%s'\n",
                   std::string(arg).c_str());
      return ExitCode::kUsage;
    }
    if (!opt.input.empty()) {
      std::fprintf(stderr,
                   "kernscope-analyze: expected exactly one input file\n");
      return ExitCode::kUsage;
    }
    opt.input = std::string(arg);
  }
  if (!opt.show_help && !opt.show_version && opt.input.empty()) {
    std::fprintf(stderr, "kernscope-analyze: missing input file\n");
    return ExitCode::kUsage;
  }
  return std::nullopt;
}

std::string read_file(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

Language detect_language(const std::string &path, const std::string &text) {
  const std::string ext = std::filesystem::path(path).extension().string();
  if (ext == ".ptx") return Language::kPtx;
  if (ext == ".s" || ext == ".isa") return Language::kAmdGpu;
  // Content sniff for unknown extensions: PTX carries .version/.target
  // header directives; AMDGPU .s text does not.
  if (text.find(".version") != std::string::npos &&
      text.find(".target") != std::string::npos)
    return Language::kPtx;
  return Language::kAmdGpu;
}

int render(const Options &opt, const kernscope::KernelReport &report);

void report_unknown_arch(const std::string &arch) {
  std::fprintf(stderr,
               "kernscope-analyze: unknown arch '%s' (known arches: %s)\n",
               arch.c_str(), kKnownArches);
}

// ---------------------------------------------------------------------------
// Launch-config discovery (manifest.json in the input's directory)
// ---------------------------------------------------------------------------

struct LaunchManifest {
  bool present = false;
  std::string arch;
  int block_size = 0;
  std::string ptxas_v;
};

// Pointer-safe object lookup. NOTE: `kernscope::db::json::Value` has non-explicit
// constructors (including Value(bool)), so passing a bare `const Value*`
// where a `const Value&` is expected silently converts the pointer to `bool`
// and binds a temporary Value(true) — lookups MUST go through this wrapper
// (same trap as tools/analyze/tests/fixture_main.cpp).
const kernscope::db::json::Value *child(const kernscope::db::json::Value *obj, const char *key) {
  return obj != nullptr ? obj->find(key) : nullptr;
}

LaunchManifest load_manifest(const std::string &input_path) {
  LaunchManifest m;
  std::filesystem::path dir = std::filesystem::path(input_path).parent_path();
  if (dir.empty()) dir = ".";
  const std::filesystem::path manifest_path = dir / "manifest.json";
  std::error_code ec;
  if (!std::filesystem::exists(manifest_path, ec)) return m;
  const std::string text = read_file(manifest_path.string());
  if (text.empty()) return m;
  std::string error;
  const std::optional<kernscope::db::json::Value> parsed =
      kernscope::db::json::Value::parse(text, error);
  if (!parsed) return m;
  const kernscope::db::json::Value *arch = child(&*parsed, "arch");
  if (arch != nullptr && arch->kind() == kernscope::db::json::Kind::String)
    m.arch = arch->as_string();
  const kernscope::db::json::Value *block = child(&*parsed, "block_size");
  if (block != nullptr && block->kind() == kernscope::db::json::Kind::Number)
    m.block_size = static_cast<int>(block->as_number());
  const kernscope::db::json::Value *ptxas = child(&*parsed, "ptxas_v_report");
  if (ptxas != nullptr && ptxas->kind() == kernscope::db::json::Kind::String)
    m.ptxas_v = ptxas->as_string();
  m.present = true;
  return m;
}

// ---------------------------------------------------------------------------
// Analysis paths
// ---------------------------------------------------------------------------

std::string kernel_name_for(const Options &opt) {
  // --source wins: the CUDA/HIP source names the kernel (PTX entry names are
  // mangled); otherwise the input file stem.
  const std::string &chosen = opt.source.empty() ? opt.input : opt.source;
  return std::filesystem::path(chosen).stem().string();
}

int run_ptx(const Options &opt, const std::string &text,
            const LaunchManifest &manifest) {
  const kernscope::PtxModule module = kernscope::parse_ptx_module(text);

  // Architecture: --arch > launch manifest > the .target directive.
  std::string arch = opt.arch;
  if (arch.empty() && manifest.present) arch = manifest.arch;
  if (arch.empty()) arch = module.target;
  if (arch.empty()) {
    std::fprintf(stderr,
                 "kernscope-analyze: no architecture: the PTX has no .target "
                 "directive and no --arch was given\n");
    return static_cast<int>(ExitCode::kError);
  }
  if (!kernscope::has_arch(arch)) {
    report_unknown_arch(arch);
    return static_cast<int>(ExitCode::kError);
  }

  kernscope::analyze::AnalyzeInputs inputs;
  inputs.kernel_name = kernel_name_for(opt);
  inputs.block_size =
      opt.block_size > 0 ? opt.block_size : manifest.block_size;
  inputs.arch_override = arch;
  inputs.ptxas_v = manifest.ptxas_v;

  kernscope::KernelReport report =
      kernscope::analyze::analyze_ptx_module(module, inputs);

  // The report model requires a non-empty limiter (schema); when occupancy
  // could not be computed (no block size / no register figure) the occupancy
  // module leaves it empty, so fall back to the documented "unknown".
  if (report.limiter.empty()) report.limiter = "unknown";

  return render(opt, report);
}

// The five analysis modules with AmdModule overloads; occupancy + the
// suggestion engine are PTX-specific and emit documented not-applicable
// findings instead (never a crash).
int run_amd(const Options &opt, const std::string &text) {
  const kernscope::AmdModule module =
      kernscope::parse_amd_module(text, opt.arch, opt.input);

  kernscope::KernelReport report;
  report.kernel_name = kernel_name_for(opt);

  kernscope::analyze::analyze_regs(module, report);
  kernscope::analyze::analyze_coalescing(module, report);
  kernscope::analyze::analyze_banks(module, report);
  kernscope::analyze::analyze_divergence(module, report);
  kernscope::analyze::analyze_mix(module, report);

  kernscope::Finding occ;
  occ.id = "occupancy_not_applicable";
  occ.severity = "info";
  occ.evidence["text"] =
      "occupancy analysis is PTX-specific: register-limited occupancy needs "
      "ptxas-level register allocation, which is absent from .s ISA text "
      "(the amdhsa next_free_vgpr figure lives in code-object notes) - not "
      "applicable to this AMDGPU input";
  report.suggestions.push_back(std::move(occ));

  kernscope::Finding sug;
  sug.id = "suggestions_not_applicable";
  sug.severity = "info";
  sug.evidence["text"] =
      "suggestion engine is PTX-specific: its rules consume the PTX CFG and "
      "the occupancy limiter - not applicable to this AMDGPU input";
  report.suggestions.push_back(std::move(sug));

  report.limiter = "not-modeled";

  return render(opt, report);
}

int render(const Options &opt, const kernscope::KernelReport &report) {
  std::string out;
  try {
    switch (opt.format) {
      case Format::kText: out = kernscope::to_text(report); break;
      case Format::kJson: out = kernscope::to_json(report); break;
      case Format::kHtml: out = kernscope::to_html(report); break;
    }
  } catch (const kernscope::SchemaViolation &e) {
    std::fprintf(stderr, "kernscope-analyze: %s\n", e.what());
    return static_cast<int>(ExitCode::kError);
  }
  std::fputs(out.c_str(), stdout);
  return report.warnings.empty() ? static_cast<int>(ExitCode::kOk)
                                 : static_cast<int>(ExitCode::kWarnings);
}

}  // namespace

int main(int argc, char **argv) {
  Options opt;
  const std::optional<ExitCode> early = parse_args(opt, argc, argv);
  if (early.has_value()) return static_cast<int>(*early);

  if (opt.show_help) {
    print_usage(stdout);
    return static_cast<int>(ExitCode::kOk);
  }
  if (opt.show_version) {
    std::printf("kernscope-analyze %s\n", KERNSCOPE_VERSION_STRING);
    return static_cast<int>(ExitCode::kOk);
  }

  // Fail fast on an unknown --arch (deterministic diagnostic before any I/O).
  if (!opt.arch.empty() && !kernscope::has_arch(opt.arch)) {
    report_unknown_arch(opt.arch);
    return static_cast<int>(ExitCode::kError);
  }

  const std::string text = read_file(opt.input);
  if (text.empty()) {
    std::fprintf(stderr, "kernscope-analyze: cannot read '%s'\n",
                 opt.input.c_str());
    return static_cast<int>(ExitCode::kError);
  }

  const Language lang = detect_language(opt.input, text);
  if (lang == Language::kAmdGpu && opt.arch.empty()) {
    std::fprintf(stderr,
                 "kernscope-analyze: --arch <gfxXXXX> is required for AMDGPU "
                 "input ('%s'): ISA text carries no architecture without it\n",
                 opt.input.c_str());
    return static_cast<int>(ExitCode::kUsage);
  }

  try {
    if (lang == Language::kPtx) {
      const LaunchManifest manifest = load_manifest(opt.input);
      return run_ptx(opt, text, manifest);
    }
    return run_amd(opt, text);
  } catch (const kernscope::PtxParseError &e) {
    // what() already names line:col ("ptx parse error at line N col M: ...").
    std::fprintf(stderr, "kernscope-analyze: %s\n", e.what());
    return static_cast<int>(ExitCode::kError);
  } catch (const kernscope::AmdParseError &e) {
    if (e.line > 0) {
      std::fprintf(stderr,
                   "kernscope-analyze: amd parse error at line %u col %u "
                   "(%s): %s\n",
                   e.line, e.column, e.target_triple.c_str(), e.detail.c_str());
    } else {
      std::fprintf(stderr, "kernscope-analyze: %s\n", e.what());
    }
    return static_cast<int>(ExitCode::kError);
  } catch (const std::exception &e) {
    // Defensive: no analysis module may take the tool down unlabeled.
    std::fprintf(stderr, "kernscope-analyze: %s\n", e.what());
    return static_cast<int>(ExitCode::kError);
  }
}
