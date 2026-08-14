// kernscope-verify-core — the todo-30 harness.
//
// Ingests CUDA/HIP source through libkernscope_verify and prints the
// per-kernel state dump + diagnostics. This is NOT the full kernscope-verify
// CLI (todo 35: verdict JSON, approved-IR handoff, committed corpus); it is
// the acceptance harness for the core pipeline:
//
//   kernscope-verify-core <src.cu|src.hip>
//       [--device-only]   degraded ingestion (device IR only -> symbolic dims)
//       [--dump-state]    print VerifierState::dumpState() after analysis
//       [--budget N]      per-function explored-state cap (default 65536)
//       [--opt O1|O2]     ingestion optimization level (default O1)
//       [--arch sm_80|gfx1101]  GPU arch (default sm_80 / gfx1101)
//       [--version]
//
// Exit codes: 0 = ingested + analyzed, no Error-severity findings;
//             1 = Error-severity findings present;
//             2 = ingestion failure (clang diagnostics forwarded verbatim).
//
// QA negative contract: a source with a syntax error must exit 2 with the
// clang diagnostic forwarded and NO verifier crash.

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string source;
  bool device_only = false;
  bool dump_state = false;
  uint64_t budget = kernscope::verify::Budget::kDefaultMaxStatesPerFunction;
  int opt_level = 1;
  std::string arch;
  bool version = false;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s <src.cu|src.hip> [--device-only] [--dump-state]\n"
               "      [--budget N] [--opt O1|O2] [--arch sm_80|gfx1101]\n"
               "  --device-only   degraded ingestion (device IR only)\n"
               "  --dump-state    print per-kernel verifier state dump\n"
               "  --budget N      per-function explored-state cap\n"
               "  --opt O1|O2     ingestion opt level (default O1)\n"
               "  --arch <arch>   GPU arch (default sm_80 / gfx1101)\n"
               "  --version\n"
               "exit codes: 0 ok | 1 error findings | 2 ingestion failure\n",
               argv0);
  return 3;
}

std::vector<std::string> parse_args(int argc, char** argv, Args* out) {
  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--device-only") {
      out->device_only = true;
    } else if (arg == "--dump-state") {
      out->dump_state = true;
    } else if (arg == "--version") {
      out->version = true;
    } else if (arg == "--budget" && i + 1 < argc) {
      out->budget = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--opt" && i + 1 < argc) {
      std::string level = argv[++i];
      out->opt_level = level == "O2" || level == "2" ? 2 : 1;
    } else if (arg == "--arch" && i + 1 < argc) {
      out->arch = argv[++i];
    } else if (arg.starts_with("--")) {
      positional.clear();
      positional.push_back("unknown option: " + arg);
      return positional;
    } else {
      positional.push_back(arg);
    }
  }
  return positional;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  std::vector<std::string> positional = parse_args(argc, argv, &args);
  if (args.version) {
    std::printf("kernscope-verify-core 0.1 (todo-30 harness; full CLI = todo 35)\n");
    return 0;
  }
  if (positional.size() != 1) {
    return usage(argv[0]);
  }
  args.source = positional[0];

  kernscope::verify::ClangIngester ingester;
  kernscope::verify::IngestionOptions options;
  options.device_only = args.device_only;
  options.opt_level = args.opt_level;
  options.gpu_arch = args.arch;
  kernscope::verify::IngestionResult result = ingester.ingest(args.source, options);
  if (!result.success) {
    // QA negative contract: forward the clang diagnostics verbatim, exit 2.
    std::fprintf(stderr, "kernscope-verify-core: ingestion failed (exit 2)\n");
    for (const std::string& line : result.clang_diagnostics) {
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    return 2;
  }

  kernscope::verify::VerifierOptions vopts;
  vopts.max_states_per_function = args.budget;
  vopts.ingestion_opt_level = args.opt_level;
  auto module = std::shared_ptr<llvm::Module>(std::move(result.module));
  kernscope::verify::VerifierState state(module, result.context, vopts);
  state.analyze();

  if (args.dump_state) {
    std::printf("%s\n", state.dumpState().c_str());
  }
  const auto& diags = state.diagnostics();
  if (!args.dump_state) {
    for (const auto& d : diags) {
      std::printf("%s\n", d.render().c_str());
    }
  }
  std::printf("kernscope-verify-core: %zu kernel(s), %zu diagnostic(s)\n",
              state.kernels().size(), diags.size());
  for (const auto& d : diags) {
    if (d.severity == kernscope::verify::Severity::Error) {
      return 1;
    }
  }
  return 0;
}
