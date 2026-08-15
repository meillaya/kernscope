// kernscope-verify — the todo-35 CLI (Wave 6 closer).
//
// The full user-facing verifier: ingests CUDA/HIP source through
// libkernscope_verify (clang -x cuda|hip, host+device merged into ONE
// module by default; --device-only secondary mode), runs the core pipeline
// AND every registered check plugin (the engine dispatches each plugin on
// each kernel), prints a verdict summary + per-check JSON with source
// locations, and performs the approved-IR handoff:
//
//   kernscope-verify <src.cu|src.hip>
//       [--device-only]   degraded ingestion (device IR only -> symbolic dims)
//       [--json]          emit the full verdict document as JSON on stdout
//       [--out <base>]    approved-IR base name (default: <src> stem, CWD)
//       [--list-checks]   enumerate registered check plugins, then exit 0
//       [--budget N]      per-function explored-state cap (default 65536)
//       [--opt O1|O2]     ingestion optimization level (default O1)
//       [--arch sm_80|gfx1101]  GPU arch (default sm_80 / gfx1101)
//       [--version]
//
// Verdict model: per (kernel, check class) the CLI reports the module-owned
// verdict vocabulary (SAFE, MAY_OOB, SYMBOLIC_DIMS, DIVERGENT_BARRIER,
// BARRIER_MISMATCH, MAY_RACE, UNKNOWN_RACE, PTR_OOB, INVALID_ADDRSPACE_CAST,
// MAY_UNINIT_READ). SAFE is the only approval-class verdict: when every
// emitted verdict is SAFE the verified module is written to <out>.verified.ll
// and the process exits 0; any non-SAFE verdict (including the degraded
// SYMBOLIC_DIMS / SCEV_RECONSTRUCTION_FAILED / BUDGET_EXCEEDED engine
// verdicts) means no bound was proven, so no approved IR is emitted and the
// process exits 1. Ingestion failure exits 2 with the clang diagnostics
// forwarded verbatim (never a crash). Usage errors exit 3.
//
// The verdict vocabulary is the modules' own (embedded in their
// "verdict_json:" diagnostic payloads; the per-check corpus tests parse the
// same payloads); the CLI surfaces those claims per kernel/class instead of
// duplicating module internals, and the lit suite diffs them against the
// committed corpus manifest (tests/verify/manifest.json) — an
// anti-self-report gate: the CLI never sees the manifest.

#include <kernscope/verify/budget.h>
#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/launch_config.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace {

// CheckClass / EngineClass (diagnostic.h) are namespaces: qualified refs only.
using kernscope::verify::CheckRegistry;
using kernscope::verify::Diagnostic;
using kernscope::verify::KernelInfo;
using kernscope::verify::LaunchConfig;
using kernscope::verify::Severity;
using kernscope::verify::VerifierState;

// ---------------------------------------------------------------------------
// Args
// ---------------------------------------------------------------------------

struct Args {
  std::string source;
  bool device_only = false;
  bool json = false;
  bool list_checks = false;
  bool version = false;
  std::string out_base;
  uint64_t budget = kernscope::verify::Budget::kDefaultMaxStatesPerFunction;
  int opt_level = 1;
  std::string arch;
};

int usage(const char* argv0) {
  std::fprintf(stderr,
               "usage: %s <src.cu|src.hip> [--device-only] [--json]\n"
               "      [--out <base>] [--list-checks] [--budget N]\n"
               "      [--opt O1|O2] [--arch sm_80|gfx1101] [--version]\n"
               "  --device-only   degraded ingestion (device IR only)\n"
               "  --json          verdict document as JSON on stdout\n"
               "  --out <base>    approved-IR base (default: <src> stem in CWD)\n"
               "  --list-checks   enumerate registered check plugins\n"
               "  --budget N      per-function explored-state cap\n"
               "  --opt O1|O2     ingestion opt level (default O1)\n"
               "  --arch <arch>   GPU arch (default sm_80 / gfx1101)\n"
               "  --version\n"
               "exit codes: 0 approved (all verdicts SAFE; IR emitted)\n"
               "            1 error-class verdict (no IR emitted)\n"
               "            2 ingestion failure\n",
               argv0);
  return 3;
}

bool parse_args(int argc, char** argv, Args* out) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device-only") {
      out->device_only = true;
    } else if (arg == "--json") {
      out->json = true;
    } else if (arg == "--list-checks") {
      out->list_checks = true;
    } else if (arg == "--version") {
      out->version = true;
    } else if (arg == "--out" && i + 1 < argc) {
      out->out_base = argv[++i];
    } else if (arg == "--budget" && i + 1 < argc) {
      out->budget = std::strtoull(argv[++i], nullptr, 10);
    } else if (arg == "--opt" && i + 1 < argc) {
      const std::string level = argv[++i];
      out->opt_level = (level == "O2" || level == "2") ? 2 : 1;
    } else if (arg == "--arch" && i + 1 < argc) {
      out->arch = argv[++i];
    } else if (arg.starts_with("--")) {
      std::fprintf(stderr, "unknown option: %s\n", arg.c_str());
      return false;
    } else if (out->source.empty()) {
      out->source = arg;
    } else {
      std::fprintf(stderr, "unexpected extra argument: %s\n", arg.c_str());
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
// JSON helpers (module payloads + the CLI document use a restricted grammar:
// objects/arrays/strings/ints/bools — same design as the corpus tests' mini
// parsers).
// ---------------------------------------------------------------------------

std::string json_escape(const std::string& s) {
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
        out += c;
    }
  }
  return out;
}

/// Extracts the `"verdict":"..."` field from a module-owned "verdict_json:"
/// payload embedded in a diagnostic message. std::nullopt when the message
/// carries no payload verdict. Module-owned vocabulary; this is the same
/// channel the per-check corpus tests parse (oob/race/ptr emit payloads;
/// barrier derives its verdict from class + severity instead).
std::optional<std::string> payload_verdict(const Diagnostic& d) {
  constexpr const char* kMarker = "verdict_json: ";
  const size_t at = d.message.find(kMarker);
  if (at == std::string::npos) {
    return std::nullopt;
  }
  const std::string payload = d.message.substr(at + std::string(kMarker).size());
  // Scan for the "verdict" key (order-insensitive), then its string value.
  const size_t key = payload.find("\"verdict\"");
  if (key == std::string::npos) {
    return std::nullopt;
  }
  size_t colon = payload.find(':', key);
  if (colon == std::string::npos) {
    return std::nullopt;
  }
  const size_t quote = payload.find('"', colon);
  if (quote == std::string::npos) {
    return std::nullopt;
  }
  const size_t end = payload.find('"', quote + 1);
  if (end == std::string::npos) {
    return std::nullopt;
  }
  return payload.substr(quote + 1, end - quote - 1);
}

// ---------------------------------------------------------------------------
// Verdict model
// ---------------------------------------------------------------------------

/// One per-(kernel, check class) verdict. `verdict` uses the module-owned
/// vocabulary; SAFE is the only approval-class verdict.
struct Verdict {
  int check = 0;
  std::string class_name;
  std::string verdict;
  Severity severity = Severity::Info;
  bool error_class = false;  // verdict != "SAFE"
  std::vector<const Diagnostic*> diagnostics;
};

/// The derived verdict for a (kernel, class) whose diagnostics carry no
/// payload: barrier + engine verdicts are reconstructed from class/severity
/// (module-owned semantics; see the per-check tests).
std::string derived_verdict(int class_id, Severity worst) {
  switch (class_id) {
    case kernscope::verify::CheckClass::kDivergentBarrier:
      return "DIVERGENT_BARRIER";
    case kernscope::verify::CheckClass::kBarrier:
      return "BARRIER_MISMATCH";
    case kernscope::verify::EngineClass::kSymbolicDims:
      return "SYMBOLIC_DIMS";
    case kernscope::verify::EngineClass::kScevReconstructionFailed:
      return "SCEV_RECONSTRUCTION_FAILED";
    case kernscope::verify::EngineClass::kBudgetExceeded:
      return "BUDGET_EXCEEDED";
    default:
      break;
  }
  return worst == Severity::Error ? "ERROR" : "WARNING";
}

/// Assembles the per-(kernel, class) verdict from all diagnostics of that
/// class attributed to that kernel. Payload verdicts (oob/race/ptr) win over
/// derived ones (barrier/engine); any non-SAFE payload beats SAFE. When the
/// class had no findings at all the verdict is SAFE (the registered plugin
/// covered it and found nothing — the engine dispatched it on this kernel).
Verdict build_verdict(int class_id, std::vector<const Diagnostic*> diags) {
  Verdict v;
  v.check = class_id;
  v.class_name = kernscope::verify::check_class_name(class_id);
  v.severity = Severity::Info;
  for (const Diagnostic* d : diags) {
    if (d->severity == Severity::Error) {
      v.severity = Severity::Error;
    } else if (d->severity == Severity::Warning && v.severity != Severity::Error) {
      v.severity = Severity::Warning;
    }
  }
  std::optional<std::string> first_payload;
  std::optional<std::string> first_non_safe;
  for (const Diagnostic* d : diags) {
    if (auto pv = payload_verdict(*d)) {
      if (!first_payload) {
        first_payload = pv;
      }
      if (*pv != "SAFE") {
        first_non_safe = pv;
        break;  // worst-case payload found; no need to scan further
      }
    }
  }
  if (first_non_safe) {
    v.verdict = *first_non_safe;
  } else if (first_payload) {
    v.verdict = *first_payload;
  } else if (!diags.empty()) {
    v.verdict = derived_verdict(class_id, v.severity);
  } else {
    v.verdict = "SAFE";  // plugin covered the class on this kernel, no findings
  }
  v.error_class = v.verdict != "SAFE";
  v.diagnostics = std::move(diags);
  return v;
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

std::string tier_name(const LaunchConfig& c) {
  return kernscope::verify::launch_tier_name(c.tier);
}

/// Canonical kernel label for the verdict document: the demangled name
/// without its parameter list (llvm::demangle yields "name(int*, int)" /
/// "name()"); the manifest + per-check corpus tests use the bare form.
std::string kernel_label(const std::string& demangled) {
  const size_t paren = demangled.find('(');
  if (paren == std::string::npos) {
    return demangled;
  }
  std::string bare = demangled.substr(0, paren);
  while (!bare.empty() && (bare.back() == ' ' || bare.back() == '\t')) {
    bare.pop_back();
  }
  return bare;
}

/// Drops dangling references to erased host functions from the module's
/// annotation globals (llvm.global_ctors / llvm.global_dtors entries and
/// llvm.used / llvm.compiler.used operands). HIP modules carry a host-side
/// global constructor that the host-code projection would otherwise dangle.
void drop_dangling_annotations(
    llvm::Module& m,
    const llvm::SmallPtrSetImpl<llvm::Function*>& dropped) {
  const char* struct_names[] = {"llvm.global_ctors", "llvm.global_dtors"};
  for (const char* name : struct_names) {
    llvm::GlobalVariable* gv = m.getNamedGlobal(name);
    if (gv == nullptr || !gv->hasInitializer()) {
      continue;
    }
    auto* arr = llvm::dyn_cast<llvm::ConstantArray>(gv->getInitializer());
    if (arr == nullptr) {
      continue;
    }
    std::vector<llvm::Constant*> keep;
    for (const llvm::Use& u : arr->operands()) {
      auto* entry = llvm::dyn_cast<llvm::ConstantStruct>(u.get());
      if (entry == nullptr || entry->getNumOperands() < 2 ||
          !dropped.contains(
              llvm::dyn_cast_or_null<llvm::Function>(entry->getOperand(1)))) {
        keep.push_back(llvm::cast<llvm::Constant>(u.get()));
      }
    }
    if (keep.empty()) {
      gv->eraseFromParent();
    } else if (keep.size() != arr->getNumOperands()) {
      llvm::ArrayType* ty =
          llvm::ArrayType::get(arr->getType()->getElementType(), keep.size());
      gv->setInitializer(llvm::ConstantArray::get(ty, keep));
    }
  }
  const char* ptr_names[] = {"llvm.used", "llvm.compiler.used"};
  for (const char* name : ptr_names) {
    llvm::GlobalVariable* gv = m.getNamedGlobal(name);
    if (gv == nullptr || !gv->hasInitializer()) {
      continue;
    }
    auto* arr = llvm::dyn_cast<llvm::ConstantArray>(gv->getInitializer());
    if (arr == nullptr) {
      continue;
    }
    std::vector<llvm::Constant*> keep;
    for (const llvm::Use& u : arr->operands()) {
      if (!dropped.contains(llvm::dyn_cast_or_null<llvm::Function>(u.get()))) {
        keep.push_back(llvm::cast<llvm::Constant>(u.get()));
      }
    }
    if (keep.empty()) {
      gv->eraseFromParent();
    } else if (keep.size() != arr->getNumOperands()) {
      llvm::ArrayType* ty =
          llvm::ArrayType::get(arr->getType()->getElementType(), keep.size());
      gv->setInitializer(llvm::ConstantArray::get(ty, keep));
    }
  }
}

/// Projects the analyzed module onto its GPU code: kernels + every function
/// reachable from them (device helpers), keeping declarations. Host
/// functions (main, launch stubs, host helpers) are erased — the approved
/// artifact is the device code the proofs apply to, and the AMDGPU backend
/// rejects addrspace(0) allocas, which only host functions use.
void strip_host_code(llvm::Module& m) {
  std::vector<llvm::Function*> worklist;
  for (llvm::Function& f : m) {
    if (f.isDeclaration()) {
      continue;
    }
    const llvm::CallingConv::ID cc = f.getCallingConv();
    if (cc == llvm::CallingConv::PTX_Kernel ||
        cc == llvm::CallingConv::AMDGPU_KERNEL ||
        f.hasFnAttribute("kernel")) {
      worklist.push_back(&f);
    }
  }
  llvm::SmallPtrSet<llvm::Function*, 32> reached(worklist.begin(), worklist.end());
  while (!worklist.empty()) {
    llvm::Function* f = worklist.back();
    worklist.pop_back();
    for (llvm::BasicBlock& bb : *f) {
      for (llvm::Instruction& i : bb) {
        if (auto* call = llvm::dyn_cast<llvm::CallBase>(&i)) {
          if (llvm::Function* callee = call->getCalledFunction()) {
            if (reached.insert(callee).second && !callee->isDeclaration()) {
              worklist.push_back(callee);
            }
          }
        }
      }
    }
  }
  std::vector<llvm::Function*> drop;
  for (llvm::Function& f : m) {
    if (!f.isDeclaration() && !reached.contains(&f)) {
      drop.push_back(&f);
    }
  }
  llvm::SmallPtrSet<llvm::Function*, 32> dropped(drop.begin(), drop.end());
  // Rewrite the annotation globals BEFORE erasing: eraseFromParent() deletes
  // the Function object, and a live constant (e.g. a global_ctors entry)
  // would dangle past it.
  drop_dangling_annotations(m, dropped);
  for (llvm::Function* f : drop) {
    f->eraseFromParent();
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!parse_args(argc, argv, &args)) {
    return usage(argv[0]);
  }
  if (args.version) {
    std::printf(
        "kernscope-verify 0.1 (todo-35 CLI: core pipeline + all registered "
        "check plugins + approved-IR handoff)\n");
    return 0;
  }
  if (args.list_checks) {
    const std::vector<CheckRegistry::Entry> checks =
        CheckRegistry::instance().enumerate();
    std::printf("kernscope-verify: %zu registered check plugin(s)\n",
                checks.size());
    for (const CheckRegistry::Entry& e : checks) {
      std::printf("  %-16s %-40s classes {", e.id.c_str(), e.name.c_str());
      for (size_t j = 0; j < e.class_ids.size(); ++j) {
        if (j) {
          std::printf(", ");
        }
        std::printf("%d", e.class_ids[j]);
      }
      std::printf("}\n");
    }
    return 0;
  }
  if (args.source.empty()) {
    return usage(argv[0]);
  }

  kernscope::verify::ClangIngester ingester;
  kernscope::verify::IngestionOptions options;
  options.device_only = args.device_only;
  options.opt_level = args.opt_level;
  options.gpu_arch = args.arch;
  kernscope::verify::IngestionResult result = ingester.ingest(args.source, options);
  if (!result.success) {
    // QA negative contract: forward clang diagnostics verbatim, exit 2.
    std::fprintf(stderr, "kernscope-verify: ingestion failed (exit 2)\n");
    for (const std::string& line : result.clang_diagnostics) {
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    return 2;
  }

  kernscope::verify::VerifierOptions vopts;
  vopts.max_states_per_function = args.budget;
  vopts.ingestion_opt_level = args.opt_level;
  auto module = std::shared_ptr<llvm::Module>(std::move(result.module));
  VerifierState state(module, result.context, vopts);
  state.analyze();

  // Classes the registered plugins cover (the engine dispatches every plugin
  // on every kernel, so each covered class yields a per-kernel verdict).
  std::set<int> plugin_classes;
  for (const auto& p : CheckRegistry::instance().plugins()) {
    for (int c : p->check_class_ids) {
      plugin_classes.insert(c);
    }
  }

  // Per kernel: verdicts grouped by class, in kernel order (contiguous).
  std::vector<std::vector<Verdict>> per_kernel;
  size_t error_class_count = 0;
  const auto& kernels = state.kernels();
  for (const KernelInfo& ki : kernels) {
    std::set<int> classes = plugin_classes;
    // Engine classes appear only when the engine emitted a diagnostic for
    // this kernel (symbolic dims / SCEV / budget).
    for (const Diagnostic& d : state.diagnostics()) {
      if ((d.function == ki.name || d.function == ki.mangled_name) &&
          d.class_id >= 100) {
        classes.insert(d.class_id);
      }
    }
    std::vector<Verdict> vk;
    for (int c : classes) {
      std::vector<const Diagnostic*> diags;
      for (const Diagnostic& d : state.diagnostics()) {
        if (d.class_id == c &&
            (d.function == ki.name || d.function == ki.mangled_name)) {
          diags.push_back(&d);
        }
      }
      Verdict v = build_verdict(c, std::move(diags));
      if (v.error_class) {
        ++error_class_count;
      }
      vk.push_back(std::move(v));
    }
    per_kernel.push_back(std::move(vk));
  }

  // Summary line: SAFE | WARNINGS | ERRORS (error-class verdicts include the
  // degraded engine verdicts — nothing proven means nothing approved).
  bool any_warning = false;
  for (const Diagnostic& d : state.diagnostics()) {
    if (d.severity == Severity::Warning) {
      any_warning = true;
      break;
    }
  }
  std::string summary = error_class_count ? "ERRORS"
                        : any_warning        ? "WARNINGS"
                                             : "SAFE";

  // Approved-IR handoff: emit ONLY when no error-class verdict exists.
  std::string approved_ir;
  if (error_class_count == 0) {
    if (args.out_base.empty()) {
      std::filesystem::path src(args.source);
      args.out_base = src.stem().string();
    }
    approved_ir = args.out_base + ".verified.ll";
    strip_host_code(*module);
    std::error_code ec;
    llvm::raw_fd_ostream os(approved_ir, ec, llvm::sys::fs::OF_Text);
    if (ec) {
      std::fprintf(stderr, "kernscope-verify: cannot write %s: %s\n",
                   approved_ir.c_str(), ec.message().c_str());
      return 2;
    }
    os << *module;
    os.close();
  }

  // Emit the verdict document (JSON or human summary).
  if (args.json) {
    llvm::outs() << "{\n";
    llvm::outs() << "  \"tool\": \"kernscope-verify\",\n";
    llvm::outs() << "  \"source\": \"" << json_escape(args.source) << "\",\n";
    llvm::outs() << "  \"device_only\": " << (args.device_only ? "true" : "false")
                 << ",\n";
    llvm::outs() << "  \"summary\": \"" << summary << "\",\n";
    llvm::outs() << "  \"error_class_verdicts\": " << error_class_count << ",\n";
    const std::vector<CheckRegistry::Entry> checks =
        CheckRegistry::instance().enumerate();
    llvm::outs() << "  \"checks\": [";
    for (size_t i = 0; i < checks.size(); ++i) {
      if (i) {
        llvm::outs() << ",";
      }
      llvm::outs() << "\n    {\"id\": \"" << json_escape(checks[i].id)
                   << "\", \"name\": \"" << json_escape(checks[i].name)
                   << "\", \"classes\": [";
      for (size_t j = 0; j < checks[i].class_ids.size(); ++j) {
        if (j) {
          llvm::outs() << ", ";
        }
        llvm::outs() << checks[i].class_ids[j];
      }
      llvm::outs() << "]}";
    }
    llvm::outs() << (checks.empty() ? "],\n" : "\n  ],\n");
    llvm::outs() << "  \"kernels\": [";
    for (size_t k = 0; k < per_kernel.size(); ++k) {
      if (k) {
        llvm::outs() << ",";
      }
      llvm::outs() << "\n    {\"kernel\": \"" << json_escape(kernel_label(kernels[k].name))
                   << "\",\n"
                   << "     \"tier\": \"" << tier_name(kernels[k].config)
                   << "\",\n"
                   << "     \"launch\": \""
                   << json_escape(kernels[k].config.render()) << "\",\n"
                   << "     \"verdicts\": [";
      for (size_t j = 0; j < per_kernel[k].size(); ++j) {
        if (j) {
          llvm::outs() << ",";
        }
        const Verdict& v = per_kernel[k][j];
        llvm::outs() << "\n       {\"check\": " << v.check
                     << ", \"class\": \"" << json_escape(v.class_name)
                     << "\", \"verdict\": \"" << v.verdict
                     << "\", \"severity\": \""
                     << kernscope::verify::severity_name(v.severity)
                     << "\", \"error_class\": " << (v.error_class ? "true" : "false")
                     << ", \"diagnostics\": [";
        for (size_t di = 0; di < v.diagnostics.size(); ++di) {
          if (di) {
            llvm::outs() << ",";
          }
          const Diagnostic& d = *v.diagnostics[di];
          llvm::outs() << "\n         {\"severity\": \""
                       << kernscope::verify::severity_name(d.severity)
                       << "\", \"class\": " << d.class_id
                       << ", \"location\": \"" << json_escape(d.loc.render())
                       << "\", \"file\": \"" << json_escape(d.loc.file)
                       << "\", \"line\": " << d.loc.line
                       << ", \"col\": " << d.loc.col
                       << ", \"function\": \"" << json_escape(d.function)
                       << "\", \"instruction\": \""
                       << json_escape(d.instruction) << "\", \"message\": \""
                       << json_escape(d.message) << "\"}";
        }
        llvm::outs() << "]}";
      }
      llvm::outs() << "]\n     }";
    }
    llvm::outs() << (per_kernel.empty() ? "],\n" : "\n  ],\n");
    llvm::outs() << "  \"approved_ir\": "
                 << (approved_ir.empty()
                         ? "null"
                         : ("\"" + json_escape(approved_ir) + "\""))
                 << ",\n";
    llvm::outs() << "  \"exit_code\": " << (error_class_count ? 1 : 0) << "\n";
    llvm::outs() << "}\n";
  } else {
    std::printf("kernscope-verify: %s (%s) — %zu kernel(s)\n",
                args.source.c_str(),
                args.device_only ? "device-only" : "host+device",
                kernels.size());
    for (size_t k = 0; k < kernels.size(); ++k) {
      const KernelInfo& ki = kernels[k];
      std::printf("  kernel %s (%s)\n", kernel_label(ki.name).c_str(),
                  ki.config.render().c_str());
      for (const Verdict& v : per_kernel[k]) {
        std::printf("    [class %d %s] %s (%s)%s\n", v.check,
                    v.class_name.c_str(), v.verdict.c_str(),
                    kernscope::verify::severity_name(v.severity),
                    v.error_class ? "  <-- error-class" : "");
        for (const Diagnostic* d : v.diagnostics) {
          std::printf("      %s\n", d->render().c_str());
        }
      }
    }
    size_t total = 0;
    for (const auto& vk : per_kernel) {
      total += vk.size();
    }
    if (approved_ir.empty()) {
      std::printf(
          "kernscope-verify: SUMMARY %s — %zu verdict(s), %zu error-class; "
          "approved IR NOT emitted\n",
          summary.c_str(), total, error_class_count);
    } else {
      std::printf("kernscope-verify: SUMMARY %s — %zu verdict(s), %zu "
                  "error-class; approved IR: %s\n",
                  summary.c_str(), total, error_class_count,
                  approved_ir.c_str());
    }
  }

  return error_class_count ? 1 : 0;
}
