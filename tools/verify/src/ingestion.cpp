#include <kernscope/verify/ingestion.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/DiagnosticPrinter.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/Path.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <unistd.h>

namespace kernscope {
namespace verify {

using llvm::DiagnosticHandler;
using llvm::DiagnosticInfo;
using llvm::DiagnosticPrinterRawOStream;
using llvm::LLVMContext;
using llvm::Module;
using llvm::StringRef;

namespace {

bool ends_with_ci(const std::string& s, const char* suffix) {
  size_t n = std::strlen(suffix);
  if (s.size() < n) {
    return false;
  }
  for (size_t i = 0; i < n; ++i) {
    if (std::tolower(static_cast<unsigned char>(s[s.size() - n + i])) !=
        std::tolower(static_cast<unsigned char>(suffix[i]))) {
      return false;
    }
  }
  return true;
}

// Captures module-level diagnostics (e.g. llvm-link data-layout warnings)
// so they never print to the process stderr unannounced.
class CaptureDiagHandler : public DiagnosticHandler {
 public:
  bool handleDiagnostics(const DiagnosticInfo& di) override {
    std::string text;
    llvm::raw_string_ostream os(text);
    DiagnosticPrinterRawOStream printer(os);
    di.print(printer);
    os.flush();
    messages.push_back(text);
    return true;
  }
  std::vector<std::string> messages;
};

// Runs `program` with `args`, capturing stdout/stderr into temp files that
// are read back and removed. Returns the process exit code (or -1 when the
// program could not be launched; *errmsg is set then).
int run_tool(const std::string& program, const std::vector<std::string>& args,
             std::string* stdout_out, std::string* stderr_out,
             std::string* errmsg) {
  llvm::SmallVector<StringRef, 24> argv;
  argv.push_back(program);
  for (const std::string& a : args) {
    argv.push_back(a);
  }
  namespace fs = std::filesystem;
  static std::atomic<uint64_t> seq{0};
  std::string tag = std::to_string(::getpid()) + "." + std::to_string(seq.fetch_add(1));
  fs::path out_path = fs::temp_directory_path() / ("kernscope-out." + tag);
  fs::path err_path = fs::temp_directory_path() / ("kernscope-err." + tag);
  {
    std::ofstream touch_out(out_path);
    std::ofstream touch_err(err_path);
  }

  const std::string out_path_str = out_path.string();
  const std::string err_path_str = err_path.string();
  std::array<std::optional<StringRef>, 3> redirects = {
      StringRef("/dev/null"), StringRef(out_path_str), StringRef(err_path_str)};
  int rc = llvm::sys::ExecuteAndWait(program, argv, std::nullopt, redirects,
                                     /*SecondsToWait=*/0, /*MemoryLimit=*/0,
                                     errmsg, /*ExecutionFailed=*/nullptr);
  if (stdout_out) {
    if (auto buf = llvm::MemoryBuffer::getFile(out_path.string())) {
      *stdout_out = buf.get()->getBuffer().str();
    }
  }
  if (stderr_out) {
    if (auto buf = llvm::MemoryBuffer::getFile(err_path.string())) {
      *stderr_out = buf.get()->getBuffer().str();
    }
  }
  (void)fs::remove(out_path);
  (void)fs::remove(err_path);
  return rc;
}

std::string truncate_diags(const std::string& stderr_text, uint64_t max_bytes) {
  if (stderr_text.size() <= max_bytes) {
    return stderr_text;
  }
  return stderr_text.substr(0, max_bytes) + "\n[truncated]";
}

}  // namespace

std::optional<SourceLanguage> infer_language(const std::string& path) {
  if (ends_with_ci(path, ".cu")) {
    return SourceLanguage::Cuda;
  }
  if (ends_with_ci(path, ".hip")) {
    return SourceLanguage::Hip;
  }
  return std::nullopt;
}

std::string locate_unwrapped_clang() {
  auto from_wrapper = [](StringRef wrapper) -> std::string {
    auto buf = llvm::MemoryBuffer::getFile(wrapper);
    if (!buf) {
      return "";
    }
    StringRef content = buf.get()->getBuffer();
    size_t pos = content.find("/nix/store/");
    while (pos != StringRef::npos) {
      size_t binpos = content.find("/bin/clang", pos);
      if (binpos != StringRef::npos && binpos - pos < 256) {
        StringRef candidate = content.slice(pos, binpos + 10);
        if (llvm::sys::fs::exists(candidate)) {
          return candidate.str();
        }
      }
      pos = content.find("/nix/store/", pos + 1);
    }
    return "";
  };
  if (auto clangxx = llvm::sys::findProgramByName("clang++")) {
    llvm::SmallString<256> wrapper(*clangxx);
    if (!llvm::sys::fs::real_path(wrapper, wrapper, true)) {
      std::string found = from_wrapper(wrapper.str());
      if (!found.empty()) {
        return found;
      }
    }
  }
  if (auto clang = llvm::sys::findProgramByName("clang")) {
    return *clang;
  }
  return "";
}

struct ClangIngester::Impl {
  // Staged single-root CUDA layout (synthetic; see file comment). Removed
  // with the ingester.
  std::string staged_cuda_root;
  // ROCm root whose include/hip/hip_runtime.h drives the HIP path
  // (resolved from the hipcc wrapper; clang 21's --hip-path).
  std::string hip_root;

  // Resolves the ROCm root from the hipcc wrapper (clr package layout).
  std::string find_hip_root() {
    if (auto hipcc = llvm::sys::findProgramByName("hipcc")) {
      llvm::SmallString<256> resolved(*hipcc);
      if (!llvm::sys::fs::real_path(resolved, resolved, true)) {
        llvm::SmallString<256> root(resolved);
        llvm::sys::path::remove_filename(root);  // .../bin
        llvm::sys::path::remove_filename(root);  // <clr root>
        if (llvm::sys::fs::exists(root + "/include/hip/hip_runtime.h")) {
          return root.str().str();
        }
      }
    }
    return "";
  }

  // Stages include/ (symlink farm) + bin/, nvvm/, crt/ + the curand shim
  // into a fresh temp dir, mirroring scripts/smoke_toolchain.sh. Returns ""
  // when staging fails.
  std::string stage_cuda_root(const std::string& cuda_home) {
    if (cuda_home.empty() || !std::filesystem::is_directory(cuda_home + "/include")) {
      return "";
    }
    namespace fs = std::filesystem;
    std::string base = (fs::temp_directory_path() / "kernscope-cuda-root-XXXXXX").string();
    // Deterministic unique suffix.
    base = base.substr(0, base.size() - 6) + std::to_string(::getpid());
    fs::path root(base);
    std::error_code ec;
    if (!fs::create_directory(root, ec) && ec) {
      return "";
    }
    fs::path inc = root / "include";
    fs::create_directory(inc, ec);
    for (const auto& entry : fs::directory_iterator(cuda_home + "/include", ec)) {
      if (ec) {
        break;
      }
      fs::path target = inc / entry.path().filename();
      std::error_code lec;
      fs::create_symlink(entry.path(), target, lec);
    }
    if (auto nvcc = llvm::sys::findProgramByName("nvcc")) {
      fs::path nvcc_bin(*nvcc);
      fs::path pkg = nvcc_bin.parent_path().parent_path();
      std::error_code lec;
      fs::create_symlink(pkg / "bin", root / "bin", lec);
      fs::create_symlink(pkg / "nvvm", root / "nvvm", lec);
      fs::create_symlink(pkg / "include" / "crt", inc / "crt", lec);
    }
    std::ofstream shim(inc / "curand_mtgp32_kernel.h");
    shim << "// kernscope smoke-test shim (NOT an NVIDIA header).\n"
         << "#ifndef __KERNSCOPE_CURAND_SHIM__\n"
         << "#define __KERNSCOPE_CURAND_SHIM__\n"
         << "#endif\n";
    return root.string();
  }
};

ClangIngester::ClangIngester() : impl_(new Impl) {
  clang_path_ = locate_unwrapped_clang();
  impl_->hip_root = impl_->find_hip_root();
  const char* cuda_home = std::getenv("CUDA_HOME");
  std::string home = cuda_home ? cuda_home : "";
  impl_->staged_cuda_root = impl_->stage_cuda_root(home);
  toolchain_ok_ = !clang_path_.empty() && !impl_->staged_cuda_root.empty() &&
                  !impl_->hip_root.empty();
  cuda_root_ = impl_->staged_cuda_root;
}

ClangIngester::~ClangIngester() = default;

IngestionResult ClangIngester::ingest(const std::string& source_path,
                                      const IngestionOptions& options) {
  IngestionResult result;
  if (!toolchain_ok_) {
    const std::string clang_status = clang_path_.empty() ? "missing" : "found";
    const std::string root_status = cuda_root_.empty() ? "missing" : "present";
    result.clang_diagnostics.push_back(
        "kernscope-verify: toolchain not usable (unwrapped clang: " +
        clang_status + ", staged CUDA root: " + root_status +
        "). Run inside `devenv shell` (CUDA_HOME set).");
    return result;
  }
  SourceLanguage lang = options.language;
  if (auto inferred = infer_language(source_path)) {
    lang = *inferred;
  }
  std::string opt_flag = options.opt_level >= 2 ? "-O2" : "-O1";
  bool is_cuda = lang == SourceLanguage::Cuda;

  namespace fs = std::filesystem;
  std::string work = (fs::temp_directory_path() / ("kernscope-ingest-" + std::to_string(::getpid()))).string();
  std::error_code mec;
  if (!fs::create_directory(work, mec) && mec) {
    result.clang_diagnostics.push_back("kernscope-verify: cannot create temp dir: " + mec.message());
    return result;
  }
  struct WorkDirGuard {
    std::string path;
    ~WorkDirGuard() { (void)std::filesystem::remove_all(path); }
  } guard{work};

  std::string arch = options.gpu_arch;
  if (arch.empty()) {
    arch = is_cuda ? "sm_80" : "gfx1101";
  }

  // HIP is driven by the UNWRAPPED clang 21 with --hip-path: the hipcc
  // wrapper's bundled hipClang emits lifetime intrinsics in the 1-arg form
  // that LLVM 21's parser rejects (Intrinsic::verify at end of module), so
  // hipcc output cannot be ingested directly.
  const std::string& compiler = clang_path_;
  if (compiler.empty()) {
    result.clang_diagnostics.push_back("kernscope-verify: no clang binary found");
    return result;
  }
  if (!is_cuda && impl_->hip_root.empty()) {
    result.clang_diagnostics.push_back(
        "kernscope-verify: HIP root not found (include/hip/hip_runtime.h "
        "unresolvable); run inside `devenv shell`");
    return result;
  }

  std::vector<std::string> base;
  base.push_back("-x");
  base.push_back(is_cuda ? "cuda" : "hip");
  if (is_cuda) {
    base.push_back("--no-cuda-version-check");
    base.push_back("--cuda-path=" + cuda_root_);
    base.push_back("-nocudalib");
  } else {
    base.push_back("--hip-path=" + impl_->hip_root);
    base.push_back("--offload-arch=" + arch);
  }
  base.push_back(opt_flag);
  base.push_back("-gline-tables-only");
  base.push_back("-emit-llvm");
  base.push_back("-S");
  if (is_cuda) {
    base.push_back("--cuda-gpu-arch=" + arch);
  }

  auto run_pass = [&](const std::string& pass_flag, const std::string& out_name,
                      std::string* diag_out) -> bool {
    std::vector<std::string> args = base;
    args.push_back(pass_flag);
    args.push_back(source_path);
    args.push_back("-o");
    args.push_back(work + "/" + out_name);
    std::string out_text;
    std::string err_text;
    std::string errmsg;
    int rc = run_tool(compiler, args, &out_text, &err_text, &errmsg);
    if (rc != 0) {
      *diag_out += "clang (pass " + pass_flag + ") exited " + std::to_string(rc) + "\n";
      if (!errmsg.empty()) {
        *diag_out += errmsg + "\n";
      }
      *diag_out += truncate_diags(err_text, options.max_diag_bytes);
      return false;
    }
    return true;
  };

  std::string diag_text;
  std::string host_ll = work + "/host.ll";
  std::string dev_ll = work + "/dev.ll";
  if (options.device_only) {
    if (!run_pass(is_cuda ? "--cuda-device-only" : "--offload-device-only", "dev.ll",
                  &diag_text)) {
      result.clang_diagnostics.push_back(diag_text);
      return result;
    }
  } else {
    if (!run_pass(is_cuda ? "--cuda-host-only" : "--offload-host-only", "host.ll",
                  &diag_text) ||
        !run_pass(is_cuda ? "--cuda-device-only" : "--offload-device-only",
                  "dev.ll", &diag_text)) {
      result.clang_diagnostics.push_back(diag_text);
      return result;
    }
  }

  auto context = std::make_shared<LLVMContext>();
  auto capture = std::make_unique<CaptureDiagHandler>();
  CaptureDiagHandler* capture_ptr = capture.get();
  context->setDiagnosticHandler(std::move(capture));

  llvm::SMDiagnostic err;
  auto load = [&](const std::string& path) -> std::unique_ptr<Module> {
    auto m = llvm::parseIRFile(path, err, *context);
    if (!m) {
      std::string text;
      llvm::raw_string_ostream os(text);
      err.print("kernscope-verify", os);
      os.flush();
      diag_text += text + "\n";
    }
    return m;
  };

  std::unique_ptr<Module> module;
  if (options.device_only) {
    module = load(dev_ll);
  } else {
    auto host = load(host_ll);
    auto dev = load(dev_ll);
    if (!host || !dev) {
      result.clang_diagnostics.push_back(diag_text);
      return result;
    }
    // HIP emits the kernel body in BOTH passes (the host module carries it
    // so host-side launch references resolve); CUDA avoids the clash with
    // __device_stub__ names. The device definition is the real kernel, so
    // drop/retarget the host copy before linking ("symbol multiply defined"
    // would otherwise abort the merge): the host's kernel is a constant
    // alias global (`@kernel = constant ptr @__device_stub__kernel`) whose
    // uses must point at the device function, and any host function body
    // under a device-defined name must go. The device module's appending
    // globals (`@llvm.compiler.used` lives in addrspace(1) on amdgcn) are
    // metadata for the backend, not analysis input, and cannot link against
    // their host counterparts ("Appending variables with different address
    // spaces need to be linked!") — drop them too.
    for (llvm::GlobalVariable& g : llvm::make_early_inc_range(dev->globals())) {
      if (g.hasAppendingLinkage()) {
        g.eraseFromParent();
      }
    }
    for (llvm::GlobalVariable& g : llvm::make_early_inc_range(host->globals())) {
      if (!dev->getNamedValue(g.getName())) {
        continue;
      }
      // The host alias's uses must reference a host-side symbol: the alias
      // initializer is the __device_stub__ function (host-defined, distinct
      // name, so no link conflict; the launch-site resolver maps the stub
      // back to the device kernel by demangling). A same-name declaration
      // does NOT work: the linker renames either side (calling-convention
      // mismatch) and the launch use dangles on the renamed copy.
      if (llvm::Value* init = g.getInitializer()) {
        if (auto* stub = llvm::dyn_cast<llvm::Function>(
                init->stripPointerCasts())) {
          g.replaceAllUsesWith(stub);
          g.eraseFromParent();
        }
      }
    }
    for (llvm::Function& df : dev->functions()) {
      if (df.isDeclaration()) {
        continue;
      }
      if (llvm::Function* hf = host->getFunction(df.getName())) {
        if (!hf->isDeclaration()) {
          hf->deleteBody();
        }
      }
    }
    if (llvm::Linker::linkModules(*host, std::move(dev))) {
      result.clang_diagnostics.push_back(
          "kernscope-verify: LLVM link of host+device IR failed\n" + diag_text);
      for (const std::string& m : capture_ptr->messages) {
        result.clang_diagnostics.push_back("linker: " + m);
      }
      return result;
    }
    module = std::move(host);
  }
  if (!module) {
    result.clang_diagnostics.push_back(diag_text);
    return result;
  }

  result.success = true;
  result.context = std::move(context);
  result.module = std::move(module);
  result.provenance = is_cuda
                          ? "clang -x cuda " + opt_flag + " host+device passes + LLVM link (arch " +
                                arch + (options.device_only ? ", device-only" : "") + ")"
                          : "clang -x hip --hip-path=" + impl_->hip_root + " " + opt_flag +
                                " host+device passes + LLVM link (arch " + arch +
                                (options.device_only ? ", device-only" : "") + ")";
  return result;
}

}  // namespace verify
}  // namespace kernscope
