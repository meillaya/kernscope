// kernscope::verify — clang ingestion wrapper (todo 30).
//
// Primary mode (DEFAULT): ingest CUDA/HIP source as ONE LLVM module
// containing BOTH host and device code:
//
//   clang -x cuda --no-cuda-version-check --cuda-path=<root> -nocudalib \
//         -gline-tables-only -O1 --cuda-host-only  -emit-llvm -S \
//         --cuda-gpu-arch=sm_80 <src> -o host.ll
//   clang -x cuda ...                                     --cuda-device-only ...
//   (host.ll + dev.ll linked in-process into one module)
//
// The host half carries the launch sites (__cudaPushCallConfiguration /
// cudaLaunchKernel) whose constant dims feed launch-config tier (a); the
// device half carries the kernels. Two clang passes + a link step are
// REQUIRED on this toolchain: clang 21's -emit-llvm runs fatbinary on the
// device bitcode and dies with "fatbinary elf mismatch" (documented in
// tests/fixtures/README.md "clang/CUDA header gate" + todo-5 evidence), so
// a single -emit-llvm invocation cannot produce host+device in one file.
//
// Secondary mode (--device-only, documented degraded path): only the device
// pass runs; launch dims are lost, so every kernel without nvvm.maxntid
// degrades to LaunchTier::Symbolic ("symbolic dims" verdict).
//
// The CUDA root is staged like scripts/smoke_toolchain.sh when the nix
// package split makes the single-root layout unavailable: include/ symlink
// farm + nvcc crt/ + bin/ + nvvm/libdevice + a test-owned curand shim
// (clang's __clang_cuda_runtime_wrapper.h force-includes the name).
// The HIP path (gfx1101, -x hip) mirrors this with its own staging.
//
// Ingestion failures NEVER crash the verifier: clang's stderr is captured
// and forwarded verbatim in IngestionResult::clang_diagnostics; the caller
// (harness/CLI) reports it and exits 2.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llvm {
class LLVMContext;
class Module;
}  // namespace llvm

namespace kernscope {
namespace verify {

/// Which language the source is (inferred from the extension).
enum class SourceLanguage : std::uint8_t { Cuda, Hip };

/// Options controlling one ingestion run.
struct IngestionOptions {
  SourceLanguage language = SourceLanguage::Cuda;
  /// Device-only secondary mode (degraded: dims become symbolic).
  bool device_only = false;
  /// Ingestion optimization level. Pin -O1 by default (verify early, before
  /// loop destruction — research brief §5.2); -O2 exercises the degraded
  /// SCEV-reconstruction path (acceptance case 3).
  int opt_level = 1;
  /// GPU arch: "sm_80" for CUDA, "gfx1101" for HIP (ROCm toolchain).
  std::string gpu_arch;
  /// clang binary. Empty: auto-locate the UNWRAPPED clang 21 binary the way
  /// scripts/smoke_toolchain.sh does (nix wrapper bakes host-only flags the
  /// nvptx64 target rejects).
  std::string clang_path;
  /// CUDA toolkit root (CUDA_HOME). Empty: read the environment. The
  /// toolchain staging step still synthesizes a single-root layout from it.
  std::string cuda_home;
  /// Maximum stderr bytes kept per clang invocation (diagnostics cap).
  uint64_t max_diag_bytes = 64 * 1024;
};

/// Result of one ingestion. `module` is null exactly when `success` is false.
struct IngestionResult {
  bool success = false;
  /// Forwarded clang/linker diagnostics (verbatim stderr, truncated at
  /// max_diag_bytes). Empty on success.
  std::vector<std::string> clang_diagnostics;
  /// The module's LLVMContext. Declared before `module` so it outlives it
  /// (destruction order is reverse declaration order).
  std::shared_ptr<llvm::LLVMContext> context;
  std::unique_ptr<llvm::Module> module;
  /// Human summary of how the module was produced (for --dump-state).
  std::string provenance;
};

/// Locates + stages the CUDA/HIP toolchain and runs the clang ingestion.
/// One instance stages its synthetic CUDA root in a temp dir held for the
/// object's lifetime (removed on destruction).
class ClangIngester {
 public:
  ClangIngester();
  ~ClangIngester();

  ClangIngester(const ClangIngester&) = delete;
  ClangIngester& operator=(const ClangIngester&) = delete;

  /// Runs ingestion for `source_path`. On failure, returns success=false with
  /// the clang diagnostics forwarded (never throws, never crashes).
  IngestionResult ingest(const std::string& source_path,
                         const IngestionOptions& options);

  /// Paths used (for diagnostics / tests): staged CUDA root, clang binary.
  const std::string& cuda_root() const { return cuda_root_; }
  const std::string& clang_path() const { return clang_path_; }

  /// True when the toolchain could be staged (clang found + CUDA root
  /// usable). ingest() returns success=false with a diagnostic when false.
  bool toolchain_ok() const { return toolchain_ok_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::string cuda_root_;
  std::string clang_path_;
  bool toolchain_ok_ = false;
};

/// Toolchain discovery helper (exposed for tests): resolves the unwrapped
/// clang 21 binary following scripts/smoke_toolchain.sh (readlink the
/// clang++ wrapper, grep the nix-store unwrapped path; fall back to plain
/// `clang` on PATH). Returns "" when nothing usable is found.
std::string locate_unwrapped_clang();

/// Infers SourceLanguage from the file extension (.cu -> Cuda, .hip -> Hip).
/// Returns std::nullopt for unknown extensions.
std::optional<SourceLanguage> infer_language(const std::string& path);

}  // namespace verify
}  // namespace kernscope
