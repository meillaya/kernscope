// kernscope::verify — VerifierState: per-kernel state + the core pipeline
// driver (todo 30).
//
// Pipeline: ingest (ClangIngester) -> module -> KernelInfo per kernel
// (launch-config tier resolution + intrinsic range propagation) -> run all
// registered check plugins with a per-function budget -> diagnostics.
//
// VerifierState also produces the --dump-state text: per-kernel module,
// launch tier, exact dim ranges (the todo-30 acceptance gate: "exact dim
// ranges visible in dumped verifier state", e.g. threadIdx.x in [0, 256),
// blockIdx.x in [0, 32) for <<<32, 256>>>).

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <llvm/IR/ConstantRange.h>

#include <kernscope/verify/budget.h>
#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/launch_config.h>

namespace llvm {
class Function;
class Instruction;
class Module;
class Value;
class ScalarEvolution;
class TargetLibraryInfo;
class AssumptionCache;
class DominatorTree;
class LoopInfo;
}  // namespace llvm

namespace kernscope {
namespace verify {

/// Engine options (dump + pipeline behaviour).
struct VerifierOptions {
  /// Per-function explored-state cap (see budget.h; default documented).
  uint64_t max_states_per_function = Budget::kDefaultMaxStatesPerFunction;
  /// Optimization level the module was ingested at (harness --opt; default
  /// 1 = verify-early mode, the supported ingestion contract).
  int ingestion_opt_level = 1;
  /// Emit degraded-verdict diagnostics for unbounded memory-access indices
  /// when the module was ingested at -O2 (degraded ingestion mode). Always
  /// false for -O1: at -O1 missing ranges are ordinary may-analysis input
  /// for the check modules, not engine verdicts.
  bool o2_reconstruction_degraded = true;
  /// Include the range-lattice summary in the state dump.
  bool dump_ranges = true;
};

/// Per-intrinsic range lane for thread/block coordinate intrinsics.
enum class Lane : std::uint8_t {
  ThreadX, ThreadY, ThreadZ,
  BlockDimX, BlockDimY, BlockDimZ,
  BlockX, BlockY, BlockZ,
  GridDimX, GridDimY, GridDimZ,
};

/// One kernel: identity, resolved LaunchConfig, propagated intrinsic ranges.
struct KernelInfo {
  std::string mangled_name;      ///< e.g. "_Z9lb_kernelPii".
  std::string name;              ///< demangled ("lb_kernel"); mangled fallback.
  llvm::Function* function = nullptr;
  LaunchConfig config;
  /// Ranges for the coordinate intrinsics inside this kernel (exact for
  /// tier (a), sound upper bounds for tier (b), empty/unbounded for (c)).
  std::vector<llvm::ConstantRange> intrinsic_ranges;  // indexed by Lane
  /// Source line of the launch site that produced tier (a), else 0.
  unsigned launch_line = 0;
  std::string launch_file;

  const llvm::ConstantRange& range(Lane lane) const {
    return intrinsic_ranges[static_cast<size_t>(lane)];
  }
  bool rangeBounded(Lane lane) const;
};

/// Diagnostic sink + budget. One per VerifierState; handed to plugins as
/// ctx.diagnostics(). Emit routes findings to the sink and enforces the
/// budget (emitting kBudgetExceeded once per function on exhaustion).
class DiagnosticEngine {
 public:
  explicit DiagnosticEngine(VerifierOptions options)
      : options_(options), budget_(options.max_states_per_function) {}

  const VerifierOptions& options() const { return options_; }
  Budget& budget() { return budget_; }
  const Budget& budget() const { return budget_; }

  /// Emits one diagnostic (sink append). Also re-checks the budget latch:
  /// when `function` has been latched exceeded, further emissions for it
  /// are still recorded (findings are cheap; only EXPLORATION is capped).
  void emit(Diagnostic d);

  /// Emits the "budget exceeded" degraded verdict for `function` (once).
  void emitBudgetExceeded(const std::string& function);

  /// All findings collected so far.
  const std::vector<Diagnostic>& diagnostics() const { return diagnostics_; }

 private:
  VerifierOptions options_;
  Budget budget_;
  std::vector<Diagnostic> diagnostics_;
  std::vector<std::string> budget_latched_functions_;
};

/// Context handed to CheckPlugin::run (one per plugin+kernel). The exact
/// surface todos 31-34 build against.
struct VerifierContext {
  llvm::Module& module;
  KernelInfo& kernel;
  DiagnosticEngine& diagnostics;
  /// Back-pointer to the owning VerifierState (SCEV cache + lattice).
  class VerifierState* state = nullptr;

  /// RANGE QUERY HELPER (SCEV-backed, launch-dim aware). Returns the proven
  /// unsigned range of `v` inside the current kernel: the propagated
  /// range-lattice answer first (exact for threadIdx/blockIdx-derived
  /// values under tier (a)), ScalarEvolution getRange second, std::nullopt
  /// when neither can bound it (caller emits its own degraded verdict /
  /// may-analysis finding). `v` must be defined in the kernel (or be a
  /// constant).
  std::optional<llvm::ConstantRange> queryRange(llvm::Value* v) const;

  /// ScalarEvolution for the kernel's function (cached; created on first
  /// use). May be null only if analysis construction failed (never in
  /// practice). Helpers for SCEV-driven checks (todos 31-34).
  llvm::ScalarEvolution* scalarEvolution() const;

  /// Convenience: the kernel's LaunchConfig (tier + exact/upper dims +
  /// origin). Same as kernel().config.
  const LaunchConfig& launchConfig() const { return kernel.config; }
  const KernelInfo& kernelInfo() const { return kernel; }
};

/// Core pipeline driver: ingest module + kernels + launch tiers + checks.
class VerifierState {
 public:
  /// Builds the state from an already-ingested module. The LLVMContext must
  /// outlive the module (both shared; IngestionResult keeps them together).
  VerifierState(std::shared_ptr<llvm::Module> module,
                const std::shared_ptr<llvm::LLVMContext>& context,
                VerifierOptions options);
  ~VerifierState();

  VerifierState(const VerifierState&) = delete;
  VerifierState& operator=(const VerifierState&) = delete;

  /// Runs the whole pipeline: kernel discovery, launch-config tier
  /// resolution, intrinsic range propagation, then every registered plugin
  /// on every kernel (each with per-function budget). Safe to call once;
  /// repeated calls re-run plugins (idempotent state updates).
  void analyze();

  /// All kernels found (populated by analyze()).
  const std::vector<KernelInfo>& kernels() const;
  /// Findings (engine + plugin diagnostics) after analyze().
  const std::vector<Diagnostic>& diagnostics() const;
  /// Diagnostic engine (budget access for tests).
  DiagnosticEngine& engine() { return *engine_; }

  /// Count of memory accesses whose index range was bounded by the core
  /// propagation/SCEV path (the -O2 reconstruction acceptance gate).
  uint64_t boundedAccessCount() const;

  /// Internal analysis state (SCEV cache, lattices) — exposed for
  /// VerifierContext and tests.
  struct Impl;
  Impl* impl() const;

  /// Per-kernel state dump: module identity, launch tier + origin, exact
  /// dim ranges (threadIdx/blockIdx/blockDim/gridDim lanes), propagated
  /// range summary, plugin enumeration, budget consumption. This is the
  /// --dump-state output.
  std::string dumpState() const;

  /// Finds kernels in the module (ptx_kernel / amdgpu_kernel / "kernel"
  /// attr), resolving mangled + demangled names. Public for tests.
  static std::vector<llvm::Function*> findKernels(llvm::Module& m);

  /// Resolves the launch config for `k`: scans host-side launch sites
  /// (__cudaPushCallConfiguration/cudaConfigureCall + cudaLaunchKernel),
  /// falls back to nvvm.maxntid, else Symbolic. Public for tests.
  static LaunchConfig resolveLaunchConfig(llvm::Module& m,
                                          llvm::Function& kernel);

 private:
  std::unique_ptr<Impl> impl_;
  std::shared_ptr<llvm::Module> module_;
  VerifierOptions options_;
  std::unique_ptr<DiagnosticEngine> engine_;
  std::vector<KernelInfo> kernels_;
  bool analyzed_ = false;
};

}  // namespace verify
}  // namespace kernscope
