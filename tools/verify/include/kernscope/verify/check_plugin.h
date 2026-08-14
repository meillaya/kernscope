// kernscope::verify — CHECK PLUGIN API (todo 30). The contract todos 31-34
// implement against.
//
// A check module (tools/verify/checks/<name>/) is a self-contained plugin:
//
//   1. Define a class deriving kernscope::verify::CheckPlugin, setting
//      `id`, `name` and `check_class_ids` in its constructor and
//      implementing `run(VerifierContext&) const` to emit Diagnostics
//      through `ctx.diagnostics().emit(...)`.
//   2. Register it with the REGISTER_CHECK(PluginClass) macro at file
//      scope. The static registrar hands a fresh instance to the global
//      CheckRegistry — NO shared file needs editing, so todos 31-34 can
//      land their modules without touching anything outside their
//      subdirectory (that is why tools/verify/checks/{oob,barrier,race,
//      ptr} are pre-declared as stub subdirectories).
//   3. Enumerate registered checks (todo 35's CLI uses this for
//      `--list-checks` and verdict JSON): CheckRegistry::instance().
//
// run() receives VerifierContext (defined in verifier_state.h) exposing:
//   * the LLVM module and the per-kernel KernelInfo (LaunchConfig: tier +
//     exact/upper-bound dims + origin),
//   * the diagnostic sink + budget counter (DiagnosticEngine),
//   * helpers to query value ranges (SCEV-backed, launch-dim aware):
//     VerifierContext::queryRange(Value*).
//
// The engine calls run() once per (plugin, kernel) pair. Plugins MUST
// acquire budget slots for every explored state
// (ctx.diagnostics().budget().acquire(ctx.kernel().function_name, n)) and
// stop exploring when acquisition fails; the engine emits the
// kBudgetExceeded diagnostic for them.
//
// Doxygen on each member states the exact contract.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <kernscope/verify/diagnostic.h>

namespace kernscope {
namespace verify {

struct VerifierContext;

/// One check plugin: id/name/classes + the per-(plugin,kernel) analysis.
/// All members are const-stable for the process lifetime.
struct CheckPlugin {
  /// Stable identifier (e.g. "global-oob"); used by --list-checks + verdict
  /// JSON. Must be unique across all plugins.
  std::string id;
  /// Human name (e.g. "Global out-of-bounds").
  std::string name;
  /// Check-class ids this plugin implements (see diagnostic.h CheckClass).
  /// May be empty for engine-only plugins.
  std::vector<int> check_class_ids;

  virtual ~CheckPlugin() = default;

  /// Runs this plugin against one kernel. Emit findings via
  /// ctx.diagnostics().emit(...) and return diagnostics directly (the
  /// engine merges both channels). Must respect the budget: check
  /// ctx.diagnostics().budget().acquire(...) per explored state and stop
  /// exploring once it fails (the engine will have emitted the
  /// budget-exceeded diagnostic).
  virtual std::vector<Diagnostic> run(VerifierContext& ctx) const = 0;
};

/// Global registry of registered check plugins (singleton; lazily
/// constructed, no static-init-order hazards).
class CheckRegistry {
 public:
  static CheckRegistry& instance();

  /// Registers one plugin (ownership taken). Called by REGISTER_CHECK.
  void registerPlugin(std::unique_ptr<CheckPlugin> plugin);

  /// All registered plugins, in registration order.
  const std::vector<std::unique_ptr<CheckPlugin>>& plugins() const;

  /// Plugins implementing `class_id` (used to dispatch per check class).
  std::vector<const CheckPlugin*> pluginsForClass(int class_id) const;

  /// Summary entry for enumeration (todo 35's --list-checks).
  struct Entry {
    std::string id;
    std::string name;
    std::vector<int> class_ids;
  };
  std::vector<Entry> enumerate() const;

 private:
  CheckRegistry() = default;
  std::vector<std::unique_ptr<CheckPlugin>> plugins_;
};

/// Registrar type used by REGISTER_CHECK (public so the macro works from any
/// TU; do not construct manually).
class CheckPluginRegistrar {
 public:
  explicit CheckPluginRegistrar(std::unique_ptr<CheckPlugin> plugin);
};

}  // namespace verify
}  // namespace kernscope

/// Self-registration: defines a static registrar that adds one instance of
/// PluginClass to the global registry. Place at file scope in the check
/// module's .cpp:
///
///   REGISTER_CHECK(GlobalOobCheck);   // class derives CheckPlugin
///
/// No shared-file edits needed — this is the registry contract todos 31-34
/// rely on.
#define REGISTER_CHECK(PluginClass)                                          \
  static ::kernscope::verify::CheckPluginRegistrar                           \
      kernscope_verify_register_##PluginClass(std::make_unique<PluginClass>())
