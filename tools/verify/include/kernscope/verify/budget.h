// kernscope::verify — path/complexity budget (todo 30).
//
// eBPF analog: the kernel verifier caps explored work (BPF_COMPLEXITY_LIMIT_
// INSNS 1M / BPF_COMPLEXITY_LIMIT_STATES 64 per instruction) and reports a
// "BPF program is too large" rejection instead of hanging. The kernscope
// budget does the same per GPU kernel *function*: every state the engine
// explores (basic block visited, instruction processed, range-lattice step)
// must be acquired from the Budget. When the cap is hit the engine stops
// exploring that function and emits an EngineClass::kBudgetExceeded
// diagnostic ("budget exceeded") — a degraded verdict, never a hang.
//
// Default: 65536 state acquisitions per function (documented; configurable
// via VerifierOptions and the --budget CLI flag). The synthetic budget test
// (verify_core) exercises a small cap on a large synthetic function and
// asserts termination < 5s with the budget diagnostic.

#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace kernscope {
namespace verify {

/// Per-function explored-state budget. Thread-affine (single-threaded use).
class Budget {
 public:
  /// Default per-function cap on explored states (documented default).
  static constexpr uint64_t kDefaultMaxStatesPerFunction = 65536;

  explicit Budget(uint64_t max_states_per_function = kDefaultMaxStatesPerFunction)
      : max_states_per_function_(max_states_per_function) {}

  /// Resets the counter for `function` (called when starting a new function).
  void reset(std::string_view function);

  /// Acquires `n` state slots for `function`. Returns false (and latches the
  /// function as exceeded) when the acquisition would cross the cap.
  bool acquire(std::string_view function, uint64_t n = 1);

  /// True once `function` has hit the cap (latched; stays true until reset).
  bool exceeded(std::string_view function) const;

  /// States consumed by `function` so far (0 if never touched).
  uint64_t consumed(std::string_view function) const;

  uint64_t max_states_per_function() const { return max_states_per_function_; }

 private:
  uint64_t max_states_per_function_;
  std::unordered_map<std::string, uint64_t> consumed_;
  std::unordered_map<std::string, bool> latched_;
};

}  // namespace verify
}  // namespace kernscope
