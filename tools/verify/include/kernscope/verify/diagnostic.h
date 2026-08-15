// kernscope::verify — core diagnostic model (todo 30).
//
// eBPF-style diagnostic: severity class, class id + name, source location
// (DebugLoc + function + instruction), human message and an optional
// path/stack of evidence steps. Check modules (todos 31-34) emit
// Diagnostics through DiagnosticEngine::emit(); the engine routes them to
// the sink and enforces the per-function budget.
//
// Check-class ids are reserved in kernscope::verify::CheckClass so the eight
// plan check classes map to stable integers across the registry, the
// diagnostics and todo 35's CLI/verdict JSON:
//   1 global OOB            5 illegal pointer arithmetic
//   2 shared-memory bounds  6 divergent barriers
//   3 barrier correctness   7 uninitialized reads
//   4 race-prone writes     8 invalid address-space casts
// Engine-degraded verdicts (symbolic dims, budget exceeded, -O2 SCEV
// failure) use ids >= 100 so they can never collide with a check class.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace kernscope {
namespace verify {

/// Reserved check-class ids (the eight plan classes; see file comment).
/// Check modules declare their classes via CheckPlugin::check_class_ids.
namespace CheckClass {
inline constexpr int kGlobalOob = 1;
inline constexpr int kSmemBounds = 2;
inline constexpr int kBarrier = 3;
inline constexpr int kRace = 4;
inline constexpr int kPtrArith = 5;
inline constexpr int kDivergentBarrier = 6;
inline constexpr int kUninitRead = 7;
inline constexpr int kAddrSpaceCast = 8;
}  // namespace CheckClass

/// Engine-degraded verdict classes (ids >= 100, never a check class).
namespace EngineClass {
/// Launch config unknown (tier Symbolic): "symbolic dims" degraded verdict.
inline constexpr int kSymbolicDims = 100;
/// -O2 ingestion: index bounds could not be reconstructed via SCEV.
inline constexpr int kScevReconstructionFailed = 101;
/// Per-function state budget exhausted (eBPF complexity-limit analog).
inline constexpr int kBudgetExceeded = 102;
}  // namespace EngineClass

/// Maps a class id to its human name ("" for unknown ids).
/// Used by Diagnostic::render and todo 35's verdict JSON.
std::string check_class_name(int class_id);

enum class Severity : std::uint8_t { Info, Warning, Error };

const char* severity_name(Severity s);

/// Source location of a finding: DebugLoc line/col + the enclosing function
/// and the offending instruction. `file` is empty when the IR carries no
/// debug info for that instruction (compiled without -gline-tables-only).
struct SourceLoc {
  std::string file;
  unsigned line = 0;
  unsigned col = 0;

  bool valid() const { return !file.empty() && line != 0; }

  /// Renders "<file>:<line>:<col>" or "<unknown>" when invalid.
  std::string render() const;
};

/// One finding from a check plugin or the engine itself.
struct Diagnostic {
  Severity severity = Severity::Warning;
  int class_id = 0;           ///< CheckClass / EngineClass id.
  std::string class_name;     ///< Human name (checked in via check_class_name).
  SourceLoc loc;              ///< DebugLoc-backed source location.
  std::string function;       ///< Enclosing LLVM function (demangled if kernel).
  std::string instruction;    ///< Short instruction description.
  std::string message;        ///< Human-readable finding text.
  std::vector<std::string> path;  ///< Optional evidence path (eBPF analog).

  /// One-line rendering: "severity [class] file:line:col (fn): message".
  std::string render() const;
};

}  // namespace verify
}  // namespace kernscope
