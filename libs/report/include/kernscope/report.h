//===- report.h - Structured kernel report model + renderers ----*- C++ -*-===//
//
// kernscope report engine (wave-1 todo 7). One structured model,
// three renderers:
//
//   * text  - the owner's example format, docs/research/owner-examples.md
//             Section A (Kernel: / Occupancy: / Main limiter: /
//             Warnings: / Suggested: with indented "- " bullets);
//   * json  - the model as JSON, validated against the committed schema
//             docs/schema/report.schema.json (draft 2020-12);
//   * html  - a self-contained single file (inline CSS/JS, embedded JSON
//             payload in <script id="report-data" type="application/json">,
//             zero network requests; polykernel/tt-flow convention).
//
// Every metric is a machine-readable field: the model carries a flat
// numeric metrics map; renderers never invent prose-only fields. The
// special evidence key "text" holds the one-line human summary used by
// the text renderer; all other evidence keys are machine-readable.
//
// All renderers validate the model first and throw SchemaViolation
// (never a silent render) when the model cannot satisfy the schema.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_REPORT_H
#define KERNSCOPE_REPORT_H

#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace kernscope {

/// One diagnostic finding (warning or suggestion).
///
/// `evidence` is a key/value string map: numeric fields stay machine-readable
/// as strings (e.g. {"stride_words": "68", "way": "4"}), matching the schema's
/// evidence value type. The special key "text" is the one-line human summary
/// the TEXT renderer prints; when absent, the text renderer falls back to
/// `id`.
struct Finding {
  std::string id;                                   ///< stable id, e.g. bank_conflict
  std::string severity;                             ///< e.g. warning, suggestion
  std::map<std::string, std::string> evidence;      ///< machine-readable key/value
};

/// The full kernel report model shared by every renderer.
///
/// `metrics` is a flat numeric map — every measured quantity lives here as a
/// machine-readable number (occupancy, registers_per_thread, ...). `limiter`
/// is the human main-limiter string (e.g. "112 registers/thread"); its
/// numeric counterpart lives in metrics.registers_per_thread.
struct KernelReport {
  std::string kernel_name;
  std::map<std::string, double> metrics;
  std::string limiter;
  std::vector<Finding> warnings;
  std::vector<Finding> suggestions;
};

/// Thrown by every renderer when the model violates
/// docs/schema/report.schema.json (e.g. a finding missing its required
/// `severity`). Renderers never silently emit an invalid report.
class SchemaViolation : public std::runtime_error {
public:
  explicit SchemaViolation(const std::string &message)
      : std::runtime_error(message) {}
};

/// Validate the model against the committed schema's required fields:
/// kernel name, finite numeric metrics, limiter, and findings with
/// id/severity/evidence. Returns false and fills `error` with the first
/// violation when the model cannot be rendered validly.
[[nodiscard]] bool validate(const KernelReport &report, std::string *error);

/// Text renderer — the owner's Section A format exactly
/// (docs/research/owner-examples.md). Throws SchemaViolation on invalid input.
std::string to_text(const KernelReport &report);

/// JSON renderer — validates against docs/schema/report.schema.json
/// (draft 2020-12). Throws SchemaViolation on invalid input.
std::string to_json(const KernelReport &report);

/// HTML renderer — self-contained single file, inline CSS/JS, embedded JSON
/// payload (extractable with python3 html.parser, see tests/report/
/// check_html.py), zero network requests. Throws SchemaViolation on invalid
/// input.
std::string to_html(const KernelReport &report);

} // namespace kernscope

#endif // KERNSCOPE_REPORT_H
