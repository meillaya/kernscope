//===- report_test.cpp - gtest suite for libs/report -------*- C++ -*-===//
//
// (a) a fixed finding set renders to an EXACT text golden — byte-identical
//     to the owner's example (docs/report/kernscope-technical-report.pdf Section A);
// (b) JSON output validates against docs/report/kernscope-technical-report.pdf
//     (draft 2020-12) via the committed minimal validator
//     tests/report/validate_json.py under python3 (no pip packages needed);
// (c) the HTML output embeds the JSON payload, extracted it equals the
//     input model byte-for-byte AND validates against the schema;
// (d) negative: a finding missing a required field (severity) is rejected
//     with SchemaViolation naming the field — never a silent render.
//
//===----------------------------------------------------------------------===//

#include "kernscope/report.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>

// CMake -D values are already string literals; re-stringifying double-quotes
// the path (STR()-on-string-literal bug), so use the macros directly.
#define KERN_SHELL_FMT \
  std::string(PYTHON3) + " " + REPORT_VALIDATOR + " " + path + " " + \
  REPORT_SCHEMA

namespace {

// ---- shared model: the owner's fused_attention example --------------------

kernscope::KernelReport OwnerExampleReport() {
  kernscope::KernelReport report;
  report.kernel_name = "fused_attention.ptx";
  report.metrics = {
      {"occupancy", 0.375},
      {"registers_per_thread", 112},
      {"blocks_per_sm", 9},
      {"warps_resident_per_sm", 18},
      {"max_warps_per_sm", 48},
      {"shared_memory_bytes", 67584},
      {"spill_stores", 0},
      {"spill_loads", 0},
  };
  report.limiter = "112 registers/thread";

  report.warnings = {
      {"bank_conflict", "warning",
       {{"text", "shared memory stride likely causes 4-way bank conflicts"},
        {"stride_words", "68"},
        {"way", "4"}}},
      {"divergent_inner_loop", "warning",
       {{"text", "divergent branch inside inner loop"},
        {"loop", "score j-loop"},
        {"predicate", "lane % 8 == j % 8"}}},
      {"coalescing", "warning",
       {{"text", "global loads not 128B coalesced"},
        {"efficiency", "0.5"},
        {"vectorized_ld_global", "0"}}},
  };
  report.suggestions = {
      {"tile_k_32", "suggestion",
       {{"text", "tile K dimension by 32"},
        {"current_tile_k", "64"},
        {"suggested_tile_k", "32"}}},
      {"split_accumulator", "suggestion",
       {{"text", "split accumulator fragment"},
        {"accumulator_elements", "64"},
        {"suggested_fragment_elements", "16"}}},
      {"hoist_predicate", "suggestion",
       {{"text", "move predicate outside loop"},
        {"current_branch_sites", "8"},
        {"suggested_branch_sites", "1"}}},
  };
  return report;
}

/// Write `content` to `path` (creating parent dirs). Returns true on success.
bool WriteFile(const std::string &path, const std::string &content) {
  std::error_code ec;
  std::filesystem::create_directories(
      std::filesystem::path(path).parent_path(), ec);
  if (ec)
    return false;
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;
  out << content;
  return static_cast<bool>(out);
}

/// Run `python3 validate_json.py <json> <schema>`; returns the exit status.
int RunValidator(const std::string &path) {
  const std::string cmd = KERN_SHELL_FMT + " > /dev/null 2>&1";
  const int status = std::system(cmd.c_str());
  if (status == -1)
    return -1;
  return WEXITSTATUS(status);
}

/// Extract the embedded JSON payload from `html` — the byte-exact text
/// between the report-data script tags (to_html embeds it with no
/// surrounding whitespace, so to_json's own trailing newline is part of
/// the payload). Returns false on malformed structure.
bool ExtractPayload(const std::string &html, std::string *payload) {
  const std::string open =
      "<script id=\"report-data\" type=\"application/json\">";
  const std::string close = "</script>";
  const size_t begin = html.find(open);
  if (begin == std::string::npos)
    return false;
  const size_t payload_begin = begin + open.size();
  const size_t end = html.find(close, payload_begin);
  if (end == std::string::npos)
    return false;
  *payload = html.substr(payload_begin, end - payload_begin);
  return true;
}

// ---- (a) exact text golden ------------------------------------------------

TEST(ReportText, OwnerExampleGolden) {
  const kernscope::KernelReport report = OwnerExampleReport();

  // Byte-identical to docs/report/kernscope-technical-report.pdf Section A (the
  // trailing newline is part of the format).
  const std::string kGolden = R"(Kernel: fused_attention.ptx
Occupancy: 37.5%
Main limiter: 112 registers/thread
Warnings:
  - shared memory stride likely causes 4-way bank conflicts
  - divergent branch inside inner loop
  - global loads not 128B coalesced
Suggested:
  - tile K dimension by 32
  - split accumulator fragment
  - move predicate outside loop
)";

  EXPECT_EQ(kernscope::to_text(report), kGolden);
}

TEST(ReportText, EmptyFindingsRenderNone) {
  kernscope::KernelReport report;
  report.kernel_name = "empty_kernel";
  report.metrics = {{"occupancy", 0.25}};
  report.limiter = "registers";
  const std::string text = kernscope::to_text(report);
  EXPECT_NE(text.find("Warnings:\n  (none)"), std::string::npos);
  EXPECT_NE(text.find("Suggested:\n  (none)"), std::string::npos);
}

// ---- (b) JSON validates against the committed schema -----------------------

TEST(ReportJson, ValidatesAgainstSchema) {
  const kernscope::KernelReport report = OwnerExampleReport();
  const std::string json = kernscope::to_json(report);
  const std::string path = std::string(REPORT_TMPDIR) + "/owner.json";
  ASSERT_TRUE(WriteFile(path, json));
  EXPECT_EQ(RunValidator(path), 0)
      << std::string("to_json output must validate against ") + REPORT_SCHEMA;
}

TEST(ReportJson, NumericMetricsAreMachineReadable) {
  const kernscope::KernelReport report = OwnerExampleReport();
  const std::string json = kernscope::to_json(report);
  // Machine-readable numeric fields: no prose-only metrics anywhere.
  EXPECT_NE(json.find("\"occupancy\": 0.375"), std::string::npos);
  EXPECT_NE(json.find("\"registers_per_thread\": 112"), std::string::npos);
  EXPECT_NE(json.find("\"way\": \"4\""), std::string::npos);
  // JSON is deterministic (std::map ordering + fixed emission order).
  EXPECT_EQ(json, kernscope::to_json(report));
}

// ---- (c) HTML embeds the JSON payload --------------------------------------

TEST(ReportHtml, EmbedsExtractablePayloadEqualToModel) {
  const kernscope::KernelReport report = OwnerExampleReport();
  const std::string html = kernscope::to_html(report);
  const std::string json = kernscope::to_json(report);

  // Self-contained: no external URLs anywhere.
  EXPECT_EQ(html.find("http://"), std::string::npos);
  EXPECT_EQ(html.find("https://"), std::string::npos);

  std::string payload;
  ASSERT_TRUE(ExtractPayload(html, &payload))
      << "HTML must embed <script id=\"report-data\" type=\"application/json\">";
  // The embedded payload is the exact model JSON (the sample model contains
  // no "</" sequences, so no escaping is applied; the payload text therefore
  // equals to_json byte-for-byte).
  EXPECT_EQ(payload, json);

  // And the extracted payload itself validates against the schema.
  const std::string path = std::string(REPORT_TMPDIR) + "/embedded.json";
  ASSERT_TRUE(WriteFile(path, payload));
  EXPECT_EQ(RunValidator(path), 0)
      << std::string("embedded HTML payload must validate against ") + REPORT_SCHEMA;
}

TEST(ReportHtml, EscapesScriptClosersInPayload) {
  // A malicious/naive evidence value containing "</script>" must not break
  // the HTML; the escape "<\/script>" round-trips through JSON parsing.
  kernscope::KernelReport report;
  report.kernel_name = "tricky";
  report.metrics = {{"occupancy", 0.5}};
  report.limiter = "registers";
  report.warnings = {
      {"weird", "warning", {{"text", "payload </script> injection probe"}}}};
  const std::string html = kernscope::to_html(report);

  std::string payload;
  ASSERT_TRUE(ExtractPayload(html, &payload));
  // Raw closers must never survive in the payload; the JSON-valid "<\/"
  // escape (decoded by JSON.parse/json.loads) must be present instead.
  EXPECT_EQ(payload.find("</script>"), std::string::npos);
  EXPECT_NE(payload.find("<\\/script>"), std::string::npos);
}

// ---- (d) negative: schema-violating models are rejected, never rendered ----

TEST(ReportRejection, MissingSeverityThrowsSchemaViolation) {
  kernscope::KernelReport report = OwnerExampleReport();
  report.warnings.push_back(
      {"bank_conflict", "", {{"text", "missing severity"}}}); // no severity

  for (const auto &renderer : {kernscope::to_text, kernscope::to_json,
                               kernscope::to_html}) {
    EXPECT_THROW(renderer(report), kernscope::SchemaViolation);
  }

  try {
    kernscope::to_json(report);
    FAIL() << "expected SchemaViolation for a finding without severity";
  } catch (const kernscope::SchemaViolation &error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("severity"), std::string::npos)
        << "violation message must name the missing field, got: " << message;
    EXPECT_NE(message.find("bank_conflict"), std::string::npos)
        << "violation message must name the offending finding, got: " << message;
  }
}

TEST(ReportRejection, MissingFindingIdThrows) {
  kernscope::KernelReport report = OwnerExampleReport();
  report.suggestions.push_back(
      {"", "suggestion", {{"text", "no id"}}});
  std::string error;
  EXPECT_FALSE(kernscope::validate(report, &error));
  EXPECT_NE(error.find("id"), std::string::npos);
  EXPECT_THROW(kernscope::to_json(report), kernscope::SchemaViolation);
}

TEST(ReportRejection, NonFiniteMetricThrows) {
  kernscope::KernelReport report = OwnerExampleReport();
  report.metrics["occupancy"] = std::numeric_limits<double>::quiet_NaN();
  std::string error;
  EXPECT_FALSE(kernscope::validate(report, &error));
  EXPECT_NE(error.find("finite"), std::string::npos);
  EXPECT_THROW(kernscope::to_json(report), kernscope::SchemaViolation);
}

} // namespace
