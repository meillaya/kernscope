//===- fixture_main.cpp - golden report artifact dumper ---------------------===//
//
// Wave-3 todo 19: produces the committed-fixture report artifacts the
// todo-20 CLI and todo-56 dashboard consume:
//   <build>/test_reports/fused_attention_report.json
//   <build>/test_reports/fused_attention_report.html
//
// Runs analyze_ptx_module on tests/fixtures/fused_attention.ptx with the
// launch config from tests/fixtures/manifest.json (block size 64, ptxas -v
// register line) + the dynamic smem opt-in from
// expected_fused_attention.json, renders the report with libs/report's
// to_json / to_html (both renderers validate the model first and throw
// SchemaViolation on an invalid model — a written artifact is therefore
// schema-valid by construction), and exits 0. Registered as the ctest
// "analyze-fixture-reports" (LABELS no-gpu). The output tree
// (<build>/test_reports) is gitignored.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "kernscope/analyze/analyze.h"
#include "kernscope/db.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace {

using kernscope::analyze::AnalyzeInputs;
using kernscope::analyze::analyze_ptx_module;
using kernscope::db::json::Value;

std::string read_file(const std::string &path) {
  std::ifstream f(path);
  if (!f) return "";
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

bool load_json(const std::string &path, Value *out, std::string *error) {
  const std::string text = read_file(path);
  if (text.empty()) {
    *error = "cannot read " + path;
    return false;
  }
  const std::optional<Value> parsed = Value::parse(text, *error);
  if (!parsed) return false;
  *out = *parsed;
  return true;
}

// Pointer-safe object lookup. NOTE: `db::json::Value` has non-explicit
// constructors (including Value(bool)); passing a bare `const Value*` where
// a `const Value&` is expected silently converts the pointer to `bool` and
// binds a temporary Value(true) — lookups MUST go through this wrapper.
const Value *child(const Value *obj, const char *key) {
  return (obj != nullptr) ? obj->find(key) : nullptr;
}

double num(const Value *v, double fallback = 0.0) {
  return (v != nullptr) ? v->as_number() : fallback;
}

int main_impl(const char *fixtures_dir, const char *reports_dir) {
  const std::string fixtures = fixtures_dir;

  Value manifest;
  Value expected;
  std::string err;
  if (!load_json(fixtures + "/manifest.json", &manifest, &err) ||
      !load_json(fixtures + "/expected_fused_attention.json", &expected,
                 &err)) {
    std::fprintf(stderr, "fixture_main: %s\n", err.c_str());
    return 1;
  }
  const Value *kern = child(&expected, "kernel");

  const std::string ptx =
      read_file(fixtures + "/fused_attention.ptx");
  if (ptx.empty()) {
    std::fprintf(stderr, "fixture_main: cannot read fused_attention.ptx\n");
    return 1;
  }
  const auto mod = kernscope::parse_ptx_module(ptx);

  AnalyzeInputs inputs;
  const Value *name = kern != nullptr ? child(kern, "name") : nullptr;
  inputs.kernel_name =
      name != nullptr ? name->as_string() : "fused_attention";
  inputs.block_size = static_cast<int>(num(child(&manifest, "block_size"), 0));
  const Value *ptxas = child(&manifest, "ptxas_v_report");
  inputs.ptxas_v = ptxas != nullptr ? ptxas->as_string() : "";
  if (kern != nullptr)
    inputs.dynamic_smem_bytes =
        static_cast<int>(num(child(kern, "dynamic_shared_bytes"), 0));

  const kernscope::KernelReport report = analyze_ptx_module(mod, inputs);

  // Render both artifacts (renderers validate the model first).
  std::string json_text = kernscope::to_json(report);
  std::string html_text = kernscope::to_html(report);

  std::error_code ec;
  std::filesystem::create_directories(reports_dir, ec);
  const std::string json_path =
      std::string(reports_dir) + "/fused_attention_report.json";
  const std::string html_path =
      std::string(reports_dir) + "/fused_attention_report.html";
  {
    std::ofstream f(json_path);
    f << json_text;
  }
  {
    std::ofstream f(html_path);
    f << html_text;
  }

  std::printf("fixture_main: wrote %s (%zu B) and %s (%zu B)\n",
              json_path.c_str(), json_text.size(), html_path.c_str(),
              html_text.size());
  std::printf("fixture_main: kernel=%s occupancy=%g limiter=%s warnings=%zu "
              "suggestions=%zu\n",
              report.kernel_name.c_str(), report.metrics.at("occupancy"),
              report.limiter.c_str(), report.warnings.size(),
              report.suggestions.size());
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  const char *fixtures = FIXTURES_DIR;
  const char *reports = TEST_REPORTS_DIR;
  // Allow explicit overrides (useful for manual runs).
  if (argc >= 2) fixtures = argv[1];
  if (argc >= 3) reports = argv[2];
  return main_impl(fixtures, reports);
}
