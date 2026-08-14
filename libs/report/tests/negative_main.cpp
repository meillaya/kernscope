//===- negative_main.cpp - schema-violation negative path demo --*- C++ -*-===//
//
// QA negative path for libs/report (plan todo 7): a Finding missing a
// required field (severity) must make the renderer return a schema-violation
// error with an asserted exit code + message — never a silent render.
//
// Behavior (asserted):
//   exit 42  - the renderer rejected the invalid finding and the message
//              names the missing field 'severity' (EXPECTED outcome);
//   exit 1   - FAIL: the renderer silently rendered the invalid report;
//   exit 2   - FAIL: a violation was reported but the message does not name
//              the missing field.
//
// The ctest test "report-negative" registers this with WILL_FAIL TRUE, so the
// suite passes exactly when the renderer refuses to emit the invalid report.
//
//===----------------------------------------------------------------------===//

#include "kernscope/report.h"

#include <cstdio>
#include <string>

int main() {
  kernscope::KernelReport report;
  report.kernel_name = "bad_kernel";
  report.metrics = {{"occupancy", 0.375}};
  report.limiter = "registers";
  // Finding WITHOUT the required severity field.
  report.warnings.push_back(
      {"bank_conflict", "", {{"text", "shared memory stride"}}});

  try {
    const std::string json = kernscope::to_json(report);
    std::fprintf(stderr,
                 "FAIL: renderer silently rendered a schema-violating report:\n%s",
                 json.c_str());
    return 1;
  } catch (const kernscope::SchemaViolation &violation) {
    const std::string message = violation.what();
    std::printf("SCHEMA_VIOLATION: %s\n", message.c_str());
    if (message.find("severity") == std::string::npos) {
      std::fprintf(stderr,
                   "FAIL: violation message does not name the missing field:\n%s\n",
                   message.c_str());
      return 2;
    }
    std::printf("OK: renderer rejected the finding; exit code 42 (asserted)\n");
    return 42; // asserted exit code (non-zero: WILL_FAIL ctest passes on this)
  }
}
