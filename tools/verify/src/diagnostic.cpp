#include <kernscope/verify/diagnostic.h>

#include <string>

namespace kernscope {
namespace verify {

std::string check_class_name(int class_id) {
  switch (class_id) {
    case CheckClass::kGlobalOob:
      return "global out-of-bounds";
    case CheckClass::kSmemBounds:
      return "shared-memory bounds";
    case CheckClass::kBarrier:
      return "barrier correctness";
    case CheckClass::kRace:
      return "race-prone shared writes";
    case CheckClass::kPtrArith:
      return "illegal pointer arithmetic";
    case CheckClass::kDivergentBarrier:
      return "divergent barriers";
    case CheckClass::kUninitRead:
      return "uninitialized reads";
    case CheckClass::kAddrSpaceCast:
      return "invalid address-space casts";
    case EngineClass::kSymbolicDims:
      return "symbolic launch dims";
    case EngineClass::kScevReconstructionFailed:
      return "SCEV reconstruction failed";
    case EngineClass::kBudgetExceeded:
      return "budget exceeded";
    default:
      return "";
  }
}

const char* severity_name(Severity s) {
  switch (s) {
    case Severity::Info:
      return "info";
    case Severity::Warning:
      return "warning";
    case Severity::Error:
      return "error";
  }
  return "unknown";
}

std::string SourceLoc::render() const {
  if (!valid()) {
    return "<unknown>";
  }
  return file + ":" + std::to_string(line) + ":" + std::to_string(col);
}

std::string Diagnostic::render() const {
  std::string name = class_name.empty() ? check_class_name(class_id) : class_name;
  if (name.empty()) {
    name = "class " + std::to_string(class_id);
  }
  std::string out = severity_name(severity);
  out += " [";
  out += name;
  out += "] ";
  out += loc.render();
  if (!function.empty()) {
    out += " (" + function + ")";
  }
  out += ": ";
  out += message;
  if (!instruction.empty()) {
    out += " [at " + instruction + "]";
  }
  for (const std::string& step : path) {
    out += "\n    path: ";
    out += step;
  }
  return out;
}

}  // namespace verify
}  // namespace kernscope
