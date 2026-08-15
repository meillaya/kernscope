//===- analyze_regs.cpp - register pressure + spill detection -------------===//
//
// Wave-3 todo 13. See regs.h for the full design rationale and citations.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/regs.h"

#include <cctype>
#include <cstdlib>
#include <regex>
#include <string>

namespace kernscope::analyze {
namespace {

// Extract the declared register count from a .reg variable name:
//   "%f<5510>"  -> 5510   (register array)
//   "%r1"       -> 1      (single register)
// Returns 0 when the name carries no register token (defensive).
int register_count_of(const std::string &name) noexcept {
  const auto lt = name.find('<');
  const auto gt = name.rfind('>');
  if (lt != std::string::npos && gt != std::string::npos && gt > lt) {
    const std::string n = name.substr(lt + 1, gt - lt - 1);
    char *end = nullptr;
    const long v = std::strtol(n.c_str(), &end, 10);
    if (end != n.c_str() && v > 0) return static_cast<int>(v);
  }
  return name.empty() ? 0 : 1;
}

// Sum of .local array shapes "[NNN]" for one kernel body (stack frame bytes).
int local_frame_bytes(const PtxKernel &kernel) noexcept {
  int total = 0;
  for (const PtxVariable &v : kernel.variables) {
    if (v.space != StateSpace::kLocal) continue;
    if (v.shape.size() >= 2 && v.shape.front() == '[' && v.shape.back() == ']') {
      const std::string n = v.shape.substr(1, v.shape.size() - 2);
      char *end = nullptr;
      const long bytes = std::strtol(n.c_str(), &end, 10);
      if (end != n.c_str() && bytes > 0) total += static_cast<int>(bytes);
    }
  }
  return total;
}

// Count spill traffic from PTX: ld.local/st.local (also atom/red .local,
// counted by the parser's local_access_count).
struct SpillTraffic {
  int stores = 0;
  int loads = 0;
  int total = 0;
};
SpillTraffic count_spill_traffic(const PtxKernel &kernel) noexcept {
  SpillTraffic t;
  for (const PtxBlock &block : kernel.cfg) {
    for (const PtxInstruction &insn : block.instructions) {
      if (insn.mem_space != StateSpace::kLocal) continue;
      if (insn.cls == InstructionClass::kStore) ++t.stores;
      else if (insn.cls == InstructionClass::kLoad) ++t.loads;
    }
  }
  t.total = t.stores + t.loads;
  return t;
}

void add_spill_finding(KernelReport &report, int regs_per_thread,
                       int stack_bytes, int spill_stores,
                       int spill_loads, int spill_bytes) {
  Finding f;
  f.id = "spill";
  f.severity = "warning";
  f.evidence["text"] = "kernel spills to local memory (ld.local/st.local)";
  f.evidence["regs_per_thread"] = std::to_string(regs_per_thread);
  f.evidence["spill_bytes"] = std::to_string(spill_bytes);
  f.evidence["stack_bytes"] = std::to_string(stack_bytes);
  f.evidence["spill_stores"] = std::to_string(spill_stores);
  f.evidence["spill_loads"] = std::to_string(spill_loads);
  report.warnings.push_back(std::move(f));
}

// Shared filling logic given a regs figure + spill traffic.
// spill_stores/spill_loads are instruction COUNTS; spill_bytes is a byte
// figure (ptxas spill store/load bytes + stack frame; the static proxy has
// no per-op byte counts, so it uses the .local frame size).
void fill_report(KernelReport &report, int regs_per_thread, int stack_bytes,
                 int spill_stores, int spill_loads, int spill_bytes) {
  report.metrics["registers_per_thread"] = regs_per_thread;
  report.metrics["spill_stores"] = spill_stores;
  report.metrics["spill_loads"] = spill_loads;
  report.metrics["local_stack_bytes"] = stack_bytes;
  report.metrics["spill_bytes"] = spill_bytes;
  if (spill_stores > 0 || spill_loads > 0 || stack_bytes > 0) {
    add_spill_finding(report, regs_per_thread, stack_bytes, spill_stores,
                      spill_loads, spill_bytes);
  }
}

void add_cannot_estimate_finding(KernelReport &report) {
  Finding f;
  f.id = "cannot_estimate_registers";
  f.severity = "warning";
  f.evidence["text"] =
      "cannot estimate register pressure: kernel declares no .reg registers";
  report.warnings.push_back(std::move(f));
}

}  // namespace

PtxasVReport parse_ptxas_v(std::string_view text) noexcept {
  PtxasVReport out;
  const std::string s(text);

  // All regexes here are constant and valid, so no throw is possible in
  // practice; the guard keeps the noexcept contract honest (a throw inside a
  // noexcept function would terminate instead of returning {ok=false}).
  try {
    // Validity anchor: the "N registers" token appears in BOTH ptxas formats
    // [brief §6.1 L401-408; polykernel PtxasParser oracle]. No register line
    // -> the input is not a ptxas log -> {ok=false}.
    std::smatch m;
    const std::regex registers_re(R"(([0-9]+) registers)");
    if (!std::regex_search(s, m, registers_re)) return out;
    out.regs_per_thread = std::atoi(m[1].str().c_str());
    out.ok = true;

    // Optional fields (independent regex per field; absent -> 0).
    const auto grab = [&s](const char *pattern) -> int {
      std::smatch mm;
      if (std::regex_search(s, mm, std::regex(pattern))) {
        return std::atoi(mm[1].str().c_str());
      }
      return 0;
    };
    out.stack_frame_bytes = grab(R"(([0-9]+) bytes stack frame)");
    out.spill_stores_bytes = grab(R"(([0-9]+) bytes spill stores)");
    out.spill_loads_bytes = grab(R"(([0-9]+) bytes spill loads)");
    out.smem_bytes = grab(R"(([0-9]+) bytes smem)");
    out.cmem_bytes = grab(R"(([0-9]+) bytes cmem)");
  } catch (...) {
    return out;
  }
  return out;
}

RegPressureEstimate estimate_reg_pressure(const PtxKernel &kernel) noexcept {
  RegPressureEstimate est;
  for (const PtxVariable &v : kernel.variables) {
    if (v.space != StateSpace::kReg) continue;
    est.declared_regs += register_count_of(v.name);
  }
  est.stack_frame_bytes = local_frame_bytes(kernel);
  const SpillTraffic t = count_spill_traffic(kernel);
  est.spill_stores = t.stores;
  est.spill_loads = t.loads;
  est.local_access_count = t.total;
  est.can_estimate = est.declared_regs > 0;
  return est;
}

void analyze_regs(const PtxModule &module, KernelReport &report) {
  for (const PtxKernel &kernel : module.kernels()) {
    const RegPressureEstimate est = estimate_reg_pressure(kernel);
    if (!est.can_estimate) {
      add_cannot_estimate_finding(report);
      return;
    }
    fill_report(report, est.declared_regs, est.stack_frame_bytes,
                est.spill_stores, est.spill_loads, est.stack_frame_bytes);
    return;  // first .entry / .func analyzed (todo 20 picks the entry)
  }
  add_cannot_estimate_finding(report);
}

void analyze_regs(const PtxModule &module, std::string_view ptxas_v,
                  KernelReport &report) {
  const PtxasVReport v = parse_ptxas_v(ptxas_v);
  if (v.ok) {
    // ptxas -v is authoritative when present (cubin build / manifest).
    const int spill_bytes =
        v.stack_frame_bytes + v.spill_stores_bytes + v.spill_loads_bytes;
    fill_report(report, v.regs_per_thread, v.stack_frame_bytes,
                v.spill_stores_bytes, v.spill_loads_bytes, spill_bytes);
    return;
  }
  analyze_regs(module, report);
}

void analyze_regs(const AmdModule &module, KernelReport &report) {
  // AMD: scratch_load/scratch_store are the spill class [brief §6.2
  // L414-426]. amdhsa next_free_vgpr lives in the code-object notes
  // (llvm-readobj --notes), not in the .s text, so the .s-level signal is
  // the scratch op census; the vgpr figure is code-object-only.
  int scratch_stores = 0;
  int scratch_loads = 0;
  for (const AmdInstruction &insn : module.instructions) {
    if (insn.klass != InstrClass::Scratch) continue;
    if (insn.is_store) ++scratch_stores;
    else if (insn.is_load) ++scratch_loads;
  }
  // No register figure is derivable from assembly text alone: report the
  // scratch census as the spill evidence and mark register pressure as
  // unavailable (never fabricate a vgpr count).
  report.metrics["spill_stores"] = scratch_stores;
  report.metrics["spill_loads"] = scratch_loads;
  report.metrics["local_stack_bytes"] = 0;
  report.metrics["spill_bytes"] = scratch_stores + scratch_loads;
  if (scratch_stores > 0 || scratch_loads > 0) {
    Finding f;
    f.id = "spill";
    f.severity = "warning";
    f.evidence["text"] =
        "kernel spills via scratch_load/scratch_store (amdhsa "
        "next_free_vgpr is code-object-only, see llvm-readobj --notes)";
    f.evidence["spill_stores"] = std::to_string(scratch_stores);
    f.evidence["spill_loads"] = std::to_string(scratch_loads);
    f.evidence["spill_bytes"] =
        std::to_string(scratch_stores + scratch_loads);
    report.warnings.push_back(std::move(f));
  }
}

}  // namespace kernscope::analyze
