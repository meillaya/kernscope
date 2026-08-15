//===- analyze_coalescing.cpp - global coalescing sector model ------------===//
//
// Wave-3 todo 15. See coalescing.h for the design rationale.
//
// Method: a small PTX index-arithmetic resolver turns each ld.global /
// st.global address into a per-lane affine `base + coeff*lane + const`
// (threadIdx.x is lane). Sectors per warp follow the 32B-sector model of
// brief §3.1 (L216-238): each lane's access touches floor(addr/32) sectors;
// efficiency = useful_bytes / (sectors x 32). Classification and the
// warning threshold are pinned to the brief's own definitions.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/coalescing.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

namespace kernscope::analyze {
namespace {

// Pin the warning threshold to the brief's definition of fully-coalesced:
// brief §3.1 (L218-226) defines the fully-coalesced case as 100% sector
// utilization ("100% utilization", "four 32-byte memory transactions" for a
// 128B warp span). Anything below 100% is "not 128B coalesced" (the owner's
// warning wording) and triggers the "coalescing" warning.
constexpr double kCoalescingWarningThreshold = 1.0;

constexpr int kSectorSizeBytes = 32;  // brief §3.1 (L218)
constexpr int kWarpSize = 32;         // NVIDIA warp (CPG Table 30)

// Per-lane affine address: base + coeff*lane + const_off.
struct AffineExpr {
  bool known = false;
  bool has_base = false;  // includes an unresolved base pointer
  int coeff = 0;          // bytes per lane
  long long const_off = 0;
};

using ExprMap = std::map<std::string, AffineExpr>;

long long parse_int(const std::string &s) noexcept {
  char *end = nullptr;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  return end != s.c_str() ? v : 0;
}

std::string operand_reg(const std::string &text) {
  if (text.empty() || text[0] != '%') return "";
  std::string out = "%";
  for (std::size_t i = 1; i < text.size(); ++i) {
    const char c = text[i];
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '.' && c != '<' &&
        c != '>')
      break;
    out += c;
  }
  return out;
}

AffineExpr tid_expr() {
  AffineExpr e;
  e.known = true;
  e.coeff = 1;
  return e;
}

AffineExpr base_pointer_expr() {
  AffineExpr e;
  e.known = true;
  e.has_base = true;
  return e;
}

// Record one instruction's register definition (first-definition-wins so
// loop-carried redefinitions that preserve the tid coefficient keep the
// entry value).
void record_def(const PtxInstruction &insn, ExprMap &exprs) {
  if (insn.operands.empty()) return;
  const std::string dest = operand_reg(insn.operands[0].text);
  if (dest.empty()) return;
  if (exprs.contains(dest)) return;

  const std::string &mn = insn.mnemonic;
  if (mn == "mov" || mn == "mov.u32" || mn == "mov.u64" || mn == "mov.b64" ||
      mn == "mov.b32") {
    if (insn.operands.size() < 2) return;
    const std::string &src = insn.operands[1].text;
    if (src == "%tid.x" || src == "%laneid") {
      exprs[dest] = tid_expr();
    } else if (src == "%tid.y" || src == "%tid.z" || src == "%ctaid.x" ||
               src == "%ctaid.y" || src == "%ctaid.z" || src == "%ntid.x") {
      exprs[dest] = base_pointer_expr();  // uniform across a warp
    } else if (src[0] == '%') {
      auto it = exprs.find(src);
      if (it != exprs.end()) exprs[dest] = it->second;
    }
    return;
  }

  if (mn.rfind("cvta.to.", 0) == 0 && insn.operands.size() >= 2) {
    exprs[dest] = base_pointer_expr();
    return;
  }
  if (mn.rfind("ld.param", 0) == 0) {
    exprs[dest] = base_pointer_expr();
    return;
  }

  const bool is_shl = mn.rfind("shl", 0) == 0;
  const bool is_mul = mn.rfind("mul", 0) == 0;
  const bool is_add = mn.rfind("add", 0) == 0;
  if (!is_shl && !is_mul && !is_add) return;
  if (insn.operands.size() < 3) return;

  if (is_shl) {
    const std::string src = operand_reg(insn.operands[1].text);
    auto it = exprs.find(src);
    if (it == exprs.end()) return;
    const long long shift = parse_int(insn.operands[2].text);
    AffineExpr e = it->second;
    e.coeff = static_cast<int>(e.coeff * (1LL << shift));
    e.const_off *= (1LL << shift);
    exprs[dest] = e;
    return;
  }

  if (is_mul) {
    const std::string src = operand_reg(insn.operands[1].text);
    auto it = exprs.find(src);
    if (it == exprs.end()) return;
    const long long factor = parse_int(insn.operands[2].text);
    AffineExpr e = it->second;
    e.coeff = static_cast<int>(e.coeff * factor);
    e.const_off *= factor;
    exprs[dest] = e;
    return;
  }

  if (is_add) {
    const std::string s1 = operand_reg(insn.operands[1].text);
    const std::string s2 = operand_reg(insn.operands[2].text);
    auto it1 = exprs.find(s1);
    auto it2 = exprs.find(s2);
    if (!s2.empty()) {
      // reg + reg. If either term is an unresolved register the result is
      // unknown (never treat a register name as an immediate -> that would
      // fabricate a "base+0" broadcast and a false coalesced verdict).
      if (it1 == exprs.end() || it2 == exprs.end()) return;
      AffineExpr e = it1->second;
      e.coeff += it2->second.coeff;
      e.const_off += it2->second.const_off;
      e.has_base = e.has_base || it2->second.has_base;
      e.known = it1->second.known && it2->second.known;
      if (e.known) exprs[dest] = e;
      return;
    }
    const long long c = parse_int(insn.operands[2].text);
    if (it1 == exprs.end() && !s1.empty()) return;
    if (it1 != exprs.end()) {
      AffineExpr e = it1->second;
      e.const_off += c;
      exprs[dest] = e;
    }
    return;
  }
}

// Parse an address operand text "[%rd10+4]" -> (register, const offset).
std::pair<std::string, long long> parse_address(const std::string &text) {
  std::string inner = text;
  if (!inner.empty() && inner.front() == '[') inner.erase(0, 1);
  if (!inner.empty() && inner.back() == ']') inner.pop_back();
  const std::string reg = operand_reg(inner);
  std::string rest;
  std::size_t i = reg.size();
  while (i < inner.size() &&
         (inner[i] == '+' || inner[i] == '-' ||
          std::isdigit(static_cast<unsigned char>(inner[i])))) {
    rest += inner[i++];
  }
  return {reg, parse_int(rest)};
}

// Classify one global access from its per-lane affine + access width.
struct PatternResult {
  std::string pattern;  // contiguous-128B | contiguous-partial | strided |
                        // unknown-pattern
  int sectors = 0;
  int bytes_used = 0;
  int bytes_transferred = 0;
  double efficiency = 0.0;
  bool resolved = false;
};

PatternResult classify_pattern(const AffineExpr &e, int width_bytes) {
  PatternResult r;
  if (!e.known || e.coeff < 0) {
    r.pattern = "unknown-pattern";
    return r;
  }
  if (width_bytes <= 0) width_bytes = 4;

  if (e.coeff == 0) {
    // All lanes hit the same base -> broadcast, single sector span.
    const int span = width_bytes;
    r.sectors = (span + kSectorSizeBytes - 1) / kSectorSizeBytes;
    r.bytes_used = kWarpSize * width_bytes;
    r.bytes_transferred = r.sectors * kSectorSizeBytes;
    r.efficiency = 1.0;
    r.resolved = true;
    r.pattern = "contiguous-128B";
    return r;
  }

  std::set<long long> sectors;
  for (int lane = 0; lane < kWarpSize; ++lane) {
    const long long base =
        e.const_off + (static_cast<long long>(e.coeff) * lane);
    sectors.insert(base / kSectorSizeBytes);
    sectors.insert((base + width_bytes - 1) / kSectorSizeBytes);
  }
  r.sectors = static_cast<int>(sectors.size());
  r.bytes_used = kWarpSize * width_bytes;
  r.bytes_transferred = r.sectors * kSectorSizeBytes;
  r.efficiency = r.bytes_transferred > 0
                     ? static_cast<double>(r.bytes_used) /
                           static_cast<double>(r.bytes_transferred)
                     : 0.0;
  r.resolved = true;

  if (r.efficiency >= 1.0 - 1e-12) {
    r.pattern = "contiguous-128B";
  } else if (r.efficiency >= 0.5) {
    r.pattern = "contiguous-partial";
  } else {
    r.pattern = "strided";
  }
  return r;
}

struct CoalescingReport {
  std::optional<PatternResult> worst;
  int vectorized_ld_global = 0;
  int total_global_accesses = 0;
  bool any_unknown = false;
};

void analyze_kernel(const PtxKernel &kernel, CoalescingReport &out) {
  ExprMap exprs;
  auto consider = [&out](const PatternResult &p) {
    if (!p.resolved) return;
    if (!out.worst || p.efficiency < out.worst->efficiency) out.worst = p;
  };

  for (const PtxBlock &block : kernel.cfg) {
    for (const PtxInstruction &insn : block.instructions) {
      const bool is_mem =
          insn.cls == InstructionClass::kLoad ||
          insn.cls == InstructionClass::kStore;
      if (!is_mem || insn.mem_space != StateSpace::kGlobal) {
        record_def(insn, exprs);
        continue;
      }
      ++out.total_global_accesses;
      if (insn.mem_width_bits >= 64 &&
          insn.mnemonic.find(".v") != std::string::npos)
        ++out.vectorized_ld_global;

      bool found_address = false;
      for (const PtxOperand &op : insn.operands) {
        if (!op.is_address) continue;
        found_address = true;
        const auto [reg, off] = parse_address(op.text);
        AffineExpr e;
        if (!reg.empty()) {
          auto it = exprs.find(reg);
          if (it != exprs.end()) e = it->second;
        } else {
          e = base_pointer_expr();
        }
        if (e.known) {
          e.const_off += off;
          const int width =
              insn.mem_width_bits > 0 ? insn.mem_width_bits / 8 : 4;
          const PatternResult p = classify_pattern(e, width);
          if (p.pattern == "unknown-pattern") out.any_unknown = true;
          consider(p);
        } else {
          out.any_unknown = true;
        }
      }
      if (!found_address) record_def(insn, exprs);
    }
  }
}

void emit_finding(const CoalescingReport &c, KernelReport &report) {
  // QA contract: if ANY global access is unresolvable (indirect/
  // data-dependent index) the module must report severity "unknown-pattern"
  // and never claim a coalesced verdict for the kernel — a single
  // data-dependent access invalidates the whole coalescing analysis.
  if (c.any_unknown) {
    Finding f;
    f.id = "coalescing_unknown";
    f.severity = "unknown-pattern";
    f.evidence["text"] =
        "global access address pattern could not be resolved statically "
        "(indirect/data-dependent index); coalescing not assumed";
    f.evidence["vectorized_ld_global"] = std::to_string(c.vectorized_ld_global);
    report.warnings.push_back(std::move(f));
    return;
  }
  if (!c.worst) return;
  const PatternResult &p = *c.worst;

  report.metrics["global_sectors_per_warp"] = p.sectors;
  report.metrics["global_bytes_used_per_warp"] = p.bytes_used;
  report.metrics["global_bytes_transferred_per_warp"] = p.bytes_transferred;
  report.metrics["global_coalescing_efficiency"] = p.efficiency;
  report.metrics["vectorized_ld_global"] = c.vectorized_ld_global;

  if (p.efficiency < kCoalescingWarningThreshold) {
    Finding f;
    f.id = "coalescing";
    f.severity = "warning";
    f.evidence["text"] =
        "global loads not 128B coalesced (brief §3.1 threshold: 100% sector "
        "utilization required)";
    f.evidence["pattern"] = p.pattern;
    f.evidence["bytes_used_per_warp"] = std::to_string(p.bytes_used);
    f.evidence["bytes_transferred_per_warp"] =
        std::to_string(p.bytes_transferred);
    f.evidence["sectors_per_warp"] = std::to_string(p.sectors);
    f.evidence["efficiency"] = std::to_string(p.efficiency);
    f.evidence["vectorized_ld_global"] = std::to_string(c.vectorized_ld_global);
    report.warnings.push_back(std::move(f));
  }
}

}  // namespace

void analyze_coalescing(const PtxModule &module, KernelReport &report) {
  CoalescingReport cr;
  for (const PtxKernel &kernel : module.kernels()) {
    analyze_kernel(kernel, cr);
    if (cr.total_global_accesses > 0) break;
  }
  emit_finding(cr, report);
}

void analyze_coalescing(const kernscope::AmdModule &module,
                        KernelReport &report) {
  // AMD: the LLVM-MC parser gives instruction records without resolved
  // index arithmetic (no SCEV on .s text here), so the honest static
  // verdict for register-indexed loads is "unknown-pattern" - never a
  // fabricated "coalesced". Constant-offset accesses get the sector model.
  int global_loads = 0;
  int vectorized = 0;
  long long bytes_used = 0;
  long long bytes_transferred = 0;
  bool any_unknown = false;

  for (const AmdInstruction &insn : module.instructions) {
    const bool is_global = insn.klass == InstrClass::Global ||
                           insn.klass == InstrClass::Buffer ||
                           insn.klass == InstrClass::Flat;
    if (!is_global) continue;
    ++global_loads;
    bool has_data_index = false;
    for (std::size_t i = 1; i < insn.operands.size(); ++i) {
      const std::string &t = insn.operands[i].text;
      if (!t.empty() &&
          (t[0] == 'v' || t[0] == 's' || t[0] == 'a' || t[0] == 'f')) {
        has_data_index = true;
        break;
      }
    }
    if (has_data_index) {
      any_unknown = true;
    } else {
      const int width = 4;
      const int sectors = (width + kSectorSizeBytes - 1) / kSectorSizeBytes;
      bytes_used += kWarpSize * width;
      bytes_transferred += sectors * kSectorSizeBytes;
    }
  }
  if (global_loads == 0) return;

  if (any_unknown) {
    Finding f;
    f.id = "coalescing_unknown";
    f.severity = "unknown-pattern";
    f.evidence["text"] =
        "AMD global accesses use data-dependent (register-indexed) addresses "
        "that the static .s analysis cannot resolve; coalescing not assumed";
    f.evidence["vectorized_ld_global"] = std::to_string(vectorized);
    report.warnings.push_back(std::move(f));
    return;
  }
  const double eff =
      bytes_transferred > 0
          ? static_cast<double>(bytes_used) /
                static_cast<double>(bytes_transferred)
          : 1.0;
  report.metrics["global_coalescing_efficiency"] = eff;
  report.metrics["vectorized_ld_global"] = vectorized;
  if (eff < kCoalescingWarningThreshold) {
    Finding f;
    f.id = "coalescing";
    f.severity = "warning";
    f.evidence["text"] = "global loads not fully coalesced";
    f.evidence["efficiency"] = std::to_string(eff);
    f.evidence["vectorized_ld_global"] = std::to_string(vectorized);
    report.warnings.push_back(std::move(f));
  }
}

}  // namespace kernscope::analyze
