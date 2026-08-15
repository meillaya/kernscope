//===- analyze_banks.cpp - shared-memory bank conflict analysis -----------===//
//
// Wave-3 todo 16. See banks.h for the design rationale.
//
// Method: reuse the per-lane affine resolver idea from the coalescing module
// (ld.shared/st.shared address = base + stride_bytes*lane + const; tid is
// lane). Bank model per brief §2.1 (L158-184): 32 banks x 4-byte words,
// bank(word) = (word_addr/4) mod 32; count multiplicity per bank over a
// warp. way = max multiplicity; distinct = number of touched banks.
//   stride_words 68 -> stride_mod_32 4 -> banks {0,4,...,28} -> 8 distinct,
//   4-way (owner's golden, cross-checked against expected JSON).
// Padded stride 69 (+1 pad idiom, brief §2.1 L173-179): 69 mod 32 = 5,
// gcd(5,32)=1 -> every lane on a distinct bank -> conflict-free.
// A data-dependent (unresolvable) index is classified "unknown" — never
// conflict-free, never a fabricated bank count (QA failure path).
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/banks.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kernscope::analyze {
namespace {

// NVIDIA: 32 shared-memory banks, 4 bytes each (brief §2.1 L159-171,
// CPG §2.3.4.2 + Table 31). RDNA3 LDS is 32-bank DWORD too (brief §2.2).
constexpr int kNumBanks = 32;
constexpr int kWarpSize = 32;

// Same affine resolver shape as the coalescing module (kept local so each
// analysis module stays self-contained; the fused_attention shared loads go
// through the same mul/shl/add/mad chains as its global loads).
struct AffineExpr {
  bool known = false;
  bool has_base = false;
  int coeff = 0;  // bytes per lane
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
    } else if (src == "smem" || src == "%tid.y" || src == "%tid.z" ||
               src == "%ctaid.x" || src == "%ctaid.y" || src == "%ctaid.z" ||
               src == "%ntid.x") {
      exprs[dest] = base_pointer_expr();  // smem/blockIdx: uniform per warp
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
  const bool is_mad = mn.rfind("mad", 0) == 0;
  const bool is_add = mn.rfind("add", 0) == 0;
  if (!is_shl && !is_mul && !is_mad && !is_add) return;

  if (is_shl) {
    if (insn.operands.size() < 3) return;
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

  if (is_mul || is_mad) {
    if (insn.operands.size() < 3) return;
    const std::string src = operand_reg(insn.operands[1].text);
    auto it = exprs.find(src);
    if (it == exprs.end()) return;
    const long long factor = parse_int(insn.operands[2].text);
    AffineExpr e = it->second;
    e.coeff = static_cast<int>(e.coeff * factor);
    e.const_off *= factor;
    if (is_mad && insn.operands.size() >= 4) {
      const std::string b = operand_reg(insn.operands[3].text);
      auto itb = exprs.find(b);
      if (itb == exprs.end()) return;
      e.coeff += itb->second.coeff;
      e.const_off += itb->second.const_off;
    }
    exprs[dest] = e;
    return;
  }

  if (is_add) {
    if (insn.operands.size() < 3) return;
    const std::string s1 = operand_reg(insn.operands[1].text);
    const std::string s2 = operand_reg(insn.operands[2].text);
    auto it1 = exprs.find(s1);
    auto it2 = exprs.find(s2);
    if (!s2.empty()) {
      // reg + reg. If either term is an unresolved register the result is
      // unknown (never treat a register name as an immediate).
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

struct BankResult {
  bool resolved = false;
  int stride_words = 0;
  int stride_mod_32 = 0;
  int distinct_banks = 0;
  int way = 0;  // max multiplicity per bank (1 = conflict-free)
};

// Bank footprint over a warp: word address = (const + stride*lane)/4.
// bank(word) = word mod 32. way = max count per bank; distinct = banks
// touched. stride 0 (broadcast): all lanes hit the SAME word -> every lane
// reads it conflict-free (broadcast, brief §2.1 L164-169) -> way 1.
BankResult compute_banks(int stride_bytes, long long base_word) {
  BankResult r;
  const int stride_words = stride_bytes / 4;
  r.stride_words = stride_words;
  r.stride_mod_32 = ((stride_words % kNumBanks) + kNumBanks) % kNumBanks;

  if (stride_words == 0) {
    r.distinct_banks = 1;
    r.way = 1;  // broadcast: conflict-free
    r.resolved = true;
    return r;
  }

  std::map<int, int> bank_count;
  for (int lane = 0; lane < kWarpSize; ++lane) {
    const long long word =
        base_word + (static_cast<long long>(stride_words) * lane);
    const int bank =
        static_cast<int>((word % kNumBanks + kNumBanks) % kNumBanks);
    bank_count[bank]++;
  }
  r.distinct_banks = static_cast<int>(bank_count.size());
  r.way = 0;
  for (const auto &[bank, count] : bank_count) r.way = std::max(r.way, count);
  r.resolved = true;
  return r;
}

struct BanksReport {
  std::optional<BankResult> worst;
  int accesses = 0;
  bool any_unknown = false;
};

void analyze_kernel(const PtxKernel &kernel, BanksReport &out) {
  ExprMap exprs;
  auto consider = [&out](const BankResult &b) {
    if (!b.resolved) return;
    if (!out.worst || b.way > out.worst->way) out.worst = b;
  };

  for (const PtxBlock &block : kernel.cfg) {
    for (const PtxInstruction &insn : block.instructions) {
      const bool is_mem = insn.cls == InstructionClass::kLoad ||
                          insn.cls == InstructionClass::kStore;
      if (!is_mem || insn.mem_space != StateSpace::kShared) {
        record_def(insn, exprs);
        continue;
      }
      ++out.accesses;

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
        if (!e.known) {
          out.any_unknown = true;
          continue;
        }
        e.const_off += off;
        const long long base_word = e.const_off / 4;
        consider(compute_banks(e.coeff, base_word));
      }
      if (!found_address) record_def(insn, exprs);
    }
  }
}

void emit_finding(const BanksReport &b, KernelReport &report) {
  if (b.accesses == 0) return;
  // QA contract: a data-dependent smem index is "unknown", never silently
  // conflict-free. If a REAL conflict resolved, report it (fused_attention
  // has both patterns); otherwise any unknown access wins over a
  // conflict-free reading (a broadcast index-read is not evidence that the
  // kernel is conflict-free).
  if (b.worst && b.worst->way > 1) {
    const BankResult &r = *b.worst;
    report.metrics["shared_stride_words"] = r.stride_words;
    report.metrics["shared_stride_mod_32"] = r.stride_mod_32;
    report.metrics["shared_distinct_banks"] = r.distinct_banks;
    report.metrics["shared_bank_way"] = r.way;

    Finding f;
    f.id = "bank_conflict";
    f.severity = "warning";
    f.evidence["text"] = "shared memory stride likely causes " +
                         std::to_string(r.way) + "-way bank conflicts";
    f.evidence["stride_words"] = std::to_string(r.stride_words);
    f.evidence["stride_mod_32"] = std::to_string(r.stride_mod_32);
    f.evidence["banks_per_sm"] = std::to_string(kNumBanks);
    f.evidence["distinct_banks"] = std::to_string(r.distinct_banks);
    f.evidence["way"] = std::to_string(r.way);
    f.evidence["suggestion"] =
        "pad the shared array row by one element (+1 pad) or apply an XOR "
        "swizzle (brief §2.1 L173-179)";
    report.warnings.push_back(std::move(f));
    return;
  }

  if (b.any_unknown) {
    Finding f;
    f.id = "bank_conflict_unknown";
    f.severity = "unknown-pattern";
    f.evidence["text"] =
        "shared-memory access index is data-dependent; bank footprint "
        "cannot be determined statically";
    report.warnings.push_back(std::move(f));
    return;
  }

  // Conflict-free (or broadcast): report the clean verdict. `worst` is
  // engaged whenever accesses > 0 and nothing is unknown (every known-
  // address shared access runs consider() with a resolved result); the guard
  // keeps the deref honest for the analyzer.
  if (b.worst) {
    const BankResult &r = *b.worst;
    report.metrics["shared_stride_words"] = r.stride_words;
    report.metrics["shared_stride_mod_32"] = r.stride_mod_32;
    report.metrics["shared_distinct_banks"] = r.distinct_banks;
    report.metrics["shared_bank_way"] = r.way;
  }
}

}  // namespace

void analyze_banks(const PtxModule &module, KernelReport &report) {
  BanksReport br;
  for (const PtxKernel &kernel : module.kernels()) {
    analyze_kernel(kernel, br);
    if (br.accesses > 0) break;
  }
  emit_finding(br, report);
}

void analyze_banks(const kernscope::AmdModule &module, KernelReport &report) {
  // AMD: ds_read_*/ds_write_* on the same 32-bank DWORD LDS model
  // (brief §2.2 L188-194; wave32 RDNA3 groups 32 lanes). The LLVM-MC
  // records expose the constant offset only when the operand is an
  // immediate; register-indexed addresses are data-dependent -> unknown.
  std::optional<BankResult> worst;
  bool any_unknown = false;
  int accesses = 0;

  for (const AmdInstruction &insn : module.instructions) {
    if (insn.klass != InstrClass::LDS) continue;
    ++accesses;
    bool has_data_index = false;
    for (std::size_t i = 1; i < insn.operands.size(); ++i) {
      const std::string &t = insn.operands[i].text;
      if (!t.empty() && (t[0] == 'v' || t[0] == 's')) {
        has_data_index = true;
        break;
      }
    }
    if (has_data_index) {
      any_unknown = true;
      continue;
    }
    long long offset_bytes = 0;
    for (const AmdOperand &op : insn.operands) {
      if (op.kind == AmdOperand::Kind::Immediate) {
        offset_bytes = parse_int(op.text);
      }
    }
    const BankResult b = compute_banks(0, offset_bytes / 4);
    if (!worst || b.way > worst->way) worst = b;
  }
  if (accesses == 0) return;

  if (!worst) {
    if (any_unknown) {
      Finding f;
      f.id = "bank_conflict_unknown";
      f.severity = "unknown-pattern";
      f.evidence["text"] =
          "LDS access index is data-dependent; bank footprint cannot be "
          "determined statically";
      report.warnings.push_back(std::move(f));
    }
    return;
  }
  const BankResult &r = *worst;
  report.metrics["shared_distinct_banks"] = r.distinct_banks;
  report.metrics["shared_bank_way"] = r.way;
  if (r.way > 1) {
    Finding f;
    f.id = "bank_conflict";
    f.severity = "warning";
    f.evidence["text"] = "LDS stride likely causes " +
                         std::to_string(r.way) + "-way bank conflicts";
    f.evidence["banks_per_sm"] = std::to_string(kNumBanks);
    f.evidence["distinct_banks"] = std::to_string(r.distinct_banks);
    f.evidence["way"] = std::to_string(r.way);
    report.warnings.push_back(std::move(f));
  }
}

}  // namespace kernscope::analyze
