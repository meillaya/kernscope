// kernscope::verify::ptr — PointerDisciplineCheckPlugin (todo 34): check
// classes 5 (illegal pointer arithmetic) + 7 (uninitialized reads) + 8
// (invalid address-space casts).
//
// CLASS 5 (PTR_OOB, Warning) — illegal pointer arithmetic, the eBPF
// sanitize_ptr_alu typed-pointer discipline analog (research brief §1):
//   1. Every GEP in the kernel is resolved to its allocation root (device
//      global / shared array / extern shared / local alloca) and its byte
//      span is computed from the GEP index expressions via
//      VerifierContext::queryRange (lattice first, ScalarEvolution second).
//   2. A pointer VALUE escaping its statically sized allocation is PTR_OOB
//      (pointer arithmetic must stay within the allocation bounds; a
//      one-past-end pointer is legal C, but spans whose max offset exceeds
//      the allocation by the recovered element width are conservatively
//      flagged — see README.md scope).
//   3. Unbounded (data-dependent) index expressions are reported PTR_OOB —
//      may-analysis: no in-bounds claim is made (soundness over false
//      safety).
//   4. Subtracting pointers rooted in DIFFERENT allocations (ptrtoint pairs
//      under a `sub`) is flagged PTR_OOB (unrelated-pointer subtraction is
//      UB); same-allocation subtraction stays SAFE.
// Kernel parameters (runtime allocation size) are never the subject of a
// static escape claim.
//
// Registered via REGISTER_CHECK: the static registrar hands an instance to
// the global CheckRegistry — no shared file edits (todo 30 contract).
//
// Skeleton commit 81cedfe registered id "ptr" with classes {5,7,8} and an
// empty run(); class analyses land in follow-up commits (5 -> 8 -> 7).

#include "ptr_common.h"

#include <kernscope/verify/check_plugin.h>

#include <llvm/ADT/SmallString.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Operator.h>

#include <optional>
#include <string>
#include <vector>

namespace kernscope {
namespace verify {
namespace ptr {

namespace {

using llvm::ConstantRange;
using llvm::GetElementPtrInst;
using llvm::Instruction;
using llvm::Value;

// DebugLoc-backed source location (empty when the IR carries no debug info).
SourceLoc loc_of(const Instruction* i) {
  SourceLoc loc;
  if (const llvm::DebugLoc dl = i->getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}

// Short instruction rendering for the diagnostic.
std::string short_instruction(const Instruction& i) {
  std::string s;
  llvm::raw_string_ostream os(s);
  i.print(os);
  os.flush();
  if (s.size() > 120) {
    s.resize(120);
    s += "...";
  }
  return s;
}

// One class-5 finding.
struct PtrFinding {
  enum class Kind {
    InBounds,     ///< pointer arithmetic proven inside the allocation.
    Escaped,      ///< pointer VALUE provably outside its allocation.
    Unbounded,    ///< index range unbounded -> may-analysis PTR_OOB.
    UnrelatedSub, ///< subtraction of pointers from different allocations.
  };
  Kind kind = Kind::InBounds;
  SourceLoc loc;
  std::string instruction;
  std::string base;
  uint64_t bound = 0;
  std::string index_range;
  std::string span;
  std::string detail;
};

// Class-5 analysis of one GEP (pointer arithmetic escaping an allocation).
// GEPOperator covers both GEP instructions and ConstantExpr GEPs (clang 21
// folds `g_buf + 20000` into a ConstantExpr inside the store operand).
std::optional<PtrFinding> analyze_gep(llvm::GEPOperator& gep,
                                      VerifierContext& ctx) {
  const AllocationInfo alloc = resolveAllocation(&gep, ctx.module);
  if (!alloc.known_size) {
    return std::nullopt;  // kernel param / unknown root: no static claim.
  }
  PtrFinding f;
  if (const auto* inst = llvm::dyn_cast<Instruction>(&gep)) {
    f.loc = loc_of(inst);
    f.instruction = short_instruction(*inst);
  } else {
    f.instruction = "constant-expr GEP";  // folded operand; no own DebugLoc.
  }
  f.base = alloc.name;
  f.bound = alloc.size_bytes;
  std::string index_range;
  uint64_t stride = 0;
  const std::optional<ConstantRange> span =
      gepByteSpan(&gep, ctx, index_range, stride);
  f.index_range = index_range;
  if (!span) {
    // Unbounded index (data-dependent): may-analysis, never SAFE.
    f.kind = PtrFinding::Kind::Unbounded;
    f.detail = "index expression cannot be bounded (lattice/SCEV): "
               "may-analysis — pointer escape cannot be ruled out";
    return f;
  }
  f.span = range_render(*span);
  // Pointer-VALUE check: the exclusive span upper holds max_offset +
  // element_width; recover the max possible pointer offset. Constant-index
  // GEPs record no stride, so the conservative (max_offset + width - 1) is
  // used — one-past-end pointers are legal C but are conservatively
  // flagged when the element width is unrecoverable (documented scope).
  llvm::APInt max_off = span->getUpper();
  max_off -= stride > 0 ? llvm::APInt(64, stride) : llvm::APInt(64, 1);
  const bool in_bounds = max_off.ule(alloc.size_bytes);
  f.kind = in_bounds ? PtrFinding::Kind::InBounds : PtrFinding::Kind::Escaped;
  f.detail = in_bounds
                 ? std::string("pointer arithmetic stays within allocation ") +
                       alloc.name + " (" +
                       std::to_string(alloc.size_bytes) + " bytes)"
                 : std::string("pointer arithmetic escapes allocation ") +
                       alloc.name + " (" +
                       std::to_string(alloc.size_bytes) + " bytes): max "
                       "offset " +
                       [&] {
                         llvm::SmallString<24> s;
                         max_off.toString(s, 10, false);
                         return std::string(s);
                       }() + " > size";
  return f;
}

// Resolves the allocation a ptrtoint-derived value is rooted in.
std::optional<AllocationInfo> ptrtoint_root(Value* v, const llvm::Module& m) {
  if (auto* pt = llvm::dyn_cast<llvm::PtrToIntInst>(v)) {
    return resolveAllocation(pt->getPointerOperand(), m);
  }
  if (auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(v)) {
    if (ce->getOpcode() == Instruction::PtrToInt) {
      return resolveAllocation(ce->getOperand(0), m);
    }
  }
  return std::nullopt;
}

// Class-5 analysis of one pointer-difference (`sub` of ptrtoint pairs):
// subtracting pointers rooted in DIFFERENT known allocations is UB.
std::optional<PtrFinding> analyze_ptr_diff(llvm::BinaryOperator& sub,
                                           VerifierContext& ctx) {
  if (sub.getOpcode() != llvm::Instruction::Sub) {
    return std::nullopt;
  }
  const std::optional<AllocationInfo> a =
      ptrtoint_root(sub.getOperand(0), ctx.module);
  const std::optional<AllocationInfo> b =
      ptrtoint_root(sub.getOperand(1), ctx.module);
  if (!a || !b || !a->known_size || !b->known_size) {
    return std::nullopt;  // not provably unrelated known allocations
  }
  if (a->name == b->name && a->size_bytes == b->size_bytes) {
    return std::nullopt;  // same allocation: legal subtraction.
  }
  PtrFinding f;
  f.kind = PtrFinding::Kind::UnrelatedSub;
  f.loc = loc_of(&sub);
  f.instruction = short_instruction(sub);
  f.base = a->name;
  f.bound = a->size_bytes;
  f.detail = "subtraction of pointers rooted in unrelated allocations "
             "(" + a->name + " vs " + b->name + ") — undefined behavior";
  return f;
}

const char* kind_name(PtrFinding::Kind k) {
  switch (k) {
    case PtrFinding::Kind::InBounds:
      return "IN_BOUNDS";
    case PtrFinding::Kind::Escaped:
      return "ESCAPED";
    case PtrFinding::Kind::Unbounded:
      return "UNBOUNDED";
    case PtrFinding::Kind::UnrelatedSub:
      return "UNRELATED_SUB";
  }
  return "?";
}

// JSON rendering of one class-5 finding (module-owned format).
std::string finding_json(const PtrFinding& f) {
  std::string out = "{\"op\":\"ptr_arith\"";
  out += ",\"line\":" + std::to_string(f.loc.line);
  out += ",\"alloc\":\"" + jsonEscape(f.base) + "\"";
  out += ",\"bound\":" + std::to_string(f.bound);
  out += ",\"range\":\"" + jsonEscape(f.index_range) + "\"";
  out += ",\"span\":\"" + jsonEscape(f.span) + "\"";
  out += ",\"kind\":\"" + std::string(kind_name(f.kind)) + "\"";
  out += ",\"note\":\"" + jsonEscape(f.detail) + "\"}";
  return out;
}

}  // namespace

class PointerDisciplineCheckPlugin : public CheckPlugin {
 public:
  PointerDisciplineCheckPlugin() {
    id = "ptr";
    name = "Pointer discipline: arithmetic bounds + address-space casts + "
           "uninitialized reads";
    check_class_ids = {CheckClass::kPtrArith, CheckClass::kUninitRead,
                       CheckClass::kAddrSpaceCast};
  }

  std::vector<Diagnostic> run(VerifierContext& ctx) const override {
    std::vector<Diagnostic> out;
    llvm::Function& F = *ctx.kernel.function;
    const std::string& fn = ctx.kernel.name;
    Budget& budget = ctx.diagnostics.budget();

    // --- class 5: illegal pointer arithmetic ------------------------------
    std::vector<PtrFinding> findings;
    llvm::SmallPtrSet<const llvm::Value*, 16> seen_geps;
    for (llvm::BasicBlock& bb : F) {
      for (Instruction& i : bb) {
        if (!budget.acquire(fn, 1)) {
          break;
        }
        // GEP instructions themselves.
        if (auto* gep = llvm::dyn_cast<llvm::GEPOperator>(&i)) {
          if (seen_geps.insert(gep).second) {
            if (auto f = analyze_gep(*gep, ctx)) {
              findings.push_back(std::move(*f));
            }
          }
        }
        // ConstantExpr GEPs folded into operands (clang 21: `g_buf + K`
        // becomes an inline GEP expression inside the store operand).
        for (const llvm::Use& u : i.operands()) {
          auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(u.get());
          if (ce == nullptr || ce->getOpcode() != llvm::Instruction::GetElementPtr) {
            continue;
          }
          if (!seen_geps.insert(ce).second) {
            continue;
          }
          auto* cgep = llvm::dyn_cast<llvm::GEPOperator>(
              static_cast<llvm::Value*>(ce));
          if (cgep == nullptr) {
            continue;
          }
          if (auto f = analyze_gep(*cgep, ctx)) {
            // Attach the enclosing instruction's location to the folded
            // GEP finding (ConstantExprs carry no DebugLoc of their own).
            if (!f->loc.valid()) {
              f->loc = loc_of(&i);
            }
            findings.push_back(std::move(*f));
          }
        }
        if (auto* bin = llvm::dyn_cast<llvm::BinaryOperator>(&i)) {
          if (auto f = analyze_ptr_diff(*bin, ctx)) {
            findings.push_back(std::move(*f));
          }
        }
      }
    }
    if (!findings.empty()) {
      const size_t bad = std::count_if(
          findings.begin(), findings.end(), [](const PtrFinding& f) {
            return f.kind != PtrFinding::Kind::InBounds;
          });
      const bool safe = bad == 0;
      Diagnostic d;
      d.severity = safe ? Severity::Info : Severity::Warning;
      d.class_id = CheckClass::kPtrArith;
      d.class_name = check_class_name(d.class_id);
      d.function = fn;
      d.loc = findings.front().loc;
      std::string findings_json = "[";
      for (size_t k = 0; k < findings.size(); ++k) {
        if (k) {
          findings_json += ",";
        }
        findings_json += finding_json(findings[k]);
      }
      findings_json += "]";
      d.message = "pointer-arithmetic discipline check: " +
                  std::to_string(findings.size()) + " pointer expression(s), " +
                  std::to_string(bad) + " possibly escaping or unrelated; "
                  "verdict_json: {\"check\":" +
                  std::to_string(CheckClass::kPtrArith) + ",\"kernel\":\"" +
                  fn + "\",\"verdict\":\"" +
                  (safe ? kVerdictSafe : kVerdictPtrOob) +
                  "\",\"pointers\":" + std::to_string(findings.size()) +
                  ",\"bad\":" + std::to_string(bad) + ",\"findings\":" +
                  findings_json + "}";
      out.push_back(std::move(d));
    }

    // class 8 + class 7 land in follow-up commits.
    return out;
  }
};

REGISTER_CHECK(PointerDisciplineCheckPlugin);

}  // namespace ptr
}  // namespace verify
}  // namespace kernscope
