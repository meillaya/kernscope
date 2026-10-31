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

#include <cstdint>
#include <kernscope/verify/check_plugin.h>

#include <llvm/ADT/PostOrderIterator.h>
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
using llvm::Instruction;
using llvm::Value;

// DebugLoc-backed source location (empty when the IR carries no debug info).
SourceLoc loc_of(const Instruction* i) {
  SourceLoc loc;
  if (const llvm::DebugLoc& dl = i->getDebugLoc()) {
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
  enum class Kind : std::uint8_t {
    InBounds,     ///< pointer arithmetic proven inside the allocation.
    Escaped,      ///< pointer VALUE provably outside its allocation.
    Unbounded,    ///< index range unbounded -> may-analysis PTR_OOB.
    UnrelatedSub, ///< subtraction of pointers from different allocations.
    ParamBounded, ///< runtime kernel-param allocation: index bounded, no
                  ///< static escape claim (not counted as bad).
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
    if (alloc.root == AllocationInfo::Root::KernelParam) {
      PtrFinding f;
      f.kind = PtrFinding::Kind::ParamBounded;
      if (const auto* inst = llvm::dyn_cast<Instruction>(&gep)) {
        f.loc = loc_of(inst);
        f.instruction = short_instruction(*inst);
      } else {
        f.instruction = "constant-expr GEP";
      }
      f.base = alloc.name;
      f.detail = "runtime kernel parameter allocation: index expression "
                 "bounded, static pointer-escape proof not possible "
                 "(may-analysis scope)";
      return f;
    }
    return std::nullopt;  // unknown root: no static claim.
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
    case PtrFinding::Kind::ParamBounded:
      return "PARAM_BOUNDED";
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

// ---------------------------------------------------------------------------
// Class 8: invalid address-space casts (INVALID_ADDRSPACE_CAST, Error).
//
// LEGALITY per the VERIFIED tables (docs/report/kernscope-technical-report.pdf
// §2.1/§2.2, mirrored in ptr_common.h): CUDA-level pointer casts may only
// move between generic(0) and one of the specific spaces — NVPTX
// {1 global, 3 shared, 4 constant, 5 local} (plus 101 entry-param in
// kernel-argument position); AMDGPU {1 global, 3 local/LDS, 4 constant,
// 5 private} — via the flat generic space. A DIRECT addrspacecast between
// two non-generic spaces (e.g. shared(3) -> global(1)) has no hardware
// instruction and is poison per LangRef §addrspacecast; it cannot be
// expressed in well-formed CUDA, so its presence marks IR from legacy
// toolchains / hand-written sources.
//
// Expressing it in corpus CUDA: `__attribute__((address_space(1)))` pointer
// casts compile to direct `addrspacecast ptr addrspace(3) to ptr
// addrspace(1)` ConstantExprs (verified against clang-21 -O1). The
// generic-mediated chain (shared -> generic -> global, the QA-negative
// pattern) emits two LEGAL casts and must stay unflagged.
// ---------------------------------------------------------------------------

// One class-8 finding.
struct CastFinding {
  unsigned src_as = 0;
  unsigned dst_as = 0;
  SourceLoc loc;
  std::string instruction;
  std::string detail;
};

// NVPTX/AMDGPU space names for the diagnostic (from the §2 tables).
const char* as_name_nvptx(unsigned as) {
  switch (as) {
    case 0: return "generic";
    case 1: return "global";
    case 2: return "internal";
    case 3: return "shared";
    case 4: return "constant";
    case 5: return "local";
    case 6: return "tensor";
    case 7: return "shared-cluster";
    case 101: return "entry-param";
    default: return "?";
  }
}
const char* as_name_amdgpu(unsigned as) {
  switch (as) {
    case 0: return "generic(flat)";
    case 1: return "global";
    case 2: return "region";
    case 3: return "local(LDS)";
    case 4: return "constant";
    case 5: return "private";
    case 6: return "const32";
    case 7: return "buffer-fat";
    case 8: return "buffer-resource";
    default: return "?";
  }
}

// Class-8 analysis of one addrspacecast: only DIRECT non-generic <-> non-
// generic pairs are flagged (both sides within the target's known space set).
std::optional<CastFinding> analyze_cast(unsigned src_as, unsigned dst_as,
                                        const llvm::Instruction* enclosing) {
  if (src_as == dst_as || src_as == 0 || dst_as == 0) {
    return std::nullopt;  // no-op / generic-mediated: legal.
  }
  const bool known_src = src_as == 1 || src_as == 3 || src_as == 4 ||
                         src_as == 5 || src_as == 101;
  const bool known_dst = dst_as == 1 || dst_as == 3 || dst_as == 4 ||
                         dst_as == 5 || dst_as == 101;
  if (!known_src || !known_dst) {
    return std::nullopt;  // exotic space: no table claim.
  }
  CastFinding f;
  f.src_as = src_as;
  f.dst_as = dst_as;
  if (enclosing != nullptr) {
    f.loc = loc_of(enclosing);
    f.instruction = short_instruction(*enclosing);
  } else {
    f.instruction = "constant-expr addrspacecast";
  }
  return f;
}

// Verdict for one kernel's cast findings (class 8). Emitted whenever the
// kernel contains at least one addrspacecast; SAFE when none is illegal.
void emit_cast_verdict(VerifierContext& ctx,
                       const std::vector<CastFinding>& casts, size_t total_casts,
                       std::vector<Diagnostic>& out) {
  if (total_casts == 0) {
    return;
  }
  const bool amdgpu =
      ctx.module.getTargetTriple().getArchName().starts_with("amdgcn");
  const char* table = amdgpu ? kTableAmdgpu : kTableNvptx;
  size_t bad = 0;
  std::string findings = "[";
  for (size_t k = 0; k < casts.size(); ++k) {
    if (k) {
      findings += ",";
    }
    const CastFinding& c = casts[k];
    const char* src =
        amdgpu ? as_name_amdgpu(c.src_as) : as_name_nvptx(c.src_as);
    const char* dst =
        amdgpu ? as_name_amdgpu(c.dst_as) : as_name_nvptx(c.dst_as);
    findings += "{\"op\":\"addrspacecast\",\"line\":" +
                std::to_string(c.loc.line) + ",\"src_as\":" +
                std::to_string(c.src_as) + ",\"dst_as\":" +
                std::to_string(c.dst_as) + ",\"src\":\"" + src +
                "\",\"dst\":\"" + dst + "\"}";
    if (c.src_as != 0 && c.dst_as != 0) {
      ++bad;
    }
  }
  findings += "]";
  const bool safe = bad == 0;
  Diagnostic d;
  d.severity = safe ? Severity::Info : Severity::Error;
  d.class_id = CheckClass::kAddrSpaceCast;
  d.class_name = check_class_name(d.class_id);
  d.function = ctx.kernel.name;
  if (!casts.empty()) {
    d.loc = casts.front().loc;
  }
  d.message = std::string("address-space cast check: ") +
              std::to_string(casts.size()) + " cast(s), " +
              std::to_string(bad) + " direct non-generic pair(s); " +
              (safe ? "all casts are generic-mediated (legal)"
                    : "direct casts between non-generic address spaces have "
                      "no hardware instruction (LangRef §addrspacecast: "
                      "unsupported casts yield poison) — " +
                          std::string(table));
  d.path.push_back(
      "legal CUDA casts: generic(0) <-> {global(1), shared(3), constant(4), "
      "local(5)} only (NVPTX) / generic(0) <-> {global(1), local(3), "
      "constant(4), private(5)} (AMDGPU, flat-mediated); direct "
      "shared<->global or any non-generic<->non-generic pair is invalid");
  d.message += " verdict_json: {\"check\":" +
               std::to_string(CheckClass::kAddrSpaceCast) + ",\"kernel\":\"" +
               ctx.kernel.name + "\",\"verdict\":\"" +
               (safe ? kVerdictSafe : kVerdictInvalidAddrspaceCast) +
               "\",\"casts\":" + std::to_string(total_casts) +
               ",\"bad\":" + std::to_string(bad) + ",\"findings\":" +
               findings + "}";
  out.push_back(std::move(d));
}

// ---------------------------------------------------------------------------
// Class 7: uninitialized reads (MAY_UNINIT_READ, Warning).
//
// Must/may-initialized lattice dataflow over the kernel body (GPUVerify-style
// must-init, at per-allocation granularity):
//   * initial states: kernel parameters (and their pointees) and
//     __device__/__constant__ globals are INITIALIZED on entry (host-side
//     setup before launch); allocas and __shared__ globals start
//     UNINITIALIZED;
//   * per object, two bits flow: must_written (every reaching path wrote it)
//     and may_written (some path wrote it);
//   * a load is flagged MAY_UNINIT_READ iff must_written is false at that
//     point (some path reaches the read without a write) — with an evidence
//     path; a store sets both bits;
//   * __syncthreads() is a PHASE BOUNDARY: at a barrier, every shared-memory
//     object written by ANY thread before the barrier (may_written) becomes
//     readable by ALL threads after it (must_written := may_written for
//     addrspace(3) objects only — local memory stays per-thread private).
//
// DOCUMENTED SCOPE (kernel-boundary + granularity assumptions, also in
// checks/ptr/README.md):
//   * The analysis NEVER reasons across kernel launches: a kernel's shared
//     memory is assumed uninitialized at entry regardless of any prior
//     kernel (CUDA gives no such guarantee without explicit host-side
//     memset/cudaMemset tracking, out of scope).
//   * Per-allocation granularity: writing ANY element marks the whole object
//     written (span-precision is future work); cross-thread index patterns
//     within one phase are the race module's (class 4) domain.
//   * Only the kernel body is scanned; reads through pointers whose base
//     cannot be resolved (inttoptr, unknown calls) are not claimed on.
// ---------------------------------------------------------------------------

// One class-7 finding.
struct UninitFinding {
  SourceLoc loc;
  std::string instruction;
  std::string base;
  bool must_written = false;
  std::string detail;
};

// Per-object flow state.
struct InitBits {
  bool must = false;
  bool may = false;
  bool operator!=(const InitBits& o) const {
    return must != o.must || may != o.may;
  }
};

// The kernel's tracked memory objects (allocation base -> initial bits).
using ObjMap = llvm::DenseMap<const llvm::Value*, InitBits>;

// Resolves the TRACKED memory object an access pointer is rooted in, plus
// its initial bits. std::nullopt when the object is not tracked.
std::optional<std::pair<const llvm::Value*, InitBits>> track_object(
    const llvm::Value* ptr, const llvm::Module& m) {
  const AllocationInfo alloc = resolveAllocation(ptr, m);
  InitBits init;
  switch (alloc.root) {
    case AllocationInfo::Root::Local:
    case AllocationInfo::Root::SharedArray:
    case AllocationInfo::Root::SharedDynamic:
      init = {false, false};  // alloca/smem: uninitialized at kernel entry.
      break;
    case AllocationInfo::Root::DeviceGlobal:
    case AllocationInfo::Root::KernelParam:
      init = {true, true};  // device/constant globals + params: host-provided.
      break;
    case AllocationInfo::Root::Unknown:
      return std::nullopt;
  }
  // The object key is the BASE value of the allocation.
  const llvm::Value* base = ptr;
  while (true) {
    if (const auto* bc = llvm::dyn_cast<llvm::BitCastInst>(base)) {
      base = bc->getOperand(0);
    } else if (const auto* ac = llvm::dyn_cast<llvm::AddrSpaceCastInst>(base)) {
      base = ac->getOperand(0);
    } else if (const auto* gep = llvm::dyn_cast<llvm::GEPOperator>(base)) {
      base = gep->getPointerOperand();
    } else if (const auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(base)) {
      const unsigned op = ce->getOpcode();
      if (op == llvm::Instruction::BitCast ||
          op == llvm::Instruction::AddrSpaceCast ||
          op == llvm::Instruction::GetElementPtr) {
        base = ce->getOperand(0);
      } else {
        break;
      }
    } else {
      break;
    }
  }
  return std::make_pair(base, init);
}

// Maps an object to the basic blocks where it is read before any write
// (the evidence path of a finding).
struct UninitResult {
  std::vector<UninitFinding> findings;
  size_t analyzed_accesses = 0;  ///< loads/stores resolved to tracked objects.
};

// Runs the class-7 dataflow on one kernel.
UninitResult analyze_uninit(llvm::Function& F, VerifierContext& ctx) {
  UninitResult result;
  ObjMap entry_objects;
  // Initial states: allocas + smem globals UNINITIALIZED; device/constant
  // globals + kernel params INITIALIZED (host-side setup before launch).
  for (llvm::BasicBlock& bb : F) {
    for (Instruction& i : bb) {
      if (auto* alloca = llvm::dyn_cast<llvm::AllocaInst>(&i)) {
        entry_objects[alloca] = {false, false};
      }
    }
  }
  for (llvm::GlobalVariable& g : ctx.module.globals()) {
    if (g.getAddressSpace() == 3) {
      entry_objects[&g] = {false, false};
    } else if (g.getAddressSpace() == 1 || g.getAddressSpace() == 4) {
      entry_objects[&g] = {true, true};
    }
  }
  for (llvm::Argument& a : F.args()) {
    if (a.getType()->isPointerTy()) {
      entry_objects[&a] = {true, true};
    }
  }

  // Collect access sites per block (load/store with a resolvable object).
  struct AccessSite {
    llvm::LoadInst* load = nullptr;
    llvm::StoreInst* store = nullptr;
    const llvm::Value* obj = nullptr;
  };
  llvm::DenseMap<const llvm::BasicBlock*, llvm::SmallVector<AccessSite, 8>>
      sites;
  for (llvm::BasicBlock& bb : F) {
    for (Instruction& i : bb) {
      llvm::Value* ptr = nullptr;
      llvm::StoreInst* store = nullptr;
      llvm::LoadInst* load = nullptr;
      load = llvm::dyn_cast<llvm::LoadInst>(&i);
      if (load != nullptr) {
        ptr = load->getPointerOperand();
      } else {
        store = llvm::dyn_cast<llvm::StoreInst>(&i);
        if (store != nullptr) {
          ptr = store->getPointerOperand();
        } else {
          continue;
        }
      }
      const auto obj = track_object(ptr, ctx.module);
      if (!obj) {
        continue;
      }
      ++result.analyzed_accesses;
      sites[&bb].push_back({load, store, obj->first});
    }
  }

  // RPO dataflow: in-state per block = {must,may} per object; join at
  // merges: must = AND, may = OR. Transfer is INSTRUCTION-ORDER sensitive
  // within each block (store -> written; barrier -> promote smem; the
  // reporting pass re-walks the same order against the converged in-state).
  std::vector<const llvm::BasicBlock*> rpo;
  for (const llvm::BasicBlock* bb :
       llvm::ReversePostOrderTraversal<const llvm::Function*>(&F)) {
    rpo.push_back(bb);
  }
  auto transfer = [&](const ObjMap& in, const llvm::BasicBlock* bb) {
    ObjMap out = in;
    for (const Instruction& i : *bb) {
      if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&i)) {
        const auto obj = track_object(store->getPointerOperand(), ctx.module);
        if (obj) {
          out[obj->first] = {true, true};
        }
      } else if (const auto* ci = llvm::dyn_cast<llvm::CallInst>(&i)) {
        const llvm::Function* cf = ci->getCalledFunction();
        if (cf == nullptr || !cf->isIntrinsic()) {
          continue;
        }
        const llvm::StringRef n = cf->getName();
        if (!n.starts_with("llvm.nvvm.barrier") &&
            n != "llvm.amdgcn.s.barrier") {
          continue;
        }
        for (auto& [obj, bits] : out) {
          unsigned as = 0;
          if (const auto* gv =
                  llvm::dyn_cast<const llvm::GlobalVariable>(obj)) {
            as = gv->getAddressSpace();
          } else if (const auto* alloca =
                         llvm::dyn_cast<const llvm::AllocaInst>(obj)) {
            as = alloca->getAddressSpace();
          }
          if (as == 3 && !bits.must && bits.may) {
            bits.must = true;
          }
        }
      }
    }
    return out;
  };
  llvm::DenseMap<const llvm::BasicBlock*, ObjMap> in_state;
  bool changed = true;
  int iterations = 0;
  const int kMaxIterations = 32;
  while (changed && iterations++ < kMaxIterations) {
    changed = false;
    for (const llvm::BasicBlock* bb : rpo) {
      ObjMap in = entry_objects;
      for (const llvm::BasicBlock* p : llvm::predecessors(bb)) {
        const auto it = in_state.find(p);
        if (it == in_state.end()) {
          continue;  // first-iteration ordering.
        }
        for (const auto& [obj, bits] : it->second) {
          InitBits& cur = in[obj];
          if (cur.must != bits.must || cur.may != bits.may) {
            cur.must = cur.must && bits.must;
            cur.may = cur.may || bits.may;
            changed = true;
          }
        }
      }
      ObjMap out = transfer(in, bb);
      auto it = in_state.find(bb);
      if (it == in_state.end()) {
        in_state[bb] = std::move(out);
        changed = true;
      } else if (it->second != out) {
        it->second = std::move(out);
        changed = true;
      }
    }
  }

  // Reporting pass: recompute each block's IN state (join of predecessor
  // OUT states) and re-walk in instruction order; a load whose object is
  // not must-written at that point is a MAY_UNINIT_READ with an evidence
  // path.
  for (const llvm::BasicBlock* bb : rpo) {
    ObjMap cur = entry_objects;
    for (const llvm::BasicBlock* p : llvm::predecessors(bb)) {
      const auto it = in_state.find(p);
      if (it == in_state.end()) {
        continue;
      }
      for (const auto& [obj, bits] : it->second) {
        InitBits& c = cur[obj];
        c.must = c.must && bits.must;
        c.may = c.may || bits.may;
      }
    }
    for (const Instruction& i : *bb) {
      if (const auto* store = llvm::dyn_cast<llvm::StoreInst>(&i)) {
        const auto obj = track_object(store->getPointerOperand(), ctx.module);
        if (obj) {
          cur[obj->first] = {true, true};
        }
        continue;
      }
      if (const auto* ci = llvm::dyn_cast<llvm::CallInst>(&i)) {
        const llvm::Function* cf = ci->getCalledFunction();
        if (cf != nullptr && cf->isIntrinsic()) {
          const llvm::StringRef n = cf->getName();
          if (n.starts_with("llvm.nvvm.barrier") ||
              n == "llvm.amdgcn.s.barrier") {
            for (auto& [obj, bits] : cur) {
              unsigned as = 0;
              if (const auto* gv =
                      llvm::dyn_cast<const llvm::GlobalVariable>(obj)) {
                as = gv->getAddressSpace();
              } else if (const auto* alloca =
                             llvm::dyn_cast<const llvm::AllocaInst>(obj)) {
                as = alloca->getAddressSpace();
              }
              if (as == 3 && !bits.must && bits.may) {
                bits.must = true;
              }
            }
          }
        }
        continue;
      }
      const auto* load = llvm::dyn_cast<llvm::LoadInst>(&i);
      if (load == nullptr) {
        continue;
      }
      const auto obj = track_object(load->getPointerOperand(), ctx.module);
      if (!obj) {
        continue;
      }
      const auto cur_it = cur.find(obj->first);
      if (cur_it == cur.end() || cur_it->second.must) {
        continue;
      }
      UninitFinding f;
      f.loc = loc_of(load);
      f.instruction = short_instruction(*load);
      std::string name;
      llvm::raw_string_ostream os(name);
      obj->first->printAsOperand(os, false);
      os.flush();
      f.base = name;
      f.must_written = false;
      f.detail = "read reaches an object that no path has written: "
                 "may-analysis — uninitialized read (must-write=false)";
      result.findings.push_back(std::move(f));
    }
  }
  return result;
}

const char* uninit_kind_name() { return "MAY_UNINIT_READ"; }

// JSON rendering of one class-7 finding.
std::string uninit_finding_json(const UninitFinding& f) {
  std::string out = "{\"op\":\"load\"";
  out += ",\"line\":" + std::to_string(f.loc.line);
  out += ",\"alloc\":\"" + jsonEscape(f.base) + "\"";
  out += ",\"kind\":\"" + std::string(uninit_kind_name()) + "\"";
  out += ",\"must_written\":false";
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
            return f.kind != PtrFinding::Kind::InBounds &&
                 f.kind != PtrFinding::Kind::ParamBounded;
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

    // --- class 8: invalid address-space casts ------------------------------
    std::vector<CastFinding> casts;
    size_t total_casts = 0;
    llvm::SmallPtrSet<const llvm::Value*, 16> seen_casts;
    for (llvm::BasicBlock& bb : F) {
      for (Instruction& i : bb) {
        if (!budget.acquire(fn, 1)) {
          break;
        }
        if (auto* ac = llvm::dyn_cast<llvm::AddrSpaceCastInst>(&i)) {
          if (seen_casts.insert(ac).second) {
            ++total_casts;
            if (auto f = analyze_cast(ac->getSrcAddressSpace(),
                                      ac->getDestAddressSpace(), &i)) {
              casts.push_back(std::move(*f));
            }
          }
        }
        for (const llvm::Use& u : i.operands()) {
          auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(u.get());
          if (ce == nullptr ||
              ce->getOpcode() != llvm::Instruction::AddrSpaceCast) {
            continue;
          }
          if (!seen_casts.insert(ce).second) {
            continue;
          }
          ++total_casts;
          auto* cac = llvm::dyn_cast<llvm::ConstantExpr>(ce);
          unsigned src = 0;
          unsigned dst = 0;
          if (const auto* src_ty = llvm::dyn_cast<llvm::PointerType>(
                  cac->getOperand(0)->getType())) {
            src = src_ty->getAddressSpace();
          }
          if (const auto* dst_ty =
                  llvm::dyn_cast<llvm::PointerType>(cac->getType())) {
            dst = dst_ty->getAddressSpace();
          }
          if (auto f = analyze_cast(src, dst, &i)) {
            if (!f->loc.valid()) {
              f->loc = loc_of(&i);
            }
            casts.push_back(std::move(*f));
          }
        }
      }
    }
    emit_cast_verdict(ctx, casts, total_casts, out);

    // --- class 7: uninitialized reads --------------------------------------
    const UninitResult uninit = analyze_uninit(F, ctx);
    if (uninit.analyzed_accesses > 0) {
      const bool safe = uninit.findings.empty();
      Diagnostic d;
      d.severity = safe ? Severity::Info : Severity::Warning;
      d.class_id = CheckClass::kUninitRead;
      d.class_name = check_class_name(d.class_id);
      d.function = fn;
      if (!uninit.findings.empty()) {
        d.loc = uninit.findings.front().loc;
        d.instruction = uninit.findings.front().instruction;
      }
      std::string findings_json = "[";
      for (size_t k = 0; k < uninit.findings.size(); ++k) {
        if (k) {
          findings_json += ",";
        }
        findings_json += uninit_finding_json(uninit.findings[k]);
      }
      findings_json += "]";
      d.message = "uninitialized-read check: " +
                  std::to_string(uninit.analyzed_accesses) +
                  " memory access(es), " +
                  std::to_string(uninit.findings.size()) +
                  " read(s) reaching unwritten memory; "
                  "verdict_json: {\"check\":" +
                  std::to_string(CheckClass::kUninitRead) + ",\"kernel\":\"" +
                  fn + "\",\"verdict\":\"" +
                  (safe ? kVerdictSafe : kVerdictMayUninitRead) +
                  "\",\"accesses\":" +
                  std::to_string(uninit.analyzed_accesses) + ",\"bad\":" +
                  std::to_string(uninit.findings.size()) + ",\"findings\":" +
                  findings_json + "}";
      if (!safe) {
        d.path.push_back("read site: " + uninit.findings.front().instruction);
        d.path.push_back(
            "kernel-boundary assumption: shared memory and allocas are "
            "uninitialized at kernel entry — no cross-kernel initialization "
            "is analyzed (CUDA gives no guarantee without host-side "
            "cudaMemset tracking)");
      }
      out.push_back(std::move(d));
    }

    return out;
  }
};

// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_CHECK(PointerDisciplineCheckPlugin);

}  // namespace ptr
}  // namespace verify
}  // namespace kernscope
