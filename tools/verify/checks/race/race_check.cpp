// kernscope::verify::race — race-prone shared-memory write analysis
// (plan todo 33; check class 4).
//
// OWNS check class:
//   4  race-prone shared-memory writes  (MAY_RACE / UNKNOWN_RACE warnings)
//
// METHOD
// ------
// A shared-memory data race needs: two accesses to the same __shared__
// allocation (>= 1 write) performed by DIFFERENT logical threads, with no
// barrier synchronizing them.  The check is a MAY analysis (GPUVerify-style
// barrier-interval reasoning, TOPLAS'15 §2.1/§3): it reports a race
// candidate when it CANNOT PROVE one of the three separators:
//
//   1. PHASE separation (barrier-delimited phase partitioning).  Barrier
//      calls (llvm.nvvm.barrier0* / llvm.nvvm.barrier.cta.* /
//      llvm.amdgcn.s.barrier) split each basic block into segments; the
//      segment graph (within-block adjacency + CFG edges) is partitioned
//      into connected components = barrier phases.  Two accesses in
//      DIFFERENT components have a barrier on every path between them and
//      are therefore proven race-free.  Same component = same barrier
//      phase = MAY-be-simultaneous (different threads can drift anywhere
//      inside a phase; GPUVerify's fully-async-within-barriers model, sound
//      under lockstep and independent-thread-scheduling hardware).
//
//   2. THREAD separation (plan scope: thread-id-dependent indices).  The
//      plan contract scopes the analysis to accesses whose address depends
//      on the thread id (tid.x/y/z / laneid / workitem.id.x/y/z — detected
//      by a use-chain walk over the GEP index expression).  Pairs where
//      BOTH indices are thread-uniform are broadcasts and are skipped
//      (documented false-negative class: uniform-address writes with
//      thread-dependent VALUES, e.g. s[0] = tid, are not modeled).
//      Pairs on MUTUALLY EXCLUSIVE paths of a block-uniform branch
//      (condition derived from blockIdx/blockDim/gridDim — no thread-id
//      leaf) are proven non-simultaneous via dominance (DT/PDT region
//      test, mirroring the barrier module's divergent-region logic) and
//      are skipped.
//
//   3. ADDRESS separation (SCEV may-alias).  For every GEP level of the
//      access, VerifierContext::queryRange (launch-dim-aware range lattice
//      first, ScalarEvolution second — research brief §4.1) yields the
//      unsigned index range; the access BYTE SPAN is the linearized
//      offset range (index * DataLayout stride per level).  Disjoint spans
//      prove the addresses can never coincide -> SAFE.  Overlapping (or
//      unbounded) spans -> MAY_RACE.
//
// Verdicts (module-owned vocabulary, embedded in "verdict_json:" payloads
// like the oob module; the core has no verdict enum for check classes):
//   SAFE         — no same-phase race candidate with a thread-dependent or
//                  data-dependent index.
//   MAY_RACE     — >= 1 pair of same-phase accesses (>= 1 write, >= 1
//                  thread-dependent index) whose byte spans overlap (or
//                  cannot be bounded).  Warning with BOTH source sites +
//                  the missing-barrier path in the diagnostic.
//   UNKNOWN_RACE — >= 1 same-phase pair with a DATA-DEPENDENT index
//                  (derived from a memory load / unknown call / kernel
//                  argument): the analysis cannot determine which
//                  addresses different threads touch (GPUVerify-style
//                  adversarial reads).  Soundness naming: NOT SAFE (a race
//                  cannot be excluded), NOT MAY_RACE (addresses are
//                  unknowable, not merely overlapping).
//
// SOUNDNESS SCOPE (may-analysis — this check does NOT claim DRF freedom):
// documented in docs/verify-soundness.md.  Known false-positive classes:
// identical thread-indexed expressions (s[t] = a; s[t] = b — per-thread
// sequential stores reported MAY_RACE), same-expression write-read
// (s[t] = v; v2 = s[t]), loop-carried phases (a barrier inside a loop does
// not distinguish iterations), unknown-range thread indices.  Known
// false-negative classes: data-dependent indices (UNKNOWN_RACE, never
// SAFE, but not a race proof either way), uniform-address value races
// (s[0] = tid), divergent barriers (still treated as phase separators —
// class 6 flags them as UB), atomics/__threadfence/__syncwarp (not
// modeled).  HGRD (2026, MLIR + CP-SAT, research brief §3.1) is noted as
// the FUTURE upgrade path for precise index-alias solving — NOT
// implemented here.
//
// Registered via REGISTER_CHECK: no shared-file edits (todo 30 contract).

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace kernscope {
namespace verify {
namespace race {

/// Verdict strings (this module's vocabulary for manifest fragments +
/// diagnostics; no core-header additions per the todo-33 contract).
inline constexpr const char* kVerdictSafe = "SAFE";
inline constexpr const char* kVerdictMayRace = "MAY_RACE";
inline constexpr const char* kVerdictUnknownRace = "UNKNOWN_RACE";

namespace {

using llvm::ConstantRange;
using llvm::GlobalVariable;
using llvm::Instruction;
using llvm::Value;

// ---------------------------------------------------------------------------
// Small shared helpers (duplicated into this module per the isolation
// contract — the oob/barrier modules' helpers are READ-ONLY).
// ---------------------------------------------------------------------------

SourceLoc loc_of(const Instruction& i) {
  SourceLoc loc;
  if (const llvm::DebugLoc dl = i.getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}

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

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (const char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += c;
    }
  }
  return out;
}

std::string range_render(const ConstantRange& r) {
  llvm::SmallString<16> lo;
  llvm::SmallString<16> hi;
  r.getLower().toStringUnsigned(lo);
  r.getUpper().toStringUnsigned(hi);
  std::string out = "[";
  out += lo;
  out += ",";
  out += hi;
  out += ")";
  return out;
}

/// True for a block-wide barrier intrinsic: CUDA __syncthreads() lowers to
/// llvm.nvvm.barrier0 (older clang) or llvm.nvvm.barrier.cta.* (clang 21 /
/// CUDA 12.6 emits llvm.nvvm.barrier.cta.sync.aligned.all); HIP
/// __syncthreads() lowers to llvm.amdgcn.s.barrier.  __syncwarp and
/// __threadfence_block are NOT block barriers and are excluded (documented
/// in the soundness scope).
bool is_barrier_call(const llvm::CallInst& ci) {
  const llvm::Function* cf = ci.getCalledFunction();
  if (cf == nullptr || !cf->isIntrinsic()) {
    return false;
  }
  const llvm::StringRef n = cf->getName();
  if (n.starts_with("llvm.nvvm.barrier0")) {
    return true;
  }
  if (n.contains("llvm.nvvm.barrier.cta.")) {
    return true;
  }
  if (n == "llvm.amdgcn.s.barrier") {
    return true;
  }
  return false;
}

/// Thread-id coordinate intrinsics (per-thread-varying address leaves).
bool is_tid_intrinsic(const llvm::StringRef& n) {
  return n == "llvm.nvvm.read.ptx.sreg.tid.x" ||
         n == "llvm.nvvm.read.ptx.sreg.tid.y" ||
         n == "llvm.nvvm.read.ptx.sreg.tid.z" ||
         n == "llvm.nvvm.read.ptx.sreg.laneid" ||
         n == "llvm.amdgcn.workitem.id.x" ||
         n == "llvm.amdgcn.workitem.id.y" ||
         n == "llvm.amdgcn.workitem.id.z";
}

/// Block-uniform coordinate intrinsics (blockIdx/blockDim/gridDim +
/// AMDGPU workgroup.* — identical for every thread of the block; a
/// blockIdx-dependent index is a broadcast inside one block's shared
/// memory).
bool is_block_uniform_intrinsic(const llvm::StringRef& n) {
  return n == "llvm.nvvm.read.ptx.sreg.ctaid.x" ||
         n == "llvm.nvvm.read.ptx.sreg.ctaid.y" ||
         n == "llvm.nvvm.read.ptx.sreg.ctaid.z" ||
         n == "llvm.nvvm.read.ptx.sreg.nctaid.x" ||
         n == "llvm.nvvm.read.ptx.sreg.nctaid.y" ||
         n == "llvm.nvvm.read.ptx.sreg.nctaid.z" ||
         n == "llvm.nvvm.read.ptx.sreg.ntid.x" ||
         n == "llvm.nvvm.read.ptx.sreg.ntid.y" ||
         n == "llvm.nvvm.read.ptx.sreg.ntid.z" ||
         n == "llvm.amdgcn.workgroup.id.x" ||
         n == "llvm.amdgcn.workgroup.id.y" ||
         n == "llvm.amdgcn.workgroup.id.z" ||
         n == "llvm.amdgcn.workgroup.size.x" ||
         n == "llvm.amdgcn.workgroup.size.y" ||
         n == "llvm.amdgcn.workgroup.size.z";
}

// ---------------------------------------------------------------------------
// Index classification: how does the address expression vary per thread?
// ---------------------------------------------------------------------------

enum class IndexClass { Uniform, Thread, Data };

/// Classifies an index expression by walking its SSA use chain:
///   Thread  — reaches a thread-id intrinsic (tid/laneid/workitem.id):
///             different threads can evaluate DIFFERENT addresses.
///   Data    — reaches a memory load, an unknown call, or a kernel
///             argument: the analysis cannot determine the address at all
///             (GPUVerify-style adversarial reads); MAY-alias with
///             anything.
///   Uniform — everything else (constants, blockIdx/blockDim/gridDim
///             intrinsics, loop induction variables).
/// Precedence: Data > Thread > Uniform.
IndexClass classify_index(Value* v) {
  llvm::SmallPtrSet<Value*, 16> visited;
  std::vector<Value*> stack{v};
  bool thread = false;
  bool data = false;
  while (!stack.empty()) {
    Value* cur = stack.back();
    stack.pop_back();
    if (!visited.insert(cur).second) {
      continue;
    }
    if (const auto* ci = llvm::dyn_cast<const llvm::CallInst>(cur)) {
      if (const llvm::Function* cf = ci->getCalledFunction()) {
        const llvm::StringRef n = cf->getName();
        if (is_tid_intrinsic(n)) {
          thread = true;
          continue;
        }
        if (is_block_uniform_intrinsic(n)) {
          continue;  // block-uniform coordinate: not an address variable
        }
      }
      // Unknown call / unmodeled intrinsic: adversarial (data-dependent).
      data = true;
      continue;
    }
    if (llvm::isa<llvm::LoadInst>(cur)) {
      data = true;
      continue;
    }
    if (llvm::isa<llvm::Argument>(cur)) {
      // Kernel parameter: runtime value, uniform across the block but
      // unknowable statically (an smem index derived from it cannot be
      // bounded).
      data = true;
      continue;
    }
    if (llvm::isa<llvm::Constant>(cur)) {
      continue;
    }
    if (auto* inst = llvm::dyn_cast<llvm::Instruction>(cur)) {
      for (const llvm::Use& u : inst->operands()) {
        stack.push_back(u.get());
      }
    }
    // Other non-instruction values (none expected in index expressions).
  }
  if (data) {
    return IndexClass::Data;
  }
  if (thread) {
    return IndexClass::Thread;
  }
  return IndexClass::Uniform;
}

const char* index_class_name(IndexClass c) {
  switch (c) {
    case IndexClass::Thread:
      return "thread-dependent";
    case IndexClass::Data:
      return "data-dependent";
    case IndexClass::Uniform:
      return "uniform";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// Shared-memory access collection
// ---------------------------------------------------------------------------

/// Resolves the __shared__ allocation a pointer expression is rooted in.
/// Walks the GEP pointer-operand chain and pointer casts (incl. clang 21's
/// addrspacecast (addrspace(3) -> generic) of the smem global) to the
/// addrspace(3) global.  Returns std::nullopt for non-shared pointers.
struct SmemRoot {
  GlobalVariable* gv = nullptr;
  /// extern __shared__ (dynamic smem): zero-sized global.
  bool dynamic = false;
};

std::optional<SmemRoot> smem_root(Value* ptr) {
  Value* v = ptr;
  while (true) {
    v = v->stripPointerCasts();
    if (auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(v)) {
      v = gep->getPointerOperand();
      continue;
    }
    break;
  }
  if (auto* gv = llvm::dyn_cast<GlobalVariable>(v)) {
    if (gv->getAddressSpace() == 3) {
      return SmemRoot{gv, gv->getInitializer() == nullptr};
    }
  }
  return std::nullopt;
}

/// One GEP level of an access: the index value and its DataLayout stride in
/// bytes (the type the index strides over, via gep_type_iterator).
struct IndexLevel {
  Value* index = nullptr;
  uint64_t stride = 0;
  IndexClass cls = IndexClass::Uniform;
};

/// One analyzed shared-memory access.
struct SmemAccess {
  Instruction* inst = nullptr;
  bool is_store = false;
  std::string op;  // "store" / "load"
  SourceLoc loc;
  std::string base_name;
  std::vector<IndexLevel> levels;
  IndexClass idx_class = IndexClass::Uniform;
  /// Byte span [lo, hi) of the accessed address range (linearized GEP
  /// offsets); std::nullopt when any level is unbounded or arithmetic
  /// overflows.  Direct (non-GEP) scalar accesses use [0, width).
  std::optional<ConstantRange> byte_span;
  /// Barrier-phase component id (set after the segment partition).
  int component = -1;
};

/// Extracts the GEP index levels of `ptr` (innermost GEP chain), with
/// DataLayout strides, into `levels`.  Returns true when the chain was
/// walked (ptr is GEP-rooted); false for direct scalar-global accesses.
///
/// Strides follow LLVM GEP semantics (gep_type_iterator): index k's byte
/// contribution is idx_k * allocSize(type indexed at k) — the leading
/// clang-21 `i64 0` array/struct index contributes 0 with stride
/// allocSize(whole array/struct), and each element index strides its own
/// element type.
bool collect_levels(Value* ptr, const llvm::DataLayout& dl,
                    std::vector<IndexLevel>& levels) {
  std::vector<llvm::GetElementPtrInst*> geps;
  Value* v = ptr;
  while (true) {
    v = v->stripPointerCasts();
    auto* gep = llvm::dyn_cast<llvm::GetElementPtrInst>(v);
    if (gep == nullptr) {
      break;
    }
    geps.push_back(gep);
    v = gep->getPointerOperand();
  }
  if (geps.empty()) {
    return false;
  }
  // Outermost first (walk the collected list in reverse).
  for (auto it = geps.rbegin(); it != geps.rend(); ++it) {
    const llvm::GetElementPtrInst& gep = **it;
    for (auto gi = llvm::gep_type_begin(gep); gi != llvm::gep_type_end(gep);
         ++gi) {
      // Index k strides the type it indexes into (getIndexedType()).
      levels.push_back(IndexLevel{gi.getOperand(),
                                  dl.getTypeAllocSize(gi.getIndexedType()),
                                  classify_index(gi.getOperand())});
    }
  }
  return true;
}

/// Linearizes the per-level index ranges into one byte span.  Uses 128-bit
/// intermediates so byte-offset overflow degrades to "unknown" instead of
/// wrapping into a wrong span.  CONSTANT indices contribute a fixed offset
/// (e.g. the leading `i64 0` of an array GEP); only VARIABLE indices
/// contribute range widths (a range [lo,hi) covers contiguous elements
/// [lo*stride, hi*stride)).  Returns std::nullopt when any variable level
/// is unbounded / wrapped / overflowing (may-overlap: the conservative
/// direction for a MAY analysis).
std::optional<ConstantRange> byte_span(const SmemAccess& a, uint64_t width,
                                       VerifierContext& ctx) {
  if (a.levels.empty()) {
    // Direct access to a scalar __shared__ global: one element.
    return ConstantRange::getNonEmpty(llvm::APInt(64, 0),
                                      llvm::APInt(64, width));
  }
  unsigned __int128 lo = 0;
  unsigned __int128 hi = 0;
  for (const IndexLevel& lv : a.levels) {
    const uint64_t stride = lv.stride;
    if (const auto* c = llvm::dyn_cast<llvm::ConstantInt>(lv.index)) {
      // Fixed offset: the leading array/struct index (always 0 in clang's
      // smem GEPs) and constant field indices contribute no range width.
      lo += static_cast<unsigned __int128>(c->getValue().getZExtValue()) *
            stride;
      hi += static_cast<unsigned __int128>(c->getValue().getZExtValue()) *
            stride;
      if (lo > UINT64_MAX || hi > UINT64_MAX) {
        return std::nullopt;
      }
      continue;
    }
    std::optional<ConstantRange> r = ctx.queryRange(lv.index);
    if (!r) {
      return std::nullopt;
    }
    // Wrapped/full ranges carry no usable linear span.
    if (r->isWrappedSet() || r->isFullSet()) {
      return std::nullopt;
    }
    const llvm::APInt max = r->getUnsignedMax();
    if (max == llvm::APInt::getMaxValue(max.getBitWidth())) {
      return std::nullopt;  // hi = max + 1 would overflow
    }
    lo += static_cast<unsigned __int128>(r->getUnsignedMin().getZExtValue()) *
          stride;
    hi += static_cast<unsigned __int128>((max.getZExtValue() + 1)) * stride;
    if (lo > UINT64_MAX || hi > UINT64_MAX) {
      return std::nullopt;
    }
  }
  if (hi <= lo) {
    return std::nullopt;
  }
  return ConstantRange::getNonEmpty(llvm::APInt(64, static_cast<uint64_t>(lo)),
                                    llvm::APInt(64, static_cast<uint64_t>(hi)));
}

/// Byte spans overlap iff neither is entirely to the left of the other
/// (inclusive/exclusive intervals: [a0,a1) vs [b0,b1), overlap iff
/// a0 < b1 && b0 < a1).
bool spans_overlap(const ConstantRange& a, const ConstantRange& b) {
  return a.getLower().ult(b.getUpper()) &&
         b.getLower().ult(a.getUpper());
}

// ---------------------------------------------------------------------------
// Barrier-phase partition (segments + union-find)
// ---------------------------------------------------------------------------

/// One instruction-range segment of a block (a maximal barrier-free
/// subsequence; barrier calls are segment boundaries).
struct Segment {
  llvm::BasicBlock* block = nullptr;
  int id = 0;
};

/// Union-find over segment ids (path compression).
class SegmentUF {
 public:
  explicit SegmentUF(int n) : parent_(n) {
    for (int i = 0; i < n; ++i) {
      parent_[i] = i;
    }
  }
  int find(int x) {
    while (parent_[x] != x) {
      parent_[x] = parent_[parent_[x]];
      x = parent_[x];
    }
    return x;
  }
  void unite(int a, int b) { parent_[find(a)] = find(b); }

 private:
  std::vector<int> parent_;
};

/// A conditional branch whose condition is BLOCK-UNIFORM (no thread-id
/// leaf): every thread of a block takes the same side, so accesses on the
/// two sides are mutually exclusive (never simultaneously active).
struct UniformBranch {
  llvm::BasicBlock* branch = nullptr;
  llvm::BasicBlock* succ0 = nullptr;
  llvm::BasicBlock* succ1 = nullptr;
  llvm::BasicBlock* join = nullptr;  // immediate post-dominator (may be null)
};

/// Is `bb` inside the region headed by `succ` (dominated by succ, not yet
/// past the join)?
bool in_region(llvm::BasicBlock* bb, llvm::BasicBlock* succ,
               llvm::BasicBlock* join, const llvm::DominatorTree& dt) {
  if (!dt.dominates(succ, bb)) {
    return false;
  }
  if (join != nullptr && dt.dominates(join, bb)) {
    return false;  // past the reconvergence point
  }
  return true;
}

/// True when the two access blocks lie on mutually exclusive sides of some
/// uniform conditional branch (different sides -> cannot be simultaneously
/// active -> proven race-free).
bool mutually_exclusive(llvm::BasicBlock* a, llvm::BasicBlock* b,
                        const std::vector<UniformBranch>& branches,
                        const llvm::DominatorTree& dt) {
  for (const UniformBranch& ub : branches) {
    const bool a0 = in_region(a, ub.succ0, ub.join, dt);
    const bool a1 = in_region(a, ub.succ1, ub.join, dt);
    if (!a0 && !a1) {
      continue;
    }
    const bool b0 = in_region(b, ub.succ0, ub.join, dt);
    const bool b1 = in_region(b, ub.succ1, ub.join, dt);
    if ((a0 && b1) || (a1 && b0)) {
      return true;
    }
  }
  return false;
}

/// One race-candidate pair finding (MAY_RACE or UNKNOWN_RACE).
struct PairFinding {
  const SmemAccess* a = nullptr;
  const SmemAccess* b = nullptr;
  bool may = false;  // true = MAY_RACE, false = UNKNOWN_RACE
  std::string why;
};

}  // namespace

/// The plugin: id "race", class {4}.
class RaceCheck : public CheckPlugin {
 public:
  RaceCheck() {
    id = "race";
    name = "Race-prone shared-memory writes";
    check_class_ids = {CheckClass::kRace};
  }

  std::vector<Diagnostic> run(VerifierContext& ctx) const override;
};

std::vector<Diagnostic> RaceCheck::run(VerifierContext& ctx) const {
  std::vector<Diagnostic> out;
  llvm::Function& F = *ctx.kernel.function;
  const std::string& fn = ctx.kernel.name;
  Budget& budget = ctx.diagnostics.budget();
  const llvm::DataLayout& dl = ctx.module.getDataLayout();

  // --- pass 1: segment split + access collection (budgeted per access) ----
  std::vector<Segment> segments;
  std::vector<SmemAccess> accesses;
  llvm::DenseMap<const llvm::BasicBlock*, int> first_seg;
  llvm::DenseMap<const llvm::BasicBlock*, int> last_seg;
  for (llvm::BasicBlock& bb : F) {
    int seg = static_cast<int>(segments.size());
    segments.push_back({&bb, seg});
    first_seg[&bb] = seg;
    for (Instruction& i : bb) {
      if (const auto* ci = llvm::dyn_cast<llvm::CallInst>(&i)) {
        if (is_barrier_call(*ci)) {
          seg = static_cast<int>(segments.size());
          segments.push_back({&bb, seg});
          continue;
        }
      }
      if (!budget.acquire(fn, 1)) {
        return out;  // budget latched; engine emits the degraded verdict
      }
      Value* addr = nullptr;
      bool is_store = false;
      uint64_t width = 0;
      if (auto* load = llvm::dyn_cast<llvm::LoadInst>(&i)) {
        addr = load->getPointerOperand();
        width = dl.getTypeAllocSize(load->getType());
      } else if (auto* store = llvm::dyn_cast<llvm::StoreInst>(&i)) {
        addr = store->getPointerOperand();
        width = dl.getTypeAllocSize(store->getValueOperand()->getType());
        is_store = true;
      }
      if (addr == nullptr) {
        continue;
      }
      std::optional<SmemRoot> root = smem_root(addr);
      if (!root) {
        continue;  // not shared memory (out of scope)
      }
      SmemAccess a;
      a.inst = &i;
      a.is_store = is_store;
      a.op = is_store ? "store" : "load";
      a.loc = loc_of(i);
      a.base_name = root->gv->getName().str();
      collect_levels(addr, dl, a.levels);
      // Class = worst of the levels (Data > Thread > Uniform).
      IndexClass cls = IndexClass::Uniform;
      for (const IndexLevel& lv : a.levels) {
        if (lv.cls == IndexClass::Data) {
          cls = IndexClass::Data;
        } else if (lv.cls == IndexClass::Thread && cls != IndexClass::Data) {
          cls = IndexClass::Thread;
        }
      }
      a.idx_class = cls;
      a.byte_span = byte_span(a, width, ctx);
      a.component = seg;
      accesses.push_back(std::move(a));
    }
    last_seg[&bb] = seg;
  }
  if (accesses.empty()) {
    return out;  // no shared-memory accesses to race-check for this kernel
  }

  // --- pass 2: barrier-phase components (segment union-find) --------------
  SegmentUF uf(static_cast<int>(segments.size()));
  for (const llvm::BasicBlock& bb : F) {
    // CFG edges only: a block's last segment flows into each successor's
    // first segment.  Consecutive segments WITHIN a block are separated by
    // a barrier call (that is why they are split) and must NOT be united —
    // an access before the barrier and one after it are in different
    // phases.
    for (const llvm::BasicBlock* succ : llvm::successors(&bb)) {
      uf.unite(last_seg[&bb], first_seg[succ]);
    }
  }
  for (SmemAccess& a : accesses) {
    a.component = uf.find(a.component);
  }

  // --- pass 3: uniform-branch exclusivity (DT/PDT) ------------------------
  llvm::DominatorTree dt(F);
  llvm::PostDominatorTree pdt(F);
  std::vector<UniformBranch> uniform_branches;
  for (llvm::BasicBlock& bb : F) {
    const auto* br = llvm::dyn_cast<llvm::BranchInst>(bb.getTerminator());
    if (br == nullptr || !br->isConditional()) {
      continue;
    }
    if (classify_index(br->getCondition()) != IndexClass::Uniform) {
      continue;  // thread-dependent or data-dependent condition: no
                 // exclusivity (different threads can take either side)
    }
    UniformBranch ub;
    ub.branch = &bb;
    ub.succ0 = br->getSuccessor(0);
    ub.succ1 = br->getSuccessor(1);
    if (const auto* node = pdt.getNode(&bb)) {
      if (const auto* idom = node->getIDom()) {
        ub.join = idom->getBlock();
      }
    }
    uniform_branches.push_back(ub);
  }

  // --- pass 4: pairwise race-candidate analysis (budgeted per pair) -------
  std::vector<PairFinding> findings;
  bool budget_ok = true;
  for (size_t i = 0; i < accesses.size() && budget_ok; ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      if (!budget.acquire(fn, 1)) {
        budget_ok = false;  // budget latched; engine emits the degraded
                            // verdict. Findings so far stay.
        break;
      }
      const SmemAccess& a = accesses[i];
      const SmemAccess& b = accesses[j];
      // Same allocation?
      if (a.base_name != b.base_name) {
        continue;
      }
      // >= 1 write?
      if (!a.is_store && !b.is_store) {
        continue;
      }
      // Plan scope: at least one thread-dependent (or data-dependent)
      // index.  Both-uniform pairs are broadcasts, not race candidates.
      if (a.idx_class == IndexClass::Uniform &&
          b.idx_class == IndexClass::Uniform) {
        continue;
      }
      // Phase separation: a barrier on every path between the accesses
      // proves them race-free.
      if (a.component != b.component) {
        continue;
      }
      // Uniform-branch exclusivity: mutually exclusive paths.
      if (mutually_exclusive(a.inst->getParent(), b.inst->getParent(),
                             uniform_branches, dt)) {
        continue;
      }
      PairFinding f;
      f.a = &a;
      f.b = &b;
      // Address separation: data-dependent indices are unknowable
      // (UNKNOWN_RACE); analyzable indices are tested for span overlap.
      if (a.idx_class == IndexClass::Data || b.idx_class == IndexClass::Data) {
        f.may = false;
        f.why = std::string("index expression is data-dependent (") +
                (a.idx_class == IndexClass::Data
                     ? index_class_name(a.idx_class)
                     : index_class_name(b.idx_class)) +
                "): the analysis cannot determine which addresses "
                "different threads touch";
      } else if (a.byte_span && b.byte_span &&
                 !spans_overlap(*a.byte_span, *b.byte_span)) {
        continue;  // ScalarEvolution-proven disjoint spans: addresses can
                   // never coincide -> SAFE.
      } else {
        f.may = true;
        f.why = "byte spans overlap (or are unbounded): different threads "
                "may access the same address in the same barrier phase";
      }
      findings.push_back(std::move(f));
    }
  }

  // --- pass 5: emit the per-kernel verdict + per-pair findings ------------
  const std::string tier =
      ctx.launchConfig().bounded() ? "UpperBound" : "Exact";
  size_t may_count = 0;
  size_t unknown_count = 0;
  for (const PairFinding& f : findings) {
    if (f.may) {
      ++may_count;
    } else {
      ++unknown_count;
    }
  }
  const std::string verdict = may_count > 0
                                  ? kVerdictMayRace
                                  : (unknown_count > 0 ? kVerdictUnknownRace
                                                       : kVerdictSafe);

  // Per-kernel verdict diagnostic (FIRST: the corpus test matches the first
  // class-4 "verdict_json:" payload for the kernel).
  {
    std::string pairs = "[";
    for (size_t k = 0; k < findings.size(); ++k) {
      const PairFinding& f = findings[k];
      const SmemAccess& a = *f.a;
      const SmemAccess& b = *f.b;
      if (k) {
        pairs += ",";
      }
      pairs += "{\"verdict\":\"" + std::string(f.may ? kVerdictMayRace
                                                     : kVerdictUnknownRace) +
               "\",\"op1\":\"" + a.op + "\",\"l1\":" +
               std::to_string(a.loc.line) + ",\"op2\":\"" + b.op +
               "\",\"l2\":" + std::to_string(b.loc.line) + ",\"base\":\"" +
               json_escape(a.base_name) + "\",\"span1\":\"" +
               (a.byte_span ? range_render(*a.byte_span) : "unbounded") +
               "\",\"span2\":\"" +
               (b.byte_span ? range_render(*b.byte_span) : "unbounded") +
               "\",\"idx1\":\"" + index_class_name(a.idx_class) +
               "\",\"idx2\":\"" + index_class_name(b.idx_class) +
               "\",\"phase\":" + std::to_string(a.component) +
               ",\"why\":\"" + json_escape(f.why) + "\"}";
    }
    pairs += "]";
    Diagnostic d;
    d.severity = verdict == kVerdictSafe ? Severity::Info : Severity::Warning;
    d.class_id = CheckClass::kRace;
    d.class_name = check_class_name(d.class_id);
    d.function = fn;
    d.loc = accesses.front().loc;
    d.instruction = "shared-memory race analysis";
    d.message = "shared-memory race check: " +
                std::to_string(accesses.size()) + " access(es), " +
                std::to_string(findings.size()) +
                " race candidate(s) (" + std::to_string(may_count) +
                " MAY_RACE, " + std::to_string(unknown_count) +
                " UNKNOWN_RACE); verdict_json: {\"check\":" +
                std::to_string(CheckClass::kRace) + ",\"kernel\":\"" + fn +
                "\",\"verdict\":\"" + verdict + "\",\"tier\":\"" + tier +
                "\",\"accesses\":" + std::to_string(accesses.size()) +
                ",\"pairs\":" + pairs + "}";
    out.push_back(std::move(d));
  }

  // Per-pair findings: warning with BOTH source sites + the missing-barrier
  // path (the plan's "warning with both sites + the missing-barrier path").
  for (const PairFinding& f : findings) {
    const SmemAccess& a = *f.a;
    const SmemAccess& b = *f.b;
    Diagnostic d;
    d.severity = Severity::Warning;
    d.class_id = CheckClass::kRace;
    d.class_name = check_class_name(d.class_id);
    d.loc = a.loc;
    d.function = fn;
    d.instruction = short_instruction(*a.inst);
    d.message = std::string(f.may ? kVerdictMayRace : kVerdictUnknownRace) +
                ": " + a.op + " at line " + std::to_string(a.loc.line) +
                " and " + b.op + " at line " + std::to_string(b.loc.line) +
                " of shared memory '" + a.base_name +
                "' may be performed by different threads in the same "
                "barrier phase with no intervening __syncthreads()/bar.sync"
                " (phase " + std::to_string(a.component) + "): " + f.why +
                "; verdict_json: {\"check\":" +
                std::to_string(CheckClass::kRace) + ",\"kernel\":\"" + fn +
                "\",\"verdict\":\"" + (f.may ? kVerdictMayRace
                                             : kVerdictUnknownRace) +
                "\",\"pair\":{\"op1\":\"" + a.op + "\",\"l1\":" +
                std::to_string(a.loc.line) + ",\"op2\":\"" + b.op +
                "\",\"l2\":" + std::to_string(b.loc.line) + ",\"base\":\"" +
                json_escape(a.base_name) + "\"}}";
    d.path.push_back("site 1: " + a.loc.render() + " " + a.op + " " +
                     short_instruction(*a.inst));
    d.path.push_back("site 2: " + b.loc.render() + " " + b.op + " " +
                     short_instruction(*b.inst));
    d.path.push_back(
        "missing barrier: both accesses lie in the same barrier phase "
        "(phase " +
        std::to_string(a.component) +
        ") — no __syncthreads()/bar.sync separates them; a barrier between "
        "the sites is required for correctness");
    out.push_back(std::move(d));
  }

  return out;
}

// REGISTER_CHECK token-pastes the class name, so it must be expanded in the
// scope where RaceCheck is unqualified.
REGISTER_CHECK(RaceCheck);

}  // namespace race
}  // namespace verify
}  // namespace kernscope
