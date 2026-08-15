// kernscope::verify — barrier check module (plan todo 32; classes 3 + 6).
//
// OWNS check classes:
//   3  barrier correctness     (BARRIER_MISMATCH warnings)
//   6  divergent barriers      (DIVERGENT_BARRIER errors — CUDA UB)
//
// METHOD
// ------
// Class 6 (divergent __syncthreads, Error). A barrier call site is
// divergent when its block lies inside the divergence region of a
// thread-dependent conditional branch. Thread-dependence is decided with
// LLVM's UniformityInfo (the POPL'21 abstract-interpretation uniformity
// analysis; the successor of the legacy DivergenceAnalysis and the basis
// for ConvergenceRegions). UniformityInfo needs divergence *roots*: we seed
// them through a TargetTransformInfo hook (isSourceOfDivergence /
// isAlwaysUniform / hasBranchDivergence) declaring:
//   * divergence sources: the NVPTX/AMDGPU thread-ID intrinsics
//     (llvm.nvvm.read.ptx.sreg.tid.* / llvm.amdgcn.workitem.id.* — the
//     CUDA threadIdx), loads from non-constant address spaces and results
//     of unknown calls (GPUVerify-style adversarial reads: a static
//     analysis cannot prove two threads read the same value);
//   * always-uniform overrides: blockIdx/gridDim/blockDim coordinates
//     (llvm.nvvm.read.ptx.sreg.{ctaid,nctaid,ntid}.* and the AMDGPU
//     workgroup.id/workgroup.size counterparts — identical for every thread
//     of a block).
// UniformityInfo then propagates divergence through SSA uses and divergent
// joins (including cycle/loop handling) exactly as the NVPTX/AMDGPU
// backends use it. The divergent-region test for a barrier in block b:
//   exists conditional branch B with a divergent condition such that
//     B strictly dominates b AND the immediate post-dominator of B (the
//     reconvergence point) does not dominate b.
// The LLVM IR-level analysis is value-based (a barrier call itself is
// never "divergent"), so this dominance test is the control-flow part of
// the check (research brief §4.2 recipe item 4: "barrier not post-dominated
// by entry of its enclosing divergent region").
//
// Class 3 (barrier count correctness, Warning). Path-sensitive barrier-count
// dataflow (GPUVerify's predicated-execution barrier-divergence check,
// TOPLAS'15 §2.2, at basic-block granularity): for every block we compute
// the set of barrier counts with which threads can reach it, merging only
// non-backedge paths (loop backedges are temporal, not simultaneous — the
// uniform-loop regression guard). A merge point whose simultaneously-active
// paths carry different counts is a BARRIER_MISMATCH warning; a barrier
// site reached with a multi-count simultaneously-active set is likewise
// flagged. Uniform branches (blockIdx-dependent, uniform loop-carried
// conditions) never merge into a mismatch, and count-set overflow
// (unknown counts) suppresses the report — both guard the classic
// over-reporting trap. Barriers inside a loop with a divergent early exit
// are additionally flagged (early-exit divergence).
//
// CUDA UB citation (research brief §2.1, GPUVerify TOPLAS'15 §2.2 quote):
// CUDA C Programming Guide B.6 — "__syncthreads() is allowed in conditional
// code but only if the conditional evaluates identically across the entire
// thread block, otherwise the code execution is likely to hang or produce
// unintended side effects".

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/PostOrderIterator.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Analysis/CycleAnalysis.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetTransformInfoImpl.h>
#include <llvm/Analysis/UniformityAnalysis.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/DebugLoc.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/PassInstrumentation.h>
#include <llvm/IR/PassManager.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace kernscope {
namespace verify {
namespace barrier {

/// Verdict strings (this module's vocabulary for manifest fragments +
/// diagnostics; no core-header additions per the todo-32 contract).
inline constexpr const char* kVerdictDivergentBarrier = "DIVERGENT_BARRIER";
inline constexpr const char* kVerdictBarrierMismatch = "BARRIER_MISMATCH";
inline constexpr const char* kVerdictSafe = "SAFE";

/// CUDA C Programming Guide B.6 wording (thread synchronization rules for
/// __syncthreads in conditional code).
inline constexpr const char* kCudaGuideB6 =
    "CUDA C Programming Guide B.6: \"__syncthreads() is allowed in "
    "conditional code but only if the conditional evaluates identically "
    "across the entire thread block, otherwise the code execution is likely "
    "to hang or produce unintended side effects\"";

/// TargetTransformInfo hook seeding the uniformity analysis with GPU
/// thread-dependence roots (see the file comment). Without a target that
/// declares sources, UniformityInfo marks everything uniform (verified
/// against LLVM 21.1.8: UniformityInfoAnalysis::run and
/// GenericUniformityAnalysisImpl<SSAContext>::initialize consult exactly
/// these three hooks).
class ThreadDivergenceTTIImpl : public llvm::TargetTransformInfoImplBase {
 public:
  explicit ThreadDivergenceTTIImpl(const llvm::DataLayout& DL)
      : llvm::TargetTransformInfoImplBase(DL) {}

  bool hasBranchDivergence(const llvm::Function* /*F*/) const override {
    return true;
  }

  bool isSourceOfDivergence(const llvm::Value* V) const override {
    if (const auto* CI = llvm::dyn_cast<const llvm::CallInst>(V)) {
      if (const llvm::Function* CF = CI->getCalledFunction()) {
        const llvm::StringRef n = CF->getName();
        // CUDA threadIdx / HIP workitem.id: per-thread-varying coordinates.
        if (n == "llvm.nvvm.read.ptx.sreg.tid.x" ||
            n == "llvm.nvvm.read.ptx.sreg.tid.y" ||
            n == "llvm.nvvm.read.ptx.sreg.tid.z" ||
            n == "llvm.amdgcn.workitem.id.x" ||
            n == "llvm.amdgcn.workitem.id.y" ||
            n == "llvm.amdgcn.workitem.id.z") {
          return true;
        }
        // Other hardware intrinsics (ctaid/ntid/nctaid + AMDGPU
        // workgroup.*): block-uniform, handled by isAlwaysUniform below.
        if (n.starts_with("llvm.nvvm.") || n.starts_with("llvm.amdgcn.")) {
          return false;
        }
        // Unknown function: adversarial result (GPUVerify-style — a thread
        // may observe a different value than another thread).
        return true;
      }
      // Indirect call: adversarial.
      return true;
    }
    if (const auto* LI = llvm::dyn_cast<const llvm::LoadInst>(V)) {
      // Constant address space is uniform; any other load may differ per
      // thread (conservative, sound direction for divergence detection).
      return LI->getPointerAddressSpace() != 4;
    }
    return false;
  }

  bool isAlwaysUniform(const llvm::Value* V) const override {
    if (const auto* CI = llvm::dyn_cast<const llvm::CallInst>(V)) {
      if (const llvm::Function* CF = CI->getCalledFunction()) {
        const llvm::StringRef n = CF->getName();
        // blockIdx (ctaid) / gridDim (nctaid) / blockDim (ntid) and the
        // AMDGPU workgroup.id / workgroup.size: identical for every thread
        // of the block.
        if (n == "llvm.nvvm.read.ptx.sreg.ctaid.x" ||
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
            n == "llvm.amdgcn.workgroup.size.z") {
          return true;
        }
      }
    }
    return false;
  }
};

/// True for a block-wide barrier intrinsic: CUDA __syncthreads() lowers to
/// llvm.nvvm.barrier0 (older clang) or llvm.nvvm.barrier.cta.sync.* (clang
/// 21 / CUDA 12.6 emits llvm.nvvm.barrier.cta.sync.aligned.all); HIP
/// __syncthreads() lowers to llvm.amdgcn.s.barrier. The cluster barriers
/// (llvm.nvvm.barrier.cluster.*) are not block barriers and are excluded.
namespace {
bool isBarrierCall(const llvm::CallInst& CI) {
  const llvm::Function* CF = CI.getCalledFunction();
  if (CF == nullptr || !CF->isIntrinsic()) {
    return false;
  }
  const llvm::StringRef n = CF->getName();
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
}  // namespace

/// Runs UniformityInfo (LLVM 21: GenericUniformityInfo<SSAContext>, the
/// successor of DivergenceAnalysis + ConvergenceRegions) on `F` with the
/// thread-dependence roots seeded through ThreadDivergenceTTIImpl.
namespace {
llvm::UniformityInfo computeUniformity(llvm::Function& F) {
  llvm::FunctionAnalysisManager FAM;
  FAM.registerPass([] { return llvm::PassInstrumentationAnalysis(); });
  FAM.registerPass([] { return llvm::DominatorTreeAnalysis(); });
  FAM.registerPass([] { return llvm::CycleAnalysis(); });
  FAM.registerPass([&] {
    return llvm::TargetIRAnalysis([](const llvm::Function& FF) {
      return llvm::TargetTransformInfo(
          std::make_unique<const ThreadDivergenceTTIImpl>(
              FF.getParent()->getDataLayout()));
    });
  });
  FAM.registerPass([] { return llvm::UniformityInfoAnalysis(); });
  return std::move(FAM.getResult<llvm::UniformityInfoAnalysis>(F));
}
}  // namespace

namespace {
SourceLoc sourceLocOf(const llvm::Instruction& I) {
  SourceLoc loc;
  if (const llvm::DebugLoc& dl = I.getDebugLoc()) {
    loc.file = dl->getFilename().str();
    loc.line = dl->getLine();
    loc.col = dl->getColumn();
  }
  return loc;
}
}  // namespace

/// One barrier call site inside the kernel under analysis.
struct BarrierSite {
  llvm::CallInst* call = nullptr;
  llvm::BasicBlock* block = nullptr;
};

/// Path-sensitive barrier-count state (class 3 dataflow).
/// `counts`: sorted unique barrier counts with which threads can reach the
/// block; empty = overflow (unknown counts — mismatch reporting is
/// suppressed, the uniform-loop regression guard).
/// `active`: simultaneously-active skew — threads at this point can carry
/// different counts (introduced by a divergent branch or carried through
/// joins).
struct CountState {
  /// Sorted unique barrier counts with which threads can reach the block.
  llvm::SmallVector<unsigned, 8> counts;
  /// Simultaneously-active skew: threads at this point can carry different
  /// counts (introduced by a divergent branch or carried through joins).
  bool active = false;
  /// Overflow (unknown counts): mismatch reporting is suppressed, the
  /// uniform-loop regression guard.
  bool overflow = false;
  /// Never merged yet (the merge target's initial state).
  bool valid = false;

  /// Merges another state into this one (counts = capped union; active = OR).
  void mergeWith(const CountState& o) {
    if (!o.valid) {
      return;  // Unvisited predecessor (first-iteration ordering).
    }
    active = active || o.active;
    if (!valid) {
      *this = o;
      return;
    }
    if (overflow || o.overflow) {
      overflow = true;
      counts.clear();
      return;
    }
    llvm::SmallVector<unsigned, 8> merged;
    merged.reserve(counts.size() + o.counts.size());
    std::set_union(counts.begin(), counts.end(), o.counts.begin(),
                   o.counts.end(), std::back_inserter(merged));
    // Cap distinct counts: more than this means "unknown counts" -> suppress
    // mismatch reports rather than guessing (sound for the uniform-loop
    // regression guard).
    if (merged.size() > 8) {
      overflow = true;
      counts.clear();
      return;
    }
    counts = std::move(merged);
  }

  bool operator!=(const CountState& o) const {
    return active != o.active || overflow != o.overflow ||
           valid != o.valid || counts != o.counts;
  }
};

/// Class 6: is `bb` inside the divergence region of some thread-dependent
/// conditional branch?  (branch strictly dominates bb; the reconvergence
/// point — the branch's immediate post-dominator — does not dominate bb).
/// Returns the diverging branch's block, or nullptr when bb is uniform.
namespace {
const llvm::BasicBlock* inDivergentRegion(
    const llvm::BasicBlock& bb, const llvm::DominatorTree& DT,
    const llvm::PostDominatorTree& PDT,
    const std::vector<llvm::BasicBlock*>& divergentBranches) {
  for (const llvm::BasicBlock* B : divergentBranches) {
    if (B == &bb) {
      continue;  // Instructions before the terminator are pre-divergence.
    }
    if (!DT.dominates(B, &bb)) {
      continue;  // Not on a path under this branch's influence.
    }
    const llvm::BasicBlock* reconverge = nullptr;
    if (const auto* node = PDT.getNode(B)) {
      if (const auto* idom = node->getIDom()) {
        reconverge = idom->getBlock();
      }
    }
    if (reconverge == nullptr) {
      return B;  // No post-dominator: paths never fully reconverge.
    }
    if (!DT.dominates(reconverge, &bb)) {
      return B;  // Still inside the region (before the join).
    }
  }
  return nullptr;
}
}  // namespace

/// The plugin: id "barrier", classes {3, 6}.
class BarrierCheck : public CheckPlugin {
 public:
  BarrierCheck() {
    id = "barrier";
    name = "Barrier correctness + divergent barriers";
    check_class_ids = {CheckClass::kBarrier, CheckClass::kDivergentBarrier};
  }

  std::vector<Diagnostic> run(VerifierContext& ctx) const override;
};

std::vector<Diagnostic> BarrierCheck::run(VerifierContext& ctx) const {
  std::vector<Diagnostic> out;
  llvm::Function& F = *ctx.kernel.function;
  const std::string& budget_key = ctx.kernel.name;
  Budget& budget = ctx.diagnostics.budget();

  // --- collect barrier call sites ----------------------------------------
  std::vector<BarrierSite> sites;
  for (llvm::BasicBlock& BB : F) {
    for (llvm::Instruction& I : BB) {
      if (const auto* CI = llvm::dyn_cast<llvm::CallInst>(&I)) {
        if (isBarrierCall(*CI)) {
          sites.push_back({const_cast<llvm::CallInst*>(CI), &BB});
        }
      }
    }
  }
  if (sites.empty()) {
    return out;  // No barriers -> nothing to check.
  }

  if (!budget.acquire(budget_key, sites.size())) {
    return out;  // Budget latch tripped; the engine emits the degraded
                 // verdict. Findings so far (none) stay.
  }

  llvm::DominatorTree DT(F);
  llvm::PostDominatorTree PDT(F);
  llvm::LoopInfo LI(DT);
  llvm::UniformityInfo U = computeUniformity(F);

  // --- class 6: divergent barriers ---------------------------------------
  std::vector<llvm::BasicBlock*> divergentBranches;
  for (llvm::BasicBlock& BB : F) {
    if (U.hasDivergentTerminator(BB)) {
      divergentBranches.push_back(&BB);
    }
  }
  // Dedupe sites reported as class-6 errors (class 3's loop rule must not
  // double-flag them).
  llvm::SmallPtrSet<llvm::CallInst*, 8> class6Sites;
  for (const BarrierSite& s : sites) {
    const llvm::BasicBlock* divBranch =
        inDivergentRegion(*s.block, DT, PDT, divergentBranches);
    if (divBranch != nullptr) {
      Diagnostic d;
      d.severity = Severity::Error;
      d.class_id = CheckClass::kDivergentBarrier;
      d.loc = sourceLocOf(*s.call);
      d.function = ctx.kernel.name;
      d.instruction = "call to " + s.call->getCalledFunction()->getName().str() +
                      " (__syncthreads/bar.sync)";
      d.message = std::string(kVerdictDivergentBarrier) +
                  ": barrier call site is reachable only under divergent "
                  "control flow (thread-id-dependent condition); " +
                  kCudaGuideB6 + " — behavior is undefined";
      d.path.push_back("divergence source: " +
                       llvm::Twine(divBranch->getName()).str() +
                       " (terminator with thread-dependent condition)");
      class6Sites.insert(s.call);
      out.push_back(std::move(d));
    }
  }

  // --- class 3: path-sensitive barrier-count dataflow ---------------------
  // in-state per block; out-state = in + barriers(block). Merges at joins
  // ignore backedge contributions (temporal, not simultaneous). The
  // simultaneously-active flag turns on when a predecessor's terminator is
  // divergent (its outgoing edges are taken by disjoint thread subsets).
  llvm::DenseMap<const llvm::BasicBlock*, CountState> inState;
  llvm::DenseMap<const llvm::BasicBlock*, CountState> outState;
  std::vector<const llvm::BasicBlock*> rpo;
  for (const llvm::BasicBlock* BB :
       llvm::ReversePostOrderTraversal<const llvm::Function*>(&F)) {
    rpo.push_back(BB);
  }

  bool changed = true;
  int iterations = 0;
  const int kMaxIterations = 64;
  while (changed && iterations++ < kMaxIterations) {
    changed = false;
    for (const llvm::BasicBlock* BB : rpo) {
      CountState in;
      if (BB == &F.getEntryBlock()) {
        in.valid = true;
        in.counts.push_back(0);
      }
      for (const llvm::BasicBlock* P : llvm::predecessors(BB)) {
        // Backedge (BB dominates P) contributions are temporal: the same
        // threads pass through the loop head once per iteration, never
        // simultaneously — skip them (uniform-loop regression guard).
        if (DT.dominates(BB, P)) {
          continue;
        }
        in.mergeWith(outState[P]);
        if (U.hasDivergentTerminator(*P)) {
          in.active = true;  // Divergent edge: taken by a thread subset.
        }
      }
      // Transfer: barriers in this block add 1 to every count.
      CountState out = in;
      if (!out.overflow) {
        unsigned barriersInBlock = 0;
        for (const BarrierSite& s : sites) {
          if (s.block == BB) {
            ++barriersInBlock;
          }
        }
        for (unsigned& c : out.counts) {
          c += barriersInBlock;
        }
      }
      if (out != outState[BB]) {
        inState[BB] = std::move(in);
        outState[BB] = std::move(out);
        changed = true;
      }
    }
  }

  // Reporting pass. A merge point (>= 2 non-backedge predecessors) whose
  // in-state carries >= 2 counts with simultaneously-active skew is a
  // BARRIER_MISMATCH; a barrier site reached with the same skew (block not
  // a reported join) is flagged likewise. Uniform branches keep `active`
  // false, so equal-evaluation conditionals stay SAFE.
  for (const llvm::BasicBlock* BB : rpo) {
    const CountState& in = inState[BB];
    if (in.overflow || in.counts.size() < 2 || !in.active) {
      continue;
    }
    unsigned includedPreds = 0;
    for (const llvm::BasicBlock* P : llvm::predecessors(BB)) {
      if (!DT.dominates(BB, P)) {
        ++includedPreds;
      }
    }
    const bool isJoin = includedPreds >= 2;
    if (isJoin) {
      Diagnostic d;
      d.severity = Severity::Warning;
      d.class_id = CheckClass::kBarrier;
      d.loc = sourceLocOf(*BB->getTerminator());
      d.function = ctx.kernel.name;
      d.instruction = "join point with unequal barrier counts";
      d.message = std::string(kVerdictBarrierMismatch) +
                  ": paths reaching this join execute unequal numbers of "
                  "barriers (" +
                  llvm::Twine(in.counts[0]).str() + " vs " +
                  llvm::Twine(in.counts[1]).str() +
                  "); threads arrive with different barrier counts — " +
                  kCudaGuideB6;
      d.path.push_back("join point: " + llvm::Twine(BB->getName()).str());
      out.push_back(std::move(d));
      continue;  // The join's own barrier (if any) is covered by this report.
    }
    // Non-join barrier site with a multi-count active set.
    for (const BarrierSite& s : sites) {
      if (s.block == BB && !class6Sites.contains(s.call)) {
        Diagnostic d;
        d.severity = Severity::Warning;
        d.class_id = CheckClass::kBarrier;
        d.loc = sourceLocOf(*s.call);
        d.function = ctx.kernel.name;
        d.instruction =
            "call to " + s.call->getCalledFunction()->getName().str() +
            " (__syncthreads/bar.sync)";
        d.message = std::string(kVerdictBarrierMismatch) +
                    ": threads reach this barrier having executed different "
                    "numbers of barriers (" +
                    llvm::Twine(in.counts[0]).str() + " vs " +
                    llvm::Twine(in.counts[1]).str() + "); " + kCudaGuideB6;
        out.push_back(std::move(d));
        break;
      }
    }
  }

  // --- class 3: barrier inside a loop with a divergent early exit ---------
  // A loop whose exiting block has a divergent terminator can be left by a
  // subset of threads at a different iteration -> the barrier count for the
  // remaining threads skews. Flag only sites that are not already class-6
  // errors (those are reported with the UB citation instead).
  for (const BarrierSite& s : sites) {
    if (class6Sites.contains(s.call)) {
      continue;
    }
    for (const llvm::Loop* L = LI.getLoopFor(s.block); L != nullptr;
         L = L->getParentLoop()) {
      bool divergentExit = false;
      llvm::SmallVector<llvm::BasicBlock*, 4> exiting;
      L->getExitingBlocks(exiting);
      for (const llvm::BasicBlock* E : exiting) {
        if (U.hasDivergentTerminator(*E)) {
          divergentExit = true;
          break;
        }
      }
      if (divergentExit) {
        Diagnostic d;
        d.severity = Severity::Warning;
        d.class_id = CheckClass::kBarrier;
        d.loc = sourceLocOf(*s.call);
        d.function = ctx.kernel.name;
        d.instruction =
            "call to " + s.call->getCalledFunction()->getName().str() +
            " (__syncthreads/bar.sync)";
        d.message = std::string(kVerdictBarrierMismatch) +
                    ": barrier inside a loop with a divergent early exit — "
                    "threads can leave the loop at different iterations and "
                    "miss subsequent barriers; " +
                    kCudaGuideB6;
        out.push_back(std::move(d));
        break;
      }
    }
  }

  return out;
}

// REGISTER_CHECK token-pastes the class name, so it must be expanded in the
// scope where BarrierCheck is unqualified.
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_CHECK(BarrierCheck);

}  // namespace barrier
}  // namespace verify
}  // namespace kernscope
