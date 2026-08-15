// kernscope::pass — kernscope-rle (todo 39). Design notes:
// include/kernscope/pass/rle.h — the two soundness rules:
//
//   (1) WITHIN a phase (no barrier between the loads) plain AA + MemorySSA
//       is sound for concurrent threads: GPU writes by other threads only
//       become visible at barriers.
//   (2) ACROSS a barrier elimination is sound ONLY for globally-invariant
//       locations (no MemoryDef anywhere in the function may-modifies it).
//
// Walk shape (per candidate load L2): walk the block's MemorySSA ACCESS
// LIST backward from L2 in program order (the use-def chain alone never
// reaches earlier loads — sibling uses hang off the same def). MemoryUse
// -> must-alias plain load found, else transparent; MemoryDef that does
// not may-modify -> continue; MemoryDef that is a barrier (convergent
// call / unknown callee) -> continue only if the location is globally
// invariant; MemoryPhi (head of list) -> continue only if globally
// invariant and every incoming edge walks to the same must-alias load;
// volatile/atomic ops -> STOP. All candidates are collected before any
// mutation (MemorySSA is stale after rewrites).

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/MemorySSA.h>
#include <llvm/IR/Attributes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/ModRef.h>

#include <kernscope/pass/register_pass.h>
#include <kernscope/pass/rle.h>

using namespace llvm;

namespace kernscope {
namespace pass {
namespace {

/// True when the instruction carries ordering/visibility semantics that the
/// location-value walk must not cross (volatile/atomic accesses). These
/// are MemoryDefs; the explicit check makes them block even where plain AA
/// would treat them as Ref-only (e.g. a volatile load).
bool isBlockingOp(const Instruction *I) {
  if (const auto *LI = dyn_cast<LoadInst>(I)) {
    return LI->isVolatile() || LI->isAtomic();
  }
  if (const auto *SI = dyn_cast<StoreInst>(I)) {
    return SI->isVolatile() || SI->isAtomic();
  }
  return isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
         isa<FenceInst>(I);
}

/// True when I is a GPU barrier: a call whose callee carries the
/// `convergent` attribute (llvm.nvvm.barrier0, llvm.ptx.bar.sync,
/// llvm.amdgcn.s.barrier, __syncthreads, ...), or a call with an unknown
/// callee (indirect call / inline asm — conservative barrier).
bool isBarrier(const Instruction *I) {
  const auto *CB = dyn_cast<CallBase>(I);
  if (!CB) return false;
  const Function *Callee = CB->getCalledFunction();
  if (!Callee) return true;  // unknown callee: conservative barrier
  return Callee->hasFnAttribute(Attribute::Convergent) ||
         CB->hasFnAttr(Attribute::Convergent);
}

/// Walks the MemorySSA access lists backward (program order) from the
/// candidate load L2 to find an earlier must-alias load.
///
/// Soundness: the walk crosses only MemoryDefs that do not may-modify the
/// location (AA), so the memory state at every reached access equals the
/// state at L2; any reached must-alias load therefore loads the same
/// value. Barriers (convergent calls / unknown callees) may only be
/// crossed for globally-invariant locations (rule 2 in rle.h); volatile /
/// atomic accesses block the walk outright.
class RLEWalk {
 public:
  RLEWalk(Function &F, MemorySSA &MSSA, AAResults &AA,
          const MemoryLocation &Loc)
      : F_(F), MSSA_(MSSA), AA_(AA), Loc_(Loc) {}

  /// Returns the must-alias load found earlier than L2, or nullptr.
  LoadInst *findEarlier(LoadInst *L2) {
    BasicBlock *BB = L2->getParent();
    MemoryAccess *L2MA = MSSA_.getMemoryAccess(L2);
    if (!L2MA) return nullptr;
    const MemorySSA::AccessList *List = MSSA_.getBlockAccesses(BB);
    if (!List) return nullptr;
    int Idx = 0;
    int L2Idx = -1;
    for (const MemoryAccess &MA : *List) {
      if (&MA == L2MA) {
        L2Idx = Idx;
        break;
      }
      ++Idx;
    }
    if (L2Idx < 0) return nullptr;
    if (L2Idx == 0) return nullptr;  // nothing earlier in this block
    return walkAccesses(BB, L2Idx - 1, 0);
  }

 private:
  static constexpr unsigned kMaxDepth = 64;

  /// Walk the access list of BB backward, ending at (and including)
  /// index EndIdx (EndIdx == -1: start from the block's last access).
// NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR tree recursion
  LoadInst *walkAccesses(BasicBlock *BB, int EndIdx, unsigned Depth) {
    if (!BB || Depth > kMaxDepth) return nullptr;
    const MemorySSA::AccessList *List = MSSA_.getBlockAccesses(BB);
    if (!List) return nullptr;
    SmallVector<MemoryAccess *, 32> AccessList;
    for (const MemoryAccess &MA : *List) {
      AccessList.push_back(const_cast<MemoryAccess *>(&MA));
    }
    int End = EndIdx < 0 ? static_cast<int>(AccessList.size()) - 1 : EndIdx;
    for (int i = End; i >= 0; --i) {
      MemoryAccess *MA = AccessList[i];
      if (auto *Use = dyn_cast<MemoryUse>(MA)) {
        auto *LI = dyn_cast<LoadInst>(Use->getMemoryInst());
        if (LI && !LI->isVolatile() && !LI->isAtomic() &&
            AA_.isMustAlias(Loc_, MemoryLocation::get(LI))) {
          return LI;
        }
        // Any other use is transparent: keep walking.
      } else if (auto *Def = dyn_cast<MemoryDef>(MA)) {
        Instruction *I = Def->getMemoryInst();
        if (!I) continue;  // liveOnEntry: nothing above, keep walking
        if (isBlockingOp(I)) return nullptr;
        if (isBarrier(I) && !isGloballyInvariant()) return nullptr;
        if (isModSet(AA_.getModRefInfo(I, Loc_))) return nullptr;
        // Transparent def: keep walking.
      } else if (auto *Phi = dyn_cast<MemoryPhi>(MA)) {
        if (i != 0) continue;  // defensive: phis head the list
        return walkPhi(Phi, Depth + 1);
      }
    }
    // End of the block's list without a phi: the state flows in from
    // predecessors (or entry) — conservative stop, no candidate.
    return nullptr;
  }

  /// A phi merges the states of its incoming edges. Only globally
  /// invariant locations may be tracked through a phi, and only when every
  /// edge walks to the SAME must-alias load (the chosen replacement then
  /// dominates all paths to L2).
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR tree recursion
  LoadInst *walkPhi(MemoryPhi *Phi, unsigned Depth) {
    if (!isGloballyInvariant()) return nullptr;
    if (!PhiVisited_.insert(Phi).second) return nullptr;  // cycle
    LoadInst *Common = nullptr;
    for (unsigned i = 0; i < Phi->getNumIncomingValues(); ++i) {
      LoadInst *R = walkAccesses(Phi->getIncomingBlock(i), -1, Depth + 1);
      if (!R || (Common && Common != R)) {
        Common = nullptr;
        break;
      }
      Common = R;
    }
    PhiVisited_.erase(Phi);
    return Common;
  }

  /// No MemoryDef in the whole function may-modifies the location (AA
  /// getModRefInfo over every def — scanned via the function's
  /// instructions, since LLVM 21 exposes no defs() range on MemorySSA).
  /// Read-only/constant objects and never-escaping allocas qualify;
  /// shared globals do not.
  bool isGloballyInvariant() {
    if (InvariantComputed) return InvariantCached;
    bool Invariant = true;
    for (Instruction &I : instructions(F_)) {
      auto *Def = dyn_cast_or_null<MemoryDef>(MSSA_.getMemoryAccess(&I));
      // liveOnEntry carries a null instruction and modifies nothing.
      if (Def && isModSet(AA_.getModRefInfo(&I, Loc_))) {
        Invariant = false;
        break;
      }
    }
    InvariantCached = Invariant;
    InvariantComputed = true;
    return Invariant;
  }

  Function &F_;
  MemorySSA &MSSA_;
  AAResults &AA_;
  const MemoryLocation &Loc_;
  SmallPtrSet<const MemoryPhi *, 8> PhiVisited_;
  bool InvariantComputed = false;
  bool InvariantCached = false;
};

}  // namespace

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
PreservedAnalyses RedundantLoadEliminationPass::run(Function &F,
                                                    FunctionAnalysisManager &FAM) {
  MemorySSA &MSSA = FAM.getResult<MemorySSAAnalysis>(F).getMSSA();
  AAResults &AA = FAM.getResult<AAManager>(F);

  // Pass 1: collect candidates. Never mutate MemorySSA mid-walk — it is
  // stale after the first rewrite and must not be re-queried.
  SmallVector<std::pair<LoadInst *, LoadInst *>, 8> Candidates;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *L2 = dyn_cast<LoadInst>(&I);
      if (!L2 || L2->isVolatile() || L2->isAtomic()) continue;
      RLEWalk Walker(F, MSSA, AA, MemoryLocation::get(L2));
      if (LoadInst *L1 = Walker.findEarlier(L2)) {
        if (L1 != L2) Candidates.push_back({L2, L1});
      }
    }
  }
  if (Candidates.empty()) return PreservedAnalyses::all();

  // Pass 2: apply. All replacements first, then all erasures — a load may
  // be the replacement for one candidate and the erased load of another
  // (chains of redundant loads).
  for (auto &[L2, L1] : Candidates) L2->replaceAllUsesWith(L1);
  for (auto &[L2, L1] : Candidates) {
    if (L2->use_empty()) L2->eraseFromParent();
  }
  return PreservedAnalyses::none();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseRLEName(StringRef Name, ModulePassManager &MPM,
                  ArrayRef<PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-rle") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::RedundantLoadEliminationPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h).
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseRLEName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::RedundantLoadEliminationPass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::RedundantLoadEliminationPass());
      });
});
