// kernscope::pass — vectorized load/store rewrite (todo 37, Wave 7).
//
// Rewrites chains of scalar addrspace(1) (NVPTX global) loads/stores into
// vector loads/stores of width 2 or 4 when the following are ALL provable:
//
//   contiguity: consecutive chain addresses differ by exactly the element
//     size. Proven with SCEV: SE.getMinusSCEV of two pointer SCEVs folds to
//     a compile-time constant. This is what makes loops work — the base of
//     the chain is an induction expression (e.g. a + 16*i), so the offsets
//     are not textual constants, but their differences are.
//
//   alignment:  the chain head must be aligned to the full vector byte
//     width (NVPTX ld.global.v4/st.global.v4 need naturally-aligned
//     addresses). Proven as min(known alignment of the underlying object
//     (getKnownAlignment — global align / argument align attribute),
//     offset alignment from SCEV (2^SE.getMinTrailingZeros(offset SCEV))).
//     For a strided-contiguous loop whose address advances 16*e bytes per
//     iteration the head offset {0,+,16} has trailing zeros 4 -> 16-byte
//     alignment -> <4 x float> is legal.
//
//   address space: v1 rewrites addrspace(1) (global) ONLY; addrspace(3)
//     (shared) / addrspace(4) (constant) chains are left untouched (shared
//     vectorization is flagged future work).
//
//   aliasing:    a chain is rewritten only when no intervening memory
//     instruction (store/load/call/atomic/fence between the first and last
//     merged access) may alias ANY of the merged locations — AAResults
//     (BasicAA etc. from the pass pipeline) does the proof, mirroring
//     LoadStoreVectorizer's intervening-op discipline (research brief
//     §4.2). Chains are additionally grouped per underlying object
//     (getUnderlyingObject), so accesses to distinct objects are never
//     merged even when adjacent. MUST NOT vectorize across possibly-
//     aliasing stores — this is that guard.
//
// Report hook (todo 42): opt -kernscope-vectorize-report prints
// "kernscope-vectorize: <fn>: vectorized N loads, M stores (K chains)" to
// stderr per function with rewrites; alternatively the VectorizeStats
// returned by vectorizeFunction() is the same contract the unit test
// asserts on.

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdint>

#include <kernscope/pass/register_pass.h>
#include <kernscope/pass/vectorize.h>

using namespace llvm;

namespace {

cl::opt<bool> VectorizeReportFlag(
    "kernscope-vectorize-report",
    cl::desc("Print kernscope-vectorize rewrite counts to stderr"),
    cl::init(false));

}  // namespace

namespace kernscope {
namespace pass {
namespace {

// One candidate access: the load/store plus the SCEV of its address
// relative to the group's underlying object (in ELEMENT units, since all
// accesses of a group share the element type).
struct Access {
  Instruction *Inst;
  const SCEV *Offset;
};

// All scalar global accesses with the same underlying object + element
// type in one basic block.
struct Group {
  Value *Base;  // getUnderlyingObject result shared by all accesses
  Type *EltTy;
  unsigned ElemBytes;
  SmallVector<Access> Loads;
  SmallVector<Access> Stores;
};

Group *findGroup(SmallVectorImpl<Group> &Groups, Value *Base, Type *EltTy) {
  for (Group &G : Groups) {
    if (G.Base == Base && G.EltTy == EltTy) {
      return &G;
    }
  }
  Groups.push_back(Group{Base, EltTy, 0, {}, {}});
  return &Groups.back();
}

/// Vector widths to try, largest first. Total vector width w*e must land in
/// [4, 16] bytes (NVPTX vector loads/stores are v2/v4 of 8..64-bit
/// elements; a 2-byte v2.i8 is never worth it).
unsigned pickWidth(unsigned ChainLen, unsigned ElemBytes, uint64_t HeadAlign) {
  for (unsigned W : {4u, 2u}) {
    if (W > ChainLen) {
      continue;
    }
    unsigned Total = W * ElemBytes;
    if (Total < 4 || Total > 16) {
      continue;
    }
    if (HeadAlign >= Total) {
      return W;
    }
  }
  return 0;
}

/// Alignment of the chain head: the chain may be vectorized to W*e bytes
/// only if the head address is W*e-aligned.
uint64_t computeHeadAlign(const Group &G, const Access &Head,
                          const DataLayout &DL, ScalarEvolution &SE) {
  uint64_t BaseAlign = getKnownAlignment(G.Base, DL).value();
  unsigned TZ = SE.getMinTrailingZeros(Head.Offset);
  uint64_t OffsetAlign = (TZ >= 63) ? ~0ULL : (1ULL << TZ);
  return std::min(BaseAlign, OffsetAlign);
}

bool isConsecutive(const Access &A, const Access &B, unsigned ElemBytes,
                   ScalarEvolution &SE) {
  const SCEV *Diff = SE.getMinusSCEV(B.Offset, A.Offset);
  const auto *C = dyn_cast<SCEVConstant>(Diff);
  return C && C->getAPInt().getSExtValue() == static_cast<int64_t>(ElemBytes);
}

/// Intervening-op legality (the AA guard, plan MUST-NOT: never vectorize
/// across possibly-aliasing memory ops). Every instruction between the
/// first and the last merged access must be side-effect-free with respect
/// to ALL merged locations; merged accesses themselves are skipped (their
/// relative order is what the rewrite erases).
bool chainSafe(ArrayRef<Access> Chain, bool IsStoreChain,
               const SmallPtrSetImpl<const Instruction *> &ChainSet,
               AAResults &AA) {
  SmallVector<MemoryLocation, 4> Locs;
  for (const Access &A : Chain) {
    if (const auto *LI = dyn_cast<LoadInst>(A.Inst)) {
      Locs.push_back(MemoryLocation::get(LI));
    } else {
      Locs.push_back(MemoryLocation::get(cast<StoreInst>(A.Inst)));
    }
  }

  Instruction *First = Chain.front().Inst;
  Instruction *Last = Chain.back().Inst;
  for (auto It = First->getIterator(), End = Last->getIterator(); It != End;
       ++It) {
    Instruction &I = *It;
    if (ChainSet.count(&I)) {
      continue;
    }
    if (auto *CB = dyn_cast<CallBase>(&I)) {
      for (const MemoryLocation &Loc : Locs) {
        if (AA.getModRefInfo(CB, Loc) != ModRefInfo::NoModRef) {
          return false;
        }
      }
      continue;
    }
    if (auto *SI = dyn_cast<StoreInst>(&I)) {
      if (SI->isVolatile() || SI->isAtomic()) {
        return false;
      }
      for (const MemoryLocation &Loc : Locs) {
        if (AA.getModRefInfo(SI, Loc) != ModRefInfo::NoModRef) {
          return false;
        }
      }
      continue;
    }
    if (auto *LI = dyn_cast<LoadInst>(&I)) {
      if (LI->isVolatile() || LI->isAtomic()) {
        return false;
      }
      if (IsStoreChain) {
        for (const MemoryLocation &Loc : Locs) {
          if (AA.getModRefInfo(LI, Loc) != ModRefInfo::NoModRef) {
            return false;
          }
        }
      }
      continue;
    }
    if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I) ||
        isa<FenceInst>(&I)) {
      return false;
    }
    // Everything else (arithmetic, GEPs, ...) has no memory effects.
  }
  return true;
}

/// Replace Chain[0..W) with one vector load at the first access's position.
void rewriteLoadChain(ArrayRef<Access> Chain, unsigned W, unsigned ElemBytes,
                      VectorizeStats &Stats) {
  auto *FirstLoad = cast<LoadInst>(Chain[0].Inst);
  IRBuilder<> B(FirstLoad);
  auto *VecTy = FixedVectorType::get(FirstLoad->getType(), W);
  LoadInst *VLoad = B.CreateLoad(VecTy, FirstLoad->getPointerOperand(),
                                 /*isVolatile=*/false, "kernscope.vload");
  VLoad->setAlignment(Align(W * ElemBytes));
  for (unsigned J = 0; J < W; ++J) {
    auto *LI = cast<LoadInst>(Chain[J].Inst);
    IRBuilder<> BJ(LI);
    Value *Ext = BJ.CreateExtractElement(
        VLoad, ConstantInt::get(Type::getInt32Ty(LI->getContext()), J),
        "kernscope.vext");
    LI->replaceAllUsesWith(Ext);
    LI->eraseFromParent();
  }
  Stats.loadsVectorized += W;
}

/// Replace Chain[0..W) with one vector store at the LAST access's position
/// (all store values dominate that point).
void rewriteStoreChain(ArrayRef<Access> Chain, unsigned W, unsigned ElemBytes,
                       VectorizeStats &Stats) {
  auto *LastStore = cast<StoreInst>(Chain[W - 1].Inst);
  IRBuilder<> B(LastStore);
  auto *VecTy = FixedVectorType::get(LastStore->getValueOperand()->getType(),
                                     W);
  Value *Vec = PoisonValue::get(VecTy);
  for (unsigned J = 0; J < W; ++J) {
    auto *SI = cast<StoreInst>(Chain[J].Inst);
    Vec = B.CreateInsertElement(
        Vec, SI->getValueOperand(),
        ConstantInt::get(Type::getInt32Ty(SI->getContext()), J),
        "kernscope.vins");
  }
  StoreInst *VStore =
      B.CreateStore(Vec, cast<StoreInst>(Chain[0].Inst)->getPointerOperand(),
                    /*isVolatile=*/false);
  VStore->setAlignment(Align(W * ElemBytes));
  for (unsigned J = 0; J < W; ++J) {
    Chain[J].Inst->eraseFromParent();
  }
  Stats.storesVectorized += W;
}

VectorizeStats processAccessList(SmallVectorImpl<Access> &Accs,
                                 bool IsStoreChain, Group &G,
                                 const DataLayout &DL, ScalarEvolution &SE,
                                 AAResults &AA) {
  VectorizeStats Stats;
  if (Accs.size() < 2) {
    return Stats;
  }

  // Constant-offset chains (unrolled bodies) are sorted by address so the
  // rewrite does not depend on emission order; inside a loop the offsets
  // are induction expressions, in which case program order is kept (clang
  // emits the canonical idiom in address order) and only +e diffs merge.
  bool AllConstant =
      all_of(Accs, [](const Access &A) { return isa<SCEVConstant>(A.Offset); });
  if (AllConstant) {
    stable_sort(Accs, [](const Access &A, const Access &B) {
      return cast<SCEVConstant>(A.Offset)->getAPInt().getSExtValue() <
             cast<SCEVConstant>(B.Offset)->getAPInt().getSExtValue();
    });
  }

  size_t I = 0;
  while (I + 1 < Accs.size()) {
    // Maximal contiguous run: each consecutive pair differs by e.
    size_t J = I + 1;
    while (J < Accs.size() &&
           isConsecutive(Accs[J - 1], Accs[J], G.ElemBytes, SE)) {
      ++J;
    }
    const size_t Len = J - I;
    if (Len >= 2) {
      uint64_t HeadAlign = computeHeadAlign(G, Accs[I], DL, SE);
      unsigned W = pickWidth(Len, G.ElemBytes, HeadAlign);
      if (W >= 2) {
        SmallVector<Access> Chain(Accs.begin() + I, Accs.begin() + I + W);
        SmallPtrSet<const Instruction *, 8> ChainSet;
        for (const Access &A : Chain) {
          ChainSet.insert(A.Inst);
        }
        if (chainSafe(Chain, IsStoreChain, ChainSet, AA)) {
          if (IsStoreChain) {
            rewriteStoreChain(Chain, W, G.ElemBytes, Stats);
          } else {
            rewriteLoadChain(Chain, W, G.ElemBytes, Stats);
          }
          ++Stats.chainsRewritten;
          I += W;  // leftover accesses of the same run may still merge
          continue;
        }
      }
    }
    I = J;
  }
  return Stats;
}

VectorizeStats processGroup(Group &G, const DataLayout &DL,
                            ScalarEvolution &SE, AAResults &AA) {
  VectorizeStats Stats = processAccessList(G.Loads, /*IsStoreChain=*/false, G,
                                           DL, SE, AA);
  VectorizeStats StoreStats = processAccessList(
      G.Stores, /*IsStoreChain=*/true, G, DL, SE, AA);
  Stats.storesVectorized += StoreStats.storesVectorized;
  Stats.chainsRewritten += StoreStats.chainsRewritten;
  return Stats;
}

}  // namespace

VectorizeStats vectorizeFunction(Function &F, ScalarEvolution &SE,
                                 AAResults &AA) {
  VectorizeStats Stats;
  const DataLayout &DL = F.getParent()->getDataLayout();

  for (BasicBlock &BB : F) {
    SmallVector<Group, 4> Groups;
    for (Instruction &I : BB) {
      Value *Ptr = nullptr;
      Type *EltTy = nullptr;
      bool IsStore = false;
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (!LI->isSimple()) {
          continue;
        }
        Ptr = LI->getPointerOperand();
        EltTy = LI->getType();
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (!SI->isSimple()) {
          continue;
        }
        Ptr = SI->getPointerOperand();
        EltTy = SI->getValueOperand()->getType();
        IsStore = true;
      } else {
        continue;
      }

      // v1 scope: addrspace(1) (NVPTX global) only. Shared/constant/generic
      // chains stay scalar (shared vectorization is future work).
      if (Ptr->getType()->getPointerAddressSpace() != 1) {
        continue;
      }
      // Scalar int/float elements only; no i1/bool, no pointers, no vectors.
      if (!EltTy->isIntegerTy() && !EltTy->isFloatingPointTy()) {
        continue;
      }
      const unsigned Bits = DL.getTypeSizeInBits(EltTy);
      if (Bits != 8 && Bits != 16 && Bits != 32 && Bits != 64) {
        continue;
      }

      Value *Base = getUnderlyingObject(Ptr, /*MaxLookup=*/6);
      Group *G = findGroup(Groups, Base, EltTy);
      G->ElemBytes = Bits / 8;
      const SCEV *Offset =
          SE.getMinusSCEV(SE.getSCEV(Ptr), SE.getSCEV(Base));
      (IsStore ? G->Stores : G->Loads).push_back(Access{&I, Offset});
    }

    for (Group &G : Groups) {
      VectorizeStats GroupStats = processGroup(G, DL, SE, AA);
      Stats.loadsVectorized += GroupStats.loadsVectorized;
      Stats.storesVectorized += GroupStats.storesVectorized;
      Stats.chainsRewritten += GroupStats.chainsRewritten;
    }
  }
  return Stats;
}

llvm::PreservedAnalyses VectorizeLoadStorePass::run(
    llvm::Function &F, llvm::FunctionAnalysisManager &FAM) {
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  AAResults &AA = FAM.getResult<AAManager>(F);

  VectorizeStats Stats = vectorizeFunction(F, SE, AA);

  // Report hook for the todo-42 coalescing-analysis report: one line per
  // function with rewrites, on stderr (captured with opt ... 2>&1 or
  // -print-after-all style wrappers).
  if (VectorizeReportFlag &&
      (Stats.loadsVectorized != 0 || Stats.storesVectorized != 0)) {
    errs() << "kernscope-vectorize: " << F.getName() << ": vectorized "
           << Stats.loadsVectorized << " loads, " << Stats.storesVectorized
           << " stores (" << Stats.chainsRewritten << " chains)\n";
  }

  if (Stats.chainsRewritten == 0) {
    return PreservedAnalyses::all();
  }
  // IR changed: all analyses are invalidated.
  return PreservedAnalyses::none();
}

}  // namespace pass
}  // namespace kernscope

#ifndef KERNSCOPE_VECTORIZE_AS_LIB
namespace {

bool parseVectorizePass(llvm::StringRef Name, llvm::ModulePassManager &MPM,
                        llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "kernscope-vectorize") {
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
        kernscope::pass::VectorizeLoadStorePass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h): textual name for
// opt -passes=kernscope-vectorize plus the EP callbacks that make
// clang -fpass-plugin / opt -passes=default<O2> run the pass. Skipped when
// this TU is compiled into the unit test binary (KERNSCOPE_VECTORIZE_AS_LIB:
// the registrar lives in the plugin's src/, which the test does not link;
// the test drives vectorizeFunction() directly).
REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseVectorizePass);

  PB.registerPipelineStartEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
            kernscope::pass::VectorizeLoadStorePass()));
      });
  PB.registerPeepholeEPCallback(
      [](llvm::FunctionPassManager &FPM, llvm::OptimizationLevel) {
        FPM.addPass(kernscope::pass::VectorizeLoadStorePass());
      });
});
#endif  // KERNSCOPE_VECTORIZE_AS_LIB
