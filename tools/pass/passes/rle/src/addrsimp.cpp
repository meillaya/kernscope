// kernscope::pass — kernscope-addrsimp (todo 39). Design notes:
// include/kernscope/pass/addrsimp.h — three rewrites:
//
//   (1) mul X, 2^k -> shl X, k (SCEV-proven power-of-two constant;
//       k == 0 erases the mul; nsw/nuw preserved — a mul by 2^k and a shl
//       by k overflow identically for k >= 1).
//   (2) GEP folding via llvm::simplifyInstruction (InstSimplify).
//   (3) Loop-invariant address-computation hoisting to preheaders
//       (children loops first), whitelisted to non-inbounds GEPs and
//       integer arith without nsw/nuw/exact so no UB/poison can move to
//       an earlier point.

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/Analysis/InstructionSimplify.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Operator.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Transforms/Utils/Local.h>

#include <kernscope/pass/addrsimp.h>
#include <kernscope/pass/register_pass.h>
#include <algorithm>

using namespace llvm;

namespace kernscope {
namespace pass {
namespace {

/// mul X, C -> shl X, log2(C) when C is a SCEV-proven power-of-two
/// ConstantInt (v1: the constant operand's SCEV folds to a SCEVConstant
/// with a power-of-two APInt). k == 0 (mul X, 1) erases the mul: X is the
/// same value and nsw/nuw are trivially satisfied. Returns true when the
/// instruction was rewritten/erased.
bool simplifyMulToShl(BinaryOperator &M, ScalarEvolution &SE) {
  ConstantInt *C = nullptr;
  Value *X = nullptr;
  if (auto *CI = dyn_cast<ConstantInt>(M.getOperand(0))) {
    C = CI;
    X = M.getOperand(1);
  } else if (auto *CI = dyn_cast<ConstantInt>(M.getOperand(1))) {
    C = CI;
    X = M.getOperand(0);
  }
  if (!C || !X) return false;
  const auto *SConst = dyn_cast<SCEVConstant>(SE.getSCEV(C));
  if (!SConst) return false;
  const APInt &V = SConst->getAPInt();
  if (V.getBitWidth() == 0 || !V.isPowerOf2()) return false;
  unsigned K = V.logBase2();
  if (K == 0) {
    M.replaceAllUsesWith(X);
    M.eraseFromParent();
    return true;
  }
  auto *Shl = BinaryOperator::CreateShl(X, ConstantInt::get(C->getType(), K));
  // mul nuw/nsw by 2^k and shl nuw/nsw by k overflow identically for
  // k >= 1, so the flags stay truthful.
  if (M.hasNoUnsignedWrap()) Shl->setHasNoUnsignedWrap();
  if (M.hasNoSignedWrap()) Shl->setHasNoSignedWrap();
  Shl->insertAfter(&M);
  Shl->setName(M.getName());
  M.replaceAllUsesWith(Shl);
  M.eraseFromParent();
  return true;
}

/// True when the instruction may be hoisted out of L without changing
/// semantics: whitelisted pure ops (non-inbounds GEP — hoisting an
/// inbounds GEP would require an in-bounds proof; integer arith without
/// nsw/nuw/exact — a flag-carrying op could move poison earlier), no side
/// effects / memory access, and all operands defined outside the loop.
bool isHoistable(const Instruction &I, const Loop *L) {
  if (I.isTerminator()) return false;
  if (I.mayHaveSideEffects() || I.mayReadOrWriteMemory()) return false;
  switch (I.getOpcode()) {
    case Instruction::GetElementPtr:
      if (cast<GetElementPtrInst>(&I)->isInBounds()) return false;
      break;
    case Instruction::Add:
    case Instruction::Sub:
    case Instruction::Mul:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor: {
      const auto *BO = cast<BinaryOperator>(&I);
      const auto *Exact = dyn_cast<PossiblyExactOperator>(BO);
      if (BO->hasNoSignedWrap() || BO->hasNoUnsignedWrap() ||
          (Exact && Exact->isExact())) {
        return false;
      }
      break;
    }
    default:
      return false;
  }
  return std::ranges::all_of(I.operands(), [&](const Use &U) {
    const auto *OpI = dyn_cast<Instruction>(U.get());
    return OpI == nullptr || !L->contains(OpI->getParent());
  });
  return true;
}

/// Hoist invariant address-computation instructions of L to its preheader
/// (children loops first, so each instruction lands in the innermost
/// preheader whose operands permit it). Blocks belonging to nested loops
/// are handled by the nested loop's own pass.
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR/SCEV tree recursion
bool hoistInvariant(Loop *L, ScalarEvolution &SE, unsigned &Count) {
  bool Changed = false;
  for (Loop *Child : L->getSubLoops()) {
    if (hoistInvariant(Child, SE, Count)) Changed = true;
  }
  BasicBlock *Preheader = L->getLoopPreheader();
  if (!Preheader) return Changed;  // no preheader (irreducible, etc.)
  SmallVector<Instruction *, 8> ToHoist;
  for (BasicBlock *BB : L->getBlocks()) {
    bool InChild = any_of(L->getSubLoops(),
                          [&](Loop *C) { return C->contains(BB); });
    if (InChild) continue;
    for (Instruction &I : *BB) {
      if (isHoistable(I, L) && SE.isLoopInvariant(SE.getSCEV(&I), L)) {
        ToHoist.push_back(&I);
      }
    }
  }
  for (Instruction *I : ToHoist) {
    I->moveBefore(Preheader->getTerminator()->getIterator());
    ++Count;
    Changed = true;
  }
  return Changed;
}

}  // namespace

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
PreservedAnalyses AddressSimplificationPass::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  const DataLayout &DL = F.getParent()->getDataLayout();
  bool Changed = false;

  // 1. mul X, 2^k -> shl X, k. Collect first: we erase inside the loop.
  SmallVector<BinaryOperator *, 8> Muls;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (auto *M = dyn_cast<BinaryOperator>(&I)) {
        if (M->getOpcode() == Instruction::Mul) Muls.push_back(M);
      }
    }
  }
  for (BinaryOperator *M : Muls) {
    if (simplifyMulToShl(*M, SE)) Changed = true;
  }

  // 2. GEP folding via InstSimplify (zero-offset GEPs, constant-indexed
  //    chains, ...).
  SmallVector<Instruction *, 8> Geps;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      if (isa<GetElementPtrInst>(&I)) Geps.push_back(&I);
    }
  }
  for (Instruction *G : Geps) {
    SimplifyQuery SQ(DL);
    SQ.CxtI = G;
    if (Value *R = simplifyInstruction(G, SQ)) {
      if (R != G) {
        G->replaceAllUsesWith(R);
        G->eraseFromParent();
        Changed = true;
      }
    }
  }

  // 3. Loop-invariant address-chain hoisting (children first).
  unsigned Hoisted = 0;
  for (Loop *L : LI) {
    if (hoistInvariant(L, SE, Hoisted)) Changed = true;
  }

  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parseAddrSimpName(StringRef Name, ModulePassManager &MPM,
                       ArrayRef<PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-addrsimp") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::AddressSimplificationPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h).
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseAddrSimpName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::AddressSimplificationPass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::AddressSimplificationPass());
      });
});
