// kernscope::pass — kernscope-as1: NVPTX generic-address-space
// canonicalization (todo 42, Wave 7).
//
// WHY: clang 21 (LLVM 21.1.8, store build) lowers ALL device memory
// accesses through the generic (flat) address space p0:
//
//   * __device__ globals: `addrspacecast (ptr addrspace(1) @G to ptr)`
//     followed by p0 GEPs/loads/stores;
//   * kernel pointer params: `ptr` (p0) directly — no cast in the IR.
//
// The committed v1 passes (todos 37-41) are scoped to direct AS1/AS3
// accesses — their fixtures (tests/pass/Inputs/{vectorize,promote,tile}/)
// are written in that form, which clang 19- emitted natively. On real
// clang-21 IR every one of those passes sees ZERO global accesses. This
// is the documented plan-vs-actuality note of todo 42: the artifact
// pipeline runs kernscope-as1 FIRST to restore the canonical direct-AS
// form (semantics-preserving on NVPTX: the flat address of a global
// object IS its AS1 address; kernel params live in global memory), then
// the five committed passes consume it exactly as their fixtures do.
//
// Rewrites, per memory access:
//   (1) load/store pointer == addrspacecast(ASx -> p0)  -> the ASx operand
//       directly (GEP chains above the cast are rebuilt on the ASx base,
//       carrying source element type + indices + inbounds/nuw flags).
//   (2) load/store pointer (or its GEP base) == p0 function ARGUMENT (a
//       kernel param): a p0->AS1 addrspacecast is inserted at function
//       entry and the access is rewritten to it (one cast per param).
//
// Analysis-only + rewrite; runs ONLY via explicit
// opt -passes='kernscope-as1' (the pipeline script) — deliberately NOT
// registered on the default-pipeline EP callbacks: it exists to feed the
// todo-42 before/after artifact pipeline, not to alter normal compiles.

#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallVector.h>

#include <kernscope/pass/coalesce.h>
#include <kernscope/pass/register_pass.h>

#include <llvm/Support/raw_ostream.h>

using namespace llvm;

namespace kernscope {
namespace pass {
namespace {

constexpr unsigned kGenericAS = 0;
constexpr unsigned kGlobalAS = 1;
constexpr unsigned kSharedAS = 3;

// The direct-AS source of a p0 pointer: unwrap a leading addrspacecast
// (p0 -> ASx), whether it is an INSTRUCTION (kernel-param casts, runtime
// sources) or a CONSTANT EXPRESSION (clang 21 emits global-address casts
// as `ptr addrspacecast (ptr addrspace(1) @G to ptr)` constant expressions
// folded into the GEP operand). Returns the ASx operand, or nullptr.
Value *unwrapFlatCast(Value *V) {
  Value *Src = nullptr;
  unsigned SrcAS = kGenericAS;
  if (auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    Src = ASC->getPointerOperand();
    SrcAS = ASC->getSrcAddressSpace();
  } else if (auto *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::AddrSpaceCast &&
        CE->getType()->isPointerTy() &&
        CE->getType()->getPointerAddressSpace() == kGenericAS) {
      Src = CE->getOperand(0);
      SrcAS = Src->getType()->getPointerAddressSpace();
    }
  }
  if (!Src || (SrcAS != kGlobalAS && SrcAS != kSharedAS)) {
    return nullptr;
  }
  return Src;
}

/// Rebuild the p0 GEP chain rooted at `Base` (the cast's ASx source) with
/// the same element types + indices, inserting before `InsertBefore`.
/// Returns the rebuilt outermost pointer (ASx-typed). The chain includes
/// the outermost GEP `Top` itself (the access pointer).
Value *rebuildGepChain(GetElementPtrInst *Top, Value *Base,
                       Instruction *InsertBefore) {
  SmallVector<GetElementPtrInst *, 4> Chain;
  Value *Ptr = Top;
  while (auto *G = dyn_cast<GetElementPtrInst>(Ptr)) {
    Chain.push_back(G);
    Ptr = G->getPointerOperand();
  }
  // `Ptr` walked here is the cast the chain is rooted on; `Base` is its
  // ASx source — the chain rebuild starts from `Base`.
  Value *NewPtr = Base;
  for (auto It = Chain.rbegin(); It != Chain.rend(); ++It) {
    GetElementPtrInst *G = *It;
    SmallVector<Value *, 4> Idx(G->indices());
    // Copy the GEP's inbounds/nuw flags wholesale (GEPNoWrapFlags, LLVM 21)
    // and insert via BasicBlock::iterator (the Instruction* overload is
    // deprecated).
    auto *NG = GetElementPtrInst::Create(
        G->getSourceElementType(), NewPtr, Idx, G->getNoWrapFlags(), "",
        InsertBefore->getIterator());
    NewPtr = NG;
  }
  return NewPtr;
}

/// Restore the direct-AS pointer for one access, or nullptr when the
/// access is already in a direct AS (nothing to do) — and `Changed` is set
/// only when a rewrite happens.
Value *canonicalizePointer(Value *Ptr, Function &F,
                           DenseMap<Argument *, Value *> &ParamCasts,
                           Instruction *InsertBefore, bool &Changed) {
  // (1) The access pointer is the cast itself.
  if (Value *Src = unwrapFlatCast(Ptr)) {
    Changed = true;
    return Src;
  }
  // (1b) The access pointer is a GEP chain rooted at a cast.
  if (auto *Top = dyn_cast<GetElementPtrInst>(Ptr)) {
    Value *Base = Top->getPointerOperand();
    while (auto *G = dyn_cast<GetElementPtrInst>(Base)) {
      Base = G->getPointerOperand();
    }
    if (Value *Src = unwrapFlatCast(Base)) {
      Changed = true;
      return rebuildGepChain(Top, Src, InsertBefore);
    }
  }
  // (2) Kernel param: p0 function argument (possibly under a GEP chain).
  Value *Arg = Ptr;
  while (auto *G = dyn_cast<GetElementPtrInst>(Arg)) {
    Arg = G->getPointerOperand();
  }
  auto *A = dyn_cast<Argument>(Arg);
  if (A && A->getType()->isPointerTy() &&
      A->getType()->getPointerAddressSpace() == kGenericAS) {
    Value *&Cast = ParamCasts[A];
    if (!Cast) {
      IRBuilder<> B(&F.getEntryBlock(), F.getEntryBlock().getFirstInsertionPt());
      Cast = B.CreateAddrSpaceCast(A,
                                   PointerType::get(F.getContext(), kGlobalAS));
      Changed = true;
    }
    if (Ptr == Arg) {
      return Cast;
    }
    // GEP chain on the param: rebuild on the cast (AS1 base).
    return rebuildGepChain(cast<GetElementPtrInst>(Ptr), Cast, InsertBefore);
  }
  return nullptr;
}

}  // namespace

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
llvm::PreservedAnalyses As1CanonPass::run(llvm::Function &F,
                                          llvm::FunctionAnalysisManager & /*unused*/) {
  DenseMap<Argument *, Value *> ParamCasts;
  bool Changed = false;

  for (BasicBlock &BB : F) {
    SmallVector<Instruction *, 16> Worklist;
    for (Instruction &I : BB) {
      if (isa<LoadInst>(&I) || isa<StoreInst>(&I)) {
        Worklist.push_back(&I);
      }
    }
    for (Instruction *I : Worklist) {
      Value *Ptr = isa<LoadInst>(I) ? cast<LoadInst>(I)->getPointerOperand()
                                    : cast<StoreInst>(I)->getPointerOperand();
      Instruction* InsertBefore = I;
      if (getenv("KERNSCOPE_AS1_DEBUG")) {
        errs() << "as1: " << (isa<LoadInst>(I) ? "load" : "store") << " ptr: "
               << *Ptr << " AS=" << Ptr->getType()->getPointerAddressSpace()
               << "\n";
      }
      Value *NewPtr = canonicalizePointer(Ptr, F, ParamCasts, InsertBefore,
                                          Changed);
      if (!NewPtr) {
        continue;
      }
      if (auto *LI = dyn_cast<LoadInst>(I)) {
        LI->setOperand(llvm::LoadInst::getPointerOperandIndex(), NewPtr);
      } else {
        auto *SI = cast<StoreInst>(I);
        SI->setOperand(llvm::StoreInst::getPointerOperandIndex(), NewPtr);
      }
    }
  }

  // Dead flat casts + p0 GEPs are erased (the chain rebuild leaves them
  // unused); the entry param casts stay (they feed the rewrites).
  if (Changed) {
    SmallVector<Instruction *, 8> Dead;
    for (BasicBlock &BB : F) {
      for (Instruction &I : BB) {
        if (isa<AddrSpaceCastInst>(&I) || isa<GetElementPtrInst>(&I)) {
          if (I.use_empty()) {
            Dead.push_back(&I);
          }
        }
      }
    }
    for (Instruction *I : Dead) {
      I->eraseFromParent();
    }
  }
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope
