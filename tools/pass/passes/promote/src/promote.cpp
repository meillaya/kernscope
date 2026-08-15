// kernscope::pass — global-to-shared memory promotion (todo 40, Wave 7).
//
// kernscope-promote rewrites tiled/loop-wide addrspace(1) (global) read
// patterns that are reused by every thread of the block into a cooperative
// block load: a fresh addrspace(3) (shared) tile is allocated, the block
// cooperatively fills it (thread tid loads element tid, strided by
// blockDim.x — each element of the tile exactly once), a barrier makes the
// loads visible, and the original reads are redirected to the tile. The
// "scattered global loads -> cooperative block load -> shared tile ->
// vectorized compute" story (docs/research/owner-examples.md); there is no
// upstream LLVM-IR-level equivalent (IREE GPUPromoteMatmulOperands / MLIR
// gpu MemoryPromotion are MLIR-only — research brief §4.7, §5.1).
//
// Soundness rules (the plan's MUST-NOTs) — full design in
// include/kernscope/pass/promote.h:
//
//   (1) Tile coverage: every rewritten load's SCEV element offset must be
//       a constant c in [0, N) or an {s,+,1}-AddRec in its own loop with a
//       constant trip count T, bounding it to [s, s+T) in [0, N); the tile
//       extent N is the max reachable element over the whole group. The
//       cooperative loop copies [0, N) — every element any read can touch.
//   (2) AA guard: no memory op in the function may-modify the tile range
//       [base, base + N*elemSize) (AAResults over every instruction) — a
//       cross-iteration write to the same global would be observed by a
//       later rewritten read and is a decline, IR untouched.
//   (3) Budget guard: N*elemSize <= the arch per-block smem budget
//       (libs/arch kernscope::get_arch/has_arch, never hardcoded).
//   (4) Uniform execution: the cooperative load + barrier are inserted at
//       function entry (all threads pass them before any divergence); a
//       candidate read in the entry block itself is NOT dominated by the
//       barrier and declines the group.
//
// v1 scope: unit-stride contiguous tiles ({s,+,1}); at most one dynamic
// GEP index per address expression (all constants non-negative); scalar
// int/float 8/16/32/64-bit elements; the entry terminator must be an
// unconditional branch; candidate loads must sit in the same loop as
// their IV (nested-AddRec 2D tiling declines).

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/MemoryLocation.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/Analysis/ValueTracking.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/ModRef.h>
#include <llvm/Support/raw_ostream.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>

#include <kernscope/arch.h>
#include <kernscope/pass/promote.h>
#include <kernscope/pass/register_pass.h>

using namespace llvm;

namespace kernscope {
namespace pass {
namespace {

cl::opt<bool> PromoteReportFlag(
    "kernscope-promote-report",
    cl::desc("Print kernscope-promote rewrite counts to stderr"),
    cl::init(false));

// ---------------------------------------------------------------------------
// Arch resolution + smem budget (libs/arch reuse — never hardcoded). The
// same pattern as the todo-38 pad module (pad_pass.cpp); duplicated here
// because tools/pass/include is READ-ONLY and pad's copy is TU-local.
// ---------------------------------------------------------------------------

std::optional<kernscope::ArchInfo> resolveArch(const Module &M,
                                               const Function *F) {
  if (F) {
    const Attribute A = F->getFnAttribute("kernscope.promote.arch");
    if (A.isStringAttribute() && kernscope::has_arch(A.getValueAsString()))
      return kernscope::get_arch(A.getValueAsString());
  }
  for (const char *Key : {"nvvmarch", "amdgpu-arch"}) {
    const Metadata *Flag = M.getModuleFlag(Key);
    if (Flag && isa<MDString>(Flag)) {
      const std::string Name = cast<MDString>(Flag)->getString().str();
      if (kernscope::has_arch(Name)) return kernscope::get_arch(Name);
    }
  }
  return std::nullopt;
}

int64_t smemBudget(const kernscope::ArchInfo &A) {
  return A.smem_per_block_max_bytes > 0 ? A.smem_per_block_max_bytes
                                        : A.smem_per_sm_bytes;
}

// ---------------------------------------------------------------------------
// Backend intrinsics (by target triple): the barrier makes the cooperative
// loads visible to every thread; threadIdx.x + blockDim.x drive the
// cooperative loop (each thread loads elements tid, tid+bdx, ...).
// ---------------------------------------------------------------------------

struct BackendIntrinsics {
  std::string Barrier;    // llvm.nvvm.barrier0 / llvm.amdgcn.s.barrier
  std::string ThreadIdx;  // read.ptx.sreg.tid.x / amdgcn.workitem.id.x
  std::string BlockDim;   // read.ptx.sreg.ntid.x / amdgcn.workgroup.size.x
};

bool backendOf(const Module &M, BackendIntrinsics &Out) {
  const StringRef T = M.getTargetTriple().getTriple();
  if (T.starts_with("nvptx")) {
    Out = {"llvm.nvvm.barrier0", "llvm.nvvm.read.ptx.sreg.tid.x",
           "llvm.nvvm.read.ptx.sreg.ntid.x"};
    return true;
  }
  if (T.starts_with("amdgcn") || T.starts_with("r600")) {
    Out = {"llvm.amdgcn.s.barrier", "llvm.amdgcn.workitem.id.x",
           "llvm.amdgcn.workgroup.size.x"};
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// Candidate classification.
// ---------------------------------------------------------------------------

// Element-offset classification of one load (offset in ELEMENTS of the
// group's element type, relative to the underlying object).
enum class OffsetKind : std::uint8_t {
  kConstant,    // constant c: reads element c only
  kUnitAddRec,  // {s,+,1} in the load's own loop with constant trip count
  kUnknown,     // not statically bounded -> declines the group
};

struct ElemOffset {
  OffsetKind Kind = OffsetKind::kUnknown;
  uint64_t Const = 0;  // kConstant value / kUnitAddRec start
  const Loop *L = nullptr;  // kUnitAddRec loop
};

// Divide a byte-offset SCEV by ElementBytes, returning the element-offset
// SCEV, or nullptr when the division is not statically exact. Handles
// constants, casts (zext/sext/trunc), mul-by-constant, and affine
// AddRecs with constant start/step (every part must divide evenly).
// NOLINTNEXTLINE(misc-no-recursion)  // deliberate SCEV tree recursion
const SCEV *divSCEV(ScalarEvolution &SE, const SCEV *S, uint64_t D) {
  if (isa<SCEVIntegralCastExpr>(S))
    return divSCEV(SE, cast<SCEVIntegralCastExpr>(S)->getOperand(), D);
  if (const auto *C = dyn_cast<SCEVConstant>(S)) {
    const APInt &V = C->getAPInt();
    if (V.isNegative() || V.urem(D) != 0) return nullptr;
    return SE.getConstant(V.udiv(D));
  }
  if (const auto *M = dyn_cast<SCEVMulExpr>(S)) {
    const SCEV *Inner = nullptr;
    uint64_t Coeff = 1;
    for (const SCEV *Op : M->operands()) {
      if (const auto *OC = dyn_cast<SCEVConstant>(Op)) {
        if (OC->getAPInt().isNegative() ||
            Coeff > std::numeric_limits<uint64_t>::max() /
                        OC->getAPInt().getZExtValue())
          return nullptr;
        Coeff *= OC->getAPInt().getZExtValue();
      } else if (Inner) {
        return nullptr;
      } else {
        Inner = Op;
      }
    }
    if (Coeff != D || !Inner) return nullptr;
    return divSCEV(SE, Inner, 1);
  }
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    const auto *StartC = dyn_cast<SCEVConstant>(AR->getStart());
    const auto *StepC = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    if (!StartC || !StepC) return nullptr;
    const APInt S0 = StartC->getAPInt();
    const APInt St = StepC->getAPInt();
    if (S0.isNegative() || St.isNegative() || S0.urem(D) != 0 ||
        St.urem(D) != 0)
      return nullptr;
    return SE.getAddRecExpr(SE.getConstant(S0.udiv(D)),
                            SE.getConstant(St.udiv(D)), AR->getLoop(),
                            AR->getNoWrapFlags());
  }
  return nullptr;
}

// Classify the element-offset SCEV of a load whose innermost loop is
// `LoadLoop` (null for loads outside any loop). kUnitAddRec is accepted
// only when the AddRec belongs to the load's own loop and the trip count
// is a compile-time constant.
ElemOffset classifyOffset(ScalarEvolution &SE, const SCEV *ElemSCEV,
                          const Loop *LoadLoop) {
  ElemOffset R;
  if (const auto *C = dyn_cast<SCEVConstant>(ElemSCEV)) {
    if (!C->getAPInt().isNegative()) {
      R.Kind = OffsetKind::kConstant;
      R.Const = C->getAPInt().getZExtValue();
    }
    return R;
  }
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(ElemSCEV)) {
    const auto *StepC = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
    const auto *StartC = dyn_cast<SCEVConstant>(AR->getStart());
    if (!StepC || !StartC || StepC->getAPInt().isNegative() ||
        StartC->getAPInt().isNegative() ||
        StepC->getAPInt().getZExtValue() != 1)
      return R;
    if (AR->getLoop() != LoadLoop) return R;  // nested-AddRec: out of v1
    if (LoadLoop && SE.getSmallConstantTripCount(LoadLoop)) {
      R.Kind = OffsetKind::kUnitAddRec;
      R.Const = StartC->getAPInt().getZExtValue();
      R.L = LoadLoop;
    }
  }
  return R;
}

// True when the GEP chain between the load pointer and Base is
// rewrite-ready: every index a non-negative constant, and at most one
// dynamic index (any integer type — the element-scale of whatever index
// carries it is verified by the SCEV gate, which is the authority).
bool gepShapeOk(Value *Ptr, Value *Base) {
  Value *Cur = Ptr;
  unsigned Depth = 0;
  unsigned Dyn = 0;
  while (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
    if (Depth++ > 8) return false;
    for (unsigned J = 0; J < GEP->getNumIndices(); ++J) {
      Value *Idx = GEP->getOperand(J + 1);
      if (auto *C = dyn_cast<ConstantInt>(Idx)) {
        if (C->getSExtValue() < 0) return false;
        continue;
      }
      if (++Dyn > 1) return false;
    }
    Cur = GEP->getPointerOperand();
    if (Cur == Base) break;
  }
  return Cur == Base;
}

// One candidate load: the instruction plus its classified element offset.
struct CandidateLoad {
  LoadInst *LI;
  ElemOffset Off;
};

// A promotable group: all candidate loads sharing one underlying object
// and element type. `Extent` = max reachable element (tile size).
// `TileAlloca` is filled during the rewrite phase (read redirection).
struct Group {
  Value *Base;
  Type *EltTy;
  uint64_t ElemBytes = 0;
  uint64_t Extent = 0;
  AllocaInst *TileAlloca = nullptr;
  SmallVector<CandidateLoad, 4> Loads;
};

Group *findGroup(SmallVectorImpl<Group> &Groups, Value *Base, Type *EltTy) {
  for (Group &G : Groups) {
    if (G.Base == Base && G.EltTy == EltTy) return &G;
  }
  Groups.push_back(Group{Base, EltTy, 0, 0, nullptr, {}});
  return &Groups.back();
}

// ---------------------------------------------------------------------------
// Rewrite (only reached after ALL analysis passed — two-phase, a late
// decline leaves IR untouched).
// ---------------------------------------------------------------------------

enum class DeclineReason : std::uint8_t {
  kNoPattern,     // no promotable group (no reuse) — silent
  kNoArch,        // no arch attribute/flag -> cannot verify the budget
  kNoBackend,     // triple has no supported GPU backend
  kBudget,        // tile would exceed the arch smem budget
  kMayAliasWrite, // AA guard: a memory op may-modify the tile range
  kEntryTerm,     // entry terminator is not an unconditional branch
  kEntryRead,     // a candidate read is in the entry block (pre-barrier)
};

const char *declineReasonText(DeclineReason R) {
  switch (R) {
    case DeclineReason::kNoArch:
      return "no arch (kernscope.promote.arch attribute or nvvmarch/"
             "amdgpu-arch module flag); smem budget unverifiable";
    case DeclineReason::kNoBackend:
      return "target triple has no supported GPU backend (NVPTX/AMDGPU)";
    case DeclineReason::kBudget:
      return "exceeds smem budget";
    case DeclineReason::kMayAliasWrite:
      return "possibly-alias write to the tile range";
    case DeclineReason::kEntryTerm:
      return "entry terminator is not an unconditional branch";
    case DeclineReason::kEntryRead:
      return "candidate read in the entry block is not barrier-dominated";
    case DeclineReason::kNoPattern:
      return "no promotable pattern";
  }
  return "unknown";
}

// Insert the cooperative block load for one group between AnchorTerm
// (whose block's successor is redirected through the new barrier block)
// and AnchorTerm's successor. Returns the barrier block's terminator —
// the anchor for chained groups. On failure nothing is mutated.
Instruction *insertCooperativeLoad(Function &F, Group &G,
                                   const BackendIntrinsics &BI,
                                   Instruction *AnchorTerm,
                                   uint64_t GroupIdx) {
  Module &M = *F.getParent();
  LLVMContext &Ctx = F.getContext();
  BasicBlock *AnchorBB = AnchorTerm->getParent();
  BasicBlock *OrigSucc = AnchorTerm->getSuccessor(0);

  auto *I64 = Type::getInt64Ty(Ctx);
  auto *I32 = Type::getInt32Ty(Ctx);
  auto *TileTy = ArrayType::get(G.EltTy, G.Extent);

  // The tile alloca lives at the top of the ENTRY block (it must dominate
  // every read); align 4 matches clang's __shared__ scalar arrays.
  BasicBlock &Entry = F.getEntryBlock();
  IRBuilder<> AllocB(&Entry, Entry.getFirstInsertionPt());
  G.TileAlloca =
      AllocB.CreateAlloca(TileTy, /*AddrSpace=*/3, nullptr, "kernscope.tile");
  G.TileAlloca->setAlignment(Align(4));

  // threadIdx.x / blockDim.x (fresh calls — pure intrinsics).
  IRBuilder<> AnchorB(AnchorBB, AnchorTerm->getIterator());
  Value *Tid32 = AnchorB.CreateCall(M.getOrInsertFunction(BI.ThreadIdx, FunctionType::get(I32, false)), {},
      "kernscope.coop.tid");
  Value *Bdx32 = AnchorB.CreateCall(M.getOrInsertFunction(BI.BlockDim, FunctionType::get(I32, false)), {},
      "kernscope.coop.ntid");
  Value *Tid = AnchorB.CreateZExt(Tid32, I64, "kernscope.coop.tid64");
  Value *Bdx = AnchorB.CreateZExt(Bdx32, I64, "kernscope.coop.ntid64");

  // New blocks: coop.cond <-> coop.body, then coop.exit (the barrier),
  // placed between the anchor block and the original successor.
  const std::string Suffix = GroupIdx == 0 ? "" : Twine(GroupIdx).str();
  BasicBlock *Cond = BasicBlock::Create(Ctx, "kernscope.coop.cond" + Suffix,
                                        &F, AnchorBB->getNextNode());
  BasicBlock *Body = BasicBlock::Create(Ctx, "kernscope.coop.body" + Suffix,
                                        &F, Cond->getNextNode());
  BasicBlock *Exit = BasicBlock::Create(Ctx, "kernscope.coop.exit" + Suffix,
                                        &F, Body->getNextNode());

  // Cond: phi (entry: tid, body: next), trip test against the extent.
  IRBuilder<> CondB(Cond);
  PHINode *Idx = CondB.CreatePHI(I64, 2, "kernscope.coop.idx");
  Idx->addIncoming(Tid, AnchorBB);
  Value *Cmp = CondB.CreateICmpULT(Idx, ConstantInt::get(I64, G.Extent),
                                   "kernscope.coop.cmp");
  CondB.CreateCondBr(Cmp, Body, Exit);

  // Body: tile[idx] = global[base + idx]; idx += blockDim.
  IRBuilder<> BodyB(Body);
  Value *GPtr = BodyB.CreateGEP(G.EltTy, G.Base, Idx, "kernscope.coop.gptr");
  LoadInst *GVal = BodyB.CreateLoad(G.EltTy, GPtr, "kernscope.coop.val");
  GVal->setAlignment(Align(4));
  Value *SPtr = BodyB.CreateInBoundsGEP(
      TileTy, G.TileAlloca, {ConstantInt::get(I64, 0), Idx},
      "kernscope.coop.sptr");
  BodyB.CreateStore(GVal, SPtr);
  Value *Next = BodyB.CreateAdd(Idx, Bdx, "kernscope.coop.next");
  Idx->addIncoming(Next, Body);
  BodyB.CreateBr(Cond);

  // Exit: barrier, then resume the displaced successor.
  IRBuilder<> ExitB(Exit);
  ExitB.CreateCall(M.getOrInsertFunction(BI.Barrier,
                                         FunctionType::get(
                                             Type::getVoidTy(Ctx), false)),
                   {});
  ExitB.CreateBr(OrigSucc);

  // Displace the anchor terminator: the anchor block now enters the
  // cooperative loop instead of branching straight to OrigSucc. Phis in
  // the displaced successors that name the anchor block as an incoming
  // block must be retargeted to the barrier block (the edge now comes
  // from there; the carried value is unchanged).
  for (BasicBlock *Succ : successors(AnchorTerm)) {
    for (PHINode &PN : Succ->phis()) {
      for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
        if (PN.getIncomingBlock(I) == AnchorBB) PN.setIncomingBlock(I, Exit);
      }
    }
  }
  AnchorTerm->eraseFromParent();
  IRBuilder<> A(AnchorBB);
  A.CreateBr(Cond);
  return Exit->getTerminator();
}

}  // namespace

namespace {
PromoteStats promoteFunction(Function &F, ScalarEvolution &SE, AAResults &AA,
                             LoopInfo &LI) {
  PromoteStats Stats;
  const DataLayout &DL = F.getParent()->getDataLayout();

  // Pass 1 (analysis): collect groups of candidate loads.
  SmallVector<Group, 4> Groups;
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *L = dyn_cast<LoadInst>(&I);
      if (!L || !L->isSimple()) continue;
      Value *Ptr = L->getPointerOperand();
      if (Ptr->getType()->getPointerAddressSpace() != 1) continue;
      Type *EltTy = L->getType();
      if (!EltTy->isIntegerTy() && !EltTy->isFloatingPointTy()) continue;
      const unsigned Bits = DL.getTypeSizeInBits(EltTy);
      if (Bits != 8 && Bits != 16 && Bits != 32 && Bits != 64) continue;

      Value *Base = getUnderlyingObject(Ptr, /*MaxLookup=*/6);
      if (Base->getType()->getPointerAddressSpace() != 1) continue;

      // Element-offset SCEV: byte offset / element size.
      const SCEV *ByteOff =
          SE.getMinusSCEV(SE.getSCEV(Ptr), SE.getSCEV(Base));
      const SCEV *ElemOff = divSCEV(SE, ByteOff, Bits / 8);
      if (!ElemOff) continue;  // not statically exact: not a candidate
      ElemOffset Off = classifyOffset(SE, ElemOff, LI.getLoopFor(&BB));
      if (Off.Kind == OffsetKind::kUnknown) continue;
      if (!gepShapeOk(Ptr, Base)) continue;  // index shape not rewriteable

      Group *G = findGroup(Groups, Base, EltTy);
      G->ElemBytes = Bits / 8;
      G->Loads.push_back(CandidateLoad{L, Off});
    }
  }

  // Extent = max reachable element per group (tile size); groups whose
  // extent is not provable are dropped (not candidates).
  SmallVector<Group, 4> Promotable;
  for (Group &G : Groups) {
    uint64_t Extent = 0;
    bool Ok = true;
    for (const CandidateLoad &C : G.Loads) {
      uint64_t Hi;
      if (C.Off.Kind == OffsetKind::kConstant) {
        Hi = C.Off.Const + 1;
      } else {
        const unsigned Trip = SE.getSmallConstantTripCount(C.Off.L);
        if (Trip == 0 ||
            C.Off.Const > std::numeric_limits<uint64_t>::max() - Trip) {
          Ok = false;
          break;
        }
        Hi = C.Off.Const + Trip;
      }
      if (Hi < C.Off.Const) {  // overflow
        Ok = false;
        break;
      }
      Extent = std::max(Extent, Hi);
    }
    if (Ok && Extent > 0) {
      G.Extent = Extent;
      Promotable.push_back(G);
    }
  }
  if (Promotable.empty()) return Stats;

  // Pass 2 (gates, then rewrite — all analysis before the first
  // mutation, so a late decline leaves the IR untouched).
  const std::optional<kernscope::ArchInfo> Arch =
      resolveArch(*F.getParent(), &F);
  BackendIntrinsics BI;
  const bool HasBackend = backendOf(*F.getParent(), BI);
  BasicBlock &Entry = F.getEntryBlock();
  const auto *EntryTerm = dyn_cast<BranchInst>(Entry.getTerminator());
  const bool EntryStraight = EntryTerm && EntryTerm->isUnconditional();

  Instruction *Anchor = Entry.getTerminator();
  for (Group &G : Promotable) {
    const uint64_t TileBytes = G.Extent * G.ElemBytes;
    DeclineReason Reason = DeclineReason::kNoPattern;

    if (!HasBackend) {
      Reason = DeclineReason::kNoBackend;
    } else if (!EntryStraight) {
      Reason = DeclineReason::kEntryTerm;
    } else if (!Arch) {
      Reason = DeclineReason::kNoArch;
    } else if (TileBytes >
               static_cast<uint64_t>(smemBudget(*Arch))) {
      Reason = DeclineReason::kBudget;
    } else {
      // AA guard: no memory op in the function may-modify the tile range
      // (a cross-iteration write to the same global would be observed by
      // a later rewritten read — the plan's MUST-NOT).
      const MemoryLocation TileLoc(G.Base, LocationSize::precise(TileBytes));
      bool MayWrite = false;
      for (Instruction &I : instructions(F)) {
        if (isModSet(AA.getModRefInfo(&I, TileLoc))) {
          MayWrite = true;
          break;
        }
      }
      if (MayWrite) {
        Reason = DeclineReason::kMayAliasWrite;
      } else {
        // Barrier dominance: a candidate read in the entry block would
        // execute before the cooperative load. All other blocks are
        // dominated by the barrier (entry's only successor is
        // redirected through it).
        for (const CandidateLoad &C : G.Loads) {
          if (C.LI->getParent() == &Entry) {
            Reason = DeclineReason::kEntryRead;
            break;
          }
        }
      }
    }

    if (Reason != DeclineReason::kNoPattern) {
      errs() << "kernscope-promote: declining promotion of "
             << G.Base->getName() << " in @" << F.getName() << ": "
             << declineReasonText(Reason);
      if (Reason == DeclineReason::kBudget)
        errs() << " (" << TileBytes << " bytes, arch budget "
               << smemBudget(*Arch) << " for " << Arch->name << ")";
      errs() << "\n";
      continue;
    }

    // ---- Rewrite: cooperative load + barrier + read redirection. ----
    Anchor = insertCooperativeLoad(F, G, BI, Anchor, Stats.groupsPromoted);
    Stats.groupsPromoted += 1;
    Stats.tileElements = std::max<uint64_t>(Stats.tileElements, G.Extent);

    // Redirect the reads: pointer operand -> smem GEP. The offset value
    // is rebuilt from the GEP indices (each index scaled by the element
    // count of the type it indexes into); the SCEV gate above already
    // proved containment in [0, Extent).
    SmallPtrSet<GetElementPtrInst *, 8> ChainGEPs;
    for (const CandidateLoad &C : G.Loads) {
      IRBuilder<> B(C.LI);
      Value *Off = nullptr;
      Value *Cur = C.LI->getPointerOperand();
      while (auto *GEP = dyn_cast<GetElementPtrInst>(Cur)) {
        ChainGEPs.insert(GEP);
        for (auto IT = gep_type_begin(GEP), ET = gep_type_end(GEP);
             IT != ET; ++IT) {
          Value *Idx = IT.getOperand();
          const uint64_t Scale =
              DL.getTypeAllocSize(IT.getIndexedType()) / G.ElemBytes;
          Value *Term = nullptr;
          if (auto *CI = dyn_cast<ConstantInt>(Idx)) {
            Term = ConstantInt::get(Type::getInt64Ty(F.getContext()),
                                    Scale * CI->getZExtValue());
          } else {
            Term = Idx;
            if (Term->getType()->getIntegerBitWidth() != 64)
              Term = B.CreateZExt(Term, Type::getInt64Ty(F.getContext()));
            if (Scale != 1)
              Term = B.CreateMul(
                  Term, ConstantInt::get(Type::getInt64Ty(F.getContext()),
                                         Scale));
          }
          Off = Off ? B.CreateAdd(Off, Term) : Term;
        }
        Cur = GEP->getPointerOperand();
        if (Cur == G.Base) break;
      }
      if (!Off) Off = ConstantInt::get(Type::getInt64Ty(F.getContext()), 0);
      auto *TileTy = ArrayType::get(G.EltTy, G.Extent);
      Value *SPtr = B.CreateInBoundsGEP(
          TileTy, G.TileAlloca, {ConstantInt::get(Type::getInt64Ty(
                                     F.getContext()), 0), Off},
          "kernscope.sptr");
      C.LI->setOperand(llvm::LoadInst::getPointerOperandIndex(), SPtr);
      Stats.loadsPromoted += 1;
    }
    while (true) {
      GetElementPtrInst *Dead = nullptr;
      for (GetElementPtrInst *GEP : ChainGEPs)
        if (GEP->use_empty()) {
          Dead = GEP;
          break;
        }
      if (!Dead) break;
      ChainGEPs.erase(Dead);
      Dead->eraseFromParent();
    }
  }

  return Stats;
}
}  // namespace

}  // namespace pass
}  // namespace kernscope

namespace kernscope {
namespace pass {

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
PreservedAnalyses GlobalToSharedPromotePass::run(Function &F,
                                                 FunctionAnalysisManager &FAM) {
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  AAResults &AA = FAM.getResult<AAManager>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);

  PromoteStats Stats = promoteFunction(F, SE, AA, LI);

  // Report hook (todo 42): one line per function with rewrites, stderr.
  if (PromoteReportFlag && Stats.loadsPromoted != 0) {
    errs() << "kernscope-promote: " << F.getName() << ": promoted "
           << Stats.loadsPromoted << " loads to shared ("
           << Stats.tileElements << "-element tile)\n";
  }

  if (Stats.loadsPromoted == 0) return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

}  // namespace pass
}  // namespace kernscope

namespace {

bool parsePromoteName(StringRef Name, ModulePassManager &MPM,
                      ArrayRef<PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-promote") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::GlobalToSharedPromotePass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h). Function-scoped
// transform: textual name + pipeline-start EP + peephole EP (the same set
// as the todo-39 rle module).
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parsePromoteName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::GlobalToSharedPromotePass()));
      });
  PB.registerPeepholeEPCallback(
      [](FunctionPassManager &FPM, OptimizationLevel) {
        FPM.addPass(kernscope::pass::GlobalToSharedPromotePass());
      });
});
