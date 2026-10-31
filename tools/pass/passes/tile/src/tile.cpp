// kernscope::pass — loop tiling (todo 41, Wave 7).
//
// kernscope-tile tiles innermost GPU loops by arch-derived tile sizes
// (libs/arch smem budget + register budget estimates) so that
// kernscope-promote (todo 40) can consume constant-trip-count,
// unit-stride {s,+,1} windows — the "must write" LLVM-IR-level tiler of
// docs/report/kernscope-technical-report.pdf/§4.9. Full design +
// soundness rules in include/kernscope/pass/tile.h; the essentials:
//
//   for (i = 0; i < N; ++i) body(i);
//     ==>
//   for (t = 0; t < N/TS; ++t)          // tile loop (trip N/TS, constant)
//     for (i = t*TS; i < t*TS+TS; ++i)  // full-tile loop (trip TS, constant)
//       body(i);
//   for (r = 0; r < N%TS; ++r)          // remainder loop (trip N%TS,
//     body(Tfull*TS + r);               //  constant) — only when N%TS != 0
//
// The remainder loop has a CONSTANT start and trip count: exactly the
// {s,+,1} window kernscope-promote v1 consumes (the pipeline fixture
// design rule). Non-IV loop-carried phis (reduction accumulators) are
// carried across tiles via a tile-loop-header phi + LCSSA phi in the
// tile latch; the remainder loop continues the carried value; exit phis
// are retargeted to the final carrier.
//
// Soundness gates (any failure = decline, IR untouched — two-phase:
// ALL analysis for ALL candidate loops first, mutation after):
//   1. innermost, LoopSimplify, single-block body (v1 scope);
//   2. constant trip count N = SE.getSmallConstantTripCount (SCEV-driven
//      bound handling; non-divisible N yields the remainder loop);
//   3. N > tile size TS (N < TS: tiling pointless; N == TS: single-tile
//      no-op — both decline, which also keeps the pass idempotent);
//   4. the plan's MUST-NOT — no cross-iteration memory dependences:
//      LoopAccessAnalysis must report canVectorizeMemory(); no convergent
//      op (barrier), no volatile/atomic op, no call in the body;
//   5. arch-derived TS (largest pow2 of {32..1024} with TS*elemBytes <=
//      smem budget AND TS <= max_regs_per_thread; libs/arch, never
//      hardcoded); the "kernscope.tile.arch" function attribute or
//      nvvmarch/amdgpu-arch module flags name the arch; the
//      "kernscope.tile.size" attribute pins TS for tests;
//   6. canonical {0,+,1} IV with a `inc = i+1; cmp = icmp ult inc, N`
//      latch test, and exit phis whose values are carried header-phi
//      latch values (v1 shapes).

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/Twine.h>
#include <llvm/Analysis/LoopAccessAnalysis.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/Module.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/ValueMapper.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

#include <kernscope/arch.h>
#include <kernscope/pass/register_pass.h>
#include <kernscope/pass/tile.h>

using namespace llvm;

namespace kernscope {
namespace pass {
namespace {

cl::opt<bool> TileReportFlag(
    "kernscope-tile-report",
    cl::desc("Print kernscope-tile rewrite counts to stderr"),
    cl::init(false));

// ---------------------------------------------------------------------------
// Arch resolution + tile-size derivation (libs/arch reuse — never
// hardcoded; the same pattern as the todo-38 pad / todo-40 promote
// modules, TU-local because tools/pass/include is READ-ONLY).
// ---------------------------------------------------------------------------

std::optional<kernscope::ArchInfo> resolveArch(const Module &M,
                                               const Function *F) {
  if (F) {
    const Attribute A = F->getFnAttribute("kernscope.tile.arch");
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

// Largest power-of-two tile in {32, 64, 128, 256, 512, 1024} elements that
// fits both the per-block smem budget (a whole tile promotable by
// kernscope-promote) and the per-thread register budget (a whole tile
// addressable in registers). ElemBytes = the loop's widest scalar access
// element (default 4).
uint64_t deriveTileSize(const kernscope::ArchInfo &A, uint64_t ElemBytes) {
  const uint64_t Cap = std::min<uint64_t>(
      static_cast<uint64_t>(smemBudget(A)) / ElemBytes,
      static_cast<uint64_t>(A.max_regs_per_thread));
  static constexpr uint64_t kCandidates[] = {32, 64, 128, 256, 512, 1024};
  for (int I = 5; I >= 0; --I) {
    if (kCandidates[I] <= Cap) return kCandidates[I];
  }
  return 0;  // even the smallest candidate does not fit
}

// ---------------------------------------------------------------------------
// Decline bookkeeping (mirrors the todo-40 promote annotation contract).
// ---------------------------------------------------------------------------

enum class DeclineReason : std::uint8_t {
  kStructure,      // not innermost/simplify-form/single-block (silent)
  kNoConstantTrip, // trip count not a compile-time constant (SCEV)
  kNoArch,         // no arch attribute/flag -> cannot derive a tile size
  kNoTileFits,     // no candidate tile fits the arch budgets
  kTripBelowTile,  // N < TS: tiling pointless
  kSingleTile,     // N == TS: single-tile no-op (keeps the pass idempotent)
  kDependence,     // LAI: cross-iteration memory dependence (MUST-NOT)
  kConvergent,     // barrier/convergent op in the body
  kVolatileAtomic, // volatile/atomic memory op in the body
  kCall,           // call in the body (v1 scope)
  kNoCanonicalIV,  // no {0,+,1} induction phi
  kLatchShape,     // latch is not `inc = i+1; icmp ult inc, N`
  kExitPhi,        // exit phi value is not a carried header-phi value
};

const char *declineReasonText(DeclineReason R) {
  switch (R) {
    case DeclineReason::kStructure:
      return "not an innermost LoopSimplify single-block loop (v1 scope)";
    case DeclineReason::kNoConstantTrip:
      return "trip count is not a compile-time constant (SCEV)";
    case DeclineReason::kNoArch:
      return "no arch (kernscope.tile.arch attribute or nvvmarch/"
             "amdgpu-arch module flag); tile size underivable";
    case DeclineReason::kNoTileFits:
      return "no candidate tile size fits the arch smem/register budget";
    case DeclineReason::kTripBelowTile:
      return "trip count < tile size; tiling pointless";
    case DeclineReason::kSingleTile:
      return "trip count == tile size; single-tile tiling is a no-op";
    case DeclineReason::kDependence:
      return "cross-iteration memory dependence (LAI)";
    case DeclineReason::kConvergent:
      return "convergent/barrier op in the loop body";
    case DeclineReason::kVolatileAtomic:
      return "volatile/atomic memory op in the loop body";
    case DeclineReason::kCall:
      return "call in the loop body (v1 scope)";
    case DeclineReason::kNoCanonicalIV:
      return "no canonical {0,+,1} induction phi";
    case DeclineReason::kLatchShape:
      return "latch is not `inc = i+1; icmp ult inc, N`";
    case DeclineReason::kExitPhi:
      return "exit phi value is not a carried header-phi latch value";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// Per-loop analysis (pass 1 — no mutation). One TilePlan per candidate;
// everything the rewrite needs is captured up front so all loops can be
// analyzed before the first mutation (promote's two-phase pattern).
// ---------------------------------------------------------------------------

struct TilePlan {
  Loop *L = nullptr;
  BasicBlock *LoopBlock = nullptr;  // header == latch (single-block v1)
  PHINode *IV = nullptr;
  ICmpInst *Cmp = nullptr;  // icmp ult %inc, N
  BasicBlock *Exit = nullptr;  // Cmp's other successor
  uint64_t N = 0;       // trip count
  uint64_t TS = 0;      // tile size (elements)
  uint64_t Tfull = 0;   // N / TS
  uint64_t R = 0;       // N % TS
  Type *IVTy = nullptr;
  // Header phis other than the IV: (phi, (entry value, latch value)).
  SmallVector<std::pair<PHINode *, std::pair<Value *, Value *>>, 4> Carried;
  uint64_t ElemBytes = 4;  // widest scalar access element
};

std::optional<TilePlan> analyzeLoop(Loop *L, ScalarEvolution &SE,
                                    DeclineReason &Reason) {
  if (!L->isLoopSimplifyForm() || !L->isInnermost() ||
      L->getNumBlocks() != 1) {
    Reason = DeclineReason::kStructure;
    return std::nullopt;
  }
  const unsigned Trip = SE.getSmallConstantTripCount(L);
  if (Trip == 0) {
    Reason = DeclineReason::kNoConstantTrip;
    return std::nullopt;
  }

  TilePlan Plan;
  Plan.L = L;
  Plan.N = Trip;
  Plan.LoopBlock = L->getHeader();

  // Canonical {0,+,1} IV.
  for (PHINode &PN : Plan.LoopBlock->phis()) {
    if (const auto *AR = dyn_cast<SCEVAddRecExpr>(SE.getSCEV(&PN))) {
      if (AR->getLoop() == L) {
        const auto *StartC = dyn_cast<SCEVConstant>(AR->getStart());
        const auto *StepC =
            dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
        if (StartC && StepC && StartC->getAPInt().isZero() &&
            StepC->getAPInt().getZExtValue() == 1) {
          Plan.IV = &PN;
          Plan.IVTy = PN.getType();
          break;
        }
      }
    }
  }
  if (!Plan.IV) {
    Reason = DeclineReason::kNoCanonicalIV;
    return std::nullopt;
  }

  // Carried phis (everything except the IV); each must have exactly the
  // preheader and backedge incoming.
  BasicBlock *Preheader = L->getLoopPreheader();
  for (PHINode &PN : Plan.LoopBlock->phis()) {
    if (&PN == Plan.IV) continue;
    Value *EntryV = nullptr;
    Value *LatchV = nullptr;
    for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
      if (PN.getIncomingBlock(I) == Preheader) {
        EntryV = PN.getIncomingValue(I);
      } else if (PN.getIncomingBlock(I) == Plan.LoopBlock) {
        LatchV = PN.getIncomingValue(I);
      }
    }
    if (!EntryV || !LatchV) {
      Reason = DeclineReason::kStructure;
      return std::nullopt;
    }
    Plan.Carried.push_back({&PN, {EntryV, LatchV}});
  }

  // Latch shape: `%inc = add i64 %i, 1; %cmp = icmp ult i64 %inc, N`.
  const auto *Term = dyn_cast<BranchInst>(Plan.LoopBlock->getTerminator());
  if (!Term || !Term->isConditional() || Term->getNumSuccessors() != 2 ||
      Term->getSuccessor(0) != Plan.LoopBlock) {
    Reason = DeclineReason::kLatchShape;
    return std::nullopt;
  }
  Plan.Cmp = dyn_cast<ICmpInst>(Term->getCondition());
  if (!Plan.Cmp || Plan.Cmp->getPredicate() != ICmpInst::ICMP_ULT ||
      Plan.Cmp->getOperand(0) == Plan.IV) {
    Reason = DeclineReason::kLatchShape;
    return std::nullopt;
  }
  // The compared value must be IV+1 (add in either operand order) and the
  // bound must be the constant N.
  auto *Inc = dyn_cast<Instruction>(Plan.Cmp->getOperand(0));
  const auto *BC = dyn_cast<ConstantInt>(Plan.Cmp->getOperand(1));
  bool IncOk = false;
  if (Inc && Inc->getOpcode() == Instruction::Add) {
    for (Value *Op : Inc->operands()) {
      if (Op == Plan.IV) IncOk = true;
    }
    if (IncOk) {
      Value *Other = Inc->getOperand(0) == Plan.IV ? Inc->getOperand(1)
                                                   : Inc->getOperand(0);
      IncOk = isa<ConstantInt>(Other) &&
              cast<ConstantInt>(Other)->getZExtValue() == 1;
    }
  }
  if (!IncOk || !BC || BC->getZExtValue() != Plan.N) {
    Reason = DeclineReason::kLatchShape;
    return std::nullopt;
  }
  Plan.Exit = Term->getSuccessor(1);

  // Exit phis: every incoming from the loop block must be a carried
  // header-phi latch value (the value flows through the outer carrier).
  for (PHINode &PN : Plan.Exit->phis()) {
    for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
      if (PN.getIncomingBlock(I) != Plan.LoopBlock) continue;
      Value *V = PN.getIncomingValue(I);
      bool Found = false;
      for (auto &[P, Vals] : Plan.Carried) {
        if (Vals.second == V) {
          Found = true;
          break;
        }
      }
      if (!Found) {
        Reason = DeclineReason::kExitPhi;
        return std::nullopt;
      }
    }
  }

  // Widest scalar access element size (for the smem budget estimate).
  const DataLayout &DL = L->getHeader()->getModule()->getDataLayout();
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        Plan.ElemBytes = std::max<uint64_t>(
            Plan.ElemBytes, DL.getTypeAllocSize(LI->getType()));
      } else if (auto *SI = dyn_cast<StoreInst>(&I)) {
        Plan.ElemBytes = std::max<uint64_t>(
            Plan.ElemBytes,
            DL.getTypeAllocSize(SI->getValueOperand()->getType()));
      }
    }
  }
  return Plan;
}

// ---------------------------------------------------------------------------
// Rewrite (pass 2 — only reached after ALL analysis passed).
// ---------------------------------------------------------------------------

void tileOneLoop(Function &F, const TilePlan &Plan) {
  LLVMContext &Ctx = F.getContext();
  BasicBlock *LoopBlock = Plan.LoopBlock;
  BasicBlock *Preheader = Plan.L->getLoopPreheader();
  BasicBlock *Exit = Plan.Exit;
  const uint64_t Tfull = Plan.Tfull;
  const uint64_t R = Plan.R;
  const uint64_t TS = Plan.TS;

  auto *TSConst = ConstantInt::get(Plan.IVTy, TS);
  auto *TfullConst = ConstantInt::get(Plan.IVTy, Tfull);

  // New blocks, in layout order: outer preheader, outer header, inner
  // preheader (before the loop block); tile latch (after it); remainder
  // preheader/check/body/latch (before the exit).
  BasicBlock *OuterPre = BasicBlock::Create(Ctx, "tile.pre", &F, LoopBlock);
  BasicBlock *OuterCheck = BasicBlock::Create(Ctx, "tile.check", &F,
                                              OuterPre->getNextNode());
  BasicBlock *InnerPre = BasicBlock::Create(Ctx, "tile.inner.pre", &F,
                                            OuterCheck->getNextNode());
  BasicBlock *TileLatch = BasicBlock::Create(Ctx, "tile.latch", &F,
                                             LoopBlock->getNextNode());
  BasicBlock *RemPre = nullptr;
  BasicBlock *RemCheck = nullptr;
  BasicBlock *RemBody = nullptr;
  BasicBlock *RemLatch = nullptr;
  if (R > 0) {
    RemPre = BasicBlock::Create(Ctx, "rem.pre", &F, Exit);
    RemCheck = BasicBlock::Create(Ctx, "rem.check", &F, Exit);
    RemBody = BasicBlock::Create(Ctx, "rem.body", &F, Exit);
    RemLatch = BasicBlock::Create(Ctx, "rem.latch", &F, Exit);
  }

  // The preheader now enters the outer loop's preheader, which enters the
  // outer header (the outer phis' first incoming block is tile.pre).
  Preheader->getTerminator()->eraseFromParent();
  IRBuilder<> PreB(Preheader);
  PreB.CreateBr(OuterPre);
  IRBuilder<> OuterPreB(OuterPre);
  OuterPreB.CreateBr(OuterCheck);

  // Outer header: tile IV + one carried phi per non-IV header phi.
  IRBuilder<> CheckB(OuterCheck);
  PHINode *TileIV =
      CheckB.CreatePHI(Plan.IVTy, 2, Plan.IV->getName() + ".t");
  TileIV->addIncoming(ConstantInt::get(Plan.IVTy, 0), OuterPre);
  SmallVector<std::pair<PHINode *, PHINode *>, 4> OuterCarried;
  for (const auto &[P, Vals] : Plan.Carried) {
    PHINode *Outer = CheckB.CreatePHI(P->getType(), 2, P->getName() + ".t");
    Outer->addIncoming(Vals.first, OuterPre);
    OuterCarried.push_back({P, Outer});
  }
  Value *TileCmp = CheckB.CreateICmpULT(TileIV, TfullConst, "tile.cmp");
  CheckB.CreateCondBr(TileCmp, InnerPre, R > 0 ? RemPre : Exit);

  // Inner preheader: tile start/end for the full-tile loop.
  IRBuilder<> InnerB(InnerPre);
  Value *Start =
      InnerB.CreateMul(TileIV, TSConst, Plan.IV->getName() + ".start");
  Value *End =
      InnerB.CreateAdd(Start, TSConst, Plan.IV->getName() + ".end");
  InnerB.CreateBr(LoopBlock);

  // Retarget the header phis: the preheader edge now comes from the inner
  // preheader (IV -> tile start; carried -> outer phi).
  for (PHINode &PN : LoopBlock->phis()) {
    for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
      if (PN.getIncomingBlock(I) == Preheader) {
        PN.setIncomingBlock(I, InnerPre);
        if (&PN == Plan.IV) {
          PN.setIncomingValue(I, Start);
        } else {
          for (auto &[P, Outer] : OuterCarried) {
            if (&PN == P) PN.setIncomingValue(I, Outer);
          }
        }
      }
    }
  }

  // The latch test now bounds against the inner end.
  Plan.Cmp->setOperand(1, End);

  // Tile latch: LCSSA phi per carried value + next tile index.
  IRBuilder<> LatchB(TileLatch);
  SmallVector<PHINode *, 4> LcssaPhis;
  for (const auto &[P, Vals] : Plan.Carried) {
    PHINode *LC =
        LatchB.CreatePHI(P->getType(), 1, P->getName() + ".lcssa");
    LC->addIncoming(Vals.second, LoopBlock);
    LcssaPhis.push_back(LC);
  }
  Value *TileNext =
      LatchB.CreateAdd(TileIV, ConstantInt::get(Plan.IVTy, 1), "tile.inc");
  for (unsigned I = 0; I < OuterCarried.size(); ++I)
    OuterCarried[I].second->addIncoming(LcssaPhis[I], TileLatch);
  TileIV->addIncoming(TileNext, TileLatch);
  LatchB.CreateBr(OuterCheck);

  // Retarget the loop terminator: on exit it now enters the tile latch.
  auto *LoopTerm = cast<BranchInst>(LoopBlock->getTerminator());
  LoopTerm->setSuccessor(1, TileLatch);

  if (R > 0) {
    // Remainder preheader.
    IRBuilder<> RemPreB(RemPre);
    RemPreB.CreateBr(RemCheck);

    // Remainder check: IV + carried phis, bound R (constant).
    IRBuilder<> RemCheckB(RemCheck);
    PHINode *RemIV =
        RemCheckB.CreatePHI(Plan.IVTy, 2, Plan.IV->getName() + ".r");
    RemIV->addIncoming(ConstantInt::get(Plan.IVTy, 0), RemPre);
    SmallVector<PHINode *, 4> RemCarried;
    for (const auto &[P, Vals] : Plan.Carried) {
      PHINode *RP = RemCheckB.CreatePHI(P->getType(), 2,
                                        P->getName() + ".rem");
      RP->addIncoming(Vals.first, RemPre);  // continues the outer carrier
      RemCarried.push_back(RP);
    }
    Value *RemCmp =
        RemCheckB.CreateICmpULT(RemIV, ConstantInt::get(Plan.IVTy, Plan.R),
                                "rem.cmp");
    RemCheckB.CreateCondBr(RemCmp, RemBody, Exit);

    // Remainder body: clone of the loop body with the IV -> (base + r)
    // and carried phis -> remainder phis.
    IRBuilder<> RemBodyB(RemBody);
    Value *Base = ConstantInt::get(Plan.IVTy, Tfull * TS);
    Value *RIdx =
        RemBodyB.CreateAdd(Base, RemIV, Plan.IV->getName() + ".idx");
    ValueToValueMapTy VMap;
    VMap[Plan.IV] = RIdx;
    for (unsigned I = 0; I < Plan.Carried.size(); ++I)
      VMap[Plan.Carried[I].first] = RemCarried[I];
    SmallVector<Instruction *, 8> Clones;
    Instruction *IncInst = cast<Instruction>(Plan.Cmp->getOperand(0));
    for (Instruction &I : *LoopBlock) {
      // Skip the header phis, the loop terminator, and the latch test
      // (inc + cmp): the remainder body reuses the same IV and the test
      // is dead here (and would reference the non-dominating inner end).
      if (isa<PHINode>(I) || isa<BranchInst>(I)) continue;
      if (&I == Plan.Cmp || &I == IncInst) continue;
      Instruction *C = I.clone();
      C->setName(I.getName() + ".rem");
      VMap[&I] = C;
      Clones.push_back(C);
    }
    for (Instruction *C : Clones) RemapInstruction(C, VMap, RF_None);
    for (Instruction *C : Clones) RemBodyB.Insert(C);
    RemBodyB.CreateBr(RemLatch);

    // Remainder latch: next index; carried values from the cloned body.
    IRBuilder<> RemLatchB(RemLatch);
    Value *RemNext = RemLatchB.CreateAdd(
        RemIV, ConstantInt::get(Plan.IVTy, 1), "rem.inc");
    for (unsigned I = 0; I < Plan.Carried.size(); ++I) {
      Value *OrigV = Plan.Carried[I].second.second;
      Value *CloneV = VMap[OrigV];
      RemCarried[I]->addIncoming(CloneV, RemLatch);
    }
    RemIV->addIncoming(RemNext, RemLatch);
    RemLatchB.CreateBr(RemCheck);

    // Exit phis: the value now arrives from the remainder check (the only
    // block branching to the exit), carrying the remainder's final value.
    for (PHINode &PN : Exit->phis()) {
      for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
        if (PN.getIncomingBlock(I) == LoopBlock) {
          Value *V = PN.getIncomingValue(I);
          for (unsigned C = 0; C < Plan.Carried.size(); ++C) {
            if (Plan.Carried[C].second.second == V) {
              PN.setIncomingBlock(I, RemCheck);
              PN.setIncomingValue(I, RemCarried[C]);
              break;
            }
          }
        }
      }
    }
  } else {
    // No remainder: the exit phis come straight from the outer check,
    // carrying the outer phi (final value after the last tile).
    for (PHINode &PN : Exit->phis()) {
      for (unsigned I = 0; I < PN.getNumIncomingValues(); ++I) {
        if (PN.getIncomingBlock(I) == LoopBlock) {
          Value *V = PN.getIncomingValue(I);
          for (unsigned C = 0; C < Plan.Carried.size(); ++C) {
            if (Plan.Carried[C].second.second == V) {
              PN.setIncomingBlock(I, OuterCheck);
              PN.setIncomingValue(I, OuterCarried[C].second);
              break;
            }
          }
        }
      }
    }
  }
}

// True when every memory op in the loop is simple (non-volatile,
// non-atomic) and the body has no calls (v1: remainder-body cloning must
// not duplicate side-effecting semantics).
bool bodyHasSimpleMemoryOnly(Loop *L, DeclineReason &Reason) {
  for (BasicBlock *BB : L->blocks()) {
    for (Instruction &I : *BB) {
      if (isa<CallBase>(I)) {
        Reason = DeclineReason::kCall;
        return false;
      }
      if (auto *LI = dyn_cast<LoadInst>(&I)) {
        if (LI->isVolatile() || LI->isAtomic()) {
          Reason = DeclineReason::kVolatileAtomic;
          return false;
        }
        continue;
      }
      if (auto *SI = dyn_cast<StoreInst>(&I)) {
        if (SI->isVolatile() || SI->isAtomic()) {
          Reason = DeclineReason::kVolatileAtomic;
          return false;
        }
        continue;
      }
      if (isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) ||
          isa<FenceInst>(I)) {
        Reason = DeclineReason::kVolatileAtomic;
        return false;
      }
    }
  }
  return true;
}

}  // namespace

namespace {
TileStats tileFunction(Function &F, ScalarEvolution &SE, LoopInfo &LI,
                       LoopAccessInfoManager &LAIM,
                       const std::optional<kernscope::ArchInfo> &Arch,
                       const std::optional<uint64_t> &PinnedTS) {
  TileStats Stats;

  // Pass 1 (analysis): collect plans for every candidate loop.
  SmallVector<TilePlan, 4> Plans;
  for (Loop *L : LI.getLoopsInPreorder()) {
    if (!L->isInnermost()) continue;  // outer nest loops are left as-is
    DeclineReason Reason = DeclineReason::kStructure;
    std::optional<TilePlan> Plan = analyzeLoop(L, SE, Reason);
    if (!Plan) {
      // Structural ineligibility is silent; real declines print (the
      // todo-40 promote annotation contract).
      if (Reason != DeclineReason::kStructure) {
        errs() << "kernscope-tile: declining tiling of loop @"
               << L->getHeader()->getName() << " in @" << F.getName()
               << ": " << declineReasonText(Reason) << "\n";
      }
      continue;
    }
    if (!Arch) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kNoArch) << "\n";
      continue;
    }
    const uint64_t TS =
        PinnedTS ? *PinnedTS : deriveTileSize(*Arch, Plan->ElemBytes);
    if (TS == 0) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kNoTileFits)
             << "\n";
      continue;
    }
    if (Plan->N < TS) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kTripBelowTile)
             << " (" << Plan->N << " < " << TS << ")\n";
      continue;
    }
    if (Plan->N == TS) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kSingleTile)
             << "\n";
      continue;
    }
    // Dependence gate (the plan's MUST-NOT): LAI must prove the loop
    // memory-safe (no cross-iteration dependencies) and barrier-free.
    const LoopAccessInfo &LAI = LAIM.getInfo(*L);
    if (!LAI.canVectorizeMemory()) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kDependence)
             << "\n";
      continue;
    }
    if (LAI.hasConvergentOp()) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(DeclineReason::kConvergent)
             << "\n";
      continue;
    }
    DeclineReason MemReason = DeclineReason::kStructure;
    if (!bodyHasSimpleMemoryOnly(L, MemReason)) {
      errs() << "kernscope-tile: declining tiling of loop @"
             << L->getHeader()->getName() << " in @" << F.getName()
             << ": " << declineReasonText(MemReason) << "\n";
      continue;
    }

    Plan->TS = TS;
    Plan->Tfull = Plan->N / TS;
    Plan->R = Plan->N % TS;
    Plans.push_back(*Plan);
  }

  // Pass 2 (rewrite): only after all analysis passed — a late decline
  // leaves the IR untouched.
  for (const TilePlan &Plan : Plans) {
    tileOneLoop(F, Plan);
    Stats.loopsTiled += 1;
    Stats.fullTiles += Plan.Tfull;
    Stats.remainderIters += Plan.R;
    if (TileReportFlag) {
      errs() << "kernscope-tile: " << F.getName() << ": tiled loop @"
             << Plan.L->getHeader()->getName() << " (" << Plan.N
             << " -> " << Plan.Tfull << " tiles of " << Plan.TS << " + "
             << Plan.R << ")\n";
    }
  }
  return Stats;
}
}  // namespace

}  // namespace pass
}  // namespace kernscope

// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
PreservedAnalyses kernscope::pass::LoopTilingPass::run(
    Function &F, FunctionAnalysisManager &FAM) {
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  LoopInfo &LI = FAM.getResult<LoopAnalysis>(F);
  LoopAccessInfoManager &LAIM = FAM.getResult<LoopAccessAnalysis>(F);

  const std::optional<kernscope::ArchInfo> Arch =
      kernscope::pass::resolveArch(*F.getParent(), &F);
  std::optional<uint64_t> PinnedTS;
  const Attribute Pin = F.getFnAttribute("kernscope.tile.size");
  if (Pin.isStringAttribute()) {
    uint64_t V = 0;
    if (!Pin.getValueAsString().getAsInteger(10, V) && V != 0) PinnedTS = V;
  }

  TileStats Stats = tileFunction(F, SE, LI, LAIM, Arch, PinnedTS);

  if (Stats.loopsTiled == 0) return PreservedAnalyses::all();
  return PreservedAnalyses::none();
}

namespace {

bool parseTileName(StringRef Name, ModulePassManager &MPM,
                   ArrayRef<PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-tile") {
    MPM.addPass(createModuleToFunctionPassAdaptor(
        kernscope::pass::LoopTilingPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h). Textual name +
// pipeline-start EP; NO peephole EP (a loop restructure re-run after every
// InstCombine would be useless — documented in tile.h).
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseTileName);

  PB.registerPipelineStartEPCallback(
      [](ModulePassManager &MPM, OptimizationLevel) {
        MPM.addPass(createModuleToFunctionPassAdaptor(
            kernscope::pass::LoopTilingPass()));
      });
});
