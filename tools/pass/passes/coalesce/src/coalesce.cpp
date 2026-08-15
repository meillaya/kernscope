// kernscope::pass — coalescing analysis (todo 42, Wave 7). SKELETON.
//
// Analysis-only pass: classifies every addrspace(1) (NVPTX global) memory
// op of each kernel function into the todo-15 vocabulary (contiguous-128B /
// contiguous-partial / strided / unknown-pattern) under the 32B-sector
// model, and exposes per-function counts for the todo-42 before/after
// report. NO transforms: run() returns PreservedAnalyses::all().
//
// IR-vs-PTX split (honest documentation, same plan-vs-actuality handling
// as todo 38's bank-math note — verified 2026-08-15 that libs/ has NO
// shared coalescing classifier; the analyzer's math lives in
// tools/analyze/src/analyze_coalescing.cpp):
//
//   * THIS pass = IR-level consumer. The per-lane coefficient is extracted
//     from SCEV: the threadIdx.x / workitem.id.x intrinsic call value
//     enters the pointer SCEV as a SCEVUnknown term, and the coefficient
//     of that term (via SCEVMulExpr) is the per-lane byte stride. This is
//     the LLVM-IR analog of the analyzer's textual mov/add/shl/mul
//     resolver (analyze_coalescing.cpp: record_def).
//   * tools/analyze = PTX-level consumer of the same concept with the same
//     vocabulary (classify_pattern: contiguous-128B | contiguous-partial |
//     strided | unknown-pattern) and the same 32B-sector efficiency model.
//   * The two levels never round-trip numbers through a shared lib; they
//     agree by construction on vocabulary + model, so an IR-level report
//     and a PTX-level report of the same kernel are comparable. If a
//     shared classifier is ever extracted to libs/, this module becomes
//     its consumer (the include/kernscope/pass/coalesce.h API is the
//     seam).
//
// Honesty rule (mirrors the analyzer's coalescing_unknown finding): an
// access whose SCEV is not affine in the lane value is classified
// unknown-pattern — never optimistically claimed coalesced.

#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/Analysis/ScalarEvolutionExpressions.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <set>

#include <kernscope/pass/coalesce.h>
#include <kernscope/pass/register_pass.h>
#include <algorithm>

using namespace llvm;

namespace {

cl::opt<bool> CoalesceReportFlag(
    "kernscope-coalesce-report",
    cl::desc("Print kernscope-coalesce per-function classification counts "
             "to stderr"),
    cl::init(false));

}  // namespace

namespace kernscope {
namespace pass {
namespace {

// Backend lane intrinsics: threadIdx.x on NVPTX, workitem.id.x on AMDGCN —
// the IR analog of the analyzer's %tid.x register (analyze_coalescing.cpp:
// tid_expr()).
bool isLaneIntrinsic(const CallBase &CB) {
  const Function *F = CB.getCalledFunction();
  if (!F) {
    return false;
  }
  // Backend lane intrinsics by name (the per-target intrinsic enum tables
  // are TableGen-gated; name matching mirrors the analyzer's %tid.x
  // mnemonic matching on PTX).
  return F->getName() == "llvm.nvvm.read.ptx.sreg.tid.x" ||
         F->getName() == "llvm.amdgcn.workitem.id.x";
}

// The function's lane value (the tid intrinsic call), or nullptr when the
// function has none (then every access is uniform across the warp — the
// analyzer's coeff==0 case).
Value *findLaneValue(Function &F) {
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (CB && isLaneIntrinsic(*CB)) {
        return CB;
      }
    }
  }
  return nullptr;
}

// A value that is safe to treat as a uniform (warp-invariant) address
// term: kernel parameters (the analyzer's ld.param -> base_pointer_expr),
// globals (cvta.to.global), constants, and the BLOCK index intrinsics
// (ctaid/workgroup.id — per-BLOCK values, uniform across any warp by
// construction; the analyzer's %ctaid.x -> base_pointer_expr case).
// Anything data-derived (loads, calls, computed pointers, threadIdx.y/z —
// warp-varying whenever blockDim.x is not a multiple of 32, and blockDim
// is not a compile-time constant in IR) makes the per-lane affine
// UNRESOLVABLE — the analyzer's !e.known -> unknown-pattern honesty rule
// (analyze_coalescing.cpp: "indirect/data-dependent index; coalescing not
// assumed"). NEVER claimed coalesced without proof.
//
// Divergence from the analyzer, documented: analyze_coalescing.cpp treats
// %tid.y/%tid.z as uniform bases; this IR-level classifier treats them as
// warp-varying -> unknown-pattern, because blockDim.x is not provably a
// warp multiple at IR level. Stricter = honest.
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR/SCEV tree recursion
bool isUniformTerm(const Value *V) {
  if (isa<Argument>(V) || isa<GlobalValue>(V) || isa<Constant>(V)) {
    return true;
  }
  if (const auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
    return isUniformTerm(GEP->getPointerOperand());
  }
  // An addrspacecast preserves warp-uniformity (the kernscope-as1 param
  // casts land here).
  if (const auto *ASC = dyn_cast<AddrSpaceCastInst>(V)) {
    return isUniformTerm(ASC->getPointerOperand());
  }
  if (const auto *CB = dyn_cast<CallBase>(V)) {
    const Function *F = CB->getCalledFunction();
    if (!F) {
      return false;
    }
    const StringRef N = F->getName();
    // Block/block-dimension ids: uniform across the whole block, hence
    // across any warp.
    return N == "llvm.nvvm.read.ptx.sreg.ctaid.x" ||
           N == "llvm.nvvm.read.ptx.sreg.ctaid.y" ||
           N == "llvm.nvvm.read.ptx.sreg.ctaid.z" ||
           N == "llvm.nvvm.read.ptx.sreg.ntid.x" ||
           N == "llvm.nvvm.read.ptx.sreg.ntid.y" ||
           N == "llvm.nvvm.read.ptx.sreg.ntid.z" ||
           N == "llvm.amdgcn.workgroup.id.x" ||
           N == "llvm.amdgcn.workgroup.id.y" ||
           N == "llvm.amdgcn.workgroup.id.z" ||
           N == "llvm.amdgcn.workgroup.size.x" ||
           N == "llvm.amdgcn.workgroup.size.y" ||
           N == "llvm.amdgcn.workgroup.size.z";
  }
  return false;
}

// Does the SCEV node mention the lane value at all?
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR/SCEV tree recursion
bool mentionsLane(const SCEV *S, const Value *LaneValue) {
  if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
    return U->getValue() == LaneValue;
  }
  return std::ranges::any_of(
      S->operands(),
      [&](const SCEV *Op) { return mentionsLane(Op, LaneValue); });  // NOLINT(misc-no-recursion)  // deliberate IR walk
  return false;
}

// The bare lane term, possibly through zext/sext/trunc (clang emits
// zext(tid.x) before the GEP index arithmetic).
bool isBareLaneTerm(const SCEV *S, const Value *LaneValue) {
  while (const auto *Ext = dyn_cast<SCEVZeroExtendExpr>(S)) {
    S = Ext->getOperand();
  }
  while (const auto *Ext = dyn_cast<SCEVSignExtendExpr>(S)) {
    S = Ext->getOperand();
  }
  while (const auto *T = dyn_cast<SCEVTruncateExpr>(S)) {
    S = T->getOperand();
  }
  if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
    return U->getValue() == LaneValue;
  }
  return false;
}

// A SCEV node with no lane inside and no data-dependent unknown inside —
// a warp-uniform address component (constants, args, globals, canonical
// loop AddRecs). Such components shift the whole warp by the same amount
// and never affect the per-lane stride.
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR/SCEV tree recursion
bool isKnownUniformSCEV(const SCEV *S) {
  if (isa<SCEVConstant>(S) || isa<SCEVCouldNotCompute>(S)) {
    return true;
  }
  if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
    return isUniformTerm(U->getValue());
  }
  return std::ranges::all_of(
      S->operands(),
      [&](const SCEV *Op) { return isKnownUniformSCEV(Op); });  // NOLINT(misc-no-recursion)  // deliberate IR walk
  return true;
}

// Collect the lane coefficient from one SCEV node: bytes per lane that the
// access address advances per tid unit. `ok` flips false when the SCEV is
// not affine in the lane (or depends on data), which classifies the access
// unknown-pattern.
  // NOLINTNEXTLINE(misc-no-recursion)  // deliberate IR/SCEV tree recursion
void collectLaneCoeff(const SCEV *S, const Value *LaneValue, int64_t &Coeff,
                      bool &Ok) {
  if (!Ok) {
    return;
  }
  if (isa<SCEVConstant>(S) || isa<SCEVCouldNotCompute>(S)) {
    return;
  }
  if (const auto *U = dyn_cast<SCEVUnknown>(S)) {
    if (U->getValue() == LaneValue) {
      Coeff += 1;  // a bare lane term: coefficient 1 (bytes)
      return;
    }
    if (!isUniformTerm(U->getValue())) {
      Ok = false;  // data-dependent term -> per-lane affine unresolvable
    }
    return;
  }
  if (const auto *Add = dyn_cast<SCEVAddExpr>(S)) {
    for (const SCEV *Op : Add->operands()) {
      collectLaneCoeff(Op, LaneValue, Coeff, Ok);
    }
    return;
  }
  if (const auto *Mul = dyn_cast<SCEVMulExpr>(S)) {
    // Affine-in-lane shape: the product must decompose into ONE factor
    // containing the lane (constant or affine-in-lane — nested AddExpr
    // factors distribute: (base + tid)*4 = 4*base + 4*tid) and constants.
    // Two lane-containing factors would be quadratic -> non-affine.
    // A uniform-but-non-constant factor beside the lane (e.g. a loop
    // AddRec or ctaid) makes the per-lane stride iteration/block-dependent
    // -> not a compile-time constant -> unknown.
    if (!mentionsLane(S, LaneValue)) {
      if (!isKnownUniformSCEV(S)) {
        Ok = false;  // data-dependent term
      }
      return;
    }
    int64_t Factor = 1;
    int LaneFactors = 0;
    for (const SCEV *Op : Mul->operands()) {
      if (const auto *C = dyn_cast<SCEVConstant>(Op)) {
        Factor *= C->getAPInt().getSExtValue();
        continue;
      }
      if (isBareLaneTerm(Op, LaneValue)) {
        ++LaneFactors;
        continue;
      }
      if (mentionsLane(Op, LaneValue)) {
        // Distribute: extract the lane coefficient of the nested factor.
        int64_t Inner = 0;
        collectLaneCoeff(Op, LaneValue, Inner, Ok);
        if (!Ok) {
          return;
        }
        Factor *= Inner;
        ++LaneFactors;
        continue;
      }
      // Uniform-but-non-constant factor beside a lane factor.
      Ok = false;
      return;
    }
    if (LaneFactors > 1) {
      Ok = false;  // quadratic in the lane
      return;
    }
    Coeff += Factor;
    return;
  }
  // zext/sext/trunc are coefficient-transparent: ext(base + c*lane) has
  // the same per-lane coefficient as its operand (the byte stride is
  // decided by the address arithmetic around the extension, not by the
  // extension itself). Recurse into the operand.
  if (const auto *Ext = dyn_cast<SCEVZeroExtendExpr>(S)) {
    collectLaneCoeff(Ext->getOperand(), LaneValue, Coeff, Ok);
    return;
  }
  if (const auto *Ext = dyn_cast<SCEVSignExtendExpr>(S)) {
    collectLaneCoeff(Ext->getOperand(), LaneValue, Coeff, Ok);
    return;
  }
  if (const auto *T = dyn_cast<SCEVTruncateExpr>(S)) {
    collectLaneCoeff(T->getOperand(), LaneValue, Coeff, Ok);
    return;
  }
  if (const auto *AR = dyn_cast<SCEVAddRecExpr>(S)) {
    // Loop IVs are warp-uniform (all lanes execute the same iteration), so
    // an AddRec contributes to the BASE, not the stride. A lane term in the
    // START is a plain per-lane offset (distribute it); a lane term in the
    // STEP would make the per-lane stride iteration-dependent -> not a
    // compile-time constant -> unknown. (clang-21 shape: the loop-carried
    // address is {base + tid,+,64}-style for row-strided inner loops.)
    if (AR->getNumOperands() < 2) {
      Ok = false;
      return;
    }
    const SCEV *Start = AR->getOperand(0);
    const SCEV *Step = AR->getOperand(1);
    if (mentionsLane(Start, LaneValue)) {
      collectLaneCoeff(Start, LaneValue, Coeff, Ok);
    } else if (!isKnownUniformSCEV(Start)) {
      Ok = false;
      return;
    }
    if (mentionsLane(Step, LaneValue)) {
      Ok = false;  // iteration-dependent per-lane stride
      return;
    }
    if (!isKnownUniformSCEV(Step)) {
      Ok = false;
    }
    return;
  }
  // Any other node (min/max, ...): lane inside -> the per-lane stride is
  // not a compile-time constant; no lane -> the node must still be free of
  // data-dependent unknowns, else the address is unresolvable.
  if (mentionsLane(S, LaneValue)) {
    Ok = false;
    return;
  }
  if (!isKnownUniformSCEV(S)) {
    Ok = false;
  }
}

constexpr int kSectorSizeBytes = 32;  // brief §3.1 (L218)
constexpr int kWarpSize = 32;         // NVIDIA warp (CPG Table 30)

}  // namespace

CoalescingClass classifyGlobalAccess(Value *Ptr, unsigned WidthBytes,
                                     ScalarEvolution &SE, Value *LaneValue) {
  CoalescingClass C;
  if (WidthBytes == 0) {
    WidthBytes = 4;
  }

  const SCEV *S = SE.getSCEV(Ptr);

  // No lane value in the function: every access is uniform across the warp.
  int64_t Coeff = 0;
  bool Ok = true;
  collectLaneCoeff(S, LaneValue, Coeff, Ok);
  if (!Ok) {
    // Per-lane affine unresolvable -> unknown-pattern (honesty rule).
    C.pattern = "unknown-pattern";
    C.cls = "unknown";
    return C;
  }

  if (Coeff == 0) {
    // Uniform / broadcast: all lanes hit the same address (analyzer's
    // coeff==0 case — analyze_coalescing.cpp classify_pattern). A single
    // span of WidthBytes, efficiency 1.0.
    C.sectors = static_cast<int>((WidthBytes + kSectorSizeBytes - 1) / kSectorSizeBytes);
    C.efficiency = 1.0;
    C.lane_stride_elements = 0;
    C.pattern = "contiguous-128B";
    C.cls = "fully";
    return C;
  }

  // Per-lane stride Coeff bytes: simulate one warp under the 32B-sector
  // model (brief §3.1, analyzer classify_pattern).
  std::set<int64_t> Sectors;
  for (int Lane = 0; Lane < kWarpSize; ++Lane) {
    const int64_t Base = Coeff * Lane;
    Sectors.insert(Base / kSectorSizeBytes);
    Sectors.insert((Base + WidthBytes - 1) / kSectorSizeBytes);
  }
  C.sectors = static_cast<int>(Sectors.size());
  const double Used = static_cast<double>(kWarpSize) * WidthBytes;
  const double Transferred = static_cast<double>(C.sectors) * kSectorSizeBytes;
  C.efficiency = Transferred > 0.0 ? Used / Transferred : 0.0;
  C.lane_stride_elements = 0;

  if (C.efficiency >= 1.0 - 1e-12) {
    C.pattern = "contiguous-128B";
    C.cls = "fully";
  } else if (C.efficiency >= 0.5) {
    C.pattern = "contiguous-partial";
    C.cls = "partial";
  } else {
    C.pattern = "strided";
    C.cls = "uncoalesced";
  }
  return C;
}

CoalesceStats analyzeFunction(Function &F, ScalarEvolution &SE) {
  CoalesceStats Stats;
  const DataLayout &DL = F.getParent()->getDataLayout();
  Value *LaneValue = findLaneValue(F);

  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      Value *Ptr = nullptr;
      Type *EltTy = nullptr;
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
      } else {
        continue;
      }
      if (Ptr->getType()->getPointerAddressSpace() != 1) {
        continue;
      }
      const unsigned Width = DL.getTypeStoreSize(EltTy);
      const unsigned ScalarWidth = DL.getTypeStoreSize(EltTy->getScalarType());
      ++Stats.total;
      Stats.maxWidthBytes = std::max(Stats.maxWidthBytes, Width);
      if (Width > ScalarWidth) {
        ++Stats.vectorized;
      }

      const CoalescingClass C =
          classifyGlobalAccess(Ptr, Width, SE, LaneValue);
      if (C.cls == "fully") {
        ++Stats.fully;
      } else if (C.cls == "partial") {
        ++Stats.partial;
      } else if (C.cls == "uncoalesced") {
        ++Stats.uncoalesced;
      } else {
        ++Stats.unknown;
      }
    }
  }
  return Stats;
}
// NOLINTNEXTLINE(readability-convert-member-functions-to-static)  // LLVM pass-framework run() entry point
llvm::PreservedAnalyses CoalescePass::run(llvm::Function &F,
                                          llvm::FunctionAnalysisManager &FAM) {
  ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
  CoalesceStats Stats = analyzeFunction(F, SE);

  if (CoalesceReportFlag) {
    errs() << "kernscope-coalesce: " << F.getName() << ": " << Stats.total
           << " global accesses (" << Stats.fully << " fully, "
           << Stats.partial << " partial, " << Stats.uncoalesced
           << " uncoalesced, " << Stats.unknown << " unknown); widest access "
           << Stats.maxWidthBytes << " B; vectorized " << Stats.vectorized
           << "\n";
  }
  // Analysis only — nothing invalidated.
  return PreservedAnalyses::all();
}

}  // namespace pass
}  // namespace kernscope

#ifndef KERNSCOPE_COALESCE_AS_LIB
namespace {

bool parseCoalescePass(llvm::StringRef Name, llvm::ModulePassManager &MPM,
                       llvm::ArrayRef<llvm::PassBuilder::PipelineElement> /*unused*/) {
  if (Name == "kernscope-coalesce") {
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
        kernscope::pass::CoalescePass()));
    return true;
  }
  // The canonicalization step of the todo-42 artifact pipeline (see
  // src/as1.cpp): restores direct AS1/AS3 accesses from clang-21's generic
  // flat-space lowering. Deliberately NOT on the EP callbacks below.
  if (Name == "kernscope-as1") {
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
        kernscope::pass::As1CanonPass()));
    return true;
  }
  return false;
}

}  // namespace

// Self-registration (todo-36 contract — register_pass.h): textual name for
// opt -passes=kernscope-coalesce plus the EP callbacks (analysis-only; it
// counts accesses and invalidates nothing, so it is harmless in default
// pipelines). Skipped when this TU is compiled into the unit test binary
// (KERNSCOPE_COALESCE_AS_LIB — the test drives classifyGlobalAccess /
// analyzeFunction directly).
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_PASS_CALLBACK([](llvm::PassBuilder &PB) {
  PB.registerPipelineParsingCallback(parseCoalescePass);

  PB.registerPipelineStartEPCallback(
      [](llvm::ModulePassManager &MPM, llvm::OptimizationLevel) {
        MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
            kernscope::pass::CoalescePass()));
      });
  PB.registerPeepholeEPCallback(
      [](llvm::FunctionPassManager &FPM, llvm::OptimizationLevel) {
        FPM.addPass(kernscope::pass::CoalescePass());
      });
});
#endif  // KERNSCOPE_COALESCE_AS_LIB
