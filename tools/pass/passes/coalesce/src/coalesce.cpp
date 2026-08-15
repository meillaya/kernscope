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
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Passes/OptimizationLevel.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Passes/PassPlugin.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>

#include <kernscope/pass/coalesce.h>
#include <kernscope/pass/register_pass.h>

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

CoalescingClass classifyGlobalAccess(Value *Ptr, unsigned WidthBytes,
                                     ScalarEvolution &SE, Value *LaneValue) {
  // SKELETON: full SCEV classifier lands in the next commit. For now every
  // access is conservatively unknown-pattern (never claimed coalesced).
  (void)Ptr;
  (void)WidthBytes;
  (void)SE;
  (void)LaneValue;
  CoalescingClass C;
  C.pattern = "unknown-pattern";
  C.cls = "unknown";
  return C;
}

CoalesceStats analyzeFunction(Function &F, ScalarEvolution &SE) {
  (void)SE;  // consumed by the SCEV classifier (next commit)
  CoalesceStats Stats;
  const DataLayout &DL = F.getParent()->getDataLayout();
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
      ++Stats.total;
      Stats.maxWidthBytes = std::max(Stats.maxWidthBytes, Width);
      if (Width > DL.getTypeStoreSize(EltTy->getScalarType())) {
        ++Stats.vectorized;
      }
      // Skeleton: classification counts filled by the classifier commit.
    }
  }
  return Stats;
}

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
                       llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "kernscope-coalesce") {
    MPM.addPass(llvm::createModuleToFunctionPassAdaptor(
        kernscope::pass::CoalescePass()));
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
