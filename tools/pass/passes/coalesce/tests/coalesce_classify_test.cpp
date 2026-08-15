// kernscope::pass — coalescing classifier unit test (todo 42, Wave 7).
//
// Drives classifyGlobalAccess()/analyzeFunction() directly (the verify-track
// pattern: the module sources are linked into this binary, not the plugin —
// KERNSCOPE_COALESCE_AS_LIB skips the registrar). Fixtures live in
// tests/pass/Inputs/coalesce/ (COALESCE_FIXTURES_DIR).
//
// Skeleton stage: the classifier is not yet SCEV-aware, so every access is
// conservatively unknown-pattern (the honesty rule — never claim coalesced
// without proof). The full classification assertions land with the SCEV
// classifier commit.

#include <gtest/gtest.h>

#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>

#include <algorithm>
#include <memory>
#include <string>

#include <kernscope/pass/coalesce.h>

#ifndef COALESCE_FIXTURES_DIR
#error "COALESCE_FIXTURES_DIR must be defined (CMake)"
#endif

namespace {

std::unique_ptr<llvm::Module> loadModule(llvm::LLVMContext &Ctx,
                                      llvm::StringRef Name) {
  llvm::SMDiagnostic Err;
  const std::string Path =
      std::string(COALESCE_FIXTURES_DIR) + "/" + Name.str();
  std::unique_ptr<llvm::Module> M = llvm::parseIRFile(Path, Err, Ctx);
  EXPECT_NE(M, nullptr) << "failed to parse fixture: " << Path << ": "
                        << Err.getMessage().str();
  return M;
}

// Run analyzeFunction over every definition with a PassBuilder-wired
// FunctionAnalysisManager (mirrors how the pass wrapper under opt obtains
// ScalarEvolution).
kernscope::pass::CoalesceStats runAnalyze(llvm::Module &M) {
  llvm::LoopAnalysisManager LAM;
  llvm::FunctionAnalysisManager FAM;
  llvm::CGSCCAnalysisManager CGAM;
  llvm::ModuleAnalysisManager MAM;
  llvm::PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  kernscope::pass::CoalesceStats Total;
  for (llvm::Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    llvm::ScalarEvolution &SE = FAM.getResult<llvm::ScalarEvolutionAnalysis>(F);
    kernscope::pass::CoalesceStats S =
        kernscope::pass::analyzeFunction(F, SE);
    Total.total += S.total;
    Total.maxWidthBytes = std::max(Total.maxWidthBytes, S.maxWidthBytes);
  }
  return Total;
}

TEST(CoalesceClassify, SkeletonCountsFullyCoalescedFixture) {
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M = loadModule(Ctx, "fully_coalesced.ll");
  ASSERT_NE(M, nullptr);

  // Skeleton stage: the analyzer collects the raw counts (2 global ops,
  // widest access 4 B) before the SCEV classifier lands.
  kernscope::pass::CoalesceStats Stats = runAnalyze(*M);
  EXPECT_EQ(Stats.total, 2u);
  EXPECT_EQ(Stats.maxWidthBytes, 4u);
}

}  // namespace
