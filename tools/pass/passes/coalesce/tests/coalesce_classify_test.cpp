// kernscope::pass — coalescing classifier unit test (todo 42, Wave 7).
//
// Drives classifyGlobalAccess()/analyzeFunction() directly (the verify-track
// pattern: the module sources are linked into this binary, not the plugin —
// KERNSCOPE_COALESCE_AS_LIB skips the registrar). Fixtures live in
// tests/pass/Inputs/coalesce/ (COALESCE_FIXTURES_DIR).
//
// Acceptance-relevant assertions:
//   * fully_coalesced.ll — tid-stride-1 float load+store -> contiguous-128B
//     (efficiency 1.0, 4 sectors — the brief §3.1 fully-coalesced golden,
//     same exact counts the todo-15 analyzer asserts at PTX level).
//   * strided.ll        — tid-stride-8 float -> strided (32 sectors,
//     12.5% efficiency — the todo-15 32-sector golden).
//   * unknown.ll        — indirect (loaded) index -> unknown-pattern, never
//     claimed coalesced (the analyzer's coalescing_unknown honesty rule).
//     The gather-load itself stays affine (contiguous-128B).

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
    Total.fully += S.fully;
    Total.partial += S.partial;
    Total.uncoalesced += S.uncoalesced;
    Total.unknown += S.unknown;
    Total.vectorized += S.vectorized;
    Total.maxWidthBytes = std::max(Total.maxWidthBytes, S.maxWidthBytes);
  }
  return Total;
}

TEST(CoalesceClassify, FullyCoalescedStride1) {
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M = loadModule(Ctx, "fully_coalesced.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::CoalesceStats Stats = runAnalyze(*M);
  // 2 float ops (load + store), both contiguous-128B: 32 lanes x 4 B = 128 B
  // = exactly 4 sectors of 32 B -> efficiency 1.0 (brief §3.1 golden).
  EXPECT_EQ(Stats.total, 2u);
  EXPECT_EQ(Stats.fully, 2u);
  EXPECT_EQ(Stats.partial, 0u);
  EXPECT_EQ(Stats.uncoalesced, 0u);
  EXPECT_EQ(Stats.unknown, 0u);
  EXPECT_EQ(Stats.maxWidthBytes, 4u);
}

TEST(CoalesceClassify, Strided8IsUncoalesced) {
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M = loadModule(Ctx, "strided.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::CoalesceStats Stats = runAnalyze(*M);
  // tid-stride-8 float: per-lane stride 32 B -> 32 sectors per warp for
  // 128 B of useful data (the todo-15 32-sector golden) -> 12.5% efficiency.
  EXPECT_EQ(Stats.total, 2u);
  EXPECT_EQ(Stats.fully, 0u);
  EXPECT_EQ(Stats.uncoalesced, 2u);
  EXPECT_EQ(Stats.unknown, 0u);
}

TEST(CoalesceClassify, IndirectIndexIsUnknown) {
  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M = loadModule(Ctx, "unknown.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::CoalesceStats Stats = runAnalyze(*M);
  // The gather load (%idx[tid]) is affine -> fully; the two a[%j] accesses
  // through a LOADED index are data-dependent -> unknown-pattern. The
  // honesty rule: unresolvable accesses are NEVER claimed coalesced.
  EXPECT_EQ(Stats.total, 3u);
  EXPECT_EQ(Stats.fully, 1u);
  EXPECT_EQ(Stats.unknown, 2u);
  EXPECT_EQ(Stats.uncoalesced, 0u);
}

}  // namespace
