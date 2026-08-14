// kernscope::pass — vectorize-rewrite-test (todo 37, Wave 7).
//
// Unit-level acceptance for the vectorized load/store rewrite pass: asserts
// the rewrite counts (the same VectorizeStats contract the todo-42 report
// consumes) on the three committed fixtures:
//
//   strided_contig_loop.ll — strided-contiguous loop: 4 consecutive float
//     loads + 4 consecutive float stores per iteration at a 16-float
//     stride; SCEV proves contiguity (offset diffs fold to 1) and 16-byte
//     alignment (head offset {0,+,16} has min trailing zeros 4, base arg
//     is align 16) -> one <4 x float> load + one <4 x float> store.
//
//   misaligned.ll          — same shape but 4-float stride: the head is
//     only provably 4-byte aligned -> no vector width is legal -> pass
//     must leave the IR untouched (counts all zero). The lit suite asserts
//     byte-identical output via llvm-diff on top of this.
//
//   addrspace3.ll          — identical contiguous + aligned chain in
//     addrspace(3) (shared memory): out of v1 scope -> untouched.
//
// The test runs the pass core directly (vectorizeFunction) with a
// PassBuilder-wired FunctionAnalysisManager, mirroring how the pass
// wrapper obtains ScalarEvolution + AAResults under opt.

#include <gtest/gtest.h>

#include <llvm/Analysis/AliasAnalysis.h>
#include <llvm/Analysis/LoopAnalysisManager.h>
#include <llvm/Analysis/ScalarEvolution.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/SourceMgr.h>

#include <memory>
#include <string>

#include <kernscope/pass/vectorize.h>

using namespace llvm;

namespace {

std::unique_ptr<Module> loadFixture(LLVMContext &Ctx,
                                    const std::string &FileName) {
  SMDiagnostic Err;
  const std::string Path = std::string(VEC_FIXTURES_DIR) + "/" + FileName;
  std::unique_ptr<Module> M = parseIRFile(Path, Err, Ctx);
  EXPECT_NE(M, nullptr) << "failed to parse fixture: " << Path << ": "
                        << Err.getMessage().str();
  return M;
}

kernscope::pass::VectorizeStats runVectorize(Module &M) {
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PassBuilder PB;
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  kernscope::pass::VectorizeStats Total;
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }
    ScalarEvolution &SE = FAM.getResult<ScalarEvolutionAnalysis>(F);
    AAResults &AA = FAM.getResult<AAManager>(F);
    kernscope::pass::VectorizeStats S =
        kernscope::pass::vectorizeFunction(F, SE, AA);
    Total.loadsVectorized += S.loadsVectorized;
    Total.storesVectorized += S.storesVectorized;
    Total.chainsRewritten += S.chainsRewritten;
  }
  return Total;
}

}  // namespace

TEST(VectorizeRewrite, StridedContiguousLoop) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = loadFixture(Ctx, "strided_contig_loop.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::VectorizeStats Stats = runVectorize(*M);

  EXPECT_EQ(Stats.loadsVectorized, 4u) << "4 scalar loads -> <4 x float>";
  EXPECT_EQ(Stats.storesVectorized, 4u) << "4 scalar stores -> <4 x float>";
  EXPECT_EQ(Stats.chainsRewritten, 2u) << "one load chain + one store chain";
}

TEST(VectorizeRewrite, MisalignedChainUntouched) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = loadFixture(Ctx, "misaligned.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::VectorizeStats Stats = runVectorize(*M);

  // 4-float stride -> head provably 4-byte aligned only -> no legal width.
  EXPECT_EQ(Stats.loadsVectorized, 0u);
  EXPECT_EQ(Stats.storesVectorized, 0u);
  EXPECT_EQ(Stats.chainsRewritten, 0u);
}

TEST(VectorizeRewrite, Addrspace3Untouched) {
  LLVMContext Ctx;
  std::unique_ptr<Module> M = loadFixture(Ctx, "addrspace3.ll");
  ASSERT_NE(M, nullptr);

  kernscope::pass::VectorizeStats Stats = runVectorize(*M);

  // Shared-memory chains are out of v1 scope (global-only).
  EXPECT_EQ(Stats.loadsVectorized, 0u);
  EXPECT_EQ(Stats.storesVectorized, 0u);
  EXPECT_EQ(Stats.chainsRewritten, 0u);
}
