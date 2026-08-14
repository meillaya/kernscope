// verify_ptr_test.cpp — corpus-backed acceptance tests for the ptr check
// module (todo 34): check classes 5 (illegal pointer arithmetic) + 7
// (uninitialized reads) + 8 (invalid address-space casts).
//
// SKELETON COMMIT: registry contract + ingestion smoke test only. The
// manifest-driven verdict tests land with the class implementations
// (5 -> 8 -> 7), each with corpus fixtures + manifest.ptr.json entries.

#include "ptr_common.h"

#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/Module.h>

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace kernscope::verify;

namespace {

std::string corpusDir() { return std::string(PTR_CORPUS_DIR); }

struct Analyzed {
  std::shared_ptr<llvm::LLVMContext> context;
  std::shared_ptr<llvm::Module> module;
  std::unique_ptr<VerifierState> state;
};

Analyzed runAnalyze(const std::string& file, IngestionOptions iopts) {
  ClangIngester ingester;
  IngestionResult result = ingester.ingest(corpusDir() + "/" + file, iopts);
  Analyzed out;
  if (!result.success) {
    return out;
  }
  out.context = result.context;
  out.module = std::move(result.module);
  out.state = std::make_unique<VerifierState>(out.module, out.context,
                                              VerifierOptions());
  out.state->analyze();
  return out;
}

}  // namespace

// --- registry contract -----------------------------------------------------
TEST(VerifyPtr, PluginRegisteredWithClasses578) {
  bool found = false;
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    if (e.id == "ptr") {
      found = true;
      EXPECT_EQ(e.class_ids.size(), 3u);
      EXPECT_EQ(e.class_ids[0], CheckClass::kPtrArith);
      EXPECT_EQ(e.class_ids[1], CheckClass::kUninitRead);
      EXPECT_EQ(e.class_ids[2], CheckClass::kAddrSpaceCast);
    }
  }
  EXPECT_TRUE(found) << "ptr plugin not registered (REGISTER_CHECK)";
  EXPECT_EQ(CheckRegistry::instance().pluginsForClass(CheckClass::kPtrArith)
                .size(),
            1u);
  EXPECT_EQ(CheckRegistry::instance().pluginsForClass(CheckClass::kUninitRead)
                .size(),
            1u);
  EXPECT_EQ(
      CheckRegistry::instance().pluginsForClass(CheckClass::kAddrSpaceCast)
          .size(),
      1u);
}

// --- ingestion smoke: the clean fixture must ingest and analyze ------------
TEST(VerifyPtr, CleanFixtureIngests) {
  const Analyzed a = runAnalyze("ptr_clean.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr) << "ingestion of ptr_clean.cu failed";
  bool found = false;
  for (const KernelInfo& k : a.state->kernels()) {
    if (k.name.find("ptr_clean_kernel") != std::string::npos) {
      found = true;
    }
  }
  EXPECT_TRUE(found) << "ptr_clean_kernel not found";
}
