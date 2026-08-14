// barrier_test.cpp — kernscope-verify barrier check tests (todo 32).
//
// Acceptance coverage (all no-gpu; clang runs GPU-less, -emit-llvm only):
//   * manifest-driven verdicts (tests/verify/corpus/manifest.barrier.json,
//     asserted by the verify_barrier ctest, `ctest -R verify_barrier`):
//       - barrier_divergent.cu    divergent_kernel   -> DIVERGENT_BARRIER
//         (error, with a source location)
//       - barrier_uniform.cu      uniform_kernel     -> SAFE
//       - barrier_mismatch.cu     mismatch_kernel    -> BARRIER_MISMATCH
//         (warning)
//       - barrier_uniform_loop.cu uniform_loop_kernel -> SAFE
//   * QA negative (regression guard against over-reporting): a barrier
//     under a UNIFORM loop-carried condition must stay SAFE — asserted
//     explicitly here AND in the manifest entry above.
//   * plugin registration contract: id "barrier" with classes {3, 6}.
//
// Verdict semantics: expected_verdict DIVERGENT_BARRIER -> at least one
// class-6 (divergent barrier) Error finding with a valid source location;
// BARRIER_MISMATCH -> at least one class-3 (barrier correctness) Warning;
// SAFE -> no class-3/class-6 findings at all for that kernel.

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/Module.h>
#include <llvm/Support/JSON.h>

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace kernscope::verify;

namespace {

struct ManifestEntry {
  std::string file;
  std::string kernel;
  int check = 0;
  std::string verdict;
};

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::string text((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());
  return text;
}

std::vector<ManifestEntry> readManifest() {
  const std::string path = std::string(BARRIER_CORPUS_DIR) +
                           "/manifest.barrier.json";
  llvm::json::Value root = nullptr;
  {
    llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(readFile(path));
    EXPECT_TRUE(static_cast<bool>(parsed)) << "cannot parse " << path;
    if (!parsed) {
      return {};
    }
    root = std::move(*parsed);
  }
  std::vector<ManifestEntry> entries;
  const llvm::json::Array* array = root.getAsArray();
  EXPECT_TRUE(array != nullptr) << "manifest root must be a JSON array";
  if (array == nullptr) {
    return {};
  }
  for (const llvm::json::Value& item : *array) {
    const llvm::json::Object* obj = item.getAsObject();
    EXPECT_TRUE(obj != nullptr);
    if (obj == nullptr) {
      continue;
    }
    ManifestEntry e;
    e.file = obj->getString("file")->str();
    e.kernel = obj->getString("kernel")->str();
    if (const std::optional<double> check = obj->getNumber("check")) {
      e.check = static_cast<int>(*check);
    }
    e.verdict = obj->getString("expected_verdict")->str();
    entries.push_back(std::move(e));
  }
  return entries;
}

std::string fixture(const std::string& name) {
  return std::string(BARRIER_CORPUS_DIR) + "/" + name;
}

// Runs the full pipeline on one fixture (ingestion cached per file).
struct Analyzed {
  std::shared_ptr<llvm::LLVMContext> context;
  std::shared_ptr<llvm::Module> module;
  std::unique_ptr<VerifierState> state;
};

std::map<std::string, std::unique_ptr<Analyzed>> g_cache;

const Analyzed* runAnalyze(const std::string& file) {
  auto it = g_cache.find(file);
  if (it != g_cache.end()) {
    return it->second.get();
  }
  auto out = std::make_unique<Analyzed>();
  ClangIngester ingester;
  IngestionResult result = ingester.ingest(fixture(file), IngestionOptions());
  if (!result.success) {
    ADD_FAILURE() << "ingestion of " << file
                  << " failed (toolchain staging broken?)";
    return nullptr;
  }
  out->context = result.context;
  out->module = std::shared_ptr<llvm::Module>(std::move(result.module));
  out->state = std::make_unique<VerifierState>(out->module, out->context,
                                               VerifierOptions());
  out->state->analyze();
  const Analyzed* ref = out.get();
  g_cache[file] = std::move(out);
  return ref;
}

const KernelInfo* findKernel(const Analyzed& a, const std::string& needle) {
  for (const KernelInfo& k : a.state->kernels()) {
    if (k.name.find(needle) != std::string::npos) {
      return &k;
    }
  }
  return nullptr;
}

// Diagnostics of `class_id` whose function contains `kernel` (demangled).
std::vector<const Diagnostic*> findingsFor(const Analyzed& a, int class_id,
                                           const std::string& kernel) {
  std::vector<const Diagnostic*> out;
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id == class_id &&
        d.function.find(kernel) != std::string::npos) {
      out.push_back(&d);
    }
  }
  return out;
}

// Asserts one manifest entry against the analyzed fixture.
void assertVerdict(const ManifestEntry& e, const Analyzed& a) {
  const KernelInfo* k = findKernel(a, e.kernel);
  ASSERT_TRUE(k != nullptr) << "kernel " << e.kernel << " not found in "
                            << e.file;
  if (e.verdict == "DIVERGENT_BARRIER") {
    const auto found = findingsFor(a, CheckClass::kDivergentBarrier, e.kernel);
    EXPECT_FALSE(found.empty()) << e.file << ": expected DIVERGENT_BARRIER";
    for (const Diagnostic* d : found) {
      EXPECT_EQ(d->severity, Severity::Error);
      EXPECT_TRUE(d->loc.valid()) << e.file << ": verdict needs a source loc";
    }
  } else if (e.verdict == "BARRIER_MISMATCH") {
    const auto found = findingsFor(a, CheckClass::kBarrier, e.kernel);
    EXPECT_FALSE(found.empty()) << e.file << ": expected BARRIER_MISMATCH";
    for (const Diagnostic* d : found) {
      EXPECT_EQ(d->severity, Severity::Warning);
    }
  } else if (e.verdict == "SAFE") {
    EXPECT_TRUE(findingsFor(a, CheckClass::kBarrier, e.kernel).empty())
        << e.file << "/" << e.kernel << ": expected SAFE (class 3)";
    EXPECT_TRUE(
        findingsFor(a, CheckClass::kDivergentBarrier, e.kernel).empty())
        << e.file << "/" << e.kernel << ": expected SAFE (class 6)";
  } else {
    FAIL() << "unknown expected_verdict " << e.verdict;
  }
}

}  // namespace

// --- acceptance: manifest-driven verdicts (ctest -R verify_barrier) ------
TEST(BarrierCheck, ManifestVerdicts) {
  const std::vector<ManifestEntry> entries = readManifest();
  ASSERT_EQ(entries.size(), 4u) << "manifest.barrier.json must carry the "
                                   "four todo-32 acceptance cases";
  for (const ManifestEntry& e : entries) {
    const Analyzed* a = runAnalyze(e.file);
    ASSERT_TRUE(a != nullptr) << "ingestion failed for " << e.file;
    assertVerdict(e, *a);
  }
}

// --- QA negative: uniform loop-carried condition stays SAFE ---------------
// Regression guard against over-reporting (the classic false-positive
// trap): the barrier under `if (j == 2)` inside `for (j = 0; j < 4; ++j)`
// must produce neither a class-3 warning nor a class-6 error, and the
// post-loop barrier likewise. Asserted in detail here (the manifest entry
// asserts the same verdict more compactly).
TEST(BarrierCheck, UniformLoopCarriedConditionStaysSafe) {
  const Analyzed* ap = runAnalyze("barrier_uniform_loop.cu");
  ASSERT_TRUE(ap != nullptr);
  const Analyzed& a = *ap;
  EXPECT_TRUE(findingsFor(a, CheckClass::kBarrier, "uniform_loop_kernel")
                  .empty());
  EXPECT_TRUE(
      findingsFor(a, CheckClass::kDivergentBarrier, "uniform_loop_kernel")
          .empty());
}

// --- QA negative: uniform blockIdx condition stays SAFE -------------------
TEST(BarrierCheck, UniformBlockIdxConditionStaysSafe) {
  const Analyzed* ap = runAnalyze("barrier_uniform.cu");
  ASSERT_TRUE(ap != nullptr);
  const Analyzed& a = *ap;
  EXPECT_TRUE(findingsFor(a, CheckClass::kBarrier, "uniform_kernel").empty());
  EXPECT_TRUE(
      findingsFor(a, CheckClass::kDivergentBarrier, "uniform_kernel").empty());
}

// --- acceptance detail: DIVERGENT_BARRIER carries a source location -------
TEST(BarrierCheck, DivergentBarrierHasSourceLocation) {
  const Analyzed* ap = runAnalyze("barrier_divergent.cu");
  ASSERT_TRUE(ap != nullptr);
  const Analyzed& a = *ap;
  const auto found =
      findingsFor(a, CheckClass::kDivergentBarrier, "divergent_kernel");
  ASSERT_FALSE(found.empty());
  for (const Diagnostic* d : found) {
    EXPECT_TRUE(d->loc.valid());
    EXPECT_FALSE(d->loc.file.empty());
    EXPECT_NE(d->loc.line, 0u);
    EXPECT_NE(d->message.find("CUDA C Programming Guide B.6"),
              std::string::npos)
        << "UB citation must be part of the diagnostic text";
  }
}

// --- registry contract -----------------------------------------------------
TEST(BarrierCheck, PluginRegisteredWithClasses36) {
  bool found = false;
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    if (e.id == "barrier") {
      found = true;
      EXPECT_EQ(e.class_ids.size(), 2u);
      EXPECT_EQ(e.class_ids[0], CheckClass::kBarrier);
      EXPECT_EQ(e.class_ids[1], CheckClass::kDivergentBarrier);
    }
  }
  EXPECT_TRUE(found) << "barrier plugin not registered (REGISTER_CHECK)";
  EXPECT_EQ(CheckRegistry::instance().pluginsForClass(CheckClass::kBarrier)
                .size(),
            1u);
  EXPECT_EQ(CheckRegistry::instance()
                .pluginsForClass(CheckClass::kDivergentBarrier)
                .size(),
            1u);
}
