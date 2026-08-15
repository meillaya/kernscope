// verify_core_test.cpp — kernscope-verify core pipeline tests (todo 30).
//
// Acceptance coverage (plan todo 30, all no-gpu — clang runs GPU-less):
//   1. constant-<<<32,256>>> kernel -> EXACT dim ranges visible in the
//      dumped verifier state (threadIdx.x in [0, 256), blockIdx.x in
//      [0, 32)) + tier UpperBound via __launch_bounds__;
//   2. the SAME kernel via --device-only -> "symbolic dims" degraded
//      verdict;
//   3. -O2 unrolled case: bounds reconstructed (bounded access count > 0)
//      AND the degraded SCEV-reconstruction verdict for an opaque index
//      (both paths);
//   4. budget-exceeded synthetic case terminates < 5s with the budget
//      diagnostic;
//   plus: check-plugin registry contract (REGISTER_CHECK), ingestion
//   failure (syntax error -> forwarded clang diagnostics, no crash), HIP
//   ingestion compile-gated.

#include <kernscope/verify/budget.h>
#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/launch_config.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/Program.h>

#include <chrono>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include <algorithm>

using namespace kernscope::verify;

namespace {

std::string fixture(const std::string& name) {
  return std::string(VERIFY_FIXTURES_DIR) + "/" + name;
}

// Runs the full pipeline on a fixture; skips nothing — the CUDA toolchain
// staging is the todo-5 gate (asserted present in the tests).
struct Analyzed {
  std::shared_ptr<llvm::LLVMContext> context;
  std::shared_ptr<llvm::Module> module;
  std::unique_ptr<VerifierState> state;
  std::string dump;
};

Analyzed run_analyze(const std::string& fixture_name, VerifierOptions vopts,
                     const IngestionOptions& iopts = IngestionOptions()) {
  ClangIngester ingester;
  IngestionResult result = ingester.ingest(fixture(fixture_name), iopts);
  Analyzed out;
  if (!result.success) {
    return out;  // caller asserts
  }
  out.context = result.context;
  out.module = std::move(result.module);
  out.state = std::make_unique<VerifierState>(out.module, out.context, vopts);
  out.state->analyze();
  out.dump = out.state->dumpState();
  return out;
}

const KernelInfo* find_kernel(const Analyzed& a, const std::string& needle) {
  for (const KernelInfo& k : a.state->kernels()) {
    if (k.name.find(needle) != std::string::npos) {
      return &k;
    }
  }
  return nullptr;
}

bool has_diagnostic(const Analyzed& a, int class_id) {
  return std::ranges::any_of(a.state->diagnostics(),
                             [&](const Diagnostic& d) { return d.class_id == class_id; });
}

// Registry-contract test plugin (passive; registered via REGISTER_CHECK so
// it lands without editing any shared file, exactly like todos 31-34).
class TestPlugin : public CheckPlugin {
 public:
  TestPlugin() {
    id = "test-plugin";
    name = "Test plugin (registry contract)";
    check_class_ids = {CheckClass::kGlobalOob};
  }
  std::vector<Diagnostic> run(VerifierContext& /*ctx*/) const override { return {}; }
};
// NOLINTNEXTLINE(misc-use-anonymous-namespace)  // macro-expanded plugin registry entry (must be namespace-scope static)
REGISTER_CHECK(TestPlugin);

}  // namespace

// --- 1. constant-<<<32,256>>> -> EXACT dims in the dumped state -----------
TEST(VerifyCore, ExactDimsVisibleInDump) {
  VerifierOptions vopts;
  Analyzed a = run_analyze("launch_const.cu", vopts);
  ASSERT_NE(a.state, nullptr) << "ingestion of launch_const.cu failed";
  const KernelInfo* k = find_kernel(a, "exact_kernel");
  ASSERT_NE(k, nullptr);
  ASSERT_EQ(k->config.tier, LaunchTier::Exact);
  EXPECT_EQ(k->config.block.x, 256u);
  EXPECT_EQ(k->config.block.y, 1u);
  EXPECT_EQ(k->config.grid.x, 32u);
  EXPECT_EQ(k->config.grid.y, 1u);
  EXPECT_EQ(k->config.origin.find("launch site"), 0u);
  EXPECT_EQ(k->range(Lane::ThreadX), llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, 256)));
  EXPECT_EQ(k->range(Lane::BlockX), llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, 32)));
  EXPECT_EQ(k->range(Lane::BlockDimX), llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 256), llvm::APInt(32, 257)));
  EXPECT_TRUE(k->rangeBounded(Lane::ThreadX));
  EXPECT_TRUE(k->rangeBounded(Lane::BlockX));
  // Acceptance: exact dim ranges visible in the dumped verifier state.
  EXPECT_NE(a.dump.find("threadIdx.x in [0,256)"), std::string::npos);
  EXPECT_NE(a.dump.find("blockIdx.x in [0,32)"), std::string::npos);
  EXPECT_NE(a.dump.find("blockDim.x in [256,257)"), std::string::npos);
  EXPECT_NE(a.dump.find("launch tier: Exact"), std::string::npos);
}

// --- 1b. __launch_bounds__ -> sound upper bound ----------------------------
TEST(VerifyCore, LaunchBoundsUpperBoundTier) {
  VerifierOptions vopts;
  Analyzed a = run_analyze("launch_const.cu", vopts);
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = find_kernel(a, "bound_kernel");
  ASSERT_NE(k, nullptr);
  ASSERT_EQ(k->config.tier, LaunchTier::UpperBound);
  EXPECT_EQ(k->config.block.x, 256u);
  EXPECT_EQ(k->config.origin.find("nvvm.maxntid"), 0u);
  // Sound upper bound: threadIdx.x in [0, 256) even with runtime launch dims.
  EXPECT_EQ(k->range(Lane::ThreadX), llvm::ConstantRange::getNonEmpty(llvm::APInt(32, 0), llvm::APInt(32, 256)));
  EXPECT_FALSE(k->rangeBounded(Lane::BlockX));  // grid unknown -> unbounded
}

// --- 1c. runtime dims -> tier (c) symbolic + degraded verdict --------------
TEST(VerifyCore, SymbolicDimsDegradedVerdict) {
  VerifierOptions vopts;
  Analyzed a = run_analyze("launch_const.cu", vopts);
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = find_kernel(a, "symbolic_kernel");
  ASSERT_NE(k, nullptr);
  ASSERT_EQ(k->config.tier, LaunchTier::Symbolic);
  EXPECT_FALSE(k->config.runtime_check_suggestion.empty());
  EXPECT_TRUE(has_diagnostic(a, EngineClass::kSymbolicDims));
}

// --- 2. --device-only: the SAME exact-launch kernel degrades to symbolic --
TEST(VerifyCore, DeviceOnlySymbolicDims) {
  IngestionOptions iopts;
  iopts.device_only = true;
  VerifierOptions vopts;
  Analyzed a = run_analyze("launch_const.cu", vopts, iopts);
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = find_kernel(a, "exact_kernel");
  ASSERT_NE(k, nullptr);
  ASSERT_EQ(k->config.tier, LaunchTier::Symbolic);  // degraded
  EXPECT_NE(a.dump.find("symbolic dims"), std::string::npos) << a.dump;
  EXPECT_TRUE(has_diagnostic(a, EngineClass::kSymbolicDims));
  // __launch_bounds__ survives device-only: still a sound upper bound.
  const KernelInfo* bound = find_kernel(a, "bound_kernel");
  ASSERT_NE(bound, nullptr);
  EXPECT_EQ(bound->config.tier, LaunchTier::UpperBound);
}

// --- 3. -O2: SCEV reconstruction AND degraded verdict (both paths) --------
TEST(VerifyCore, O2ReconstructionAndDegradedVerdict) {
  IngestionOptions iopts;
  iopts.opt_level = 2;
  VerifierOptions vopts;
  vopts.ingestion_opt_level = 2;
  Analyzed a = run_analyze("o2_unrolled.cu", vopts, iopts);
  ASSERT_NE(a.state, nullptr);
  // Path A: unrolled constant-offset index is bounded (reconstruction).
  EXPECT_GT(a.state->boundedAccessCount(), 0u);
  // Path B: opaque runtime index cannot be reconstructed -> degraded
  // verdict with a source location.
  EXPECT_TRUE(has_diagnostic(a, EngineClass::kScevReconstructionFailed));
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id == EngineClass::kScevReconstructionFailed) {
      EXPECT_EQ(d.severity, Severity::Warning);
    }
  }
}

// --- 3b. -O2: the SCEV/lattice-assisted queryRange helper reconstructs a
// bounded range for the unrolled (constant-offset) index -------------------
TEST(VerifyCore, O2ReconstructionViaQueryRange) {
  IngestionOptions iopts;
  iopts.opt_level = 2;
  VerifierOptions vopts;
  vopts.ingestion_opt_level = 2;
  Analyzed a = run_analyze("o2_unrolled.cu", vopts, iopts);
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = find_kernel(a, "reconstruct_kernel");
  ASSERT_NE(k, nullptr);
  llvm::GetElementPtrInst* gep = nullptr;
  for (llvm::BasicBlock& bb : *k->function) {
    for (llvm::Instruction& i : bb) {
      if (auto* g = llvm::dyn_cast<llvm::GetElementPtrInst>(&i)) {
        gep = g;
        break;
      }
    }
    if (gep) {
      break;
    }
  }
  ASSERT_NE(gep, nullptr);
  VerifierContext ctx{*a.module, *const_cast<KernelInfo*>(k), a.state->engine()};
  ctx.state = a.state.get();
  auto r = ctx.queryRange(gep);
  if (!r.has_value()) {
    FAIL() << "no range";
    return;
  }
  EXPECT_FALSE(r.value().isFullSet());
  EXPECT_FALSE(r.value().isEmptySet());
}

// --- 4. budget-exceeded synthetic case: < 5s + budget diagnostic ----------
TEST(VerifyCore, BudgetExceededTerminates) {
  auto context = std::make_shared<llvm::LLVMContext>();
  auto module = std::make_shared<llvm::Module>("budget_synth", *context);
  llvm::IRBuilder<> builder(*context);
  llvm::FunctionType* fty =
      llvm::FunctionType::get(builder.getVoidTy(), {builder.getInt32Ty()}, false);
  llvm::Function* f = llvm::Function::Create(
      fty, llvm::GlobalValue::ExternalLinkage, "synthetic_budget_kernel",
      module.get());
  f->addFnAttr("kernel");
  llvm::BasicBlock* entry = llvm::BasicBlock::Create(*context, "entry", f);
  std::vector<llvm::BasicBlock*> blocks;
  blocks.push_back(entry);
  for (int i = 0; i < 1000; ++i) {
    blocks.push_back(llvm::BasicBlock::Create(*context, "b" + std::to_string(i), f));
  }
  llvm::BasicBlock* exit = llvm::BasicBlock::Create(*context, "exit", f);
  builder.SetInsertPoint(entry);
  builder.CreateBr(blocks[1]);
  llvm::Value* acc = f->getArg(0);
  for (int i = 1; i <= 1000; ++i) {
    builder.SetInsertPoint(blocks[i]);
    for (int j = 0; j < 50; ++j) {
      acc = builder.CreateAdd(acc, builder.getInt32(j + 1));
    }
    builder.CreateBr(i < 1000 ? blocks[i + 1] : exit);
  }
  builder.SetInsertPoint(exit);
  builder.CreateRetVoid();

  VerifierOptions vopts;
  vopts.max_states_per_function = 4096;
  VerifierState state(module, context, vopts);
  const auto t0 = std::chrono::steady_clock::now();
  state.analyze();
  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed =
      std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
  EXPECT_LT(elapsed, 5.0);
  bool budget_diag = false;
  for (const Diagnostic& d : state.diagnostics()) {
    if (d.class_id == EngineClass::kBudgetExceeded) {
      budget_diag = true;
      EXPECT_NE(d.message.find("budget exceeded"), std::string::npos);
    }
  }
  EXPECT_TRUE(budget_diag);
}

// --- registry contract: REGISTER_CHECK without shared-file edits ----------
TEST(VerifyCore, CheckPluginRegistry) {
  bool found = false;
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    if (e.id == "test-plugin") {
      found = true;
      EXPECT_EQ(e.class_ids.size(), 1u);
      EXPECT_EQ(e.class_ids[0], CheckClass::kGlobalOob);
    }
  }
  EXPECT_TRUE(found);
  EXPECT_EQ(CheckRegistry::instance().pluginsForClass(CheckClass::kGlobalOob).size(),
            1u);
}

// --- QA negative: syntax error -> forwarded clang diag, no crash ----------
TEST(VerifyCore, IngestionFailureForwardsClangDiagnostics) {
  ClangIngester ingester;
  IngestionResult result = ingester.ingest(fixture("bad_syntax.cu"),
                                           IngestionOptions());
  EXPECT_FALSE(result.success);
  EXPECT_EQ(result.module, nullptr);
  ASSERT_FALSE(result.clang_diagnostics.empty());
  EXPECT_NE(result.clang_diagnostics[0].find("error"), std::string::npos);
}

// --- HIP path (compile-gated on the ROCm toolchain) ------------------------
TEST(VerifyCore, HipIngestionCompileGated) {
  if (!llvm::sys::findProgramByName("hipcc")) {
    GTEST_SKIP() << "hipcc not on PATH; HIP ingestion compile-gated off";
  }
  IngestionOptions iopts;
  iopts.language = SourceLanguage::Hip;
  VerifierOptions vopts;
  Analyzed a = run_analyze("hip_ingest.hip", vopts, iopts);
  if (!a.state) {
    GTEST_SKIP() << "HIP ingestion unusable on this host (compile-gated)";
  }
  const KernelInfo* k = find_kernel(a, "hip_ingest_kernel");
  ASSERT_NE(k, nullptr);
  EXPECT_EQ(k->config.tier, LaunchTier::Exact);
  EXPECT_EQ(k->config.block.x, 64u);
  EXPECT_EQ(k->config.grid.x, 2u);
}
