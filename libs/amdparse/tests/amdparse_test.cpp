//===-- amdparse_test.cpp - AMDGPU ISA parser unit tests -----------------===//
//
// todo-10 acceptance coverage (gtest, LABELS no-gpu):
//   1. hipcc-generated layernorm gfx1101 asm (tests/amdparse/corpus/
//      layernorm_gfx1101.s): ds_* count > 0, global_load_* count > 0,
//      CFG with >= 1 branch.
//   2. v_wmma fixture -> tensor-class tag (WMMA), all six RDNA3 16x16x16
//      variants.
//   3. v_mfma fixture (gfx90a) -> tensor-class tag (MFMA).
//   4. unknown mnemonic -> AmdParseError naming the line + target triple
//      (negative QA: parse must fail, diagnostic carries line + triple).
//   5. taxonomy spot-checks: LDS / scratch (spill class) / buffer / global /
//      SALU / branch classification.
//
// Corpus generation (committed, reproducible): compile the golden HIP app
// with hipcc --offload-arch=gfx1101 -S (see tests/amdparse/corpus/README).

#include "kernscope/amd.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

namespace {

using namespace kernscope;

std::string read_file(const std::string &path) {
  std::ifstream In(path);
  EXPECT_TRUE(In.good()) << "missing fixture: " << path;
  std::ostringstream Ss;
  Ss << In.rdbuf();
  return Ss.str();
}

// Path to tests/amdparse/corpus/ relative to the test executable's build dir
// (build/libs/amdparse/... -> two levels up to the repo root).
const std::string kCorpus = TEST_CORPUS_DIR;

TEST(AmdParse, LayernormGfx1101ParseAndCounts) {
  const std::string Asm = read_file((kCorpus + "/layernorm_gfx1101.s"));
  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");

  // Acceptance: ds_* > 0 (LDS traffic; layernorm block reduction runs through
  // shared memory), global_load_* > 0, CFG with >= 1 branch.
  const std::size_t DsCount = Mod.count_class(InstrClass::LDS);
  EXPECT_GT(DsCount, 0u) << "expected ds_* (LDS) instructions in layernorm";
  EXPECT_GT(Mod.count_class(InstrClass::Global), 0u)
      << "expected global_load_* instructions in layernorm";

  std::size_t BranchCount = 0;
  for (const AmdInstruction &I : Mod.instructions)
    if (I.klass == InstrClass::Branch)
      ++BranchCount;
  EXPECT_GT(BranchCount, 0u) << "expected s_branch/s_cbranch instructions";

  bool FoundCfgBranch = false;
  for (const AmdBlock &B : Mod.blocks)
    if (!B.successors.empty())
      FoundCfgBranch = true;
  EXPECT_TRUE(FoundCfgBranch) << "CFG must contain a branch edge";
  EXPECT_FALSE(Mod.blocks.empty());

  // The parse must carry the right target triple.
  EXPECT_EQ(Mod.target_triple, "amdgcn-amd-amdhsa--gfx1101");
}

TEST(AmdParse, LayernormGfx1101GlobalLoadCount) {
  const std::string Asm = read_file((kCorpus + "/layernorm_gfx1101.s"));
  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");

  std::size_t GlobalLoads = 0;
  for (const AmdInstruction &I : Mod.instructions)
    if (I.klass == InstrClass::Global && I.is_load)
      ++GlobalLoads;
  EXPECT_GT(GlobalLoads, 0u) << "expected global_load_* in layernorm";
}

TEST(AmdParse, WmmaFixtureTensorClass) {
  const std::string Asm = read_file((kCorpus + "/wmma_fixture.s"));
  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");

  EXPECT_TRUE(Mod.has_tensor_instruction());
  EXPECT_EQ(Mod.count_class(InstrClass::WMMA), 6u);
  // Every wmma mnemonic must carry the tensor flag.
  for (const AmdInstruction &I : Mod.instructions)
    if (I.klass == InstrClass::WMMA)
      EXPECT_TRUE(I.is_tensor);
  // All six RDNA3 16x16x16 variants present.
  const auto M = Mod.mnemonics(InstrClass::WMMA);
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_f32_16x16x16_f16"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_f32_16x16x16_bf16"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_f16_16x16x16_f16"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_bf16_16x16x16_bf16"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_i32_16x16x16_iu8"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(),
                      "v_wmma_i32_16x16x16_iu4"),
            M.end());
}

TEST(AmdParse, MfmaFixtureTensorClass) {
  const std::string Asm = read_file((kCorpus + "/mfma_fixture.s"));
  const AmdModule Mod = parse_amd_module(Asm, "gfx90a");

  EXPECT_TRUE(Mod.has_tensor_instruction());
  EXPECT_EQ(Mod.count_class(InstrClass::MFMA), 4u);
  for (const AmdInstruction &I : Mod.instructions)
    if (I.klass == InstrClass::MFMA)
      EXPECT_TRUE(I.is_tensor);
  const auto M = Mod.mnemonics(InstrClass::MFMA);
  EXPECT_NE(std::find(M.begin(), M.end(), "v_mfma_f32_32x32x8f16"),
            M.end());
  EXPECT_NE(std::find(M.begin(), M.end(), "v_mfma_f32_16x16x16f16"),
            M.end());
}

TEST(AmdParse, UnknownMnemonicThrowsWithLineAndTriple) {
  const std::string Asm =
      ".text\n"
      "  s_waitcnt vmcnt(0)\n"
      "  v_frobnicate_bogus v0, v1\n"  // unknown mnemonic, line 3
      "  s_endpgm\n";
  EXPECT_THROW(parse_amd_module(Asm, "gfx1101", "<mem>"), AmdParseError);
  try {
    parse_amd_module(Asm, "gfx1101", "<mem>");
    FAIL() << "expected AmdParseError";
  } catch (const AmdParseError &E) {
    EXPECT_EQ(E.line, 3u) << "diagnostic must name the offending line";
    EXPECT_EQ(E.target_triple, "amdgcn-amd-amdhsa--gfx1101");
    // The what() message should also mention line + triple.
    EXPECT_NE(std::string(E.what()).find("line 3"), std::string::npos);
    EXPECT_NE(std::string(E.what()).find("gfx1101"), std::string::npos);
  }
}

TEST(AmdParse, FullTripleTargetAccepted) {
  const std::string Asm = read_file((kCorpus + "/layernorm_gfx1101.s"));
  // Passing the full triple (as .amdgcn_target emits it) must also work.
  const AmdModule Mod =
      parse_amd_module(Asm, "amdgcn-amd-amdhsa--gfx1101");
  EXPECT_EQ(Mod.target_triple, "amdgcn-amd-amdhsa--gfx1101");
  EXPECT_GT(Mod.count_class(InstrClass::LDS), 0u);
}

TEST(AmdParse, ScratchIsSpillClass) {
  // scratch_load_*/scratch_store_* = SPILL class (brief §6). Hand-written
  // scratch ops use the gfx9+ scratch mnemonic with a vector base+offset.
  const std::string Asm =
      ".text\n"
      "  scratch_load_b32 v0, v1, off\n"
      "  scratch_store_b32 v2, v3, off\n"
      "  s_endpgm\n";
  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");
  EXPECT_EQ(Mod.count_class(InstrClass::Scratch), 2u);
  for (const AmdInstruction &I : Mod.instructions)
    if (I.klass == InstrClass::Scratch)
      EXPECT_TRUE(I.is_scratch);
}

TEST(AmdParse, TaxonomyClasses) {
  const std::string Asm =
      ".text\n"
      "  v_mov_b32_e32 v0, 0\n"                  // VALU
      "  s_mov_b32 s0, exec_lo\n"                // SALU
      "  ds_read_b32 v1, v2\n"                   // LDS
      "  global_load_u16 v3, v[0:1], off\n"      // Global
      "  buffer_load_b32 v4, off, s[0:3], s4\n"  // Buffer
      "  flat_load_b32 v5, v[6:7]\n"             // Flat
      "  s_branch .Ltarget\n"                    // Branch (unconditional)
      ".Ltarget:\n"
      "  s_cbranch_execz .Ltarget\n"             // Branch (conditional)
      "  s_barrier\n"                            // Barrier
      "  s_waitcnt vmcnt(0)\n"                   // Wait
      "  s_endpgm\n";                            // EndPgm

  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");
  EXPECT_EQ(Mod.count_class(InstrClass::VALU), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::SALU), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::LDS), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Global), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Buffer), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Flat), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Branch), 2u);
  EXPECT_EQ(Mod.count_class(InstrClass::Barrier), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Wait), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::EndPgm), 1u);
  EXPECT_EQ(Mod.count_class(InstrClass::Unknown), 0u);
}

TEST(AmdParse, CfgEdgesAndFallthrough) {
  const std::string Asm =
      ".text\n"
      "start:\n"
      "  v_mov_b32_e32 v0, 0\n"
      "  s_cbranch_execz .Ldone\n"  // conditional: .Ldone + fallthrough
      "  v_add_f32_e32 v0, 1.0, v0\n"
      "  s_branch .Ldone\n"         // unconditional: .Ldone only
      ".Ldone:\n"
      "  s_endpgm\n";
  const AmdModule Mod = parse_amd_module(Asm, "gfx1101");

  const AmdBlock *Start = nullptr;
  const AmdBlock *Done = nullptr;
  for (const AmdBlock &B : Mod.blocks) {
    if (B.name == "start")
      Start = &B;
    if (B.name == ".Ldone")
      Done = &B;
  }
  ASSERT_NE(Start, nullptr);
  ASSERT_NE(Done, nullptr);
  // start: conditional branch -> two successors (.Ldone + fall-through bb).
  ASSERT_EQ(Start->successors.size(), 2u);
  EXPECT_EQ(Start->successors[0], ".Ldone");
  EXPECT_TRUE(Start->has_fallthrough);
  // .Ldone is the s_endpgm block: no successors.
  EXPECT_TRUE(Done->successors.empty());
  EXPECT_FALSE(Done->has_fallthrough);
}

} // namespace
