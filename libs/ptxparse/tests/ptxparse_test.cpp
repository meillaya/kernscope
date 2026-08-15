//===- ptxparse_test.cpp - PTX parser gtest suite ---------------*- C++ -*-===//
//
// Wave-2 todo 9 acceptance:
//   * every corpus fixture (tests/ptxparse/corpus/) parses and its kernel
//     statistics match manifest.json EXACTLY (instruction count, CFG edge
//     count, bar.sync, per-state-space access counts, tensor class tags;
//     the hand-written tcgen05 fixture carries parse_only: true);
//   * the real-world fixture tests/fixtures/fused_attention.ptx parses
//     cleanly (instruction_count > 0, bar.sync >= 1, shared >= 1);
//   * malformed input throws PtxParseError with 1-based line:col and never
//     crashes (no silent partial parses).
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kernscope/db.h"
#include "kernscope/ptx.h"

namespace {

std::string read_file(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return "";
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

std::string corpus_path(const std::string &name) {
  return std::string(TEST_CORPUS_DIR) + "/" + name;
}

// ---------------------------------------------------------------------------
// Manifest model (mirrors tests/ptxparse/corpus/manifest.json)
// ---------------------------------------------------------------------------

struct KernelExpect {
  std::string name;
  bool entry = false;
  std::int64_t instruction_count = -1;
  std::int64_t cfg_edge_count = -1;
  std::int64_t bar_sync_count = -1;
  std::int64_t shared_access_count = -1;
  std::int64_t global_access_count = -1;
  std::int64_t local_access_count = -1;
  std::int64_t const_access_count = -1;
  std::int64_t tensor_ops = -1;
  std::vector<std::string> tensor_classes;
};

struct FixtureExpect {
  std::string file;
  bool parse_only = false;
  std::vector<KernelExpect> kernels;
};

std::int64_t json_int(const kernscope::db::json::Value *v, std::int64_t dflt) {  if (v == nullptr) return dflt;
  return static_cast<std::int64_t>(v->as_number());
}

std::vector<FixtureExpect> load_manifest() {
  const std::string text = read_file(corpus_path("manifest.json"));
  EXPECT_FALSE(text.empty()) << "manifest.json not found under " << TEST_CORPUS_DIR;

  std::vector<FixtureExpect> out;
  std::string error;
  auto parsed = kernscope::db::json::Value::parse(text, error);
  if (!parsed) {
    EXPECT_TRUE(false) << "manifest.json is not valid JSON: " << error;
    return out;
  }
  const kernscope::db::json::Value *fixtures = parsed->find("fixtures");
  if (fixtures == nullptr) return out;
  for (const auto &f : fixtures->as_array()) {
    FixtureExpect fe;
    if (const auto *p = f.find("file")) fe.file = p->as_string();
    if (const auto *p = f.find("parse_only")) fe.parse_only = p->as_bool();
    const kernscope::db::json::Value *kernels = f.find("kernels");
    if (kernels != nullptr) {
      for (const auto &k : kernels->as_array()) {
        KernelExpect ke;
        if (const auto *p = k.find("name")) ke.name = p->as_string();
        if (const auto *p = k.find("entry")) ke.entry = p->as_bool();
        ke.instruction_count = json_int(k.find("instruction_count"), -1);
        ke.cfg_edge_count = json_int(k.find("cfg_edge_count"), -1);
        ke.bar_sync_count = json_int(k.find("bar_sync_count"), -1);
        ke.shared_access_count = json_int(k.find("shared_access_count"), -1);
        ke.global_access_count = json_int(k.find("global_access_count"), -1);
        ke.local_access_count = json_int(k.find("local_access_count"), -1);
        ke.const_access_count = json_int(k.find("const_access_count"), -1);
        ke.tensor_ops = json_int(k.find("tensor_ops"), -1);
        if (const auto *p = k.find("tensor_classes")) {
          for (const auto &c : p->as_array())
            ke.tensor_classes.push_back(c.as_string());
        }
        fe.kernels.push_back(std::move(ke));
      }
    }
    out.push_back(std::move(fe));
  }
  return out;
}

std::vector<std::string> corpus_fixture_files() {
  std::vector<std::string> files;
  for (const auto &fe : load_manifest()) files.push_back(fe.file);
  return files;
}

const FixtureExpect *find_fixture(const std::vector<FixtureExpect> &manifest,
                                  const std::string &file) {
  for (const auto &fe : manifest)
    if (fe.file == file) return &fe;
  return nullptr;
}

void assert_kernel_matches(const kernscope::PtxKernel &k,
                           const KernelExpect &ke) {
  EXPECT_EQ(ke.name, k.name);
  EXPECT_EQ(ke.entry, k.entry_point);
  EXPECT_EQ(ke.instruction_count, k.instruction_count);
  EXPECT_EQ(ke.cfg_edge_count, k.cfg_edge_count);
  EXPECT_EQ(ke.bar_sync_count, k.bar_sync_count);
  EXPECT_EQ(ke.shared_access_count, k.shared_access_count);
  EXPECT_EQ(ke.global_access_count, k.global_access_count);
  EXPECT_EQ(ke.local_access_count, k.local_access_count);
  EXPECT_EQ(ke.const_access_count, k.const_access_count);

  EXPECT_EQ(ke.tensor_ops,
            static_cast<std::int64_t>(k.tensor_ops.size()));
  std::vector<std::string> got_classes;
  got_classes.reserve(k.tensor_ops.size());
  for (const auto &op : k.tensor_ops) got_classes.push_back(op.class_tag);
  std::vector<std::string> want_classes = ke.tensor_classes;
  std::sort(got_classes.begin(), got_classes.end());
  std::sort(want_classes.begin(), want_classes.end());
  EXPECT_EQ(want_classes, got_classes);
}

}  // namespace

// ---------------------------------------------------------------------------
// Corpus fixtures: exact per-manifest assertions
// ---------------------------------------------------------------------------

class PtxCorpusTest : public ::testing::TestWithParam<std::string> {};

INSTANTIATE_TEST_SUITE_P(Ptxparse, PtxCorpusTest,
                         ::testing::ValuesIn(corpus_fixture_files()),
                         [](const ::testing::TestParamInfo<std::string> &info) {
                           std::string name = info.param;
                           std::replace(name.begin(), name.end(), '.', '_');
                           return name;
                         });

TEST_P(PtxCorpusTest, MatchesManifest) {
  const std::string& file = GetParam();
  const auto manifest = load_manifest();
  const FixtureExpect *fe = find_fixture(manifest, file);
  ASSERT_NE(fe, nullptr) << "no manifest entry for " << file;

  const std::string text = read_file(corpus_path(file));
  ASSERT_FALSE(text.empty()) << "fixture missing: " << corpus_path(file);

  kernscope::PtxModule mod;
  ASSERT_NO_THROW({ mod = kernscope::parse_ptx_module(text); })
      << "fixture failed to parse: " << file;

  ASSERT_EQ(fe->kernels.size(), mod.kernels().size())
      << "kernel count mismatch for " << file;
  for (const auto &ke : fe->kernels) {
    const kernscope::PtxKernel *k = nullptr;
    for (const auto &cand : mod.kernels()) {
      if (cand.name == ke.name) {
        k = &cand;
        break;
      }
    }
    ASSERT_NE(k, nullptr) << "kernel '" << ke.name << "' missing in " << file;
    assert_kernel_matches(*k, ke);
  }

  // The parse_only annotation: the hand-written tcgen05 fixture is declared
  // parse-only in the manifest AND every tcgen05-family tensor op the parser
  // emits must carry parse_only == true (nvcc 12.6 cannot target sm_100, so
  // tcgen05 input is by definition hand-written).
  for (const auto &k : mod.kernels()) {
    for (const auto &op : k.tensor_ops) {
      EXPECT_EQ(fe->parse_only, op.parse_only)
          << "tensor op parse_only mismatch in " << file << ": " << op.mnemonic;
    }
  }
}

// ---------------------------------------------------------------------------
// Real-world fixture: must parse cleanly with the acceptance thresholds
// ---------------------------------------------------------------------------

TEST(Ptxparse, RealFusedAttentionParsesCleanly) {
  const std::string text = read_file(REAL_FIXTURE);
  ASSERT_FALSE(text.empty()) << "fused_attention.ptx missing: " << REAL_FIXTURE;

  kernscope::PtxModule mod;
  ASSERT_NO_THROW({ mod = kernscope::parse_ptx_module(text); });

  ASSERT_FALSE(mod.kernels().empty());
  const auto &k = mod.kernels().front();
  EXPECT_EQ("_Z15fused_attentionPKfS0_S0_Pfi", k.name);
  EXPECT_TRUE(k.entry_point);
  EXPECT_GT(k.instruction_count, 0);
  EXPECT_GE(k.bar_sync_count, 1);
  EXPECT_GE(k.shared_access_count, 1);
  EXPECT_GE(k.global_access_count, 1);
  EXPECT_GT(k.cfg_edge_count, 0);
}

// ---------------------------------------------------------------------------
// Malformed input: PtxParseError with line:col, never a crash
// ---------------------------------------------------------------------------

TEST(Ptxparse, TruncatedCorpusFixtureThrowsWithLineCol) {
  const std::string text = read_file(corpus_path("truncated_kernel.ptx"));
  ASSERT_FALSE(text.empty());
  EXPECT_THROW(
      {
        try {
          (void)kernscope::parse_ptx_module(text);
        } catch (const kernscope::PtxParseError &e) {
          EXPECT_GT(e.line(), 0);
          EXPECT_GT(e.col(), 0);
          EXPECT_NE(std::string(e.what()).find("line"), std::string::npos);
          throw;
        }
      },
      kernscope::PtxParseError);
}

TEST(Ptxparse, TruncatedRealFixtureAtOffsetsThrowsWithLineCol) {
  const std::string full = read_file(REAL_FIXTURE);
  ASSERT_FALSE(full.empty());
  for (const double frac : {0.37, 0.9}) {
    const std::size_t cut = static_cast<std::size_t>(
        static_cast<double>(full.size()) * frac);
    const std::string truncated = full.substr(0, cut);
    EXPECT_THROW(
        {
          try {
            (void)kernscope::parse_ptx_module(truncated);
          } catch (const kernscope::PtxParseError &e) {
            EXPECT_GT(e.line(), 0);
            EXPECT_GT(e.col(), 0);
            throw;
          }
        },
        kernscope::PtxParseError)
        << "expected PtxParseError for truncation at " << cut << " bytes";
  }
}

TEST(Ptxparse, GarbageInputThrows) {
  EXPECT_THROW((void)kernscope::parse_ptx_module("this is not ptx at all"),
               kernscope::PtxParseError);
}

TEST(Ptxparse, EmptyInputThrows) {
  EXPECT_THROW((void)kernscope::parse_ptx_module(""), kernscope::PtxParseError);
}

TEST(Ptxparse, UnbalancedBodyThrows) {
  const std::string text =
      ".version 8.5\n"
      ".target sm_86\n"
      ".visible .entry f()\n"
      "{\n"
      "  mov.u32 %r1, 1;\n";  // missing closing '}'
  EXPECT_THROW((void)kernscope::parse_ptx_module(text),
               kernscope::PtxParseError);
}

TEST(Ptxparse, MissingMnemonicAfterPredicateThrows) {
  const std::string text =
      ".version 8.5\n"
      ".visible .entry f()\n"
      "{\n"
      "  @%p1 ;\n"
      "}\n";
  EXPECT_THROW((void)kernscope::parse_ptx_module(text),
               kernscope::PtxParseError);
}
