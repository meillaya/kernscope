// race_test.cpp — kernscope-verify race check tests (todo 33, class 4).
//
// The plugin (RaceCheck, registered via REGISTER_CHECK in race_check.cpp)
// is linked into THIS binary, so the core engine runs it for every kernel.
// The tests ingest the committed corpus fixtures (tests/verify/corpus/),
// collect the plugin's "verdict_json:" diagnostics and assert them against
// the manifest fragment (tests/verify/corpus/manifest.race.json — a JSON
// array of {file, kernel, check, expected_verdict}).
//
// Coverage (plan todo 33 acceptance + QA):
//   1. manifest verdicts (ctest verify_race):
//        race_writewrite.cu  race_writewrite_kernel -> MAY_RACE (both
//          source sites in the diagnostic JSON: two same-phase writes to
//          s[t], no barrier);
//        race_writeread.cu   race_writeread_kernel  -> MAY_RACE (write
//          s[t] / read s[255-t], no barrier, spans overlap);
//        race_clean.cu       race_clean_kernel      -> SAFE (barrier
//          separates the write phase from the reduction-read phase);
//        race_clean.cu       race_clean_disjoint_kernel -> SAFE (SCEV-
//          proven disjoint byte spans);
//        race_clean.cu       race_clean_uniform_kernel -> SAFE (mutually
//          exclusive paths of a block-uniform branch);
//        race_dataindex.cu   race_data_kernel       -> UNKNOWN_RACE
//          (data-dependent index: soundness naming — NOT SAFE, NOT
//          MAY_RACE).
//   2. QA negative (asserted in detail + evidence): data-dependent index
//      race candidate -> UNKNOWN_RACE with BOTH sites in the diagnostic.
//   3. MAY_RACE pairs carry both source sites (op1/l1 + op2/l2 in the
//      verdict JSON) and the per-pair warning lists the missing-barrier
//      path.
//   4. registry contract: id "race", class {4}.

#include <kernscope/verify/check_plugin.h>
#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

using namespace kernscope::verify;

namespace {

// --------------------------------------------------------------------------
// Minimal JSON parser for the module-owned verdict payloads + the manifest
// fragment (the plugin emits a restricted grammar: objects/arrays/strings/
// ints/bools; the corpus test is the only consumer — same design as the
// oob module's corpus test).
// --------------------------------------------------------------------------

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  int64_t number = 0;
  std::string string;
  std::vector<Json> array;
  std::vector<std::pair<std::string, std::unique_ptr<Json>>> object;

  const Json* find(const std::string& key) const {
    if (type != Type::Object) {
      return nullptr;
    }
    for (const auto& member : object) {
      if (member.first == key) {
        return member.second.get();
      }
    }
    return nullptr;
  }
  std::string getString(const std::string& key,
                        const std::string& def = "") const {
    const Json* v = find(key);
    return (v && v->type == Type::String) ? v->string : def;
  }
  int64_t getInt(const std::string& key, int64_t def = 0) const {
    const Json* v = find(key);
    return (v && v->type == Type::Number) ? v->number : def;
  }
};

class JsonParser {
 public:
  explicit JsonParser(const std::string& text) : text_(text) {}

  Json parse() {
    pos_ = skipWs(0);
    Json v = parseValue();
    pos_ = skipWs(pos_);
    return v;
  }

 private:
  const std::string& text_;
  size_t pos_ = 0;

  static bool isWs(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
  }
  size_t skipWs(size_t p) const {
    while (p < text_.size() && isWs(text_[p])) {
      ++p;
    }
    return p;
  }
  char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
  char next() { return pos_ < text_.size() ? text_[pos_++] : '\0'; }
  [[noreturn]] void fail(const std::string& why) const {
    std::cerr << "json parse error at " << pos_ << ": " << why << "\n";
    std::abort();
  }

  Json parseValue() {
    pos_ = skipWs(pos_);
    const char c = peek();
    switch (c) {
      case '{':
        return parseObject();
      case '[':
        return parseArray();
      case '"':
        return parseString();
      case 't':
      case 'f':
        return parseBool();
      default:
        if (c == '-' || (c >= '0' && c <= '9')) {
          return parseNumber();
        }
        if (c == 'n') {
          for (const char* lit = "null"; *lit; ++lit) {
            if (next() != *lit) {
              fail("bad null literal");
            }
          }
          return Json{};
        }
        fail("unexpected character");
    }
  }

  Json parseObject() {
    Json out;
    out.type = Json::Type::Object;
    next();  // '{'
    pos_ = skipWs(pos_);
    if (peek() == '}') {
      next();
      return out;
    }
    while (true) {
      pos_ = skipWs(pos_);
      if (peek() != '"') {
        fail("expected string key");
      }
      std::string key = parseString().string;
      pos_ = skipWs(pos_);
      if (next() != ':') {
        fail("expected ':'");
      }
      out.object.emplace_back(std::move(key),
                              std::make_unique<Json>(parseValue()));
      pos_ = skipWs(pos_);
      const char sep = next();
      if (sep == '}') {
        break;
      }
      if (sep != ',') {
        fail("expected ',' or '}'");
      }
    }
    return out;
  }

  Json parseArray() {
    Json out;
    out.type = Json::Type::Array;
    next();  // '['
    pos_ = skipWs(pos_);
    if (peek() == ']') {
      next();
      return out;
    }
    while (true) {
      out.array.push_back(parseValue());
      pos_ = skipWs(pos_);
      const char sep = next();
      if (sep == ']') {
        break;
      }
      if (sep != ',') {
        fail("expected ',' or ']'");
      }
    }
    return out;
  }

  Json parseString() {
    Json out;
    out.type = Json::Type::String;
    next();  // '"'
    std::string s;
    while (true) {
      const char c = next();
      if (c == '\0' || c == '"') {
        break;
      }
      if (c == '\\') {
        const char e = next();
        switch (e) {
          case '"':
            s += '"';
            break;
          case '\\':
            s += '\\';
            break;
          case '/':
            s += '/';
            break;
          case 'n':
            s += '\n';
            break;
          case 'r':
            s += '\r';
            break;
          case 't':
            s += '\t';
            break;
          default:
            fail("unsupported escape");
        }
      } else {
        s += c;
      }
    }
    out.string = std::move(s);
    return out;
  }

  Json parseNumber() {
    Json out;
    out.type = Json::Type::Number;
    std::string s;
    if (peek() == '-') {
      s += next();
    }
    while (peek() >= '0' && peek() <= '9') {
      s += next();
    }
    if (s.empty() || s == "-") {
      fail("bad number");
    }
    out.number = std::stoll(s);
    return out;
  }

  Json parseBool() {
    Json out;
    out.type = Json::Type::Bool;
    if (peek() == 't') {
      for (const char* lit = "true"; *lit; ++lit) {
        if (next() != *lit) {
          fail("bad literal");
        }
      }
      out.boolean = true;
    } else {
      for (const char* lit = "false"; *lit; ++lit) {
        if (next() != *lit) {
          fail("bad literal");
        }
      }
      out.boolean = false;
    }
    return out;
  }
};

Json parseJsonText(const std::string& text) { return JsonParser(text).parse(); }

// --------------------------------------------------------------------------
// Corpus harness
// --------------------------------------------------------------------------

std::string corpusDir() { return std::string(RACE_CORPUS_DIR); }

std::string readFile(const std::string& path) {
  std::ifstream in(path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct Analyzed {
  std::shared_ptr<llvm::LLVMContext> context;
  std::shared_ptr<llvm::Module> module;
  std::unique_ptr<VerifierState> state;
};

Analyzed runAnalyze(const std::string& file, IngestionOptions iopts,
                    VerifierOptions vopts = VerifierOptions()) {
  ClangIngester ingester;
  IngestionResult result = ingester.ingest(corpusDir() + "/" + file, iopts);
  Analyzed out;
  if (!result.success) {
    return out;
  }
  out.context = result.context;
  out.module = std::move(result.module);
  out.state = std::make_unique<VerifierState>(out.module, out.context, vopts);
  out.state->analyze();
  return out;
}

const KernelInfo* findKernel(const Analyzed& a, const std::string& needle) {
  if (!a.state) {
    return nullptr;
  }
  for (const KernelInfo& k : a.state->kernels()) {
    if (k.name.find(needle) != std::string::npos) {
      return &k;
    }
  }
  return nullptr;
}

/// The plugin's per-kernel verdict for check class 4 (parsed from the FIRST
/// class-4 "verdict_json:" payload — the per-kernel verdict diagnostic is
/// emitted before any per-pair finding).  std::nullopt when none was
/// emitted.
struct Verdict {
  std::string verdict;
  Json payload;
};
std::optional<Verdict> pluginVerdict(const Analyzed& a,
                                     const std::string& kernel) {
  constexpr const char* kMarker = "verdict_json: ";
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id != CheckClass::kRace || d.function != kernel) {
      continue;
    }
    const size_t at = d.message.find(kMarker);
    if (at == std::string::npos) {
      continue;
    }
    Verdict v;
    v.payload = parseJsonText(d.message.substr(at + std::string(kMarker).size()));
    v.verdict = v.payload.getString("verdict");
    return v;
  }
  return std::nullopt;
}

struct ManifestEntry {
  std::string file;
  std::string kernel;
  int check = 0;
  std::string expected;
};

std::vector<ManifestEntry> loadManifest() {
  const Json root = parseJsonText(readFile(corpusDir() + "/manifest.race.json"));
  std::vector<ManifestEntry> out;
  EXPECT_EQ(root.type, Json::Type::Array);
  for (const Json& e : root.array) {
    ManifestEntry m;
    m.file = e.getString("file");
    m.kernel = e.getString("kernel");
    m.check = static_cast<int>(e.getInt("check"));
    m.expected = e.getString("expected_verdict");
    EXPECT_FALSE(m.file.empty());
    EXPECT_FALSE(m.kernel.empty());
    out.push_back(std::move(m));
  }
  return out;
}

/// All verdict payloads for one analyzed file (printed for evidence).
std::string verdictJsonSummary(const Analyzed& a, const std::string& file) {
  constexpr const char* kMarker = "verdict_json: ";
  std::string out;
  for (const Diagnostic& d : a.state->diagnostics()) {
    const size_t at = d.message.find(kMarker);
    if (at == std::string::npos) {
      continue;
    }
    const std::string payload =
        d.message.substr(at + std::string(kMarker).size());
    if (!out.empty()) {
      out += ",\n";
    }
    out += "  {\"file\":\"" + file + "\",\"diagnostic\":" + payload + "}";
  }
  return out;
}

}  // namespace

// --------------------------------------------------------------------------
// Acceptance tests
// --------------------------------------------------------------------------

// 1. Corpus manifest: every {file, kernel, check, expected_verdict} entry
// matches the plugin's emitted verdict (default ingestion: host+device).
TEST(RaceCheck, ManifestVerdictsMatch) {
  const std::vector<ManifestEntry> manifest = loadManifest();
  ASSERT_FALSE(manifest.empty());

  std::string evidence = "[\n";
  std::string seen_files;
  for (const ManifestEntry& m : manifest) {
    if (seen_files.find(m.file + ";") == std::string::npos) {
      seen_files += m.file + ";";
      const Analyzed a = runAnalyze(m.file, IngestionOptions());
      ASSERT_NE(a.state, nullptr) << "ingestion of " << m.file << " failed";
      const std::string summary = verdictJsonSummary(a, m.file);
      if (!summary.empty()) {
        evidence += summary + ",\n";
      }
    }
    const Analyzed a = runAnalyze(m.file, IngestionOptions());
    const KernelInfo* k = findKernel(a, m.kernel);
    ASSERT_NE(k, nullptr) << "kernel " << m.kernel << " not found in "
                          << m.file;
    const std::optional<Verdict> v = pluginVerdict(a, k->name);
    ASSERT_TRUE(v.has_value())
        << m.file << ":" << m.kernel << ": no race verdict emitted";
    EXPECT_EQ(v->verdict, m.expected)
        << m.file << ":" << m.kernel << " check " << m.check;
  }
  if (evidence.size() > 2) {
    evidence.resize(evidence.size() - 2);  // drop trailing ",\n"
  }
  evidence += "\n]\n";
  std::cout << "=== corpus verdict JSON (evidence) ===\n" << evidence;
}

// 2. MAY_RACE fixtures: the per-kernel verdict payload carries BOTH source
// sites (op1/l1 + op2/l2) per pair, and each pair explains the missing
// barrier (same barrier phase).
TEST(RaceCheck, MayRacePairsCarryBothSites) {
  const struct {
    const char* file;
    const char* kernel;
  } cases[] = {{"race_writewrite.cu", "race_writewrite_kernel"},
               {"race_writeread.cu", "race_writeread_kernel"}};
  for (const auto& c : cases) {
    const Analyzed a = runAnalyze(c.file, IngestionOptions());
    ASSERT_NE(a.state, nullptr) << "ingestion of " << c.file << " failed";
    const KernelInfo* k = findKernel(a, c.kernel);
    ASSERT_NE(k, nullptr) << "kernel " << c.kernel << " not found";
    const std::optional<Verdict> v = pluginVerdict(a, k->name);
    ASSERT_TRUE(v.has_value());
    EXPECT_EQ(v->verdict, "MAY_RACE") << c.file;
    const Json* pairs = v->payload.find("pairs");
    ASSERT_NE(pairs, nullptr) << c.file << ": verdict JSON has no pairs array";
    ASSERT_FALSE(pairs->array.empty()) << c.file << ": no race pairs";
    for (const Json& p : pairs->array) {
      EXPECT_EQ(p.getString("verdict"), "MAY_RACE");
      // Acceptance: BOTH source sites in the diagnostic JSON.
      EXPECT_GT(p.getInt("l1"), 0) << c.file << ": site 1 missing";
      EXPECT_GT(p.getInt("l2"), 0) << c.file << ": site 2 missing";
      EXPECT_NE(p.getInt("l1"), p.getInt("l2"))
          << c.file << ": sites must be distinct source locations";
      EXPECT_FALSE(p.getString("op1").empty());
      EXPECT_FALSE(p.getString("op2").empty());
      EXPECT_FALSE(p.getString("why").empty());
    }
  }
}

// 3. SAFE fixtures stay SAFE (barrier separation, SCEV-disjoint spans,
// uniform-branch mutual exclusion).
TEST(RaceCheck, CleanFixturesStaySafe) {
  const Analyzed a = runAnalyze("race_clean.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  for (const char* kernel :
       {"race_clean_kernel", "race_clean_disjoint_kernel",
        "race_clean_uniform_kernel"}) {
    const KernelInfo* k = findKernel(a, kernel);
    ASSERT_NE(k, nullptr) << "kernel " << kernel << " not found";
    const std::optional<Verdict> v = pluginVerdict(a, k->name);
    ASSERT_TRUE(v.has_value()) << kernel << ": no verdict emitted";
    EXPECT_EQ(v->verdict, "SAFE") << kernel;
  }
}

// 4. QA negative: data-dependent index race candidate -> UNKNOWN_RACE
// (soundness naming: NOT SAFE, NOT MAY_RACE — the analysis cannot even
// bound the addresses).
TEST(RaceCheck, DataDependentIndexIsUnknownRace) {
  const Analyzed a = runAnalyze("race_dataindex.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "race_data_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v = pluginVerdict(a, k->name);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, "UNKNOWN_RACE");
  EXPECT_NE(v->verdict, "SAFE");
  EXPECT_NE(v->verdict, "MAY_RACE");
  const Json* pairs = v->payload.find("pairs");
  ASSERT_NE(pairs, nullptr);
  ASSERT_FALSE(pairs->array.empty());
  for (const Json& p : pairs->array) {
    EXPECT_EQ(p.getString("verdict"), "UNKNOWN_RACE");
    EXPECT_GT(p.getInt("l1"), 0);
    EXPECT_GT(p.getInt("l2"), 0);
    EXPECT_EQ(p.getString("idx1"), "data-dependent");
    EXPECT_EQ(p.getString("idx2"), "data-dependent");
  }
  std::cout << "=== QA negative: " << k->name << " -> " << v->verdict << "\n"
            << verdictJsonSummary(a, "race_dataindex.cu") << "\n";
}

// 5. Per-pair warnings carry the missing-barrier path (both sites + the
// phase explanation).
TEST(RaceCheck, PerPairWarningsListBothSitesAndMissingBarrier) {
  const Analyzed a = runAnalyze("race_writeread.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "race_writeread");
  ASSERT_NE(k, nullptr);
  bool saw_pair_warning = false;
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id != CheckClass::kRace || d.function != k->name ||
        d.message.find("MAY_RACE:") == std::string::npos) {
      continue;
    }
    saw_pair_warning = true;
    EXPECT_EQ(d.severity, Severity::Warning);
    EXPECT_EQ(d.path.size(), 3u) << "path must list both sites + the "
                                    "missing-barrier explanation";
    EXPECT_NE(d.path[0].find("site 1:"), std::string::npos);
    EXPECT_NE(d.path[1].find("site 2:"), std::string::npos);
    EXPECT_NE(d.path[2].find("missing barrier"), std::string::npos);
    EXPECT_NE(d.message.find("line "), std::string::npos);
  }
  EXPECT_TRUE(saw_pair_warning) << "no per-pair MAY_RACE warning emitted";
}

// 6. Registry contract: id "race", check class {4} only.
TEST(RaceCheck, PluginRegisteredWithClass4) {
  bool found = false;
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    if (e.id == "race") {
      found = true;
      ASSERT_EQ(e.class_ids.size(), 1u);
      EXPECT_EQ(e.class_ids[0], CheckClass::kRace);
    }
  }
  EXPECT_TRUE(found) << "race plugin not registered (REGISTER_CHECK)";
  EXPECT_EQ(CheckRegistry::instance().pluginsForClass(CheckClass::kRace)
                .size(),
            1u);
}
