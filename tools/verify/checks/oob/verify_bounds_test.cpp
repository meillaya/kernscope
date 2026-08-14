// verify_bounds_test.cpp — corpus-backed acceptance tests for the oob check
// module (todo 31): check classes 1 (global OOB) + 2 (shared-memory bounds).
//
// The plugin (BoundsCheckPlugin, registered via REGISTER_CHECK in
// memory_bounds_check.cpp) is linked into THIS binary, so the core engine
// runs it for every kernel.  The tests ingest the committed corpus fixtures
// (tests/verify/corpus/), collect the plugin's "verdict_json:" diagnostics
// and assert them against the manifest fragment
// (tests/verify/corpus/manifest.oob.json — the schema: a JSON array of
// {file, kernel, check, expected_verdict}).
//
// Coverage (plan todo 31 acceptance):
//   1. clean kernel (constant launch, provably in-bounds
//      i = blockIdx.x*blockDim.x + threadIdx.x) -> SAFE with the SCEV range
//      in the proof field;
//   2. OOB kernel (index = tid*N beyond the allocation) -> MAY_OOB with
//      range + bound in the diagnostic JSON;
//   3. smem overflow pair: access past the declared __shared__ size ->
//      MAY_OOB (smem); in-bounds smem -> SAFE;
//   4. dynamic smem (extern __shared__) launch-param handling;
//   5. the manifest fragment entries all match the emitted verdicts;
//   6. QA negative: --device-only ingestion removes the launch site ->
//      verdict degrades to SYMBOLIC_DIMS, never SAFE.

#include "oob_common.h"

#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

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
// ints/bools; the corpus test is the only consumer).
// --------------------------------------------------------------------------

struct Json {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool boolean = false;
  int64_t number = 0;
  std::string string;
  std::vector<Json> array;
  // Object members are held through unique_ptr: a by-value
  // pair<string, Json> member would need Json complete while Json is still
  // being defined (libstdc++ instantiates the vector destructor eagerly).
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

  static bool isWs(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }
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

std::string corpusDir() { return std::string(OOB_CORPUS_DIR); }

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
  if (std::getenv("OOB_DEBUG_IR")) {
    llvm::errs() << "=== IR of " << file << " ===\n" << *out.module << "\n";
    for (const auto& [fn, sz] : oob::scanLaunchSmem(*out.module)) {
      llvm::errs() << "smem-scan: " << fn->getName() << " -> "
                   << (sz ? std::to_string(*sz) : std::string("unknown"))
                   << "\n";
    }
  }
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

// The plugin's per-kernel verdict for one check class (parsed from the
// "verdict_json: {...}" payload).  std::nullopt when no verdict was emitted.
struct Verdict {
  std::string verdict;
  Json payload;
};
std::optional<Verdict> pluginVerdict(const Analyzed& a, const std::string& kernel,
                                     int check) {
  constexpr const char* kMarker = "verdict_json: ";
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id != check || d.function != kernel) {
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
  const Json root = parseJsonText(readFile(corpusDir() + "/manifest.oob.json"));
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

// All verdict payloads for one analyzed file (printed for evidence).
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
    out += "  {\"file\":\"" + oob::jsonEscape(file) + "\",\"diagnostic\":" +
           payload + "}";
  }
  return out;
}

}  // namespace

// --------------------------------------------------------------------------
// Acceptance tests
// --------------------------------------------------------------------------

// 1+5. Corpus manifest: every {file, kernel, check, expected_verdict} entry
// matches the plugin's emitted verdict (default ingestion: host+device).
TEST(VerifyBounds, ManifestVerdictsMatch) {
  const std::vector<ManifestEntry> manifest = loadManifest();
  ASSERT_FALSE(manifest.empty());

  std::string evidence = "[\n";
  bool any_missing = false;
  std::string seen_files;
  for (const ManifestEntry& m : manifest) {
    if (seen_files.find(m.file + ";") == std::string::npos) {
      seen_files += m.file + ";";
      const Analyzed a = runAnalyze(m.file, IngestionOptions());
      ASSERT_NE(a.state, nullptr) << "ingestion of " << m.file << " failed";
      std::string summary = verdictJsonSummary(a, m.file);
      if (!summary.empty()) {
        evidence += summary + ",\n";
      }
    }
    const Analyzed a = runAnalyze(m.file, IngestionOptions());
    const KernelInfo* k = findKernel(a, m.kernel);
    ASSERT_NE(k, nullptr) << "kernel " << m.kernel << " not found in " << m.file;
    const std::optional<Verdict> v = pluginVerdict(a, k->name, m.check);
    if (!v) {
      any_missing = true;
      ADD_FAILURE() << m.file << ":" << m.kernel << " check " << m.check
                    << ": no plugin verdict emitted";
      continue;
    }
    EXPECT_EQ(v->verdict, m.expected)
        << m.file << ":" << m.kernel << " check " << m.check;
  }
  // Emitted verdicts that are NOT in the manifest must not silently appear
  // (every kernel in the corpus should be covered).
  EXPECT_FALSE(any_missing);
  if (evidence.size() > 2) {
    evidence.resize(evidence.size() - 2);  // drop trailing ",\n"
  }
  evidence += "\n]\n";
  std::cout << "=== corpus verdict JSON (evidence) ===\n" << evidence;
}

// 1. Clean kernel: SAFE verdict whose proof names the SCEV-derived range.
TEST(VerifyBounds, CleanKernelSafeWithScevRangeProof) {
  const Analyzed a = runAnalyze("oob_clean.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "oob_clean_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v = pluginVerdict(a, k->name, CheckClass::kGlobalOob);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, oob::kVerdictSafe);
  const Json* findings = v->payload.find("findings");
  ASSERT_NE(findings, nullptr);
  ASSERT_FALSE(findings->array.empty());
  const Json& f = findings->array.front();
  // Proof field: the SCEV range of the innermost GEP index, e.g. [0,8224).
  const std::string range = f.getString("range");
  EXPECT_EQ(f.getString("kind"), "IN_BOUNDS");
  EXPECT_NE(range.find("[0,"), std::string::npos) << "range = " << range;
  EXPECT_NE(range.find(")"), std::string::npos);
  // The bound is the statically known allocation size (65536 bytes).
  EXPECT_EQ(f.getInt("bound"), 65536);
  EXPECT_TRUE(f.getInt("bound") > 0);
}

// 2. OOB kernel: MAY_OOB verdict with range + bound in the diagnostic JSON.
TEST(VerifyBounds, OobKernelMayOobWithRangeAndBound) {
  const Analyzed a = runAnalyze("oob_may.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "oob_may_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v = pluginVerdict(a, k->name, CheckClass::kGlobalOob);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, oob::kVerdictMayOob);
  EXPECT_GT(v->payload.getInt("may_oob"), 0);
  const Json* findings = v->payload.find("findings");
  ASSERT_NE(findings, nullptr);
  ASSERT_FALSE(findings->array.empty());
  const Json& f = findings->array.front();
  EXPECT_EQ(f.getString("kind"), "MAY_OOB");
  EXPECT_EQ(f.getString("alloc"), "g_buf");
  EXPECT_EQ(f.getInt("bound"), 4096);  // g_buf[1024] = 4096 bytes
  // The SCEV range for tid*1024 over 256 threads starts at [0,...).
  EXPECT_NE(f.getString("range").find("[0,"), std::string::npos);
}

// 3. smem overflow pair: past the declared __shared__ size -> MAY_OOB;
// in-bounds smem -> SAFE.
TEST(VerifyBounds, SmemOverflowPair) {
  const Analyzed a = runAnalyze("smem_overflow.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* bug = findKernel(a, "smem_overflow_kernel");
  ASSERT_NE(bug, nullptr);
  const std::optional<Verdict> vbug =
      pluginVerdict(a, bug->name, CheckClass::kSmemBounds);
  ASSERT_TRUE(vbug.has_value());
  EXPECT_EQ(vbug->verdict, oob::kVerdictMayOob);
  const Json* fbug = vbug->payload.find("findings");
  ASSERT_NE(fbug, nullptr);
  ASSERT_FALSE(fbug->array.empty());
  EXPECT_EQ(fbug->array.front().getInt("bound"), 256);  // s[64] = 256 bytes

  const KernelInfo* clean = findKernel(a, "smem_safe_wide_kernel");
  ASSERT_NE(clean, nullptr);
  const std::optional<Verdict> vclean =
      pluginVerdict(a, clean->name, CheckClass::kSmemBounds);
  ASSERT_TRUE(vclean.has_value());
  EXPECT_EQ(vclean->verdict, oob::kVerdictSafe);
}

// 4. Dynamic smem (extern __shared__): the bound comes from the launch-site
// smem argument, per kernel.
TEST(VerifyBounds, DynamicSmemLaunchParamHandling) {
  const Analyzed a = runAnalyze("smem_dynamic.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* safe = findKernel(a, "smem_dyn_safe_kernel");
  ASSERT_NE(safe, nullptr);
  const std::optional<Verdict> vsafe =
      pluginVerdict(a, safe->name, CheckClass::kSmemBounds);
  ASSERT_TRUE(vsafe.has_value());
  EXPECT_EQ(vsafe->verdict, oob::kVerdictSafe);
  const Json* fsafe = vsafe->payload.find("findings");
  ASSERT_NE(fsafe, nullptr);
  ASSERT_FALSE(fsafe->array.empty());
  // <<<1, 64, 256, 0>>>: 256 bytes of dynamic smem bound the access.
  EXPECT_EQ(fsafe->array.front().getInt("bound"), 256);

  const KernelInfo* bug = findKernel(a, "smem_dyn_oob_kernel");
  ASSERT_NE(bug, nullptr);
  const std::optional<Verdict> vbug =
      pluginVerdict(a, bug->name, CheckClass::kSmemBounds);
  ASSERT_TRUE(vbug.has_value());
  EXPECT_EQ(vbug->verdict, oob::kVerdictMayOob);
  const Json* fbug = vbug->payload.find("findings");
  ASSERT_NE(fbug, nullptr);
  ASSERT_FALSE(fbug->array.empty());
  EXPECT_EQ(fbug->array.front().getInt("bound"), 256);
}

// QA negative: removing the launch site (--device-only ingestion) degrades
// the verdict to SYMBOLIC_DIMS — never SAFE (soundness over false safety).
TEST(VerifyBounds, DeviceOnlyDegradesToSymbolicDims) {
  IngestionOptions iopts;
  iopts.device_only = true;
  const Analyzed a = runAnalyze("oob_clean.cu", iopts);
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "oob_clean_kernel");
  ASSERT_NE(k, nullptr);
  ASSERT_EQ(k->config.tier, LaunchTier::Symbolic);
  const std::optional<Verdict> v = pluginVerdict(a, k->name, CheckClass::kGlobalOob);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, oob::kVerdictSymbolicDims);
  EXPECT_NE(v->verdict, oob::kVerdictSafe);
  std::cout << "device-only degraded verdict: "
            << v->payload.getString("verdict") << " (check "
            << v->payload.getInt("check") << ", kernel "
            << v->payload.getString("kernel") << ")\n";
}

// The plugin is registered and dispatched for both check classes.
TEST(VerifyBounds, PluginRegistryContract) {
  bool found = false;
  for (const CheckRegistry::Entry& e : CheckRegistry::instance().enumerate()) {
    if (e.id == "global-oob") {
      found = true;
      EXPECT_EQ(e.class_ids.size(), 2u);
      EXPECT_EQ(e.class_ids[0], CheckClass::kGlobalOob);
      EXPECT_EQ(e.class_ids[1], CheckClass::kSmemBounds);
    }
  }
  EXPECT_TRUE(found);
}
