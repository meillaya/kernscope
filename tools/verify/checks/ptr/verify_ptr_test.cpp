// verify_ptr_test.cpp — corpus-backed acceptance tests for the ptr check
// module (todo 34): check classes 5 (illegal pointer arithmetic) + 7
// (uninitialized reads) + 8 (invalid address-space casts).
//
// The plugin (PointerDisciplineCheckPlugin, registered via REGISTER_CHECK in
// ptr_check.cpp) is linked into THIS binary, so the core engine runs it for
// every kernel.  The tests ingest the committed corpus fixtures
// (tests/verify/corpus/), collect the plugin's "verdict_json:" diagnostics
// and assert them against the manifest fragment
// (tests/verify/corpus/manifest.ptr.json — the schema: a JSON array of
// {file, kernel, check, expected_verdict}).
//
// Coverage (plan todo 34 acceptance, `ctest -R verify_ptr`):
//   1. manifest-driven verdicts:
//        ptr_clean.cu  ptr_clean_kernel        check 5 -> SAFE
//        ptr_oob.cu    ptr_oob_escape_kernel   check 5 -> PTR_OOB
//        ptr_oob.cu    ptr_oob_range_kernel    check 5 -> PTR_OOB
//        ptr_oob.cu    ptr_sub_unrelated_kernel check 5 -> PTR_OOB
//        ptr_cast_bad.cu ...                   check 8 -> INVALID_ADDRSPACE_CAST
//        ptr_uninit.cu ...                     check 7 -> MAY_UNINIT_READ
//   2. QA negative: the LEGAL generic-mediated cast (shared -> generic ->
//      global via __cvta_generic_to_global) must NOT be flagged — the
//      manifest entry asserts this, and the test guards it explicitly.
//   3. plugin registration contract: id "ptr" with classes {5, 7, 8}.

#include "ptr_common.h"

#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/ingestion.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/Module.h>
#include <llvm/Support/raw_ostream.h>

#include <gtest/gtest.h>

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
// ints/bools; the corpus test is the only consumer). Mirrors the oob
// module's test parser.
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
          case 'n':
            s += '\n';
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

std::string corpusDir() { return std::string(PTR_CORPUS_DIR); }

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
    llvm::errs() << "[ingest-fail] " << file << " diags="
                 << result.clang_diagnostics.size() << "\n";
    for (const std::string& d : result.clang_diagnostics) {
      llvm::errs() << "[clang] " << d << "\n";
    }
    return out;
  }
  out.context = result.context;
  out.module = std::move(result.module);
  if (std::getenv("PTR_DEBUG_IR")) {
    llvm::errs() << "=== IR of " << file << " ===\n" << *out.module << "\n";
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
  const Json root = parseJsonText(readFile(corpusDir() + "/manifest.ptr.json"));
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
    out += "  {\"file\":\"" + ptr::jsonEscape(file) + "\",\"diagnostic\":" +
           payload + "}";
  }
  return out;
}

}  // namespace

// --- acceptance: manifest-driven verdicts (ctest -R verify_ptr) -----------
TEST(VerifyPtr, ManifestVerdictsMatch) {
  const std::vector<ManifestEntry> manifest = loadManifest();
  ASSERT_FALSE(manifest.empty());

  std::string evidence = "[\n";
  std::string seen_files;
  bool any_missing = false;
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
  EXPECT_FALSE(any_missing);
  if (evidence.size() > 2) {
    evidence.resize(evidence.size() - 2);  // drop trailing ",\n"
  }
  evidence += "\n]\n";
  std::cout << "=== corpus verdict JSON (evidence) ===\n" << evidence;
}

// --- class-5 detail: escape kernel carries span + bound in the payload ----
TEST(VerifyPtr, PtrOobEscapeHasSpanAndBound) {
  const Analyzed a = runAnalyze("ptr_oob.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "ptr_oob_escape_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v =
      pluginVerdict(a, k->name, CheckClass::kPtrArith);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, ptr::kVerdictPtrOob);
  EXPECT_GT(v->payload.getInt("bad"), 0);
  const Json* findings = v->payload.find("findings");
  ASSERT_NE(findings, nullptr);
  ASSERT_FALSE(findings->array.empty());
  const Json& f = findings->array.front();
  EXPECT_EQ(f.getString("alloc"), "g_buf");
  EXPECT_EQ(f.getInt("bound"), 65536);
  EXPECT_EQ(f.getString("kind"), "ESCAPED");
  // g_buf + 20000: span [80000, 80004) — the escape is proven statically.
  EXPECT_NE(f.getString("span").find("80000"), std::string::npos)
      << "span = " << f.getString("span");
}

// --- class-5 detail: unrelated pointer subtraction is flagged -------------
TEST(VerifyPtr, PtrSubUnrelatedFlagged) {
  const Analyzed a = runAnalyze("ptr_oob.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "ptr_sub_unrelated_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v =
      pluginVerdict(a, k->name, CheckClass::kPtrArith);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, ptr::kVerdictPtrOob);
  const Json* findings = v->payload.find("findings");
  ASSERT_NE(findings, nullptr);
  bool saw_unrelated = false;
  for (const Json& f : findings->array) {
    if (f.getString("kind") == "UNRELATED_SUB") {
      saw_unrelated = true;
      EXPECT_EQ(f.getString("alloc"), "g_buf");
    }
  }
  EXPECT_TRUE(saw_unrelated) << "no UNRELATED_SUB finding emitted";
}

// --- class-8 detail: the bad cast is an Error citing the table ------------
TEST(VerifyPtr, BadCastErrorCitesTable) {
  const Analyzed a = runAnalyze("ptr_cast_bad.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "cast_bad_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v =
      pluginVerdict(a, k->name, CheckClass::kAddrSpaceCast);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, ptr::kVerdictInvalidAddrspaceCast);
  EXPECT_GT(v->payload.getInt("bad"), 0);
  const Json* findings = v->payload.find("findings");
  ASSERT_NE(findings, nullptr);
  ASSERT_FALSE(findings->array.empty());
  const Json& f = findings->array.front();
  EXPECT_EQ(f.getInt("src_as"), 3);
  EXPECT_EQ(f.getInt("dst_as"), 1);
  EXPECT_EQ(f.getString("src"), "shared");
  EXPECT_EQ(f.getString("dst"), "global");
  // The diagnostic message must cite the verified table
  // (docs/research/gpu-memory-pass-research.md §2.1).
  bool cited = false;
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id == CheckClass::kAddrSpaceCast &&
        d.function.find("cast_bad_kernel") != std::string::npos) {
      EXPECT_EQ(d.severity, Severity::Error);
      EXPECT_NE(d.message.find("gpu-memory-pass-research.md"),
                std::string::npos)
          << "class-8 diagnostic must cite the research table";
      EXPECT_NE(d.message.find("§2.1"), std::string::npos);
      cited = true;
    }
  }
  EXPECT_TRUE(cited);
}

// --- QA negative: generic-mediated chain stays SAFE ------------------------
// The shared -> generic -> global chain (the pattern the plan names via
// __cvta_generic_to_global; expressed here with explicit attribute casts —
// the cvta builtins are host-stubbed in the staged CUDA headers and fail
// the host pass — producing IR-identical addrspacecast 3->0 + 0->1) must
// NOT be flagged: both individual casts are legal per the table.
TEST(VerifyPtr, LegalGenericMediatedChainNotFlagged) {
  const Analyzed a = runAnalyze("ptr_cast_bad.cu", IngestionOptions());
  ASSERT_NE(a.state, nullptr);
  const KernelInfo* k = findKernel(a, "cast_legal_kernel");
  ASSERT_NE(k, nullptr);
  const std::optional<Verdict> v =
      pluginVerdict(a, k->name, CheckClass::kAddrSpaceCast);
  ASSERT_TRUE(v.has_value());
  EXPECT_EQ(v->verdict, ptr::kVerdictSafe);
  EXPECT_EQ(v->payload.getInt("bad"), 0);
  // No class-8 Error may exist for this kernel (precision guard).
  for (const Diagnostic& d : a.state->diagnostics()) {
    if (d.class_id == CheckClass::kAddrSpaceCast &&
        d.function.find("cast_legal_kernel") != std::string::npos) {
      EXPECT_NE(d.severity, Severity::Error)
          << "legal generic-mediated cast must not be flagged";
    }
  }
}

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
