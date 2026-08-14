//===- db.cpp - JSONL benchmark/config database -------------------*- C++ -*-===//
//
// Implementation of kernscope::db. The JSON value model is a self-contained
// recursive-descent parser + canonical (sorted-member) writer; see db.h for
// the rationale (nlohmann/json is not in the devenv profile and todo 8 must
// not modify devenv files; llvm::json would drag in LLVMSupport).
//
//===----------------------------------------------------------------------===//

#include "kernscope/db.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <system_error>

namespace kernscope::db {
namespace json {
namespace {

constexpr int kMaxNestingDepth = 128;

void AppendEscaped(std::string& out, std::string_view s) {
  out.push_back('"');
  for (const char c : s) {
    switch (c) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        static constexpr char kHex[] = "0123456789abcdef";
        out += "\\u00";
        out.push_back(kHex[(c >> 4) & 0xF]);
        out.push_back(kHex[c & 0xF]);
      } else {
        out.push_back(c); // UTF-8 bytes pass through verbatim
      }
    }
  }
  out.push_back('"');
}

void AppendUtf8(std::string& out, std::uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

class Parser {
public:
  explicit Parser(std::string_view input) : in_(input) {}

  bool ParseValue(Value& out, int depth) {
    SkipWs();
    if (depth > kMaxNestingDepth)
      return Fail("nesting deeper than 128 levels");
    if (pos_ >= in_.size())
      return Fail("unexpected end of input");
    const char c = in_[pos_];
    if (c == '{')
      return ParseObject(out, depth);
    if (c == '[')
      return ParseArray(out, depth);
    if (c == '"') {
      std::string s;
      if (!ParseString(s))
        return false;
      out = Value(std::move(s));
      return true;
    }
    if (c == 't') {
      if (!MatchLiteral("true"))
        return false;
      out = Value(true);
      return true;
    }
    if (c == 'f') {
      if (!MatchLiteral("false"))
        return false;
      out = Value(false);
      return true;
    }
    if (c == 'n') {
      if (!MatchLiteral("null"))
        return false;
      out = Value();
      return true;
    }
    if (c == '-' || (c >= '0' && c <= '9')) {
      double n = 0.0;
      if (!ParseNumber(n))
        return false;
      out = Value(n);
      return true;
    }
    return Fail(std::string("unexpected character '") + c + "'");
  }

  void SkipWs() {
    while (pos_ < in_.size() &&
           (in_[pos_] == ' ' || in_[pos_] == '\t' || in_[pos_] == '\n' ||
            in_[pos_] == '\r'))
      ++pos_;
  }

  bool AtEnd() const { return pos_ == in_.size(); }
  std::size_t Position() const { return pos_; }
  const std::string& Error() const { return error_; }

private:
  bool Fail(std::string message) {
    if (error_.empty())
      error_ = "byte " + std::to_string(pos_) + ": " + std::move(message);
    return false;
  }

  bool MatchLiteral(std::string_view literal) {
    if (in_.substr(pos_, literal.size()) != literal)
      return Fail(std::string("invalid literal (expected ") +
                  std::string(literal) + ")");
    pos_ += literal.size();
    return true;
  }

  bool ParseObject(Value& out, int depth) {
    ++pos_; // consume '{'
    std::vector<Value::Member> members;
    SkipWs();
    if (pos_ < in_.size() && in_[pos_] == '}') {
      ++pos_;
      out = Value(std::move(members));
      return true;
    }
    while (true) {
      SkipWs();
      if (pos_ >= in_.size() || in_[pos_] != '"')
        return Fail("expected string key in object");
      std::string key;
      if (!ParseString(key))
        return false;
      SkipWs();
      if (pos_ >= in_.size() || in_[pos_] != ':')
        return Fail("expected ':' after object key");
      ++pos_;
      Value v;
      if (!ParseValue(v, depth + 1))
        return false;
      members.emplace_back(std::move(key), std::move(v));
      SkipWs();
      if (pos_ >= in_.size())
        return Fail("unterminated object");
      if (in_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (in_[pos_] == '}') {
        ++pos_;
        out = Value(std::move(members));
        return true;
      }
      return Fail("expected ',' or '}' in object");
    }
  }

  bool ParseArray(Value& out, int depth) {
    ++pos_; // consume '['
    std::vector<Value> elements;
    SkipWs();
    if (pos_ < in_.size() && in_[pos_] == ']') {
      ++pos_;
      out = Value(std::move(elements));
      return true;
    }
    while (true) {
      Value v;
      if (!ParseValue(v, depth + 1))
        return false;
      elements.push_back(std::move(v));
      SkipWs();
      if (pos_ >= in_.size())
        return Fail("unterminated array");
      if (in_[pos_] == ',') {
        ++pos_;
        continue;
      }
      if (in_[pos_] == ']') {
        ++pos_;
        out = Value(std::move(elements));
        return true;
      }
      return Fail("expected ',' or ']' in array");
    }
  }

  bool ParseString(std::string& out) {
    ++pos_; // consume opening quote
    out.clear();
    while (true) {
      if (pos_ >= in_.size())
        return Fail("unterminated string");
      const char c = in_[pos_];
      if (c == '"') {
        ++pos_;
        return true;
      }
      if (c == '\\') {
        ++pos_;
        if (pos_ >= in_.size())
          return Fail("unterminated escape sequence");
        const char e = in_[pos_++];
        switch (e) {
        case '"':
          out.push_back('"');
          break;
        case '\\':
          out.push_back('\\');
          break;
        case '/':
          out.push_back('/');
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        case 'u': {
          std::uint32_t cp = 0;
          if (!ParseHex4(cp))
            return false;
          if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate: require low
            if (pos_ + 1 >= in_.size() || in_[pos_] != '\\' ||
                in_[pos_ + 1] != 'u')
              return Fail("lone high surrogate in \\u escape");
            pos_ += 2;
            std::uint32_t low = 0;
            if (!ParseHex4(low))
              return false;
            if (low < 0xDC00 || low > 0xDFFF)
              return Fail("invalid low surrogate in \\u escape");
            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
            return Fail("lone low surrogate in \\u escape");
          }
          AppendUtf8(out, cp);
          break;
        }
        default:
          return Fail(std::string("invalid escape '\\") + e + "'");
        }
        continue;
      }
      if (static_cast<unsigned char>(c) < 0x20)
        return Fail("unescaped control character in string");
      out.push_back(c);
      ++pos_;
    }
  }

  bool ParseHex4(std::uint32_t& out) {
    if (pos_ + 4 > in_.size())
      return Fail("truncated \\u escape");
    std::uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      const char c = in_[pos_++];
      v <<= 4;
      if (c >= '0' && c <= '9')
        v |= static_cast<std::uint32_t>(c - '0');
      else if (c >= 'a' && c <= 'f')
        v |= static_cast<std::uint32_t>(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F')
        v |= static_cast<std::uint32_t>(c - 'A' + 10);
      else
        return Fail("invalid hex digit in \\u escape");
    }
    out = v;
    return true;
  }

  bool ParseNumber(double& out) {
    const std::size_t start = pos_;
    if (pos_ < in_.size() && in_[pos_] == '-')
      ++pos_;
    if (pos_ >= in_.size() || !IsDigit(in_[pos_]))
      return Fail("invalid number");
    if (in_[pos_] == '0') {
      ++pos_;
      if (pos_ < in_.size() && IsDigit(in_[pos_]))
        return Fail("invalid number: leading zeros");
    } else {
      while (pos_ < in_.size() && IsDigit(in_[pos_]))
        ++pos_;
    }
    if (pos_ < in_.size() && in_[pos_] == '.') {
      ++pos_;
      if (pos_ >= in_.size() || !IsDigit(in_[pos_]))
        return Fail("invalid number: expected digit after '.'");
      while (pos_ < in_.size() && IsDigit(in_[pos_]))
        ++pos_;
    }
    if (pos_ < in_.size() && (in_[pos_] == 'e' || in_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < in_.size() && (in_[pos_] == '+' || in_[pos_] == '-'))
        ++pos_;
      if (pos_ >= in_.size() || !IsDigit(in_[pos_]))
        return Fail("invalid number: expected digit in exponent");
      while (pos_ < in_.size() && IsDigit(in_[pos_]))
        ++pos_;
    }
    const std::string_view token = in_.substr(start, pos_ - start);
    const auto res = std::from_chars(token.data(), token.data() + token.size(),
                                     out);
    if (res.ec != std::errc())
      return Fail("invalid number '" + std::string(token) + "'");
    return true;
  }

  static bool IsDigit(char c) { return c >= '0' && c <= '9'; }

  std::string_view in_;
  std::size_t pos_ = 0;
  std::string error_;
};

} // namespace

Value::Value() = default;

Value::Value(bool b) : kind_(Kind::Bool), bool_(b) {}

Value::Value(int n) : kind_(Kind::Number), number_(static_cast<double>(n)) {}

Value::Value(double n) : kind_(Kind::Number), number_(n) {}

Value::Value(std::string s) : kind_(Kind::String), string_(std::move(s)) {}

Value::Value(const char* s) : Value(std::string(s)) {}

Value::Value(std::vector<Member> object)
    : kind_(Kind::Object), object_(std::move(object)) {}

Value::Value(std::vector<Value> array)
    : kind_(Kind::Array), array_(std::move(array)) {}

bool Value::as_bool() const { return bool_; }
double Value::as_number() const { return number_; }
const std::string& Value::as_string() const { return string_; }
const std::vector<Value>& Value::as_array() const { return array_; }
const std::vector<Value::Member>& Value::as_object() const { return object_; }

const Value* Value::find(std::string_view key) const {
  if (kind_ != Kind::Object)
    return nullptr;
  for (const auto& member : object_)
    if (member.first == key)
      return &member.second;
  return nullptr;
}

std::string Value::dump() const {
  std::string out;
  dump_into(out);
  return out;
}

void Value::dump_into(std::string& out) const {
  switch (kind_) {
  case Kind::Null:
    out += "null";
    break;
  case Kind::Bool:
    out += bool_ ? "true" : "false";
    break;
  case Kind::Number: {
    char buf[32];
    const auto res =
        std::to_chars(buf, buf + sizeof(buf), number_, std::chars_format::general);
    if (res.ec != std::errc())
      throw std::runtime_error(
          "kernscope::db: cannot serialize a non-finite number");
    out.append(buf, res.ptr);
    break;
  }
  case Kind::String:
    AppendEscaped(out, string_);
    break;
  case Kind::Array:
    out.push_back('[');
    for (std::size_t i = 0; i < array_.size(); ++i) {
      if (i != 0)
        out.push_back(',');
      array_[i].dump_into(out);
    }
    out.push_back(']');
    break;
  case Kind::Object: {
    out.push_back('{');
    std::vector<Member> sorted = object_;
    std::sort(sorted.begin(), sorted.end(),
              [](const Member& a, const Member& b) { return a.first < b.first; });
    for (std::size_t i = 0; i < sorted.size(); ++i) {
      if (i != 0)
        out.push_back(',');
      AppendEscaped(out, sorted[i].first);
      out.push_back(':');
      sorted[i].second.dump_into(out);
    }
    out.push_back('}');
    break;
  }
  }
}

std::optional<Value> Value::parse(std::string_view text, std::string& error) {
  Parser p(text);
  Value v;
  if (!p.ParseValue(v, 0)) {
    error = p.Error();
    return std::nullopt;
  }
  p.SkipWs();
  if (!p.AtEnd()) {
    error = "trailing content after value at byte " +
            std::to_string(p.Position());
    return std::nullopt;
  }
  return v;
}

} // namespace json

namespace {

json::Value RecordToJson(const Record& r) {
  return json::object({
      {"schema_version", json::Value(r.schema_version)},
      {"gpu_arch", json::Value(r.gpu_arch)},
      {"kernel_family", json::Value(r.kernel_family)},
      {"shape_hash", json::Value(r.shape_hash)},
      {"config", r.config},
      {"timings", json::object({{"median", json::Value(r.timings.median)},
                                {"p20", json::Value(r.timings.p20)},
                                {"p80", json::Value(r.timings.p80)}})},
      {"provenance",
       json::object({{"timestamp", json::Value(r.provenance.timestamp)},
                     {"tool_version", json::Value(r.provenance.tool_version)},
                     {"mode", json::Value(r.provenance.mode)}})},
  });
}

bool GetString(const json::Value& obj, std::string_view key, std::string& out,
               std::string& error) {
  const json::Value* v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::String) {
    error = "field \"" + std::string(key) + "\" is not a string";
    return false;
  }
  out = v->as_string();
  return true;
}

bool GetNumber(const json::Value& obj, std::string_view key, double& out,
               std::string& error) {
  const json::Value* v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::Number) {
    error = "field \"" + std::string(key) + "\" is not a number";
    return false;
  }
  out = v->as_number();
  return true;
}

bool GetObject(const json::Value& obj, std::string_view key,
               const json::Value*& out, std::string& error) {
  const json::Value* v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::Object) {
    error = "field \"" + std::string(key) + "\" is not an object";
    return false;
  }
  out = v;
  return true;
}

// Parses the (already version-gated) record object into a Record. All fields
// are required; provenance.mode must be "measured" or "projected".
bool RecordFromJson(const json::Value& obj, Record& r, std::string& error) {
  const json::Value* sv = obj.find("schema_version");
  if (sv == nullptr || sv->kind() != json::Kind::Number) {
    error = "missing or non-numeric \"schema_version\"";
    return false;
  }
  const double d = sv->as_number();
  if (d != std::floor(d)) {
    error = "\"schema_version\" must be an integer";
    return false;
  }
  r.schema_version = static_cast<int>(d);
  if (!GetString(obj, "gpu_arch", r.gpu_arch, error) ||
      !GetString(obj, "kernel_family", r.kernel_family, error) ||
      !GetString(obj, "shape_hash", r.shape_hash, error))
    return false;
  const json::Value* cfg = nullptr;
  if (!GetObject(obj, "config", cfg, error))
    return false;
  r.config = *cfg;
  const json::Value* timings = nullptr;
  if (!GetObject(obj, "timings", timings, error))
    return false;
  if (!GetNumber(*timings, "median", r.timings.median, error) ||
      !GetNumber(*timings, "p20", r.timings.p20, error) ||
      !GetNumber(*timings, "p80", r.timings.p80, error))
    return false;
  const json::Value* prov = nullptr;
  if (!GetObject(obj, "provenance", prov, error))
    return false;
  if (!GetString(*prov, "timestamp", r.provenance.timestamp, error) ||
      !GetString(*prov, "tool_version", r.provenance.tool_version, error) ||
      !GetString(*prov, "mode", r.provenance.mode, error))
    return false;
  if (r.provenance.mode != "measured" && r.provenance.mode != "projected") {
    error = "\"provenance.mode\" must be \"measured\" or \"projected\" (got \"" +
            r.provenance.mode + "\")";
    return false;
  }
  return true;
}

// JSON has no NaN/Inf literal; reject them so save() can never emit them.
bool AllNumbersFinite(const json::Value& v) {
  switch (v.kind()) {
  case json::Kind::Number:
    return std::isfinite(v.as_number());
  case json::Kind::Array:
    for (const auto& e : v.as_array())
      if (!AllNumbersFinite(e))
        return false;
    return true;
  case json::Kind::Object:
    for (const auto& member : v.as_object())
      if (!AllNumbersFinite(member.second))
        return false;
    return true;
  default:
    return true;
  }
}

bool IsBlank(std::string_view line) {
  for (const char c : line)
    if (c != ' ' && c != '\t')
      return false;
  return true;
}

} // namespace

Database::Key Database::make_key(const Record& record) {
  return Key{record.gpu_arch, record.kernel_family, record.shape_hash,
             record.config.dump()};
}

void Database::put(Record record) {
  if (record.config.kind() != json::Kind::Object)
    throw std::invalid_argument(
        "kernscope::db: record.config must be a JSON object");
  if (!std::isfinite(record.timings.median) ||
      !std::isfinite(record.timings.p20) || !std::isfinite(record.timings.p80))
    throw std::invalid_argument(
        "kernscope::db: record.timings must be finite numbers");
  if (!AllNumbersFinite(record.config))
    throw std::invalid_argument(
        "kernscope::db: record.config must contain only finite numbers");
  const Key key = make_key(record);
  const auto [it, inserted] = index_.emplace(key, records_.size());
  if (inserted) {
    records_.push_back(std::move(record));
  } else {
    records_[it->second] = std::move(record);
  }
}

std::optional<Record> Database::query_best(std::string_view gpu_arch,
                                           std::string_view kernel_family,
                                           std::string_view shape_hash) const {
  // Keys sharing the query prefix are contiguous in map order (config is the
  // last component), so one lower_bound spans exactly the matching records.
  const Key prefix{std::string(gpu_arch), std::string(kernel_family),
                   std::string(shape_hash), ""};
  const Record* best = nullptr;
  for (auto it = index_.lower_bound(prefix); it != index_.end(); ++it) {
    const Key& key = it->first;
    if (key.gpu_arch != gpu_arch || key.kernel_family != kernel_family ||
        key.shape_hash != shape_hash)
      break;
    const Record& r = records_[it->second];
    if (best == nullptr || r.timings.median < best->timings.median)
      best = &r;
  }
  if (best == nullptr)
    return std::nullopt;
  return *best;
}

bool Database::load(const std::string& path, std::vector<LoadWarning>& warnings) {
  std::ifstream in(path);
  if (!in)
    return false;
  std::string line;
  std::size_t lineno = 0;
  while (std::getline(in, line)) {
    ++lineno;
    if (!line.empty() && line.back() == '\r')
      line.pop_back(); // CRLF tolerance
    if (IsBlank(line))
      continue; // blank lines are tolerated, not reported

    std::string error;
    auto parsed = json::Value::parse(line, error);
    if (!parsed) {
      warnings.push_back(
          {lineno, "line " + std::to_string(lineno) +
                       ": malformed JSON line skipped: " + error});
      continue;
    }
    if (parsed->kind() != json::Kind::Object) {
      warnings.push_back(
          {lineno, "line " + std::to_string(lineno) +
                       ": line is not a JSON object - skipped"});
      continue;
    }
    const json::Value* sv = parsed->find("schema_version");
    if (sv == nullptr || sv->kind() != json::Kind::Number) {
      warnings.push_back(
          {lineno, "line " + std::to_string(lineno) +
                       ": record rejected: missing or non-numeric "
                       "\"schema_version\""});
      continue;
    }
    const int schema = static_cast<int>(sv->as_number());
    if (schema > kCurrentSchemaVersion) {
      warnings.push_back(
          {lineno, "line " + std::to_string(lineno) + ": record schema_version " +
                       std::to_string(schema) + " > current " +
                       std::to_string(kCurrentSchemaVersion) +
                       " - forward-incompatible, skipped"});
      continue;
    }
    Record r;
    if (!RecordFromJson(*parsed, r, error)) {
      warnings.push_back(
          {lineno, "line " + std::to_string(lineno) + ": record rejected: " +
                       error});
      continue;
    }
    put(std::move(r));
  }
  return true;
}

bool Database::save(const std::string& path) const {
  std::ofstream out(path);
  if (!out)
    return false;
  for (const Record& r : records_) {
    out << RecordToJson(r).dump() << '\n';
    if (!out)
      return false;
  }
  return out.good();
}

} // namespace kernscope::db
