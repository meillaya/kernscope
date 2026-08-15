//===- db.h - JSONL benchmark/config database ---------------------*- C++ -*-===//
//
// kernscope benchmark/config database: an append-only JSONL store of
// per-(gpu_arch, kernel_family, shape_hash, config) tuning records with a
// best-by-median query. Design follows the Triton autotune disk cache
// (docs/research/autotuner-research-brief.md §1: key = hash over config +
// tuning key, configs_timings pairs re-picked by min timing without
// re-benchmarking) and the TVM MetaSchedule JSONDatabase (§2: newline-
// delimited JSON records keyed by workload hash, warm-start seeding,
// forward-compatible loading).
//
// Record shape (one JSON object per line; serialization is canonical, i.e.
// object members are emitted in sorted order, so identical records are
// byte-identical):
//   {
//     "schema_version": 1,
//     "gpu_arch": "sm_90",
//     "kernel_family": "matmul",
//     "shape_hash": "sha256:...",
//     "config": { ... },                    // arbitrary tuning-knob object
//     "timings": { "median": 0.412, "p20": 0.398, "p80": 0.430 },
//     "provenance": { "timestamp": "...", "tool_version": "...",
//                     "mode": "measured" }  // "measured" | "projected"
//   }
//
// Resilience contract (plan todo 8, audited by F1):
//   * malformed line       -> warning naming the 1-based line number, skipped
//   * schema_version > 1   -> warning naming the line, skipped (forward-
//                             compatible loader); loading ALWAYS continues.
//
// No SQLite / external DB by design: JSONL is portable and git-diffable
// (plan todo 8 MUST NOT).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_DB_H
#define KERNSCOPE_DB_H

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kernscope::db {

// Current schema version this library reads and writes. Loaders are
// forward-compatible: any record whose schema_version exceeds this is
// warning-skipped (with its line number) and loading continues.
inline constexpr int kCurrentSchemaVersion = 1;

// ---------------------------------------------------------------------------
// Minimal dependency-free JSON value model.
//
// WHY NOT nlohmann/json or llvm::json:
//   * nlohmann/json is NOT part of the devenv profile (verified 2026-08-14:
//     no nlohmann/json.hpp reachable from the profile; CMAKE_PREFIX_PATH
//     empty) and todo 8 MUST NOT modify devenv files, so it cannot be added
//     without violating the todo's constraint set;
//   * llvm::json (the polykernel precedent) would couple this portable,
//     diffable store to the LLVM Support library for a fixed, small record
//     envelope;
//   * the record shape is a fixed envelope around an arbitrary `config`
//     object - a self-contained ~350-line value model covers it, keeps
//     libs/db zero-dependency, and its roundtrip/diffability contract is
//     fully gtest-covered (libs/db/tests/db_test.cpp).
// ---------------------------------------------------------------------------
namespace json {

enum class Kind : std::uint8_t { Null, Bool, Number, String, Array, Object };

// Value is a recursive variant-like JSON node (vector<Value> members): its
// implicit copy/assign/dtor recursion is the design, not a bug.
// NOLINTNEXTLINE(misc-no-recursion)
class Value {
public:
  using Member = std::pair<std::string, Value>;

  Value();                     // JSON null
  Value(bool b);
  Value(int n);
  Value(double n);
  Value(std::string s);
  Value(const char* s);
  Value(std::vector<Member> object);
  Value(std::vector<Value> array);

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::Null; }
  bool as_bool() const;                    // precondition: kind() == Kind::Bool
  double as_number() const;                // precondition: kind() == Kind::Number
  const std::string& as_string() const;    // precondition: kind() == Kind::String
  const std::vector<Value>& as_array() const;    // precondition: kind() == Kind::Array
  const std::vector<Member>& as_object() const;  // precondition: kind() == Kind::Object

  // Object member lookup; nullptr when this is not an object or the key is
  // absent. Duplicate keys in parsed input: last one wins.
  const Value* find(std::string_view key) const;

  // Canonical serialization: object members are emitted in lexicographic key
  // order (recursively) and numbers in shortest round-trip form, so identical
  // logical records produce byte-identical output - the property Database's
  // (gpu_arch, kernel_family, shape_hash, config) dedupe key relies on.
  // Throws std::runtime_error if a number is non-finite (JSON has no
  // NaN/Inf; Database::put rejects such records up front).
  std::string dump() const;

  // Parses exactly one JSON value from `text`; trailing content is an error.
  // On failure returns std::nullopt and fills `error` with a byte-offset
  // description. Nesting is capped at 128 levels (adversarial-input guard).
  static std::optional<Value> parse(std::string_view text, std::string& error);

private:
  void dump_into(std::string& out) const;

  Kind kind_ = Kind::Null;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<Member> object_;
  std::vector<Value> array_;
};

// Convenience builders: object({{"BM", 128}, {"BK", 32}, ...}),
// array({Value(1.0), Value("x"), ...}).
inline Value object(std::initializer_list<Value::Member> members) {
  return Value(std::vector<Value::Member>(members.begin(), members.end()));
}
inline Value array(std::initializer_list<Value> elements) {
  return Value(std::vector<Value>(elements.begin(), elements.end()));
}

} // namespace json

// Per-record timing quantiles (all in ms; must be finite).
struct Timings {
  double median = 0.0;
  double p20 = 0.0;
  double p80 = 0.0;
};

struct Provenance {
  std::string timestamp;     // ISO-8601 UTC, e.g. "2026-08-14T04:00:00Z"
  std::string tool_version;  // producer tool version
  std::string mode;          // "measured" | "projected"
};

// One benchmark/config record; `config` carries the arbitrary tuning-knob
// object (Triton configs_timings pattern: one record per config, winner
// re-picked by median without re-benchmarking).
struct Record {
  int schema_version = kCurrentSchemaVersion;
  std::string gpu_arch;
  std::string kernel_family;
  std::string shape_hash;    // workload key (MetaSchedule JSONDatabase pattern)
  json::Value config;
  Timings timings;
  Provenance provenance;
};

// One skipped line reported by load(); `line` is 1-based and `message` also
// names it (asserted by the corrupt-line and schema-skew tests).
struct LoadWarning {
  std::size_t line = 0;
  std::string message;
};

class Database {
public:
  // Inserts `record`, or REPLACES the existing record with an identical
  // (gpu_arch, kernel_family, shape_hash, config) key, keeping its position
  // in insertion order. Rejects (throws std::invalid_argument): non-object
  // config, non-finite timings, non-finite numbers inside config.
  void put(Record record);

  // The record with the minimal `timings.median` among all records sharing
  // (gpu_arch, kernel_family, shape_hash); ties broken deterministically by
  // config key order. std::nullopt when no record matches.
  std::optional<Record> query_best(std::string_view gpu_arch,
                                   std::string_view kernel_family,
                                   std::string_view shape_hash) const;

  // Append-only JSONL reader: one JSON object per line. Blank lines are
  // ignored; a malformed line or a record with schema_version >
  // kCurrentSchemaVersion is reported in `warnings` (line number named) and
  // skipped; loading always continues. Records merge into the current
  // database (same dedupe/replace policy as put). Returns false only when
  // the file could not be opened.
  bool load(const std::string& path, std::vector<LoadWarning>& warnings);

  // Writes every record as one JSON object per line, in insertion order
  // (canonical serialization - deterministic, git-diffable output).
  // Returns false when the file could not be written.
  bool save(const std::string& path) const;

  std::size_t size() const { return records_.size(); }

  // Records in insertion order (used by tests and tooling).
  const std::vector<Record>& records() const { return records_; }

private:
  // Dedupe/replace key. config_canonical is the canonical (sorted-member)
  // serialization of the config object, so logically identical configs in
  // any member order collide - the Triton/MetaSchedule "hash the config into
  // the key" pattern, without an external hash library.
  struct Key {
    std::string gpu_arch;
    std::string kernel_family;
    std::string shape_hash;
    std::string config_canonical;
    auto operator<=>(const Key&) const = default;
  };
  static Key make_key(const Record& record);

  std::vector<Record> records_;               // insertion order
  std::map<Key, std::size_t> index_;          // key -> position in records_
};

} // namespace kernscope::db

#endif // KERNSCOPE_DB_H
