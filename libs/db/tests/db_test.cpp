//===- db_test.cpp - tests for the JSONL benchmark/config DB ----*- C++ -*-===//
//
// gtest coverage for plan todo 8 acceptance criteria:
//   (a) 1000-record roundtrip (put all, save, load fresh) < 1s;
//   (b) query_best returns the config with the minimal median;
//   (c) hand-corrupted middle line -> skipped, count == N-1, warning names
//       the line number (asserted);
//   (d) schema-version skew (schema_version 99) -> skipped with warning,
//       count reflects it.
//
//===----------------------------------------------------------------------===//

#include "kernscope/db.h"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace db = kernscope::db;
namespace fs = std::filesystem;

namespace {

fs::path TempPath(const std::string& name) {
  return fs::temp_directory_path() /
         ("kernscope_db_test_" + name + "_" + std::to_string(::getpid()) +
          ".jsonl");
}

db::Record MakeRecord(int i, double median) {
  db::Record r;
  r.gpu_arch = (i % 2 == 0) ? "gfx1101" : "sm_90";
  r.kernel_family = "matmul";
  r.shape_hash = "sha256:rec" + std::to_string(i);
  r.config = db::json::object({{"BM", 128}, {"BN", 128}, {"BK", 32}, {"unroll", i}});
  r.timings.median = median;
  r.timings.p20 = median * 0.9;
  r.timings.p80 = median * 1.1;
  r.provenance.timestamp = "2026-08-14T00:00:00Z";
  r.provenance.tool_version = "0.1.0-test";
  r.provenance.mode = "measured";
  return r;
}

db::Record MakeRecordWith(const char* arch, const char* family,
                          const char* shape, double bk, double median) {
  db::Record r;
  r.gpu_arch = arch;
  r.kernel_family = family;
  r.shape_hash = shape;
  r.config = db::json::object({{"BM", 128}, {"BN", 128}, {"BK", bk}});
  r.timings.median = median;
  r.timings.p20 = median * 0.9;
  r.timings.p80 = median * 1.1;
  r.provenance.timestamp = "2026-08-14T00:00:00Z";
  r.provenance.tool_version = "0.1.0-test";
  r.provenance.mode = "measured";
  return r;
}

// Committed 5-record clean fixture (hand-written, non-canonical member order
// on purpose: the loader must be member-order independent).
const fs::path kFixture =
    fs::path(__FILE__).parent_path() / "fixtures" / "benchmarks.jsonl";

} // namespace

// (a) 1000-record roundtrip under one second, plus the dedupe/replace policy.
TEST(DbTest, Roundtrip1000RecordsUnderOneSecond) {
  constexpr int kCount = 1000;
  const auto t0 = std::chrono::steady_clock::now();

  db::Database store;
  for (int i = 0; i < kCount; ++i)
    store.put(MakeRecord(i, 0.4 + (i % 7) * 0.01));

  // Dedupe/replace: re-putting an identical (arch, family, shape, config)
  // key must not grow the store, and must update the record in place.
  store.put(MakeRecord(0, 0.999));
  ASSERT_EQ(store.size(), static_cast<std::size_t>(kCount));

  const fs::path path = TempPath("roundtrip");
  ASSERT_TRUE(store.save(path.string()));

  db::Database fresh;
  std::vector<db::LoadWarning> warnings;
  ASSERT_TRUE(fresh.load(path.string(), warnings));

  const auto t1 = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(t1 - t0).count();
  fs::remove(path);

  EXPECT_EQ(fresh.size(), static_cast<std::size_t>(kCount));
  EXPECT_TRUE(warnings.empty());
  EXPECT_LT(elapsed, 1.0) << "1000-record roundtrip took " << elapsed << "s";
  // Round-trip fidelity, including the replaced record.
  EXPECT_EQ(fresh.records()[0].timings.median, 0.999);
  EXPECT_DOUBLE_EQ(fresh.records()[0].timings.median,
                   store.records()[0].timings.median);
  EXPECT_EQ(fresh.records()[999].shape_hash, "sha256:rec999");
}

// (b) query_best returns the config with the minimal median.
TEST(DbTest, QueryBestReturnsConfigWithMinimalMedian) {
  db::Database store;
  const char* arch = "sm_90";
  const char* fam = "matmul";
  const char* shape = "sha256:matmul_4096";
  store.put(MakeRecordWith(arch, fam, shape, 128, 1.20));
  store.put(MakeRecordWith(arch, fam, shape, 256, 0.90));
  store.put(MakeRecordWith(arch, fam, shape, 64, 0.35)); // minimal median
  store.put(MakeRecordWith(arch, fam, shape, 32, 2.00));
  store.put(MakeRecordWith(arch, fam, shape, 512, 0.60));
  // A different arch with a tiny median must not leak into the query.
  store.put(MakeRecordWith("gfx1101", fam, shape, 64, 0.001));

  const auto best = store.query_best(arch, fam, shape);
  ASSERT_TRUE(best.has_value());
  EXPECT_DOUBLE_EQ(best->config.find("BK")->as_number(), 64);
  EXPECT_DOUBLE_EQ(best->timings.median, 0.35);

  EXPECT_FALSE(store.query_best("sm_100", fam, shape).has_value());
}

// (c) hand-corrupted middle line: skipped, count == N-1, warning names the
// line number; records before AND after the corruption survive.
TEST(DbTest, CorruptMiddleLineSkippedAndWarningNamesLine) {
  ASSERT_TRUE(fs::exists(kFixture)) << "missing fixture: " << kFixture;
  const fs::path path = TempPath("corrupt");
  {
    std::ifstream in(kFixture);
    ASSERT_TRUE(in.good());
    std::ofstream out(path);
    std::string line;
    int lineno = 0;
    while (std::getline(in, line)) {
      ++lineno;
      if (lineno == 3) {
        out << "this line is deliberately not json {{{" << '\n';
      } else {
        out << line << '\n';
      }
    }
    out << '\n'; // trailing blank line must be tolerated, not reported
  }

  db::Database store;
  std::vector<db::LoadWarning> warnings;
  ASSERT_TRUE(store.load(path.string(), warnings));
  fs::remove(path);

  EXPECT_EQ(store.size(), 4u); // N=5 fixture records, one skipped -> N-1
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].line, 3u); // the corrupted line's number is named
  EXPECT_NE(warnings[0].message.find("line 3"), std::string::npos);
  // Records before and after the corruption survive, in order.
  ASSERT_EQ(store.records().size(), 4u);
  EXPECT_EQ(store.records()[0].shape_hash, "sha256:3f4a"); // fixture line 1
  EXPECT_EQ(store.records()[2].shape_hash, "sha256:c0a8"); // fixture line 4
  EXPECT_EQ(store.records()[3].shape_hash, "sha256:51f7"); // fixture line 5
}

// (d) schema-version skew: a schema_version 99 record is skipped with a
// warning, and the loaded count reflects it.
TEST(DbTest, NewerSchemaVersionSkippedWithWarning) {
  ASSERT_TRUE(fs::exists(kFixture));
  const fs::path path = TempPath("skew");
  {
    std::ifstream in(kFixture);
    std::ofstream out(path);
    std::string line;
    while (std::getline(in, line))
      out << line << '\n'; // 5 valid fixture records
    out << R"({"schema_version": 99, "gpu_arch": "sm_90", "kernel_family": "future_kernel", "shape_hash": "sha256:future", "config": {}, "timings": {"median": 0.1, "p20": 0.09, "p80": 0.11}, "provenance": {"timestamp": "2026-08-14T00:00:00Z", "tool_version": "9.9", "mode": "measured"}})"
        << '\n';
  }

  db::Database store;
  std::vector<db::LoadWarning> warnings;
  ASSERT_TRUE(store.load(path.string(), warnings));
  fs::remove(path);

  EXPECT_EQ(store.size(), 5u); // skew record skipped; count reflects it
  ASSERT_EQ(warnings.size(), 1u);
  EXPECT_EQ(warnings[0].line, 6u); // fixture has 5 lines; skew is line 6
  EXPECT_NE(warnings[0].message.find("schema_version"), std::string::npos);
  EXPECT_NE(warnings[0].message.find("99"), std::string::npos);
}
