//===- tune_driver_test.cpp - tuning driver acceptance suite ----------------===//
//
// kernscope tools/tune todo 28. Two compile configurations share this source
// (see tools/tune/CMakeLists.txt):
//
//   * no-gpu (LABELS no-gpu, no KERNSCOPE_TUNE_LIVE): the negative test —
//     a min-occupancy floor of 1.0 makes every config infeasible at the
//     GPU-free pre-screen; the driver must exit with "no viable config" and a
//     JSON reason, never crash, and never touch the GPU.
//   * gpu-amd (LABELS gpu-amd, KERNSCOPE_TUNE_LIVE): the LIVE tune — matmul
//     512^3 bf16 on gfx1101 over the 24-config auto grid:
//       - JSON evidence shows pre-screen rejections >= 1 (the 128x128
//         register/smem over-budget configs);
//       - a test-hook-injected variant (rewrite_source) that compiles but
//         writes garbage is DISCARDED by the golden gate and recorded
//         (golden_gate_failed == 1);
//       - the best config is selected with median == min of the measured
//         medians;
//       - the driver writes libs/db records queryable after the run
//         (query_best == the selected median);
//       - re-running with the same tunecache compiles 0 new variants (cache
//         hit; the winner is re-picked without re-benchmarking).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/driver.h"

#include "kernscope/db.h"

#include <gtest/gtest.h>

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>

namespace kernscope::tune {
namespace {

DriverOptions matmul_options(const std::string &base_dir, const char *tag) {
  DriverOptions o;
  o.kernel_family = "matmul";
  o.shape = {512, 512, 512};
  o.arch = "gfx1101";
  o.grid_preset = "auto";
  o.db_path = base_dir + "/" + tag + ".jsonl";
  o.cache_path = base_dir + "/" + tag + ".tunecache.json";
  o.workdir = base_dir + "/work";
  return o;
}

std::string make_workdir(const char *tag) {
  static int counter = 0;
  const std::string dir = testing::TempDir() + "/tune_driver_" + tag + "_" +
                          std::to_string(::getpid()) + "_" +
                          std::to_string(++counter);
  std::error_code ec;
  std::filesystem::create_directories(dir, ec);
  EXPECT_FALSE(ec) << "cannot create workdir " << dir;
  return dir;
}

#ifdef KERNSCOPE_TUNE_LIVE

// A source that compiles under hipcc --genco for gfx1101 and exports the
// fixed matmul entry symbol, but writes only zeros to a handful of elements —
// the golden gate MUST fail (verdict not pass:true) and the driver MUST
// discard the variant + record the failure.
const char *kSabotageSource = R"(// injected golden-gate-failing variant (test hook)
#include "kernscope/kernel_traits.h"
extern "C" __global__ void kernscope_tune_matmul(
    const ks_bf16* a, const ks_bf16* b, ks_bf16* out, int m, int n, int k) {
  const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
  if (idx < m * n) out[idx] = ks_float_to_bf16(0.0f);
}
)";

// --- gpu-amd: the full live tune --------------------------------------------

TEST(TuneDriverLive, Matmul512CubeGfx1101CorrectnessGatedTune) {
  const std::string base = make_workdir("live");
  DriverOptions opts = matmul_options(base, "live");

  // Inject exactly one bad variant: the first config that reaches the
  // compile step gets its source replaced by the zero-writing kernel.
  DriverHooks hooks;
  hooks.rewrite_source = [](const DriverOptions &, const db::json::Value &,
                            const std::string &) -> std::string {
    static int injected = 0;
    return injected++ == 0 ? std::string(kSabotageSource) : std::string();
  };

  const DriverResult r = tune(opts, &hooks);
  ASSERT_EQ(r.status, DriverStatus::kOk)
      << "live tune must select a best config: " << r.message;
  ASSERT_TRUE(r.has_best);

  // Grid >= 20 configs (the acceptance floor).
  EXPECT_GE(r.variants_total, 20);
  // Pre-screen rejections >= 1 (128x128 register/smem over-budget configs).
  EXPECT_GE(r.prescreen_rejected, 1);
  // The injected bad variant was discarded by the golden gate + recorded.
  EXPECT_EQ(r.golden_failed, 1)
      << "the injected bad variant must be discarded by the golden gate";
  EXPECT_GT(r.benchmarked, 0) << "at least one variant must be measured";
  EXPECT_EQ(r.compiled, r.benchmarked + r.golden_failed + r.bench_failed)
      << "everything that compiled went to the gate or the bench";

  // Best selected with median == min of the measured medians.
  ASSERT_FALSE(r.measured_medians.empty());
  const double min_measured =
      *std::min_element(r.measured_medians.begin(), r.measured_medians.end());
  EXPECT_DOUBLE_EQ(r.best_median_ms, min_measured)
      << "best config median must equal the min measured median";

  // The JSON evidence records the pre-screen rejections + the gate discard.
  std::string jerr;
  const auto evidence = db::json::Value::parse(r.evidence_json, jerr);
  if (!evidence) {
    FAIL() << "evidence JSON must parse: " << jerr;
    return;
  }
  const auto *summary = evidence->find("summary");
  if (summary == nullptr || summary->kind() != db::json::Kind::Object) {
    FAIL() << "summary object missing";
    return;
  }
  const auto *prescreen = summary->find("prescreen_rejected");
  const auto *gate_fail = summary->find("golden_gate_failed");
  if (prescreen == nullptr || gate_fail == nullptr) {
    FAIL() << "summary counters missing";
    return;
  }
  EXPECT_GE(prescreen->as_number(), 1.0);
  EXPECT_EQ(gate_fail->as_number(), 1.0);
  const auto *best = evidence->find("best");
  if (best == nullptr) {
    FAIL() << "evidence must carry the selected best";
    return;
  }
  EXPECT_TRUE(evidence->find("cache_key"));
  EXPECT_TRUE(evidence->find("roofline"));
  EXPECT_TRUE(evidence->find("shape_hash"));

  // The driver wrote libs/db records; query_best must return the selected
  // config's median.
  kernscope::db::Database db;
  std::vector<kernscope::db::LoadWarning> warnings;
  ASSERT_TRUE(db.load(opts.db_path, warnings))
      << "db file must exist and load: " << opts.db_path;
  const auto db_best =
      db.query_best("gfx1101", "matmul", r.shape_hash);
  if (!db_best.has_value()) {
    FAIL() << "db must hold a (gfx1101, matmul, shape_hash) record";
    return;
  }
  EXPECT_DOUBLE_EQ(db_best.value().timings.median, r.best_median_ms)
      << "query_best must agree with the driver's selection";

  // Cache hit on re-run: the same (arch, family, shape) + tunecache must
  // compile 0 new variants and re-pick the same winner from the cache.
  const DriverResult r2 = tune(opts, /*hooks=*/nullptr);
  ASSERT_EQ(r2.status, DriverStatus::kOk) << r2.message;
  EXPECT_EQ(r2.compiled, 0) << "second run must compile 0 new variants";
  EXPECT_EQ(r2.cached, r2.variants_total)
      << "every config must be a cache hit on the second run";
  ASSERT_TRUE(r2.has_best);
  EXPECT_DOUBLE_EQ(r2.best_median_ms, r.best_median_ms)
      << "the cached winner must match the first run's selection";
}

#else // !KERNSCOPE_TUNE_LIVE — the no-gpu negative test

// Every config is infeasible at the GPU-free pre-screen: min_occupancy = 1.0
// is unachievable, so the occupancy pre-screen rejects every config (after
// the generator's register/smem budget rejects its own share). The driver
// must exit with "no viable config", a JSON reason, and no crash — and must
// never reach the compile/gate/bench phases (no GPU touched).
TEST(TuneDriver, AllConfigsInfeasible) {
  const std::string base = make_workdir("negative");
  DriverOptions opts = matmul_options(base, "negative");
  opts.min_occupancy = 1.0;

  const DriverResult r = tune(opts, /*hooks=*/nullptr);
  ASSERT_EQ(r.status, DriverStatus::kNoViableConfig)
      << "expected 'no viable config', got: " << r.message;
  EXPECT_NE(r.message.find("no viable config"), std::string::npos) << r.message;
  EXPECT_EQ(r.prescreen_rejected, r.variants_total)
      << "every config must be pre-screen rejected";
  EXPECT_EQ(r.compiled, 0) << "no config may be compiled";
  EXPECT_EQ(r.benchmarked, 0) << "no config may be benchmarked";
  EXPECT_FALSE(r.has_best);

  std::string jerr;
  const auto evidence = db::json::Value::parse(r.evidence_json, jerr);
  if (!evidence) {
    FAIL() << "evidence JSON must parse: " << jerr;
    return;
  }
  const auto *summary = evidence->find("summary");
  if (summary == nullptr || summary->kind() != db::json::Kind::Object) {
    FAIL() << "summary object missing";
    return;
  }
  const auto *rejected = summary->find("prescreen_rejected");
  if (rejected == nullptr) {
    FAIL() << "prescreen_rejected counter missing";
    return;
  }
  EXPECT_EQ(rejected->as_number(), static_cast<double>(r.variants_total));
  // Every rejected variant carries a reason.
  const auto *variants = evidence->find("variants");
  if (variants == nullptr || variants->kind() != db::json::Kind::Array) {
    FAIL() << "variants array missing";
    return;
  }
  ASSERT_EQ(variants->as_array().size(),
            static_cast<std::size_t>(r.variants_total));
  for (const auto &v : variants->as_array()) {
    ASSERT_TRUE(v.find("reason")) << "rejected variant must carry a reason";
    EXPECT_EQ(v.find("status")->as_string(), "prescreen_rejected");
  }
}

#endif // KERNSCOPE_TUNE_LIVE

} // namespace
} // namespace kernscope::tune
