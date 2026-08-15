//===- bench_driver.cpp - todo-42 showcase bench on gfx1101 ----------------===//
//
// The AMD analog of the todo-42 artifact pipeline's measured half (Metis
// L-6 venue rule: NVIDIA timings are rental-gated; the local measured
// numbers are the AMD side of the same kernels). Each showcase kernel is
// compiled by hipcc --genco (offload-arch=gfx1101) from the SAME portable
// source the clang pipeline consumed, loaded through the todo-22 harness
// (libs/gpuruntime + libs/bench), timed with the validated methodology
// (time-based warmup/rep, L2 flush, event timing, adaptive trials), and
// recorded into libs/db JSONL WITH PROVENANCE — the
// reports/peaks/gfx1101-bench.jsonl format mirrored: device name,
// gcnArchName, rocm-smi clocks as-measured, tool version, measured=true.
// Clocks are NEVER locked (no root) — recorded as-measured.
//
//   usage: showcase-bench --hsaco-dir <dir> [--db <path>]
//
// Exit codes: 0 = all three measured + records written; 2 = environment
// error (no live HIP device, module load failure).
//
// libs/* are REUSED (headers + sources compiled here), never modified.

#include "kernscope/arch.h"
#include "kernscope/bench.h"
#include "kernscope/db.h"
#include "kernscope/gpu.h"
#include "kernscope/version.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#ifndef KERNSCOPE_BENCH_HSACO_DIR
#define KERNSCOPE_BENCH_HSACO_DIR "."
#endif
#ifndef KERNSCOPE_BENCH_DB
#define KERNSCOPE_BENCH_DB "reports/passes/bench/gfx1101-showcase-bench.jsonl"
#endif

namespace {

using kernscope::Backend;
using kernscope::GpuError;
using kernscope::GpuRuntime;
using kernscope::GpuStatus;
using kernscope::Kernel;
using kernscope::KernelArg;
using kernscope::LaunchConfig;
using kernscope::Module;
using kernscope::create_runtime;
using kernscope::bench::Options;
using kernscope::bench::Result;
using kernscope::bench::run;
using kernscope::db::Database;
using kernscope::db::Provenance;
using kernscope::db::Record;
using kernscope::db::Timings;
using namespace kernscope::db::json; // NOLINT

std::string IsoTimestampUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[24];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return buf;
}

std::string IsoDateUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900,
                tm.tm_mon + 1, tm.tm_mday);
  return buf;
}

// rocm-smi clocks as-measured (never locked — no root).
std::string ReadClocks() {
  std::string out;
  std::array<char, 256> line;
  std::unique_ptr<std::FILE, decltype(&::pclose)> pipe(
      ::popen("rocm-smi --showclocks 2>/dev/null", "r"), &::pclose);
  if (pipe) {
    while (std::fgets(line.data(), static_cast<int>(line.size()), pipe.get()) !=
           nullptr) {
      out += line.data();
    }
  }
  return out;
}

struct BenchSpec {
  const char *family;      // kernel_family (libs/db key component)
  const char *shape_hash;  // workload key
  const char *kernel_name; // hsaco symbol
  const char *hsaco;       // module file
  LaunchConfig cfg;
  std::uint32_t arg_count;
};

struct DeviceState {
  std::string name;
  std::string gcn_arch;
  std::string gcn_arch_full;
  std::string clocks;
};

DeviceState QueryDevice(GpuRuntime &rt) {
  DeviceState S;
  auto devs = rt.list_devices();
  if (!devs.ok() || devs.value.empty()) {
    std::fprintf(stderr, "error: no live HIP device: %s\n",
                 devs.message.c_str());
    std::exit(2);
  }
  S.name = devs.value.front().name;
  auto props = rt.device_properties(S.name);
  if (props.ok()) {
    S.gcn_arch = props.value.gcn_arch;
    S.gcn_arch_full = props.value.gcn_arch_full;
  }
  S.clocks = ReadClocks();
  return S;
}

int BenchOne(GpuRuntime &rt, const DeviceState &Dev, Database &Db,
             const std::string &HsacoDir, const BenchSpec &Spec,
             const std::vector<KernelArg> &Args,
             const std::vector<std::uint8_t> & /*InputBytes*/) {
  const std::string Path = HsacoDir + "/" + Spec.hsaco;
  auto mod = Module::load(rt, Path);
  if (!mod.ok()) {
    std::fprintf(stderr, "error: load %s: %s\n", Path.c_str(),
                 mod.message.c_str());
    return 2;
  }
  auto kernel = Kernel::create(rt, *mod.value, Spec.kernel_name);
  if (!kernel.ok()) {
    std::fprintf(stderr, "error: kernel %s in %s: %s\n", Spec.kernel_name,
                 Path.c_str(), kernel.message.c_str());
    return 2;
  }

  Options Opts;  // todo-22 defaults: 25 ms warmup / 100 ms reps, L2 flush,
                 // adaptive trials (5% noise, min 10 samples).
  const Result R = run(rt, Dev.name, [&]() {
    return kernel.value->launch(Spec.cfg, Args);
  });
  if (R.error != GpuError::kOk) {
    std::fprintf(stderr, "error: bench %s: %s\n", Spec.family,
                 R.error_message.c_str());
    return 1;
  }

  Record Rec;
  Rec.gpu_arch = Dev.gcn_arch;
  Rec.kernel_family = Spec.family;
  Rec.shape_hash = Spec.shape_hash;
  Rec.config = ::kernscope::db::json::object({
      {"grid_x", ::kernscope::db::json::Value(static_cast<int>(Spec.cfg.grid_x))},
      {"grid_y", ::kernscope::db::json::Value(static_cast<int>(Spec.cfg.grid_y))},
      {"block_x", ::kernscope::db::json::Value(static_cast<int>(Spec.cfg.block_x))},
      {"block_y", ::kernscope::db::json::Value(static_cast<int>(Spec.cfg.block_y))},
      {"device_name", ::kernscope::db::json::Value(Dev.name)},
      {"gcn_arch", ::kernscope::db::json::Value(Dev.gcn_arch)},
      {"gcn_arch_full", ::kernscope::db::json::Value(Dev.gcn_arch_full)},
      {"date", ::kernscope::db::json::Value(IsoDateUtc())},
      {"measured", ::kernscope::db::json::Value(true)},
      {"clocks", ::kernscope::db::json::Value(Dev.clocks)},
  });
  Rec.timings = Timings{R.median_ms, R.p20_ms, R.p80_ms};
  Rec.provenance = Provenance{
      IsoTimestampUtc(), "kernscope-pass-bench " KERNSCOPE_VERSION_STRING,
      "measured"};
  Db.put(Rec);

  std::printf("showcase-bench %-11s %-20s median %.4f ms (p20 %.4f, p80 "
              "%.4f, %d reps, stdev %.3f)\n",
              Spec.family, Spec.shape_hash, R.median_ms, R.p20_ms, R.p80_ms,
              R.n_reps, R.relative_stdev);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  std::string HsacoDir = KERNSCOPE_BENCH_HSACO_DIR;
  std::string DbPath = KERNSCOPE_BENCH_DB;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--hsaco-dir") == 0 && i + 1 < argc) {
      HsacoDir = argv[++i];
    } else if (std::strcmp(argv[i], "--db") == 0 && i + 1 < argc) {
      DbPath = argv[++i];
    } else if (std::strcmp(argv[i], "--version") == 0) {
      std::printf("showcase-bench " KERNSCOPE_VERSION_STRING "\n");
      return 0;
    }
  }

  auto rt = create_runtime(Backend::kHip);
  if (!rt.ok()) {
    std::fprintf(stderr, "error: HIP runtime unavailable: %s\n",
                 rt.message.c_str());
    return 2;
  }
  const DeviceState Dev = QueryDevice(*rt.value);
  if (Dev.gcn_arch != "gfx1101") {
    std::fprintf(stderr, "error: expected gfx1101, got '%s'\n",
                 Dev.gcn_arch.c_str());
    return 2;
  }

  Database Db;
  std::vector<kernscope::db::LoadWarning> Warnings;
  if (std::filesystem::exists(DbPath)) {
    if (!Db.load(DbPath, Warnings)) {
      std::fprintf(stderr, "error: cannot read %s\n", DbPath.c_str());
      return 2;
    }
  }

  // Buffers (sizes in elements; the kernels read/write these).
  // gemm: C[64][64] = A[64][100] x B[100][64] — one output element/thread.
  auto a = rt.value->device_malloc(64 * 100 * 4);
  auto b = rt.value->device_malloc(100 * 64 * 4);
  auto c = rt.value->device_malloc(64 * 64 * 4);
  // softmax: x/y rows x cols (64 x 64), n = cols.
  auto x = rt.value->device_malloc(64 * 64 * 4);
  auto y = rt.value->device_malloc(64 * 64 * 4);
  if (!a.ok() || !b.ok() || !c.ok() || !x.ok() || !y.ok()) {
    std::fprintf(stderr, "error: device_malloc failed\n");
    return 2;
  }
  const auto Zeros = std::vector<std::uint8_t>(4096, 0);
  for (const auto &Buf : {a.value, b.value, c.value, x.value, y.value}) {
    rt.value->memset_device(Buf, 0, 4096);
  }

  const std::uint64_t a_p = a.value.value, b_p = b.value.value,
                      c_p = c.value.value, x_p = x.value.value,
                      y_p = y.value.value;
  const int softmax_n = 64;

  int Exit = 0;

  {
    const BenchSpec Spec{"showcase_gemm", "M=64,N=64,K=100",
                         "gemm_kernel", "gemm.hsaco",
                         LaunchConfig{4, 4, 1, 16, 16, 1, 0}, 3};
    const std::vector<KernelArg> Args = {
        KernelArg{&a_p, 8}, KernelArg{&b_p, 8}, KernelArg{&c_p, 8}};
    Exit |= BenchOne(*rt.value, Dev, Db, HsacoDir, Spec, Args, Zeros);
  }
  {
    const BenchSpec Spec{"showcase_layernorm", "rows=64,cols=64",
                         "layernorm_kernel", "layernorm.hsaco",
                         LaunchConfig{1, 1, 1, 64, 1, 1, 0}, 0};
    Exit |= BenchOne(*rt.value, Dev, Db, HsacoDir, Spec, {}, Zeros);
  }
  {
    const BenchSpec Spec{"showcase_softmax", "rows=64,cols=64",
                         "softmax_kernel", "softmax.hsaco",
                         LaunchConfig{1, 1, 1, 64, 1, 1, 0}, 3};
    const std::vector<KernelArg> Args = {
        KernelArg{&x_p, 8}, KernelArg{&y_p, 8},
        KernelArg{&softmax_n, sizeof(softmax_n)}};
    Exit |= BenchOne(*rt.value, Dev, Db, HsacoDir, Spec, Args, Zeros);
  }

  if (!Db.save(DbPath)) {
    std::fprintf(stderr, "error: cannot write %s\n", DbPath.c_str());
    return 2;
  }
  std::printf("showcase-bench: %zu records written to %s (gfx1101, "
              "measured)\n",
              Db.size(), DbPath.c_str());
  return Exit;
}
