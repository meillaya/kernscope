//===- main.cpp - kernscope-bench CLI (gfx1101 roofline peak measurement) ---===//
//
// Wave-4 todo 22: measures the gfx1101 roofline peaks on this machine (RX
// 7800 XT) with the validated kernscope::bench methodology (time-based
// warmup/rep, L2 flush, event timing, adaptive trials —
// docs/research/autotuner-research-brief.md §6) and writes libs/db JSONL
// records WITH PROVENANCE: device name, gcnArchName, rocm-smi clocks as
// measured at run time, tool version, measured=true. Clocks are NEVER locked
// (no root) — they are recorded as-measured (plan todo 22 MUST NOT).
//
//   usage: kernscope-bench <command> [--arch gfx1101] [options]
//     peak-bw     DRAM bandwidth: D2D-copy + read-only sweep -> GB/s
//     peak-flops  fp32 FFMA peak -> TFLOPS
//     peak-lds    shared-memory bandwidth -> GB/s
//
//   options:
//     --arch <arch>   arch to measure (default gfx1101; the peaks.hsaco is
//                     built for gfx1101)
//     --sizes <csv>   peak-bw sizes in MiB (default 1,4,16,64,256,512)
//     --no-flush      disable the L2 flush between reps
//     --db <path>     JSONL result db (default reports/peaks/gfx1101-bench.jsonl)
//     --json          emit the summary as one JSON object
//     --version
//
// Exit codes: 0 = measured + floors passed; 1 = measured but a sanity floor
// (peak-bw > 300 GB/s, peak-flops fp32 > 10 TFLOPS) FAILED; 2 = environment
// or usage error (no live HIP device, module load failure, unknown arch).
//
//===----------------------------------------------------------------------===//

#include "kernscope/arch.h"
#include "kernscope/bench.h"
#include "kernscope/db.h"
#include "kernscope/gpu.h"
#include "kernscope/version.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef KERNSCOPE_BENCH_HSACO
#define KERNSCOPE_BENCH_HSACO ""
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

// Launch shape: 480 blocks x 256 threads over 60 CUs = 4 blocks/CU, 8 RDNA3
// wave32 waves/block = full occupancy on the RX 7800 XT.
constexpr std::uint32_t kGrid = 480;
constexpr std::uint32_t kBlock = 256;
// FFMA chains per iteration (matches libs/bench/kernels/peaks.hip).
constexpr int kFmaChains = 8;
// Small sweep sizes are L2/Infinity-Cache resident on the RX 7800 XT (6 MiB
// L2 + 64 MiB Infinity Cache), so their bandwidth is NOT a DRAM number. The
// reported peak is the max over sizes >= 2x the cache hierarchy (128 MiB),
// i.e. unambiguously DRAM-resident traffic.
constexpr std::int64_t kDramResidentFloorBytes = std::int64_t{128} << 20;

struct Clocks {
  std::string raw;      // full `rocm-smi --showclocks` output
  std::string sclk_mhz; // current sclk (as measured)
  std::string mclk_mhz; // current mclk (as measured)
};

std::string IsoDateUtc() {
  const std::time_t t = std::time(nullptr);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[16];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02d", tm.tm_year + 1900,
                tm.tm_mon + 1, tm.tm_mday);
  return buf;
}

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

// Read `rocm-smi --showclocks` and the current sclk/mclk values. Recorded
// as-measured; clocks are never locked (no root).
Clocks ReadClocks() {
  Clocks out;
  std::array<char, 256> line;
  std::unique_ptr<std::FILE, decltype(&::pclose)> pipe(
      ::popen("rocm-smi --showclocks 2>/dev/null", "r"), &::pclose);
  if (pipe) {
    std::ostringstream all;
    while (std::fgets(line.data(), static_cast<int>(line.size()), pipe.get()) !=
           nullptr) {
      all << line.data();
      const std::string s(line.data());
      // "GPU[0]		: sclk clock level: S: (36Mhz)" -> value inside parens.
      const std::size_t paren = s.find('(');
      const std::size_t close =
          s.find("Mhz", paren == std::string::npos ? 0 : paren);
      if (paren != std::string::npos && close != std::string::npos &&
          close > paren) {
        const std::string val = s.substr(paren + 1, close - paren - 1);
        if (s.find("sclk") != std::string::npos) out.sclk_mhz = val;
        if (s.find("mclk") != std::string::npos) out.mclk_mhz = val;
      }
    }
    out.raw = all.str();
  }
  return out;
}

// One measured result (family + quantiles + derived peak value).
struct Measurement {
  std::string family;          // "peak_bw_d2d" | "peak_bw_read" | "peak_flops" |
                               // "peak_lds"
  std::string label;           // human label, e.g. "D2D copy 256 MiB"
  std::int64_t size_bytes = 0; // traffic counted for this measurement
  double median_ms = 0.0;
  double p20_ms = 0.0;
  double p80_ms = 0.0;
  double value = 0.0; // GB/s (bw/lds) or TFLOPS (flops)
  bool timed_out = false;
};

struct Peaks {
  double max_bw_gbps = 0.0; // best of D2D + read sweep
  double max_tflops = 0.0;
  double lds_gbps = 0.0;
};

class Runner {
public:
  Runner(std::string device, std::string gcn_full, std::string db_path)
      : device_(std::move(device)), gcn_full_(std::move(gcn_full)),
        db_path_(std::move(db_path)) {}

  const Clocks &clocks() const { return clocks_; }
  void set_clocks(Clocks c) { clocks_ = std::move(c); }
  const std::string &device() const { return device_; }

  // One harness measurement of `kernel`; returns the measured Result.
  Result Measure(GpuRuntime &rt, Kernel &kernel, const LaunchConfig &cfg,
                 const std::vector<KernelArg> &args, const Options &opts) const {
    return run(rt, device_, [&]() { return kernel.launch(cfg, args); }, opts);
  }

  void Record(const Measurement &m) {
    using kernscope::db::json::Value;
    Value config = Value(std::vector<Value::Member>{
        {"size_bytes", Value(static_cast<double>(m.size_bytes))},
        {"device_name", Value(device_)},
        {"gcn_arch", Value("gfx1101")},
        {"gcn_arch_full", Value(gcn_full_)},
        {"date", Value(IsoDateUtc())},
        {"clocks", Value(clocks_.raw)},
        {"sclk_mhz", Value(clocks_.sclk_mhz)},
        {"mclk_mhz", Value(clocks_.mclk_mhz)},
        {"measured", Value(true)},
    });
    kernscope::db::Record rec;
    rec.gpu_arch = "gfx1101";
    rec.kernel_family = m.family;
    rec.shape_hash = "size=" + std::to_string(m.size_bytes);
    rec.config = std::move(config);
    rec.timings = {m.median_ms, m.p20_ms, m.p80_ms};
    rec.provenance.timestamp = IsoTimestampUtc();
    rec.provenance.tool_version = "kernscope-bench " KERNSCOPE_VERSION_STRING;
    rec.provenance.mode = "measured";
    db_.put(std::move(rec));
  }

  // Merge + write all records into the JSONL db (append-only; re-runs merge
  // on the (gpu_arch, kernel_family, shape_hash) key).
  void Flush() {
    const std::filesystem::path dir =
        std::filesystem::path(db_path_).parent_path();
    if (!dir.empty() && !std::filesystem::exists(dir)) {
      std::error_code ec;
      std::filesystem::create_directories(dir, ec);
      if (ec) {
        std::fprintf(stderr, "error: create %s: %s\n", dir.c_str(),
                     ec.message().c_str());
        std::exit(2);
      }
    }
    kernscope::db::Database merged;
    std::vector<kernscope::db::LoadWarning> warnings;
    if (std::filesystem::exists(db_path_)) {
      if (!merged.load(db_path_, warnings)) {
        std::fprintf(stderr, "warning: could not read existing %s\n",
                     db_path_.c_str());
      }
    }
    for (const kernscope::db::Record &r : db_.records())
      merged.put(r);
    if (!merged.save(db_path_)) {
      std::fprintf(stderr, "error: could not write db file %s\n",
                   db_path_.c_str());
      std::exit(2);
    }
  }

private:
  std::string device_;
  std::string gcn_full_;
  std::string db_path_;
  Clocks clocks_;
  kernscope::db::Database db_;
};

// --- argument parsing -------------------------------------------------------

struct Args {
  std::string command;
  std::string arch = "gfx1101";
  std::vector<std::int64_t> sizes_bytes{
      std::int64_t{1} << 20, std::int64_t{4} << 20, std::int64_t{16} << 20,
      std::int64_t{64} << 20, std::int64_t{256} << 20, std::int64_t{512} << 20};
  bool flush = true;
  std::string db_path = "reports/peaks/gfx1101-bench.jsonl";
  bool json = false;
  std::string hsaco = KERNSCOPE_BENCH_HSACO;
};

void PrintUsage(std::FILE *out) {
  std::fprintf(out,
               "usage: kernscope-bench <peak-bw|peak-flops|peak-lds> "
               "[--arch gfx1101] [--sizes 1,4,16,64,256,512] [--no-flush] "
               "[--db <path>] [--json]\n"
               "  measures gfx1101 roofline peaks on the local GPU (RX 7800 XT)\n"
               "  peak-bw     DRAM bandwidth: D2D-copy + read-only sweep -> GB/s\n"
               "  peak-flops  fp32 FFMA peak -> TFLOPS\n"
               "  peak-lds    shared-memory bandwidth -> GB/s\n"
               "  clocks recorded as-measured via rocm-smi (never locked)\n");
}

bool ParseSizes(const std::string &csv, std::vector<std::int64_t> *out) {
  std::vector<std::int64_t> sizes;
  std::size_t pos = 0;
  while (pos < csv.size()) {
    const std::size_t comma = csv.find(',', pos);
    const std::string tok =
        csv.substr(pos, comma == std::string::npos ? std::string::npos
                                                   : comma - pos);
    try {
      const long v = std::stol(tok);
      if (v <= 0) return false;
      sizes.push_back(static_cast<std::int64_t>(v) << 20);
    } catch (...) {
      return false;
    }
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  if (sizes.empty()) return false;
  *out = std::move(sizes);
  return true;
}

bool ParseArgs(int argc, char **argv, Args *out) {
  if (argc < 2) return false;
  // --version / --help may be the FIRST argument (no command).
  if (std::string_view(argv[1]) == "--version") {
    std::printf("kernscope-bench " KERNSCOPE_VERSION_STRING "\n");
    std::exit(0);
  }
  if (std::string_view(argv[1]) == "--help") {
    PrintUsage(stdout);
    std::exit(0);
  }
  out->command = argv[1];
  for (int i = 2; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--arch") {
      if (++i >= argc) return false;
      out->arch = argv[i];
    } else if (a == "--sizes") {
      if (++i >= argc) return false;
      if (!ParseSizes(argv[i], &out->sizes_bytes)) {
        std::fprintf(stderr, "error: invalid --sizes list '%s'\n", argv[i]);
        return false;
      }
    } else if (a == "--no-flush") {
      out->flush = false;
    } else if (a == "--db") {
      if (++i >= argc) return false;
      out->db_path = argv[i];
    } else if (a == "--json") {
      out->json = true;
    } else if (a == "--hsaco") {
      if (++i >= argc) return false;
      out->hsaco = argv[i];
    } else if (a == "--version") {
      std::printf("kernscope-bench " KERNSCOPE_VERSION_STRING "\n");
      std::exit(0);
    } else {
      std::fprintf(stderr, "error: unknown option '%s'\n",
                   std::string(a).c_str());
      return false;
    }
  }
  return out->command == "peak-bw" || out->command == "peak-flops" ||
         out->command == "peak-lds";
}

// --- measurements -----------------------------------------------------------

// peak-bw: D2D-copy (2*size traffic) + read-only (1*size traffic) sweep.
int RunPeakBw(GpuRuntime &rt, Runner &runner, const Args &args, Peaks *peaks,
              std::vector<Measurement> *measured) {
  auto mod = Module::load(rt, args.hsaco);
  if (!mod.ok()) {
    std::fprintf(stderr, "error: load %s: %s\n", args.hsaco.c_str(),
                 mod.message.c_str());
    return 2;
  }
  auto copy = Kernel::create(rt, *mod.value, "kernscope_peak_memcpy");
  auto read = Kernel::create(rt, *mod.value, "kernscope_peak_read");
  if (!copy.ok() || !read.ok()) {
    std::fprintf(stderr, "error: peak-bw kernels not in module: %s\n",
                 (copy.ok() ? read : copy).message.c_str());
    return 2;
  }

  Options opts;
  opts.l2_flush = args.flush;
  opts.min_samples = 10;
  opts.min_time_ms = 50.0;
  opts.max_noise = 0.05;
  opts.timeout_ms = 3000.0;

  for (const std::int64_t size : args.sizes_bytes) {
    // --- D2D copy ---
    auto src = rt.device_malloc(static_cast<std::size_t>(size));
    auto dst = rt.device_malloc(static_cast<std::size_t>(size));
    if (!src.ok() || !dst.ok()) {
      std::fprintf(stderr, "error: device_malloc(%lld): %s\n",
                   static_cast<long long>(size),
                   (src.ok() ? dst : src).message.c_str());
      return 2;
    }
    GpuStatus s = rt.memset_device(src.value, 1, size);
    if (!s.ok()) {
      std::fprintf(stderr, "error: memset: %s\n", s.message.c_str());
      return 2;
    }

    const std::uint64_t src_p = src.value.value;
    const std::uint64_t dst_p = dst.value.value;
    const std::int64_t n4 = size / 16;
    const std::vector<KernelArg> copy_args = {
        KernelArg{&src_p, sizeof(src_p)},
        KernelArg{&dst_p, sizeof(dst_p)},
        KernelArg{&n4, sizeof(n4)},
    };
    LaunchConfig cfg;
    cfg.grid_x = kGrid;
    cfg.block_x = kBlock;

    Result r = runner.Measure(rt, *copy.value, cfg, copy_args, opts);
    if (r.error != GpuError::kOk) {
      std::fprintf(stderr, "error: peak-bw D2D (%lld MiB): %s\n",
                   static_cast<long long>(size >> 20), r.error_message.c_str());
      return 2;
    }
    Measurement m;
    m.family = "peak_bw_d2d";
    m.label = "D2D copy " + std::to_string(size >> 20) + " MiB";
    m.size_bytes = 2 * size; // read + write
    m.median_ms = r.median_ms;
    m.p20_ms = r.p20_ms;
    m.p80_ms = r.p80_ms;
    m.timed_out = r.timed_out;
    m.value = static_cast<double>(m.size_bytes) / (r.median_ms / 1000.0) / 1e9;
    measured->push_back(m);
    if (size >= kDramResidentFloorBytes)
      peaks->max_bw_gbps = std::max(peaks->max_bw_gbps, m.value);
    runner.Record(m);

    // --- read-only ---
    Result rr = runner.Measure(rt, *read.value, cfg, copy_args, opts);
    if (rr.error != GpuError::kOk) {
      std::fprintf(stderr, "error: peak-bw read (%lld MiB): %s\n",
                   static_cast<long long>(size >> 20), rr.error_message.c_str());
      return 2;
    }
    Measurement mr;
    mr.family = "peak_bw_read";
    mr.label = "read-only " + std::to_string(size >> 20) + " MiB";
    mr.size_bytes = size; // reads only
    mr.median_ms = rr.median_ms;
    mr.p20_ms = rr.p20_ms;
    mr.p80_ms = rr.p80_ms;
    mr.timed_out = rr.timed_out;
    mr.value = static_cast<double>(mr.size_bytes) / (rr.median_ms / 1000.0) / 1e9;
    measured->push_back(mr);
    if (size >= kDramResidentFloorBytes)
      peaks->max_bw_gbps = std::max(peaks->max_bw_gbps, mr.value);
    runner.Record(mr);

    (void)rt.device_free(src.value);
    (void)rt.device_free(dst.value);
  }
  return 0;
}

// peak-flops: register-resident fp32 FFMA peak.
int RunPeakFlops(GpuRuntime &rt, Runner &runner, const Args &args, Peaks *peaks,
                 std::vector<Measurement> *measured) {
  auto mod = Module::load(rt, args.hsaco);
  if (!mod.ok()) {
    std::fprintf(stderr, "error: load %s: %s\n", args.hsaco.c_str(),
                 mod.message.c_str());
    return 2;
  }
  auto fma = Kernel::create(rt, *mod.value, "kernscope_peak_fma");
  if (!fma.ok()) {
    std::fprintf(stderr, "error: peak-flops kernel not in module: %s\n",
                 fma.message.c_str());
    return 2;
  }

  auto out = rt.device_malloc(sizeof(float));
  if (!out.ok()) {
    std::fprintf(stderr, "error: device_malloc: %s\n", out.message.c_str());
    return 2;
  }
  const std::uint64_t out_p = out.value.value;
  const int iters = 100000;
  const std::vector<KernelArg> args_k = {
      KernelArg{&out_p, sizeof(out_p)},
      KernelArg{&iters, sizeof(iters)},
  };
  LaunchConfig cfg;
  cfg.grid_x = kGrid;
  cfg.block_x = kBlock;

  Options opts;
  opts.l2_flush = args.flush;
  opts.min_samples = 10;
  opts.min_time_ms = 50.0;
  opts.max_noise = 0.05;
  opts.timeout_ms = 3000.0;

  const Result r = runner.Measure(rt, *fma.value, cfg, args_k, opts);
  if (r.error != GpuError::kOk) {
    std::fprintf(stderr, "error: peak-flops: %s\n", r.error_message.c_str());
    return 2;
  }

  Measurement m;
  m.family = "peak_flops";
  m.label = "fp32 FFMA";
  // FLOPs = grid * block * iters * (chains + 1 varying-op update) * 2.
  const double flops =
      static_cast<double>(kGrid) * kBlock * iters * (kFmaChains + 1) * 2.0;
  m.size_bytes = 0;
  m.median_ms = r.median_ms;
  m.p20_ms = r.p20_ms;
  m.p80_ms = r.p80_ms;
  m.timed_out = r.timed_out;
  m.value = flops / (r.median_ms / 1000.0) / 1e12; // TFLOPS
  measured->push_back(m);
  peaks->max_tflops = m.value;
  runner.Record(m);

  (void)rt.device_free(out.value);
  return 0;
}

// peak-lds: shared-memory bandwidth.
int RunPeakLds(GpuRuntime &rt, Runner &runner, const Args &args, Peaks *peaks,
               std::vector<Measurement> *measured) {
  auto mod = Module::load(rt, args.hsaco);
  if (!mod.ok()) {
    std::fprintf(stderr, "error: load %s: %s\n", args.hsaco.c_str(),
                 mod.message.c_str());
    return 2;
  }
  auto lds = Kernel::create(rt, *mod.value, "kernscope_peak_lds");
  if (!lds.ok()) {
    std::fprintf(stderr, "error: peak-lds kernel not in module: %s\n",
                 lds.message.c_str());
    return 2;
  }

  auto out = rt.device_malloc(sizeof(float));
  if (!out.ok()) {
    std::fprintf(stderr, "error: device_malloc: %s\n", out.message.c_str());
    return 2;
  }
  const std::uint64_t out_p = out.value.value;
  const int iters = 100000;
  const std::vector<KernelArg> args_k = {
      KernelArg{&out_p, sizeof(out_p)},
      KernelArg{&iters, sizeof(iters)},
  };
  LaunchConfig cfg;
  cfg.grid_x = kGrid;
  cfg.block_x = kBlock;

  Options opts;
  opts.l2_flush = args.flush;
  opts.min_samples = 10;
  opts.min_time_ms = 50.0;
  opts.max_noise = 0.05;
  opts.timeout_ms = 3000.0;

  const Result r = runner.Measure(rt, *lds.value, cfg, args_k, opts);
  if (r.error != GpuError::kOk) {
    std::fprintf(stderr, "error: peak-lds: %s\n", r.error_message.c_str());
    return 2;
  }

  Measurement m;
  m.family = "peak_lds";
  m.label = "LDS read+write";
  // Bytes = blocks * blockDim * iters * (4 B read + 4 B write per lane).
  const double bytes = static_cast<double>(kGrid) * kBlock * iters * 8.0;
  m.size_bytes = static_cast<std::int64_t>(bytes);
  m.median_ms = r.median_ms;
  m.p20_ms = r.p20_ms;
  m.p80_ms = r.p80_ms;
  m.timed_out = r.timed_out;
  m.value = bytes / (r.median_ms / 1000.0) / 1e9; // GB/s
  measured->push_back(m);
  peaks->lds_gbps = m.value;
  runner.Record(m);

  (void)rt.device_free(out.value);
  return 0;
}

void PrintReport(const Args &args, const Runner &runner,
                 const std::vector<Measurement> &measured, const Peaks &peaks,
                 bool floors_passed) {
  if (args.json) {
    std::printf("{\"arch\":\"%s\",\"device\":\"%s\",\"date\":\"%s\","
                "\"sclk_mhz\":\"%s\",\"mclk_mhz\":\"%s\","
                "\"peak_bw_gbps\":%.1f,\"peak_flops_tflops\":%.2f,"
                "\"peak_lds_gbps\":%.1f,\"floors_passed\":%s,",
                args.arch.c_str(), runner.device().c_str(), IsoDateUtc().c_str(),
                runner.clocks().sclk_mhz.c_str(), runner.clocks().mclk_mhz.c_str(),
                peaks.max_bw_gbps, peaks.max_tflops, peaks.lds_gbps,
                floors_passed ? "true" : "false");
    std::printf("\"measurements\":[");
    for (std::size_t i = 0; i < measured.size(); ++i) {
      if (i > 0) std::printf(",");
      const Measurement &m = measured[i];
      std::printf("{\"family\":\"%s\",\"label\":\"%s\",\"value\":%.1f,"
                  "\"median_ms\":%.4f,\"p20_ms\":%.4f,\"p80_ms\":%.4f}",
                  m.family.c_str(), m.label.c_str(), m.value, m.median_ms,
                  m.p20_ms, m.p80_ms);
    }
    std::printf("]}\n");
    return;
  }
  std::printf("kernscope-bench %s on %s (%s, sclk %s MHz, mclk %s MHz)\n",
              args.command.c_str(), runner.device().c_str(), args.arch.c_str(),
              runner.clocks().sclk_mhz.c_str(), runner.clocks().mclk_mhz.c_str());
  for (const Measurement &m : measured) {
    std::printf("  %-22s %8.1f (median %.4f ms, p20 %.4f, p80 %.4f)%s\n",
                m.label.c_str(), m.value, m.median_ms, m.p20_ms, m.p80_ms,
                m.timed_out ? " [timeout]" : "");
  }
  std::printf("peak_bw   %8.1f GB/s\n", peaks.max_bw_gbps);
  std::printf("peak_flops %8.2f TFLOPS fp32\n", peaks.max_tflops);
  std::printf("peak_lds  %8.1f GB/s\n", peaks.lds_gbps);
  std::printf("floors: %s\n", floors_passed ? "PASS" : "FAIL");
}

} // namespace

int main(int argc, char **argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    PrintUsage(stderr);
    return 2;
  }
  if (args.arch != "gfx1101") {
    std::fprintf(stderr,
                 "error: kernscope-bench measures gfx1101 only (the "
                 "peaks.hsaco is built for gfx1101); got '%s'\n",
                 args.arch.c_str());
    return 2;
  }
  if (args.hsaco.empty()) {
    std::fprintf(stderr,
                 "error: peaks.hsaco path not set (KERNSCOPE_BENCH_HSACO); "
                 "run `devenv shell -- cmake --build build` first\n");
    return 2;
  }

  auto rt = create_runtime(Backend::kAuto);
  if (!rt.ok()) {
    std::fprintf(stderr, "error: %s\n", rt.message.c_str());
    return 2;
  }
  if (rt.value->backend_name() != "hip") {
    std::fprintf(stderr, "error: kernscope-bench needs a live HIP device; "
                         "got backend '%s'\n",
                 std::string(rt.value->backend_name()).c_str());
    return 2;
  }
  auto devs = rt.value->list_devices();
  if (!devs.ok() || devs.value.empty()) {
    std::fprintf(stderr, "error: no HIP devices: %s\n", devs.message.c_str());
    return 2;
  }
  const std::string device = devs.value.front().name;
  auto props = rt.value->device_properties(device);
  if (!props.ok()) {
    std::fprintf(stderr, "error: device_properties(%s): %s\n", device.c_str(),
                 props.message.c_str());
    return 2;
  }

  Runner runner(device, props.value.gcn_arch_full, args.db_path);
  runner.set_clocks(ReadClocks());

  Peaks peaks;
  std::vector<Measurement> measured;
  int rc = 0;
  if (args.command == "peak-bw")
    rc = RunPeakBw(*rt.value, runner, args, &peaks, &measured);
  else if (args.command == "peak-flops")
    rc = RunPeakFlops(*rt.value, runner, args, &peaks, &measured);
  else
    rc = RunPeakLds(*rt.value, runner, args, &peaks, &measured);
  if (rc != 0)
    return rc;

  // Sanity floors for the RX 7800 XT (todo-22 acceptance): > 300 GB/s DRAM,
  // > 10 TFLOPS fp32. Below the floor -> exit 1 (measured, but suspicious).
  bool floors_passed = true;
  if (args.command == "peak-bw" && !(peaks.max_bw_gbps > 300.0)) {
    std::fprintf(stderr,
                 "error: peak-bw %.1f GB/s is below the 300 GB/s sanity floor\n",
                 peaks.max_bw_gbps);
    floors_passed = false;
  }
  if (args.command == "peak-flops" && !(peaks.max_tflops > 10.0)) {
    std::fprintf(stderr,
                 "error: peak-flops %.2f TFLOPS is below the 10 TFLOPS "
                 "sanity floor\n",
                 peaks.max_tflops);
    floors_passed = false;
  }

  PrintReport(args, runner, measured, peaks, floors_passed);
  runner.Flush();
  std::printf("db: %s\n", args.db_path.c_str());
  return floors_passed ? 0 : 1;
}
