//===- cli_pipeline.cpp - kernscope-profile CLI pipeline (todo 52) ---------*- C++ -*-===//
//
// Implementation of the todo-52 pipeline (see cli_pipeline.h): trace under
// the todo-49 HIP tracer -> parse the todo-50 enriched JSONL -> todo-51 rank
// + diagnose -> libs/report-shaped outputs (owner Section B text shape,
// profile JSON, libs/report HTML).
//
//===----------------------------------------------------------------------===//

#include "cli_pipeline.h"

#include "kernscope/db.h"
#include "kernscope/profile/kernel_record.h" // json_escape
#include "kernscope/report.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace kernscope::profile::cli {

namespace {

//===--- small helpers --------------------------------------------------------===//

std::string FmtNumber(double value) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.10g", value);
  return std::string(buf);
}

std::string ReadFile(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    return "";
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

//===--- vendor detection (auto mode) ------------------------------------------===//
// `ldd` on the target: libamdhip64/hip -> hip; libcudart/libcuda -> cuda.
// Neither -> "hip" (the local default; the tracer is inert for non-GPU
// binaries, which is exactly the no-kernels QA negative path).

std::string RunCapture(const std::vector<std::string> &argv) {
  std::string cmd;
  for (std::size_t i = 0; i < argv.size(); ++i) {
    if (i > 0) {
      cmd += " ";
    }
    cmd += "'" + argv[i] + "'";
  }
  cmd += " 2>/dev/null";
  std::FILE *pipe = ::popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return "";
  }
  std::string out;
  char buf[4096];
  std::size_t n = 0;
  while ((n = std::fread(buf, 1, sizeof(buf), pipe)) > 0) {
    out.append(buf, n);
  }
  (void)::pclose(pipe);
  return out;
}

} // namespace

//===--- vendor detection -------------------------------------------------------===//

std::string detect_vendor(const std::string &binary_path) {
  const std::string ldd = RunCapture({"ldd", binary_path});
  const std::string lower = [&] {
    std::string s = ldd;
    for (char &c : s) {
      if (c >= 'A' && c <= 'Z') {
        c = static_cast<char>(c - 'A' + 'a');
      }
    }
    return s;
  }();
  if (lower.find("libamdhip64") != std::string::npos ||
      lower.find("libhip") != std::string::npos) {
    return "hip";
  }
  if (lower.find("libcudart") != std::string::npos ||
      lower.find("libcuda") != std::string::npos) {
    return "cuda";
  }
  return "hip"; // local default (see the header)
}

//===--- target resolution ------------------------------------------------------===//

std::optional<std::string> resolve_target(const std::string &target,
                                          std::string *error) {
  std::error_code ec;
  if (!std::filesystem::exists(target, ec)) {
    *error = "target does not exist: '" + target + "'";
    return std::nullopt;
  }
  if (std::filesystem::is_regular_file(target, ec)) {
    return target;
  }
  if (!std::filesystem::is_directory(target, ec)) {
    *error = "target is neither a file nor a directory: '" + target + "'";
    return std::nullopt;
  }
  // A directory (the committed fixture build dir, or a capture repro-dir):
  // scan for the executable. The todo-5 fixture's standalone build produces
  // layernorm_app; prefer it, then any executable file.
  std::vector<std::string> candidates;
  for (const auto &entry : std::filesystem::directory_iterator(target, ec)) {
    std::error_code fec;
    if (entry.is_regular_file(fec) && entry.path().extension().empty()) {
      candidates.push_back(entry.path().string());
    }
  }
  if (candidates.empty()) {
    *error = "directory contains no executable target: '" + target + "'";
    return std::nullopt;
  }
  std::sort(candidates.begin(), candidates.end());
  for (const std::string &c : candidates) {
    if (std::filesystem::path(c).filename().string().find("layernorm_app") !=
        std::string::npos) {
      return c;
    }
  }
  return candidates.front();
}

//===--- scratch input seeding (layernorm_fused fixture contract) ---------------===//
// The committed fixture app (tests/fixtures/layernorm_fused/layernorm.hip)
// reads input.bin from its CWD by default: [int32 M][int32 N][uint16 X:
// M*N bf16][uint16 gamma: N bf16][uint16 beta: N bf16]. The e2e golden runs
// it with default args, so the CLI seeds that input (M=1024 x N=128,
// deterministic bf16 values — profiling never depends on the payload).

void SeedLayernormInput(const std::filesystem::path &scratch_dir) {
  constexpr std::int32_t kM = 1024;
  constexpr std::int32_t kN = 128;
  std::mt19937 rng(20260815); // deterministic; payload values are irrelevant
  std::uniform_int_distribution<std::uint16_t> dist(0, 0xFFFF);

  std::ofstream f(scratch_dir / "input.bin", std::ios::binary);
  if (!f) {
    return;
  }
  const std::int32_t m = kM, n = kN;
  f.write(reinterpret_cast<const char *>(&m), sizeof(m));
  f.write(reinterpret_cast<const char *>(&n), sizeof(n));
  std::vector<std::uint16_t> buf;
  buf.reserve(static_cast<std::size_t>(kM) * kN);
  for (std::size_t i = 0; i < static_cast<std::size_t>(kM) * kN; ++i) {
    buf.push_back(static_cast<std::uint16_t>(dist(rng)));
  }
  f.write(reinterpret_cast<const char *>(buf.data()),
          static_cast<std::streamsize>(buf.size() * sizeof(std::uint16_t)));
  std::vector<std::uint16_t> vec;
  vec.reserve(static_cast<std::size_t>(kN));
  for (int i = 0; i < kN; ++i) {
    vec.push_back(static_cast<std::uint16_t>(dist(rng)));
  }
  f.write(reinterpret_cast<const char *>(vec.data()),
          static_cast<std::streamsize>(vec.size() * sizeof(std::uint16_t)));
  f.write(reinterpret_cast<const char *>(vec.data()),
          static_cast<std::streamsize>(vec.size() * sizeof(std::uint16_t)));
}

//===--- trace runner ------------------------------------------------------------===//

TraceResult run_trace(const TraceOptions &options) {
  TraceResult out;
  std::string error;
  const std::optional<std::string> resolved =
      resolve_target(options.target, &error);
  if (!resolved) {
    out.error = error;
    return out;
  }
  // The child exec's from the scratch dir (chdir), so the target must be an
  // absolute path — a relative CLI argument would break after the chdir.
  const std::string binary = std::filesystem::absolute(*resolved).string();

  if (options.vendor == "cuda") {
    out.error =
        "CUDA runtime tracing is rental-gated (compile+unit only on this "
        "host; see scripts/rental-validate.sh)";
    return out;
  }
  if (options.hip_tool.empty() || !std::filesystem::exists(options.hip_tool)) {
    out.error =
        "HIP tracer tool not found ('" + options.hip_tool +
        "') — rebuild with the rocprofiler-sdk present (devenv shell) or set "
        "KERNSCOPE_PROFILE_HIP_TOOL";
    return out;
  }

  // Scratch dir: the app runs here (default input.bin/output.bin contract).
  std::string tmpl = (std::filesystem::temp_directory_path() /
                      "kernscope-profile-XXXXXX")
                         .string();
  std::vector<char> writable(tmpl.begin(), tmpl.end());
  writable.push_back('\0');
  char *dir = ::mkdtemp(writable.data());
  if (dir == nullptr) {
    out.error = "scratch dir creation failed";
    return out;
  }
  const std::filesystem::path scratch(dir);
  const std::string jsonl_path = (scratch / "profile.jsonl").string();
  out.jsonl_path = jsonl_path;

  // The fixture's input contract (see above): only for the layernorm app.
  const std::string stem = std::filesystem::path(binary).stem().string();
  if (stem.find("layernorm") != std::string::npos) {
    SeedLayernormInput(scratch);
  }

  std::vector<std::string> argv = {binary};
  argv.insert(argv.end(), options.app_args.begin(), options.app_args.end());

  const pid_t pid = ::fork();
  if (pid < 0) {
    out.error = "fork failed";
    (void)std::filesystem::remove_all(scratch);
    return out;
  }
  if (pid == 0) {
    // Child: env + cwd, then exec. 5-minute guard against a hung target.
    // The app's stdout/stderr are CAPTURED to files (never inherited): the
    // CLI's own stdout must stay reserved for the report stream (a noisy
    // target would corrupt --json/--text output).
    const std::string out_log = (scratch / "app_stdout.log").string();
    const std::string err_log = (scratch / "app_stderr.log").string();
    const int fd_out = ::open(out_log.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                              0644);
    if (fd_out >= 0) {
      (void)::dup2(fd_out, STDOUT_FILENO);
      (void)::close(fd_out);
    }
    const int fd_err = ::open(err_log.c_str(), O_WRONLY | O_CREAT | O_TRUNC,
                              0644);
    if (fd_err >= 0) {
      (void)::dup2(fd_err, STDERR_FILENO);
      (void)::close(fd_err);
    }
    ::setenv("LD_PRELOAD", options.hip_tool.c_str(), 1);
    ::setenv("KERNSCOPE_PROFILE_OUT", jsonl_path.c_str(), 1);
    (void)::chdir(scratch.c_str());
    (void)::alarm(300);
    std::vector<char *> cargv;
    cargv.reserve(argv.size() + 1);
    for (const std::string &a : argv) {
      cargv.push_back(const_cast<char *>(a.c_str()));
    }
    cargv.push_back(nullptr);
    ::execv(argv[0].c_str(), cargv.data());
    std::fprintf(stderr,
                 "[kernscope-profile] error: cannot execute '%s': %s\n",
                 argv[0].c_str(), std::strerror(errno));
    std::exit(127);
  }

  int status = 0;
  (void)::waitpid(pid, &status, 0);
  const bool killed_by_alarm =
      WIFSIGNALED(status) && (WTERMSIG(status) == SIGALRM);
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    // Surface the tail of the captured stderr so the failure is debuggable.
    const std::string err_log = (scratch / "app_stderr.log").string();
    const std::string err_text = ReadFile(err_log);
    const std::string tail =
        err_text.size() > 2000 ? err_text.substr(err_text.size() - 2000)
                               : err_text;
    out.error = "target exited abnormally (status " +
                std::to_string(WIFEXITED(status) ? WEXITSTATUS(status)
                                                 : -WTERMSIG(status)) +
                (killed_by_alarm ? "; killed by the 300s timeout" : "") +
                ") — a partial trace is never trusted";
    if (!tail.empty()) {
      out.error += "; captured stderr tail: " + tail;
    }
    (void)std::filesystem::remove_all(scratch);
    return out;
  }

  const std::string jsonl = ReadFile(jsonl_path);
  parse_profile_jsonl(jsonl, &out.kernels, &out.api_records);
  out.ok = true;
  (void)std::filesystem::remove_all(scratch);
  return out;
}

//===--- JSONL parsing ------------------------------------------------------------===//

namespace {

// Pointer-safe object lookup (db::json::Value has non-explicit constructors;
// see tools/analyze/main.cpp's child() rationale).
const db::json::Value *Child(const db::json::Value *obj, const char *key) {
  return obj != nullptr ? obj->find(key) : nullptr;
}

std::int64_t AsInt(const db::json::Value *v, std::int64_t fallback) {
  if (v == nullptr || v->kind() != db::json::Kind::Number) {
    return fallback;
  }
  return static_cast<std::int64_t>(v->as_number());
}

void ParseVec3(const db::json::Value *v, std::int64_t *x, std::int64_t *y,
               std::int64_t *z) {
  *x = *y = *z = 1;
  if (v == nullptr || v->kind() != db::json::Kind::Array) {
    return;
  }
  const auto &arr = v->as_array();
  if (!arr.empty()) {
    *x = AsInt(&arr[0], 1);
  }
  if (arr.size() > 1) {
    *y = AsInt(&arr[1], 1);
  }
  if (arr.size() > 2) {
    *z = AsInt(&arr[2], 1);
  }
}

} // namespace

void parse_profile_jsonl(const std::string &jsonl_text,
                         std::vector<EnrichedKernel> *kernels,
                         std::size_t *api_records) {
  kernels->clear();
  if (api_records != nullptr) {
    *api_records = 0;
  }
  std::istringstream in(jsonl_text);
  std::string line;
  while (std::getline(in, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    std::string error;
    const std::optional<db::json::Value> parsed =
        db::json::Value::parse(line, error);
    if (!parsed) {
      continue; // malformed line: skip, never fatal
    }
    const db::json::Value *kind = Child(&*parsed, "kind");
    if (kind == nullptr || kind->kind() != db::json::Kind::String) {
      continue;
    }
    if (kind->as_string() == "api") {
      if (api_records != nullptr) {
        ++*api_records;
      }
      continue;
    }
    if (kind->as_string() != "kernel") {
      continue;
    }

    EnrichedKernel ek;
    const db::json::Value *name = Child(&*parsed, "name");
    if (name != nullptr && name->kind() == db::json::Kind::String) {
      ek.record.kernel_name = name->as_string();
    }
    ek.record.duration_ns =
        static_cast<std::uint64_t>(AsInt(Child(&*parsed, "duration_ns"), 0));
    ParseVec3(Child(&*parsed, "grid"), &ek.record.grid_x, &ek.record.grid_y,
              &ek.record.grid_z);
    ParseVec3(Child(&*parsed, "workgroup"), &ek.record.block_x,
              &ek.record.block_y, &ek.record.block_z);
    ek.record.static_shared_bytes = static_cast<std::int32_t>(
        AsInt(Child(&*parsed, "static_shared_bytes"), 0));
    ek.record.registers_per_thread = static_cast<std::uint16_t>(
        AsInt(Child(&*parsed, "registers_per_thread"), 0));

    // The tracer enriched the record before emission (todo 50); the CLI
    // carries those fields through verbatim — never recomputes, never
    // fabricates.
    const db::json::Value *occ = Child(&*parsed, "occupancy");
    if (occ != nullptr && occ->kind() == db::json::Kind::Number) {
      ek.occupancy = occ->as_number();
      ek.occupancy_unknown = false;
    } else {
      ek.occupancy = -1.0;
      ek.occupancy_unknown = true; // "unknown-resources" marker
    }
    const db::json::Value *limiter = Child(&*parsed, "limiter");
    if (limiter != nullptr && limiter->kind() == db::json::Kind::String) {
      ek.limiter = limiter->as_string();
    }
    ek.blocks_per_sm =
        static_cast<int>(AsInt(Child(&*parsed, "blocks_per_sm"), 0));
    ek.warps_resident =
        static_cast<int>(AsInt(Child(&*parsed, "warps_resident"), 0));
    ek.max_warps_per_sm =
        static_cast<int>(AsInt(Child(&*parsed, "max_warps_per_sm"), 0));
    ek.arch_vgpr_count =
        static_cast<int>(AsInt(Child(&*parsed, "vgpr"), -1));
    ek.sgpr_count = static_cast<int>(AsInt(Child(&*parsed, "sgpr"), -1));
    ek.accum_vgpr_count =
        static_cast<int>(AsInt(Child(&*parsed, "accum_vgpr"), -1));
    ek.group_segment_bytes =
        static_cast<int>(AsInt(Child(&*parsed, "lds_bytes"), -1));
    ek.private_segment_bytes =
        static_cast<int>(AsInt(Child(&*parsed, "scratch_bytes"), -1));
    ek.kernel_address = static_cast<std::uint64_t>(
        AsInt(Child(&*parsed, "kernel_address"), 0));
    const db::json::Value *ptx = Child(&*parsed, "ptx_available");
    if (ptx != nullptr && ptx->kind() == db::json::Kind::Bool) {
      ek.ptx_available = ptx->as_bool();
    }
    kernels->push_back(std::move(ek));
  }
}

//===--- report assembly (rank + diagnose, todo 51) ------------------------------===//

ProfileReport build_report(const std::vector<EnrichedKernel> &kernels,
                           const TraceOptions &options) {
  ProfileReport report;
  report.vendor = options.vendor;
  report.target = options.target;
  report.kernels = kernels;
  if (kernels.empty()) {
    report.no_kernels_traced = true;
    return report;
  }

  report.ranking = rank_hot_kernels(kernels, options.top_n);
  for (const HotKernelEntry &e : report.ranking) {
    report.wall_time_ns += e.total_ns;
  }

  // AMD has no PTX-level diagnostics path (ptx_available == false): the
  // analyzer merge cannot run; the rules engine degrades honestly.
  DiagnosticsOptions dopts;
  dopts.arch_name = options.arch_name;
  dopts.analyzer_cli = options.analyzer_cli;
  dopts.top_n = options.top_n;
  report.diagnostics.reserve(kernels.size());
  for (const EnrichedKernel &k : kernels) {
    report.diagnostics.push_back(diagnose_kernel(k, "", dopts));
  }
  return report;
}

//===--- JSON renderer -------------------------------------------------------------===//

//===--- JSON renderer -------------------------------------------------------------===//

namespace {

// The showcase kernel for the owner-style renderings (owner_example JSON
// field + the --html page): the first hot kernel whose name contains
// "layernorm" — the golden fixture's profiler example
// (docs/research/owner-examples.md Section B) — falling back to the hottest
// kernel. Internal runtime kernels (e.g. the rocclr memcpy helper) never
// take the showcase.
std::size_t ShowcaseIndex(const ProfileReport &report) {
  if (report.ranking.empty()) {
    return 0;
  }
  const std::string &fallback = report.ranking.front().kernel_name;
  for (std::size_t i = 0; i < report.kernels.size(); ++i) {
    if (report.kernels[i].record.kernel_name.find("layernorm") !=
        std::string::npos) {
      return i;
    }
  }
  for (std::size_t i = 0; i < report.kernels.size(); ++i) {
    if (report.kernels[i].record.kernel_name == fallback) {
      return i;
    }
  }
  return 0;
}

} // namespace

std::size_t showcase_kernel_index(const ProfileReport &report) {
  return ShowcaseIndex(report);
}

std::string report_to_json(const ProfileReport &report) {
  std::ostringstream out;
  out << "{\n";
  out << "  \"schema_version\": \"1.0\",\n";
  out << "  \"report_type\": \"kernscope_profile_report\",\n";
  out << "  \"vendor\": \"" << json_escape(report.vendor) << "\",\n";
  out << "  \"target\": \"" << json_escape(report.target) << "\",\n";
  out << "  \"no_kernels_traced\": "
      << (report.no_kernels_traced ? "true" : "false") << ",\n";
  out << "  \"kernel_count\": " << report.kernels.size() << ",\n";
  out << "  \"wall_time_ns\": " << report.wall_time_ns << ",\n";

  out << "  \"kernels\": [\n";
  for (std::size_t i = 0; i < report.kernels.size(); ++i) {
    const EnrichedKernel &k = report.kernels[i];
    const KernelDiagnostics &d =
        i < report.diagnostics.size() ? report.diagnostics[i]
                                      : KernelDiagnostics{};
    out << "    {\n";
    out << "      \"kernel_name\": \"" << json_escape(k.record.kernel_name)
        << "\",\n";
    out << "      \"duration_ns\": " << k.record.duration_ns << ",\n";
    out << "      \"grid\": [" << k.record.grid_x << ", " << k.record.grid_y
        << ", " << k.record.grid_z << "],\n";
    out << "      \"workgroup\": [" << k.record.block_x << ", "
        << k.record.block_y << ", " << k.record.block_z << "],\n";
    out << "      \"registers_per_thread\": "
        << k.record.registers_per_thread << ",\n";
    out << "      \"shared_bytes\": "
        << (static_cast<std::int64_t>(k.record.static_shared_bytes) +
            static_cast<std::int64_t>(k.record.dynamic_shared_bytes))
        << ",\n";
    if (k.occupancy_unknown) {
      out << "      \"occupancy\": \"unknown-resources\",\n";
    } else {
      out << "      \"occupancy\": " << FmtNumber(k.occupancy) << ",\n";
    }
    out << "      \"limiter\": \"" << json_escape(k.limiter) << "\",\n";
    out << "      \"blocks_per_sm\": " << k.blocks_per_sm << ",\n";
    out << "      \"ptx_available\": "
        << (k.ptx_available ? "true" : "false") << ",\n";
    out << "      \"limiter_diag\": \"" << json_escape(d.limiter) << "\",\n";
    out << "      \"hints\": [\n";
    for (std::size_t j = 0; j < d.hints.size(); ++j) {
      const kernscope::Finding &f = d.hints[j];
      out << "        {\"id\": \"" << json_escape(f.id)
          << "\", \"severity\": \"" << json_escape(f.severity)
          << "\", \"evidence\": {";
      bool first = true;
      for (const auto &[key, value] : f.evidence) {
        if (!first) {
          out << ", ";
        }
        first = false;
        out << "\"" << json_escape(key) << "\": \"" << json_escape(value)
            << "\"";
      }
      out << "}}";
      if (j + 1 < d.hints.size()) {
        out << ",";
      }
      out << "\n";
    }
    out << "      ]\n";
    out << "    }";
    if (i + 1 < report.kernels.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n";

  out << "  \"ranking\": [\n";
  for (std::size_t i = 0; i < report.ranking.size(); ++i) {
    const HotKernelEntry &e = report.ranking[i];
    out << "    {\"kernel_name\": \"" << json_escape(e.kernel_name)
        << "\", \"launches\": " << e.launches
        << ", \"total_ns\": " << e.total_ns
        << ", \"median_ns\": " << e.median_ns
        << ", \"pct_wall_time\": " << FmtNumber(e.pct_wall_time) << "}";
    if (i + 1 < report.ranking.size()) {
      out << ",";
    }
    out << "\n";
  }
  out << "  ],\n";

  // The owner's Section B example shape (docs/research/owner-examples.md)
  // with LOCALLY-MEASURED values for the fixture's showcase kernel: the
  // first hot kernel whose name contains "layernorm" (the golden e2e target;
  // internal runtime kernels like the rocclr memcpy never take the
  // showcase), falling back to the hottest kernel.
  out << "  \"owner_example\": ";
  if (report.ranking.empty() || report.kernels.empty()) {
    out << "null\n";
  } else {
    const std::size_t idx = ShowcaseIndex(report);
    const EnrichedKernel &show = report.kernels[idx];
    const HotKernelEntry *hot = nullptr;
    for (const HotKernelEntry &e : report.ranking) {
      if (e.kernel_name == show.record.kernel_name) {
        hot = &e;
        break;
      }
    }
    const KernelDiagnostics *d = nullptr;
    for (const KernelDiagnostics &cand : report.diagnostics) {
      if (cand.kernel_name == show.record.kernel_name) {
        d = &cand;
        break;
      }
    }
    const std::uint64_t time_ns =
        hot != nullptr ? hot->median_ns : show.record.duration_ns;
    out << "{\n";
    out << "    \"kernel\": \"" << json_escape(show.record.kernel_name)
        << "\",\n";
    out << "    \"time_us\": " << FmtNumber(static_cast<double>(time_ns) / 1000.0)
        << ",\n";
    out << "    \"registers_per_thread\": "
        << static_cast<int>(show.record.registers_per_thread) << ",\n";
    out << "    \"shared_memory_block_kb\": "
        << FmtNumber(static_cast<double>(show.record.static_shared_bytes +
                                        show.record.dynamic_shared_bytes) /
                     1024.0)
        << ",\n";
    out << "    \"likely_limiter\": \"" << (d != nullptr ? d->limiter : "")
        << "\",\n";
    out << "    \"compiler_hints\": [";
    bool first_hint = true;
    if (d != nullptr) {
      for (const kernscope::Finding &f : d->hints) {
        if (!first_hint) {
          out << ", ";
        }
        first_hint = false;
        const auto it = f.evidence.find("text");
        const std::string text =
            (it != f.evidence.end() && !it->second.empty()) ? it->second
                                                            : f.id;
        out << "\"" << json_escape(text) << "\"";
      }
    }
    out << "]\n";
    out << "  }\n";
  }
  out << "}\n";
  return out.str();
}

//===--- text renderer (owner Section B shape) --------------------------------------===//

namespace {

const std::string &FindingText(const kernscope::Finding &f) {
  const auto it = f.evidence.find("text");
  if (it != f.evidence.end() && !it->second.empty()) {
    return it->second;
  }
  return f.id;
}

} // namespace

std::string report_to_text(const ProfileReport &report) {
  std::ostringstream out;
  if (report.no_kernels_traced) {
    out << "no kernels traced\n";
    return out.str();
  }

  out << "kernscope-profile report (vendor: " << report.vendor
      << ", target: " << report.target << ")\n";
  out << report.kernels.size() << " kernel record(s), wall time "
      << report.wall_time_ns << " ns\n\n";

  // Owner Section B shape per hot kernel (docs/research/owner-examples.md),
  // with locally-measured values only.
  for (std::size_t i = 0; i < report.ranking.size(); ++i) {
    const HotKernelEntry &e = report.ranking[i];
    const EnrichedKernel *k = nullptr;
    for (const EnrichedKernel &cand : report.kernels) {
      if (cand.record.kernel_name == e.kernel_name) {
        k = &cand;
        break;
      }
    }
    const KernelDiagnostics *d = nullptr;
    if (k != nullptr) {
      for (const KernelDiagnostics &cand : report.diagnostics) {
        if (cand.kernel_name == e.kernel_name) {
          d = &cand;
          break;
        }
      }
    }
    out << "== hot kernel " << (i + 1) << ": " << e.kernel_name << " ==\n";
    out << "Kernel: " << e.kernel_name << "\n";
    out << "Time: " << FmtNumber(static_cast<double>(e.median_ns) / 1000.0)
        << " us\n";
    if (k != nullptr) {
      out << "Registers/thread: " << k->record.registers_per_thread << "\n";
      out << "Shared memory/block: "
          << FmtNumber(static_cast<double>(k->record.static_shared_bytes +
                                          k->record.dynamic_shared_bytes) /
                       1024.0)
          << " KB\n";
    }
    out << "Likely limiter: " << (d != nullptr ? d->limiter : "unknown")
        << "\n";
    if (d != nullptr && !d->hints.empty()) {
      out << "Compiler hint:\n";
      for (const kernscope::Finding &f : d->hints) {
        out << "  " << FindingText(f) << "\n";
      }
    } else {
      out << "Compiler hint:\n  (none)\n";
    }
    out << "\n";
  }

  out << "== ranking ==\n";
  for (std::size_t i = 0; i < report.ranking.size(); ++i) {
    const HotKernelEntry &e = report.ranking[i];
    out << (i + 1) << ". " << e.kernel_name << ": " << e.launches
        << " launch(es), total " << e.total_ns << " ns, median "
        << e.median_ns << " ns, " << FmtNumber(e.pct_wall_time)
        << "% of wall time\n";
  }
  return out.str();
}

//===--- libs/report model builder (the HTML render path) ---------------------------===//

kernscope::KernelReport build_kernel_report_model(const ProfileReport &report,
                                                  std::size_t kernel_index) {
  kernscope::KernelReport model;
  if (kernel_index >= report.kernels.size()) {
    model.kernel_name = "(no kernels traced)";
    model.limiter = "unknown";
    return model;
  }
  const EnrichedKernel &k = report.kernels[kernel_index];
  const KernelDiagnostics &d =
      kernel_index < report.diagnostics.size()
          ? report.diagnostics[kernel_index]
          : KernelDiagnostics{};
  model.kernel_name = k.record.kernel_name;
  if (!k.occupancy_unknown) {
    model.metrics["occupancy"] = k.occupancy;
  }
  model.metrics["duration_ns"] = static_cast<double>(k.record.duration_ns);
  model.metrics["registers_per_thread"] =
      static_cast<double>(k.record.registers_per_thread);
  model.metrics["shared_bytes"] = static_cast<double>(
      k.record.static_shared_bytes + k.record.dynamic_shared_bytes);
  model.metrics["blocks_per_sm"] = static_cast<double>(k.blocks_per_sm);
  model.limiter = d.limiter.empty() ? k.limiter : d.limiter;
  if (model.limiter.empty()) {
    model.limiter = "unknown";
  }
  for (const kernscope::Finding &f : d.hints) {
    if (f.severity == "suggestion") {
      model.suggestions.push_back(f);
    } else {
      model.warnings.push_back(f);
    }
  }
  return model;
}

} // namespace kernscope::profile::cli
