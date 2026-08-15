//===- main.cpp - kernscope-capture CLI (todo 48: record|replay|bench) ------===//
//
// kernscope capture/replay track. The C4 tool entry point. Todo 47 shipped
// `emit-repro`; todo 48 appends the live replay track:
//
//   kernscope-capture record <app> [app-args...]    run an app under the
//       capture mechanism (vendor auto-detect) -> bundle + blobs
//   kernscope-capture emit-repro <bundle> [...]     standalone reproducer
//   kernscope-capture replay <bundle> [...]         live replay + validation
//       (bitwise default; --mode tolerance) + profiling hook records
//   kernscope-capture bench <bundle> [...]          replay-as-benchmark:
//       N reps, L2 flush, median -> libs/db record (measured provenance)
//   kernscope-capture db-query <db-file> [...]      query the JSONL db
//
// <bundle> is a v1 *.kcap/ directory OR a v0 text bundle (BundleReader
// compatibility); v0 requires --vendor/--arch.
//
// Exit codes: 0 success; 1 replay ran but validation FAILED / db-query found
// nothing; 2 usage/environment/generation failure.
//
//===----------------------------------------------------------------------===//

#include "replay.h"
#include "repro_gen.h"

#include "kernscope/bundle.h"
#include "kernscope/db.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// The process environment (POSIX): declared at GLOBAL scope so it keeps
// external linkage (an `extern` inside an anonymous namespace would be an
// internal-linkage undefined symbol under -Werror).


namespace {

using kernscope::BenchOptions;
using kernscope::ReplayOptions;
using kernscope::ReplayResult;

void Usage(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture <command> [args]\n"
               "\n"
               "kernscope capture/replay tool (C4).\n"
               "\n"
               "commands:\n"
               "  record <app> [app-args...]  run an app under the capture\n"
               "                              mechanism (vendor auto-detect)\n"
               "  emit-repro <bundle>          emit a STANDALONE C++\n"
               "                              reproducer project (todo 47)\n"
               "  replay <bundle>              live replay + output validation\n"
               "                              (bitwise default, --mode tolerance)\n"
               "  bench <bundle>               replay-as-benchmark: N reps, L2\n"
               "                              flush, median -> libs/db record\n"
               "  db-query <db-file>           query the JSONL benchmark db\n"
               "\n"
               "run 'kernscope-capture <command> --help' for command options\n");
}

bool FileExists(const std::string &path) {
  std::error_code ec;
  return std::filesystem::exists(path, ec) &&
         std::filesystem::is_regular_file(path, ec);
}

std::string CliDir(const char *argv0) {
  std::filesystem::path p(argv0);
  std::error_code ec;
  const std::filesystem::path abs =
      std::filesystem::absolute(p, ec).parent_path();
  return ec ? std::string(".") : abs.string();
}

//--- vendor auto-detect --------------------------------------------------------//
// --vendor wins; else $KERNSCOPE_VENDOR; else the first capture .so next to
// the CLI (hip preferred: this machine's local loop); else an error naming
// both possibilities.

std::string AutoDetectVendor(const std::string &dir) {
  const char *env = std::getenv("KERNSCOPE_VENDOR");
  if (env != nullptr && env[0] != '\0') {
    return env;
  }
  if (FileExists(dir + "/libkernscope_capture_hip.so")) {
    return "amd";
  }
  if (FileExists(dir + "/libkernscope_intercept.so")) {
    return "nvidia";
  }
  return "";
}

//--- record --------------------------------------------------------------------//

void UsageRecord(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture record <app> [app-args...] [options]\n"
               "\n"
               "Runs <app> under the capture mechanism (LD_PRELOAD of the\n"
               "vendor capture tool) and leaves a bundle + blobs behind.\n"
               "\n"
               "options:\n"
               "  --vendor <v>      amd | nvidia | auto (default: auto;\n"
               "                    $KERNSCOPE_VENDOR overrides; else the\n"
               "                    capture .so next to this CLI decides)\n"
               "  --bundle <path>   bundle output path (default: ./capture.bundle)\n"
               "  --blob-dir <dir>  blob output dir (default: ./blobs)\n"
               "  --env K=V         extra environment for the app (repeatable)\n"
               "  -h, --help        this help\n"
               "\n"
               "The app's exit code is propagated; 2 = could not spawn.\n");
}

struct RecordArgs {
  std::string vendor = "auto";
  std::string bundle = "capture.bundle";
  std::string blob_dir = "blobs";
  std::vector<std::string> env;
  std::vector<std::string> app; // app + args
};

bool ParseRecordArgs(int argc, char **argv, RecordArgs *out) {
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-capture: %s requires a value\n", flag);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--vendor") {
      out->vendor = next("--vendor");
    } else if (a == "--bundle") {
      out->bundle = next("--bundle");
    } else if (a == "--blob-dir") {
      out->blob_dir = next("--blob-dir");
    } else if (a == "--env") {
      out->env.push_back(next("--env"));
    } else if (a == "-h" || a == "--help") {
      UsageRecord(stdout);
      std::exit(0);
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "kernscope-capture: unknown option '%s'\n",
                   a.c_str());
      return false;
    } else {
      out->app.push_back(a);
    }
  }
  if (out->app.empty()) {
    std::fprintf(stderr, "kernscope-capture: record requires <app>\n");
    UsageRecord(stderr);
    return false;
  }
  if (out->vendor != "auto" && out->vendor != "amd" && out->vendor != "nvidia") {
    std::fprintf(stderr, "kernscope-capture: --vendor must be auto|amd|nvidia\n");
    return false;
  }
  return true;
}

int RunRecord(const char *argv0, const RecordArgs &args) {
  const std::string dir = CliDir(argv0);
  std::string vendor = args.vendor;
  if (vendor == "auto") {
    vendor = AutoDetectVendor(dir);
    if (vendor.empty()) {
      std::fprintf(stderr,
                   "kernscope-capture: record: cannot auto-detect vendor: no "
                   "capture .so next to %s (pass --vendor amd|nvidia)\n",
                   argv0);
      return 2;
    }
    std::printf("kernscope-capture: record: vendor auto-detect -> %s\n",
                vendor.c_str());
  }

  std::string preload;
  std::string bundle_var;
  std::string blob_var;
  if (vendor == "amd") {
    preload = dir + "/libkernscope_capture_hip.so";
    bundle_var = "KERNSCOPE_HIP_BUNDLE=" + args.bundle;
    blob_var = "KERNSCOPE_HIP_BLOB_DIR=" + args.blob_dir;
  } else {
    preload = dir + "/libkernscope_intercept.so";
    bundle_var = "KERNSCOPE_CAPTURE_LOG=" + args.bundle;
    blob_var.clear();
  }
  if (!FileExists(preload)) {
    std::fprintf(stderr, "kernscope-capture: record: capture tool not found: "
                         "%s (build it first)\n",
                 preload.c_str());
    return 2;
  }

  std::vector<std::string> envp;
  for (char **e = environ; *e != nullptr; ++e) {
    envp.push_back(*e);
  }
  const auto set_env = [&](const std::string &kv) {
    const std::size_t eq = kv.find('=');
    const std::string key = kv.substr(0, eq);
    for (std::string &existing : envp) {
      if (existing.rfind(key + "=", 0) == 0) {
        existing = kv;
        return;
      }
    }
    envp.push_back(kv);
  };
  set_env("LD_PRELOAD=" + preload);
  set_env(bundle_var);
  if (!blob_var.empty()) {
    set_env(blob_var);
  }
  for (const std::string &kv : args.env) {
    set_env(kv);
  }
  std::vector<char *> exec_env;
  exec_env.reserve(envp.size() + 1);
  for (std::string &kv : envp) {
    exec_env.push_back(kv.data());
  }
  exec_env.push_back(nullptr);

  std::vector<char *> exec_argv;
  exec_argv.reserve(args.app.size() + 1);
  for (const std::string &a : args.app) {
    exec_argv.push_back(const_cast<char *>(a.c_str()));
  }
  exec_argv.push_back(nullptr);

  const pid_t pid = fork();
  if (pid < 0) {
    std::fprintf(stderr, "kernscope-capture: record: fork: %s\n",
                 std::strerror(errno));
    return 2;
  }
  if (pid == 0) {
    execvpe(exec_argv[0], exec_argv.data(), exec_env.data());
    std::fprintf(stderr, "kernscope-capture: record: cannot exec %s: %s\n",
                 exec_argv[0], std::strerror(errno));
    std::exit(127);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  if (WIFEXITED(status)) {
    return WEXITSTATUS(status);
  }
  std::fprintf(stderr, "kernscope-capture: record: app terminated abnormally\n");
  return 1;
}

//--- emit-repro (todo 47) ------------------------------------------------------//

void UsageEmitRepro(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture emit-repro <bundle> [options]\n"
               "\n"
               "Emits a STANDALONE plain-CMake C++ project replaying the\n"
               "captured launch (embedded kernel binary + input buffers,\n"
               "captured grid/block/args, output dump). Zero repo deps.\n"
               "\n"
               "options:\n"
               "  --out <dir>      output project dir (default: ./reproducer)\n"
               "  --blob-dir <dir> sidecar blob dir for v0 bundles\n"
               "  --vendor <v>     amd | nvidia (REQUIRED for v0 bundles;\n"
               "                   v1 reads the manifest)\n"
               "  --arch <arch>    gfx1101, sm_80, ... (REQUIRED for v0)\n"
               "  --kernel <name>  replay a specific kernel launch\n"
               "  --force          overwrite a non-empty out dir\n"
               "  -h, --help       this help\n");
}

//--- replay --------------------------------------------------------------------//

void UsageReplay(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture replay <bundle> [options]\n"
               "\n"
               "Restores the captured launch on the local device (fresh\n"
               "allocations + remap/patch table; original VAs where the\n"
               "platform allows — see docs/capture.md) and VALIDATES the\n"
               "captured outputs: BITWISE by default (sha256 + byte\n"
               "compare), tolerance mode (cosine/max_rel_err/pcc, golden\n"
               "contract) with --mode tolerance. Exit 0 = validation PASS;\n"
               "1 = replay ran but validation FAILED.\n"
               "\n"
               "options:\n"
               "  --vendor <v>     amd | nvidia (REQUIRED for v0; v1 wins)\n"
               "  --arch <arch>    REQUIRED for v0\n"
               "  --blob-dir <dir> v0 sidecar blobs\n"
               "  --kernel <name>  specific launch (default: auto)\n"
               "  --mode <m>       bitwise (default) | tolerance\n"
               "  --ref <file>     reference output file(s), in output\n"
               "                   buffer order (v0 bundles; repeatable;\n"
               "                   overrides v1 bundle snapshots)\n"
               "  --dump-dir <dir> dump the replayed outputs\n"
               "  --profile-out <path>  JSONL of timestamped launch records\n"
               "                   (the todo-49 profiling hook contract)\n"
               "  -h, --help       this help\n");
}

int RunReplay(const ReplayOptions &args) {
  ReplayResult result;
  std::string error;
  if (!kernscope::run_replay(args, result, &error)) {
    std::fprintf(stderr, "kernscope-capture: replay: %s\n", error.c_str());
    return 2;
  }
  std::printf("kernscope-capture: replay %s (%s/%s) on %s\n",
              result.kernel_name.c_str(), result.vendor.c_str(),
              result.arch.c_str(), result.device_name.c_str());
  for (const auto &o : result.outputs) {
    if (result.tolerance) {
      std::printf("  output %-12s ref=%s replay=%s cos=%.6f max_rel_err=%.3e "
                  "pcc=%.6f -> %s (ref: %s)\n",
                  o.buffer_id.c_str(), o.ref_sha.substr(0, 12).c_str(),
                  o.replay_sha.substr(0, 12).c_str(), o.tolerance.cosine,
                  o.tolerance.max_rel_err, o.tolerance.pcc,
                  o.tolerance_pass ? "PASS" : "FAIL", o.ref_source.c_str());
    } else {
      std::printf("  output %-12s ref=%s replay=%s %s -> %s (ref: %s)\n",
                  o.buffer_id.c_str(), o.ref_sha.substr(0, 12).c_str(),
                  o.replay_sha.substr(0, 12).c_str(),
                  o.equal ? "bitwise EQUAL" : "BITWISE DIFF",
                  o.equal ? "PASS" : "FAIL", o.ref_source.c_str());
    }
  }
  if (!result.has_reference) {
    std::fprintf(stderr,
                 "kernscope-capture: replay: no reference output for "
                 "validation (v0 bundles: pass --ref <captured-output>)\n");
    return 1;
  }
  std::printf("validation: %s\n", result.validation_pass ? "PASS" : "FAIL");
  if (!result.launches.empty()) {
    std::printf("launch records: %zu (profiling hook contract, "
                "docs/capture.md §profiling-hooks)\n",
                result.launches.size());
  }
  return result.validation_pass ? 0 : 1;
}

//--- bench ---------------------------------------------------------------------//

void UsageBench(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture bench <bundle> [options]\n"
               "\n"
               "Replay-as-benchmark: validates the captured launch once\n"
               "(bitwise), then times it with the libs/bench methodology\n"
               "(N reps, L2 flush between reps, median + p20/p80) and writes\n"
               "ONE libs/db JSONL record with measured provenance.\n"
               "\n"
               "options:\n"
               "  --vendor <v>, --arch <arch>, --blob-dir <dir>,\n"
               "  --kernel <name>         bundle selection (as replay)\n"
               "  --ref <file>            reference output (repeatable)\n"
               "  --reps <n>              timed samples (default 10)\n"
               "  --no-flush              disable L2 flush between reps\n"
               "  --no-validate            skip the validation pass\n"
               "  --db <path>             JSONL db (default ./capture-replay.jsonl)\n"
               "  --profile-out <path>    launch records JSONL\n"
               "  -h, --help              this help\n");
}

int RunBench(const BenchOptions &args) {
  kernscope::BenchResult result;
  std::string error;
  if (!kernscope::run_replay_bench(args, result, &error)) {
    std::fprintf(stderr, "kernscope-capture: bench: %s\n", error.c_str());
    return 2;
  }
  std::printf("bench: %s median %.4f ms (p20 %.4f, p80 %.4f) over %d reps "
              "(L2 flush %s) -> db record %s\n",
              result.gpu_arch.c_str(), result.median_ms, result.p20_ms,
              result.p80_ms, result.n_reps, args.l2_flush ? "on" : "off",
              args.db_path.c_str());
  return 0;
}

//--- db-query ------------------------------------------------------------------//

void UsageDbQuery(FILE *out) {
  std::fprintf(out,
               "usage: kernscope-capture db-query <db-file> [options]\n"
               "\n"
               "Loads a libs/db JSONL file and reports every record matching\n"
               "the filters. Exit 0 = at least one record found; 1 = none.\n"
               "\n"
               "options:\n"
               "  --arch <a>     filter by gpu_arch (default: any)\n"
               "  --family <f>   filter by kernel_family (default: any)\n"
               "  --shape <s>    filter by shape_hash (default: any)\n"
               "  -h, --help     this help\n");
}

int RunDbQuery(const std::string &path, const std::string &arch,
               const std::string &family, const std::string &shape) {
  kernscope::db::Database db;
  std::vector<kernscope::db::LoadWarning> warnings;
  if (!db.load(path, warnings)) {
    std::fprintf(stderr, "kernscope-capture: db-query: cannot open %s\n",
                 path.c_str());
    return 2;
  }
  for (const auto &w : warnings) {
    std::fprintf(stderr, "kernscope-capture: db-query: %s\n", w.message.c_str());
  }
  int found = 0;
  for (const auto &r : db.records()) {
    if (!arch.empty() && r.gpu_arch != arch) {
      continue;
    }
    if (!family.empty() && r.kernel_family != family) {
      continue;
    }
    if (!shape.empty() && r.shape_hash != shape) {
      continue;
    }
    std::printf("record: gpu_arch=%s kernel_family=%s shape_hash=%s "
                "median_ms=%.4f p20=%.4f p80=%.4f mode=%s\n",
                r.gpu_arch.c_str(), r.kernel_family.c_str(),
                r.shape_hash.c_str(), r.timings.median, r.timings.p20,
                r.timings.p80, r.provenance.mode.c_str());
    ++found;
  }
  if (found == 0) {
    std::printf("db-query: no matching records in %s\n", path.c_str());
    return 1;
  }
  std::printf("db-query: %d record(s) found in %s\n", found, path.c_str());
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    Usage(stderr);
    return 1;
  }
  const std::string cmd = argv[1];
  if (cmd == "-h" || cmd == "--help") {
    Usage(stdout);
    return 0;
  }

  if (cmd == "record") {
    RecordArgs args;
    if (!ParseRecordArgs(argc, argv, &args)) {
      return 1;
    }
    return RunRecord(argv[0], args);
  }

  if (cmd == "emit-repro") {
    kernscope::ReproduceOptions opts;
    bool have_bundle = false;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&](const char *flag) -> const char * {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "kernscope-capture: %s requires a value\n",
                       flag);
          std::exit(1);
        }
        return argv[++i];
      };
      if (a == "--out") {
        opts.out_dir = next("--out");
      } else if (a == "--blob-dir") {
        opts.blob_dir = next("--blob-dir");
      } else if (a == "--vendor") {
        opts.vendor = next("--vendor");
      } else if (a == "--arch") {
        opts.arch = next("--arch");
      } else if (a == "--kernel") {
        opts.kernel = next("--kernel");
      } else if (a == "--force") {
        opts.overwrite = true;
      } else if (a == "-h" || a == "--help") {
        UsageEmitRepro(stdout);
        return 0;
      } else if (!a.empty() && a[0] == '-') {
        std::fprintf(stderr, "kernscope-capture: unknown option '%s'\n",
                     a.c_str());
        return 1;
      } else if (!have_bundle) {
        opts.bundle_path = a;
        have_bundle = true;
      } else {
        std::fprintf(stderr, "kernscope-capture: unexpected argument '%s'\n",
                     a.c_str());
        return 1;
      }
    }
    if (!have_bundle) {
      std::fprintf(stderr, "kernscope-capture: emit-repro requires <bundle>\n");
      UsageEmitRepro(stderr);
      return 1;
    }
    if (opts.out_dir.empty()) {
      opts.out_dir = "reproducer";
    }
    std::string error;
    if (!kernscope::emit_reproducer(opts, &error)) {
      std::fprintf(stderr, "kernscope-capture: emit-repro: %s\n",
                   error.c_str());
      return 2;
    }
    std::printf("kernscope-capture: reproducer emitted to %s\n",
                opts.out_dir.c_str());
    return 0;
  }

  if (cmd == "replay" || cmd == "bench") {
    kernscope::CaptureBundleOptions common;
    std::vector<std::string> ref_files;
    std::string dump_dir;
    std::string profile_out;
    std::string mode;
    std::string db_path = "capture-replay.jsonl";
    int reps = 10;
    bool l2_flush = true;
    bool validate = true;
    bool have_bundle = false;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&](const char *flag) -> const char * {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "kernscope-capture: %s requires a value\n",
                       flag);
          std::exit(1);
        }
        return argv[++i];
      };
      if (a == "--vendor") {
        common.vendor = next("--vendor");
      } else if (a == "--arch") {
        common.arch = next("--arch");
      } else if (a == "--blob-dir") {
        common.blob_dir = next("--blob-dir");
      } else if (a == "--kernel") {
        common.kernel = next("--kernel");
      } else if (a == "--mode") {
        mode = next("--mode");
      } else if (a == "--ref") {
        ref_files.push_back(next("--ref"));
      } else if (a == "--dump-dir") {
        dump_dir = next("--dump-dir");
      } else if (a == "--reps") {
        reps = std::atoi(next("--reps"));
        if (reps <= 0) {
          std::fprintf(stderr, "kernscope-capture: --reps must be > 0\n");
          return 1;
        }
      } else if (a == "--no-flush") {
        l2_flush = false;
      } else if (a == "--no-validate") {
        validate = false;
      } else if (a == "--db") {
        db_path = next("--db");
      } else if (a == "--profile-out") {
        profile_out = next("--profile-out");
      } else if (a == "-h" || a == "--help") {
        (cmd == "replay" ? UsageReplay : UsageBench)(stdout);
        return 0;
      } else if (!a.empty() && a[0] == '-') {
        std::fprintf(stderr, "kernscope-capture: unknown option '%s'\n",
                     a.c_str());
        return 1;
      } else if (!have_bundle) {
        common.bundle_path = a;
        have_bundle = true;
      } else {
        std::fprintf(stderr, "kernscope-capture: unexpected argument '%s'\n",
                     a.c_str());
        return 1;
      }
    }
    if (!have_bundle) {
      std::fprintf(stderr, "kernscope-capture: %s requires <bundle>\n",
                   cmd.c_str());
      (cmd == "replay" ? UsageReplay : UsageBench)(stderr);
      return 1;
    }
    if (cmd == "replay") {
      ReplayOptions opts;
      static_cast<kernscope::CaptureBundleOptions &>(opts) = common;
      opts.ref_files = ref_files;
      opts.dump_dir = dump_dir;
      opts.profile_out = profile_out;
      if (!mode.empty()) {
        if (mode == "bitwise") {
          opts.tolerance = false;
        } else if (mode == "tolerance") {
          opts.tolerance = true;
        } else {
          std::fprintf(stderr,
                       "kernscope-capture: --mode must be bitwise|tolerance\n");
          return 1;
        }
      }
      return RunReplay(opts);
    }
    BenchOptions bopts;
    static_cast<kernscope::CaptureBundleOptions &>(bopts) = common;
    bopts.ref_files = ref_files;
    bopts.reps = reps;
    bopts.l2_flush = l2_flush;
    bopts.validate = validate;
    bopts.db_path = db_path;
    bopts.profile_out = profile_out;
    return RunBench(bopts);
  }

  if (cmd == "db-query") {
    std::string path;
    std::string arch;
    std::string family;
    std::string shape;
    bool have_path = false;
    for (int i = 2; i < argc; ++i) {
      const std::string a = argv[i];
      const auto next = [&](const char *flag) -> const char * {
        if (i + 1 >= argc) {
          std::fprintf(stderr, "kernscope-capture: %s requires a value\n",
                       flag);
          std::exit(1);
        }
        return argv[++i];
      };
      if (a == "--arch") {
        arch = next("--arch");
      } else if (a == "--family") {
        family = next("--family");
      } else if (a == "--shape") {
        shape = next("--shape");
      } else if (a == "-h" || a == "--help") {
        UsageDbQuery(stdout);
        return 0;
      } else if (!a.empty() && a[0] == '-') {
        std::fprintf(stderr, "kernscope-capture: unknown option '%s'\n",
                     a.c_str());
        return 1;
      } else if (!have_path) {
        path = a;
        have_path = true;
      } else {
        std::fprintf(stderr, "kernscope-capture: unexpected argument '%s'\n",
                     a.c_str());
        return 1;
      }
    }
    if (!have_path) {
      std::fprintf(stderr, "kernscope-capture: db-query requires <db-file>\n");
      UsageDbQuery(stderr);
      return 1;
    }
    return RunDbQuery(path, arch, family, shape);
  }

  std::fprintf(stderr, "kernscope-capture: unknown subcommand '%s'\n",
               cmd.c_str());
  Usage(stderr);
  return 1;
}
