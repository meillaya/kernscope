//===- main.cpp - kernscope-capture CLI (todo 47: emit-repro) ---------------===//
//
// kernscope capture/replay track. The C4 tool entry point. Todo 47 ships the
// `emit-repro` subcommand (standalone reproducer generation); todos 48/49
// append record|replay|bench.
//
//   kernscope-capture emit-repro <bundle> [--out <dir>] [--blob-dir <dir>]
//       [--vendor amd|nvidia] [--arch <arch>] [--kernel <name>] [--force]
//
// <bundle> is a v1 *.kcap/ directory OR a v0 text bundle file (BundleReader
// compatibility mode). For v0 bundles --vendor/--arch are REQUIRED (the v0
// format has no envelope); v1 reads them from the manifest.
//
// Exit codes: 0 success, 1 usage/argument error, 2 generation failure.
//
//===----------------------------------------------------------------------===//

#include "repro_gen.h"

#include "kernscope/bundle.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

void Usage(FILE *out) {
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
               "                   (default: the bundle file's directory)\n"
               "  --vendor <v>     amd | nvidia (REQUIRED for v0 bundles;\n"
               "                   v1 reads the manifest)\n"
               "  --arch <arch>    gfx1101, sm_80, ... (REQUIRED for v0)\n"
               "  --kernel <name>  replay a specific kernel launch\n"
               "                   (default: the first non-internal launch)\n"
               "  --force          overwrite a non-empty out dir\n"
               "  -h, --help       this help\n");
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
  if (cmd != "emit-repro") {
    std::fprintf(stderr, "kernscope-capture: unknown subcommand '%s'\n",
                 cmd.c_str());
    Usage(stderr);
    return 1;
  }

  kernscope::ReproduceOptions opts;
  bool have_bundle = false;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&](const char *flag) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-capture: %s requires a value\n", flag);
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
      Usage(stdout);
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
    Usage(stderr);
    return 1;
  }
  if (opts.out_dir.empty()) {
    opts.out_dir = "reproducer";
  }

  std::string error;
  if (!kernscope::emit_reproducer(opts, &error)) {
    std::fprintf(stderr, "kernscope-capture: emit-repro: %s\n", error.c_str());
    return 2;
  }
  std::printf("kernscope-capture: reproducer emitted to %s\n",
              opts.out_dir.c_str());
  return 0;
}
