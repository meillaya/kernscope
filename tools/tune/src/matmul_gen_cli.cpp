//===- matmul_gen_cli.cpp - kernscope-tune-gen CLI ------------------------===//
//
// kernscope tools/tune todo 25: the `kernscope-tune-gen` command-line driver
// for the matmul template generator. Emits the generated single-source .hip
// kernel for one (arch, config) to a file, or exits non-zero with the
// pre-compile validity reason when the config is rejected.
//
//   kernscope-tune-gen --arch sm_80|gfx1101|... --bm 64 --bn 64 --bk 16
//                      --warps 1 --stages 1 --vec 1 --layout padded|xor
//                      --unroll 1 --out kernel.hip
//
// The problem shape is NOT a CLI input: M/N/K are runtime kernel arguments
// (the kernel is bounds-checked), so validation depends only on the arch
// budget + the config; a nominal 64x64x64 problem is used internally.
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/matmul_gen.h"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --arch <sm_80|sm_90|gfx1101|...> --bm 64|128"
               " --bn 64|128 --bk 8|16|32|64\n"
               "             --warps 1|2|4|8 --stages 1|2|3|4"
               " --vec 1|2|4 --layout padded|xor\n"
               "             --unroll 1|2|4 --out <file>\n",
               argv0);
}

bool parse_int(const char *s, int *out) {
  if (!s || !*s)
    return false;
  char *end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0' || v < -0x7FFFFFFF || v > 0x7FFFFFFF)
    return false;
  *out = static_cast<int>(v);
  return true;
}

} // namespace

int main(int argc, char **argv) {
  kernscope::tune::MatmulConfig cfg;
  std::string arch;
  std::string out_path;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-tune-gen: %s requires an argument\n",
                     name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--arch")
      arch = need("--arch");
    else if (a == "--bm")
      parse_int(need("--bm"), &cfg.tile_m);
    else if (a == "--bn")
      parse_int(need("--bn"), &cfg.tile_n);
    else if (a == "--bk")
      parse_int(need("--bk"), &cfg.tile_k);
    else if (a == "--warps")
      parse_int(need("--warps"), &cfg.warps);
    else if (a == "--stages")
      parse_int(need("--stages"), &cfg.stages);
    else if (a == "--vec")
      parse_int(need("--vec"), &cfg.vec_width);
    else if (a == "--layout") {
      const std::string lay = need("--layout");
      if (lay == "padded" || lay == "padding") {
        cfg.smem_layout = kernscope::tune::SmemLayout::kPadding;
      } else if (lay == "xor") {
        cfg.smem_layout = kernscope::tune::SmemLayout::kXorSwizzle;
      } else {
        std::fprintf(stderr, "kernscope-tune-gen: bad --layout '%s' "
                             "(padded|xor)\n", lay.c_str());
        return 1;
      }
    } else if (a == "--unroll")
      parse_int(need("--unroll"), &cfg.unroll);
    else if (a == "--out")
      out_path = need("--out");
    else {
      std::fprintf(stderr, "kernscope-tune-gen: unknown option '%s'\n",
                   a.c_str());
      usage(argv[0]);
      return 1;
    }
  }

  if (arch.empty() || out_path.empty()) {
    usage(argv[0]);
    return 1;
  }

  kernscope::tune::MatmulProblem prob;
  prob.m = 64;
  prob.n = 64;
  prob.k = 64;
  prob.arch = arch;
  prob.dtype_bytes = 2;

  const kernscope::tune::ConfigVerdict verdict =
      kernscope::tune::validate_config(cfg, prob);
  if (!verdict.valid) {
    std::fprintf(stderr, "kernscope-tune-gen: invalid matmul config: %s\n",
                 verdict.reason.c_str());
    return 1;
  }

  std::string err;
  if (!kernscope::tune::generate_to_file(cfg, prob, out_path, &err)) {
    std::fprintf(stderr, "kernscope-tune-gen: %s\n", err.c_str());
    return 1;
  }
  std::printf("kernscope-tune-gen: wrote %s (symbol %s)\n", out_path.c_str(),
              kernscope::tune::kernel_symbol(cfg).c_str());
  return 0;
}
