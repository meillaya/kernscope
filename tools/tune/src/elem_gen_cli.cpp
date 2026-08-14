//===- elem_gen_cli.cpp - elementwise/reduction generator CLI -------------===//
//
// kernscope tools/tune todo 26: `kernscope-tune-elem-gen` — the command-line
// front end for the four elementwise/reduction template generators
// (bias_gelu, softmax, layernorm, reduction). One invocation emits ONE
// single-source .hip kernel for a validated config; invalid configs are
// rejected PRE-COMPILE with a reason string and a non-zero exit (the failure
// mode the tuner driver, todo 28, depends on).
//
// Usage:
//   kernscope-tune-elem-gen --kernel <family> --arch <key> --block <N>
//       --vec <N> [--mode onepass|twopass|strided|tree] --shape R,C
//       --out <file.hip>
//
//   --mode defaults per family: bias_gelu (unused), softmax/layernorm ->
//   onepass, reduction -> strided. --shape R,C is the problem the kernel is
//   validated and generated for (the tuner passes candidate shapes too).
//
//===----------------------------------------------------------------------===//

#include "kernscope/tune/bias_gelu_gen.h"
#include "kernscope/tune/layernorm_gen.h"
#include "kernscope/tune/reduction_gen.h"
#include "kernscope/tune/softmax_gen.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

void usage(const char *argv0) {
  std::fprintf(stderr,
               "usage: %s --kernel <family> --arch <key> --block <N> "
               "--vec <N> [--mode onepass|twopass|strided|tree] "
               "--shape R,C --out <file>\n"
               "  family: bias_gelu (mode unused) | softmax (onepass|twopass)\n"
               "        | layernorm (onepass|twopass) | reduction "
               "(strided|tree)\n",
               argv0);
}

const char *kFamilies[] = {"bias_gelu", "softmax", "layernorm", "reduction"};

bool parse_shape(const std::string &s, std::int64_t *rows,
                 std::int64_t *cols) {
  const std::size_t comma = s.find(',');
  if (comma == std::string::npos)
    return false;
  try {
    *rows = std::stoll(s.substr(0, comma));
    *cols = std::stoll(s.substr(comma + 1));
  } catch (...) {
    return false;
  }
  return *rows > 0 && *cols > 0;
}

} // namespace

int main(int argc, char **argv) {
  std::string kernel, arch, mode, shape, out_path;
  int block = 0, vec = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "kernscope-tune-elem-gen: %s requires an "
                             "argument\n", name);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--kernel")
      kernel = need("--kernel");
    else if (a == "--arch")
      arch = need("--arch");
    else if (a == "--block")
      block = std::atoi(need("--block"));
    else if (a == "--vec")
      vec = std::atoi(need("--vec"));
    else if (a == "--mode")
      mode = need("--mode");
    else if (a == "--shape")
      shape = need("--shape");
    else if (a == "--out")
      out_path = need("--out");
    else {
      std::fprintf(stderr, "kernscope-tune-elem-gen: unknown option %s\n",
                   a.c_str());
      usage(argv[0]);
      return 2;
    }
  }

  bool known = false;
  for (const char *f : kFamilies)
    known = known || (kernel == f);
  if (!known) {
    std::fprintf(stderr, "kernscope-tune-elem-gen: unknown --kernel '%s'\n",
                 kernel.c_str());
    usage(argv[0]);
    return 2;
  }
  if (arch.empty() || block <= 0 || vec <= 0 || out_path.empty()) {
    usage(argv[0]);
    return 2;
  }
  std::int64_t rows = 0, cols = 0;
  if (!parse_shape(shape, &rows, &cols)) {
    std::fprintf(stderr, "kernscope-tune-elem-gen: --shape must be R,C "
                         "(positive integers)\n");
    return 2;
  }

  kernscope::tune::ElemProblem prob;
  prob.rows = rows;
  prob.cols = cols;
  prob.arch = arch;
  prob.dtype_bytes = 2;

  std::string symbol;
  std::string err;
  bool ok = false;

  if (kernel == "bias_gelu") {
    kernscope::tune::BiasGeluConfig cfg;
    cfg.block = block;
    cfg.vec_width = vec;
    const auto vd = kernscope::tune::validate_config(cfg, prob);
    if (!vd.valid) {
      std::fprintf(stderr, "kernscope-tune-elem-gen: rejected: %s\n",
                   vd.reason.c_str());
      return 1;
    }
    symbol = kernscope::tune::kernel_symbol(cfg);
    ok = kernscope::tune::generate_to_file(cfg, prob, out_path, &err);
  } else if (kernel == "softmax") {
    kernscope::tune::SoftmaxConfig cfg;
    cfg.block = block;
    cfg.vec_width = vec;
    if (!mode.empty() && mode != "onepass" && mode != "twopass") {
      std::fprintf(stderr, "kernscope-tune-elem-gen: softmax --mode must be "
                           "onepass or twopass (got '%s')\n", mode.c_str());
      return 1;
    }
    cfg.mode = mode == "twopass" ? kernscope::tune::ElemMode::kTwoPass
                                 : kernscope::tune::ElemMode::kOnePass;
    const auto vd = kernscope::tune::validate_config(cfg, prob);
    if (!vd.valid) {
      std::fprintf(stderr, "kernscope-tune-elem-gen: rejected: %s\n",
                   vd.reason.c_str());
      return 1;
    }
    symbol = kernscope::tune::kernel_symbol(cfg);
    ok = kernscope::tune::generate_to_file(cfg, prob, out_path, &err);
  } else if (kernel == "layernorm") {
    kernscope::tune::LayerNormConfig cfg;
    cfg.block = block;
    cfg.vec_width = vec;
    if (!mode.empty() && mode != "onepass" && mode != "twopass") {
      std::fprintf(stderr, "kernscope-tune-elem-gen: layernorm --mode must be "
                           "onepass or twopass (got '%s')\n", mode.c_str());
      return 1;
    }
    cfg.mode = mode == "twopass" ? kernscope::tune::ElemMode::kTwoPass
                                 : kernscope::tune::ElemMode::kOnePass;
    const auto vd = kernscope::tune::validate_config(cfg, prob);
    if (!vd.valid) {
      std::fprintf(stderr, "kernscope-tune-elem-gen: rejected: %s\n",
                   vd.reason.c_str());
      return 1;
    }
    symbol = kernscope::tune::kernel_symbol(cfg);
    ok = kernscope::tune::generate_to_file(cfg, prob, out_path, &err);
  } else {  // reduction
    kernscope::tune::ReductionConfig cfg;
    cfg.block = block;
    cfg.vec_width = vec;
    if (!mode.empty() && mode != "strided" && mode != "tree") {
      std::fprintf(stderr, "kernscope-tune-elem-gen: reduction --mode must be "
                           "strided or tree (got '%s')\n", mode.c_str());
      return 1;
    }
    cfg.mode = mode == "tree" ? kernscope::tune::ElemMode::kTree
                              : kernscope::tune::ElemMode::kStrided;
    const auto vd = kernscope::tune::validate_config(cfg, prob);
    if (!vd.valid) {
      std::fprintf(stderr, "kernscope-tune-elem-gen: rejected: %s\n",
                   vd.reason.c_str());
      return 1;
    }
    symbol = kernscope::tune::kernel_symbol(cfg);
    ok = kernscope::tune::generate_to_file(cfg, prob, out_path, &err);
  }

  if (!ok) {
    std::fprintf(stderr, "kernscope-tune-elem-gen: %s\n", err.c_str());
    return 1;
  }
  std::printf("kernscope-tune-elem-gen: ok kernel=%s symbol=%s out=%s\n",
              kernel.c_str(), symbol.c_str(), out_path.c_str());
  return 0;
}
