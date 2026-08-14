//===- bias_gelu_gen.h - bias+GELU template generator ---------------------===//
//
// kernscope tools/tune todo 26: the fused bias+GELU elementwise template
// generator. Emits a single-source portable .hip kernel
// (kernels/templates convention) implementing
//
//     y[i] = GELU(x[i] + bias[i % cols])   (bf16 in, fp32 add + erf, bf16 out)
//
// with the EXACT erf GELU 0.5*t*(1+erf(t/sqrt(2))) matching the golden
// reference (tests/golden refs.bias_gelu: the bias add happens in fp32 after
// both operands are bf16-rounded, the sum is bf16-rounded at the fusion
// point, and the erf is the C-libm exact form — NOT the tanh approximation;
// max_rel_err is sensitive to this, see refs.py L27-29). The generated
// kernel reproduces that pipeline step for step.
//
// Search space: block {128,256,512,1024} x vec_width {1,2,4} (elements per
// thread). The grid covers ceil(total / (block*vec)) elements; every access
// is bounds-checked against the runtime `total`. Validity rules (libs/arch,
// rejected PRE-COMPILE with a reason): search-space membership, arch
// existence, block <= the arch's max threads/block, positive shape, and the
// vector-width divisibility rule cols % vec_width == 0.
//
// Kernel parameter contract (kernels/templates/README.md + runner):
// (const ks_bf16* x, const ks_bf16* bias, ks_bf16* out, int total, int cols)
// — two input device pointers, one output pointer, two int32 scalars
// (flat element count + row length for the bias broadcast).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_BIAS_GELU_GEN_H
#define KERNSCOPE_TUNE_BIAS_GELU_GEN_H

#include "kernscope/tune/elem_common.h"

#include <cstdint>
#include <string>

namespace kernscope::tune {

struct BiasGeluConfig {
  int block = 256;       // threads per block
  int vec_width = 1;     // elements per thread
};

ConfigVerdict validate_config(const BiasGeluConfig &cfg,
                              const ElemProblem &prob) noexcept;
LaunchGeometry launch_geometry(const BiasGeluConfig &cfg,
                               const ElemProblem &prob) noexcept;
std::string kernel_symbol(const BiasGeluConfig &cfg);
std::string generate_source(const BiasGeluConfig &cfg, const ElemProblem &prob,
                            std::string *error);
bool generate_to_file(const BiasGeluConfig &cfg, const ElemProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_BIAS_GELU_GEN_H
