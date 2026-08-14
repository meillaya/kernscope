//===- layernorm_gen.h - layer-norm template generator --------------------===//
//
// kernscope tools/tune todo 26: the layer-norm template generator. Emits a
// single-source portable .hip kernel implementing normalization over the
// last axis (bf16 in, fp32 mean/variance, bf16 out)
//
//     y[r,c] = (x[r,c] - mean_r) / sqrt(var_r + eps) * gamma[c] + beta[c]
//
// matching the golden reference (tests/golden refs.layernorm): biased
// population variance (E[(x-E[x])^2]) and eps pinned to 1e-5. One block per
// row; row length cols == block * vec_width (unguarded, rejected pre-compile
// otherwise). The `mode` dimension chooses the algorithm:
//   * kOnePass: one walk over the row (register-resident) computing the
//     sum and sum of squares, combined with warp-shuffle reductions
//     (mean = s1/n, var = s2/n - mean^2);
//   * kTwoPass: the classic two-pass formula — pass 1 reduces sum -> mean
//     (shared-memory tree over the smem-staged row), pass 2 reduces the
//     squared deviations -> var. This is the formula the golden reference
//     describes ("matching the two-pass formula a device kernel computes").
//
// Search space: block {128,256,512,1024} x vec_width {1,2,4} x mode
// {kOnePass, kTwoPass}. Validity rules (libs/arch, rejected PRE-COMPILE with
// a reason): search-space membership, arch existence, block <= max
// threads/block, positive rows, cols % (block*vec_width) == 0, and the
// smem-staged row (block*vec_width fp32) within the arch's per-block smem
// budget.
//
// Kernel parameter contract (kernels/templates/README.md + runner):
// (const ks_bf16* x, const ks_bf16* gamma, const ks_bf16* beta, ks_bf16* out,
//  int rows, int cols) — three input pointers, one output pointer, two
// int32 scalars.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_LAYERNORM_GEN_H
#define KERNSCOPE_TUNE_LAYERNORM_GEN_H

#include "kernscope/tune/elem_common.h"

#include <cstdint>
#include <string>

namespace kernscope::tune {

struct LayerNormConfig {
  int block = 256;        // threads per block (one block per row)
  int vec_width = 1;      // elements per thread (row = block*vec)
  ElemMode mode = ElemMode::kOnePass;  // one-pass vs two-pass variance
};

ConfigVerdict validate_config(const LayerNormConfig &cfg,
                              const ElemProblem &prob) noexcept;
LaunchGeometry launch_geometry(const LayerNormConfig &cfg,
                               const ElemProblem &prob) noexcept;
std::string kernel_symbol(const LayerNormConfig &cfg);
std::string generate_source(const LayerNormConfig &cfg, const ElemProblem &prob,
                            std::string *error);
bool generate_to_file(const LayerNormConfig &cfg, const ElemProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_LAYERNORM_GEN_H
