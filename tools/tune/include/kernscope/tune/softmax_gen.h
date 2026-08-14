//===- softmax_gen.h - softmax template generator -------------------------===//
//
// kernscope tools/tune todo 26: the row-softmax template generator. Emits a
// single-source portable .hip kernel implementing the numerically stable
// row softmax over the last axis (bf16 in, fp32 max/exp/sum, bf16 out)
//
//     y[r,c] = exp(x[r,c] - rowmax) / rowsum(exp(x - rowmax))
//
// matching the golden reference (tests/golden refs.softmax). One block per
// row; the row length cols == block * vec_width (unguarded by design, like
// the flash lite kernel — a non-divisible shape is rejected pre-compile, not
// padded). The `mode` dimension chooses the block reduction strategy:
//   * kOnePass: warp-shuffle reduction (per-warp __shfl_xor tree, one
//     cross-warp shared-memory combine) — the register-resident single walk;
//   * kTwoPass: shared-memory partial strategy (each thread's partial lands
//     in smem and a halving tree completes the reduction).
//
// Search space: block {128,256,512,1024} x vec_width {1,2,4} x mode
// {kOnePass, kTwoPass}. Validity rules (libs/arch, rejected PRE-COMPILE with
// a reason): search-space membership, arch existence, block <= max
// threads/block, positive rows, and cols % (block*vec_width) == 0.
//
// Kernel parameter contract (kernels/templates/README.md + runner):
// (const ks_bf16* x, ks_bf16* out, int rows, int cols) — one input pointer,
// one output pointer, two int32 scalars.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_SOFTMAX_GEN_H
#define KERNSCOPE_TUNE_SOFTMAX_GEN_H

#include "kernscope/tune/elem_common.h"

#include <cstdint>
#include <string>

namespace kernscope::tune {

struct SoftmaxConfig {
  int block = 256;        // threads per block (one block per row)
  int vec_width = 1;      // elements per thread (row = block*vec)
  ElemMode mode = ElemMode::kOnePass;  // reduction strategy
};

ConfigVerdict validate_config(const SoftmaxConfig &cfg,
                              const ElemProblem &prob) noexcept;
LaunchGeometry launch_geometry(const SoftmaxConfig &cfg,
                               const ElemProblem &prob) noexcept;
std::string kernel_symbol(const SoftmaxConfig &cfg);
std::string generate_source(const SoftmaxConfig &cfg, const ElemProblem &prob,
                            std::string *error);
bool generate_to_file(const SoftmaxConfig &cfg, const ElemProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_SOFTMAX_GEN_H
