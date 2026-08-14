//===- reduction_gen.h - axis-0 reduction template generator --------------===//
//
// kernscope tools/tune todo 26: the leading-axis sum-reduction template
// generator. Emits a single-source portable .hip kernel implementing
//
//     y[j] = sum_i x[i,j]   (reduce axis 0, fp32 accumulate, bf16 out)
//
// matching the golden reference (tests/golden refs.reduction). Each block
// reduces vec_width columns (grid = cols / vec_width); the row loop is
// block-strided over the leading axis with a per-access bound check, so any
// positive row count is valid. The `mode` dimension chooses the combine:
//   * kStrided: warp-shuffle butterfly (column-aware __shfl_xor offsets,
//     multiples of vec_width) + one cross-warp shared-memory combine;
//   * kTree: every thread's partial lands in shared memory and a halving
//     tree (offsets multiples of vec_width) completes the reduction.
//
// Search space: block {128,256,512,1024} x vec_width {1,2,4} (columns per
// block) x mode {kStrided, kTree}. Validity rules (libs/arch, rejected
// PRE-COMPILE with a reason): search-space membership, arch existence,
// block <= max threads/block, positive shape, cols % vec_width == 0 (each
// block's column set must be exact; a non-divisible shape is rejected, not
// padded).
//
// Kernel parameter contract (kernels/templates/README.md + runner):
// (const ks_bf16* x, ks_bf16* out, int rows, int cols) — one input pointer,
// one output pointer, two int32 scalars.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_REDUCTION_GEN_H
#define KERNSCOPE_TUNE_REDUCTION_GEN_H

#include "kernscope/tune/elem_common.h"

#include <cstdint>
#include <string>

namespace kernscope::tune {

struct ReductionConfig {
  int block = 256;        // threads per block
  int vec_width = 1;      // columns per block
  ElemMode mode = ElemMode::kStrided;  // combine strategy
};

ConfigVerdict validate_config(const ReductionConfig &cfg,
                              const ElemProblem &prob) noexcept;
LaunchGeometry launch_geometry(const ReductionConfig &cfg,
                               const ElemProblem &prob) noexcept;
std::string kernel_symbol(const ReductionConfig &cfg);
std::string generate_source(const ReductionConfig &cfg, const ElemProblem &prob,
                            std::string *error);
bool generate_to_file(const ReductionConfig &cfg, const ElemProblem &prob,
                      const std::string &path, std::string *error);

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_REDUCTION_GEN_H
