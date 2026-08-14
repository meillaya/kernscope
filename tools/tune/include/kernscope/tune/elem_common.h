//===- elem_common.h - elementwise/reduction template generator (shared) ---===//
//
// kernscope tools/tune todo 26: shared types + inline helpers for the four
// elementwise/reduction template generators (bias_gelu, softmax, layernorm,
// reduction). Each family's header (bias_gelu_gen.h, softmax_gen.h,
// layernorm_gen.h, reduction_gen.h) builds on this file, mirroring the
// flash_attention_gen.h API shape (todo 27): a config struct, a pre-compile
// ConfigVerdict, launch geometry, a stable kernel symbol, and a source
// generator emitting single-source portable .hip kernels
// (kernels/templates convention: includes only kernscope/kernel_traits.h,
// compiles under nvcc -x cu for sm_80/sm_90 AND hipcc for gfx1101).
//
// ALGORITHM CONTRACT (must match tests/golden/kernscope_golden/refs.py):
//   * bias_gelu   y = GELU(x + bias): fp32 add of the bf16 operands, the sum
//                 bf16-rounded (the golden's fusion point), then the EXACT
//                 erf GELU 0.5*t*(1+erf(t/sqrt(2))) — the golden uses
//                 math.erf (C libm), NOT the tanh approximation;
//   * softmax     y = exp(x - rowmax) / rowsum(exp(x - rowmax)) over the last
//                 axis (stable, fp32), bf16 out;
//   * layernorm   y = (x-mean)/sqrt(var+eps)*gamma+beta over the last axis,
//                 biased population variance, eps = 1e-5 pinned (refs.py);
//   * reduction   y[j] = sum_i x[i,j] over axis 0 (fp32 accumulate).
// All families: bf16 IO, fp32 accumulate, bf16 output rounded exactly once
// (the todo-23 rounding contract).
//
// SEARCH SPACE (shared): block size {128, 256, 512, 1024}, vector width
// {1, 2, 4} (elements per thread for bias_gelu/softmax/layernorm; columns
// per block for reduction), plus a per-family reduction strategy
// (ElemMode). Validity rules (libs/arch, rejected PRE-COMPILE with a reason
// string): search-space membership, arch existence, block <= the arch's max
// threads/block, shape divisibility by the vector width, and — for the
// row-per-block families — the row length divisible by block*vec.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_TUNE_ELEM_COMMON_H
#define KERNSCOPE_TUNE_ELEM_COMMON_H

#include "kernscope/arch.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <string>

namespace kernscope::tune {

// --- reduction strategy (the per-family 'mode' search dimension) -----------
enum class ElemMode : std::uint8_t {
  kOnePass,  // softmax/layernorm: warp-shuffle reduction strategy
  kTwoPass,  // softmax/layernorm: shared-memory partial/tree strategy
  kStrided,  // reduction: block-strided loop + warp-shuffle combine
  kTree      // reduction: block-strided loop + shared-memory tree combine
};

inline const char *elem_mode_name(ElemMode mode) noexcept {
  switch (mode) {
  case ElemMode::kOnePass:
    return "onepass";
  case ElemMode::kTwoPass:
    return "twopass";
  case ElemMode::kStrided:
    return "strided";
  case ElemMode::kTree:
    return "tree";
  }
  return "unknown";
}

// --- the problem the generated kernel is validated against -----------------
struct ElemProblem {
  std::int64_t rows = 0;  // leading dim (rows to reduce over for reduction)
  std::int64_t cols = 0;  // last dim (row length / columns)
  std::string arch;       // libs/arch key ("gfx1101", "sm_80", ...)
  int dtype_bytes = 2;    // bytes/element (bf16)
};

// --- pre-compile validity verdict ------------------------------------------
struct ConfigVerdict {
  bool valid = true;
  std::string reason;  // empty when valid; the reject reason otherwise
};

// Launch geometry the generated kernels expect (grid covers the leading
// work in x; block is 1-D): see each family's header for the mapping.
struct LaunchGeometry {
  int grid_x = 1;
  int grid_y = 1;
  int block_x = 1;
  int block_y = 1;
};

// --- search space ----------------------------------------------------------
inline constexpr std::array<int, 4> kElemBlockChoices{128, 256, 512, 1024};
inline constexpr std::array<int, 3> kElemVecWidthChoices{1, 2, 4};

// --- shared implementation helpers (used by the four generators) -----------
namespace detail {

// Search-space membership + arch existence + block-limit + positivity checks
// shared by every family. `family` names the caller for the reason string.
inline ConfigVerdict elem_common_validate(const char *family, int block,
                                          int vec_width,
                                          const std::string &arch,
                                          std::int64_t rows,
                                          std::int64_t cols) noexcept {
  bool block_ok = false;
  for (const int b : kElemBlockChoices)
    block_ok = block_ok || (b == block);
  if (!block_ok)
    return {false, std::string(family) + ": block must be 128, 256, 512 or "
                   "1024 (search space)"};
  bool vec_ok = false;
  for (const int v : kElemVecWidthChoices)
    vec_ok = vec_ok || (v == vec_width);
  if (!vec_ok)
    return {false, std::string(family) + ": vec_width must be 1, 2 or 4 "
                   "(search space)"};

  if (rows <= 0 || cols <= 0)
    return {false, std::string(family) + ": shape (rows, cols) must be "
                   "positive (got " + std::to_string(rows) + ", " +
                       std::to_string(cols) + ")"};

  const ArchInfo *a = nullptr;
  try {
    a = &get_arch(arch);
  } catch (const std::out_of_range &) {
    return {false, "unknown arch '" + arch + "' (libs/arch DB)"};
  }
  if (block > 1024)
    return {false, std::string(family) + ": block " + std::to_string(block) +
                       " exceeds the 1024-thread hardware limit"};
  if (a->max_threads_per_sm > 0 && block > a->max_threads_per_sm)
    return {false, std::string(family) + ": block " + std::to_string(block) +
                       " > max threads/block " +
                       std::to_string(a->max_threads_per_sm) + " on " +
                       std::string(a->name)};
  return {true, ""};
}

// Write `content` to `path` (truncate). Returns false + *error on IO failure.
inline bool elem_write_file(const std::string &path,
                            const std::string &content,
                            std::string *error) {
  std::ofstream fh(path, std::ios::binary | std::ios::trunc);
  if (!fh) {
    if (error)
      *error = "cannot open " + path + " for writing";
    return false;
  }
  fh << content;
  if (!fh) {
    if (error)
      *error = "write failed for " + path;
    return false;
  }
  return true;
}

} // namespace detail

} // namespace kernscope::tune

#endif // KERNSCOPE_TUNE_ELEM_COMMON_H
