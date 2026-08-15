//===- tolerance.h - golden-contract tolerance comparison (todo 48) -------*- C++ -*-===//
//
// kernscope capture/replay track.
//
// The opt-in tolerance comparison mode of the replay engine (todo 48),
// reusing the todo-23 golden-math CONTRACT of tests/golden/kernscope_golden/
// metrics.py (the repo's "libs/golden" reference implementation):
//
//   * comparison view: FLATTENED fp32 upcast of the compared tensors. The
//     captured kernel outputs are bf16 (2-byte chunks, little-endian), so
//     every chunk is upcast bf16 -> fp32 (rounding contract pinned by the
//     plan, todo 23);
//   * cosine similarity, max relative error (eps = 1e-6 in the denom) and
//     Pearson correlation coefficient, computed EXACTLY as metrics.py;
//   * thresholds are INCLUSIVE (>= / <=): cosine >= 0.999, max_rel_err
//     <= 1e-2, pcc >= 0.99 — PASS iff all three hold.
//
// The e2e suite cross-validates this C++ port against the Python golden
// implementation on the same byte pair (asserted equal within fp64 noise).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CAPTURE_TOLERANCE_H
#define KERNSCOPE_CAPTURE_TOLERANCE_H

#include <cstddef>
#include <cstdint>
#include <span>

namespace kernscope {

struct ToleranceMetrics {
  double cosine = 0.0;
  double max_rel_err = 0.0;
  double pcc = 0.0;
};

struct ToleranceVerdict {
  bool pass = false;
  ToleranceMetrics metrics;
  // Number of bf16 elements compared; odd trailing bytes are ignored
  // (documented: a captured buffer must be a whole number of bf16s).
  std::size_t elements = 0;
};

/// Tolerance-mode verdict over two byte buffers (bf16-chunked fp32-upcast
/// comparison view). Returns pass=true iff all three golden thresholds hold.
/// Empty input yields pass=false with zero metrics.
ToleranceVerdict tolerance_verdict(std::span<const std::uint8_t> a,
                                   std::span<const std::uint8_t> b);

} // namespace kernscope

#endif // KERNSCOPE_CAPTURE_TOLERANCE_H
