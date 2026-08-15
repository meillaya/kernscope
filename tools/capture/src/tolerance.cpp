//===- tolerance.cpp - golden-contract tolerance comparison (todo 48) ------===//
//
// kernscope capture/replay track. See tolerance.h for the contract.
//
// Faithful C++ port of tests/golden/kernscope_golden/metrics.py — same
// formulas, same degenerate cases, same inclusive thresholds. The e2e suite
// asserts equality against the Python implementation on the same inputs.
//
// bf16 -> fp32 upcast: the bf16 value is the top 16 bits of an fp32; shifting
// the 16-bit pattern left by 16 yields the exact fp32 value (round-to-nearest
// is exact for this representation — bf16 is a truncated fp32 mantissa).
//
//===----------------------------------------------------------------------===//

#include "tolerance.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kernscope {
namespace {

inline constexpr double kCosineThreshold = 0.999;
inline constexpr double kMaxRelErrThreshold = 1e-2;
inline constexpr double kPccThreshold = 0.99;
inline constexpr double kRelErrEps = 1e-6;

std::vector<float> Bf16ToFp32(std::span<const std::uint8_t> bytes) {
  std::vector<float> out;
  out.reserve(bytes.size() / 2);
  for (std::size_t i = 0; i + 1 < bytes.size(); i += 2) {
    const std::uint16_t raw =
        static_cast<std::uint16_t>(bytes[i]) |
        (static_cast<std::uint16_t>(bytes[i + 1]) << 8);
    const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16;
    float f;
    static_assert(sizeof(f) == sizeof(bits));
    std::memcpy(&f, &bits, sizeof(f));
    out.push_back(f);
  }
  return out;
}

double Cosine(const std::vector<float> &x, const std::vector<float> &y) {
  double dot = 0.0, nx = 0.0, ny = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    dot += static_cast<double>(x[i]) * y[i];
    nx += static_cast<double>(x[i]) * x[i];
    ny += static_cast<double>(y[i]) * y[i];
  }
  const double denom = std::sqrt(nx) * std::sqrt(ny);
  if (denom == 0.0) {
    return (nx == 0.0 && ny == 0.0) ? 1.0 : 0.0;
  }
  return dot / denom;
}

double MaxRelErr(const std::vector<float> &x, const std::vector<float> &y) {
  double worst = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double d = std::abs(static_cast<double>(x[i]) - y[i]);
    worst = std::max(worst, d / (std::abs(static_cast<double>(y[i])) + kRelErrEps));
  }
  return worst;
}

double Pcc(const std::vector<float> &x, const std::vector<float> &y) {
  double mx = 0.0, my = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    mx += x[i];
    my += y[i];
  }
  mx /= static_cast<double>(x.size());
  my /= static_cast<double>(y.size());
  double cov = 0.0, vx = 0.0, vy = 0.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const double xc = static_cast<double>(x[i]) - mx;
    const double yc = static_cast<double>(y[i]) - my;
    cov += xc * yc;
    vx += xc * xc;
    vy += yc * yc;
  }
  const double denom = std::sqrt(vx) * std::sqrt(vy);
  if (denom == 0.0) {
    // metrics.py degenerate case: zero-variance tensors are perfectly
    // correlated when elementwise-equal, else uncorrelated
    return (x == y) ? 1.0 : 0.0;
  }
  return cov / denom;
}

} // namespace

ToleranceVerdict tolerance_verdict(std::span<const std::uint8_t> a,
                                   std::span<const std::uint8_t> b) {
  ToleranceVerdict out;
  if (a.size() < 2 || b.size() < 2) {
    return out;
  }
  const std::vector<float> x = Bf16ToFp32(a);
  const std::vector<float> y = Bf16ToFp32(b);
  const std::size_t n = std::min(x.size(), y.size());
  out.elements = n;
  if (n == 0) {
    return out;
  }
  const std::vector<float> xc(x.begin(), x.begin() + static_cast<std::ptrdiff_t>(n));
  const std::vector<float> yc(y.begin(), y.begin() + static_cast<std::ptrdiff_t>(n));
  out.metrics.cosine = Cosine(xc, yc);
  out.metrics.max_rel_err = MaxRelErr(xc, yc);
  out.metrics.pcc = Pcc(xc, yc);
  out.pass = out.metrics.cosine >= kCosineThreshold &&
             out.metrics.max_rel_err <= kMaxRelErrThreshold &&
             out.metrics.pcc >= kPccThreshold;
  return out;
}

} // namespace kernscope
