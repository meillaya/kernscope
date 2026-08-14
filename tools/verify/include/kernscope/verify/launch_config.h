// kernscope::verify — kernel launch-config tiers (todo 30).
//
// Per research brief §4.6 the launch configuration lives on the HOST side of
// the LLVM IR: clang lowers `kernel<<<grid,block>>>` to
// `__cudaPushCallConfiguration(grid, block, smem, stream)` +
// `cudaLaunchKernel(&__device_stub__kernel, ...)` (or the legacy
// `cudaConfigureCall`). The verifier therefore ingests host+device in ONE
// module by default and resolves, per kernel, one of three tiers:
//
//   (a) LaunchTier::Exact      — constant dims at the launch site: exact
//                                ranges propagate to threadIdx/blockIdx
//                                intrinsics (threadIdx.x in [0, blockDim.x),
//                                blockIdx.x in [0, gridDim.x)).
//   (b) LaunchTier::UpperBound — no exact site (or device-only mode) but the
//                                kernel carries `__launch_bounds__` /
//                                nvvm.maxntid: sound UPPER bound on blockDim.
//   (c) LaunchTier::Symbolic   — dims come from runtime values: degraded
//                                "symbolic dims" verdict + a runtime-check
//                                suggestion string instead of exact proof.
//
// Tier (a) is the default ingestion contract; --device-only (secondary,
// documented mode) can only reach (b)/(c).

#pragma once

#include <cstdint>
#include <string>

namespace kernscope {
namespace verify {

/// One launch dimension (grid or block), defaulting to 1 like CUDA dim3.
struct Dim3 {
  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;

  bool operator==(const Dim3& o) const { return x == o.x && y == o.y && z == o.z; }
  bool operator!=(const Dim3& o) const { return !(*this == o); }

  /// "32x1x1" style rendering.
  std::string render() const;
};

enum class LaunchTier {
  Exact,      ///< (a) constant dims at the launch site — exact ranges.
  UpperBound, ///< (b) nvvm.maxntid / __launch_bounds__ — sound upper bound.
  Symbolic,   ///< (c) runtime dims — degraded verdict + runtime check.
};

const char* launch_tier_name(LaunchTier t);

/// Resolved launch configuration for one kernel (verifier state dump input).
struct LaunchConfig {
  LaunchTier tier = LaunchTier::Symbolic;
  Dim3 block;   ///< Exact dims (tier a) or sound upper bound (tier b).
  Dim3 grid;    ///< Exact dims (tier a); meaningful only there.
  /// Provenance: "launch site <file>:<line>" (a), "nvvm.maxntid" (b),
  /// "symbolic (no constant launch site)" (c).
  std::string origin;
  /// Tier (c) only: suggestion for a runtime check that would make the
  /// config exact (e.g. "assert(blockDim.x <= 256) before launch").
  std::string runtime_check_suggestion;

  bool exact() const { return tier == LaunchTier::Exact; }
  bool bounded() const { return tier == LaunchTier::UpperBound; }
  bool symbolic() const { return tier == LaunchTier::Symbolic; }

  /// "Exact block 256x1x1 grid 32x1x1 @ launch site foo.cu:12"
  std::string render() const;
};

}  // namespace verify
}  // namespace kernscope
