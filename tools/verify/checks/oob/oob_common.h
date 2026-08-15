// kernscope::verify::oob — shared helpers for the bounds check module
// (todo 31: check classes 1 global OOB + 2 shared-memory bounds).
//
// SOUNDNESS SCOPE (may-analysis — see tools/verify/checks/oob/README.md):
//   * Bounds are proven only for STATICALLY SIZED allocations:
//       - __device__ globals (addrspace(1)) with a fixed array type,
//       - __shared__ arrays (addrspace(3)) with a fixed array type,
//       - extern __shared__ (dynamic smem, addrspace(3), zero-sized global)
//         whose size is a CONSTANT launch-site smem argument.
//   * Accesses rooted at kernel pointer parameters have a RUNTIME allocation
//     size: the check proves the INDEX is bounded (SCEV range) and emits an
//     info note; it NEVER claims a static allocation proof for them.
//   * DATA-DEPENDENT indices (index not bounded by the lattice or SCEV) are
//     reported as MAY_OOB — the check does NOT claim completeness for them
//     (soundness over false safety; per-kernel verdict never SAFE when any
//     such access exists).
//   * The byte-span arithmetic is unsigned (negative GEP indices wrap into a
//     huge unsigned range and are conservatively flagged MAY_OOB).

#pragma once

#include <kernscope/verify/diagnostic.h>
#include <kernscope/verify/verifier_state.h>

#include <llvm/IR/ConstantRange.h>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace kernscope {
namespace verify {
namespace oob {

/// Verdict strings (this module's own vocabulary — the core has no verdict
/// enum for check classes; todo 35's CLI maps these into its verdict JSON).
inline constexpr const char* kVerdictSafe = "SAFE";
inline constexpr const char* kVerdictMayOob = "MAY_OOB";
inline constexpr const char* kVerdictSymbolicDims = "SYMBOLIC_DIMS";

/// One memory-access finding (per load/store the plugin inspects).
struct AccessFinding {
  enum class Kind : std::uint8_t {
    InBounds,      ///< proven inside a statically sized allocation.
    MayOob,        ///< proven (or conservatively possible) outside it.
    Unbounded,     ///< index range unbounded -> cannot prove (may-analysis).
    ParamBounded,  ///< runtime allocation (kernel param); index proven
                   ///< bounded; static allocation proof not possible.
  };
  Kind kind = Kind::Unbounded;
  int check_class = 0;      ///< CheckClass::kGlobalOob or kSmemBounds.
  bool is_store = false;
  SourceLoc loc;
  std::string alloc;        ///< base allocation name ("" when none).
  bool alloc_known = false; ///< statically sized allocation?
  uint64_t bound_bytes = 0; ///< allocation size in bytes (when known).
  std::string index_range;  ///< "[0,8224)" render of the innermost GEP index
                            ///< SCEV/lattice range ("" when unbounded).
  std::string span;         ///< "[0,32896)" byte-span render ("" unknown).
  std::string detail;       ///< human note.
};

/// Classification + size of the allocation a pointer expression is rooted in.
struct AllocationInfo {
  enum class Root : std::uint8_t {
    Unknown,  ///< not recognized (call result, addrspacecast, ...).
    DeviceGlobal,   ///< addrspace(1) __device__ global.
    SharedArray,    ///< addrspace(3) __shared__ array.
    SharedDynamic,  ///< addrspace(3) extern __shared__ (size from launch).
    KernelParam,    ///< pointer kernel argument (runtime size).
    Local,          ///< alloca (local memory) — not checked by this module.
  };
  Root root = Root::Unknown;
  std::string name;
  bool known_size = false;
  uint64_t size_bytes = 0;
};

/// Resolves the allocation a memory-access pointer is rooted in (walks
/// bitcast/addrspacecast chains and GEP pointer operands to the base).
AllocationInfo resolveAllocation(llvm::Value* ptr, const llvm::Module& m);

/// True when `v` (after stripping bitcast/addrspacecast) is a GEP chain.
bool isGepChain(llvm::Value* v);

/// Computes the accessed BYTE-span [first, last+1) of a pointer expression
/// (GEP chain; element sizes from the module DataLayout; index ranges via
/// VerifierContext::queryRange — lattice then SCEV).  Returns std::nullopt
/// when any index is unbounded or the expression is not a GEP chain.
/// `out_index_range` receives the render of the innermost element-index
/// range ("" when not applicable) for the diagnostic proof field;
/// `out_innermost_stride` receives the byte width of the accessed element
/// (the width the in-bounds check must add for the last byte).
std::optional<llvm::ConstantRange> gepByteSpan(llvm::Value* ptr,
                                               VerifierContext& ctx,
                                               std::string& out_index_range,
                                               uint64_t& out_innermost_stride);

/// Byte width of one access (load/store of `ty`), for direct (non-GEP)
/// accesses to scalar globals.
uint64_t accessByteWidth(llvm::Type* ty, const llvm::Module& m);

/// Launch-site scan: per kernel, the dynamic shared-memory size in bytes.
/// Value is std::nullopt when the kernel is launched with a non-constant
/// smem argument (or not launched) — the bound is then unknowable.
/// Mirrors the core's launch-record scan (launch_config.cpp) for the
/// cudaConfigureCall / __cudaPushCallConfiguration / cudaLaunchKernel forms.
using SmemSizeMap = std::unordered_map<const llvm::Function*, std::optional<uint64_t>>;
SmemSizeMap scanLaunchSmem(llvm::Module& m);

/// JSON escaping for diagnostic payload strings (module owns the format).
std::string jsonEscape(const std::string& s);

/// Renders a ConstantRange the way the core's state dump does ("[0,8224)").
std::string range_render(const llvm::ConstantRange& r);

}  // namespace oob
}  // namespace verify
}  // namespace kernscope
