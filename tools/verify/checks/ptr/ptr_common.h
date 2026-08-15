// kernscope::verify::ptr — shared helpers for the pointer-discipline check
// module (todo 34: check classes 5 illegal pointer arithmetic + 7
// uninitialized reads + 8 invalid address-space casts).
//
// SOUNDNESS SCOPE (may-analysis — see tools/verify/checks/ptr/README.md):
//   * Class 5 (PTR_OOB): a pointer expression's byte span is proven against
//     its allocation only when the base resolves to a STATICALLY SIZED
//     allocation (device global / shared array / extern shared with constant
//     launch-site smem) AND every variable GEP index is bounded by the
//     core's range lattice / ScalarEvolution. Unbounded index expressions
//     are reported PTR_OOB (may-analysis: no in-bounds claim).
//   * Class 8 (INVALID_ADDRSPACE_CAST): the addrspacecast legality table is
//     the VERIFIED NVPTX/AMDGPU address-space table of
//     docs/research/gpu-memory-pass-research.md §2 — NVPTX: only
//     generic(0) <-> {global(1), shared(3), const(4), local(5)} are legal
//     CUDA-level casts (all non-generic <-> non-generic pairs are invalid
//     at the hardware level); AMDGPU: only generic(0) <-> {global(1),
//     local(3), constant(4), private(5)} (flat-mediated). Clang never
//     emits the direct illegal pairs for well-formed CUDA, but IR from
//     hand-written/legacy sources or backend lowering can.
//   * Class 7 (MAY_UNINIT_READ): must/may-initialized lattice over the
//     kernel body. Kernel parameters and device globals start initialized;
//     allocas start uninitialized; __shared__ arrays start UNINITIALIZED
//     (their initialization is a CTA-cooperative act — the check tracks
//     per-address writes BEFORE reads along intra-kernel paths, with
//     __syncthreads() marking the smem write-before-read phase boundary).
//     KERNEL-BOUNDARY ASSUMPTION (documented; see README.md): the analysis
//     never reasons across kernel launches — a kernel's smem is assumed
//     uninitialized at entry regardless of prior kernels.
//
// The byte-span arithmetic is unsigned (negative GEP indices wrap into a
// huge unsigned range and are conservatively flagged PTR_OOB).

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
namespace ptr {

/// Verdict strings (this module's own vocabulary — the core has no verdict
/// enum for check classes; todo 35's CLI maps these into its verdict JSON).
inline constexpr const char* kVerdictSafe = "SAFE";
inline constexpr const char* kVerdictPtrOob = "PTR_OOB";
inline constexpr const char* kVerdictInvalidAddrspaceCast =
    "INVALID_ADDRSPACE_CAST";
inline constexpr const char* kVerdictMayUninitRead = "MAY_UNINIT_READ";

/// The verified address-space tables the class-8 diagnostic cites
/// (docs/research/gpu-memory-pass-research.md §2.1/§2.2; NVPTXUsage.rst +
/// AMDGPUUsage.rst canonical tables).
inline constexpr const char* kTableNvptx =
    "NVPTX address-space table (docs/research/gpu-memory-pass-research.md "
    "§2.1; NVPTXUsage.rst): 0=generic 1=global 2=internal 3=shared "
    "4=constant 5=local 6=tensor 7=shared-cluster 101=entry-param";
inline constexpr const char* kTableAmdgpu =
    "AMDGPU address-space table (docs/research/gpu-memory-pass-research.md "
    "§2.2; AMDGPUUsage.rst): 0=generic(flat) 1=global 2=region 3=local(LDS) "
    "4=constant 5=private 6=const32 7=buffer-fat 8=buffer-resource";

/// Classification of the allocation a pointer expression is rooted in.
struct AllocationInfo {
  enum class Root : std::uint8_t {
    Unknown,       ///< not recognized (call result, ..., ).
    DeviceGlobal,  ///< addrspace(1) __device__ global.
    SharedArray,   ///< addrspace(3) __shared__ array.
    SharedDynamic, ///< addrspace(3) extern __shared__ (size from launch).
    KernelParam,   ///< pointer kernel argument (runtime size).
    Local,         ///< alloca (local memory).
  };
  Root root = Root::Unknown;
  std::string name;
  bool known_size = false;
  uint64_t size_bytes = 0;
};

/// Resolves the allocation a pointer is rooted in (walks bitcast /
/// addrspacecast chains and GEP pointer operands to the base). Mirror of the
/// oob module's resolveAllocation — kept module-local per the isolation
/// contract.
AllocationInfo resolveAllocation(const llvm::Value* ptr, const llvm::Module& m);

/// True when `v` (after stripping bitcast/addrspacecast) is a GEP chain.
bool isGepChain(llvm::Value* v);

/// Computes the accessed BYTE-span [first, last+1) of a pointer expression
/// (GEP chain; element sizes from the module DataLayout; index ranges via
/// VerifierContext::queryRange — lattice then SCEV). Returns std::nullopt
/// when any index is unbounded or the expression is not a GEP chain.
/// `out_index_range` receives the render of the innermost element-index
/// range; `out_innermost_stride` the byte width of the accessed element.
std::optional<llvm::ConstantRange> gepByteSpan(llvm::Value* ptr,
                                               VerifierContext& ctx,
                                               std::string& out_index_range,
                                               uint64_t& out_innermost_stride);

/// Byte width of one access (load/store of `ty`), for direct (non-GEP)
/// accesses to scalar globals.
uint64_t accessByteWidth(llvm::Type* ty, const llvm::Module& m);

/// Launch-site scan: per kernel, the dynamic shared-memory size in bytes.
/// Value is std::nullopt when the kernel is launched with a non-constant
/// smem argument (or not launched). Mirror of the oob module's scan.
using SmemSizeMap =
    std::unordered_map<const llvm::Function*, std::optional<uint64_t>>;
SmemSizeMap scanLaunchSmem(llvm::Module& m);

/// JSON escaping for diagnostic payload strings (module owns the format).
std::string jsonEscape(const std::string& s);

/// Renders a ConstantRange the way the core's state dump does ("[0,8224)").
std::string range_render(const llvm::ConstantRange& r);

}  // namespace ptr
}  // namespace verify
}  // namespace kernscope
