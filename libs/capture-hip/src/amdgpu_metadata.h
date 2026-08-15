//===- amdgpu_metadata.h - HSACO kernel argument layout -------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Extracts the per-argument layout of a kernel from an AMDGPU code object's
// NT_AMD_AMDGPU_METADATA note (msgpack-encoded, emitted by LLVM's
// amdgpu code object backend). The layout drives the byte-exact per-argument
// capture of libs/capture-hip: the HIP launch path (hipLaunchKernel) exposes
// only a host-side void** of argument pointers; the code object metadata is
// where the per-argument SIZES live (the same source the AMDGPU toolchain
// uses for kernarg layout).
//
// Layout of the note (verified against the todo-5 layernorm_fused HSACO on
// ROCm 7.2.3):
//
//   .note section, note { name: "AMDGPU", type: 7 (NT_AMD_AMDGPU_METADATA),
//     desc: msgpack object {
//       "amdhsa.version": [1, 2],
//       "amdhsa.kernels": [ { ".name": "_Z...", ".args": [
//           { ".offset": 0, ".size": 8, ".value_kind": "global_buffer", ... },
//           ... ] }, ... ] } }
//
// Self-contained (no msgpack/ELF dependency); pure host-side parsing.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_AMDGPU_METADATA_H
#define KERNSCOPE_AMDGPU_METADATA_H

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope {

/// One kernel argument slot from the AMDGPU metadata `.args` entry.
struct AmdArgLayout {
  std::uint64_t offset = 0;
  std::uint64_t size = 0;
  std::string value_kind; // "global_buffer", "by_value", "hidden_*", ...
  bool is_pointer = false; // value_kind denotes an address (global_buffer,
                           // hidden_global_offset_*, hidden_none, ...)
};

/// The parsed arg layout of one kernel (keyed by the MANGLED name, with the
/// trailing ".kd" that the dispatch layer appends stripped by the caller).
struct AmdKernelLayout {
  std::string name; // ".name" from the metadata (mangled)
  std::vector<AmdArgLayout> args;
};

/// Parses `hsaco` (an ELF64 AMDGPU code object) and returns the kernel arg
/// layouts from its NT_AMD_AMDGPU_METADATA note. Returns an empty vector when
/// the object is not a parseable AMDGPU ELF or has no metadata note (callers
/// degrade gracefully - arg capture is best-effort).
std::vector<AmdKernelLayout> parse_amdgpu_metadata(
    std::span<const std::uint8_t> hsaco);

} // namespace kernscope

#endif // KERNSCOPE_AMDGPU_METADATA_H
