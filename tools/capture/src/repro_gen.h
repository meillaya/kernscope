//===- repro_gen.h - kernscope-capture: standalone reproducer generator -----===//
//
// kernscope capture/replay track, todo 47.
//
// `emit-repro <bundle>` turns a capture bundle (v1 *.kcap/ directory OR the
// v0 text bundle todos 43/44/45 emit, BundleReader compatibility mode) into a
// STANDALONE plain-CMake C++ project that replays the captured launch:
//
//   * main.hip (vendor=amd) / main.cu (vendor=nvidia): embeds the captured
//     kernel binary (HSACO/cubin bytes) and the captured input buffers,
//     loads the code object through the module API (hipModuleLoadData /
//     cuModuleLoadData), patches the captured kernel arguments (pointer
//     slots are re-pointed at freshly allocated device buffers), launches
//     with the captured grid/block/shared configuration and dumps the
//     captured output buffers for diffing.
//   * CMakeLists.txt pinned to the bundle manifest's vendor/arch (toolkit
//     versions recorded in the provenance header; HIP builds the executable
//     with hipcc --offload-arch=<arch>, CUDA is COMPILE-ONLY with nvcc -c -
//     runtime validation on NVIDIA is owner-gated).
//   * buffers as embedded byte arrays (kernel_binary.inc / input_<n>.inc),
//     zero repo dependencies: the reproducer depends ONLY on the toolkit.
//
// Size-unknown buffers (a device pointer that is neither in the captured
// allocation map nor covered by a copy record with a known byte count) are
// replaced with a CLEARLY-MARKED synthetic input: the emitted source carries
// `KERNSCOPE_SYNTHETIC_INPUT` + a warning header and the generated buffer is
// a deterministic fill pattern (QA negative path - the reproducer still
// compiles and runs; it can never reproduce such a buffer bit-exactly, and
// the warning says so).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_REPRO_GEN_H
#define KERNSCOPE_REPRO_GEN_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace kernscope {

struct ReproduceOptions {
  std::string bundle_path;      // v1 *.kcap/ dir OR v0 text bundle file
  std::string out_dir;          // emitted project directory (created)
  std::string blob_dir;         // sidecar blob dir (v0); default: bundle dir
  std::string vendor;           // "amd" | "nvidia"; v1: manifest wins
  std::string arch;             // gfx1101 / sm_80; v1: manifest wins
  std::string kernel;           // select a launch by kernel name ("" = auto)
  bool overwrite = false;       // allow overwriting an existing out_dir
};

/// Emits the standalone reproducer project for the bundle's target launch.
/// Returns false + a clear `error` on any failure (never crashes).
bool emit_reproducer(const ReproduceOptions &opts, std::string *error);

} // namespace kernscope

#endif // KERNSCOPE_REPRO_GEN_H
