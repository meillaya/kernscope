//===- replay_model.h - shared capture-bundle replay plan (todo 48) -----*- C++ -*-===//
//
// kernscope capture/replay track.
//
// The SHARED model builder for the todo-48 replay engine and the todo-47
// standalone reproducer generator: both consume a capture bundle (v1 *.kcap/
// directory OR a v0 text bundle with sidecar blobs) and need the same plan:
//
//   * the target launch (grid/block/shared, args) — auto-selected (first
//     non-internal launch) or named with --kernel;
//   * the device-buffer plan: exact sizes from allocation records, input
//     content from H2D copy blobs / v1 input snapshots, output buffers from
//     D2H copy records / v1 output snapshots (with their captured bytes —
//     the replay VALIDATION reference, todo 48);
//   * the kernel binary (hsaco/cubin) with its sha256 provenance;
//   * the argument marshalling plan: pointer slots (original VA -> plan
//     buffer index) vs scalar slots (byte-exact LE value).
//
// The v0 text grammar here is PERMISSIVE on empty arg bytes (runtime-internal
// dispatches carry `[48:]` slots), which BundleReader's compatibility mode
// rejects — the model owns its own v0 parser exactly like repro_gen did
// before the extraction.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CAPTURE_REPLAY_MODEL_H
#define KERNSCOPE_CAPTURE_REPLAY_MODEL_H

#include "kernscope/bundle.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace kernscope {

//===--- the replay plan ----------------------------------------------------===//

enum class BufKind : std::uint8_t { kInput, kOutput, kSynthetic };

struct DevBuffer {
  std::string id;          // "b0001" / "input-buffer-1.bin" / ptr hex
  std::uint64_t va = 0;    // ORIGINAL device VA (0 = none)
  BufKind kind = BufKind::kSynthetic;
  std::size_t size = 0;    // byte count (synthetic: the planned size)
  std::vector<std::uint8_t> bytes; // embedded input content
  bool size_unknown = false;       // QA-negative marker
  bool blob_missing = false;       // input had no captured blob
  std::string warning;             // reported when synthetic
  // v1 only: the embedded-pointer patch table for this snapshot (todo 46).
  // Replay applies it through apply_remap (libs/capture-core) when the
  // snapshot is remapped to fresh VAs (docs/report/kernscope-technical-report.pdf).
  std::vector<PatchEntry> patches;
};

struct ArgPlan {
  std::size_t size = 0;     // slot byte size
  bool is_pointer = false;
  std::uint64_t pointer_va = 0; // captured pointer value (ORIGINAL VA)
  int buffer = -1;              // index into the plan's buffers (-1 = scalar)
  std::uint64_t scalar = 0;     // scalar value (LE, low `size` bytes)
};

/// One captured OUTPUT snapshot usable as the replay validation reference.
/// v1 bundles carry it (BufferRole::kOutput blobs); v0 bundles do not (the
/// caller supplies --ref files instead).
struct OutputRef {
  std::vector<std::uint8_t> bytes;
  std::string sha256;   // mandatory provenance
  std::string source;   // human description ("b0003" / "blobs/...")
};

struct Plan {
  std::string vendor, arch, mechanism, tool_version, timestamp, kernel_name;
  std::string bundle_path;
  // The target launch, OWNED by the plan: the source records live in the
  // BundleReader / v0 model which are destroyed when BuildPlan returns, so
  // a pointer into them would dangle. launch_index locates the record in
  // the source bundle's launch list (for arg-table lookups).
  LaunchRecord launch_rec;
  std::size_t launch_index = 0;
  std::vector<DevBuffer> buffers;
  std::vector<ArgPlan> args;
  std::vector<std::uint8_t> kernel_binary; // empty => not captured
  std::string kernel_binary_sha;
  bool has_synthetic = false;
  std::string out_sha; // captured output hash, when derivable ("" = n/a)
  // Captured OUTPUT snapshot bytes (v1), indexed like `buffers` (empty
  // vector for v0 bundles / buffers without a snapshot).
  std::vector<OutputRef> output_refs;
};

//===--- helpers -------------------------------------------------------------===//

bool IsInternalKernel(std::string_view name);

/// Strips the HIP ".kd" kernel-descriptor suffix (the symbol name in the
/// code object is the descriptor-less base name).
std::string StripKernelDescriptor(std::string_view name);

struct PlanOptions {
  std::string vendor;   // amd | nvidia — REQUIRED for v0 (v1: manifest wins)
  std::string arch;     // gfx1101 / sm_80 — REQUIRED for v0
  std::string blob_dir; // v0 sidecar blob dir (default: bundle's directory)
  std::string kernel;   // "" = auto-select the first non-internal launch
};

/// Builds the full replay plan for a bundle (v1 *.kcap/ dir or v0 text).
/// Returns false + a clear `error` on any failure (never crashes).
bool BuildPlan(const std::string &bundle_path, const PlanOptions &opts,
               Plan &plan, std::string *error);

} // namespace kernscope

#endif // KERNSCOPE_CAPTURE_REPLAY_MODEL_H
