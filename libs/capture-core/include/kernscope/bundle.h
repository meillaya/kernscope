//===- bundle.h - capture-core: v1 bundle format + reader (todo 46) -------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// The FORMAL capture bundle format (docs/capture-format.md, todo 46): a
// directory-based `*.kcap/` bundle with a JSON manifest, a formalized event
// log with an arg marshalling table, and VA-faithful buffer blobs carrying
// mandatory sha256 provenance and embedded-pointer patch tables (the Kerncap
// address-space-closure policy, research brief §5). This header is the
// writer/reader API; the v0 BundleWriter (capture.h) is UNCHANGED - v1 is an
// upgrade layer the existing mechanisms (shim/cupti-sub/capture-hip) are not
// migrated to here (their record semantics are frozen; BundleReader loads
// their v0 output - see docs/capture-format.md §7 for the compatibility
// decision).
//
// JSON is produced/parsed with the repo's dependency-free model
// (kernscope::db::json::Value, libs/db - the todo 8 conventions): canonical
// lexicographic-key serialization and byte-offset parse errors, which the
// reader's schema-violation reporting (truncation QA) is built on.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_BUNDLE_H
#define KERNSCOPE_BUNDLE_H

#include "kernscope/capture.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace kernscope {

//===--- v1 format constants --------------------------------------------------===//
// The bundle directory suffix is gitignored repo-wide (`*.kcap/`,
// .gitignore §5.2 of docs/capture-format.md).

inline constexpr std::string_view kBundleFormatId = "kernscope-capture";
inline constexpr int kBundleSchemaVersion = 1;
/// --max-snapshot-bytes default: 1 GiB, configurable per write (plan todo 46
/// pins the default; the todo-48 CLI exposes the flag).
inline constexpr std::size_t kDefaultMaxSnapshotBytes = (std::size_t{1} << 30);

//===--- VA-faithful buffer blobs ----------------------------------------------===//

/// What a captured blob IS (the plan's cubin/hsaco + input/output classes).
enum class BufferRole : std::uint8_t {
  kKernelBinary, // cubin / hsaco code object
  kInput,        // input buffer snapshot
  kOutput,       // output buffer snapshot
  kArg,          // kernel-argument-referenced buffer
  kState,        // other device state
};

const char *to_string(BufferRole role);

/// One embedded-pointer patch candidate: a deep pointer (T** indirection
/// table entry, array slot, ...) inside a blob whose 8-byte little-endian
/// value at `offset` equals the VA of a known allocation (docs/capture-format
/// .md §4.3 - the zero-type-info scan heuristic).
struct PatchEntry {
  std::uint64_t offset = 0;    // byte offset inside the blob
  std::uint64_t target_va = 0; // ORIGINAL VA of the pointed-to allocation
};

/// One blob record: bytes + ORIGINAL virtual address + provenance + patches.
struct BufferSnapshot {
  std::string id;                       // "b0001"... (write order)
  BufferRole role = BufferRole::kInput;
  std::uint64_t va = 0;                 // ORIGINAL VA (VA-faithful); 0 = n/a
  std::uint64_t size = 0;               // byte count
  std::string sha256;                   // MANDATORY provenance (sha256 of bytes)
  std::string file;                     // path relative to the bundle dir
  std::vector<PatchEntry> patches;      // embedded-pointer patch table
};

//===--- write options / size governance ---------------------------------------===//

struct BundleWriteOptions {
  /// Size governance (docs/capture-format.md §5.1): total blob bytes budget;
  /// exceeded -> WriteStatus::kAbortedSizeCap + partial manifest. Default
  /// pinned at 1 GiB.
  std::size_t max_snapshot_bytes = kDefaultMaxSnapshotBytes;
  std::string vendor = "unknown";            // nvidia | amd | mock
  std::string arch;                          // sm_80 / gfx1101 / ...
  std::string tool_version = "kernscope-capture/1.0";
  std::string capture_mechanism = "preload"; // cupti | preload | hip-sdk
  std::string timestamp;                     // ISO-8601 UTC; empty -> auto
  /// Snapshot requests (VA-faithful buffer blobs). `va` is the buffer's
  /// ORIGINAL device VA; `read_fn` must return its bytes.
  std::vector<BufferSnapshot> snapshots;
  /// Fetches the bytes for a requested snapshot. Return false -> kIoError.
  std::function<bool(const BufferSnapshot &, std::vector<std::uint8_t> &)> read_fn;
  /// Run the embedded-pointer scan over every snapped blob (§4.3).
  bool scan_embedded_pointers = true;
};

enum class WriteStatus {
  kOk,
  kAbortedSizeCap, // snapshot exceeded --max-snapshot-bytes; partial manifest
  kIoError,
  kInvalid,        // bad options / session
};

/// Deterministic embedded-pointer scan: every byte offset of `blob` is
/// checked for an 8-byte little-endian value exactly equal to any VA in
/// `vas`; each hit becomes a PatchEntry (docs/capture-format.md §4.3).
std::vector<PatchEntry> scan_embedded_pointers(
    std::span<const std::uint8_t> blob, std::span<const std::uint64_t> vas);

/// Applies a remap table to `blob` in place: for every patch entry whose
/// target_va is present in `remap`, the 8 bytes at `offset` are rewritten to
/// the new VA (little-endian). Entries whose target was not remapped are
/// left untouched (the patch table is a candidate table by design).
void apply_remap(std::span<std::uint8_t> blob,
                 const std::vector<PatchEntry> &patches,
                 const std::unordered_map<std::uint64_t, std::uint64_t> &remap);

/// Writes a v1 directory bundle (manifest.json, events.json, blobs.json,
/// blobs/). `dir` is the bundle directory (".kcap" suffix appended when
/// absent). Blob budgets are checked BEFORE each blob write: an over-cap
/// snapshot aborts with a partial manifest (aborted=true + abort_reason) and
/// returns kAbortedSizeCap - the partial bundle stays loadable.
WriteStatus write_bundle_v1(const CaptureSession &session, const std::string &dir,
                            const BundleWriteOptions &opts, std::string *error);

/// A captured launch's arg-marshalling enrichment (docs/capture-format.md
/// §3.1): the reader keeps the byte-exact KernelArgSlot in LaunchRecord and
/// exposes the slot KIND (scalar|pointer) + original VA separately, so
/// field-exact roundtrips compare against the session model unchanged.
struct ArgSlotInfo {
  bool is_pointer = false;
  std::uint64_t va = 0; // original device VA when is_pointer
};

/// One formalized param_info record (v1 event; the v0 writer dropped it).
struct ParamInfoRecord {
  std::uint64_t cufunction = 0;
  std::size_t param_index = 0;
  std::size_t offset = 0;
  std::size_t size = 0;
};

//===--- reader ----------------------------------------------------------------===//
// Loads v1 directory bundles AND v0 line-oriented text bundles (the
// todos-43/44/45 output shape; docs/capture-format.md §7). Errors carry byte
// offsets for JSON ("byte N: ...") and line/column for v0 text - the
// truncated-bundle QA (no UB, no crash). Two corruption defenses (F3):
// the v0 parser rejects an unterminated final line (a 1-byte trailing cut
// of the 'end-capture-v0' terminator is a schema violation, never a clean
// EOF), and v1 blob provenance is verified EAGERLY at open (sha256 + size
// against the recorded blobs.json reference; mismatch/truncation names the
// blob path + offset).

class BundleReader {
public:
  struct Impl; // loader helpers live out-of-line and need access (the same
               // public-data pattern CaptureSession uses)

  /// Opens `path`: a v1 *.kcap/ directory OR a v0 text bundle file (or a
  /// directory containing one). nullptr + `error` on any failure.
  static std::unique_ptr<BundleReader> open(const std::string &path,
                                            std::string *error);

  BundleReader();
  ~BundleReader();
  BundleReader(const BundleReader &) = delete;
  BundleReader &operator=(const BundleReader &) = delete;

  // --- manifest ----------------------------------------------------------
  bool is_v1() const; // false => v0 text bundle (compatibility mode)
  std::string format_id() const;
  int schema_version() const;
  std::string tool_version() const;
  std::string vendor() const;
  std::string arch() const;
  std::string capture_mechanism() const;
  std::string timestamp() const;
  std::string first_kernel() const;
  std::uint64_t max_snapshot_bytes() const;
  std::uint64_t total_blob_bytes() const;
  bool aborted() const;              // size-cap abort: partial bundle
  std::string abort_reason() const;

  // --- event log (same record structs as CaptureSession) ------------------
  const std::vector<ApiTableEntry> &api() const;
  const std::vector<AllocRecord> &allocs() const;
  const std::vector<UnknownFreeRecord> &unknown_frees() const;
  const std::vector<LaunchRecord> &launches() const;
  const std::vector<CopyRecord> &copies() const;
  const std::vector<MemsetRecord> &memsets() const;
  const std::vector<StreamRecord> &streams() const;
  const std::vector<ModuleRecord> &modules() const;
  const std::vector<ParamInfoRecord> &param_infos() const;
  /// Arg-marshalling enrichment, indexed like launches(): per-launch
  /// slot kind/VA. Empty when the bundle carried no arg tables (v0).
  const std::vector<std::vector<ArgSlotInfo>> &launch_arg_tables() const;

  // --- blobs (v1) ----------------------------------------------------------
  const std::vector<BufferSnapshot> &blobs() const;
  /// Reads the blob's bytes AND verifies its sha256 provenance; false + a
  /// clear error on I/O failure or hash mismatch (never silent corruption).
  /// NOTE: open() already verified every recorded blob eagerly (sha256 +
  /// size vs blobs.json), so this is a defense-in-depth re-check.
  bool read_blob(const BufferSnapshot &blob, std::vector<std::uint8_t> &out,
                 std::string *error) const;

private:
  std::unique_ptr<Impl> impl_;
};

} // namespace kernscope

#endif // KERNSCOPE_BUNDLE_H
