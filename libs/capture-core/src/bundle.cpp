//===- bundle.cpp - v1 bundle writer/reader implementation (todo 46) -------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Implements docs/capture-format.md: the v1 directory bundle (manifest.json,
// events.json with the arg marshalling table, blobs.json + blobs/ with
// VA-faithful snapshots, mandatory sha256 provenance and embedded-pointer
// patch tables) and the dual-format BundleReader (v1 directories + v0 text
// bundles from todos 43/44/45 - the compatibility decision, format doc §7).
//
// JSON uses kernscope::db::json::Value (todo 8 conventions): canonical dump
// and byte-offset parse errors. All 64-bit fields are hex strings
// (format doc §1.1). Pure host-side; no CUDA/HIP headers.
//
//===----------------------------------------------------------------------===//

#include "kernscope/bundle.h"

#include "kernscope/db.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <system_error>

namespace kernscope {

namespace {

namespace json = kernscope::db::json;
namespace fs = std::filesystem;

//===--- small helpers ---------------------------------------------------------===//

std::string Hex64(std::uint64_t v) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out = "0x";
  for (int shift = 60; shift >= 0; shift -= 4) {
    out.push_back(kHex[(v >> shift) & 0xF]);
  }
  // Trim leading zero nibbles, keeping at least one digit ("0x0" not "0x").
  while (out.size() > 3 && out[2] == '0') {
    out.erase(out.begin() + 2);
  }
  return out;
}

std::string HexBytes(std::span<const std::uint8_t> bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (std::uint8_t b : bytes) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

std::vector<std::uint8_t> Unhex(std::string_view hex, std::string &error) {
  std::vector<std::uint8_t> out;
  out.reserve(hex.size() / 2);
  auto nibble = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  if (hex.size() % 2 != 0) {
    error = "odd-length hex string";
    return {};
  }
  for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
    const int hi = nibble(hex[i]);
    const int lo = nibble(hex[i + 1]);
    if (hi < 0 || lo < 0) {
      error = "invalid hex byte at offset " + std::to_string(i);
      return {};
    }
    out.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
  }
  return out;
}

/// Parses "0x<hex>" (v1 JSON convention) OR "<decimal>" (v0 text numbers).
bool ParseU64(std::string_view tok, std::uint64_t &out, std::string &error) {
  std::string_view digits = tok;
  int base = 10;
  if (digits.size() > 2 && digits[0] == '0' &&
      (digits[1] == 'x' || digits[1] == 'X')) {
    digits.remove_prefix(2);
    base = 16;
  }
  std::uint64_t v = 0;
  const auto res =
      std::from_chars(digits.data(), digits.data() + digits.size(), v, base);
  if (res.ec != std::errc() || res.ptr != digits.data() + digits.size()) {
    error = "invalid u64 '" + std::string(tok) + "'";
    return false;
  }
  out = v;
  return true;
}

bool ParseI64(std::string_view tok, std::int64_t &out, std::string &error) {
  std::string_view digits = tok;
  int base = 10;
  if (digits.size() > 2 && digits[0] == '0' &&
      (digits[1] == 'x' || digits[1] == 'X')) {
    digits.remove_prefix(2);
    base = 16;
  }
  std::uint64_t v = 0;
  const auto res =
      std::from_chars(digits.data(), digits.data() + digits.size(), v, base);
  if (res.ec != std::errc() || res.ptr != digits.data() + digits.size()) {
    error = "invalid integer '" + std::string(tok) + "'";
    return false;
  }
  out = static_cast<std::int64_t>(v);
  return true;
}

std::string NowIso8601Utc() {
  const auto now = std::chrono::system_clock::now();
  const std::time_t t = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  gmtime_r(&t, &tm);
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02dZ",
                tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour,
                tm.tm_min, tm.tm_sec);
  return std::string(buf);
}

/// Table-kind lookup by stable name (unknown -> kCount; the raw name is kept
/// by the caller for kCount entries).
ApiKind KindForApiName(const std::string &name) {
  for (int k = 0; k < static_cast<int>(ApiKind::kCount); ++k) {
    if (name == api_name(static_cast<ApiKind>(k))) {
      return static_cast<ApiKind>(k);
    }
  }
  return ApiKind::kCount;
}

/// ApiTableEntry::name is `const char*`; names read from JSON/v0 text are
/// owned here (std::deque: insertion never invalidates c_str() pointers).
const char *OwnedApiName(std::deque<std::string> &pool, const std::string &name) {
  pool.push_back(name);
  return pool.back().c_str();
}

//===--- required-field helpers (db conventions: "missing required field X") ---===//

bool GetString(const json::Value &obj, std::string_view key, std::string &out,
               std::string &error) {
  const json::Value *v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::String) {
    error = "field \"" + std::string(key) + "\" is not a string";
    return false;
  }
  out = v->as_string();
  return true;
}

bool GetBool(const json::Value &obj, std::string_view key, bool &out,
             std::string &error) {
  const json::Value *v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::Bool) {
    error = "field \"" + std::string(key) + "\" is not a bool";
    return false;
  }
  out = v->as_bool();
  return true;
}

bool GetInt(const json::Value &obj, std::string_view key, std::int64_t &out,
            std::string &error) {
  const json::Value *v = obj.find(key);
  if (v == nullptr) {
    error = "missing required field \"" + std::string(key) + "\"";
    return false;
  }
  if (v->kind() != json::Kind::Number) {
    error = "field \"" + std::string(key) + "\" is not a number";
    return false;
  }
  out = static_cast<std::int64_t>(v->as_number());
  return true;
}

bool GetU64(const json::Value &obj, std::string_view key, std::uint64_t &out,
            std::string &error) {
  std::string s;
  if (!GetString(obj, key, s, error)) {
    return false;
  }
  return ParseU64(s, out, error);
}

//===--- events -> JSON ---------------------------------------------------------===//

std::vector<std::uint64_t> KnownAllocVas(const CaptureSession &session) {
  std::vector<std::uint64_t> vas;
  vas.reserve(session.allocs().size());
  for (const AllocRecord &a : session.allocs()) {
    vas.push_back(a.ptr);
  }
  return vas;
}

json::Value ArgSlotToJson(const KernelArgSlot &slot, bool is_pointer,
                          std::uint64_t va) {
  std::vector<json::Value::Member> m = {
      {"kind", json::Value(is_pointer ? "pointer" : "scalar")},
      {"size", json::Value(static_cast<int>(slot.size))},
      {"bytes", json::Value(HexBytes(slot.bytes))},
  };
  if (is_pointer) {
    m.emplace_back("va", json::Value(Hex64(va)));
  }
  return json::Value(std::move(m));
}

json::Value LaunchToJson(const LaunchRecord &l,
                         const std::vector<ArgSlotInfo> &arg_info) {
  std::vector<json::Value> args;
  args.reserve(l.args.size());
  for (std::size_t i = 0; i < l.args.size(); ++i) {
    const bool is_ptr = i < arg_info.size() && arg_info[i].is_pointer;
    const std::uint64_t va = i < arg_info.size() ? arg_info[i].va : 0;
    args.push_back(ArgSlotToJson(l.args[i], is_ptr, va));
  }
  return json::object({
      {"type", json::Value("launch")},
      {"kernel_name", json::Value(l.kernel_name)},
      {"via", json::Value(to_string(l.via))},
      {"grid", json::array({json::Value(static_cast<int>(l.grid_x)),
                            json::Value(static_cast<int>(l.grid_y)),
                            json::Value(static_cast<int>(l.grid_z))})},
      {"block", json::array({json::Value(static_cast<int>(l.block_x)),
                             json::Value(static_cast<int>(l.block_y)),
                             json::Value(static_cast<int>(l.block_z))})},
      {"shared_mem_bytes", json::Value(static_cast<int>(l.shared_mem_bytes))},
      {"stream", json::Value(Hex64(l.stream))},
      {"graph_handle", json::Value(Hex64(l.graph_handle))},
      {"seq", json::Value(static_cast<int>(l.seq))},
      {"args", json::Value(std::move(args))},
  });
}

/// Computes the arg-marshalling enrichment for every launch: an 8-byte slot
/// whose little-endian value equals a known allocation VA is a pointer slot
/// (capture-mapping §4.3 applied at write time; format doc §3.1).
std::vector<std::vector<ArgSlotInfo>> ClassifyArgSlots(
    const CaptureSession &session) {
  std::vector<std::vector<ArgSlotInfo>> tables;
  std::unordered_map<std::uint64_t, bool> known;
  for (std::uint64_t va : KnownAllocVas(session)) {
    known[va] = true;
  }
  tables.reserve(session.launches().size());
  for (const LaunchRecord &l : session.launches()) {
    std::vector<ArgSlotInfo> table;
    table.reserve(l.args.size());
    for (const KernelArgSlot &slot : l.args) {
      ArgSlotInfo info;
      if (slot.size == sizeof(std::uint64_t) &&
          slot.bytes.size() >= sizeof(std::uint64_t)) {
        std::uint64_t value = 0;
        std::memcpy(&value, slot.bytes.data(), sizeof(value));
        if (known.contains(value)) {
          info.is_pointer = true;
          info.va = value;
        }
      }
      table.push_back(info);
    }
    tables.push_back(std::move(table));
  }
  return tables;
}

json::Value SessionToEvents(const CaptureSession &session,
                            const std::vector<std::vector<ArgSlotInfo>> &tables) {
  std::vector<json::Value> events;
  events.reserve(session.api_table().size() + session.allocs().size() +
                 session.launches().size() + session.copies().size() +
                 session.memsets().size() + session.streams().size() +
                 session.modules().size());
  for (const ApiTableEntry &e : session.api_table()) {
    events.push_back(json::object({
        {"type", json::Value("api")},
        {"name", json::Value(e.name)},
        {"calls", json::Value(static_cast<int>(e.calls))},
    }));
  }
  for (const AllocRecord &a : session.allocs()) {
    events.push_back(json::object({
        {"type", json::Value(a.is_alloc ? "alloc" : "free")},
        {"ptr", json::Value(Hex64(a.ptr))},
        {"size", json::Value(Hex64(a.size))},
        {"is_alloc", json::Value(a.is_alloc)},
        {"seq", json::Value(static_cast<int>(a.seq))},
    }));
  }
  for (const UnknownFreeRecord &u : session.unknown_frees()) {
    events.push_back(json::object({
        {"type", json::Value("unknown-free")},
        {"ptr", json::Value(Hex64(u.ptr))},
        {"forwarded", json::Value(u.forwarded)},
        {"seq", json::Value(static_cast<int>(u.seq))},
    }));
  }
  std::size_t launch_index = 0;
  for (const LaunchRecord &l : session.launches()) {
    static const std::vector<ArgSlotInfo> kEmptyArgTable;
    const std::vector<ArgSlotInfo> &table =
        launch_index < tables.size() ? tables[launch_index] : kEmptyArgTable;
    events.push_back(LaunchToJson(l, table));
    ++launch_index;
  }
  for (const CopyRecord &c : session.copies()) {
    events.push_back(json::object({
        {"type", json::Value("copy")},
        {"kind", json::Value(std::string(c.kind))},
        {"dst", json::Value(Hex64(c.dst))},
        {"src", json::Value(Hex64(c.src))},
        {"bytes", json::Value(Hex64(c.bytes))},
        {"cuda_kind", json::Value(static_cast<int>(c.cuda_kind))},
        {"dst_size", json::Value(Hex64(c.dst_size))},
        {"src_size", json::Value(Hex64(c.src_size))},
        {"stream", json::Value(Hex64(c.stream))},
        {"seq", json::Value(static_cast<int>(c.seq))},
    }));
  }
  for (const MemsetRecord &m : session.memsets()) {
    events.push_back(json::object({
        {"type", json::Value("memset")},
        {"ptr", json::Value(Hex64(m.ptr))},
        {"value", json::Value(static_cast<int>(m.value))},
        {"bytes", json::Value(Hex64(m.bytes))},
        {"ptr_size", json::Value(Hex64(m.ptr_size))},
        {"stream", json::Value(Hex64(m.stream))},
        {"seq", json::Value(static_cast<int>(m.seq))},
    }));
  }
  for (const StreamRecord &s : session.streams()) {
    events.push_back(json::object({
        {"type", json::Value("stream")},
        {"sync", json::Value(s.is_sync)},
        {"handle", json::Value(Hex64(s.handle))},
        {"id", json::Value(static_cast<int>(s.id))},
        {"seq", json::Value(static_cast<int>(s.seq))},
    }));
  }
  for (const ModuleRecord &m : session.modules()) {
    std::vector<json::Value> kernels;
    kernels.reserve(m.kernels.size());
    for (const std::string &k : m.kernels) {
      kernels.push_back(json::Value(k));
    }
    std::vector<json::Value::Member> members = {
        {"type", json::Value(m.kind == ModuleRecord::Kind::kFatBinary
                                 ? "fatbin"
                                 : "cubin")},
        {"mod", json::Value(static_cast<int>(m.handle))},
        {"sha256", json::Value(m.sha256)},
        {"kernels", json::Value(std::move(kernels))},
        {"seq", json::Value(static_cast<int>(m.seq))},
    };
    if (!m.name.empty()) {
      members.emplace_back("path", json::Value(m.name));
    }
    events.push_back(json::Value(std::move(members)));
  }
  for (const auto &[cufunction, infos] : session.param_infos()) {
    for (std::size_t i = 0; i < infos.size(); ++i) {
      events.push_back(json::object({
          {"type", json::Value("param_info")},
          {"cufunction", json::Value(Hex64(cufunction))},
          {"param_index", json::Value(static_cast<int>(i))},
          {"offset", json::Value(static_cast<int>(infos[i].offset))},
          {"size", json::Value(static_cast<int>(infos[i].size))},
      }));
    }
  }
  return json::Value(std::move(events));
}

bool WriteFile(const fs::path &path, const std::string &content,
               std::string &error) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) {
    error = "cannot open " + path.string();
    return false;
  }
  out.write(content.data(), static_cast<std::streamsize>(content.size()));
  if (!out) {
    error = "write failed: " + path.string();
    return false;
  }
  return true;
}

bool ReadWholeFile(const fs::path &path, std::string &out, std::string &error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    error = "cannot open " + path.string();
    return false;
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  if (in.bad()) {
    error = "read failed: " + path.string();
    return false;
  }
  out = ss.str();
  return true;
}

} // namespace

//===--- public patch-table API --------------------------------------------------===//

const char *to_string(BufferRole role) {
  switch (role) {
  case BufferRole::kKernelBinary: return "kernel-binary";
  case BufferRole::kInput: return "input";
  case BufferRole::kOutput: return "output";
  case BufferRole::kArg: return "arg";
  case BufferRole::kState: return "state";
  }
  return "?";
}

std::vector<PatchEntry> scan_embedded_pointers(
    std::span<const std::uint8_t> blob, std::span<const std::uint64_t> vas) {
  std::vector<PatchEntry> patches;
  if (blob.size() < sizeof(std::uint64_t)) {
    return patches;
  }
  std::unordered_map<std::uint64_t, bool> known;
  known.reserve(vas.size());
  for (std::uint64_t va : vas) {
    known[va] = true;
  }
  // Byte-granular scan: a T** indirection table lives at an arbitrary offset
  // inside the blob (docs/capture-format.md §4.3).
  for (std::size_t offset = 0; offset + sizeof(std::uint64_t) <= blob.size();
       ++offset) {
    std::uint64_t value = 0;
    std::memcpy(&value, blob.data() + offset, sizeof(value));
    if (known.contains(value)) {
      patches.push_back(PatchEntry{static_cast<std::uint64_t>(offset), value});
    }
  }
  return patches;
}

void apply_remap(std::span<std::uint8_t> blob,
                 const std::vector<PatchEntry> &patches,
                 const std::unordered_map<std::uint64_t, std::uint64_t> &remap) {
  for (const PatchEntry &p : patches) {
    const auto it = remap.find(p.target_va);
    if (it == remap.end()) {
      continue; // candidate not remapped -> untouched
    }
    if (p.offset + sizeof(std::uint64_t) <= blob.size()) {
      const std::uint64_t new_va = it->second;
      std::memcpy(blob.data() + p.offset, &new_va, sizeof(new_va));
    }
  }
}

//===--- v1 writer ----------------------------------------------------------------===//

WriteStatus write_bundle_v1(const CaptureSession &session, const std::string &dir,
                            const BundleWriteOptions &opts, std::string *error) {
  std::string err;
  auto fail = [&](WriteStatus st, const std::string &msg) {
    if (error != nullptr) {
      *error = msg;
    }
    return st;
  };
  if (opts.read_fn == nullptr) {
    return fail(WriteStatus::kInvalid,
                "write_bundle_v1: read_fn is required (snapshot provider)");
  }
  if (opts.capture_mechanism != "cupti" && opts.capture_mechanism != "preload" &&
      opts.capture_mechanism != "hip-sdk") {
    return fail(WriteStatus::kInvalid,
                "write_bundle_v1: capture_mechanism must be cupti|preload|hip-sdk");
  }
  fs::path root(dir);
  if (root.extension() != ".kcap") {
    root += ".kcap";
  }
  std::error_code ec;
  fs::remove_all(root, ec);
  ec.clear();
  fs::create_directories(root / "blobs", ec);
  if (ec) {
    return fail(WriteStatus::kIoError,
                "cannot create bundle dir " + root.string());
  }

  const std::string timestamp =
      opts.timestamp.empty() ? NowIso8601Utc() : opts.timestamp;

  // events.json (formalized event log + arg marshalling tables).
  const auto arg_tables = ClassifyArgSlots(session);
  const std::string events_json = SessionToEvents(session, arg_tables).dump();
  if (!WriteFile(root / "events.json", events_json, err)) {
    return fail(WriteStatus::kIoError, err);
  }

  // Blobs: budget-check BEFORE each write (size governance, format doc §5.1).
  std::vector<json::Value> blob_entries;
  std::uint64_t total = 0;
  bool aborted = false;
  std::string abort_reason;
  const std::vector<std::uint64_t> known_vas = KnownAllocVas(session);
  std::size_t index = 0;
  for (const BufferSnapshot &req : opts.snapshots) {
    const std::uint64_t next = total + req.size;
    if (next > opts.max_snapshot_bytes) {
      aborted = true;
      abort_reason = "snapshot exceeds --max-snapshot-bytes=" +
                     std::to_string(opts.max_snapshot_bytes) + " (total " +
                     std::to_string(next) + " bytes); capture aborted";
      break;
    }
    BufferSnapshot blob = req;
    if (blob.id.empty()) {
      std::ostringstream id;
      id << "b" << std::setfill('0') << std::setw(4) << (index + 1);
      blob.id = id.str();
    }
    if (blob.file.empty()) {
      blob.file = "blobs/" + blob.id + ".bin";
    }
    std::vector<std::uint8_t> bytes;
    if (!opts.read_fn(blob, bytes)) {
      return fail(WriteStatus::kIoError,
                  "read_fn failed for snapshot " + blob.id);
    }
    if (bytes.size() != req.size) {
      return fail(WriteStatus::kIoError,
                  "snapshot " + blob.id + ": read_fn returned " +
                      std::to_string(bytes.size()) + " bytes, expected " +
                      std::to_string(req.size));
    }
    if (opts.scan_embedded_pointers) {
      blob.patches = scan_embedded_pointers(bytes, known_vas);
    }
    blob.sha256 = sha256_hex(bytes);
    blob.size = bytes.size();
    const std::string raw(bytes.begin(), bytes.end());
    if (!WriteFile(root / blob.file, raw, err)) {
      return fail(WriteStatus::kIoError, err);
    }
    std::vector<json::Value> patches;
    patches.reserve(blob.patches.size());
    for (const PatchEntry &p : blob.patches) {
      patches.push_back(json::object({
          {"offset", json::Value(static_cast<int>(p.offset))},
          {"target_va", json::Value(Hex64(p.target_va))},
      }));
    }
    blob_entries.push_back(json::object({
        {"id", json::Value(blob.id)},
        {"file", json::Value(blob.file)},
        {"role", json::Value(to_string(blob.role))},
        {"va", json::Value(Hex64(blob.va))},
        {"size", json::Value(Hex64(blob.size))},
        {"sha256", json::Value(blob.sha256)},
        {"patches", json::Value(std::move(patches))},
    }));
    total += blob.size;
    ++index;
  }

  // blobs.json
  const std::string blobs_json = json::object({
      {"schema_version", json::Value("1.0")},
      {"blobs", json::Value(std::move(blob_entries))},
  }).dump();
  if (!WriteFile(root / "blobs.json", blobs_json, err)) {
    return fail(WriteStatus::kIoError, err);
  }

  // manifest.json (written LAST: it is the commit marker of a complete write
  // and carries the size-governance verdict).
  std::string first_kernel;
  if (!session.launches().empty()) {
    first_kernel = session.launches().front().kernel_name;
  }
  const std::string manifest_json = json::object({
      {"format_id", json::Value(std::string(kBundleFormatId))},
      {"schema_version", json::Value("1.0")},
      {"tool_version", json::Value(opts.tool_version)},
      {"vendor", json::Value(opts.vendor)},
      {"arch", json::Value(opts.arch)},
      {"capture_mechanism", json::Value(opts.capture_mechanism)},
      {"timestamp", json::Value(timestamp)},
      {"snapshot_policy", json::Value("va-faithful")},
      {"first_kernel", json::Value(first_kernel)},
      {"size_governance",
       json::object({
           {"max_snapshot_bytes",
            json::Value(static_cast<int>(opts.max_snapshot_bytes))},
           {"total_blob_bytes", json::Value(static_cast<int>(total))},
           {"aborted", json::Value(aborted)},
           {"abort_reason", json::Value(abort_reason)},
       })},
  }).dump();
  if (!WriteFile(root / "manifest.json", manifest_json, err)) {
    return fail(WriteStatus::kIoError, err);
  }

  if (aborted) {
    return fail(WriteStatus::kAbortedSizeCap, abort_reason);
  }
  return WriteStatus::kOk;
}

//===--- v0 text parser (todos 43/44/45 output shape) ----------------------------===//
// Accepts exactly the line grammar the v0 BundleWriter emits (capture.cpp):
//   kernscope-capture-v0
//   api <name> calls=<n>                      (name is a bare token)
//   alloc|free ptr=0x.. size=<n>
//   unknown-free ptr=0x.. forwarded=true|false
//   launch kernel=".." via=.. grid=AxBxC block=AxBxC shared=<n>
//          stream=0x.. graph=0x.. args=<n> [<size>:<hex>]...   (args bare)
//   copy kind=.. dst=0x.. src=0x.. bytes=<n> cuda-kind=<n>
//        dst-size=<n> src-size=<n> stream=0x..
//   memset ptr=0x.. value=<n> bytes=<n> ptr-size=<n> stream=0x..
//   stream create|sync handle=0x.. id=<n>     (create|sync bare)
//   fatbin|cubin mod=<n> sha256=.. [path=".."] [kernels="a","b"]
//   end-capture-v0
// Truncated input is reported with line+byte offsets (QA: schema violation
// WITH OFFSET, no UB, no crash). v0 carries no `seq`; the reader assigns
// parse order (format doc §7).
//
// Truncation hardening (F3 review): a v0 bundle is a NEWLINE-TERMINATED
// record stream, so an unterminated final line is always a truncation -
// in particular a 1-byte trailing cut of the 'end-capture-v0' terminator
// (its trailing '\n' removed) must be reported as a schema violation with
// the byte offset, never tolerated as a clean end-of-file. The only
// tolerated leniency is content AFTER a complete, newline-terminated
// terminator.

namespace {

struct Line {
  std::string keyword; // owns its storage (tokens are rebuilt per line)
  std::vector<std::string> toks; // bare tokens (api name, stream create|sync,
                                 // [size:hex] arg slots)
  std::vector<std::pair<std::string, std::string>> kv; // key=value pairs
};

std::vector<std::string> SplitWhitespace(std::string_view line) {
  std::vector<std::string> toks;
  std::size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() &&
           (line[i] == ' ' || line[i] == '\t' || line[i] == '\r')) {
      ++i;
    }
    if (i >= line.size()) {
      break;
    }
    const std::size_t start = i;
    while (i < line.size() && line[i] != ' ' && line[i] != '\t') {
      ++i;
    }
    toks.emplace_back(line.substr(start, i - start));
  }
  return toks;
}

bool ParseLine(std::string_view line, Line &out) {
  const std::vector<std::string> toks = SplitWhitespace(line);
  if (toks.empty()) {
    return false;
  }
  out.keyword = toks[0];
  for (std::size_t i = 1; i < toks.size(); ++i) {
    const std::string &tok = toks[i];
    const std::size_t eq = tok.find('=');
    if (eq == std::string::npos) {
      out.toks.push_back(tok); // bare token
      continue;
    }
    std::string value = tok.substr(eq + 1);
    // Strip quotes only for simple quoted values ("vec_add"); leave
    // comma-separated quoted lists ("a","b") intact for the kernels parser.
    if (value.size() >= 2 && value.front() == '"' && value.back() == '"' &&
        value.find('"', 1) == value.size() - 1) {
      value = value.substr(1, value.size() - 2);
    }
    out.kv.emplace_back(tok.substr(0, eq), std::move(value));
  }
  return true;
}

bool KvGet(const Line &line, std::string_view key, std::string &out,
           std::string &error) {
  for (const auto &[k, v] : line.kv) {
    if (k == key) {
      out = v;
      return true;
    }
  }
  error = "line '" + std::string(line.keyword) + "': missing key '" +
          std::string(key) + "'";
  return false;
}

bool KvGetU64(const Line &line, std::string_view key, std::uint64_t &out,
              std::string &error) {
  std::string s;
  if (!KvGet(line, key, s, error)) {
    return false;
  }
  return ParseU64(s, out, error);
}

bool KvGetI64(const Line &line, std::string_view key, std::int64_t &out,
              std::string &error) {
  std::string s;
  if (!KvGet(line, key, s, error)) {
    return false;
  }
  return ParseI64(s, out, error);
}

std::string V0Error(std::size_t line_no, std::size_t byte,
                    const std::string &msg) {
  return "schema violation: v0 parse error at line " +
         std::to_string(line_no) + " byte " + std::to_string(byte) + ": " +
         msg;
}

struct V0Model {
  std::vector<ApiTableEntry> api;
  std::vector<AllocRecord> allocs;
  std::vector<UnknownFreeRecord> unknown_frees;
  std::vector<LaunchRecord> launches;
  std::vector<std::vector<ArgSlotInfo>> arg_tables;
  std::vector<CopyRecord> copies;
  std::vector<MemsetRecord> memsets;
  std::vector<StreamRecord> streams;
  std::vector<ModuleRecord> modules;
  std::deque<std::string> name_pool;
  std::deque<std::string> string_pool; // CopyRecord::kind (string_view) storage
};

bool ParseDims(const std::string &s, std::uint32_t *out, std::string &error) {
  std::string_view sv = s;
  for (int i = 0; i < 3; ++i) {
    const std::size_t x = sv.find('x');
    const std::string_view part =
        x == std::string_view::npos ? sv : sv.substr(0, x);
    std::uint64_t v = 0;
    if (!ParseU64(part, v, error) || v > 0xFFFFFFFFULL) {
      return false;
    }
    out[i] = static_cast<std::uint32_t>(v);
    if (x == std::string_view::npos) {
      break;
    }
    sv.remove_prefix(x + 1);
  }
  return true;
}

bool ParseV0Text(std::string_view text, V0Model &model, std::string &error) {
  std::uint64_t seq = 0;
  std::size_t line_start = 0;
  bool saw_header = false;
  bool saw_end = false;
  std::size_t line_no = 0;
  std::size_t last_content_line = 0;
  while (line_start <= text.size()) {
    const std::size_t nl = text.find('\n', line_start);
    const std::size_t end = nl == std::string_view::npos ? text.size() : nl;
    const std::string_view line = text.substr(line_start, end - line_start);
    ++line_no;
    std::string_view l = line;
    if (!l.empty() && l.back() == '\r') {
      l.remove_suffix(1);
    }
    if (!l.empty()) {
      last_content_line = line_no;
      if (saw_end) {
        break; // tolerate trailing content after end-capture-v0
      }
      Line parsed;
      if (!ParseLine(l, parsed)) {
        error =
            V0Error(line_no, line_start, "malformed line '" + std::string(l) + "'");
        return false;
      }
      if (!saw_header && parsed.keyword != "kernscope-capture-v0") {
        error = V0Error(line_no, line_start,
                        "expected 'kernscope-capture-v0' header");
        return false;
      }
      if (parsed.keyword == "kernscope-capture-v0") {
        saw_header = true;
      } else if (parsed.keyword == "end-capture-v0") {
        saw_end = true;
      } else if (parsed.keyword == "api") {
        if (parsed.toks.empty() ||
            parsed.toks[0].find('=') != std::string::npos) {
          error = V0Error(line_no, line_start, "api record missing name");
          return false;
        }
        std::int64_t calls = 0;
        if (!KvGetI64(parsed, "calls", calls, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        ApiTableEntry e;
        e.kind = KindForApiName(parsed.toks[0]);
        e.name = e.kind == ApiKind::kCount
                     ? OwnedApiName(model.name_pool, parsed.toks[0])
                     : api_name(e.kind);
        e.calls = static_cast<std::uint64_t>(calls);
        model.api.push_back(e);
      } else if (parsed.keyword == "alloc" || parsed.keyword == "free") {
        std::uint64_t ptr = 0;
        std::uint64_t size = 0;
        if (!KvGetU64(parsed, "ptr", ptr, error) ||
            !KvGetU64(parsed, "size", size, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        model.allocs.push_back(AllocRecord{
            ptr, static_cast<std::size_t>(size), parsed.keyword == "alloc",
            ++seq});
      } else if (parsed.keyword == "unknown-free") {
        std::uint64_t ptr = 0;
        std::string forwarded;
        if (!KvGetU64(parsed, "ptr", ptr, error) ||
            !KvGet(parsed, "forwarded", forwarded, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        model.unknown_frees.push_back(
            UnknownFreeRecord{ptr, forwarded == "true", ++seq});
      } else if (parsed.keyword == "launch") {
        std::string kernel_name;
        std::string via;
        std::string grid_s;
        std::string block_s;
        std::int64_t shared = 0;
        std::uint64_t stream = 0;
        std::uint64_t graph = 0;
        std::uint64_t args = 0;
        if (!KvGet(parsed, "kernel", kernel_name, error) ||
            !KvGet(parsed, "via", via, error) ||
            !KvGet(parsed, "grid", grid_s, error) ||
            !KvGet(parsed, "block", block_s, error) ||
            !KvGetI64(parsed, "shared", shared, error) ||
            !KvGetU64(parsed, "stream", stream, error) ||
            !KvGetU64(parsed, "graph", graph, error) ||
            !KvGetU64(parsed, "args", args, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        LaunchRecord rec;
        rec.kernel_name = kernel_name;
        if (via == "runtime") {
          rec.via = LaunchVia::kRuntimeApi;
        } else if (via == "runtime-ex") {
          rec.via = LaunchVia::kRuntimeApiEx;
        } else if (via == "graph") {
          rec.via = LaunchVia::kGraph;
        } else if (via == "driver-ex") {
          rec.via = LaunchVia::kDriverApiEx;
        } else {
          rec.via = LaunchVia::kDriverApi;
        }
        std::uint32_t dims[3] = {1, 1, 1};
        if (!ParseDims(grid_s, dims, error)) {
          error = V0Error(line_no, line_start, "invalid grid '" + grid_s + "'");
          return false;
        }
        rec.grid_x = dims[0];
        rec.grid_y = dims[1];
        rec.grid_z = dims[2];
        if (!ParseDims(block_s, dims, error)) {
          error =
              V0Error(line_no, line_start, "invalid block '" + block_s + "'");
          return false;
        }
        rec.block_x = dims[0];
        rec.block_y = dims[1];
        rec.block_z = dims[2];
        rec.shared_mem_bytes = static_cast<std::uint32_t>(shared);
        rec.stream = stream;
        rec.graph_handle = graph;
        rec.seq = ++seq;
        std::vector<ArgSlotInfo> table;
        for (const std::string &tok : parsed.toks) {
          if (tok.empty() || tok.front() != '[') {
            continue; // only [<size>:<hex>] arg slots are bare tokens here
          }
          if (tok.back() != ']') {
            error = V0Error(line_no, line_start, "bad arg token '" + tok + "'");
            return false;
          }
          const std::string_view inner(tok.data() + 1, tok.size() - 2);
          const std::size_t colon = inner.find(':');
          if (colon == std::string_view::npos) {
            error = V0Error(line_no, line_start, "bad arg token '" + tok + "'");
            return false;
          }
          std::uint64_t slot_size = 0;
          if (!ParseU64(inner.substr(0, colon), slot_size, error)) {
            error = V0Error(line_no, line_start, error);
            return false;
          }
          const std::vector<std::uint8_t> bytes =
              Unhex(inner.substr(colon + 1), error);
          if (!error.empty()) {
            error = V0Error(line_no, line_start, error);
            return false;
          }
          if (bytes.size() != slot_size) {
            error = V0Error(line_no, line_start,
                            "arg size mismatch (" +
                                std::to_string(bytes.size()) + " != " +
                                std::to_string(slot_size) + ")");
            return false;
          }
          KernelArgSlot slot;
          slot.size = static_cast<std::size_t>(slot_size);
          slot.bytes = bytes;
          // Pointer-slot classification for v0: same heuristic as v1
          // (8-byte slot whose LE value matches a known allocation VA).
          ArgSlotInfo info;
          if (slot.size == sizeof(std::uint64_t) &&
              slot.bytes.size() >= sizeof(std::uint64_t)) {
            std::uint64_t value = 0;
            std::memcpy(&value, slot.bytes.data(), sizeof(value));
            bool known = false;
            for (const AllocRecord &a : model.allocs) {
              if (a.ptr == value) {
                known = true;
                break;
              }
            }
            if (known) {
              info.is_pointer = true;
              info.va = value;
            }
          }
          rec.args.push_back(std::move(slot));
          table.push_back(info);
        }
        if (rec.args.size() != args) {
          error = V0Error(line_no, line_start,
                          "args count mismatch (" +
                              std::to_string(rec.args.size()) + " != " +
                              std::to_string(args) + ")");
          return false;
        }
        model.launches.push_back(std::move(rec));
        model.arg_tables.push_back(std::move(table));
      } else if (parsed.keyword == "copy") {
        std::string kind;
        std::uint64_t dst = 0;
        std::uint64_t src = 0;
        std::uint64_t bytes = 0;
        std::uint64_t stream = 0;
        std::int64_t cuda_kind = 0;
        std::int64_t dst_size = 0;
        std::int64_t src_size = 0;
        if (!KvGet(parsed, "kind", kind, error) ||
            !KvGetU64(parsed, "dst", dst, error) ||
            !KvGetU64(parsed, "src", src, error) ||
            !KvGetU64(parsed, "bytes", bytes, error) ||
            !KvGetI64(parsed, "cuda-kind", cuda_kind, error) ||
            !KvGetI64(parsed, "dst-size", dst_size, error) ||
            !KvGetI64(parsed, "src-size", src_size, error) ||
            !KvGetU64(parsed, "stream", stream, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        CopyRecord rec;
        // CopyRecord::kind is a string_view: keep the text alive in the
        // model's pool (deque: insertion never invalidates).
        rec.kind = model.string_pool.emplace_back(kind);
        rec.dst = dst;
        rec.src = src;
        rec.bytes = static_cast<std::size_t>(bytes);
        rec.cuda_kind = static_cast<int>(cuda_kind);
        rec.dst_size = static_cast<std::size_t>(dst_size);
        rec.src_size = static_cast<std::size_t>(src_size);
        rec.stream = stream;
        rec.seq = ++seq;
        model.copies.push_back(rec);
      } else if (parsed.keyword == "memset") {
        std::uint64_t ptr = 0;
        std::uint64_t bytes = 0;
        std::uint64_t stream = 0;
        std::int64_t value = 0;
        std::int64_t ptr_size = 0;
        if (!KvGetU64(parsed, "ptr", ptr, error) ||
            !KvGetI64(parsed, "value", value, error) ||
            !KvGetU64(parsed, "bytes", bytes, error) ||
            !KvGetI64(parsed, "ptr-size", ptr_size, error) ||
            !KvGetU64(parsed, "stream", stream, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        MemsetRecord rec;
        rec.ptr = ptr;
        rec.value = static_cast<int>(value);
        rec.bytes = static_cast<std::size_t>(bytes);
        rec.ptr_size = static_cast<std::size_t>(ptr_size);
        rec.stream = stream;
        rec.seq = ++seq;
        model.memsets.push_back(rec);
      } else if (parsed.keyword == "stream") {
        // v0 emits: stream create|sync handle=0x.. id=<n>  (create|sync bare)
        if (parsed.toks.empty() ||
            (parsed.toks[0] != "create" && parsed.toks[0] != "sync")) {
          error = V0Error(line_no, line_start,
                          "stream record missing create|sync kind");
          return false;
        }
        std::uint64_t handle = 0;
        std::int64_t id = 0;
        if (!KvGetU64(parsed, "handle", handle, error) ||
            !KvGetI64(parsed, "id", id, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        StreamRecord rec;
        rec.handle = handle;
        rec.id = static_cast<std::uint64_t>(id);
        rec.is_sync = parsed.toks[0] == "sync";
        rec.seq = ++seq;
        model.streams.push_back(rec);
      } else if (parsed.keyword == "fatbin" || parsed.keyword == "cubin") {
        std::uint64_t mod = 0;
        std::string sha256;
        if (!KvGetU64(parsed, "mod", mod, error) ||
            !KvGet(parsed, "sha256", sha256, error)) {
          error = V0Error(line_no, line_start, error);
          return false;
        }
        ModuleRecord rec;
        rec.kind = parsed.keyword == "fatbin" ? ModuleRecord::Kind::kFatBinary
                                              : ModuleRecord::Kind::kCubin;
        rec.handle = mod;
        rec.sha256 = sha256;
        for (const auto &[k, v] : parsed.kv) {
          if (k == "path") {
            rec.name = v;
          } else if (k == "kernels") {
            // comma-separated quoted list: "a","b"
            std::string_view sv = v;
            while (!sv.empty()) {
              while (!sv.empty() && sv.front() == ',') {
                sv.remove_prefix(1);
              }
              if (sv.empty()) {
                break;
              }
              const std::size_t comma = sv.find(',');
              const std::string_view name =
                  comma == std::string_view::npos ? sv : sv.substr(0, comma);
              std::string n(name);
              if (n.size() >= 2 && n.front() == '"' && n.back() == '"') {
                n = n.substr(1, n.size() - 2);
              }
              rec.kernels.push_back(n);
              if (comma == std::string_view::npos) {
                break;
              }
              sv.remove_prefix(comma + 1);
            }
          }
        }
        rec.seq = ++seq;
        model.modules.push_back(rec);
      } else {
        error = V0Error(line_no, line_start,
                        "unknown record '" + std::string(parsed.keyword) + "'");
        return false;
      }
    }
    if (nl == std::string_view::npos) {
      if (!l.empty()) {
        // Unterminated final line: the record stream was cut mid-write
        // (every v0 record, including the terminator, is newline-terminated).
        // A 1-byte trailing truncation of 'end-capture-v0' lands here and
        // must be a schema violation with the byte offset, not a clean EOF.
        if (saw_end) {
          error = V0Error(line_no, line_start,
                          "truncated 'end-capture-v0' terminator (missing "
                          "trailing newline; file ends at byte " +
                              std::to_string(text.size()) + ")");
          return false;
        }
      }
      break;
    }
    line_start = nl + 1;
  }
  if (!saw_header) {
    error = "schema violation: v0 parse error: missing 'kernscope-capture-v0' "
            "header";
    return false;
  }
  if (!saw_end) {
    error = V0Error(last_content_line, text.size(),
                    "missing 'end-capture-v0' terminator (truncated bundle)");
    return false;
  }
  return true;
}

} // namespace

//===--- BundleReader ------------------------------------------------------------===//

struct BundleReader::Impl {
  bool v1 = false;
  std::string root; // bundle directory (v1) or parent dir (v0 sidecars)

  // manifest
  std::string tool_version, vendor, arch, capture_mechanism, timestamp;
  std::string first_kernel;
  std::uint64_t max_snapshot_bytes = 0;
  std::uint64_t total_blob_bytes = 0;
  bool aborted = false;
  std::string abort_reason;

  // events
  std::vector<ApiTableEntry> api;
  std::vector<AllocRecord> allocs;
  std::vector<UnknownFreeRecord> unknown_frees;
  std::vector<LaunchRecord> launches;
  std::vector<std::vector<ArgSlotInfo>> arg_tables;
  std::vector<CopyRecord> copies;
  std::vector<MemsetRecord> memsets;
  std::vector<StreamRecord> streams;
  std::vector<ModuleRecord> modules;
  std::vector<ParamInfoRecord> param_infos;
  std::deque<std::string> api_name_pool;
  std::deque<std::string> string_pool; // CopyRecord::kind (string_view) storage

  // blobs
  std::vector<BufferSnapshot> blobs;
};

BundleReader::BundleReader() : impl_(std::make_unique<Impl>()) {}
BundleReader::~BundleReader() = default;

bool BundleReader::is_v1() const { return impl_->v1; }
std::string BundleReader::format_id() const {
  return impl_->v1 ? std::string(kBundleFormatId) : "kernscope-capture-v0";
}
int BundleReader::schema_version() const {
  return impl_->v1 ? kBundleSchemaVersion : 0;
}
std::string BundleReader::tool_version() const { return impl_->tool_version; }
std::string BundleReader::vendor() const { return impl_->vendor; }
std::string BundleReader::arch() const { return impl_->arch; }
std::string BundleReader::capture_mechanism() const {
  return impl_->capture_mechanism;
}
std::string BundleReader::timestamp() const { return impl_->timestamp; }
std::string BundleReader::first_kernel() const { return impl_->first_kernel; }
std::uint64_t BundleReader::max_snapshot_bytes() const {
  return impl_->max_snapshot_bytes;
}
std::uint64_t BundleReader::total_blob_bytes() const {
  return impl_->total_blob_bytes;
}
bool BundleReader::aborted() const { return impl_->aborted; }
std::string BundleReader::abort_reason() const { return impl_->abort_reason; }

const std::vector<ApiTableEntry> &BundleReader::api() const {
  return impl_->api;
}
const std::vector<AllocRecord> &BundleReader::allocs() const {
  return impl_->allocs;
}
const std::vector<UnknownFreeRecord> &BundleReader::unknown_frees() const {
  return impl_->unknown_frees;
}
const std::vector<LaunchRecord> &BundleReader::launches() const {
  return impl_->launches;
}
const std::vector<CopyRecord> &BundleReader::copies() const {
  return impl_->copies;
}
const std::vector<MemsetRecord> &BundleReader::memsets() const {
  return impl_->memsets;
}
const std::vector<StreamRecord> &BundleReader::streams() const {
  return impl_->streams;
}
const std::vector<ModuleRecord> &BundleReader::modules() const {
  return impl_->modules;
}
const std::vector<ParamInfoRecord> &BundleReader::param_infos() const {
  return impl_->param_infos;
}
const std::vector<std::vector<ArgSlotInfo>> &
BundleReader::launch_arg_tables() const {
  return impl_->arg_tables;
}
const std::vector<BufferSnapshot> &BundleReader::blobs() const {
  return impl_->blobs;
}

bool BundleReader::read_blob(const BufferSnapshot &blob,
                             std::vector<std::uint8_t> &out,
                             std::string *error) const {
  std::string bytes;
  std::string err;
  if (!ReadWholeFile(fs::path(impl_->root) / blob.file, bytes, err)) {
    if (error != nullptr) {
      *error = err;
    }
    return false;
  }
  out.assign(bytes.begin(), bytes.end());
  const std::string actual = sha256_hex(out);
  if (actual != blob.sha256) {
    if (error != nullptr) {
      *error = "schema violation: sha256 mismatch for '" + blob.file +
               "' (expected " + blob.sha256 + ", got " + actual;
      if (out.size() != blob.size) {
        *error += ", size " + std::to_string(out.size()) + " != recorded " +
                  std::to_string(blob.size) + " (truncated at byte " +
                  std::to_string(out.size()) + ")";
      }
      *error += ")";
    }
    return false;
  }
  return true;
}

namespace {

bool LoadManifest(const fs::path &root, BundleReader::Impl &impl,
                  std::string &error) {
  std::string text;
  if (!ReadWholeFile(root / "manifest.json", text, error)) {
    return false;
  }
  const auto parsed = json::Value::parse(text, error);
  if (!parsed.has_value()) {
    error = "schema violation: manifest.json: " + error;
    return false;
  }
  const json::Value &doc = *parsed;
  if (doc.kind() != json::Kind::Object) {
    error = "schema violation: manifest.json is not an object";
    return false;
  }
  std::string format_id;
  if (!GetString(doc, "format_id", format_id, error) ||
      format_id != kBundleFormatId) {
    error = "schema violation: manifest.json: bad format_id (expected \"" +
            std::string(kBundleFormatId) + "\")";
    return false;
  }
  std::string schema_version;
  if (!GetString(doc, "schema_version", schema_version, error)) {
    return false;
  }
  if (schema_version != "1.0") {
    error = "schema violation: manifest.json: unsupported schema_version \"" +
            schema_version + "\"";
    return false;
  }
  if (!GetString(doc, "tool_version", impl.tool_version, error) ||
      !GetString(doc, "vendor", impl.vendor, error) ||
      !GetString(doc, "arch", impl.arch, error) ||
      !GetString(doc, "capture_mechanism", impl.capture_mechanism, error) ||
      !GetString(doc, "timestamp", impl.timestamp, error)) {
    return false;
  }
  const json::Value *sg = doc.find("size_governance");
  if (sg == nullptr || sg->kind() != json::Kind::Object) {
    error = "schema violation: manifest.json: missing required field "
            "\"size_governance\"";
    return false;
  }
  std::int64_t max_bytes = 0;
  std::int64_t total = 0;
  if (!GetInt(*sg, "max_snapshot_bytes", max_bytes, error) ||
      !GetInt(*sg, "total_blob_bytes", total, error) ||
      !GetBool(*sg, "aborted", impl.aborted, error) ||
      !GetString(*sg, "abort_reason", impl.abort_reason, error)) {
    return false;
  }
  impl.max_snapshot_bytes = static_cast<std::uint64_t>(max_bytes);
  impl.total_blob_bytes = static_cast<std::uint64_t>(total);
  const json::Value *fk = doc.find("first_kernel");
  if (fk != nullptr && fk->kind() == json::Kind::String) {
    impl.first_kernel = fk->as_string();
  }
  return true;
}

bool LoadEvents(const fs::path &root, BundleReader::Impl &impl,
                std::string &error) {
  std::string text;
  if (!ReadWholeFile(root / "events.json", text, error)) {
    return false;
  }
  const auto parsed = json::Value::parse(text, error);
  if (!parsed.has_value()) {
    error = "schema violation: events.json: " + error;
    return false;
  }
  const json::Value &arr = *parsed;
  if (arr.kind() != json::Kind::Array) {
    error = "schema violation: events.json is not an array";
    return false;
  }
  for (const json::Value &ev : arr.as_array()) {
    if (ev.kind() != json::Kind::Object) {
      error = "schema violation: events.json: non-object event";
      return false;
    }
    std::string type;
    if (!GetString(ev, "type", type, error)) {
      return false;
    }
    std::int64_t seq = 0;
    error.clear();
    if (!GetInt(ev, "seq", seq, error)) {
      // seq is optional-tolerant here: v1 writers always carry it, but the
      // reader degrades gracefully to 0 rather than rejecting.
      error.clear();
      seq = 0;
    }
    if (type == "api") {
      std::string name;
      std::int64_t calls = 0;
      if (!GetString(ev, "name", name, error) ||
          !GetInt(ev, "calls", calls, error)) {
        return false;
      }
      ApiTableEntry e;
      e.kind = KindForApiName(name);
      e.name = e.kind == ApiKind::kCount
                   ? OwnedApiName(impl.api_name_pool, name)
                   : api_name(e.kind);
      e.calls = static_cast<std::uint64_t>(calls);
      impl.api.push_back(e);
    } else if (type == "alloc" || type == "free") {
      std::uint64_t ptr = 0;
      std::uint64_t size = 0;
      bool is_alloc = type == "alloc";
      if (!GetU64(ev, "ptr", ptr, error) || !GetU64(ev, "size", size, error) ||
          !GetBool(ev, "is_alloc", is_alloc, error)) {
        return false;
      }
      impl.allocs.push_back(AllocRecord{
          ptr, static_cast<std::size_t>(size), is_alloc,
          static_cast<std::uint64_t>(seq)});
    } else if (type == "unknown-free") {
      std::uint64_t ptr = 0;
      bool forwarded = true;
      if (!GetU64(ev, "ptr", ptr, error) ||
          !GetBool(ev, "forwarded", forwarded, error)) {
        return false;
      }
      impl.unknown_frees.push_back(
          UnknownFreeRecord{ptr, forwarded, static_cast<std::uint64_t>(seq)});
    } else if (type == "launch") {
      LaunchRecord rec;
      std::string via;
      std::int64_t shared = 0;
      if (!GetString(ev, "kernel_name", rec.kernel_name, error) ||
          !GetString(ev, "via", via, error) ||
          !GetU64(ev, "stream", rec.stream, error) ||
          !GetU64(ev, "graph_handle", rec.graph_handle, error) ||
          !GetInt(ev, "shared_mem_bytes", shared, error)) {
        return false;
      }
      rec.shared_mem_bytes = static_cast<std::uint32_t>(shared);
      if (via == "runtime") {
        rec.via = LaunchVia::kRuntimeApi;
      } else if (via == "runtime-ex") {
        rec.via = LaunchVia::kRuntimeApiEx;
      } else if (via == "graph") {
        rec.via = LaunchVia::kGraph;
      } else if (via == "driver-ex") {
        rec.via = LaunchVia::kDriverApiEx;
      } else {
        rec.via = LaunchVia::kDriverApi;
      }
      const json::Value *g = ev.find("grid");
      const json::Value *b = ev.find("block");
      auto read_dims = [&](const json::Value *d, std::uint32_t &x,
                           std::uint32_t &y, std::uint32_t &z) -> bool {
        if (d == nullptr || d->kind() != json::Kind::Array ||
            d->as_array().size() != 3) {
          error = "schema violation: launch grid/block must be [x,y,z]";
          return false;
        }
        x = static_cast<std::uint32_t>(d->as_array()[0].as_number());
        y = static_cast<std::uint32_t>(d->as_array()[1].as_number());
        z = static_cast<std::uint32_t>(d->as_array()[2].as_number());
        return true;
      };
      if (!read_dims(g, rec.grid_x, rec.grid_y, rec.grid_z) ||
          !read_dims(b, rec.block_x, rec.block_y, rec.block_z)) {
        return false;
      }
      std::vector<ArgSlotInfo> table;
      const json::Value *args = ev.find("args");
      if (args != nullptr && args->kind() == json::Kind::Array) {
        for (const json::Value &slot : args->as_array()) {
          std::string kind;
          std::string bytes;
          std::int64_t size = 0;
          if (slot.kind() != json::Kind::Object ||
              !GetString(slot, "kind", kind, error) ||
              !GetInt(slot, "size", size, error) ||
              !GetString(slot, "bytes", bytes, error)) {
            return false;
          }
          KernelArgSlot ks;
          ks.size = static_cast<std::size_t>(size);
          std::string uerr;
          ks.bytes = Unhex(bytes, uerr);
          if (!uerr.empty()) {
            error = "schema violation: launch arg bytes: " + uerr;
            return false;
          }
          ArgSlotInfo info;
          if (kind == "pointer") {
            info.is_pointer = true;
            if (!GetU64(slot, "va", info.va, error)) {
              return false;
            }
          } else if (kind != "scalar") {
            error = "schema violation: unknown arg kind '" + kind + "'";
            return false;
          }
          rec.args.push_back(std::move(ks));
          table.push_back(info);
        }
      }
      rec.seq = static_cast<std::uint64_t>(seq);
      impl.launches.push_back(std::move(rec));
      impl.arg_tables.push_back(std::move(table));
    } else if (type == "copy") {
      CopyRecord rec;
      std::string kind;
      std::int64_t cuda_kind = 0;
      if (!GetString(ev, "kind", kind, error) ||
          !GetU64(ev, "dst", rec.dst, error) ||
          !GetU64(ev, "src", rec.src, error) ||
          !GetU64(ev, "bytes", rec.bytes, error) ||
          !GetInt(ev, "cuda_kind", cuda_kind, error) ||
          !GetU64(ev, "dst_size", rec.dst_size, error) ||
          !GetU64(ev, "src_size", rec.src_size, error) ||
          !GetU64(ev, "stream", rec.stream, error)) {
        return false;
      }
      rec.kind = impl.string_pool.emplace_back(kind);
      rec.cuda_kind = static_cast<int>(cuda_kind);
      rec.seq = static_cast<std::uint64_t>(seq);
      impl.copies.push_back(rec);
    } else if (type == "memset") {
      MemsetRecord rec;
      std::int64_t value = 0;
      if (!GetU64(ev, "ptr", rec.ptr, error) ||
          !GetInt(ev, "value", value, error) ||
          !GetU64(ev, "bytes", rec.bytes, error) ||
          !GetU64(ev, "ptr_size", rec.ptr_size, error) ||
          !GetU64(ev, "stream", rec.stream, error)) {
        return false;
      }
      rec.value = static_cast<int>(value);
      rec.seq = static_cast<std::uint64_t>(seq);
      impl.memsets.push_back(rec);
    } else if (type == "stream") {
      StreamRecord rec;
      std::int64_t id = 0;
      if (!GetBool(ev, "sync", rec.is_sync, error) ||
          !GetU64(ev, "handle", rec.handle, error) ||
          !GetInt(ev, "id", id, error)) {
        return false;
      }
      rec.id = static_cast<std::uint64_t>(id);
      rec.seq = static_cast<std::uint64_t>(seq);
      impl.streams.push_back(rec);
    } else if (type == "fatbin" || type == "cubin") {
      ModuleRecord rec;
      std::int64_t mod = 0;
      if (!GetInt(ev, "mod", mod, error) ||
          !GetString(ev, "sha256", rec.sha256, error)) {
        return false;
      }
      rec.handle = static_cast<std::uint64_t>(mod);
      rec.kind = type == "fatbin" ? ModuleRecord::Kind::kFatBinary
                                  : ModuleRecord::Kind::kCubin;
      const json::Value *path = ev.find("path");
      if (path != nullptr && path->kind() == json::Kind::String) {
        rec.name = path->as_string();
      }
      const json::Value *kernels = ev.find("kernels");
      if (kernels != nullptr && kernels->kind() == json::Kind::Array) {
        for (const json::Value &k : kernels->as_array()) {
          if (k.kind() != json::Kind::String) {
            error = "schema violation: kernels must be strings";
            return false;
          }
          rec.kernels.push_back(k.as_string());
        }
      }
      rec.seq = static_cast<std::uint64_t>(seq);
      impl.modules.push_back(rec);
    } else if (type == "param_info") {
      ParamInfoRecord rec;
      std::int64_t param_index = 0;
      std::int64_t offset = 0;
      std::int64_t size = 0;
      if (!GetU64(ev, "cufunction", rec.cufunction, error) ||
          !GetInt(ev, "param_index", param_index, error) ||
          !GetInt(ev, "offset", offset, error) ||
          !GetInt(ev, "size", size, error)) {
        return false;
      }
      rec.param_index = static_cast<std::size_t>(param_index);
      rec.offset = static_cast<std::size_t>(offset);
      rec.size = static_cast<std::size_t>(size);
      impl.param_infos.push_back(rec);
    } else {
      error = "schema violation: events.json: unknown event type '" + type +
              "'";
      return false;
    }
    error.clear();
  }
  return true;
}

bool LoadBlobs(const fs::path &root, BundleReader::Impl &impl,
               std::string &error) {
  std::string text;
  if (!ReadWholeFile(root / "blobs.json", text, error)) {
    return false;
  }
  const auto parsed = json::Value::parse(text, error);
  if (!parsed.has_value()) {
    error = "schema violation: blobs.json: " + error;
    return false;
  }
  const json::Value &doc = *parsed;
  if (doc.kind() != json::Kind::Object) {
    error = "schema violation: blobs.json is not an object";
    return false;
  }
  const json::Value *arr = doc.find("blobs");
  if (arr == nullptr || arr->kind() != json::Kind::Array) {
    error = "schema violation: blobs.json: missing required field \"blobs\"";
    return false;
  }
  for (const json::Value &b : arr->as_array()) {
    BufferSnapshot blob;
    if (b.kind() != json::Kind::Object ||
        !GetString(b, "id", blob.id, error) ||
        !GetString(b, "file", blob.file, error)) {
      return false;
    }
    std::string role;
    if (!GetString(b, "role", role, error)) {
      return false;
    }
    if (role == "kernel-binary") {
      blob.role = BufferRole::kKernelBinary;
    } else if (role == "input") {
      blob.role = BufferRole::kInput;
    } else if (role == "output") {
      blob.role = BufferRole::kOutput;
    } else if (role == "arg") {
      blob.role = BufferRole::kArg;
    } else if (role == "state") {
      blob.role = BufferRole::kState;
    } else {
      error = "schema violation: blobs.json: unknown role '" + role + "'";
      return false;
    }
    if (!GetU64(b, "va", blob.va, error) ||
        !GetU64(b, "size", blob.size, error) ||
        !GetString(b, "sha256", blob.sha256, error)) {
      return false;
    }
    const json::Value *patches = b.find("patches");
    if (patches != nullptr && patches->kind() == json::Kind::Array) {
      for (const json::Value &p : patches->as_array()) {
        PatchEntry pe;
        std::int64_t offset = 0;
        if (p.kind() != json::Kind::Object ||
            !GetInt(p, "offset", offset, error) ||
            !GetU64(p, "target_va", pe.target_va, error)) {
          return false;
        }
        pe.offset = static_cast<std::uint64_t>(offset);
        blob.patches.push_back(pe);
      }
    }
    impl.blobs.push_back(std::move(blob));
  }
  // Eager provenance verification (F3): each recorded blob is checked
  // against its on-disk bytes AT LOAD TIME. sha256 + size are mandatory
  // per the format spec, so a corrupted/truncated blob file fails open
  // with a schema violation naming the blob path + offset instead of
  // surfacing lazily at read_blob.
  for (const BufferSnapshot &blob : impl.blobs) {
    std::string bytes;
    std::string rerr;
    if (!ReadWholeFile(root / blob.file, bytes, rerr)) {
      error = "schema violation: blobs.json: blob file '" + blob.file +
              "': " + rerr;
      return false;
    }
    const std::string actual = sha256_hex(std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
    if (actual != blob.sha256) {
      error = "schema violation: blobs.json: sha256 mismatch for '" +
              blob.file + "' (expected " + blob.sha256 + ", got " + actual;
      if (bytes.size() != blob.size) {
        error += ", size " + std::to_string(bytes.size()) + " != recorded " +
                 std::to_string(blob.size) + " (truncated at byte " +
                 std::to_string(bytes.size()) + ")";
      }
      error += ")";
      return false;
    }
  }
  return true;
}

} // namespace

std::unique_ptr<BundleReader> BundleReader::open(const std::string &path,
                                                 std::string *error) {
  auto reader = std::make_unique<BundleReader>();
  Impl &impl = *reader->impl_;
  std::string err;
  const fs::path p(path);
  std::error_code ec;
  const bool is_dir = fs::is_directory(p, ec);

  if (is_dir) {
    impl.root = path;
    if (fs::exists(p / "manifest.json")) {
      impl.v1 = true;
      if (!LoadManifest(p, impl, err) || !LoadEvents(p, impl, err) ||
          !LoadBlobs(p, impl, err)) {
        if (error != nullptr) {
          *error = err;
        }
        return nullptr;
      }
      return reader;
    }
    // v0 bundle directory: find the text file carrying the v0 header and
    // load *.bin sidecars (the todo-44 equivalence-bundle shape:
    // input-buffer-<n>.bin next to the text).
    std::string v0_text;
    fs::path v0_file;
    ec.clear();
    for (const auto &entry : fs::directory_iterator(p, ec)) {
      if (ec) {
        break;
      }
      if (!entry.is_regular_file(ec)) {
        continue;
      }
      std::string content;
      std::string rerr;
      if (!ReadWholeFile(entry.path(), content, rerr)) {
        continue;
      }
      if (content.rfind("kernscope-capture-v0", 0) == 0) {
        v0_text = std::move(content);
        v0_file = entry.path();
        break;
      }
    }
    if (v0_file.empty()) {
      if (error != nullptr) {
        *error = "cannot open bundle '" + path +
                 "': not a v1 bundle (no manifest.json) and no v0 text "
                 "found in the directory";
      }
      return nullptr;
    }
    V0Model model;
    if (!ParseV0Text(v0_text, model, err)) {
      if (error != nullptr) {
        *error = err + " (file: " + v0_file.string() + ")";
      }
      return nullptr;
    }
    impl.api = std::move(model.api);
    impl.api_name_pool = std::move(model.name_pool);
    impl.string_pool = std::move(model.string_pool);
    impl.allocs = std::move(model.allocs);
    impl.unknown_frees = std::move(model.unknown_frees);
    impl.launches = std::move(model.launches);
    impl.arg_tables = std::move(model.arg_tables);
    impl.copies = std::move(model.copies);
    impl.memsets = std::move(model.memsets);
    impl.streams = std::move(model.streams);
    impl.modules = std::move(model.modules);
    // Sidecar blobs: every *.bin sibling of the v0 text.
    std::vector<fs::path> bins;
    ec.clear();
    for (const auto &entry : fs::directory_iterator(p, ec)) {
      if (ec) {
        break;
      }
      if (entry.is_regular_file(ec) && entry.path().extension() == ".bin") {
        bins.push_back(entry.path());
      }
    }
    std::sort(bins.begin(), bins.end());
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < bins.size(); ++i) {
      std::string bytes;
      if (!ReadWholeFile(bins[i], bytes, err)) {
        if (error != nullptr) {
          *error = err;
        }
        return nullptr;
      }
      BufferSnapshot blob;
      std::ostringstream id;
      id << "b" << std::setfill('0') << std::setw(4) << (i + 1);
      blob.id = id.str();
      blob.role = BufferRole::kInput; // capture-mapping M11 input blobs
      blob.size = bytes.size();
      blob.sha256 = sha256_hex(std::span<const std::uint8_t>(
          reinterpret_cast<const std::uint8_t *>(bytes.data()), bytes.size()));
      blob.file = bins[i].filename().string();
      const std::size_t blob_bytes = blob.size;
      impl.blobs.push_back(std::move(blob));
      total += blob_bytes;
    }
    impl.total_blob_bytes = total;
    impl.max_snapshot_bytes = 0; // v0 bundles carry no governance
    if (!impl.launches.empty()) {
      impl.first_kernel = impl.launches.front().kernel_name;
    }
    return reader;
  }

  // Single file: must be v0 text (v1 bundles are directories).
  if (!fs::exists(p, ec)) {
    if (error != nullptr) {
      *error = "cannot open bundle '" + path + "': no such file or directory";
    }
    return nullptr;
  }
  std::string text;
  if (!ReadWholeFile(p, text, err)) {
    if (error != nullptr) {
      *error = err;
    }
    return nullptr;
  }
  if (text.rfind("kernscope-capture-v0", 0) != 0) {
    if (error != nullptr) {
      *error = "cannot open bundle '" + path +
               "': not a v0 text bundle (missing 'kernscope-capture-v0' "
               "header) and not a v1 directory";
    }
    return nullptr;
  }
  V0Model model;
  if (!ParseV0Text(text, model, err)) {
    if (error != nullptr) {
      *error = err + " (file: " + p.string() + ")";
    }
    return nullptr;
  }
  impl.root = p.parent_path().string();
  impl.api = std::move(model.api);
  impl.api_name_pool = std::move(model.name_pool);
    impl.string_pool = std::move(model.string_pool);
  impl.allocs = std::move(model.allocs);
  impl.unknown_frees = std::move(model.unknown_frees);
  impl.launches = std::move(model.launches);
  impl.arg_tables = std::move(model.arg_tables);
  impl.copies = std::move(model.copies);
  impl.memsets = std::move(model.memsets);
  impl.streams = std::move(model.streams);
  impl.modules = std::move(model.modules);
  if (!impl.launches.empty()) {
    impl.first_kernel = impl.launches.front().kernel_name;
  }
  return reader;
}

} // namespace kernscope
