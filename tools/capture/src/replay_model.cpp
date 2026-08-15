//===- replay_model.cpp - shared capture-bundle replay plan (todo 48) -----===//
//
// kernscope capture/replay track. See replay_model.h for the contract.
//
// Extracted from the todo-47 reproducer generator (tools/capture/src/
// repro_gen.cpp): the model-building half (bundle load -> launch selection ->
// device-buffer/arg plan) is shared verbatim between `emit-repro` (which
// embeds the plan into a standalone project) and the todo-48 replay engine
// (which executes it live). The emission half stays in repro_gen.cpp.
//
// Pipeline:
//   1. LOAD: v1 *.kcap/ dirs through BundleReader; v0 text bundles through
//      the permissive local grammar (empty arg bytes of runtime-internal
//      dispatches are NOT a schema violation here — they are never targets).
//   2. SELECT: the target launch (auto: first non-internal; --kernel names
//      one). Runtime-internal dispatches (__amd_rocclr_*) are skipped.
//   3. MODEL: allocation records give exact sizes; H2D copy records attach
//      the dumped input blobs; D2H copy records mark output buffers; v1
//      output snapshots become the replay validation references. Pointer
//      arg slots resolve to plan buffers (reader classification + an
//      independent copy-based classification); unresolvable buffers become
//      synthetic inputs (QA-negative path, clearly warned).
//
//===----------------------------------------------------------------------===//

#include "replay_model.h"

#include "kernscope/capture.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

namespace kernscope {
namespace {

namespace fs = std::filesystem;

constexpr std::size_t kSyntheticDefaultBytes = 4096;
constexpr std::uint8_t kSyntheticFillByte = 0xA5;

//===--- v0 text bundle model -------------------------------------------------===//
// v0 text is parsed with a PERMISSIVE grammar: runtime-internal dispatches
// carry arg slots with a size but NO bytes (`[48:]`), which BundleReader's
// compatibility mode treats as a schema violation. Those launches are never
// replay/reproducer targets (internal-kernel filter), so empty-byte slots
// are accepted here. The grammar is otherwise identical to the reader's.

struct V0Bundle {
  std::vector<AllocRecord> allocs;
  std::vector<CopyRecord> copies;
  std::vector<MemsetRecord> memsets;
  std::vector<LaunchRecord> launches;
  std::vector<std::vector<ArgSlotInfo>> arg_tables;
  std::vector<ModuleRecord> modules;
};

bool ParseV0BundleText(std::string_view text, V0Bundle &out,
                       std::string *error) {
  const auto kv_of = [](std::string_view line, std::string_view key,
                        std::string *value) {
    // token scan: <key>=<value>
    std::size_t at = 0;
    while (at < line.size()) {
      const std::size_t sp = line.find(' ', at);
      const std::string_view tok =
          sp == std::string_view::npos ? line.substr(at) : line.substr(at, sp - at);
      const std::size_t eq = tok.find('=');
      if (eq != std::string_view::npos &&
          tok.substr(0, eq) == key) {
        *value = std::string(tok.substr(eq + 1));
        // v0 writers quote scalar string values (kernel="vec_add") but NOT
        // the kernels="a","b" list (per-name quotes) - strip only when the
        // whole value is exactly one quoted string.
        if (value->size() >= 2 && value->front() == '"' &&
            value->back() == '"' &&
            std::count(value->begin(), value->end(), '"') == 2) {
          value->erase(value->begin());
          value->pop_back();
        }
        return true;
      }
      if (sp == std::string_view::npos) {
        break;
      }
      at = sp + 1;
    }
    return false;
  };
  const auto parse_u64 = [](std::string_view s, std::uint64_t *v) {
    // 0x hex or decimal (v0 convention)
    std::size_t consumed = 0;
    try {
      if (s.size() > 2 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) {
        *v = std::stoull(std::string(s), &consumed, 16);
      } else {
        *v = std::stoull(std::string(s), &consumed, 10);
      }
    } catch (...) {
      return false;
    }
    return consumed == s.size();
  };
  const auto parse_u64kv = [&](std::string_view line, std::string_view key,
                               std::uint64_t *v, std::string *err) {
    std::string s;
    if (!kv_of(line, key, &s) || !parse_u64(s, v)) {
      *err = "v0: bad or missing '" + std::string(key) + "'";
      return false;
    }
    return true;
  };
  const auto parse_dims = [](std::string_view s, std::uint32_t d[3]) {
    std::size_t at = 0;
    for (int i = 0; i < 3; ++i) {
      const std::size_t x = s.find('x', at);
      const std::string_view part = x == std::string_view::npos
                                        ? s.substr(at)
                                        : s.substr(at, x - at);
      if (part.empty()) {
        return false;
      }
      try {
        d[i] = static_cast<std::uint32_t>(std::stoul(std::string(part)));
      } catch (...) {
        return false;
      }
      if (x == std::string_view::npos) {
        return i == 2;
      }
      at = x + 1;
    }
    return true;
  };

  std::size_t line_start = 0;
  std::size_t line_no = 0;
  bool saw_header = false, saw_end = false;
  std::uint64_t seq = 0;
  while (line_start <= text.size()) {
    const std::size_t nl = text.find('\n', line_start);
    const std::size_t end = nl == std::string_view::npos ? text.size() : nl;
    std::string_view line = text.substr(line_start, end - line_start);
    ++line_no;
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    const auto fail = [&](std::string msg) {
      *error = "v0 parse error at line " + std::to_string(line_no) + ": " + msg;
      return false;
    };
    if (line.empty()) {
      if (nl == std::string_view::npos) {
        break;
      }
      line_start = nl + 1;
      continue;
    }
    if (saw_end) {
      break;
    }
    const std::size_t sp = line.find(' ');
    const std::string_view kw = sp == std::string_view::npos ? line : line.substr(0, sp);
    if (kw == "kernscope-capture-v0") {
      saw_header = true;
    } else if (kw == "end-capture-v0") {
      saw_end = true;
    } else if (kw == "alloc" || kw == "free") {
      std::uint64_t ptr = 0, size = 0;
      if (!parse_u64kv(line, "ptr", &ptr, error) ||
          !parse_u64kv(line, "size", &size, error)) {
        return fail(*error);
      }
      out.allocs.push_back(AllocRecord{ptr, static_cast<std::size_t>(size),
                                       kw == "alloc", ++seq});
    } else if (kw == "launch") {
      std::string name, via, grid_s, block_s;
      std::uint64_t shared = 0, stream = 0, graph = 0, nargs = 0;
      if (!kv_of(line, "kernel", &name) || !kv_of(line, "via", &via) ||
          !kv_of(line, "grid", &grid_s) || !kv_of(line, "block", &block_s) ||
          !parse_u64kv(line, "shared", &shared, error) ||
          !parse_u64kv(line, "stream", &stream, error) ||
          !parse_u64kv(line, "graph", &graph, error) ||
          !parse_u64kv(line, "args", &nargs, error)) {
        return fail("malformed launch record");
      }
      LaunchRecord rec;
      rec.kernel_name = name;
      rec.via = via == "runtime-ex" ? LaunchVia::kRuntimeApiEx
               : via == "graph"     ? LaunchVia::kGraph
               : via == "driver"    ? LaunchVia::kDriverApi
               : via == "driver-ex" ? LaunchVia::kDriverApiEx
                                    : LaunchVia::kRuntimeApi;
      std::uint32_t d[3] = {1, 1, 1};
      if (!parse_dims(grid_s, d)) {
        return fail("invalid grid '" + grid_s + "'");
      }
      rec.grid_x = d[0]; rec.grid_y = d[1]; rec.grid_z = d[2];
      if (!parse_dims(block_s, d)) {
        return fail("invalid block '" + block_s + "'");
      }
      rec.block_x = d[0]; rec.block_y = d[1]; rec.block_z = d[2];
      rec.shared_mem_bytes = static_cast<std::uint32_t>(shared);
      rec.stream = stream;
      rec.graph_handle = graph;
      rec.seq = ++seq;
      std::vector<ArgSlotInfo> table;
      std::size_t at = sp == std::string_view::npos ? line.size() : sp + 1;
      while (at < line.size()) {
        const std::size_t sp2 = line.find(' ', at);
        const std::string_view tok =
            sp2 == std::string_view::npos ? line.substr(at) : line.substr(at, sp2 - at);
        if (tok.size() >= 3 && tok.front() == '[' && tok.back() == ']') {
          const std::string_view inner = tok.substr(1, tok.size() - 2);
          const std::size_t colon = inner.find(':');
          if (colon == std::string_view::npos) {
            return fail("bad arg token '" + std::string(tok) + "'");
          }
          std::uint64_t slot_size = 0;
          if (!parse_u64(inner.substr(0, colon), &slot_size)) {
            return fail("bad arg size '" + std::string(tok) + "'");
          }
          KernelArgSlot slot;
          slot.size = static_cast<std::size_t>(slot_size);
          // PERMISSIVE: empty bytes allowed (runtime-internal dispatches
          // have no kernel_params); bad hex is still a schema violation.
          std::string hex = std::string(inner.substr(colon + 1));
          if (!hex.empty()) {
            if (hex.size() % 2 != 0) {
              return fail("odd-length arg hex '" + std::string(tok) + "'");
            }
            const auto nibble = [](char c) -> int {
              if (c >= '0' && c <= '9') return c - '0';
              if (c >= 'a' && c <= 'f') return c - 'a' + 10;
              if (c >= 'A' && c <= 'F') return c - 'A' + 10;
              return -1;
            };
            slot.bytes.reserve(hex.size() / 2);
            for (std::size_t i = 0; i + 1 < hex.size(); i += 2) {
              const int hi = nibble(hex[i]), lo = nibble(hex[i + 1]);
              if (hi < 0 || lo < 0) {
                return fail("invalid arg hex '" + std::string(tok) + "'");
              }
              slot.bytes.push_back(static_cast<std::uint8_t>((hi << 4) | lo));
            }
          }
          ArgSlotInfo info;
          rec.args.push_back(std::move(slot));
          table.push_back(info);
        }
        if (sp2 == std::string_view::npos) {
          break;
        }
        at = sp2 + 1;
      }
      if (rec.args.size() != nargs) {
        return fail("args count mismatch (" + std::to_string(rec.args.size()) +
                    " != " + std::to_string(nargs) + ")");
      }
      out.launches.push_back(std::move(rec));
      out.arg_tables.push_back(std::move(table));
    } else if (kw == "copy") {
      std::string kind;
      std::uint64_t dst = 0, src = 0, bytes = 0, stream = 0;
      std::uint64_t cuda_kind = 0, dst_size = 0, src_size = 0;
      if (!kv_of(line, "kind", &kind) ||
          !parse_u64kv(line, "dst", &dst, error) ||
          !parse_u64kv(line, "src", &src, error) ||
          !parse_u64kv(line, "bytes", &bytes, error) ||
          !parse_u64kv(line, "cuda-kind", &cuda_kind, error) ||
          !parse_u64kv(line, "dst-size", &dst_size, error) ||
          !parse_u64kv(line, "src-size", &src_size, error) ||
          !parse_u64kv(line, "stream", &stream, error)) {
        return fail(*error);
      }
      CopyRecord rec;
      rec.kind = kind;
      rec.dst = dst;
      rec.src = src;
      rec.bytes = static_cast<std::size_t>(bytes);
      rec.cuda_kind = static_cast<int>(cuda_kind);
      rec.dst_size = static_cast<std::size_t>(dst_size);
      rec.src_size = static_cast<std::size_t>(src_size);
      rec.stream = stream;
      rec.seq = ++seq;
      out.copies.push_back(rec);
    } else if (kw == "memset") {
      std::uint64_t ptr = 0, bytes = 0, stream = 0, value = 0, ptr_size = 0;
      if (!parse_u64kv(line, "ptr", &ptr, error) ||
          !parse_u64kv(line, "value", &value, error) ||
          !parse_u64kv(line, "bytes", &bytes, error) ||
          !parse_u64kv(line, "ptr-size", &ptr_size, error) ||
          !parse_u64kv(line, "stream", &stream, error)) {
        return fail(*error);
      }
      MemsetRecord rec;
      rec.ptr = ptr;
      rec.value = static_cast<int>(value);
      rec.bytes = static_cast<std::size_t>(bytes);
      rec.ptr_size = static_cast<std::size_t>(ptr_size);
      rec.stream = stream;
      rec.seq = ++seq;
      out.memsets.push_back(rec);
    } else if (kw == "fatbin" || kw == "cubin") {
      std::uint64_t mod = 0;
      std::string sha256;
      if (!parse_u64kv(line, "mod", &mod, error) ||
          !kv_of(line, "sha256", &sha256)) {
        return fail("malformed module record");
      }
      ModuleRecord rec;
      rec.kind = kw == "fatbin" ? ModuleRecord::Kind::kFatBinary
                                : ModuleRecord::Kind::kCubin;
      rec.handle = mod;
      rec.sha256 = sha256;
      std::string path;
      if (kv_of(line, "path", &path)) {
        rec.name = path;
      }
      std::string kernels;
      if (kv_of(line, "kernels", &kernels)) {
        std::string_view sv = kernels;
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
      rec.seq = ++seq;
      out.modules.push_back(rec);
    }
    // api / stream / unknown-free records: parsed-and-ignored (the replay
    // model does not need them).
    if (nl == std::string_view::npos) {
      break;
    }
    line_start = nl + 1;
  }
  if (!saw_header) {
    *error = "v0 parse error: missing 'kernscope-capture-v0' header";
    return false;
  }
  if (!saw_end) {
    *error = "v0 parse error: missing 'end-capture-v0' terminator "
             "(truncated bundle)";
    return false;
  }
  return true;
}

//===--- v0 sidecar blobs ------------------------------------------------------===//

struct V0Sidecars {
  // input-buffer-<n>.bin, in numeric order
  std::vector<std::vector<std::uint8_t>> input_blobs;
  // code-object-<id>.hsaco blobs, indexed by sha256
  std::unordered_map<std::string, std::vector<std::uint8_t>> binaries;
};

bool LoadV0Sidecars(const fs::path &blob_dir, V0Sidecars &out,
                    std::string *error) {
  std::error_code ec;
  if (!fs::exists(blob_dir, ec)) {
    *error = "blob dir not found: " + blob_dir.string();
    return false;
  }
  std::vector<fs::path> inputs;
  for (const auto &e : fs::directory_iterator(blob_dir, ec)) {
    if (!e.is_regular_file(ec)) {
      continue;
    }
    const std::string name = e.path().filename().string();
    if (name.rfind("input-buffer-", 0) == 0 && name.ends_with(".bin")) {
      inputs.push_back(e.path());
    } else if (name.rfind("code-object-", 0) == 0 && name.ends_with(".hsaco")) {
      std::ifstream in(e.path(), std::ios::binary);
      std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
      if (in.bad()) {
        *error = "cannot read blob " + e.path().string();
        return false;
      }
      out.binaries[sha256_hex(data)] = std::move(data);
    }
  }
  if (ec) {
    *error = "blob dir scan failed: " + blob_dir.string();
    return false;
  }
  // numeric order (input-buffer-2.bin < input-buffer-10.bin)
  std::sort(inputs.begin(), inputs.end(), [](const fs::path &a, const fs::path &b) {
    const auto num = [](const fs::path &p) {
      std::string n = p.filename().string();
      n.erase(0, std::string("input-buffer-").size());
      n.erase(n.size() - std::string(".bin").size());
      try {
        return std::stoull(n);
      } catch (...) {
        return 0ull;
      }
    };
    return num(a) < num(b);
  });
  for (const fs::path &p : inputs) {
    std::ifstream in(p, std::ios::binary);
    std::vector<std::uint8_t> data((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
    if (in.bad()) {
      *error = "cannot read blob " + p.string();
      return false;
    }
    out.input_blobs.push_back(std::move(data));
  }
  return true;
}

//===--- model construction (v0 sidecar world + v1 blob world) ----------------===//

/// Builds the device-buffer plan for a v0 bundle (sidecar blobs).
bool BuildPlanV0(const V0Bundle &model, const V0Sidecars &side, Plan &plan) {
  const auto &allocs = model.allocs;
  const auto &copies = model.copies;
  const auto &launches = model.launches;

  // allocation map (alloc records), exact sizes
  std::unordered_map<std::uint64_t, std::size_t> alloc_size;
  for (const AllocRecord &a : allocs) {
    if (a.is_alloc) {
      alloc_size[a.ptr] = a.size;
    }
  }

  // device-side copy facts: ptr -> (bytes, is_output)
  struct CopyFact {
    std::size_t bytes = 0;
    bool is_output = false; // D2H: device src copied to host
    bool is_input = false;  // H2D: host src copied into device dst
  };
  std::map<std::uint64_t, CopyFact> copy_facts;
  std::vector<std::uint64_t> h2d_order; // H2D dst ptrs, in record order
  for (const CopyRecord &c : copies) {
    CopyFact &fact = copy_facts[c.dst];
    fact.bytes = std::max(fact.bytes, static_cast<std::size_t>(c.bytes));
    if (c.cuda_kind == 1) { // H2D: dst is device
      fact.is_input = true;
      h2d_order.push_back(c.dst);
    } else if (c.cuda_kind == 2) { // D2H: src is device
      CopyFact &sf = copy_facts[c.src];
      sf.bytes = std::max(sf.bytes, static_cast<std::size_t>(c.bytes));
      sf.is_output = true;
    } else if (c.cuda_kind == 3) { // D2D: both device
      CopyFact &df = copy_facts[c.dst];
      df.bytes = std::max(df.bytes, static_cast<std::size_t>(c.bytes));
      df.is_input = true; // content produced device-side; treat as input
    }
  }

  const auto buffer_for = [&](std::uint64_t va, bool *unknown_size) -> int {
    *unknown_size = false;
    for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
      if (plan.buffers[i].va == va) {
        *unknown_size = plan.buffers[i].size_unknown;
        return static_cast<int>(i);
      }
    }
    // new buffer
    DevBuffer b;
    b.va = va;
    std::ostringstream id;
    id << "0x" << std::hex << va;
    b.id = id.str();
    b.size = 0;
    const auto as = alloc_size.find(va);
    if (as != alloc_size.end()) {
      b.size = as->second;
    } else {
      b.size_unknown = true;
      *unknown_size = true;
    }
    const auto cf = copy_facts.find(va);
    if (cf != copy_facts.end()) {
      b.size = std::max(b.size, cf->second.bytes);
      if (cf->second.is_output) {
        b.kind = BufKind::kOutput;
      } else if (cf->second.is_input) {
        b.kind = BufKind::kInput;
      }
    }
    if (b.kind == BufKind::kSynthetic) {
      b.kind = BufKind::kInput; // device buffer referenced by a launch
    }
    plan.buffers.push_back(std::move(b));
    return static_cast<int>(plan.buffers.size() - 1);
  };

  // associate H2D input blobs (record order == blob order)
  std::size_t blob_idx = 0;
  for (const std::uint64_t dst : h2d_order) {
    bool unknown = false;
    const int bi = buffer_for(dst, &unknown);
    DevBuffer &b = plan.buffers[static_cast<std::size_t>(bi)];
    if (b.bytes.empty() && blob_idx < side.input_blobs.size()) {
      b.bytes = side.input_blobs[blob_idx];
      b.size = std::max(b.size, b.bytes.size());
    } else if (b.bytes.empty()) {
      b.blob_missing = true;
    }
    ++blob_idx;
  }
  if (blob_idx < side.input_blobs.size()) {
    // more blobs than H2D copies: they are orphaned; the count mismatch is
    // a provenance warning.
    std::fprintf(stderr,
                 "[kernscope-capture] warning: %zu input blobs for %zu H2D "
                 "copy records\n",
                 side.input_blobs.size(), h2d_order.size());
  }

  // kernel binary: the module record carrying the target kernel name
  const auto &mods = model.modules;
  for (const ModuleRecord &m : mods) {
    for (const std::string &k : m.kernels) {
      if (k == plan.kernel_name || k == plan.kernel_name + ".kd" ||
          StripKernelDescriptor(k) == StripKernelDescriptor(plan.kernel_name)) {
        const auto it = side.binaries.find(m.sha256);
        if (it != side.binaries.end()) {
          plan.kernel_binary = it->second;
          plan.kernel_binary_sha = m.sha256;
        } else {
          std::fprintf(stderr,
                       "[kernscope-capture] warning: kernel binary for '%s' "
                       "missing in blob dir (sha256=%s)\n",
                       plan.kernel_name.c_str(), m.sha256.c_str());
        }
        break;
      }
    }
  }

  // argument marshalling
  const auto &arg_tables = model.arg_tables;
  std::vector<ArgSlotInfo> table;
  if (plan.launch != nullptr && !arg_tables.empty()) {
    const std::size_t li = static_cast<std::size_t>(
        plan.launch - launches.data());
    if (li < arg_tables.size()) {
      table = arg_tables[li];
    }
  }
  // pointer classification pass: alloc-map hit (like the reader) OR any
  // copy record's device-side pointer (buffers allocated via unhooked APIs
  // - the size-unknown QA path).
  std::set<std::uint64_t> device_ptrs;
  for (const CopyRecord &c : copies) {
    if (c.cuda_kind == 1 || c.cuda_kind == 3) {
      device_ptrs.insert(c.dst);
    }
    if (c.cuda_kind == 2 || c.cuda_kind == 3) {
      device_ptrs.insert(c.src);
    }
  }
  const std::size_t nargs = plan.launch != nullptr ? plan.launch->args.size() : 0;
  for (std::size_t i = 0; i < nargs; ++i) {
    const KernelArgSlot &slot = plan.launch->args[i];
    ArgPlan ap;
    ap.size = slot.size;
    const bool reader_pointer =
        i < table.size() && table[i].is_pointer;
    std::uint64_t slot_value = 0;
    if (slot.size == sizeof(std::uint64_t) && slot.bytes.size() >= 8) {
      std::memcpy(&slot_value, slot.bytes.data(), 8);
    }
    const bool copy_pointer =
        slot.size == sizeof(std::uint64_t) &&
        (alloc_size.count(slot_value) != 0 ||
         device_ptrs.count(slot_value) != 0);
    if (reader_pointer || copy_pointer) {
      ap.is_pointer = true;
      ap.pointer_va = slot_value;
      bool unknown = false;
      ap.buffer = buffer_for(slot_value, &unknown);
      // size-unknown marker: allocation map miss AND no copy byte count
      DevBuffer &b = plan.buffers[static_cast<std::size_t>(ap.buffer)];
      if (b.size == 0 || unknown) {
        b.size_unknown = true;
        b.size = std::max<std::size_t>(b.size, kSyntheticDefaultBytes);
        b.kind = BufKind::kSynthetic;
        b.bytes.assign(b.size, kSyntheticFillByte);
        b.warning = "buffer size-unknown (ptr=0x" +
                    [](std::uint64_t v) {
                      std::ostringstream os;
                      os << std::hex << v;
                      return os.str();
                    }(slot_value) +
                    "): not allocated via a captured allocation API and no "
                    "copy record fixes its size - SYNTHETIC input (deterministic "
                    "fill, cannot reproduce the original bytes)";
        plan.has_synthetic = true;
      } else if (b.bytes.empty() && b.kind == BufKind::kInput) {
        b.blob_missing = true;
        b.bytes.assign(b.size, kSyntheticFillByte);
        b.warning =
            "input buffer has no captured blob (H2D content not dumped by "
            "the capture mechanism) - SYNTHETIC input (deterministic fill, "
            "cannot reproduce the original bytes)";
        plan.has_synthetic = true;
      } else if (b.bytes.empty() && b.kind == BufKind::kOutput) {
        // output buffers are allocated + dumped, no embedded content needed
      }
    } else {
      ap.scalar = slot_value;
      if (slot.bytes.size() < 8) {
        ap.scalar = 0;
        std::memcpy(&ap.scalar, slot.bytes.data(), slot.bytes.size());
      }
    }
    plan.args.push_back(std::move(ap));
  }

  // captured output hash: not present in v0 bundles; the replay engine
  // validates against --ref files supplied by the caller.
  plan.out_sha = "";
  return true;
}

/// Builds the device-buffer plan for a v1 bundle (blobs carry VA + role).
bool BuildPlanV1(const BundleReader &reader, Plan &plan, std::string *error) {
  const auto &blobs = reader.blobs();
  std::unordered_map<std::uint64_t, std::size_t> blob_by_va;
  std::unordered_map<std::uint64_t, std::size_t> input_by_va;
  std::unordered_map<std::uint64_t, std::size_t> output_by_va;
  for (std::size_t i = 0; i < blobs.size(); ++i) {
    const BufferSnapshot &b = blobs[i];
    if (b.role == BufferRole::kKernelBinary) {
      std::vector<std::uint8_t> data;
      if (!reader.read_blob(b, data, error)) {
        return false;
      }
      plan.kernel_binary = std::move(data);
      plan.kernel_binary_sha = b.sha256;
      continue;
    }
    blob_by_va[b.va] = i;
    if (b.role == BufferRole::kInput) {
      input_by_va[b.va] = i;
    } else if (b.role == BufferRole::kOutput) {
      output_by_va[b.va] = i;
    }
  }
  const auto add_buffer = [&](std::uint64_t va, bool output) -> int {
    for (std::size_t i = 0; i < plan.buffers.size(); ++i) {
      if (plan.buffers[i].va == va) {
        return static_cast<int>(i);
      }
    }
    DevBuffer b;
    b.va = va;
    std::ostringstream id;
    id << "0x" << std::hex << va;
    b.id = id.str();
    b.kind = output ? BufKind::kOutput : BufKind::kInput;
    const auto it = blob_by_va.find(va);
    if (it != blob_by_va.end()) {
      const BufferSnapshot &blob = blobs[it->second];
      b.size = blob.size;
      b.patches = blob.patches; // embedded-pointer patch table (todo 46)
      if (!output) {
        std::vector<std::uint8_t> data;
        if (!reader.read_blob(blob, data, error)) {
          return -1;
        }
        b.bytes = std::move(data);
      }
      // output bytes: read after push_back so output_refs stays aligned
    }
    plan.buffers.push_back(std::move(b));
    const int idx = static_cast<int>(plan.buffers.size() - 1);
    if (output && it != blob_by_va.end()) {
      OutputRef ref;
      if (!reader.read_blob(blobs[it->second], ref.bytes, error)) {
        return -1;
      }
      ref.sha256 = blobs[it->second].sha256;
      ref.source = blobs[it->second].id + " (" +
                   blobs[it->second].file + ")";
      plan.output_refs.push_back(std::move(ref));
    } else {
      plan.output_refs.emplace_back(); // keep aligned with `buffers`
    }
    return idx;
  };
  const auto &launch_arg_tables = reader.launch_arg_tables();
  if (plan.launch != nullptr && !launch_arg_tables.empty()) {
    const auto &launches = reader.launches();
    const std::size_t li =
        static_cast<std::size_t>(plan.launch - launches.data());
    if (li < launch_arg_tables.size()) {
      const auto &table = launch_arg_tables[li];
      for (std::size_t i = 0; i < plan.launch->args.size(); ++i) {
        const KernelArgSlot &slot = plan.launch->args[i];
        ArgPlan ap;
        ap.size = slot.size;
        if (i < table.size() && table[i].is_pointer) {
          ap.is_pointer = true;
          ap.pointer_va = table[i].va;
          const bool out = output_by_va.count(table[i].va) != 0;
          ap.buffer = add_buffer(table[i].va, out);
          if (ap.buffer < 0) {
            return false;
          }
          DevBuffer &b = plan.buffers[static_cast<std::size_t>(ap.buffer)];
          if (b.size == 0) {
            b.size_unknown = true;
            b.size = kSyntheticDefaultBytes;
            b.kind = BufKind::kSynthetic;
            b.bytes.assign(b.size, kSyntheticFillByte);
            b.warning = "buffer has no snapshot blob in the bundle "
                        "(size-unknown) - SYNTHETIC input";
            plan.has_synthetic = true;
          } else if (b.bytes.empty() && !out) {
            b.blob_missing = true;
            b.bytes.assign(b.size, kSyntheticFillByte);
            b.warning = "input buffer snapshot missing - SYNTHETIC input";
            plan.has_synthetic = true;
          }
        } else {
          ap.scalar = 0;
          if (slot.bytes.size() <= 8) {
            std::memcpy(&ap.scalar, slot.bytes.data(), slot.bytes.size());
          }
        }
        plan.args.push_back(std::move(ap));
      }
    }
  }
  return true;
}

} // namespace

//===--- helpers ---------------------------------------------------------------===//

bool IsInternalKernel(std::string_view name) {
  // The HIP runtime's own dispatches (rocclr copy/fill kernels behind
  // hipMemcpy/hipMemset) are captured like any other launch; they are not
  // user kernels and never a replay/reproducer target.
  return name.rfind("__amd_rocclr", 0) == 0;
}

std::string StripKernelDescriptor(std::string_view name) {
  constexpr std::string_view kKd = ".kd";
  std::string s(name);
  if (s.size() > kKd.size() &&
      s.compare(s.size() - kKd.size(), kKd.size(), kKd) == 0) {
    s.resize(s.size() - kKd.size());
  }
  return s;
}

//===--- entry point -----------------------------------------------------------===//

bool BuildPlan(const std::string &bundle_path, const PlanOptions &opts,
               Plan &plan, std::string *error) {
  if (bundle_path.empty()) {
    *error = "bundle path is required";
    return false;
  }
  plan.bundle_path = bundle_path;

  // v0 text bundles are detected by their header BEFORE BundleReader::open:
  // the reader's v0 compatibility mode rejects empty-byte arg slots
  // (runtime-internal dispatches), so v0 text is parsed with the permissive
  // grammar below. BundleReader stays the v1 loader.
  const fs::path bpath(bundle_path);
  const bool is_v0_text = !fs::is_directory(bpath);
  if (is_v0_text) {
    std::ifstream probe(bpath, std::ios::binary);
    std::string first;
    std::getline(probe, first);
    if (first != "kernscope-capture-v0") {
      *error = "unrecognized bundle (not a v0 text bundle and not a v1 "
               "*.kcap/ directory): " + bundle_path;
      return false;
    }
  }

  std::unique_ptr<BundleReader> reader;
  V0Bundle v0model;
  const std::vector<LaunchRecord> *launches = nullptr;

  if (!is_v0_text) {
    reader = BundleReader::open(bundle_path, error);
    if (!reader) {
      return false;
    }
    plan.tool_version = reader->tool_version();
    plan.mechanism = reader->capture_mechanism();
    plan.timestamp = reader->timestamp();
    plan.vendor = reader->vendor();
    plan.arch = reader->arch();
    if (plan.vendor != "amd" && plan.vendor != "nvidia") {
      *error = "unsupported vendor in bundle manifest: " + plan.vendor;
      return false;
    }
    launches = &reader->launches();
  } else {
    plan.vendor = opts.vendor;
    plan.arch = opts.arch;
    if (plan.vendor != "amd" && plan.vendor != "nvidia") {
      *error =
          "v0 bundle has no manifest; --vendor amd|nvidia is required";
      return false;
    }
    if (plan.arch.empty()) {
      *error = "v0 bundle has no manifest; --arch (e.g. gfx1101, sm_80) is "
               "required";
      return false;
    }
    plan.tool_version = "kernscope-capture (v0 text bundle)";
    plan.mechanism = "unknown (v0 has no envelope)";
    std::ifstream in(bpath, std::ios::binary);
    if (!in) {
      *error = "cannot open bundle " + bundle_path;
      return false;
    }
    const std::string text((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    if (!ParseV0BundleText(text, v0model, error)) {
      return false;
    }
    launches = &v0model.launches;
  }

  // select the launch
  const LaunchRecord *target = nullptr;
  for (const LaunchRecord &l : *launches) {
    if (!opts.kernel.empty()) {
      if (l.kernel_name == opts.kernel ||
          StripKernelDescriptor(l.kernel_name) ==
              StripKernelDescriptor(opts.kernel)) {
        target = &l;
        break;
      }
    } else if (!IsInternalKernel(l.kernel_name)) {
      target = &l;
      break;
    }
  }
  if (target == nullptr) {
    *error = "no launch record found in bundle (kernel '"
             + std::string(opts.kernel) + "')";
    return false;
  }
  plan.launch = target;
  plan.kernel_name = target->kernel_name;

  // model
  if (!is_v0_text) {
    if (!BuildPlanV1(*reader, plan, error)) {
      return false;
    }
  } else {
    fs::path blob_dir = opts.blob_dir.empty()
                            ? fs::path(bundle_path).parent_path()
                            : fs::path(opts.blob_dir);
    V0Sidecars side;
    if (!LoadV0Sidecars(blob_dir, side, error)) {
      return false;
    }
    if (!BuildPlanV0(v0model, side, plan)) {
      return false;
    }
  }
  return true;
}

} // namespace kernscope
