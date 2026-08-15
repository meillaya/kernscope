//===- repro_gen.cpp - standalone reproducer generator (todo 47) -----------===//
//
// kernscope capture/replay track. See repro_gen.h for the contract.
//
// Pipeline (documented here because the emitted project IS the deliverable):
//
//   1. LOAD: BundleReader opens the bundle (v1 or v0 compatibility mode).
//      vendor/arch come from the v1 manifest; for v0 text bundles (todos
//      43/44/45 output) they are CLI-supplied (--vendor/--arch) since the
//      v0 format carries no envelope.
//   2. SELECT: the target launch. Auto-selection skips runtime-internal
//      dispatches (the HIP runtime's __amd_rocclr_* copy kernels behind
//      hipMemcpy are captured too and are NOT user kernels); --kernel
//      overrides.
//   3. MODEL: the device-buffer plan. Allocation records give exact sizes;
//      H2D copy records (cuda-kind=1) associate the dumped input blobs with
//      their device buffers in record order; D2H copy records (cuda-kind=2)
//      mark the output buffers. Kernel-argument pointer slots are resolved
//      against this plan (reader classification + an independent copy-based
//      classification for buffers allocated through unhooked APIs - the
//      size-unknown QA path). Unresolvable buffers become synthetic inputs.
//   4. EMBED: kernel binary + input blobs become byte arrays in .inc files
//      (deterministic hex dumps). The emitted main.<cu|hip> marshals the
//      captured arguments, patches pointer slots to the NEW device VAs,
//      module-loads the code object, launches and dumps the outputs.
//   5. PIN: CMakeLists.txt hard-codes vendor + arch from the bundle (v1
//      manifest / CLI for v0) and records the capture provenance; HIP
//      produces an executable, CUDA is compile-only (no NVIDIA hardware
//      locally; runtime validation is owner-gated via rental-validate.sh).
//
// Standalone contract: the emitted project contains NO reference to kernscope
// libraries - only <hip/*> / <cuda*>/<cu*> toolkit headers. (The provenance
// comment mentions the tool that generated it; that is documentation.)
//
//===----------------------------------------------------------------------===//

#include "repro_gen.h"

#include "kernscope/bundle.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <unordered_map>

namespace kernscope {
namespace {

namespace fs = std::filesystem;

constexpr std::string_view kSyntheticMarker = "KERNSCOPE_SYNTHETIC_INPUT";
constexpr std::size_t kSyntheticDefaultBytes = 4096;
constexpr std::uint8_t kSyntheticFillByte = 0xA5;

//===--- helpers -------------------------------------------------------------===//

bool WriteFile(const fs::path &path, std::string_view content,
               std::string *error) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    *error = "cannot write " + path.string();
    return false;
  }
  out << content;
  if (!out) {
    *error = "write failed: " + path.string();
    return false;
  }
  return true;
}

std::string HexLine(std::span<const std::uint8_t> bytes, std::size_t &at,
                    std::size_t per_line) {
  std::ostringstream os;
  os << "  ";
  const std::size_t end = std::min(at + per_line, bytes.size());
  for (; at < end; ++at) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "0x%02x,", bytes[at]);
    os << buf;
  }
  return os.str();
}

/// Deterministic .inc file: 16 bytes per line, hex literals.
bool WriteInc(const fs::path &path, std::span<const std::uint8_t> bytes,
              std::string *error) {
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    *error = "cannot write " + path.string();
    return false;
  }
  std::size_t at = 0;
  while (at < bytes.size()) {
    out << HexLine(bytes, at, 16) << "\n";
  }
  return out.good();
}

bool IsInternalKernel(std::string_view name) {
  // The HIP runtime's own dispatches (rocclr copy/fill kernels behind
  // hipMemcpy/hipMemset) are captured like any other launch; they are not
  // user kernels and never the reproducer target.
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

//===--- v0 text bundle model -------------------------------------------------===//
// Why a parser here instead of BundleReader's v0 compatibility mode: the
// reader rejects the real todo-45 HIP output - runtime-internal dispatches
// (__amd_rocclr_* behind hipMemcpy) carry arg slots with a size but NO
// bytes (`[48:]`), which the reader treats as a schema violation. Those
// launches are never reproducer targets (internal-kernel filter), so this
// parser accepts empty-byte slots. The grammar is otherwise identical to
// the reader's. libs/capture-core is off-limits for this todo.

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
    // api / stream / unknown-free records: parsed-and-ignored (the
    // reproducer model does not need them).
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

//===--- the device-buffer plan -----------------------------------------------===//

enum class BufKind { kInput, kOutput, kSynthetic };

struct DevBuffer {
  std::string id;          // "b0001" / "input-buffer-1.bin" / ptr hex
  std::uint64_t va = 0;    // ORIGINAL device VA (0 = none)
  BufKind kind = BufKind::kSynthetic;
  std::size_t size = 0;    // byte count (synthetic: the planned size)
  std::vector<std::uint8_t> bytes; // embedded input content
  bool size_unknown = false;       // QA-negative marker
  bool blob_missing = false;       // input had no captured blob
  std::string warning;             // emitted as a comment + runtime note
};

struct ArgPlan {
  std::size_t size = 0;     // slot byte size
  bool is_pointer = false;
  std::uint64_t pointer_va = 0; // captured pointer value (ORIGINAL VA)
  int buffer = -1;              // index into the plan's buffers (-1 = scalar)
  std::uint64_t scalar = 0;     // scalar value (LE, low `size` bytes)
};

struct Plan {
  std::string vendor, arch, mechanism, tool_version, timestamp, kernel_name;
  std::string bundle_path;
  const LaunchRecord *launch = nullptr;
  std::vector<DevBuffer> buffers;
  std::vector<ArgPlan> args;
  std::vector<std::uint8_t> kernel_binary; // empty => not embedded
  std::string kernel_binary_sha;
  bool has_synthetic = false;
  std::string out_sha; // captured output hash, when derivable ("" = n/a)
};

//===--- model construction (v0 sidecar world + v1 blob world) ----------------===//

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
    // more blobs than H2D copies: attach to the first input buffer? No -
    // they are orphaned; the count mismatch is a provenance warning.
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

  // captured output hash: not present in v0 bundles; the caller asserts the
  // reproducer dump against the live app output hash itself.
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
      if (!output) {
        std::vector<std::uint8_t> data;
        if (!reader.read_blob(blob, data, error)) {
          return -1;
        }
        b.bytes = std::move(data);
      }
    }
    plan.buffers.push_back(std::move(b));
    return static_cast<int>(plan.buffers.size() - 1);
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

//===--- emission --------------------------------------------------------------===//

std::string BytesComment(std::span<const std::uint8_t> bytes) {
  std::ostringstream os;
  os << bytes.size() << " bytes";
  return os.str();
}

std::string EmitMainSource(const Plan &plan) {
  const bool hip = plan.vendor == "amd";
  const std::string guard_inc = hip ? "hip/hip_runtime.h" : "cuda_runtime.h";
  std::ostringstream os;
  os << "//===- main." << (hip ? "hip" : "cu")
     << " - standalone kernscope reproducer (GENERATED) -----------------===//\n";
  os << "//\n";
  os << "// Generated by `kernscope-capture emit-repro` (kernscope capture\n";
  os << "// track, todo 47). DO NOT EDIT - re-emit from the bundle instead.\n";
  os << "//\n";
  os << "// Provenance:\n";
  os << "//   bundle:              " << plan.bundle_path << "\n";
  os << "//   capture tool:        " << plan.tool_version << "\n";
  os << "//   capture mechanism:   " << plan.mechanism << "\n";
  os << "//   vendor/arch:         " << plan.vendor << " / " << plan.arch << "\n";
  os << "//   kernel:              " << plan.kernel_name << "\n";
  if (plan.launch != nullptr) {
    os << "//   grid:                " << plan.launch->grid_x << "x"
       << plan.launch->grid_y << "x" << plan.launch->grid_z << "\n";
    os << "//   block:               " << plan.launch->block_x << "x"
       << plan.launch->block_y << "x" << plan.launch->block_z << "\n";
    os << "//   shared_mem_bytes:    " << plan.launch->shared_mem_bytes << "\n";
  }
  if (!plan.kernel_binary_sha.empty()) {
    os << "//   kernel binary sha256: " << plan.kernel_binary_sha << "\n";
  }
  os << "//\n";
  os << "// THIS REPRODUCER IS STANDALONE: it depends ONLY on the "
     << (hip ? "HIP" : "CUDA") << " toolkit\n";
  os << "// (arch pinned in CMakeLists.txt). It embeds the captured kernel\n";
  os << "// binary and input buffers, replays the captured launch through the\n";
  os << "// module API and dumps the captured output buffers for diffing. No\n";
  os << "// kernscope code is included or linked.\n";
  if (plan.has_synthetic) {
    os << "//\n";
    os << "// WARNING: this bundle contains size-unknown / unblobbed buffers.\n";
    os << "// Their original contents were NOT captured; they are replaced by\n";
    os << "// SYNTHETIC inputs (deterministic fill, see KERNSCOPE_SYNTHETIC_"
          "INPUT\n";
    os << "// below). The launch replays but the dumped output CANNOT match\n";
    os << "// the original for these buffers.\n";
  }
  os << "//===----------------------------------------------------------------===//\n";
  os << "\n";
  if (plan.has_synthetic) {
    os << "// Marked synthetic-input reproducer (QA negative path): original\n";
    os << "// buffer contents were not captured.\n";
    os << "#define " << kSyntheticMarker << " 1\n\n";
  }
  os << "#include <" << guard_inc << ">\n";
  if (hip) {
    os << "#include <hip/hip_runtime_api.h>\n";
  } else {
    os << "#include <cuda.h>\n";
  }
  os << "#include <cstdint>\n#include <cstdio>\n#include <cstdlib>\n"
        "#include <cstring>\n#include <vector>\n\n";

  if (!plan.kernel_binary.empty()) {
    os << "//--- embedded kernel binary (sha256 " << plan.kernel_binary_sha
       << ", " << BytesComment(plan.kernel_binary) << ") --------------------\n";
    os << "static const unsigned char kKernelBinary[] = {\n"
          "#include \"kernel_binary.inc\"\n"
          "};\n";
    os << "static const std::size_t kKernelBinaryBytes = sizeof(kKernelBinary);\n\n";
  }

  int in_idx = 0;
  for (const DevBuffer &b : plan.buffers) {
    if (b.kind == BufKind::kOutput || b.bytes.empty()) {
      continue;
    }
    os << "//--- embedded input buffer " << b.id << " (" << b.size
       << " bytes"
       << (b.size_unknown ? ", SYNTHETIC" : "") << ") -----------------------\n";
    os << "static const unsigned char kInput_" << in_idx << "[] = {\n"
       << "#include \"input_" << in_idx << ".inc\"\n"
       << "};\n\n";
    ++in_idx;
  }

  // buffer plan table
  os << "//--- device-buffer plan --------------------------------------------\n";
  os << "namespace {\n";
  os << "enum class BufKind { kInput, kOutput, kSynthetic };\n";
  os << "struct BufPlan {\n";
  os << "  const char *id;\n";
  os << "  BufKind kind;\n";
  os << "  std::size_t size;\n";
  os << "  const unsigned char *data; // embedded bytes (inputs only)\n";
  os << "  bool synthetic;\n";
  os << "  std::uint64_t old_va; // original VA (provenance)\n";
  os << "};\n";
  os << "const BufPlan kBufs[] = {\n";
  int data_src = 0;
  for (const DevBuffer &b : plan.buffers) {
    const char *kind = b.kind == BufKind::kOutput
                           ? "BufKind::kOutput"
                           : (b.kind == BufKind::kSynthetic
                                  ? "BufKind::kSynthetic"
                                  : "BufKind::kInput");
    std::ostringstream dataref;
    if (!b.bytes.empty()) {
      dataref << "kInput_" << data_src++;
    }
    os << "  { \"" << b.id << "\", " << kind << ", " << b.size << ", "
       << (b.bytes.empty() ? std::string("nullptr") : dataref.str()) << ", "
       << (b.size_unknown || b.blob_missing ? "true" : "false") << ", 0x"
       << std::hex << b.va << std::dec << " },\n";
  }
  os << "};\n";
  os << "const int kNumBufs = " << plan.buffers.size() << ";\n";
  os << "} // namespace\n\n";

  // argument plan
  os << "//--- captured kernel arguments (pointer slots re-pointed) -----------\n";
  os << "namespace {\n";
  os << "struct ArgPlan {\n";
  os << "  std::size_t size;\n";
  os << "  bool is_pointer;\n";
  os << "  int buf_index; // kBufs index for pointer slots\n";
  os << "  std::uint64_t scalar;\n";
  os << "};\n";
  os << "const ArgPlan kArgs[] = {\n";
  for (const ArgPlan &a : plan.args) {
    os << "  { " << a.size << ", " << (a.is_pointer ? "true" : "false")
       << ", " << a.buffer << ", 0x" << std::hex << a.scalar << std::dec
       << " },\n";
  }
  os << "};\n";
  os << "const int kNumArgs = " << plan.args.size() << ";\n";
  os << "} // namespace\n\n";

  os << "static int die(const char *msg, " << (hip ? "hipError_t" : "cudaError_t")
     << " err) {\n";
  if (hip) {
    os << "  std::fprintf(stderr, \"[repro] %s: %s\\n\", msg,\n"
          "              hipGetErrorString(err));\n";
  } else {
    os << "  std::fprintf(stderr, \"[repro] %s: %s\\n\", msg,\n"
          "              cudaGetErrorString(err));\n";
  }
  os << "  return 1;\n";
  os << "}\n";
  if (!hip) {
    os << "// driver-API error helper (cuModuleLoadData / cuLaunchKernel)\n";
    os << "static int die_drv(const char *msg, CUresult err) {\n";
    os << "  const char *s = nullptr;\n";
    os << "  cuGetErrorString(err, &s);\n";
    os << "  std::fprintf(stderr, \"[repro] %s: %s\\n\", msg, s ? s : \"?\");\n";
    os << "  return 1;\n";
    os << "}\n";
  }
  os << "\n";

  os << "int main(int argc, char **argv) {\n";
  os << "  const char *out_path = argc > 1 ? argv[1] : \"output.bin\";\n";
  for (const DevBuffer &b : plan.buffers) {
    if (!b.warning.empty()) {
      os << "  std::fprintf(stderr, \"[repro] WARNING: " << b.warning
         << "\\n\");\n";
    }
  }
  os << "  if (" << (hip ? "hipSetDevice(0)" : "cudaSetDevice(0)")
     << " != " << (hip ? "hipSuccess" : "cudaSuccess")
     << ") { std::fprintf(stderr, \"[repro] no CUDA/HIP device\\n\"); return "
        "1; }\n";
  os << "\n";
  os << "  // device buffers\n";
  os << "  std::vector<void *> dev(kNumBufs, nullptr);\n";
  os << "  for (int i = 0; i < kNumBufs; ++i) {\n";
  os << "    const auto err = " << (hip ? "hipMalloc" : "cudaMalloc")
     << "(&dev[i], kBufs[i].size);\n";
  os << "    if (err != " << (hip ? "hipSuccess" : "cudaSuccess") << ") {\n";
  os << "      return die(\"malloc\", err);\n";
  os << "    }\n";
  os << "  }\n";
  os << "  // upload inputs\n";
  os << "  for (int i = 0; i < kNumBufs; ++i) {\n";
  os << "    if (kBufs[i].data == nullptr) {\n";
  os << "      continue;\n";
  os << "    }\n";
  os << "    const auto err = " << (hip ? "hipMemcpy" : "cudaMemcpy")
     << "(dev[i], kBufs[i].data, kBufs[i].size, "
     << (hip ? "hipMemcpyHostToDevice" : "cudaMemcpyHostToDevice") << ");\n";
  os << "    if (err != " << (hip ? "hipSuccess" : "cudaSuccess") << ") {\n";
  os << "      return die(\"memcpy H2D\", err);\n";
  os << "    }\n";
  os << "  }\n";
  os << "\n";
  os << "  // kernel binary\n";
  if (plan.kernel_binary.empty()) {
    os << "  std::fprintf(stderr,\n"
          "    \"[repro] FATAL: the bundle has no kernel binary blob for '%s' "
          "(missing cubin/hsaco) - nothing to launch. \"\n"
          "    \"Compile-only reproducer.\\n\", \"" << plan.kernel_name
       << "\");\n";
    os << "  return 1;\n";
  } else {
    os << "  " << (hip ? "hipModule_t" : "CUmodule") << " mod;\n";
    os << "  " << (hip ? "hipFunction_t" : "CUfunction") << " fn;\n";
    os << "  auto err = " << (hip ? "hipModuleLoadData" : "cuModuleLoadData")
       << "(&mod, kKernelBinary);\n";
    os << "  if (err != " << (hip ? "hipSuccess" : "CUDA_SUCCESS")
       << (hip ? ") { return die(\"module load\", err); }\n" : ") { return die_drv(\"module load\", err); }\n");
    os << "  err = " << (hip ? "hipModuleGetFunction" : "cuModuleGetFunction")
       << "(&fn, mod, \"" << StripKernelDescriptor(plan.kernel_name) << "\");\n";
    os << "  if (err != " << (hip ? "hipSuccess" : "CUDA_SUCCESS")
       << (hip ? ") { return die(\"function get\", err); }\n" : ") { return die_drv(\"function get\", err); }\n");
    os << "\n";
    os << "  // marshalled arguments: pointer slots carry the NEW device VA\n";
    os << "  std::vector<std::uint8_t> slot_data(static_cast<std::size_t>("
          "kNumArgs) * 8, 0);\n";
    os << "  std::vector<void *> kparams(kNumArgs);\n";
    os << "  for (int i = 0; i < kNumArgs; ++i) {\n";
    os << "    if (kArgs[i].is_pointer) {\n";
    os << "      const std::uint64_t v = reinterpret_cast<std::uint64_t>("
          "dev[kArgs[i].buf_index]);\n";
    os << "      std::memcpy(&slot_data[static_cast<std::size_t>(i) * 8], &v, "
          "sizeof(v));\n";
    os << "    } else {\n";
    os << "      std::memcpy(&slot_data[static_cast<std::size_t>(i) * 8], "
          "&kArgs[i].scalar, kArgs[i].size);\n";
    os << "    }\n";
    os << "    kparams[i] = &slot_data[static_cast<std::size_t>(i) * 8];\n";
    os << "  }\n";
    os << "\n";
    if (hip) {
      os << "  err = hipModuleLaunchKernel(fn, " << plan.launch->grid_x << ", "
         << plan.launch->grid_y << ", " << plan.launch->grid_z << ", "
         << plan.launch->block_x << ", " << plan.launch->block_y << ", "
         << plan.launch->block_z << ", " << plan.launch->shared_mem_bytes
         << ", nullptr, kparams.data(), nullptr);\n";
    } else {
      os << "  err = cuLaunchKernel(fn, " << plan.launch->grid_x << ", "
         << plan.launch->grid_y << ", " << plan.launch->grid_z << ", "
         << plan.launch->block_x << ", " << plan.launch->block_y << ", "
         << plan.launch->block_z << ", " << plan.launch->shared_mem_bytes
         << ", nullptr, kparams.data(), nullptr);\n";
    }
    os << "  if (err != " << (hip ? "hipSuccess" : "CUDA_SUCCESS")
       << (hip ? ") { return die(\"launch\", err); }\n" : ") { return die_drv(\"launch\", err); }\n");
    os << "  if (" << (hip ? "hipDeviceSynchronize" : "cudaDeviceSynchronize")
       << "() != " << (hip ? "hipSuccess" : "cudaSuccess")
       << ") { return 1; }\n";
    os << "\n";
    os << "  // dump captured output buffers\n";
    os << "  int out_idx = 0;\n";
    os << "  for (int i = 0; i < kNumBufs; ++i) {\n";
    os << "    if (kBufs[i].kind != BufKind::kOutput) {\n";
    os << "      continue;\n";
    os << "    }\n";
    os << "    std::vector<std::uint8_t> host(kBufs[i].size);\n";
    os << "    const auto derr = " << (hip ? "hipMemcpy" : "cudaMemcpy")
       << "(host.data(), dev[i], host.size(), "
       << (hip ? "hipMemcpyDeviceToHost" : "cudaMemcpyDeviceToHost") << ");\n";
    os << "    if (derr != " << (hip ? "hipSuccess" : "cudaSuccess")
       << ") { return die(\"memcpy D2H\", derr); }\n";
    os << "    std::string p(out_path);\n";
    os << "    if (out_idx > 0) { p += \"-\" + std::to_string(out_idx); }\n";
    os << "    FILE *fp = std::fopen(p.c_str(), \"wb\");\n";
    os << "    if (!fp) { std::fprintf(stderr, \"[repro] cannot open %s\\n\", "
          "p.c_str()); return 1; }\n";
    os << "    std::fwrite(host.data(), 1, host.size(), fp);\n";
    os << "    std::fclose(fp);\n";
    os << "    std::printf(\"[repro] dumped %s (%zu bytes)\\n\", p.c_str(), "
          "host.size());\n";
    os << "    ++out_idx;\n";
    os << "  }\n";
    os << "  return 0;\n";
  }
  os << "}\n";
  return os.str();
}

std::string EmitCmake(const Plan &plan) {
  std::ostringstream os;
  os << "# CMakeLists.txt - standalone kernscope reproducer (GENERATED).\n";
  os << "#\n";
  os << "# Generated by `kernscope-capture emit-repro` (todo 47). Pinned to\n";
  os << "# the bundle's capture envelope: vendor=" << plan.vendor
     << " arch=" << plan.arch << ".\n";
  os << "#\n";
  os << "# Standalone by construction: this project depends ONLY on the "
     << (plan.vendor == "amd" ? "HIP" : "CUDA")
     << " toolkit - no kernscope code.\n";
  os << "#\n";
  os << "# Build (HIP, gfx1101):  cmake -S . -B build && cmake --build build\n";
  os << "# Build (CUDA):          cmake -S . -B build && cmake --build build\n";
  os << "#                         (COMPILE-ONLY: the emitted main.cu is\n";
  os << "#                         compiled with nvcc -c; NVIDIA runtime\n";
  os << "#                         validation is owner-gated).\n";
  os << "#\n";
  os << "# Toolkit provenance: captured with " << plan.tool_version << "\n";
  os << "# (capture mechanism " << plan.mechanism << "); kernel '"
     << plan.kernel_name << "'.\n";
  os << "#\n";
  os << "cmake_minimum_required(VERSION 3.20)\n";
  os << "project(kernscope_repro_" << StripKernelDescriptor(plan.kernel_name)
     << " LANGUAGES CXX)\n";
  os << "\n";
  os << "set(KERNSCOPE_REPRO_VENDOR \"" << plan.vendor << "\")\n";
  os << "set(KERNSCOPE_REPRO_ARCH \"" << plan.arch << "\")\n";
  os << "set(KERNSCOPE_REPRO_KERNEL \"" << plan.kernel_name << "\")\n";
  os << "\n";
  if (plan.vendor == "amd") {
    os << "# HIP executable (the full reproducer: embed -> load -> launch).\n";
    os << "find_program(KERNSCOPE_REPRO_HIPCC hipcc REQUIRED)\n";
    os << "message(STATUS \"kernscope reproducer: hipcc=${KERNSCOPE_REPRO_HIPCC} "
          "arch=${KERNSCOPE_REPRO_ARCH}\")\n";
    os << "add_custom_command(OUTPUT kernscope_repro\n";
    os << "  COMMAND ${KERNSCOPE_REPRO_HIPCC} -std=c++17\n";
    os << "          --offload-arch=${KERNSCOPE_REPRO_ARCH}\n";
    os << "          ${CMAKE_CURRENT_SOURCE_DIR}/main.hip\n";
    os << "          -o ${CMAKE_CURRENT_BINARY_DIR}/kernscope_repro\n";
    os << "  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.hip\n";
    os << "  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}\n";
    os << "  COMMENT \"hipcc: standalone reproducer for ${KERNSCOPE_REPRO_KERNEL}\"\n";
    os << "  VERBATIM)\n";
    os << "add_custom_target(kernscope_repro ALL\n";
    os << "  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernscope_repro)\n";
  } else {
    os << "# CUDA COMPILE-ONLY (no NVIDIA hardware in this environment; the\n";
    os << "# emitted object is the CI gate. Runtime validation is owner-gated\n";
    os << "# via scripts/rental-validate.sh).\n";
    os << "find_program(KERNSCOPE_REPRO_NVCC nvcc REQUIRED)\n";
    os << "message(STATUS \"kernscope reproducer: nvcc=${KERNSCOPE_REPRO_NVCC} "
          "arch=${KERNSCOPE_REPRO_ARCH} (compile-only)\")\n";
    os << "add_custom_command(OUTPUT kernscope_repro.cu.o\n";
    os << "  COMMAND ${KERNSCOPE_REPRO_NVCC} -std=c++17\n";
    os << "          -arch=${KERNSCOPE_REPRO_ARCH} -c\n";
    os << "          ${CMAKE_CURRENT_SOURCE_DIR}/main.cu\n";
    os << "          -o ${CMAKE_CURRENT_BINARY_DIR}/kernscope_repro.cu.o\n";
    os << "  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/main.cu\n";
    os << "  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}\n";
    os << "  COMMENT \"nvcc: compile-only reproducer for ${KERNSCOPE_REPRO_KERNEL}\"\n";
    os << "  VERBATIM)\n";
    os << "add_custom_target(kernscope_repro ALL\n";
    os << "  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/kernscope_repro.cu.o)\n";
  }
  return os.str();
}

std::string EmitReadme(const Plan &plan) {
  std::ostringstream os;
  os << "# kernscope standalone reproducer: " << plan.kernel_name << "\n\n";
  os << "Generated by `kernscope-capture emit-repro` (todo 47) from\n"
        "`" << plan.bundle_path << "`.\n\n";
  os << "| field | value |\n|---|---|\n";
  os << "| vendor / arch | " << plan.vendor << " / " << plan.arch << " |\n";
  os << "| capture tool | " << plan.tool_version << " |\n";
  os << "| capture mechanism | " << plan.mechanism << " |\n";
  os << "| kernel | " << plan.kernel_name << " |\n";
  if (plan.launch != nullptr) {
    os << "| grid | " << plan.launch->grid_x << "x" << plan.launch->grid_y
       << "x" << plan.launch->grid_z << " |\n";
    os << "| block | " << plan.launch->block_x << "x" << plan.launch->block_y
       << "x" << plan.launch->block_z << " |\n";
    os << "| shared mem | " << plan.launch->shared_mem_bytes << " B |\n";
  }
  if (!plan.kernel_binary_sha.empty()) {
    os << "| kernel binary sha256 | `" << plan.kernel_binary_sha << "` |\n";
  }
  if (!plan.timestamp.empty()) {
    os << "| captured at | " << plan.timestamp << " |\n";
  }
  os << "\n";
  os << "This project is STANDALONE (zero kernscope dependencies): build with\n";
  if (plan.vendor == "amd") {
    os << "the HIP toolkit (`cmake -S . -B build && cmake --build build`),\n";
  } else {
    os << "the CUDA toolkit (`cmake -S . -B build && cmake --build build` -\n";
    os << "compile-only gate; runtime validation is owner-gated),\n";
  }
  os << "then run `./build/kernscope_repro <output.bin>` to reproduce the\n";
  os << "captured launch. Compare the dumped output against the captured\n";
  os << "output hash.\n";
  if (plan.has_synthetic) {
    os << "\n## Synthetic inputs (warning)\n\n";
    os << "This bundle contained size-unknown or unblobbed buffers; their\n";
    os << "original contents were not captured. The emitted reproducer\n";
    os << "substitutes deterministic fills and flags each one at runtime and\n";
    os << "in the source (`KERNSCOPE_SYNTHETIC_INPUT`). The output cannot\n";
    os << "match the original for these buffers.\n";
  }
  return os.str();
}

} // namespace

//===--- entry point -----------------------------------------------------------===//

bool emit_reproducer(const ReproduceOptions &opts, std::string *error) {
  if (opts.bundle_path.empty() || opts.out_dir.empty()) {
    *error = "bundle path and out dir are required";
    return false;
  }

  Plan plan;
  plan.bundle_path = opts.bundle_path;

  // v0 text bundles are detected by their header BEFORE BundleReader::open:
  // the reader's v0 compatibility mode rejects empty-byte arg slots
  // (runtime-internal dispatches), so v0 text is parsed with the permissive
  // grammar below. BundleReader stays the v1 loader.
  const fs::path bpath(opts.bundle_path);
  const bool is_v0_text = !fs::is_directory(bpath);
  if (is_v0_text) {
    std::ifstream probe(bpath, std::ios::binary);
    std::string first;
    std::getline(probe, first);
    if (first != "kernscope-capture-v0") {
      *error = "unrecognized bundle (not a v0 text bundle and not a v1 "
               "*.kcap/ directory): " + opts.bundle_path;
      return false;
    }
  }

  std::unique_ptr<BundleReader> reader;
  V0Bundle v0model;
  const std::vector<LaunchRecord> *launches = nullptr;

  if (!is_v0_text) {
    reader = BundleReader::open(opts.bundle_path, error);
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
      *error = "cannot open bundle " + opts.bundle_path;
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
                            ? fs::path(opts.bundle_path).parent_path()
                            : fs::path(opts.blob_dir);
    V0Sidecars side;
    if (!LoadV0Sidecars(blob_dir, side, error)) {
      return false;
    }
    if (!BuildPlanV0(v0model, side, plan)) {
      return false;
    }
  }

  // emit
  std::error_code ec;
  if (!opts.overwrite && fs::exists(opts.out_dir, ec) &&
      !fs::is_empty(opts.out_dir, ec)) {
    *error = "out dir exists and is not empty: " + opts.out_dir +
             " (pass --force to overwrite)";
    return false;
  }
  fs::create_directories(opts.out_dir, ec);
  if (ec) {
    *error = "cannot create out dir " + opts.out_dir + ": " + ec.message();
    return false;
  }
  const fs::path out(opts.out_dir);
  if (!plan.kernel_binary.empty()) {
    if (!WriteInc(out / "kernel_binary.inc", plan.kernel_binary, error)) {
      return false;
    }
  }
  int in_idx = 0;
  for (const DevBuffer &b : plan.buffers) {
    if (b.kind == BufKind::kOutput || b.bytes.empty()) {
      continue;
    }
    if (!WriteInc(out / ("input_" + std::to_string(in_idx) + ".inc"),
                  b.bytes, error)) {
      return false;
    }
    ++in_idx;
  }
  if (!WriteFile(out / (plan.vendor == "amd" ? "main.hip" : "main.cu"),
                 EmitMainSource(plan), error)) {
    return false;
  }
  if (!WriteFile(out / "CMakeLists.txt", EmitCmake(plan), error)) {
    return false;
  }
  if (!WriteFile(out / "README.md", EmitReadme(plan), error)) {
    return false;
  }
  return true;
}

} // namespace kernscope
