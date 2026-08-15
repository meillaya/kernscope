//===- amdgpu_metadata.cpp - HSACO kernel argument layout ---------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Minimal, self-contained msgpack + ELF-note reader scoped to exactly what
// the AMDGPU metadata note contains. Every format assumption below is
// verified against the todo-5 layernorm_fused HSACO on ROCm 7.2.3; unparsable
// input degrades to an empty result (the caller treats arg capture as
// best-effort and never crashes).
//
//===----------------------------------------------------------------------===//

#include "amdgpu_metadata.h"

#include <cstring>

#include <span>

namespace kernscope {
namespace {

//===--- minimal msgpack reader ---------------------------------------------===//
// Values: nil | bool | int | float | str | bin | array | map. The AMDGPU
// metadata uses fixint/int64 (offsets, sizes), fixstr/str8 (names, value
// kinds), fixarray/array16 (kernels, args, versions) and fixmap/map16.

struct MpValue {
  enum class Type : std::uint8_t { kNil, kBool, kInt, kFloat, kStr, kBin, kArray, kMap };
  Type type = Type::kNil;
  bool b = false;
  std::int64_t i = 0;
  double f = 0.0;
  std::string_view str;
  std::vector<MpValue> items;   // array items
  std::vector<MpValue> entries; // map entries, flattened k1,v1,k2,v2,...

  bool is_str(std::string_view s) const {
    return type == Type::kStr && str == s;
  }
};

class MpReader {
public:
  explicit MpReader(std::span<const std::uint8_t> data) : p_(data.data()), end_(data.data() + data.size()) {}

  bool ok() const { return ok_; }

  MpValue parse() {
    if (p_ >= end_) {
      ok_ = false;
      return {};
    }
    const std::uint8_t b = *p_++;
    if (b <= 0x7F) { return Int(b); }
    if (b >= 0xE0) { return Int(static_cast<std::int64_t>(b) - 0x100); }
    if (b >= 0x80 && b <= 0x8F) { return Map(b & 0x0F); }
    if (b >= 0x90 && b <= 0x9F) { return Array(b & 0x0F); }
    if (b >= 0xA0 && b <= 0xBF) { return Str(Raw(b & 0x1F)); }
    switch (b) {
    case 0xC0: return {};                       // nil
    case 0xC2: return Bool(false);
    case 0xC3: return Bool(true);
    case 0xCC: return Int(Read<std::uint8_t>());
    case 0xCD: return Int(Read<std::uint16_t>());
    case 0xCE: return Int(Read<std::uint32_t>());
    case 0xCF: return Int(static_cast<std::int64_t>(Read<std::uint64_t>()));
    case 0xD0: return Int(static_cast<std::int64_t>(static_cast<std::int8_t>(Read<std::uint8_t>())));
    case 0xD1: return Int(static_cast<std::int64_t>(static_cast<std::int16_t>(Read<std::uint16_t>())));
    case 0xD2: return Int(static_cast<std::int64_t>(static_cast<std::int32_t>(Read<std::uint32_t>())));
    case 0xD3: return Int(static_cast<std::int64_t>(Read<std::uint64_t>()));
    case 0xCA: return Float(Read<std::uint32_t>());
    case 0xCB: return Float(Read<std::uint64_t>());
    case 0xD9: return Str(Raw(Read<std::uint8_t>()));
    case 0xDA: return Str(Raw(Read<std::uint16_t>()));
    case 0xDB: return Str(Raw(Read<std::uint32_t>()));
    case 0xC4: return Bin(Raw(Read<std::uint8_t>()));
    case 0xC5: return Bin(Raw(Read<std::uint16_t>()));
    case 0xC6: return Bin(Raw(Read<std::uint32_t>()));
    case 0xDC: return Array(Read<std::uint16_t>());
    case 0xDD: return Array(Read<std::uint32_t>());
    case 0xDE: return Map(Read<std::uint16_t>());
    case 0xDF: return Map(Read<std::uint32_t>());
    default:
      ok_ = false; // unsupported marker; degrade
      return {};
    }
  }

private:
  MpValue Int(std::int64_t v) { MpValue r; r.type = MpValue::Type::kInt; r.i = v; return r; }
  MpValue Bool(bool v) { MpValue r; r.type = MpValue::Type::kBool; r.b = v; return r; }
  MpValue Float(std::uint64_t raw) {
    MpValue r; r.type = MpValue::Type::kFloat;
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::memcpy(&r.f, &raw, sizeof(raw));
    return r;
  }
  MpValue Str(std::span<const std::uint8_t> s) {
    MpValue r; r.type = MpValue::Type::kStr;
    r.str = std::string_view(reinterpret_cast<const char *>(s.data()), s.size());
    return r;
  }
  MpValue Bin(std::span<const std::uint8_t> s) {
    MpValue r; r.type = MpValue::Type::kBin;
    r.str = std::string_view(reinterpret_cast<const char *>(s.data()), s.size());
    return r;
  }
  MpValue Array(std::size_t n) {
    MpValue r; r.type = MpValue::Type::kArray;
    r.items.reserve(n);
    for (std::size_t i = 0; i < n && ok_; ++i) {
      r.items.push_back(parse());
    }
    return r;
  }
  MpValue Map(std::size_t n) {
    MpValue r; r.type = MpValue::Type::kMap;
    r.entries.reserve(n * 2);
    for (std::size_t i = 0; i < n && ok_; ++i) {
      r.entries.push_back(parse());
      r.entries.push_back(parse());
    }
    return r;
  }

  template <typename T> std::uint64_t Read() {
    if (end_ - p_ < static_cast<std::ptrdiff_t>(sizeof(T))) {
      ok_ = false;
      return 0;
    }
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < sizeof(T); ++i) {
      v = (v << 8) | p_[i];
    }
    p_ += sizeof(T);
    return v;
  }
  std::span<const std::uint8_t> Raw(std::size_t n) {
    if (static_cast<std::size_t>(end_ - p_) < n) {
      ok_ = false;
      return {};
    }
    std::span<const std::uint8_t> s(p_, n);
    p_ += n;
    return s;
  }

  const std::uint8_t *p_;
  const std::uint8_t *end_;
  bool ok_ = true;
};

//===--- ELF64 note walker ---------------------------------------------------===//
// Finds the NT_AMD_AMDGPU_METADATA (type 7) note in the ".note" section.

const MpValue *FindMapEntry(const MpValue &map, std::string_view key) {
  if (map.type != MpValue::Type::kMap) {
    return nullptr;
  }
  for (std::size_t i = 0; i + 1 < map.entries.size(); i += 2) {
    if (map.entries[i].is_str(key)) {
      return &map.entries[i + 1];
    }
  }
  return nullptr;
}

std::span<const std::uint8_t> FindMetadataNote(std::span<const std::uint8_t> elf) {
  constexpr std::size_t kEhdrSize = 64;
  constexpr std::uint32_t kElf64 = 2;
  constexpr std::uint32_t kShtNote = 7;
  constexpr std::uint32_t kNoteAmdAmdgpuMetadata = 32; // NT_AMDGPU_METADATA
  if (elf.size() < kEhdrSize) {
    return {};
  }
  // ELF64 little-endian header: e_ident[EI_CLASS] at 4, e_shoff at 40,
  // e_shentsize at 58, e_shnum at 60, e_shstrndx at 62.
  if (elf[4] != kElf64) {
    return {};
  }
  auto rd32 = [&](std::size_t off) -> std::uint32_t {
    if (off + 4 > elf.size()) {
      return 0;
    }
    return static_cast<std::uint32_t>(elf[off]) |
           (static_cast<std::uint32_t>(elf[off + 1]) << 8) |
           (static_cast<std::uint32_t>(elf[off + 2]) << 16) |
           (static_cast<std::uint32_t>(elf[off + 3]) << 24);
  };
  auto rd64 = [&](std::size_t off) -> std::uint64_t {
    std::uint64_t v = 0;
    for (std::size_t i = 0; i < 8 && off + i < elf.size(); ++i) {
      v |= static_cast<std::uint64_t>(elf[off + i]) << (8 * i);
    }
    return v;
  };
  const std::uint64_t shoff = rd64(40);
  const std::uint16_t shentsize = static_cast<std::uint16_t>(rd32(58) & 0xFFFF);
  const std::uint16_t shnum = static_cast<std::uint16_t>(rd32(60) & 0xFFFF);
  const std::uint16_t shstrndx = static_cast<std::uint16_t>(rd32(62) & 0xFFFF);
  if (shoff == 0 || shentsize < 64 || shnum == 0 || shstrndx >= shnum) {
    return {};
  }
  const std::size_t shstr_off =
      static_cast<std::size_t>(shoff) + static_cast<std::size_t>(shstrndx) * shentsize;
  const std::uint64_t shstr_offset = rd64(shstr_off + 24);
  const std::uint64_t shstr_size = rd64(shstr_off + 32);
  auto section_name = [&](std::uint64_t name_off) -> std::string_view {
    if (name_off >= shstr_size) {
      return {};
    }
    const std::size_t start = static_cast<std::size_t>(shstr_offset + name_off);
    if (start >= elf.size()) {
      return {};
    }
    std::size_t len = 0;
    while (start + len < elf.size() && elf[start + len] != 0) {
      ++len;
    }
    return std::string_view(reinterpret_cast<const char *>(elf.data() + start), len);
  };
  for (std::uint16_t i = 0; i < shnum; ++i) {
    const std::size_t sh = static_cast<std::size_t>(shoff) + static_cast<std::size_t>(i) * shentsize;
    if (sh + 64 > elf.size()) {
      break;
    }
    if (rd32(sh + 4) != kShtNote) {
      continue;
    }
    if (section_name(rd32(sh)) != ".note") {
      continue;
    }
    const std::uint64_t offset = rd64(sh + 24);
    const std::uint64_t size = rd64(sh + 32);
    if (offset + size > elf.size()) {
      return {};
    }
    const std::uint8_t *note = elf.data() + offset;
    const std::uint8_t *note_end = note + size;
    while (note + 12 <= note_end) {
      const auto namesz = static_cast<std::uint32_t>(note[0]) |
                          (static_cast<std::uint32_t>(note[1]) << 8) |
                          (static_cast<std::uint32_t>(note[2]) << 16) |
                          (static_cast<std::uint32_t>(note[3]) << 24);
      const auto descsz = static_cast<std::uint32_t>(note[4]) |
                          (static_cast<std::uint32_t>(note[5]) << 8) |
                          (static_cast<std::uint32_t>(note[6]) << 16) |
                          (static_cast<std::uint32_t>(note[7]) << 24);
      const auto ntype = static_cast<std::uint32_t>(note[8]) |
                         (static_cast<std::uint32_t>(note[9]) << 8) |
                         (static_cast<std::uint32_t>(note[10]) << 16) |
                         (static_cast<std::uint32_t>(note[11]) << 24);
      const std::uint32_t name_pad = (namesz + 3) & ~3u;
      const std::uint32_t desc_pad = (descsz + 3) & ~3u;
      const std::uint8_t *name = note + 12;
      const std::uint8_t *desc = name + name_pad;
      const std::uint8_t *next = desc + desc_pad;
      if (next > note_end) {
        return {};
      }
      if (ntype == kNoteAmdAmdgpuMetadata &&
          namesz >= 6 &&
          std::memcmp(name, "AMDGPU", 6) == 0) {
        return std::span<const std::uint8_t>(desc, descsz);
      }
      note = next;
    }
    return {};
  }
  return {};
}

} // namespace

std::vector<AmdKernelLayout> parse_amdgpu_metadata(
    std::span<const std::uint8_t> hsaco) {
  std::vector<AmdKernelLayout> out;
  const std::span<const std::uint8_t> note = FindMetadataNote(hsaco);
  if (note.empty()) {
    return out;
  }
  MpReader reader(note);
  const MpValue root = reader.parse();
  if (!reader.ok() || root.type != MpValue::Type::kMap) {
    return out;
  }
  const MpValue *kernels = FindMapEntry(root, "amdhsa.kernels");
  if (kernels == nullptr || kernels->type != MpValue::Type::kArray) {
    return out;
  }
  for (const MpValue &kernel : kernels->items) {
    if (kernel.type != MpValue::Type::kMap) {
      continue;
    }
    const MpValue *name = FindMapEntry(kernel, ".name");
    if (name == nullptr || name->type != MpValue::Type::kStr) {
      continue;
    }
    AmdKernelLayout layout;
    layout.name.assign(name->str);
    const MpValue *args = FindMapEntry(kernel, ".args");
    if (args != nullptr && args->type == MpValue::Type::kArray) {
      layout.args.reserve(args->items.size());
      for (const MpValue &arg : args->items) {
        if (arg.type != MpValue::Type::kMap) {
          continue;
        }
        AmdArgLayout a;
        if (const MpValue *off = FindMapEntry(arg, ".offset");
            off != nullptr && off->type == MpValue::Type::kInt) {
          a.offset = static_cast<std::uint64_t>(off->i);
        }
        if (const MpValue *sz = FindMapEntry(arg, ".size");
            sz != nullptr && sz->type == MpValue::Type::kInt) {
          a.size = static_cast<std::uint64_t>(sz->i);
        }
        if (const MpValue *vk = FindMapEntry(arg, ".value_kind");
            vk != nullptr && vk->type == MpValue::Type::kStr) {
          a.value_kind.assign(vk->str);
          // Address-valued args (global buffers, hidden pointer args); their
          // pointee must be in the allocation map for the capture to know the
          // buffer size (the QA negative path marks them size-unknown).
          a.is_pointer = a.value_kind.rfind("hidden_", 0) == 0 ||
                         a.value_kind == "global_buffer";
        }
        layout.args.push_back(std::move(a));
      }
    }
    out.push_back(std::move(layout));
  }
  return out;
}

} // namespace kernscope
