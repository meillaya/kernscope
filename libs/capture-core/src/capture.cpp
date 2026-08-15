//===- capture.cpp - capture-core implementation --------------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Implements the CaptureSession (interception table, ptr->size allocation
// map, event log), the SHA-256 digest (FIPS 180-4) and the v0 BundleWriter.
// Pure host-side, no CUDA headers - the shim (shim/intercept.cpp) maps CUDA
// types onto the plain integers of capture.h.
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture.h"

#include <array>
#include <cstring>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <ostream>
#include <sstream>
#include <unordered_map>

namespace kernscope {

//===--- SHA-256 (FIPS 180-4) ------------------------------------------------===//
namespace {

class Sha256 {
public:
  void update(std::span<const std::uint8_t> data) {
    for (std::uint8_t b : data) {
      buf_[buf_len_++] = b;
      total_bits_ += 8;
      if (buf_len_ == 64) {
        compress(buf_.data());
        buf_len_ = 0;
      }
    }
  }

  std::array<std::uint8_t, 32> finalize() {
    const std::uint64_t bits = total_bits_;
    // Append 0x80, pad with zeros, append the 64-bit bit length (big-endian).
    buf_[buf_len_++] = 0x80;
    if (buf_len_ > 56) {
      std::memset(buf_.data() + buf_len_, 0, 64 - buf_len_);
      compress(buf_.data());
      buf_len_ = 0;
    }
    std::memset(buf_.data() + buf_len_, 0, 56 - buf_len_);
    for (int i = 0; i < 8; ++i) {
      buf_[63 - i] = static_cast<std::uint8_t>(bits >> (8 * i));
    }
    compress(buf_.data());

    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[4 * i + 0] = static_cast<std::uint8_t>(h_[i] >> 24);
      out[4 * i + 1] = static_cast<std::uint8_t>(h_[i] >> 16);
      out[4 * i + 2] = static_cast<std::uint8_t>(h_[i] >> 8);
      out[4 * i + 3] = static_cast<std::uint8_t>(h_[i]);
    }
    return out;
  }

private:
  static constexpr std::uint32_t Rotr(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
  }

  void compress(const std::uint8_t *block) {
    static constexpr std::array<std::uint32_t, 64> k = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::array<std::uint32_t, 64> w{};
    for (int i = 0; i < 16; ++i) {
      w[i] = (static_cast<std::uint32_t>(block[4 * i]) << 24) |
             (static_cast<std::uint32_t>(block[4 * i + 1]) << 16) |
             (static_cast<std::uint32_t>(block[4 * i + 2]) << 8) |
             static_cast<std::uint32_t>(block[4 * i + 3]);
    }
    for (int i = 16; i < 64; ++i) {
      const std::uint32_t s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^
                               (w[i - 15] >> 3);
      const std::uint32_t s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^
                               (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
      const std::uint32_t s1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
      const std::uint32_t ch = (e & f) ^ (~e & g);
      const std::uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const std::uint32_t s0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
      const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
    h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
  }

  std::array<std::uint32_t, 8> h_ = {
      0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  std::array<std::uint8_t, 64> buf_{};
  std::uint64_t total_bits_ = 0;
  std::size_t buf_len_ = 0;
};

std::string ToHex(const std::array<std::uint8_t, 32> &bytes) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(64);
  for (std::uint8_t b : bytes) {
    out.push_back(kHex[b >> 4]);
    out.push_back(kHex[b & 0xF]);
  }
  return out;
}

} // namespace

std::string sha256_hex(std::span<const std::uint8_t> data) {
  Sha256 s;
  s.update(data);
  return ToHex(s.finalize());
}

//===--- interception table --------------------------------------------------===//

namespace {

struct ApiNameEntry {
  ApiKind kind;
  const char *name;
};

constexpr ApiNameEntry kApiNames[] = {
    {ApiKind::kCudaLaunchKernel, "cudaLaunchKernel"},
    {ApiKind::kCudaLaunchKernelExC, "cudaLaunchKernelExC"},
    {ApiKind::kCudaGraphLaunch, "cudaGraphLaunch"},
    {ApiKind::kCudaMalloc, "cudaMalloc"},
    {ApiKind::kCudaFree, "cudaFree"},
    {ApiKind::kCudaMemcpy, "cudaMemcpy"},
    {ApiKind::kCudaMemcpyAsync, "cudaMemcpyAsync"},
    {ApiKind::kCudaMemcpy2D, "cudaMemcpy2D"},
    {ApiKind::kCudaMemcpy2DAsync, "cudaMemcpy2DAsync"},
    {ApiKind::kCudaMemset, "cudaMemset"},
    {ApiKind::kCudaMemsetAsync, "cudaMemsetAsync"},
    {ApiKind::kCudaStreamCreate, "cudaStreamCreate"},
    {ApiKind::kCudaStreamSynchronize, "cudaStreamSynchronize"},
    {ApiKind::kRegisterFatBinary, "__cudaRegisterFatBinary"},
    {ApiKind::kRegisterFunction, "__cudaRegisterFunction"},
    {ApiKind::kCuModuleLoad, "cuModuleLoad"},
    {ApiKind::kCuModuleLoadData, "cuModuleLoadData"},
    {ApiKind::kCuModuleLoadDataEx, "cuModuleLoadDataEx"},
    {ApiKind::kCuModuleGetFunction, "cuModuleGetFunction"},
    {ApiKind::kCuFuncGetParamInfo, "cuFuncGetParamInfo"},
    {ApiKind::kCuLaunchKernel, "cuLaunchKernel"},
    {ApiKind::kCuLaunchKernelEx, "cuLaunchKernelEx"},
};

} // namespace

const char *api_name(ApiKind kind) {
  for (const auto &e : kApiNames) {
    if (e.kind == kind) {
      return e.name;
    }
  }
  return "?";
}

const char *to_string(LaunchVia via) {
  switch (via) {
  case LaunchVia::kRuntimeApi: return "runtime";
  case LaunchVia::kRuntimeApiEx: return "runtime-ex";
  case LaunchVia::kGraph: return "graph";
  case LaunchVia::kDriverApi: return "driver";
  case LaunchVia::kDriverApiEx: return "driver-ex";
  }
  return "?";
}

//===--- CaptureSession ------------------------------------------------------===//

CaptureSession &CaptureSession::instance() {
  // Leaked on purpose: see the header contract comment on instance().
  static CaptureSession *session = new CaptureSession();
  return *session;
}

CaptureSession::CaptureSession() {
  table_.reserve(std::size(kApiNames));
  for (const auto &e : kApiNames) {
    table_.push_back(ApiTableEntry{e.kind, e.name, 0});
  }
}

void CaptureSession::note_api(ApiKind kind) {
  std::lock_guard<std::mutex> lock(mu_);
  for (auto &entry : table_) {
    if (entry.kind == kind) {
      ++entry.calls;
      return;
    }
  }
}

void CaptureSession::record_alloc(std::uint64_t ptr, std::size_t size) {
  std::lock_guard<std::mutex> lock(mu_);
  allocations_[ptr] = size;
  allocs_.push_back(AllocRecord{ptr, size, true, ++seq_});
}

void CaptureSession::record_free(std::uint64_t ptr, bool forwarded) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    // QA negative path: never crash on an unknown pointer (cudaFree contract:
    // freeing a pointer cudaMalloc did not return is undefined -> we record
    // and forward).
    unknown_frees_.push_back(UnknownFreeRecord{ptr, forwarded, ++seq_});
    return;
  }
  const std::size_t size = it->second;
  allocations_.erase(it);
  allocs_.push_back(AllocRecord{ptr, size, false, ++seq_});
}

void CaptureSession::record_copy(const CopyRecord &rec) {
  std::lock_guard<std::mutex> lock(mu_);
  CopyRecord r = rec;
  r.seq = ++seq_;
  const auto dst = allocations_.find(r.dst);
  const auto src = allocations_.find(r.src);
  // cudaMemcpyKind (CUDA 12.6 driver_types.h): 1 = HostToDevice, 2 =
  // DeviceToHost, 3 = DeviceToDevice - the device-side pointer is `dst` for
  // kinds 1/3 and `src` for kinds 2/3; attribute sizes from the allocation
  // map where the device pointer is known (0 = unknown ptr).
  if (r.cuda_kind == 1 || r.cuda_kind == 3) {
    r.dst_size = dst == allocations_.end() ? 0 : dst->second;
  }
  if (r.cuda_kind == 2 || r.cuda_kind == 3) {
    r.src_size = src == allocations_.end() ? 0 : src->second;
  }
  copies_.push_back(std::move(r));
}

void CaptureSession::record_memset(const MemsetRecord &rec) {
  std::lock_guard<std::mutex> lock(mu_);
  MemsetRecord r = rec;
  r.seq = ++seq_;
  const auto it = allocations_.find(r.ptr);
  r.ptr_size = it == allocations_.end() ? 0 : it->second;
  memsets_.push_back(std::move(r));
}

std::uint64_t CaptureSession::record_stream_create(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = stream_ids_.find(handle);
  if (it != stream_ids_.end()) {
    streams_.push_back(StreamRecord{handle, it->second, false, ++seq_});
    return it->second;
  }
  const std::uint64_t id = stream_ids_.size() + 1;
  stream_ids_[handle] = id;
  streams_.push_back(StreamRecord{handle, id, false, ++seq_});
  return id;
}

void CaptureSession::record_stream_sync(std::uint64_t handle) {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = stream_ids_.find(handle);
  streams_.push_back(
      StreamRecord{handle, it == stream_ids_.end() ? 0 : it->second, true,
                   ++seq_});
}

std::uint64_t CaptureSession::record_launch(LaunchRecord rec) {
  std::lock_guard<std::mutex> lock(mu_);
  rec.seq = ++seq_;
  launches_.push_back(std::move(rec));
  return launches_.back().seq;
}

std::uint64_t CaptureSession::register_fatbinary(
    std::span<const std::uint8_t> blob) {
  std::lock_guard<std::mutex> lock(mu_);
  const std::uint64_t id = modules_.size() + 1;
  modules_.push_back(
      ModuleRecord{ModuleRecord::Kind::kFatBinary, id, sha256_hex(blob), "",
                   {}, ++seq_});
  return id;
}

bool CaptureSession::register_function(std::uint64_t reg_id,
                                       std::string_view kernel_name) {
  std::lock_guard<std::mutex> lock(mu_);
  if (reg_id == 0 || reg_id > modules_.size()) {
    return false;
  }
  ModuleRecord &mod = modules_[reg_id - 1];
  if (mod.kind != ModuleRecord::Kind::kFatBinary) {
    return false;
  }
  mod.kernels.push_back(std::string(kernel_name));
  return true;
}

std::uint64_t CaptureSession::register_cubin(
    std::span<const std::uint8_t> image) {
  std::lock_guard<std::mutex> lock(mu_);
  const std::uint64_t id = modules_.size() + 1;
  modules_.push_back(
      ModuleRecord{ModuleRecord::Kind::kCubin, id, sha256_hex(image), "",
                   {}, ++seq_});
  return id;
}

std::uint64_t CaptureSession::register_module_file(std::string_view path) {
  std::lock_guard<std::mutex> lock(mu_);
  const std::uint64_t id = modules_.size() + 1;
  modules_.push_back(ModuleRecord{ModuleRecord::Kind::kCubin, id, "", std::string(path),
                                  {}, ++seq_});
  return id;
}

void CaptureSession::record_param_info(std::uint64_t cufunction,
                                       std::size_t param_index,
                                       std::size_t offset, std::size_t size) {
  std::lock_guard<std::mutex> lock(mu_);
  auto &list = param_infos_[cufunction];
  if (param_index >= list.size()) {
    list.resize(param_index + 1);
  }
  list[param_index] = ParamInfo{offset, size};
}

std::optional<std::size_t> CaptureSession::param_size(
    std::uint64_t cufunction, std::size_t param_index) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = param_infos_.find(cufunction);
  if (it == param_infos_.end() || param_index >= it->second.size()) {
    return std::nullopt;
  }
  const ParamInfo &p = it->second[param_index];
  return p.size == 0 ? std::nullopt : std::optional<std::size_t>(p.size);
}

const std::unordered_map<std::uint64_t, std::vector<ParamInfo>> &
CaptureSession::param_infos() const {
  return param_infos_;
}

void CaptureSession::register_host_function(std::uint64_t host_fun,
                                            std::string_view kernel_name) {
  std::lock_guard<std::mutex> lock(mu_);
  function_names_[host_fun] = std::string(kernel_name);
}

void CaptureSession::register_cufunction(std::uint64_t cufunction,
                                         std::string_view kernel_name) {
  std::lock_guard<std::mutex> lock(mu_);
  cufunction_names_[cufunction] = std::string(kernel_name);
}

std::string CaptureSession::kernel_name_for_host_function(
    std::uint64_t host_fun) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = function_names_.find(host_fun);
  return it == function_names_.end() ? std::string{} : it->second;
}

std::string CaptureSession::kernel_name_for_cufunction(
    std::uint64_t cufunction) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = cufunction_names_.find(cufunction);
  return it == cufunction_names_.end() ? std::string{} : it->second;
}

std::optional<std::size_t> CaptureSession::allocation_size(
    std::uint64_t ptr) const {
  std::lock_guard<std::mutex> lock(mu_);
  const auto it = allocations_.find(ptr);
  if (it == allocations_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::size_t CaptureSession::allocation_count() const {
  std::lock_guard<std::mutex> lock(mu_);
  return allocations_.size();
}

const std::vector<ApiTableEntry> &CaptureSession::api_table() const {
  return table_;
}

std::uint64_t CaptureSession::api_count(ApiKind kind) const {
  for (const auto &entry : table_) {
    if (entry.kind == kind) {
      return entry.calls;
    }
  }
  return 0;
}

bool CaptureSession::empty() const {
  return launches_.empty() && copies_.empty() && memsets_.empty() &&
         allocs_.empty() && streams_.empty() && unknown_frees_.empty() &&
         modules_.empty();
}

void CaptureSession::clear() {
  std::lock_guard<std::mutex> lock(mu_);
  table_.clear();
  launches_.clear();
  copies_.clear();
  memsets_.clear();
  allocs_.clear();
  streams_.clear();
  unknown_frees_.clear();
  modules_.clear();
  allocations_.clear();
  stream_ids_.clear();
  param_infos_.clear();
  function_names_.clear();
  cufunction_names_.clear();
  seq_ = 0;
}

//===--- v0 bundle writer ----------------------------------------------------===//
// Deterministic line-oriented text so the smoke test can assert records.
// todo 46 replaces this format (JSON schema) behind BundleWriter.

namespace {

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

std::string Hex(std::uint64_t v) {
  std::ostringstream os;
  os << "0x" << std::hex << v;
  return os.str();
}

void WriteLaunch(std::ostream &out, const LaunchRecord &l) {
  out << "launch kernel=\"" << l.kernel_name << "\" via=" << to_string(l.via)
      << " grid=" << l.grid_x << "x" << l.grid_y << "x" << l.grid_z
      << " block=" << l.block_x << "x" << l.block_y << "x" << l.block_z
      << " shared=" << l.shared_mem_bytes << " stream=" << Hex(l.stream)
      << " graph=" << Hex(l.graph_handle) << " args=" << l.args.size();
  for (const KernelArgSlot &slot : l.args) {
    out << " [" << slot.size << ":" << HexBytes(slot.bytes) << "]";
  }
  out << "\n";
}

class V0BundleWriter : public BundleWriter {
public:
  bool write(const CaptureSession &session, std::ostream &out) const override {
    out << "kernscope-capture-v0\n";
    for (const ApiTableEntry &e : session.api_table()) {
      out << "api " << e.name << " calls=" << e.calls << "\n";
    }
    for (const AllocRecord &a : session.allocs()) {
      out << (a.is_alloc ? "alloc" : "free") << " ptr=" << Hex(a.ptr)
          << " size=" << a.size << "\n";
    }
    for (const UnknownFreeRecord &u : session.unknown_frees()) {
      out << "unknown-free ptr=" << Hex(u.ptr)
          << " forwarded=" << (u.forwarded ? "true" : "false") << "\n";
    }
    for (const LaunchRecord &l : session.launches()) {
      WriteLaunch(out, l);
    }
    for (const CopyRecord &c : session.copies()) {
      out << "copy kind=" << c.kind << " dst=" << Hex(c.dst)
          << " src=" << Hex(c.src) << " bytes=" << c.bytes
          << " cuda-kind=" << c.cuda_kind << " dst-size=" << c.dst_size
          << " src-size=" << c.src_size << " stream=" << Hex(c.stream) << "\n";
    }
    for (const MemsetRecord &m : session.memsets()) {
      out << "memset ptr=" << Hex(m.ptr) << " value=" << m.value
          << " bytes=" << m.bytes << " ptr-size=" << m.ptr_size
          << " stream=" << Hex(m.stream) << "\n";
    }
    for (const StreamRecord &s : session.streams()) {
      out << "stream " << (s.is_sync ? "sync" : "create") << " handle="
          << Hex(s.handle) << " id=" << s.id << "\n";
    }
    for (const ModuleRecord &m : session.modules()) {
      out << (m.kind == ModuleRecord::Kind::kFatBinary ? "fatbin" : "cubin")
          << " mod=" << m.handle << " sha256=" << m.sha256;
      if (!m.name.empty()) {
        out << " path=\"" << m.name << "\"";
      }
      if (!m.kernels.empty()) {
        out << " kernels=";
        for (std::size_t i = 0; i < m.kernels.size(); ++i) {
          if (i > 0) {
            out << ",";
          }
          out << "\"" << m.kernels[i] << "\"";
        }
      }
      out << "\n";
    }
    out << "end-capture-v0\n";
    return bool(out);
  }
};

} // namespace

std::unique_ptr<BundleWriter> BundleWriter::create_v0() {
  return std::make_unique<V0BundleWriter>();
}

} // namespace kernscope
