//===- gpu.cpp - CUDA/HIP runtime abstraction (single source) ----*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libs/gpuruntime implementation. ONE translation unit, backend selected by
// the portable #ifdef convention (pattern proven in
// /home/mei/projects/polykernel/lib/Runtime/DeviceDetect.cpp):
//   * __HIP_PLATFORM_AMD__  defined (hipcc, or clang++ with the CMake
//     target definition)  -> HIP backend (hip/hip_runtime_api.h).
//   * KERNSCOPE_BACKEND_CUDA defined (plain C++ with CUDA_HOME headers) ->
//     CUDA backend (cuda.h driver API + cuda_runtime_api.h).
//   * neither               -> compile-only (no backend headers at all).
// The backend headers are included HERE only, so the GPU-free gtest never
// needs ROCm/CUDA headers on its include path.
//
// Thinness contract: only what the capture/profile tracks need (todo 21) -
// device query, module/kernel load+launch, events, malloc/free, memcpy/memset.
// No streams/async, no texture/surface, no graphs: deliberately out of scope.
//
//===----------------------------------------------------------------------===//

#include "kernscope/gpu.h"

#include <cstring>
#include <sstream>

//===--- backend selection --------------------------------------------------===//
#if defined(__HIP_PLATFORM_AMD__) || defined(KERNSCOPE_HIP_BACKEND_BUILD)
#define KERNSCOPE_HAVE_HIP 1
#include <hip/hip_runtime_api.h>
#elif defined(KERNSCOPE_BACKEND_CUDA)
#define KERNSCOPE_HAVE_CUDA 1
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

namespace kernscope {

//===--- typed error model --------------------------------------------------===//

std::string_view to_string(GpuError err) noexcept {
  switch (err) {
  case GpuError::kOk: return "ok";
  case GpuError::kNoDevice: return "no-device";
  case GpuError::kNoDriver: return "no-driver";
  case GpuError::kModuleLoadFailed: return "module-load-failed";
  case GpuError::kKernelNotFound: return "kernel-not-found";
  case GpuError::kLaunchFailed: return "launch-failed";
  case GpuError::kMemoryOpFailed: return "memory-op-failed";
  case GpuError::kCopyOpFailed: return "copy-op-failed";
  case GpuError::kEventOpFailed: return "event-op-failed";
  case GpuError::kDeviceNotFound: return "device-not-found";
  case GpuError::kUnsupportedBackend: return "unsupported-backend";
  case GpuError::kOther: return "other";
  }
  return "unknown";
}

namespace {

//===--- helpers -------------------------------------------------------------===//

/// Bare arch token of a gcnArchName: the "gfx<digits>[<letter>]" run up to the
/// first non-alphanumeric (ROCm appends flags as "gfx1101:sramecc+:xnack-").
/// Returns "" when the string has no leading gfx<digit> arch.
#if defined(KERNSCOPE_HAVE_HIP)
std::string ParseGcnArch(std::string_view arch) {
  const std::size_t pos = arch.find("gfx");
  if (pos == std::string_view::npos) {
    return {};
  }
  std::size_t end = pos + 3;
  if (end >= arch.size() || arch[end] < '0' || arch[end] > '9') {
    return {}; // "gfx" not followed by a digit is not an arch token.
  }
  while (end < arch.size()) {
    const char c = arch[end];
    const bool alnum = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'z');
    if (!alnum) {
      break;
    }
    ++end;
  }
  return std::string(arch.substr(pos, end - pos));
}
#endif // KERNSCOPE_HAVE_HIP

#if defined(KERNSCOPE_HAVE_CUDA)
std::string SmArch(int major, int minor) {
  return "sm_" + std::to_string(major) + std::to_string(minor);
}
#endif // KERNSCOPE_HAVE_CUDA

} // namespace

//===--- Mock backend --------------------------------------------------------===//
// Unit-test backend: records every op into an inspectable log and retains
// BYTE-EXACT buffer contents for every captured allocation (review-mandated:
// todo 44's cross-vendor equivalence test asserts exact input-blob bytes).
// Mock semantics are fixed and documented; test expectations cite the CUDA/HIP
// API contract docs (Metis M-5.i) - e.g. cudaMemcpy copies the exact byte
// sequence of the source region (CUDA Runtime API, memcpy section) - so the
// mock is validated against the documented contract, never against itself.

namespace {

class MockModule : public Module {
public:
  explicit MockModule(std::string path) : path_(std::move(path)) {}
  const std::string &path() const noexcept { return path_; }

private:
  std::string path_;
};

class MockKernel : public Kernel {
public:
  MockKernel(MockRuntime &rt, std::string name) : rt_(rt), name_(std::move(name)) {}

  GpuStatus launch(const LaunchConfig &cfg, std::span<const KernelArg> args) override {
    std::vector<std::uint8_t> blob;
    std::size_t total = 0;
    for (const KernelArg &a : args) {
      total += a.size;
      if (a.size > 0 && a.data != nullptr) {
        const auto *bytes = static_cast<const std::uint8_t *>(a.data);
        blob.insert(blob.end(), bytes, bytes + a.size);
      } else {
        blob.insert(blob.end(), a.size, std::uint8_t{0});
      }
    }
    // The mock cannot execute kernels; per the CUDA/HIP launch contract the
    // argument blob must reach the kernel byte-exactly (CUDA Runtime API,
    // cuLaunchKernel kernelParams) - we record it for byte-exact assertions.
    rt_.record(MockRuntime::Op::Type::kLaunch, name_, total, cfg, std::move(blob));
    return {};
  }

private:
  MockRuntime &rt_;
  std::string name_;
};

class MockEvent : public Event {
public:
  explicit MockEvent(MockRuntime &rt) : rt_(rt) {}

  GpuStatus record() override {
    rt_.record(MockRuntime::Op::Type::kEventRecord, "event");
    return {};
  }
  GpuStatus synchronize() override {
    rt_.record(MockRuntime::Op::Type::kEventSynchronize, "event");
    return {};
  }
  GpuResult<double> elapsed_ms(const Event &) const override {
    rt_.record(MockRuntime::Op::Type::kEventElapsed, "event");
    // Deterministic mock timing: the >0 contract is asserted against the live
    // backend (gpu-amd suite); the mock pins a fixed value so elapsed()-based
    // logic is reproducible in unit tests.
    return GpuResult<double>{0.5};
  }

private:
  MockRuntime &rt_;
};

} // namespace

void MockRuntime::record(Op::Type type, std::string detail, std::size_t size,
                         LaunchConfig cfg, std::vector<std::uint8_t> payload) {
  ops_.push_back(Op{type, std::move(detail), size, cfg, std::move(payload)});
}

GpuResult<MockRuntime::Buffer *> MockRuntime::find_buffer(DevicePtr ptr) noexcept {
  for (auto &kv : buffers_) {
    if (kv.first == ptr) {
      return GpuResult<Buffer *>{&kv.second};
    }
  }
  return GpuResult<Buffer *>{GpuError::kMemoryOpFailed,
                             "mock: unknown device handle"};
}

std::span<const std::uint8_t> MockRuntime::buffer(DevicePtr ptr) const noexcept {
  for (const auto &kv : buffers_) {
    if (kv.first == ptr) {
      return kv.second.bytes;
    }
  }
  return {};
}

GpuResult<std::vector<DeviceInfo>> MockRuntime::list_devices() {
  record(Op::Type::kDeviceQuery, "list_devices");
  DeviceInfo dev;
  dev.name = "mock-gpu-0";
  dev.arch = "gfx1101";
  dev.memory_bytes = std::uint64_t{64} << 20; // 64 MiB
  dev.vendor = "mock";
  return GpuResult<std::vector<DeviceInfo>>{{dev}};
}

GpuResult<DeviceProperties> MockRuntime::device_properties(const std::string &name) {
  record(Op::Type::kDeviceQuery, "device_properties:" + name);
  if (name != "mock-gpu-0") {
    return GpuResult<DeviceProperties>{GpuError::kDeviceNotFound,
                                       "mock: no device named '" + name + "'"};
  }
  DeviceProperties props;
  props.name = name;
  props.gcn_arch = "gfx1101";
  props.gcn_arch_full = "gfx1101:sramecc+:xnack-";
  props.memory_bytes = std::uint64_t{64} << 20;
  props.vendor = "mock";
  return GpuResult<DeviceProperties>{std::move(props)};
}

GpuResult<std::unique_ptr<Module>> MockRuntime::load_module(const std::string &path) {
  // The mock has no file system: loading is recorded and always succeeds (the
  // file-reading contract of hipModuleLoad/cuModuleLoad is exercised by the
  // live backend; the mock's job is op-order + byte-exactness).
  record(Op::Type::kModuleLoad, path);
  return GpuResult<std::unique_ptr<Module>>{std::make_unique<MockModule>(path)};
}

GpuResult<std::unique_ptr<Kernel>> MockRuntime::create_kernel(Module &module, const std::string &name) {
  record(Op::Type::kKernelCreate, name);
  (void)module; // mock accepts any name (no fat-binary lookup)
  return GpuResult<std::unique_ptr<Kernel>>{std::make_unique<MockKernel>(*this, name)};
}

GpuResult<DevicePtr> MockRuntime::device_malloc(std::size_t bytes) {
  record(Op::Type::kMalloc, "", bytes);
  const DevicePtr handle{next_handle_++};
  buffers_.push_back({handle, Buffer{std::vector<std::uint8_t>(bytes, 0)}});
  return GpuResult<DevicePtr>{handle};
}

GpuStatus MockRuntime::device_free(DevicePtr ptr) {
  record(Op::Type::kFree, "", 0);
  for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
    if (it->first == ptr) {
      buffers_.erase(it);
      return {};
    }
  }
  return GpuStatus{GpuError::kMemoryOpFailed, "mock: free of unknown handle"};
}

GpuStatus MockRuntime::memcpy_htod(DevicePtr dst, const void *src, std::size_t bytes) {
  auto found = find_buffer(dst);
  if (!found.ok()) {
    return GpuStatus{found.err, found.message};
  }
  if (src == nullptr) {
    return GpuStatus{GpuError::kCopyOpFailed, "mock: null source"};
  }
  if (bytes > found.value->bytes.size()) {
    return GpuStatus{GpuError::kCopyOpFailed,
                     "mock: htod of " + std::to_string(bytes) + " bytes into a " +
                         std::to_string(found.value->bytes.size()) + "-byte buffer"};
  }
  // Byte-exact copy (cudaMemcpy contract: the destination receives the exact
  // byte sequence of the source region).
  std::memcpy(found.value->bytes.data(), src, bytes);
  const auto *raw = static_cast<const std::uint8_t *>(src);
  record(Op::Type::kMemcpyHtoD, "", bytes, {},
         std::vector<std::uint8_t>(raw, raw + bytes));
  return {};
}

GpuStatus MockRuntime::memcpy_dtoh(void *dst, DevicePtr src, std::size_t bytes) {
  auto found = find_buffer(src);
  if (!found.ok()) {
    return GpuStatus{found.err, found.message};
  }
  if (dst == nullptr) {
    return GpuStatus{GpuError::kCopyOpFailed, "mock: null destination"};
  }
  if (bytes > found.value->bytes.size()) {
    return GpuStatus{GpuError::kCopyOpFailed,
                     "mock: dtoh of " + std::to_string(bytes) + " bytes from a " +
                         std::to_string(found.value->bytes.size()) + "-byte buffer"};
  }
  std::memcpy(dst, found.value->bytes.data(), bytes);
  record(Op::Type::kMemcpyDtoH, "", bytes, {},
         std::vector<std::uint8_t>(found.value->bytes.begin(),
                                   found.value->bytes.begin() + bytes));
  return {};
}

GpuStatus MockRuntime::memset_device(DevicePtr dst, std::uint8_t value, std::size_t bytes) {
  auto found = find_buffer(dst);
  if (!found.ok()) {
    return GpuStatus{found.err, found.message};
  }
  if (bytes > found.value->bytes.size()) {
    return GpuStatus{GpuError::kMemoryOpFailed,
                     "mock: memset of " + std::to_string(bytes) + " bytes into a " +
                         std::to_string(found.value->bytes.size()) + "-byte buffer"};
  }
  // Byte-exact fill (cudaMemset contract); contents are inspected via
  // MockRuntime::buffer(), which always returns the retained bytes.
  std::memset(found.value->bytes.data(), value, bytes);
  record(Op::Type::kMemset, "", bytes);
  return {};
}

GpuResult<std::unique_ptr<Event>> MockRuntime::create_event() {
  return GpuResult<std::unique_ptr<Event>>{std::make_unique<MockEvent>(*this)};
}

GpuStatus MockRuntime::synchronize() {
  record(Op::Type::kSynchronize, "");
  return {};
}

//===--- CompileOnly backend -------------------------------------------------===//
// Deterministic no-GPU runtime: every call returns a typed GpuError::kNoDevice
// result (never crashes), so the unit suite runs on machines with no GPU or
// driver at all. On this AMD-only machine the CUDA driver is also absent, so
// the CUDA path behaves exactly like this through the driver's own typed
// errors (cudaErrorNoDevice -> kNoDevice).

namespace {

class CompileOnlyRuntime : public GpuRuntime {
public:
  std::string_view backend_name() const noexcept override { return "compile-only"; }

  static GpuStatus NoDevice(std::string_view op) {
    return GpuStatus{GpuError::kNoDevice,
                     std::string("compile-only runtime: no GPU/driver available (") +
                         std::string(op) + ")"};
  }

  GpuResult<std::vector<DeviceInfo>> list_devices() override {
    return GpuResult<std::vector<DeviceInfo>>{NoDevice("list_devices")};
  }
  GpuResult<DeviceProperties> device_properties(const std::string &) override {
    return GpuResult<DeviceProperties>{NoDevice("device_properties")};
  }
  GpuResult<std::unique_ptr<Module>> load_module(const std::string &) override {
    return GpuResult<std::unique_ptr<Module>>{NoDevice("load_module")};
  }
  GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module &, const std::string &) override {
    return GpuResult<std::unique_ptr<Kernel>>{NoDevice("create_kernel")};
  }
  GpuResult<DevicePtr> device_malloc(std::size_t) override {
    return GpuResult<DevicePtr>{NoDevice("device_malloc")};
  }
  GpuStatus device_free(DevicePtr) override { return NoDevice("device_free"); }
  GpuStatus memcpy_htod(DevicePtr, const void *, std::size_t) override {
    return NoDevice("memcpy_htod");
  }
  GpuStatus memcpy_dtoh(void *, DevicePtr, std::size_t) override {
    return NoDevice("memcpy_dtoh");
  }
  GpuStatus memset_device(DevicePtr, std::uint8_t, std::size_t) override {
    return NoDevice("memset_device");
  }
  GpuResult<std::unique_ptr<Event>> create_event() override {
    return GpuResult<std::unique_ptr<Event>>{NoDevice("create_event")};
  }
  GpuStatus synchronize() override { return NoDevice("synchronize"); }
};

} // namespace

//===--- Real backends (CUDA / HIP) -------------------------------------------===//
// Exactly one of these translation units is active per build (the #ifdef at
// the top of the file); on a machine whose driver is absent the driver's own
// typed errors (cudaErrorNoDevice / hipErrorNoDevice) map to kNoDevice, so the
// real backends degrade to compile-only behavior without any special casing.

#if defined(KERNSCOPE_HAVE_HIP)

namespace {

std::string HipErrorText(hipError_t e) {
  return std::string(hipGetErrorString(e));
}

GpuStatus HipStatus(hipError_t e, std::string_view op) {
  if (e == hipSuccess) {
    return {};
  }
  GpuError err = GpuError::kOther;
  switch (e) {
  case hipErrorNoDevice: err = GpuError::kNoDevice; break;
  case hipErrorNoBinaryForGpu: err = GpuError::kNoDevice; break;
  case hipErrorInsufficientDriver: err = GpuError::kNoDriver; break;
  case hipErrorInvalidDevice: err = GpuError::kDeviceNotFound; break;
  case hipErrorFileNotFound: err = GpuError::kModuleLoadFailed; break;
  case hipErrorNotFound: err = GpuError::kKernelNotFound; break;
  case hipErrorLaunchFailure: err = GpuError::kLaunchFailed; break;
  case hipErrorOutOfMemory: err = GpuError::kMemoryOpFailed; break;
  case hipErrorInvalidValue: err = GpuError::kOther; break;
  default: err = GpuError::kOther; break;
  }
  return GpuStatus{err, std::string(op) + ": " + HipErrorText(e)};
}

class HipModule : public Module {
public:
  explicit HipModule(hipModule_t m) : handle_(m) {}
  ~HipModule() override {
    if (handle_ != nullptr) {
      (void)hipModuleUnload(handle_);
    }
  }
  hipModule_t handle_ = nullptr;
};

class HipKernel : public Kernel {
public:
  explicit HipKernel(hipFunction_t fn) : fn_(fn) {}
  GpuStatus launch(const LaunchConfig &cfg, std::span<const KernelArg> args) override {
    // Driver-API launch: kernelParams holds a pointer to each parameter's
    // host-side value (HIP API, hipLaunchKernel). The device pointer args are
    // passed BY VALUE (uintptr), matching the kernel signature.
    std::vector<void *> params;
    params.reserve(args.size());
    for (const KernelArg &a : args) {
      params.push_back(const_cast<void *>(a.data));
    }
    const hipError_t e =
        hipLaunchKernel(fn_, dim3(cfg.grid_x, cfg.grid_y, cfg.grid_z),
                        dim3(cfg.block_x, cfg.block_y, cfg.block_z),
                        params.data(), cfg.shared_mem_bytes, nullptr);
    return HipStatus(e, "hipLaunchKernel");
  }

private:
  hipFunction_t fn_ = nullptr;
};

class HipEvent : public Event {
public:
  HipEvent() { (void)hipEventCreate(&handle_); }
  ~HipEvent() override {
    if (handle_ != nullptr) {
      (void)hipEventDestroy(handle_);
    }
  }
  GpuStatus record() override { return HipStatus(hipEventRecord(handle_, nullptr), "hipEventRecord"); }
  GpuStatus synchronize() override { return HipStatus(hipEventSynchronize(handle_), "hipEventSynchronize"); }
  GpuResult<double> elapsed_ms(const Event &other) const override {
    const auto *o = dynamic_cast<const HipEvent *>(&other);
    if (o == nullptr) {
      return GpuResult<double>{GpuError::kEventOpFailed,
                               "elapsed_ms: event from a different runtime"};
    }
    float ms = 0.0f;
    const hipError_t e = hipEventElapsedTime(&ms, handle_, o->handle_);
    if (e != hipSuccess) {
      return GpuResult<double>{HipStatus(e, "hipEventElapsedTime")};
    }
    return GpuResult<double>{static_cast<double>(ms)};
  }

private:
  hipEvent_t handle_ = nullptr;
};

class HipRuntime : public GpuRuntime {
public:
  static bool Live() {
    int count = 0;
    return hipGetDeviceCount(&count) == hipSuccess && count > 0;
  }

  std::string_view backend_name() const noexcept override { return "hip"; }

  GpuResult<std::vector<DeviceInfo>> list_devices() override {
    int count = 0;
    if (hipError_t e = hipGetDeviceCount(&count); e != hipSuccess) {
      return GpuResult<std::vector<DeviceInfo>>{HipStatus(e, "hipGetDeviceCount")};
    }
    if (count <= 0) {
      return GpuResult<std::vector<DeviceInfo>>{GpuError::kNoDevice,
                                                "hipGetDeviceCount: 0 devices"};
    }
    std::vector<DeviceInfo> devs;
    devs.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t prop{};
      if (hipError_t e = hipGetDeviceProperties(&prop, i); e != hipSuccess) {
        return GpuResult<std::vector<DeviceInfo>>{HipStatus(e, "hipGetDeviceProperties")};
      }
      DeviceInfo info;
      info.name = prop.name;
      info.arch = ParseGcnArch(prop.gcnArchName);
      info.memory_bytes = prop.totalGlobalMem;
      info.vendor = "amd";
      devs.push_back(std::move(info));
    }
    return GpuResult<std::vector<DeviceInfo>>{std::move(devs)};
  }

  GpuResult<DeviceProperties> device_properties(const std::string &name) override {
    // Query per-ordinal props and match on the marketing name (same matching
    // rule the CUDA path uses); the prop carries the full gcnArchName.
    int count = 0;
    if (hipError_t e = hipGetDeviceCount(&count); e != hipSuccess) {
      return GpuResult<DeviceProperties>{HipStatus(e, "hipGetDeviceCount")};
    }
    for (int i = 0; i < count; ++i) {
      hipDeviceProp_t prop{};
      if (hipError_t e = hipGetDeviceProperties(&prop, i); e != hipSuccess) {
        return GpuResult<DeviceProperties>{HipStatus(e, "hipGetDeviceProperties")};
      }
      if (std::string(prop.name) != name) {
        continue;
      }
      DeviceProperties props;
      props.name = prop.name;
      props.compute_capability_major = prop.major; // 0 on AMD (CUDA-only field)
      props.compute_capability_minor = prop.minor;
      props.gcn_arch = ParseGcnArch(prop.gcnArchName);
      props.gcn_arch_full = prop.gcnArchName;
      props.memory_bytes = prop.totalGlobalMem;
      props.vendor = "amd";
      return GpuResult<DeviceProperties>{std::move(props)};
    }
    return GpuResult<DeviceProperties>{GpuError::kDeviceNotFound,
                                       "no HIP device named '" + name + "'"};
  }

  GpuResult<std::unique_ptr<Module>> load_module(const std::string &path) override {
    hipModule_t mod = nullptr;
    const hipError_t e = hipModuleLoad(&mod, path.c_str());
    if (e != hipSuccess) {
      return GpuResult<std::unique_ptr<Module>>{HipStatus(e, "hipModuleLoad")};
    }
    return GpuResult<std::unique_ptr<Module>>{std::make_unique<HipModule>(mod)};
  }

  GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module &module, const std::string &name) override {
    auto *m = dynamic_cast<HipModule *>(&module);
    if (m == nullptr) {
      return GpuResult<std::unique_ptr<Kernel>>{GpuError::kOther,
                                                "create_kernel: module from a different runtime"};
    }
    hipFunction_t fn = nullptr;
    const hipError_t e = hipModuleGetFunction(&fn, m->handle_, name.c_str());
    if (e != hipSuccess) {
      return GpuResult<std::unique_ptr<Kernel>>{HipStatus(e, "hipModuleGetFunction")};
    }
    return GpuResult<std::unique_ptr<Kernel>>{std::make_unique<HipKernel>(fn)};
  }

  GpuResult<DevicePtr> device_malloc(std::size_t bytes) override {
    void *p = nullptr;
    const hipError_t e = hipMalloc(&p, bytes);
    if (e != hipSuccess) {
      return GpuResult<DevicePtr>{HipStatus(e, "hipMalloc")};
    }
    return GpuResult<DevicePtr>{DevicePtr{reinterpret_cast<std::uint64_t>(p)}};
  }

  GpuStatus device_free(DevicePtr ptr) override {
    return HipStatus(hipFree(reinterpret_cast<void *>(ptr.value)), "hipFree");
  }

  GpuStatus memcpy_htod(DevicePtr dst, const void *src, std::size_t bytes) override {
    return HipStatus(hipMemcpyHtoD(reinterpret_cast<void *>(dst.value),
                                   const_cast<void *>(src), bytes),
                     "hipMemcpyHtoD");
  }

  GpuStatus memcpy_dtoh(void *dst, DevicePtr src, std::size_t bytes) override {
    return HipStatus(hipMemcpyDtoH(dst, reinterpret_cast<void *>(src.value), bytes),
                     "hipMemcpyDtoH");
  }

  GpuStatus memset_device(DevicePtr dst, std::uint8_t value, std::size_t bytes) override {
    return HipStatus(hipMemset(reinterpret_cast<void *>(dst.value), value, bytes),
                     "hipMemset");
  }

  GpuResult<std::unique_ptr<Event>> create_event() override {
    return GpuResult<std::unique_ptr<Event>>{std::make_unique<HipEvent>()};
  }

  GpuStatus synchronize() override { return HipStatus(hipDeviceSynchronize(), "hipDeviceSynchronize"); }
};

} // namespace

#elif defined(KERNSCOPE_HAVE_CUDA)

namespace {

std::string CudaErrorText(cudaError_t e) {
  return std::string(cudaGetErrorString(e));
}

GpuStatus CudaStatus(cudaError_t e, std::string_view op) {
  if (e == cudaSuccess) {
    return {};
  }
  GpuError err = GpuError::kOther;
  switch (e) {
  case cudaErrorNoDevice: err = GpuError::kNoDevice; break;
  case cudaErrorInsufficientDriver: err = GpuError::kNoDriver; break;
  case cudaErrorInvalidDevice: err = GpuError::kDeviceNotFound; break;
  case cudaErrorFileNotFound: err = GpuError::kModuleLoadFailed; break;
  case cudaErrorLaunchFailure: err = GpuError::kLaunchFailed; break;
  case cudaErrorMemoryAllocation: err = GpuError::kMemoryOpFailed; break;
  default: err = GpuError::kOther; break;
  }
  return GpuStatus{err, std::string(op) + ": " + CudaErrorText(e)};
}

GpuStatus CuStatus(CUresult r, std::string_view op) {
  if (r == CUDA_SUCCESS) {
    return {};
  }
  const char *text = "unknown CUDA driver error";
  cuGetErrorString(r, &text);
  GpuError err = GpuError::kOther;
  switch (r) {
  case CUDA_ERROR_NO_DEVICE: err = GpuError::kNoDevice; break;
  case CUDA_ERROR_SYSTEM_DRIVER_MISMATCH: err = GpuError::kNoDriver; break;
  case CUDA_ERROR_INVALID_DEVICE: err = GpuError::kDeviceNotFound; break;
  case CUDA_ERROR_FILE_NOT_FOUND: err = GpuError::kModuleLoadFailed; break;
  case CUDA_ERROR_NOT_FOUND: err = GpuError::kKernelNotFound; break;
  case CUDA_ERROR_LAUNCH_FAILED: err = GpuError::kLaunchFailed; break;
  case CUDA_ERROR_OUT_OF_MEMORY: err = GpuError::kMemoryOpFailed; break;
  default: err = GpuError::kOther; break;
  }
  return GpuStatus{err, std::string(op) + ": " + std::string(text)};
}

class CudaModule : public Module {
public:
  explicit CudaModule(CUmodule m) : handle_(m) {}
  ~CudaModule() override {
    if (handle_ != nullptr) {
      (void)cuModuleUnload(handle_);
    }
  }
  CUmodule handle_ = nullptr;
};

class CudaKernel : public Kernel {
public:
  explicit CudaKernel(CUfunction fn) : fn_(fn) {}
  GpuStatus launch(const LaunchConfig &cfg, std::span<const KernelArg> args) override {
    // Driver-API launch: kernelParams holds a pointer to each parameter's
    // host-side value (CUDA Driver API, cuLaunchKernel).
    std::vector<void *> params;
    params.reserve(args.size());
    for (const KernelArg &a : args) {
      params.push_back(const_cast<void *>(a.data));
    }
    const CUresult r = cuLaunchKernel(fn_, cfg.grid_x, cfg.grid_y, cfg.grid_z,
                                      cfg.block_x, cfg.block_y, cfg.block_z,
                                      cfg.shared_mem_bytes, nullptr,
                                      params.data(), nullptr);
    return CuStatus(r, "cuLaunchKernel");
  }

private:
  CUfunction fn_ = nullptr;
};

class CudaEvent : public Event {
public:
  CudaEvent() { (void)cudaEventCreate(&handle_); }
  ~CudaEvent() override {
    if (handle_ != nullptr) {
      (void)cudaEventDestroy(handle_);
    }
  }
  GpuStatus record() override { return CudaStatus(cudaEventRecord(handle_, nullptr), "cudaEventRecord"); }
  GpuStatus synchronize() override { return CudaStatus(cudaEventSynchronize(handle_), "cudaEventSynchronize"); }
  GpuResult<double> elapsed_ms(const Event &other) const override {
    const auto *o = dynamic_cast<const CudaEvent *>(&other);
    if (o == nullptr) {
      return GpuResult<double>{GpuError::kEventOpFailed,
                               "elapsed_ms: event from a different runtime"};
    }
    float ms = 0.0f;
    const cudaError_t e = cudaEventElapsedTime(&ms, handle_, o->handle_);
    if (e != cudaSuccess) {
      return GpuResult<double>{CudaStatus(e, "cudaEventElapsedTime")};
    }
    return GpuResult<double>{static_cast<double>(ms)};
  }

private:
  cudaEvent_t handle_ = nullptr;
};

class CudaRuntime : public GpuRuntime {
public:
  static bool Live() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
  }

  std::string_view backend_name() const noexcept override { return "cuda"; }

  GpuResult<std::vector<DeviceInfo>> list_devices() override {
    int count = 0;
    if (cudaError_t e = cudaGetDeviceCount(&count); e != cudaSuccess) {
      return GpuResult<std::vector<DeviceInfo>>{CudaStatus(e, "cudaGetDeviceCount")};
    }
    if (count <= 0) {
      return GpuResult<std::vector<DeviceInfo>>{GpuError::kNoDevice,
                                                "cudaGetDeviceCount: 0 devices"};
    }
    std::vector<DeviceInfo> devs;
    devs.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      cudaDeviceProp prop{};
      if (cudaError_t e = cudaGetDeviceProperties(&prop, i); e != cudaSuccess) {
        return GpuResult<std::vector<DeviceInfo>>{CudaStatus(e, "cudaGetDeviceProperties")};
      }
      DeviceInfo info;
      info.name = prop.name;
      info.arch = SmArch(prop.major, prop.minor);
      info.memory_bytes = prop.totalGlobalMem;
      info.vendor = "nvidia";
      devs.push_back(std::move(info));
    }
    return GpuResult<std::vector<DeviceInfo>>{std::move(devs)};
  }

  GpuResult<DeviceProperties> device_properties(const std::string &name) override {
    auto devs = list_devices();
    if (!devs.ok()) {
      return GpuResult<DeviceProperties>{devs.err, devs.message};
    }
    for (const DeviceInfo &d : devs.value) {
      if (d.name == name) {
        DeviceProperties props;
        props.name = d.name;
        // compute capability is encoded in the arch token "sm_<M><m>".
        const std::string prefix = "sm_";
        if (d.arch.size() == prefix.size() + 2 &&
            d.arch.rfind(prefix, 0) == 0) {
          props.compute_capability_major = d.arch[prefix.size()] - '0';
          props.compute_capability_minor = d.arch[prefix.size() + 1] - '0';
        }
        props.memory_bytes = d.memory_bytes;
        props.vendor = d.vendor;
        return GpuResult<DeviceProperties>{std::move(props)};
      }
    }
    return GpuResult<DeviceProperties>{GpuError::kDeviceNotFound,
                                       "no CUDA device named '" + name + "'"};
  }

  GpuResult<std::unique_ptr<Module>> load_module(const std::string &path) override {
    if (cuInit(0) != CUDA_SUCCESS) {
      return GpuResult<std::unique_ptr<Module>>{GpuError::kNoDriver,
                                                "cuInit: CUDA driver unavailable"};
    }
    CUmodule mod = nullptr;
    const CUresult r = cuModuleLoad(&mod, path.c_str());
    if (r != CUDA_SUCCESS) {
      return GpuResult<std::unique_ptr<Module>>{CuStatus(r, "cuModuleLoad")};
    }
    return GpuResult<std::unique_ptr<Module>>{std::make_unique<CudaModule>(mod)};
  }

  GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module &module, const std::string &name) override {
    auto *m = dynamic_cast<CudaModule *>(&module);
    if (m == nullptr) {
      return GpuResult<std::unique_ptr<Kernel>>{GpuError::kOther,
                                                "create_kernel: module from a different runtime"};
    }
    CUfunction fn = nullptr;
    const CUresult r = cuModuleGetFunction(&fn, m->handle_, name.c_str());
    if (r != CUDA_SUCCESS) {
      return GpuResult<std::unique_ptr<Kernel>>{CuStatus(r, "cuModuleGetFunction")};
    }
    return GpuResult<std::unique_ptr<Kernel>>{std::make_unique<CudaKernel>(fn)};
  }

  GpuResult<DevicePtr> device_malloc(std::size_t bytes) override {
    void *p = nullptr;
    const cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
      return GpuResult<DevicePtr>{CudaStatus(e, "cudaMalloc")};
    }
    return GpuResult<DevicePtr>{DevicePtr{reinterpret_cast<std::uint64_t>(p)}};
  }

  GpuStatus device_free(DevicePtr ptr) override {
    return CudaStatus(cudaFree(reinterpret_cast<void *>(ptr.value)), "cudaFree");
  }

  GpuStatus memcpy_htod(DevicePtr dst, const void *src, std::size_t bytes) override {
    return CudaStatus(cudaMemcpy(reinterpret_cast<void *>(dst.value), src, bytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy(HostToDevice)");
  }

  GpuStatus memcpy_dtoh(void *dst, DevicePtr src, std::size_t bytes) override {
    return CudaStatus(cudaMemcpy(dst, reinterpret_cast<void *>(src.value), bytes,
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy(DeviceToHost)");
  }

  GpuStatus memset_device(DevicePtr dst, std::uint8_t value, std::size_t bytes) override {
    return CudaStatus(cudaMemset(reinterpret_cast<void *>(dst.value), value, bytes),
                      "cudaMemset");
  }

  GpuResult<std::unique_ptr<Event>> create_event() override {
    return GpuResult<std::unique_ptr<Event>>{std::make_unique<CudaEvent>()};
  }

  GpuStatus synchronize() override {
    return CudaStatus(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  }
};

} // namespace

#endif // KERNSCOPE_HAVE_HIP / KERNSCOPE_HAVE_CUDA

//===--- static factories (Module::load / Kernel::create / Event::create) ---===//

GpuResult<std::unique_ptr<Module>> Module::load(GpuRuntime &rt, const std::string &path) {
  return rt.load_module(path);
}

GpuResult<std::unique_ptr<Kernel>> Kernel::create(GpuRuntime &rt, Module &module,
                                                  const std::string &name) {
  return rt.create_kernel(module, name);
}

GpuResult<std::unique_ptr<Event>> Event::create(GpuRuntime &rt) {
  return rt.create_event();
}

//===--- factory --------------------------------------------------------------===//

GpuResult<std::unique_ptr<GpuRuntime>> create_runtime(Backend backend) {
  switch (backend) {
  case Backend::kMock:
    return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<MockRuntime>()};
  case Backend::kCompileOnly:
    return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<CompileOnlyRuntime>()};
  case Backend::kHip:
#if defined(KERNSCOPE_HAVE_HIP)
    if (HipRuntime::Live()) {
      return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<HipRuntime>()};
    }
    return GpuResult<std::unique_ptr<GpuRuntime>>{
        GpuError::kNoDevice, "hip: no live HIP device (hipGetDeviceCount)"};
#else
    return GpuResult<std::unique_ptr<GpuRuntime>>{
        GpuError::kUnsupportedBackend,
        "hip backend not compiled into this build (no ROCm headers at "
        "configure time)"};
#endif
  case Backend::kCuda:
#if defined(KERNSCOPE_HAVE_CUDA)
    if (CudaRuntime::Live()) {
      return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<CudaRuntime>()};
    }
    return GpuResult<std::unique_ptr<GpuRuntime>>{
        GpuError::kNoDevice, "cuda: no live CUDA device (cudaGetDeviceCount)"};
#else
    return GpuResult<std::unique_ptr<GpuRuntime>>{
        GpuError::kUnsupportedBackend,
        "cuda backend not compiled into this build (no CUDA_HOME headers at "
        "configure time)"};
#endif
  case Backend::kAuto:
    break;
  }
  // kAuto: first compiled backend with a live device wins; otherwise the
  // deterministic compile-only runtime (ok(), so no-device states surface as
  // typed per-call errors instead of a factory failure).
#if defined(KERNSCOPE_HAVE_HIP)
  if (HipRuntime::Live()) {
    return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<HipRuntime>()};
  }
#endif
#if defined(KERNSCOPE_HAVE_CUDA)
  if (CudaRuntime::Live()) {
    return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<CudaRuntime>()};
  }
#endif
  return GpuResult<std::unique_ptr<GpuRuntime>>{std::make_unique<CompileOnlyRuntime>()};
}

} // namespace kernscope
