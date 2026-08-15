//===- gpu.h - Thin CUDA/HIP runtime abstraction for kernscope ----*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libs/gpuruntime: a THIN host-side abstraction over the CUDA and HIP
// runtime/driver-level APIs, covering exactly what the capture/profile tracks
// need (todo 21): device enumeration + properties, module/kernel loading and
// launch, event timing, device malloc/free, and memcpy/memset. No device-code
// generation, no kernel libraries: every operation is a host call, so the
// implementation is a single translation unit (src/gpu.cpp) that selects the
// backend with the portable #ifdef convention
//   #if defined(__HIP_PLATFORM_AMD__) -> HIP backend
//   #elif defined(KERNSCOPE_BACKEND_CUDA) -> CUDA backend
// (pattern proven in /home/mei/projects/polykernel/lib/Runtime/
// DeviceDetect.cpp: the backend headers are included in the .cpp, NEVER here,
// so this header and the GPU-free gtests never need ROCm/CUDA headers on the
// include path).
//
// Backends:
//   * HipRuntime   - live HIP (rocmPackages.clr): the local backend on this
//                    machine (gfx1101, rocminfo-verified).
//   * CudaRuntime  - CUDA (cudaPackages_12_6): compiled whenever CUDA headers
//                    are available; on this AMD-only machine it is a
//                    COMPILE-ONLY path (no NVIDIA driver -> every call
//                    returns a typed no-device error from the driver).
//   * MockRuntime  - unit-test backend: records every operation into an
//                    inspectable op log AND retains BYTE-EXACT buffer contents
//                    for every captured allocation (review-mandated: todo 44's
//                    cross-vendor equivalence test asserts exact input-blob
//                    bytes; expectations cite the CUDA/HIP API contract docs,
//                    Metis M-5.i - no circular validation against the mock).
//   * CompileOnlyRuntime - deterministic no-GPU backend: every call returns a
//                    typed GpuError::kNoDevice result (never crashes), so the
//                    unit suite runs on machines with no GPU/driver at all.
//
// GpuRuntime instances are obtained from create_runtime(), which
// auto-detects: a live compiled backend wins, otherwise CompileOnlyRuntime.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_GPU_H
#define KERNSCOPE_GPU_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope {

// Which backend the library was built with (set by the CMake target compile
// definitions of kernscope_gpuruntime, PUBLIC so consumers/tests see it).
#ifdef KERNSCOPE_HIP_BACKEND_BUILD
inline constexpr bool kHipBackendCompiled = true;
inline constexpr bool kCudaBackendCompiled = false;
inline constexpr bool kCompileOnlyBuild = false;
#elif defined(KERNSCOPE_CUDA_BACKEND_BUILD)
inline constexpr bool kHipBackendCompiled = false;
inline constexpr bool kCudaBackendCompiled = true;
inline constexpr bool kCompileOnlyBuild = false;
#else
inline constexpr bool kHipBackendCompiled = false;
inline constexpr bool kCudaBackendCompiled = false;
inline constexpr bool kCompileOnlyBuild = true;
#endif

//===--- typed error model --------------------------------------------------===//
// Every backend call returns either GpuStatus (void ops) or GpuResult<T>
// (value-returning ops) carrying a typed GpuError + human message. The
// "no-device" states are first-class errors, so compile-only machines exercise
// exactly the same code paths (Metis M-5.i: typed errors asserted by the unit
// suite, never crashes).

enum class GpuError : std::uint8_t {
  kOk = 0,
  kNoDevice,        // no GPU present / no driver / backend not compiled in
  kNoDriver,        // driver present but unusable (e.g. insufficient version)
  kModuleLoadFailed, // module file unreadable or rejected by the backend
  kKernelNotFound,  // named kernel/symbol not found in the module
  kLaunchFailed,    // kernel launch rejected
  kMemoryOpFailed,  // device malloc/free/memset failed
  kCopyOpFailed,    // memcpy failed
  kEventOpFailed,   // event create/record/sync/elapsed failed
  kDeviceNotFound,  // device_properties() name did not match any device
  kUnsupportedBackend, // requested backend not compiled into this build
  kOther,
};

/// Human-readable name of a GpuError (stable, for diagnostics/tests).
std::string_view to_string(GpuError err) noexcept;

/// Result of a void operation.
struct GpuStatus {
  GpuError err = GpuError::kOk;
  std::string message; // backend-provided detail on failure (never on success)

  GpuStatus() = default;
  GpuStatus(GpuError e, std::string m) : err(e), message(std::move(m)) {}

  bool ok() const noexcept { return err == GpuError::kOk; }
  explicit operator bool() const noexcept { return ok(); }
};

/// Result of a value-returning operation (value is valid only when ok()).
template <typename T> struct GpuResult {
  GpuError err = GpuError::kOk;
  std::string message;
  T value{};

  GpuResult() = default;
  GpuResult(T v) : value(std::move(v)) {}
  GpuResult(GpuError e, std::string m) : err(e), message(std::move(m)) {}
  GpuResult(GpuStatus s) : err(s.err), message(std::move(s.message)) {}

  bool ok() const noexcept { return err == GpuError::kOk; }
  explicit operator bool() const noexcept { return ok(); }
};

//===--- device model -------------------------------------------------------===//

struct DeviceInfo {
  std::string name;         // vendor marketing name (e.g. "AMD Radeon RX 7800 XT")
  std::string arch;         // "sm_120" (CUDA) | "gfx1101" (HIP)
  std::uint64_t memory_bytes = 0;
  std::string vendor;       // "nvidia" | "amd" | "mock"
};

struct DeviceProperties {
  std::string name;
  // CUDA: compute capability (cudaDeviceProp.major/minor; 0 when n/a).
  int compute_capability_major = 0;
  int compute_capability_minor = 0;
  // HIP: gcnArchName (e.g. "gfx1101" / full "gfx1101:sramecc+:xnack-").
  std::string gcn_arch;      // bare cache-key form ("gfx1101")
  std::string gcn_arch_full; // raw gcnArchName string
  std::uint64_t memory_bytes = 0; // totalGlobalMem
  std::string vendor;
};

/// Opaque device-memory handle (address-like value; backend-defined).
struct DevicePtr {
  std::uint64_t value = 0;
  bool operator==(const DevicePtr &o) const noexcept { return value == o.value; }
  bool operator!=(const DevicePtr &o) const noexcept { return value != o.value; }
  explicit operator bool() const noexcept { return value != 0; }
};

//===--- launch model -------------------------------------------------------===//

struct LaunchConfig {
  std::uint32_t grid_x = 1, grid_y = 1, grid_z = 1;
  std::uint32_t block_x = 1, block_y = 1, block_z = 1;
  std::uint32_t shared_mem_bytes = 0;
};

/// One kernel parameter passed BY VALUE (driver-API convention): `data` points
/// at the host-side value of the parameter (e.g. the uintptr of a device
/// pointer, or the 4-byte int for a scalar); `size` is its byte width. The
/// backend copies these bytes into the launch parameter buffer, and the mock
/// records them byte-exactly into the op log.
struct KernelArg {
  const void *data = nullptr;
  std::size_t size = 0;
};

//===--- forward declarations ------------------------------------------------===//
class GpuRuntime;
class Module;
class Kernel;
class Event;
class MockRuntime;

//===--- GpuRuntime interface ------------------------------------------------===//

/// Backend-agnostic host-side GPU runtime. All operations are synchronous on
/// the default stream (thin: streams/async are deliberately out of scope until
/// a consumer needs them - MUST NOT wrap driver features beyond the
/// capture/profile tracks' needs).
class GpuRuntime {
public:
  virtual ~GpuRuntime() = default;

  /// Backend tag: "cuda" | "hip" | "mock" | "compile-only".
  virtual std::string_view backend_name() const noexcept = 0;

  // Device enumeration + properties.
  virtual GpuResult<std::vector<DeviceInfo>> list_devices() = 0;
  virtual GpuResult<DeviceProperties>
  device_properties(const std::string &name) = 0;

  // Module/kernel loading.
  virtual GpuResult<std::unique_ptr<Module>> load_module(const std::string &path) = 0;
  virtual GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module &module, const std::string &name) = 0;

  // Device memory + copies (driver-API semantics: cudaMalloc/hipMalloc,
  // cudaMemcpy{HtoD,DtoH}/hipMemcpy{HtoD,DtoH}, cudaMemset/hipMemset).
  virtual GpuResult<DevicePtr> device_malloc(std::size_t bytes) = 0;
  virtual GpuStatus device_free(DevicePtr ptr) = 0;
  virtual GpuStatus memcpy_htod(DevicePtr dst, const void *src, std::size_t bytes) = 0;
  virtual GpuStatus memcpy_dtoh(void *dst, DevicePtr src, std::size_t bytes) = 0;
  virtual GpuStatus memset_device(DevicePtr dst, std::uint8_t value, std::size_t bytes) = 0;

  // Event timing (default stream).
  virtual GpuResult<std::unique_ptr<Event>> create_event() = 0;

  /// Block until all device work is done (hipDeviceSynchronize /
  /// cudaDeviceSynchronize / mock no-op / compile-only typed error).
  virtual GpuStatus synchronize() = 0;
};

enum class Backend : std::uint8_t { kAuto, kCuda, kHip, kMock, kCompileOnly };

/// Factory. kAuto: the first compiled backend that reports a live device wins;
/// with no live backend a CompileOnlyRuntime is returned (ok(), never an error
/// - the caller can then detect the no-device state via typed errors). An
/// explicit kCuda/kHip on a build that did not compile that backend returns a
/// typed GpuError::kUnsupportedBackend result.
GpuResult<std::unique_ptr<GpuRuntime>> create_runtime(Backend backend = Backend::kAuto);

//===--- Module / Kernel / Event --------------------------------------------===//
// Static factories keep the deliverable's call shape (Module::load(path),
// Kernel::create(module, name), Event::record/elapsed) while every object
// stays bound to the runtime that created it (multiple backends can coexist
// in one process, e.g. mock + live in the test binary).

class Module {
public:
  virtual ~Module() = default;
  static GpuResult<std::unique_ptr<Module>> load(GpuRuntime &rt, const std::string &path);
};

class Kernel {
public:
  virtual ~Kernel() = default;
  static GpuResult<std::unique_ptr<Kernel>>
  create(GpuRuntime &rt, Module &module, const std::string &name);
  virtual GpuStatus launch(const LaunchConfig &cfg, std::span<const KernelArg> args) = 0;
};

class Event {
public:
  virtual ~Event() = default;
  static GpuResult<std::unique_ptr<Event>> create(GpuRuntime &rt);
  /// Record this event on the default stream (driver semantics:
  /// cudaEventRecord/hipEventRecord on the null stream).
  virtual GpuStatus record() = 0;
  virtual GpuStatus synchronize() = 0;
  /// Wall-clock ms between *this* (earlier) and `other` (later), as reported
  /// by the backend (cudaEventElapsedTime/hipEventElapsedTime).
  virtual GpuResult<double> elapsed_ms(const Event &other) const = 0;
};

//===--- MockRuntime (unit-test backend) ------------------------------------===//
// Records every operation into an inspectable op log and retains BYTE-EXACT
// buffer contents for every captured allocation (review-mandated; todo 44's
// cross-vendor equivalence test reads these bytes). All expectations in tests
// cite the CUDA/HIP API contract docs (Metis M-5.i), never the mock's own
// behavior.

class MockRuntime : public GpuRuntime {
public:
  /// One recorded operation. `payload` carries the exact bytes that crossed
  /// the API boundary (kernel args for a launch; the copied blob for a
  /// memcpy), so tests can assert byte-exact contents.
  struct Op {
    enum class Type : std::uint8_t {
      kDeviceQuery,  // list_devices / device_properties
      kModuleLoad,
      kKernelCreate,
      kLaunch,
      kMalloc,
      kFree,
      kMemcpyHtoD,
      kMemcpyDtoH,
      kMemset,
      kEventRecord,
      kEventSynchronize,
      kEventElapsed,
      kSynchronize,
    };
    Type type;
    std::string detail; // kernel name / module path / device name
    std::size_t size = 0; // byte count of the op (alloc/copy/arg blob)
    LaunchConfig config;  // launch ops only
    std::vector<std::uint8_t> payload; // byte-exact captured bytes (may be empty)
  };

  /// The recorded ops, in call order. Tests assert the exact sequence
  /// (QA: an unexpected op order fails the test citing the API contract).
  const std::vector<Op> &ops() const noexcept { return ops_; }

  /// Byte-exact contents of a mock allocation (the bytes written by
  /// memcpy_htod/memset_device, exactly as stored). Empty span for an
  /// unknown/freed handle.
  std::span<const std::uint8_t> buffer(DevicePtr ptr) const noexcept;

  /// Append an op to the log (used by the mock's own kernel/event objects;
  /// tests treat the log as append-only and only read it).
  void record(Op::Type type, std::string detail, std::size_t size = 0,
              LaunchConfig cfg = {}, std::vector<std::uint8_t> payload = {});

  // GpuRuntime interface (mock semantics documented in gpu.cpp).
  std::string_view backend_name() const noexcept override { return "mock"; }
  GpuResult<std::vector<DeviceInfo>> list_devices() override;
  GpuResult<DeviceProperties> device_properties(const std::string &name) override;
  GpuResult<std::unique_ptr<Module>> load_module(const std::string &path) override;
  GpuResult<std::unique_ptr<Kernel>>
  create_kernel(Module &module, const std::string &name) override;
  GpuResult<DevicePtr> device_malloc(std::size_t bytes) override;
  GpuStatus device_free(DevicePtr ptr) override;
  GpuStatus memcpy_htod(DevicePtr dst, const void *src, std::size_t bytes) override;
  GpuStatus memcpy_dtoh(void *dst, DevicePtr src, std::size_t bytes) override;
  GpuStatus memset_device(DevicePtr dst, std::uint8_t value, std::size_t bytes) override;
  GpuResult<std::unique_ptr<Event>> create_event() override;
  GpuStatus synchronize() override;

private:
  struct Buffer {
    std::vector<std::uint8_t> bytes;
  };
  std::vector<Op> ops_;
  std::vector<std::pair<DevicePtr, Buffer>> buffers_; // stable: never reallocates Buffer
  std::uint64_t next_handle_ = 1;

  GpuResult<Buffer *> find_buffer(DevicePtr ptr) noexcept;
};

} // namespace kernscope

#endif // KERNSCOPE_GPU_H
