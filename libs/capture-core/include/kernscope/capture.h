//===- capture.h - capture-core: interception foundation ---------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libs/capture-core (todo 43): the shared interception foundation of the
// capture/replay track (C4). It owns the FOUR pieces every capture mechanism
// (LD_PRELOAD shim today, CUPTI primary in todo 44, HIP path in todo 45)
// records into:
//
//   1. the INTERCEPTION TABLE (Metis C-1): the fixed, versioned list of every
//      CUDA runtime/driver API the capture layer can observe, with per-API
//      call counters. The LD_PRELOAD shim (libkernscope_intercept.so) defines
//      exactly this table; todo 44's CUPTI subscription enables the same
//      cbids. A bundle written from a session therefore always cites a table
//      entry the reader can validate against.
//   2. the ptr->size ALLOCATION MAP (GPGPU-Sim `g_mallocPtr_Size` pattern,
//      docs/research/capture-replay-research-brief.md §5 heuristic 1): learned
//      from intercepted cudaMalloc/cudaFree (and cuMemAlloc-free equivalents
//      later), queried by memcpy/memset/launch records to attribute sizes.
//      THIS is how the bundle's buffer blobs get their byte counts without
//      type info.
//   3. the EVENT LOG: append-only, sequenced records of launches, copies,
//      memsets, streams, module/fatbin registrations and param introspection -
//      everything the shim observes at the API boundary, in call order.
//   4. the BUNDLE WRITER INTERFACE: a v0 minimal serializer the shim records
//      through (deterministic text). The FORMAL format/schema (JSON schema,
//      VA-faithful snapshots, provenance hashes) is todo 46's deliverable;
//      this interface is deliberately tiny so todo 46 swaps the writer
//      without touching the session model or the shim.
//
// KNOWN LIMITATIONS of the LD_PRELOAD mechanism (recorded here per
// docs/research/capture-replay-research-brief.md §2 pitfalls 1-5; they are
// exactly why todo 44's CUPTI callback/injection path is the NVIDIA-PRIMARY
// mechanism and this shim is NOT):
//   * driver-vs-runtime split (pitfall 1): the shim intercepts BOTH libcudart
//     and libcuda entry points, but driver entry points fetched BY ADDRESS via
//     cuGetProcAddress (CUDA Driver API, "Driver Entry Point Access") bypass
//     our PLT-level symbols. CUPTI's injection hooks from inside the driver
//     and does not have this hole.
//   * static cudart (pitfall 5): apps built with `-cudart static` never bind
//     the dynamic symbols the shim interposes on. CUPTI injection still sees
//     them.
//   * symbol-level coverage only: per-thread-default-stream variants
//     (cudaLaunchKernel_ptsz / cuLaunchKernel_ptsz) are not interposed unless
//     the app binds the unversioned names.
//   * argument introspection is driver-API-only: the runtime-API launches
//     (cudaLaunchKernel / cudaLaunchKernelExC) carry an opaque args array with
//     NO per-argument sizes at the boundary (the sizes live in the kernel
//     image); cuFuncGetParamInfo - intercepted here - resolves the layout for
//     the driver-API path (cuLaunchKernel/cuLaunchKernelEx). Byte-exact arg
//     capture for runtime-API launches is a todo-44 CUPTI functionParams
//     concern.
//
// This header is deliberately CUDA-header-free (like libs/gpuruntime): tests
// and consumers never need CUDA on the include path. The shim maps CUDA types
// to the plain integers below in intercept.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CAPTURE_H
#define KERNSCOPE_CAPTURE_H

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace kernscope {

//===--- SHA-256 (FIPS 180-4) ------------------------------------------------===//
// todo 46 requires a sha256 provenance hash per captured blob
// (docs/capture-format.md plan: "MUST NOT store buffers without provenance
// hashes (sha256 per blob)"); the todo-43 fatbin/cubin capture acceptance
// already demands a "cubin bytes HASH", so the digest is part of the core.
/// Hex-encoded lowercase SHA-256 of `data` (64 chars).
std::string sha256_hex(std::span<const std::uint8_t> data);

//===--- 1. interception table -----------------------------------------------===//
// The fixed API table the shim interposes on (Metis C-1). Order is stable:
// sessions serialize `api <name> calls=<n>` lines in this order.
enum class ApiKind : std::uint8_t {
  kCudaLaunchKernel,      // cudaLaunchKernel (runtime launch)
  kCudaLaunchKernelExC,   // cudaLaunchKernelExC (runtime launch, <<<>>> on 12.x)
  kCudaGraphLaunch,       // cudaGraphLaunch (graph launch)
  kCudaMalloc,            // cudaMalloc (alloc -> map)
  kCudaFree,              // cudaFree (free -> map; unknown ptr -> unknown-free)
  kCudaMemcpy,            // cudaMemcpy
  kCudaMemcpyAsync,       // cudaMemcpyAsync
  kCudaMemcpy2D,          // cudaMemcpy2D
  kCudaMemcpy2DAsync,     // cudaMemcpy2DAsync
  kCudaMemset,            // cudaMemset
  kCudaMemsetAsync,       // cudaMemsetAsync
  kCudaStreamCreate,      // cudaStreamCreate (stream info)
  kCudaStreamSynchronize, // cudaStreamSynchronize (stream info)
  kRegisterFatBinary,     // __cudaRegisterFatBinary (fatbin capture)
  kRegisterFunction,      // __cudaRegisterFunction (kernel name table)
  kCuModuleLoad,          // cuModuleLoad (cubin file capture)
  kCuModuleLoadData,      // cuModuleLoadData (cubin blob capture)
  kCuModuleLoadDataEx,    // cuModuleLoadDataEx (cubin blob capture)
  kCuModuleGetFunction,   // cuModuleGetFunction (CUfunction -> name for
                          // driver-API launch records; without it the bundle's
                          // kernel-name field is unsatisfiable on the driver
                          // path)
  kCuFuncGetParamInfo,    // cuFuncGetParamInfo (arg introspection)
  kCuLaunchKernel,        // cuLaunchKernel (driver launch)
  kCuLaunchKernelEx,      // cuLaunchKernelEx (driver launch)
  kCount,
};

/// Stable table name (also the `api` line token in the v0 bundle).
const char *api_name(ApiKind kind);

/// One row of the interception table: what was interposed + how often it ran.
struct ApiTableEntry {
  ApiKind kind;
  const char *name;     // == api_name(kind)
  std::uint64_t calls = 0;
};

//===--- 2/3. event log records ----------------------------------------------===//

/// How a launch record reached the session (the driver-vs-runtime split in
/// the brief §2 pitfall 1 makes this a first-class attribute: runtime-API
/// records carry no arg bytes; driver-API records may, via cuFuncGetParamInfo).
enum class LaunchVia : std::uint8_t {
  kRuntimeApi,   // cudaLaunchKernel
  kRuntimeApiEx, // cudaLaunchKernelExC
  kGraph,        // cudaGraphLaunch (kernel names unreachable at this boundary;
                 // todo 44's CUPTI resource/launch callbacks walk graph nodes)
  kDriverApi,    // cuLaunchKernel
  kDriverApiEx,  // cuLaunchKernelEx
};

const char *to_string(LaunchVia via);

/// One kernel parameter BY VALUE: `size` bytes copied from the host side of
/// the parameter (driver-API convention, cuLaunchKernel kernelParams; the
/// byte-exact copy contract is asserted against the mock in the tests).
struct KernelArgSlot {
  std::size_t size = 0;
  std::vector<std::uint8_t> bytes; // empty when size==0 or size unknown
};

struct LaunchRecord {
  std::string kernel_name;    // "" for graph launches
  LaunchVia via = LaunchVia::kDriverApi;
  std::uint32_t grid_x = 1, grid_y = 1, grid_z = 1;
  std::uint32_t block_x = 1, block_y = 1, block_z = 1;
  std::uint32_t shared_mem_bytes = 0;
  std::uint64_t stream = 0;        // raw stream handle value
  std::uint64_t graph_handle = 0;  // raw cudaGraphExec_t value (kGraph only)
  std::uint64_t seq = 0;           // monotonic event order
  std::vector<KernelArgSlot> args; // may be empty (see header limitation note)
};

/// cudaMemcpyKind value (CUDA 12.6 driver_types.h: 1 H2D, 2 D2H, 3 D2D) plus
/// the allocation-map sizes that were KNOWN at the boundary (0 = unknown ptr).
struct CopyRecord {
  std::string_view kind = "memcpy"; // memcpy|memcpy2d|memcpy-async|memcpy2d-async
  std::uint64_t dst = 0, src = 0;
  std::size_t bytes = 0;        // count (memcpy) or width*height (2D)
  int cuda_kind = 0;            // cudaMemcpyKind value
  std::size_t dst_size = 0;     // allocation_map[dst] when kind is H2D/D2D
  std::size_t src_size = 0;     // allocation_map[src] when kind is D2H/D2D
  std::uint64_t stream = 0;
  std::uint64_t seq = 0;
};

struct MemsetRecord {
  std::uint64_t ptr = 0;
  int value = 0;
  std::size_t bytes = 0;
  std::size_t ptr_size = 0; // allocation_map[ptr] (0 = unknown)
  std::uint64_t stream = 0;
  std::uint64_t seq = 0;
};

struct AllocRecord { // one per successful alloc (is_alloc) or known free
  std::uint64_t ptr = 0;
  std::size_t size = 0; // size at alloc; remembered size at free
  bool is_alloc = true;
  std::uint64_t seq = 0;
};

struct StreamRecord {
  std::uint64_t handle = 0;
  std::uint64_t id = 0; // logical id assigned at create (0 = unknown handle)
  bool is_sync = false;
  std::uint64_t seq = 0;
};

/// Negative-path event (QA-asserted): cudaFree on a pointer the allocation
/// map does not know. The CUDA contract says freeing a pointer that was not
/// returned by cudaMalloc is undefined; the shim therefore logs this record
/// and forwards - it must NEVER crash on it.
struct UnknownFreeRecord {
  std::uint64_t ptr = 0;
  bool forwarded = true; // shim forwarded to the real cudaFree
  std::uint64_t seq = 0;
};

/// One (CUfunction, paramIndex) -> (offset, size) layout fact learned from
/// cuFuncGetParamInfo (driver-API arg introspection, brief §5).
struct ParamInfo {
  std::size_t offset = 0;
  std::size_t size = 0;
};

struct ModuleRecord {
  enum class Kind : std::uint8_t { kFatBinary, kCubin };
  Kind kind = Kind::kFatBinary;
  std::uint64_t handle = 0; // fatbin registration id / cubin module id
  std::string sha256;       // blob hash (cubin bytes; fatbin cubin payload)
  std::string name;         // "" for blobs; file path for cuModuleLoad
  std::vector<std::string> kernels; // names registered via __cudaRegisterFunction
  std::uint64_t seq = 0;
};

//===--- 4. the session -------------------------------------------------------===//
// The single recording surface the shim (and any future mechanism) writes
// into. All recorders are thread-safe (the shim is preloaded into arbitrary
// multi-threaded apps).

class CaptureSession {
public:
  /// Process-wide singleton for the shim; deliberately leaked (a preload
  /// library must not destroy capture state at exit - atexit flush handlers
  /// run before function-local statics of a .so are destroyed, and apps may
  /// dlclose the shim).
  static CaptureSession &instance();

  // --- recorders (called by the shim / tests) -------------------------------

  void note_api(ApiKind kind);

  void record_alloc(std::uint64_t ptr, std::size_t size);
  /// Unknown ptr -> UnknownFreeRecord (the QA negative path), never a crash.
  void record_free(std::uint64_t ptr, bool forwarded = true);

  void record_copy(const CopyRecord &rec);
  void record_memset(const MemsetRecord &rec);

  /// Returns the logical stream id (1-based; each handle gets one).
  std::uint64_t record_stream_create(std::uint64_t handle);
  void record_stream_sync(std::uint64_t handle);

  std::uint64_t record_launch(LaunchRecord rec);

  /// Registers a fatbin (shim feeds the cubin bytes extracted from the
  /// __fatBinCHeader image, GPGPU-Sim pattern; unparsable -> the raw header
  /// bytes it did parse). Returns the registration id (1-based).
  std::uint64_t register_fatbinary(std::span<const std::uint8_t> blob);
  /// Associates a kernel name with fatbin registration `reg_id` (from
  /// __cudaRegisterFunction's deviceName). Returns false on unknown reg id.
  bool register_function(std::uint64_t reg_id, std::string_view kernel_name);
  /// cuModuleLoadData(Ex) cubin image bytes. Returns the module id.
  std::uint64_t register_cubin(std::span<const std::uint8_t> image);
  /// cuModuleLoad file path (shim hashes the file itself; "" sha = unreadable).
  std::uint64_t register_module_file(std::string_view path);

  /// cuFuncGetParamInfo interception: (func, idx) -> (offset, size).
  void record_param_info(std::uint64_t cufunction, std::size_t param_index,
                         std::size_t offset, std::size_t size);

  /// Kernel-name registries: the runtime path names launches by the host
  /// function pointer seen in __cudaRegisterFunction; the driver path by the
  /// CUfunction seen in cuModuleGetFunction. Lookups return "" when unknown.
  void register_host_function(std::uint64_t host_fun,
                              std::string_view kernel_name);
  void register_cufunction(std::uint64_t cufunction,
                           std::string_view kernel_name);
  std::string kernel_name_for_host_function(std::uint64_t host_fun) const;
  std::string kernel_name_for_cufunction(std::uint64_t cufunction) const;

  // --- queries (tests) -------------------------------------------------------

  /// Allocation map: EXACT size for a live allocation; nullopt if unknown.
  std::optional<std::size_t> allocation_size(std::uint64_t ptr) const;
  std::size_t allocation_count() const;

  const std::vector<ApiTableEntry> &api_table() const;
  std::uint64_t api_count(ApiKind kind) const;

  const std::vector<LaunchRecord> &launches() const { return launches_; }
  const std::vector<CopyRecord> &copies() const { return copies_; }
  const std::vector<MemsetRecord> &memsets() const { return memsets_; }
  const std::vector<AllocRecord> &allocs() const { return allocs_; }
  const std::vector<StreamRecord> &streams() const { return streams_; }
  const std::vector<UnknownFreeRecord> &unknown_frees() const {
    return unknown_frees_;
  }
  const std::vector<ModuleRecord> &modules() const { return modules_; }
  /// Param layout learned for `cufunction` (by param_index), if any.
  std::optional<std::size_t> param_size(std::uint64_t cufunction,
                                        std::size_t param_index) const;

  bool empty() const;
  void clear();

public:
  // Constructible (tests create fresh sessions); the shim uses instance().
  CaptureSession();

  // All recorders lock `mu_` (the shim runs inside arbitrary multi-threaded
  // apps). Read accessors intentionally do NOT lock - same convention as the
  // mock runtime's ops() (libs/gpuruntime): the read side is the unit tests
  // and the bundle writer, which run when the session is quiescent (explicit
  // flush call or atexit). Recorders and readers must not run concurrently;
  // documented contract.
  mutable std::mutex mu_;

  std::vector<ApiTableEntry> table_;
  std::vector<LaunchRecord> launches_;
  std::vector<CopyRecord> copies_;
  std::vector<MemsetRecord> memsets_;
  std::vector<AllocRecord> allocs_;
  std::vector<StreamRecord> streams_;
  std::vector<UnknownFreeRecord> unknown_frees_;
  std::vector<ModuleRecord> modules_;
  std::uint64_t seq_ = 0;
  // ptr -> EXACT allocation size (GPGPU-Sim g_mallocPtr_Size pattern).
  std::unordered_map<std::uint64_t, std::size_t> allocations_;
  // stream handle -> logical id (1-based, assigned at create).
  std::unordered_map<std::uint64_t, std::uint64_t> stream_ids_;
  // (CUfunction, paramIndex) -> ParamInfo, learned from cuFuncGetParamInfo.
  std::unordered_map<std::uint64_t, std::vector<ParamInfo>> param_infos_;
  // host function pointer -> kernel name (__cudaRegisterFunction).
  std::unordered_map<std::uint64_t, std::string> function_names_;
  // CUfunction -> kernel name (cuModuleGetFunction).
  std::unordered_map<std::uint64_t, std::string> cufunction_names_;
};

//===--- bundle writer INTERFACE (v0 minimal) ---------------------------------===//
// The shim records INTO a CaptureSession; the writer serializes a session to
// a bundle. v0 = deterministic line-oriented text so the smoke test can assert
// records and the mechanism can be validated end-to-end. todo 46 replaces the
// FORMAT (JSON schema + VA-faithful blobs + provenance) behind this interface
// without touching the session model or the shim.

class BundleWriter {
public:
  virtual ~BundleWriter() = default;
  /// Serialize the whole session state to `out`. Returns false only on I/O
  /// failure; never on content (records are plain data).
  virtual bool write(const CaptureSession &session, std::ostream &out) const = 0;
  /// v0 deterministic text writer (todo 46 swaps this factory's implementation).
  static std::unique_ptr<BundleWriter> create_v0();
};

} // namespace kernscope

#endif // KERNSCOPE_CAPTURE_H
