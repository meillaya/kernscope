//===- capture_hip.cpp - rocprofiler-sdk HIP capture tool ---------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libkernscope_capture_hip.so (todo 45): the LOCAL LIVE HIP capture loop.
// A rocprofiler-sdk tool exporting `rocprofiler_configure`. Loading:
//
//   LD_PRELOAD=$PWD/build/libkernscope_capture_hip.so <app>       (or)
//   ROCPROFILER_LIBRARY=$PWD/build/libkernscope_capture_hip.so <app>
//
// The .so NEEDs librocprofiler-sdk.so, so the SDK is always present when the
// tool is loaded; rocprofiler-sdk discovers the tool's rocprofiler_configure
// at initialization and calls it. The tool then:
//
//   * registers for the HIP runtime intercept table
//     (rocprofiler_at_intercept_table_registration) and wraps the allocation
//     hooks, copies/memsets and launches (Kerncap/intellikit pattern - MIT,
//     studied only, THIRD-PARTY.md);
//   * subscribes kernel-dispatch callback tracing (grid/workgroup from the
//     HSA layer) and code-object tracing (HSACO blobs + kernel symbols);
//   * records everything into kernscope::CaptureSession (libs/capture-core,
//     NEVER modified) and flushes a v0 bundle at exit through the v0
//     BundleWriter.
//
// The deprecated tracer is NOT used (rocprofiler-sdk is the blessed API).
//
// Recording contract ("MUST NOT modify application behavior beyond
// observation"): every wrapper forwards the ORIGINAL arguments to the stored
// real table entry and returns its exact result; recording happens around the
// forward. Wrapper code never calls HIP APIs through the table (no recursion).
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture.h"

#include "amdgpu_metadata.h"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/intercept_table.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <hip/amd_detail/hip_api_trace.hpp>
#include <hip/hip_runtime.h>

#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using kernscope::ApiKind;
using kernscope::CopyRecord;
using kernscope::LaunchRecord;
using kernscope::LaunchVia;
using kernscope::MemsetRecord;

//===--- interception state --------------------------------------------------===//
// The stored REAL table entries (captured at intercept-table registration)
// and the session recorder.

kernscope::CaptureSession &Session() {
  return kernscope::CaptureSession::instance();
}

struct RealTable {
  t_hipMalloc hipMalloc = nullptr;
  t_hipFree hipFree = nullptr;
  t_hipMemcpy hipMemcpy = nullptr;
  t_hipMemcpyAsync hipMemcpyAsync = nullptr;
  t_hipMemset hipMemset = nullptr;
  t_hipMemsetAsync hipMemsetAsync = nullptr;
  t_hipLaunchKernel hipLaunchKernel = nullptr;
  t_hipStreamSynchronize hipStreamSynchronize = nullptr;
  t_hipDeviceSynchronize hipDeviceSynchronize = nullptr;
};
RealTable g_real{};

// The HIP runtime implements hipMemcpy through hipMemcpyAsync internally
// (and hipMemset through hipMemsetAsync); thread-local guards suppress the
// duplicate record of the runtime's own re-dispatch - the same dedup the
// todo-43 shim applies for the runtime-vs-driver split.
thread_local bool g_in_memcpy = false;
thread_local bool g_in_memset = false;

//===--- pending launch merge (HIP wrapper <-> kernel dispatch) --------------===//
// The HIP intercept-table launch (hipLaunchKernel) sees grid/block/shared/
// stream and the host-side kernelParams array, but NOT the kernel name; the
// kernel-dispatch callback (HSA layer) sees kernel_id (-> name via the
// code-object symbol table) and the runtime grid/workgroup. For a sequential
// single-GPU app the two are strictly ordered: pending-launch #N is completed
// by dispatch #N (FIFO). This is the documented v0 merge; correlation-id
// hardening is a todo-48/49 concern.

struct PendingLaunch {
  std::uint32_t grid_x = 1, grid_y = 1, grid_z = 1;
  std::uint32_t block_x = 1, block_y = 1, block_z = 1;
  std::uint32_t shared_mem_bytes = 0;
  std::uint64_t stream = 0;
  void **kernel_params = nullptr; // host-side arg pointer array (hipLaunchKernel)
  std::size_t seq = 0;
};

std::mutex g_mu;                             // guards the maps + pending queue
std::vector<PendingLaunch> g_pending_launches{};
std::uint64_t g_pending_seq = 0;

struct KernelSymbol {
  std::string name; // kernel symbol name (mangled, from the code object)
  std::uint32_t kernarg_segment_size = 0;
};
std::unordered_map<std::uint64_t, KernelSymbol> g_kernels; // kernel_id -> symbol
std::unordered_map<std::uint64_t, std::uint64_t> g_co_to_mod; // code_object_id -> module reg id
// Per-kernel argument layout from the code object's AMDGPU metadata note
// (keyed by the mangled kernel name, ".kd" suffix stripped): drives the
// byte-exact per-argument capture of the launch path.
std::unordered_map<std::string, std::vector<kernscope::AmdArgLayout>>
    g_arg_layouts;
std::uint64_t g_input_buffer_seq = 0; // input-buffer-<n>.bin numbering

void WriteBlob(const std::string &name, const void *data, std::size_t size); // fwd

/// Kernel-name key for the layout map: the dispatch-layer name carries a
/// trailing ".kd" (kernel descriptor suffix) the metadata ".name" does not.
std::string LayoutKey(std::string_view dispatch_name) {
  std::string key(dispatch_name);
  constexpr std::string_view kKd = ".kd";
  if (key.size() > kKd.size() && key.compare(key.size() - kKd.size(), kKd.size(), kKd) == 0) {
    key.resize(key.size() - kKd.size());
  }
  return key;
}

/// Dumps the host-side source of a hipMemcpyHostToDevice call to
/// input-buffer-<n>.bin (v0-scope of the Kerncap address-space closure): the
/// acceptance hash-matches these blobs against the app's known inputs. The
/// host source is valid at the API boundary (for the async variant the app
/// must keep it alive until completion by contract, so reading it at call
/// time is sound). Best-effort; capped so a runaway app cannot fill the disk.
void DumpH2DInput(const void *src, std::size_t bytes) {
  constexpr std::size_t kMaxDumpBytes = 512u << 20; // 512 MiB sanity cap
  if (src == nullptr || bytes == 0) {
    return;
  }
  if (bytes > kMaxDumpBytes) {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: H2D input dump skipped "
                 "(%zu bytes > 512 MiB cap)\n", bytes);
    return;
  }
  std::lock_guard<std::mutex> lock(g_mu);
  const std::string name =
      "input-buffer-" + std::to_string(++g_input_buffer_seq) + ".bin";
  WriteBlob(name, src, bytes);
}

//===--- blob output ----------------------------------------------------------===//
// Blob directory for code-object-<id>.hsaco and input-buffer-<n>.bin.
// Resolved once at first use from KERNSCOPE_HIP_BLOB_DIR (default ".").

std::string BlobDir() {
  const char *dir = std::getenv("KERNSCOPE_HIP_BLOB_DIR");
  return (dir != nullptr && dir[0] != '\0') ? dir : ".";
}

void WriteBlob(const std::string &name, const void *data, std::size_t size) {
  const std::string path = BlobDir() + "/" + name;
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (out) {
    out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  } else {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: cannot write blob %s\n",
                 path.c_str());
  }
}

//===--- wrappers -------------------------------------------------------------===//

hipError_t WrapHipMalloc(void **ptr, size_t size) {
  Session().note_api(ApiKind::kCudaMalloc);
  hipError_t rc = hipErrorNoDevice;
  if (g_real.hipMalloc != nullptr) {
    rc = g_real.hipMalloc(ptr, size);
  }
  if (rc == hipSuccess && ptr != nullptr && *ptr != nullptr) {
    Session().record_alloc(reinterpret_cast<std::uint64_t>(*ptr), size);
  }
  return rc;
}

hipError_t WrapHipFree(void *ptr) {
  Session().note_api(ApiKind::kCudaFree);
  // Record BEFORE the free (the pointer must be looked up while live).
  Session().record_free(reinterpret_cast<std::uint64_t>(ptr),
                        g_real.hipFree != nullptr);
  return g_real.hipFree != nullptr ? g_real.hipFree(ptr) : hipErrorNoDevice;
}

void RecordCopy(CopyRecord rec) { Session().record_copy(std::move(rec)); }

/// QA-negative instrumentation (asserted in capture_hip_negative_check.py):
/// when the device-side pointer of a copy is not in the ptr->size allocation
/// map (allocated via an API the capture does not hook, e.g. hipMallocPitch),
/// the bundle marks the buffer size-unknown (dst-size/src-size = 0, the v0
/// convention) and the capture logs a warning - NEVER a crash.
void WarnIfDevicePtrUnknown(const CopyRecord &rec) {
  const bool dst_is_dev = (rec.cuda_kind == 1 || rec.cuda_kind == 3);
  const bool src_is_dev = (rec.cuda_kind == 2 || rec.cuda_kind == 3);
  const auto warn = [](std::uint64_t ptr) {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: buffer size-unknown "
                 "(ptr=0x%llx): not allocated via a captured HIP allocation "
                 "API\n",
                 static_cast<unsigned long long>(ptr));
  };
  if (dst_is_dev && !Session().allocation_size(rec.dst).has_value()) {
    warn(rec.dst);
  }
  if (src_is_dev && !Session().allocation_size(rec.src).has_value()) {
    warn(rec.src);
  }
}

hipError_t WrapHipMemcpy(void *dst, const void *src, size_t sizeBytes,
                         hipMemcpyKind kind) {
  Session().note_api(ApiKind::kCudaMemcpy);
  if (g_in_memcpy) { // runtime re-dispatch of the async variant
    return g_real.hipMemcpy(dst, src, sizeBytes, kind);
  }
  if (kind == hipMemcpyHostToDevice) {
    DumpH2DInput(src, sizeBytes);
  }
  g_in_memcpy = true;
  hipError_t rc = g_real.hipMemcpy(dst, src, sizeBytes, kind);
  g_in_memcpy = false;
  if (rc == hipSuccess) {
    CopyRecord rec;
    rec.kind = "memcpy";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = sizeBytes;
    rec.cuda_kind = static_cast<int>(kind);
    WarnIfDevicePtrUnknown(rec);
    RecordCopy(std::move(rec));
  }
  return rc;
}

hipError_t WrapHipMemcpyAsync(void *dst, const void *src, size_t sizeBytes,
                              hipMemcpyKind kind, hipStream_t stream) {
  Session().note_api(ApiKind::kCudaMemcpyAsync);
  if (g_in_memcpy) { // sync variant already recorded this logical copy
    return g_real.hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  }
  if (kind == hipMemcpyHostToDevice) {
    DumpH2DInput(src, sizeBytes);
  }
  g_in_memcpy = true;
  hipError_t rc = g_real.hipMemcpyAsync(dst, src, sizeBytes, kind, stream);
  g_in_memcpy = false;
  if (rc == hipSuccess) {
    CopyRecord rec;
    rec.kind = "memcpy-async";
    rec.dst = reinterpret_cast<std::uint64_t>(dst);
    rec.src = reinterpret_cast<std::uint64_t>(src);
    rec.bytes = sizeBytes;
    rec.cuda_kind = static_cast<int>(kind);
    rec.stream = reinterpret_cast<std::uint64_t>(stream);
    WarnIfDevicePtrUnknown(rec);
    RecordCopy(std::move(rec));
  }
  return rc;
}

hipError_t WrapHipMemset(void *dst, int value, size_t sizeBytes) {
  Session().note_api(ApiKind::kCudaMemset);
  if (g_in_memset) {
    return g_real.hipMemset(dst, value, sizeBytes);
  }
  g_in_memset = true;
  hipError_t rc = g_real.hipMemset(dst, value, sizeBytes);
  g_in_memset = false;
  if (rc == hipSuccess) {
    kernscope::MemsetRecord rec;
    rec.ptr = reinterpret_cast<std::uint64_t>(dst);
    rec.value = value;
    rec.bytes = sizeBytes;
    Session().record_memset(std::move(rec));
  }
  return rc;
}

hipError_t WrapHipMemsetAsync(void *dst, int value, size_t sizeBytes,
                              hipStream_t stream) {
  Session().note_api(ApiKind::kCudaMemsetAsync);
  if (g_in_memset) {
    return g_real.hipMemsetAsync(dst, value, sizeBytes, stream);
  }
  g_in_memset = true;
  hipError_t rc = g_real.hipMemsetAsync(dst, value, sizeBytes, stream);
  g_in_memset = false;
  if (rc == hipSuccess) {
    kernscope::MemsetRecord rec;
    rec.ptr = reinterpret_cast<std::uint64_t>(dst);
    rec.value = value;
    rec.bytes = sizeBytes;
    rec.stream = reinterpret_cast<std::uint64_t>(stream);
    Session().record_memset(std::move(rec));
  }
  return rc;
}

hipError_t WrapHipLaunchKernel(const void *function_address, dim3 numBlocks,
                               dim3 dimBlocks, void **args,
                               size_t sharedMemBytes, hipStream_t stream) {
  Session().note_api(ApiKind::kCudaLaunchKernel);
  // The pending launch is queued BEFORE the forward: the kernel-dispatch
  // callback fires while the real hipLaunchKernel is still enqueueing (the
  // SDK processes the packet on its own thread), so recording after the
  // return races the dispatch. On failure the entry is removed again.
  PendingLaunch p;
  p.grid_x = numBlocks.x; p.grid_y = numBlocks.y; p.grid_z = numBlocks.z;
  p.block_x = dimBlocks.x; p.block_y = dimBlocks.y; p.block_z = dimBlocks.z;
  p.shared_mem_bytes = static_cast<std::uint32_t>(sharedMemBytes);
  p.stream = reinterpret_cast<std::uint64_t>(stream);
  p.kernel_params = args;
  p.seq = ++g_pending_seq;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    g_pending_launches.push_back(p);
  }
  const hipError_t rc =
      g_real.hipLaunchKernel(function_address, numBlocks, dimBlocks, args,
                             sharedMemBytes, stream);
  if (rc != hipSuccess) {
    std::lock_guard<std::mutex> lock(g_mu);
    for (auto it = g_pending_launches.begin(); it != g_pending_launches.end();
         ++it) {
      if (it->seq == p.seq) {
        g_pending_launches.erase(it);
        break;
      }
    }
  }
  return rc;
}

hipError_t WrapHipStreamSynchronize(hipStream_t stream) {
  Session().note_api(ApiKind::kCudaStreamSynchronize);
  Session().record_stream_sync(reinterpret_cast<std::uint64_t>(stream));
  return g_real.hipStreamSynchronize(stream);
}

hipError_t WrapHipDeviceSynchronize() {
  Session().note_api(ApiKind::kCudaStreamSynchronize);
  Session().record_stream_sync(0); // default stream handle 0
  return g_real.hipDeviceSynchronize();
}

//===--- intercept-table registration -----------------------------------------===//
// The HIP runtime hands its live dispatch table to rocprofiler-sdk, which
// invokes this callback; the tool stores the real entries and installs the
// wrappers above (Kerncap/intellikit pattern; attribution THIRD-PARTY.md).

void HipTableCallback(rocprofiler_intercept_table_t type, uint64_t lib_version,
                      uint64_t /*lib_instance*/, void **tables,
                      uint64_t num_tables, void * /*user_data*/) {
  if (type != ROCPROFILER_HIP_RUNTIME_TABLE || tables == nullptr ||
      num_tables == 0) {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: unexpected intercept table "
                 "(type=%d tables=%p n=%llu)\n",
                 static_cast<int>(type), static_cast<void *>(tables),
                 static_cast<unsigned long long>(num_tables));
    return;
  }
  auto *table = static_cast<HipDispatchTable *>(tables[0]);
  if (table == nullptr) {
    return;
  }
  g_real.hipMalloc = table->hipMalloc_fn;
  g_real.hipFree = table->hipFree_fn;
  g_real.hipMemcpy = table->hipMemcpy_fn;
  g_real.hipMemcpyAsync = table->hipMemcpyAsync_fn;
  g_real.hipMemset = table->hipMemset_fn;
  g_real.hipMemsetAsync = table->hipMemsetAsync_fn;
  g_real.hipLaunchKernel = table->hipLaunchKernel_fn;
  g_real.hipStreamSynchronize = table->hipStreamSynchronize_fn;
  g_real.hipDeviceSynchronize = table->hipDeviceSynchronize_fn;

  table->hipMalloc_fn = WrapHipMalloc;
  table->hipFree_fn = WrapHipFree;
  table->hipMemcpy_fn = WrapHipMemcpy;
  table->hipMemcpyAsync_fn = WrapHipMemcpyAsync;
  table->hipMemset_fn = WrapHipMemset;
  table->hipMemsetAsync_fn = WrapHipMemsetAsync;
  table->hipLaunchKernel_fn = WrapHipLaunchKernel;
  table->hipStreamSynchronize_fn = WrapHipStreamSynchronize;
  table->hipDeviceSynchronize_fn = WrapHipDeviceSynchronize;

  std::fprintf(stderr,
               "[kernscope-capture-hip] info: HIP runtime intercept table "
               "registered (lib v%llu.%llu.%llu)\n",
               static_cast<unsigned long long>(lib_version / 10000),
               static_cast<unsigned long long>((lib_version % 10000) / 100),
               static_cast<unsigned long long>(lib_version % 100));
}

//===--- callback tracing: kernel dispatch ------------------------------------===//
// Grid/workgroup + kernel identity at the HSA dispatch layer. The kernel name
// is resolved from the code-object kernel-symbol table (g_kernels). The
// pending HIP launch (FIFO) is completed here with the name; the full
// per-argument capture fills in from the code-object kernel metadata.

void KernelDispatchCallback(rocprofiler_callback_tracing_record_t record,
                            rocprofiler_user_data_t * /*user_data*/,
                            void * /*callback_data*/) {
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  auto *data = static_cast<rocprofiler_callback_tracing_kernel_dispatch_data_t *>(
      record.payload);
  if (data == nullptr) {
    return;
  }
  const rocprofiler_kernel_dispatch_info_t &info = data->dispatch_info;
  std::lock_guard<std::mutex> lock(g_mu);

  // The dispatch grid is in WORK-ITEMS (observed on ROCm 7.2.3: layernorm
  // M=1024 blocks x 128 threads -> grid_size=131072); the v0 LaunchRecord
  // grid is in BLOCKS (CUDA/HIP launch semantics), so convert by dividing
  // through the workgroup size (a zero workgroup dim means unknown -> 1).
  const auto blocks_of = [](std::uint64_t grid, std::uint64_t wg) {
    return wg > 0 ? static_cast<std::uint32_t>(grid / wg) : 1u;
  };
  const std::uint32_t blocks_x = blocks_of(info.grid_size.x, info.workgroup_size.x);
  const std::uint32_t blocks_y = blocks_of(info.grid_size.y, info.workgroup_size.y);
  const std::uint32_t blocks_z = blocks_of(info.grid_size.z, info.workgroup_size.z);

  // Merge with the pending HIP launch ONLY when it matches this dispatch
  // (grid/workgroup). The runtime's own internal dispatches (e.g. the
  // __amd_rocclr_copyBuffer kernels behind hipMemcpy) never pass through the
  // wrapped hipLaunchKernel, so they must not consume the pending queue.
  PendingLaunch pending{};
  bool have_pending = false;
  if (!g_pending_launches.empty()) {
    PendingLaunch &front = g_pending_launches.front();
    if (front.grid_x == blocks_x && front.grid_y == blocks_y &&
        front.grid_z == blocks_z && front.block_x == info.workgroup_size.x &&
        front.block_y == info.workgroup_size.y &&
        front.block_z == info.workgroup_size.z) {
      pending = front;
      g_pending_launches.erase(g_pending_launches.begin());
      have_pending = true;
    }
  }

  LaunchRecord launch;
  launch.via = LaunchVia::kRuntimeApi;
  const auto kit = g_kernels.find(info.kernel_id);
  if (kit != g_kernels.end()) {
    launch.kernel_name = kit->second.name;
  }
  // Runtime dispatch view (blocks) - the authoritative grid/workgroup from
  // the HSA dispatch layer; falls back to the HIP-side request.
  launch.grid_x = blocks_x;
  launch.grid_y = blocks_y;
  launch.grid_z = blocks_z;
  launch.block_x = info.workgroup_size.x;
  launch.block_y = info.workgroup_size.y;
  launch.block_z = info.workgroup_size.z;
  if (have_pending) {
    launch.shared_mem_bytes = pending.shared_mem_bytes;
    launch.stream = pending.stream;
  }
  // Per-argument byte-exact capture: the code object's AMDGPU metadata gives
  // each argument's size; the wrapped hipLaunchKernel exposed the host-side
  // argument pointer array (pending.kernel_params). Copy size bytes from each
  // pointer - the same layout the toolchain uses to build the kernarg
  // segment. Pointer-valued args whose pointee is NOT in the allocation map
  // trigger the size-unknown warning (QA negative), but the slot still
  // carries the raw pointer value bytes.
  const auto layout_it = g_arg_layouts.find(LayoutKey(launch.kernel_name));
  if (layout_it != g_arg_layouts.end() && have_pending &&
      pending.kernel_params != nullptr) {
    std::size_t idx = 0;
    for (const kernscope::AmdArgLayout &a : layout_it->second) {
      // Compiler-added hidden args (hidden_global_offset_x, printf buffers,
      // ...) occupy kernarg space but have NO kernel_params entry: the
      // hipLaunchKernel array covers user args only. Skipping them also
      // keeps the captured arg count == the user-callable parameter count.
      if (a.value_kind.rfind("hidden_", 0) == 0) {
        continue;
      }
      if (a.size == 0) {
        continue;
      }
      kernscope::KernelArgSlot slot;
      slot.size = static_cast<std::size_t>(a.size);
      if (idx < 64 && pending.kernel_params[idx] != nullptr) {
        const auto *bytes = static_cast<const std::uint8_t *>(
            pending.kernel_params[idx]);
        slot.bytes.assign(bytes, bytes + slot.size);
        if (a.is_pointer) {
          std::uint64_t pointee = 0;
          if (a.size >= sizeof(std::uint64_t)) {
            std::memcpy(&pointee, bytes, sizeof(pointee));
          }
          if (!Session().allocation_size(pointee).has_value()) {
            std::fprintf(stderr,
                         "[kernscope-capture-hip] warning: buffer "
                         "size-unknown (kernel arg #%zu, ptr=0x%llx): not "
                         "allocated via a captured HIP allocation API\n",
                         idx, static_cast<unsigned long long>(pointee));
          }
        }
      }
      launch.args.push_back(std::move(slot));
      ++idx;
    }
  } else if (kit != g_kernels.end() &&
             kit->second.kernarg_segment_size > 0) {
    // No metadata layout (or no pending launch): fall back to the raw
    // kernarg segment size as a single slot (sizes unavailable - bytes stay
    // empty). Applies to dispatch-only launches (runtime-internal kernels).
    kernscope::KernelArgSlot slot;
    slot.size = kit->second.kernarg_segment_size;
    launch.args.push_back(std::move(slot));
  }
  Session().record_launch(std::move(launch));
}

//===--- callback tracing: code objects ---------------------------------------===//
// Code-object LOAD: HSACO blob -> session module record (sha256) + on-disk
// blob. KERNEL_SYMBOL_REGISTER: kernel_id -> name (+ kernarg size) table and
// kernel name attachment to the module record.

void CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                        rocprofiler_user_data_t * /*user_data*/,
                        void * /*callback_data*/) {
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  switch (record.operation) {
  case ROCPROFILER_CODE_OBJECT_LOAD: {
    auto *data =
        static_cast<rocprofiler_callback_tracing_code_object_load_data_t *>(
            record.payload);
    if (data == nullptr) {
      return;
    }
    std::vector<std::uint8_t> blob;
    if (data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) {
      const int fd = data->storage_file;
      if (fd >= 0) {
        // Size from fstat, then a full pread (the fd is a regular file or
        // memfd; either way pread from offset 0 is valid).
        struct stat st;
        if (::fstat(fd, &st) == 0 && st.st_size > 0) {
          blob.resize(static_cast<std::size_t>(st.st_size));
          std::size_t got = 0;
          while (got < blob.size()) {
            const ssize_t n = ::pread(fd, blob.data() + got,
                                      blob.size() - got, got);
            if (n <= 0) {
              break;
            }
            got += static_cast<std::size_t>(n);
          }
          blob.resize(got);
        }
      }
    } else if (data->storage_type ==
               ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
      const auto *base = reinterpret_cast<const std::uint8_t *>(
          data->memory_base);
      const std::size_t size = static_cast<std::size_t>(data->memory_size);
      if (base != nullptr && size > 0) {
        blob.assign(base, base + size);
      }
    }
    if (blob.empty()) {
      std::fprintf(stderr,
                   "[kernscope-capture-hip] warning: code object %llu "
                   "unreadable (storage_type=%d)\n",
                   static_cast<unsigned long long>(data->code_object_id),
                   static_cast<int>(data->storage_type));
      return;
    }
    const std::vector<kernscope::AmdKernelLayout> layouts =
        kernscope::parse_amdgpu_metadata(blob);
    std::lock_guard<std::mutex> lock(g_mu);
    for (const kernscope::AmdKernelLayout &k : layouts) {
      g_arg_layouts[k.name] = k.args;
    }
    const std::uint64_t mod =
        Session().register_fatbinary(blob); // sha256 module record
    g_co_to_mod[data->code_object_id] = mod;
    WriteBlob("code-object-" + std::to_string(data->code_object_id) +
                  ".hsaco",
              blob.data(), blob.size());
    break;
  }
  case ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER:
  case ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER: {
    auto *data = static_cast<
        rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
        record.payload);
    if (data == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(g_mu);
    KernelSymbol sym;
    sym.name = data->kernel_name != nullptr ? data->kernel_name : "";
    sym.kernarg_segment_size = data->kernarg_segment_size;
    g_kernels[data->kernel_id] = std::move(sym);
    const auto it = g_co_to_mod.find(data->code_object_id);
    if (it != g_co_to_mod.end()) {
      (void)Session().register_function(it->second, data->kernel_name);
    }
    break;
  }
  default:
    break;
  }
}

//===--- flush (bundle emission) -----------------------------------------------===//
// Emits the v0 bundle through libs/capture-core's BundleWriter. Idempotent:
// runs at rocprofiler-sdk finalize and via atexit (whichever comes first).

std::atomic<bool> g_flushed{false};

void FlushBundle() {
  if (g_flushed.exchange(true)) {
    return;
  }
  const char *path = std::getenv("KERNSCOPE_HIP_BUNDLE");
  if (path == nullptr || path[0] == '\0') {
    path = std::getenv("KERNSCOPE_CAPTURE_LOG");
  }
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  kernscope::CaptureSession &session = Session();
  if (session.empty()) {
    return; // never litter for processes that made no HIP calls
  }
  std::ofstream out(path, std::ios::trunc);
  if (!out) {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: cannot write bundle %s\n",
                 path);
    return;
  }
  kernscope::BundleWriter::create_v0()->write(session, out);
  std::fprintf(stderr,
               "[kernscope-capture-hip] info: bundle written to %s\n", path);
}

void AtExitFlush() { FlushBundle(); }

void ToolFinalize(void * /*tool_data*/) { FlushBundle(); }

/// Contexts + services are created here, NOT in rocprofiler_configure:
/// rocprofiler-sdk assigns the tool's client identifier only after every
/// tool's rocprofiler_configure has returned, and rocprofiler_create_context
/// asserts on a missing client id (observed on ROCm 7.2.3 when the HIP
/// runtime's own table registration triggers lazy SDK initialization).
int ToolInitialize(rocprofiler_client_finalize_t /*finalize_func*/,
                   void * /*tool_data*/) {
  rocprofiler_context_id_t ctx{};
  if (rocprofiler_create_context(&ctx) != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "[kernscope-capture-hip] warning: context creation failed; "
                 "kernel-dispatch/code-object tracing disabled\n");
    return -1;
  }
  (void)rocprofiler_configure_callback_tracing_service(
      ctx, ROCPROFILER_CALLBACK_TRACING_KERNEL_DISPATCH, nullptr, 0,
      KernelDispatchCallback, nullptr);
  (void)rocprofiler_configure_callback_tracing_service(
      ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
      CodeObjectCallback, nullptr);
  (void)rocprofiler_start_context(ctx);
  return 0;
}

} // namespace

//===--- tool entry point ------------------------------------------------------===//

extern "C" rocprofiler_tool_configure_result_t *rocprofiler_configure(
    uint32_t /*version*/, const char * /*runtime_version*/, uint32_t /*priority*/,
    rocprofiler_client_id_t *id) {
  id->name = const_cast<char *>("kernscope-capture-hip");

  (void)rocprofiler_at_intercept_table_registration(
      HipTableCallback, ROCPROFILER_HIP_RUNTIME_TABLE, nullptr);

  std::atexit(AtExitFlush);

  static rocprofiler_tool_configure_result_t cfg = {
      sizeof(rocprofiler_tool_configure_result_t), ToolInitialize,
      ToolFinalize, nullptr};
  return &cfg;
}
