//===- cupti_tracer.cpp - CUPTI activity tracer (todo 49) -----------------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// Implementation of src/tracer_cupti.h. See that header + the versioned
// adapter (include/kernscope/profile/record_adapter.h) for the design and
// the CUDA-13 migration surface.
//
// The REUSE contract with libs/cupti-sub (todo 44, NEVER modified): every
// record from a delivered buffer is handed to
// kernscope::cupti::handle_activity_record() FIRST — the shared subscription
// library consumes the MEMORY-kind records into its ptr->size allocation map
// and ignores the rest ("everything else is ignored by the capture (the
// todo-49 profiler consumes the other kinds through this same lib)" — its
// own header). The tracer then parses the kernel/API/overhead records it
// cares about from the same records. One subscription, two consumers.
//
// The libcupti.so loading follows the todo-44 pattern (dlopen by path,
// dlsym per symbol; NOT a link dependency) so the negative path degrades
// gracefully on a driverless box: start_tracing() logs the SHARED fallback
// line (kernscope::cupti::kFallbackLogLine) and returns false.
//
//===----------------------------------------------------------------------===//

#include "tracer_cupti.h"

#include "kernscope/capture_cupti.h" // the todo-44 shared subscription library

#include <dlfcn.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace kernscope::profile::cupti {

namespace {

//===--- the dlopened CUPTI entry points -------------------------------------===//
// Resolved once at first start_tracing() call; kept for the process lifetime
// (dlclose would race CUPTI's worker thread).

struct CuptiApi {
  void *handle = nullptr;
  CUptiResult (*activity_register_callbacks)(
      CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc) = nullptr;
  CUptiResult (*activity_enable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*activity_disable)(CUpti_ActivityKind) = nullptr;
  CUptiResult (*activity_flush_all)(uint32_t) = nullptr;
  CUptiResult (*activity_get_num_dropped)(CUcontext, uint32_t, size_t *) = nullptr;
};
CuptiApi g_cupti{};

// The candidate libcupti.so paths (the todo-44 pattern): CUPTI_HOME/lib/
// first (devenv), then the loader's search path.
std::vector<const char *> CuptiLibCandidates() {
  static std::string home_path;
  if (const char *home = std::getenv("CUPTI_HOME");
      home != nullptr && home[0] != '\0') {
    home_path = std::string(home) + "/lib/libcupti.so";
  }
  return {home_path.c_str(), "libcupti.so.12", "libcupti.so"};
}

bool LoadCupti() {
  if (g_cupti.handle != nullptr) {
    return true;
  }
  void *handle = nullptr;
  for (const char *candidate : CuptiLibCandidates()) {
    if (candidate == nullptr || candidate[0] == '\0') {
      continue;
    }
    if (void *h = dlopen(candidate, RTLD_NOW | RTLD_LOCAL)) {
      handle = h;
      break;
    }
  }
  if (handle == nullptr) {
    return false;
  }
  g_cupti.handle = handle;
  g_cupti.activity_register_callbacks = reinterpret_cast<decltype(g_cupti.activity_register_callbacks)>(
      dlsym(handle, "cuptiActivityRegisterCallbacks"));
  g_cupti.activity_enable = reinterpret_cast<decltype(g_cupti.activity_enable)>(
      dlsym(handle, "cuptiActivityEnable"));
  g_cupti.activity_disable = reinterpret_cast<decltype(g_cupti.activity_disable)>(
      dlsym(handle, "cuptiActivityDisable"));
  g_cupti.activity_flush_all = reinterpret_cast<decltype(g_cupti.activity_flush_all)>(
      dlsym(handle, "cuptiActivityFlushAll"));
  g_cupti.activity_get_num_dropped = reinterpret_cast<decltype(g_cupti.activity_get_num_dropped)>(
      dlsym(handle, "cuptiActivityGetNumDroppedRecords"));
  if (g_cupti.activity_register_callbacks == nullptr ||
      g_cupti.activity_enable == nullptr || g_cupti.activity_flush_all == nullptr) {
    std::fprintf(stderr,
                 "[kernscope-profile-cupti] libcupti.so loaded but the activity "
                 "API symbols are missing\n");
    return false;
  }
  return true;
}

//===--- tracer state ---------------------------------------------------------===//

std::mutex g_mu; // guards the sinks + stats (CUPTI worker thread vs app)
bool g_active = false;
KernelSink g_kernel_sink;
ApiSink g_api_sink;
TracerStats g_stats;

void LogFallback(const char *detail) {
  std::fprintf(stderr, "[kernscope-profile-cupti] %s\n", detail);
  // The shared todo-44 degradation contract: the same fallback line the
  // subscription library logs when CUPTI is unavailable.
  std::fprintf(stderr, "%s\n", kernscope::cupti::kFallbackLogLine);
}

//===--- the CUPTI buffer callbacks (C linkage, CUPTI worker thread) ----------===//

void BufferRequested(std::uint8_t **buffer, std::size_t *size, std::size_t * /*unused*/) {
  if (buffer == nullptr || size == nullptr) {
    return;
  }
  *buffer = static_cast<std::uint8_t *>(std::malloc(kBufferSize));
  if (*buffer == nullptr) {
    *size = 0;
    return;
  }
  std::memset(*buffer, 0, kBufferSize);
  *size = kBufferSize;
}

void BufferCompleted(CUcontext /*context*/, std::uint32_t /*streamId*/,
                     std::uint8_t *buffer, std::size_t /*size*/,
                     std::size_t valid_size) {
  if (buffer == nullptr) {
    return;
  }
  // Records to deliver: every record first through the todo-44 shared
  // subscription library (MEMORY records feed its allocation map; everything
  // else is ignored by it), then through the tracer's own dispatch.
  const BufferIterationResult iter = iterate_activity_buffer(
      buffer, valid_size,
      [](CUpti_ActivityKind kind, const CUpti_Activity *record, std::size_t) {
        kernscope::cupti::handle_activity_record(record); // the REUSE
        switch (kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
        case CUPTI_ACTIVITY_KIND_KERNEL: { // kind 3 is never ENABLED; parse if seen
          KernelRecord out;
          if (parse_cupti_kernel_record(
                  reinterpret_cast<const ActivityKernelT *>(record), &out)) {
            std::lock_guard<std::mutex> lock(g_mu);
            ++g_stats.kernels;
            if (g_kernel_sink) {
              g_kernel_sink(out);
            }
          }
          break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_DRIVER: {
          const auto *api = reinterpret_cast<const CUpti_ActivityAPI *>(record);
          ApiRecord rec;
          rec.kind = api->kind;
          rec.cbid = api->cbid;
          rec.start_ns = api->start;
          rec.end_ns = api->end;
          rec.process_id = api->processId;
          rec.thread_id = api->threadId;
          rec.correlation_id = api->correlationId;
          rec.return_value = api->returnValue;
          std::lock_guard<std::mutex> lock(g_mu);
          ++g_stats.apis;
          if (g_api_sink) {
            g_api_sink(rec);
          }
          break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD: {
          std::lock_guard<std::mutex> lock(g_mu);
          ++g_stats.overheads;
          break;
        }
        default: // MEMORY + anything else: already routed to cupti-sub above
          break;
        }
      });

  {
    std::lock_guard<std::mutex> lock(g_mu);
    ++g_stats.buffers_delivered;
    g_stats.records_total += iter.records_parsed;
    g_stats.malformed += iter.malformed;
  }
  std::free(buffer);
}

} // namespace

//===--- public API ------------------------------------------------------------===//

bool start_tracing(KernelSink kernel_sink, ApiSink api_sink, TracerStats *stats) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_active) {
    if (stats != nullptr) {
      *stats = g_stats;
    }
    return true; // idempotent
  }
  if (!LoadCupti()) {
    LogFallback("cannot dlopen libcupti.so (no CUPTI runtime on this host?)");
    return false;
  }
  const CUptiResult reg =
      g_cupti.activity_register_callbacks(BufferRequested, BufferCompleted);
  if (reg != CUPTI_SUCCESS) {
    LogFallback("cuptiActivityRegisterCallbacks failed");
    return false;
  }
  for (std::size_t i = 0; i < kEnabledKindCount; ++i) {
    if (g_cupti.activity_enable(kEnabledKinds[i]) != CUPTI_SUCCESS) {
      LogFallback("cuptiActivityEnable failed (kind not supported?)");
      return false;
    }
  }
  g_kernel_sink = std::move(kernel_sink);
  g_api_sink = std::move(api_sink);
  g_active = true;
  if (stats != nullptr) {
    *stats = g_stats;
  }
  return true;
}

bool tracing_active() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_active;
}

void flush() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_active || g_cupti.activity_flush_all == nullptr) {
    return;
  }
  (void)g_cupti.activity_flush_all(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
}

void stop() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_active) {
    return;
  }
  if (g_cupti.activity_flush_all != nullptr) {
    (void)g_cupti.activity_flush_all(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
  }
  if (g_cupti.activity_get_num_dropped != nullptr) {
    size_t dropped = 0;
    if (g_cupti.activity_get_num_dropped(nullptr, 0, &dropped) == CUPTI_SUCCESS) {
      g_stats.dropped = dropped;
    }
  }
  if (g_cupti.activity_disable != nullptr) {
    for (std::size_t i = 0; i < kEnabledKindCount; ++i) {
      (void)g_cupti.activity_disable(kEnabledKinds[i]);
    }
  }
  g_kernel_sink = nullptr;
  g_api_sink = nullptr;
  g_active = false;
}

TracerStats stats_snapshot() {
  std::lock_guard<std::mutex> lock(g_mu);
  return g_stats;
}

} // namespace kernscope::profile::cupti
