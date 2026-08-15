//===- tracer_cupti.h - CUPTI activity tracer (todo 49) ------------------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// NOTE ON THE NAME (todo-49 CI-gate fix): this header used to be named
// `cupti_tracer` + `.h` — the NVIDIA-header filename guard in scripts/ci.sh
// (regex `(^|/)(cuda_occupancy|cupti_[a-z_]*|cuda[a-z_]*|nvtx[a-z_]*)\.h$`)
// matched the old basename as if it were a vendored NVIDIA header, a FALSE
// POSITIVE on the merged-tree gate. Same fix pattern as todo 44's
// `cupti_capture` -> `capture_cupti` rename (commit 667840d): keep the guard
// as-is, rename the project header so the basename no longer starts with
// `cupti_`. The .cpp keeps its name (the guard only matches `.h$`).
//
// The NVIDIA tracer: CUPTI activity tracing with the kineto enable-set and
// the todo-44 shared subscription library (libs/cupti-sub, NEVER modified)
// on the MEMORY-record side:
//
//   1. Subscription: cuptiActivityRegisterCallbacks (buffer request/completed)
//      + cuptiActivityEnable for CONCURRENT_KERNEL (kind 10, low-overhead —
//      NEVER the serializing KERNEL kind 3) + RUNTIME + DRIVER + OVERHEAD —
//      the kineto enable-set (docs/research/profiler-research-brief.md §1.1,
//      §1.3). No cuptiSubscribe: activity-only tracing needs no subscriber.
//   2. Buffer completed -> iterate_activity_buffer (the versioned adapter's
//      iterator, mirroring cuptiActivityGetNextRecord) and per record:
//      - MEMORY records -> kernscope::cupti::handle_activity_record (the
//        REUSE: the shared lib feeds its ptr->size allocation map and ignores
//        everything else — the todo-44 header documents exactly this split);
//      - kernel records (CONCURRENT_KERNEL) -> the versioned adapter
//        (record_adapter.h) -> KernelRecord sink;
//      - RUNTIME/DRIVER records -> ApiRecord sink (correlation_id joins them
//        to kernel records);
//      - OVERHEAD records -> counted (kineto enables them for attribution);
//      - truncated/unknown records -> reported malformed + skipped (QA
//        negative, asserted by tests/cupti_tracer_test.cpp).
//   3. Flush: cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) at
//      stop — the kineto stop pattern (forced flush returns buffers with
//      incomplete records too).
//
// Negative path (first-class, same contract as todo 44): libcupti.so is
// dlopened (NOT a link dependency); on a driverless box or with a second
// subscriber active, start_tracing() logs kernscope::cupti::kFallbackLogLine
// and returns false — never crashes, never hangs.
//
// Compile+unit ONLY locally (cuda_cupti 12.6 headers); real-NVIDIA behavior
// is rental-gated (scripts/rental-validate.sh).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_CUPTI_TRACER_H
#define KERNSCOPE_PROFILE_CUPTI_TRACER_H

#include "kernscope/profile/record_adapter.h"

#include <cstddef>
#include <cstdint>
#include <functional>

namespace kernscope::profile::cupti {

/// The kineto enable-set (brief §1.1/§1.3): CONCURRENT_KERNEL (kind 10) +
/// RUNTIME + DRIVER + OVERHEAD. CUPTI_ACTIVITY_KIND_KERNEL (3) is NEVER
/// enabled — it serializes kernel executions on the GPU (a behavior change,
/// not just overhead). CONCURRENT_KERNEL only.
inline constexpr CUpti_ActivityKind kEnabledKinds[] = {
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL,
    CUPTI_ACTIVITY_KIND_RUNTIME,
    CUPTI_ACTIVITY_KIND_DRIVER,
    CUPTI_ACTIVITY_KIND_OVERHEAD,
};
inline constexpr std::size_t kEnabledKindCount = 4;

/// Activity buffer size (8 MiB — the cuptisamples activity_trace_async flow).
inline constexpr std::size_t kBufferSize = 8u << 20;

/// One RUNTIME/DRIVER activity record — joins kernel records by correlationId.
struct ApiRecord {
  CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_RUNTIME;
  CUpti_CallbackId cbid = 0; // the API function id (cupti_runtime_cbid.h / _driver_cbid.h)
  std::uint64_t start_ns = 0;
  std::uint64_t end_ns = 0;
  std::uint32_t process_id = 0;
  std::uint32_t thread_id = 0;
  std::uint32_t correlation_id = 0;
  std::uint32_t return_value = 0;
};

/// Tracer counters — surfaced to tests and the future CLI (todo 51+).
struct TracerStats {
  std::size_t buffers_delivered = 0; // completed buffers processed
  std::size_t records_total = 0;     // well-formed records seen
  std::size_t kernels = 0;           // CONCURRENT_KERNEL records
  std::size_t apis = 0;              // RUNTIME/DRIVER records
  std::size_t overheads = 0;         // OVERHEAD records
  std::size_t malformed = 0;         // truncated/unknown-kind (skipped)
  std::size_t dropped = 0;           // CUPTI dropped-record count at flush
};

using KernelSink = std::function<void(const KernelRecord &)>;
using ApiSink = std::function<void(const ApiRecord &)>;

/// Establish the activity subscription (dlopen libcupti, register the buffer
/// callbacks, enable the kineto kinds). On ANY failure logs the todo-44
/// fallback line and returns false — never crashes, never hangs. Safe to
/// call more than once (idempotent). `kernel_sink`/`api_sink` are invoked
/// from the CUPTI worker thread while buffers are delivered.
bool start_tracing(KernelSink kernel_sink, ApiSink api_sink, TracerStats *stats);

/// True once the activity subscription is live.
bool tracing_active();

/// cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED) — delivers all
/// pending buffers including those with incomplete records (kineto stop
/// pattern). Called implicitly by stop().
void flush();

/// Disable the kinds + forced flush. Idempotent.
void stop();

/// Copy of the running counters (thread-safe snapshot).
TracerStats stats_snapshot();

} // namespace kernscope::profile::cupti

#endif // KERNSCOPE_PROFILE_CUPTI_TRACER_H
