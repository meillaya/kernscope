//===- record_adapter.h - versioned CUPTI record adapter (todo 49) -*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// This header is THE todo-49 deliverable: the VERSIONED ADAPTER that pins the
// record parsing to CUpti_ActivityKernel9 (the CUDA 12.6 struct, Metis L-5)
// while DOCUMENTING the CUDA-13 migration surface (Metis H-2). It is the ONLY
// place in the profile track that reads raw CUPTI activity records, so a
// header bump (12.6 -> 13.x) shows up as a compile error here instead of a
// silent parse shift.
//
// ## CUPTI 12.x vs 13.x kernel records
//
// CUDA 12.x delivers CUpti_ActivityKernel9 for both CUPTI_ACTIVITY_KIND_KERNEL
// (3) and CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL (10); CUDA 13.x delivers
// CUpti_ActivityKernel10 (the 12.x struct moved to cupti_activity_deprecated.h,
// docs/research/profiler-research-brief.md §1.2/§1.5). The two kinds use the
// SAME record struct; they differ only in collection behavior - KERNEL
// SERIALIZES all kernel executions on the GPU (a behavior change), while
// CONCURRENT_KERNEL does not. This tracer ENABLES CONCURRENT_KERNEL ONLY
// (MUST NOT serialize kernels; kineto does the same).
//
// Kernel10 == Kernel9 + one new field at the TAIL:
//
//     uint8_t isDeviceLaunched;   // + 3 padding bytes to restore alignment
//
// Every field this adapter reads sits at an IDENTICAL offset in both structs
// (pinned by the static_asserts below), so the same parser works against
// either header. That is the versioning mechanism:
// `parse_cupti_kernel_record` is a template over the delivered struct
// (member-name access), instantiated with the build's own `ActivityKernelT`
// typedef below; `isDeviceLaunched` is read through SFINAE member detection,
// so a CUDA-12 build compiles against Kernel9 (field absent ->
// is_device_launched stays 0) and a CUDA-13 build compiles against Kernel10
// (field read). The unit test pins the mapping with a Kernel10View mirror
// struct (tests/cupti_tracer_test.cpp).
//
// ## CUDA-13 migration surface (documented; nothing here calls it)
//
// - CUPTI Event API (cupti_events.h) and Metric API (cupti_metrics.h) are
//   REMOVED in CUDA 13 (calls return CUPTI_ERROR_LEGACY_PROFILER_NOT_SUPPORTED
//   = 38). This tracer MUST NOT use them; all profiling data comes from the
//   Activity API only (per-launch fields: timestamps, grid/block, regs,
//   static+dynamic smem, local memory, correlation id).
// - The PC-sampling ACTIVITY kind is removed (replaced by cupti_pcsampling.h);
//   nvprof and NVIDIA Visual Profiler are removed entirely in CUDA 13.0.
// - The Profiling API (cupti_profiler_target.h) and the PerfWorks Metric
//   API's NVIDIA host header are deprecated -> CUPTI Range Profiling API
//   (cupti_range_profiler.h) + Profiler Host API (cupti_profiler_host.h).
// - Single-subscriber scope (12.x) -> multi-subscriber V2 APIs
//   (cuptiSubscribe_v2, cuptiActivityEnable_v2, cuptiActivityRegisterCallbacks_v2)
//   arrive with CUDA 13.3+; the 12.x subscription path used here degrades to
//   the fallback when a second subscriber is active (the todo-44 single-tool
//   scope contract, libs/cupti-sub).
// - CUPTI_API_VERSION: 24 == CUDA 12.6; CUDA 13.0 reports 130001 (brief §1.5).
//
// ## Record iteration contract
//
// CUPTI records have NO self-describing size field: the size is implied by
// the kind (activity_record_size below, pinned with sizeof on the build's own
// structs). This header implements `iterate_activity_buffer` - the same
// semantics as cuptiActivityGetNextRecord (12.6 header: iterate until
// MAX_LIMIT_REACHED, stop on error) - so the parser is unit-testable without
// libcupti AND so a truncated record inside a delivered buffer is DETECTABLE:
// the iterator reports it as malformed and SKIPS the remainder of the buffer
// (QA negative contract: assert + skip, never crash). Records are packed at
// 8-byte alignment (every parsed record struct's largest member is uint64_t;
// the synthetic fixture in the unit test pins this convention).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_RECORD_ADAPTER_H
#define KERNSCOPE_PROFILE_RECORD_ADAPTER_H

#include "kernel_record.h" // the vendor-neutral record model + JSONL helpers

#include <cupti.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace kernscope::profile {

//===--- which struct the build's CUPTI delivers ------------------------------===//
// 12.6 headers (this repo's local compile surface) define Kernel9; CUDA 13
// headers deliver Kernel10. The parser is a template over this typedef, so
// swapping the build's CUPTI headers is the ONLY change a 12->13 bump needs.
#if CUPTI_API_VERSION > 24  // 24 == CUDA 12.6; 13.0 reports 130001
using ActivityKernelT = CUpti_ActivityKernel10;
#else
using ActivityKernelT = CUpti_ActivityKernel9;
#endif

//===--- SFINAE member detection: the Kernel10 tail field ---------------------===//
// isDeviceLaunched exists in Kernel10 only; read it when the delivered type
// has it, else report 0. This is what makes one parser source compile against
// both struct generations (the unit test's Kernel10View exercises the
// present-member branch on 12.6 headers).

template <typename KernelT, typename = void>
struct has_is_device_launched : std::false_type {};
template <typename KernelT>
struct has_is_device_launched<
    KernelT, std::void_t<decltype(std::declval<const KernelT &>().isDeviceLaunched)>>
    : std::true_type {};

//===--- the Kernel9 <-> Kernel10 field map (documented + pinned) --------------===//
// The fields the adapter reads, in struct order, with their offsets in the
// CUDA 12.6 Kernel9 layout (verified against the vendored cuda_cupti 12.6
// headers). Every entry is pinned by the static_asserts below, so a CUPTI
// header bump that MOVES a field fails the compile instead of mis-parsing.
// CUDA-13 delta summarized: NO field moved, NO field removed, ONE field added
// at the tail (isDeviceLaunched, uint8_t + 3 pad bytes -> sizeof 212).

struct KernelFieldEntry {
  const char *name;       // vendor member name
  std::size_t offset;     // offsetof in the 12.6 Kernel9 layout (pinned)
};

inline constexpr KernelFieldEntry kKernelFieldMap[] = {
    {"kind", 0},
    {"registersPerThread", 6},
    {"start", 16},
    {"end", 24},
    {"completed", 32},
    {"deviceId", 40},
    {"contextId", 44},
    {"streamId", 48},
    {"gridX", 52},
    {"gridY", 56},
    {"gridZ", 60},
    {"blockX", 64},
    {"blockY", 68},
    {"blockZ", 72},
    {"staticSharedMemory", 76},
    {"dynamicSharedMemory", 80},
    {"localMemoryPerThread", 84},
    {"correlationId", 92},
    {"gridId", 96},
    {"name", 104},
    {"launchType", 136},
    {"sharedMemoryExecuted", 140},
    {"localMemoryTotal_v2", 192},
};

/// The Kernel10-only tail field (CUDA 13): the structural delta the adapter
/// documents. Kernel10 == Kernel9 + isDeviceLaunched + 3 pad bytes.
inline constexpr struct {
  const char *name = "isDeviceLaunched";
  std::size_t size = sizeof(std::uint8_t); // + 3 pad bytes (8-byte alignment)
  bool added_in_13 = true;   // new field, Kernel10 (CUDA 13)
  bool moved_in_13 = false;  // no Kernel9 field moved
  bool removed_in_13 = false; // no Kernel9 field removed
} kKernel10TailDelta{};

// Layout pins (Metis L-5): every field the adapter reads must sit at the
// documented offset in BOTH struct generations (Kernel9 in 12.6, Kernel10 in
// CUDA 13 - identical offsets; only the tail differs). A header bump that
// reorders the struct fails here, loudly, at compile time.
#define KERNSCOPE_PIN_OFFSET(FIELD, OFF)                                       \
  static_assert(offsetof(ActivityKernelT, FIELD) == (OFF),                      \
                "kernscope profile: Kernel9/10 layout pin for " #FIELD)
KERNSCOPE_PIN_OFFSET(kind, 0);
KERNSCOPE_PIN_OFFSET(registersPerThread, 6);
KERNSCOPE_PIN_OFFSET(start, 16);
KERNSCOPE_PIN_OFFSET(end, 24);
KERNSCOPE_PIN_OFFSET(completed, 32);
KERNSCOPE_PIN_OFFSET(deviceId, 40);
KERNSCOPE_PIN_OFFSET(contextId, 44);
KERNSCOPE_PIN_OFFSET(streamId, 48);
KERNSCOPE_PIN_OFFSET(gridX, 52);
KERNSCOPE_PIN_OFFSET(gridY, 56);
KERNSCOPE_PIN_OFFSET(gridZ, 60);
KERNSCOPE_PIN_OFFSET(blockX, 64);
KERNSCOPE_PIN_OFFSET(blockY, 68);
KERNSCOPE_PIN_OFFSET(blockZ, 72);
KERNSCOPE_PIN_OFFSET(staticSharedMemory, 76);
KERNSCOPE_PIN_OFFSET(dynamicSharedMemory, 80);
KERNSCOPE_PIN_OFFSET(localMemoryPerThread, 84);
KERNSCOPE_PIN_OFFSET(correlationId, 92);
KERNSCOPE_PIN_OFFSET(gridId, 96);
KERNSCOPE_PIN_OFFSET(name, 104);
KERNSCOPE_PIN_OFFSET(launchType, 136);
KERNSCOPE_PIN_OFFSET(sharedMemoryExecuted, 140);
KERNSCOPE_PIN_OFFSET(localMemoryTotal_v2, 192);
#undef KERNSCOPE_PIN_OFFSET

//===--- parse: CUPTI kernel record -> KernelRecord ----------------------------===//
// Template over the DELIVERED struct (ActivityKernelT on the build's headers,
// Kernel10View in the mapping test). Member-name access keeps it compiling
// against both struct generations — the versioned-adapter mechanism.

template <typename KernelT>
inline bool parse_cupti_kernel_record(const KernelT *raw, KernelRecord *out) {
  if (raw == nullptr || out == nullptr) {
    return false;
  }
  out->kernel_name = raw->name != nullptr ? raw->name : "";
  out->start_ns = raw->start;
  out->end_ns = raw->end;
  out->completed_ns = raw->completed;
  out->duration_ns = raw->end >= raw->start ? raw->end - raw->start : 0;
  out->grid_x = raw->gridX;
  out->grid_y = raw->gridY;
  out->grid_z = raw->gridZ;
  out->block_x = raw->blockX;
  out->block_y = raw->blockY;
  out->block_z = raw->blockZ;
  out->static_shared_bytes = raw->staticSharedMemory;
  out->dynamic_shared_bytes = raw->dynamicSharedMemory;
  out->shared_memory_executed = raw->sharedMemoryExecuted;
  out->registers_per_thread = raw->registersPerThread;
  out->local_memory_per_thread = raw->localMemoryPerThread;
  out->local_memory_total = raw->localMemoryTotal_v2;
  out->correlation_id = raw->correlationId;
  out->device_id = raw->deviceId;
  out->context_id = raw->contextId;
  out->stream_id = raw->streamId;
  out->grid_id = raw->gridId;
  out->launch_type = raw->launchType;
  if constexpr (has_is_device_launched<KernelT>::value) {
    out->is_device_launched = raw->isDeviceLaunched; // Kernel10 (CUDA 13)
  }
  return true;
}

/// The one-line entry: parse the build's own delivered struct (no template
/// argument needed at call sites).
inline bool parse_kernel_record(const ActivityKernelT *raw, KernelRecord *out) {
  return parse_cupti_kernel_record<ActivityKernelT>(raw, out);
}

//===--- per-kind record sizes (no self-describing size in CUPTI) --------------===//
// Only the kinds the tracer enables (kineto pattern) + MEMORY (routed through
// the todo-44 shared lib) are sized; any other kind reports unknown so the
// buffer iterator can flag it as malformed instead of mis-parsing.

inline bool activity_record_size(CUpti_ActivityKind kind, std::size_t *size) {
  switch (kind) {
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
  case CUPTI_ACTIVITY_KIND_KERNEL: // never ENABLED (serializes); parsed if seen
    *size = sizeof(ActivityKernelT);
    return true;
  case CUPTI_ACTIVITY_KIND_DRIVER:
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    *size = sizeof(CUpti_ActivityAPI);
    return true;
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    *size = sizeof(CUpti_ActivityOverhead);
    return true;
  case CUPTI_ACTIVITY_KIND_MEMORY: // MEMORY records route to cupti-sub
    *size = sizeof(CUpti_ActivityMemory4);
    return true;
  default:
    return false; // unknown/unsupported kind
  }
}

//===--- buffer iteration (mirrors cuptiActivityGetNextRecord) -----------------===//
// 8-byte record alignment (see file header); a record that does not fit in
// the remaining bytes is TRUNCATED -> reported as malformed and the iterator
// STOPS (skips the remainder) — the asserted QA-negative behavior.

struct BufferIterationResult {
  std::size_t records_parsed = 0; // well-formed records delivered to visitor
  std::size_t malformed = 0;      // truncated/unknown-kind records skipped
  bool ok = true;                 // false when the iteration was cut short
};

template <typename Visitor>
inline BufferIterationResult iterate_activity_buffer(const std::uint8_t *buffer,
                                                     std::size_t size,
                                                     Visitor &&visitor) {
  BufferIterationResult result;
  constexpr std::size_t kAlign = 8;
  std::size_t pos = 0;
  while (pos + sizeof(CUpti_ActivityKind) <= size) {
    CUpti_ActivityKind kind;
    std::memcpy(&kind, buffer + pos, sizeof(kind));
    std::size_t record_size = 0;
    if (!activity_record_size(kind, &record_size) ||
        pos + record_size > size) {
      // Truncated or unknown-kind record: report + skip the remainder.
      ++result.malformed;
      result.ok = false;
      break;
    }
    visitor(kind, reinterpret_cast<const CUpti_Activity *>(buffer + pos),
            record_size);
    ++result.records_parsed;
    const std::size_t advance = (record_size + kAlign - 1) / kAlign * kAlign;
    pos += advance;
  }
  return result;
}

} // namespace kernscope::profile

#endif // KERNSCOPE_PROFILE_RECORD_ADAPTER_H
