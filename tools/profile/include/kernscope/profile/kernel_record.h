//===- kernel_record.h - vendor-neutral profile record model (todo 49) ---*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// The vendor-neutral per-kernel profile record + JSONL emission shared by
// BOTH tracers: the CUPTI tracer fills it through the versioned adapter
// (record_adapter.h), the AMD tracer fills the same fields from
// rocprofiler_kernel_dispatch_info_t. Deliberately CUPTI-free (no CUDA
// headers): the AMD tool compiles this header together with HIP/rocprofiler
// headers, whose vector types would collide with CUDA's (cuda.h is pulled in
// by cupti.h) — keeping the record model separate is what makes the AMD
// tracer buildable without that collision.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_KERNEL_RECORD_H
#define KERNSCOPE_PROFILE_KERNEL_RECORD_H

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>

namespace kernscope::profile {

/// The single record shape both tracers emit (CUPTI on NVIDIA, buffer-tracing
/// on AMD). Field correspondence:
///   NVIDIA: CUpti_ActivityKernel9/10 via record_adapter.h
///   AMD:    rocprofiler_kernel_dispatch_info_t — start/end_timestamp,
///           correlation_id, workgroup_size -> block_*, grid_size (WORK
///           ITEMS) / workgroup_size -> grid_* (blocks, the CUDA/HIP launch
///           semantics), group_segment_size -> LDS (static+dynamic),
///           private_segment_size -> scratch per work-item.
struct KernelRecord {
  std::string kernel_name;          // mangled kernel name
  std::uint64_t start_ns = 0;       // GPU timestamps (ns); 0/0 = not collected
  std::uint64_t end_ns = 0;
  std::uint64_t completed_ns = 0;   // incl. child kernels (CUDA Dynamic Parallelism)
  std::uint64_t duration_ns = 0;    // end - start (0 when no timestamps)
  std::int64_t grid_x = 1, grid_y = 1, grid_z = 1;     // grid dims (blocks)
  std::int64_t block_x = 1, block_y = 1, block_z = 1;  // block dims (threads)
  std::int32_t static_shared_bytes = 0;   // compile-time smem
  std::int32_t dynamic_shared_bytes = 0;  // launch-time smem
  std::uint32_t shared_memory_executed = 0; // smem actually set by the driver
  std::uint16_t registers_per_thread = 0;
  std::uint32_t local_memory_per_thread = 0; // register-spill indicator
  std::uint64_t local_memory_total = 0;      // Kernel9: localMemoryTotal_v2
  std::uint32_t correlation_id = 0; // joins kernel record <-> API record
  std::uint32_t device_id = 0, context_id = 0, stream_id = 0;
  std::int64_t grid_id = 0;
  std::uint8_t launch_type = 0;     // regular vs cooperative
  std::uint8_t is_device_launched = 0; // Kernel10 only (CUDA 13); 0 on 12.x
};

//===--- JSONL emission (shared by both tracers' sinks) ------------------------===//
// One line per record, same shape for NVIDIA and AMD (vendor field differs).
// Used by the AMD live check (tests/profile_amd_check.py parses it).

inline std::string json_escape(const std::string &s) {
  std::string out;
  out.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
    case '"': out += "\\\""; break;
    case '\\': out += "\\\\"; break;
    case '\n': out += "\\n"; break;
    case '\t': out += "\\t"; break;
    default: out += c; break;
    }
  }
  return out;
}

inline std::string kernel_record_to_json(const char *vendor,
                                         const KernelRecord &r) {
  char buf[512];
  const int n = std::snprintf(
      buf, sizeof(buf),
      "{\"vendor\":\"%s\",\"kind\":\"kernel\",\"name\":\"%s\","
      "\"start_ns\":%llu,\"end_ns\":%llu,\"duration_ns\":%llu,"
      "\"grid\":[%lld,%lld,%lld],\"workgroup\":[%lld,%lld,%lld],"
      "\"static_shared_bytes\":%d,\"dynamic_shared_bytes\":%d,"
      "\"shared_memory_executed\":%u,\"registers_per_thread\":%u,"
      "\"local_memory_per_thread\":%u,\"local_memory_total\":%llu,"
      "\"correlation_id\":%u,\"device_id\":%u,\"context_id\":%u,"
      "\"stream_id\":%u,\"grid_id\":%lld,\"launch_type\":%u,"
      "\"is_device_launched\":%u}",
      vendor, json_escape(r.kernel_name).c_str(),
      static_cast<unsigned long long>(r.start_ns),
      static_cast<unsigned long long>(r.end_ns),
      static_cast<unsigned long long>(r.duration_ns),
      static_cast<long long>(r.grid_x), static_cast<long long>(r.grid_y),
      static_cast<long long>(r.grid_z), static_cast<long long>(r.block_x),
      static_cast<long long>(r.block_y), static_cast<long long>(r.block_z),
      r.static_shared_bytes, r.dynamic_shared_bytes,
      r.shared_memory_executed, r.registers_per_thread,
      r.local_memory_per_thread,
      static_cast<unsigned long long>(r.local_memory_total),
      r.correlation_id, r.device_id, r.context_id, r.stream_id,
      static_cast<long long>(r.grid_id), r.launch_type,
      r.is_device_launched);
  return (n > 0) ? std::string(buf, static_cast<std::size_t>(n)) : std::string();
}

} // namespace kernscope::profile

#endif // KERNSCOPE_PROFILE_KERNEL_RECORD_H
