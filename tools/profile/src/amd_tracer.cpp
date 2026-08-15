//===- amd_tracer.cpp - rocprofiler-sdk buffer-tracing tool (todo 49) ----*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// libkernscope_profile_hip.so: the LIVE local AMD tracer. A rocprofiler-sdk
// TOOL (exports rocprofiler_configure) that profiles any HIP application
// through BUFFER TRACING (async, low-overhead — records delivered by the
// SDK's worker thread, mirroring the todo-45 capture-hip usage pattern):
//
//   1. BUFFER TRACING KERNEL_DISPATCH: every kernel dispatch yields a
//      rocprofiler_buffer_tracing_kernel_dispatch_record_t with
//      start/end_timestamp (ns), correlation_id, thread_id and the
//      rocprofiler_kernel_dispatch_info_t (agent/queue/kernel/dispatch ids,
//      private_segment_size = scratch/work-item, group_segment_size = LDS per
//      work-group (static+dynamic), workgroup_size, grid_size in WORK ITEMS).
//      Grid is converted to BLOCKS (the CUDA/HIP launch semantics, same
//      conversion as todo-45): blocks = grid_size / workgroup_size per dim.
//   2. BUFFER TRACING HIP_RUNTIME_API: the HIP API calls that enqueue work,
//      each with its correlation_id — the JOIN KEY: a kernel dispatch
//      record's correlation_id identifies the API call that enqueued it, so
//      the acceptance asserts the layernorm kernel record links to the
//      hipLaunchKernel API record through the shared correlation_id.
//   3. CALLBACK TRACING CODE_OBJECT (kernel-symbol register): kernel_id ->
//      kernel_name map (buffer-tracing dispatch records carry only the
//      kernel_id; the name comes from the loaded code object's symbol table —
//      the same join the todo-45 tool performs).
//
// Output: one JSONL line per record to KERNSCOPE_PROFILE_OUT (default:
// stdout) in the shared record shape of the profile track (vendor "amd").
//
// Loading (same as todo-45): LD_PRELOAD=<this .so> <app> or
// ROCPROFILER_LIBRARY=<this .so> <app>.
//
// MUST NOT use roctracer (deprecated); MUST NOT use nvprof-style counters —
// this tracer is pure buffer-tracing (timestamps + dispatch info).
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/kernel_record.h" // vendor-neutral model + JSONL

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

//===--- output ---------------------------------------------------------------===//
// JSONL to KERNSCOPE_PROFILE_OUT (default stdout). Opened lazily on the first
// record; flushed + closed at finalize.

std::mutex g_out_mu;
std::FILE *g_out = nullptr;

std::FILE *Output() {
  if (g_out == nullptr) {
    const char *path = std::getenv("KERNSCOPE_PROFILE_OUT");
    g_out = (path != nullptr && path[0] != '\0') ? std::fopen(path, "w")
                                                 : stdout;
  }
  return g_out;
}

void WriteJsonl(const char *line) {
  std::lock_guard<std::mutex> lock(g_out_mu);
  if (std::FILE *out = Output(); out != nullptr) {
    std::fputs(line, out);
    std::fputc('\n', out);
  }
}

//===--- kernel_id -> name table (code-object kernel-symbol callbacks) ---------===//

std::mutex g_kernels_mu;
std::unordered_map<std::uint64_t, std::string> g_kernels; // kernel_id -> name

void CodeObjectCallback(rocprofiler_callback_tracing_record_t record,
                        rocprofiler_user_data_t * /*user_data*/,
                        void * /*callback_data*/) {
  if (record.phase != ROCPROFILER_CALLBACK_PHASE_ENTER) {
    return;
  }
  if (record.operation != ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER &&
      record.operation != ROCPROFILER_CODE_OBJECT_HOST_KERNEL_SYMBOL_REGISTER) {
    return;
  }
  auto *data = static_cast<
      rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t *>(
      record.payload);
  if (data == nullptr) {
    return;
  }
  const char *name = data->kernel_name != nullptr ? data->kernel_name : "";
  std::lock_guard<std::mutex> lock(g_kernels_mu);
  g_kernels[data->kernel_id] = name;
}

//===--- the buffer callback (SDK worker thread) -------------------------------===//

std::atomic<std::size_t> g_kernel_count{0};
std::atomic<std::size_t> g_api_count{0};

void BufferCallback(rocprofiler_context_id_t /*context*/,
                    rocprofiler_buffer_id_t /*buffer_id*/,
                    rocprofiler_record_header_t **headers, std::size_t num_headers,
                    void * /*data*/, std::uint64_t /*drop_count*/) {
  for (std::size_t i = 0; i < num_headers; ++i) {
    const rocprofiler_record_header_t *hdr = headers[i];
    if (hdr == nullptr || hdr->category != ROCPROFILER_BUFFER_CATEGORY_TRACING) {
      continue;
    }
    switch (hdr->kind) {
    case ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH: {
      const auto *rec = static_cast<
          const rocprofiler_buffer_tracing_kernel_dispatch_record_t *>(hdr->payload);
      if (rec == nullptr) {
        continue;
      }
      const rocprofiler_kernel_dispatch_info_t &info = rec->dispatch_info;
      // grid_size is in WORK ITEMS (observed on ROCm 7.2.3: layernorm
      // M=1024 blocks x 128 threads -> grid_size=131072); report blocks with
      // the same conversion as the todo-45 tool.
      const auto blocks_of = [](std::uint64_t grid, std::uint64_t wg) {
        return wg > 0 ? static_cast<std::int64_t>(grid / wg) : 1;
      };
      std::string name;
      {
        std::lock_guard<std::mutex> lock(g_kernels_mu);
        const auto it = g_kernels.find(info.kernel_id);
        if (it != g_kernels.end()) {
          name = it->second;
        }
      }
      const std::uint64_t duration =
          rec->end_timestamp >= rec->start_timestamp
              ? rec->end_timestamp - rec->start_timestamp
              : 0;
      // correlation_id is rocprofiler_async_correlation_id_t {internal,
      // ancestor}: `internal` is the API call's correlation id — the join
      // key linking this dispatch to the HIP API record that enqueued it.
      char line[1024];
      const int n = std::snprintf(
          line, sizeof(line),
          "{\"vendor\":\"amd\",\"kind\":\"kernel\",\"name\":\"%s\","
          "\"start_ns\":%" PRIu64 ",\"end_ns\":%" PRIu64
          ",\"duration_ns\":%" PRIu64 ",\"grid\":[%lld,%lld,%lld],"
          "\"workgroup\":[%lld,%lld,%lld],\"grid_size\":[%" PRIu64 ",%" PRIu64
          ",%" PRIu64 "],\"lds_bytes\":%u,\"scratch_bytes\":%u,"
          "\"correlation_id\":%" PRIu64 ",\"kernel_id\":%" PRIu64
          ",\"agent_id\":%" PRIu64 ",\"queue_id\":%" PRIu64
          ",\"dispatch_id\":%" PRIu64 ",\"thread_id\":%" PRIu64 "}",
          kernscope::profile::json_escape(name).c_str(), rec->start_timestamp,
          rec->end_timestamp, duration, blocks_of(info.grid_size.x, info.workgroup_size.x),
          blocks_of(info.grid_size.y, info.workgroup_size.y),
          blocks_of(info.grid_size.z, info.workgroup_size.z),
          static_cast<long long>(info.workgroup_size.x),
          static_cast<long long>(info.workgroup_size.y),
          static_cast<long long>(info.workgroup_size.z), info.grid_size.x,
          info.grid_size.y, info.grid_size.z, info.group_segment_size,
          info.private_segment_size, rec->correlation_id.internal,
          info.kernel_id, info.agent_id.handle, info.queue_id.handle,
          info.dispatch_id, rec->thread_id);
      if (n > 0) {
        WriteJsonl(line);
      }
      ++g_kernel_count;
      break;
    }
    case ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API: {
      const auto *rec = static_cast<
          const rocprofiler_buffer_tracing_hip_api_record_t *>(hdr->payload);
      if (rec == nullptr) {
        continue;
      }
      const char *op_name = "hip?";
      uint64_t name_len = 0;
      (void)rocprofiler_query_buffer_tracing_kind_operation_name(
          static_cast<rocprofiler_buffer_tracing_kind_t>(hdr->kind),
          rec->operation, &op_name, &name_len);
      char line[512];
      const int n = std::snprintf(
          line, sizeof(line),
          "{\"vendor\":\"amd\",\"kind\":\"api\",\"operation\":\"%s\","
          "\"correlation_id\":%" PRIu64 ",\"start_ns\":%" PRIu64
          ",\"end_ns\":%" PRIu64 ",\"thread_id\":%" PRIu64 "}",
          op_name, rec->correlation_id.internal, rec->start_timestamp,
          rec->end_timestamp, rec->thread_id);
      if (n > 0) {
        WriteJsonl(line);
      }
      ++g_api_count;
      break;
    }
    default:
      break;
    }
  }
}

//===--- context + services (created in ToolInitialize, NOT rocprofiler_configure) ===//
// Same reason as todo-45: the SDK assigns the tool's client id only after
// every rocprofiler_configure has returned.

int ToolInitialize(rocprofiler_client_finalize_t /*finalize_func*/,
                   void * /*tool_data*/) {
  rocprofiler_context_id_t ctx{};
  if (rocprofiler_create_context(&ctx) != ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "[kernscope-profile-hip] warning: context creation failed; "
                 "tracing disabled\n");
    return -1;
  }
  rocprofiler_buffer_id_t buffer{};
  constexpr std::size_t kBufferBytes = 8u << 20;
  if (rocprofiler_create_buffer(ctx, kBufferBytes, 0,
                                ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                BufferCallback, nullptr, &buffer) !=
      ROCPROFILER_STATUS_SUCCESS) {
    std::fprintf(stderr,
                 "[kernscope-profile-hip] warning: buffer creation failed; "
                 "tracing disabled\n");
    return -1;
  }
  // KERNEL_DISPATCH (per-kernel records) + HIP_RUNTIME_API (the correlation
  // join side) on the same buffer.
  (void)rocprofiler_configure_buffer_tracing_service(
      ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, buffer);
  (void)rocprofiler_configure_buffer_tracing_service(
      ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, buffer);
  // kernel_id -> name via the code-object kernel-symbol callbacks.
  (void)rocprofiler_configure_callback_tracing_service(
      ctx, ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT, nullptr, 0,
      CodeObjectCallback, nullptr);
  (void)rocprofiler_start_context(ctx);
  return 0;
}

void ToolFinalize(void * /*tool_data*/) {
  std::lock_guard<std::mutex> lock(g_out_mu);
  if (g_out != nullptr && g_out != stdout) {
    std::fclose(g_out);
    g_out = nullptr;
  }
  std::fprintf(stderr,
               "[kernscope-profile-hip] info: %zu kernel dispatch record(s), "
               "%zu HIP API record(s)\n",
               g_kernel_count.load(), g_api_count.load());
}

} // namespace

//===--- tool entry point ------------------------------------------------------===//

extern "C" rocprofiler_tool_configure_result_t *rocprofiler_configure(
    uint32_t /*version*/, const char * /*runtime_version*/, uint32_t /*priority*/,
    rocprofiler_client_id_t *id) {
  id->name = const_cast<char *>("kernscope-profile-hip");
  static rocprofiler_tool_configure_result_t cfg = {
      sizeof(rocprofiler_tool_configure_result_t), ToolInitialize,
      ToolFinalize, nullptr};
  return &cfg;
}
