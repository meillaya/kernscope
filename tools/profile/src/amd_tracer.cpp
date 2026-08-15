//===- amd_tracer.cpp - rocprofiler-sdk buffer-tracing tool (todo 49/50) -*- C++ -*-===//
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
//      kernel name AND the STATIC RESOURCE FACTS the code-object service
//      reports per symbol (todo 50): sgpr_count, arch_vgpr_count,
//      accum_vgpr_count, group_segment_size (static LDS), private_segment_size
//      (scratch/work-item) and kernel_address.
//
// Enrichment (todo 50): every kernel record is enriched by the
// vendor-neutral kernscope::profile::enrich_kernel() layer — theoretical
// occupancy + limiter computed by LIBS/OCCUPANCY (todo 11) fed with
// registersPerThread (= the code-object arch_vgpr_count), LDS bytes and the
// block dims from the dispatch — and emitted in the enriched JSONL shape
// (the todo-49 fields preserved field-for-field plus occupancy/limiter/
// vgpr/sgpr/accum_vgpr/lds/scratch/kernel_address/ptx_available).
//
// Output: one JSONL line per record to KERNSCOPE_PROFILE_OUT (default:
// stdout) in the shared record shape of the profile track (vendor "amd").
//
// Loading (same as todo-45): LD_PRELOAD=<this .so> <app> or
// ROCPROFILER_LIBRARY=<this .so> <app>.
//
// MUST NOT use the deprecated tracer; MUST NOT use nvprof-style counters —
// this tracer is pure buffer-tracing (timestamps + dispatch info).
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/enrich.h"     // vendor-neutral model + enrichment
#include "kernscope/profile/kernel_record.h"

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

//===--- kernel_id -> code-object facts (kernel-symbol callbacks) ------------===//
// Todo 50: the code-object service reports, per registered kernel symbol, the
// STATIC resource facts (sgpr_count/arch_vgpr_count/accum_vgpr_count/
// group_segment_size/private_segment_size) + kernel_address. The dispatch
// record carries only the kernel_id, so the resources join here by id.

struct KernelSymbolInfo {
  std::string name;
  std::uint32_t sgpr_count = 0;
  std::uint32_t arch_vgpr_count = 0;
  std::uint32_t accum_vgpr_count = 0;
  std::uint32_t group_segment_size = 0;   // static LDS per work-group
  std::uint32_t private_segment_size = 0; // scratch per work-item
  std::uint64_t kernel_address = 0;
};

std::mutex g_kernels_mu;
std::unordered_map<std::uint64_t, KernelSymbolInfo> g_kernels;

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
  KernelSymbolInfo info;
  info.name = data->kernel_name != nullptr ? data->kernel_name : "";
  info.sgpr_count = data->sgpr_count;
  info.arch_vgpr_count = data->arch_vgpr_count;
  info.accum_vgpr_count = data->accum_vgpr_count;
  info.group_segment_size = data->group_segment_size;
  info.private_segment_size = data->private_segment_size;
  info.kernel_address = data->kernel_address.handle;
  std::lock_guard<std::mutex> lock(g_kernels_mu);
  g_kernels[data->kernel_id] = std::move(info);
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
      KernelSymbolInfo kinfo;
      bool have_symbol = false;
      {
        std::lock_guard<std::mutex> lock(g_kernels_mu);
        const auto it = g_kernels.find(info.kernel_id);
        if (it != g_kernels.end()) {
          kinfo = it->second;
          have_symbol = true;
        }
      }
      // Build the vendor-neutral record (todo 49 model), then enrich it
      // (todo 50): registersPerThread <- the code-object arch_vgpr_count,
      // LDS <- dispatch group_segment_size, occupancy <- libs/occupancy.
      kernscope::profile::KernelRecord rec_model;
      rec_model.kernel_name = have_symbol ? kinfo.name : "";
      rec_model.start_ns = rec->start_timestamp;
      rec_model.end_ns = rec->end_timestamp;
      rec_model.duration_ns =
          rec->end_timestamp >= rec->start_timestamp
              ? rec->end_timestamp - rec->start_timestamp
              : 0;
      rec_model.grid_x = blocks_of(info.grid_size.x, info.workgroup_size.x);
      rec_model.grid_y = blocks_of(info.grid_size.y, info.workgroup_size.y);
      rec_model.grid_z = blocks_of(info.grid_size.z, info.workgroup_size.z);
      rec_model.block_x = static_cast<std::int64_t>(info.workgroup_size.x);
      rec_model.block_y = static_cast<std::int64_t>(info.workgroup_size.y);
      rec_model.block_z = static_cast<std::int64_t>(info.workgroup_size.z);
      rec_model.static_shared_bytes =
          static_cast<std::int32_t>(info.group_segment_size);
      rec_model.registers_per_thread =
          static_cast<std::uint16_t>(have_symbol ? kinfo.arch_vgpr_count : 0);
      rec_model.correlation_id = rec->correlation_id.internal;

      kernscope::profile::EnrichOptions opts;
      opts.arch_name = "gfx1101"; // the local live target (todo-5 fixture)
      opts.sgprs_per_thread =
          static_cast<int>(have_symbol ? kinfo.sgpr_count : 0);
      kernscope::profile::EnrichedKernel ek =
          kernscope::profile::enrich_kernel(rec_model, opts);
      if (have_symbol) {
        // The code-object service facts (todo 50) ride along in the record.
        ek.arch_vgpr_count = static_cast<int>(kinfo.arch_vgpr_count);
        ek.sgpr_count = static_cast<int>(kinfo.sgpr_count);
        ek.accum_vgpr_count = static_cast<int>(kinfo.accum_vgpr_count);
        ek.group_segment_bytes = static_cast<int>(kinfo.group_segment_size);
        ek.private_segment_bytes = static_cast<int>(kinfo.private_segment_size);
        ek.kernel_address = kinfo.kernel_address;
      } else {
        // Degradation: no code-object facts for this kernel id yet -> the
        // record stays honest (vgpr/sgpr -1, occupancy unknown-resources
        // since registersPerThread is 0); never fabricate.
        ek.group_segment_bytes = static_cast<int>(info.group_segment_size);
        ek.private_segment_bytes = static_cast<int>(info.private_segment_size);
      }
      const std::string line =
          kernscope::profile::enriched_kernel_to_json("amd", ek);
      if (!line.empty()) {
        WriteJsonl(line.c_str());
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
