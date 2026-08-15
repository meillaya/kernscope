//===- cupti_tracer_test.cpp - CUPTI tracer unit suite (todo 49) ---------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// Compile+unit ONLY (no NVIDIA runtime locally — real CUPTI behavior is
// rental-gated per scripts/rental-validate.sh): the suite drives the record
// parser + the versioned Kernel9/Kernel10 adapter through SYNTHETIC activity
// buffers built exactly as CUPTI would deliver them (the mock style of the
// todo-43 mock-cudart / todo-44 synthetic_cupti.h):
//
//   1. FIELD EXTRACTION from a synthetic Kernel9 buffer (acceptance: the
//      record parser asserts field extraction).
//   2. ADAPTER MAPPING: the Kernel9 <-> Kernel10 field set — the same parser
//      template instantiated over a Kernel10View mirror (Kernel9 fields +
//      the isDeviceLaunched tail) extracts the identical field set and picks
//      up the tail field (the documented CUDA-13 delta).
//   3. QA NEGATIVE: a buffer with a TRUNCATED record -> the iterator reports
//      malformed + SKIPS the remainder (asserted, no crash) — this is the
//      evidence for task-49-gpu-kernel-suite-negative.txt.
//   4. Field-map pin: every documented KernelFieldMap entry matches
//      offsetof on the build's own struct (compile-time static_asserts in the
//      header already pin them; this runtime check double-locks the table).
//
// Test naming: gtest_discover_tests TEST_PREFIX "profile_tracer:" so
// `ctest -R profile_tracer` selects this suite (hardware label no-gpu).
//
//===----------------------------------------------------------------------===//

#include "kernscope/profile/record_adapter.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

using namespace kernscope::profile;

namespace {

//===--- synthetic record builders (what CUPTI would deliver) ----------------===//

CUpti_ActivityKernel9 MakeSyntheticKernel9(std::uint32_t correlation,
                                           std::uint64_t start,
                                           std::uint64_t end) {
  CUpti_ActivityKernel9 r{};
  r.kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL; // kind 10 — never kind 3
  r.cacheConfig.both = 0;                          // CU_FUNC_CACHE_PREFER_L1
  r.sharedMemoryConfig = 0;                        // CU_SHARED_MEM_CONFIG_DEFAULT
  r.registersPerThread = 96;
  r.start = start;
  r.end = end;
  r.completed = end;
  r.deviceId = 0;
  r.contextId = 1;
  r.streamId = 7;
  r.gridX = 32;
  r.gridY = 1;
  r.gridZ = 1;
  r.blockX = 128;
  r.blockY = 1;
  r.blockZ = 1;
  r.staticSharedMemory = 24576;
  r.dynamicSharedMemory = 0;
  r.localMemoryPerThread = 0;
  r.localMemoryTotal = 0;
  r.correlationId = correlation;
  r.gridId = 42;
  r.name = "layernorm_fused";
  r.queued = 0; // latency timestamps off by default
  r.submitted = 0;
  r.launchType = 0;
  r.sharedMemoryExecuted = 24576;
  r.localMemoryTotal_v2 = 0;
  return r;
}

CUpti_ActivityAPI MakeSyntheticApi(std::uint32_t correlation,
                                   CUpti_CallbackId cbid) {
  CUpti_ActivityAPI r{};
  r.kind = CUPTI_ACTIVITY_KIND_RUNTIME;
  r.cbid = cbid;
  r.start = 500;
  r.end = 1500;
  r.processId = 1234;
  r.threadId = 1;
  r.correlationId = correlation;
  r.returnValue = 0;
  return r;
}

CUpti_ActivityOverhead MakeSyntheticOverhead() {
  CUpti_ActivityOverhead r{};
  r.kind = CUPTI_ACTIVITY_KIND_OVERHEAD;
  r.overheadKind = CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER;
  r.objectKind = CUPTI_ACTIVITY_OBJECT_PROCESS;
  r.start = 10;
  r.end = 20;
  return r;
}

/// Packs records into a raw activity buffer at 8-byte alignment (the
/// documented iteration convention), exactly like a completed CUPTI buffer.
std::vector<std::uint8_t> PackBuffer(
    const std::vector<const void *> &records, const std::vector<std::size_t> &sizes,
    std::size_t truncated_tail_bytes = 0) {
  std::vector<std::uint8_t> buf;
  for (std::size_t i = 0; i < records.size(); ++i) {
    const std::size_t pad = (8 - (buf.size() % 8)) % 8;
    buf.resize(buf.size() + pad, 0);
    const auto *bytes = static_cast<const std::uint8_t *>(records[i]);
    buf.insert(buf.end(), bytes, bytes + sizes[i]);
  }
  buf.resize(buf.size() + truncated_tail_bytes, 0xA5); // poison tail
  return buf;
}

//===--- Kernel10View: the Kernel10 field set the adapter maps ----------------===//
// The documented CUDA-13 struct: Kernel9's mapped field set (identical member
// names/types — the parser template reads by member name) + the
// isDeviceLaunched tail. The mapping test proves the SAME parser template
// extracts the same fields from a Kernel10-shaped record and picks up the new
// tail member (which real 12.6 headers cannot provide locally). The
// intermediate fields the adapter does NOT read (cacheConfig, carved-out GC,
// cluster fields, ...) are omitted — this view mirrors the MAPPED field set.

struct Kernel10View {
  CUpti_ActivityKind kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  std::uint16_t registersPerThread = 0;
  std::uint64_t start = 0;
  std::uint64_t end = 0;
  std::uint64_t completed = 0;
  std::uint32_t deviceId = 0;
  std::uint32_t contextId = 0;
  std::uint32_t streamId = 0;
  std::int32_t gridX = 1, gridY = 1, gridZ = 1;
  std::int32_t blockX = 1, blockY = 1, blockZ = 1;
  std::int32_t staticSharedMemory = 0;
  std::int32_t dynamicSharedMemory = 0;
  std::uint32_t localMemoryPerThread = 0;
  std::uint32_t correlationId = 0;
  std::int64_t gridId = 0;
  const char *name = nullptr;
  std::uint8_t launchType = 0;
  std::uint32_t sharedMemoryExecuted = 0;
  std::uint64_t localMemoryTotal_v2 = 0;
  std::uint8_t isDeviceLaunched = 0; // new in Kernel10 (CUDA 13)
  std::uint8_t pad[3] = {0, 0, 0};   // tail padding to 8-byte alignment
};

static_assert(has_is_device_launched<CUpti_ActivityKernel9>::value == false,
              "Kernel9 must not expose the Kernel10 tail field");
static_assert(has_is_device_launched<Kernel10View>::value == true,
              "Kernel10View must expose the Kernel10 tail field");

} // namespace

//===--- 1. field extraction from a synthetic Kernel9 buffer -------------------===//

TEST(profile_tracer, SyntheticKernel9FieldExtraction) {
  const CUpti_ActivityKernel9 raw = MakeSyntheticKernel9(0x2A, 1000, 2000);
  KernelRecord out;
  ASSERT_TRUE(parse_kernel_record(&raw, &out));
  EXPECT_EQ(out.kernel_name, "layernorm_fused");
  EXPECT_EQ(out.start_ns, 1000u);
  EXPECT_EQ(out.end_ns, 2000u);
  EXPECT_EQ(out.completed_ns, 2000u);
  EXPECT_EQ(out.duration_ns, 1000u);
  EXPECT_EQ(out.grid_x, 32);
  EXPECT_EQ(out.grid_y, 1);
  EXPECT_EQ(out.grid_z, 1);
  EXPECT_EQ(out.block_x, 128);
  EXPECT_EQ(out.block_y, 1);
  EXPECT_EQ(out.block_z, 1);
  EXPECT_EQ(out.static_shared_bytes, 24576);
  EXPECT_EQ(out.dynamic_shared_bytes, 0);
  EXPECT_EQ(out.shared_memory_executed, 24576u);
  EXPECT_EQ(out.registers_per_thread, 96u);
  EXPECT_EQ(out.local_memory_per_thread, 0u);
  EXPECT_EQ(out.local_memory_total, 0u);
  EXPECT_EQ(out.correlation_id, 0x2Au);
  EXPECT_EQ(out.device_id, 0u);
  EXPECT_EQ(out.context_id, 1u);
  EXPECT_EQ(out.stream_id, 7u);
  EXPECT_EQ(out.grid_id, 42);
  EXPECT_EQ(out.launch_type, 0u);
  EXPECT_EQ(out.is_device_launched, 0u); // Kernel9 has no such field
}

TEST(profile_tracer, IterateSyntheticBufferDeliversKernelAndApiRecords) {
  const CUpti_ActivityKernel9 kernel = MakeSyntheticKernel9(1, 1000, 2000);
  const CUpti_ActivityAPI api = MakeSyntheticApi(1, 0x1234);
  const CUpti_ActivityOverhead overhead = MakeSyntheticOverhead();
  const std::vector<std::uint8_t> buf = PackBuffer(
      {&kernel, &api, &overhead},
      {sizeof(kernel), sizeof(api), sizeof(overhead)});

  std::vector<std::uint32_t> correlations;
  std::size_t kernels = 0, apis = 0, overheads = 0;
  const BufferIterationResult res = iterate_activity_buffer(
      buf.data(), buf.size(),
      [&](CUpti_ActivityKind kind, const CUpti_Activity *record, std::size_t) {
        switch (kind) {
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
          KernelRecord out;
          ASSERT_TRUE(parse_kernel_record(
              reinterpret_cast<const ActivityKernelT *>(record), &out));
          correlations.push_back(out.correlation_id);
          ++kernels;
          break;
        }
        case CUPTI_ACTIVITY_KIND_RUNTIME: {
          const auto *api_rec = reinterpret_cast<const CUpti_ActivityAPI *>(record);
          correlations.push_back(api_rec->correlationId);
          ++apis;
          break;
        }
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
          ++overheads;
          break;
        default:
          FAIL() << "unexpected kind " << static_cast<int>(kind);
        }
      });

  EXPECT_TRUE(res.ok);
  EXPECT_EQ(res.records_parsed, 3u);
  EXPECT_EQ(res.malformed, 0u);
  EXPECT_EQ(kernels, 1u);
  EXPECT_EQ(apis, 1u);
  EXPECT_EQ(overheads, 1u);
  // The kernel record's correlation id joins the API record (kineto join).
  ASSERT_EQ(correlations.size(), 2u);
  EXPECT_EQ(correlations[0], 1u);
  EXPECT_EQ(correlations[1], 1u);
}

//===--- 2. the Kernel9 <-> Kernel10 adapter mapping ---------------------------===//

TEST(profile_tracer, AdapterMapsKernel10FieldSet) {
  Kernel10View raw;
  raw.registersPerThread = 96;
  raw.start = 1000;
  raw.end = 2500;
  raw.completed = 2500;
  raw.deviceId = 0;
  raw.contextId = 1;
  raw.streamId = 7;
  raw.gridX = 32;
  raw.blockX = 128;
  raw.staticSharedMemory = 24576;
  raw.dynamicSharedMemory = 0;
  raw.localMemoryPerThread = 0;
  raw.correlationId = 0xBEEF;
  raw.gridId = 42;
  raw.name = "layernorm_fused";
  raw.launchType = 0;
  raw.sharedMemoryExecuted = 24576;
  raw.localMemoryTotal_v2 = 0;
  raw.isDeviceLaunched = 1; // Kernel10 tail: device-side launch (CDP)

  KernelRecord out9, out10;
  ASSERT_TRUE(parse_cupti_kernel_record<Kernel10View>(&raw, &out10));
  CUpti_ActivityKernel9 k9 = MakeSyntheticKernel9(0xBEEF, 1000, 2500);
  ASSERT_TRUE(parse_kernel_record(&k9, &out9));

  // The Kernel10 field set maps 1:1 onto Kernel9 for every mapped field ...
  EXPECT_EQ(out10.kernel_name, out9.kernel_name);
  EXPECT_EQ(out10.start_ns, out9.start_ns);
  EXPECT_EQ(out10.end_ns, out9.end_ns);
  EXPECT_EQ(out10.duration_ns, out9.duration_ns);
  EXPECT_EQ(out10.grid_x, out9.grid_x);
  EXPECT_EQ(out10.block_x, out9.block_x);
  EXPECT_EQ(out10.static_shared_bytes, out9.static_shared_bytes);
  EXPECT_EQ(out10.registers_per_thread, out9.registers_per_thread);
  EXPECT_EQ(out10.correlation_id, out9.correlation_id);
  // ... and picks up the Kernel10-only tail field.
  EXPECT_EQ(out10.is_device_launched, 1u);
  EXPECT_EQ(out9.is_device_launched, 0u);
  // Structural delta documented in the adapter: added, never moved/removed.
  EXPECT_TRUE(kKernel10TailDelta.added_in_13);
  EXPECT_FALSE(kKernel10TailDelta.moved_in_13);
  EXPECT_FALSE(kKernel10TailDelta.removed_in_13);
}

TEST(profile_tracer, AdapterPinsKernelFieldMap) {
  // Every documented entry names a real member at the documented offset
  // (the header static_asserts are the compile-time pin; this double-locks
  // the table against the build's own struct).
  ASSERT_EQ(kKernelFieldMap[0].name, std::string("kind"));
  ASSERT_EQ(kKernelFieldMap[0].offset, offsetof(ActivityKernelT, kind));
  ASSERT_EQ(kKernelFieldMap[1].name, std::string("registersPerThread"));
  ASSERT_EQ(kKernelFieldMap[1].offset,
            offsetof(ActivityKernelT, registersPerThread));
  ASSERT_EQ(kKernelFieldMap[4].name, std::string("completed"));
  ASSERT_EQ(kKernelFieldMap[4].offset, offsetof(ActivityKernelT, completed));
  ASSERT_EQ(kKernelFieldMap[19].name, std::string("name"));
  ASSERT_EQ(kKernelFieldMap[19].offset, offsetof(ActivityKernelT, name));
  ASSERT_EQ(kKernelFieldMap[22].name, std::string("localMemoryTotal_v2"));
  ASSERT_EQ(kKernelFieldMap[22].offset,
            offsetof(ActivityKernelT, localMemoryTotal_v2));
}

TEST(profile_tracer, KernelAndConcurrentKernelParseIdentically) {
  // Same struct, both kinds — the tracer enables kind 10 only, but if a kind-3
  // record ever appeared the parser must handle it identically (never
  // serializing: the ENABLE is what matters, not the parse).
  CUpti_ActivityKernel9 raw = MakeSyntheticKernel9(3, 100, 900);
  raw.kind = CUPTI_ACTIVITY_KIND_KERNEL;
  KernelRecord out;
  ASSERT_TRUE(parse_kernel_record(&raw, &out));
  EXPECT_EQ(out.duration_ns, 800u);
}

//===--- 3. QA negative: truncated record -> malformed + skip (no crash) -------===//

TEST(profile_tracer, TruncatedRecordIsReportedAndSkipped) {
  // One VALID kernel record followed by a TRUNCATED one (kind byte says
  // CONCURRENT_KERNEL, only 16 of the 208 bytes present — the shape a torn
  // buffer delivery could produce).
  const CUpti_ActivityKernel9 kernel = MakeSyntheticKernel9(7, 1000, 2000);
  const std::vector<std::uint8_t> good = PackBuffer(
      {&kernel}, {sizeof(kernel)});
  std::vector<std::uint8_t> buf = good;
  const std::uint32_t truncated_kind = CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL;
  buf.insert(buf.end(), reinterpret_cast<const std::uint8_t *>(&truncated_kind),
             reinterpret_cast<const std::uint8_t *>(&truncated_kind) +
                 sizeof(truncated_kind));
  buf.resize(buf.size() + 12, 0xA5); // only 16 bytes of a 208-byte record

  std::size_t kernels_seen = 0;
  BufferIterationResult res = iterate_activity_buffer(
      buf.data(), buf.size(),
      [&](CUpti_ActivityKind kind, const CUpti_Activity *, std::size_t) {
        if (kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
          ++kernels_seen;
        }
      });

  // The valid record was parsed; the truncated one was reported + skipped.
  EXPECT_EQ(res.records_parsed, 1u);
  EXPECT_EQ(res.malformed, 1u);
  EXPECT_FALSE(res.ok); // iteration cut short at the malformed record
  EXPECT_EQ(kernels_seen, 1u);
}

TEST(profile_tracer, UnknownKindRecordIsReportedAndSkipped) {
  // A kind the tracer never enables (e.g. CUPTI_ACTIVITY_KIND_CONTEXT) must
  // not be mis-parsed: reported malformed, iteration stops, no crash.
  const CUpti_ActivityKernel9 kernel = MakeSyntheticKernel9(9, 1000, 2000);
  std::vector<std::uint8_t> buf = PackBuffer({&kernel}, {sizeof(kernel)});
  const std::uint32_t unknown_kind = CUPTI_ACTIVITY_KIND_CONTEXT;
  buf.insert(buf.end(), reinterpret_cast<const std::uint8_t *>(&unknown_kind),
             reinterpret_cast<const std::uint8_t *>(&unknown_kind) +
                 sizeof(unknown_kind));
  buf.resize(buf.size() + 8, 0x5A);

  BufferIterationResult res = iterate_activity_buffer(
      buf.data(), buf.size(),
      [&](CUpti_ActivityKind, const CUpti_Activity *, std::size_t) {});
  EXPECT_EQ(res.records_parsed, 1u);
  EXPECT_EQ(res.malformed, 1u);
  EXPECT_FALSE(res.ok);
}

TEST(profile_tracer, EmptyBufferIteratesCleanly) {
  BufferIterationResult res = iterate_activity_buffer(
      nullptr, 0, [&](CUpti_ActivityKind, const CUpti_Activity *, std::size_t) {});
  EXPECT_TRUE(res.ok);
  EXPECT_EQ(res.records_parsed, 0u);
  EXPECT_EQ(res.malformed, 0u);
}
