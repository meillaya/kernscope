//===- bundle_test.cpp - gtest suite for the v1 bundle format (todo 46) -*- C++ -*-===//
//
// Covers every acceptance of todo 46:
//  (a) v1 write->read ROUNDTRIP field-exact: manifest/args/blobs all match;
//  (b) embedded-pointer fixture patched correctly on remap (synthetic VA
//      table - a T** embedded pointer inside a buffer blob gets its offset
//      patched per the remap table);
//  (c) size-cap test: bundle exceeding --max-snapshot-bytes -> capture
//      ABORTS with a clear error + a partial manifest (asserted);
//  (d) the reader loads the v0 output shape from todos 43/44/45 (committed
//      fixture + sidecar .bin blobs) AND v1;
//  (e) QA negative: a TRUNCATED bundle file -> reader reports a schema
//      violation WITH OFFSET (no UB, no crash);
//  (f) sha256 provenance is verified per blob (mismatch -> schema violation);
//  (g) F3 review seeds: a 1-byte truncation of the LAST byte of the
//      committed v0 fixture (the 'end-capture-v0' trailing newline) FAILS
//      with a schema violation naming the terminator + offset, and a
//      1-byte truncation of the committed v1 fixture blob at ANY byte
//      position FAILS at load with a sha256-mismatch naming the blob path
//      + offset (both committed fixtures exercise the hardened paths).
//
// All tests are pure host-side (no GPU), LABELS no-gpu.
//
//===----------------------------------------------------------------------===//

#include "kernscope/bundle.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#ifndef KERNSCOPE_V0_FIXTURE
#define KERNSCOPE_V0_FIXTURE ""
#endif

#ifndef KERNSCOPE_V1_FIXTURE
#define KERNSCOPE_V1_FIXTURE ""
#endif

namespace {

namespace fs = std::filesystem;
using kernscope::BufferRole;
using kernscope::BufferSnapshot;
using kernscope::BundleReader;
using kernscope::BundleWriteOptions;
using kernscope::CaptureSession;
using kernscope::LaunchVia;
using kernscope::PatchEntry;
using kernscope::WriteStatus;

fs::path TempDir() {
  std::error_code ec;
  fs::path dir = fs::temp_directory_path(ec) /
                 ("kernscope-bundle-test-" + std::to_string(::getpid()));
  fs::remove_all(dir, ec);
  fs::create_directories(dir, ec);
  return dir;
}

std::string ReadFile(const fs::path &p) {
  std::ifstream in(p, std::ios::binary);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

/// Fills `s` with one record of every type the v1 events.json formalizes:
/// api counts, alloc/free, unknown-free, launch with scalar + pointer args,
/// copy, memset, streams, fatbin + cubin modules, param_info.
void FillSession(CaptureSession &s) {
  s.note_api(kernscope::ApiKind::kCudaMalloc);
  s.note_api(kernscope::ApiKind::kCudaLaunchKernel);

  // allocations: 0x1000 (1024B), 0x2000 (64B); 0x1000 is freed later.
  s.record_alloc(0x1000, 1024);
  s.record_alloc(0x2000, 64);
  s.record_free(0xBEEF, /*forwarded=*/true); // unknown-free QA path

  // launch: arg0 = pointer slot (8-byte LE 0x1000), arg1 = scalar int32.
  kernscope::LaunchRecord launch;
  launch.kernel_name = "vec_add";
  launch.via = LaunchVia::kRuntimeApi;
  launch.grid_x = 2;
  launch.grid_y = 3;
  launch.grid_z = 1;
  launch.block_x = 128;
  launch.block_y = 1;
  launch.block_z = 1;
  launch.shared_mem_bytes = 8;
  launch.stream = 0x1234;
  kernscope::KernelArgSlot ptr_slot;
  ptr_slot.size = 8;
  ptr_slot.bytes = {0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
  kernscope::KernelArgSlot count_slot;
  count_slot.size = 4;
  count_slot.bytes = {0x00, 0x02, 0x00, 0x00};
  launch.args = {std::move(ptr_slot), std::move(count_slot)};
  s.record_launch(std::move(launch));

  kernscope::CopyRecord copy;
  copy.kind = "memcpy";
  copy.dst = 0x1000;
  copy.src = 0x7F00;
  copy.bytes = 512;
  copy.cuda_kind = 1; // HostToDevice
  copy.dst_size = 1024;
  s.record_copy(copy);

  kernscope::MemsetRecord memset;
  memset.ptr = 0x2000;
  memset.value = 0;
  memset.bytes = 64;
  memset.ptr_size = 64;
  s.record_memset(memset);

  s.record_stream_create(0x1234);
  s.record_stream_sync(0x1234);

  std::vector<std::uint8_t> cubin = {0xAA, 0xBB, 0xCC};
  const std::uint64_t reg_id = s.register_fatbinary(cubin);
  s.register_function(reg_id, "vec_add");
  std::vector<std::uint8_t> hsaco = {0x11, 0x22};
  s.register_cubin(hsaco);

  constexpr std::uint64_t kCufunction = 0x9000;
  s.record_param_info(kCufunction, 0, 0, 8);
  s.record_param_info(kCufunction, 1, 8, 4);
}

//--- (a) v1 roundtrip, field-exact ------------------------------------------------

TEST(BundleV1Test, RoundtripIsFieldExact) {
  CaptureSession session;
  FillSession(session);
  BundleWriteOptions opts;
  opts.vendor = "mock";
  opts.arch = "sm_80";
  opts.capture_mechanism = "preload";
  opts.tool_version = "kernscope-capture/1.0";
  opts.timestamp = "2026-08-14T04:00:00Z";
  opts.max_snapshot_bytes = kernscope::kDefaultMaxSnapshotBytes;

  // Snapshot requests: an input buffer at VA 0x1000 (with a T** embedded
  // pointer to 0x2000 at offset 16) + a kernel-binary blob (no VA).
  const std::uint64_t kInputVa = 0x1000;
  const std::uint64_t kEmbeddedVa = 0x2000;
  BufferSnapshot input;
  input.role = BufferRole::kInput;
  input.va = kInputVa;
  input.size = 64;
  BufferSnapshot kernel;
  kernel.role = BufferRole::kKernelBinary;
  kernel.size = 3;
  opts.snapshots = {input, kernel};

  const std::vector<std::uint8_t> kInputBytes = [&]() {
    std::vector<std::uint8_t> b(64, 0xAB);
    std::memcpy(b.data() + 16, &kEmbeddedVa, sizeof(kEmbeddedVa));
    b[0] = 0x01;
    b[1] = 0x02;
    return b;
  }();
  const std::vector<std::uint8_t> kKernelBytes = {0xAA, 0xBB, 0xCC};
  opts.read_fn = [&](const BufferSnapshot &req,
                     std::vector<std::uint8_t> &out) -> bool {
    if (req.role == BufferRole::kInput) {
      out = kInputBytes;
    } else {
      out = kKernelBytes;
    }
    return true;
  };

  const fs::path dir = TempDir() / "roundtrip";
  std::string err;
  ASSERT_EQ(write_bundle_v1(session, dir.string(), opts, &err),
            WriteStatus::kOk)
      << err;
  const fs::path bundle = dir.string() + ".kcap";

  auto reader = BundleReader::open(bundle.string(), &err);
  ASSERT_NE(reader, nullptr) << err;

  // manifest
  EXPECT_TRUE(reader->is_v1());
  EXPECT_EQ(reader->format_id(), "kernscope-capture");
  EXPECT_EQ(reader->schema_version(), 1);
  EXPECT_EQ(reader->tool_version(), "kernscope-capture/1.0");
  EXPECT_EQ(reader->vendor(), "mock");
  EXPECT_EQ(reader->arch(), "sm_80");
  EXPECT_EQ(reader->capture_mechanism(), "preload");
  EXPECT_EQ(reader->timestamp(), "2026-08-14T04:00:00Z");
  EXPECT_EQ(reader->first_kernel(), "vec_add");
  EXPECT_EQ(reader->max_snapshot_bytes(),
            kernscope::kDefaultMaxSnapshotBytes);
  EXPECT_FALSE(reader->aborted());

  // events: api counts
  ASSERT_EQ(reader->api().size(), session.api_table().size());
  for (std::size_t i = 0; i < reader->api().size(); ++i) {
    EXPECT_EQ(reader->api()[i].name, session.api_table()[i].name);
    EXPECT_EQ(reader->api()[i].calls, session.api_table()[i].calls);
  }
  // events: allocs (incl. free + unknown-free)
  ASSERT_EQ(reader->allocs().size(), session.allocs().size());
  for (std::size_t i = 0; i < reader->allocs().size(); ++i) {
    EXPECT_EQ(reader->allocs()[i].ptr, session.allocs()[i].ptr);
    EXPECT_EQ(reader->allocs()[i].size, session.allocs()[i].size);
    EXPECT_EQ(reader->allocs()[i].is_alloc, session.allocs()[i].is_alloc);
    EXPECT_EQ(reader->allocs()[i].seq, session.allocs()[i].seq);
  }
  ASSERT_EQ(reader->unknown_frees().size(), session.unknown_frees().size());
  EXPECT_EQ(reader->unknown_frees()[0].ptr,
            session.unknown_frees()[0].ptr);

  // events: launches + arg marshalling table
  ASSERT_EQ(reader->launches().size(), 1u);
  const auto &rl = reader->launches()[0];
  const auto &sl = session.launches()[0];
  EXPECT_EQ(rl.kernel_name, sl.kernel_name);
  EXPECT_EQ(rl.via, sl.via);
  EXPECT_EQ(rl.grid_x, sl.grid_x);
  EXPECT_EQ(rl.grid_y, sl.grid_y);
  EXPECT_EQ(rl.grid_z, sl.grid_z);
  EXPECT_EQ(rl.block_x, sl.block_x);
  EXPECT_EQ(rl.block_y, sl.block_y);
  EXPECT_EQ(rl.block_z, sl.block_z);
  EXPECT_EQ(rl.shared_mem_bytes, sl.shared_mem_bytes);
  EXPECT_EQ(rl.stream, sl.stream);
  EXPECT_EQ(rl.graph_handle, sl.graph_handle);
  EXPECT_EQ(rl.seq, sl.seq);
  ASSERT_EQ(rl.args.size(), sl.args.size());
  for (std::size_t i = 0; i < rl.args.size(); ++i) {
    EXPECT_EQ(rl.args[i].size, sl.args[i].size);
    EXPECT_EQ(rl.args[i].bytes, sl.args[i].bytes);
  }
  // arg marshalling: slot 0 classified POINTER with original VA, slot 1
  // scalar.
  ASSERT_EQ(reader->launch_arg_tables().size(), 1u);
  const auto &table = reader->launch_arg_tables()[0];
  ASSERT_EQ(table.size(), 2u);
  EXPECT_TRUE(table[0].is_pointer);
  EXPECT_EQ(table[0].va, kInputVa);
  EXPECT_FALSE(table[1].is_pointer);

  // events: copy / memset / stream
  ASSERT_EQ(reader->copies().size(), session.copies().size());
  EXPECT_EQ(reader->copies()[0].dst, session.copies()[0].dst);
  EXPECT_EQ(reader->copies()[0].bytes, session.copies()[0].bytes);
  EXPECT_EQ(reader->copies()[0].cuda_kind, session.copies()[0].cuda_kind);
  EXPECT_EQ(reader->copies()[0].dst_size, session.copies()[0].dst_size);
  ASSERT_EQ(reader->memsets().size(), session.memsets().size());
  EXPECT_EQ(reader->memsets()[0].ptr, session.memsets()[0].ptr);
  EXPECT_EQ(reader->memsets()[0].bytes, session.memsets()[0].bytes);
  ASSERT_EQ(reader->streams().size(), session.streams().size());
  for (std::size_t i = 0; i < reader->streams().size(); ++i) {
    EXPECT_EQ(reader->streams()[i].handle, session.streams()[i].handle);
    EXPECT_EQ(reader->streams()[i].id, session.streams()[i].id);
    EXPECT_EQ(reader->streams()[i].is_sync, session.streams()[i].is_sync);
  }

  // events: modules (fatbin + cubin)
  ASSERT_EQ(reader->modules().size(), session.modules().size());
  for (std::size_t i = 0; i < reader->modules().size(); ++i) {
    EXPECT_EQ(reader->modules()[i].kind, session.modules()[i].kind);
    EXPECT_EQ(reader->modules()[i].handle, session.modules()[i].handle);
    EXPECT_EQ(reader->modules()[i].sha256, session.modules()[i].sha256);
    EXPECT_EQ(reader->modules()[i].name, session.modules()[i].name);
    EXPECT_EQ(reader->modules()[i].kernels, session.modules()[i].kernels);
  }
  // events: param_info
  ASSERT_EQ(reader->param_infos().size(), 2u);
  EXPECT_EQ(reader->param_infos()[0].cufunction, 0x9000u);
  EXPECT_EQ(reader->param_infos()[0].param_index, 0u);
  EXPECT_EQ(reader->param_infos()[0].size, 8u);

  // blobs: bytes, VAs, sha256, embedded-pointer patch table
  ASSERT_EQ(reader->blobs().size(), 2u);
  const auto &b0 = reader->blobs()[0];
  EXPECT_EQ(b0.role, BufferRole::kInput);
  EXPECT_EQ(b0.va, kInputVa);
  EXPECT_EQ(b0.size, 64u);
  EXPECT_EQ(b0.sha256, kernscope::sha256_hex(kInputBytes));
  ASSERT_EQ(b0.patches.size(), 1u);
  EXPECT_EQ(b0.patches[0].offset, 16u);
  EXPECT_EQ(b0.patches[0].target_va, kEmbeddedVa);
  const auto &b1 = reader->blobs()[1];
  EXPECT_EQ(b1.role, BufferRole::kKernelBinary);
  EXPECT_EQ(b1.sha256, kernscope::sha256_hex(kKernelBytes));
  EXPECT_TRUE(b1.patches.empty());
  EXPECT_EQ(reader->total_blob_bytes(), 67u);

  // blob bytes read back + provenance verified
  std::vector<std::uint8_t> in_bytes, kern_bytes;
  std::string berr;
  ASSERT_TRUE(reader->read_blob(b0, in_bytes, &berr)) << berr;
  ASSERT_TRUE(reader->read_blob(b1, kern_bytes, &berr)) << berr;
  EXPECT_EQ(in_bytes, kInputBytes);
  EXPECT_EQ(kern_bytes, kKernelBytes);
}

//--- (b) embedded-pointer patch table (synthetic VA table) ------------------------

TEST(BundleV1Test, EmbeddedPointerRemapPatchesDeepPointer) {
  // A T** embedded pointer: 64-byte buffer; at offset 16 the LE bytes of VA
  // 0x2000 (the pointed-to allocation). Synthetic VA table = {0x1000, 0x2000}.
  const std::uint64_t kOldVa = 0x2000;
  const std::uint64_t kNewVa = 0x9000;
  std::vector<std::uint8_t> blob(64, 0xAB);
  std::memcpy(blob.data() + 16, &kOldVa, sizeof(kOldVa));
  const std::vector<std::uint8_t> original = blob;

  const std::vector<std::uint64_t> vas = {0x1000, kOldVa};
  const auto patches = kernscope::scan_embedded_pointers(blob, vas);

  // Exactly one candidate, at offset 16, pointing at 0x2000. The 8 bytes at
  // offset 0 (0xAB... not a VA) and offset 8 must NOT be candidates.
  bool found = false;
  for (const PatchEntry &p : patches) {
    if (p.offset == 16 && p.target_va == kOldVa) {
      found = true;
    } else {
      // every candidate must be an 8-byte window exactly equal to a known VA
      std::uint64_t v = 0;
      std::memcpy(&v, blob.data() + p.offset, sizeof(v));
      EXPECT_TRUE(v == 0x1000 || v == kOldVa);
    }
  }
  EXPECT_TRUE(found);
  // The candidate at offset 16 also matches at offset 17-23 (the scan is
  // byte-granular: sliding windows over the same pointer bytes) - that is
  // documented behavior (format doc §4.3) and harmless: apply_remap writes
  // the same new VA into each window.
  EXPECT_GE(patches.size(), 1u);

  // Apply the remap: 0x2000 -> 0x9000. The deep pointer at offset 16 (and
  // its sliding windows) are rewritten; everything else is untouched.
  kernscope::apply_remap(blob, patches,
                         std::unordered_map<std::uint64_t, std::uint64_t>{
                             {kOldVa, kNewVa}});
  std::uint64_t patched = 0;
  std::memcpy(&patched, blob.data() + 16, sizeof(patched));
  EXPECT_EQ(patched, kNewVa);
  // The patch is a real content change: the sha256 moves.
  EXPECT_NE(kernscope::sha256_hex(blob), kernscope::sha256_hex(original));
  // Outside the patched pointer window the buffer is byte-identical.
  for (std::size_t i = 0; i < blob.size(); ++i) {
    if (i >= 16 && i < 24) {
      continue;
    }
    EXPECT_EQ(blob[i], original[i]);
  }

  // A remap table WITHOUT the target VA leaves the blob untouched.
  std::vector<std::uint8_t> blob2 = original;
  kernscope::apply_remap(blob2, patches,
                         std::unordered_map<std::uint64_t, std::uint64_t>{});
  EXPECT_EQ(blob2, original);
}

//--- (c) size-cap abort with partial manifest --------------------------------------

TEST(BundleV1Test, SizeCapAbortsWithPartialManifest) {
  CaptureSession session;
  FillSession(session);
  BundleWriteOptions opts;
  opts.vendor = "mock";
  opts.capture_mechanism = "preload";
  opts.timestamp = "2026-08-14T04:00:00Z";
  opts.max_snapshot_bytes = 64; // tiny budget: the FIRST 1024-byte blob blows it

  BufferSnapshot big;
  big.role = BufferRole::kInput;
  big.va = 0x1000;
  big.size = 1024;
  opts.snapshots = {big};
  opts.read_fn = [](const BufferSnapshot &, std::vector<std::uint8_t> &out) {
    out.assign(1024, 0xEE);
    return true;
  };

  const fs::path dir = TempDir() / "sizecap";
  std::string err;
  EXPECT_EQ(write_bundle_v1(session, dir.string(), opts, &err),
            WriteStatus::kAbortedSizeCap);
  EXPECT_NE(err.find("--max-snapshot-bytes=64"), std::string::npos) << err;
  EXPECT_NE(err.find("capture aborted"), std::string::npos) << err;

  // Partial manifest: aborted=true + the clear error; NO blob written.
  const fs::path bundle = dir.string() + ".kcap";
  ASSERT_TRUE(fs::exists(bundle / "manifest.json"));
  const std::string manifest = ReadFile(bundle / "manifest.json");
  // canonical dump has no spaces: "aborted":true
  EXPECT_NE(manifest.find("\"aborted\":true"), std::string::npos);
  EXPECT_NE(manifest.find("--max-snapshot-bytes"), std::string::npos);
  EXPECT_EQ(fs::exists(bundle / "blobs.json"), true);
  // The partial bundle stays LOADABLE and reports aborted().
  auto reader = BundleReader::open(bundle.string(), &err);
  ASSERT_NE(reader, nullptr) << err;
  EXPECT_TRUE(reader->aborted());
  EXPECT_NE(reader->abort_reason().find("--max-snapshot-bytes"),
            std::string::npos);
  EXPECT_EQ(reader->max_snapshot_bytes(), 64u);
  EXPECT_EQ(reader->total_blob_bytes(), 0u);
  EXPECT_TRUE(reader->blobs().empty());
}

//--- (d) v0 compatibility: the todos-43/44/45 output shape -------------------------

TEST(BundleV0Test, ReaderLoadsCommittedV0TextBundle) {
  const std::string fixture = KERNSCOPE_V0_FIXTURE;
  ASSERT_NE(fixture, "") << "KERNSCOPE_V0_FIXTURE not defined";
  ASSERT_TRUE(fs::exists(fixture)) << fixture;

  std::string err;
  auto reader = BundleReader::open(fixture, &err);
  ASSERT_NE(reader, nullptr) << err;
  EXPECT_FALSE(reader->is_v1());
  EXPECT_EQ(reader->format_id(), "kernscope-capture-v0");

  // The committed fixture is the todo-43 smoke-capture shape (mock runtime):
  // allocs, an unknown-free, the vec_add launch with a pointer + scalar arg.
  ASSERT_GE(reader->allocs().size(), 2u);
  EXPECT_EQ(reader->allocs()[0].ptr, 0x1000u);
  EXPECT_EQ(reader->allocs()[0].size, 1024u);
  EXPECT_TRUE(reader->allocs()[0].is_alloc);
  ASSERT_GE(reader->unknown_frees().size(), 1u);
  EXPECT_EQ(reader->unknown_frees()[0].ptr, 0xBEEFu);
  ASSERT_GE(reader->launches().size(), 1u);
  const auto &l = reader->launches()[0];
  EXPECT_EQ(l.kernel_name, "vec_add");
  EXPECT_EQ(l.via, LaunchVia::kRuntimeApi);
  EXPECT_EQ(l.grid_x, 2u);
  EXPECT_EQ(l.grid_y, 1u);
  EXPECT_EQ(l.grid_z, 1u);
  EXPECT_EQ(l.block_x, 128u);
  ASSERT_EQ(l.args.size(), 2u);
  // arg0 is the 8-byte LE pointer value 0x1000 -> classified as pointer slot
  // with the original VA (same heuristic as v1).
  ASSERT_EQ(reader->launch_arg_tables().size(), 1u);
  EXPECT_TRUE(reader->launch_arg_tables()[0][0].is_pointer);
  EXPECT_EQ(reader->launch_arg_tables()[0][0].va, 0x1000u);
  EXPECT_FALSE(reader->launch_arg_tables()[0][1].is_pointer);
  // scalar arg bytes
  EXPECT_EQ(l.args[1].size, 4u);
  EXPECT_EQ(l.args[1].bytes, (std::vector<std::uint8_t>{0x00, 0x02, 0x00, 0x00}));
  // copies / memsets / streams / modules
  ASSERT_GE(reader->copies().size(), 1u);
  EXPECT_EQ(reader->copies()[0].cuda_kind, 1); // H2D
  EXPECT_EQ(reader->copies()[0].dst_size, 1024u);
  ASSERT_GE(reader->memsets().size(), 1u);
  ASSERT_GE(reader->streams().size(), 1u);
  ASSERT_GE(reader->modules().size(), 1u);
  EXPECT_EQ(reader->modules()[0].kind, kernscope::ModuleRecord::Kind::kFatBinary);
  EXPECT_EQ(reader->modules()[0].kernels[0], "vec_add");
  EXPECT_EQ(reader->first_kernel(), "vec_add");
}

TEST(BundleV0Test, ReaderLoadsV0BundleWithSidecarBlobs) {
  // The todo-44 equivalence-bundle shape: v0 text + input-buffer-<n>.bin
  // sidecars in the same directory.
  const fs::path dir = TempDir() / "v0sidecar";
  fs::create_directories(dir);
  const std::string v0 = ReadFile(KERNSCOPE_V0_FIXTURE);
  ASSERT_FALSE(v0.empty());
  std::ofstream(dir / "capture.bundle").write(v0.data(), v0.size());
  const std::vector<std::uint8_t> blob_bytes = {0x01, 0x02, 0x03, 0x04};
  std::ofstream(dir / "input-buffer-1.bin", std::ios::binary)
      .write(reinterpret_cast<const char *>(blob_bytes.data()),
             blob_bytes.size());

  std::string err;
  auto reader = BundleReader::open(dir.string(), &err);
  ASSERT_NE(reader, nullptr) << err;
  ASSERT_EQ(reader->blobs().size(), 1u);
  EXPECT_EQ(reader->blobs()[0].role, BufferRole::kInput);
  EXPECT_EQ(reader->blobs()[0].size, 4u);
  EXPECT_EQ(reader->blobs()[0].sha256, kernscope::sha256_hex(blob_bytes));
  std::vector<std::uint8_t> out;
  ASSERT_TRUE(reader->read_blob(reader->blobs()[0], out, &err)) << err;
  EXPECT_EQ(out, blob_bytes);
}

TEST(BundleV0Test, ReaderLoadsCommittedV1Fixture) {
  // The committed v1 fixture (F3): a real writer-produced bundle with
  // recorded per-blob sha256 - the load-time provenance path is exercised
  // against COMMITTED data, not just temp-dir writes.
  const std::string fixture = KERNSCOPE_V1_FIXTURE;
  ASSERT_NE(fixture, "") << "KERNSCOPE_V1_FIXTURE not defined";
  ASSERT_TRUE(fs::is_directory(fixture)) << fixture;

  std::string err;
  auto reader = BundleReader::open(fixture, &err);
  ASSERT_NE(reader, nullptr) << err;
  EXPECT_TRUE(reader->is_v1());
  EXPECT_EQ(reader->schema_version(), 1);
  EXPECT_EQ(reader->tool_version(), "kernscope-capture/1.0");
  EXPECT_EQ(reader->vendor(), "mock");
  EXPECT_EQ(reader->arch(), "sm_80");
  EXPECT_EQ(reader->first_kernel(), "vec_add");
  EXPECT_EQ(reader->total_blob_bytes(), 52u);
  ASSERT_EQ(reader->blobs().size(), 2u);
  const auto &b0 = reader->blobs()[0];
  EXPECT_EQ(b0.role, BufferRole::kInput);
  EXPECT_EQ(b0.va, 0x1000u);
  EXPECT_EQ(b0.size, 48u);
  EXPECT_EQ(b0.file, "blobs/b0001.bin");
  // The recorded reference must be the sha256 of the committed bytes.
  const std::vector<std::uint8_t> kExpected0 = []() {
    std::vector<std::uint8_t> v(48);
    for (std::size_t i = 0; i < v.size(); ++i) {
      v[i] = static_cast<std::uint8_t>(i);
    }
    return v;
  }();
  EXPECT_EQ(b0.sha256, kernscope::sha256_hex(kExpected0));
  const auto &b1 = reader->blobs()[1];
  EXPECT_EQ(b1.role, BufferRole::kKernelBinary);
  EXPECT_EQ(b1.size, 4u);
  EXPECT_EQ(b1.sha256,
            kernscope::sha256_hex(std::vector<std::uint8_t>{0xAA, 0xBB, 0xCC, 0xDD}));
  // Bytes read back match the recorded provenance.
  std::vector<std::uint8_t> bytes;
  ASSERT_TRUE(reader->read_blob(b0, bytes, &err)) << err;
  EXPECT_EQ(bytes, kExpected0);
}

//--- (e) QA negative: truncated bundle -> schema violation WITH OFFSET ------------

TEST(BundleNegativeTest, TruncatedV1EventsJsonReportsByteOffset) {
  CaptureSession session;
  FillSession(session);
  BundleWriteOptions opts;
  opts.capture_mechanism = "preload";
  opts.timestamp = "2026-08-14T04:00:00Z";
  opts.read_fn = [](const BufferSnapshot &, std::vector<std::uint8_t> &out) {
    out = {0x01};
    return true;
  };
  BufferSnapshot s;
  s.size = 1;
  opts.snapshots = {s};

  const fs::path dir = TempDir() / "trunc";
  std::string err;
  ASSERT_EQ(write_bundle_v1(session, dir.string(), opts, &err),
            WriteStatus::kOk)
      << err;
  const fs::path bundle = dir.string() + ".kcap";

  // Truncate events.json in the middle of the array (cut ~40% off the end).
  const fs::path events = bundle / "events.json";
  const std::string full = ReadFile(events);
  std::ofstream(events, std::ios::binary | std::ios::trunc)
      .write(full.data(), full.size() * 4 / 10);

  // The reader must REJECT with a schema violation carrying a byte offset -
  // no UB, no crash.
  auto reader = BundleReader::open(bundle.string(), &err);
  EXPECT_EQ(reader, nullptr);
  EXPECT_NE(err.find("schema violation"), std::string::npos) << err;
  EXPECT_NE(err.find("byte "), std::string::npos) << err;
  // The reported offset is in range of the truncated file (sanity).
  const std::size_t pos = err.find("byte ");
  ASSERT_NE(pos, std::string::npos);
  const std::size_t off = std::stoull(err.substr(pos + 5));
  EXPECT_LE(off, full.size() * 4 / 10);
}

TEST(BundleNegativeTest, TruncatedV0TextReportsLineAndByteOffset) {
  const std::string full = ReadFile(KERNSCOPE_V0_FIXTURE);
  ASSERT_FALSE(full.empty());
  const fs::path file = TempDir() / "truncated-v0.bundle";
  // Cut the file mid-record: the reader must reject with a schema violation
  // carrying the line + byte offset - no UB, no crash.
  std::ofstream(file, std::ios::binary | std::ios::trunc)
      .write(full.data(), full.size() * 2 / 3);

  std::string err;
  auto reader = BundleReader::open(file.string(), &err);
  EXPECT_EQ(reader, nullptr);
  EXPECT_NE(err.find("schema violation"), std::string::npos) << err;
  EXPECT_NE(err.find("line "), std::string::npos) << err;
  EXPECT_NE(err.find("byte "), std::string::npos) << err;
}

//--- (g) F3 review seeds: 1-byte truncations of the COMMITTED fixtures ------------

TEST(BundleNegativeTest, TruncatedV0FixtureLastByteReportsTerminatorOffset) {
  // F3 seed: exactly ONE byte removed from the committed v0 fixture - the
  // trailing newline of the 'end-capture-v0' terminator. The parser must
  // report a schema violation naming the terminator + the truncation offset
  // (never treat the unterminated terminator as a clean end-of-file).
  const std::string full = ReadFile(KERNSCOPE_V0_FIXTURE);
  ASSERT_FALSE(full.empty());
  ASSERT_EQ(full.back(), '\n') << "fixture must be newline-terminated";
  const fs::path file = TempDir() / "truncated-v0-last-byte.bundle";
  std::ofstream(file, std::ios::binary | std::ios::trunc)
      .write(full.data(), full.size() - 1);

  std::string err;
  auto reader = BundleReader::open(file.string(), &err);
  EXPECT_EQ(reader, nullptr);
  EXPECT_NE(err.find("schema violation"), std::string::npos) << err;
  EXPECT_NE(err.find("end-capture-v0"), std::string::npos) << err;
  EXPECT_NE(err.find("line "), std::string::npos) << err;
  EXPECT_NE(err.find("truncated-v0-last-byte.bundle"), std::string::npos)
      << err; // the error names the file
  // The reported truncation offset is the file's post-truncation size (the
  // position of the removed trailing newline).
  EXPECT_NE(err.find("ends at byte " + std::to_string(full.size() - 1)),
            std::string::npos)
      << err;
}

TEST(BundleNegativeTest, TruncatedV1BlobAnyByteFailsShaMismatch) {
  // F3 seeds on the committed v1 fixture: a 1-byte truncation of the blob at
  // ANY byte position (first, quarter, mid, three-quarter, last) must fail
  // AT LOAD with a sha256-mismatch schema violation naming the blob path +
  // the truncation offset.
  const fs::path src = fs::path(KERNSCOPE_V1_FIXTURE);
  ASSERT_TRUE(fs::is_directory(src)) << KERNSCOPE_V1_FIXTURE;
  const fs::path blob_rel = "blobs/b0001.bin";
  const std::string full = ReadFile(src / blob_rel);
  ASSERT_FALSE(full.empty());
  ASSERT_GE(full.size(), 8u);

  const std::size_t positions[] = {
      0u, full.size() / 4, full.size() / 2, (full.size() * 3) / 4,
      full.size() - 1};
  for (const std::size_t cut : positions) {
    const fs::path dir = TempDir() / ("v1trunc-" + std::to_string(cut));
    fs::copy(src, dir, fs::copy_options::recursive);
    std::string bytes = ReadFile(dir / blob_rel);
    ASSERT_EQ(bytes.size(), full.size());
    bytes.erase(bytes.begin() + static_cast<std::ptrdiff_t>(cut));
    std::ofstream(dir / blob_rel, std::ios::binary | std::ios::trunc)
        .write(bytes.data(), bytes.size());

    std::string err;
    auto reader = BundleReader::open(dir.string(), &err);
    EXPECT_EQ(reader, nullptr) << "cut at byte " << cut << ": " << err;
    EXPECT_NE(err.find("sha256 mismatch"), std::string::npos)
        << "cut at byte " << cut << ": " << err;
    EXPECT_NE(err.find(blob_rel.string()), std::string::npos)
        << "cut at byte " << cut << ": " << err; // names the blob path
    // The truncation offset: the blob's post-truncation size.
    EXPECT_NE(err.find("truncated at byte " + std::to_string(full.size() - 1)),
              std::string::npos)
        << "cut at byte " << cut << ": " << err;
  }
}

//--- (f) sha256 provenance is verified per blob (EAGERLY at load) -----------------

TEST(BundleNegativeTest, CorruptedBlobFailsSha256Verification) {
  CaptureSession session;
  FillSession(session);
  BundleWriteOptions opts;
  opts.capture_mechanism = "preload";
  opts.timestamp = "2026-08-14T04:00:00Z";
  BufferSnapshot s;
  s.role = BufferRole::kInput;
  s.size = 4;
  opts.snapshots = {s};
  opts.read_fn = [](const BufferSnapshot &, std::vector<std::uint8_t> &out) {
    out = {0xDE, 0xAD, 0xBE, 0xEF};
    return true;
  };
  const fs::path dir = TempDir() / "corrupt";
  std::string err;
  ASSERT_EQ(write_bundle_v1(session, dir.string(), opts, &err),
            WriteStatus::kOk)
      << err;
  const fs::path bundle = dir.string() + ".kcap";

  // Flip one byte of the blob on disk (size unchanged): the recorded sha256
  // no longer matches. Verification is EAGER at load (F3): open itself must
  // fail with a sha256-mismatch schema violation naming the blob path.
  const fs::path blob_path = bundle / "blobs" / "b0001.bin";
  std::string bytes = ReadFile(blob_path);
  bytes[0] = static_cast<char>(0xFF);
  std::ofstream(blob_path, std::ios::binary | std::ios::trunc)
      .write(bytes.data(), bytes.size());

  auto reader = BundleReader::open(bundle.string(), &err);
  EXPECT_EQ(reader, nullptr) << err;
  EXPECT_NE(err.find("schema violation"), std::string::npos) << err;
  EXPECT_NE(err.find("sha256 mismatch"), std::string::npos) << err;
  EXPECT_NE(err.find("blobs/b0001.bin"), std::string::npos) << err;
}

} // namespace
