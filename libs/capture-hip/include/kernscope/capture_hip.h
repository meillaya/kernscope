//===- capture_hip.h - capture-hip: rocprofiler-sdk HIP capture -*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libs/capture-hip (todo 45): the LOCAL LIVE HIP capture loop on gfx1101.
// This library is a rocprofiler-sdk TOOL: it exports `rocprofiler_configure`
// and is loaded into the captured process via LD_PRELOAD (or
// ROCPROFILER_LIBRARY). It records the HIP application's observable behavior
// into libs/capture-core's CaptureSession - the SAME session model the
// todo-43 LD_PRELOAD shim and todo-44 CUPTI path use - so a bundle written
// from any mechanism parses identically.
//
// Mechanism (per docs/report/kernscope-technical-report.pdf and the
// Kerncap/intellikit HSA-layer precedent, MIT - pattern only, attribution in
// THIRD-PARTY.md):
//   1. INTERCEPT TABLE: rocprofiler_at_intercept_table_registration(
//      ROCPROFILER_HIP_RUNTIME_TABLE) hands the tool the live HipDispatchTable
//      of libamdhip64; the tool wraps the allocation hooks (hipMalloc ->
//      ptr->size allocation map, hipFree), copies/memsets and the launch
//      entry points. This is the "HIP allocation hooks" of the plan.
//   2. KERNEL DISPATCH callback tracing: grid/workgroup at the HSA dispatch
//      layer; the kernel name is resolved through the code-object
//      kernel-symbol registration callback (kernel_id -> name table).
//   3. CODE OBJECT callback tracing: the loaded HSACO is registered into the
//      session (sha256 provenance in the v0 module record) and dumped next to
//      the bundle as code-object-<id>.hsaco (the kernel binary blob).
//   4. H2D MEMCPY CONTENT: the host source of every hipMemcpyHostToDevice is
//      dumped as input-buffer-<n>.bin (v0-scope of the Kerncap address-space
//      closure; the VA-faithful snapshot policy is todo 46's deliverable).
//
// The deprecated tracer is NOT used.
//
// ENVIRONMENT VARIABLES (all optional):
//   KERNSCOPE_HIP_BUNDLE    bundle output path (v0 text via BundleWriter)
//   KERNSCOPE_CAPTURE_LOG   fallback bundle path (shim convention)
//   KERNSCOPE_HIP_BLOB_DIR  directory for code-object-*.hsaco /
//                           input-buffer-*.bin (default: "." )
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CAPTURE_HIP_H
#define KERNSCOPE_CAPTURE_HIP_H

#endif // KERNSCOPE_CAPTURE_HIP_H
