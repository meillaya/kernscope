//===- capture_cupti.h - cupti-sub: shared CUPTI subscription lib -*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// libs/cupti-sub (todo 44): the NVIDIA-PRIMARY capture mechanism of the
// capture/replay track (C4) and the shared CUPTI subscription library the
// todo-49 profiler reuses. It records the SAME session model as the todo-43
// LD_PRELOAD shim (libs/capture-core, NEVER modified) and the todo-45 HIP
// path (libs/capture-hip, NEVER modified), so a bundle written from any
// mechanism parses identically. The todo-44 dual-mechanism equivalence test
// (ctest -R capture_equiv -L gpu-amd) consumes BOTH this library and
// libkernscope_capture_hip.so against the todo-24 portable kernel; the field
// correspondence it enforces is documented in docs/capture-mapping.md (this
// todo's deliverable).
//
// Mechanism (per docs/research/capture-replay-research-brief.md §1/§2 build
// recommendation 1 - CUPTI injection is the PRIMARY NVIDIA mechanism because
// it hooks from inside the driver, closing the driver-vs-runtime split and
// static-cudart holes of LD_PRELOAD):
//
//   1. SUBSCRIPTION: cuptiSubscribe + cuptiEnableCallback for the RUNTIME and
//      DRIVER callback domains (and the RESOURCE domain for module loads).
//      The library is transported via CUDA_INJECTION64_PATH (the driver loads
//      the .so and calls its exported InitializeInjection()), or by direct
//      link for local tools. Single-subscriber scope only (documented; see
//      "single-tool scope" below) - CUPTI 12.6 rejects a second concurrent
//      subscriber (CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED); v2
//      multi-subscriber APIs arrive in CUDA 13.3 and are a documented
//      migration surface (docs/capture-mapping.md).
//   2. CALLBACK CAPTURE: RUNTIME+DRIVER domain callbacks at CUPTI_API_ENTER
//      / CUPTI_API_EXIT. CUpti_CallbackData.functionParams points at the
//      actual API argument struct (generated_cuda_runtime_api_meta.h /
//      generated_cuda_meta.h layouts) and .symbolName carries the kernel name
//      on launch callbacks. Runtime launches (cudaLaunchKernel) expose
//      grid/block/shared/stream + the opaque args pointer array; per-argument
//      SIZES come from cuFuncGetParamInfo (driver domain EXIT callback) -
//      the documented layout resolver (brief §5) - and are merged per kernel.
//   3. ACTIVITY MEMORY RECORDS: CUpti_ActivityMemory4 records report
//      alloc/free with address + bytes WITHOUT any allocation hook - this is
//      the ptr->size allocation-map learner (GPGPU-Sim g_mallocPtr_Size
//      pattern, brief §5 heuristic 2).
//   4. MODULE LOADS: the RESOURCE domain CUPTI_CBID_RESOURCE_MODULE_LOADED
//      callback passes CUpti_ModuleResourceData (cubin bytes + size) -> the
//      v0 module record (sha256 provenance) + on-disk blob; cuModuleGetFunction
//      EXIT maps CUfunction -> kernel name for driver-API launches.
//
// Loading: as an injection library (CUDA_INJECTION64_PATH=<this .so>) the
// .so exports InitializeInjection(); when loaded into a process without
// injection (LD_PRELOAD of the todo-43 shim, local tools, tests) the same
// subscription code runs from try_initialize(). The negative path is a
// FIRST-CLASS contract: when the subscription cannot be established (no
// driver, another CUPTI subscriber active, libcupti missing) the library
// logs "CUPTI injection unavailable, falling back to LD_PRELOAD" and
// degrades - NEVER crashes, NEVER hangs (QA-asserted by the negative test).
//
// Recording contract ("MUST NOT alter application behavior beyond
// observation"): callbacks only READ the param structs CUPTI hands us and
// record into the session; the library never calls CUDA/CUPTI APIs that
// could re-enter the intercepted call (CUPTI docs warn the context can
// change between ENTER and EXIT of a runtime call - we never touch CUDA
// state from callbacks).
//
// SINGLE-TOOL SCOPE (documented, per the plan's MUST NOT): this library
// assumes it is the ONLY CUPTI subscriber in the captured process. No
// exclusivity is REQUIRED to build/unit/equiv-test it (the synthetic
// harness drives the dispatcher directly); the subscription establishment
// merely degrades to the fallback if CUPTI reports a concurrent subscriber.
//
// ENVIRONMENT VARIABLES (all optional; KERNSCOPE_* convention from the
// todo-43 shim / todo-45 HIP path):
//   KERNSCOPE_CUPTI_BUNDLE   bundle output path (v0 text via BundleWriter)
//   KERNSCOPE_CAPTURE_LOG    fallback bundle path (shim convention)
//   KERNSCOPE_CUPTI_BLOB_DIR directory for cubin-<id>.cubin /
//                             input-buffer-<n>.bin (default: ".")
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_CUPTI_CAPTURE_H
#define KERNSCOPE_CUPTI_CAPTURE_H

#include "kernscope/capture.h"

// The library's contract IS the CUPTI callback API (unlike capture-core,
// which is deliberately CUDA-header-free because the shim maps types): the
// callback dispatcher below is typed with CUPTI's own enums/structs so the
// synthetic test harness (tests/synthetic_cupti.h) builds exactly the
// objects CUPTI would deliver.
#include <cupti.h>

// Export macro: the library builds with hidden visibility (no symbol
// conflicts when injected/linked into arbitrary apps), so the documented API
// is re-exported explicitly - the same discipline as the injection entry.
#if defined(_WIN32)
#define KERNSCOPE_CUPTI_API
#else
#define KERNSCOPE_CUPTI_API __attribute__((visibility("default")))
#endif

namespace kernscope::cupti {

//===--- environment contract -------------------------------------------------===//

inline constexpr const char *kBundleEnv = "KERNSCOPE_CUPTI_BUNDLE";
inline constexpr const char *kCaptureLogEnv = "KERNSCOPE_CAPTURE_LOG";
inline constexpr const char *kBlobDirEnv = "KERNSCOPE_CUPTI_BLOB_DIR";
inline constexpr const char *kInjectEnv = "CUDA_INJECTION64_PATH";

/// Which transport established the capture (first-class in the bundle's
/// provenance story; todo 46 formalizes it in the manifest).
enum class Mechanism : std::uint8_t {
  kNone,             // not initialized
  kCuptiInjected,    // CUDA_INJECTION64_PATH (InitializeInjection entry)
  kCuptiLinked,      // try_initialize() from a directly-linked consumer
  kLdPreloadFallback // subscription failed -> caller falls back to the shim
};

/// Log line emitted verbatim on the negative path (QA-asserted).
inline constexpr const char *kFallbackLogLine =
    "CUPTI injection unavailable, falling back to LD_PRELOAD";

//===--- lifecycle ------------------------------------------------------------===//

/// Establish the CUPTI subscription (cuptiSubscribe + cuptiEnableCallback
/// for RUNTIME/DRIVER/RESOURCE domains + cuptiActivityEnable(MEMORY) for the
/// allocation-map learner). The CUPTI library is dlopened by path (it is not
/// a link dependency), so a driverless box or a missing libcupti degrades
/// gracefully: on ANY failure the function logs kFallbackLogLine and returns
/// false; it never crashes and never blocks (QA negative contract). Safe to
/// call more than once (idempotent; returns the current state).
bool KERNSCOPE_CUPTI_API try_initialize();

/// The injection entry's work (see injection.cpp). Same contract as
/// try_initialize(); the mechanism is reported as kCuptiInjected.
bool KERNSCOPE_CUPTI_API initialize_from_injection();

/// Synthetic-harness mode: arm the dispatcher with `session` WITHOUT
/// establishing a real CUPTI subscription. Unit tests and the equivalence
/// test's CUDA side drive handle_callback()/handle_activity_record() through
/// the synthetic harness (tests/synthetic_cupti.h) - the same dispatcher
/// CUPTI calls on hardware - while the real subscription establishment stays
/// reserved for try_initialize(). Idempotent; re-arming with a new session
/// replaces the recording target (tests only; never called by the injection
/// path).
void KERNSCOPE_CUPTI_API arm_session(CaptureSession *session);

/// True once a subscription is live.
bool KERNSCOPE_CUPTI_API initialized();

/// The transport that established the capture (kNone before try_initialize).
Mechanism KERNSCOPE_CUPTI_API mechanism();

/// The recording session (the process-wide CaptureSession by default; a
/// caller may pass its own session to try_initialize for hermetic tests).
CaptureSession &KERNSCOPE_CUPTI_API session();

//===--- the callback dispatcher ----------------------------------------------===//

/// The CUPTI callback installed via cuptiSubscribe: dispatches on
/// (domain, cbid, callbackSite) and records into session(). It is PUBLIC so
/// the synthetic harness (tests/synthetic_cupti.h, mock style of the
/// todo-43 mock-cudart) and the unit tests drive it without any GPU - the
/// same function the real CUPTI calls on hardware (compile+unit only locally;
/// real-NVIDIA behavior is rental-gated). `data` is the callback payload:
/// CUpti_CallbackData for the RUNTIME/DRIVER domains, CUpti_ResourceData for
/// the RESOURCE domain (cupti_callbacks.h - the real CUPTI delivers each
/// domain's own struct through this slot).
void KERNSCOPE_CUPTI_API handle_callback(CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
                     const void *data);

/// One activity record delivered from a completed CUPTI activity buffer
/// (cuptiActivityGetNextRecord). MEMORY records (CUpti_ActivityMemory4) feed
/// the ptr->size allocation map; everything else is ignored by the capture
/// (the todo-49 profiler consumes the other kinds through this same lib).
void KERNSCOPE_CUPTI_API handle_activity_record(const CUpti_Activity *record);

//===--- flush -----------------------------------------------------------------===//

/// Serialize the session through the v0 BundleWriter to
/// KERNSCOPE_CUPTI_BUNDLE (fallback KERNSCOPE_CAPTURE_LOG). Idempotent
/// (runs at atexit and from explicit callers; whichever comes first).
void KERNSCOPE_CUPTI_API flush_bundle();

} // namespace kernscope::cupti

#endif // KERNSCOPE_CUPTI_CAPTURE_H
