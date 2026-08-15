//===- injection.cpp - CUDA_INJECTION64_PATH entry point ---------------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// The CUPTI injection contract (docs/research/capture-replay-research-brief.md
// §1): when CUDA_INJECTION64_PATH=<this .so> is set, the CUDA driver loads
// the library during CUDA init and calls its exported InitializeInjection().
// This is the NVIDIA-PRIMARY transport (the todo-43 LD_PRELOAD shim is NOT the
// primary mechanism): injection hooks from inside the driver, so it sees both
// runtime and driver API traffic, works with `-cudart static` apps and has no
// PLT-interposition holes (brief §2 pitfalls 1/5 - "THIS IS THE FLIP").
//
// The entry must be a plain extern "C" function with default visibility (the
// driver resolves it by name from an arbitrary .so). It must never throw and
// never fail hard: if the subscription cannot be established the library logs
// "CUPTI injection unavailable, falling back to LD_PRELOAD" and returns - the
// captured process keeps running (QA negative contract).
//
// The injection .so deliberately does NOT link libcupti.so (it is dlopened by
// cupti_capture.cpp): the same library binary therefore works as an injection
// library, an LD_PRELOAD'd capture and a directly-linked consumer.
//
//===----------------------------------------------------------------------===//

#include "kernscope/cupti_capture.h"

extern "C" {

/// CUDA driver injection entry point (CUDA_INJECTION64_PATH contract).
/// Called once during CUDA init; the driver ignores the return value, so we
/// use it only for logging.
__attribute__((visibility("default"))) void InitializeInjection() {
  kernscope::cupti::initialize_from_injection();
}

/// NVTX injection is out of scope (the capture does not consume NVTX); a
/// stub is exported so the driver's NVTX_INJECTION64_PATH probing never
/// fails to resolve.
__attribute__((visibility("default"))) void InitializeInjectionNvtx() {}

} // extern "C"
