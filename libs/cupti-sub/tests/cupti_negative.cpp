//===- cupti_negative.cpp - negative QA: CUPTI injection unavailable -----*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// QA negative (todo 44, acceptance-asserted): when the CUPTI subscription
// cannot be established - the injection entry never ran (no
// CUDA_INJECTION64_PATH), no NVIDIA driver on this host, or another CUPTI
// subscriber is active - the capture must log
//     "CUPTI injection unavailable, falling back to LD_PRELOAD"
// and degrade: NO crash, NO hang (the ctest TIMEOUT guards the hang side).
//
// This process exercises the REAL negative path on this driverless box:
// try_initialize() dlopens the REAL libcupti.so from CUPTI_HOME and calls the
// REAL cuptiSubscribe; with no NVIDIA driver the subscription fails, the
// fallback line is logged to stderr, and the process must still exit 0.
// (The positive subscription path is exercised by the synthetic-harness unit
// tests; real-NVIDIA behavior is rental-gated.)
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture_cupti.h"

#include <cstdio>

int main() {
  std::fprintf(stderr, "[cupti-negative] attempting CUPTI subscription on a "
                       "driverless host...\n");
  const bool ok = kernscope::cupti::try_initialize();
  std::fprintf(stderr, "[cupti-negative] try_initialize() returned %s\n",
               ok ? "true" : "false");
  // The contract: graceful degradation - we REACH the end of main (no crash,
  // no hang) and report the fallback state.
  return ok ? 1 : 0;
}
