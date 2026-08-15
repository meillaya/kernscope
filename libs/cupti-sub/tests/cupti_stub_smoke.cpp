//===- cupti_stub_smoke.cpp - cupti-sub link smoke -----------------------*- C++ -*-===//
//
// kernscope shared-core library tree.
//
// Acceptance: "cupti-sub library links into a stub without symbol conflicts".
// This stub plays the app-side role: it defines its OWN dummy functions with
// names a captured app might also export (kernscope-related + generic), links
// libkernscope_cupti_sub.so and exercises the public API surface. If the .so
// leaked conflicting global symbols, linking or running this would fail.
//
//===----------------------------------------------------------------------===//

#include "kernscope/capture_cupti.h"

#include <cstdio>

// App-side symbols that must not collide with the subscription library.
namespace kernscope {
namespace cupti {
namespace {
// An app-level redefinition of a namespace-local name would be a conflict
// if the library exported it - it does not (hidden visibility). We also
// define our own `session`-adjacent helper to prove name-space isolation.
inline int app_helper(int x) { return x + 1; }
} // namespace
} // namespace cupti
} // namespace kernscope

extern "C" int kernscope_app_side_exported() { return 42; }

int main() {
  // Exercise the public API surface (link + call smoke).
  const kernscope::cupti::Mechanism m = kernscope::cupti::mechanism();
  const bool init = kernscope::cupti::initialized();
  kernscope::CaptureSession &s = kernscope::cupti::session();
  s.record_alloc(0x1000, 64);
  kernscope::cupti::flush_bundle(); // no env set -> no-op, must not crash
  std::printf("cupti-stub-smoke: mechanism=%d initialized=%d app_helper=%d "
              "exported=%d allocs=%zu\n",
              static_cast<int>(m), init ? 1 : 0,
              kernscope::cupti::app_helper(1), kernscope_app_side_exported(),
              s.allocs().size());
  return s.allocs().size() == 1 ? 0 : 1;
}
