// CUPTI subscribe stub — proves $CUPTI_HOME headers + lib are usable for
// compile+link. Used by todo 2's acceptance gate (kernscope profiler/capture
// modules will build on this).
//
// Build (inside `devenv shell`):
//   nvcc tests/env/cupti_stub.cu -o /tmp/cupti_stub \
//     -I"$CUPTI_HOME/include" -L"$CUPTI_HOME/lib64" -lcupti
//
// This machine has no NVIDIA GPU, so cuptiSubscribe is expected to return an
// error status at runtime — the assertion is compile+link + clean run, not a
// successful subscription.
#include <cstdio>
#include <cupti.h>

int main() {
  CUpti_SubscriberHandle subscriber = nullptr;
  CUptiResult rc = cuptiSubscribe(&subscriber, nullptr, nullptr);
  // cuptiGetResultString is an OUT-param API: (result, const char** str).
  const char *msg = nullptr;
  cuptiGetResultString(rc, &msg);
  std::printf("cuptiSubscribe -> %d (%s)\n", (int)rc, msg ? msg : "?");
  if (rc != CUPTI_SUCCESS) {
    // Compile+link gate only: any runtime status is acceptable here, but we
    // still exit 0 so the gate asserts toolchain usability, not hardware.
    std::printf("note: no NVIDIA GPU on this host; subscription expected to fail at runtime\n");
  }
  return 0;
}
