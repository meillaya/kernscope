// rocprofiler-sdk tool-registration stub — proves rocmPackages.rocprofiler-sdk
// headers + lib are usable for compile+link. In rocprofiler-sdk the
// "intercept table" is the tool entry point rocprofiler_configure()
// (rocprofiler-sdk/registration.h): the runtime (librocprofiler-sdk.so)
// dlsym()s it at process start and calls the returned initialize/finalize
// callbacks. This stub defines that entry point (the exact API the
// capture/replay + profiler modules will use) and calls a registration query
// so the linker must resolve the SDK lib.
//
// Build (inside `devenv shell`):
//   cc tests/env/rocprof_stub.c -o /tmp/rocprof_stub \
//     -I"$ROCPROF_SDK_DEV/include" -L"$ROCPROF_SDK_LIB" -lrocprofiler-sdk
// where ROCPROF_SDK_DEV/ROCPROF_SDK_LIB point at the package's dev/out
// outputs (set in check_env.sh's compile gate).
#include <rocprofiler-sdk/rocprofiler.h>
#include <rocprofiler-sdk/registration.h>

#include <stddef.h>
#include <stdio.h>

static int tool_initialize(rocprofiler_client_finalize_t finalize_func,
                           void *tool_data) {
  (void)finalize_func;
  (void)tool_data;
  return 0;
}

static void tool_finalize(void *tool_data) { (void)tool_data; }

// Tool entry point — looked up (dlsym) by the rocprofiler runtime.
rocprofiler_tool_configure_result_t *rocprofiler_configure(
    uint32_t version, const char *runtime_version, uint32_t priority,
    rocprofiler_client_id_t *client_id) {
  (void)version;
  (void)runtime_version;
  (void)priority;
  if (client_id)
    client_id->name = "kernscope-stub";
  static rocprofiler_tool_configure_result_t result = {
      .size = sizeof(result),
      .initialize = tool_initialize,
      .finalize = tool_finalize,
      .tool_data = NULL,
  };
  return &result;
}

int main(void) {
  // Forces the linker to resolve librocprofiler-sdk.
  int status = -1;
  rocprofiler_status_t rc = rocprofiler_is_initialized(&status);
  printf("rocprofiler-sdk stub OK: is_initialized rc=%d status=%d\n",
         (int)rc, status);
  return 0;
}
