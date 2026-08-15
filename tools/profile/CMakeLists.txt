# tools/profile — kernscope-profile (C6 profiler track, todo 49).
#
# Todo 49 ships the two TRACERS the profiler is built on:
#
#   1. NVIDIA tracer (src/cupti_tracer.{h,cpp}): CUPTI activity tracing with
#      the kineto enable-set (CONCURRENT_KERNEL kind 10 + RUNTIME + DRIVER +
#      OVERHEAD; NEVER the serializing KERNEL kind 3), buffer request/
#      completed callbacks, cuptiActivityFlushAll(FORCED) at stop, record
#      parsing pinned to CUpti_ActivityKernel9 behind the VERSIONED ADAPTER
#      (include/kernscope/profile/record_adapter.h — the todo-49 deliverable:
#      Kernel9 <-> Kernel10 field mapping + the documented CUDA-13 migration
#      surface). Reuses libs/cupti-sub (todo 44): the shared subscription
#      library processes the MEMORY-kind records through its
#      handle_activity_record() and owns the session/allocation-map side; the
#      tracer owns the kernel-record parsing. Compile+unit ONLY locally
#      (cuda_cupti 12.6 headers); real-NVIDIA behavior is rental-gated.
#
#   2. AMD tracer (src/amd_tracer.cpp): a rocprofiler-sdk TOOL (exports
#      rocprofiler_configure) doing buffer-tracing KERNEL_DISPATCH records
#      (start/end timestamps, correlation_id, grid/workgroup, LDS/scratch
#      sizes) + HIP_RUNTIME_API tracing (correlation_id join), LIVE on
#      gfx1101 against the todo-5 layernorm_fused fixture.
#
# MUST NOT: use nvprof/Event/Metric APIs (removed/deprecated in CUDA 13);
# enable CUPTI_ACTIVITY_KIND_KERNEL (serializes kernels). CONCURRENT_KERNEL
# only.
#
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.

# --- CUPTI tracer (compile+unit locally; the adapter + parser) ----------------
# Gated on the todo-44 shared subscription library (which itself requires
# $CUPTI_HOME at configure time — configure inside `devenv shell`).
if(NOT TARGET kernscope::cupti_sub)
  message(STATUS "tools/profile: CUPTI tracer DISABLED (kernscope::cupti_sub "
                 "unavailable — no CUPTI_HOME at configure time)")
else()
  find_package(GTest REQUIRED)
  add_executable(profile-cupti-test tests/cupti_tracer_test.cpp)
  target_include_directories(profile-cupti-test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # CUPTI headers are SYSTEM so the project-wide -Werror never trips on
  # vendor declarations.
  target_include_directories(profile-cupti-test SYSTEM PRIVATE
    "$ENV{CUPTI_HOME}/include")
  # The reuse contract: the tracer's tests link the todo-44 shared
  # subscription library (kernscope::cupti_sub); the tracer itself (commit 2)
  # links it for the MEMORY-record routing + the fallback contract.
  target_link_libraries(profile-cupti-test PRIVATE
    kernscope::cupti_sub GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(profile-cupti-test TEST_PREFIX "profile_tracer:"
    PROPERTIES LABELS "no-gpu")
endif()
