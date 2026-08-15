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
  # The tracer itself: dlopen-based CUPTI activity subscription (kineto
  # enable-set), buffer request/completed callbacks, forced flush. Links the
  # todo-44 shared subscription library for the MEMORY-record routing and the
  # shared fallback contract.
  add_library(kernscope_profile_cupti STATIC src/cupti_tracer.cpp)
  target_include_directories(kernscope_profile_cupti PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(kernscope_profile_cupti SYSTEM PRIVATE
    "$ENV{CUPTI_HOME}/include")
  target_link_libraries(kernscope_profile_cupti PRIVATE
    kernscope::cupti_sub ${CMAKE_DL_LIBS})
  add_library(kernscope::profile_cupti ALIAS kernscope_profile_cupti)

  find_package(GTest REQUIRED)
  add_executable(profile-cupti-test tests/cupti_tracer_test.cpp)
  target_include_directories(profile-cupti-test PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(profile-cupti-test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src) # tracer_cupti.h lives with the impl
  target_include_directories(profile-cupti-test SYSTEM PRIVATE
    "$ENV{CUPTI_HOME}/include")
  target_link_libraries(profile-cupti-test PRIVATE
    kernscope::profile_cupti kernscope::cupti_sub GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(profile-cupti-test TEST_PREFIX "profile_tracer:"
    PROPERTIES LABELS "no-gpu")
endif()

# --- AMD tracer (live gfx1101) -------------------------------------------------
# The rocprofiler-sdk TOOL .so + the live acceptance against the todo-5
# layernorm_fused fixture. Gated on the SDK dev tree (same gate as
# libs/capture-hip, todo 45).
find_program(HIPCONFIG hipconfig REQUIRED)
execute_process(COMMAND ${HIPCONFIG} --includedir
  OUTPUT_VARIABLE HIP_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT DEFINED ENV{ROCPROF_SDK_DEV} OR
   NOT EXISTS "$ENV{ROCPROF_SDK_DEV}/include/rocprofiler-sdk/rocprofiler.h" OR
   NOT DEFINED ENV{ROCPROF_SDK_OUT} OR
   NOT EXISTS "$ENV{ROCPROF_SDK_OUT}/lib/librocprofiler-sdk.so")
  message(STATUS "tools/profile: AMD tracer DISABLED (rocprofiler-sdk not "
                 "found; set ROCPROF_SDK_DEV/ROCPROF_SDK_OUT, build inside "
                 "`devenv shell`)")
else()
  add_library(kernscope_profile_hip SHARED src/amd_tracer.cpp)
  target_include_directories(kernscope_profile_hip PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_include_directories(kernscope_profile_hip SYSTEM PRIVATE
    "${HIP_INCLUDE_DIR}" "$ENV{ROCPROF_SDK_DEV}/include")
  target_compile_definitions(kernscope_profile_hip PRIVATE __HIP_PLATFORM_AMD__)
  target_link_libraries(kernscope_profile_hip PRIVATE
    "$ENV{ROCPROF_SDK_OUT}/lib/librocprofiler-sdk.so")
  set_target_properties(kernscope_profile_hip PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  add_library(kernscope::profile_hip ALIAS kernscope_profile_hip)

  # The live check: build the layernorm app (hipcc gfx1101), run it under
  # LD_PRELOAD of the tool, parse the JSONL profile output and assert >= 1
  # kernel record with duration_ns > 0, grid/workgroup matching the launch
  # (M=1024 blocks x 128 threads) and the correlation_id linking the kernel
  # dispatch record to the hipLaunchKernel HIP-API record.
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME profile_tracer_amd
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profile_amd_check.py
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(profile_tracer_amd PROPERTIES LABELS "gpu-amd"
    TIMEOUT 600)
endif()
