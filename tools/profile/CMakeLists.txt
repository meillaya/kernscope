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

# --- per-kernel enrichment layer (todo 50) --------------------------------------
# The vendor-neutral enrichment library: theoretical occupancy + limiter via
# libs/occupancy (todo 11 — the ONLY occupancy engine, no cuda_occupancy.h
# port), the cuobjdump -res-usage / PTX -lineinfo / llvm-objdump -l parsers
# and the enriched JSONL shape. Deliberately CUPTI/HIP-free (pure libs/arch +
# libs/occupancy + std) so BOTH tracers and the report path can link it.
add_library(kernscope_profile_enrich STATIC src/enrich.cpp)
target_include_directories(kernscope_profile_enrich PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kernscope_profile_enrich PUBLIC kernscope::occupancy)
add_library(kernscope::profile_enrich ALIAS kernscope_profile_enrich)

# --- ranking + compiler-aware diagnostics engine (todo 51) ----------------------
# Hot-kernel ranking + the SOL-style "why is this slow?" rules engine
# (docs/research/profiler-research-brief.md §6): occupancy resource hints,
# libs/roofline (todo 12) memory-traffic classification for known families,
# the insufficient-evidence degradation, and the kernscope-analyze (todo 20)
# subprocess merge on extracted PTX (the profiler<->analyzer cross-tool
# multiplier). Reuses libs/roofline + libs/report + libs/db via their headers
# (MUST NOT modify libs/*). Pure host-side: no CUPTI/HIP/GPU required.
add_library(kernscope_profile_diag STATIC src/diagnostics.cpp)
target_include_directories(kernscope_profile_diag PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kernscope_profile_diag PUBLIC
  kernscope::profile_enrich kernscope::roofline kernscope::report
  kernscope::db)
add_library(kernscope::profile_diag ALIAS kernscope_profile_diag)

find_package(GTest REQUIRED)
add_executable(profile-diag-test tests/profile_diag_test.cpp)
target_include_directories(profile-diag-test PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(profile-diag-test PRIVATE
  PROFILE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/profile"
  KERNSCOPE_ANALYZE_CLI="$<TARGET_FILE:kernscope-analyze>")
target_link_libraries(profile-diag-test PRIVATE
  kernscope::profile_diag kernscope::profile_enrich kernscope::roofline
  kernscope::report kernscope::db GTest::gtest_main)
# The analyzer-merge acceptance INVOKES the REAL kernscope-analyze CLI (todo
# 20) on the extracted-PTX fixture: the test binary must be built after it.
# The analyzer itself is never modified (MUST NOT touch tools/analyze).
add_dependencies(profile-diag-test kernscope-analyze)
include(GoogleTest)
gtest_discover_tests(profile-diag-test TEST_PREFIX "profile_diag:"
  PROPERTIES LABELS "no-gpu")

# --- enrichment unit suite ------------------------------------------------------
# The todo-50 acceptance (no-gpu): occupancy field == libs/occupancy output
# (asserted EQUAL), the -lineinfo fixture -> source-mapping entries >= 1, the
# cubin-only fixture -> ptx_available == false with no PTX diagnostics, the
# QA negative -> registersPerThread==0 marks "unknown-resources" (never 0%),
# plus the cuobjdump -res-usage and llvm-objdump -l parsers against the
# committed fixtures.
find_package(GTest REQUIRED)
add_executable(profile-enrich-test tests/enrich_test.cpp)
target_include_directories(profile-enrich-test PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(profile-enrich-test PRIVATE
  PROFILE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/profile")
target_link_libraries(profile-enrich-test PRIVATE
  kernscope::profile_enrich GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(profile-enrich-test TEST_PREFIX "profile_enrich:"
  PROPERTIES LABELS "no-gpu")

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
    kernscope_profile_enrich
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

  # The todo-50 live enrichment check: the same layernorm run, additionally
  # asserting the code-object service facts on the kernel record — vgpr/sgpr
  # non-zero + LDS non-zero (the fixture uses __shared__), matching the
  # hipcc build — plus the enriched occupancy/limiter fields.
  add_test(NAME profile_enrich_amd
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profile_enrich_amd_check.py
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(profile_enrich_amd PROPERTIES LABELS "gpu-amd"
    TIMEOUT 600)

  # The todo-52 e2e GOLDEN (local HIP): build the layernorm_fused app into
  # build/tests/fixtures/layernorm_fused/, run `kernscope-profile <dir>
  # --json` piped through tests/fixtures/check_profile.py (numeric/range
  # assertions vs expected_layernorm_profile.json — no absolute durations),
  # plus the --text owner Section B shape and the --html libs/report
  # embedded-JSON parse check.
  add_test(NAME profile_cli_amd
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profile_cli_amd_check.py
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(profile_cli_amd PROPERTIES LABELS "gpu-amd"
    TIMEOUT 600)

  # The todo-52 QA negative: profile a binary with NO GPU kernels -> the
  # report says "no kernels traced" + exit 0 (not an error).
  add_test(NAME profile_cli_no_kernels
    COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/profile_cli_negative_check.py
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR})
  set_tests_properties(profile_cli_no_kernels PROPERTIES LABELS "gpu-amd"
    TIMEOUT 300)
endif()

# --- kernscope-profile CLI (todo 52) -------------------------------------------
# The Wave-9 closer: one command running trace (49) -> enrich (50) -> rank +
# diagnose (51) -> report (libs/report reuse). Pure host-side (the HIP tracer
# is invoked via LD_PRELOAD at runtime), so the CLI + its no-gpu units build
# unconditionally — the CUDA path is compile+unit only (rental-gated runtime).
add_executable(kernscope-profile src/main.cpp src/cli_pipeline.cpp)
target_include_directories(kernscope-profile PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(kernscope-profile PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src) # cli_pipeline.h lives with the impl
target_link_libraries(kernscope-profile PRIVATE
  kernscope::profile_diag kernscope::profile_enrich kernscope::roofline
  kernscope::report kernscope::db)
target_compile_definitions(kernscope-profile PRIVATE
  PROFILE_HIP_TOOL="${CMAKE_BINARY_DIR}/libkernscope_profile_hip.so"
  KERNSCOPE_ANALYZE_CLI="$<TARGET_FILE:kernscope-analyze>")
# The diag engine invokes the REAL kernscope-analyze CLI (todo 20) on PTX;
# AMD has no PTX path, but the CLI must still be built after it.
add_dependencies(kernscope-profile kernscope-analyze)
# Land the binary in <build>/bin/ (tt-flow convention, same as kernscope-analyze).
set_target_properties(kernscope-profile PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_test(NAME profile-cli-version COMMAND kernscope-profile --version)
set_tests_properties(profile-cli-version PROPERTIES LABELS "no-gpu")

# --- CLI unit suite (no-gpu) ---------------------------------------------------
# The todo-52 acceptance units: enriched-JSONL parsing back into the
# EnrichedKernel model, the "no kernels traced" QA negative (exit-0 report,
# never an error), the profile JSON + owner Section B text renderers, and the
# libs/report HTML path (embedded JSON payload). No GPU required.
find_package(GTest REQUIRED)
add_executable(profile-cli-test tests/profile_cli_test.cpp
  src/cli_pipeline.cpp)
target_include_directories(profile-cli-test PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(profile-cli-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(profile-cli-test PRIVATE
  kernscope::profile_diag kernscope::profile_enrich kernscope::roofline
  kernscope::report kernscope::db GTest::gtest_main)
target_compile_definitions(profile-cli-test PRIVATE
  KERNSCOPE_ANALYZE_CLI="$<TARGET_FILE:kernscope-analyze>")
include(GoogleTest)
gtest_discover_tests(profile-cli-test TEST_PREFIX "profile_cli:"
  PROPERTIES LABELS "no-gpu")
