# kernscope shared-core library tree.
#
# libs/cupti-sub (todo 44): the NVIDIA-PRIMARY capture mechanism and the
# shared CUPTI subscription library the todo-49 profiler reuses. It records
# into libs/capture-core's CaptureSession (NEVER modified) the same way the
# todo-43 LD_PRELOAD shim and the todo-45 HIP path do, so a bundle written
# from any mechanism parses identically. The dual-mechanism equivalence test
# (ctest -R capture_equiv -L gpu-amd) consumes BOTH this library and
# libkernscope_capture_hip.so against the todo-24 portable kernel; the field
# correspondence is docs/capture-mapping.md.
#
# Build contract: compiled against the cuda_cupti 12.6 headers ($CUPTI_HOME,
# resolved by the dev shell) - compile+unit ONLY locally (no NVIDIA runtime
# here; real-NVIDIA CUPTI is rental-gated per scripts/rental-validate.sh).
# libcupti.so is NOT a link dependency: it is dlopened at runtime by
# cupti_capture.cpp so the negative path (injection unavailable) degrades
# gracefully on a driverless box.
#
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.

if(NOT DEFINED ENV{CUPTI_HOME} OR
   NOT EXISTS "$ENV{CUPTI_HOME}/include/cupti.h")
  message(STATUS "cupti-sub DISABLED (no CUPTI_HOME headers at configure "
                 "time; configure inside `devenv shell`)")
  return()
endif()

# --- the shared subscription library ---------------------------------------
# Hidden visibility: only the documented API + InitializeInjection are
# exported, so the .so links into a stub / into captured apps without symbol
# conflicts (acceptance: link smoke).
add_library(kernscope_cupti_sub SHARED
  src/cupti_capture.cpp src/injection.cpp)
target_include_directories(kernscope_cupti_sub PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
# CUPTI headers are SYSTEM so the project-wide -Werror never trips on vendor
# declarations.
target_include_directories(kernscope_cupti_sub SYSTEM PRIVATE
  "$ENV{CUPTI_HOME}/include")
# capture-core is STATICALLY linked in (no runtime dependency); -ldl for
# dlopen/dlsym of libcupti.so. PUBLIC so the capture-core include dir (the
# public header includes kernscope/capture.h) propagates to consumers.
target_link_libraries(kernscope_cupti_sub PUBLIC
  kernscope::capture_core)
target_link_libraries(kernscope_cupti_sub PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(kernscope_cupti_sub PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_library(kernscope::cupti_sub ALIAS kernscope_cupti_sub)

# --- unit tests (gtest), synthetic CUPTI-callback harness --------------------
# TEST_PREFIX "capture_cupti:" so `ctest -R capture_cupti` selects this
# suite. The harness builds the exact objects CUPTI would deliver
# (tests/synthetic_cupti.h) and drives the SAME dispatcher function CUPTI
# calls on hardware - the mock style of the todo-43 mock-cudart.
find_package(GTest REQUIRED)
add_executable(cupti-sub-test tests/cupti_capture_test.cpp)
target_include_directories(cupti-sub-test SYSTEM PRIVATE
  "$ENV{CUPTI_HOME}/include")
target_link_libraries(cupti-sub-test PRIVATE
  kernscope::cupti_sub kernscope::capture_core GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(cupti-sub-test TEST_PREFIX "capture_cupti:"
  PROPERTIES LABELS "no-gpu")

# --- link smoke --------------------------------------------------------------
# The stub defines its own dummy symbols (the app-side role) and links the
# .so: proves cupti-sub links without symbol conflicts and its API is usable
# from a plain consumer.
add_executable(cupti-stub-smoke tests/cupti_stub_smoke.cpp)
target_include_directories(cupti-stub-smoke SYSTEM PRIVATE
  "$ENV{CUPTI_HOME}/include")
target_link_libraries(cupti-stub-smoke PRIVATE kernscope::cupti_sub)
add_test(NAME capture_cupti_stub_link COMMAND cupti-stub-smoke)
set_tests_properties(capture_cupti_stub_link PROPERTIES LABELS "no-gpu")

# --- negative QA: injection unavailable ---------------------------------------
# Driverless box: try_initialize() dlopens the REAL libcupti.so and calls the
# REAL cuptiSubscribe; with no NVIDIA driver the subscription fails and the
# library must log "CUPTI injection unavailable, falling back to LD_PRELOAD"
# and return false - no crash, no hang (asserted; test timeout guards hang).
add_executable(cupti-negative tests/cupti_negative.cpp)
target_include_directories(cupti-negative SYSTEM PRIVATE
  "$ENV{CUPTI_HOME}/include")
target_link_libraries(cupti-negative PRIVATE kernscope::cupti_sub)
add_test(NAME capture_cupti_negative COMMAND cupti-negative)
set_tests_properties(capture_cupti_negative PROPERTIES LABELS "no-gpu"
  TIMEOUT 60)
