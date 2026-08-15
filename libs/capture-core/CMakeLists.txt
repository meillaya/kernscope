# kernscope shared-core library tree.
#
# libs/capture-core (todo 43): the shared interception foundation of the
# capture/replay track (C4): interception table, ptr->size allocation map,
# event log and the minimal v0 bundle-writer interface (formal schema is todo
# 46's deliverable). Consumed by the LD_PRELOAD shim (libkernscope_intercept.so,
# built here), and later by the todo-44 CUPTI subscription and todo-45 HIP
# path. The unit tests run against the todo-21 mock runtime (libs/gpuruntime),
# no NVIDIA hardware needed -> no-gpu labels.
#
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.

# --- capture-core static library --------------------------------------------
# PIC because it is linked into the shim .so; hidden dynamic visibility so the
# shim's copies never interpose with an app/test binary's own copies (the shim
# exports ONLY its interposed API table + flush entry point). Static consumers
# (unit tests, smoke) are unaffected by the visibility preset.
add_library(kernscope_capture_core STATIC src/capture.cpp)
target_include_directories(kernscope_capture_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_capture_core PUBLIC cxx_std_20)
set_target_properties(kernscope_capture_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
add_library(kernscope::capture_core ALIAS kernscope_capture_core)

# --- unit tests (gtest), against the mock runtime (todo 21) -------------------
# TEST_PREFIX "capture_shim:" so `ctest -R capture_shim` selects this suite.
find_package(GTest REQUIRED)
add_executable(capture-shim-test tests/capture_core_test.cpp)
target_link_libraries(capture-shim-test PRIVATE
  kernscope::capture_core kernscope::gpuruntime GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(capture-shim-test TEST_PREFIX "capture_shim:"
  PROPERTIES LABELS "no-gpu")

# --- LD_PRELOAD shim + mock-cudart smoke --------------------------------------
# Compiled against the CUDA 12.6 headers ($CUDA_HOME, resolved by the dev
# shell) so every interposed signature is contract-checked at build time.
# The shim is COMPILE-ONLY here (no NVIDIA driver): the smoke proves the shim
# loads, intercepts and forwards against a binary that EXPORTS the cuda* and
# cu* symbols itself (the todo-21 mock-provides-the-symbols scenario). Real
# NVIDIA interception is rental-gated (scripts/rental-validate.sh).
# The .so lands at build/libkernscope_intercept.so (the acceptance command
# LD_PRELOAD=$PWD/build/libkernscope_intercept.so).
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime_api.h")
  add_library(kernscope_intercept SHARED shim/intercept.cpp)
  target_include_directories(kernscope_intercept SYSTEM PRIVATE
    "$ENV{CUDA_HOME}/include")
  # -ldl for dlsym(RTLD_NEXT); capture-core is STATICALLY linked in (no
  # runtime dependency, preload ordering unaffected).
  target_link_libraries(kernscope_intercept PRIVATE
    kernscope::capture_core ${CMAKE_DL_LIBS})
  set_target_properties(kernscope_intercept PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  add_library(kernscope::intercept ALIAS kernscope_intercept)

  # Mock-cudart shared library: exports the same cuda*/cu* symbols the shim
  # interposes on, backed by a deterministic fake device arena. The smoke
  # executable LINKS this library so its cuda* references are UNDEFINED in
  # the executable and resolved by the dynamic linker - an executable's own
  # DEFINITIONS would bind internally and could not be preloaded over (the
  # "mock provides the symbols" scenario; the real CUDA runtime is absent).
  add_library(kernscope_mockcudart SHARED tests/mockcudart.cpp)
  target_include_directories(kernscope_mockcudart SYSTEM PRIVATE
    "$ENV{CUDA_HOME}/include")
  add_library(kernscope::mockcudart ALIAS kernscope_mockcudart)

  # The smoke binary links the shim (DT_NEEDED: flush symbol + interposition
  # even without LD_PRELOAD) and the mock (the fake targets for RTLD_NEXT).
  # NEEDED order matters: the shim must precede the mock so the executable's
  # cuda* references resolve to the interposed symbols.
  add_executable(capture-shim-smoke tests/shim_smoke.cpp)
  target_include_directories(capture-shim-smoke SYSTEM PRIVATE
    "$ENV{CUDA_HOME}/include")
  target_link_libraries(capture-shim-smoke PRIVATE
    kernscope::intercept kernscope::capture_core kernscope::mockcudart)
  set_target_properties(capture-shim-smoke PROPERTIES
    BUILD_RPATH "$ORIGIN;${CMAKE_BINARY_DIR}")
  # Self-checking smoke: preloaded shim + fake runtime, asserts the capture
  # log contents itself. Runs with LD_PRELOAD set (the acceptance command
  # shape) and exits nonzero if interception produced no records.
  add_test(NAME capture_shim_smoke
    COMMAND capture-shim-smoke ${CMAKE_BINARY_DIR}/capture-smoke.log)
  set_tests_properties(capture_shim_smoke PROPERTIES
    LABELS "no-gpu"
    ENVIRONMENT "LD_PRELOAD=${CMAKE_BINARY_DIR}/libkernscope_intercept.so")
else()
  message(STATUS "capture-core: LD_PRELOAD shim DISABLED (no CUDA_HOME "
                 "headers at configure time; configure inside `devenv shell`)")
endif()
