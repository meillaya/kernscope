# kernscope-pass module: coalesce (todo 42, Wave 7).
#
# The coalescing ANALYSIS pass (IR-level consumer of the todo-15
# classification concept; no transforms). Self-registers via
# REGISTER_PASS_CALLBACK (tools/pass/include/kernscope/pass/register_pass.h)
# from its OWN translation unit — no edits outside this directory (todo-36
# isolation contract).
#
# IR-vs-PTX split: libs/ has NO shared coalescing classifier (verified
# 2026-08-15); the todo-15 analyzer keeps its classifier in
# tools/analyze/src/analyze_coalescing.cpp and this module implements the
# IR-level classifier itself (include/kernscope/pass/coalesce.h — the seam
# for a future libs/ extraction). Same vocabulary + same 32B-sector model
# on both levels; documented in src/coalesce.cpp.
#
# Unit test (passes/coalesce/tests/): compiles the pass sources directly
# (the verify-track pattern — links the module sources into the test
# binary) so it can drive classifyGlobalAccess()/analyzeFunction() with its
# own analysis managers and assert the classification classes on the
# committed fixtures (tests/pass/Inputs/coalesce/).
target_sources(kernscope-pass PRIVATE src/coalesce.cpp)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- unit tests (gtest) ----------------------------------------------------
# ctest hardware-gating convention (todo 3): exactly one of no-gpu |
# gpu-amd | gpu-nvidia-rental. opt-on-.ll only -> no-gpu.
find_package(GTest REQUIRED)

add_executable(coalesce-classify-test
  tests/coalesce_classify_test.cpp
  src/coalesce.cpp
)
target_include_directories(coalesce-classify-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(coalesce-classify-test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS})
target_compile_features(coalesce-classify-test PRIVATE cxx_std_20)
target_link_libraries(coalesce-classify-test PRIVATE LLVM GTest::gtest_main)
target_compile_definitions(coalesce-classify-test PRIVATE
  KERNSCOPE_COALESCE_AS_LIB
  COALESCE_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/pass/Inputs/coalesce")
set_target_properties(coalesce-classify-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}")

add_test(NAME coalesce_classify COMMAND coalesce-classify-test)
set_tests_properties(coalesce_classify PROPERTIES LABELS "no-gpu")
