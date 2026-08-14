# kernscope-pass module: vectorize (todo 37).
#
# The vectorized load/store rewrite pass. Self-registers via
# REGISTER_PASS_CALLBACK (tools/pass/include/kernscope/pass/
# register_pass.h) from its OWN translation unit — no edits outside this
# directory are allowed (todo-36 isolation contract).
#
# Sources are added to the plugin target only; the unit test compiles the
# pass sources directly (the verify-track pattern: checks/oob links plugin
# sources into verify-bounds-test) so it can call vectorizeFunction() with
# its own analysis managers and assert the rewrite counts on the three
# committed fixtures (tests/pass/Inputs/vectorize/):
#   * strided_contig_loop.ll  — 4 loads + 4 stores vectorized
#   * misaligned.ll           — alignment not provable -> untouched
#   * addrspace3.ll           — shared memory -> untouched (v1 scope)
target_sources(kernscope-pass PRIVATE src/vectorize.cpp)
target_include_directories(kernscope-pass PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- unit tests (gtest) ----------------------------------------------------
# ctest hardware-gating convention (todo 3): exactly one of no-gpu |
# gpu-amd | gpu-nvidia-rental. opt-on-.ll only -> no-gpu.
find_package(GTest REQUIRED)

add_executable(vectorize-rewrite-test
  tests/vectorize_rewrite_test.cpp
  src/vectorize.cpp
)
target_include_directories(vectorize-rewrite-test PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(vectorize-rewrite-test SYSTEM PRIVATE
  ${LLVM_INCLUDE_DIRS})
target_compile_features(vectorize-rewrite-test PRIVATE cxx_std_20)
target_link_libraries(vectorize-rewrite-test PRIVATE LLVM GTest::gtest_main)
target_compile_definitions(vectorize-rewrite-test PRIVATE
  KERNSCOPE_VECTORIZE_AS_LIB
  VEC_FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/pass/Inputs/vectorize")
set_target_properties(vectorize-rewrite-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}")

add_test(NAME vectorize_rewrite COMMAND vectorize-rewrite-test)
set_tests_properties(vectorize_rewrite PROPERTIES LABELS "no-gpu")
