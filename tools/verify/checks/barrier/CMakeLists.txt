# kernscope-verify check module: barrier (todo 32).
#
# OWNS check classes 3 (barrier correctness) + 6 (divergent barriers) via
# the CheckPlugin contract (tools/verify/include/kernscope/verify/
# check_plugin.h — REGISTER_CHECK). Self-contained per the todo-32
# isolation contract: the parent tools/verify/CMakeLists.txt is untouched;
# this file registers the plugin library and its ctest (verify_barrier)
# only.
#
# The plugin sources link into the test binary directly (the static
# registrar in barrier_check.cpp registers the plugin in the global
# CheckRegistry at load time — no shared-file edits).

add_library(kernscope_verify_barrier STATIC barrier_check.cpp)
target_include_directories(kernscope_verify_barrier PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(kernscope_verify_barrier SYSTEM PUBLIC
  ${LLVM_INCLUDE_DIRS})
target_compile_features(kernscope_verify_barrier PUBLIC cxx_std_20)
target_link_libraries(kernscope_verify_barrier PUBLIC kernscope::verify LLVM)

# --- barrier check tests ---------------------------------------------------
# Acceptance (plan todo 32): manifest-driven verdicts over the corpus
# fixtures (tests/verify/corpus/barrier_*.cu + manifest.barrier.json) +
# QA-negative regression guards (uniform loop-carried conditions stay SAFE).
# ctest -R verify_barrier runs this binary (LABELS no-gpu: clang runs
# GPU-less, -emit-llvm only).
#
# NOTE: barrier_check.cpp is compiled into the test executable DIRECTLY (not
# only via the archive): the REGISTER_CHECK static registrar has no external
# references, so an archive member would never be extracted by the linker
# and the plugin would silently never run.
find_package(GTest REQUIRED)

add_executable(verify-barrier-test barrier_test.cpp barrier_check.cpp)
target_link_libraries(verify-barrier-test PRIVATE
  kernscope::verify GTest::gtest_main)
target_compile_definitions(verify-barrier-test PRIVATE
  BARRIER_CORPUS_DIR="${CMAKE_SOURCE_DIR}/tests/verify/corpus")
set_target_properties(verify-barrier-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}")
add_test(NAME verify_barrier COMMAND verify-barrier-test)
set_tests_properties(verify_barrier PROPERTIES LABELS "no-gpu")
