# kernscope-verify check module: race (todo 33).
#
# OWNS check class 4 (race-prone shared-memory writes) via the CheckPlugin
# contract (tools/verify/include/kernscope/verify/check_plugin.h —
# REGISTER_CHECK). Self-contained per the todo-33 isolation contract: the
# parent tools/verify/CMakeLists.txt is untouched; this file registers the
# plugin library and its ctest (verify_race) only.
#
# The plugin sources link into the test binary directly (the static
# registrar in race_check.cpp registers the plugin in the global
# CheckRegistry at load time — no shared-file edits).

add_library(kernscope_verify_race STATIC race_check.cpp)
target_include_directories(kernscope_verify_race PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include)
target_include_directories(kernscope_verify_race SYSTEM PUBLIC
  ${LLVM_INCLUDE_DIRS})
target_compile_features(kernscope_verify_race PUBLIC cxx_std_20)
target_link_libraries(kernscope_verify_race PUBLIC kernscope::verify LLVM)

# --- race check tests -------------------------------------------------------
# Acceptance (plan todo 33): manifest-driven verdicts over the corpus
# fixtures (tests/verify/corpus/race_*.cu + manifest.race.json) +
# QA-negative regression guards (data-dependent index -> UNKNOWN_RACE;
# uniform-branch exclusivity and SCEV-disjoint spans stay SAFE).
# ctest -R verify_race runs this binary (LABELS no-gpu: clang runs
# GPU-less, -emit-llvm only).
#
# NOTE: race_check.cpp is compiled into the test executable DIRECTLY (not
# only via the archive): the REGISTER_CHECK static registrar has no external
# references, so an archive member would never be extracted by the linker
# and the plugin would silently never run.
find_package(GTest REQUIRED)

add_executable(verify-race-test race_test.cpp race_check.cpp)
target_link_libraries(verify-race-test PRIVATE
  kernscope::verify GTest::gtest_main)
target_compile_definitions(verify-race-test PRIVATE
  RACE_CORPUS_DIR="${CMAKE_SOURCE_DIR}/tests/verify/corpus")
set_target_properties(verify-race-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}")
add_test(NAME verify_race COMMAND verify-race-test)
set_tests_properties(verify_race PROPERTIES LABELS "no-gpu")
