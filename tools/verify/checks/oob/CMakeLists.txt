# kernscope-verify check module: oob (todo 31).
#
# Check classes 1 (global OOB) + 2 (shared-memory bounds):
#   * kernscope_check_oob  — the BoundsCheckPlugin sources (registered into
#     the global CheckRegistry via REGISTER_CHECK; linked into the harness
#     and test binaries so the core engine dispatches it).
#   * verify-bounds-test   — corpus-backed acceptance tests (ctest name
#     `verify_bounds`): ingests the committed corpus fixtures
#     tests/verify/corpus/ and asserts the plugin's verdicts against the
#     manifest fragment manifest.oob.json (schema: JSON array of
#     {file, kernel, check, expected_verdict}).
#
# Sources register themselves (check_plugin.h REGISTER_CHECK), so this file
# is the ONLY shared surface the module owns — no edits outside this
# directory (todo 30 isolation contract).

# The plugin library (self-registering via REGISTER_CHECK).  NOTE: a static
# archive only pulls objects that resolve referenced symbols, which would
# drop the plugin's static registrar — consumers must either link the
# sources directly (like verify-bounds-test below) or use --whole-archive.
add_library(kernscope_check_oob STATIC
  memory_bounds_check.cpp
  oob_common.cpp
)
target_link_libraries(kernscope_check_oob PUBLIC kernscope::verify)

find_package(GTest REQUIRED)

# Plugin sources are compiled into the test executable DIRECTLY so the
# REGISTER_CHECK static registrar is linked in and the core engine dispatches
# the plugin for every kernel.
add_executable(verify-bounds-test
  verify_bounds_test.cpp
  memory_bounds_check.cpp
  oob_common.cpp
)
target_link_libraries(verify-bounds-test PRIVATE
  kernscope::verify
  GTest::gtest_main
)
target_compile_definitions(verify-bounds-test PRIVATE
  OOB_CORPUS_DIR="${CMAKE_SOURCE_DIR}/tests/verify/corpus"
)
set_target_properties(verify-bounds-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}"
)

add_test(NAME verify_bounds COMMAND verify-bounds-test)
set_tests_properties(verify_bounds PROPERTIES LABELS "no-gpu")
