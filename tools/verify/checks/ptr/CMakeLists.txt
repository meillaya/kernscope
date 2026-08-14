# kernscope-verify check module: ptr (todo 34).
#
# Check classes 5 (illegal pointer arithmetic) + 7 (uninitialized reads) + 8
# (invalid address-space casts):
#   * kernscope_check_ptr     — the PointerDisciplineCheckPlugin sources
#     (registered into the global CheckRegistry via REGISTER_CHECK; linked
#     into the harness and test binaries so the core engine dispatches it).
#   * verify-ptr-test         — corpus-backed acceptance tests (ctest name
#     `verify_ptr`): ingests the committed corpus fixtures
#     tests/verify/corpus/ and asserts the plugin's verdicts against the
#     manifest fragment manifest.ptr.json (schema: JSON array of
#     {file, kernel, check, expected_verdict}).
#
# Sources register themselves (check_plugin.h REGISTER_CHECK), so this file
# is the ONLY shared surface the module owns — no edits outside this
# directory (todo 30 isolation contract).

# The plugin library (self-registering via REGISTER_CHECK).  NOTE: a static
# archive only pulls objects that resolve referenced symbols, which would
# drop the plugin's static registrar — consumers must either link the
# sources directly (like verify-ptr-test below) or use --whole-archive.
add_library(kernscope_check_ptr STATIC
  ptr_check.cpp
  ptr_common.cpp
)
target_link_libraries(kernscope_check_ptr PUBLIC kernscope::verify)

find_package(GTest REQUIRED)

# Plugin sources are compiled into the test executable DIRECTLY so the
# REGISTER_CHECK static registrar is linked in and the core engine dispatches
# the plugin for every kernel.
add_executable(verify-ptr-test
  verify_ptr_test.cpp
  ptr_check.cpp
  ptr_common.cpp
)
target_link_libraries(verify-ptr-test PRIVATE
  kernscope::verify
  GTest::gtest_main
)
target_compile_definitions(verify-ptr-test PRIVATE
  PTR_CORPUS_DIR="${CMAKE_SOURCE_DIR}/tests/verify/corpus"
)
set_target_properties(verify-ptr-test PROPERTIES
  BUILD_RPATH "${LLVM_LIBRARY_DIR}"
)

add_test(NAME verify_ptr COMMAND verify-ptr-test)
set_tests_properties(verify_ptr PROPERTIES LABELS "no-gpu")
