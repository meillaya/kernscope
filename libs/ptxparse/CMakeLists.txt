# kernscope shared-core library tree.
#
# libs/ptxparse: the PTX 8.x text parser (wave-2 todo 9). A hand-written
# lexer+parser (ptx_lexer.cpp/ptx_parser.cpp) turning nvcc/ptxas PTX source
# into a typed IR: directives, variable declarations with state space +
# alignment, .entry/.func bodies, instructions with opcode/operands/@p
# predicates, a per-kernel CFG (basic blocks + branch/fallthrough edges), and
# a tensor-core taxonomy (mma.sync/wmma/wgmma/tcgen05). Parse-only: never
# assembles, never executes, never touches NVIDIA bytecode tooling.
#
# The kernscope-ptxdump tool target lives here (source in tools/ptxdump/
# main.cpp) so the parse library and its consumer ship together; it is placed
# in <build>/bin/ (tt-flow convention) like kernscope-analyze.
add_library(kernscope_ptxparse STATIC src/ptx_lexer.cpp src/ptx_parser.cpp)
target_include_directories(kernscope_ptxparse PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_ptxparse PUBLIC cxx_std_20)
add_library(kernscope::ptxparse ALIAS kernscope_ptxparse)

# --- kernscope-ptxdump: JSON dump of a parsed module -------------------------
# Reuses the self-contained JSON value model from libs/db (no external JSON
# dependency; canonical serialization is gtest-covered there).
add_executable(kernscope-ptxdump ${CMAKE_SOURCE_DIR}/tools/ptxdump/main.cpp)
target_link_libraries(kernscope-ptxdump PRIVATE kernscope::ptxparse kernscope::db)
set_target_properties(kernscope-ptxdump PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# Parsing PTX text is pure host-side (no GPU required) -> no-gpu.
find_package(GTest REQUIRED)
add_executable(ptxparse-test tests/ptxparse_test.cpp)
target_link_libraries(ptxparse-test PRIVATE kernscope::ptxparse kernscope::db
  GTest::gtest_main)
target_compile_definitions(ptxparse-test PRIVATE
  TEST_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ptxparse/corpus"
  REAL_FIXTURE="${CMAKE_SOURCE_DIR}/tests/fixtures/fused_attention.ptx"
)
include(GoogleTest)
gtest_discover_tests(ptxparse-test
  TEST_PREFIX "ptxparse:"
  PROPERTIES LABELS "no-gpu"
)

# --- ctest: ptxdump acceptance gates ------------------------------------------
# 1) The real nvcc fixture must parse (exit 0 implies success).
add_test(NAME ptxparse-ptxdump-real-fixture
  COMMAND $<TARGET_FILE:kernscope-ptxdump>
          ${CMAKE_SOURCE_DIR}/tests/fixtures/fused_attention.ptx)
set_tests_properties(ptxparse-ptxdump-real-fixture PROPERTIES LABELS "no-gpu")

# 2) The JSON shape asserted (python3; committed checker).
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME ptxparse-ptxdump-json-check
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tests/ptxparse/check_dump.py
          $<TARGET_FILE:kernscope-ptxdump>
          ${CMAKE_SOURCE_DIR}/tests/fixtures/fused_attention.ptx)
set_tests_properties(ptxparse-ptxdump-json-check PROPERTIES
  LABELS "no-gpu"
  DEPENDS ptxparse-ptxdump-real-fixture
)

# 3) Negative path: malformed/truncated input must exit non-zero with a
#    line:col diagnostic (WILL_FAIL inverts: ctest passes exactly when the
#    tool refuses to emit a partial dump - the report-negative pattern).
add_test(NAME ptxparse-ptxdump-negative
  COMMAND $<TARGET_FILE:kernscope-ptxdump>
          ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/ptxparse/corpus/truncated_kernel.ptx)
set_tests_properties(ptxparse-ptxdump-negative PROPERTIES
  LABELS "no-gpu"
  WILL_FAIL TRUE
)
