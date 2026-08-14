# libs/report: the report engine (wave-1 todo 7).
#
# One structured model (KernelReport/Finding) + three renderers:
#   * text - owner's example format (docs/research/owner-examples.md Section A)
#   * json - validated against docs/schema/report.schema.json (draft 2020-12)
#   * html - self-contained single file (inline CSS/JS, embedded JSON payload,
#            zero network requests; polykernel/tt-flow convention)
#
# Every metric is a machine-readable field; renderers throw SchemaViolation
# instead of silently emitting an invalid report (negative-path QA).
#
# ctest wiring (all LABELS no-gpu, i.e. runs on any machine):
#   report-fixture    - generates build/test_reports/{sample.html,sample.json}
#   report-html-check - python3 tests/report/check_html.py on the sample
#                       (parses embedded JSON with html.parser, compares to
#                       the expected input JSON; reused by todo 56)
#   report-negative   - a finding missing a required field (severity) must be
#                       rejected with exit code 42 + a message (WILL_FAIL
#                       inverts: ctest passes exactly when the renderer
#                       refuses to render)
#   gtest suite       - discovered per-case by gtest_discover_tests, all
#                       LABELS no-gpu

add_library(kernscope_report src/report.cpp)
target_include_directories(kernscope_report PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_report PUBLIC cxx_std_20)
add_library(kernscope::report ALIAS kernscope_report)

# Python3 (check_html.py + validate_json.py gates) and gtest.
# nixpkgs' gtest ships no CMake config in CMAKE_PREFIX_PATH; the devenv
# exposes it via pkg-config (gtest-1.17.0-dev in PKG_CONFIG_PATH), so
# pkg_check_modules is the deterministic path inside `devenv shell`.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GTEST REQUIRED gtest gtest_main)
find_package(Threads REQUIRED)

# --- gtest unit suite -------------------------------------------------------
add_executable(kernscope-report-test tests/report_test.cpp)
# CMake 4.3.4 (nixpkgs) BUG WORKAROUND: after a pkg_check_modules call,
# variable-sourced target_include_directories() is silently dropped from the
# generated Ninja INCLUDES (verified with a minimal repro: plain set() var,
# $CACHE{} var and even pkg vars all vanish; hardcoded literals survive).
# The -I therefore travels through target_compile_options (FLAGS), which is
# proven to survive; GTEST_CFLAGS_OTHER carries -DGTEST_HAS_PTHREAD=1.
# -isystem (not -I): gtest-1.17.0 headers trip -Wcharacter-conversion under
# clang-21 -Wextra, and third-party headers must not fail the -Werror gate.
target_compile_options(kernscope-report-test PRIVATE
  "-isystem${GTEST_INCLUDE_DIRS}" ${GTEST_CFLAGS_OTHER})
target_link_libraries(kernscope-report-test PRIVATE
  kernscope::report ${GTEST_LINK_LIBRARIES} Threads::Threads)
# Absolute paths the test shells out to (schema validation runs under
# python3; the validator is committed, no pip packages needed).
target_compile_definitions(kernscope-report-test PRIVATE
  PYTHON3="${Python3_EXECUTABLE}"
  REPORT_VALIDATOR="${CMAKE_SOURCE_DIR}/tests/report/validate_json.py"
  REPORT_SCHEMA="${CMAKE_SOURCE_DIR}/docs/schema/report.schema.json"
  REPORT_TMPDIR="${CMAKE_BINARY_DIR}/report-test-tmp"
)
include(GoogleTest)
# TEST_PREFIX keeps every discovered case matching `ctest -R report`
# (ctest -R is a case-sensitive regex; gtest suite names are CamelCase).
gtest_discover_tests(kernscope-report-test
  TEST_PREFIX "report:"
  PROPERTIES LABELS "no-gpu"
)

# --- sample report generator (feeds the check_html.py gate) -----------------
add_executable(kernscope-report-fixture tests/fixture_main.cpp)
target_link_libraries(kernscope-report-fixture PRIVATE kernscope::report)

add_test(NAME report-fixture
  COMMAND kernscope-report-fixture ${CMAKE_BINARY_DIR}/test_reports)
set_tests_properties(report-fixture PROPERTIES LABELS "no-gpu")

# Acceptance form: python3 tests/report/check_html.py build/test_reports/
# sample.html (expected JSON defaults to the sibling sample.json).
add_test(NAME report-html-check
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_SOURCE_DIR}/tests/report/check_html.py
          ${CMAKE_BINARY_DIR}/test_reports/sample.html)
set_tests_properties(report-html-check PROPERTIES
  LABELS "no-gpu"
  DEPENDS report-fixture
)

# --- negative path: schema-violating model must never render silently -------
# The tool exits 42 (asserted code) when the renderer rejects the finding
# with a message naming the missing field, 1 if it silently rendered.
# WILL_FAIL inverts ctest's expectation: the test passes exactly when the
# renderer refuses to emit the invalid report.
add_executable(kernscope-report-negative tests/negative_main.cpp)
target_link_libraries(kernscope-report-negative PRIVATE kernscope::report)

add_test(NAME report-negative COMMAND kernscope-report-negative)
set_tests_properties(report-negative PROPERTIES LABELS "no-gpu" WILL_FAIL TRUE)
