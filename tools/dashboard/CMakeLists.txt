# tools/dashboard — kernscope-dashboard (todo 56): the pure-Python aggregator
# that folds the committed per-module reports (analyzer goldens, tuning runs,
# verify corpus, capture/replay evidence, profile traces, pass deltas) into
# ONE self-contained dashboard HTML (libs/report conventions: inline
# CSS/JS, embedded JSON payload, zero network requests; validated with the
# todo-7 pattern — python3 html.parser payload extraction — via
# tests/report/check_dashboard.py + check_html.py).
#
# Reuses libs/report and the six module tools by CONSUMING their committed
# reports (reports/, tests/); it never regenerates module artifacts. Pure
# Python3 stdlib — no C++ targets; the CLI is copied to <build>/bin/ next to
# the other kernscope-* binaries so `kernscope-dashboard` resolves like any
# other tool binary.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(KERNSCOPE_DASH_SRC ${CMAKE_CURRENT_SOURCE_DIR}/kernscope-dashboard)
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${KERNSCOPE_DASH_SRC} ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard
  COMMAND chmod +x ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard
  DEPENDS ${KERNSCOPE_DASH_SRC}
  COMMENT "install kernscope-dashboard -> build/bin")
add_custom_target(kernscope-dashboard-bin ALL
  DEPENDS ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard)

# --- no-gpu ctest suite ------------------------------------------------------
# dashboard-version: CLI answers --version.
# dashboard-html:    happy path — generate + assert all six module sections,
#                    provenance.measured on each, zero external assets.
# dashboard-negative: QA negative — --missing verify,profile renders "no data"
#                    panels + available:false JSON flags, page stays valid.
add_test(NAME dashboard-version
  COMMAND ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard --version)
set_tests_properties(dashboard-version PROPERTIES LABELS "no-gpu")

add_test(NAME dashboard-html
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_check.sh
          ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard
          ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/report/check_dashboard.py
          ${CMAKE_CURRENT_BINARY_DIR}/gen-happy happy)
set_tests_properties(dashboard-html PROPERTIES LABELS "no-gpu" TIMEOUT 120)

add_test(NAME dashboard-negative
  COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_check.sh
          ${CMAKE_BINARY_DIR}/bin/kernscope-dashboard
          ${CMAKE_CURRENT_SOURCE_DIR}/../../tests/report/check_dashboard.py
          ${CMAKE_CURRENT_BINARY_DIR}/gen-negative negative)
set_tests_properties(dashboard-negative PROPERTIES LABELS "no-gpu" TIMEOUT 120)
