# kernscope-analyze: the static GPU-kernel analyzer.
#
# Wave-3 todos 13-20 build the ten analyses (register pressure, occupancy,
# coalescing, bank conflicts, divergence, barriers, instruction mix, tensor
# core / MFMA / WMMA usage, spills, latency bottlenecks) + the suggestion
# engine. Todos 13-18 land the first six analysis modules as the shared
# kernscope_analyze library; each module ships its own gtest executable with
# a TEST_PREFIX so `ctest -R analyze_<name>` runs exactly that module.
#
# The six modules share one parsed-PTX/AMDGPU IR (libs/ptxparse +
# libs/amdparse), one report model (libs/report), one arch model (libs/arch)
# and one occupancy engine (libs/occupancy) — all consumed via headers, never
# modified here.

# --- the analysis library ---------------------------------------------------
add_library(kernscope_analyze STATIC
  src/analyze_regs.cpp
  src/analyze_occupancy.cpp
  src/analyze_coalescing.cpp
  src/analyze_banks.cpp
  src/analyze_divergence.cpp
  src/analyze_mix.cpp
)
target_include_directories(kernscope_analyze PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_analyze PUBLIC cxx_std_20)
add_library(kernscope::analyze ALIAS kernscope_analyze)
target_link_libraries(kernscope_analyze PUBLIC
  kernscope::ptxparse
  kernscope::amdparse
  kernscope::arch
  kernscope::occupancy
  kernscope::report
)

# --- the CLI tool (todo-3 stub, kept building; todo 20 wires the CLI) -------
# The stub proves the target graph and the ctest LABELS convention under
# -Wall -Wextra -Werror. It links the analysis library so the tool target
# stays the single entry point once todo 20 lands.
add_executable(kernscope-analyze main.cpp)
target_link_libraries(kernscope-analyze PRIVATE kernscope::analyze)

# Land the binary in <build>/bin/ (tt-flow convention) so lit's
# KERNSCOPE_TOOLS_DIR (${CMAKE_BINARY_DIR}/bin) finds it directly.
set_target_properties(kernscope-analyze PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ctest hardware-gating convention: every test carries a LABELS property of
# exactly one of no-gpu | gpu-amd | gpu-nvidia-rental. This stub runs on any
# machine, so it is labeled no-gpu.
add_test(NAME analyze-version COMMAND kernscope-analyze --version)
set_tests_properties(analyze-version PROPERTIES LABELS "no-gpu")

# --- unit tests (gtest) -----------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# All six analyzer modules parse host-side text and run no GPU -> no-gpu.
find_package(GTest REQUIRED)

# Shared compile definitions: the committed fixture corpus + the real
# fused_attention.ptx fixture + the expected golden JSON + its manifest.
set(ANALYZE_TEST_DEFS
  TEST_CORPUS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/analyze/corpus"
  FIXTURES_DIR="${CMAKE_SOURCE_DIR}/tests/fixtures"
)

# --- TODO 13: register pressure + spill detection --------------------------
add_executable(analyze-regs-test tests/analyze_regs_test.cpp)
target_link_libraries(analyze-regs-test PRIVATE kernscope::analyze
  GTest::gtest_main)
target_compile_definitions(analyze-regs-test PRIVATE ${ANALYZE_TEST_DEFS})
include(GoogleTest)
gtest_discover_tests(analyze-regs-test
  TEST_PREFIX "analyze_regs:"
  PROPERTIES LABELS "no-gpu"
)

# --- TODO 14: occupancy limits + main limiter -------------------------------
add_executable(analyze-occ-test tests/analyze_occupancy_test.cpp)
target_link_libraries(analyze-occ-test PRIVATE kernscope::analyze
  GTest::gtest_main)
target_compile_definitions(analyze-occ-test PRIVATE ${ANALYZE_TEST_DEFS})
gtest_discover_tests(analyze-occ-test
  TEST_PREFIX "analyze_occ:"
  PROPERTIES LABELS "no-gpu"
)

# --- TODO 15: global memory coalescing --------------------------------------
# (test executable added when tests/analyze_coalescing_test.cpp lands with the
# todo-15 commit)

# --- TODO 16: shared-memory bank conflicts ----------------------------------
# (test executable added when tests/analyze_banks_test.cpp lands with the
# todo-16 commit)

# --- TODO 17: warp divergence + barrier usage -------------------------------
# (test executable added when tests/analyze_divergence_test.cpp lands with the
# todo-17 commit)

# --- TODO 18: instruction mix + tensor core + latency -----------------------
# (test executable added when tests/analyze_mix_test.cpp lands with the
# todo-18 commit)
