# kernscope shared-core library tree.
#
# libs/arch: the GPU architecture-model database (constants + occupancy
# formulas for NVIDIA sm_75..sm_120 and AMD gfx90a/gfx942/gfx1101) consumed by
# tools/analyze and the rest of the suite, plus the version header every tool
# stamps. Wave-1 todo 6 implements the DB (include/kernscope/arch.h) and pins
# every constant with gtest cases against docs/research/
# gpu-static-analyzer-research.md (and docs/research/rdna3-constants.md for
# the ISA-resolved RDNA3 VGPR file).
add_library(kernscope_arch INTERFACE)
target_include_directories(kernscope_arch INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_arch INTERFACE cxx_std_20)
add_library(kernscope::arch ALIAS kernscope_arch)

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# The arch DB is pure host-side constants (no GPU required) -> no-gpu.
find_package(GTest REQUIRED)
add_executable(arch-test tests/arch_test.cpp)
target_link_libraries(arch-test PRIVATE kernscope::arch GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(arch-test PROPERTIES LABELS "no-gpu")

# --- todo-22: gfx1101 measured-peaks validator -------------------------------
# python3 tests/arch/check_peaks.py asserts the COMMITTED gfx1101 peak entries
# (libs/arch + libs/roofline + reports/peaks/gfx1101-bench.jsonl provenance)
# pass the todo-22 sanity floors (bw > 300 GB/s, fp32 > 10 TFLOPS) with
# non-empty provenance fields. Pure file validation -> no-gpu.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
add_test(NAME arch-check-peaks
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/arch/check_peaks.py)
set_tests_properties(arch-check-peaks PROPERTIES LABELS "no-gpu")
