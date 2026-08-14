# kernscope shared-core library tree.
#
# libs/roofline: the roofline engine (wave-2 todo 12). Per-family
# arithmetic-intensity models for the six C2 autotuner kernel families (GEMM,
# fused bias+GELU, softmax, layernorm, reduction, FlashAttention) from
# docs/research/autotuner-research-brief.md §7 (Williams/Waterman/Patterson
# CACM 2009, DOI 10.1145/1498765.1498785) + a machine-peaks table (A100, RTX
# 4090, MI250X, MI300X from brief §7 L127-130; gfx1101 "pending-measurement"
# placeholders that todo 22 fills with measured values + provenance — querying
# gfx1101 returns projected==true, never a fabricated number) + roofline
# position + bound classification (compute/memory/latency).
#
# Depends on libs/arch for arch-name validation (has_arch), so this
# add_subdirectory must come after libs/arch in the superbuild.
add_library(kernscope_roofline src/roofline.cpp)
target_include_directories(kernscope_roofline PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_roofline PUBLIC cxx_std_20)
add_library(kernscope::roofline ALIAS kernscope_roofline)
target_link_libraries(kernscope_roofline PUBLIC kernscope::arch)

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# The roofline engine is pure host-side math over brief-pinned constants (no
# GPU required) -> no-gpu.
find_package(GTest REQUIRED)
add_executable(roofline-test tests/roofline_test.cpp)
target_link_libraries(roofline-test PRIVATE kernscope::roofline GTest::gtest_main)
include(GoogleTest)
# TEST_PREFIX keeps every discovered case matching `ctest -R roofline`
# (ctest -R is a case-sensitive regex; gtest suite names are CamelCase).
gtest_discover_tests(roofline-test
  TEST_PREFIX "roofline:"
  PROPERTIES LABELS "no-gpu"
)
