# kernscope shared-core library tree.
#
# libs/occupancy: the theoretical-occupancy engine (wave-2 todo 11). Computes
# resident blocks/warps + the binding limiter from (arch, regs_per_thread,
# smem_per_block, block_size) with the min-wins rule (docs/research/
# gpu-static-analyzer-research.md §1.1 L31-45):
#   NVIDIA path: 256-register warp granule on 65536 regs/SM (brief §1.2
#     L91-102); blocks by regs/smem/threads/blocks-cap, min wins.
#   AMD path: VGPR waves/SIMD via libs/arch (CDNA granule 8, RDNA3 granule
#     24 wave32 / 12 wave64 per docs/research/rdna3-constants.md) clamped to
#     max waves/SIMD, SGPR pool ~800/SIMD granule 16 (CDNA only, 5-arg
#     overload), LDS 64 KB/CU with 512-byte granule (brief §1.3 L117-127),
#     then CU-level min.
# Depends on libs/arch for the ISA-resolved constants (todo 6); comes after
# libs/arch in the superbuild.
#
# NVIDIA oracle wiring: the test TU includes cuda_occupancy.h from
# $CUDA_HOME/include (cudaPackages_12_6, consumed from the nix store at BUILD
# time — NEVER committed; the ci.sh vendored-header guard must stay clean).
# The include path is read from the CUDA_HOME env var at CONFIGURE time
# ($ENV{CUDA_HOME}) and passed via -isystem so third-party NVIDIA headers
# cannot trip the global -Werror/-Wextra gate. If the header is absent at
# configure time (e.g. configure ran outside the devenv shell) the oracle
# tests compile with KERNSCOPE_NO_CUDA_OCCUPANCY and GTEST_SKIP() at runtime
# instead of failing the build.
add_library(kernscope_occupancy src/occupancy.cpp)
target_include_directories(kernscope_occupancy PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_occupancy PUBLIC cxx_std_20)
add_library(kernscope::occupancy ALIAS kernscope_occupancy)
target_link_libraries(kernscope_occupancy PUBLIC kernscope::arch)

# --- unit tests (gtest) -------------------------------------------------------
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# The occupancy engine is pure host-side math (no GPU required) -> no-gpu.
find_package(GTest REQUIRED)
add_executable(occupancy-test tests/occupancy_test.cpp)
target_link_libraries(occupancy-test PRIVATE kernscope::occupancy GTest::gtest_main)

if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_occupancy.h")
  # -isystem (not -I): NVIDIA's cuda_occupancy.h predates clang-21 -Wextra;
  # third-party headers must not fail the -Werror gate (report-lib pattern).
  target_compile_options(occupancy-test PRIVATE "-isystem$ENV{CUDA_HOME}/include")
  message(STATUS "occupancy: NVIDIA oracle ENABLED "
                 "(cuda_occupancy.h from $ENV{CUDA_HOME}/include)")
else()
  target_compile_definitions(occupancy-test PRIVATE KERNSCOPE_NO_CUDA_OCCUPANCY=1)
  message(STATUS "occupancy: NVIDIA oracle DISABLED (no CUDA_HOME/"
                 "include/cuda_occupancy.h at configure time; oracle gtests "
                 "will SKIP). Configure inside `devenv shell` to enable.")
endif()

include(GoogleTest)
# TEST_PREFIX keeps every discovered case matching `ctest -R occupancy`
# (ctest -R is a case-sensitive regex; gtest suite names are CamelCase).
gtest_discover_tests(occupancy-test
  TEST_PREFIX "occupancy:"
  PROPERTIES LABELS "no-gpu"
)
