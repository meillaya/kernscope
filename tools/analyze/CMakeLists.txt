# kernscope-analyze: the static GPU-kernel analyzer (stub).
#
# Wave-3 todos 13-20 build the ten analyses (register pressure, occupancy,
# coalescing, bank conflicts, divergence, barriers, instruction mix, tensor
# core / MFMA / WMMA usage, spills, latency bottlenecks) + the suggestion
# engine. For now: a trivial executable that prints the kernscope version,
# proving the CMake target graph and the ctest LABELS convention end to end.
add_executable(kernscope-analyze main.cpp)
target_link_libraries(kernscope-analyze PRIVATE kernscope::arch)

# Land the binary in <build>/bin/ (tt-flow convention) so lit's
# KERNSCOPE_TOOLS_DIR (${CMAKE_BINARY_DIR}/bin) finds it directly.
set_target_properties(kernscope-analyze PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ctest hardware-gating convention: every test carries a LABELS property of
# exactly one of no-gpu | gpu-amd | gpu-nvidia-rental. This stub runs on any
# machine, so it is labeled no-gpu.
add_test(NAME analyze-version COMMAND kernscope-analyze --version)
set_tests_properties(analyze-version PROPERTIES LABELS "no-gpu")
