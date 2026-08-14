# kernscope shared-core library tree (stub).
#
# libs/arch: the GPU architecture-model database (constants + occupancy
# formulas for NVIDIA sm_* and AMD gfx90a/gfx942/gfx1101). Wave-1 todo 6
# fills this in; for now it provides the version header every tool stamps.
add_library(kernscope_arch INTERFACE)
target_include_directories(kernscope_arch INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_arch INTERFACE cxx_std_20)
add_library(kernscope::arch ALIAS kernscope_arch)
