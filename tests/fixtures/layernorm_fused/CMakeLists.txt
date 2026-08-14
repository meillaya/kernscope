# Standalone CMake target for the layernorm_fused golden app.
#
# Not wired into the root superbuild (that file is owned by foundation todo 3);
# this target is configured standalone:
#
#   cmake -B build -S tests/fixtures/layernorm_fused
#   cmake --build build            -> build/layernorm_app
#
# The build is a plain hipcc driver call (no HIP cmake config dependency),
# so it works inside the devenv without find_package(HIP) fragility.

cmake_minimum_required(VERSION 3.20)
project(layernorm_fused NONE)

find_program(HIPCC hipcc REQUIRED)

set(OFFLOAD_ARCH "gfx1101" CACHE STRING "AMDGPU offload arch for the fixture")

add_custom_target(layernorm_fused ALL
    COMMAND ${HIPCC} --offload-arch=${OFFLOAD_ARCH} -std=c++17 -O3
            ${CMAKE_CURRENT_SOURCE_DIR}/layernorm.hip
            -o ${CMAKE_CURRENT_BINARY_DIR}/layernorm_app
    BYPRODUCTS layernorm_app
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "hipcc layernorm_fused (gfx1101)"
    VERBATIM)
