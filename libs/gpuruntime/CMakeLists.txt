# kernscope shared-core library tree.
#
# libs/gpuruntime: thin CUDA/HIP runtime abstraction (todo 21) consumed by the
# capture/profile tracks. ONE translation unit (src/gpu.cpp) selects the
# backend with the portable #ifdef convention (pattern proven in
# /home/mei/projects/polykernel/lib/Runtime/DeviceDetect.cpp):
#   * HIP backend:   plain clang++ + -D__HIP_PLATFORM_AMD__ against the ROCm
#                    host headers (libamdhip64 at link time) - NO device code,
#                    so neither hipcc nor --offload-arch is needed for the lib.
#                    hipcc is only used to --genco the live-test kernel module
#                    (tests/kernels/add_kernel.hip -> add_kernel.hsaco).
#   * CUDA backend:  compiled whenever CUDA headers are available; on this
#                    AMD-only machine the CUDA path is COMPILE-ONLY (no NVIDIA
#                    driver), enforced by an OBJECT library gate below that
#                    never links.
#   * neither        -> compile-only backend (deterministic typed no-device
#                    errors; the unit suite still runs).
#
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
# The mock + compile-only suites need no device -> no-gpu; the live HIP
# launch/roundtrip suite -> gpu-amd.

find_program(KERNSCOPE_HIPCC_EXECUTABLE hipcc)

# ROCm host headers + lib live at <clr>/include and <clr>/lib (rocmPackages.clr
# in the dev shell; HIP_PATH points at the same root).
set(KERNSCOPE_HAVE_HIP OFF)
if(KERNSCOPE_HIPCC_EXECUTABLE)
  get_filename_component(_GPURUNTIME_CLR_BIN "${KERNSCOPE_HIPCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(_GPURUNTIME_CLR_ROOT "${_GPURUNTIME_CLR_BIN}" DIRECTORY)
  set(_GPURUNTIME_HIP_INCLUDE "${_GPURUNTIME_CLR_ROOT}/include")
  set(_GPURUNTIME_HIP_LIBDIR "${_GPURUNTIME_CLR_ROOT}/lib")
  if(NOT EXISTS "${_GPURUNTIME_HIP_INCLUDE}/hip/hip_runtime_api.h"
     AND DEFINED ENV{HIP_PATH})
    # hipcc on PATH but not under <clr>/include (e.g. wrapped): trust HIP_PATH.
    set(_GPURUNTIME_HIP_INCLUDE "$ENV{HIP_PATH}/include")
    set(_GPURUNTIME_HIP_LIBDIR "$ENV{HIP_PATH}/lib")
  endif()
  if(EXISTS "${_GPURUNTIME_HIP_INCLUDE}/hip/hip_runtime_api.h"
     AND EXISTS "${_GPURUNTIME_HIP_LIBDIR}/libamdhip64.so")
    set(KERNSCOPE_HAVE_HIP ON)
  endif()
endif()

add_library(kernscope_gpuruntime STATIC src/gpu.cpp)
target_include_directories(kernscope_gpuruntime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(kernscope_gpuruntime PUBLIC cxx_std_20)
add_library(kernscope::gpuruntime ALIAS kernscope_gpuruntime)

if(KERNSCOPE_HAVE_HIP)
  # HIP host API: plain clang++ compiles it once __HIP_PLATFORM_AMD__ selects
  # the AMD platform (hipcc defines this implicitly; set for the host compiler).
  target_compile_definitions(kernscope_gpuruntime PRIVATE __HIP_PLATFORM_AMD__=1)
  # PUBLIC: consumers/tests need to know which backend the build compiled
  # (asserted by the factory typed-error tests).
  target_compile_definitions(kernscope_gpuruntime PUBLIC KERNSCOPE_HIP_BACKEND_BUILD=1)
  # -isystem (not -I): ROCm headers are third-party; they must not trip the
  # global -Wall -Wextra -Werror gate (occupancy-lib pattern).
  target_include_directories(kernscope_gpuruntime SYSTEM PRIVATE
    "${_GPURUNTIME_HIP_INCLUDE}")
  target_link_directories(kernscope_gpuruntime INTERFACE "${_GPURUNTIME_HIP_LIBDIR}")
  target_link_libraries(kernscope_gpuruntime PUBLIC amdhip64)
  message(STATUS "gpuruntime: HIP backend enabled "
                 "(hipcc=${KERNSCOPE_HIPCC_EXECUTABLE})")
elseif(DEFINED ENV{CUDA_HOME}
       AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime_api.h"
       AND EXISTS "$ENV{CUDA_HOME}/lib64/libcuda.so")
  # CUDA-live machine (NVIDIA driver present): build the lib as the CUDA
  # backend (runtime API + driver API for modules/kernels).
  target_compile_definitions(kernscope_gpuruntime PRIVATE KERNSCOPE_BACKEND_CUDA=1)
  target_compile_definitions(kernscope_gpuruntime PUBLIC KERNSCOPE_CUDA_BACKEND_BUILD=1)
  target_include_directories(kernscope_gpuruntime SYSTEM PRIVATE "$ENV{CUDA_HOME}/include")
  target_link_directories(kernscope_gpuruntime INTERFACE "$ENV{CUDA_HOME}/lib64")
  target_link_libraries(kernscope_gpuruntime PUBLIC cudart)
  target_link_libraries(kernscope_gpuruntime PUBLIC cuda)
  message(STATUS "gpuruntime: CUDA backend enabled (CUDA_HOME=$ENV{CUDA_HOME})")
else()
  target_compile_definitions(kernscope_gpuruntime PUBLIC KERNSCOPE_COMPILE_ONLY_BUILD=1)
  message(STATUS "gpuruntime: compile-only backend (no HIP/CUDA headers at "
                 "configure time; configure inside `devenv shell` to enable)")
endif()

# --- CUDA compile-only gate (AMD-only machine) -------------------------------
# The CUDA backend must COMPILE cleanly (headers from cudaPackages_12_6 via
# $CUDA_HOME) even though it is never linked here (no NVIDIA driver): an
# OBJECT library under the global -Wall -Wextra -Werror gate is the proof.
if(DEFINED ENV{CUDA_HOME} AND EXISTS "$ENV{CUDA_HOME}/include/cuda_runtime_api.h")
  if(NOT KERNSCOPE_HAVE_HIP)
    message(STATUS "gpuruntime: CUDA is already the lib backend - compile gate "
                   "redundant")
  else()
    add_library(kernscope_gpuruntime_cuda_compilecheck OBJECT src/gpu.cpp)
    target_compile_definitions(kernscope_gpuruntime_cuda_compilecheck
      PRIVATE KERNSCOPE_BACKEND_CUDA=1)
    target_include_directories(kernscope_gpuruntime_cuda_compilecheck
      SYSTEM PRIVATE "$ENV{CUDA_HOME}/include")
    target_include_directories(kernscope_gpuruntime_cuda_compilecheck
      PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_compile_features(kernscope_gpuruntime_cuda_compilecheck
      PRIVATE cxx_std_20)
    message(STATUS "gpuruntime: CUDA compile-only gate ENABLED (gpu.cpp as "
                   "CUDA backend, headers from $ENV{CUDA_HOME}/include)")
  endif()
else()
  message(STATUS "gpuruntime: CUDA compile-only gate DISABLED (no "
                 "CUDA_HOME/include at configure time)")
endif()

# --- live HIP test kernel (hipcc --genco, gfx1101) ----------------------------
if(KERNSCOPE_HAVE_HIP)
  set(_GPURUNTIME_HSACO "${CMAKE_CURRENT_BINARY_DIR}/kernels/add_kernel.hsaco")
  add_custom_command(OUTPUT "${_GPURUNTIME_HSACO}"
    COMMAND "${KERNSCOPE_HIPCC_EXECUTABLE}" --genco --offload-arch=gfx1101
            "-I${_GPURUNTIME_HIP_INCLUDE}"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/add_kernel.hip"
            -o "${_GPURUNTIME_HSACO}"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/kernels/add_kernel.hip"
    COMMENT "hipcc --genco tests/kernels/add_kernel.hip (gfx1101) -> add_kernel.hsaco")
  add_custom_target(kernscope_gpuruntime_hsaco DEPENDS "${_GPURUNTIME_HSACO}")
endif()

# --- unit tests (gtest) -------------------------------------------------------
find_package(GTest REQUIRED)

add_executable(gpuruntime-test tests/gpuruntime_test.cpp)
target_link_libraries(gpuruntime-test PRIVATE kernscope::gpuruntime GTest::gtest_main)
include(GoogleTest)
# Mock + compile-only backends: no device needed -> no-gpu.
gtest_discover_tests(gpuruntime-test TEST_PREFIX "gpuruntime:"
  PROPERTIES LABELS "no-gpu")

# Live HIP suite: gfx1101 launch + roundtrip + event timing -> gpu-amd.
add_executable(gpuruntime-hip-test tests/gpuruntime_hip_test.cpp)
target_link_libraries(gpuruntime-hip-test PRIVATE kernscope::gpuruntime GTest::gtest_main)
if(KERNSCOPE_HAVE_HIP)
  target_compile_definitions(gpuruntime-hip-test PRIVATE
    KERNSCOPE_TEST_HSACO="${_GPURUNTIME_HSACO}")
  add_dependencies(gpuruntime-hip-test kernscope_gpuruntime_hsaco)
else()
  # No hipcc at configure time: the live suite SKIPs (KERNSCOPE_NO_LIVE_HIP).
  target_compile_definitions(gpuruntime-hip-test PRIVATE
    KERNSCOPE_NO_LIVE_HIP=1 KERNSCOPE_TEST_HSACO="")
endif()
include(GoogleTest)
gtest_discover_tests(gpuruntime-hip-test TEST_PREFIX "gpuruntime-hip:"
  PROPERTIES LABELS "gpu-amd")
