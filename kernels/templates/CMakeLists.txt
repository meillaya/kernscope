# kernscope kernels/templates: portable CUDA/HIP kernel template convention
# (todo 24).
#
# Establishes the single-source kernel template infrastructure that todos
# 25-27 instantiate (tune track) and todo 44 consumes (capture circularity
# breaker). Two CMake functions encapsulate the compile gates:
#
#   kernscope_add_cuda_kernel(name SOURCES <file.hip>...)
#     nvcc COMPILE-ONLY gates: -arch=sm_80 -lineinfo and -arch=sm_90
#     -lineinfo (no NVIDIA GPU needed; sm_80/sm_90 are the 12.6-supported
#     targets on this machine, NOT sm_100). The .hip sources are forced
#     through nvcc with -x cu. Registers one ctest test per arch
#     (LABELS no-gpu) whose command rebuilds the gate target.
#
#   kernscope_add_hip_kernel(name SOURCES <file.hip>... [RUN_SYMBOL <sym>]
#                            [GOLDEN_FAMILY <family>] [GOLDEN_SHAPE <m,n,...>]
#                            [GRID <W[xH]>] [BLOCK <X[xY]>])
#     hipcc gate: --genco --offload-arch=gfx1101 -gline-tables-only
#     -D__HIP_PLATFORM_AMD__=1 (hipcc 7.2 no longer defines the platform
#     macro implicitly - verified via -dM; the portable guard contract
#     requires it). Registers a ctest compile gate (LABELS no-gpu). With
#     RUN_SYMBOL + GOLDEN_FAMILY + GOLDEN_SHAPE + GRID + BLOCK the function
#     additionally registers the LIVE gate (LABELS gpu-amd): tests/run_hip_gate.py
#     runs the kernel on this machine (gfx1101) through the
#     kernscope_gpuruntime-backed hip_template_runner and requires a
#     kernscope_golden verdict pass:true.
#
# ctest hardware-gating convention (todo 3): every test carries LABELS of
# exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
#
# NOTE: kernels/templates must be added AFTER libs/gpuruntime (the runner
# links kernscope::gpuruntime).

# --- toolchain discovery (libs/gpuruntime pattern) --------------------------
find_program(KERNSCOPE_NVCC_EXECUTABLE nvcc)
find_program(KERNSCOPE_HIPCC_EXECUTABLE hipcc)

set(KERNSCOPE_TEMPLATES_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# ROCm host headers live at <clr>/include (rocmPackages.clr in the dev shell;
# HIP_PATH points at the same root).
set(KERNSCOPE_TPL_HAVE_HIP OFF)
if(KERNSCOPE_HIPCC_EXECUTABLE)
  get_filename_component(_TPL_CLR_BIN "${KERNSCOPE_HIPCC_EXECUTABLE}" DIRECTORY)
  get_filename_component(_TPL_CLR_ROOT "${_TPL_CLR_BIN}" DIRECTORY)
  set(_TPL_HIP_INCLUDE "${_TPL_CLR_ROOT}/include")
  if(NOT EXISTS "${_TPL_HIP_INCLUDE}/hip/hip_runtime_api.h"
     AND DEFINED ENV{HIP_PATH})
    set(_TPL_HIP_INCLUDE "$ENV{HIP_PATH}/include")
  endif()
  if(EXISTS "${_TPL_HIP_INCLUDE}/hip/hip_runtime_api.h")
    set(KERNSCOPE_TPL_HAVE_HIP ON)
  endif()
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  set(_TPL_PYTHON "${Python3_EXECUTABLE}")
else()
  set(_TPL_PYTHON "python3")
endif()

# --- kernscope_add_cuda_kernel ----------------------------------------------
# nvcc compile-only gates: -arch=sm_80 -lineinfo and -arch=sm_90 -lineinfo,
# one ctest test per arch (LABELS no-gpu -- compile-only, GPU-less).
function(kernscope_add_cuda_kernel name)
  cmake_parse_arguments(ARG "" "" "SOURCES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "kernscope_add_cuda_kernel(${name}): SOURCES required")
  endif()
  if(NOT KERNSCOPE_NVCC_EXECUTABLE)
    message(STATUS "kernscope_add_cuda_kernel(${name}): nvcc not found - "
                   "CUDA compile gates skipped")
    return()
  endif()
  foreach(arch sm_80 sm_90)
    set(_objs "")
    foreach(src IN LISTS ARG_SOURCES)
      get_filename_component(_base "${src}" NAME_WE)
      set(_obj "${CMAKE_CURRENT_BINARY_DIR}/cuda/${_base}_${arch}.o")
      # -x cu: the template sources carry the .hip extension (hipcc's), so
      # nvcc must be told explicitly to compile them as CUDA.
      add_custom_command(OUTPUT "${_obj}"
        COMMAND "${KERNSCOPE_NVCC_EXECUTABLE}" -arch=${arch} -lineinfo
                -x cu -std=c++20 "-I${KERNSCOPE_TEMPLATES_INCLUDE}"
                -c "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${_obj}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
        COMMENT "nvcc ${arch} compile-only gate: ${name}/${src}")
      list(APPEND _objs "${_obj}")
    endforeach()
    add_custom_target(kernscope_templates_${name}_cuda_${arch} DEPENDS ${_objs})
    add_test(NAME "kernscope.templates.${name}.cuda.${arch}"
      COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
              --target kernscope_templates_${name}_cuda_${arch})
    set_tests_properties("kernscope.templates.${name}.cuda.${arch}" PROPERTIES
      LABELS "no-gpu")
  endforeach()
endfunction()

# --- kernscope_add_hip_kernel ------------------------------------------------
# hipcc gate: --genco --offload-arch=gfx1101 -gline-tables-only (compile).
# With the RUN_* arguments: ALSO register the live gate (gpu-amd) that runs
# the kernel on this machine and requires a golden verdict pass:true.
function(kernscope_add_hip_kernel name)
  cmake_parse_arguments(ARG "" "RUN_SYMBOL;GOLDEN_FAMILY;GOLDEN_SHAPE;GRID;BLOCK"
    "SOURCES" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "kernscope_add_hip_kernel(${name}): SOURCES required")
  endif()
  if(NOT KERNSCOPE_TPL_HAVE_HIP)
    message(STATUS "kernscope_add_hip_kernel(${name}): hipcc/ROCm headers not "
                   "found - HIP gates skipped")
    return()
  endif()
  set(_hsacos "")
  set(_live_hsaco "")
  foreach(src IN LISTS ARG_SOURCES)
    get_filename_component(_base "${src}" NAME_WE)
    set(_hsaco "${CMAKE_CURRENT_BINARY_DIR}/hip/${_base}.hsaco")
    if(NOT _live_hsaco)
      set(_live_hsaco "${_hsaco}")
    endif()
    # -D__HIP_PLATFORM_AMD__=1: hipcc 7.2 does NOT define the platform macro
    # implicitly anymore (verified via -dM on the host and device passes);
    # the portable guard convention requires it (see kernel_traits.h).
    add_custom_command(OUTPUT "${_hsaco}"
      COMMAND "${KERNSCOPE_HIPCC_EXECUTABLE}" --genco --offload-arch=gfx1101
              -gline-tables-only -D__HIP_PLATFORM_AMD__=1
              "-I${_TPL_HIP_INCLUDE}" "-I${KERNSCOPE_TEMPLATES_INCLUDE}"
              "${CMAKE_CURRENT_SOURCE_DIR}/${src}" -o "${_hsaco}"
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${src}"
      COMMENT "hipcc --genco gfx1101 compile gate: ${name}/${src}")
    list(APPEND _hsacos "${_hsaco}")
  endforeach()
  add_custom_target(kernscope_templates_${name}_hip_genco DEPENDS ${_hsacos})
  add_test(NAME "kernscope.templates.${name}.hip.genco"
    COMMAND "${CMAKE_COMMAND}" --build "${CMAKE_BINARY_DIR}"
            --target kernscope_templates_${name}_hip_genco)
  set_tests_properties("kernscope.templates.${name}.hip.genco" PROPERTIES
    LABELS "no-gpu")

  if(ARG_RUN_SYMBOL)
    if(NOT ARG_GOLDEN_FAMILY OR NOT ARG_GOLDEN_SHAPE OR NOT ARG_GRID OR NOT ARG_BLOCK)
      message(FATAL_ERROR "kernscope_add_hip_kernel(${name}): RUN_SYMBOL "
              "requires GOLDEN_FAMILY, GOLDEN_SHAPE, GRID and BLOCK")
    endif()
    if(NOT TARGET kernscope_templates_hip_runner)
      add_executable(kernscope_templates_hip_runner
        tools/hip_template_runner.cpp)
      target_link_libraries(kernscope_templates_hip_runner PRIVATE
        kernscope::gpuruntime)
      target_compile_features(kernscope_templates_hip_runner PRIVATE cxx_std_20)
    endif()
    # `cmake --build build` (ci stage a) builds the runner and, through this
    # dependency, the hsaco; ctest (stage b) then runs the live gate against
    # the built artifacts.
    add_dependencies(kernscope_templates_hip_runner
      kernscope_templates_${name}_hip_genco)
    add_test(NAME "kernscope.templates.${name}.hip.live"
      COMMAND "${_TPL_PYTHON}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_hip_gate.py"
              --root "${CMAKE_SOURCE_DIR}"
              --workdir "${CMAKE_CURRENT_BINARY_DIR}/hip/live/${name}"
              --hsaco "${_live_hsaco}"
              --runner "$<TARGET_FILE:kernscope_templates_hip_runner>"
              --symbol "${ARG_RUN_SYMBOL}"
              --family "${ARG_GOLDEN_FAMILY}"
              --shape "${ARG_GOLDEN_SHAPE}"
              --grid "${ARG_GRID}"
              --block "${ARG_BLOCK}")
    set_tests_properties("kernscope.templates.${name}.hip.live" PROPERTIES
      LABELS "gpu-amd")
  endif()
endfunction()

# --- reference templates (todo 24) -------------------------------------------
# elementwise_add: full treatment - nvcc sm_80/sm_90 compile-only gates, hipcc
# gfx1101 genco gate, and the LIVE gate: run on this machine + golden verdict
# pass:true. Golden mapping: `reduction` shape 2,N (reference y[j] = sum_i
# x[i,j] over a 2-row tensor == x[0]+x[1], the elementwise add of two
# vectors; see README.md "Golden mapping").
kernscope_add_cuda_kernel(elementwise_add
  SOURCES reference/elementwise_add.hip)
kernscope_add_hip_kernel(elementwise_add
  SOURCES reference/elementwise_add.hip
  RUN_SYMBOL kernscope_templates_elementwise_add
  GOLDEN_FAMILY reduction
  GOLDEN_SHAPE "2,1024"
  GRID "4"
  BLOCK "256")

# matmul_skeleton: same gates minus the live run (the tuned matmul variants
# of todos 25/27 carry the live golden validation; this todo pins the
# compile-only gates for both archs and the hipcc genco gate).
kernscope_add_cuda_kernel(matmul_skeleton
  SOURCES reference/matmul_skeleton.hip)
kernscope_add_hip_kernel(matmul_skeleton
  SOURCES reference/matmul_skeleton.hip)

