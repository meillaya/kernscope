# kernscope shared-core library tree.
#
# libs/capture-hip (todo 45): the LOCAL LIVE HIP capture loop on gfx1101.
# A rocprofiler-sdk TOOL (shared library exporting rocprofiler_configure) that
# captures a HIP application into the libs/capture-core session model:
#
#   1. INTERCEPT-TABLE REGISTRATION (rocprofiler_at_intercept_table_registration,
#      ROCPROFILER_HIP_RUNTIME_TABLE): the HIP runtime hands over its live
#      dispatch table (HipDispatchTable) and we wrap the allocation hooks
#      (hipMalloc -> ptr->size map, hipFree), copies/memsets and the kernel
#      launch entry point. Kerncap/intellikit precedent (MIT, THIRD-PARTY.md).
#   2. KERNEL-DISPATCH callback tracing: grid/workgroup from the HSA dispatch
#      layer, kernel name resolved through the code-object kernel-symbol
#      registration callback (kernel_id -> name).
#   3. CODE-OBJECT callback tracing: the loaded HSACO blob is registered into
#      the session (sha256 provenance via the v0 module record) and dumped to
#      disk next to the bundle for todo 46/47 (kernel binary blob).
#   4. H2D memcpy content capture: the host-side source of every
#      hipMemcpyHostToDevice call is dumped to a blob file so the acceptance
#      can HASH-MATCH the captured input buffers against the app's known
#      inputs (v0-scope of the Kerncap address-space closure; the full
#      VA-faithful snapshot policy is todo 46's deliverable).
#
# The bundle is emitted through libs/capture-core's v0 BundleWriter (never
# touched here). roctracer is NOT used (deprecated; rocprofiler-sdk is the
# blessed API per docs/research/capture-replay-research-brief.md §7).
#
# Build contract: links kernscope::capture_core statically (no runtime
# dependency on capture-core) and librocprofiler-sdk.so (NEEDED: the SDK must
# be present in the captured process for the tool to be configured; loading
# via LD_PRELOAD of this .so or ROCPROFILER_LIBRARY both work).

# --- the tool library --------------------------------------------------------
find_program(HIPCONFIG hipconfig REQUIRED)
execute_process(COMMAND ${HIPCONFIG} --includedir
  OUTPUT_VARIABLE HIP_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "capture-hip: HIP headers at ${HIP_INCLUDE_DIR}")

# rocprofiler-sdk dev tree ($ROCPROF_SDK_DEV in the devenv). The shipped
# cmake package self-checks a lib subdir that the nix dev split does not
# contain, so the tool links the SDK directly by path instead of
# find_package (headers SYSTEM so the project-wide -Werror never trips on
# vendor declarations).
if(NOT DEFINED ENV{ROCPROF_SDK_DEV} OR
   NOT EXISTS "$ENV{ROCPROF_SDK_DEV}/include/rocprofiler-sdk/rocprofiler.h" OR
   NOT DEFINED ENV{ROCPROF_SDK_OUT} OR
   NOT EXISTS "$ENV{ROCPROF_SDK_OUT}/lib/librocprofiler-sdk.so")
  message(STATUS "capture-hip: rocprofiler-sdk NOT found (set ROCPROF_SDK_DEV/"
                 "ROCPROF_SDK_OUT; build inside `devenv shell`)")
  return()
endif()

add_library(kernscope_capture_hip SHARED src/capture_hip.cpp src/amdgpu_metadata.cpp)
target_include_directories(kernscope_capture_hip PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include)
# HIP + rocprofiler-sdk headers are SYSTEM so the project-wide -Werror never
# trips on vendor declarations.
target_include_directories(kernscope_capture_hip SYSTEM PRIVATE
  "${HIP_INCLUDE_DIR}" "$ENV{ROCPROF_SDK_DEV}/include")
target_compile_definitions(kernscope_capture_hip PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(kernscope_capture_hip PRIVATE
  kernscope::capture_core "$ENV{ROCPROF_SDK_OUT}/lib/librocprofiler-sdk.so")
set_target_properties(kernscope_capture_hip PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_library(kernscope::capture_hip ALIAS kernscope_capture_hip)

# --- live capture test (gfx1101; todo-5 layernorm_fused fixture) -------------
# ctest hardware-gating convention (todo 3): gpu-amd label.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(NAME capture_hip_live
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_hip_check.py
          ${CMAKE_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(capture_hip_live PROPERTIES LABELS "gpu-amd"
  TIMEOUT 600)

# --- negative QA app + test (unsupported-API allocation -> size-unknown) -----
# The app allocates via hipMallocPitch (NOT among the wrapped hooks) and then
# copies into the buffer + launches with it; the bundle must mark the buffer
# size-unknown and the capture must log a warning WITHOUT crashing.
find_program(HIPCC hipcc REQUIRED)
add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/capture_hip_negative_app
  COMMAND ${HIPCC} --offload-arch=gfx1101 -std=c++17 -O3
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_hip_negative.cpp
          -o ${CMAKE_CURRENT_BINARY_DIR}/capture_hip_negative_app
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_hip_negative.cpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT "hipcc capture_hip_negative app (gfx1101)"
  VERBATIM)
add_custom_target(capture_hip_negative_build ALL
  DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/capture_hip_negative_app)
add_test(NAME capture_hip_negative
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_hip_negative_check.py
          ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_BINARY_DIR})
set_tests_properties(capture_hip_negative PROPERTIES LABELS "gpu-amd"
  TIMEOUT 600)
