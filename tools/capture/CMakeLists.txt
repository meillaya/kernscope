# tools/capture — kernscope-capture CLI (C4 capture/replay track, todo 47).
#
# Todo 47 ships the `emit-repro` subcommand: the STANDALONE C++ reproducer
# generator. It consumes the bundle formats through libs/capture-core's
# BundleReader (v1 *.kcap/ dirs + v0 text bundles with sidecar blobs) and
# emits a plain-CMake project (embedded kernel binary + input buffers,
# captured grid/block/args, output dump) with ZERO repo dependencies at
# runtime. Todos 48/49 append record|replay|bench here.
#
# Wired from tools/CMakeLists.txt (the designed hook point - capture/pass/
# profile were listed there from the start); the tool itself NEVER touches
# libs/capture-core or libs/capture-hip (consume via headers/CLI only).

add_executable(kernscope-capture
  src/main.cpp
  src/repro_gen.cpp)
target_include_directories(kernscope-capture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
# BundleReader + sha256 provenance (libs/capture-core, todo 46) and the
# dependency-free json model (libs/db, todo 8).
target_link_libraries(kernscope-capture PRIVATE
  kernscope::capture_core kernscope::db)
# Land at build/kernscope-capture (the acceptance command shape).
set_target_properties(kernscope-capture PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# --- check-reproducers: emitted-reproducer compile gates ---------------------
# The reproducer for the todo-45 layernorm bundle must COMPILE standalone in
# CI (HIP: full hipcc build; CUDA: nvcc compile-only - no NVIDIA hardware
# locally, runtime validation is owner-gated). The emitted projects are
# generated from COMMITTED fixtures so the gate is GPU-less.
#
#   cmake --build build --target check-reproducers
#
# ctest hardware-gating convention (todo 3): every registered test carries a
# LABELS property of exactly one of no-gpu | gpu-amd | gpu-nvidia-rental.
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(KERNSCOPE_CAPTURE_FIXTURE_DIR ${CMAKE_SOURCE_DIR}/tests/fixtures/capture)
set(KERNSCOPE_REPROS_DIR ${CMAKE_BINARY_DIR}/repros)

# --- HIP reproducer: committed todo-45 layernorm live capture ----------------
set(KERNSCOPE_REPRO_LAYERNORM_DIR ${KERNSCOPE_REPROS_DIR}/layernorm-hip)
add_custom_command(OUTPUT ${KERNSCOPE_REPRO_LAYERNORM_DIR}/main.hip
  COMMAND ${CMAKE_BINARY_DIR}/kernscope-capture emit-repro
          ${KERNSCOPE_CAPTURE_FIXTURE_DIR}/layernorm-live/bundle.v0
          --blob-dir ${KERNSCOPE_CAPTURE_FIXTURE_DIR}/layernorm-live/blobs
          --vendor amd --arch gfx1101 --force
          --out ${KERNSCOPE_REPRO_LAYERNORM_DIR}
  DEPENDS kernscope-capture
          ${KERNSCOPE_CAPTURE_FIXTURE_DIR}/layernorm-live/bundle.v0
  COMMENT "emit-repro: layernorm_fused HIP reproducer"
  VERBATIM)
add_custom_command(OUTPUT ${KERNSCOPE_REPRO_LAYERNORM_DIR}/build/kernscope_repro
  COMMAND ${CMAKE_COMMAND} -S ${KERNSCOPE_REPRO_LAYERNORM_DIR}
          -B ${KERNSCOPE_REPRO_LAYERNORM_DIR}/build
  COMMAND ${CMAKE_COMMAND} --build ${KERNSCOPE_REPRO_LAYERNORM_DIR}/build
  DEPENDS ${KERNSCOPE_REPRO_LAYERNORM_DIR}/main.hip
  WORKING_DIRECTORY ${KERNSCOPE_REPRO_LAYERNORM_DIR}
  COMMENT "build: layernorm_fused HIP reproducer (hipcc gfx1101)"
  VERBATIM)
add_custom_target(repro-layernorm-hip
  DEPENDS ${KERNSCOPE_REPRO_LAYERNORM_DIR}/build/kernscope_repro)

# --- CUDA reproducer: committed v0-sample fixture (compile-only) -------------
set(KERNSCOPE_REPRO_VECADD_DIR ${KERNSCOPE_REPROS_DIR}/vecadd-cuda)
add_custom_command(OUTPUT ${KERNSCOPE_REPRO_VECADD_DIR}/main.cu
  COMMAND ${CMAKE_BINARY_DIR}/kernscope-capture emit-repro
          ${KERNSCOPE_CAPTURE_FIXTURE_DIR}/v0-sample.bundle
          --vendor nvidia --arch sm_80 --force
          --out ${KERNSCOPE_REPRO_VECADD_DIR}
  DEPENDS kernscope-capture
          ${KERNSCOPE_CAPTURE_FIXTURE_DIR}/v0-sample.bundle
  COMMENT "emit-repro: vec_add CUDA reproducer (compile-only)"
  VERBATIM)
add_custom_command(OUTPUT ${KERNSCOPE_REPRO_VECADD_DIR}/build/kernscope_repro.cu.o
  COMMAND ${CMAKE_COMMAND} -S ${KERNSCOPE_REPRO_VECADD_DIR}
          -B ${KERNSCOPE_REPRO_VECADD_DIR}/build
  COMMAND ${CMAKE_COMMAND} --build ${KERNSCOPE_REPRO_VECADD_DIR}/build
  DEPENDS ${KERNSCOPE_REPRO_VECADD_DIR}/main.cu
  WORKING_DIRECTORY ${KERNSCOPE_REPRO_VECADD_DIR}
  COMMENT "build: vec_add CUDA reproducer (nvcc sm_80 compile-only)"
  VERBATIM)
add_custom_target(repro-vecadd-cuda
  DEPENDS ${KERNSCOPE_REPRO_VECADD_DIR}/build/kernscope_repro.cu.o)

# --- the aggregating compile gate --------------------------------------------
add_custom_target(check-reproducers
  DEPENDS repro-layernorm-hip repro-vecadd-cuda
  COMMENT "check-reproducers: emitted reproducers compile standalone "
          "(HIP gfx1101 full build + CUDA sm_80 compile-only)")

# --- negative QA: size-unknown buffer -> synthetic input (no-gpu) ------------
# The committed negative fixture carries a launch arg pointing at a device
# buffer that is neither in the allocation map nor covered by a copy record
# with a known size. The generator must emit a CLEARLY-MARKED synthetic input
# + warning header and STILL compile (never crash). Emit + compile only - no
# GPU needed (the reproducer builds; running it would just replay synthetic
# inputs).
add_test(NAME capture_repro_negative
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_repro_negative_check.py
          ${CMAKE_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR}
          ${KERNSCOPE_CAPTURE_FIXTURE_DIR})
set_tests_properties(capture_repro_negative PROPERTIES LABELS "no-gpu"
  TIMEOUT 600)

# --- live acceptance: layernorm bundle -> emit -> build -> run -> hash match --
# gpu-amd (todo 3 convention): runs the todo-45 live capture of the
# layernorm_fused app on gfx1101, emits the reproducer from the FRESH bundle,
# builds it standalone, runs it and asserts the reproducer's output hash ==
# the captured app output hash.
add_test(NAME capture_repro_gen
  COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/capture_repro_check.py
          ${CMAKE_BINARY_DIR}
          ${CMAKE_CURRENT_SOURCE_DIR})
set_tests_properties(capture_repro_gen PROPERTIES LABELS "gpu-amd"
  TIMEOUT 900)
