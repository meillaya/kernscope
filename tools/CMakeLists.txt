# tools/ — kernscope tool entry points.
#
# One-time addition (todo 30): the verify track. Each tool keeps its own
# CMakeLists.txt; the root superbuild registers this directory so future
# tools (capture/pass/profile) can hook in here instead of touching the
# root CMakeLists.txt again.
add_subdirectory(verify)
# One-time addition (todo 36): the pass track (kernscope-pass, the
# out-of-tree LLVM 21 pass-plugin). Owns its lit wiring for tests/pass/
# inside tools/pass/CMakeLists.txt.
add_subdirectory(pass)
# One-time addition (todo 47): the capture track (kernscope-capture, the
# C4 CLI). Owns emit-repro + check-reproducers + the reproducer ctests
# inside tools/capture/CMakeLists.txt. Must stay after libs/capture-core
# (the CLI links kernscope::capture_core for BundleReader).
add_subdirectory(capture)
# One-time addition (todo 49): the profile track (kernscope-profile, the
# C6 profiler). Owns the CUPTI tracer (compile+unit; reuses
# libs/cupti-sub) + the rocprofiler-sdk buffer-tracing tracer (live
# gfx1101) inside tools/profile/CMakeLists.txt. Must stay after
# libs/cupti-sub (the CUPTI tracer reuses the todo-44 subscription library).
add_subdirectory(profile)
