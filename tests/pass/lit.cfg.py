# -*- Python -*-

import os
import re
import shutil

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'kernscope-pass' lit suite (todo 36).
#
# Part of the single check-kernscope target: test/CMakeLists.txt passes
# ${CMAKE_BINARY_DIR}/tests/pass (where configure_lit_site_cfg wrote
# lit.site.cfg.py — see tools/pass/CMakeLists.txt) to the same
# add_lit_testsuite as the main test/ suite and tests/verify, so one lit
# invocation runs all three.
#
# Tests (suffixes: .test):
#   * hello_pass.test          — opt -load-pass-plugin smoke: runs the
#     HelloPass by textual name (kernscope-hello) AND via the EP callbacks
#     (opt -passes=default<O2>), FileCheck asserts the "kernscope.hello"
#     marker attribute in the emitted IR.
#   * clang_fpass_plugin.test  — clang -fpass-plugin smoke: compiles a tiny
#     device-only .cu through the clang CUDA path with the plugin active
#     (synthetic root — -nocudainc -nocudalib, no real CUDA headers, the
#     todo-5 fallback idiom), FileCheck asserts the marker in the emitted
#     IR. This exercises the EP callbacks exactly as todos 37-41's passes
#     will run under real clang GPU compiles.
#
# REQUIRES: no-gpu — both tests run GPU-less (opt on .ll, clang
# --cuda-device-only -emit-llvm), no GPU hardware involved.

config.name = "kernscope-pass"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".test"]

# test_source_root: the source directory holding the .test files (this dir).
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: where lit materializes %T/%t; the obj root mirrors the
# source layout (build/tests/pass).
config.test_exec_root = os.path.join(config.kernscope_obj_root, "tests", "pass")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ["HOME", "INCLUDE", "LIB", "TMP", "TEMP", "CUDA_HOME", "CUDA_PATH",
     "ROCM_PATH", "HIP_PATH", "HIP_DEVICE_LIB_PATH", "HSA_PATH",
     "NIX_STORE", "LLVM_DIR", "MLIR_DIR"]
)

llvm_config.use_default_substitutions()

# lit 18 does not provide %llvm_tools_dir out of the box; define it so tests
# can reference the installed LLVM tool dir (${LLVM_TOOLS_BINARY_DIR}) where
# llc/opt/FileCheck/count/not live.
config.substitutions.append(("%llvm_tools_dir", config.llvm_tools_dir))

# Repo root (test_source_root = <root>/tests/pass -> two dirnames up), so
# tests can reference committed fixtures by source path instead of fragile
# %S-relative paths.
config.substitutions.append(
    ("%{kernscope_source_dir}",
     os.path.dirname(os.path.dirname(config.test_source_root)))
)

# The built plugin shared library (MODULE output, no "lib" prefix on Linux:
# build/lib/kernscope-pass.so). Pin it as a literal substitution so RUN
# lines stay short.
config.substitutions.append(
    ("%kernscope-pass-plugin", config.kernscope_pass_plugin)
)

# clang for the -fpass-plugin .cu smoke: resolved from the devenv PATH (the
# nixpkgs llvm-21.1.8 clang with NVPTX targets registered). The nix clang on
# PATH is a WRAPPER that injects host-only flags (-mtls-dialect=,
# -fzero-call-used-regs=) which the nvptx64 target rejects, so the REAL
# clang is located by scanning the wrapper script for a /nix/store/.../bin/
# clang — the same unwrap the verify ingester performs
# (tools/verify/src/ingestion.cpp locate_unwrapped_clang). Fail loudly at
# config time rather than producing a mysteriously failing test.
def _unwrapped_clang():
    wrapper = shutil.which("clang")
    if not wrapper:
        return None
    try:
        with open(wrapper, "r", errors="replace") as f:
            content = f.read()
    except OSError:
        return wrapper
    if "/nix/store/" in content:
        for m in re.finditer(r"/nix/store/[^\s\"']*?/bin/clang", content):
            if os.path.exists(m.group(0)):
                return m.group(0)
    return wrapper

_clang = _unwrapped_clang()
if _clang is None:
    raise RuntimeError(
        "kernscope-pass lit suite: 'clang' not found on PATH (run inside "
        "`devenv shell` — the suite needs the llvm-21.1.8 clang)"
    )
config.substitutions.append(("%clang", _clang))

# Synthetic CUDA root for the clang smoke (todo-5 fallback idiom, mirroring
# the verify ingester's stage_cuda_root): clang 21 only enables the CUDA
# language (the __global__ keyword etc.) when it can DETECT a CUDA
# installation, so a staged root is materialized under the build dir —
# include/ symlinked from $CUDA_HOME plus bin/ + nvvm/ from the nvcc
# package + the curand shim the builtin __clang_cuda_runtime_wrapper.h
# requires. The clang test is gated on the "cuda-root" feature.
def _stage_cuda_root():
    cuda_home = os.environ.get("CUDA_HOME", "")
    if not cuda_home or not os.path.isdir(os.path.join(cuda_home, "include")):
        return None
    root = os.path.join(config.kernscope_obj_root, "tests", "pass", "cuda_root")
    shutil.rmtree(root, ignore_errors=True)
    inc = os.path.join(root, "include")
    os.makedirs(inc)
    for f in os.listdir(os.path.join(cuda_home, "include")):
        os.symlink(os.path.join(cuda_home, "include", f),
                   os.path.join(inc, f))
    nvcc = shutil.which("nvcc")
    if nvcc:
        pkg = os.path.dirname(os.path.dirname(nvcc))
        for sub in ("bin", "nvvm"):
            src = os.path.join(pkg, sub)
            if os.path.isdir(src):
                os.symlink(src, os.path.join(root, sub))
        crt = os.path.join(pkg, "include", "crt")
        if os.path.isdir(crt):
            os.symlink(crt, os.path.join(inc, "crt"))
    with open(os.path.join(inc, "curand_mtgp32_kernel.h"), "w") as f:
        f.write("// kernscope smoke-test shim (NOT an NVIDIA header).\n"
                "#ifndef __KERNSCOPE_CURAND_SHIM__\n"
                "#define __KERNSCOPE_CURAND_SHIM__\n"
                "#endif\n")
    return root

_cuda_root = _stage_cuda_root()
if _cuda_root is not None:
    config.substitutions.append(("%cuda_root", _cuda_root))
    config.available_features.add("cuda-root")

# excludes: auxiliary inputs for the tests.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
]

# Tweak the PATH to include the LLVM tools dir (opt, FileCheck, count, not).
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.kernscope_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["opt", "FileCheck", "count", "not"], tool_dirs
)

# Hardware feature detection mirrors the sibling suites (both tests run
# GPU-less -> always no-gpu capable).
config.available_features.add("no-gpu")

if os.path.exists("/dev/kfd"):
    config.available_features.add("gpu-amd")

if shutil.which("nvcc") is not None:
    config.available_features.add("gpu-nvidia-rental")
