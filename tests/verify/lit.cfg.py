# -*- Python -*-

import os
import shutil
import subprocess

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'kernscope-verify' lit suite (todo 35).
#
# Part of the single check-kernscope target: test/CMakeLists.txt passes
# ${CMAKE_BINARY_DIR}/tests/verify (where configure_lit_site_cfg wrote
# lit.site.cfg.py — see tools/verify/CMakeLists.txt) to the same
# add_lit_testsuite as the main test/ suite, so one lit invocation runs both.
#
# Tests (suffixes: .test):
#   * verify_manifest.test  — runs kernscope-verify --json over EVERY corpus
#     kernel and diffs the emitted verdicts against tests/verify/manifest.json
#     (read LIVE from disk: corrupting a verdict fails the suite NAMING the
#     kernel — an anti-self-report gate, the CLI never sees the manifest).
#   * e2e_approved_ir.test  — approved-IR handoff e2e: SAFE kernels emit
#     <out>.verified.ll which llc compiles for nvptx64-nvidia-cuda (+ amdgcn
#     smoke); error kernels exit 1 and emit NO .verified.ll (asserted).

config.name = "kernscope-verify"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".test"]

# test_source_root: the source directory holding the .test files (this dir).
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: where lit materializes %T/%t; the obj root mirrors the
# source layout (build/tests/verify).
config.test_exec_root = os.path.join(config.kernscope_obj_root, "tests", "verify")

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

# Repo root (test_source_root = <root>/tests/verify -> two dirnames up), so
# tests can reference committed fixtures by source path instead of fragile
# %S-relative paths.
config.substitutions.append(
    ("%{kernscope_source_dir}",
     os.path.dirname(os.path.dirname(config.test_source_root)))
)

# Exact literal substitution for our tool (see the main suite's comment:
# add_tool_substitutions keys are word-boundary regexes that would leave a
# stray '%' inside "%kernscope-verify").
config.substitutions.append(
    ("%kernscope-verify",
     os.path.join(config.kernscope_tools_dir, "kernscope-verify"))
)

# excludes: auxiliary inputs for the tests.
config.excludes = [
    "corpus",
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
]

# Tweak the PATH to include the LLVM tools dir (llc, FileCheck, count, not).
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.kernscope_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["kernscope-verify", "opt", "FileCheck", "count", "not"], tool_dirs
)

# Hardware feature detection mirrors the main suite (verify runs clang
# GPU-less: -emit-llvm only, no GPU needed -> always no-gpu capable).
config.available_features.add("no-gpu")

if os.path.exists("/dev/kfd"):
    config.available_features.add("gpu-amd")

if shutil.which("nvcc") is not None:
    config.available_features.add("gpu-nvidia-rental")
