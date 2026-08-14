# -*- Python -*-

import os
import shutil
import subprocess

import lit.formats

from lit.llvm import llvm_config

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = "kernscope"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".ll", ".test", ".mlir"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.kernscope_obj_root, "test")

config.substitutions.append(("%PATH%", config.environment["PATH"]))
config.substitutions.append(("%shlibext", config.llvm_shlib_ext))

llvm_config.with_system_environment(["HOME", "INCLUDE", "LIB", "TMP", "TEMP"])

llvm_config.use_default_substitutions()

# lit 18 does not provide %llvm_tools_dir out of the box; define it so tests
# can reference the installed LLVM tool dir (${LLVM_TOOLS_BINARY_DIR}) where
# opt/FileCheck/count/not live.
config.substitutions.append(("%llvm_tools_dir", config.llvm_tools_dir))

# Repo root, so tests can reference committed fixtures (tests/fixtures/,
# tests/analyze/corpus/, tests/amdparse/corpus/) by source path instead of
# fragile %S-relative paths. test_source_root = <root>/test, so one dirname
# up is the repo root. Used by test/analyze/*.test (todo 20 CLI e2e suite).
config.substitutions.append(
    ("%{kernscope_source_dir}", os.path.dirname(config.test_source_root))
)

# Exact literal substitution for our own tool. llvm_config's
# add_tool_substitutions keys are word-boundary regexes (\bkernscope-analyze\b)
# that also match INSIDE "%kernscope-analyze" and would leave a stray '%'
# behind; a plain literal key (as %{kernscope_source_dir} above) replaces the
# token exactly. kernscope_tools_dir is NOT on PATH (only llvm_tools_dir is),
# so tests must reference the binary through this substitution.
config.substitutions.append(
    ("%kernscope-analyze",
     os.path.join(config.kernscope_tools_dir, "kernscope-analyze"))
)

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = [
    "Inputs",
    "Examples",
    "CMakeLists.txt",
    "README.txt",
    "LICENSE.txt",
]

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

# Kernscope build tools first, then the installed LLVM tools (FileCheck is
# discovered via ${LLVM_TOOLS_BINARY_DIR} -> config.llvm_tools_dir).
tool_dirs = [config.kernscope_tools_dir, config.llvm_tools_dir]
llvm_config.add_tool_substitutions(
    ["kernscope-analyze", "opt", "FileCheck", "count", "not"], tool_dirs
)

# --- Hardware feature detection (mirrors the ctest LABELS convention) -------
#
# no-gpu:            always available -- IR-level / compile-only tests run
#                    everywhere, including rented GPU hosts.
# gpu-amd:           an AMD GPU is present and usable (ROCm kernel driver
#                    node /dev/kfd exists; the 7800 XT box satisfies this).
# gpu-nvidia-rental: the NVIDIA toolchain (nvcc) is present. Compile-only
#                    tests may require this; actual NVIDIA runtime execution
#                    happens only on owner-run rented hardware.
# gfx1101:           the local AMD GPU is a Radeon RX 7800 XT (gfx1101).
config.available_features.add("no-gpu")

if os.path.exists("/dev/kfd"):
    config.available_features.add("gpu-amd")
    rocminfo = shutil.which("rocminfo")
    if rocminfo is not None:
        try:
            out = subprocess.run(
                [rocminfo], capture_output=True, text=True, timeout=30
            )
            if "gfx1101" in out.stdout:
                config.available_features.add("gfx1101")
        except (OSError, subprocess.TimeoutExpired):
            pass

if shutil.which("nvcc") is not None:
    config.available_features.add("gpu-nvidia-rental")
