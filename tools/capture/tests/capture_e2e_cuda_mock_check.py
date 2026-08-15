#!/usr/bin/env python3
"""capture_e2e_cuda_mock_check.py — todo-48 CUDA mock e2e: mock-runtime
record -> bundle -> reproducer compile.

The NVIDIA runtime path is owner-gated (scripts/rental-validate.sh); the
LOCAL gate proves the mechanism chain end-to-end on the mock:
  1. the todo-43 mock-CUDA app (capture-shim-smoke, linked against
     libkernscope_mockcudart.so) runs under LD_PRELOAD of the todo-43
     LD_PRELOAD shim -> a FRESH v0 text bundle (the mock provides the
     cuda*/cu* symbols; no NVIDIA runtime anywhere).
  2. kernscope-capture emit-repro <bundle> --vendor nvidia --arch sm_80
     -> a standalone CUDA project (main.cu + CMakeLists.txt).
  3. nvcc compile-only build (the accepted local CUDA gate; running the
     reproducer on real NVIDIA hardware is owner-gated).

Usage: capture_e2e_cuda_mock_check.py <build-dir> <source-dir>
"""

import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
SMOKE = BUILD_DIR / "libs" / "capture-core" / "capture-shim-smoke"
SHIM = BUILD_DIR / "libkernscope_intercept.so"
CLI = BUILD_DIR / "kernscope-capture"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not SMOKE.is_file() or not SHIM.is_file():
        print(f"[FAIL] mock shim chain missing: {SMOKE} / {SHIM} "
              f"(build with CUDA_HOME available)")
        return 1
    if not CLI.is_file():
        print(f"[FAIL] kernscope-capture CLI missing: {CLI}")
        return 1

    with tempfile.TemporaryDirectory(prefix="capture_e2e_cuda_mock_") as td:
        work = Path(td)
        bundle = work / "mock-capture.bundle"
        import os
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(SHIM)

        # 1. mock-runtime record -> bundle (the shim flushes the v0 bundle
        #    at exit via KERNSCOPE_CAPTURE_LOG)
        run = subprocess.run([str(SMOKE), str(bundle)], env=env,
                             capture_output=True, text=True, cwd=work)
        print(run.stdout, end="")
        print(run.stderr, end="")
        check(run.returncode == 0, "mock app ran to completion (exit 0)")
        check(bundle.is_file(), "mock record produced a bundle file")
        if bundle.is_file():
            head = bundle.read_text().splitlines()
            check(head and head[0] == "kernscope-capture-v0",
                  "bundle is v0 text format")
            check(any("launch kernel=" in l for l in head),
                  "bundle carries a launch record")

        # 2. emit-repro -> standalone CUDA project (compile-only gate)
        repro = work / "reproducer"
        emit = subprocess.run(
            [str(CLI), "emit-repro", str(bundle),
             "--vendor", "nvidia", "--arch", "sm_80",
             "--out", str(repro), "--force"],
            capture_output=True, text=True, cwd=work)
        print(emit.stdout, end="")
        print(emit.stderr, end="")
        check(emit.returncode == 0, "emit-repro (CUDA) exited 0")
        check((repro / "main.cu").is_file() and
              (repro / "CMakeLists.txt").is_file(),
              "emitted standalone CUDA project (main.cu + CMakeLists.txt)")

        # 3. nvcc compile-only (the local CUDA gate; runtime is rental-gated)
        cfg = subprocess.run(["cmake", "-S", str(repro), "-B", str(repro / "build")],
                             capture_output=True, text=True, cwd=work)
        check(cfg.returncode == 0, "reproducer cmake configure exit 0")
        bld = subprocess.run(["cmake", "--build", str(repro / "build")],
                             capture_output=True, text=True, cwd=work)
        print(bld.stdout, end="")
        print(bld.stderr, end="")
        check(bld.returncode == 0, "reproducer compiles (nvcc sm_80, "
                                   "compile-only)")
        check((repro / "build" / "kernscope_repro.cu.o").is_file(),
              "compiled reproducer object produced")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_e2e_cuda_mock: mock record -> bundle -> "
          "reproducer compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
