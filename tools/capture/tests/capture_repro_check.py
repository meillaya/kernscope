#!/usr/bin/env python3
"""capture_repro_check.py — todo-47 LIVE acceptance: layernorm bundle ->
emit-repro -> standalone build -> run -> captured-output hash match.

Flow (all asserted, exit 0 on pass):
  1. build the todo-5 layernorm_fused HIP app and run it on gfx1101 under
     libkernscope_capture_hip.so (todo-45 live capture) -> FRESH v0 bundle +
     sidecar blobs; record the app's output hash (the "captured" output).
  2. `kernscope-capture emit-repro <bundle> --vendor amd --arch gfx1101`
     -> standalone project dir.
  3. STANDALONE check: the emitted project contains no kernscope references
     (no kernscope include paths, no kernscope link libraries) - zero repo
     deps at runtime by construction.
  4. build the emitted project with plain cmake + hipcc (compile gate).
  5. RUN the reproducer on gfx1101; the reproducer's dumped output hash MUST
     equal the captured output hash (asserted).

Usage: capture_repro_check.py <build-dir> <capture-source-dir>
"""

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE_DIR = SRC_DIR.parent.parent / "tests" / "fixtures" / "layernorm_fused"
CAPTURE_FIXTURE = SRC_DIR.parent.parent / "tests" / "fixtures" / "capture"
TOOL = BUILD_DIR / "libkernscope_capture_hip.so"
CLI = BUILD_DIR / "kernscope-capture"
APP_SRC = FIXTURE_DIR / "layernorm.hip"
N, THREADS = 128, 128

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] tool library missing: {TOOL} (run cmake --build)")
        return 1
    if not CLI.is_file():
        print(f"[FAIL] kernscope-capture CLI missing: {CLI} (run cmake --build)")
        return 1

    import numpy as np

    inputs = np.load(FIXTURE_DIR / "golden_inputs.npz")
    x = inputs["X"].astype(np.uint16)
    gamma = inputs["gamma"].astype(np.uint16)
    beta = inputs["beta"].astype(np.uint16)
    M = x.shape[0]

    with tempfile.TemporaryDirectory(prefix="capture_repro_") as td:
        work = Path(td)
        in_bin = work / "input.bin"
        with in_bin.open("wb") as f:
            f.write(np.int32(M).tobytes())
            f.write(np.int32(N).tobytes())
            f.write(x.tobytes())
            f.write(gamma.tobytes())
            f.write(beta.tobytes())

        app = work / "layernorm_app"
        subprocess.run(
            ["hipcc", "--offload-arch=gfx1101", "-std=c++17", "-O3",
             str(APP_SRC), "-o", str(app)],
            cwd=work, check=True, capture_output=True)

        # 1. live capture (fresh bundle + blobs + output)
        blob_dir = work / "blobs"
        blob_dir.mkdir()
        bundle = work / "capture.bundle"
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(TOOL)
        env["KERNSCOPE_HIP_BUNDLE"] = str(bundle)
        env["KERNSCOPE_HIP_BLOB_DIR"] = str(blob_dir)
        out_bin = work / "output.bin"
        run = subprocess.run(
            [str(app), str(in_bin), str(out_bin)], env=env,
            capture_output=True, text=True, cwd=work)
        print(run.stderr, end="")
        check(run.returncode == 0, "captured app ran to completion")
        check(bundle.is_file() and out_bin.is_file(),
              "live capture produced bundle + output")
        captured_sha = sha(out_bin)
        print(f"[INFO] captured output sha256={captured_sha}")

        # 2. emit-repro -> standalone project
        repro = work / "reproducer"
        emit = subprocess.run(
            [str(CLI), "emit-repro", str(bundle),
             "--blob-dir", str(blob_dir),
             "--vendor", "amd", "--arch", "gfx1101",
             "--out", str(repro), "--force"],
            capture_output=True, text=True, cwd=work)
        print(emit.stderr, end="")
        print(emit.stdout, end="")
        check(emit.returncode == 0, "emit-repro exited 0")
        main_src = repro / "main.hip"
        check(main_src.is_file() and (repro / "CMakeLists.txt").is_file(),
              "emitted standalone project (main.hip + CMakeLists.txt)")

        # 3. standalone: zero kernscope dependencies in the emitted project.
        #    Dependency markers = include paths, header names, linked libs;
        #    provenance DOC comments may mention the tool name.
        emitted = "\n".join(p.read_text() for p in
                            [main_src, repro / "CMakeLists.txt",
                             repro / "README.md"])
        dep_markers = ["#include <kernscope", "#include \"kernscope",
                       "kernscope/capture.h", "kernscope/bundle.h",
                       "kernscope::", "capture_core", "libkernscope"]
        hits = [m for m in dep_markers if m in emitted]
        check(not hits,
              f"no kernscope dependency markers in emitted project "
              f"(found: {hits if hits else 'none'})")

        # 4. standalone build (plain cmake, hipcc)
        build = subprocess.run(
            ["cmake", "-S", str(repro), "-B", str(repro / "build")],
            capture_output=True, text=True, cwd=work)
        check(build.returncode == 0, "reproducer cmake configure exit 0")
        build = subprocess.run(
            ["cmake", "--build", str(repro / "build")],
            capture_output=True, text=True, cwd=work)
        print(build.stdout, end="")
        check(build.returncode == 0, "reproducer compiles standalone (hipcc)")
        repro_bin = repro / "build" / "kernscope_repro"
        check(repro_bin.is_file(), f"reproducer binary built ({repro_bin})")

        # 5. RUN on gfx1101 + hash-match the captured output
        repro_out = work / "repro_output.bin"
        rrun = subprocess.run([str(repro_bin), str(repro_out)],
                              capture_output=True, text=True, cwd=work)
        print(rrun.stdout, end="")
        print(rrun.stderr, end="")
        check(rrun.returncode == 0, "reproducer ran on gfx1101 (exit 0)")
        check(repro_out.is_file(), "reproducer dumped output file")
        if repro_out.is_file():
            repro_sha = sha(repro_out)
            print(f"[INFO] reproducer output sha256={repro_sha}")
            check(repro_sha == captured_sha,
                  f"OUTPUT HASH MATCH: reproducer {repro_sha[:16]}... == "
                  f"captured {captured_sha[:16]}...")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_repro_gen: standalone reproducer emitted, "
          "compiled, ran and hash-matched the captured output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
