#!/usr/bin/env python3
"""capture_hip_check.py — todo-45 LIVE acceptance: capture the todo-5
layernorm_fused HIP app on gfx1101 under libkernscope_capture_hip.so and
validate the produced bundle.

Checks (all asserted, exit 0 on pass):
  1. the app runs unmodified under capture and its output still passes the
     golden comparator (capture must not alter app behavior);
  2. a COMPLETE v0 bundle is emitted (kernscope-capture-v0 ... end-capture-v0);
  3. the launch record carries the kernel name, grid=1024x1x1, block=128x1x1
     (M=1024 rows, THREADS=128), arg count 6 (dX dG dB dY eps n);
  4. the allocation map + H2D copy records carry EXACTLY the known input
     buffer byte counts (X = M*N*2, gamma = N*2, beta = N*2, Y = M*N*2);
  5. INPUT BUFFER BYTES HASH-MATCH: input-buffer-*.bin blobs (captured H2D
     host sources) sha256 == sha256 of the app's known inputs from
     golden_inputs.npz;
  6. KERNEL BINARY BLOB present: code-object-*.hsaco dumped and the bundle's
     module record sha256 matches the blob on disk.

Usage: capture_hip_check.py <build-dir> <capture-hip-source-dir>
"""

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE_DIR = SRC_DIR.parent.parent / "tests" / "fixtures" / "layernorm_fused"
HERE = SRC_DIR / "tests"
TOOL = BUILD_DIR / "libkernscope_capture_hip.so"
APP_SRC = FIXTURE_DIR / "layernorm.hip"
N, THREADS = 128, 128

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def bundle_lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] tool library missing: {TOOL} (run cmake --build)")
        return 1

    inputs = np.load(FIXTURE_DIR / "golden_inputs.npz")
    x = inputs["X"].astype(np.uint16)
    gamma = inputs["gamma"].astype(np.uint16)
    beta = inputs["beta"].astype(np.uint16)
    M = x.shape[0]

    with tempfile.TemporaryDirectory(prefix="capture_hip_") as td:
        work = Path(td)
        in_bin = work / "input.bin"
        with in_bin.open("wb") as f:
            f.write(np.int32(M).tobytes())
            f.write(np.int32(N).tobytes())
            f.write(x.tobytes())
            f.write(gamma.tobytes())
            f.write(beta.tobytes())

        # known input buffers (the app's known inputs, per todo-5 goldens)
        known = {
            "X": (x.tobytes(), x.size * 2),
            "gamma": (gamma.tobytes(), gamma.size * 2),
            "beta": (beta.tobytes(), beta.size * 2),
        }

        app = work / "layernorm_app"
        subprocess.run(
            ["hipcc", "--offload-arch=gfx1101", "-std=c++17", "-O3",
             str(APP_SRC), "-o", str(app)],
            cwd=work, check=True, capture_output=True)

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

        # 1. app behavior unchanged: output equals the golden (bit-exact bf16)
        y_dev = np.frombuffer(out_bin.read_bytes(), dtype=np.uint16)
        gold = np.load(FIXTURE_DIR / "golden_outputs.npz")["Y"]
        gold_u16 = (gold.astype(np.uint32) & 0xFFFF).astype(np.uint16)
        check(np.array_equal(y_dev, gold_u16),
              "captured app output == golden (behavior unchanged)")

        # 2. complete v0 bundle
        if not bundle.is_file():
            print("[FAIL] bundle missing")
            print(run.stderr)
            return 1
        lines = bundle_lines(bundle)
        check(lines and lines[0] == "kernscope-capture-v0",
              "bundle starts with kernscope-capture-v0")
        check(lines and lines[-1] == "end-capture-v0",
              "bundle ends with end-capture-v0")
        text = "\n".join(lines)

        # 3. launch record: kernel name, grid, block, arg count
        launch = [l for l in lines if l.startswith("launch ")]
        check(len(launch) == 1, f"exactly one launch record (got {len(launch)})")
        if launch:
            m = re.search(r'kernel="([^"]*)"', launch[0])
            check(m is not None and "layernorm_fused" in m.group(1),
                  f"kernel name present + names layernorm_fused (got "
                  f"{m.group(1) if m else None})")
            check("grid=1024x1x1" in launch[0], "grid=1024x1x1")
            check("block=128x1x1" in launch[0], "block=128x1x1")
            am = re.search(r"args=(\d+)", launch[0])
            check(am is not None and int(am.group(1)) == 6,
                  f"arg count == 6 (got {am.group(1) if am else None})")

        # 4. allocation map + copy byte counts == known input sizes
        allocs = [l for l in lines if l.startswith("alloc ")]
        sizes = sorted(int(re.search(r"size=(\d+)", l).group(1))
                       for l in allocs)
        expected_sizes = sorted([x.size * 2, gamma.size * 2, beta.size * 2,
                                 x.size * 2])
        check(sizes == expected_sizes,
              f"allocation sizes match known buffers {expected_sizes}")

        copies = [l for l in lines if l.startswith("copy ")]
        h2d = [c for c in copies if "cuda-kind=1" in c]
        check(len(h2d) == 3, "three H2D copies (X, gamma, beta)")
        copy_bytes = sorted(int(re.search(r"bytes=(\d+)", c).group(1))
                            for c in h2d)
        known_bytes = sorted([x.size * 2, gamma.size * 2, beta.size * 2])
        check(copy_bytes == known_bytes,
              f"H2D copy byte counts == known inputs {known_bytes}")

        # 5. INPUT BUFFER BYTES HASH-MATCH
        blobs = sorted(blob_dir.glob("input-buffer-*.bin"))
        check(len(blobs) == 3, f"three input-buffer blobs (got {len(blobs)})")
        blob_hashes = {}
        for b in blobs:
            h = hashlib.sha256(b.read_bytes()).hexdigest()
            blob_hashes[b.name] = h
            print(f"[INFO] {b.name}: sha256={h} size={b.stat().st_size}")
        known_hashes = {name: hashlib.sha256(data).hexdigest()
                        for name, (data, _) in known.items()}
        check(set(blob_hashes.values()) == set(known_hashes.values()),
              "input-buffer blobs hash-match the app's known inputs "
              f"(X={known_hashes['X'][:16]}... gamma={known_hashes['gamma'][:16]}... "
              f"beta={known_hashes['beta'][:16]}...)")

        # 6. kernel binary blob present + provenance hash in bundle
        co = sorted(blob_dir.glob("code-object-*.hsaco"))
        check(len(co) == 1, f"one code-object hsaco blob (got {len(co)})")
        if co:
            co_hash = hashlib.sha256(co[0].read_bytes()).hexdigest()
            mods = [l for l in lines if l.startswith("fatbin ")]
            check(any(f"sha256={co_hash}" in l for l in mods),
                  "bundle module record sha256 == hsaco blob sha256 "
                  f"({co_hash[:16]}...)")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_hip_live: complete bundle + hash-matched inputs "
          "+ kernel binary blob")
    return 0


if __name__ == "__main__":
    sys.exit(main())
