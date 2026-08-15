#!/usr/bin/env python3
"""profile_amd_check.py — todo-49 LIVE acceptance: profile the layernorm_fused
app on gfx1101 through the rocprofiler-sdk buffer-tracing tracer and assert
per-kernel records.

Flow (all asserted, exit 0 on pass):
  1. build the todo-5 layernorm_fused HIP app (hipcc --offload-arch=gfx1101)
     + a golden-shaped input (M=1024 rows x N=128 cols, bf16).
  2. run it under LD_PRELOAD=libkernscope_profile_hip.so with
     KERNSCOPE_PROFILE_OUT=<work>/profile.jsonl (rocprofiler-sdk discovers
     the tool's rocprofiler_configure at init).
  3. parse the JSONL: assert
     - >= 1 kernel dispatch record with duration_ns > 0;
     - grid/workgroup MATCHING the launch: the app launches
       layernorm_fused<<<M=1024, 128>>> -> grid=[1024,1,1] (blocks),
       workgroup=[128,1,1] (threads);
     - correlation_id linking: the kernel dispatch record's correlation_id
       equals the correlation_id of a HIP API record (the hipLaunchKernel
       that enqueued it).

Usage: profile_amd_check.py <build-dir> <profile-src-dir>
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE_DIR = SRC_DIR.parent.parent / "tests" / "fixtures" / "layernorm_fused"
TOOL = BUILD_DIR / "libkernscope_profile_hip.so"
APP_SRC = FIXTURE_DIR / "layernorm.hip"
M, N = 1024, 128  # launch: layernorm_fused<<<M, THREADS=128>>>

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] tracer tool library missing: {TOOL} (run cmake --build)")
        return 1

    import numpy as np

    rng = np.random.default_rng(20260815)
    x = rng.normal(0.0, 1.0, size=(M, N)).astype(np.float32)
    gamma = np.ones(N, dtype=np.float32)
    beta = np.zeros(N, dtype=np.float32)

    def bf16(u32: np.ndarray) -> np.ndarray:
        return (u32.astype(np.uint32) >> 16).astype(np.uint16)

    with tempfile.TemporaryDirectory(prefix="profile_amd_") as td:
        work = Path(td)
        in_bin = work / "input.bin"
        with in_bin.open("wb") as f:
            f.write(np.int32(M).tobytes())
            f.write(np.int32(N).tobytes())
            f.write(bf16(x).tobytes())
            f.write(bf16(gamma).tobytes())
            f.write(bf16(beta).tobytes())

        app = work / "layernorm_app"
        build = subprocess.run(
            ["hipcc", "--offload-arch=gfx1101", "-std=c++17", "-O3",
             str(APP_SRC), "-o", str(app)],
            cwd=work, capture_output=True, text=True)
        print(build.stderr, end="")
        check(build.returncode == 0, "layernorm_fused app built (hipcc gfx1101)")
        if build.returncode != 0:
            return 1

        profile_out = work / "profile.jsonl"
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(TOOL)
        env["KERNSCOPE_PROFILE_OUT"] = str(profile_out)
        out_bin = work / "output.bin"
        run = subprocess.run(
            [str(app), str(in_bin), str(out_bin)], env=env,
            capture_output=True, text=True, cwd=work)
        print(run.stderr, end="")
        check(run.returncode == 0, "app ran to completion under the tracer")
        check(profile_out.is_file(), "JSONL profile output written")

        records = []
        if profile_out.is_file():
            for line in profile_out.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    check(False, f"JSONL line parses: {line[:80]!r}")
        kernels = [r for r in records if r.get("kind") == "kernel"]
        apis = [r for r in records if r.get("kind") == "api"]
        print(f"[INFO] {len(kernels)} kernel record(s), {len(apis)} API "
              f"record(s)")

        check(len(kernels) >= 1, ">= 1 kernel dispatch record")

        target = None
        for k in kernels:
            if "layernorm_fused" in k.get("name", ""):
                target = k
                break
        if target is None and kernels:
            target = kernels[0]  # fall back to the first (mangled) kernel
        check(target is not None, "a kernel record is present")

        if target is not None:
            d = target.get("duration_ns", 0)
            print(f"[INFO] kernel {target.get('name')}: duration_ns={d} "
                  f"grid={target.get('grid')} workgroup={target.get('workgroup')} "
                  f"correlation_id={target.get('correlation_id')} "
                  f"lds_bytes={target.get('lds_bytes')} "
                  f"scratch_bytes={target.get('scratch_bytes')}")
            check(d > 0, "kernel record has duration_ns > 0")
            check(target.get("grid") == [M, 1, 1],
                  f"grid matches the launch (M={M} blocks)")
            check(target.get("workgroup") == [N, 1, 1],
                  f"workgroup matches the launch ({N} threads)")

            cid = target.get("correlation_id")
            linked = [a for a in apis if a.get("correlation_id") == cid]
            check(len(linked) >= 1,
                  f"correlation_id {cid} links to a HIP API record")
            if linked:
                print(f"[INFO] linked API record: {linked[0].get('operation')} "
                      f"correlation_id={linked[0].get('correlation_id')}")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] profile_tracer_amd: layernorm_fused profiled live on "
          "gfx1101; kernel records have duration > 0, matching "
          "grid/workgroup and a correlation_id joining the HIP API record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
