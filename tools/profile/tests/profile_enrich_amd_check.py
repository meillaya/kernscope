#!/usr/bin/env python3
"""profile_enrich_amd_check.py — todo-50 LIVE enrichment acceptance on gfx1101:
profile the layernorm_fused app through the rocprofiler-sdk buffer-tracing
tracer and assert the per-kernel ENRICHMENT fields on the kernel record.

Flow (all asserted, exit 0 on pass):
  1. build the todo-5 layernorm_fused HIP app (hipcc --offload-arch=gfx1101)
     + a golden-shaped input (M=1024 rows x N=128 cols, bf16) — the same
     fixture as the todo-49 check.
  2. run it under LD_PRELOAD=libkernscope_profile_hip.so with
     KERNSCOPE_PROFILE_OUT=<work>/profile.jsonl.
  3. parse the JSONL and assert on the layernorm kernel record (enriched
     shape, todo 50):
     - RESOURCES FROM THE CODE-OBJECT SERVICE: vgpr > 0 AND sgpr > 0 AND
       lds_bytes > 0 (the fixture kernel uses __shared__ -> static LDS),
       accum_vgpr present, kernel_address != 0, scratch_bytes >= 0 — the
       values matching the hipcc gfx1101 build (vgpr in 1..256, sgpr in
       1..128, LDS >= the 2 x 128 x 4 B static arrays);
     - OCCUPANCY VIA LIBS/OCCUPANCY (todo 11): occupancy in (0, 1] and NOT
       the "unknown-resources" marker, limiter in the contract set
       {registers, shared, threads, blocks}, blocks_per_sm > 0;
     - HONEST DEGRADATION: ptx_available == false (AMD has no PTX-level
       diagnostics path — never fabricated);
     - the todo-49 fields preserved: duration_ns > 0, grid == [M,1,1],
       workgroup == [N,1,1], correlation_id linking to the hipLaunchKernel
       HIP-API record.

Usage: profile_enrich_amd_check.py <build-dir> <profile-src-dir>
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

    with tempfile.TemporaryDirectory(prefix="profile_enrich_amd_") as td:
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
            target = kernels[0]
        check(target is not None, "a kernel record is present")

        if target is not None:
            print(f"[INFO] kernel {target.get('name')}: "
                  f"vgpr={target.get('vgpr')} sgpr={target.get('sgpr')} "
                  f"accum_vgpr={target.get('accum_vgpr')} "
                  f"lds_bytes={target.get('lds_bytes')} "
                  f"scratch_bytes={target.get('scratch_bytes')} "
                  f"kernel_address={target.get('kernel_address')} "
                  f"occupancy={target.get('occupancy')} "
                  f"limiter={target.get('limiter')} "
                  f"blocks_per_sm={target.get('blocks_per_sm')}")

            # --- code-object service facts (todo 50, acceptance) -------------
            vgpr = target.get("vgpr", -1)
            sgpr = target.get("sgpr", -1)
            lds = target.get("lds_bytes", -1)
            check(vgpr > 0, "arch_vgpr_count non-zero (code-object service)")
            check(sgpr > 0, "sgpr_count non-zero (code-object service)")
            check(lds > 0, "LDS non-zero (fixture uses __shared__)")
            check(1 <= vgpr <= 256,
                  f"vgpr in the hipcc gfx1101 build range ({vgpr})")
            check(1 <= sgpr <= 128,
                  f"sgpr in the hipcc gfx1101 build range ({sgpr})")
            # static LDS: 2 x THREADS x 4 B fp32 arrays (+ padding) >= 1024
            check(lds >= 2 * N * 4, f"LDS >= the static __shared__ arrays ({lds})")
            check(target.get("accum_vgpr", -1) >= 0, "accum_vgpr present")
            check(target.get("scratch_bytes", -1) >= 0, "scratch_bytes present")
            check(target.get("kernel_address", 0) != 0,
                  "kernel_address non-zero (PC-sampling hook)")

            # --- occupancy via libs/occupancy (todo 50, acceptance) ----------
            occ = target.get("occupancy")
            check(occ != "unknown-resources",
                  "occupancy computed (NOT unknown-resources)")
            check(isinstance(occ, (int, float)) and 0 < occ <= 1,
                  f"occupancy in (0,1] ({occ})")
            lim = target.get("limiter", "")
            check(lim in ("registers", "shared", "threads", "blocks"),
                  f"limiter in the contract set ({lim})")
            check(target.get("blocks_per_sm", 0) > 0, "blocks_per_sm > 0")

            # --- honest degradation (todo 50) ---------------------------------
            check(target.get("ptx_available") is False,
                  "ptx_available == false on AMD (no PTX diagnostics path)")

            # --- todo-49 fields preserved -------------------------------------
            check(target.get("duration_ns", 0) > 0, "duration_ns > 0")
            check(target.get("grid") == [M, 1, 1],
                  f"grid matches the launch (M={M} blocks)")
            check(target.get("workgroup") == [N, 1, 1],
                  f"workgroup matches the launch ({N} threads)")
            cid = target.get("correlation_id")
            linked = [a for a in apis if a.get("correlation_id") == cid]
            check(len(linked) >= 1,
                  f"correlation_id {cid} links to a HIP API record")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] profile_enrich_amd: layernorm_fused profiled live on "
          "gfx1101; kernel record enriched with code-object vgpr/sgpr/LDS "
          "(non-zero, matching the hipcc build), occupancy via libs/occupancy "
          "(0,1] with limiter, and honest ptx_available == false")
    return 0


if __name__ == "__main__":
    sys.exit(main())
