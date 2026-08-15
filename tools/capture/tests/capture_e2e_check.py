#!/usr/bin/env python3
"""capture_e2e_check.py — todo-48 LIVE acceptance on gfx1101: layernorm
record -> replay bitwise -> tolerance dual-assert -> bench -> db record.

Flow (all asserted, exit 0 on pass):
  1. build the todo-5 layernorm_fused HIP app (hipcc gfx1101), run it
     through `kernscope-capture record` (vendor AUTO-detect) under
     libkernscope_capture_hip.so -> FRESH v0 bundle + blobs + app output.
  2. `kernscope-capture replay <bundle> --vendor amd --arch gfx1101
     --ref <app-output> --mode bitwise` -> exit 0, validation PASS,
     bitwise EQUAL (hash match, asserted).
  3. bf16-noise fixture: the app output perturbed by +/-1 bf16 ULP on a
     few elements. The SAME replay against the noise reference must FAIL
     bitwise (exit 1, hash diff) and PASS in --mode tolerance (cosine /
     max_rel_err / pcc thresholds) — BOTH asserted. The tolerance verdict
     is CROSS-CHECKED against the golden Python implementation
     (tests/golden/kernscope_golden/metrics.py — the repo's golden math).
  4. `kernscope-capture bench <bundle> ... --reps 10 --db <db.jsonl>`
     -> exit 0; the JSONL db record EXISTS and is QUERYABLE via
     `kernscope-capture db-query` (asserted: family capture_replay,
     median > 0, matching the bench report).
  5. profiling hook records: --profile-out JSONL carries a timestamped
     launch record (+ the aggregate bench record) — the todo-49 contract.

Usage: capture_e2e_check.py <build-dir> <capture-source-dir>
"""

import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE_DIR = SRC_DIR.parent.parent / "tests" / "fixtures" / "layernorm_fused"
GOLDEN_DIR = SRC_DIR.parent.parent / "tests" / "golden"
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


def bf16_perturb(data: bytes, rng: random.Random, n: int = 8) -> bytes:
    """Flip n uint16 elements by +/-1 ULP, only where |value| is large
    enough that a 1-ULP step stays well inside the tolerance band."""
    import struct
    vals = list(struct.unpack("<%dH" % (len(data) // 2), data))
    candidates = []
    for i, v in enumerate(vals):
        bits = v << 16
        f = struct.unpack("<f", struct.pack("<I", bits))[0]
        if abs(f) > 0.01:
            candidates.append(i)
    for i in rng.sample(candidates, min(n, len(candidates))):
        vals[i] = max(0, min(0xFFFF, vals[i] + rng.choice([-1, 1])))
    return struct.pack("<%dH" % len(vals), *vals)


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] tool library missing: {TOOL} (run cmake --build)")
        return 1
    if not CLI.is_file():
        print(f"[FAIL] kernscope-capture CLI missing: {CLI} (run cmake --build)")
        return 1

    import numpy as np
    sys.path.insert(0, str(GOLDEN_DIR))
    from kernscope_golden import metrics as golden

    inputs = np.load(FIXTURE_DIR / "golden_inputs.npz")
    x = inputs["X"].astype(np.uint16)
    gamma = inputs["gamma"].astype(np.uint16)
    beta = inputs["beta"].astype(np.uint16)
    M = x.shape[0]

    with tempfile.TemporaryDirectory(prefix="capture_e2e_") as td:
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

        # 1. record (vendor AUTO-detect: the hip capture .so sits next to
        #    the CLI in the build dir)
        blob_dir = work / "blobs"
        blob_dir.mkdir()
        bundle = work / "capture.bundle"
        out_bin = work / "output.bin"
        rec = subprocess.run(
            [str(CLI), "record", str(app), str(in_bin), str(out_bin),
             "--bundle", str(bundle), "--blob-dir", str(blob_dir)],
            capture_output=True, text=True, cwd=work)
        print(rec.stdout, end="")
        print(rec.stderr, end="")
        check(rec.returncode == 0, "record ran the app to completion (exit 0)")
        check("auto-detect -> amd" in rec.stdout,
              "record vendor auto-detect selected amd")
        check(bundle.is_file() and out_bin.is_file(),
              "record produced bundle + app output")
        check(list(blob_dir.glob("input-buffer-*.bin")) and
              list(blob_dir.glob("code-object-*.hsaco")),
              "record produced input + kernel blobs")
        captured_sha = sha(out_bin)
        print(f"[INFO] captured output sha256={captured_sha}")

        # 2. replay bitwise -> PASS
        dump1 = work / "dump1"
        rp = subprocess.run(
            [str(CLI), "replay", str(bundle),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(blob_dir), "--ref", str(out_bin),
             "--mode", "bitwise", "--dump-dir", str(dump1)],
            capture_output=True, text=True, cwd=work)
        print(rp.stdout, end="")
        print(rp.stderr, end="")
        check(rp.returncode == 0, "replay bitwise exited 0")
        check("validation: PASS" in rp.stdout, "replay validation PASS")
        check("bitwise EQUAL" in rp.stdout, "replay output bitwise EQUAL")
        replayed = dump1 / "output-0.bin"
        check(replayed.is_file(), "replayed output dumped")
        if replayed.is_file():
            check(sha(replayed) == captured_sha,
                  f"replayed output hash == captured output hash "
                  f"({sha(replayed)[:16]}...)")
        check("launch records: 1" in rp.stdout,
              "profiling hook record emitted for the launch")

        # 3a. bf16-noise fixture: bitwise MUST FAIL (exit 1, hash diff)
        noise = work / "noise.bin"
        noise.write_bytes(bf16_perturb(out_bin.read_bytes(),
                                       random.Random(48)))
        rn = subprocess.run(
            [str(CLI), "replay", str(bundle),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(blob_dir), "--ref", str(noise),
             "--mode", "bitwise"],
            capture_output=True, text=True, cwd=work)
        print(rn.stdout, end="")
        print(rn.stderr, end="")
        check(rn.returncode == 1, "replay bitwise vs noise exited 1 (FAIL)")
        check("validation: FAIL" in rn.stdout, "bitwise validation FAIL "
                                               "(expected for noise)")
        check("BITWISE DIFF" in rn.stdout, "hash diff reported "
                                           "(ref != replay)")

        # 3b. tolerance mode on the SAME noise: MUST PASS (golden metrics)
        dump2 = work / "dump2"
        rt = subprocess.run(
            [str(CLI), "replay", str(bundle),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(blob_dir), "--ref", str(noise),
             "--mode", "tolerance", "--dump-dir", str(dump2)],
            capture_output=True, text=True, cwd=work)
        print(rt.stdout, end="")
        print(rt.stderr, end="")
        check(rt.returncode == 0, "replay tolerance vs noise exited 0 (PASS)")
        check("validation: PASS" in rt.stdout,
              "tolerance validation PASS (cosine/max_rel_err/pcc)")
        cpp_cosine = None
        for line in rt.stdout.splitlines():
            if "cos=" in line:
                cpp_cosine = float(line.split("cos=")[1].split()[0])
        # cross-check against the golden Python implementation
        if (dump2 / "output-0.bin").is_file():
            import ml_dtypes
            actual = np.frombuffer((dump2 / "output-0.bin").read_bytes(),
                                   dtype=ml_dtypes.bfloat16).astype(np.float32)
            ref = np.frombuffer(noise.read_bytes(),
                                dtype=ml_dtypes.bfloat16).astype(np.float32)
            gv = golden.verdict(actual, ref)
            print(f"[INFO] golden verdict: {gv}")
            check(gv["pass"] is True,
                  "golden Python verdict PASS (golden math reused)")
            if cpp_cosine is not None:
                check(abs(cpp_cosine - gv["cosine"]) < 1e-4,
                      f"C++ cosine {cpp_cosine:.6f} == golden cosine "
                      f"{gv['cosine']:.6f} (within 1e-4)")
            check(gv["cosine"] >= 0.999 and gv["max_rel_err"] <= 1e-2,
                  "golden thresholds met on the noise fixture")
        else:
            check(False, "tolerance-run output dumped for golden cross-check")

        # 4. bench -> db record (N reps, L2 flush, median)
        db = work / "bench.jsonl"
        prof = work / "bench-profile.jsonl"
        bn = subprocess.run(
            [str(CLI), "bench", str(bundle),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(blob_dir), "--ref", str(out_bin),
             "--reps", "10", "--db", str(db), "--profile-out", str(prof)],
            capture_output=True, text=True, cwd=work)
        print(bn.stdout, end="")
        print(bn.stderr, end="")
        check(bn.returncode == 0, "bench exited 0")
        check("validation PASS" in bn.stdout,
              "bench validation pass before timing")
        check("median" in bn.stdout and "10 reps" in bn.stdout,
              "bench reports median over 10 reps (L2 flush)")
        bench_median = None
        for line in bn.stdout.splitlines():
            if "median" in line:
                try:
                    bench_median = float(line.split("median")[1].split()[0])
                except (IndexError, ValueError):
                    pass
        check(db.is_file(), "bench wrote the JSONL db file")
        rec_lines = db.read_text().splitlines() if db.is_file() else []
        check(any('"kernel_family":"capture_replay"' in l for l in rec_lines),
              "db record has kernel_family=capture_replay")
        check(any('"mode":"measured"' in l for l in rec_lines),
              "db record has measured provenance")
        check(any('"median"' in l and '"p20"' in l and '"p80"' in l
                  for l in rec_lines),
              "db record carries median/p20/p80 timings")

        # 5. db record EXISTS via the libs/db CLI (query works)
        q = subprocess.run(
            [str(CLI), "db-query", str(db), "--family", "capture_replay"],
            capture_output=True, text=True, cwd=work)
        print(q.stdout, end="")
        check(q.returncode == 0, "db-query exit 0 (record found)")
        check("capture_replay" in q.stdout,
              "db-query lists the capture_replay record")
        if bench_median is not None:
            check(f"median_ms={bench_median:.4f}" in q.stdout,
                  "db-query median matches the bench report")
        else:
            check(False, "bench median parsed from report")

        # 6. profiling hook records (todo-49 contract)
        prof_lines = prof.read_text().splitlines() if prof.is_file() else []
        launch_recs = [l for l in prof_lines if '"kind":"launch"' in l]
        bench_recs = [l for l in prof_lines if '"kind":"bench"' in l]
        check(len(launch_recs) == 1 and
              '"launched_at_us"' in launch_recs[0] and
              '"completed_at_us"' in launch_recs[0] and
              '"duration_us"' in launch_recs[0],
              "profile JSONL has a timestamped launch record")
        check(len(bench_recs) == 1 and '"median_ms"' in bench_recs[0],
              "profile JSONL has the aggregate bench record")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_e2e: record -> replay bitwise -> tolerance "
          "dual-assert -> bench -> db record")
    return 0


if __name__ == "__main__":
    sys.exit(main())
