#!/usr/bin/env python3
"""integ_capture_tune_check.py — todo-53 LIVE acceptance on gfx1101:
capture->tune replay-based launch-config tuning over the layernorm_fused
capture bundle (tests/fixtures/capture/layernorm-live/, committed todo-45
capture; output.bin is the captured reference).

Flow (all asserted, exit 0 on pass):
  1. HAPPY RUN (no injection): kernscope-tune --from-capture <bundle>
     --vendor amd --arch gfx1101 --blob-dir <blobs> --ref <output.bin>
     --json --db <db.jsonl>:
       * exit 0, JSON status "ok";
       * >= 3 candidates explored (validated), the FIRST is the captured
         baseline config (grid 1024x1x1, block 128x1x1, is_baseline true) —
         baseline-always-first guarantees best.speedup >= 1.0;
       * every explored candidate has validation PASS (bitwise vs the
         captured reference) and is timed via libs/bench;
       * best.speedup >= 1.0 vs the baseline median;
       * db record(s) written and QUERYABLE via the libs/db CLI
         (kernscope-capture db-query) under the (arch, kernel-name-hash,
         capture-bundle-hash) key — the hashes recomputed here from the
         bundle and the kernel name; the query's min-median record matches
         evidence.best.
  2. INJECTED BROKEN CONFIG: --inject-config 128,1,1:2048,1,1 (block 2048
     exceeds the launch limit 1024 threads/block) -> that candidate is
     REJECTED PRE-REPLAY (status pre_screen_rejected, reason names the
     limit), the other candidates still validate, exit 0.
  3. QA NEGATIVE: --ref <garbage> (a reference the replayed output cannot
     match) -> EVERY candidate fails bitwise validation -> the pipeline
     reports "no valid config": exit 1, JSON status no_viable_config, and
     NO db file is written (asserted).

Usage: integ_capture_tune_check.py <build-dir> <tune-source-dir>
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD_DIR = Path(sys.argv[1]).resolve()
SRC_DIR = Path(sys.argv[2]).resolve()
FIXTURE_DIR = SRC_DIR / "tests" / "fixtures" / "capture" / "layernorm-live"
TOOL = BUILD_DIR / "tools" / "tune" / "kernscope-tune"
DB_CLI = BUILD_DIR / "kernscope-capture"
BUNDLE = FIXTURE_DIR / "bundle.v0"
BLOB_DIR = FIXTURE_DIR / "blobs"
REF = FIXTURE_DIR / "output.bin"
KERNEL_NAME = "_Z15layernorm_fusedPKtS0_S0_Ptfi.kd"

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    if not TOOL.is_file():
        print(f"[FAIL] kernscope-tune binary missing: {TOOL} (run cmake --build)")
        return 1
    if not DB_CLI.is_file():
        print(f"[FAIL] kernscope-capture binary missing: {DB_CLI} (run cmake --build)")
        return 1
    if not BUNDLE.is_file() or not REF.is_file():
        print(f"[FAIL] layernorm-live fixture missing: {BUNDLE} / {REF}")
        return 1

    # the db key parts, recomputed independently of the tool
    kernel_hash = "kernel=" + sha256_bytes(KERNEL_NAME.encode())
    bundle_hash = "bundle=" + sha256_bytes(BUNDLE.read_bytes())

    with tempfile.TemporaryDirectory(prefix="integ_capture_tune_") as td:
        work = Path(td)

        # --- 1. HAPPY RUN -----------------------------------------------------
        db_path = work / "tune.jsonl"
        r = subprocess.run(
            [str(TOOL), "--from-capture", str(BUNDLE),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(BLOB_DIR), "--ref", str(REF),
             "--json", "--db", str(db_path)],
            capture_output=True, text=True)
        print(r.stdout, end="")
        print(r.stderr, end="")
        check(r.returncode == 0, "happy run exits 0")

        ev = json.loads(r.stdout[r.stdout.index("{"):]) if r.stdout.strip() else {}
        check(ev.get("status") == "ok", "JSON status ok")
        check(ev.get("mode") == "capture_tune", "JSON mode capture_tune")
        check(ev.get("kernel_name") == KERNEL_NAME,
              f"kernel resolved from the bundle ({KERNEL_NAME})")
        check(ev.get("arch") == "gfx1101" and ev.get("vendor") == "amd",
              "arch/vendor from the bundle args (gfx1101/amd)")

        cands = ev.get("candidates", [])
        explored = [c for c in cands if c.get("status") == "validated"]
        check(len(explored) >= 3,
              f">= 3 candidates explored (got {len(explored)}: "
              f"{[c['config']['block'][0] for c in explored]})")
        check(cands and cands[0].get("config", {}).get("is_baseline") is True,
              "captured baseline config is candidate #1 (explored first)")
        base_cfg = cands[0]["config"]
        check(base_cfg.get("grid") == [1024, 1, 1] and
              base_cfg.get("block") == [128, 1, 1],
              f"baseline config is the captured grid/block (1024x1x1, 128x1x1)")
        check(all(c.get("validation", {}).get("pass") is True
                  for c in explored),
              "validation PASS for every explored config (bitwise vs capture)")
        check(all("timings" in c and c["timings"].get("median", 0) > 0
                  for c in explored),
              "every explored config timed (libs/bench median > 0)")
        speedup = ev.get("speedup", 0)
        check(speedup >= 1.0,
              f"best.speedup >= 1.0 vs baseline (got {speedup:.4f})")
        best = ev.get("best", {})
        check(best.get("config") is not None and
              ev.get("baseline", {}).get("timings", {}).get("median", 0) > 0,
              "evidence has best + baseline with timings")
        # best == min median among the measured candidates
        medians = [c["timings"]["median"] for c in explored]
        check(abs(best["timings"]["median"] - min(medians)) < 1e-9,
              "best == min of measured medians")

        # db record(s) QUERYABLE via the libs/db CLI under the exact key
        check(db_path.is_file(), "db file written")
        q = subprocess.run(
            [str(DB_CLI), "db-query", str(db_path),
             "--arch", "gfx1101",
             "--family", kernel_hash,
             "--shape", bundle_hash],
            capture_output=True, text=True)
        print(q.stdout, end="")
        check(q.returncode == 0, "db-query under (arch, kernel-name-hash, "
                                 "capture-bundle-hash) exits 0")
        check(f"gpu_arch=gfx1101 kernel_family={kernel_hash} "
              f"shape_hash={bundle_hash}" in q.stdout,
              "db-query reports the exact (arch, kernel-name-hash, bundle-hash) key")
        # the query's minimum-median record == evidence.best (db-query prints
        # %.4f, so compare within the printed precision)
        rec_medians = []
        for line in q.stdout.splitlines():
            if line.startswith("record:"):
                parts = dict(tok.split("=", 1) for tok in line.split(" ")[1:]
                             if "=" in tok)
                rec_medians.append(float(parts["median_ms"]))
        check(rec_medians and abs(min(rec_medians) -
                                  best["timings"]["median"]) < 0.001,
              "db record with the best (min-median) config is queryable")

        # --- 2. INJECTED BROKEN CONFIG ---------------------------------------
        db_path2 = work / "tune-inj.jsonl"
        r = subprocess.run(
            [str(TOOL), "--from-capture", str(BUNDLE),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(BLOB_DIR), "--ref", str(REF),
             "--inject-config", "128,1,1:2048,1,1",
             "--json", "--db", str(db_path2)],
            capture_output=True, text=True)
        print(r.stdout, end="")
        print(r.stderr, end="")
        check(r.returncode == 0, "injected run exits 0 (best still found)")
        ev2 = json.loads(r.stdout[r.stdout.index("{"):]) if r.stdout.strip() else {}
        inj = [c for c in ev2.get("candidates", [])
               if c["config"]["block"] == [2048, 1, 1]]
        check(len(inj) == 1 and inj[0]["status"] == "pre_screen_rejected",
              "injected block-2048 config REJECTED PRE-REPLAY")
        check(inj and "exceeds the launch limit 1024" in inj[0].get("reason", ""),
              "rejection reason names the arch launch limit (1024 threads/block)")
        check(inj and inj[0].get("validation", {}).get("pass") is False,
              "rejected candidate was never replayed (no validation verdict)")
        check(ev2.get("summary", {}).get("pre_screen_rejected", 0) == 1,
              "summary counts the pre-screen rejection")

        # --- 3. QA NEGATIVE: no valid config ----------------------------------
        garbage = work / "garbage.bin"
        garbage.write_bytes(b"\xAB" * 262144)  # cannot match the replayed output
        db_neg = work / "tune-neg.jsonl"
        r = subprocess.run(
            [str(TOOL), "--from-capture", str(BUNDLE),
             "--vendor", "amd", "--arch", "gfx1101",
             "--blob-dir", str(BLOB_DIR), "--ref", str(garbage),
             "--json", "--db", str(db_neg)],
            capture_output=True, text=True)
        print(r.stdout, end="")
        print(r.stderr, end="")
        check(r.returncode == 1, "no-valid-config run exits 1")
        ev3 = json.loads(r.stdout[r.stdout.index("{"):]) if r.stdout.strip() else {}
        check(ev3.get("status") == "no_viable_config",
              "JSON status no_viable_config")
        check("no valid config" in ev3.get("message", ""),
              "message reports 'no valid config'")
        check(ev3.get("summary", {}).get("validation_failed", 0) >= 3,
              "every candidate failed bitwise validation (>= 3 recorded)")
        check(not db_neg.exists(), "NO db write on no-valid-config (asserted)")

    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("integ_capture_tune: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
