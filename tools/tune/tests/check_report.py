#!/usr/bin/env python3
"""kernscope tune report check (ctest NAME tune_report, LABELS no-gpu).

Runs the full `kernscope-tune` pipeline for matmul 512^3 on gfx1101 and
asserts the todo-29 numeric report contract on the --json evidence:

  * best.config present with ALL family config keys (the matmul schema:
    tile_m, tile_n, tile_k, warps, stages, vec_width, smem_layout, unroll);
  * roofline.position in (0, 1]  (fraction of the compute roof);
  * roofline.bound in {compute, memory, latency};
  * occupancy in (0, 1] with a non-empty occupancy.limiter;
  * variants array >= 20 (the 24-config auto grid), the measured variants
    sorted by median ascending;
  * speedup >= 1.0 vs the naive baseline (the first enumerated measured
    config), baseline.median present.

Exit 0 on pass; non-zero (with the failing assertion named) on failure.
The tune is tunecache-backed: a warm cache (reports/tune/*.tunecache.json)
replays without recompiling, so a repeat run is fast.
"""

from __future__ import annotations

import json
import subprocess
import sys

MATMUL_CONFIG_KEYS = [
    "tile_m", "tile_n", "tile_k", "warps", "stages", "vec_width",
    "smem_layout", "unroll",
]

BOUND_CLASSES = ("compute", "memory", "latency")


def fail(msg: str) -> int:
    print(f"tune_report: FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else "build/tools/tune/kernscope-tune"
    repo_root = sys.argv[2] if len(sys.argv) > 2 else "."

    cmd = [binary, "--kernel", "matmul", "--shape", "512,512,512",
           "--arch", "gfx1101", "--json"]
    print(f"tune_report: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        return fail(f"kernscope-tune exited {proc.returncode}: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}")
    try:
        ev = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        return fail(f"evidence JSON does not parse: {e}")

    best = ev.get("best")
    if not isinstance(best, dict):
        return fail("best missing from the evidence (no viable config?)")
    best_cfg = best.get("config")
    if not isinstance(best_cfg, dict):
        return fail("best.config missing")
    missing = [k for k in MATMUL_CONFIG_KEYS if k not in best_cfg]
    if missing:
        return fail(f"best.config missing config keys: {missing}")

    rl = ev.get("roofline")
    if not isinstance(rl, dict) or not rl.get("ok"):
        return fail("roofline missing / not ok (peaks must be measured)")
    pos = rl.get("position")
    if not (isinstance(pos, (int, float)) and 0.0 < pos <= 1.0):
        return fail(f"roofline.position {pos!r} not in (0, 1]")
    bound = rl.get("bound")
    if bound not in BOUND_CLASSES:
        return fail(f"roofline.bound {bound!r} not in {BOUND_CLASSES}")

    occ = ev.get("occupancy")
    if not (isinstance(occ, (int, float)) and 0.0 < occ <= 1.0):
        return fail(f"occupancy {occ!r} not in (0, 1]")
    limiter = ev.get("occupancy.limiter")
    if not isinstance(limiter, str) or not limiter:
        return fail("occupancy.limiter missing or empty")

    variants = ev.get("variants")
    if not isinstance(variants, list) or len(variants) < 20:
        return fail(f"variants array has {len(variants) if isinstance(variants, list) else '?'} entries (< 20)")
    measured = [v for v in variants if isinstance(v.get("timings"), dict)]
    if len(measured) < 1:
        return fail("no measured variant with timings")
    medians = [v["timings"]["median"] for v in measured]
    if any(b < a for a, b in zip(medians, medians[1:])):
        return fail("measured variants not sorted by median ascending")
    if any(m <= 0.0 for m in medians):
        return fail("a measured median is <= 0")

    speedup = ev.get("speedup")
    if not (isinstance(speedup, (int, float)) and speedup >= 1.0):
        return fail(f"speedup {speedup!r} < 1.0 vs the naive baseline")
    baseline = ev.get("baseline")
    if not isinstance(baseline, dict) or not isinstance(baseline.get("timings"), dict):
        return fail("baseline missing or malformed")
    base_median = baseline["timings"].get("median")
    if not (isinstance(base_median, (int, float)) and base_median > 0.0):
        return fail(f"baseline median {base_median!r} invalid")

    print(f"tune_report: PASS (best {best_cfg}, median "
          f"{best['timings']['median']:.4f} ms, roofline position {pos:.3f} "
          f"bound {bound}, occupancy {occ:.3f} limiter {limiter}, "
          f"speedup {speedup:.3f}x, {len(measured)} measured of "
          f"{len(variants)} variants)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
