#!/usr/bin/env python3
"""validate_manifest.py — assert tests/fixtures/manifest.json pins the
review-verified golden combo.

Checks (all MUST hold, else exit 1 with a message):
  * occupancy_expected == 0.375 exactly (pinned golden)
  * limiter == "registers"
  * registers_per_thread within [registers_min, registers_max] = [97, 112]
  * warning_ids == ["bank_conflict", "divergent_inner_loop", "coalescing"]
  * suggestion_ids == ["tile_k_32", "split_accumulator", "hoist_predicate"]
  * the derived occupancy arithmetic reproduces 0.375
    (9 blocks x 2 warps / 48 max warps)

Exit 0 on pass, non-zero + message on fail.
Owned by todo 5 (golden fixture materialization).
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

EXPECTED_WARNINGS = ["bank_conflict", "divergent_inner_loop", "coalescing"]
EXPECTED_SUGGESTIONS = ["tile_k_32", "split_accumulator", "hoist_predicate"]
EXPECTED_OCCUPANCY = 0.375
EXPECTED_LIMITER = "registers"
REGS_MIN, REGS_MAX = 97, 112


def fail(msg: str) -> int:
    print(f"[FAIL] validate_manifest: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    if not MANIFEST.is_file():
        return fail(f"{MANIFEST.name} not found")

    m = json.loads(MANIFEST.read_text())

    errors = []

    # --- occupancy pin -----------------------------------------------------
    occ = m.get("occupancy_expected")
    if occ != EXPECTED_OCCUPANCY:
        errors.append(f"occupancy_expected={occ!r}, expected {EXPECTED_OCCUPANCY}")

    # the occupancy must be reproducible from the pinned arithmetic
    try:
        derived = (
            m["occupancy_numerator_blocks"] * m["warps_per_block"]
        ) / m["occupancy_denominator_warps"]
        if abs(derived - EXPECTED_OCCUPANCY) > 1e-12:
            errors.append(
                f"derived occupancy {derived} != {EXPECTED_OCCUPANCY} "
                "from numerator/denominator fields"
            )
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        errors.append(f"occupancy arithmetic fields invalid: {exc}")

    # --- limiter pin -------------------------------------------------------
    if m.get("limiter") != EXPECTED_LIMITER:
        errors.append(f"limiter={m.get('limiter')!r}, expected {EXPECTED_LIMITER!r}")

    # --- register-count pin (97-112) ---------------------------------------
    regs = m.get("registers_per_thread")
    if not isinstance(regs, int) or not (REGS_MIN <= regs <= REGS_MAX):
        errors.append(
            f"registers_per_thread={regs!r} outside [{REGS_MIN}, {REGS_MAX}]"
        )

    # --- warning / suggestion id pins --------------------------------------
    if m.get("warning_ids") != EXPECTED_WARNINGS:
        errors.append(
            f"warning_ids={m.get('warning_ids')!r}, expected {EXPECTED_WARNINGS}"
        )
    if m.get("suggestion_ids") != EXPECTED_SUGGESTIONS:
        errors.append(
            f"suggestion_ids={m.get('suggestion_ids')!r}, "
            f"expected {EXPECTED_SUGGESTIONS}"
        )

    # --- schema + arch/block sanity ----------------------------------------
    if m.get("schema_version") != 1:
        errors.append(f"schema_version={m.get('schema_version')!r}, expected 1")
    if m.get("arch") != "sm_86":
        errors.append(f"arch={m.get('arch')!r}, expected sm_86")
    if m.get("block_size") != 64:
        errors.append(f"block_size={m.get('block_size')!r}, expected 64")

    if errors:
        for e in errors:
            fail(e)
        return 1

    print("[PASS] validate_manifest: manifest.json pins the golden combo "
          f"(occupancy {EXPECTED_OCCUPANCY}, limiter {EXPECTED_LIMITER}, "
          f"{regs} regs in [{REGS_MIN},{REGS_MAX}])")
    return 0


if __name__ == "__main__":
    sys.exit(main())
