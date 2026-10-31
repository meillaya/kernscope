#!/usr/bin/env python3
"""check_profile.py — todo-52 golden helper: assert a kernscope-profile --json
report NUMERICALLY against tests/fixtures/expected_layernorm_profile.json.

The kernscope-profile CLI (tools/profile, todo 52) traces the todo-5
layernorm_fused fixture on gfx1101 (trace -> enrich -> rank -> diagnose ->
report); this helper is the golden's assertion side, following the todo-5
manifest conventions (validate_manifest.py): the expected file pins RANGES,
never absolute durations (machine-dependent).

Usage (the acceptance command):

    kernscope-profile build/tests/fixtures/layernorm_fused --json \
        | python3 tests/fixtures/check_profile.py

Reads the report JSON from stdin, loads the expected-range manifest (sibling
file, like validate_manifest.py) and asserts:

  * report shape: schema_version, report_type kernscope_profile_report,
    no_kernels_traced == false, kernel_count >= 1, wall_time_ns > 0;
  * the layernorm kernel record: name CONTAINS "layernorm", duration_ns > 0,
    occupancy in (0,1] (a number, never "unknown-resources"), limiter in
    {registers, shared, threads, blocks}, registers in [min,max],
    shared_bytes >= 1024, blocks_per_sm >= 1, grid/workgroup matching the
    fixture's launch contract [1024,1,1] / [128,1,1];
  * >= 1 compiler hint, every hint carrying a non-empty evidence map (a hint
    NEVER fires without evidence — todo-51 contract);
  * the owner-style showcase (owner_example): kernel contains "layernorm",
    time_us > 0, registers/shared present — Section B shape
    (docs/report/kernscope-technical-report.pdf) with locally-measured values.

Exit codes: 0 = all numeric/range assertions passed, 1 = a check failed,
2 = IO/parse failure.

Usage: check_profile.py [expected.json]   (default: expected_layernorm_profile.json
next to this script)
"""

import json
import sys
from pathlib import Path

EXPECTED_DEFAULT = Path(__file__).parent / "expected_layernorm_profile.json"


failures: list = []


def check(cond: bool, what: str, f: list = None) -> None:
    f = failures if f is None else f
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        f.append(what)


def main() -> int:
    expected_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EXPECTED_DEFAULT
    try:
        expected = json.loads(expected_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        print(f"[FAIL] cannot load expected ranges {expected_path}: {e}")
        return 2

    try:
        report = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"[FAIL] report JSON does not parse: {e}")
        return 2

    # --- report shape --------------------------------------------------------
    check(report.get("schema_version") == expected.get("schema_version", "1.0")
          or report.get("schema_version") == "1.0",
          "schema_version present")
    check(report.get("report_type") == "kernscope_profile_report",
          "report_type == kernscope_profile_report")
    check(report.get("no_kernels_traced") is False,
          "no_kernels_traced == false (kernels were traced)")
    check(report.get("kernel_count", 0) >= 1, "kernel_count >= 1")
    check(report.get("wall_time_ns", 0) >= expected.get("wall_time_ns_min", 1),
          "wall_time_ns >= 1")

    kernels = report.get("kernels", [])
    check(len(kernels) >= 1, "kernels table non-empty")

    # --- the layernorm kernel record ------------------------------------------
    target = None
    for k in kernels:
        if expected["kernel_name_contains"] in k.get("kernel_name", ""):
            target = k
            break
    check(target is not None,
          f"a kernel record whose name contains "
          f"'{expected['kernel_name_contains']}'")
    if target is None:
        return 1

    print(f"[INFO] layernorm kernel: {target.get('kernel_name')}")

    check(target.get("duration_ns", 0) >= expected.get("duration_ns_min", 1),
          f"duration_ns >= {expected.get('duration_ns_min', 1)} "
          f"(got {target.get('duration_ns')})")

    occ = target.get("occupancy")
    check(isinstance(occ, (int, float)) and not isinstance(occ, bool)
          and expected["occupancy_min_exclusive"] < occ <= expected["occupancy_max_inclusive"],
          f"occupancy in ({expected['occupancy_min_exclusive']}, "
          f"{expected['occupancy_max_inclusive']}] (got {occ!r})")

    lim = target.get("limiter", "")
    check(lim in expected["limiter_set"],
          f"limiter in {expected['limiter_set']} (got '{lim}')")

    regs = target.get("registers_per_thread", 0)
    check(expected["registers_min"] <= regs <= expected["registers_max"],
          f"registers_per_thread in "
          f"[{expected['registers_min']},{expected['registers_max']}] "
          f"(got {regs})")

    shared = target.get("shared_bytes", 0)
    check(shared >= expected["shared_bytes_min"],
          f"shared_bytes >= {expected['shared_bytes_min']} "
          f"(got {shared})")

    check(target.get("blocks_per_sm", 0) >= expected["blocks_per_sm_min"],
          "blocks_per_sm >= 1")

    check(target.get("grid") == expected["grid"],
          f"grid == {expected['grid']} (got {target.get('grid')})")
    check(target.get("workgroup") == expected["workgroup"],
          f"workgroup == {expected['workgroup']} (got {target.get('workgroup')})")

    # --- compiler hints with evidence (todo-51 contract) ----------------------
    hints = target.get("hints", [])
    check(len(hints) >= expected.get("min_compiler_hints", 1),
          f">= {expected.get('min_compiler_hints', 1)} compiler hint(s) "
          f"(got {len(hints)})")
    for h in hints:
        check(bool(h.get("id")) and bool(h.get("severity")),
              f"hint '{h.get('id')}' carries id + severity")
        check(bool(h.get("evidence")),
              f"hint '{h.get('id')}' carries evidence "
              f"(never a hint without evidence)")

    # --- owner-style showcase (Section B shape, locally measured) --------------
    owner = report.get("owner_example")
    check(owner is not None, "owner_example block present")
    if owner is not None:
        check(expected["kernel_name_contains"] in owner.get("kernel", ""),
              "owner_example.kernel contains 'layernorm'")
        check(owner.get("time_us", 0) > 0, "owner_example.time_us > 0")
        check(owner.get("registers_per_thread", 0) > 0,
              "owner_example.registers_per_thread > 0")

    # --- ranking ---------------------------------------------------------------
    ranking = report.get("ranking", [])
    check(len(ranking) >= expected.get("min_hot_kernels", 1),
          "hot ranking non-empty")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] check_profile: layernorm_fused profile report matches "
          "the expected RANGES (durations asserted as ranges, never "
          "absolutes); occupancy/limiter/registers/shared within contract; "
          ">= 1 compiler hint with evidence")
    return 0


if __name__ == "__main__":
    sys.exit(main())
