#!/usr/bin/env python3
"""kernscope templates live HIP gate (ctest, LABELS gpu-amd).

Pipeline for ONE template kernel on this machine (gfx1101, todo 24):

    1. `gen` golden inputs with the canonical CLI invocation (cwd MUST be
       tests/golden -- sys.path[0] resolution; do NOT set PYTHONPATH);
    2. dump the npz input tensors to raw row-major .bin buffers;
    3. run the kernel through tools/hip_template_runner.cpp (the
       kernscope::gpuruntime module/kernel/launch/memcpy path);
    4. `check` the produced output .bin against the golden reference.

Exit 0 iff the verdict is pass:true (cosine >= 0.999 AND max_rel_err <= 1e-2
AND pcc >= 0.99 -- todo 23 thresholds). Every failure is loud (non-zero exit
+ message); there is NO silent-skip path.

Golden mapping (documented in README.md): the harness has six families; the
elementwise-add template validates against `reduction` with shape 2,N, whose
reference y[j] = sum_i x[i,j] (axis 0) over a 2-row tensor is exactly
x[0] + x[1] -- the elementwise add of two vectors. matmul (M,N,K) is wired
for the skeleton and the todo 25/27 matmul variants.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def _fail(msg: str) -> int:
    print(f"gate: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="repo root")
    ap.add_argument("--workdir", required=True, help="scratch dir (created)")
    ap.add_argument("--hsaco", required=True, help="hipcc --genco output")
    ap.add_argument("--runner", required=True, help="hip_template_runner binary")
    ap.add_argument("--symbol", required=True, help="kernel symbol in the hsaco")
    ap.add_argument("--family", required=True, choices=("reduction", "matmul"))
    ap.add_argument("--shape", required=True, help="comma list, e.g. 2,1024")
    ap.add_argument("--grid", required=True, help="W or WxH")
    ap.add_argument("--block", required=True, help="X or XxY")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    work = os.path.abspath(args.workdir)
    os.makedirs(work, exist_ok=True)

    if not os.path.exists(args.hsaco):
        # The gate must never silently skip: a missing module is a failure
        # (build the target first: cmake --build build).
        return _fail(f"hsaco missing (run `cmake --build build` first): {args.hsaco}")
    if not os.access(args.runner, os.X_OK):
        return _fail(f"runner missing or not executable: {args.runner}")

    # PYTHONPATH caveat (tests/golden/kernscope_golden/__main__.py): resolve
    # the package explicitly instead of relying on the environment.
    golden_dir = os.path.join(root, "tests", "golden")
    if not os.path.isdir(golden_dir):
        return _fail(f"tests/golden not found under {root}")
    sys.path.insert(0, golden_dir)
    from kernscope_golden import inputs as kinputs  # noqa: E402

    shape = tuple(int(x) for x in args.shape.split(","))
    ref_npz = os.path.join(work, "ref.npz")

    # 1. golden inputs (canonical invocation: cwd == tests/golden).
    gen = subprocess.run(
        [sys.executable, "-m", "kernscope_golden", "gen",
         "--kernel", args.family, "--shape", args.shape, "--out", ref_npz],
        cwd=golden_dir, capture_output=True, text=True,
    )
    if gen.returncode != 0:
        return _fail(f"golden gen failed (rc={gen.returncode}):\n{gen.stderr}")
    print(f"gate: golden inputs -> {ref_npz}")

    arrays = kinputs.load_npz(ref_npz, args.family)
    expected = arrays["expected"]

    # 2. npz tensors -> raw row-major bins; kernel arg layout per family.
    if args.family == "reduction":
        if len(shape) != 2 or shape[0] != 2:
            return _fail(
                f"reduction family requires shape 2,N for the elementwise-add "
                f"template (got {shape})")
        x = arrays["x"]
        a, b = x[0], x[1]
        a_bin, b_bin = os.path.join(work, "a.bin"), os.path.join(work, "b.bin")
        with open(a_bin, "wb") as fh:
            fh.write(a.tobytes())
        with open(b_bin, "wb") as fh:
            fh.write(b.tobytes())
        in_args = ["--in", a_bin, "--in", b_bin]
        int_args = ["--int", str(shape[1])]  # n
    else:  # matmul
        if len(shape) != 3:
            return _fail(f"matmul family requires shape M,N,K (got {shape})")
        m, _n, k = shape
        a_bin, b_bin = os.path.join(work, "a.bin"), os.path.join(work, "b.bin")
        with open(a_bin, "wb") as fh:
            fh.write(arrays["a"].tobytes())
        with open(b_bin, "wb") as fh:
            fh.write(arrays["b"].tobytes())
        in_args = ["--in", a_bin, "--in", b_bin]
        int_args = ["--int", str(m), "--int", str(_n), "--int", str(k)]

    # 3. run the kernel (gpuruntime module/kernel/launch/memcpy path).
    out_bin = os.path.join(work, "out.bin")
    run = subprocess.run(
        [args.runner, "--hsaco", args.hsaco, "--kernel", args.symbol,
         "--grid", args.grid, "--block", args.block,
         *in_args, "--out", out_bin, "--out-size", str(expected.nbytes),
         *int_args],
        capture_output=True, text=True,
    )
    print(run.stdout, end="")
    if run.returncode != 0:
        print(run.stderr, file=sys.stderr, end="")
        return _fail(f"kernel run failed (rc={run.returncode})")

    # 4. golden check (canonical invocation: cwd == tests/golden).
    check = subprocess.run(
        [sys.executable, "-m", "kernscope_golden", "check",
         "--kernel", args.family, "--bin", out_bin,
         "--shape", args.shape, "--inputs", ref_npz],
        cwd=golden_dir, capture_output=True, text=True,
    )
    print(check.stdout, end="")
    if check.returncode != 0:
        if check.stderr:
            print(check.stderr, file=sys.stderr, end="")
        return _fail(f"golden check failed (rc={check.returncode}, verdict not "
                     f"pass:true)")
    verdict = json.loads(check.stdout)
    if verdict["pass"] is not True:
        return _fail(f"verdict pass != true: {verdict}")
    print(f"gate: OK symbol={args.symbol} family={args.family} "
          f"shape={args.shape} verdict=pass:true")
    return 0


if __name__ == "__main__":
    sys.exit(main())
