#!/usr/bin/env python3
"""check_layernorm.py — build (if needed) + run the layernorm_fused HIP app
and compare its output against the committed NumPy golden.

Thresholds (polykernel convention, /home/mei/projects/polykernel/README.md):
    cosine      >= 0.999
    max rel err <= 1e-2
    PCC         >= 0.99

Exit 0 on pass, non-zero + message on fail.
Owned by todo 5 (golden fixture materialization).

Usage:
    python3 tests/fixtures/layernorm_fused/check_layernorm.py
    env LAYERNORM_APP=/path/to/layernorm_app python3 .../check_layernorm.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent
APP_NAME = "layernorm_app"

COSINE_MIN = 0.999
MAX_REL_ERR = 1.0e-2
PCC_MIN = 0.99


def build_app(workdir: Path) -> Path:
    """Build with hipcc into workdir; honors LAYERNORM_APP override."""
    override = os.environ.get("LAYERNORM_APP")
    if override and Path(override).is_file():
        return Path(override)
    app = workdir / APP_NAME
    if app.is_file():
        return app
    hipcc = shutil.which("hipcc")
    if not hipcc:
        raise SystemExit("[FAIL] check_layernorm: hipcc not found on PATH "
                         "(run inside `devenv shell`)")
    cmd = [hipcc, "--offload-arch=gfx1101", "-std=c++17", "-O3",
           str(HERE / "layernorm.hip"), "-o", str(app)]
    print(f"[INFO] check_layernorm: building: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=workdir, check=True)
    return app


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    denom = np.where(np.abs(b) > 1e-3, np.abs(b), 1e-3)
    max_rel = float(np.max(np.abs(a - b) / denom))
    pcc = float(np.corrcoef(a, b)[0, 1])
    return {"cosine": cos, "max_rel_err": max_rel, "pcc": pcc}


def main() -> int:
    inputs = np.load(HERE / "golden_inputs.npz")
    outputs = np.load(HERE / "golden_outputs.npz")
    x = inputs["X"].astype(np.uint16)
    gamma = inputs["gamma"].astype(np.uint16)
    beta = inputs["beta"].astype(np.uint16)
    y_gold = outputs["Y"]  # fp32 golden (pre-bf16)

    M, N = x.shape[0], x.shape[1]

    with tempfile.TemporaryDirectory(prefix="layernorm_check_") as td:
        work = Path(td)
        in_bin = work / "input.bin"
        with in_bin.open("wb") as f:
            f.write(np.int32(M).tobytes())
            f.write(np.int32(N).tobytes())
            f.write(x.tobytes())
            f.write(gamma.tobytes())
            f.write(beta.tobytes())

        app = build_app(work)
        out_bin = work / "output.bin"
        print(f"[INFO] check_layernorm: running {app.name} "
              f"(M={M}, N={N}, bf16 in/out, fp32 accumulate)")
        subprocess.run([str(app), str(in_bin), str(out_bin)], check=True)

        y_dev_raw = np.frombuffer(out_bin.read_bytes(), dtype=np.uint16)
        if y_dev_raw.size != M * N:
            raise SystemExit(
                f"[FAIL] check_layernorm: app produced {y_dev_raw.size} "
                f"values, expected {M * N}")
        # bf16 -> fp32 (exact), same as bf16_to_fp32 in the kernel
        y_dev = (y_dev_raw.astype(np.uint32) << 16).view(np.float32)
        y_dev = y_dev.reshape(M, N)

    m = metrics(y_dev, y_gold)
    print(f"[INFO] check_layernorm: cosine={m['cosine']:.6f} "
          f"max_rel_err={m['max_rel_err']:.3e} pcc={m['pcc']:.6f}")

    ok = True
    if m["cosine"] < COSINE_MIN:
        print(f"[FAIL] check_layernorm: cosine {m['cosine']:.6f} < "
              f"{COSINE_MIN}", file=sys.stderr)
        ok = False
    if m["max_rel_err"] > MAX_REL_ERR:
        print(f"[FAIL] check_layernorm: max_rel_err {m['max_rel_err']:.3e} > "
              f"{MAX_REL_ERR}", file=sys.stderr)
        ok = False
    if m["pcc"] < PCC_MIN:
        print(f"[FAIL] check_layernorm: pcc {m['pcc']:.6f} < {PCC_MIN}",
              file=sys.stderr)
        ok = False
    if not ok:
        return 1

    print("[PASS] check_layernorm: device output matches golden "
          "(cosine >= 0.999, max_rel_err <= 1e-2, pcc >= 0.99)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
