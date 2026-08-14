"""kernscope golden harness CLI.

Usage (cwd MUST be tests/golden -- sys.path[0] resolves the package; the
devenv python wrapper replaces its site-packages when the user overrides
PYTHONPATH, so do NOT set PYTHONPATH):

    cd tests/golden
    python3 -m kernscope_golden check --kernel <family> --bin <out.bin> \\
        --shape M,N,... [--dtype bf16|fp32] [--inputs ref.npz] \\
        [--seed N] [--out verdict.json]

    python3 -m kernscope_golden gen --kernel <family> --shape M,N,... \\
        --out ref.npz [--seed N]

`check` reads a device-produced output .bin (raw row-major bytes, bf16 by
default or fp32 with --dtype fp32), computes the reference from generated
(seeded, deterministic) inputs -- or from --inputs <ref.npz> when given --
and prints a JSON verdict {pass, cosine, max_rel_err, pcc, thresholds, ...}.
Exit code: 0 == pass, 1 == verdict fail, 2 == usage/IO error.

`gen` writes an .npz with the input tensors AND the reference output
(key 'expected'), enabling reproducible kernel runs followed by a check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import ml_dtypes
import numpy as np

from . import metrics, refs
from .inputs import (
    DEFAULT_SEED,
    family_arity,
    generate_inputs,
    load_npz,
    output_shape,
    reference_output,
    save_npz,
)

_BF16 = ml_dtypes.bfloat16
_F32 = np.float32


def _parse_shape(values) -> tuple:
    """Accept --shape M,N,K (comma) and/or repeated --shape M --shape N."""
    parts = []
    for v in values:
        for chunk in v.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts.append(int(chunk))
    if not parts:
        raise argparse.ArgumentTypeError("--shape requires at least one integer")
    return tuple(parts)


def _read_bin(path: str, shape: tuple, dtype: str) -> np.ndarray:
    """Load a raw row-major device output buffer, checking element count."""
    nbytes = os.path.getsize(path)
    if dtype == "bf16":
        arr = np.fromfile(path, dtype=_BF16)
        itemsize = 2
    elif dtype == "fp32":
        arr = np.fromfile(path, dtype=_F32)
        itemsize = 4
    else:  # pragma: no cover -- argparse restricts the choice
        raise ValueError(dtype)
    expected = math.prod(shape)
    if arr.size != expected:
        raise ValueError(
            f"{path}: {nbytes} bytes / {arr.size} elements (itemsize {itemsize}) "
            f"does not match --shape {shape} -> {expected} elements"
        )
    return arr.reshape(shape)


def _cmd_check(args) -> int:
    shape = _parse_shape(args.shape)
    if family_arity(args.kernel) != len(shape):
        raise ValueError(
            f"kernel {args.kernel!r} expects {family_arity(args.kernel)} "
            f"shape components (got {len(shape)} from --shape {shape})"
        )
    if args.inputs is not None:
        inputs = load_npz(args.inputs, args.kernel)
    else:
        inputs = generate_inputs(args.kernel, shape, seed=args.seed)
    ref = reference_output(args.kernel, inputs, causal=args.causal)
    actual = _read_bin(args.bin, output_shape(args.kernel, shape), args.dtype)

    v = metrics.verdict(actual, ref)
    v.update(
        kernel=args.kernel,
        shape=list(shape),
        dtype=args.dtype,
        bin=os.path.abspath(args.bin),
        ref_inputs=os.path.abspath(args.inputs) if args.inputs else None,
        seed=args.seed if args.inputs is None else None,
    )
    # Canonical field order: pass, cosine, max_rel_err, pcc, then context.
    ordered = {
        "pass": v["pass"],
        "cosine": v["cosine"],
        "max_rel_err": v["max_rel_err"],
        "pcc": v["pcc"],
        "kernel": v["kernel"],
        "shape": v["shape"],
        "dtype": v["dtype"],
        "thresholds": v["thresholds"],
        "bin": v["bin"],
        "ref_inputs": v["ref_inputs"],
        "seed": v["seed"],
    }
    print(json.dumps(ordered, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(ordered, fh, indent=2)
    return 0 if v["pass"] else 1


def _cmd_gen(args) -> int:
    shape = _parse_shape(args.shape)
    if family_arity(args.kernel) != len(shape):
        raise ValueError(
            f"kernel {args.kernel!r} expects {family_arity(args.kernel)} "
            f"shape components (got {len(shape)} from --shape {shape})"
        )
    inputs = generate_inputs(args.kernel, shape, seed=args.seed)
    arrays = save_npz(args.out, args.kernel, inputs, causal=args.causal)
    print(
        json.dumps(
            {
                "ok": True,
                "kernel": args.kernel,
                "shape": list(shape),
                "seed": args.seed,
                "causal": args.causal,
                "out": os.path.abspath(args.out),
                "arrays": sorted(arrays),
            },
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="kernscope_golden",
        description="NumPy bf16 correctness harness (reference outputs + "
        "device-buffer verdicts) for the six kernscope kernel families.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    ck = sub.add_parser("check", help="compare a device output .bin against the reference")
    ck.add_argument("--kernel", required=True, choices=refs.FAMILIES)
    ck.add_argument("--bin", required=True, help="device output buffer (.bin, row-major)")
    ck.add_argument("--shape", action="append", required=True, metavar="M,N,...")
    ck.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    ck.add_argument("--inputs", default=None, metavar="ref.npz",
                    help="reuse inputs from a gen-produced .npz (else seeded generation)")
    ck.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ck.add_argument("--causal", action="store_true",
                    help="attention family: causal-masked reference (todo 27)")
    ck.add_argument("--out", default=None, metavar="verdict.json",
                    help="also write the verdict JSON to a file")
    ck.set_defaults(func=_cmd_check)

    gn = sub.add_parser("gen", help="write seeded inputs + reference output to an .npz")
    gn.add_argument("--kernel", required=True, choices=refs.FAMILIES)
    gn.add_argument("--shape", action="append", required=True, metavar="M,N,...")
    gn.add_argument("--out", required=True, metavar="ref.npz")
    gn.add_argument("--seed", type=int, default=DEFAULT_SEED)
    gn.add_argument("--causal", action="store_true",
                    help="attention family: write the causal-masked reference "
                         "(todo 27)")
    gn.set_defaults(func=_cmd_gen)
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, KeyError, OSError) as exc:
        print(f"kernscope_golden: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
