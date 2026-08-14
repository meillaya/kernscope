"""Seeded, deterministic input generation and shape schemas per kernel family.

Every golden run starts from np.random.default_rng(seed) with a FIXED seed
(default 0) so goldens are bit-reproducible across runs, machines, and the
consumers of this harness (tuner golden gate, todo 28; capture replay,
todo 48). Inputs are generated as fp32 and rounded to bf16
round-to-nearest-even at the reference boundary (the rounding contract), so
the exact same bf16 tensors feed device kernels and references.

SHAPE SCHEMAS (--shape M,N,... on the CLI):

  * matmul     M,N,K   -> a[M,K], b[K,N]                      output [M,N]
  * bias_gelu  R,C     -> x[R,C], bias[C]                     output [R,C]
  * softmax    R,C     -> x[R,C]                              output [R,C]
  * layernorm  R,C     -> x[R,C], gamma[C], beta[C]           output [R,C]
  * reduction  R,C     -> x[R,C]  (reduce axis 0)             output [C]
  * attention  Nq,Nk,d -> q[Nq,d], k[Nk,d], v[Nk,d]           output [Nq,d]

Input distributions are chosen so every regime of each op is exercised:
mixed signs (matmul/reduction/attention), both GELU regimes (bias_gelu),
spread logits (softmax), non-degenerate variance (layernorm gamma > 0).
"""

from __future__ import annotations

import ml_dtypes
import numpy as np

from .refs import bf16, FAMILIES, REFERENCES

_BF16 = ml_dtypes.bfloat16

DEFAULT_SEED = 0

# family -> (len(shape), output_shape_fn(shape))
_SHAPES = {
    "matmul": 3,
    "bias_gelu": 2,
    "softmax": 2,
    "layernorm": 2,
    "reduction": 2,
    "attention": 3,
}


def family_arity(family: str) -> int:
    """Number of shape components a family expects (validates CLI --shape)."""
    return _SHAPES[family]


def output_shape(family: str, shape: tuple) -> tuple:
    """Output tensor shape for a family + input shape (row-major, C order)."""
    if family == "matmul":
        m, _n, _k = shape
        return (m, _n)
    if family in ("bias_gelu", "softmax", "layernorm"):
        return tuple(shape)
    if family == "reduction":
        return (shape[1],)
    if family == "attention":
        return (shape[0], shape[2])
    raise KeyError(family)


def _check_family(family: str) -> None:
    if family not in FAMILIES:
        raise KeyError(
            f"unknown kernel family {family!r}; expected one of {', '.join(FAMILIES)}"
        )


def generate_inputs(family: str, shape: tuple, seed: int = DEFAULT_SEED) -> dict:
    """Generate deterministic bf16 input tensors for `family` (see docstring)."""
    _check_family(family)
    rng = np.random.default_rng(seed)
    if family == "matmul":
        m, n, k = shape
        inputs = {"a": bf16(rng.uniform(-1.0, 1.0, (m, k))),
                  "b": bf16(rng.uniform(-1.0, 1.0, (k, n)))}
    elif family == "bias_gelu":
        r, c = shape
        inputs = {"x": bf16(rng.uniform(-1.0, 1.0, (r, c))),
                  "bias": bf16(rng.uniform(-0.25, 0.25, (c,)))}
    elif family == "softmax":
        r, c = shape
        inputs = {"x": bf16(rng.uniform(-2.0, 2.0, (r, c)))}
    elif family == "layernorm":
        r, c = shape
        inputs = {"x": bf16(rng.uniform(-1.0, 1.0, (r, c))),
                  "gamma": bf16(rng.uniform(0.5, 1.5, (c,))),
                  "beta": bf16(rng.uniform(-0.25, 0.25, (c,)))}
    elif family == "reduction":
        r, c = shape
        inputs = {"x": bf16(rng.uniform(-1.0, 1.0, (r, c)))}
    elif family == "attention":
        nq, nk, d = shape
        # Scale logits toward O(1): q,k uniform in [-1/sqrt(d), 1/sqrt(d)]
        # keeps pre-softmax scores ~ O(1) so exp() is well-conditioned.
        scale = 1.0 / np.sqrt(d)
        inputs = {"q": bf16(rng.uniform(-scale, scale, (nq, d))),
                  "k": bf16(rng.uniform(-scale, scale, (nk, d))),
                  "v": bf16(rng.uniform(-1.0, 1.0, (nk, d)))}
    else:  # pragma: no cover -- guarded by _check_family
        raise KeyError(family)
    return inputs


def reference_output(family: str, inputs: dict) -> np.ndarray:
    """Reference (expected) output array (bf16) for `family` + inputs."""
    _check_family(family)
    return REFERENCES[family](inputs)


def save_npz(path: str, family: str, inputs: dict) -> dict:
    """Persist inputs (+ computed reference output) to an .npz for replay.

    The npz contains every input tensor plus 'expected' (the reference bf16
    output), so a later `check --inputs` run reproduces the exact reference
    without regenerating random state.
    """
    _check_family(family)
    arrays = dict(inputs)
    arrays["expected"] = reference_output(family, inputs)
    np.savez_compressed(path, **arrays)
    return arrays


def load_npz(path: str, family: str) -> dict:
    """Load inputs (+ 'expected') written by save_npz, keyed by family.

    NOTE: numpy's .npz format cannot persist the ml_dtypes bfloat16 type by
    name -- bf16 arrays round-trip as raw 2-byte void arrays (|V2). They are
    restored here with a view to bf16 (byte-identical, no re-rounding), so a
    consumer of `gen`-produced inputs always sees true bf16 tensors.
    """
    _check_family(family)
    out = {}
    with np.load(path, allow_pickle=False) as z:
        for k in z.files:
            arr = z[k]
            if arr.dtype.kind == "V" and arr.dtype.itemsize == 2:
                arr = arr.view(_BF16)
            out[k] = arr
    return out
