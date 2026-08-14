"""Correctness metrics for the kernscope golden harness (fresh implementation).

All metrics operate on the FLATTENED, fp32-upcast view of the compared
tensors. The rounding contract (pinned by the plan, todo 23 / polykernel
convention, /home/mei/projects/polykernel/README.md:116-120):

  * inputs are bf16 round-to-nearest-even (ml_dtypes default);
  * all reduction/accumulation happens in fp32;
  * the kernel output is bf16, upcast to fp32 for comparison;
  * a device result is ACCEPTED iff ALL THREE thresholds hold:

        cosine        >= 0.999
        max_rel_err   <= 1e-2     (eps = 1e-6 in the relative-error denom)
        pcc           >= 0.99

THRESHOLD BOUNDARY CONTRACT (documented, tested in test_golden.py):
comparisons are INCLUSIVE (>= / <=). A metric exactly at its threshold
(e.g. cosine == 0.999) counts as a PASS. Rationale: the thresholds are
tolerance bands around an exact bf16-rounded reference; a device kernel
whose metrics land exactly on the boundary is indistinguishable from the
reference within the pinned tolerance, so excluding it would be arbitrary.
"""

from __future__ import annotations

import numpy as np

# Pinned thresholds (do not change without updating the plan / todos 24-28).
COSINE_THRESHOLD = 0.999
MAX_REL_ERR_THRESHOLD = 1e-2
PCC_THRESHOLD = 0.99
REL_ERR_EPS = 1e-6


def _flat_fp32(x: np.ndarray) -> np.ndarray:
    """Flatten and upcast to fp32 -- the comparison view of the rounding contract."""
    return np.asarray(x).astype(np.float32).reshape(-1)


def cosine(a: np.ndarray, ref: np.ndarray) -> float:
    """Cosine similarity over flattened fp32 views: dot(a,b) / (|a|*|b|).

    Degenerate case: both-zero vectors are identical (1.0); exactly one
    zero vector is maximally dissimilar (0.0).
    """
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    denom = nx * ny
    if denom == 0.0:
        return 1.0 if (nx == 0.0 and ny == 0.0) else 0.0
    return float(np.dot(x, y) / denom)


def max_rel_err(a: np.ndarray, ref: np.ndarray, eps: float = REL_ERR_EPS) -> float:
    """max(|a - ref| / (|ref| + eps)) over all elements (flattened fp32).

    eps guards division by zero for elements whose reference is exactly 0;
    a device value of 0 against a reference of 0 therefore contributes 0.
    """
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    return float(np.max(np.abs(x - y) / (np.abs(y) + eps)))


def pcc(a: np.ndarray, ref: np.ndarray) -> float:
    """Pearson correlation coefficient over flattened fp32 views.

    Degenerate case: zero-variance tensors are perfectly correlated when
    elementwise-equal, else uncorrelated (0.0).
    """
    x = _flat_fp32(a)
    y = _flat_fp32(ref)
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc)) * np.sqrt(np.sum(yc * yc)))
    if denom == 0.0:
        return 1.0 if np.array_equal(x, y) else 0.0
    return float(np.sum(xc * yc) / denom)


def metrics(actual: np.ndarray, ref: np.ndarray) -> dict:
    """Compute all three metrics; returns {cosine, max_rel_err, pcc} floats."""
    return {
        "cosine": cosine(actual, ref),
        "max_rel_err": max_rel_err(actual, ref),
        "pcc": pcc(actual, ref),
    }


def verdict(actual: np.ndarray, ref: np.ndarray) -> dict:
    """Full verdict dict: pass flag + all three metrics (always reported).

    pass == True iff cosine >= 0.999 AND max_rel_err <= 1e-2 AND pcc >= 0.99
    (inclusive comparisons -- see module docstring for the boundary contract).
    All three metrics are reported even on failure so a harness consumer can
    always tell WHICH threshold was violated.
    """
    m = metrics(actual, ref)
    ok = (
        m["cosine"] >= COSINE_THRESHOLD
        and m["max_rel_err"] <= MAX_REL_ERR_THRESHOLD
        and m["pcc"] >= PCC_THRESHOLD
    )
    return {
        "pass": bool(ok),
        "cosine": m["cosine"],
        "max_rel_err": m["max_rel_err"],
        "pcc": m["pcc"],
        "thresholds": {
            "cosine": COSINE_THRESHOLD,
            "max_rel_err": MAX_REL_ERR_THRESHOLD,
            "pcc": PCC_THRESHOLD,
        },
    }
