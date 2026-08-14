"""Reference implementations for the six kernscope kernel families.

ROUNDING CONTRACT (todo 23, polykernel convention):
  1. every input/weight is rounded to bf16 round-to-nearest-even first
     (ml_dtypes astype default -- NumPy has no native bfloat16);
  2. all accumulation / reduction happens in fp32 (upcast before computing);
  3. the OUTPUT of each reference is bf16 (so chained ops see bf16 inputs);
  4. comparisons are made on the final bf16 output upcast to fp32 (metrics.py).

The six families (must match the tuner templates, todos 25-27):

  * matmul         C[M,N] = A[M,K] @ B[K,N]                (fp32 accumulate)
  * bias_gelu      y = GELU(x + bias)                      (fused, fp32 add)
  * softmax        y = softmax(x) over the last axis       (stable, fp32)
  * layernorm      y = (x-mean)/sqrt(var+eps)*gamma+beta   (over last axis)
  * reduction      y[j] = sum_i x[i,j]                     (reduce axis 0)
  * attention      O = softmax(Q K^T / sqrt(d)) V          (FlashAttention-lite)

FlashAttention-lite convention: the reference is the ONLINE-SOFTMAX-FREE,
plain mathematical attention with the standard 1/sqrt(d) scaling factor of
the FlashAttention algorithm (Dao et al., "FlashAttention: Fast and
Memory-Efficient Exact Attention with IO-Awareness", 2022 -- the FLOPs
formula 4*N^2*d wired into the roofline entry covers exactly this scaling).
The tuner's tiled/online-softmax kernel (todo 27) is validated against THIS
reference. bf16 inputs Q/K/V, fp32 accumulate, bf16 output O.

GELU uses the exact erf definition 0.5*x*(1+erf(x/sqrt(2))); NumPy ships no
erf, so math.erf (C libm, double precision) is used via np.vectorize and the
result is downcast to fp32 -- the fusion point (bias add) stays fp32.
"""

from __future__ import annotations

import math

import ml_dtypes
import numpy as np

_BF16 = ml_dtypes.bfloat16
_F32 = np.float32
_INV_SQRT2 = 1.0 / math.sqrt(2.0)

# Vectorized exact erf over float64 (C libm), downcast after use.
_verf = np.vectorize(math.erf, otypes=[np.float64])


def bf16(x: np.ndarray) -> np.ndarray:
    """Round to bf16 round-to-nearest-even (ml_dtypes default rounding)."""
    return np.asarray(x).astype(_BF16)


def f32(x: np.ndarray) -> np.ndarray:
    """Upcast to fp32 for accumulation."""
    return np.asarray(x).astype(_F32)


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C[M,N] = A[M,K] @ B[K,N]; bf16 inputs, fp32 accumulate, bf16 output."""
    return bf16(np.matmul(f32(a), f32(b)))


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact erf-based GELU: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    xf = f32(x)
    out = 0.5 * xf * (1.0 + _verf(xf * _INV_SQRT2).astype(_F32))
    return bf16(out)


def bias_gelu(x: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Fused bias + GELU: y = GELU(x + bias), bias broadcast over last axis.

    The fusion point (bias add) is performed in fp32 after both operands are
    rounded to bf16; GELU's erf is computed at double precision and the fused
    result is rounded to bf16 exactly once.
    """
    xf = f32(x)  # x already bf16 (rounded on entry by the caller contract)
    bf = f32(bias)
    return gelu(bf16(xf + bf))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable row softmax over the last axis (fp32, bf16 out)."""
    xf = f32(x)
    shifted = xf - np.max(xf, axis=-1, keepdims=True)
    e = np.exp(shifted)
    return bf16(e / np.sum(e, axis=-1, keepdims=True))


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer norm over the last axis: (x-mean)/sqrt(var+eps)*gamma+beta.

    Variance is the biased population variance (E[(x-E[x])^2]), matching the
    two-pass formula a device kernel computes; eps is pinned to 1e-5 (the
    convention used by the tuner templates, todos 25-27).
    """
    xf = f32(x)
    mean = xf.mean(axis=-1, keepdims=True)
    var = ((xf - mean) ** 2).mean(axis=-1, keepdims=True)
    out = (xf - mean) / np.sqrt(var + eps) * f32(gamma) + f32(beta)
    return bf16(out)


def reduction(x: np.ndarray) -> np.ndarray:
    """Sum reduction over axis 0: y[j] = sum_i x[i,j] (fp32 accumulate).

    Matches a device kernel reducing the leading axis (e.g. one thread per
    column); summation order differs per implementation, which is exactly
    what the cosine / rel-err / PCC tolerances absorb.
    """
    return bf16(np.sum(f32(x), axis=0))


def attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """FlashAttention-lite forward: O = softmax(Q K^T / sqrt(d)) V.

    Q [Nq, d], K [Nk, d], V [Nk, d] -> O [Nq, d]. bf16 inputs, fp32
    accumulate (scores, exp/softmax, and the p@v matmul all in fp32),
    bf16 output. Includes the standard 1/sqrt(d) scaling (see module
    docstring for the convention citation).
    """
    qf = f32(q)
    kf = f32(k)
    vf = f32(v)
    d = qf.shape[-1]
    scores = np.matmul(qf, kf.T) / math.sqrt(d)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    e = np.exp(shifted)
    p = e / np.sum(e, axis=-1, keepdims=True)
    return bf16(np.matmul(p, vf))


# Public per-family reference: name -> callable(inputs: dict) -> output array.
REFERENCES = {
    "matmul": lambda inp: matmul(inp["a"], inp["b"]),
    "bias_gelu": lambda inp: bias_gelu(inp["x"], inp["bias"]),
    "softmax": lambda inp: softmax(inp["x"]),
    "layernorm": lambda inp: layernorm(inp["x"], inp["gamma"], inp["beta"]),
    "reduction": lambda inp: reduction(inp["x"]),
    "attention": lambda inp: attention(inp["q"], inp["k"], inp["v"]),
}

# All six families, in canonical order (used by the CLI and tests).
FAMILIES = tuple(REFERENCES)
