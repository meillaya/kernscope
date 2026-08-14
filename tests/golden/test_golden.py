"""pytest suite for the kernscope golden harness (todo 23).

Covers, for all six kernel families with small shapes:
  * reference produces deterministic expected outputs (bf16, right shape);
  * a CORRECT device-output bin -> verdict pass == true with all three
    metrics reported;
  * a deliberately-wrong (sign-bit-flipped) output bin -> verdict
    pass == false with all three metrics reported (the
    --inject-failed-correctness discipline);
  * seeded determinism (same seed -> bit-identical inputs; different seed
    -> different inputs);
  * .npz round-trip (gen -> check --inputs reproducibility);
  * threshold boundary behavior (inclusive >= / <= comparisons, documented
    in kernscope_golden/metrics.py);
  * CLI error path (wrong-size bin -> exit 2, no crash).
"""

from __future__ import annotations

import json

import ml_dtypes
import numpy as np
import pytest

from kernscope_golden import inputs as kinputs
from kernscope_golden import metrics, refs
from kernscope_golden.__main__ import main

_BF16 = ml_dtypes.bfloat16

# Small shapes per family (plan: "all six families with small shapes").
FAMILY_SHAPES = {
    "matmul": (16, 16, 16),
    "bias_gelu": (16, 32),
    "softmax": (16, 32),
    "layernorm": (16, 32),
    "reduction": (32, 16),
    "attention": (8, 8, 16),
}


def corrupt_bin(data: bytes) -> bytes:
    """Flip the sign bit of every other element (a wrong-output injection)."""
    arr = np.frombuffer(bytearray(data), dtype=_BF16).copy()
    for i in range(0, arr.size, 2):
        arr[i] = _BF16(-float(arr[i]))
    return arr.tobytes()


@pytest.fixture()
def refs_npz(tmp_path, request):
    """Generate inputs + reference output for a family, persisted to .npz."""
    family = request.param
    shape = FAMILY_SHAPES[family]
    npz = tmp_path / f"{family}.npz"
    arrays = kinputs.save_npz(str(npz), family, kinputs.generate_inputs(family, shape))
    return family, shape, npz, arrays


@pytest.mark.parametrize("refs_npz", FAMILY_SHAPES, indirect=True)
def test_reference_dtype_and_shape(refs_npz):
    """Reference outputs are bf16 with the documented output shape."""
    family, shape, _npz, arrays = refs_npz
    expected = arrays["expected"]
    assert expected.dtype == _BF16
    assert expected.shape == kinputs.output_shape(family, shape)


@pytest.mark.parametrize("refs_npz", FAMILY_SHAPES, indirect=True)
def test_correct_output_bin_passes(refs_npz, capsys, tmp_path):
    """A correct device-output bin -> verdict pass == true + all 3 metrics."""
    family, shape, _npz, arrays = refs_npz
    binpath = tmp_path / f"{family}.bin"
    binpath.write_bytes(arrays["expected"].tobytes())

    rc = main(
        ["check", "--kernel", family, "--bin", str(binpath),
         "--shape", ",".join(map(str, shape))]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["pass"] is True
    assert out["cosine"] >= metrics.COSINE_THRESHOLD
    assert out["max_rel_err"] <= metrics.MAX_REL_ERR_THRESHOLD
    assert out["pcc"] >= metrics.PCC_THRESHOLD


@pytest.mark.parametrize("refs_npz", FAMILY_SHAPES, indirect=True)
def test_wrong_output_bin_fails_with_metrics(refs_npz, capsys, tmp_path):
    """A wrong (sign-flipped) bin -> pass == false, all 3 metrics reported.

    Mirrors polykernel's --inject-failed-correctness discipline: the harness
    must report a failure verdict with the metric values, not crash.
    """
    family, shape, _npz, arrays = refs_npz
    binpath = tmp_path / f"{family}.wrong.bin"
    binpath.write_bytes(corrupt_bin(arrays["expected"].tobytes()))

    rc = main(
        ["check", "--kernel", family, "--bin", str(binpath),
         "--shape", ",".join(map(str, shape))]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 1
    assert out["pass"] is False
    for key in ("cosine", "max_rel_err", "pcc"):
        assert key in out, f"metric {key} must be reported on failure"
    assert out["cosine"] < metrics.COSINE_THRESHOLD


def test_matmul_nonsquare_shape():
    """Matmul handles non-square M/N/K (device kernels will vary shapes)."""
    shape = (8, 12, 16)
    arr = kinputs.generate_inputs("matmul", shape)
    expected = kinputs.reference_output("matmul", arr)
    assert expected.shape == (8, 12)
    assert expected.dtype == _BF16
    # Reference is exactly reproducible.
    again = kinputs.reference_output("matmul", arr)
    assert np.array_equal(expected, again)


def test_deterministic_seeding():
    """Same seed -> bit-identical inputs; different seed -> different inputs."""
    shape = (16, 16, 16)
    a1 = kinputs.generate_inputs("matmul", shape, seed=0)
    a2 = kinputs.generate_inputs("matmul", shape, seed=0)
    a3 = kinputs.generate_inputs("matmul", shape, seed=1)
    assert all(np.array_equal(a1[k], a2[k]) for k in a1)
    assert not all(np.array_equal(a1[k], a3[k]) for k in a1)


@pytest.mark.parametrize("refs_npz", FAMILY_SHAPES, indirect=True)
def test_npz_roundtrip_check_with_inputs(refs_npz, capsys, tmp_path):
    """gen .npz -> check --inputs reproduces the same reference (todo 28/48)."""
    family, shape, npz, arrays = refs_npz
    binpath = tmp_path / f"{family}.bin"
    binpath.write_bytes(arrays["expected"].tobytes())

    rc = main(
        ["check", "--kernel", family, "--bin", str(binpath),
         "--shape", ",".join(map(str, shape)), "--inputs", str(npz)]
    )
    out = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert out["pass"] is True
    assert out["ref_inputs"] == str(npz.resolve())


def test_threshold_boundary_inclusive(monkeypatch):
    """Exactly-at-threshold metrics PASS (inclusive >= / <= comparisons).

    Documented in kernscope_golden/metrics.py: a metric exactly at its
    pinned threshold (cosine == 0.999, max_rel_err == 1e-2, pcc == 0.99)
    counts as a pass; anything strictly worse fails. The predicate is what
    this pins -- the metric VALUES at the boundary follow from the formulas.
    """
    # Scale case first (uses the real metric math): scaling changes
    # max_rel_err but not cosine/pcc.
    ref = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    ok = metrics.verdict(ref * 1.005, ref)
    assert ok["pass"] is True
    assert ok["cosine"] == pytest.approx(1.0)
    bad = metrics.verdict(ref * 1.02, ref)
    assert bad["pass"] is False
    assert bad["max_rel_err"] > metrics.MAX_REL_ERR_THRESHOLD

    # Predicate at exactly-at-threshold values (inclusive) vs just-below.
    exactly_at = {"cosine": 0.999, "max_rel_err": 1e-2, "pcc": 0.99}
    monkeypatch.setattr(metrics, "metrics", lambda a, r: dict(exactly_at))
    v = metrics.verdict(None, None)
    assert v["pass"] is True

    just_below = {"cosine": 0.998999, "max_rel_err": 1e-2, "pcc": 0.99}
    monkeypatch.setattr(metrics, "metrics", lambda a, r: dict(just_below))
    assert metrics.verdict(None, None)["pass"] is False


def test_wrong_size_bin_exits_2(capsys, tmp_path):
    """Wrong-size output bin -> clear error, exit code 2 (no crash)."""
    binpath = tmp_path / "short.bin"
    binpath.write_bytes(b"\x00" * 8)  # 4 elements != 16
    rc = main(
        ["check", "--kernel", "matmul", "--bin", str(binpath),
         "--shape", "4,4,4"]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "does not match --shape" in err


def test_all_six_families_registered():
    """The canonical family list matches the tuner template contract."""
    assert tuple(refs.FAMILIES) == (
        "matmul", "bias_gelu", "softmax", "layernorm", "reduction", "attention",
    )
    assert set(refs.REFERENCES) == set(FAMILY_SHAPES)
