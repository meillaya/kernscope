"""kernscope golden harness: NumPy bf16 correctness references + verdicts.

Package layout (tests/golden/kernscope_golden/):

  metrics.py    cosine / max_rel_err / pcc and the pass verdict
  refs.py       reference implementations for the six kernel families
  inputs.py     seeded deterministic input generation + .npz persistence
  __main__.py   CLI (`python3 -m kernscope_golden check|gen`, see its docstring)

CLI entry points (run with cwd = tests/golden; see kernscope_golden.__main__
for why PYTHONPATH must not be used -- the devenv python wrapper drops its
site-packages when the user overrides PYTHONPATH):

    python3 -m kernscope_golden gen --kernel <family> --shape M,N,... \\
        --out ref.npz --seed 0
    python3 -m kernscope_golden check --kernel <family> --bin out.bin \\
        --shape M,N,... [--inputs ref.npz] [--dtype bf16|fp32]

The check prints a JSON verdict with pass/cosine/max_rel_err/pcc and exits
0 (pass) / 1 (fail) / 2 (usage or IO error).

CONSUMERS: the tuner golden gate (todo 28) and capture replay (todo 48)
invoke `check` with a device output buffer; the six families here are the
exact semantic contract for the tuner templates (todos 25-27). Determinism
is guaranteed by fixed-seed generation (inputs.py, DEFAULT_SEED = 0); the
same .npz can be pinned via --inputs for cross-run reproducibility.
"""

from __future__ import annotations

from . import inputs, metrics, refs
from .inputs import DEFAULT_SEED, generate_inputs, reference_output
from .metrics import verdict
from .refs import FAMILIES, REFERENCES

__all__ = [
    "DEFAULT_SEED",
    "FAMILIES",
    "REFERENCES",
    "generate_inputs",
    "inputs",
    "metrics",
    "reference_output",
    "refs",
    "verdict",
]

__version__ = "0.1.0"
