# Reproducibility

How to reproduce the whole kernscope plan from zero, pinned to the exact
toolchain, plus the owner-executed NVIDIA rental runbook.

## The master script: `scripts/reproduce-all.sh`

One command reproduces everything:

```bash
devenv shell                            # enter the pinned toolchain
bash scripts/reproduce-all.sh           # lock check -> ci.sh -> demo.sh -> dashboard
```

The script chains four stages and fails fast at the first non-green one:

1. **Lock check** — `devenv.lock` must exist AND be committed. The lock is
   the reproducibility contract: it pins the nix store closure of the
   toolchain (LLVM/MLIR-21, CUDA 12.6 redistributables for sm_80/sm_90,
   ROCm 7.2.3). If it is missing or untracked the script exits 1 with the
   message *"run devenv update to regenerate the lock"* — it NEVER
   regenerates silently. (`devenv.yaml` also carries `require_version:
   ">=2.2"`, which makes devenv itself refuse an older devenv.)
2. **ci.sh** — configure + build (Werror), ctest `no-gpu|gpu-amd` with a
   non-decreasing count guard (`scripts/ci-baseline.txt`), lit/FileCheck
   suite (`check-kernscope`), NumPy golden harness, generated-code compile
   checks, committed-fixture size (≤ 64 MB), and the NVIDIA-header
   vendored-file guards.
3. **demo.sh** — the five-stage end-to-end loop on local gfx1101
   (profile → capture → analyze → tune → verify), asserted under a 900 s
   budget, emitting one aggregated cross-tool report
   (`reports/demo/report.json` + self-contained HTML).
4. **Dashboard** — `kernscope-dashboard` aggregates all committed module
   reports into `reports/dashboard.html` (+ `.json` payload sibling).
   `SOURCE_DATE_EPOCH` defaults to the HEAD commit time (reproducible-builds
   convention), so regenerating on the same commit yields a byte-identical
   dashboard.

### Clean-checkout CI (the plan's final gate)

The CI gate must pass from ZERO build state — no `build/` directory, no
cached artifacts:

```bash
tmpdir=$(mktemp -d)
git clone /home/mei/projects/llvm.h "$tmpdir/repo"
cd "$tmpdir/repo"
bash scripts/ci.sh        # inside `devenv shell`; configures + builds the full tree
```

The configure step builds the full tree (the first clean run is the slowest,
roughly 10 minutes); subsequent runs reuse `build/`. Everything ci.sh needs
is committed: fixtures (≤ 64 MB total), golden inputs, lit tests, corpora —
a fresh clone contains the complete gate.

## Toolchain pinning

| Pin | Where | Enforcement |
|---|---|---|
| devenv >= 2.2 | `devenv.yaml: require_version` | devenv refuses older versions |
| nixpkgs (unstable, locked) | `devenv.lock` | committed; `reproduce-all.sh` lock check |
| CUDA 12.6 redistributables (sm_80, sm_90 only — disk-budget guard) | `devenv.yaml: nixpkgs.cuda_capabilities` | nix closure |
| ROCm 7.2.3 | `devenv.yaml: nixpkgs.rocm_support` | nix closure |
| LLVM/MLIR 21.1.8 | `devenv.yaml` inputs | nix closure; out-of-tree plugin built against it |

Changing the toolchain = editing `devenv.yaml`, running `devenv update` to
regenerate `devenv.lock`, committing BOTH, and re-running
`scripts/reproduce-all.sh`.

## Honesty rules for reports

- Reports carry `schema_version: "1.0"` and per-module
  `provenance.mode` = `measured` (live gfx1101 runs) | `projected` (static
  analysis / IR+PTX compile-only artifacts). `tests/report/check_dashboard.py`
  asserts the honesty flags; projected numbers are never presented as
  measured.
- `reports/rentals/` holds no numbers yet. Rental numbers appear there ONLY
  when the owner produces them (see the runbook below); the README and
  dashboard must not claim them meanwhile.

## Rental runbook (OWNER-executed, never auto-triggered)

NVIDIA-runtime validation runs only on a rented A100/H100-class host, only
by the owner, and only through `scripts/rental-validate.sh`:

```bash
# on the rental host, inside the same devenv shell:
bash scripts/rental-validate.sh --dry-run   # 1. print the exact command sequence
bash scripts/rental-validate.sh             # 2. exec it
```

The script owns the NVIDIA runtime paths: CUPTI capture + replay and CUPTI
profiler tracing are compile+unit tested locally but their live behavior is
validated on the rental. It also detects the CUDA-13 removals the local
build pins against (see the migration surface in `docs/architecture.md`).
Output lands under `reports/rentals/`, which the owner commits — at that
point the README may reference it.
