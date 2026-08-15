# kernscope — Comprehensive Research Report

This directory contains the research report for **kernscope**, a unified C++ GPU-kernel
tooling suite (static analyzer, autotuner, verifier, capture/replay, LLVM passes,
profiler — six CLIs sharing one library stack), grounded in what was actually built in
this repository.

## Abstract

The report (18 pages, 3 hand-drawn TikZ diagrams, 70+ references) describes how the six
tools form one closed loop — profile → capture → analyze → transform/generate → verify →
benchmark — with cross-tool edges that make the tools validate each other: the profiler's
memory-bound hint carries `analyzer_cross_refs` to analyzer findings computed on the
*captured* kernel's ISA; the tuner gates every variant through the verifier before
compiling; the LLVM passes measure before/after deltas with the analyzer's own metrics.
All measured numbers are real, provenance-labeled, and local to an AMD Radeon RX 7800 XT
(gfx1101): measured peaks of 591 GB/s DRAM read bandwidth, 16.32 TFLOPS fp32, and
9.7 TB/s LDS; best committed matmul tune variant 0.4153 ms (ledger generation 0.2027 ms,
4.2× speedup); capture/replay bitwise PASS on the layernorm hot kernel; 34-entry /
19-file / 8-check-class verifier corpus; vector-width 1→4 pass delta with 0→12
vectorized loads and −48 uncoalesced accesses. No NVIDIA runtime numbers exist anywhere;
CUDA paths are compile-only and NVIDIA validation is owner-gated via
`scripts/rental-validate.sh`.

## How to compile

```bash
# pdflatex is not on PATH; use the nix texlive toolchain:
nix shell nixpkgs#texlive.combined.scheme-small --command bash -c \
  'pdflatex -interaction=nonstopmode -halt-on-error report.tex && \
   bibtex report && \
   pdflatex -interaction=nonstopmode -halt-on-error report.tex && \
   pdflatex -interaction=nonstopmode -halt-on-error report.tex'
```

The compiled PDF (18 pages, `report.pdf`) is committed. Verification of a successful
compile: `grep -i 'error' report.log` is empty (only Overfull-hbox/Underfull warnings
may appear, currently none) and `grep 'Output written' report.log` reports
`report.pdf (18 pages)`.

> If `nix shell nixpkgs#texlive.combined.scheme-small` does not resolve, invoke the
> store binaries directly:
> `/nix/store/<hash>-texlive-combined-small-2025-final/bin/{pdflatex,bibtex}` (the store
> path of the built derivation). The report needs: tikz/pgf, natbib (plainnat), booktabs,
> tabularx, hyperref, graphicx, caption, microtype. `enumitem` and `subcaption` are NOT
> required (deliberately not used).

## Files

| File | Contents |
|---|---|
| `report.tex` | Full LaTeX source (article class, 10pt, 18 pages) |
| `report.bib` | Bibliography (70+ entries, bibtex/plainnat) |
| `report.pdf` | Compiled PDF (18 pages, zero errors) |
| `figures/loop.tikz` | The closed tool loop diagram |
| `figures/libgraph.tikz` | libs/ → tools/ dependency graph |
| `figures/dualpath.tikz` | CUDA-compile-only vs HIP-live dual-path strategy |

## Citation-verification status

All citations were mined from the project's own research briefs
(`docs/research/*.md`) and/or verified externally (web) at write time (2026-08-15):

- **Verified externally (web)**: FlashAttention (arXiv:2205.14135), FlashDecoding++
  (arXiv:2311.01282), TVM (arXiv:1802.04799), Ansor (arXiv:2006.06762), Halide
  (PLDI'13 DOI 10.1145/2491956.2462176), Triton (MAPL'19 DOI 10.1145/3315508.3329973),
  SCALE-Sim (arXiv:1811.02883), GPUVerify OOPSLA'12 (DOI 10.1145/2384616.2384625) and
  TOPLAS'15 (DOI 10.1145/2743017), Kerncap (arXiv:2605.03208 — *Kerncap: Automated
  Kernel Extraction and Isolation for AMD GPUs*, Ramos & Lowery, AMD Research),
  CUDAsmith (COMPSAC 2020, no arXiv exists).
- **Taken as-is from the repo's own briefs** (project-internal provenance, not
  independently re-verified): R3 (arXiv:2604.11109), HGRD (arXiv:2604.02106),
  MODEL26 (arXiv:2605.04178), HiRace (arXiv:2401.04701), Tally (arXiv:2410.07381),
  SIP (arXiv:2403.16863), KPerfIR (arXiv:2505.21661), Kernel Tuner
  (arXiv:2407.11488), plus all DOI-only entries (PUG, SESA, GKLEE, GRace, GMRace, CURD,
  BARRACUDA, ScoRD, iGUARD, LLOV, Faial, ESBMC-GPU, PhASAR, SVF, etc. — these DOIs are
  quoted verbatim from the briefs).
- **UNVERIFIED**: `gist-unverified` — "Gist (CUDA kernel instantiator)" was requested
  as a reference but no authoritative paper/arXiv/DOI could be located by web search;
  it is marked `% UNVERIFIED` in `report.bib` and is not cited in the report body.
  Triton's arXiv:2310.10564 was also requested but could not be confirmed and was
  therefore cited via its confirmed MAPL'19 DOI instead.
- All 70+ `\citep`/`\citet` keys in `report.tex` resolve (bibtex exit 0, zero warnings,
  zero undefined citations in the final log).

## Measured numbers cited (with provenance)

Every number in the report is quoted verbatim from a committed artifact:

| Number | Source |
|---|---|
| 591 GB/s DRAM read, 16.32 TF fp32, 9.7 TB/s LDS, 476 GB/s D2D | `reports/peaks/gfx1101-bench.jsonl` (kernscope-bench 0.1.0, measured 2026-08-14, RX 7800 XT) + `libs/arch/include/kernscope/arch.h` + `libs/roofline/src/roofline.cpp` |
| vector width 1→4, vectorized loads 0→12, uncoalesced 64→16 | `reports/passes/layernorm/report.json`, `coalesce_before/after.txt`, `pass_reports.txt`, `reports/demo/stage-5-pass-delta.json` |
| 5-stage loop, memory_bound hint (AI 1.375, ridge 27.61), capture bitwise PASS (sha256 a0027ed9…), best capture-tune 0.011880 ms, verify gate (1 gated / 12 compiled) | `reports/demo/report.json`, `stage-1-profile.json`, `stage-4-capture-tune.json`, `stage-4-tune.jsonl`, `stage-5-verify-tune.json` |
| matmul best 0.4153 ms (15 variants), pre-screen rejections, occupancy 0.03125–0.875 | `reports/tune/matmul-fnv1a64:e93d312338ce8fd7.jsonl(.tunecache.json)` |
| matmul 0.2027 ms / 4.2× / occupancy 0.875 (earlier generation) | `.omo/evidence/task-29-gpu-kernel-suite.txt` + `.omo/start-work/ledger.jsonl` (quoted as such — not averaged with the committed JSONL) |
| fused_attention occupancy 0.375 @ sm_86/64-thread, limiter registers, 102 regs/thread | `reports/dashboard.json` + `tests/fixtures/expected_fused_attention.json` |
| verify corpus: 34 entries, 19 files, 8 classes, verdict mix | `tests/verify/manifest.json` + `reports/dashboard.json` |
| profile live: layernorm 9200 ns, 24 regs/thread, occupancy 1, limiter registers+memory | `reports/demo/stage-1-profile.json`, `reports/dashboard.json` |

No number in the report is fabricated; the only interpretive decision is that the two
generations of the matmul benchmark (0.2027 ms ledger vs 0.4153 ms committed JSONL) are
reported side by side with their distinct provenance rather than conflated.
