# kernscope technical report

`kernscope-technical-report.pdf` is the consolidated technical report for the
project. It supersedes the per-tool Markdown references and the earlier
research report.

## Sources

- `kernscope-technical-report.tex`: main document.
- `sections/`: tool, operation, and appendix chapters.
- `figures/`: TikZ diagrams included by the main document.
- `references.bib`: bibliography.
- `tikzflora.sty`, `tikzflora-boot.lua`, `lua/flora/`: vendored MIT-licensed
  [tikz-flora](https://flora.alpacasoft.dev) package. See
  `tikzflora-LICENSE` and `tikzflora-README.md`.

## Build

The report uses LuaLaTeX, TikZ, and the flora package. It needs a TeX
distribution with LuaLaTeX, `natbib`, `cleveref`, `booktabs`, `tabularx`,
`longtable`, `enumitem`, `listings`, `fancyhdr`, `titlesec`, `caption`,
and `grfext`.

```bash
lualatex kernscope-technical-report.tex
bibtex kernscope-technical-report
lualatex kernscope-technical-report.tex
lualatex kernscope-technical-report.tex
```

`latexmk` is also configured; run `latexmk` from this directory. The PDF is
written next to the source. The flora is used on the title page, contents
page, and part openers only, to keep the file size bounded.

The editable Excalidraw diagrams used by the README remain in
`docs/diagrams/`.
