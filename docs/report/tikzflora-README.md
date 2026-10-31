# tikz-flora

Watercolour flora grown in the margins of LuaLaTeX documents. Every page
grows its own plant around the text it holds: stems, leaves, blooms, sprays,
gilt, and if you like, bees, butterflies, birds and snails. Everything is
computed in Lua and drawn through PGF, with no images.

Site, live mixer and full guide: **https://flora.alpacasoft.dev**

## Origins

This project would not exist without two others:

- The r/LaTeX thread
  [How do you make these flowers in LaTeX?](https://old.reddit.com/r/LaTeX/comments/1xixznw/how_do_you_make_these_flowers_in_latex/),
  whose photographed page is the reference this renderer is measured
  against.
- Kevin Raymond's [florilegium](https://github.com/kevinraymond/florilegium)
  (MIT), whose grown margins, watch-it-grow animation and visitors this
  package follows.

The ink colours are Pilot iroshizuku ramps measured by
[madoromi](https://madoromi.moonwit.ch/#palettes/colours). The growth follows
Wong, Zongker and Salesin,
[Computer-Generated Floral Ornament](https://grail.cs.washington.edu/projects/ornament/ornament.pdf)
(SIGGRAPH 1998).

## Setup

You need a TeX distribution with **LuaLaTeX** (TeX Live 2023 or later,
MiKTeX, or TinyTeX). Check with `lualatex --version`. The example also uses
`lipsum` (`tlmgr install lipsum` on TinyTeX).

1. Download `tikzflora.zip` from the site, or build it with `just dist`.
2. Unzip it. It holds `tikzflora.sty`, `tikzflora-boot.lua`, the generator
   in `lua/flora/`, and `example.tex`.
3. Put your document next to `tikzflora.sty` (or copy the `.sty`, the boot
   file and `lua/` next to your document).
4. Compile with LuaLaTeX: `latexmk example` (the included `latexmkrc`
   selects LuaLaTeX), or `lualatex example` twice.

To install it for every document instead, copy the three items into
`TEXMFHOME/tex/lualatex/tikzflora/` (`kpsewhich -var-value TEXMFHOME` prints
the path).

**Overleaf:** New Project, Upload Project, choose `tikzflora.zip`. The
`latexmkrc` in the zip sets the compiler to LuaLaTeX; otherwise choose it
under Menu, Compiler.

A minimal document:

```latex
\documentclass{article}
\usepackage[left=5.5cm,right=2cm]{geometry} % room for the flora
\usepackage{tikzflora}
\florasetup{seed=7}
\begin{document}
Text.
\end{document}
```

## Settings

`\florasetup{...}` sets keys for the rest of the document, from the page it
is on. `\florathispage{...}` sets them for one page only. Mistakes stop the
compile with a message naming the key.

| Key | Values | Effect |
|---|---|---|
| `layout` | `left`, `right`, `centered`, `split`, `title`, `surround` | Where the flora grows (below) |
| `seed` | a number, or `random` | Which garden; each page varies it |
| `mix` | `{kind=weight, ...}` | How much of each kind: 1 usual, 0 none, 2 about double |
| `density` | 0 to 10 | Fuller (above 1) or airier (below 1) |
| `visitors` | 0 to 10 | The weight of every visitor at once |
| `palette` | `{part=ink, ...}` | What paints the `hero`, `blooms`, `leaves`, `leaflets`, `stems` |
| `spin` | a whole number | Picks other inks for parts set to `random`; the shapes stay |

Mix kinds: `bloom`, `soft`, `lily`, `ghost`, `umbel`, `cyme`
(forget-me-nots), `gyps` (baby's breath), `leaves`, `leaflets`, `sprays`,
`curls`, `density`, `bee`, `butterfly`, `bird`, `snail`. Visitors are off
unless you give them weight.

Palette values are `look` (the default gradients), an ink group (`blue`,
`blue-grey`, `teal`, `green`, `violet`, `warm`, `earth`, `gold`) or one ink
by name, such as `kon-peki` (all of them are in `lua/flora/inks.lua`).
`random` paints that part in one ink picked per page. The pick has its own
random stream, so changing `spin` changes only the colours: every stem,
leaf and bloom stays where it was.

```latex
\florasetup{layout=split, visitors=0.5, mix={lily=2, umbel=0},
  palette={leaves=green}}
```

### Layouts

- `left`: the left margin (the default).
- `right`: the right margin.
- `centered`: both side margins, with two different plants.
- `split`: the outer margin of a two-sided document. It is on the right of
  odd pages and the left of even ones.
- `title`: a crown over a title. Wrap the title in
  `\begin{floratitle} ... \end{floratitle}`; without it the first block
  of lines is crowned.
- `surround`: a wreath round the text block, for title pages and contents.

The flora fills whatever margins the page has, so give it room with
`geometry` (about 5 cm for a side margin, or 3 cm all round for
`surround`).

It also grows into gaps in the text. A gap between lines taller than about
three lines (40 pt), such as a `\vspace` left for a figure, opens across
the text's full width and the flora reaches into it. The space below a
short last page fills the same way.

`tex/lorem.tex` (`example.tex` in the zip) uses each of these, with a
comment at every setting.

## Working on tikz-flora

The repository uses [just](https://github.com/casey/just). These recipes
add the TinyTeX binaries to `PATH`:

- `just build NAME [SEED]`: compile `tex/NAME.tex` to `build/` with a PNG
  preview. Use `just build lorem` for the example and `just specimens` for
  one of each element.
- `just grow NAME SEED PAGE`: build `build/grow-NAME-PAGE.mp4` and `.gif`
  of a page as it grows. It needs ffmpeg, poppler (`pdftocairo`) and
  ImageMagick.
- `just dist`: build `build/tikzflora.zip`. `just test-dist` compiles the
  example from the unpacked zip alone.
- `just test`: run the Lua and Python unit tests. `just test-tex` and
  `just test-preview` check against real compiles.
- `just site-bake`, `just site-dev`, `just test-site`, `just site-deploy`:
  build, test and publish the site. `site/README.md` explains them.
- `just tuner`: run a local web app with a slider for every setting in
  `lua/flora/tune.lua`. It needs Python 3 with numpy and Pillow, and
  Node.

## Licence

MIT. See `LICENSE`.
