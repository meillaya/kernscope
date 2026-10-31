# Architecture diagrams

The `.excalidraw` files are the sources for the PNGs used in the README.
The technical report uses TikZ figures instead. They use Excalifont, the hand-drawn font bundled with
the Excalidraw CLI. Regenerating them needs Node 22 or newer.

## Regenerate

Install the CLI into a local directory, not globally:

```bash
npm install --prefix .tools/excalidraw-cli @excalidraw/cli@0.2.0
```

Rebuild the scene JSON with the standard library:

```bash
python3 docs/diagrams/build_diagrams.py
```

Render each scene with an installed Chrome, Chromium, Edge, or Firefox:

```bash
export EXCALIDRAW_BROWSER_PATH=/path/to/chrome
.tools/excalidraw-cli/node_modules/.bin/excalidraw render \
  docs/diagrams/loop.excalidraw --out docs/diagrams/loop.png \
  --scale 1 --max-width 2048 --max-height 1600
```

Use the same command for `stack.excalidraw` and `hardware.excalidraw`. The
sources also open in [Excalidraw](https://excalidraw.com) if you want to edit
them by hand and export the PNG from there.
