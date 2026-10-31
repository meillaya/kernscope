#!/usr/bin/env python3
"""Build the kernscope architecture scenes as .excalidraw files.

The scenes are hand-written JSON: Excalidraw's file format is stable enough
for that, and it keeps the diagrams reproducible without a browser in the
generation step. Render with:

    excalidraw render docs/diagrams/loop.excalidraw --out docs/diagrams/loop.png

The fonts are embedded in the Excalidraw CLI, so the PNGs do not depend on
system fonts.
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent

# Blog-style hand-drawn palette: pastel fill, saturated stroke, dark ink.
INK = "#1e1e1e"
MUTED = "#495057"
PAPER = "#ffffff"
GRID = "#dee2e6"

RED = "#c92a2a"
RED_FILL = "#ffe3e3"
ORANGE = "#d9480f"
ORANGE_FILL = "#ffe8cc"
AMBER = "#e67700"
AMBER_FILL = "#fff3bf"
GREEN = "#2b8a3e"
GREEN_FILL = "#d3f9d8"
BLUE = "#1971c2"
BLUE_FILL = "#d0ebff"
VIOLET = "#6741d9"
VIOLET_FILL = "#e5dbff"
TEAL = "#0c8599"
TEAL_FILL = "#c5f6fa"
GRAY_FILL = "#f1f3f5"

EXCALIFONT = 5
CASCADIA = 3

_seed = 1000


def _next_seed() -> int:
    global _seed
    _seed += 1
    return _seed


def _element(kind: str, ident: str, x: float, y: float, w: float, h: float, **kw):
    element = {
        "id": ident,
        "type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": INK,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": None,
        "seed": _next_seed(),
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": [],
        "updated": 1,
        "link": None,
        "locked": False,
    }
    element.update(kw)
    return element


def box(ident, x, y, w, h, fill, stroke=INK, width=2, dash=False, rough=1):
    return _element(
        "rectangle",
        ident,
        x,
        y,
        w,
        h,
        backgroundColor=fill,
        strokeColor=stroke,
        strokeWidth=width,
        strokeStyle="dashed" if dash else "solid",
        roughness=rough,
        roundness={"type": 3},
        fillStyle="solid",
    )


def pill(ident, x, y, w, h, label, color, fill=PAPER, size=24):
    elements = [
        box(ident + "-box", x, y, w, h, fill, color, width=2),
        text(ident + "-label", x, y + (h - size * 1.25) / 2 - 2, w, label, size, color,
             EXCALIFONT, "center"),
    ]
    return elements


def text(ident, x, y, w, body, size=26, color=INK, font=EXCALIFONT,
         align="left", line_height=1.25):
    lines = body.count("\n") + 1
    return _element(
        "text",
        ident,
        x,
        y,
        w,
        size * line_height * lines,
        text=body,
        originalText=body,
        fontSize=size,
        fontFamily=font,
        textAlign=align,
        verticalAlign="top",
        lineHeight=line_height,
        strokeColor=color,
        roughness=1,
        backgroundColor="transparent",
    )


def centered_text(ident, cx, y, body, size, color=INK, font=EXCALIFONT):
    widest = max(line for line in body.split("\n"))
    w = len(widest) * size * 0.56 * 1.12
    return text(ident, cx - w / 2, y, w, body, size, color, font, "center")


def polyline(ident, points, color=INK, width=2, dash=False, arrow=True,
             arrowhead="arrow", rough=1, end_size=18):
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_x = max(p[0] for p in points)
    max_y = max(p[1] for p in points)
    local = [[p[0] - min_x, p[1] - min_y] for p in points]

    return _element(
        "arrow",
        ident,
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
        points=local,
        lastCommittedPoint=None,
        startBinding=None,
        endBinding=None,
        startArrowhead=None,
        endArrowhead=arrowhead if arrow else None,
        strokeColor=color,
        strokeWidth=width,
        strokeStyle="dashed" if dash else "solid",
        roughness=rough,
        roundness={"type": 2} if len(points) > 2 else None,
    )


def underline(ident, x1, y, x2, color, width=4):
    return polyline(ident, [(x1, y), (x2, y)], color, width, arrow=False, rough=2)


def dot(ident, cx, cy, r, color):
    return _element(
        "ellipse", ident, cx - r, cy - r, r * 2, r * 2,
        backgroundColor=color, strokeColor=color, strokeWidth=1, roughness=1,
    )


def scene(elements):
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "gridSize": None,
            "viewBackgroundColor": PAPER,
            "exportBackground": True,
            "exportWithDarkMode": False,
        },
        "files": {},
    }


def write(name, elements):
    path = OUT / name
    path.write_text(json.dumps(scene(elements), indent=2) + "\n")
    print(f"wrote {path} ({len(elements)} elements)")


# ---------------------------------------------------------------------------
# 1. The loop
# ---------------------------------------------------------------------------

def build_loop():
    els = []
    els.append(text("title", 80, 40, 1100, "one closed loop", 64, INK))
    els.append(underline("title-rule", 84, 122, 700, BLUE, 5))
    els.append(text("subtitle", 80, 138, 1500,
                    "each stage consumes the previous stage's output", 28, MUTED))

    cards = [
        {
            "x": 50, "y": 215, "fill": RED_FILL, "stroke": RED, "tag": "C6",
            "name": "profile", "cli": "kernscope-profile",
            "body": "ranks hot kernels and\nexplains why they stall",
        },
        {
            "x": 670, "y": 215, "fill": ORANGE_FILL, "stroke": ORANGE, "tag": "C4",
            "name": "capture", "cli": "kernscope-capture",
            "body": "records a launch into a\nbitwise-replayable bundle",
        },
        {
            "x": 1290, "y": 215, "fill": AMBER_FILL, "stroke": AMBER, "tag": "C1",
            "name": "analyze", "cli": "kernscope-analyze",
            "body": "runs 10 checks on the\ncaptured ISA",
        },
        {
            "x": 1290, "y": 680, "fill": GREEN_FILL, "stroke": GREEN, "tag": "C2+C5",
            "name": "transform", "cli": "kernscope-tune\nkernscope-pass-report",
            "body": "builds template variants\nand rewrites LLVM IR",
        },
        {
            "x": 670, "y": 680, "fill": BLUE_FILL, "stroke": BLUE, "tag": "C3",
            "name": "verify", "cli": "kernscope-verify",
            "body": "checks 8 safety classes\nbefore anything compiles",
        },
        {
            "x": 50, "y": 680, "fill": VIOLET_FILL, "stroke": VIOLET, "tag": "lib",
            "name": "benchmark", "cli": "libs/bench",
            "body": "times correct variants\nwith p20/p80 quantiles",
        },
    ]

    for i, card in enumerate(cards):
        x, y = card["x"], card["y"]
        tag = card["tag"]
        tag_w = 98 if len(tag) > 2 else 64
        els.append(box(f"card-{i}", x, y, 460, 250, card["fill"], card["stroke"], width=2))
        els.extend(pill(f"tag-{i}", x + 22, y + 20, tag_w, 42, tag, card["stroke"], PAPER, 20))
        els.append(text(f"name-{i}", x + 22 + tag_w + 16, y + 19, 360,
                        card["name"], 40, INK))
        cli_lines = card["cli"].count("\n") + 1
        els.append(text(f"cli-{i}", x + 24, y + 84, 430, card["cli"], 22,
                        card["stroke"], CASCADIA))
        els.append(text(f"body-{i}", x + 24, y + 150,
                        412, card["body"], 25, INK))

    # Forward path.
    els.append(polyline("a1", [(510, 340), (670, 340)], RED, 2.5))
    els.append(text("a1-label", 516, 298, 150, "hot kernel", 21, RED))
    els.append(polyline("a2", [(1130, 340), (1290, 340)], ORANGE, 2.5))
    els.append(text("a2-label", 1136, 298, 150, "bundle", 21, ORANGE))

    els.append(polyline("a3", [(1520, 465), (1520, 680)], AMBER, 2.5))
    els.append(text("a3-label", 1540, 548, 150, "findings", 21, AMBER))

    els.append(polyline("a4", [(1290, 805), (1130, 805)], GREEN, 2.5))
    els.append(text("a4-label", 1136, 763, 150, "variants", 21, GREEN))

    els.append(polyline("a5", [(670, 805), (510, 805)], BLUE, 2.5))
    els.append(text("a5-label", 516, 763, 150, "approved", 21, BLUE))

    # Delta feedback runs around the outside so it does not cross the cards.
    els.append(polyline("feedback", [(280, 930), (280, 1005), (42, 1005),
                                     (42, 340), (50, 340)], VIOLET, 3, dash=True))
    els.append(text("feedback-label", 330, 962, 900,
                    "delta reports start the next pass", 24, VIOLET))

    write("loop.excalidraw", els)


# ---------------------------------------------------------------------------
# 2. The shared stack
# ---------------------------------------------------------------------------

def build_stack():
    els = []
    els.append(text("title", 80, 40, 1100, "one shared stack", 64, INK))
    els.append(underline("title-rule", 84, 122, 740, GREEN, 5))
    els.append(text("subtitle", 80, 138, 1580,
                    "six tools, one build; each links only what it needs",
                    28, MUTED))

    tools = [
        ("C1", "analyze", "kernscope-analyze",
         "10 checks on PTX / AMDGPU ISA", "parsers · model · evidence"),
        ("C2", "tune", "kernscope-tune",
         "correctness-gated config search", "model · evidence · capture"),
        ("C3", "verify", "kernscope-verify",
         "8 IR checks; approved IR or reject", "LLVM"),
        ("C4", "capture", "kernscope-capture",
         "bundle + bitwise replay", "capture · evidence · model"),
        ("C5", "pass", "kernscope-pass-report",
         "LLVM IR rewrites + delta report", "LLVM"),
        ("C6", "profile", "kernscope-profile",
         "hot kernels + why-slow hints", "model · evidence · capture"),
    ]
    positions = [(80, 215), (660, 215), (1240, 215),
                 (80, 430), (660, 430), (1240, 430)]

    for i, (tag, name, cli, body, links) in enumerate(tools):
        x, y = positions[i]
        color = [RED, ORANGE, AMBER, GREEN, BLUE, VIOLET][i]
        fill = [RED_FILL, ORANGE_FILL, AMBER_FILL, GREEN_FILL, BLUE_FILL, VIOLET_FILL][i]
        els.append(box(f"tool-{i}", x, y, 480, 185, fill, color, width=2))
        els.extend(pill(f"tool-tag-{i}", x + 22, y + 18, 64, 40, tag, color, PAPER, 20))
        els.append(text(f"tool-name-{i}", x + 102, y + 17, 340, name, 38, INK))
        els.append(text(f"tool-cli-{i}", x + 24, y + 76, 430, cli, 21, color, CASCADIA))
        els.append(text(f"tool-body-{i}", x + 24, y + 112, 430, body, 24, INK))
        els.append(text(f"tool-links-{i}", x + 24, y + 148, 430,
                        "links: " + links, 20, MUTED))

    # One bracket groups the six tools, then a single arrow enters the stack.
    els.append(polyline("tool-bracket", [(80, 660), (1720, 660)], MUTED, 2, arrow=False))
    els.append(polyline("tool-drop", [(900, 660), (900, 716)], MUTED, 2.5))
    els.append(text("tool-drop-label", 920, 674, 140, "link", 21, MUTED))

    # Shared libraries, grouped by what they know how to do.
    els.append(box("libs-frame", 60, 720, 1680, 400, "#ffffff", MUTED, width=2, dash=True))

    groups = [
        {
            "x": 90, "stroke": TEAL, "fill": TEAL_FILL, "title": "parse + model",
            "chips": [
                "ptxparse · PTX",
                "amdparse · AMDGPU",
                "arch · gfx1101 / sm_80 / sm_90",
                "occupancy · limiter + caps",
                "roofline · bound class",
            ],
            "note": None,
        },
        {
            "x": 650, "stroke": ORANGE, "fill": ORANGE_FILL, "title": "measure + report",
            "chips": [
                "bench · L2 flush, p20/p80",
                "db · JSON records",
                "report · schema 1.0, provenance",
                "gpuruntime · CUDA / HIP shim",
            ],
            "note": None,
        },
        {
            "x": 1210, "stroke": VIOLET, "fill": VIOLET_FILL, "title": "capture",
            "chips": [
                "capture-core · bundle model",
                "capture-hip · rocprofiler-sdk",
                "cupti-sub · CUPTI subscription",
            ],
            "note": "CUDA paths compile-tested;\nlive runs are rental-gated",
        },
    ]

    for i, group in enumerate(groups):
        x = group["x"]
        els.append(box(f"lib-{i}", x, 760, 520, 340, group["fill"], group["stroke"], width=2))
        els.append(text(f"lib-title-{i}", x + 20, 772, 480, group["title"], 30, group["stroke"]))
        for j, chip in enumerate(group["chips"]):
            cy = 826 + j * 52
            els.append(box(f"chip-{i}-{j}", x + 20, cy, 480, 42, PAPER, group["stroke"], width=1.5))
            els.append(text(f"chip-text-{i}-{j}", x + 34, cy + 6, 450, chip, 21, INK))
        if group["note"]:
            els.append(text(f"lib-note-{i}", x + 20, 1000, 480, group["note"], 19, MUTED))

    # External dependencies sit under the stack they support.
    els.append(box("foundation", 90, 1160, 1620, 78, GRAY_FILL, MUTED, width=2))
    els.append(text("foundation-label", 112, 1180, 1580,
                    "LLVM / MLIR 21  ·  CUDA 12.6 headers  ·  ROCm 7.2.3  ·  pinned by devenv.lock",
                    24, INK))

    write("stack.excalidraw", els)


# ---------------------------------------------------------------------------
# 3. Where the code runs
# ---------------------------------------------------------------------------

def build_hardware():
    els = []
    els.append(text("title", 80, 40, 1100, "where the code runs", 64, INK))
    els.append(underline("title-rule", 84, 122, 800, ORANGE, 5))
    els.append(text("subtitle", 80, 138, 1580,
                    "one source tree, three venues, and a provenance flag on every report",
                    28, MUTED))

    # One brace over the three venues: same source, different execution facts.
    els.append(polyline("brace", [(80, 198), (1720, 198)], MUTED, 2, arrow=False))
    for x in (320, 900, 1480):
        els.append(polyline(f"brace-drop-{x}", [(x, 198), (x, 232)], MUTED, 2))

    columns = [
        {
            "x": 80, "stroke": GREEN, "fill": GREEN_FILL,
            "kicker": "AMD · local", "name": "RX 7800 XT", "sub": "gfx1101",
            "rows": ["compile + run", "capture + replay", "profile",
                     "tune + benchmark", "numbers: measured"],
            "note": "the live path on this machine",
        },
        {
            "x": 660, "stroke": BLUE, "fill": BLUE_FILL,
            "kicker": "NVIDIA · compile", "name": "sm_80 / sm_90", "sub": "nvcc + CUDA 12.6 headers",
            "rows": ["compile kernels", "parse PTX", "static analysis",
                     "capture / profile unit tests", "numbers: projected"],
            "note": "no CUDA runtime on this machine",
        },
        {
            "x": 1240, "stroke": ORANGE, "fill": ORANGE_FILL,
            "kicker": "NVIDIA · rental", "name": "A100 / H100", "sub": "owner-gated",
            "rows": ["live CUPTI capture", "live CUPTI profile",
                     "runtime benchmarks", "rental-validate.sh", "numbers: measured, if produced"],
            "note": "reports/rentals/ holds no numbers yet",
        },
    ]

    for i, column in enumerate(columns):
        x = column["x"]
        y = 240
        els.append(box(f"col-{i}", x, y, 480, 580, column["fill"], column["stroke"], width=2))
        els.append(text(f"kicker-{i}", x + 24, y + 20, 430, column["kicker"], 30, column["stroke"]))
        els.append(text(f"name-{i}", x + 24, y + 60, 430, column["name"], 40, INK))
        els.append(text(f"sub-{i}", x + 24, y + 112, 430, column["sub"], 23, MUTED))
        els.append(polyline(f"rule-{i}", [(x + 24, y + 152), (x + 456, y + 152)],
                            column["stroke"], 1.5, arrow=False, rough=2))
        for j, row in enumerate(column["rows"]):
            ry = y + 178 + j * 62
            els.append(dot(f"dot-{i}-{j}", x + 38, ry + 12, 7, column["stroke"]))
            els.append(text(f"row-{i}-{j}", x + 58, ry, 390, row, 24, INK))
        els.append(text(f"note-{i}", x + 24, y + 512, 430, column["note"], 20, MUTED))

    # Provenance rule: the one line that keeps the three venues honest.
    els.append(box("prov", 80, 860, 1640, 92, GRAY_FILL, MUTED, width=2))
    els.append(dot("prov-measured", 110, 892, 7, GREEN))
    els.append(text("prov-measured-text", 128, 880, 240, "measured", 23, GREEN))
    els.append(dot("prov-projected", 270, 892, 7, BLUE))
    els.append(text("prov-projected-text", 288, 880, 260, "projected", 23, BLUE))
    els.append(text("prov-rule", 520, 878, 1180,
                    "provenance.mode is asserted by tests/report/check_dashboard.py",
                    23, INK))
    els.append(text("prov-rule-2", 520, 908, 1180,
                    "projected numbers stay in the projected slot",
                    21, MUTED))

    write("hardware.excalidraw", els)


if __name__ == "__main__":
    build_loop()
    build_stack()
    build_hardware()
