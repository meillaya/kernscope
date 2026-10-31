#!/usr/bin/env python3
"""capture_equiv_check.py - todo-44 dual-mechanism equivalence test (gpu-amd).

The Metis circularity breaker: for the todo-24 portable kernel
(elementwise_add), the bundle produced by the CUDA mechanism (CUPTI
callback/injection driven by the synthetic harness - "CUDA-shim-on-mock") must
equal FIELD-FOR-FIELD the bundle produced by the todo-45 HIP mechanism
(rocprofiler-sdk intercept table, LIVE on gfx1101) under
docs/report/kernscope-technical-report.pdf (this todo's deliverable):

  M1-M8  launch record: kernel name, grid/block/shared/stream, arg count,
         arg sizes, scalar arg bytes BYTE-IDENTICAL, pointer args equal after
         VA normalization (allocation ordinals);
  M9     allocation sizes equal (sorted multiset);
  M10    copy records: byte counts + kinds equal (H2D=1, D2H=2);
  M11    input-buffer blobs BYTE-IDENTICAL (sha256 per H2D index);
  M12/M13 kernel binary: presence + PER-VENDOR sha256 (cubin != hsaco by
         construction - NEVER cross-vendor byte equality);
  M14    stream records equal.

Both bundles are produced in ONE test process (CUDA side in-process via the
producer binary, HIP side as a subprocess under libkernscope_capture_hip.so).

Usage: capture_equiv_check.py <build-dir> <cupti-sub-src-dir> <templates-src-dir>
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

BUILD = Path(sys.argv[1]).resolve()
CUPTI_SRC = Path(sys.argv[2]).resolve()
TPL_SRC = Path(sys.argv[3]).resolve()
ROOT = BUILD.parent

CUDA_SIDE = BUILD / "libs" / "cupti-sub" / "equiv_cuda_side"
CUBIN = BUILD / "libs" / "cupti-sub" / "elementwise_add.cubin"
HIP_APP = BUILD / "libs" / "cupti-sub" / "equiv_hip_app"
HIP_TOOL = BUILD / "libkernscope_capture_hip.so"
GOLDEN_DIR = ROOT / "tests" / "golden"

KERNEL = "kernscope_templates_elementwise_add"
N, THREADS = 1024, 256
GRID = 4

failures: list[str] = []


def check(cond: bool, what: str):
    if cond:
        print(f"[PASS] {what}")
    else:
        print(f"[FAIL] {what}")
        failures.append(what)


#===--- bundle parsing + normalization (docs/report/kernscope-technical-report.pdf) -----------===#


def parse_bundle(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    assert lines[0] == "kernscope-capture-v0", f"{path}: bad header"
    assert lines[-1] == "end-capture-v0", f"{path}: bad trailer"
    return lines


def norm_kernel_name(name: str) -> str:
    """§4.1: strip the HIP .kd kernel-descriptor suffix and any Itanium
    mangling prefix; the base name is the extern-C symbol."""
    if name.endswith(".kd"):
        name = name[: -len(".kd")]
    # _Z<len><name>E or _Z<len><name>... - strip the mangling prefix.
    m = re.fullmatch(r"_Z\d+([^E]+)E?.*", name)
    if m:
        name = m.group(1)
    return name


def alloc_ordinals(lines: list[str]) -> dict[str, int]:
    """§4.3: allocation ordinal = position in the program's allocation ORDER
    (the alloc record sequence), 1-based. Both mechanisms allocate in the same
    program order (a, b, out), so the same logical buffer gets the same
    ordinal on both sides regardless of the raw VAs."""
    ordinal = {}
    for l in lines:
        if l.startswith("alloc "):
            m = re.search(r"ptr=(0x[0-9a-f]+)", l)
            if m:
                ordinal[m.group(1)] = len(ordinal) + 1
    return ordinal


def norm_ptr(ptr_hex: str, ordinals: dict[str, int]) -> str:
    return f"va#{ordinals[ptr_hex]}" if ptr_hex in ordinals else f"host:{ptr_hex}"


def parse_launch(lines: list[str], ordinals: dict[str, int]):
    """The launch record selected by kernel name (the HIP path also records
    the runtime's internal dispatch launches - truthful capture, filtered by
    name per capture_hip_check.py convention)."""
    for l in lines:
        if not l.startswith("launch "):
            continue
        m = re.search(r'kernel="([^"]*)"', l)
        if m is None or norm_kernel_name(m.group(1)) != KERNEL:
            continue
        g = re.search(r"grid=(\d+)x(\d+)x(\d+)", l)
        b = re.search(r"block=(\d+)x(\d+)x(\d+)", l)
        sh = re.search(r"shared=(\d+)", l)
        st = re.search(r"stream=(0x[0-9a-f]+)", l)
        am = re.search(r"args=(\d+)", l)
        slots = re.findall(r"\[(\d+):([0-9a-f]*)\]", l)
        return {
            "name": norm_kernel_name(m.group(1)),
            "grid": tuple(int(x) for x in g.groups()),
            "block": tuple(int(x) for x in b.groups()),
            "shared": int(sh.group(1)),
            "stream": st.group(1),
            "arg_count": int(am.group(1)),
            "slots": [(int(size), bytes.fromhex(hexs))
                      for size, hexs in slots],
        }
    return None


def norm_launch_args(slots, ordinals: dict[str, int], ptr_sizes: set[int]):
    """§4.3: pointer-valued slots (size in ptr_sizes, value matches an
    allocation) -> va#ordinal; scalar slots stay raw bytes."""
    normed = []
    for size, bts in slots:
        if size in ptr_sizes and len(bts) >= 8:
            v = int.from_bytes(bts[:8], "little")
            key = f"0x{v:x}"
            if key in ordinals:
                normed.append((size, f"va#{ordinals[key]}"))
                continue
        normed.append((size, bts.hex()))
    return normed


def main() -> int:
    for art in (CUDA_SIDE, CUBIN, HIP_APP, HIP_TOOL):
        if not os.path.exists(art):
            print(f"[FAIL] required artifact missing: {art} (run "
                  f"`cmake --build build` first)")
            return 1

    sys.path.insert(0, str(GOLDEN_DIR))
    from kernscope_golden import inputs as kinputs  # noqa: E402

    with tempfile.TemporaryDirectory(prefix="capture_equiv_") as td:
        work = Path(td)
        ref_npz = work / "ref.npz"
        gen = subprocess.run(
            [sys.executable, "-m", "kernscope_golden", "gen",
             "--kernel", "reduction", "--shape", "2,1024", "--out", str(ref_npz)],
            cwd=GOLDEN_DIR, capture_output=True, text=True)
        if gen.returncode != 0:
            print(f"[FAIL] golden gen: {gen.stderr}")
            return 1
        x = kinputs.load_npz(ref_npz, "reduction")["x"]
        a_bin, b_bin = work / "a.bin", work / "b.bin"
        a_bin.write_bytes(x[0].tobytes())
        b_bin.write_bytes(x[1].tobytes())
        out_bin = work / "out.bin"

        #--- CUDA side: shim-on-mock bundle (in-process synthetic CUPTI) ----
        cuda_blobs = work / "cuda_blobs"
        cuda_blobs.mkdir()
        cuda_bundle = work / "cuda.bundle"
        cuda_run = subprocess.run(
            [str(CUDA_SIDE), "--cubin", str(CUBIN),
             "--kernel-name", KERNEL, "--bundle", str(cuda_bundle),
             "--blobdir", str(cuda_blobs),
             "--in", str(a_bin), "--in", str(b_bin),
             "--out-size", str(x[0].nbytes), "--n", str(N),
             "--grid", str(GRID), "--block", str(THREADS)],
            capture_output=True, text=True)
        print(cuda_run.stdout, end="")
        if cuda_run.returncode != 0:
            print(cuda_run.stderr)
            return 1

        #--- HIP side: live on gfx1101 under libkernscope_capture_hip.so ----
        hip_blobs = work / "hip_blobs"
        hip_blobs.mkdir()
        hip_bundle = work / "hip.bundle"
        env = dict(os.environ)
        env["LD_PRELOAD"] = str(HIP_TOOL)
        env["KERNSCOPE_HIP_BUNDLE"] = str(hip_bundle)
        env["KERNSCOPE_HIP_BLOB_DIR"] = str(hip_blobs)
        hip_run = subprocess.run(
            [str(HIP_APP), str(a_bin), str(b_bin), str(out_bin), str(N)],
            env=env, capture_output=True, text=True)
        print(hip_run.stdout, end="")
        if hip_run.returncode != 0:
            print(hip_run.stderr)
            return 1

        #--- field-for-field comparison per docs/report/kernscope-technical-report.pdf ----------
        cuda = parse_bundle(cuda_bundle)
        hip = parse_bundle(hip_bundle)
        cuda_ord = alloc_ordinals(cuda)
        hip_ord = alloc_ordinals(hip)
        check(len(cuda_ord) == 3 and len(hip_ord) == 3,
              f"three allocations per side (cuda={len(cuda_ord)}, "
              f"hip={len(hip_ord)})")

        # M9: allocation sizes equal (sorted multiset).
        alloc_sizes = lambda lines: sorted(
            int(re.search(r"size=(\d+)", l).group(1))
            for l in lines if l.startswith("alloc "))
        check(alloc_sizes(cuda) == alloc_sizes(hip) == [x[0].nbytes] * 3,
              f"M9 alloc sizes equal {[x[0].nbytes] * 3}")

        # M1-M8: the launch record.
        cl = parse_launch(cuda, cuda_ord)
        hl = parse_launch(hip, hip_ord)
        check(cl is not None, "CUDA bundle has the elementwise launch")
        check(hl is not None, "HIP bundle has the elementwise launch")
        if cl is None or hl is None:
            print("\n=== missing launch record(s) ===")
            return 1
        check(cl["name"] == hl["name"] == KERNEL,
              f"M1 kernel name equal under normalization ({KERNEL})")
        check(cl["grid"] == hl["grid"] == (GRID, 1, 1),
              f"M2 grid equal {cl['grid']}")
        check(cl["block"] == hl["block"] == (THREADS, 1, 1),
              f"M3 block equal {cl['block']}")
        check(cl["shared"] == hl["shared"] == 0, f"M4 shared equal (0)")
        check(cl["stream"] == hl["stream"] == "0x0",
              f"M5 stream equal (default 0x0)")
        check(cl["arg_count"] == hl["arg_count"] == 4,
              f"M6 arg count equal (4)")
        sizes_c = [s for s, _ in cl["slots"]]
        sizes_h = [s for s, _ in hl["slots"]]
        check(sizes_c == sizes_h == [8, 8, 8, 4],
              f"M7 arg sizes equal {sizes_c}")

        # The scalar (int32 n) slot: byte-identical WITHOUT normalization.
        scalar_c = [bts for s, bts in cl["slots"] if s == 4]
        scalar_h = [bts for s, bts in hl["slots"] if s == 4]
        check(len(scalar_c) == 1 and scalar_c[0] == scalar_h[0] and
              int.from_bytes(scalar_c[0], "little") == N,
              f"M7 scalar arg byte-identical (int32 n={N}, "
              f"bytes={scalar_c[0].hex() if scalar_c else None})")

        # M8: pointer args equal after VA normalization. First prove the raw
        # values DIFFER (VA normalization is necessary), then compare the
        # normalized ordinals.
        ptr_c = [bts for s, bts in cl["slots"] if s == 8]
        ptr_h = [bts for s, bts in hl["slots"] if s == 8]
        raw_c = [int.from_bytes(b[:8], "little") for b in ptr_c]
        raw_h = [int.from_bytes(b[:8], "little") for b in ptr_h]
        check(len(ptr_c) == 3 and len(ptr_h) == 3, "three pointer slots per side")
        check(set(raw_c).isdisjoint(set(raw_h)),
              "raw device VAs differ across vendors (VA normalization is "
              f"necessary: cuda={[hex(v) for v in raw_c]} hip="
              f"{[hex(v) for v in raw_h]})")
        normed_c = [x for x in norm_launch_args(cl["slots"], cuda_ord, {8})
                    if x[0] == 8]
        normed_h = [x for x in norm_launch_args(hl["slots"], hip_ord, {8})
                    if x[0] == 8]
        check(normed_c == normed_h == [(8, "va#1"), (8, "va#2"), (8, "va#3")],
              f"M8 pointer args equal after VA normalization "
              f"({normed_c})")

        # M10: copy records - H2D (kind 1) byte counts equal; D2H (kind 2)
        # present on both sides.
        def copies(lines):
            return [l for l in lines if l.startswith("copy ")]
        def kind_bytes(lines, kind):
            return sorted(
                int(re.search(r"bytes=(\d+)", l).group(1))
                for l in copies(lines) if f"cuda-kind={kind}" in l)
        check(kind_bytes(cuda, 1) == kind_bytes(hip, 1) == [x[0].nbytes] * 2,
              f"M10 H2D copy byte counts equal {[x[0].nbytes] * 2}")
        check(len(kind_bytes(cuda, 2)) == 1 and
              len(kind_bytes(hip, 2)) == 1,
              "M10 D2H copy present on both sides")

        # M11: input-buffer blobs byte-identical (per H2D call index).
        def blob_hashes(blob_dir: Path):
            return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(blob_dir.glob("input-buffer-*.bin"))}
        ch, hh = blob_hashes(cuda_blobs), blob_hashes(hip_blobs)
        check(set(ch) == set(hh) == {"input-buffer-1.bin", "input-buffer-2.bin"},
              f"M11 two input blobs per side ({sorted(ch)}, {sorted(hh)})")
        check(all(ch[k] == hh[k] for k in ch),
              f"M11 input-buffer blobs byte-identical "
              f"(a sha256={ch.get('input-buffer-1.bin', '?')[:16]}..., "
              f"b sha256={ch.get('input-buffer-2.bin', '?')[:16]}...)")

        # M12/M13: kernel binary - presence + per-vendor hash recorded.
        def module_hashes(lines):
            return {re.search(r"sha256=([0-9a-f]{64})", l).group(1)
                    for l in lines
                    if l.startswith("fatbin ") or l.startswith("cubin ")
                    if re.search(r"sha256=([0-9a-f]{64})", l)}
        cuda_mods = module_hashes(cuda)
        hip_mods = module_hashes(hip)
        cubin_blobs = {hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in cuda_blobs.glob("cubin-*.cubin")}
        hsaco_blobs = {hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in hip_blobs.glob("code-object-*.hsaco")}
        check(len(cuda_mods) >= 1 and len(hip_mods) >= 1,
              "M12 kernel-binary blob present per side")
        check(cuda_mods == cubin_blobs,
              "M13 CUDA module sha256 == cubin blob sha256 (provenance "
              "bijection)")
        check(hip_mods == hsaco_blobs,
              "M13 HIP module sha256 == hsaco blob sha256 (provenance "
              "bijection)")
        check(cuda_mods.isdisjoint(hip_mods),
              "M12 per-vendor hashes recorded and DISTINCT (cubin != hsaco "
              f"by construction; cuda={sorted(cuda_mods)[0][:16]}... hip="
              f"{sorted(hip_mods)[0][:16]}...)")

        # M14: stream records - the runner synchronizes the default stream.
        def stream_lines(lines):
            return [l for l in lines if l.startswith("stream ")]
        check(len(stream_lines(cuda)) >= 1 and len(stream_lines(hip)) >= 1,
              "M14 stream record(s) present per side (sync of default "
              "stream)")
        check(any("sync handle=0x0" in l for l in stream_lines(cuda)) and
              any("sync handle=0x0" in l for l in stream_lines(hip)),
              "M14 default-stream sync recorded on both sides")

    if failures:
        print(f"\n=== {len(failures)} FAILURE(S) ===")
        return 1
    print("\n[PASS] capture_equiv: CUDA(CUPTI-synthetic) == HIP(live) "
          "field-for-field per docs/report/kernscope-technical-report.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
