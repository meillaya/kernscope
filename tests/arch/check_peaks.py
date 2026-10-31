#!/usr/bin/env python3
# == check_peaks.py - validate the committed gfx1101 measured peak entries ====
#
# kernscope todo-22 helper (owned by libs/bench): validates that the gfx1101
# roofline peaks are MEASURED and committed with provenance, never projected
# or fabricated (plan todo 12 MUST NOT / Metis M-6). Asserts:
#   * libs/arch kGfx1101 row: measured == true, peak_bw_gbps > 300 GB/s,
#     peak_fp32_tflops > 10 TFLOPS (sanity floors for a 7800 XT), and the
#     peak_source tag is "measured-<date>-gfx1101";
#   * libs/roofline gfx1101 row: projected == false, fp32_peak_flops >
#     10e12 FLOP/s, bandwidth_bytes_per_s > 300e9 B/s, provenance mentions
#     "measured";
#   * tests/fixtures/peaks/gfx1101-bench.jsonl records: a DRAM-resident read record
#     (>= 128 MiB, beyond the 64 MiB Infinity Cache) with derived bandwidth
#     > 300 GB/s, a peak_flops record, and every record carries the required
#     provenance fields non-empty (device name, gcnArchName, date, clocks,
#     measured=true).
#
# Exit 0 on pass, 1 on any assertion failure. Runs without a GPU (ctest
# LABELS no-gpu).

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

failures = []


def check(ok, msg):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {msg}")
    if not ok:
        failures.append(msg)


def parse_value(row_text, field):
    # .field = <value>,  with an optional trailing // comment
    m = re.search(rf"\.{field}\s*=\s*([^,\n]+)", row_text)
    return m.group(1).strip() if m else None


# --- 1. libs/arch kGfx1101 row ----------------------------------------------
arch_h = (ROOT / "libs/arch/include/kernscope/arch.h").read_text()
m = re.search(r"constexpr ArchInfo kGfx1101\{(.*?)\};", arch_h, re.S)
check(m is not None, "arch.h contains the kGfx1101 row")
row = m.group(1) if m else ""

bw = parse_value(row, "peak_bw_gbps")
tflops = parse_value(row, "peak_fp32_tflops")
measured = parse_value(row, "measured")
source = parse_value(row, "peak_source")
check(measured == "true", f"arch.h gfx1101 measured == true (got {measured})")
try:
    check(float(bw) > 300.0, f"arch.h gfx1101 peak_bw_gbps {bw} > 300 GB/s")
except (TypeError, ValueError):
    check(False, f"arch.h gfx1101 peak_bw_gbps parseable ({bw})")
try:
    check(float(tflops) > 10.0, f"arch.h gfx1101 peak_fp32_tflops {tflops} > 10 TFLOPS")
except (TypeError, ValueError):
    check(False, f"arch.h gfx1101 peak_fp32_tflops parseable ({tflops})")
if source:
    src = source.strip('"')
    check(
        re.fullmatch(r"measured-\d{4}-\d{2}-\d{2}-gfx1101", src) is not None,
        f"arch.h gfx1101 peak_source is measured-<date>-gfx1101 (got {src})",
    )
else:
    check(False, "arch.h gfx1101 peak_source non-empty")

# --- 2. libs/roofline gfx1101 row -------------------------------------------
roofline_cpp = (ROOT / "libs/roofline/src/roofline.cpp").read_text()
m2 = re.search(r'\{"gfx1101", "RX 7800 XT \(RDNA3\)".*?\},', roofline_cpp, re.S)
check(m2 is not None, "roofline.cpp contains the gfx1101 MachinePeak row")
rrow = m2.group(0) if m2 else ""
check("/*projected=*/false" in rrow,
      "roofline.cpp gfx1101 row projected == false")
# fp32 peak and bandwidth appear as non-zero measured constants (16.32e12 /
# 591.0e9) with a measured provenance string.
check("16.32e12" in rrow, "roofline.cpp gfx1101 fp32_peak_flops == 16.32e12")
check("591.0e9" in rrow, "roofline.cpp gfx1101 bandwidth_bytes_per_s == 591.0e9")
check("MEASURED" in rrow or "measured" in rrow.lower(),
      "roofline.cpp gfx1101 provenance marks the values as measured")

# --- 3. committed JSONL provenance records -----------------------------------
jsonl = ROOT / "tests/fixtures/peaks/gfx1101-bench.jsonl"
check(jsonl.exists(), f"tests/fixtures/peaks/gfx1101-bench.jsonl exists ({jsonl})")
records = []
if jsonl.exists():
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

check(len(records) > 0, "at least one measured record committed")

read_recs = [r for r in records
             if r.get("kernel_family") == "peak_bw_read"
             and r.get("config", {}).get("size_bytes", 0) >= (128 << 20)]
check(len(read_recs) > 0,
      "a DRAM-resident (>= 128 MiB) peak_bw_read record exists")
flops_recs = [r for r in records if r.get("kernel_family") == "peak_flops"]
check(len(flops_recs) > 0, "a peak_flops record exists")

for name, rec in (("peak-bw", read_recs[-1] if read_recs else None),
                  ("peak-flops", flops_recs[-1] if flops_recs else None)):
    if rec is None:
        continue
    cfg = rec.get("config", {})
    prov = rec.get("provenance", {})
    check(cfg.get("gcn_arch") == "gfx1101", f"{name} record arch == gfx1101")
    for field in ("device_name", "gcn_arch_full", "date", "clocks"):
        check(str(cfg.get(field, "")).strip() != "",
              f"{name} record provenance field '{field}' non-empty")
    check(cfg.get("measured") is True, f"{name} record measured == true")
    check(prov.get("mode") == "measured", f"{name} record provenance.mode == measured")

if read_recs:
    rec = read_recs[-1]
    size = float(rec["config"]["size_bytes"])
    bw = size / (rec["timings"]["median"] / 1000.0) / 1e9
    check(bw > 300.0, f"committed DRAM read bandwidth {bw:.1f} GB/s > 300 GB/s")

print()
if failures:
    print(f"check_peaks: FAIL ({len(failures)} assertion(s))")
    sys.exit(1)
print("check_peaks: PASS — gfx1101 peaks measured, committed, provenance intact")
sys.exit(0)
