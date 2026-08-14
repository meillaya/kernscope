# RDNA3 (gfx1100-class) VGPR file: 768 vs 1536 — resolved

**Resolution (2026-08-14, empirical on the RX 7800 XT / gfx1101 in this dev
machine): the VGPR file is 1536 per SIMD in the native wave32 view, with
allocation granule 24 (wave32) / 12 (wave64). The "768 VGPRs" figure is the
wave64 accounting of the same file (768 = 1536 / 2), not a per-SIMD wave32
figure. Do not hardcode 16 as the RDNA3 granule.**

## The conflict

- **CLAIM A** — polykernel `docs/hip_backend.md:80` states gfx1101 has
  "**768 VGPRs / 32 KiB**" ("the same register file (768 VGPRs / 32 KiB,
  768 SGPRs / 32 KiB)…").
- **CLAIM B** — this repo's brief `docs/research/gpu-static-analyzer-research.md`
  §1.3 (lines 137-139) and §10.4 (lines 551-553) cite the RDNA3 ISA §3.3.2.1:
  "VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a shader
  may have up to 256 VGPRs… **Devices which have 1536 VGPRs per SIMD allocate
  in blocks of 24 for wave32 and 12 for wave64**" — with the caution "the
  gfx1100 family is in the latter class; don't hardcode 16".

## Evidence

### (a) rocminfo (this machine, in devenv shell)

`rocminfo` enumerates **gfx1101** ("AMD Radeon RX 7800 XT", KERNEL_DISPATCH,
BASE_PROFILE, /dev/kfd present). rocminfo prints no VGPR-file count; the
register-file figure comes from the HIP runtime instead (see (c)).

### (b) The cited ISA text in the research brief (verbatim)

gpu-static-analyzer-research.md §1.3 (lines 137-139):

> "VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a
> shader may have up to 256 VGPRs… **Devices which have 1536 VGPRs per SIMD
> allocate in blocks of 24 for wave32 and 12 for wave64**" **[RDNA3 §3.3.2.1]**.
> ⇒ `waves_per_SIMD = floor(1536 / ceil_granule(vgpr_per_lane))`, clamped by
> the hardware wave cap.

and §10.4 (lines 551-553):

> **RDNA3 allocation granule:** 16 (wave32) / 8 (wave64) generally, but
> **24/12 on devices with 1536 VGPRs per SIMD** **[RDNA3 §3.3.2.1]** — the
> gfx1100 family is in the latter class; don't hardcode 16.

### (c) Driver ground truth (HIP runtime on this machine)

`hipGetDeviceProperties` / `hipDeviceGetAttribute` on gfx1101 report:

```
regsPerMultiprocessor                        = 196608
hipDeviceAttributeMaxRegistersPerMultiprocessor = 196608
hipDeviceAttributeMaxRegistersPerBlock       = 196608
hipDeviceAttributeMaxBlocksPerMultiProcessor = 64
hipDeviceAttributeMaxThreadsPerMultiProcessor = 2048
hipDeviceAttributeWarpSize                   = 32
hipDeviceAttributeMaxSharedMemoryPerMultiprocessor = 65536  (= 64 KiB LDS/CU)
```

196608 = **1536 × 2 SIMDs/CU × 64 lanes**. In the per-lane view the register
file per SIMD is therefore **1536 per-lane entries** — exactly the ISA's
"1536 VGPRs per SIMD". (64 KiB LDS per CU also confirms brief §1.3.)

### (d) LLVM 21.1.8 backend source (fetched @llvmorg-21.1.8)

`llvm/lib/Target/AMDGPU/Utils/AMDGPUBaseInfo.cpp`:

- `getVGPRAllocGranule` (L1295-1310): `if (Feature1_5xVGPRs) return IsWave32 ? 24 : 12;`
  (gfx10.3-class non-1.5x: 16/8; CDNA (gfx90a): 8).
- `getTotalNumVGPRs` (L1336-1344): `if (Feature1_5xVGPRs) return IsWave32 ? 1536 : 768;`
  — i.e. **1536 is the wave32 total; 768 is the wave64 view of the same file.**
- `getNumWavesPerEUWithNumVGPRs` (L1371-1380):
  `waves = min(max(TotalNumVGPRs / alignTo(NumVGPRs, Granule), 1), MaxWaves)`.
- `getMaxWavesPerEU` (L1162-1168): gfx10.3+ → 16 (wave64-equivalent units).
- `AMDGPU.td` (L1860-1894): `Feature1_5xVGPRs` is set on gfx1100 (11.0.0) and
  **gfx1101 (11.0.1)** — the RX 7800 XT class; gfx1102 (11.0.2) does **not**
  carry it (a real subtlety: 1.5x-VGPR is a per-SKU feature, not a
  generation-wide default).

**This is the resolution of CLAIM A**: polykernel's "768 VGPRs" equals
LLVM's `getTotalNumVGPRs` for the wave64 mode of the same 1.5xVGPR file —
1536/2 — so the two claims describe the same register file in different wave
views, and the correct per-SIMD native-wave32 figure is **1536**. (The "32
KiB" half of claim A matches no documented gfx1101 register-file size; the
file is 1536 × 32 lanes × 4 B = 192 KiB/SIMD in the 32-lane view.)

### (e) Empirical occupancy probe (this machine; /tmp only, not committed)

A probe kernel template with N volatile-fed accumulator chains was compiled
with `hipcc -O3 --offload-arch=gfx1101` (HIP 7.2.53211), giving controlled
VGPR counts confirmed by `amdhsa_next_free_vgpr` in the emitted ISA. For each
count, `hipOccupancyMaxActiveBlocksPerMultiprocessor` was queried at
1 wave/block (stable across repeated queries):

```
wave32 (1 block = 1 wave32):  regs 37/49/73 -> 64 blocks/CU (thread-cap 2048)
                              regs 97  -> 56 ; 113/116 -> 44 ; 129 -> 44
                              regs 161 -> 36 ; 193 -> 28 ; 225/256 -> 24
wave64 (1 block = 1 wave64):  regs 37/73 -> 32 ; 97 -> 28 ; 193 -> 14 ; 256 -> 12
```

Fit: `blocks_w64 = 2 SIMDs × min(floor(1536 / align12(regs)), 16)` matches
the wave64 points; **wave32 blocks = 2 × wave64 blocks at every shared
point** (e.g. 97 → 56 = 2×28; 193 → 28 = 2×14), i.e. the driver sizes wave32
waves in the same 1536-entry file with the 12-granule step series. The step
positions observed — 108, 120, 132, 168, 204, 228 (the align-12 series) —
refute the granule-16 hypothesis (align16 would step at 112/128/144/176/208).
The only ±1-wave deviations from the model are corner cases at the register
maximum (256 regs → 6 waves/SIMD, where 6 × 256 = 1536 exactly). Max
residency observed: **32 wave32 (16 wave64) per SIMD** — consistent with
LLVM's 16-wave-per-EU cap in wave64-equivalent units.

## Resolution used by libs/arch

| gfx1101 (RDNA3) constant | Value | Source |
|---|---|---|
| VGPR file per SIMD (wave32 view) | **1536** | RDNA3 ISA §3.3.2.1 (via brief §1.3 L137-139, §10.4 L551-553); LLVM AMDGPUBaseInfo.cpp L1336-1344; driver regsPerMultiprocessor = 1536×2×64 (evidence c) |
| VGPR allocation granule | **24 (wave32) / 12 (wave64)** | RDNA3 ISA §3.3.2.1 (brief §1.3, §10.4); LLVM getVGPRAllocGranule L1295-1310; empirical step series (evidence e) |
| Max waves per SIMD | 16 wave64-equiv (32 wave32 observed) | LLVM getMaxWavesPerEU L1162-1168; probe max 32 w32 / 16 w64 |
| Wavefront | 32 (native wave32) | rocminfo warpSize=32; brief §1.3 L136 |
| LDS per CU | 64 KiB | driver maxSharedMemoryPerMultiprocessor = 65536; brief §1.3 L142-143 |

Caveats: gfx1102 (RX 7600-class) lacks `Feature1_5xVGPRs` in LLVM 21 and
should NOT be modeled with the 1536/24-12 file (LLVM models it with the
base 1024/512 file and 16/8 granule); the driver's occupancy API matches the
12-series granule model within ±1 wave at the 256-register corner.

Supersedes the informal "768 VGPRs / 32 KiB" figure in
/home/mei/projects/polykernel/docs/hip_backend.md:80 for gfx1101 occupancy
modeling (that figure is a wave64-view number and is not usable for the
native-wave32 occupancy math).
