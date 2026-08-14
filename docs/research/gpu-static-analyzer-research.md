# Static Performance Analyzer for PTX / AMDGPU Assembly — Research Brief

**Date:** 2026-08-13 · **Scope:** official ISA docs, CUDA Programming Guide, PTX ISA, peer-reviewed
microbenchmark papers, reputable GitHub tooling. Every constant below carries a citation;
conflicts between sources are flagged in §10.

Abbreviations used in citations:
- **[CPG]** CUDA C++ Programming Guide v13.3 (updated 2026-05-27), https://docs.nvidia.com/cuda/cuda-programming-guide/ (PDF: `.../pdf/cuda-programming-guide.pdf`)
- **[PTX]** PTX ISA reference, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- **[NVCC]** CUDA Compiler Driver NVCC docs, https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html
- **[VOLTA]** Jia et al., *Dissecting the NVidia Volta GPU Architecture via Microbenchmarking*, arXiv:1804.06826
- **[TURING]** Jia et al., *Dissecting the NVidia Turing T4 GPU via Microbenchmarking*, arXiv:1903.07486
- **[HOPPER]** Luo et al., *Dissecting the NVIDIA Hopper Architecture through Microbenchmarking and Multiple Level Analysis*, arXiv:2501.12084 (IPDPS'24 version: doi:10.1109/IPDPS.2024.00070; code: https://github.com/HPMLL/NVIDIA-Hopper-Benchmark)
- **[A100WP]** NVIDIA A100 Tensor Core GPU Architecture whitepaper, https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
- **[HOPPERBLOG]** NVIDIA Hopper Architecture In-Depth, https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
- **[CDNA2]** AMD Instinct MI200 (CDNA2) ISA, https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf
- **[CDNA3]** AMD Instinct MI300 (CDNA3) ISA (rev 2025-08-05, 553 pp), https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
- **[RDNA3]** AMD RDNA3 ISA, doc #70650, https://docs.amd.com/v/u/en-US/rdna3-shader-instruction-set-architecture-feb-2023_0 (mirror: vgamuseum.info/images/doc/amd/rdna3-shader-instruction-set-architecture-feb-2023.pdf)
- **[CDNA3WP]** AMD CDNA3 whitepaper, https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf
- **[OCCBLOG]** ROCm blog, *Occupancy Math on the AMD MI355X GPU (CDNA4)* (2026-07), https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html
- **[LLVM]** LLVM project, commit `27eeb73` (AMDGPU backend), https://github.com/llvm/llvm-project
- **[OMNIPERF]** ROCm Compute Profiler (omniperf) performance model docs, https://rocm.github.io/rocprofiler-compute/performance_model.html
- **[PCOUNTERS]** MI300/MI200 performance counters, https://rocm.docs.amd.com/en/latest/reference/gpu-arch/mi300-mi200-performance-counters.html
- **[MODEL26]** Jarmusch & Chandrasekaran, *Microbenchmark-Driven Analytical Performance Modeling Across Modern GPU Architectures*, arXiv:2605.04178 (2026)
- **[MALL25]** Tee et al., *The MALL is Open: Exploring Shared Caches and Latency in AMD CDNA3 GPUs*, MEMO'25 @ SC25, https://escholarship.org/uc/item/3b8311qq

---

## 1. Occupancy model

### 1.1 General formula (both vendors)

Occupancy = resident warps (waves) ÷ hardware-max warps (waves) per SM (CU). The achievable
number of blocks per SM is the **min** over resource limiters:

```
warps_per_block  = ceil(block_threads / warp_size)                    # warp_size: NV=32 [CPG Table 30], AMD=64 (CDNA) / 32 (RDNA3 native)
limit_threads    = floor(max_threads_per_SM / block_threads)
limit_warps      = floor(max_warps_per_SM  / warps_per_block)
limit_smem       = floor(smem_per_SM       / smem_per_block)
limit_regs       = floor(regs_per_SM       / (regs_per_thread × block_threads))   # with allocation granularity, see below
limit_blocks     = max_blocks_per_SM (hardware cap)
blocks_resident  = min(all of the above)
occupancy        = blocks_resident × warps_per_block / max_warps_per_SM
```

NVIDIA side: the CUDA guide defines occupancy as "the ratio of the number of active warps to the
maximum number of active warps supported by the SM" and walks worked examples using exactly the
device-property limits above (`maxBlocksPerMultiProcessor`, `sharedMemPerMultiprocessor`,
`regsPerMultiprocessor`, `maxThreadsPerMultiProcessor`, `sharedMemPerBlock`, `regsPerBlock`)
**[CPG §2.3.7 "Kernel Launch and Occupancy"]**. The runtime API
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` implements this **[CPG §2.3.7 / archived guide §5.2.3.1]**.

AMD side: occupancy = min of four limiters — VGPRs, SGPRs, LDS, workgroup/barrier slots — then
clamped by hardware caps (8 waves/SIMD, 32 waves/CU on CDNA) **[OCCBLOG, Part 2]**. LLVM's AMDGPU
backend computes the same thing: `GCNSubtarget::computeOccupancy()` takes the min of
workgroup-size-derived occupancy, `getOccupancyWithNumSGPRs`, and `getOccupancyWithNumVGPRs`
**[LLVM: llvm/lib/Target/AMDGPU/GCNSubtarget.cpp L515-531,
https://github.com/llvm/llvm-project/blob/27eeb7370281e4fdefdddeaa4b3cbb8ca11ea71f/llvm/lib/Target/AMDGPU/GCNSubtarget.cpp#L515-L531]**.
LLVM now ships a standalone tool, **`llvm-calc-occupancy`**, "a thin front-end over the same
occupancy math the AMDGPU backend uses (GCNSubtarget)" **[LLVM docs:
https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html]** — e.g. for gfx90a it reports
wavefront size 64, max 8 waves/EU (SIMD), 4 EUs/CU. A static analyzer can literally shell out to
it or re-implement the same formulas.

### 1.2 NVIDIA per-architecture constants (sm_75 → sm_120)

From **[CPG, Appendix "Compute Capabilities", Tables 30/31/32]**
(https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html).
"KB" = KiB in the guide. All listed compute capabilities: warp size 32; 64K 32-bit registers/SM;
64K registers max per thread block; **255 registers max per thread**; 512 KB local memory/thread;
64 KB constant memory; **32 shared-memory banks**; max 1024 threads/block.

| CC (arch) | max threads/SM | max warps/SM | max blocks/SM | smem/SM | smem/block (opt-in max) | unified L1+smem | smem carve-out options |
|---|---|---|---|---|---|---|---|
| 7.5 (Turing TU1xx) | 1024 | 32 | 16 | 64 KB | 64 KB | 96 KB | 32, 64 |
| 8.0 (A100, GA100) | 2048 | 64 | 32 | 164 KB | 163 KB | 192 KB | 0,8,16,32,64,100,132,164 |
| 8.6 (GA10x) | 1536 | 48 | 16 | 100 KB | 99 KB | 128 KB | 0,8,16,32,64,100 |
| 8.7 (Orin GA10B) | 1536 | 48 | 16 | 164 KB | 163 KB | 192 KB | 0,8,16,32,64,100,132,164 |
| 8.9 (Ada AD10x) | 1536 | 48 | 24 | 100 KB | 99 KB | 128 KB | 0,8,16,32,64,100 |
| 9.0 (Hopper H100) | 2048 | 64 | 32 | 228 KB | 227 KB | 256 KB | 0,8,…,228 |
| 10.0/10.3 (Blackwell datacenter B100/B200/B300) | 2048 | 64 | 32 | 228 KB | 227 KB | 256 KB | 0,8,…,228 |
| 11.0 (next datacenter) | 2048 | 64 | 32 | 228 KB | 227 KB | 256 KB | 0,8,…,228 |
| 12.x (Blackwell consumer RTX 50) | 1536 | 48 | 24 | 100 KB | 99 KB | 128 KB | 0,8,16,32,64,100 |

(Merged-cell mapping in CPG Tables 30-32: 8.6/8.7/8.9 share the 1536-thread row; 9.0/10.x/11.0
share the 2048-thread row. Blocks/SM: 7.5=16, 8.0=32, 8.6/8.7=16, 8.9=24, 9.0/10.x/11.0=32,
12.x=24.) The guide's own CC 10.0 worked example lists `sharedMemPerMultiprocessor = 233472`
bytes = 228 KiB, confirming the table **[CPG §2.3.7, Table 2]**.

**Register allocation granularity.** NVIDIA's current documentation no longer publishes the warp
allocation granule. The best citable statement is from microbenchmarking: "Turing and Volta use a
physical register file of 16,384 32-bit elements in each processing block. Thread-visible logical
registers are allocated in increments of 8, or aggregate increments of 256 for all 32 threads in a
warp" **[TURING §3.5.1]**. Hence the analyzer should use `regs_per_warp = ceil8(regs_per_thread) × 32`
(equivalently round per-thread usage up to a multiple of 8 → 256-register warp granule), and
4 sub-partitions × 16,384 = 65,536 registers/SM. Note [TURING] also shows the register file is
2-bank × 64-bit on Volta/Turing (vs 4-bank × 32-bit on Pascal/Maxwell), which drives register bank
conflicts at SASS level **[TURING §3.5.1, VOLTA §3.5]**.

Worked register-limited example (sm_90): kernel using 128 regs/thread, 256-thread blocks →
128×256 = 32,768 regs/block → floor(65,536/32,768) = 2 blocks → 16/64 warps = 25% occupancy.

### 1.3 AMD constants

**CDNA2 (MI200/MI210, gfx90a) and CDNA3 (MI300, gfx94x):**
- CU = 4 SIMDs ("execution units"); wavefront = 64 **[CDNA2 ch.1; llvm-calc-occupancy gfx90a output]**.
- **VGPRs:** "VGPRs are allocated in groups of eight Dwords… VGPRs are allocated out of two pools:
  regular VGPRs and accumulation VGPRs… A wave may have up to 512 total VGPRs, 256 of each type.
  When a wave has fewer than 512 total VGPRs, the number of each type is flexible"
  **[CDNA2 §3.6.4; CDNA3 §3.6.4 — identical wording]**.
  ⇒ VGPR limiter: `waves_per_SIMD = floor(512 / ceil8(vgpr_total))`, where `vgpr_total = arch + acc`.
  Equivalent threshold table (as used by ROCm's Triton occupancy script): ≤64→8 waves, 65–72→7,
  73–80→6, 81–96→5, 97–128→4, 129–168→3, 169–256→2, >256→1 wave/SIMD
  **[ROCm/triton scripts/amd/occ.sh,
  https://github.com/ROCm/triton/blob/cf44637139ba441342b909977f51f7cfec2c9963/scripts/amd/occ.sh]**.
- **SGPRs:** "A wavefront can be allocated 16 to 102 SGPRs, in units of 16 GPRs (Dwords)"
  **[CDNA3 §3.6.2-3.6.3]**. The per-SIMD SGPR pool size is not stated in the ISA docs; LLVM and the
  ROCm blog use ≈800 SGPRs/SIMD (llvm-calc-occupancy example: gfx90a, 30 SGPRs → SGPR limiter 8
  waves, i.e. floor(800/30)=26 clamped by the 8-wave cap; [OCCBLOG] states "~800 per SIMD,
  ≤102/wave" for CDNA4 and "SGPR granularity 16").
- **Max waves:** 8 waves/SIMD, 32 waves/CU **[llvm-calc-occupancy gfx90a output; OCCBLOG]**.
- **LDS:** 64 kB per CU **[CDNA2 glossary & §11; CDNA3 §12]**. Allocation granularity: CDNA2
  "contiguous blocks of 128 Dwords on 128-Dword alignment" (=512 B) **[CDNA2 §3.6.5]**; CDNA3
  "contiguous blocks of 512 bytes on 512-byte alignment" **[CDNA3 §3.6.5]**.
  ⇒ LDS limiter: `workgroups_per_CU = floor(64 KB / ceil512B(lds_per_workgroup))`, then convert to
  waves/SIMD via the workgroup's wave count spread one-per-SIMD **[OCCBLOG Part 2]**.
- **Workgroup/barrier slots** are a fourth limiter (hard cap + barrier slots per CU) **[OCCBLOG]**.
- MI300 packaging: XCD = 40 CUs (38 active), 4 MB L2 per XCD; MI300X = up to 8 XCDs (304 CUs),
  256 MB Infinity Cache, 5.3 TB/s HBM3 **[CDNA3WP; ROCm MI300 microarchitecture page,
  https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300.html]**.

**RDNA3 (gfx1100–1102):**
- Hierarchy: WGP = 2 CUs; "CU — one half of a WGP. Contains 2 SIMD32's that share one path to
  memory"; waves of a work-group are issued to the same WGP and can run on any of its 4 SIMD32s
  **[RDNA3 glossary, §"Compute Unit", §12.1.2]**. Native wave32 (wave64 supported).
- **VGPRs:** "VGPRs are allocated in blocks of 16 for wave32 or 8 for wave64, and a shader may have
  up to 256 VGPRs… Devices which have 1536 VGPRs per SIMD allocate in blocks of 24 for wave32 and
  12 for wave64" **[RDNA3 §3.3.2.1]**. ⇒ `waves_per_SIMD = floor(1536 / ceil_granule(vgpr_per_lane))`,
  clamped by the hardware wave cap (LLVM's GCNSubtarget max-waves-per-EU for gfx11xx is 16 —
  as implemented in LLVM; the ISA doc itself does not print a waves/SIMD cap).
- **LDS:** "128 kB of memory per work-group processor split up into 64 banks of DWORD-wide RAMs…
  One work-group can request up to 64 kB memory" **[RDNA3 §12.1]** (see §2.2).
- **Work-group size:** up to 1024 work-items = 16 wave64s or 32 wave32s **[RDNA3 §…work-groups]**.

### 1.4 Occupancy-related tooling to reuse

- `cudaOccupancyMaxActiveBlocksPerMultiprocessor` / `cudaOccupancyMaxPotentialBlockSize` **[CPG §2.3.7]**.
- `llvm-calc-occupancy` (AMDGPU; `--print-limits` prints per-occupancy VGPR/SGPR ceilings)
  **[https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html; tool source:
  https://github.com/llvm/llvm-project/blob/11d7c9c9b4e8ebaf5f9bfb6ba80aecd498aa25ca/llvm/tools/llvm-calc-occupancy/llvm-calc-occupancy.cpp]**.
- HIP occupancy API mirrors CUDA (`hipOccupancyMaxActiveBlocksPerMultiprocessor`) **[HIP docs,
  https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/occupancy.html]**.

---

## 2. Shared-memory banks

### 2.1 NVIDIA: 32 banks × 4 bytes, broadcast rules, padding

"Shared memory has 32 banks that are organized such that successive 32-bit words map to successive
banks. Each bank has a bandwidth of 32 bits per clock cycle. When multiple threads in the same warp
attempt to access different elements in the same bank, a bank conflict occurs… the access… will be
serialized. The two exceptions… happen when multiple threads in the same warp are accessing the same
shared memory location. For read accesses, the word is **broadcast** to the requesting threads. For
write accesses, each shared memory address is written by only one of the threads (which thread…
undefined)." **[CPG §2.3.4.2 "Shared Memory Access Patterns"]**. The guide illustrates stride
conflicts (stride-1 conflict-free, stride-2 two-way conflict, stride-3 conflict-free) and broadcast
cases (multiple threads reading one word = conflict-free) **[CPG Figs. 15-16]**. 32 banks per SM is
also in CPG Table 31; [VOLTA Table 3.1 / TURING Table 2] measure 32 banks × 4 B on every
architecture tested (Kepler GK210 bank width 8 B is the exception in [TURING Table 2]).

**Padding idiom:** the canonical fix for column-wise (transpose) conflicts is padding the tile row,
e.g. `__shared__ float tile[TILE_DIM][TILE_DIM+1]` — the guide's matrix-transpose section shows a
32-way conflict on unpadded column access and eliminates it with the +1 pad **[CPG §2.3.4.2.1,
Figs. 17-19]**; the same idiom lives in NVIDIA's cuda-samples transpose kernel
(https://github.com/NVIDIA/cuda-samples, Samples/0_Introduction/simpleTransposeShared). On Hopper,
TMA adds hardware **swizzle** modes (`CU_TENSOR_MAP_SWIZZLE_128B`) specifically to avoid bank
conflicts when staging tiles **[CPG §4.11.2.2.5 "Shared-Memory Bank Swizzling"]**.

Static-analysis rule set: for each `ld.shared`/`st.shared` (PTX) with a derivable per-lane address
affine in `threadIdx`, compute `bank = (addr/4) mod 32` over the warp; count multiplicity per bank;
same-address reads = broadcast (free), same-address writes = undefined-but-1. This is exactly the
model Nsight Compute's shared-memory analysis assumes (see §7).

### 2.2 AMD: how LDS differs

**CDNA2/CDNA3:** "There are 64 kB memory per compute unit, segmented into 32 banks of 512 Dwords,
each bank being 32 bits wide. Dwords are placed in the banks serially, but all banks can execute a
store or load simultaneously… if more than one access attempt is made to the same bank at the same
time, a bank conflict occurs. In this case, for indexed and atomic operations, hardware prevents the
attempted concurrent accesses to the same bank by serializing them… operations can complete in as
little as two cycles, or take as many as 64 cycles, depending upon the number of bank conflicts"
**[CDNA3 §12.1 "Overview"; near-identical in CDNA2 §11]**. LDS also has built-in atomic/ALU ops and
GDS export paths **[CDNA3 §12.5]**.

Key differences vs NVIDIA for the analyzer:
1. Same 32-bank × 4-byte interleaving, but the access group is a **wave64** (64 lanes), so conflict
   counting spans 64 lanes (two 32-lane halves can be serviced by different phases); ROCm's
   profiler exposes `LDSBankConflict` = "% of GPU time LDS is stalled by bank conflicts"
   **[PCOUNTERS derived metrics]**.
2. LDS is **per CU** (one shared pool), allocated to workgroups in 512 B granules **[CDNA3 §3.6.5]**.
3. **RDNA3:** 128 kB per **WGP**, "64 banks of DWORD-wide RAMs… sub-divided into two sets of
   32-banks each where 32 of the banks are affiliated with a pair of SIMD32's, and the other 32
   banks are affiliated with the other pair of SIMD32's within the WGP. Each bank is a 512×32
   two-port RAM (1R/1W per clock cycle)" **[RDNA3 §12.1]**. In CU-mode dispatch the LDS is
   "effectively split into a separate upper and lower LDS, each serving two SIMD32's" **[RDNA3
   §12.1.2]** — so conflict analysis must account for which half of the LDS a workgroup lands in,
   and wave32 groups map to one SIMD32 pair.
4. AMD LDS supports 2-address (64-bit) load/store forms (`load_2addr/store_2addr`) **[RDNA3 §12.1]**.

---

## 3. Coalescing

### 3.1 NVIDIA rules

"Global memory is accessed via **32-byte memory transactions**. When a CUDA thread requests a word
of data from global memory, the relevant warp coalesces the memory requests from all the threads in
that warp into the number of memory transactions necessary to satisfy the request… if consecutive
threads in the warp request consecutive 4-byte words in memory, then the warp will request 128 bytes
of memory total, and this 128 bytes… will be fetched in **four 32-byte memory transactions**…
100% utilization. …The pathologically worst case… consecutive threads access data elements that are
32 bytes or more apart… the warp will be forced to issue a 32-byte memory transaction for each
thread… 1024 bytes… utilization 12.5%." **[CPG §2.3.4.1 "Coalesced Global Memory Access", Figs.
12-13]**.

So the static rules are:
- **Fully coalesced:** warp accesses a contiguous, 128 B-aligned span (32 lanes × 4 B consecutive)
  → 4 × 32 B sectors = 1 line.
- **Not 128 B coalesced:** stride ≥ 32 B per lane (each lane touches a distinct 32 B sector) → 32
  transactions; or misaligned/odd-stride patterns → sector count = number of distinct 32 B sectors
  touched. The analyzer can compute `sectors = |{ floor(addr_lane / 32) }|` and efficiency
  `= useful_bytes / (sectors × 32)`.
- Wider accesses: `ld.global.v4.f32` per thread = 4 sectors per thread-group; the guide's local
  memory note matters for spills: "local memory is organized such that consecutive 32-bit words are
  accessed by consecutive thread IDs. Accesses are therefore fully coalesced as long as all threads
  in a warp access the same relative address" **[CPG §2.3.3.4]**.

**Cache geometry per arch (measured/official):**

| Arch | L1 line / load granule (measured) | L2 line | L1 hit | L2 hit | Source |
|---|---|---|---|---|---|
| Kepler GK210 | 128 B line / 128 B load | 32 B | 35 cyc | ~200 cyc | [TURING Table 2] |
| Maxwell GM204 | 32 B | 32 B | 82 cyc | ~207 cyc | [TURING Table 2] |
| Pascal GP100/GP104 | 32 B | 32 B | 82 cyc | ~234/~216 cyc | [TURING Table 2] |
| Volta GV100 | 32 B line, 32 B load granule, 128 B update granule | 64 B | 28 cyc | ~193 cyc | [VOLTA Table 3.1] |
| Turing TU104 | same as Volta | 64 B | 32 cyc | ~188 cyc | [TURING Table 2, Fig. 6] |
| Ampere A100 | — (whitepaper: 192 KB combined L1/smem per SM) | — | 33 cyc | 202.8–408 cyc (partitioned) | [A100WP]; [HOPPER Table 3] |
| Hopper H100/H800 | 256 KB combined L1/smem per SM | 50 MB L2 (H100 SXM5) | 32 cyc | 264.5–502 cyc (partitioned L2, near/far) | [HOPPERBLOG]; [HOPPER Table 3] |
| Ada RTX 4090 | 128 KB combined | — | 32 cyc | 273 cyc | [HOPPER Table 3] |
| Blackwell B200 | TMEM 256 KB/SM; model cache latencies 5/50/150/400 cyc (L1/L2/LLC/HBM) | — | — | — | [MODEL26] |

Note the tension: NVIDIA's programmer-facing docs speak in **32-byte sectors/transactions** [CPG],
while microbenchmarks measure **32 B load granularity with 128 B update granularity** on Volta/Turing
and 64 B L2 lines ([VOLTA Table 3.1], [TURING Table 2]); "128 B cache line" is the *write/update*
and L1-tag granularity on those archs, and common lore. Hopper's L2 is physically partitioned with
near/far latency spread **[HOPPER §4.1]**. Flag this in the report rather than asserting one number.

### 3.2 AMD coalescing

- CDNA vL1D (vector L1 data cache) services wave memory requests; omniperf reports a per-kernel
  **Coalescing metric "ranging from uncoalesced (25%) to fully coalesced (100%)"** in the vL1D
  section **[OMNIPERF, vL1D chapter]** — i.e., the hardware's natural unit makes a fully uncoalesced
  wave 4× worse, consistent with 64 B lines served as 4×16 B or 32 B sectors (the docs do not print
  the sector size; treat 25% as the documented floor).
- L2 on MI100/MI200: "32 channels… at 256 B address interleaving" **[OMNIPERF L2 chapter]**; CDNA3
  L2 channels are "64-bytes wide" (16 channels per 4 MB L2 per XCD) **[CDNA3WP memory-hierarchy
  section]**; counter `WRITE_REQ_32B` counts "32-byte effective memory writes" **[PCOUNTERS]** —
  evidence that 32 B is the effective write sector on CDNA.
- Pointer-chasing latencies for MI300X/MI300A/MI250X (sL1, L2, MALL/Infinity Cache, HBM) are
  characterized in **[MALL25]**; NPS partitioning cuts MALL latency up to 1.42×.

Static rule for AMD: same sector-count analysis per wave64 (or wave32 on RDNA3), but note wave64
spans 64 lanes → a perfectly consecutive float access by a wave64 touches 256 B = 4×64 B lines.

---

## 4. Warp divergence (static detection)

### 4.1 Execution-model facts the analyzer relies on

- "A warp executes one common instruction at a time… If threads of a warp diverge via a
  data-dependent conditional branch, the warp executes each branch path taken, disabling threads
  that are not on that path. Branch divergence occurs only within a warp" **[CPG §3.2.2.1]**.
- **Independent Thread Scheduling (CC ≥ 7.0):** "the GPU maintains execution state per thread,
  including a program counter and call stack, and can yield execution at a per-thread granularity…
  threads can now diverge and reconverge at sub-warp granularity." Warp-synchronous assumptions from
  pre-Volta are invalid; use `__syncwarp()` **[CPG §3.2.2.1.1]**. Implication for a static analyzer:
  on sm_70+ you cannot assume lockstep reconvergence at the immediate post-dominator for
  *correctness* reasoning (deadlock/livelock analysis must account for per-thread scheduling), but
  the *cost* model (both paths executed serially) still holds; also SMs "issue instructions in order
  and do not perform branch prediction or speculative execution" **[CPG §3.2.2]**.
- PTX exposes explicit convergence ops: `bar.warp.sync membermask` (waits until all threads in
  membermask have arrived) **[PTX §9.7.14.2]** and block barriers `bar.sync` — "Each CTA instance
  has sixteen barriers numbered 0..15" **[PTX §9.7.14.1]**. Predicated execution (`@p` predicates)
  is the other divergence vehicle: predicated-off instructions still consume issue slots.
- AMD: divergence is visible in the `VALUUtilization` metric — "Percentage of active vector ALU
  threads in a wave, where a lower number can mean either more thread divergence in a wave or that
  the work-group size is not a multiple of 64" **[PCOUNTERS]**. AMD waves execute under an EXEC
  mask; reconvergence is compiler-managed (LLVM "linearizes" divergent branches — see below).

### 4.2 Static detection recipe (from PTX / AMDGPU asm)

1. Rebuild the per-kernel CFG from branch instructions (`@p bra`, `bra`, conditional `s_cbranch_*`
   on AMD).
2. A branch is **potentially divergent** iff its condition (or any predicate feeding it) is not
   provably uniform across the warp/wave. Uniform sources: `blockIdx`-derived values, kernel
   parameters, constants; divergent sources: `threadIdx`-derived values, memory loads, data-dependent
   comparisons. This is precisely LLVM's **UniformityAnalysis**, which "determines the uniformity of
   value(s) computed at each instruction" and is used by the NVPTX/AMDGPU backends; it extends the
   abstract-interpretation divergence analysis of **Rosemann, Moll & Hack, "An Abstract Interpretation
   for SPMD Divergence on Reducible Control Flow Graphs," POPL 2021 (doi:10.1145/3434312)** to
   irreducible CFGs **[LLVM docs, https://llvm.org/docs/ConvergenceAndUniformity.html]**.
3. Reconvergence point = **immediate post-dominator** of the branch (classical SIMT stack behavior);
   cost of a divergent branch ≈ sum of costs of taken paths (serialize), with predicated instructions
   counted at full issue cost. Loop-carried divergence (thread-dependent trip counts) is the dominant
   case in real kernels — flag loops whose bounds depend on divergent values.
4. Barrier rules: `bar.sync`/`__syncthreads()` requires **all** threads of the block to arrive —
   a barrier inside a divergent region where some threads never reach it is UB/deadlock (detect:
   barrier not post-dominated by entry of its enclosing divergent region). `bar.warp.sync` masks let
   subsets converge **[PTX §9.7.14]**. On sm_70+, also warn about warp-synchronous idioms lacking
   `__syncwarp`/`bar.warp.sync` **[CPG §3.2.2.1.1]**.
5. Prior art for the SASS level: Ma et al. reverse-engineered Maxwell/Volta front-end behavior
   (SSY/SYNC reconvergence stack) — useful if the analyzer ever ingests SASS: Ma, Wang, et al.,
   "Understanding the GPU Microarchitecture to Achieve Bare-Metal Performance," PPoPP 2017.
   For verification-grade divergence analysis see GPUVerify (O'Sullivan et al., OOPSLA 2012).

---

## 5. Tensor cores / MFMA / WMMA

### 5.1 NVIDIA PTX families per architecture

| Family | Intro | Shapes (examples) | Target arch (per PTX ISA) |
|---|---|---|---|
| `wmma.{load,store,mma}` | PTX 6.0 / sm_70 | m16n16k16 (f16→f16/f32); m8n8k128/m16n8k128/m16n8k256 (b1); int8/int4 on sm_75 | gen-1 Tensor Cores **[PTX §9.7.15.4]** |
| `mma.sync.aligned` | sm_70+ | f16 **m8n8k4** requires sm_70; f16 **m16n8k8**, s8/u8 **m8n8k16** require sm_75; **m16n8k16 f16, m16n8k8/k16 bf16, m16n8k4/k8 tf32, m8n8k4 f64, m16n8k16/k32 s8, m16n8k32/k64 s4, m16n8k128/k256 b1** require sm_80; **e4m3/e5m2 (FP8)** requires sm_89; **e3m2/e2m3/e2m1 + block-scale (`.kind`, `.block_scale`, `.scale_vec_size`)** require sm_120a (sm_120f from PTX 8.8) | **[PTX §9.7.15.5 "Target ISA Notes"]** |
| `mma.sp` (2:4 structured sparsity) | PTX 7.1 / sm_80+ | s8 m16n8k32/k64, s4 m16n8k64/k128, f16/bf16, fp8 (PTX 8.4), ordered-metadata variants (PTX 8.5) | **[PTX §9.7.15.5 sparse table, lines ~18079-18101]** |
| `wgmma.mma_async` (warpgroup, async) | PTX 8.0 / **sm_90a** | f16/bf16 **m64nNk16** (N=8…256 step 8), tf32 **m64nNk8**, fp8 **m64nNk32**, s8 **m64nNk32**, b1 m64nNk256; sparse variants k×2 | **[PTX §9.7.16, shape table §9.7.16.2]**; async-proxy semantics **[PTX §9.7.16.4; CPG §3.2.2.3.1]** |
| `tcgen05.*` (5th-gen TensorCore + Tensor Memory) | PTX 8.6 / **sm_100a, sm_101a(→sm_110a), family sm_100f+** | `tcgen05.mma` kinds tf32/f16/f8f6f4/i8/mxf8f6f4 (Table 42); K=96 only sm_103a; TMEM = 512 columns × 128 lanes × 32-bit per CTA on sm_100a/sm_100f; alloc unit 32 columns | **[PTX §9.7.17, Tables 42-46, Target ISA Notes]** |

Tensor-core input dtypes per CC are tabulated in **[CPG Table 33]** (7.5: f16/s8/s4; 8.0 adds
f64/tf32/bf16; 8.9 adds FP8; 10.0 adds FP6/FP4; 12.x: tf32/bf16/f16/fp8/fp6/fp4/int8).

**MAC counting for the report:** each instruction computes M×N×K multiply-accumulates
(2·M·N·K FLOPs): `mma.m16n8k16.f16` = 2,048 MACs/warp; `wgmma.m64n128k16` = 131,072 MACs/warpgroup;
`tcgen05.mma` per Table 42 shape; sparsity variants halve stored A operands but keep the same MAC
count (effective throughput ×2 vs dense at same K). A static tool sums per-instruction MACs weighted
by static/dynamic loop trip counts (when derivable) to get a tensor-vs-VALU work split.

### 5.2 AMD MFMA (CDNA2/CDNA3) and WMMA (RDNA3)

**CDNA2 (MI200/MI210)** — VOP3P-MAI opcodes **[CDNA2 §7.1, Table 25]**:
- `V_MFMA_F32_{32x32x1, 16x16x1, 4x4x1, 32x32x2, 16x16x4}F32`
- `V_MFMA_F32_{32x32x4, 16x16x4, 4x4x4, 32x32x8, 16x16x16}F16`
- `V_MFMA_I32_{32x32x4, 16x16x4, 4x4x4, 32x32x8, 16x16x16}I8`
- `V_MFMA_F32_{32x32x2, 16x16x2, 4x4x2, 32x32x4, 16x16x8}BF16`
- `V_MFMA_{*}_BF16_1K` (4x4x4, 16x16x4, 16x16x16, 32x32x4, 32x32x8) — the "_1K" CDNA2 variants
- `V_MFMA_F64_{16x16x4, 4x4x4}F64`
Accumulation can live in a separate **AccVGPR (AGPR)** file; `V_ACCVGPR_READ/WRITE` move data
between files **[CDNA2 §7 preamble]**.

**CDNA3 (MI300)** — new naming with block suffix `_nB` and doubled rates **[CDNA3 §7.1, Table at
~L2482-2510]**:
- `V_MFMA_F32_*_F32` 32x32x1_2B / 16x16x1_2B (2 blocks, 64-pass)
- `V_MFMA_F32_*_F16` 32x32x4_2B, `V_MFMA_F32_*_BF16` 32x32x4_2B, `V_MFMA_I32_*_I8` 32x32x4_2B
- `V_MFMA_F32_*_XF32` 16x16x8 (TF32-like, 10-bit mantissa inputs **[CDNA3 §7 notes ~L3113]**)
- `V_MFMA_F64_*_F64` 16x16x4
- **FP8:** `V_MFMA_F32_*_{BF8_BF8, BF8_FP8, FP8_BF8, FP8_FP8}` 16x16x32 / 32x32x16
- **Sparse MFMA:** `V_SMFMA_*` (2:4 sparsity) **[CDNA3 §7.1, dependency tables ~L3300-3480]**
MAC counting: M×N×K per instruction per wave (e.g. `V_MFMA_F32_32x32x8F16` = 8,192 MACs).
AMD ships the **AMD Matrix Instruction Calculator** for per-instruction throughput/register usage
(https://github.com/ROCm/amd_matrix_instruction_calculator, cited inside [RDNA3 §7.9]).

**RDNA3 WMMA** **[RDNA3 §7.9, Table 33]** — all 16×16×16, VOP3P-encoded:
`V_WMMA_F32_16X16X16_F16`, `V_WMMA_F32_16X16X16_BF16`, `V_WMMA_F16_16X16X16_F16`,
`V_WMMA_BF16_16X16X16_BF16`, `V_WMMA_I32_16X16X16_IU8`, `V_WMMA_I32_16X16X16_IU4`
(4,096 MACs each; lanes 0-15 of A/B replicated into 16-31; back-to-back dependent WMMAs need a NOP
when D overlaps next A/B **[RDNA3 §7.9]**; GPUOpen guide: https://gpuopen.com/learn/wmma_on_rdna3/).

---

## 6. Spill detection

### 6.1 NVIDIA / PTX

- PTX `.local` state space: "The local state space (.local) is private memory for each thread to
  keep its own data… Use ld.local and st.local to access local variables… .local state-space
  variables must be declared within function scope and are allocated on the stack" **[PTX §5.1.5]**.
  Stack frame declarations look like `.local .b8 __func_frame[NNN];` (byte arrays).
- CUDA semantics: "Local memory is thread local storage… physical location… is in the global memory
  space… local memory accesses have the same latency and bandwidth as global memory accesses" but
  are laid out so same-relative-address access across a warp is fully coalesced **[CPG §2.3.3.4]**.
  Spill causes listed in [CPG §2.3.3.4]: non-constant-indexed arrays, large structs/arrays, and
  register overflow.
- **Compiler reporting (exact format)** — `nvcc --resource-usage` (which "implies
  `--ptxas-options=--verbose`") **[NVCC §4.2.8.16]** prints **[NVCC §8.4 "Printing Code Generation
  Statistics"]**:
  ```
  ptxas info    : Compiling entry function 'acos_main' for 'sm_80'
  ptxas info    : Function properties for acos_main
      0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
  ptxas info    : Used 6 registers, 1536 bytes smem, 32 bytes cmem[0]
  ```
  Plus diagnostic flags `--warn-on-local-memory-usage (-warn-lmem-usage)` and `--warn-on-spills
  (-warn-spills)` **[NVCC §4.2.9.1.28-29]**. A static analyzer ingesting PTX directly should:
  (a) sum `.local` declarations per function (frame bytes), (b) count/size `ld.local`/`st.local`
  (spill traffic), (c) cross-check against ptxas numbers when a cubin build is available.
- Register cap pressure: `-maxrregcount` forces spills **[CPG §2.3.3.3, §2.3.7]**.

### 6.2 AMD / GCN-RDNA

- Scratch (private) memory: per-wave `FLAT_SCRATCH` base (SGPRs 102/103 on CDNA3:
  `FLAT_SCRATCH_LO/HI`) **[CDNA3 §3.7, table ~L1570]**; `SCRATCH_EN` mode bit = "wave has scratch
  space allocated" **[CDNA3 mode register table ~L781]**; instructions `S_SCRATCH_LOAD/STORE`
  (scalar) and VMEM `SCRATCH_LOAD_DWORD/SCRATCH_STORE_DWORD` + flat scratch ops **[CDNA3 §10.5-10.7,
  §12.15.2; RDNA3 §11.1.3, §16.20.2]**. Detection in assembly: any `scratch`-form memory op or
  `buffer_*` with the scratch resource = spill traffic.
- Compiler side: the LLVM AMDGPU backend tracks scratch size and spills in the code object;
  ROCm's Triton flow greps `vgpr_count` / `vgpr_spill` from compiled kernel metadata
  **[ROCm/triton occ.sh,
  https://github.com/ROCm/triton/blob/cf44637139ba441342b909977f51f7cfec2c9963/scripts/amd/occ.sh]**;
  omniperf attributes VMEM utilization to "global/generic and spill/scratch operations" **[OMNIPERF
  VMEM section]**. `llvm-objdump --mcpu=gfxXXX` on a code object plus `llvm-readobj --notes`
  exposes `amdhsa.next_free_vgpr`, scratch size, etc. (AMDGPU code-object metadata).

---

## 7. Prior-art tools

| Tool | What it is | Relevance to a static analyzer | Link |
|---|---|---|---|
| **NVBit** (NVlabs, 344★) | dynamic binary instrumentation framework for NVIDIA SASS | dynamic ground truth for validating static predictions; not static itself | https://github.com/NVlabs/NVBit |
| **nvdisasm / cuobjdump** (CUDA toolkit) | official SASS disassemblers | the SASS ingestion path; control-code (stall/wait-barrier/reuse) decoding documented in [TURING §2] | CUDA toolkit |
| **CuAssembler** (618★) | unofficial SASS assembler for all generations | proves SASS round-tripping; useful for patching experiments | https://github.com/cloudcores/CuAssembler |
| **Nsight Compute** | NVIDIA profiler with section/rule engine (Speed-of-Light, Occupancy, Warp State, Memory Workload Analysis, Scheduler Stats…) | its *rules* (e.g., divergence, bank-conflict, occupancy-limiter explanations) are the industry reference for report semantics; dynamic, not static | https://docs.nvidia.com/nsight-compute/ |
| **Accel-Sim** (636★) | SASS-driven GPU simulator framework (extends GPGPU-Sim 4.0) | full microarch performance model fed by SASS traces; prior art for per-arch configs | https://github.com/accel-sim/accel-sim-framework |
| **GPGPU-Sim** | PTX/SASS cycle simulator | classic PTX-level dynamic model | https://github.com/gpgpu-sim/gpgpu-sim_distribution |
| **MGPUSim** (263★) | AMD GCN3-ISA multi-GPU simulator | AMD-side prior art (GCN ISA execution model) | https://github.com/sarchlab/mgpusim |
| **gem5** (2754★) | system simulator incl. AMD GCN3 GPU model | AMD timing model prior art | https://github.com/gem5/gem5 |
| **omniperf / ROCm Compute Profiler** | AMD profiler w/ roofline + performance-model docs (GCN/CDNA/CDNA2/CDNA3) | metric definitions (LDSBankConflict, VALUUtilization, Coalescing%, VMEM/LDS latency) to mirror in reports; now merged into ROCm/rocm-systems | https://github.com/ROCm/rocprofiler-compute, docs [OMNIPERF] |
| **omnitrace** (357★) | AMD tracing/profiling | complementary dynamic tooling | https://github.com/ROCm/omnitrace |
| **llvm-mca** | LLVM static machine-code analyzer (IPC, resource pressure, timeline) driven by backend scheduling models | **works for AMDGPU assembly** (AMDGPU ships scheduling models; run `llvm-mca -mtriple=amdgpu-amd-amdhsa -mcpu=gfx90a`); **no NVPTX support** (NVPTX backend has no scheduling model / llvm-mca target) — the PTX analog must be built | https://llvm.org/docs/CommandGuide/llvm-mca.html |
| **llvm-calc-occupancy** | AMDGPU occupancy calculator over GCNSubtarget math | directly reusable for §1 formulas | https://llvm.org/docs/CommandGuide/llvm-calc-occupancy.html |
| **AMD Matrix Instruction Calculator** | MFMA/WMMA shape/throughput/register-layout helper | reuse for §5 tables | https://github.com/ROCm/amd_matrix_instruction_calculator |
| **GPUVerify** | static verifier for GPU kernels (barrier/divergence soundness) | prior art for barrier-convergence checking | https://github.com/mc-imperial/GPUVerify |
| **[MODEL26]** | analytical (static, microbenchmark-parameterized) execution-time models for B200 + MI300A, validated to ~1% MAE | closest academic analog of the target tool; stage-centric model (TMA→TMEM→TensorCore→Sync) for Blackwell, occupancy-driven wavefront model for CDNA3 | arXiv:2605.04178 |

Academic PTX/SASS analysis lineage worth citing: GPGPU-Sim (Bakhoda et al., ISPASS 2009), Accel-Sim
(Khairy et al., ISCA 2020), NVBit (Vijaykumar et al. / NVlabs docs), Ma et al. PPoPP'17 (bare-metal
Maxwell/Volta microarchitecture), Jia et al. microbenchmark series ([VOLTA], [TURING]), [HOPPER].

---

## 8. Instruction latency tables & critical-path approximation

### 8.1 Published microbenchmark latencies (cycles)

**NVIDIA memory hierarchy** (all measured):

| Metric | K80 (Kepler) | M60 (Maxwell) | P4/P100 (Pascal) | V100 (Volta) | T4 (Turing) | A100 (Ampere) | RTX 4090 (Ada) | H800 (Hopper) |
|---|---|---|---|---|---|---|---|---|
| L1 hit | 35 | 82 | 82 | 28 | 32 | 33 | 32 | 32 |
| Shared mem (no conflict) | 26 | 23 | 23/24 | 19 | 19 | 29 | 30.1 | 29 |
| L2 hit | ~200 | ~207 | ~216/~234 | ~193 | ~188 | 202.8–408 | 273 | 264.5–502 |
| Global/DRAM | — | — | — | — | 616 (cold, TLB+L2 miss) | 566 | 571 | 656 |

Sources: [TURING Table 2 & Fig. 6] (T4/V100/P100/P4/M60/K80 columns incl. shared-memory no-conflict
latency row); [VOLTA Table 3.1] (Volta column); [HOPPER Table 3] (RTX4090/A100/H800, traditional
P-chase; Hopper L2 range reflects partitioned near/far L2 **[HOPPER §4.1]**).

**AMD memory hierarchy:** [MALL25] gives sL1/L2/MALL/HBM pointer-chase latencies for MI300A/MI300X
(MALL = 256 MB Infinity Cache; NPS4 slices it into 4×64 MB and cuts latency up to 1.42×) and MI250X
(no MALL). [MODEL26] uses measured cache latencies ≈ 5/50/150/400 cycles (L1/L2/LLC/HBM) for its
MI300A/B200 models, TMA copy latency ≈ 420 cycles, mbarrier wait/commit ≈ 40–50 cycles, and
`tcgen05.mma` instruction latency ≈ 11–14 cycles on B200 **[MODEL26 parameter table]**.

**Tensor-core instruction timing:**
- Hopper `wgmma` dense/sparse latency & throughput tables per shape/dtype: **[HOPPER Tables 8-10,
  15]** (e.g., at N≥64 all wgmma variants reach near-peak; latency in clock cycles per shape).
- CDNA2 MFMA issue/dependency timing: required-NOP table — e.g., VALU-write→V_MFMA needs 2 NOPs;
  same-opcode SrcC forwarding = 0 waits; SrcA/SrcB = 3; and MFMA "pass" counts (2/4/8/16-pass
  variants) govern back-to-back waits **[CDNA2 §7.2, Table 26]**. CDNA3 extends this with
  2/4/8/16-pass wait tables for MFMA/SGEMM/DGEMM and V_SMFMA **[CDNA3 §7 dependency tables,
  ~L3300-3480]**. RDNA3 WMMA: dependent WMMAs need 1 NOP when D overlaps A/B **[RDNA3 §7.9]**.

**Instruction-issue facts for the mix analysis:** Turing/Volta SASS encodes per-instruction
scheduling control (4-bit stall cycles, 6 wait/write/read dependency barriers, reuse flags, yield)
— static SASS analysis can read these directly **[TURING §2]**. Volta/Turing have no dual issue
(single dispatcher per processing block) **[TURING §2]**; warps map to schedulers by `warp_id % 4`
**[TURING Table 1]**.

### 8.2 How a static tool approximates critical-path latency

1. **Instruction mix → throughput bound:** classify PTX/asm instructions into issue classes (VALU/
   SFU/FMA/LSU/tensor/branch), divide counts by per-SM per-cycle issue rates (e.g., 128 FP32
   FMA/SM/cycle on H100 **[HOPPERBLOG]**; AMD: 1 MFMA pass per SIMD per cycle, 64-lane VALU over
   4 cycles on CDNA), take max → compute-bound estimate (roofline numerator).
2. **Memory bound:** sector-count analysis (§3) × measured latency/bandwidth tables above; count
   outstanding-loads needed to hide latency: `needed_warps ≈ latency_cycles / issue_interval`
   (Little's law), cross-checked against §1 occupancy.
3. **Critical path (latency bound):** build the inter-instruction dependence DAG within a basic
   block/loop body (register def-use + memory ordering + barrier edges); edge weights = instruction
   latencies from the tables above (fixed latency for ALU; variable-latency memory ops get
   cache-level latencies; tensor ops get per-shape pass counts). Longest path ≈ lower bound on
   iteration time; compare against (1) and (2) and report the dominant bound — this is exactly the
   "stage-centric" decomposition validated in [MODEL26] (TMA→TMEM→MMA→sync stages on Blackwell;
   occupancy-gated overlap on CDNA3). llvm-mca performs the analogous simulation for AMDGPU using
   LLVM scheduling models **[llvm-mca docs]** — a good oracle to validate the AMD side against.
4. Divergence multiplies path costs by the number of serialized taken-paths (§4); bank conflicts
   multiply shared-memory op cost by the conflict degree (§2); spills add global-latency round trips
   per ld.local/st.local (§6).

---

## 9. Recommended analyzer architecture (synthesis)

- **Front ends:** PTX parser (instructions, predicates, `.local` decls, `mma`/`wgmma`/`tcgen05`
  families, `bar.*`); AMDGPU assembly parser via LLVM MC (reuse `llvm-mc`/`llvm-objdump` parsing;
  occupancy via `llvm-calc-occupancy` math **[LLVM]**).
- **Per-kernel analyses:** register pressure → occupancy (§1); affine per-lane address analysis for
  coalescing (§3) and bank conflicts (§2); uniformity/divergence via Rosemann-Moll-Hack style
  abstract interpretation **[LLVM ConvergenceAndUniformity]**; barrier reachability; tensor-core MAC
  census (§5); spill census (§6); instruction-mix + longest-path latency estimate (§8).
- **Report:** per-kernel occupancy limiter breakdown (AMD-style: which of VGPR/SGPR/LDS/WG binds
  **[OCCBLOG]**; NVIDIA-style: regs/smem/threads/blocks **[CPG §2.3.7]**), coalescing efficiency %,
  bank-conflict degree histogram, divergence-site list with post-dominator info, tensor MAC share,
  spill bytes, top critical-path chains with latency attribution.

---

## 10. Conflicts, gaps, and cautions

1. **NVIDIA register granularity is not officially documented anymore.** The 256-registers-per-warp
   (8-per-thread) granule is measured/documented only for Volta/Turing **[TURING §3.5.1]**; assume
   Ampere..Blackwell inherit it, but mark it as inferred. (CUDA 8.0-era guide and current 12.8/13.3
   guides both defer to the occupancy APIs without printing the granule **[NVCC-era guide §5.2.3.1;
   CPG §2.3.7]**.)
2. **Cache-line sizes conflict across sources.** Programmer docs say 32 B transactions **[CPG]**;
   microbenchmarks measure 32 B load granularity + 128 B update granularity and 64 B L2 lines on
   Volta/Turing **[VOLTA Table 3.1, TURING Table 2]**; Hopper L2 latency is a *range* due to
   partitioning **[HOPPER §4.1]**. Report both, labeled.
3. **AMD VGPR-file wording.** Omniperf docs say MI2XX has "a 512 KiB VGPR file per CU, where each
   wave can dynamically request up to 256 KiB of VGPRs and an additional 256 KiB of AGPRs"
   **[OMNIPERF]** — the ISA instead says a wave gets "up to 512 total VGPRs, 256 of each type"
   (registers, not KiB; 256 regs × 64 lanes × 4 B = 64 KiB/wave) **[CDNA2 §3.6.4]**. Also [OCCBLOG]
   states CDNA3 *unified* regular+acc into one physical 512-entry file ("not the separate physical
   ACC file of MI100/MI200"), while the CDNA3 ISA text still describes two logical pools — the
   occupancy budget (512 total) is the same either way; use it.
4. **RDNA3 allocation granule:** 16 (wave32) / 8 (wave64) generally, but **24/12 on devices with
   1536 VGPRs per SIMD** **[RDNA3 §3.3.2.1]** — the gfx1100 family is in the latter class; don't
   hardcode 16.
5. **sm_120 tensor cores differ from sm_100:** consumer Blackwell (sm_120a/sm_120f) gets
   block-scale FP6/FP4 `mma` variants **[PTX §9.7.15.5 target notes]**, while `tcgen05.*` target
   notes list sm_100a/sm_101a(→sm_110a)/sm_100f-family **[PTX §9.7.17]** — verify per-instruction
   target notes before assuming tcgen05 on RTX 50 parts.
6. **Max blocks/SM traps:** sm_86=16 but sm_89=24; sm_75 threads/SM=1024 (not 2048) **[CPG Table
   30]** — easy to misremember.
7. **ptxas `-v` output format is stable but only semi-official:** the exact "X bytes stack frame,
   Y bytes spill stores, Z bytes spill loads / Used N registers…" format is shown in the NVCC docs
   example **[NVCC §8.4]**; parse defensively.
8. **llvm-mca on GPUs:** excellent for AMDGPU (scheduling models exist), absent for NVPTX — plan a
   custom PTX latency table (§8) rather than expecting llvm-mca coverage.
9. AMD L1/vL1D line/sector sizes are not printed in ISA docs; the documented proxies are omniperf's
   Coalescing% (25% floor) **[OMNIPERF]** and `WRITE_REQ_32B` **[PCOUNTERS]**.

---

## Appendix A — Source inventory (fetched & verified 2026-08-13)

- CUDA C++ Programming Guide v13.3 (HTML pages + PDF) — compute-capability tables, occupancy,
  coalescing, banks, ITS, local memory.
- PTX ISA (single-page HTML, 32k lines extracted) — state spaces, bar.*, wmma/mma/wgmma/tcgen05
  shapes + per-arch target notes.
- NVCC compiler driver docs — `--resource-usage`, `-warn-spills`, ptxas verbose example.
- AMD ISA PDFs: CDNA2 (267 pp), CDNA3 (553 pp), RDNA3 #70650 (609 pp) — extracted via pdftotext;
  VGPR/SGPR/LDS allocation rules, MFMA/WMMA opcode tables, scratch, LDS banks, dependency/NOP
  tables.
- CDNA3 whitepaper; A100 whitepaper; Hopper-in-depth blog.
- arXiv PDFs: 1804.06826 (Volta), 1903.07486 (Turing), 2501.12084 (Hopper); arXiv HTML: 2605.04178
  (analytical models B200/MI300A).
- LLVM sources @27eeb73: GCNSubtarget.cpp (computeOccupancy), llvm-calc-occupancy.cpp @11d7c9c;
  LLVM docs: llvm-mca, llvm-calc-occupancy, ConvergenceAndUniformity.
- ROCm: occupancy blog (CDNA4 math), gpu-arch docs index, MI300 microarchitecture page,
  MI300/MI200 performance counters, rocprofiler-compute performance model, triton occ.sh @cf44637.
- GitHub repo metadata via `gh api` (stars/descriptions verified today).
