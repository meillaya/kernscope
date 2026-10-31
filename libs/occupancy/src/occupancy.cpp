//===- occupancy.cpp - GPU occupancy engine (NVIDIA + AMD) ----------------===//
//
// Implements the min-wins occupancy rule from docs/report/kernscope-technical-report.pdf (L31-45) on top of the libs/arch
// constants (todo 6). All formulas carry the brief line citations used by
// the todo-6 test suite; every deviation or modeling choice is documented
// here and cross-checked by occupancy_test.cpp.
//
// llvm-calc-occupancy note (plan todo 11): LLVM 21.1.8 ships NO
// llvm-calc-occupancy tool (verified absent in the devenv shell PATH), so
// the AMD path re-implements the GCNSubtarget occupancy math it wraps:
//   waves/SIMD = clamp(floor(VGPR_file / align_granule(vgprs)), 1,
//                       max_waves_per_simd)
// which is exactly LLVM's getNumWavesPerEUWithNumVGPRs
// (AMDGPUBaseInfo.cpp L1371-1380; cited in rdna3-constants.md evidence (d)).
//
//===----------------------------------------------------------------------===//

#include "kernscope/occupancy.h"

#include <algorithm>
#include <limits>

namespace kernscope::occupancy {
namespace {

// NVIDIA: 64K 32-bit registers per SM [brief §1.2 L70]; warps allocate in
// increments of 256 registers (8 regs/thread x 32) [brief §1.2 L91-99].
constexpr int kNvidiaRegsPerSm = 65536;
constexpr int kNvidiaWarpRegGranule = 256;

// AMD LDS: 64 KB per CU [brief §1.3 L123 (CDNA), L142-143 (RDNA3)] allocated
// in 512-byte blocks [brief §1.3 L123-125: CDNA2 "contiguous blocks of 128
// Dwords" = 512 B; CDNA3 "blocks of 512 bytes"].
constexpr int kAmdLdsAllocGranule = 512;

// AMD SGPR pool ~800 per SIMD with granule 16 [brief §1.3 L117-121:
// "A wavefront can be allocated 16 to 102 SGPRs, in units of 16 GPRs";
// "LLVM and the ROCm blog use ~800 SGPRs/SIMD"]. Only CDNA pins the pool;
// RDNA3 has no pool figure in our research docs, so the SGPR limiter is
// modeled for CDNA only (see compute_occupancy 5-arg).
constexpr int kCdnaSgprPoolPerSimd = 800;

int align_up(int value, int granule) noexcept {
  if (granule <= 0) return value;
  return ((value + granule - 1) / granule) * granule;
}

}  // namespace

const char *limiter_name(Limiter limiter) noexcept {
  switch (limiter) {
    case Limiter::kRegisters:
      return "registers";
    case Limiter::kShared:
      return "shared";
    case Limiter::kThreads:
      return "threads";
    case Limiter::kBlocks:
      return "blocks";
  }
  return "threads";  // unreachable; keeps -Wreturn-type clean under clang
}

namespace {

// Deterministic tie-break so the binding limiter on equal block counts is
// stable and matches the polykernel oracle (which reports "registers" on an
// 8/8/8 tie; arch.h predict_occupancy uses the same registers-first order).
// Priority: registers > shared > threads > blocks. A negative candidate
// means "not modeled" and never wins.
void pick_min(int candidate, Limiter name, int priority, int *blocks,
              Limiter *limiter) noexcept {
  if (candidate < 0) return;  // sentinel: not modeled
  const int cur = *blocks;
  int cur_priority = 0;
  switch (*limiter) {
    case Limiter::kRegisters:
      cur_priority = 5;
      break;
    case Limiter::kShared:
      cur_priority = 3;
      break;
    case Limiter::kThreads:
      cur_priority = 2;
      break;
    case Limiter::kBlocks:
      cur_priority = 1;
      break;
  }
  if (candidate < cur || (candidate == cur && priority > cur_priority)) {
    *blocks = candidate;
    *limiter = name;
  }
}

OccupancyResult finish(const ArchInfo &a, int blocks, int warps_per_block,
                       Limiter limiter) noexcept {
  OccupancyResult out;
  out.max_warps_per_sm = max_resident_warps(a);
  out.blocks_per_sm = blocks;
  out.warps_resident = blocks * warps_per_block;
  out.occupancy = out.max_warps_per_sm > 0
                      ? static_cast<double>(out.warps_resident) /
                            static_cast<double>(out.max_warps_per_sm)
                      : 0.0;
  out.limiter = limiter;
  return out;
}

}  // namespace

OccupancyResult compute_occupancy(const ArchInfo &arch, int regs_per_thread,
                                  int smem_bytes_per_block, int block_size,
                                  int sgprs_per_thread) noexcept {
  // Block size sanity: a non-positive block cannot run.
  if (block_size <= 0) {
    OccupancyResult out;
    out.max_warps_per_sm = max_resident_warps(arch);
    out.blocks_per_sm = 0;
    out.warps_resident = 0;
    out.limiter = Limiter::kThreads;
    return out;
  }

  // Register counts are usage figures; clamp negatives to 0 so granule
  // rounding cannot turn a bogus negative into a "winner" of the min-wins.
  regs_per_thread = std::max(regs_per_thread, 0);
  sgprs_per_thread = std::max(sgprs_per_thread, 0);

  const int warp_size = arch.warp_size;
  const int warps_per_block = (block_size + warp_size - 1) / warp_size;
  int blocks = -1;  // start "unset"; pick_min requires a first candidate
  Limiter limiter = Limiter::kThreads;
  bool first = true;

  auto pick = [&](int candidate, Limiter name, int priority) {
    if (first) {
      blocks = candidate;
      limiter = name;
      first = false;
      return;
    }
    pick_min(candidate, name, priority, &blocks, &limiter);
  };

  // --- threads / waves cap: floor(max_threads_per_sm / block_size) ---------
  // [brief §1.1 L39]. For warp-aligned block sizes this coincides with the
  // max-warp cap floor(max_warps_per_sm / warps_per_block) (the two limits
  // differ only for non-warp-multiple block sizes, which the engine models
  // via the thread form per the task contract).
  const int limit_threads = arch.max_threads_per_sm / block_size;
  pick(limit_threads, Limiter::kThreads, 2);

  // --- shared memory (smem/LDS) limit --------------------------------------
  // NVIDIA [brief §1.1 L40]: floor(smem_per_sm / smem_per_block).
  // AMD [brief §1.3 L123-127]: floor(64 KB / ceil512B(lds_per_wg)) with the
  // 512-byte LDS allocation granule. smem == 0 -> not modeled.
  int limit_smem = -1;
  if (smem_bytes_per_block > 0 && arch.smem_per_sm_bytes > 0) {
    const int alloc = arch.vendor == Vendor::kAmd
                          ? align_up(smem_bytes_per_block, kAmdLdsAllocGranule)
                          : smem_bytes_per_block;
    limit_smem = arch.smem_per_sm_bytes / alloc;
  }
  if (limit_smem >= 0) pick(limit_smem, Limiter::kShared, 3);

  // --- workgroup-slot cap ---------------------------------------------------
  // [brief §1.1 L42] max_blocks_per_sm; libs/arch models AMD slots as 0
  // ("not modeled" — the ISA docs do not pin a CDNA/RDNA3 hardware block
  // cap; the gfx1101 driver figure 64 is recorded in rdna3-constants.md
  // evidence (c) but is not a libs/arch constant). Only a > 0 constant
  // limits.
  if (arch.max_blocks_per_sm > 0)
    pick(arch.max_blocks_per_sm, Limiter::kBlocks, 1);

  if (arch.vendor == Vendor::kNvidia) {
    // --- NVIDIA register limit ----------------------------------------------
    // regs_per_warp = round-up(regs_per_thread x 32, 256-reg warp granule)
    // [brief §1.2 L95-96]; blocks_by_regs = floor(65536 /
    // (regs_per_warp x warps_per_block)) [brief §1.2 L101-102]. Note the
    // engine implements the CUDA Programming Guide formula the task pins
    // (and the brief's worked examples use); cuda_occupancy.h additionally
    // floors warps per 16K-register sub-partition, which only diverges for
    // small blocks / high regs — occupancy_test.cpp documents and pins
    // both models (see NvidiaOracleSubpartitionDivergence).
    const int regs_per_warp =
        align_up(regs_per_thread * warp_size, kNvidiaWarpRegGranule);
    const int regs_per_block = regs_per_warp * warps_per_block;
    const int limit_regs = regs_per_block > 0 ? kNvidiaRegsPerSm / regs_per_block
                                              : 0;
    pick(limit_regs, Limiter::kRegisters, 5);
  } else {
    // --- AMD VGPR limit ------------------------------------------------------
    // waves/SIMD from VGPR usage via libs/arch (CDNA: floor(512/ceil8),
    // RDNA3 wave32: floor(1536/ceil24) clamped to 16 — the ISA/LLVM formula
    // from rdna3-constants.md), then blocks = waves x SIMDs / warps_per_block
    // [brief §1.3 L112, L139-141].
    const int waves_per_simd =
        waves_per_simd_from_vgprs(arch, regs_per_thread);
    const int limit_vgpr =
        (waves_per_simd * arch.simds_per_cu) / warps_per_block;
    pick(limit_vgpr, Limiter::kRegisters, 5);

    // --- AMD SGPR limit (CDNA only; opt-in via 5-arg overload) --------------
    // waves/SIMD = clamp(floor(~800 / ceil16(sgprs)), 1, max waves) [brief
    // §1.3 L117-121]. No RDNA3 SGPR pool figure is pinned in our research
    // docs, so sgprs <= 0 (or a non-CDNA arch) skips the limiter entirely.
    if (sgprs_per_thread > 0 && arch.reg_granule > 0) {
      // CDNA only: gfx90a/gfx942 pin the ~800/SIMD pool; RDNA3 leaves it
      // unpinned, so only archs with a documented CDNA pool get this limit.
      if (arch.name == "gfx90a" || arch.name == "gfx942") {
        const int sgpr_aligned = align_up(sgprs_per_thread, arch.sgpr_granule);
        int sgpr_waves = kCdnaSgprPoolPerSimd / sgpr_aligned;
        sgpr_waves = std::clamp(sgpr_waves, 1, arch.max_warps_per_sm);
        const int limit_sgpr =
            (sgpr_waves * arch.simds_per_cu) / warps_per_block;
        pick(limit_sgpr, Limiter::kRegisters, 5);
      }
    }
  }

  return finish(arch, blocks, warps_per_block, limiter);
}

OccupancyResult compute_occupancy(const ArchInfo &arch, int regs_per_thread,
                                  int smem_bytes_per_block,
                                  int block_size) noexcept {
  return compute_occupancy(arch, regs_per_thread, smem_bytes_per_block,
                           block_size, /*sgprs_per_thread=*/0);
}

}  // namespace kernscope::occupancy
