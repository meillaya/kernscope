//===- occupancy.h - GPU occupancy engine (NVIDIA + AMD) ------------------===//
//
// Wave-2 todo 11: the theoretical-occupancy engine shared by tools/analyze
// (todo 14), tools/tune (todo 28) and tools/profile (todo 50). Consumes the
// ISA-resolved architecture constants of libs/arch (todo 6) and implements
// the min-wins rule of docs/research/gpu-static-analyzer-research.md §1.1
// (L31-45) with the per-vendor allocation granules:
//
//   NVIDIA  §1.2 L91-102: regs per warp = round-up(regs_per_thread x 32) to
//           the 256-register warp granule; blocks_by_regs =
//           floor(65536 / (regs_per_warp x warps_per_block)); smem, threads
//           and blocks/SM limits as in §1.1; min wins.
//   AMD     §1.3: VGPR waves/SIMD via libs/arch waves_per_simd_from_vgprs
//           (CDNA granule 8, RDNA3 granule 24 wave32 / 12 wave64 per
//           docs/research/rdna3-constants.md) clamped to max waves/SIMD;
//           SGPR pool ~800/SIMD with granule 16 (brief §1.3 L117-121);
//           LDS 64 KB/CU with 512-byte allocation granule (§1.3 L123-125);
//           then CU-level min over VGPR/SGPR/LDS + thread + workgroup slots.
//
// llvm-calc-occupancy note: the plan's original AMD oracle idea was to shell
// out to LLVM's `llvm-calc-occupancy` (a thin GCNSubtarget front-end, brief
// §1.1 L60-64). That tool is ABSENT from LLVM 21.1.8 (this repo's pinned
// toolchain — verified `which llvm-calc-occupancy` in the devenv shell), so
// the engine implements the same math directly and the tests oracle against
// the documented GCNSubtarget formula (rdna3-constants.md evidence (d),
// LLVM AMDGPUBaseInfo.cpp L1371-1380) plus the ROCm/Triton occ.sh threshold
// table (brief §1.3 L113-116). See occupancy_test.cpp header for the full
// oracle rationale.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_LIBS_OCCUPANCY_INCLUDE_KERNSCOPE_OCCUPANCY_H
#define KERNSCOPE_LIBS_OCCUPANCY_INCLUDE_KERNSCOPE_OCCUPANCY_H

#include <cstdint>

#include "kernscope/arch.h"

namespace kernscope::occupancy {

// Which resource binds residency. The task contract pins these four names
// (registers|shared|threads|blocks); see limiter_name() for the stable
// strings. On AMD the VGPR and SGPR limits both report as kRegisters
// (the register limiter is the min of the two), and the max-waves-per-SM
// cap is folded into kThreads (threads/SM = waves cap x warp size — the
// two limits coincide for warp-aligned block sizes, which is all the engine
// models; brief §1.1 L39).
enum class Limiter : std::uint8_t {
  kRegisters,
  kShared,
  kThreads,
  kBlocks,
};

// Stable machine-readable limiter names: "registers" | "shared" | "threads"
// | "blocks" (the task-contract strings; pinned goldens assert "registers").
const char *limiter_name(Limiter limiter) noexcept;

// One occupancy result. occupancy is a FRACTION in [0, 1] (the task contract:
// "occupancy (double 0..1)"; convert to % with x100). blocks_per_sm is the
// min-wins resident block count; warps_resident = blocks x warps_per_block;
// max_warps_per_sm = max_resident_warps(arch) [arch.h L198-199]. A 0
// blocks_per_sm means the (regs, smem, block) combination cannot run.
struct OccupancyResult {
  int blocks_per_sm = 0;    // resident blocks per SM (NVIDIA) / CU (AMD)
  int warps_resident = 0;   // resident warps/wavefronts = blocks x wpb
  int max_warps_per_sm = 0; // hardware max (max_resident_warps)
  double occupancy = 0.0;   // 0..1 = warps_resident / max_warps_per_sm
  Limiter limiter = Limiter::kThreads;
};

// Compute theoretical occupancy for (regs_per_thread, smem_bytes_per_block,
// block_size) on `arch` with the min-wins rule (brief §1.1 L31-45):
//   NVIDIA: blocks = min(regs, smem, threads, blocks-cap) where
//     regs    = floor(65536 / (roundUp256(regs x 32) x warps_per_block))
//     smem    = floor(smem_per_sm / smem_per_block)        [smem > 0]
//     threads = floor(max_threads_per_sm / block_size)
//     blocks  = max_blocks_per_sm
//   AMD:     blocks = min over VGPR waves, SGPR waves (5-arg overload only),
//     LDS (64 KB/CU with 512-B granule), thread slots and workgroup slots.
// regs_per_thread is the NVIDIA per-thread register count or the AMD per-lane
// VGPR usage; smem_bytes_per_block 0 = "no shared memory requested";
// block_size must be > 0 (<= 0 yields a 0-blocks result). noexcept: never
// throws (libs/arch lookups happen before the call).
OccupancyResult compute_occupancy(const ArchInfo &arch, int regs_per_thread,
                                  int smem_bytes_per_block,
                                  int block_size) noexcept;

// AMD SGPR-aware overload. sgprs_per_thread > 0 enables the SGPR limiter
// (pool ~800 SGPRs/SIMD, allocation granule 16 — brief §1.3 L117-121);
// sgprs_per_thread <= 0 means "not modeled" (no documented pool figure was
// pinned for RDNA3, so SGPRs only limit on CDNA; see occupancy.cpp). The
// parameter is ignored on the NVIDIA path.
OccupancyResult compute_occupancy(const ArchInfo &arch, int regs_per_thread,
                                  int smem_bytes_per_block, int block_size,
                                  int sgprs_per_thread) noexcept;

}  // namespace kernscope::occupancy

#endif  // KERNSCOPE_LIBS_OCCUPANCY_INCLUDE_KERNSCOPE_OCCUPANCY_H
