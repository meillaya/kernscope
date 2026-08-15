//===- occupancy_test.cpp - occupancy engine unit tests -------------------===//
//
// Wave-2 todo 11. Two oracles + hand-pinned goldens:
//
//  (1) NVIDIA oracle = CUDA's own occupancy calculator, cuda_occupancy.h from
//      $CUDA_HOME/include (cudaPackages_12_6), consumed at BUILD time from the
//      nix store — NEVER vendored (ci.sh guard `git ls-files | grep
//      'cuda_occupancy.h$'` must stay empty; only a #include of the store
//      header is legal). We drive cudaOccMaxActiveBlocksPerMultiprocessor
//      with per-arch cudaOccDeviceProp rows built from the libs/arch
//      constants (sm_75/80/86/89/90) and assert the engine's blocks_per_sm
//      matches across a 342-case verified grid (see MakeNvidiaGrid; every
//      case independently verified to agree with the naive guide formula via
//      a pre-commit sweep). The pinned goldens are asserted with 1e-9
//      tolerance:
//        - (sm_86, 112 regs, 64-thread) -> 9 blocks x 2 warps / 48 =
//          occupancy 0.375, limiter "registers"  [brief §1.2 L101-102]
//        - (sm_90, 128 regs, 256-thread) -> 16/64 = 0.25 [brief §1.2 L101-102
//          worked example]
//      SUB-PARTITION NOTE: cuda_occupancy.h additionally floors warps per
//      16,384-register SM sub-partition (cudaOccMaxBlocksPerSMRegsLimit,
//      numWarpsPerSubPartition = floor(16384 / regs_per_warp); [TURING]
//      §3.5.1 via brief §1.2 L91-99). The kernscope engine implements the
//      CUDA Programming Guide formula the task pins (floor(65536 /
//      (regs_per_warp x warps_per_block))) — the two agree on the 342-case
//      grid and diverge only for small blocks / high regs; NvidiaOracle-
//      SubpartitionDivergence pins and documents one such case.
//
//  (2) AMD oracle = the GCNSubtarget occupancy math as DOCUMENTED, not as
//      linked. The plan's original choice was to link libLLVM and call
//      GCNSubtarget::computeOccupancy directly; that is impossible on this
//      toolchain because LLVM 21.1.8 does not INSTALL the AMDGPU backend
//      headers (GCNSubtarget.h lives in llvm/lib/Target/AMDGPU, absent from
//      the nix store include tree — verified) and ships no llvm-calc-
//      occupancy (absent from LLVM 21, documented in occupancy.cpp). We
//      therefore oracle against the documented equivalent:
//        - CDNA: ROCm/Triton occ.sh threshold table (brief §1.3 L113-116:
//          <=64->8, 65-72->7, 73-80->6, 81-96->5, 97-128->4, 129-168->3,
//          169-256->2, >256->1 wave/SIMD) asserted across the full VGPR
//          sweep;
//        - RDNA3: LLVM's getNumWavesPerEUWithNumVGPRs formula
//          (AMDGPUBaseInfo.cpp L1371-1380, quoted in rdna3-constants.md
//          evidence (d)): waves = clamp(floor(TotalNumVGPRs /
//          alignTo(VGPRs, Granule)), 1, MaxWaves) with TotalNumVGPRs=1536
//          (wave32 view), Granule=24, MaxWaves=16.
//
// Cross-validation oracle: /home/mei/projects/polykernel/lib/Analysis/
// tests/occupancy_test.cpp (sm_80/sm_90 hand-computed values read for
// validation only; no code copied).
//
//===----------------------------------------------------------------------===//

#include "kernscope/occupancy.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

// The NVIDIA oracle TU (below) #includes cuda_occupancy.h only when CMake
// found it under $ENV{CUDA_HOME}/include; otherwise KERNSCOPE_NO_CUDA_OCCUPANCY
// is defined and the oracle tests GTEST_SKIP() with the reason.
#ifndef KERNSCOPE_NO_CUDA_OCCUPANCY
#include <cuda_occupancy.h>
#endif

namespace {

using kernscope::get_arch;
using kernscope::occupancy::OccupancyResult;
using kernscope::occupancy::compute_occupancy;
using kernscope::occupancy::limiter_name;

const char *kNvidiaArchs[] = {"sm_75", "sm_80", "sm_86", "sm_89", "sm_90"};

// ---------------------------------------------------------------------------
// Hand-pinned NVIDIA goldens (brief §1.2 L101-102 + todo-5 manifest math).
// ---------------------------------------------------------------------------

TEST(occupancy, PinnedGoldenSm86_375) {
  // Review-verified golden (todo 5): sm_86, 112 regs/thread, 64-thread block
  // -> regs_per_warp = 112x32 = 3584 (already 256-aligned) -> blocks_by_regs
  // = floor(65536 / (3584 x 2)) = 9 -> 18 warps / 48 = 0.375. Limiter
  // "registers". Brief §1.2 L101-102; tolerance 1e-9.
  const auto o = compute_occupancy(get_arch("sm_86"), /*regs_per_thread=*/112,
                                   /*smem_bytes_per_block=*/0, /*block_size=*/64);
  EXPECT_EQ(o.blocks_per_sm, 9);
  EXPECT_EQ(o.warps_resident, 18);
  EXPECT_EQ(o.max_warps_per_sm, 48);
  EXPECT_NEAR(o.occupancy, 0.375, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "registers");
}

TEST(occupancy, PinnedGoldenSm90_025) {
  // Brief §1.2 worked example (L101-102): sm_90, 128 regs, 256-thread blocks
  // -> 128x256 = 32768 regs/block -> floor(65536/32768) = 2 blocks -> 16/64
  // warps = 0.25. Limiter "registers".
  const auto o = compute_occupancy(get_arch("sm_90"), /*regs_per_thread=*/128,
                                   /*smem_bytes_per_block=*/0, /*block_size=*/256);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_EQ(o.warps_resident, 16);
  EXPECT_EQ(o.max_warps_per_sm, 64);
  EXPECT_NEAR(o.occupancy, 0.25, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "registers");
}

TEST(occupancy, NvidiaSmemAndBlocksLimiters) {
  // sm_90 smem-limited: 32 regs, 100 KB smem, 256 threads -> floor(233472 /
  // 102400) = 2 blocks, shared. [polykernel oracle case 3]
  auto o = compute_occupancy(get_arch("sm_90"), 32, 100 * 1024, 256);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_NEAR(o.occupancy, 0.25, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "shared");

  // sm_90 blocks-cap: 16 regs, 0 smem, 32 threads -> 32 blocks (cap), 0.5.
  // [polykernel oracle case 5]
  o = compute_occupancy(get_arch("sm_90"), 16, 0, 32);
  EXPECT_EQ(o.blocks_per_sm, 32);
  EXPECT_NEAR(o.occupancy, 0.5, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "blocks");

  // sm_90 threads/wave-cap: 1024 threads -> floor(2048/1024) = 2 blocks ->
  // full occupancy. [polykernel oracle case 4; task contract maps the wave
  // cap to the "threads" limiter]
  o = compute_occupancy(get_arch("sm_90"), 16, 0, 1024);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_EQ(o.warps_resident, 64);
  EXPECT_NEAR(o.occupancy, 1.0, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "threads");
}

TEST(occupancy, NvidiaCannotRunCases) {
  // 255 regs x 512-thread block on sm_75: regs_per_block = 8192 x 16 =
  // 131072 > 65536 -> 0 blocks (kernel cannot run). Both the engine and the
  // oracle agree (verified grid covers 0-block cases).
  auto o = compute_occupancy(get_arch("sm_75"), 255, 0, 512);
  EXPECT_EQ(o.blocks_per_sm, 0);
  EXPECT_NEAR(o.occupancy, 0.0, 1e-9);
  // block > max threads per SM -> cannot run.
  o = compute_occupancy(get_arch("sm_86"), 16, 0, 2048);
  EXPECT_EQ(o.blocks_per_sm, 0);
  // Non-positive block size -> cannot run.
  o = compute_occupancy(get_arch("sm_90"), 16, 0, 0);
  EXPECT_EQ(o.blocks_per_sm, 0);
}

// ---------------------------------------------------------------------------
// Polykernel cross-check (oracle-only; /home/mei/projects/polykernel/lib/
// Analysis/tests/occupancy_test.cpp). Values read for validation, no code
// copied. All six hand-computed cases:
//   case1 sm_90 128/32K/256 -> 2 blk 16 warps 25%      case2 sm_80 same shape
//   case3 sm_90 32/100K/256 -> 2 blk 25% (smem)        case4 sm_90 16/0/1024
//   case5 sm_90 16/0/32 -> 32 blk 50% (blocks)         case6 sm_90 31/1024/256
// ---------------------------------------------------------------------------

TEST(occupancy, PolykernelSm90Cases) {
  auto o = compute_occupancy(get_arch("sm_90"), 128, 32 * 1024, 256);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_EQ(o.warps_resident, 16);
  EXPECT_NEAR(o.occupancy, 0.25, 1e-9);

  o = compute_occupancy(get_arch("sm_90"), 32, 100 * 1024, 256);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_NEAR(o.occupancy, 0.25, 1e-9);

  o = compute_occupancy(get_arch("sm_90"), 16, 0, 32);
  EXPECT_EQ(o.blocks_per_sm, 32);
  EXPECT_EQ(o.warps_resident, 32);
  EXPECT_NEAR(o.occupancy, 0.5, 1e-9);

  // MatmulKernelSm90: 31 regs, 1024 B, 256 thr -> 8 blocks (registers
  // tie-break on 8/8/8), 100%.
  o = compute_occupancy(get_arch("sm_90"), 31, 1024, 256);
  EXPECT_EQ(o.blocks_per_sm, 8);
  EXPECT_EQ(o.warps_resident, 64);
  EXPECT_NEAR(o.occupancy, 1.0, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "registers");
}

TEST(occupancy, PolykernelSm80Case) {
  // RegisterLimitedSm80: 128 regs, 32 KB smem, 256 thr -> 2 blocks, 25%.
  const auto o = compute_occupancy(get_arch("sm_80"), 128, 32 * 1024, 256);
  EXPECT_EQ(o.blocks_per_sm, 2);
  EXPECT_EQ(o.warps_resident, 16);
  EXPECT_EQ(o.max_warps_per_sm, 64);
  EXPECT_NEAR(o.occupancy, 0.25, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "registers");
}

// ---------------------------------------------------------------------------
// NVIDIA oracle: cuda_occupancy.h comparison across a verified grid.
// The 342-case grid (5 archs x regs{16,32,64,128,160,255} x smem{0,1024,
// 32768,65536 (<= opt-in cap)} x block{128,256,512}) was swept pre-commit:
// engine (naive guide formula) == cuda_occupancy.h on every case (0
// disagreements; see task-11 evidence). Parameterized so every case is an
// individually-listed ctest entry.
// ---------------------------------------------------------------------------

struct NvidiaGridCase {
  const char *arch;
  int regs;
  int smem;
  int block;
};

std::vector<NvidiaGridCase> MakeNvidiaGrid() {
  std::vector<NvidiaGridCase> grid;
  const int kRegs[] = {16, 32, 64, 128, 160, 255};
  const int kSmem[] = {0, 1024, 32768, 65536};
  const int kBlocks[] = {128, 256, 512};
  for (const char *arch : kNvidiaArchs) {
    const auto &a = get_arch(arch);
    const int smem_cap = a.smem_per_block_max_bytes;  // opt-in per-block max
    for (int r : kRegs)
      for (int s : kSmem) {
        if (s > smem_cap) continue;
        for (int b : kBlocks)
          if (b <= a.max_threads_per_sm) grid.push_back({arch, r, s, b});
      }
  }
  return grid;
}

class NvidiaOracleGridTest : public ::testing::TestWithParam<NvidiaGridCase> {};

TEST_P(NvidiaOracleGridTest, EngineMatchesCudaOccupancyHeader) {
#ifdef KERNSCOPE_NO_CUDA_OCCUPANCY
  GTEST_SKIP() << "cuda_occupancy.h not found at $CUDA_HOME/include at "
                  "configure time (KERNSCOPE_NO_CUDA_OCCUPANCY); NVIDIA oracle "
                  "grid skipped";
#else
  const NvidiaGridCase &c = GetParam();
  const auto &a = get_arch(c.arch);
  SCOPED_TRACE(std::string(c.arch) + " regs=" + std::to_string(c.regs) +
               " smem=" + std::to_string(c.smem) +
               " block=" + std::to_string(c.block));

  // cudaOccDeviceProp row mirroring the libs/arch constants (brief §1.2
  // L74-84). sharedMemPerBlock = opt-in max and shmemLimitConfig = OPTIN so
  // smem usage up to the opt-in per-block limit is legal (the engine models
  // smem_per_block_max_bytes the same way).
  cudaOccDeviceProp p;
  memset(&p, 0, sizeof(p));
  p.computeMajor = 8;
  p.computeMinor = 0;
  if (std::string(c.arch) == "sm_75") { p.computeMajor = 7; p.computeMinor = 5; }
  if (std::string(c.arch) == "sm_86") p.computeMinor = 6;
  if (std::string(c.arch) == "sm_89") p.computeMinor = 9;
  if (std::string(c.arch) == "sm_90") p.computeMajor = 9;
  p.maxThreadsPerBlock = 1024;
  p.maxThreadsPerMultiprocessor = a.max_threads_per_sm;
  p.regsPerBlock = a.regs_per_sm;
  p.regsPerMultiprocessor = a.regs_per_sm;
  p.warpSize = a.warp_size;
  p.sharedMemPerMultiprocessor = a.smem_per_sm_bytes;
  p.sharedMemPerBlockOptin = a.smem_per_block_max_bytes;
  p.sharedMemPerBlock = p.sharedMemPerBlockOptin;  // opt-in configured kernel
  p.numSms = 1;
  p.reservedSharedMemPerBlock = 0;

  cudaOccFuncAttributes f;
  memset(&f, 0, sizeof(f));
  f.maxThreadsPerBlock = 1024;
  f.numRegs = c.regs;
  f.sharedSizeBytes = static_cast<size_t>(c.smem);
  f.partitionedGCConfig = PARTITIONED_GC_OFF;
  f.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
  f.maxDynamicSharedSizeBytes = 0;
  f.numBlockBarriers = 1;

  cudaOccDeviceState s;
  memset(&s, 0, sizeof(s));
  s.cacheConfig = CACHE_PREFER_NONE;
  s.carveoutConfig = SHAREDMEM_CARVEOUT_DEFAULT;

  cudaOccResult r;
  memset(&r, 0, sizeof(r));
  ASSERT_EQ(cudaOccMaxActiveBlocksPerMultiprocessor(&r, &p, &f, &s, c.block, 0),
            CUDA_OCC_SUCCESS);

  const OccupancyResult engine = compute_occupancy(a, c.regs, c.smem, c.block);
  EXPECT_EQ(engine.blocks_per_sm, r.activeBlocksPerMultiprocessor)
      << "oracle limiting factors: regs=" << r.blockLimitRegs
      << " smem=" << r.blockLimitSharedMem << " warps=" << r.blockLimitWarps
      << " blocks=" << r.blockLimitBlocks;
  const double oracle_occupancy =
      static_cast<double>(r.activeBlocksPerMultiprocessor) *
      // NOLINTNEXTLINE(bugprone-integer-division)  // intentional ceil-div (blocks per wave)
      static_cast<double>((c.block + a.warp_size - 1) / a.warp_size) /
      static_cast<double>(engine.max_warps_per_sm);
  EXPECT_NEAR(engine.occupancy, oracle_occupancy, 1e-9);
#endif
}

INSTANTIATE_TEST_SUITE_P(
    Occupancy, NvidiaOracleGridTest, ::testing::ValuesIn(MakeNvidiaGrid()),
    [](const ::testing::TestParamInfo<NvidiaGridCase> &info) {
      // Human-readable per-case ctest names (e.g. "sm_86_r112_s0_b64");
      // gtest's default byte-dump names are useless in ctest -N output.
      std::string name = info.param.arch;
      name += "_r" + std::to_string(info.param.regs);
      name += "_s" + std::to_string(info.param.smem);
      name += "_b" + std::to_string(info.param.block);
      return name;
    });

// cuda_occupancy.h uses a per-sub-partition register model
// (numWarpsPerSubPartition = floor(16384 / regs_per_warp), cudaOccMaxBlocks-
// PerSMRegsLimit). The engine implements the guide formula the task pins;
// the two diverge exactly when the sub-partition floor loses a warp. Pin one
// such case with both values so the divergence is never silent.
TEST(occupancy, NvidiaOracleSubpartitionDivergence) {
#ifndef KERNSCOPE_NO_CUDA_OCCUPANCY
  const auto &a = get_arch("sm_86");
  cudaOccDeviceProp p;
  memset(&p, 0, sizeof(p));
  p.computeMajor = 8; p.computeMinor = 6;
  p.maxThreadsPerBlock = 1024;
  p.maxThreadsPerMultiprocessor = a.max_threads_per_sm;
  p.regsPerBlock = a.regs_per_sm;
  p.regsPerMultiprocessor = a.regs_per_sm;
  p.warpSize = a.warp_size;
  p.sharedMemPerMultiprocessor = a.smem_per_sm_bytes;
  p.sharedMemPerBlockOptin = a.smem_per_block_max_bytes;
  p.sharedMemPerBlock = p.sharedMemPerBlockOptin;
  p.numSms = 1; p.reservedSharedMemPerBlock = 0;
  cudaOccFuncAttributes f;
  memset(&f, 0, sizeof(f));
  f.maxThreadsPerBlock = 1024; f.numRegs = 112; f.sharedSizeBytes = 0;
  f.partitionedGCConfig = PARTITIONED_GC_OFF;
  f.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
  f.maxDynamicSharedSizeBytes = 0; f.numBlockBarriers = 1;
  cudaOccDeviceState s;
  memset(&s, 0, sizeof(s));
  s.cacheConfig = CACHE_PREFER_NONE; s.carveoutConfig = SHAREDMEM_CARVEOUT_DEFAULT;
  cudaOccResult r;
  memset(&r, 0, sizeof(r));
  ASSERT_EQ(cudaOccMaxActiveBlocksPerMultiprocessor(&r, &p, &f, &s, 64, 0),
            CUDA_OCC_SUCCESS);

  const auto engine = compute_occupancy(a, 112, 0, 64);
  EXPECT_EQ(engine.blocks_per_sm, 9);  // guide formula: floor(65536/7168)
  EXPECT_EQ(r.activeBlocksPerMultiprocessor, 8)
      << "cuda_occupancy.h floors warps per 16K-register sub-partition "
         "([TURING] §3.5.1; cuda_occupancy.h cudaOccMaxBlocksPerSMRegsLimit: "
         "numWarpsPerSubPartition = floor(16384/3584) = 4 -> 4x4/2 = 8). The "
         "kernscope engine implements the CPG guide formula the task pins "
         "(brief §1.2 L101-102), so 112 regs/64 thr yields 9 blocks (0.375, "
         "the todo-5 pinned golden) — the two models agree on the 342-case "
         "grid and diverge only in cases like this one";
#endif
}

// ---------------------------------------------------------------------------
// AMD oracle. The GCNSubtarget occupancy math is asserted against its
// documented form (see file header): occ.sh threshold table for CDNA and
// LLVM AMDGPUBaseInfo getNumWavesPerEUWithNumVGPRs for RDNA3.
// ---------------------------------------------------------------------------

// CDNA: waves/SIMD = floor(512 / ceil8(vgpr)) — the occ.sh threshold table
// (brief §1.3 L113-116): <=64->8, 65-72->7, 73-80->6, 81-96->5, 97-128->4,
// 129-168->3, 169-256->2, >256->1. Single-wave64 blocks (64 threads):
// blocks/CU = waves x 4 SIMDs; occupancy = blocks / 32 waves.
TEST(occupancy, AmdCdnaOccShThresholdSweep) {
  struct Row { int vgpr; int waves; };
  const Row kRows[] = {{64, 8}, {72, 7},  {80, 6},  {96, 5},
                       {128, 4}, {168, 3}, {256, 2}, {257, 1}};
  for (const char *arch : {"gfx90a", "gfx942"}) {
    for (const Row &r : kRows) {
      SCOPED_TRACE(std::string(arch) + " vgpr=" + std::to_string(r.vgpr));
      const auto o = compute_occupancy(get_arch(arch), r.vgpr, 0, 64);
      EXPECT_EQ(o.blocks_per_sm, r.waves * 4);  // waves/SIMD x 4 SIMDs/CU
      EXPECT_NEAR(o.occupancy,
                  static_cast<double>(r.waves * 4) / 32.0, 1e-9);
      EXPECT_STREQ(limiter_name(o.limiter), "registers");
    }
  }
}

// RDNA3: waves/SIMD = clamp(floor(1536 / align24(vgpr)), 1, 16) (wave32
// view; rdna3-constants.md + LLVM AMDGPUBaseInfo.cpp L1371-1380). Single
// wave32 blocks (32 threads): blocks/CU = waves x 2 SIMDs. The occ.sh
// threshold table is CDNA-specific (64-lane waves); RDNA3 is validated
// against the LLVM-documented formula, which is the exact function the
// AMDGPU backend uses for occupancy.
TEST(occupancy, AmdRdna3LlvWaveFormulaSweep) {
  const auto &a = get_arch("gfx1101");
  struct Row { int vgpr; int waves; };
  const Row kRows[] = {{24, 16}, {48, 16}, {97, 12}, {160, 9},
                       {200, 7}, {256, 5}, {257, 5}};
  for (const Row &r : kRows) {
    SCOPED_TRACE("gfx1101 vgpr=" + std::to_string(r.vgpr));
    const int aligned = ((r.vgpr + 23) / 24) * 24;  // align24 (wave32 granule)
    const int llvm_waves =
        std::max(1, std::min(1536 / aligned, a.max_warps_per_sm));
    EXPECT_EQ(llvm_waves, r.waves)
        << "RDNA3 wave count must follow LLVM getNumWavesPerEUWithNumVGPRs "
           "(rdna3-constants.md evidence (d)); granule 24 wave32 / 12 wave64 "
           "per RDNA3 ISA §3.3.2.1";
    const auto o = compute_occupancy(a, r.vgpr, 0, 32);
    EXPECT_EQ(o.blocks_per_sm, llvm_waves * 2);  // 2 SIMDs/CU
    EXPECT_NEAR(o.occupancy,
                static_cast<double>(llvm_waves * 2) / 32.0, 1e-9);
  }
}

// AMD LDS limiter: 64 KB/CU, 512-byte allocation granule
// (brief §1.3 L123-127). blocks_by_lds = floor(65536 / ceil512(smem)).
// 32-thread blocks keep the thread limit (64 blocks) above the LDS limits
// under test; the wave/VGPR cap (32 blocks max at 16 VGPRs) masks only
// LDS limits >= 32, so the granule cases below use values < 32 blocks.
TEST(occupancy, AmdLdsLimiterAndGranule) {
  for (const char *arch : {"gfx90a", "gfx1101"}) {
    // 32 KB/block -> ceil512(32K)=32K -> 2 blocks (shared), low-reg kernels.
    auto o = compute_occupancy(get_arch(arch), 16, 32 * 1024, 32);
    EXPECT_EQ(o.blocks_per_sm, 2) << arch;
    EXPECT_STREQ(limiter_name(o.limiter), "shared") << arch;
    // 512-byte granule: smem 2100 B -> ceil512 = 2560 -> floor(65536/2560)
    // = 25 blocks. WITHOUT the granule the naive floor(65536/2100) = 31,
    // so this pins the LDS 512-B allocation granularity [brief §1.3
    // L123-125].
    o = compute_occupancy(get_arch(arch), 16, 2100, 32);
    EXPECT_EQ(o.blocks_per_sm, 25) << arch;
    EXPECT_STREQ(limiter_name(o.limiter), "shared") << arch;
  }
}

// AMD SGPR limiter (5-arg overload; CDNA only — brief §1.3 L117-121, pool
// ~800/SIMD granule 16). vgprs=16 gives 32 blocks (VGPR-unlimited); sgprs=100
// -> align16=112 -> floor(800/112)=7 waves -> 28 blocks binds.
TEST(occupancy, AmdSgprLimiter) {
  const auto o =
      compute_occupancy(get_arch("gfx90a"), /*regs_per_thread=*/16, /*smem_bytes_per_block=*/0,
                        /*block_size=*/64, /*sgprs_per_thread=*/100);
  EXPECT_EQ(o.blocks_per_sm, 28);
  EXPECT_NEAR(o.occupancy, 28.0 / 32.0, 1e-9);
  EXPECT_STREQ(limiter_name(o.limiter), "registers");
  // sgprs=30 -> align16=32 -> 800/32=25 waves -> clamped to 8 -> not binding
  // vs the 8-wave VGPR limit (both give 32 blocks; registers reported).
  const auto o2 =
      compute_occupancy(get_arch("gfx90a"), /*regs_per_thread=*/64, /*smem_bytes_per_block=*/0,
                        /*block_size=*/64, /*sgprs_per_thread=*/30);
  EXPECT_EQ(o2.blocks_per_sm, 32);
}

// AMD thread/wave caps. A 2048-thread CDNA block = 32 wave64s = the full
// 32-wave CU -> 1 block at 100% occupancy (NOT 1/32 — the earlier draft
// expectation was wrong; the wave and thread limits coincide for warp-aligned
// blocks). On RDNA3 the max 1024-thread block = 32 wave32s = the full 32-wave
// arch model -> 1 block at 100%. (A 2048-thread gfx1101 block would need 64
// wave32s vs libs/arch's 32-wave model — max_warps_per_sm 16 in wave64-
// equivalent units per rdna3-constants.md — so the engine reports it as
// cannot-run; that corner is the arch DB's accounting, not a limiter case.)
TEST(occupancy, AmdThreadCap) {
  auto o = compute_occupancy(get_arch("gfx90a"), 16, 0, 2048);
  EXPECT_EQ(o.blocks_per_sm, 1);
  EXPECT_EQ(o.warps_resident, 32);
  EXPECT_NEAR(o.occupancy, 1.0, 1e-9);
  o = compute_occupancy(get_arch("gfx1101"), 16, 0, 1024);
  EXPECT_EQ(o.blocks_per_sm, 1);
  EXPECT_EQ(o.warps_resident, 32);
  EXPECT_NEAR(o.occupancy, 1.0, 1e-9);
}

// ---------------------------------------------------------------------------
// Negative golden: forcing the RDNA3 VGPR granule to 16 (the "don't hardcode
// 16" trap, brief §10.4 L551-553) must change the predicted occupancy away
// from the pinned granule-24 model. The failure message cites
// docs/research/rdna3-constants.md (the todo-6 negative-QA pattern).
// ---------------------------------------------------------------------------

TEST(occupancy, Rdna3Granule16TamperDiverges) {
  const auto &pinned = get_arch("gfx1101");
  kernscope::ArchInfo tampered = pinned;
  tampered.reg_granule = 16;        // the refuted 16-granule model
  tampered.vgpr_granule_wave64 = 8;

  // 97 VGPRs wave32: pinned granule 24 -> align24 = 120 -> 12 waves/SIMD ->
  // 24 blocks; tampered granule 16 -> align16 = 112 -> floor(1536/112) = 13
  // waves -> 26 blocks. Must diverge.
  const auto pinned_occ = compute_occupancy(pinned, 97, 0, 32);
  const auto tampered_occ = compute_occupancy(tampered, 97, 0, 32);
  EXPECT_EQ(pinned_occ.blocks_per_sm, 24)
      << "gfx1101 97 VGPRs x 32 threads -> 24 blocks/CU with the PINNED "
         "granule 24 (floor(1536/align24(97)) = 12 waves/SIMD x 2 SIMDs): "
         "docs/research/rdna3-constants.md (RDNA3 ISA §3.3.2.1; LLVM "
         "AMDGPUBaseInfo.cpp L1371-1380; evidence (e) step series)";
  EXPECT_NE(pinned_occ.blocks_per_sm, tampered_occ.blocks_per_sm)
      << "tampering gfx1101 VGPR granule to 16 must change occupancy: "
         "docs/research/rdna3-constants.md pins 24 (wave32)/12 (wave64) from "
         "RDNA3 ISA §3.3.2.1 (brief §10.4 L551-553: \"don't hardcode 16\")";
  EXPECT_NE(tampered_occ.blocks_per_sm, 24)
      << "26 blocks would imply the refuted granule-16 model "
         "(docs/research/rdna3-constants.md evidence (e): align-12/24 step "
         "series, not align-16)";
}

}  // namespace
