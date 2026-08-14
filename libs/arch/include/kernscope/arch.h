//===- arch.h - GPU architecture-model database (constants + occupancy) ---===//
//
// Wave-1 todo 6: the arch-model DB consumed by tools/analyze (and later the
// whole kernscope suite). Every constant is derived from
// docs/research/gpu-static-analyzer-research.md §1 (occupancy model):
//   §1.1  general occupancy formula (min-wins rule), lines 31-64
//   §1.2  NVIDIA per-architecture constants, lines 66-102
//         (CPG v13.3 compute-capability tables 30/31/32)
//   §1.3  AMD constants, lines 104-144 (CDNA2/CDNA3 + RDNA3)
// RDNA3 VGPR-file values are ISA-resolved in
// docs/research/rdna3-constants.md (1536 VGPRs/SIMD, granule 24 wave32 /
// 12 wave64 — NOT 16).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_LIBS_ARCH_INCLUDE_KERNSCOPE_ARCH_H
#define KERNSCOPE_LIBS_ARCH_INCLUDE_KERNSCOPE_ARCH_H

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace kernscope {

enum class Vendor : std::uint8_t { kNvidia, kAmd };

// One row of the architecture DB. All quantities are per-SM for NVIDIA and
// per-CU for AMD unless the field name says otherwise (AMD register files are
// per-SIMD, matching the ISA wording).
struct ArchInfo {
  // DB key, e.g. "sm_86" / "gfx1101". Also the name reported in diagnostics.
  std::string_view name;
  Vendor vendor;

  // --- common (brief §1.1: warp size NV=32 [CPG Table 30], AMD=64 (CDNA) /
  // 32 (RDNA3 native wave32) [brief §1.3 L136]) ---
  int warp_size;             // threads per warp / wavefront
  int max_threads_per_sm;    // threads per SM (NVIDIA) / per CU (AMD)
  int max_blocks_per_sm;     // workgroup slots per SM/CU; 0 = not modeled
  int smem_per_sm_bytes;     // shared mem per SM (NVIDIA) / LDS per CU (AMD)
  int smem_per_block_max_bytes;  // opt-in max per block; 0 = not modeled

  // --- register file ---
  // NVIDIA: 64K 32-bit registers/SM, 255 regs max/thread [brief §1.2 L70-71];
  // allocation granule 8 regs/thread = 256 regs/warp [brief §1.2 L91-99].
  // AMD: VGPR file per SIMD (512 CDNA [brief §1.3 L108-112],
  // 1536 RDNA3-wave32 [rdna3-constants.md]), VGPR granule (8 CDNA,
  // 24 RDNA3 wave32 / 12 wave64), max 256 VGPRs per lane [brief §1.3].
  int regs_per_sm;              // NVIDIA: regs per SM; AMD: VGPR file per SIMD
  int max_regs_per_thread;      // per-thread register ceiling
  int reg_granule;              // per-thread allocation granule
  int vgpr_granule_wave64;      // AMD only: RDNA3 wave64 granule (12); else 0
  int max_warps_per_sm;         // NVIDIA: warps/SM; AMD: max waves per SIMD
  int simds_per_cu;             // AMD only (4 CDNA, 2 RDNA3); NVIDIA: 1
  int sgpr_granule;             // AMD only [brief §1.3 L117-121]; NVIDIA: 0

  // Tensor-core input dtypes per CC, CPG Table 33 via brief §5.1 L343-344.
  // (7.5: f16/s8/s4; 8.0 adds f64/tf32/bf16; 8.9 adds FP8; 10.0 adds FP6/FP4;
  // 12.x: tf32/bf16/f16/fp8/fp6/fp4/int8.)
  std::array<std::string_view, 10> tensor_dtypes{};
  std::uint8_t tensor_dtype_count = 0;
};

// The DB itself. NVIDIA rows: brief §1.2 table L74-84 + §1.1 L70-72.
// AMD rows: brief §1.3 L104-144; gfx1101 VGPR values per rdna3-constants.md.
namespace arch_table {
constexpr ArchInfo kSm75{
    .name = "sm_75", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 1024, .max_blocks_per_sm = 16,
    .smem_per_sm_bytes = 64 * 1024, .smem_per_block_max_bytes = 64 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 32, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4"},
    .tensor_dtype_count = 3,
};
constexpr ArchInfo kSm80{
    .name = "sm_80", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 2048, .max_blocks_per_sm = 32,
    .smem_per_sm_bytes = 164 * 1024, .smem_per_block_max_bytes = 163 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 64, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4", "f64", "tf32", "bf16"},
    .tensor_dtype_count = 6,
};
constexpr ArchInfo kSm86{
    .name = "sm_86", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 1536, .max_blocks_per_sm = 16,
    .smem_per_sm_bytes = 100 * 1024, .smem_per_block_max_bytes = 99 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 48, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4", "f64", "tf32", "bf16"},
    .tensor_dtype_count = 6,
};
constexpr ArchInfo kSm89{
    .name = "sm_89", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 1536, .max_blocks_per_sm = 24,
    .smem_per_sm_bytes = 100 * 1024, .smem_per_block_max_bytes = 99 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 48, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4", "f64", "tf32", "bf16", "fp8"},
    .tensor_dtype_count = 7,
};
constexpr ArchInfo kSm90{
    .name = "sm_90", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 2048, .max_blocks_per_sm = 32,
    .smem_per_sm_bytes = 228 * 1024, .smem_per_block_max_bytes = 227 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 64, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4", "f64", "tf32", "bf16", "fp8"},
    .tensor_dtype_count = 7,
};
constexpr ArchInfo kSm100{
    .name = "sm_100", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 2048, .max_blocks_per_sm = 32,
    .smem_per_sm_bytes = 228 * 1024, .smem_per_block_max_bytes = 227 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 64, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"f16", "s8", "s4", "f64", "tf32", "bf16", "fp8", "fp6",
                      "fp4"},
    .tensor_dtype_count = 9,
};
constexpr ArchInfo kSm120{
    .name = "sm_120", .vendor = Vendor::kNvidia,
    .warp_size = 32, .max_threads_per_sm = 1536, .max_blocks_per_sm = 24,
    .smem_per_sm_bytes = 100 * 1024, .smem_per_block_max_bytes = 99 * 1024,
    .regs_per_sm = 65536, .max_regs_per_thread = 255, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 48, .simds_per_cu = 1,
    .sgpr_granule = 0,
    .tensor_dtypes = {"tf32", "bf16", "f16", "fp8", "fp6", "fp4", "int8"},
    .tensor_dtype_count = 7,
};
constexpr ArchInfo kGfx90a{
    .name = "gfx90a", .vendor = Vendor::kAmd,
    .warp_size = 64, .max_threads_per_sm = 2048, .max_blocks_per_sm = 0,
    .smem_per_sm_bytes = 64 * 1024, .smem_per_block_max_bytes = 0,
    .regs_per_sm = 512, .max_regs_per_thread = 256, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 8, .simds_per_cu = 4,
    .sgpr_granule = 16,
    .tensor_dtypes = {},
    .tensor_dtype_count = 0,
};
constexpr ArchInfo kGfx942{
    .name = "gfx942", .vendor = Vendor::kAmd,
    .warp_size = 64, .max_threads_per_sm = 2048, .max_blocks_per_sm = 0,
    .smem_per_sm_bytes = 64 * 1024, .smem_per_block_max_bytes = 0,
    .regs_per_sm = 512, .max_regs_per_thread = 256, .reg_granule = 8,
    .vgpr_granule_wave64 = 0, .max_warps_per_sm = 8, .simds_per_cu = 4,
    .sgpr_granule = 16,
    .tensor_dtypes = {},
    .tensor_dtype_count = 0,
};
constexpr ArchInfo kGfx1101{
    .name = "gfx1101", .vendor = Vendor::kAmd,
    .warp_size = 32, .max_threads_per_sm = 2048, .max_blocks_per_sm = 0,
    .smem_per_sm_bytes = 64 * 1024, .smem_per_block_max_bytes = 64 * 1024,
    .regs_per_sm = 1536, .max_regs_per_thread = 256, .reg_granule = 24,
    .vgpr_granule_wave64 = 12, .max_warps_per_sm = 16, .simds_per_cu = 2,
    .sgpr_granule = 16,
    .tensor_dtypes = {},
    .tensor_dtype_count = 0,
};
}  // namespace arch_table

inline constexpr std::array<std::string_view, 10> kArchNames{
    "sm_75", "sm_80", "sm_86", "sm_89", "sm_90", "sm_100", "sm_120",
    "gfx90a", "gfx942", "gfx1101"};

// Lookup by name ("sm_86", "gfx1101", ...). Throws std::out_of_range for
// unknown keys.
inline const ArchInfo &get_arch(std::string_view name) {
  const ArchInfo *archs[] = {&arch_table::kSm75,  &arch_table::kSm80,
                             &arch_table::kSm86,  &arch_table::kSm89,
                             &arch_table::kSm90,  &arch_table::kSm100,
                             &arch_table::kSm120, &arch_table::kGfx90a,
                             &arch_table::kGfx942, &arch_table::kGfx1101};
  for (const ArchInfo *a : archs)
    if (a->name == name) return *a;
  throw std::out_of_range(std::string("kernscope: unknown arch '") +
                          std::string(name) + "'");
}

inline bool has_arch(std::string_view name) noexcept {
  for (std::string_view n : kArchNames)
    if (n == name) return true;
  return false;
}

// Max resident warps: NVIDIA = warps/SM; AMD = max waves/SIMD x SIMDs/CU
// (CDNA: 8 x 4 = 32 waves/CU; RDNA3: 16 x 2 = 32 wave64-equiv waves/CU).
// [brief §1.1 L33-44, §1.3 L122, L140-141]
inline int max_resident_warps(const ArchInfo &a) noexcept {
  return a.max_warps_per_sm * a.simds_per_cu;
}

// Register allocation after the granule rounds up per thread: NVIDIA rounds
// per-thread usage to a multiple of 8 (256-register warp granule)
// [brief §1.2 L91-99]; AMD rounds per lane to the VGPR granule
// [brief §1.3 L108-112 (CDNA 8), L137-139 (RDNA3 24 wave32)].
inline int allocated_regs_per_thread(const ArchInfo &a,
                                     int regs_per_thread) noexcept {
  const int g = a.reg_granule > 0 ? a.reg_granule : 1;
  return ((regs_per_thread + g - 1) / g) * g;
}

// AMD: resident waves per SIMD given per-lane VGPR usage.
// CDNA: waves = floor(512 / ceil8(vgpr)) [brief §1.3 L112] — equivalent to
// ROCm's Triton occ.sh threshold table (<=64->8, 65-72->7, 73-80->6, 81-96->5,
// 97-128->4, 129-168->3, 169-256->2, >256->1) [brief §1.3 L113-116].
// RDNA3: waves = floor(1536 / ceil24(vgpr)) clamped by the wave cap
// [brief §1.3 L139-141; granule per docs/research/rdna3-constants.md].
inline int waves_per_simd_from_vgprs(const ArchInfo &a, int vgprs) noexcept {
  if (vgprs <= 0) return a.max_warps_per_sm;
  const int g = a.reg_granule > 0 ? a.reg_granule : 1;
  const int aligned = ((vgprs + g - 1) / g) * g;
  const int waves = a.regs_per_sm / aligned;  // per-SIMD file
  if (waves < 1) return 1;
  return waves < a.max_warps_per_sm ? waves : a.max_warps_per_sm;
}

struct OccupancyPrediction {
  int blocks_per_cu;      // resident workgroups per SM/CU (min-wins rule)
  int active_warps;       // resident warps = blocks x warps_per_block
  int max_warps;          // = max_resident_warps
  double occupancy_pct;   // active / max
  const char *limiter;    // "registers" | "smem" | "threads" | "warps" |
                          // "blocks"
};

// Predict resident blocks with the min-wins rule [brief §1.1 L33-45]:
//   blocks = min(threads, warps, smem, regs, blocks-cap)  (each floor-div)
// 0 means "not modeled" for a limiter (AMD workgroup-slot cap, CDNA
// per-block smem opt-in).
inline OccupancyPrediction predict_occupancy(const ArchInfo &a,
                                             int threads_per_block,
                                             int regs_per_thread,
                                             int smem_per_block_bytes) noexcept {
  OccupancyPrediction out{};
  out.max_warps = max_resident_warps(a);
  const int warps_per_block =
      (threads_per_block + a.warp_size - 1) / a.warp_size;
  const int limit_threads =
      threads_per_block > 0 ? a.max_threads_per_sm / threads_per_block : 0;
  const int limit_warps =
      warps_per_block > 0 ? out.max_warps / warps_per_block : 0;
  const int limit_smem = smem_per_block_bytes > 0 && a.smem_per_sm_bytes > 0
                             ? a.smem_per_sm_bytes / smem_per_block_bytes
                             : -1;  // no smem requested -> not a limiter
  const int limit_blocks = a.max_blocks_per_sm > 0 ? a.max_blocks_per_sm : -1;

  // Register limiter: NVIDIA 65536 / (granule-allocated regs x threads)
  // [brief §1.2 L101-102]; AMD waves-per-SIMD x SIMDs / warps-per-block
  // [brief §1.3 L112, L139].
  const int alloc = allocated_regs_per_thread(a, regs_per_thread);
  const int limit_regs =
      a.vendor == Vendor::kNvidia
          ? (alloc * threads_per_block > 0
                 ? a.regs_per_sm / (alloc * threads_per_block)
                 : 0)
          : (waves_per_simd_from_vgprs(a, regs_per_thread) * a.simds_per_cu) /
                warps_per_block;

  // Min-wins with a fixed tie-break so the binding limiter is deterministic:
  // on equal block counts prefer registers > warps > smem > threads > blocks
  // (matches the polykernel oracle, e.g. the 31-reg/256-thread sm_90 case
  // which reports "registers" on an 8/8/8 tie).
  int blocks = limit_threads;
  out.limiter = "threads";
  auto pick = [&blocks, &out](int limit, const char *limiter, int priority) {
    if (limit < 0) return;  // "not modeled" sentinel: not a real limiter
    const int cur = blocks;
    int cur_priority = 0;
    if (std::string_view(out.limiter) == "registers") cur_priority = 5;
    else if (std::string_view(out.limiter) == "warps") cur_priority = 4;
    else if (std::string_view(out.limiter) == "smem") cur_priority = 3;
    else if (std::string_view(out.limiter) == "threads") cur_priority = 2;
    else if (std::string_view(out.limiter) == "blocks") cur_priority = 1;
    if (limit < cur || (limit == cur && priority > cur_priority)) {
      blocks = limit;
      out.limiter = limiter;
    }
  };

  pick(limit_warps, "warps", 4);
  pick(limit_smem, "smem", 3);
  pick(limit_regs, "registers", 5);
  pick(limit_blocks, "blocks", 1);
  out.blocks_per_cu = blocks;
  out.active_warps = blocks * warps_per_block;
  out.occupancy_pct = out.max_warps > 0
                          ? 100.0 * static_cast<double>(out.active_warps) /
                                static_cast<double>(out.max_warps)
                          : 0.0;
  return out;
}

inline bool has_tensor_dtype(const ArchInfo &a, std::string_view dtype) noexcept {
  for (std::uint8_t i = 0; i < a.tensor_dtype_count; ++i)
    if (a.tensor_dtypes[i] == dtype) return true;
  return false;
}

}  // namespace kernscope

#endif  // KERNSCOPE_LIBS_ARCH_INCLUDE_KERNSCOPE_ARCH_H
