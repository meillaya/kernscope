//===- analyze_occupancy.cpp - occupancy limits + main limiter ------------===//
//
// Wave-3 todo 14. See occupancy.h for the design rationale.
//
// GOLDEN RECONCILIATION (documented, per tests/fixtures/README.md): the
// fused_attention kernel uses dynamic shared memory (extern .shared smem[],
// opt-in via cudaFuncSetAttribute). The extern declaration carries NO size
// in the PTX text, so the STATIC .shared bytes parsed from the module are 0
// for this kernel, and the pinned theoretical occupancy is the
// register-limited CPG-guide formula (9 blocks x 2 warps / 48 = 0.375,
// limiter "registers") exactly as tests/fixtures/manifest.json pins it. The
// dynamic opt-in size (67584 B, from the launch config) is reported as the
// shared_memory_bytes metric when the caller supplies it, and the smem
// limiter is exercised by the module whenever static .shared declarations
// carry a size (tests cover both paths). This mirrors the todo-11 engine
// golden, which also runs the pinned combo at smem=0.
//
// "Theoretical" labeling: every occupancy figure emitted here is THEORETICAL
// (a static tool never measures achieved occupancy); the metrics carry
// occupancy_is_theoretical=1 and the report text keeps the distinction.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/occupancy.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

#include "kernscope/analyze/regs.h"
#include "kernscope/occupancy.h"

namespace kernscope::analyze {
namespace {

constexpr const char *kTheoreticalMarkerMetric = "occupancy_is_theoretical";

const ArchInfo &resolve_arch(const PtxModule &module,
                             const std::string &override) {
  const std::string &name = override.empty() ? module.target : override;
  return get_arch(name);
}

// Byte width of a PTX type token (".b8"=1, ".b16"=2, ".b32"=4, ".b64"=8,
// ".f16"=2, ".f32"=4, ".f64"=8, ".u8/16/32/64", ".s8/16/32/64", ".pred"=1).
int type_width_bytes(const std::string &type) noexcept {
  if (type == ".b8" || type == ".u8" || type == ".s8" || type == ".pred")
    return 1;
  if (type == ".b16" || type == ".u16" || type == ".s16" || type == ".f16")
    return 2;
  if (type == ".b64" || type == ".u64" || type == ".s64" || type == ".f64")
    return 8;
  return 4;  // default: 32-bit word (".b32"/".u32"/".s32"/".f32")
}

// Sum the sizes of STATIC .shared array declarations in the module
// (top-level + kernel-body): shape "[N]" is an ELEMENT count, so bytes =
// N x element width. Extern/dynamic `smem[]` declarations have an empty
// shape and contribute 0 bytes.
int static_shared_bytes(const PtxModule &module) noexcept {
  int total = 0;
  auto count = [&total](const PtxVariable &v) {
    if (v.space != StateSpace::kShared) return;
    if (v.shape.size() >= 2 && v.shape.front() == '[' && v.shape.back() == ']') {
      const std::string n = v.shape.substr(1, v.shape.size() - 2);
      char *end = nullptr;
      const long elems = std::strtol(n.c_str(), &end, 10);
      if (end != n.c_str() && elems > 0)
        total += static_cast<int>(elems) * type_width_bytes(v.type);
    }
  };
  for (const PtxVariable &v : module.variables) count(v);
  for (const PtxKernel &k : module.kernels())
    for (const PtxVariable &v : k.variables) count(v);
  return total;
}

// Per-limiter headroom: the independent block-count limit each resource
// would allow (before the min-wins). Same math as the engine / arch.h
// predict_occupancy (brief §1.1 L37-43); each number is reported separately
// so the report shows WHY a limiter won.
struct Headroom {
  int threads = 0;
  int warps = 0;
  int smem = -1;  // -1 = not modeled (no smem requested)
  int regs = 0;
  int blocks_cap = -1;
};
Headroom compute_headroom(const ArchInfo &a, int regs_per_thread,
                          int smem_per_block, int block_size) noexcept {
  Headroom h;
  const int warps_per_block =
      block_size > 0 ? (block_size + a.warp_size - 1) / a.warp_size : 0;
  h.threads = block_size > 0 ? a.max_threads_per_sm / block_size : 0;
  h.warps = warps_per_block > 0 ? max_resident_warps(a) / warps_per_block : 0;
  h.smem = smem_per_block > 0 && a.smem_per_sm_bytes > 0
               ? a.smem_per_sm_bytes / smem_per_block
               : -1;
  h.blocks_cap = a.max_blocks_per_sm > 0 ? a.max_blocks_per_sm : -1;
  if (a.vendor == Vendor::kNvidia) {
    const int alloc = allocated_regs_per_thread(a, regs_per_thread);
    h.regs = alloc * block_size > 0 ? a.regs_per_sm / (alloc * block_size) : 0;
  } else {
    h.regs = warps_per_block > 0
                 ? (waves_per_simd_from_vgprs(a, regs_per_thread) *
                    a.simds_per_cu) /
                       warps_per_block
                 : 0;
  }
  return h;
}

// Emit a warning when the requested block size exceeds the arch maximum.
void warn_oversized_block(const ArchInfo &a, int block_size,
                          KernelReport &report) {
  if (block_size <= a.max_threads_per_sm) return;
  Finding f;
  f.id = "block_size_exceeds_arch_max";
  f.severity = "warning";
  f.evidence["text"] =
      "requested block size exceeds the architecture's max threads per SM; "
      "theoretical occupancy is 0";
  f.evidence["block_size"] = std::to_string(block_size);
  f.evidence["max_threads_per_sm"] = std::to_string(a.max_threads_per_sm);
  report.warnings.push_back(std::move(f));
}

void fill_report(const ArchInfo &a, int regs_per_thread, int smem_bytes,
                 int block_size, KernelReport &report) {
  const occupancy::OccupancyResult r =
      occupancy::compute_occupancy(a, regs_per_thread, smem_bytes, block_size);
  report.metrics["blocks_per_sm"] = r.blocks_per_sm;
  report.metrics["warps_resident_per_sm"] = r.warps_resident;
  report.metrics["max_warps_per_sm"] = r.max_warps_per_sm;
  report.metrics["max_blocks_per_sm"] = a.max_blocks_per_sm;
  report.metrics["occupancy"] = r.occupancy;
  report.metrics[kTheoreticalMarkerMetric] = 1.0;
  report.limiter = occupancy::limiter_name(r.limiter);
  // Manifest field: 102 regs -> 104 (granule-8) x 32 = 3328 regs/warp.
  report.metrics["registers_allocated_per_warp"] =
      static_cast<double>(allocated_regs_per_thread(a, regs_per_thread)) *
      static_cast<double>(a.warp_size);

  const Headroom h = compute_headroom(a, regs_per_thread, smem_bytes,
                                      block_size);
  report.metrics["headroom_blocks_by_threads"] = h.threads;
  report.metrics["headroom_blocks_by_warps"] = h.warps;
  report.metrics["headroom_blocks_by_smem"] =
      h.smem >= 0 ? h.smem : std::numeric_limits<double>::quiet_NaN();
  report.metrics["headroom_blocks_by_registers"] = h.regs;
  report.metrics["headroom_blocks_by_blocks_cap"] =
      h.blocks_cap >= 0 ? h.blocks_cap
                        : std::numeric_limits<double>::quiet_NaN();

  warn_oversized_block(a, block_size, report);
}

}  // namespace

void analyze_occupancy(const PtxModule &module, KernelReport &report) {
  analyze_occupancy(module, OccupancyInputs{}, report);
}

void analyze_occupancy(const PtxModule &module, const OccupancyInputs &inputs,
                       KernelReport &report) {
  // Block size is not statically present in PTX text (the parser skips
  // .reqntid); it comes from the launch manifest (todo 5 fixture) or an
  // explicit input. Without it the theoretical occupancy is not computable.
  if (inputs.block_size <= 0) {
    report.metrics[kTheoreticalMarkerMetric] = 1.0;
    Finding f;
    f.id = "block_size_unknown";
    f.severity = "warning";
    f.evidence["text"] =
        "cannot compute theoretical occupancy: block size not provided "
        "(from .reqntid or the launch manifest)";
    report.warnings.push_back(std::move(f));
    return;
  }

  const ArchInfo &arch = resolve_arch(module, inputs.arch_override);

  // Registers: prefer the todo-13 figure (ptxas -v / proxy already in the
  // report), else fall back to the declared-reg proxy.
  int regs = 0;
  const auto it = report.metrics.find("registers_per_thread");
  if (it != report.metrics.end() && it->second > 0) {
    regs = static_cast<int>(it->second);
  } else if (!module.kernels().empty()) {
    const RegPressureEstimate est = estimate_reg_pressure(module.kernels()[0]);
    regs = est.can_estimate ? est.declared_regs : 0;
  }

  // Shared memory: static .shared decls + the dynamic opt-in param.
  const int smem = static_shared_bytes(module) + inputs.dynamic_smem_bytes;
  report.metrics["shared_memory_bytes"] = smem;

  if (regs <= 0) {
    report.metrics[kTheoreticalMarkerMetric] = 1.0;
    Finding f;
    f.id = "cannot_estimate_registers";
    f.severity = "warning";
    f.evidence["text"] = "cannot compute theoretical occupancy: no register figure";
    report.warnings.push_back(std::move(f));
    return;
  }

  fill_report(arch, regs, smem, inputs.block_size, report);
}

void analyze_occupancy(const ArchInfo &arch, int regs_per_thread,
                       int smem_bytes_per_block, int block_size,
                       KernelReport &report) {
  if (block_size <= 0) {
    report.metrics[kTheoreticalMarkerMetric] = 1.0;
    Finding f;
    f.id = "block_size_unknown";
    f.severity = "warning";
    f.evidence["text"] = "block size must be > 0";
    report.warnings.push_back(std::move(f));
    return;
  }
  report.metrics["shared_memory_bytes"] = smem_bytes_per_block;
  fill_report(arch, regs_per_thread, smem_bytes_per_block, block_size, report);
}

}  // namespace kernscope::analyze
