//===- analyze_all.cpp - full-kernel analysis orchestrator ----------------===//
//
// Wave-3 todo 19. See analyze.h for the module ordering rationale.
//
// Golden reconciliation (documented, occupancy.h "GOLDEN RECONCILIATION"):
// fused_attention uses DYNAMIC shared memory (extern .shared smem[]), whose
// size exists only in the launch config, not the PTX text. The theoretical
// occupancy is therefore pinned at the register-limited combo (9 blocks x 2
// warps / 48 = 0.375, limiter "registers") exactly as
// tests/fixtures/manifest.json pins it; when the caller supplies the dynamic
// opt-in size it is recorded as the shared_memory_bytes metric (a launch-
// config datum) but NOT fed into the occupancy engine, which would otherwise
// switch the limiter to shared and break the pinned golden.
//
//===----------------------------------------------------------------------===//

#include "kernscope/analyze/analyze.h"

#include <cmath>
#include <string>

#include "kernscope/analyze/banks.h"
#include "kernscope/analyze/coalescing.h"
#include "kernscope/analyze/divergence.h"
#include "kernscope/analyze/mix.h"
#include "kernscope/analyze/occupancy.h"
#include "kernscope/analyze/regs.h"
#include "kernscope/analyze/suggest.h"

namespace kernscope::analyze {
namespace {

// The report model requires every metric to be a finite number (JSON has no
// NaN; libs/report's renderers throw SchemaViolation otherwise). The
// occupancy module records "not modeled" headrooms as quiet_NaN (e.g.
// headroom_blocks_by_smem when the kernel requests no shared memory) — such
// metrics are omitted from the final report so it always renders.
void drop_non_finite_metrics(KernelReport &report) {
  for (auto it = report.metrics.begin(); it != report.metrics.end();) {
    if (!std::isfinite(it->second))
      it = report.metrics.erase(it);
    else
      ++it;
  }
}

}  // namespace

KernelReport analyze_ptx_module(const PtxModule &module) {
  return analyze_ptx_module(module, AnalyzeInputs{});
}

KernelReport analyze_ptx_module(const PtxModule &module,
                                const AnalyzeInputs &inputs) {
  KernelReport report;

  // Kernel identity: prefer the caller's (mangled PTX names are not
  // user-facing), else the first entry kernel's PTX name.
  report.kernel_name = inputs.kernel_name;
  if (report.kernel_name.empty()) {
    for (const PtxKernel &k : module.kernels()) {
      if (k.entry_point) {
        report.kernel_name = k.name;
        break;
      }
    }
    if (report.kernel_name.empty() && !module.kernels().empty())
      report.kernel_name = module.kernels()[0].name;
  }

  // 1. registers (todo 13): ptxas -v when supplied, else the static proxy.
  if (!inputs.ptxas_v.empty())
    analyze_regs(module, inputs.ptxas_v, report);
  else
    analyze_regs(module, report);

  // 2. occupancy + limiter (todo 14). The dynamic smem opt-in is recorded as
  // a metric only (see header note) — the occupancy engine runs on the
  // static figure so the extern-dynamic golden stays register-limited.
  analyze_occupancy(
      module,
      OccupancyInputs{inputs.block_size, /*dynamic_smem_bytes=*/0,
                      inputs.arch_override},
      report);
  if (inputs.dynamic_smem_bytes > 0)
    report.metrics["shared_memory_bytes"] = inputs.dynamic_smem_bytes;

  // 3-6. coalescing, banks, divergence, mix (todos 15-18).
  analyze_coalescing(module, report);
  analyze_banks(module, report);
  analyze_divergence(module, report);
  analyze_mix(module, report);

  // 7. suggestions (todo 19) — runs last, owns report.suggestions.
  analyze_suggest(module, report);

  // Normalize: omit non-finite ("not modeled") metrics so the report always
  // satisfies libs/report's finite-number requirement and can be rendered.
  drop_non_finite_metrics(report);

  return report;
}

}  // namespace kernscope::analyze
