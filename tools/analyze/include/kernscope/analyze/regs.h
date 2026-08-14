//===- regs.h - Register pressure + spill detection analysis --------------===//
//
// Wave-3 todo 13: analyzer module #1+#9 (register pressure, spill detection).
//
// Register pressure estimate, in order of precedence:
//   (1) ptxas -v output when available — parse BOTH published formats
//       [brief §6.1 L397-409, NVCC §8.4]:
//         * cubin-build one-liner: "ptxas info    : Used 102 registers, used
//           1 barriers, 388 bytes cmem[0]" (fused_attention manifest);
//         * the NVCC doc two-part form: "N bytes stack frame, M bytes spill
//           stores, K bytes spill loads" + "Used 6 registers, 1536 bytes
//           smem, 32 bytes cmem[0]". Each field is grabbed with an
//           independent regex, so a field missing on some arch stays 0
//           (polykernel PtxasParser oracle, same extraction shape);
//   (2) static proxy from the parsed PTX: declared .reg register counts
//       (an UPPER-BOUND estimate — ptxas reuses registers, so the declared
//       sum can exceed the live count; documented, not a lie) plus the
//       .local frame size in bytes.
//
// Spill detection (PTX): ld.local/st.local presence + .local frame bytes
// [brief §6.1 L386-409]. (AMD): scratch_load/scratch_store traffic; the
// amdhsa next_free_vgpr figure lives in the code-object notes (read via
// llvm-readobj --notes, brief §6.2) and is not present in .s text — the
// AMD path therefore counts scratch ops and notes the vgpr figure as
// code-object-only evidence.
//
// Findings: warning id "spill" when any spill signal is present, with
// machine-readable evidence (regs_per_thread, spill_bytes, stack_bytes,
// spill_stores, spill_loads). When a kernel declares ZERO .reg registers
// (a non-kernel body), the module emits the graceful diagnostic
// "cannot estimate register pressure" instead of guessing (QA failure path).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_ANALYZE_REGS_H
#define KERNSCOPE_ANALYZE_REGS_H

#include <string>
#include <string_view>

#include "kernscope/amd.h"
#include "kernscope/ptx.h"
#include "kernscope/report.h"

namespace kernscope::analyze {

/// Parsed ptxas -v resource figures. `ok` is set iff a "Used N registers"
/// line was found (the validity anchor: present in both ptxas formats).
/// Each optional field defaults to 0 when absent (per-field tolerance).
struct PtxasVReport {
  bool ok = false;
  int regs_per_thread = 0;     ///< "Used N registers"
  int stack_frame_bytes = 0;   ///< "N bytes stack frame"
  int spill_stores_bytes = 0;  ///< "N bytes spill stores"
  int spill_loads_bytes = 0;   ///< "N bytes spill loads"
  int smem_bytes = 0;          ///< "N bytes smem"
  int cmem_bytes = 0;          ///< "N bytes cmem[0]"
};

/// Parse ptxas -v resource text (both formats). Returns {ok=false} when the
/// text is not a ptxas log (no "N registers" anchor) — never throws.
PtxasVReport parse_ptxas_v(std::string_view text) noexcept;

/// Static proxy register-pressure estimate from the parsed kernel body.
/// `can_estimate` is false exactly when the kernel declares ZERO .reg
/// registers (nothing to estimate from).
struct RegPressureEstimate {
  bool can_estimate = false;
  int declared_regs = 0;     ///< sum of declared .reg counts (upper bound)
  int stack_frame_bytes = 0; ///< .local frame bytes (stack size)
  int spill_stores = 0;      ///< st.local count
  int spill_loads = 0;       ///< ld.local count
  int local_access_count = 0;///< total ld/st .local traffic
};

/// Static proxy from a parsed kernel (sums declared .reg counts, .local
/// frame bytes, and ld/st.local spill traffic).
RegPressureEstimate estimate_reg_pressure(const PtxKernel &kernel) noexcept;

/// Analyze register pressure + spills from a parsed PTX module using the
/// static proxy. Emits metrics (registers_per_thread, spill_stores,
/// spill_loads, local_stack_bytes, spill_bytes) and findings into `report`.
/// Graceful "cannot estimate register pressure" warning when the kernel
/// declares no registers (never a crash, never a fabricated number).
void analyze_regs(const PtxModule &module, KernelReport &report);

/// Overload: when a ptxas -v report is available (e.g. from the cubin build
/// or the committed manifest), prefer its exact figures over the proxy.
void analyze_regs(const PtxModule &module, std::string_view ptxas_v,
                  KernelReport &report);

/// AMD overload: counts scratch_load/scratch_store spill traffic.
void analyze_regs(const AmdModule &module, KernelReport &report);

}  // namespace kernscope::analyze

#endif  // KERNSCOPE_ANALYZE_REGS_H
