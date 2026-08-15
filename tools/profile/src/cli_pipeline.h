//===- cli_pipeline.h - kernscope-profile CLI pipeline (todo 52) ----------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track), todo 52: the kernscope-profile
// CLI pipeline library.
//
// The CLI orchestrates the whole todo-49/50/51 stack into one command:
//
//   1. TRACE   (todo 49): run the target binary under the vendor tracer
//      (LD_PRELOAD of the rocprofiler-sdk tool for AMD/HIP; the CUDA path is
//      compile+unit only locally — real-NVIDIA tracing is rental-gated, see
//      scripts/rental-validate.sh) with KERNSCOPE_PROFILE_OUT pointed at a
//      scratch file; the tracer emits one enriched JSONL line per kernel
//      dispatch.
//   2. ENRICH  (todo 50): parse the JSONL records back into the EnrichedKernel
//      model (the tracer already enriched them — occupancy/limiter via
//      libs/occupancy + the code-object resource facts).
//   3. RANK    (todo 51): rank_hot_kernels() — total/median per kernel name,
//      % of wall time.
//   4. DIAGNOSE (todo 51): diagnose_kernel() per record — the SOL-style rules
//      engine emitting libs/report Findings (every hint carries evidence).
//   5. REPORT  (libs/report todo 7 reuse): --json = the profile report model
//      (this file's ProfileReport), --text = the OWNER'S SECTION B SHAPE
//      (docs/research/owner-examples.md: Kernel/Time/Registers/Shared/Limiter/
//      Hint) with locally-measured values, --html = kernscope::to_html() of
//      the hottest kernel's KernelReport (embedded JSON payload guaranteed to
//      parse).
//
// The QA negative is a first-class path: a target that runs fine but traces
// ZERO kernel records yields no_kernels_traced == true and the report says
// "no kernels traced" with EXIT 0 (never an error).
//
// MUST NOT: require GPU counters/PerfWorks; claim vendor numbers the local
// run did not measure; fabricate hints (the diag engine's degradation rules
// apply as-is).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_CLI_PIPELINE_H
#define KERNSCOPE_PROFILE_CLI_PIPELINE_H

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "kernscope/profile/diagnostics.h"
#include "kernscope/profile/enrich.h"

namespace kernscope::profile::cli {

//===--- trace options -------------------------------------------------------===//
// Everything the trace runner needs to profile one target binary.
struct TraceOptions {
  std::string target;        // binary path or directory containing the binary
  std::string vendor;        // "hip" | "cuda" (auto-detection already resolved)
  std::string hip_tool;      // path to libkernscope_profile_hip.so
  std::string arch_name;     // roofline/diagnostics arch key ("gfx1101" local)
  std::string analyzer_cli;  // kernscope-analyze CLI path (diag subprocess)
  std::vector<std::string> app_args; // args passed to the target (after --)
  int top_n = 10;            // hot ranking list size
};

//===--- the trace result -----------------------------------------------------===//
struct TraceResult {
  bool ok = false;                    // app ran to completion (exit 0)
  std::string error;                  // why not (shown on stderr, exit 2)
  std::vector<EnrichedKernel> kernels; // every "kernel" JSONL record
  std::size_t api_records = 0;        // HIP API records (correlation side)
  std::string jsonl_path;             // the captured JSONL file (kept for
                                      // evidence when non-empty)
};

//===--- the report model ------------------------------------------------------===//
// One complete kernscope-profile report: the enriched per-kernel trace table,
// the hot ranking and the per-record diagnostics. no_kernels_traced is the
// QA-negative signal (exit 0, report says "no kernels traced").
struct ProfileReport {
  std::string vendor;
  std::string target;
  bool no_kernels_traced = false;
  std::vector<EnrichedKernel> kernels;     // per-record enriched trace table
  std::vector<HotKernelEntry> ranking;     // per-name hot ranking (todo 51)
  std::vector<KernelDiagnostics> diagnostics; // per-record (todo 51)
  std::uint64_t wall_time_ns = 0;
};

//===--- pipeline stages -------------------------------------------------------===//

/// Run the target binary under the HIP tracer (LD_PRELOAD) with the JSONL sink
/// in a fresh scratch dir. Seeds the scratch dir's input.bin for the committed
/// layernorm_fused fixture contract (tests/fixtures/README.md: M=1024 rows x
/// N=128 cols bf16, deterministic) when the target stem contains "layernorm"
/// — the app's default-args contract (input.bin/output.bin in the CWD).
/// Returns ok == false (with error) when the target cannot be resolved/run or
/// exits non-zero: a partial trace is never trusted. Never throws.
TraceResult run_trace(const TraceOptions &options);

/// Parse the tracer's enriched JSONL text into kernel records ("kernel" lines
/// only; "api" lines are counted, not decoded). Malformed lines are skipped
/// (never fatal). Never throws.
void parse_profile_jsonl(const std::string &jsonl_text,
                         std::vector<EnrichedKernel> *kernels,
                         std::size_t *api_records);

/// Rank + diagnose every traced kernel (todos 51) and assemble ProfileReport.
/// The rank list is deterministic (rank_hot_kernels contract). Never throws.
ProfileReport build_report(const std::vector<EnrichedKernel> &kernels,
                           const TraceOptions &options);

//===--- renderers --------------------------------------------------------------===//

/// JSON renderer of the full ProfileReport (schema_version 1.0, report_type
/// "kernscope_profile_report"). Every numeric field machine-readable; hints
/// carry their rule ids + evidence maps. no_kernels_traced is always present.
std::string report_to_json(const ProfileReport &report);

/// Text renderer: the owner's Section B SHAPE (docs/research/owner-examples.md:
/// "Kernel: / Time: / Registers/thread: / Shared memory/block: / Likely
/// limiter: / Compiler hint:") for every hot kernel, with LOCALLY-MEASURED
/// values, plus the ranking. "no kernels traced" (exact phrase) when the
/// negative path fired. Never throws.
std::string report_to_text(const ProfileReport &report);

/// Build the libs/report KernelReport model for ONE kernel (metrics +
/// warnings/suggestions from its diagnostics) — the model to_html() renders
/// (todo 7 reuse: the HTML embeds the schema-valid JSON payload).
kernscope::KernelReport build_kernel_report_model(
    const ProfileReport &report, std::size_t kernel_index);

/// Resolve a CLI target argument: a file is used as-is; a directory is
/// scanned for its executable (prefers the committed fixture name
/// "layernorm_app", then any executable file). Returns the resolved path or
/// std::nullopt (error message in *error).
std::optional<std::string> resolve_target(const std::string &target,
                                          std::string *error);

/// Detect the vendor from the binary (auto mode): hip/cuda from ldd output;
/// the local default is "hip" (this machine is AMD; the tracer is harmless
/// for non-GPU binaries — the no-kernels negative path).
std::string detect_vendor(const std::string &binary_path);

} // namespace kernscope::profile::cli

#endif // KERNSCOPE_PROFILE_CLI_PIPELINE_H
