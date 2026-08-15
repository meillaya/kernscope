//===- enrich.h - per-kernel record enrichment (todo 50) ------------------*- C++ -*-===//
//
// kernscope tools/profile (C6 profiler track).
//
// The ENRICHMENT layer on top of the todo-49 KernelRecord model: every
// kernel record gets the theoretical occupancy + binding limiter (COMPUTED
// BY libs/occupancy — todo 11 — never a cuda_occupancy.h port, Metis M-2),
// the vendor resource facts and the source<->asm correlation:
//
//   * OCCUPANCY (both vendors): enrich_kernel() feeds libs/occupancy
//     compute_occupancy() with registersPerThread / static+dynamic shared
//     bytes / block dims from the record. DEGRADATION (QA contract): a
//     record with registersPerThread == 0 (or a non-positive block size, or
//     an unknown arch) has NO resource evidence -> occupancy is marked
//     "unknown-resources" (never a fabricated 0% figure).
//   * NVIDIA RESOURCES (fallback for cubin-loaded kernels):
//     apply_nvidia_res_usage() parses `cuobjdump -res-usage` output
//     (REG/STACK/SHARED/LOCAL per function) — the static per-kernel usage
//     straight from the ELF (profiler-research-brief §2.3 L265-267).
//   * NVIDIA SOURCE<->PTX: parse_ptx_lineinfo() maps `.file`/`.loc`
//     directives from PTX generated with `nvcc -lineinfo` to
//     (source file, line, ptx instruction) triples (brief §3.1 L246-250).
//   * AMD SOURCE<->ISA: parse_llvm_objdump_lines() maps the `; file:line`
//     comments `llvm-objdump -l` interleaves into AMDGCN disassembly
//     (brief §3.2 L287-288).
//   * PTX AVAILABILITY (Metis degradation finding): cubin-only modules
//     carry NO embedded PTX -> module_has_embedded_ptx() reports false and
//     ptx_available stays false with ptx_diagnostics EMPTY — PTX-level
//     diagnostics are skipped gracefully, never fabricated.
//
// JSONL: enriched_kernel_to_json() emits the todo-49 record shape PLUS the
// enrichment fields (occupancy|"unknown-resources", limiter, block/warp
// residency, vgpr/sgpr/accum_vgpr/lds/scratch/kernel_address for AMD,
// res_usage + source_mapping + ptx_available + ptx_diagnostics). The todo-49
// kernel_record_to_json() shape is preserved field-for-field (the todo-49
// checks keep passing unchanged).
//
// MUST NOT: vendor cuda_occupancy.h; require Nsight/PerfWorks; fabricate
// resource figures (every field is either computed by libs/occupancy,
// parsed from a tool's real output, or marked unknown).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_PROFILE_ENRICH_H
#define KERNSCOPE_PROFILE_ENRICH_H

#include <cstdint>
#include <string>
#include <vector>

#include "kernscope/profile/kernel_record.h"

namespace kernscope::profile {

//===--- enrichment data model -------------------------------------------------===//

/// One (source file, source line) <-> asm correlation entry. asm_text is the
/// PTX instruction (NVIDIA -lineinfo) or the AMDGCN instruction
/// (llvm-objdump -l) that the line maps to; empty when the .loc/comment has
/// no following instruction line.
struct SourceMappingEntry {
  std::string source_file;
  int source_line = 0;
  std::string asm_text;
};

/// Per-function static resource usage from `cuobjdump -res-usage`
/// (REG/STACK/SHARED/LOCAL per function; -1 = absent from the dump).
struct FunctionResourceUsage {
  std::string function;
  int reg = -1;
  int stack = -1;
  int shared = -1;
  int local = -1;
};

/// The enriched per-kernel view. Everything in KernelRecord is preserved
/// verbatim; the enrichment fields below are ADDED by the todo-50 layer.
struct EnrichedKernel {
  KernelRecord record;

  // --- theoretical occupancy + limiter (libs/occupancy, todo 11) -----------
  double occupancy = -1.0;       // fraction 0..1; -1 = not computed
  bool occupancy_unknown = true; // true -> NO resource evidence: reported as
                                 // "unknown-resources", never a fabricated %
  std::string limiter;           // registers|shared|threads|blocks ("" when
                                 // occupancy_unknown)
  int blocks_per_sm = 0;         // resident blocks per SM/CU (0 when unknown)
  int warps_resident = 0;
  int max_warps_per_sm = 0;

  // --- AMD code-object service facts (rocprofiler-sdk kernel-symbol
  //     register: sgpr_count/arch_vgpr_count/accum_vgpr_count/
  //     group_segment_size/private_segment_size/kernel_address). -1/0 =
  //     not observed. registers_per_thread mirrors arch_vgpr_count when the
  //     record's own reg field is unset.
  int sgpr_count = -1;
  int arch_vgpr_count = -1;
  int accum_vgpr_count = -1;
  int group_segment_bytes = -1;   // static LDS from the code object
  int private_segment_bytes = -1; // scratch per work-item from the code object
  std::uint64_t kernel_address = 0;

  // --- NVIDIA res-usage fallback (cuobjdump -res-usage on the cubin) -------
  std::vector<FunctionResourceUsage> res_usage;

  // --- source <-> asm correlation ------------------------------------------
  std::vector<SourceMappingEntry> source_mapping; // PTX .loc / AMDGCN lines
  bool ptx_available = false; // true iff embedded/extracted PTX was present;
                              // false on cubin-only modules (Metis finding)
  std::vector<std::string> ptx_diagnostics; // NEVER filled when
                                            // !ptx_available (never fabricate)
};

/// Options feeding the enrichment (pure host-side; no GPU required).
struct EnrichOptions {
  std::string arch_name;      // "sm_86", "gfx1101", ... (libs/arch key)
  int sgprs_per_thread = 0;   // AMD SGPR usage for the 5-arg occupancy
                              // overload; <= 0 = not modeled (CDNA-only limiter)
};

//===--- occupancy enrichment (both vendors) ----------------------------------===//
// Theoretical occupancy + limiter via libs/occupancy (todo 11 — the ONLY
// occupancy engine; no cuda_occupancy.h port). regs == 0 / block_size <= 0 /
// unknown arch -> occupancy_unknown == true ("unknown-resources").
EnrichedKernel enrich_kernel(const KernelRecord &record,
                             const EnrichOptions &options) noexcept;

//===--- NVIDIA: cuobjdump -res-usage parser ----------------------------------===//
// Parses the per-function REG/STACK/SHARED/LOCAL block:
//   Function _Z11tiny_kernelPKfPfi:
//     REG:10 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:372 ...
// Fills enriched.res_usage (all entries). Never throws.
void apply_nvidia_res_usage(const std::string &cuobjdump_text,
                            EnrichedKernel &enriched) noexcept;

//===--- NVIDIA: source<->PTX from -lineinfo PTX ------------------------------===//
// Parses `.file <N> "<path>"` + `\t.loc <N> <line> [<col>]` directives into
// source_mapping entries (the source<->PTX hop, brief §3.1 L246-250).
// Sets enriched.ptx_available = true. Never throws.
void apply_ptx_lineinfo(const std::string &ptx_text,
                        EnrichedKernel &enriched) noexcept;

//===--- AMD: source<->ISA from llvm-objdump -l -------------------------------===//
// Parses the `; /path/file.cu:<line>` comments llvm-objdump -l interleaves
// into AMDGCN disassembly (brief §3.2 L287-288): each comment starts a
// mapping entry whose asm_text is the next instruction line. Never throws.
void apply_llvm_objdump_lines(const std::string &objdump_text,
                              EnrichedKernel &enriched) noexcept;

//===--- PTX availability (Metis degradation finding) -------------------------===//
// True iff `cuobjdump -ptx <cubin_path>` yields non-empty PTX text. A
// cubin-only module (no embedded PTX) yields EMPTY output -> false, and the
// caller must leave ptx_available false + ptx_diagnostics empty (skip
// PTX-level diagnostics gracefully, never fabricate). Returns false also when
// cuobjdump is not on PATH. Never throws.
bool module_has_embedded_ptx(const std::string &cubin_path) noexcept;

//===--- enriched JSONL --------------------------------------------------------===//
// The todo-49 record JSON (field-for-field) PLUS the enrichment fields.
// occupancy is a JSON number (fraction) or the string "unknown-resources".
std::string enriched_kernel_to_json(const char *vendor,
                                    const EnrichedKernel &enriched);

} // namespace kernscope::profile

#endif // KERNSCOPE_PROFILE_ENRICH_H
