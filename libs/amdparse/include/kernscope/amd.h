//===-- kernscope/amd.h - AMDGPU ISA text parser (LLVM MC) ---------------===//
//
// kernscope::amd -- parse AMDGPU (GCN/CDNA/RDNA) assembly *text* into typed
// instruction records with a control-flow graph, using LLVM 21's MC layer
// (MCAsmParser + the registered AMDGPU target).
//
// Scope & design notes (documented fallback decision, todo-10):
//  * We drive LLVM's own AMDGPU assembly parser (MCAsmParser + the
//    AMDGPUAsmParser target extension). This is exactly what `llvm-mc`
//    uses to assemble .s files, so anything hipcc/clang `-S` emits for
//    gfx9/gfx11 parses by construction; hand-written snippets that are
//    real RDNA3/CDNA ISA also parse. If a hand-written snippet proves
//    brittle (an MC quirk in an exotic operand spelling), the accepted
//    scope is "parse hipcc/clang -S output only" -- that is the exact
//    contract the analyzer (tools/analyze, todo 14) consumes. We never
//    parse SASS (this is ISA TEXT), and we never link ROCm device
//    libraries at parse time.
//  * Target classification (VALU/SALU/LDS/global/buffer/flat/scratch/
//    branch/barrier/wait/MFMA/WMMA) is derived from the mnemonic prefix
//    per docs/research/gpu-static-analyzer-research.md §5 (MFMA CDNA2/3,
//    WMMA RDNA3) and §6 (scratch = spill class).
//  * Labels -> CFG: basic blocks are delimited by labels and by
//    terminators (s_branch / s_cbranch_* / s_endpgm); successors are
//    resolved from branch-target symbol operands plus implicit
//    fall-through edges.
//
// This header is the public surface. The implementation lives in
// libs/amdparse/src/amd_parser.cpp and links `LLVM` (the LLVM 21 shared
// library, which carries all registered targets incl. AMDGPU).
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_AMD_H
#define KERNSCOPE_AMD_H

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope {

//------------------------------------------------------------------------------
// Instruction taxonomy (mirrors docs/research/gpu-static-analyzer-research.md
// §5 / §6 and the analyzer's ten analyses; see amd_parser.cpp classify()).
//------------------------------------------------------------------------------
enum class InstrClass : std::uint8_t {
  Unknown = 0, ///< unclassified / not AMDGPU assembly
  VALU,        ///< v_* vector ALU
  SALU,        ///< s_* scalar ALU (s_load/s_mov/s_or/...)
  LDS,         ///< ds_* local-data-share ops (ds_read_*/ds_write_*)
  Global,      ///< global_load_*/global_store_*
  Buffer,      ///< buffer_load_*/buffer_store_*
  Flat,        ///< flat_load_*/flat_store_*
  Scratch,     ///< scratch_load_*/scratch_store_* (SPILL class, brief §6)
  Branch,      ///< s_branch / s_cbranch_* / s_call_b64 / s_swappc_b64
  Barrier,     ///< s_barrier
  Wait,        ///< s_waitcnt / s_waitcnt_depctr
  EndPgm,      ///< s_endpgm (kernel return)
  MFMA,        ///< v_mfma_* (CDNA2/3 matrix cores)
  WMMA,        ///< v_wmma_* (RDNA3 matrix cores)
};

/// A single typed operand of an AMDGPU instruction.
struct AmdOperand {
  enum class Kind : std::uint8_t {
    Register, ///< vN / sN / aN / vcc / exec / null / v[N:M]...
    Immediate, ///< integer / fp literal
    SymbolRef, ///< MCSymbolRefExpr (e.g. a branch target label)
    Expression, ///< other MCExpr (addend/reloc/...) -- kept as printed text
  };

  Kind kind = Kind::Expression;
  std::string text; ///< canonical printed form (e.g. "v[3:4]", ".LBB0_3")
};

/// A single parsed AMDGPU instruction.
struct AmdInstruction {
  std::string mnemonic;             ///< e.g. "global_load_u16"
  InstrClass klass = InstrClass::Unknown;
  unsigned line = 0;                ///< 1-based source line (0 = unknown)
  std::vector<AmdOperand> operands; ///< MC operands in source order
  std::string text;                 ///< full canonical MC printout

  // ---- control-flow metadata -------------------------------------------
  bool is_branch = false;           ///< s_branch / s_cbranch_* / call
  bool is_unconditional = false;    ///< s_branch (no fall-through)
  bool is_conditional = false;      ///< s_cbranch_* (fall-through edge)
  bool is_call = false;             ///< s_call_b64 / s_swappc_b64
  bool is_return = false;           ///< s_endpgm
  std::optional<std::string> branch_target; ///< target label, if any

  // ---- memory / tensor metadata ----------------------------------------
  bool is_load = false;             ///< *_load_* / ds_read_*
  bool is_store = false;            ///< *_store_* / ds_write_*
  bool is_scratch = false;          ///< SPILL-class (scratch_*)
  bool is_tensor = false;           ///< MFMA or WMMA
};

/// A basic block in the parsed module's CFG.
struct AmdBlock {
  std::string name;                      ///< entry label ("" for the entry block)
  std::vector<std::size_t> instruction_index; ///< indices into AmdModule::instructions
  std::vector<std::string> successors;   ///< successor block names
  std::vector<std::string> predecessors; ///< predecessor block names
  bool is_entry = false;                 ///< first block of the function
  bool has_fallthrough = false;          ///< falls through to next block
};

/// The result of parsing one AMDGPU assembly module.
struct AmdModule {
  std::string target_triple; ///< e.g. "amdgcn-amd-amdhsa--gfx1101"
  std::string target_cpu;    ///< e.g. "gfx1101"
  std::string source_name;   ///< e.g. "layernorm_gfx1101.s" or "<memory>"

  std::vector<AmdInstruction> instructions;
  std::vector<AmdBlock> blocks;

  /// Number of instructions classified as \p c.
  std::size_t count_class(InstrClass c) const;
  /// Distinct mnemonics of instructions in class \p c (sorted).
  std::vector<std::string> mnemonics(InstrClass c) const;
  /// Any MFMA/WMMA tensor instruction present?
  bool has_tensor_instruction() const;
  /// Number of MFMA/WMMA tensor instructions.
  std::size_t tensor_instruction_count() const;
};

/// Thrown when the AMDGPU assembly text cannot be parsed for the given
/// target. Carries the 1-based source line of the failure and the target
/// triple, per todo-10 QA ("diagnostic names the line and target triple").
class AmdParseError : public std::runtime_error {
public:
  AmdParseError(const std::string &what, const std::string &triple,
                const std::string &msg, unsigned ln, unsigned col);

  std::string target_triple; ///< triple the parse was attempted for
  std::string detail;        ///< raw LLVM MC diagnostic message
  unsigned line;             ///< 1-based source line (0 if unknown)
  unsigned column;           ///< 1-based column (0 if unknown)
};

/// Parse AMDGPU assembly \p asm_text for the target architecture \p target.
///
/// \p target accepts either a CPU name ("gfx1101", "gfx90a", "gfx942") or a
/// full triple ("amdgcn-amd-amdhsa--gfx1101"). The result records every
/// instruction with its taxonomy class and operands, plus a label-based CFG.
///
/// \throws AmdParseError if the MC parse fails (unknown mnemonic for the
///         target, malformed operands, ...), naming line + target triple.
AmdModule parse_amd_module(std::string_view asm_text, std::string_view target,
                           std::string_view source_name = "<memory>");

/// Parse an AMDGPU assembly file (thin wrapper over parse_amd_module).
AmdModule parse_amd_file(const std::string &path, std::string_view target);

/// Human-readable name for an instruction class (for diagnostics/tests).
const char *instr_class_name(InstrClass c);

} // namespace kernscope

#endif // KERNSCOPE_AMD_H
