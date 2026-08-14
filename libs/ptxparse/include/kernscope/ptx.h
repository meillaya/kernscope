//===- ptx.h - PTX 8.x parser: typed IR with CFG -------------*- C++ -*-===//
//
// kernscope PTX parser (wave-2 todo 9). A hand-written lexer+parser that
// turns PTX 8.x source text into a typed IR:
//
//   * directives   - .version / .target / .address_size / .entry / .func
//   * variables    - declarations with state space (.global/.shared/.local/
//                    .const/.param/.reg) + alignment + type + shape
//   * instructions - opcode, operands, optional predicate (@p), instruction
//                    class (taxonomy), state-space annotation on every memory
//                    op, ld/st width from the type suffix (8/16/32/64/128)
//   * CFG          - per-kernel basic blocks with explicit branch successors
//                    and fallthrough edges; deterministic edge count
//   * tensor ops   - mma.sync / wmma / wgmma / tcgen05 families with a
//                    class tag; tcgen05 ops are marked parse_only (this
//                    toolchain's nvcc is 12.6, which cannot target sm_100,
//                    so any tcgen05 input is by definition hand-written
//                    parse-only corpus - see the FIXTURE GENERATION RULE).
//
// The parser is STRICTLY parse-only: it never assembles, never executes, and
// never depends on NVBit/ptxas. A malformed input throws PtxParseError with
// line:col - there are no silent partial parses.
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_LIBS_PTXPARSE_INCLUDE_KERNSCOPE_PTX_H
#define KERNSCOPE_LIBS_PTXPARSE_INCLUDE_KERNSCOPE_PTX_H

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope {

/// PTX state spaces (PTX ISA §5.1). Also used to annotate every memory
/// operand of ld/st/atom/red with the space the mnemonic selects.
enum class StateSpace : std::uint8_t {
  kGlobal,
  kShared,
  kLocal,
  kConst,
  kParam,
  kReg,       // .reg register declarations
  kGeneric,   // generic (no state-space suffix on the mnemonic)
  kUndefined,
};

/// Stable single-word name for JSON output and manifests.
const char *state_space_name(StateSpace s) noexcept;

/// Instruction taxonomy. The classes are the instruction-mix buckets the
/// static analyzer consumes (cvt/mov/fma are explicit buckets per the plan;
/// tensor families are split out for the tensor-core analyses).
enum class InstructionClass : std::uint8_t {
  kLoad,          // ld / ldu
  kStore,         // st / std
  kAtomic,        // atom
  kReduction,     // red
  kBarrier,       // bar.sync / bar.warp / barrier.cluster
  kBranch,        // bra / brx
  kCall,
  kReturn,        // ret
  kExit,
  kSetp,          // setp
  kMov,           // mov
  kCvt,           // cvt
  kFma,           // fma
  kCvta,          // cvta.to.<space>
  kAlu,           // add/sub/mul/mad/div/rem/abs/neg/min/max
  kLogic,         // and/or/xor/not/cnot + shifts shl/shr
  kMath,          // ex2/lg2/sin/cos/tanh/rsqrt/sqrt/rcp/rro
  kSelect,        // selp/slct
  kLdMatrix,      // ldmatrix
  kStMatrix,      // stmatrix
  kTexture,       // tex/tld4/suld/sust/txq
  kAsyncCopy,     // cp.async
  kTensorMmaSync, // mma.sync / mma.sp
  kTensorWmma,    // wmma.*
  kTensorWgmma,   // wgmma.* (mma_async / commit_group / wait_group)
  kTensorTcgen05, // tcgen05.* (5th-gen tensor core, sm_100a)
  kOther,
};

/// Stable single-word class name (used as the JSON class_counts key).
const char *instruction_class_name(InstructionClass c) noexcept;

/// One parsed operand. `text` is the rendered operand exactly as written
/// (e.g. "%rd1+8", "{%f0, %f1, %f2, %f3}", "-32"). Memory operands carry the
/// state-space annotation from the instruction's mnemonic.
struct PtxOperand {
  std::string text;
  bool is_address = false;  // [ ... ] form
  StateSpace space = StateSpace::kUndefined;
};

/// One parsed instruction.
struct PtxInstruction {
  std::string mnemonic;            // full dotted mnemonic, e.g. "ld.shared.v4.f32"
  std::vector<std::string> segs;   // mnemonic split on '.'
  bool has_predicate = false;
  bool predicate_negated = false;
  std::string predicate;           // e.g. "%p1" (raw token text)
  std::vector<PtxOperand> operands;

  InstructionClass cls = InstructionClass::kOther;
  StateSpace mem_space = StateSpace::kUndefined;  // ld/st/atom/red annotation
  int mem_width_bits = 0;          // 8/16/32/64/128 from the type suffix

  bool is_branch = false;          // bra/brx
  std::string branch_target;       // label (or register for brx)

  int line = 0;                    // 1-based source position (diagnostics)
  int col = 0;
};

/// One basic block of the per-kernel CFG. `label` is "" for the implicit
/// entry block. `successors` are the labels of the blocks this block can
/// branch to (deduplicated), including fallthrough to the next layout block.
struct PtxBlock {
  std::string label;
  std::vector<PtxInstruction> instructions;
  std::vector<std::string> successors;
};

/// A tensor-core instruction occurrence (tensor-core usage analysis input).
struct PtxTensorOp {
  std::string mnemonic;
  std::string class_tag;   // "mma.sync" | "wmma" | "wgmma.mma_async" |
                           // "tcgen05.mma" | "wgmma.commit_group" | ...
  bool parse_only = false; // never assembled (see class doc comment)
};

/// A variable/register declaration (state space + alignment + type + shape).
struct PtxVariable {
  std::string name;
  StateSpace space = StateSpace::kUndefined;
  int alignment = 0;               // bytes from .align N; 0 = unspecified
  std::string type;                // ".f32" ".b8" ".u64" ".pred" ...
  std::string shape;               // "" | "[]" | "[N]"
};

/// A kernel (.entry) or function (.func) with its CFG and derived statistics.
///
/// All counts are computed by the parser (deterministic; the gtest corpus
/// asserts them exactly per manifest).
struct PtxKernel {
  std::string name;
  bool entry_point = false;
  std::vector<PtxBlock> cfg;       // layout order; index 0 = entry block

  // Derived statistics.
  std::int64_t instruction_count = 0;   // instructions incl. predicated ones
  std::int64_t bar_sync_count = 0;      // bar.sync (exact mnemonic)
  std::int64_t barrier_count = 0;       // all bar.* / barrier.*
  std::int64_t cfg_edge_count = 0;
  std::int64_t shared_access_count = 0; // ld/st/atom/red .shared
  std::int64_t global_access_count = 0; // ld/st/atom/red .global
  std::int64_t local_access_count = 0;  // ld/st/atom/red .local (spills)
  std::int64_t const_access_count = 0;  // ld/st/atom/red .const
  std::int64_t param_access_count = 0;  // ld.param / st.param
  std::map<int, std::int64_t> mem_width_counts;   // 8/16/32/64/128 (ld/st)
  std::map<InstructionClass, std::int64_t> class_counts;
  std::vector<PtxTensorOp> tensor_ops;
  std::vector<PtxVariable> variables;   // in-body declarations
};

/// The parsed module: directives + variables + kernels/functions.
struct PtxModule {
  std::string ptx_version;  // e.g. "8.5" ("" if no .version)
  std::string target;       // e.g. "sm_86" ("" if no .target)
  int address_size = 64;
  std::vector<PtxVariable> variables;   // top-level declarations
  std::vector<PtxKernel> kernels_;      // .entry and .func bodies

  const std::vector<PtxKernel> &kernels() const { return kernels_; }
};

/// Thrown on any malformed input. `line`/`col` are 1-based; `what()` names
/// both, e.g. "ptx parse error at line 12 col 5: expected ';'". The parser
/// never returns a partial module - it throws or it completes.
class PtxParseError : public std::runtime_error {
public:
  PtxParseError(int line, int col, const std::string &message)
      : std::runtime_error("ptx parse error at line " + std::to_string(line) +
                           " col " + std::to_string(col) + ": " + message),
        line_(line), col_(col) {}

  int line() const noexcept { return line_; }
  int col() const noexcept { return col_; }

private:
  int line_;
  int col_;
};

/// Parse PTX 8.x source text into a typed module. Throws PtxParseError
/// (with line:col) on malformed input; there are no silent partial parses.
PtxModule parse_ptx_module(std::string_view text);

}  // namespace kernscope

#endif  // KERNSCOPE_LIBS_PTXPARSE_INCLUDE_KERNSCOPE_PTX_H
