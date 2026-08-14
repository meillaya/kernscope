//===- ptx_parser.cpp - PTX 8.x parser: typed IR + CFG -----------*- C++ -*-===//
//
// Wave-2 todo 9. Recursive-descent parser over the token stream from
// ptx_lexer.cpp. Produces the PtxModule IR:
//
//   * directives     .version/.target/.address_size/.file/.loc/.pragma/...
//   * declarations   .global/.shared/.local/.const/.param/.reg with
//                    alignment, type and shape
//   * functions      .entry and .func bodies (param lists, body statements)
//   * instructions   opcode + operands + optional @p predicate, classified
//                    into the instruction-mix taxonomy; ld/st/atom/red get
//                    their state-space annotation and ld/st get their width
//                    from the type suffix (8/16/32/64/128)
//   * CFG            basic blocks split at labels, successor edges from
//                    bra/brx (incl. predicated @p bra with fallthrough),
//                    ret/exit, and fallthrough between layout blocks;
//                    cfg_edge_count is deterministic
//
// The parser throws PtxParseError (line:col) on malformed input and never
// returns a partial module. Unknown top-level directives are skipped
// (forward compatibility with newer PTX); instructions and declarations are
// validated structurally.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <vector>

#include "kernscope/ptx.h"
#include "token.h"

namespace kernscope {

using ptx::Token;
using ptx::TokKind;

namespace {

std::vector<std::string> split(const std::string &s, char sep) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : s) {
    if (c == sep) {
      out.push_back(cur);
      cur.clear();
    } else {
      cur += c;
    }
  }
  out.push_back(cur);
  return out;
}

StateSpace space_from_segment(const std::string &seg) {
  if (seg == "global") return StateSpace::kGlobal;
  if (seg == "shared") return StateSpace::kShared;
  if (seg == "local") return StateSpace::kLocal;
  if (seg == "const") return StateSpace::kConst;
  if (seg == "param") return StateSpace::kParam;
  if (seg == "generic") return StateSpace::kGeneric;
  return StateSpace::kUndefined;
}

// Scan the mnemonic segments for the first explicit state-space keyword
// (e.g. "ld.global.nc.f32" -> global, "cp.async.cg.shared.global" -> shared
// for its first address operand). Returns kUndefined when absent.
StateSpace first_space_in(const std::vector<std::string> &segs, std::size_t from) {
  for (std::size_t i = from; i < segs.size(); ++i) {
    // ".shared::cta" and ".shared::cluster" lex as one segment containing
    // "::"; strip everything after "::".
    std::string base = segs[i];
    const std::size_t ns = base.find("::");
    if (ns != std::string::npos) base = base.substr(0, ns);
    StateSpace s = space_from_segment(base);
    if (s != StateSpace::kUndefined) return s;
  }
  return StateSpace::kUndefined;
}

// Extract the access width from the last type suffix of the mnemonic
// (".b8/.b16/.b32/.b64/.b128" and ".uN/.sN/.fN"): 8/16/32/64/128, 0 if none.
int type_width_bits(const std::vector<std::string> &segs) {
  for (std::size_t i = segs.size(); i-- > 1;) {
    const std::string &seg = segs[i];
    if (seg.size() < 2 || (seg[0] != 'b' && seg[0] != 'u' && seg[0] != 's' &&
                           seg[0] != 'f'))
      continue;
    const bool all_digits =
        std::all_of(seg.begin() + 1, seg.end(), [](char c) {
          return std::isdigit(static_cast<unsigned char>(c));
        });
    if (!all_digits) continue;
    const int bits = std::atoi(seg.c_str() + 1);
    if (bits == 8 || bits == 16 || bits == 32 || bits == 64 || bits == 128)
      return bits;
  }
  return 0;
}

struct Classified {
  InstructionClass cls = InstructionClass::kOther;
  StateSpace mem_space = StateSpace::kUndefined;
  int width_bits = 0;
  bool is_branch = false;
  bool is_tensor = false;
  std::string tensor_tag;
  bool tensor_parse_only = false;
};

// The tcgen05 family requires sm_100a+, which CUDA 12.6's nvcc (the pinned
// toolchain in devenv.nix) cannot target - any tcgen05 input is therefore
// hand-written parse-only corpus (FIXTURE GENERATION RULE). Marked at parse
// time so the dump tool and tests can assert the annotation without the
// manifest being the only carrier.
Classified classify(const std::vector<std::string> &segs) {
  Classified c;
  const std::string &op = segs[0];
  c.width_bits = type_width_bits(segs);

  if (op == "ld" || op == "ldu" || op == "ldx") {
    c.cls = InstructionClass::kLoad;
    c.mem_space = first_space_in(segs, 1);
    if (c.mem_space == StateSpace::kUndefined) c.mem_space = StateSpace::kGeneric;
  } else if (op == "st" || op == "std" || op == "stx") {
    c.cls = InstructionClass::kStore;
    c.mem_space = first_space_in(segs, 1);
    if (c.mem_space == StateSpace::kUndefined) c.mem_space = StateSpace::kGeneric;
  } else if (op == "atom") {
    c.cls = InstructionClass::kAtomic;
    c.mem_space = first_space_in(segs, 1);
  } else if (op == "red") {
    c.cls = InstructionClass::kReduction;
    c.mem_space = first_space_in(segs, 1);
  } else if (op == "bar" || op == "barrier") {
    c.cls = InstructionClass::kBarrier;
  } else if (op == "bra" || op == "brx") {
    c.cls = InstructionClass::kBranch;
    c.is_branch = true;
  } else if (op == "call") {
    c.cls = InstructionClass::kCall;
  } else if (op == "ret") {
    c.cls = InstructionClass::kReturn;
  } else if (op == "exit") {
    c.cls = InstructionClass::kExit;
  } else if (op == "setp") {
    c.cls = InstructionClass::kSetp;
  } else if (op == "mov") {
    c.cls = InstructionClass::kMov;
  } else if (op == "cvt") {
    c.cls = InstructionClass::kCvt;
  } else if (op == "cvta") {
    c.cls = InstructionClass::kCvta;
    c.mem_space = first_space_in(segs, 1);
  } else if (op == "fma") {
    c.cls = InstructionClass::kFma;
  } else if (op == "mma") {
    c.cls = InstructionClass::kTensorMmaSync;
    c.is_tensor = true;
    c.tensor_tag = (segs.size() > 1 && segs[1] == "sp") ? "mma.sp" : "mma.sync";
  } else if (op == "wmma") {
    c.cls = InstructionClass::kTensorWmma;
    c.is_tensor = true;
    c.tensor_tag = "wmma";
  } else if (op == "wgmma") {
    c.cls = InstructionClass::kTensorWgmma;
    c.is_tensor = true;
    c.tensor_tag = segs.size() > 1 ? ("wgmma." + segs[1]) : "wgmma";
  } else if (op == "tcgen05") {
    c.cls = InstructionClass::kTensorTcgen05;
    c.is_tensor = true;
    c.tensor_tag =
        (segs.size() > 1 && segs[1] == "mma") ? "tcgen05.mma" : "tcgen05";
    c.tensor_parse_only = true;
  } else if (op == "ldmatrix") {
    c.cls = InstructionClass::kLdMatrix;
  } else if (op == "stmatrix") {
    c.cls = InstructionClass::kStMatrix;
  } else if (op == "cp") {
    c.cls = InstructionClass::kAsyncCopy;
    c.mem_space = first_space_in(segs, 1);
  } else if (op == "tex" || op == "tld4" || op == "suld" || op == "sust" ||
             op == "txq" || op == "tld") {
    c.cls = InstructionClass::kTexture;
  } else if (op == "add" || op == "sub" || op == "mul" || op == "mad" ||
             op == "div" || op == "rem" || op == "abs" || op == "neg" ||
             op == "min" || op == "max") {
    c.cls = InstructionClass::kAlu;
  } else if (op == "and" || op == "or" || op == "xor" || op == "not" ||
             op == "cnot" || op == "shl" || op == "shr") {
    c.cls = InstructionClass::kLogic;
  } else if (op == "ex2" || op == "lg2" || op == "sin" || op == "cos" ||
             op == "tanh" || op == "tan" || op == "rsqrt" || op == "sqrt" ||
             op == "rcp" || op == "rro") {
    c.cls = InstructionClass::kMath;
  } else if (op == "selp" || op == "slct") {
    c.cls = InstructionClass::kSelect;
  } else {
    c.cls = InstructionClass::kOther;
  }
  return c;
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

class Parser {
public:
  explicit Parser(std::vector<Token> toks) : toks_(std::move(toks)) {}

  PtxModule parse_module() {
    PtxModule mod;
    while (!at_eof()) {
      const Token &t = peek();
      if (t.kind == TokKind::kDirective) {
        const std::string &d = t.text;
        if (d == ".version") {
          parse_version(mod);
        } else if (d == ".target") {
          parse_target(mod);
        } else if (d == ".address_size") {
          parse_address_size(mod);
        } else if (d == ".entry" || d == ".func") {
          parse_function(mod, /*entry=*/d == ".entry");
        } else if (d == ".extern" || d == ".visible" || d == ".weak" ||
                   d == ".static") {
          // A qualifier may lead a function (.visible .entry ...) or a
          // variable declaration (.extern .shared ...). Look ahead past the
          // qualifier chain to tell them apart.
          std::size_t ahead = 0;
          while (is_qualifier(peek(ahead))) ++ahead;
          const std::string &nxt = peek(ahead).text;
          if (nxt == ".entry" || nxt == ".func") {
            parse_function(mod, nxt == ".entry");
          } else {
            PtxVariable var;
            parse_variable_decl(mod.variables, var);
          }
        } else if (is_variable_decl_start(d)) {
          PtxVariable var;
          parse_variable_decl(mod.variables, var);
        } else {
          skip_unknown_directive();
        }
      } else if (t.kind == TokKind::kIdent) {
        fail(t, "expected a PTX directive, got '" + t.text + "'");
      } else {
        fail(t, "expected a PTX directive");
      }
    }
    return mod;
  }

private:
  std::vector<Token> toks_;
  std::size_t pos_ = 0;

  bool at_eof() const { return toks_[pos_].kind == TokKind::kEof; }
  const Token &peek(std::size_t ahead = 0) const {
    return toks_[std::min(pos_ + ahead, toks_.size() - 1)];
  }
  const Token &next() {
    const Token &t = toks_[pos_];
    if (pos_ + 1 < toks_.size()) ++pos_;
    return t;
  }
  [[noreturn]] void fail(const Token &t, const std::string &msg) {
    throw PtxParseError(t.line, t.col, msg);
  }

  bool is_punct(const Token &t, char c) const {
    return t.kind == TokKind::kPunct && t.text.size() == 1 && t.text[0] == c;
  }
  bool is_qualifier(const Token &t) const {
    return t.kind == TokKind::kDirective &&
           (t.text == ".extern" || t.text == ".visible" || t.text == ".weak" ||
            t.text == ".static");
  }
  void expect_punct(char c) {
    if (is_punct(peek(), c)) {
      next();
      return;
    }
    fail(peek(), std::string("expected '") + c + "', got '" + peek().text + "'");
  }
  void expect_semicolon() {
    if (is_punct(peek(), ';')) {
      next();
      return;
    }
    fail(peek(), std::string("expected ';', got '") + peek().text + "'");
  }

  // --- directives ----------------------------------------------------------

  void parse_version(PtxModule &mod) {
    next();  // .version
    const Token &t = peek();
    if (t.kind == TokKind::kNumber || t.kind == TokKind::kIdent) {
      mod.ptx_version = next().text;
    } else {
      fail(t, "expected PTX version number after .version");
    }
  }

  void parse_target(PtxModule &mod) {
    next();  // .target
    const Token &t = peek();
    if (t.kind == TokKind::kIdent) {
      mod.target = next().text;
    } else if (t.kind == TokKind::kDirective) {
      // e.g. ".target .sm_86" form tolerated.
      mod.target = next().text;
    } else {
      fail(t, "expected target architecture after .target");
    }
  }

  void parse_address_size(PtxModule &mod) {
    next();  // .address_size
    const Token &t = peek();
    if (t.kind != TokKind::kNumber) fail(t, "expected address size number");
    mod.address_size = std::atoi(next().text.c_str());
  }

  // Skip a line-based directive (.loc, .file, .pragma, .section, .reqntid,
  // .maxnreg, ...). Consumes everything on the directive's line; a ';' on the
  // same line (e.g. ".pragma \"nounroll\";") is consumed as well.
  void skip_unknown_directive() {
    const Token &start = next();
    const int line = start.line;
    while (!at_eof() && peek().line == line) next();
  }

  // --- declarations --------------------------------------------------------

  bool is_state_space_directive(const std::string &d) const {
    return d == ".global" || d == ".shared" || d == ".local" || d == ".const" ||
           d == ".param" || d == ".reg";
  }

  // A directive that starts a variable declaration at statement level.
  bool is_variable_decl_start(const std::string &d) const {
    if (is_state_space_directive(d)) return true;
    return d == ".extern" || d == ".visible" || d == ".weak" || d == ".static";
  }

  // Parse ".extern? .visible? .weak? .static? <space> [.align N] .type
  // name[shape] [= ...] ;". Fills `var`; also returns the state space so the
  // caller can route .reg declarations into the register list.
  StateSpace parse_variable_decl(std::vector<PtxVariable> &out, PtxVariable &var) {
    // Leading qualifiers (some optional).
    StateSpace space = StateSpace::kUndefined;
    for (;;) {
      const std::string &d = peek().text;
      if (d == ".extern" || d == ".visible" || d == ".weak" || d == ".static") {
        next();
      } else if (d == ".reg" || d == ".shared" || d == ".local" ||
                 d == ".global" || d == ".const" || d == ".param") {
        space = space_from_segment(d.substr(1));
        next();
        break;
      } else {
        fail(peek(), "expected state-space directive in declaration");
      }
    }
    if (space == StateSpace::kUndefined)
      fail(peek(), "unknown state space in declaration");

    if (is_punct(peek(), ';')) {  // bare space, e.g. ".shared;"
      next();
      return space;
    }

    // .align N
    if (peek().text == ".align") {
      next();
      const Token &t = peek();
      if (t.kind != TokKind::kNumber) fail(t, "expected alignment value");
      var.alignment = std::atoi(next().text.c_str());
    }

    // type token, e.g. .b8/.f32/.u64/.pred
    if (peek().kind == TokKind::kDirective) {
      var.type = next().text;
    } else {
      fail(peek(), "expected a type token (e.g. .b32) in declaration");
    }

    // name [shape]
    if (peek().kind != TokKind::kIdent && peek().kind != TokKind::kRegister) {
      fail(peek(), "expected a variable name in declaration");
    }
    var.name = next().text;

    if (is_punct(peek(), '[')) {
      next();
      std::string shape = "[";
      while (!is_punct(peek(), ']')) {
        if (at_eof() || is_punct(peek(), ';')) {
          fail(peek(), "unterminated array shape in declaration");
        }
        shape += next().text;
      }
      next();  // ']'
      shape += "]";
      var.shape = shape;
    }

    // optional initializer: "= value;" or "= {...};" - consume to ';'.
    if (is_punct(peek(), '=')) {
      while (!is_punct(peek(), ';')) {
        if (at_eof()) fail(peek(), "unterminated initializer in declaration");
        next();
      }
    }

    expect_semicolon();
    var.space = space;
    out.push_back(var);
    return space;
  }

  // .reg declarations inside a body: type + comma-separated register names.
  void parse_reg_decl(PtxKernel &kern) {
    next();  // .reg
    const Token &t = peek();
    std::string type = ".reg";
    if (t.kind == TokKind::kDirective) type = next().text;
    for (;;) {
      if (peek().kind != TokKind::kRegister &&
          peek().kind != TokKind::kIdent) {
        fail(peek(), "expected a register name in .reg declaration");
      }
      PtxVariable var;
      var.space = StateSpace::kReg;
      var.type = type;
      var.name = next().text;
      kern.variables.push_back(std::move(var));
      if (is_punct(peek(), ',')) {
        next();
        continue;
      }
      break;
    }
    expect_semicolon();
  }

  // --- functions / entry points ---------------------------------------------

  void parse_function(PtxModule &mod, bool entry) {
    while (is_qualifier(peek())) next();  // .visible/.weak/.extern
    next();  // .entry / .func

    if (!entry && is_punct(peek(), '(')) skip_group('(', ')');

    if (peek().kind != TokKind::kIdent) {
      fail(peek(), "expected function name");
    }
    PtxKernel kern;
    kern.name = next().text;
    kern.entry_point = entry;

    // Optional .callprototype-ish qualifier before params (rare).
    if (peek().text == ".callprototype") skip_unknown_directive();

    // Parameter list: "(" .param decls ")". Only present when a body follows;
    // a bare ".func foo;" declaration ends here.
    if (is_punct(peek(), '(')) {
      next();
      while (!is_punct(peek(), ')')) {
        if (at_eof()) fail(peek(), "unterminated parameter list");
        if (peek().text == ".param") {
          PtxVariable var;
          parse_param_decl(var);
          kern.variables.push_back(std::move(var));
        } else if (is_punct(peek(), ',')) {
          next();
        } else {
          fail(peek(), "expected .param declaration in parameter list");
        }
      }
      next();  // ')'
    }

    if (is_punct(peek(), ';')) {  // declaration only
      next();
      mod.kernels_.push_back(std::move(kern));
      return;
    }

    expect_punct('{');
    parse_body(kern);
    expect_punct('}');

    finalize_kernel(kern);
    mod.kernels_.push_back(std::move(kern));
  }

  void parse_param_decl(PtxVariable &var) {
    next();  // .param
    if (peek().text == ".align") {
      next();
      const Token &t = peek();
      if (t.kind != TokKind::kNumber) fail(t, "expected alignment value");
      var.alignment = std::atoi(next().text.c_str());
    }
    if (peek().kind != TokKind::kDirective) {
      fail(peek(), "expected .param type token");
    }
    var.type = next().text;
    if (peek().kind != TokKind::kIdent && peek().kind != TokKind::kRegister) {
      fail(peek(), "expected parameter name");
    }
    var.name = next().text;
    var.space = StateSpace::kParam;
  }

  // Consume a balanced (...) / {...} / [...] group into `text` (used for
  // return-param groups and vector register groups).
  void skip_group(char open, char close) {
    if (!is_punct(peek(), open)) fail(peek(), "expected '" + std::string(1, open) + "'");
    next();
    int depth = 1;
    while (depth > 0) {
      if (at_eof()) fail(peek(), "unterminated '" + std::string(1, open) + "' group");
      const Token &t = next();
      if (is_punct(t, open)) ++depth;
      if (is_punct(t, close)) --depth;
    }
  }

  void parse_body(PtxKernel &kern) {
    std::vector<PtxBlock> blocks;
    PtxBlock cur;
    blocks.push_back(std::move(cur));  // implicit entry block

    while (!is_punct(peek(), '}')) {
      if (at_eof()) fail(peek(), "unexpected end of input inside function body");

      const Token &t = peek();
      if (t.kind == TokKind::kDirective) {
        const std::string &d = t.text;
        if (d == ".reg") {
          parse_reg_decl(kern);
        } else if (is_variable_decl_start(d)) {
          PtxVariable var;
          parse_variable_decl(kern.variables, var);
        } else {
          skip_unknown_directive();
        }
        continue;
      }

      if (t.kind == TokKind::kIdent && is_punct(peek(1), ':')) {
        // Label: starts a new block. If the entry block is still empty and
        // unlabeled, attach the label to it instead of opening a second one.
        if (blocks.size() == 1 && blocks[0].label.empty() &&
            blocks[0].instructions.empty()) {
          blocks[0].label = next().text;
          next();  // ':'
          continue;
        }
        std::string label = next().text;
        next();  // ':'
        blocks.push_back(PtxBlock{});
        blocks.back().label = label;
        continue;
      }

      if (t.kind == TokKind::kIdent || is_punct(t, '@')) {
        PtxInstruction ins = parse_instruction();
        blocks.back().instructions.push_back(std::move(ins));
        continue;
      }

      fail(t, "unexpected token '" + t.text + "' in function body");
    }

    // A body ending in a bare label leaves an empty trailing block; drop it.
    if (blocks.size() > 1 && blocks.back().instructions.empty() &&
        blocks.back().label.empty()) {
      blocks.pop_back();
    }

    build_cfg(blocks);
    kern.cfg = std::move(blocks);
  }

  // --- instructions ----------------------------------------------------------

  PtxInstruction parse_instruction() {
    PtxInstruction ins;

    // Optional predicate: @[!][%]pN
    if (is_punct(peek(), '@')) {
      next();
      ins.has_predicate = true;
      if (is_punct(peek(), '!')) {
        ins.predicate_negated = true;
        next();
      }
      const Token &p = peek();
      if (p.kind != TokKind::kRegister && p.kind != TokKind::kIdent) {
        fail(p, "expected a predicate register after '@'");
      }
      ins.predicate = next().text;
    }

    const Token &m = peek();
    if (m.kind != TokKind::kIdent) {
      fail(m, "expected an instruction mnemonic");
    }
    ins.mnemonic = next().text;
    ins.segs = split(ins.mnemonic, '.');
    ins.line = m.line;
    ins.col = m.col;

    Classified c = classify(ins.segs);
    ins.cls = c.cls;
    ins.mem_space = c.mem_space;
    ins.mem_width_bits = c.width_bits;
    ins.is_branch = c.is_branch;

    // Operands, separated by ',', terminated by ';'.
    while (!is_punct(peek(), ';')) {
      if (at_eof()) {
        fail(m, "unterminated instruction (missing ';') at '" + ins.mnemonic + "'");
      }
      PtxOperand op;
      if (is_punct(peek(), '{')) {
        op.text = collect_group('{', '}');
      } else if (is_punct(peek(), '[')) {
        op.text = collect_group('[', ']');
        op.is_address = true;
        op.space = ins.mem_space;
      } else {
        // Bare operand: one or more tokens until a separator.
        std::string text;
        for (;;) {
          if (at_eof() || is_punct(peek(), ';') || is_punct(peek(), ',') ||
              is_punct(peek(), '{') || is_punct(peek(), '[')) {
            break;
          }
          text += next().text;
        }
        op.text = text;
      }
      ins.operands.push_back(std::move(op));
      if (is_punct(peek(), ',')) {
        next();
        continue;
      }
      break;
    }
    expect_semicolon();

    if (ins.is_branch && !ins.operands.empty()) {
      ins.branch_target = ins.operands.back().text;
    }
    return ins;
  }

  // Collect a balanced group, e.g. "{...}" or "[...]", joining raw token
  // texts (no separators) so "[%rd1+8]" renders as "[%rd1+8]".
  std::string collect_group(char open, char close) {
    const Token &start = next();  // open
    std::string text(1, open);
    int depth = 1;
    while (depth > 0) {
      if (at_eof()) fail(start, "unterminated '" + std::string(1, open) + "' group");
      const Token &t = next();
      if (is_punct(t, open)) ++depth;
      if (is_punct(t, close)) --depth;
      text += t.text;
    }
    return text;
  }

  // --- CFG ------------------------------------------------------------------

  void build_cfg(std::vector<PtxBlock> &blocks) {
    for (std::size_t i = 0; i < blocks.size(); ++i) {
      PtxBlock &b = blocks[i];
      const PtxInstruction *term =
          b.instructions.empty() ? nullptr : &b.instructions.back();

      auto add_target = [&b](const std::string &label) {
        if (std::find(b.successors.begin(), b.successors.end(), label) ==
            b.successors.end()) {
          b.successors.push_back(label);
        }
      };

      if (term != nullptr && term->cls == InstructionClass::kBranch) {
        if (!term->branch_target.empty()) add_target(term->branch_target);
        if (term->has_predicate && i + 1 < blocks.size() &&
            !blocks[i + 1].label.empty()) {
          add_target(blocks[i + 1].label);  // fallthrough after @p bra
        }
      } else if (term != nullptr &&
                 (term->cls == InstructionClass::kReturn ||
                  term->cls == InstructionClass::kExit)) {
        // No successors for ret/exit.
      } else {
        // No terminator or non-terminating tail instruction: fallthrough.
        if (i + 1 < blocks.size() && !blocks[i + 1].label.empty()) {
          add_target(blocks[i + 1].label);
        }
      }
    }
  }

  // --- statistics ------------------------------------------------------------

  void finalize_kernel(PtxKernel &kern) {
    std::int64_t insn = 0, bar_sync = 0, barrier = 0;
    std::int64_t shared = 0, global = 0, local = 0, cst = 0, param = 0;
    for (const PtxBlock &b : kern.cfg) {
      for (const PtxInstruction &i : b.instructions) {
        ++insn;
        ++kern.class_counts[i.cls];
        if (i.cls == InstructionClass::kBarrier) {
          ++barrier;
          if (i.mnemonic == "bar.sync") ++bar_sync;
        }
        if (i.cls == InstructionClass::kLoad ||
            i.cls == InstructionClass::kStore ||
            i.cls == InstructionClass::kAtomic ||
            i.cls == InstructionClass::kReduction) {
          if (i.mem_width_bits > 0) ++kern.mem_width_counts[i.mem_width_bits];
          switch (i.mem_space) {
            case StateSpace::kShared: ++shared; break;
            case StateSpace::kGlobal: ++global; break;
            case StateSpace::kLocal: ++local; break;
            case StateSpace::kConst: ++cst; break;
            case StateSpace::kParam: ++param; break;
            default: break;
          }
        }
        if (i.cls == InstructionClass::kTensorMmaSync ||
            i.cls == InstructionClass::kTensorWmma ||
            i.cls == InstructionClass::kTensorWgmma ||
            i.cls == InstructionClass::kTensorTcgen05) {
          PtxTensorOp op;
          op.mnemonic = i.mnemonic;
          op.class_tag = tensor_tag(i);
          op.parse_only = tensor_parse_only(i);
          kern.tensor_ops.push_back(std::move(op));
        }
      }
    }
    kern.instruction_count = insn;
    kern.bar_sync_count = bar_sync;
    kern.barrier_count = barrier;
    kern.shared_access_count = shared;
    kern.global_access_count = global;
    kern.local_access_count = local;
    kern.const_access_count = cst;
    kern.param_access_count = param;

    std::int64_t edges = 0;
    for (const PtxBlock &b : kern.cfg) edges += b.successors.size();
    kern.cfg_edge_count = edges;
  }

  std::string tensor_tag(const PtxInstruction &i) const {
    if (i.cls == InstructionClass::kTensorMmaSync)
      return (i.segs.size() > 1 && i.segs[1] == "sp") ? "mma.sp" : "mma.sync";
    if (i.cls == InstructionClass::kTensorWmma) return "wmma";
    if (i.cls == InstructionClass::kTensorWgmma)
      return i.segs.size() > 1 ? ("wgmma." + i.segs[1]) : "wgmma";
    if (i.cls == InstructionClass::kTensorTcgen05)
      return (i.segs.size() > 1 && i.segs[1] == "mma") ? "tcgen05.mma"
                                                       : "tcgen05";
    return "";
  }

  bool tensor_parse_only(const PtxInstruction &i) const {
    return i.cls == InstructionClass::kTensorTcgen05;
  }
};

}  // namespace

const char *state_space_name(StateSpace s) noexcept {
  switch (s) {
    case StateSpace::kGlobal: return "global";
    case StateSpace::kShared: return "shared";
    case StateSpace::kLocal: return "local";
    case StateSpace::kConst: return "const";
    case StateSpace::kParam: return "param";
    case StateSpace::kReg: return "reg";
    case StateSpace::kGeneric: return "generic";
    case StateSpace::kUndefined: return "undefined";
  }
  return "undefined";
}

const char *instruction_class_name(InstructionClass c) noexcept {
  switch (c) {
    case InstructionClass::kLoad: return "ld";
    case InstructionClass::kStore: return "st";
    case InstructionClass::kAtomic: return "atom";
    case InstructionClass::kReduction: return "red";
    case InstructionClass::kBarrier: return "bar";
    case InstructionClass::kBranch: return "bra";
    case InstructionClass::kCall: return "call";
    case InstructionClass::kReturn: return "ret";
    case InstructionClass::kExit: return "exit";
    case InstructionClass::kSetp: return "setp";
    case InstructionClass::kMov: return "mov";
    case InstructionClass::kCvt: return "cvt";
    case InstructionClass::kFma: return "fma";
    case InstructionClass::kCvta: return "cvta";
    case InstructionClass::kAlu: return "alu";
    case InstructionClass::kLogic: return "logic";
    case InstructionClass::kMath: return "math";
    case InstructionClass::kSelect: return "select";
    case InstructionClass::kLdMatrix: return "ldmatrix";
    case InstructionClass::kStMatrix: return "stmatrix";
    case InstructionClass::kTexture: return "texture";
    case InstructionClass::kAsyncCopy: return "cp";
    case InstructionClass::kTensorMmaSync: return "mma.sync";
    case InstructionClass::kTensorWmma: return "wmma";
    case InstructionClass::kTensorWgmma: return "wgmma";
    case InstructionClass::kTensorTcgen05: return "tcgen05";
    case InstructionClass::kOther: return "other";
  }
  return "other";
}

PtxModule parse_ptx_module(std::string_view text) {
  if (text.empty()) throw PtxParseError(1, 1, "empty input");
  auto toks = ptx::lex(text);
  Parser parser(std::move(toks));
  return parser.parse_module();
}

}  // namespace kernscope
