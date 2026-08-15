//===- token.h - PTX token definitions (internal) ----------------*- C++ -*-===//
//
// Shared between ptx_lexer.cpp and ptx_parser.cpp. Not installed: this is a
// private implementation detail of libs/ptxparse.
//
// Token kinds:
//   Directive - starts with '.', e.g. ".version", ".shared::cta"
//   Ident     - identifier/mnemonic/label, e.g. "add.s32", "$L__BB0_7",
//               "cta_group::1", "_Z15fused_attention..."
//   Register  - starts with '%', e.g. "%r1", "%f<5510>", "%tid.x"
//   Number    - integer/float/hex literal, e.g. "8.5", "-32", "0x1f"
//   String    - quoted string, e.g. "/path/file.cu"
//   Punct     - single-char delimiter: , ; [ ] ( ) { } @ : ! + - < > =
//
//===----------------------------------------------------------------------===//

#ifndef KERNSCOPE_LIBS_PTXPARSE_SRC_TOKEN_H
#define KERNSCOPE_LIBS_PTXPARSE_SRC_TOKEN_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace kernscope {
namespace ptx {

enum class TokKind : std::uint8_t {
  kDirective,
  kIdent,
  kRegister,
  kNumber,
  kString,
  kPunct,
  kEof,
};

struct Token {
  TokKind kind = TokKind::kEof;
  std::string text;
  int line = 0;  // 1-based
  int col = 0;   // 1-based
};

/// Tokenize `src`. Throws PtxParseError (with line:col) on lexical errors
/// (unterminated string, unexpected character, ...).
std::vector<Token> lex(std::string_view src);

}  // namespace ptx
}  // namespace kernscope

#endif  // KERNSCOPE_LIBS_PTXPARSE_SRC_TOKEN_H
