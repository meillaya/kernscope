//===- ptx_lexer.cpp - PTX lexer (line/col tokens) ----------------*- C++ -*-===//
//
// Hand-written lexer for PTX 8.x source text (wave-2 todo 9). Produces a flat
// token stream with 1-based line/col so the parser can emit precise
// diagnostics. Handles the PTX lexical quirks the real corpus needs:
//
//   * register arrays:      %r<730>  (one token)
//   * special registers:    %tid.x   (dotted names, one token)
//   * mangled names/labels: _Z15fused_attentionPKfS0_S0_Pfi, $L__BB0_7
//   * dotted mnemonics:     ld.shared.v4.f32, fma.rn.f32 (one token)
//   * namespace suffix:     cta_group::1, kind::f16, .shared::cta (:: is
//                           kept inside the token; a lone ':' is the label
//                           terminator and stays a separate punct token)
//   * numbers:              8.5, -32, +33792, 0x1f, 1.0e-05
//   * strings:              "nounroll", "/abs/path/file.cu"
//   * comments:             // line, /* block */
//
// The lexer is permissive: every byte of a real nvcc PTX file must tokenize.
// Structure is validated by the parser, not the lexer.
//
//===----------------------------------------------------------------------===//

#include "token.h"

#include <cctype>

#include "kernscope/ptx.h"

namespace kernscope {
namespace ptx {

namespace {

struct Cursor {
  std::string_view src;
  std::size_t pos = 0;
  int line = 1;
  int col = 1;
  bool error = false;  // sticky: stop lexing after the first lexical error

  char peek(std::size_t ahead = 0) const {
    return pos + ahead < src.size() ? src[pos + ahead] : '\0';
  }
  char take() {
    char c = src[pos++];
    if (c == '\n') {
      ++line;
      col = 1;
    } else {
      ++col;
    }
    return c;
  }
  [[noreturn]] static void fail(int line, int col, const std::string &msg) {
    throw PtxParseError(line, col, msg);
  }
};

bool is_word_char(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '$';
}

bool is_hex_digit(char c) {
  return std::isxdigit(static_cast<unsigned char>(c)) != 0;
}

// Consume word chars, '.'s and '::' pairs as part of an identifier/directive
// token. A lone ':' terminates the token (the label case: "$L__BB0_7:" lexes
// as "$L__BB0_7" followed by a ':' punct token).
void lex_ident_body(Cursor &cur, std::string &text) {
  for (;;) {
    const char c = cur.peek();
    if (c == ':') {
      if (cur.peek(1) == ':') {
        text += cur.take();
        text += cur.take();
        continue;
      }
      break;
    }
    if (is_word_char(c) || c == '.') {
      text += cur.take();
      continue;
    }
    break;
  }
}

void skip_space_and_comments(Cursor &cur) {
  for (;;) {
    char c = cur.peek();
    if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
      cur.take();
      continue;
    }
    if (c == '/' && cur.peek(1) == '/') {
      while (cur.peek() != '\0' && cur.peek() != '\n') cur.take();
      continue;
    }
    if (c == '/' && cur.peek(1) == '*') {
      int l = cur.line;
      int co = cur.col;
      cur.take();
      cur.take();
      bool closed = false;
      while (cur.peek() != '\0') {
        if (cur.peek() == '*' && cur.peek(1) == '/') {
          cur.take();
          cur.take();
          closed = true;
          break;
        }
        cur.take();
      }
      if (!closed) Cursor::fail(l, co, "unterminated block comment");
      continue;
    }
    break;
  }
}

Token make_token(TokKind kind, std::string text, int line, int col) {
  Token t;
  t.kind = kind;
  t.text = std::move(text);
  t.line = line;
  t.col = col;
  return t;
}

// Lex a number: [+-]? ( 0x.. | 0d.. | 0f.. | decimal [. frac] [e[+-]exp] ).
// Note: a leading '-'/'+' is only folded into the number when followed by a
// digit (or '.'+digit / 0x); otherwise it is a punct token (e.g. inside an
// address operand "[%rd+4]" the '+' would lex as its own number "+4" - the
// parser rejoins tokens, so the rendered operand text is unaffected).
std::string lex_number(Cursor &cur) {
  std::string out;
  if (cur.peek() == '-' || cur.peek() == '+') out += cur.take();
  if (cur.peek() == '0' && (cur.peek(1) == 'x' || cur.peek(1) == 'X' ||
                            cur.peek(1) == 'd' || cur.peek(1) == 'D' ||
                            cur.peek(1) == 'f' || cur.peek(1) == 'F')) {
    out += cur.take();  // '0'
    out += cur.take();  // x/d/f
    while (is_hex_digit(cur.peek()) || cur.peek() == '.') out += cur.take();
    if (cur.peek() == 'p' || cur.peek() == 'P') {
      out += cur.take();
      if (cur.peek() == '+' || cur.peek() == '-') out += cur.take();
      while (is_hex_digit(cur.peek())) out += cur.take();
    }
    return out;
  }
  while (std::isdigit(static_cast<unsigned char>(cur.peek()))) out += cur.take();
  if (cur.peek() == '.') {
    out += cur.take();
    while (std::isdigit(static_cast<unsigned char>(cur.peek()))) out += cur.take();
  }
  if (cur.peek() == 'e' || cur.peek() == 'E') {
    out += cur.take();
    if (cur.peek() == '+' || cur.peek() == '-') out += cur.take();
    while (std::isdigit(static_cast<unsigned char>(cur.peek()))) out += cur.take();
  }
  return out;
}

}  // namespace

std::vector<Token> lex(std::string_view src) {
  std::vector<Token> toks;
  Cursor cur;
  cur.src = src;

  for (;;) {
    skip_space_and_comments(cur);
    if (cur.pos >= src.size()) break;

    const int line = cur.line;
    const int col = cur.col;
    const char c = cur.peek();

    if (c == '%') {
      std::string text(1, cur.take());
      while (is_word_char(cur.peek()) || cur.peek() == '.') text += cur.take();
      if (cur.peek() == '<') {
        text += cur.take();
        while (cur.peek() != '\0' && cur.peek() != '>') text += cur.take();
        if (cur.peek() == '>') {
          text += cur.take();
        } else {
          Cursor::fail(line, col, "unterminated register array '%r<N>'");
        }
      }
      toks.push_back(make_token(TokKind::kRegister, std::move(text), line, col));
      continue;
    }

    if (c == '.') {
      std::string text(1, cur.take());
      lex_ident_body(cur, text);
      toks.push_back(make_token(TokKind::kDirective, std::move(text), line, col));
      continue;
    }

    if (std::isalpha(static_cast<unsigned char>(c)) || c == '_' || c == '$') {
      std::string text(1, cur.take());
      lex_ident_body(cur, text);
      toks.push_back(make_token(TokKind::kIdent, std::move(text), line, col));
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(c)) ||
        ((c == '-' || c == '+') &&
         (std::isdigit(static_cast<unsigned char>(cur.peek(1))) ||
          (cur.peek(1) == '.' &&
           std::isdigit(static_cast<unsigned char>(cur.peek(2)))) ||
          (cur.peek(1) == '0' &&
           (cur.peek(2) == 'x' || cur.peek(2) == 'X'))))) {
      toks.push_back(
          make_token(TokKind::kNumber, lex_number(cur), line, col));
      continue;
    }

    if (c == '"') {
      cur.take();
      std::string text;
      bool closed = false;
      while (cur.peek() != '\0') {
        char q = cur.take();
        if (q == '"') {
          closed = true;
          break;
        }
        if (q == '\\' && cur.peek() != '\0') {
          text += q;
          text += cur.take();
          continue;
        }
        text += q;
      }
      if (!closed) Cursor::fail(line, col, "unterminated string literal");
      toks.push_back(make_token(TokKind::kString, std::move(text), line, col));
      continue;
    }

    // Single-character punctuation.
    static constexpr std::string_view kPunct = ",;[](){}@:!+-<>=~";
    if (kPunct.find(c) != std::string_view::npos) {
      cur.take();
      toks.push_back(make_token(TokKind::kPunct, std::string(1, c), line, col));
      continue;
    }

    Cursor::fail(line, col, std::string("unexpected character '") + c + "'");
  }

  toks.push_back(make_token(TokKind::kEof, "", cur.line, cur.col));
  return toks;
}

}  // namespace ptx
}  // namespace kernscope
