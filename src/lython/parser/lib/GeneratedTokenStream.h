#pragma once

#include "Token.h"
#include "lython/parser/Diagnostics.h"

#include "../pegen.h"

#include <deque>
#include <string>
#include <string_view>
#include <vector>

namespace lython::parser {

struct GeneratedTokenStream {
  std::vector<LythonCpythonToken> tokens;
  std::deque<std::string> storage;

  void reserve(std::size_t size);
  void push(int type, std::string text, SourceRange range);
  void pushView(int type, std::string_view text, SourceRange range);
  void pushToken(const Token &token);
};

bool buildGeneratedTokenStream(const std::vector<Token> &tokens,
                               bool interactive, GeneratedTokenStream &stream,
                               Diagnostics &diagnostics);

} // namespace lython::parser
