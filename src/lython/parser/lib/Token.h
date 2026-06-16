#pragma once

#include "Ast.h"

#include <map>
#include <string>
#include <string_view>

namespace lython::parser {

enum class TokenKind {
  Name,
  Keyword,
  Number,
  String,
  Op,
  Newline,
  Indent,
  Dedent,
  End,
  Invalid,
};

struct Token {
  TokenKind kind = TokenKind::Invalid;
  std::string text;
  std::string rawText;
  std::string cpythonName;
  int cpythonId = -1;
  SourceRange range;
};

struct TypeIgnoreInfo {
  int line = 1;
  std::string tag;
  SourceRange range;
};

struct TypeCommentInfo {
  int line = 1;
  std::string text;
  SourceRange range;
};

using TypeCommentMap = std::map<int, TypeCommentInfo>;

bool literalPrefixContains(std::string_view literal, char marker);

} // namespace lython::parser
