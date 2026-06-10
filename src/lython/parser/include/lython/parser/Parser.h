#pragma once

#include "lython/parser/Ast.h"
#include "lython/parser/Diagnostics.h"

#include <string>
#include <string_view>

namespace lython::parser {

enum class ParseMode { Module, Interactive, Expression, FunctionType };

struct ParseOptions {
  ParseMode mode = ParseMode::Module;
  bool typeComments = false;
};

struct ParseResult {
  NodePtr tree;
  Diagnostics diagnostics;

  bool ok() const { return diagnostics.empty() && tree != nullptr; }
};

ParseResult parse(std::string_view source, std::string filename = "<unknown>",
                  ParseOptions options = {});

} // namespace lython::parser
