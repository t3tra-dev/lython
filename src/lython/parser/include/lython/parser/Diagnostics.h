#pragma once

#include "lython/parser/Ast.h"

#include <string>
#include <vector>

namespace lython::parser {

enum class Severity { Error, Warning };

struct Diagnostic {
  Severity severity = Severity::Error;
  SourceLocation location;
  std::string message;
};

using Diagnostics = std::vector<Diagnostic>;

} // namespace lython::parser
