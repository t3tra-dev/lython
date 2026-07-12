#pragma once

#include "Ast.h"

#include <string>
#include <utility>
#include <vector>

namespace lython::parser {

enum class Severity { Error, Warning };

struct Diagnostic {
  Diagnostic() = default;
  Diagnostic(Severity severity, SourceLocation location, std::string message,
             std::string filename = {})
      : severity(severity), location(location), message(std::move(message)),
        filename(std::move(filename)) {}

  Severity severity = Severity::Error;
  SourceLocation location;
  std::string message;
  std::string filename;
};

using Diagnostics = std::vector<Diagnostic>;

} // namespace lython::parser
