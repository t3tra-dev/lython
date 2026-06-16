#pragma once

#include "Ast.h"
#include "Diagnostics.h"

#include <string_view>

namespace lython::parser {

void validateCpythonParserContract(Diagnostics &diagnostics);
void validateCpythonAstRootContract(const NodePtr &node,
                                    std::string_view rootKind,
                                    Diagnostics &diagnostics);
void validateCpythonAstContract(const NodePtr &node, Diagnostics &diagnostics);

} // namespace lython::parser
