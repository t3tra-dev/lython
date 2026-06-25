#pragma once

#include "Ast.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace lython::emitter {

llvm::SmallVector<std::string, 4>
lexicalCaptureNames(const parser::Node &callable);

std::string sanitizedSymbolPart(llvm::StringRef text);

} // namespace lython::emitter
