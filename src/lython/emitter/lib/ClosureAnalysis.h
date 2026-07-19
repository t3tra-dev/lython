#pragma once

#include "Ast.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace lython::emitter {

llvm::SmallVector<std::string, 4>
lexicalCaptureNames(const parser::Node &callable);

// Locals of `callable` that some nested function declares `nonlocal`
// (directly or through intermediate scopes that do not rebind them). These
// must be promoted to shared cells (R6).
llvm::StringSet<> nonlocalBoxedNames(const parser::Node &callable);

std::string sanitizedSymbolPart(llvm::StringRef text);

} // namespace lython::emitter
