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

// Names bound exactly once directly in `scope`'s own statement list, counting
// a loop target as more than once. Used at module scope to tell a constant
// apart from a name the module rebinds.
llvm::StringSet<> singleAssignmentNames(const parser::Node &scope);

std::string sanitizedSymbolPart(llvm::StringRef text);

} // namespace lython::emitter
