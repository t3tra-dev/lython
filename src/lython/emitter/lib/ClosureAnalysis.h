#pragma once

#include "Ast.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"

#include <string>

namespace lython::emitter {

llvm::SmallVector<std::string, 4>
lexicalCaptureNames(const parser::Node &callable);

// Every name `callable` binds in its own scope: its parameters plus every
// assignment, loop target, with-target, and nested def/class NAME in its body
// (not the bodies of those nested scopes). This is Python's local rule, so it
// answers "does this name shadow an enclosing one" -- unlike
// `collectAssignedNames`, which also reports `xs.append(...)` receivers and
// subscript containers because its question is which locals a loop must carry.
llvm::StringSet<> functionLocalNames(const parser::Node &callable);

// Locals of `callable` that some nested function declares `nonlocal`
// (directly or through intermediate scopes that do not rebind them). These
// must be promoted to shared cells (R6).
llvm::StringSet<> nonlocalBoxedNames(const parser::Node &callable);

// Names bound exactly once directly in `scope`'s own statement list, counting
// a loop target as more than once. Used at module scope to tell a constant
// apart from a name the module rebinds.
llvm::StringSet<> singleAssignmentNames(const parser::Node &scope);

// The complement: names `scope` binds MORE than once, a loop target counted
// as more than once. A lambda that captures one of these froze a value the
// scope goes on to replace.
llvm::StringSet<> reboundNames(const parser::Node &scope);

// Names that a function or lambda nested directly in `body` reads from the
// scope around it. A binding one of these names must be a CELL: the closure
// reads it when it RUNS, not when it was built.
llvm::StringSet<>
namesReadByNestedCallables(const std::vector<parser::NodePtr> *body);

std::string sanitizedSymbolPart(llvm::StringRef text);

} // namespace lython::emitter
