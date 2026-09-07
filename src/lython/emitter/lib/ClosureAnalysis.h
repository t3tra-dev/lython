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

// Locals of `callable` whose FIRST binding in its own statement list comes
// after the first nested function or lambda that reads them. Such a name has
// no value at the def site, so it needs a cell that the later binding fills --
// which is what CPython's frame gives every closed-over name.
llvm::StringSet<> namesBoundAfterNestedReader(const parser::Node &callable);

// The direct nested defs of `callable` that reference EACH OTHER, and the
// names of the enclosing frame they read between them. Members are reached by
// SYMBOL -- the way a nested def already reaches itself -- carrying that ONE
// capture list, so no member ever holds another as a value.
//
// The point is the reference CYCLE: without this, the enclosing frame's cell
// holds one function object, whose closure store holds the other, whose store
// holds the cell, and this runtime has no cycle collector (measured: 400 B per
// call of the enclosing function, unbounded in a loop).
//
// ⛔ ONE capture list for the whole group, sorted, and every member takes it
// whether it reads those names or not: a member naming a sibling has to pass
// the sibling's captures, and the only list it can be sure of is its own.
struct MutualNestedDefGroup {
  llvm::StringSet<> members;
  llvm::SmallVector<std::string, 4> captures;
};
MutualNestedDefGroup mutualNestedDefGroup(const parser::Node &callable);

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
