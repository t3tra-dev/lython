#pragma once

#include "Ast.h"

#include "llvm/ADT/DenseSet.h"

namespace lython::emitter {

// Whether CPython 3.14's compiler turns this set display into a frozenset
// constant: three or more elements, none starred, every one a constant once
// its own constant expressions are folded (within the compiler's size
// limits).
bool cpythonFoldsSetDisplay(const parser::Node &set);

// The set displays in `tree` CPython USES as that frozenset, without building
// a set from it: the iterable of a `for` or a comprehension, and the right
// operand of `in` / `not in`.
void collectFrozensetSetDisplays(const parser::Node &tree,
                                 llvm::DenseSet<const parser::Node *> &out);

} // namespace lython::emitter
