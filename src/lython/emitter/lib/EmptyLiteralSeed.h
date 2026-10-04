#pragma once

#include "Ast.h"
#include "TypeSystem.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstddef>
#include <vector>

namespace lython::emitter {

// One suite and the position in it the forward scan starts from. The scan
// reads the REST of a suite and of every suite that one sits inside, so a
// caller hands over its whole stack, innermost first.
struct SuiteCursor {
  const std::vector<parser::NodePtr> *suite = nullptr;
  std::size_t from = 0;
  // One past the last statement the scan reads; the whole rest by default.
  std::size_t to = static_cast<std::size_t>(-1);
  // Which statements of [from, to) the scan reads, by index; all when empty.
  std::vector<bool> keep;

  std::size_t end() const {
    return suite ? std::min(to, suite->size()) : 0;
  }
  bool keeps(std::size_t index) const {
    return keep.empty() || (index < keep.size() && keep[index]);
  }
};

// The element type an empty container literal bound to `name` takes from the
// operations that fill it later: appends, extends, subscript stores, a later
// rebinding to a non-empty literal. Returns a null type when nothing seeds it
// or when two seeds disagree, which leaves the element erased -- where it
// started.
//
// `literalKind` is the literal's node kind ("List", "Dict", "Set", "Tuple"),
// with the `dict()`/`set()`/`list()` call spellings mapped to it by the caller.
// `localSymbols`, when given, are the names a walk has bound so far and the
// symbol table does not hold -- what the SIGNATURE walk carries instead of
// binding into a scope. The emitter passes none: it has bound them already.
// `receiver`, when given, makes the key an ATTRIBUTE of it rather than a bare
// name: `receiver` "self" and `name` "xs" is the field `self.xs`, which is how
// a container held by a class is spelled at every operation that fills it.
//
// `depth` is the recursion this makes into ITSELF for a container that is
// another empty literal, and `subscriptDepth` is how that container is
// SPELLED: 0 is the bare name, 1 is `name[...]` -- which is what a container
// stored inside another container is called at every operation that fills it.
// Callers leave both at 0.
mlir::Type
emptyLiteralSeedTypeIn(const TypeSystem &types, llvm::StringRef name,
                       llvm::StringRef literalKind,
                       llvm::ArrayRef<SuiteCursor> suites,
                       const llvm::StringMap<mlir::Type> *localSymbols = nullptr,
                       unsigned depth = 0, unsigned subscriptDepth = 0,
                       llvm::StringRef receiver = {});

} // namespace lython::emitter
