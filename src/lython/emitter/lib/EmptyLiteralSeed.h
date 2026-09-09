#pragma once

#include "Ast.h"
#include "TypeSystem.h"

#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstddef>
#include <vector>

namespace lython::emitter {

// One suite and the position in it the forward scan starts from. The scan
// reads the REST of a suite and of every suite that one sits inside, so a
// caller hands over its whole stack, innermost first.
struct SuiteCursor {
  const std::vector<parser::NodePtr> *suite = nullptr;
  std::size_t from = 0;
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
// `depth` is the recursion this makes into ITSELF for a local that is another
// empty literal; callers leave it at 0.
mlir::Type
emptyLiteralSeedTypeIn(const TypeSystem &types, llvm::StringRef name,
                       llvm::StringRef literalKind,
                       llvm::ArrayRef<SuiteCursor> suites,
                       const llvm::StringMap<mlir::Type> *localSymbols = nullptr,
                       unsigned depth = 0);

} // namespace lython::emitter
