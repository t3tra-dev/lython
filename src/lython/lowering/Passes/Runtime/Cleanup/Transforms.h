#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace py::lowering::runtime::cleanup {

bool pointerRoundTrips(mlir::ModuleOp module);

// Empties the body of every private function no public symbol can reach,
// leaving the symbol and its attributes in place. Answers how many it emptied.
unsigned stripUnreachableManifestBodies(mlir::ModuleOp module);

// Routes each cond_br whose two successors are ONE block (a canonicalized
// empty-arm conditional) through two fresh edge blocks, so every edge into a
// block has a terminator of its own. Answers how many it split.
unsigned splitDualEdges(mlir::ModuleOp module);

// Marks every function body from which no Python exception can propagate
// with `ownership::kCannotRaiseAttr`, by a fixpoint over the call graph.
// Answers how many it marked.
unsigned markCannotRaiseFunctions(mlir::ModuleOp module);

// Gives every converted function so marked LLVM's `nounwind`.
void markCannotRaiseNoUnwind(mlir::ModuleOp module);

// On the casts between a static-shape memref and the pointer that carries it
// across a call (StaticMemRefABI.cpp).
inline constexpr llvm::StringLiteral kBarePointerCastAttr{"ly.abi.bare_pointer"};
// On such a cast whose memref the analysis found at its allocation's start:
// LYTHON_VERIFY_BARE_POINTER_ABI=1 checks it there.
inline constexpr llvm::StringLiteral kBarePointerAtStartAttr{
    "ly.abi.at_start"};

// Passes and returns the static-shape memrefs of every private function that
// is only called directly as a pointer. Answers how many functions changed.
unsigned passStaticMemRefsBare(mlir::ModuleOp module);

} // namespace py::lowering::runtime::cleanup
