#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace py::lowering::runtime::cleanup {

bool pointerRoundTrips(mlir::ModuleOp module);

// Empties the body of every private function no public symbol can reach,
// leaving the symbol and its attributes in place. Answers how many it emptied.
unsigned stripUnreachableManifestBodies(mlir::ModuleOp module);

// Routes each cond_br whose two successors are ONE block (a canonicalized
// empty-arm conditional) through two fresh edge blocks, so every edge into a
// block has a terminator of its own. Answers how many it split.
unsigned splitDualEdges(mlir::ModuleOp module);

} // namespace py::lowering::runtime::cleanup
