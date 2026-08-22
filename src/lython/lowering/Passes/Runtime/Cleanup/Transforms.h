#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace py::lowering::runtime::cleanup {

bool pointerRoundTrips(mlir::ModuleOp module);

// Empties the body of every private function no public symbol can reach,
// leaving the symbol and its attributes in place. Answers how many it emptied.
unsigned stripUnreachableManifestBodies(mlir::ModuleOp module);

} // namespace py::lowering::runtime::cleanup
