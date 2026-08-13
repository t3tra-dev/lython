#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace py::lowering::runtime::cleanup {

bool pointerRoundTrips(mlir::ModuleOp module);

} // namespace py::lowering::runtime::cleanup
