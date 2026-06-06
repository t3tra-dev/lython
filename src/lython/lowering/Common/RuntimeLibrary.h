#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace py::runtime_library {

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module);

} // namespace py::runtime_library
