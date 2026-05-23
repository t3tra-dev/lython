#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace py::lowering::runtime::eh {

void ensureFuncPersonalities(mlir::ModuleOp module);
void finalizeUnwindBlocks(mlir::ModuleOp module);
void wrapTopLevelMain(mlir::ModuleOp module);

} // namespace py::lowering::runtime::eh
