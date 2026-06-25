#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace py::lowering::runtime::cleanup {

bool unreachableBlocks(mlir::ModuleOp module);
bool pyBridgeCasts(mlir::Operation *op);
bool pyMultiCasts(mlir::Operation *op);
bool voidPyReturns(mlir::Operation *op);
bool memrefDescriptorCasts(mlir::Operation *op);
bool memrefRuntimeCalls(mlir::Operation *op);
bool pointerRoundTrips(mlir::ModuleOp module);
bool llvmFuncReturns(mlir::Operation *op);
bool finalBoundary(mlir::ModuleOp module);

} // namespace py::lowering::runtime::cleanup
