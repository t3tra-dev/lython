#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"

namespace py::lowering::runtime {

namespace cleanup {
bool unreachableBlocks(mlir::ModuleOp module);
bool pyBridgeCasts(mlir::Operation *container);
bool pyMultiCasts(mlir::Operation *container);
bool voidPyReturns(mlir::Operation *container);
bool memrefDescriptorCasts(mlir::Operation *container);
} // namespace cleanup

} // namespace py::lowering::runtime
