#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering::runtime::async {

mlir::LogicalResult verifyReturnPayloads(mlir::ModuleOp module);
mlir::LogicalResult normalizeFuncSignatures(mlir::ModuleOp module,
                                            PyLLVMTypeConverter &typeConverter);

} // namespace py::lowering::runtime::async
