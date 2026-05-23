#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Transforms/DialectConversion.h"

namespace py::lowering::runtime::memref_to_llvm::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace py::lowering::runtime::memref_to_llvm::Patterns
