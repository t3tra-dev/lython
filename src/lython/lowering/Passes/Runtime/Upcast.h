#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Transforms/DialectConversion.h"

namespace py::lowering::runtime::upcast::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace py::lowering::runtime::upcast::Patterns
