#pragma once

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering::arch::generic {

mlir::LogicalResult lowerMatmulMicroKernel(mlir::linalg::MatmulOp matmul,
                                           mlir::IRRewriter &rewriter);

} // namespace py::lowering::arch::generic
