#ifndef LYTHON_LOWERING_PASSES_RUNTIME_PRIMITIVE_TENSORPARALLEL_H
#define LYTHON_LOWERING_PASSES_RUNTIME_PRIMITIVE_TENSORPARALLEL_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include <memory>

namespace py::lowering {

// Splits large primitive matmuls into equal static row chunks inside a loop
// tagged for parallel dispatch. Runs on bufferized linalg, before the arch
// matmul pipelines.
std::unique_ptr<mlir::Pass> createMatmulParallelChunkPass();

// Outlines tagged chunk loops into private functions so the downstream
// matmul pipeline (tiling, packing, panel hoisting, arch micro-kernels)
// specializes each worker body in isolation. The call stays sequential; the
// dispatch is materialized at the LLVM layer.
std::unique_ptr<mlir::Pass> createParallelLoopOutliningPass();

// Rewrites calls to outlined parallel bodies into a context struct plus a
// LyParallelFor invocation, and synthesizes the pthread-based fork-join
// runtime. Runs on the final LLVM dialect module (function ABIs and struct
// layouts only exist there).
mlir::LogicalResult materializeParallelDispatch(mlir::ModuleOp module);

} // namespace py::lowering

#endif // LYTHON_LOWERING_PASSES_RUNTIME_PRIMITIVE_TENSORPARALLEL_H
