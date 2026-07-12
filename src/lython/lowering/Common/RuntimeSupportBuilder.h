#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace py::runtime_library {

// Builds the target-independent native runtime support module directly from the
// compiler (the former hand-written `runtime/native/support.mlir`). The
// module is composed from high-level dialects (func, arith, math, cf, scf,
// memref, ub) and only drops to the `llvm` dialect for the irreducible Itanium
// C++ exception ABI. It is lowered to LLVM and linked into every compiled
// program by `linkEmbeddedNativeRuntime`.
mlir::OwningOpRef<mlir::ModuleOp>
buildNativeRuntimeSupportModule(mlir::MLIRContext &context);

} // namespace py::runtime_library
