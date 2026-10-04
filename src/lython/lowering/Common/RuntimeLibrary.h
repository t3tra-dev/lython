#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace llvm {
class Module;
}

namespace py::runtime_library {

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module);
// Interprets the transform-dialect lowering strategies embedded in the
// module manifests against the user module.
mlir::LogicalResult applyEmbeddedLoweringStrategies(mlir::ModuleOp module);
mlir::LogicalResult linkEmbeddedNativeRuntime(llvm::Module &llvmModule);
// The part of linkEmbeddedNativeRuntime that does not depend on the program:
// every native runtime module lowered, translated and linked into `runtime`,
// an empty module carrying the target's triple and data layout. What the
// program's link still needs is kept in it as named metadata.
mlir::LogicalResult buildNativeRuntimeModule(llvm::Module &runtime);

} // namespace py::runtime_library
