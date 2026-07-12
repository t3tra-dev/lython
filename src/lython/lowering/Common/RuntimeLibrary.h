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

} // namespace py::runtime_library
