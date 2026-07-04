#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>

namespace llvm {
class Module;
}

namespace py::runtime_library {

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module);
mlir::LogicalResult linkPrelinkedRuntime(llvm::Module &llvmModule);
mlir::LogicalResult linkEmbeddedNativeRuntime(llvm::Module &llvmModule);

bool prelowerGenerationMode();
std::optional<std::string> prelinkedRuntimeIRPath();

} // namespace py::runtime_library
