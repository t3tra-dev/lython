#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>

namespace py::runtime_library {

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module);

bool prelowerGenerationMode();
std::optional<std::string> prelinkedRuntimeIRPath();

} // namespace py::runtime_library
