#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include <optional>
#include <string>

namespace py::runtime_library {

mlir::LogicalResult embedObjectModules(mlir::ModuleOp module);

// True while this process is generating the pre-lowered runtime cache
// (LYTHON_RUNTIME_PRELOWER=1): imports keep public visibility so SymbolDCE
// preserves the runtime bodies, and the cache itself is never consumed.
bool prelowerGenerationMode();

// Path of the pre-lowered runtime LLVM IR produced at lyc build time, when it
// exists and is usable. When set, embedObjectModules imports declarations and
// contracts only; the driver links this file into the translated module.
std::optional<std::string> prelinkedRuntimeIRPath();

} // namespace py::runtime_library
