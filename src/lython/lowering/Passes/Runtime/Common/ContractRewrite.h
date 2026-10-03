#pragma once

#include "PyDialectTypes.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLFunctionalExtras.h"

namespace py::lowering {

// Every contract type in the module -- in every op, block argument and
// attribute, and inside unions, protocols, callables and `type[...]` -- is
// replaced by what `rename` answers for it (itself to keep it). A contract's
// arguments are rewritten before it is asked.
void rewriteContracts(
    mlir::ModuleOp module,
    llvm::function_ref<mlir::Type(py::ContractType)> rename);

} // namespace py::lowering
