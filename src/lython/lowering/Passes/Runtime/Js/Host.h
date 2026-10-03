#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering::js {

// Every `js.*` contract the program was typed with becomes `_js.JsProxy`, in
// every type of every op, block and attribute: the host's values have one
// runtime representation, and from here on the only thing a member access
// needs is its name and the static types at its two ends -- which the ops
// carry. A module that did not import the host's `js` is left alone, so a
// program's own module of that name keeps its classes.
void eraseJsHostContracts(mlir::ModuleOp module);

} // namespace py::lowering::js
