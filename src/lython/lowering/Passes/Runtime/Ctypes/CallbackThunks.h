#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering::ctypes {

// Materializes ctypes callback thunks AFTER the final LLVM-dialect
// conversion. Runtime lowering records each `CFUNCTYPE(...)(f)` as a declared
// `() -> i64` address placeholder carrying the native signature
// (ly.callback.args/result, "s32"/"u64"/"p" codes) and the primitive-ABI
// clone symbol (ly.callback.target); this phase generates the C-ABI thunk
// llvm.func (native args -> widen to (i64, i1) pairs -> call the clone ->
// narrow the result) and fills the placeholder body with
// addressof(thunk) + ptrtoint. Function addresses only exist at the LLVM
// layer, which is why this cannot happen during runtime lowering.
mlir::LogicalResult materializeCallbackThunks(mlir::ModuleOp module);

// Fills `ly.symbol_address` placeholders (from casting a named function
// pointer to an int) with addressof+ptrtoint at the LLVM layer. Run alongside
// materializeCallbackThunks in phase 13c.
mlir::LogicalResult materializeSymbolAddresses(mlir::ModuleOp module);

} // namespace py::lowering::ctypes
