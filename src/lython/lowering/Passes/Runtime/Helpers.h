#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"

namespace py::lowering::runtime {

namespace async_args {
void mark(mlir::Operation *funcLike, llvm::ArrayRef<mlir::Type> inputs,
          const PyLLVMTypeConverter &typeConverter,
          bool trailingExceptionCell = false);
} // namespace async_args

namespace helpers {
void retainBorrowedEntryBlockReturns(mlir::ModuleOp module);
void synthesizeLocalSelf(mlir::ModuleOp module);
void synthesizePublishedBorrow(mlir::ModuleOp module);
} // namespace helpers

namespace published_borrow {
bool specialize(mlir::func::FuncOp func, unsigned argIndex);
} // namespace published_borrow

} // namespace py::lowering::runtime
