#pragma once

#include "Common/ClassLayout.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

namespace py::lowering::value::class_::Copy {

mlir::FailureOr<mlir::func::FuncOp>
ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
       mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter);

} // namespace py::lowering::value::class_::Copy

namespace py::lowering::value::class_::Eq {

mlir::FailureOr<mlir::func::FuncOp>
ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
       mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter);

} // namespace py::lowering::value::class_::Eq
