#pragma once

#include "Common/RuntimeSupport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace py::lowering::runtime::conversion {

mlir::LogicalResult
runPartial(mlir::ModuleOp module, mlir::MLIRContext *ctx,
           llvm::function_ref<void(mlir::RewritePatternSet &)> populatePatterns,
           llvm::function_ref<void(mlir::ConversionTarget &)> configureTarget,
           llvm::function_ref<mlir::LogicalResult(mlir::Diagnostic &)>
               materializationFilter);

void configurePyTarget(mlir::ConversionTarget &target);

namespace function::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace function::Patterns

namespace function::definition::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace function::definition::Patterns

namespace function::returns::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace function::returns::Patterns

namespace function::objects::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace function::objects::Patterns

namespace call::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace call::Patterns

namespace value::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace value::Patterns

namespace try_phase::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns);
} // namespace try_phase::Patterns

namespace try_ops {
bool hasStructured(mlir::ModuleOp module);
} // namespace try_ops

namespace types {
void configureValueTarget(mlir::ConversionTarget &target,
                          PyLLVMTypeConverter &typeConverter);
bool same(mlir::TypeRange lhs, llvm::ArrayRef<mlir::Type> rhs);
} // namespace types

} // namespace py::lowering::runtime::conversion
