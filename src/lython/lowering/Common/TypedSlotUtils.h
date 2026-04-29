#pragma once

#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace py {

mlir::FailureOr<mlir::Value>
packTypedSlot(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
              mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
              const PyLLVMTypeConverter &typeConverter);

mlir::FailureOr<mlir::Value>
boxTypedSlot(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
             mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
             const PyLLVMTypeConverter &typeConverter);

mlir::Value materializeTypedContainerStorageValue(
    mlir::Location loc, mlir::Value value, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter);

mlir::FailureOr<mlir::Value>
boxTypedContainerStorageValue(mlir::Location loc, mlir::Value value,
                              mlir::Type logicalType, mlir::ModuleOp module,
                              mlir::ConversionPatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter);

void emitClassSlotRefcount(mlir::Location loc, mlir::Value slot, ClassType type,
                           mlir::ModuleOp module,
                           mlir::ConversionPatternRewriter &rewriter,
                           llvm::StringRef suffix);

bool isMutableContainerArgument(mlir::Operation *op, mlir::Value container);

mlir::Value emitManagedContainerPredicate(mlir::Location loc,
                                          mlir::Value header,
                                          int64_t refcountSlot,
                                          mlir::OpBuilder &builder);

void emitManagedContainerLockAcquire(mlir::Location loc, mlir::Value header,
                                     int64_t lockSlot,
                                     mlir::OpBuilder &builder);

void emitManagedContainerLockRelease(mlir::Location loc, mlir::Value header,
                                     int64_t lockSlot,
                                     mlir::OpBuilder &builder);

} // namespace py
