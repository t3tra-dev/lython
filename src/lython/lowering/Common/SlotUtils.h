#pragma once

#include "Common/Container.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringRef.h"

namespace py {

struct Slot {
  static mlir::FailureOr<mlir::Value>
  pack(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
       mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
       const PyLLVMTypeConverter &typeConverter);

  static mlir::FailureOr<mlir::Value>
  box(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
      mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
      const PyLLVMTypeConverter &typeConverter);

  static mlir::Value bridgePointer(mlir::Location loc, mlir::Value value,
                                   mlir::ConversionPatternRewriter &rewriter);

  static mlir::Value storage(mlir::Location loc, mlir::Value value,
                             mlir::Type logicalType, mlir::ModuleOp module,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter);

  static mlir::FailureOr<mlir::Value>
  boxStorage(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
             mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
             const PyLLVMTypeConverter &typeConverter);

  static void classRefcount(
      mlir::Location loc, mlir::Value slot, ClassType type,
      mlir::ModuleOp module, mlir::OpBuilder &builder, llvm::StringRef suffix,
      bool aggregateEffect = false,
      llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken);

  static bool refcounted(mlir::Type logicalType);

  static void markTransfer(mlir::Operation *storeLike);

  static void releaseSource(mlir::Location loc, mlir::Value source,
                            mlir::Type logicalType, mlir::ModuleOp module,
                            mlir::ConversionPatternRewriter &rewriter,
                            const PyLLVMTypeConverter &typeConverter);

  static void refcount(
      mlir::Location loc, mlir::Value slot, mlir::Type logicalType,
      mlir::ModuleOp module, mlir::OpBuilder &builder,
      const PyLLVMTypeConverter &typeConverter, llvm::StringRef suffix,
      bool aggregateEffect = false,
      llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken);
};

namespace container {

struct Elements {
  static void refcount(mlir::Location loc, mlir::Type logicalType,
                       mlir::ValueRange descriptor, mlir::ModuleOp module,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter,
                       llvm::StringRef suffix);
};

struct Managed {
  static bool mutableArgument(mlir::Operation *op, mlir::Value value);
  static mlir::Value predicate(mlir::Location loc, mlir::Value header,
                               int64_t refcountSlot, mlir::OpBuilder &builder);
  static void lock(mlir::Location loc, mlir::Value header, int64_t lockSlot,
                   mlir::OpBuilder &builder);
  static void unlock(mlir::Location loc, mlir::Value header, int64_t lockSlot,
                     mlir::OpBuilder &builder);
};

} // namespace container

} // namespace py
