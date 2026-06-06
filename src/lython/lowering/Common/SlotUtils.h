#pragma once

#include "Common/Container.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace py {

struct Slot {
  struct ClassCarrierParts {
    mlir::Value header;
    llvm::SmallVector<mlir::Value, 8> payloadParts;
  };

  static mlir::FailureOr<mlir::Value>
  pack(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
       mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
       const PyLLVMTypeConverter &typeConverter);

  static mlir::FailureOr<mlir::Value>
  box(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
      mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
      const PyLLVMTypeConverter &typeConverter);

  static mlir::MemRefType
  classCarrierViewType(mlir::LLVM::LLVMStructType objectType,
                       mlir::MLIRContext *ctx);
  static mlir::Value classCarrierView(mlir::Location loc, mlir::Value items,
                                      mlir::Value index,
                                      mlir::LLVM::LLVMStructType objectType,
                                      mlir::OpBuilder &builder);
  static mlir::Value
  classCarrierFromValues(mlir::Location loc, mlir::ValueRange values,
                         mlir::LLVM::LLVMStructType objectType,
                         mlir::OpBuilder &builder);
  static mlir::func::CallOp classCarrierRefcount(
      mlir::Location loc, mlir::Value slotView, ClassType type,
      mlir::ModuleOp module, mlir::OpBuilder &builder, llvm::StringRef suffix,
      bool aggregateEffect = false,
      llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken);
  static mlir::LogicalResult
  classCarrierCopyTo(mlir::Location loc, mlir::Value destSlotView,
                     mlir::Value sourcePtr, ClassType type,
                     mlir::ModuleOp module, mlir::OpBuilder &builder);
  static mlir::LogicalResult
  classCarrierCopyObjectTo(mlir::Location loc, mlir::Value destObject,
                           mlir::Value sourcePtr, ClassType type,
                           mlir::ModuleOp module, mlir::OpBuilder &builder);
  static mlir::LogicalResult
  classCarrierCopy(mlir::Location loc, mlir::Value destSlotView,
                   mlir::Value sourceSlotView, ClassType type,
                   mlir::ModuleOp module, mlir::OpBuilder &builder);
  static mlir::LogicalResult
  classCarrierInitialize(mlir::Location loc, mlir::Value destSlotView,
                         ClassType type, mlir::ModuleOp module,
                         mlir::OpBuilder &builder,
                         const PyLLVMTypeConverter &typeConverter);
  static mlir::FailureOr<ClassCarrierParts> classCarrierInitializeParts(
      mlir::Location loc, mlir::Value destSlotView, ClassType type,
      mlir::ModuleOp module, mlir::OpBuilder &builder,
      const PyLLVMTypeConverter &typeConverter, bool transferToSlot = true);
  static mlir::Value classCarrierEqual(mlir::Location loc, mlir::Value lhs,
                                       mlir::Value rhs, ClassType type,
                                       mlir::ModuleOp module,
                                       mlir::OpBuilder &builder);

  static mlir::Value storage(mlir::Location loc, mlir::Value value,
                             mlir::Type logicalType, mlir::ModuleOp module,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter);
  static mlir::Value storage(mlir::Location loc, mlir::ValueRange values,
                             mlir::Type logicalType, mlir::ModuleOp module,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter);
  static mlir::Value storage(mlir::Location loc, mlir::Value value,
                             mlir::Type logicalType, mlir::Type storageType,
                             mlir::ModuleOp module,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter);
  static mlir::Value storage(mlir::Location loc, mlir::ValueRange values,
                             mlir::Type logicalType, mlir::Type storageType,
                             mlir::ModuleOp module,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter);
  static mlir::Value
  ownedContainerStorage(mlir::Location loc, mlir::Value value,
                        mlir::Type logicalType, mlir::Type storageType,
                        mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter);
  static mlir::Value
  ownedContainerStorage(mlir::Location loc, mlir::ValueRange values,
                        mlir::Type logicalType, mlir::Type storageType,
                        mlir::ModuleOp module,
                        mlir::ConversionPatternRewriter &rewriter,
                        const PyLLVMTypeConverter &typeConverter);

  static mlir::FailureOr<mlir::Value>
  boxStorage(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
             mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
             const PyLLVMTypeConverter &typeConverter);
  static mlir::LogicalResult
  replaceBoxedStorage(mlir::Operation *op, mlir::Value value,
                      mlir::Type logicalType, mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
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

  static mlir::LogicalResult refcount(
      mlir::Location loc, mlir::Value slot, mlir::Type logicalType,
      mlir::ModuleOp module, mlir::OpBuilder &builder,
      const PyLLVMTypeConverter &typeConverter, llvm::StringRef suffix,
      bool aggregateEffect = false,
      llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseOwnedToken);
};

namespace container {

struct Elements {
  static mlir::LogicalResult
  refcount(mlir::Location loc, mlir::Type logicalType,
           mlir::ValueRange descriptor, mlir::ModuleOp module,
           mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter,
           llvm::StringRef suffix, bool markAggregate = false);
};

struct Managed {
  static bool mutableArgument(mlir::Operation *op, mlir::Value value);
  static mlir::Value predicate(mlir::Location loc, mlir::Value header,
                               int64_t refcountSlot, mlir::OpBuilder &builder);
  static void lock(mlir::Location loc, mlir::Value lock,
                   mlir::OpBuilder &builder);
  static void unlock(mlir::Location loc, mlir::Value lock,
                     mlir::OpBuilder &builder);
};

} // namespace container

} // namespace py
