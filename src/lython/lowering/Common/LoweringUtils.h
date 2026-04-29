#pragma once

#include "PyDialectTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

namespace py {

inline mlir::Value createIndexConstant(mlir::Location loc,
                                       mlir::OpBuilder &builder,
                                       int64_t value) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, value);
}

inline mlir::Value createI64Constant(mlir::Location loc,
                                     mlir::OpBuilder &builder, int64_t value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value, 64);
}

inline std::string getClassHelperName(ClassType classType,
                                      llvm::StringRef suffix) {
  return ("__ly_class_" + suffix + "_" + classType.getClassName()).str();
}

inline mlir::LLVM::LLVMFuncOp
getOrInsertLLVMFunc(mlir::Location loc, mlir::ModuleOp module,
                    mlir::OpBuilder &builder, llvm::StringRef name,
                    mlir::Type resultType,
                    llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto fnType = mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return builder.create<mlir::LLVM::LLVMFuncOp>(loc, name, fnType);
}

enum class TypedContainerSlotPolicy {
  Unsupported,
  NativeInteger,
  NativeBool,
  NativeFloat,
  PointerBits,
};

inline TypedContainerSlotPolicy getTypedContainerSlotPolicy(mlir::Type type) {
  if (mlir::isa<IntType>(type))
    return TypedContainerSlotPolicy::NativeInteger;
  if (mlir::isa<BoolType>(type))
    return TypedContainerSlotPolicy::NativeBool;
  if (mlir::isa<FloatType>(type))
    return TypedContainerSlotPolicy::NativeFloat;
  if (mlir::isa<StrType, ObjectType, ClassType>(type))
    return TypedContainerSlotPolicy::PointerBits;
  return TypedContainerSlotPolicy::Unsupported;
}

inline bool isTypedContainerSlotSupported(mlir::Type type) {
  return getTypedContainerSlotPolicy(type) !=
         TypedContainerSlotPolicy::Unsupported;
}

inline bool usesPackedI64BootstrapSlot(mlir::Type type) {
  switch (getTypedContainerSlotPolicy(type)) {
  case TypedContainerSlotPolicy::NativeInteger:
  case TypedContainerSlotPolicy::NativeBool:
  case TypedContainerSlotPolicy::NativeFloat:
  case TypedContainerSlotPolicy::PointerBits:
    return true;
  case TypedContainerSlotPolicy::Unsupported:
    return false;
  }
  return false;
}

inline mlir::Type getTypedContainerElementStorageType(mlir::Type logicalType,
                                                      mlir::MLIRContext *ctx) {
  switch (getTypedContainerSlotPolicy(logicalType)) {
  case TypedContainerSlotPolicy::NativeInteger:
    return mlir::IntegerType::get(ctx, 64);
  case TypedContainerSlotPolicy::NativeBool:
    return mlir::IntegerType::get(ctx, 8);
  case TypedContainerSlotPolicy::NativeFloat:
    return mlir::Float64Type::get(ctx);
  case TypedContainerSlotPolicy::PointerBits:
    return mlir::IntegerType::get(ctx, 64);
  case TypedContainerSlotPolicy::Unsupported:
    return {};
  }
  return {};
}

inline mlir::MemRefType getListHeaderMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({4}, mlir::IntegerType::get(ctx, 64));
}

inline mlir::MemRefType getListItemsMemRefType(mlir::Type elementType,
                                               mlir::MLIRContext *ctx) {
  mlir::Type storageType =
      getTypedContainerElementStorageType(elementType, ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getTupleHeaderMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({3}, mlir::IntegerType::get(ctx, 64));
}

inline mlir::Type getTupleItemsStorageType(TupleType tupleType,
                                           mlir::MLIRContext *ctx) {
  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty())
    return mlir::IntegerType::get(ctx, 64);
  mlir::Type firstStorage =
      getTypedContainerElementStorageType(elementTypes.front(), ctx);
  if (!firstStorage)
    return mlir::IntegerType::get(ctx, 64);
  for (mlir::Type elementType : elementTypes.drop_front()) {
    mlir::Type storage = getTypedContainerElementStorageType(elementType, ctx);
    if (storage != firstStorage)
      return mlir::IntegerType::get(ctx, 64);
  }
  return firstStorage;
}

inline mlir::MemRefType getTupleItemsMemRefType(TupleType tupleType,
                                                mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                               getTupleItemsStorageType(tupleType, ctx));
}

inline mlir::MemRefType getDictHeaderMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({5}, mlir::IntegerType::get(ctx, 64));
}

inline mlir::MemRefType getDictKeysMemRefType(DictType dictType,
                                              mlir::MLIRContext *ctx) {
  mlir::Type storageType =
      getTypedContainerElementStorageType(dictType.getKeyType(), ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getDictValuesMemRefType(DictType dictType,
                                                mlir::MLIRContext *ctx) {
  mlir::Type storageType =
      getTypedContainerElementStorageType(dictType.getValueType(), ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getDictStatesMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                               mlir::IntegerType::get(ctx, 8));
}

inline bool isMemRefSlotCompatibleScalarType(mlir::Type type) {
  return isTypedContainerSlotSupported(type);
}

inline bool isCompilerOwnedMemRefListType(mlir::Type type) {
  auto listType = mlir::dyn_cast<ListType>(type);
  return listType &&
         isMemRefSlotCompatibleScalarType(listType.getElementType());
}

inline bool isCompilerOwnedMemRefDictType(mlir::Type type) {
  auto dictType = mlir::dyn_cast<DictType>(type);
  return dictType && isMemRefSlotCompatibleScalarType(dictType.getKeyType()) &&
         isMemRefSlotCompatibleScalarType(dictType.getValueType());
}

inline bool isCompilerOwnedMemRefTupleType(mlir::Type type) {
  return static_cast<bool>(mlir::dyn_cast<TupleType>(type));
}

} // namespace py
