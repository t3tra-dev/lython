#pragma once

#include "Common/Container.h"
#include "PyDialectTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

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

inline bool isEntryBorrowedValue(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
    }
    if (auto ptrToInt = value.getDefiningOp<mlir::LLVM::PtrToIntOp>()) {
      value = ptrToInt.getArg();
      continue;
    }
    if (auto intToPtr = value.getDefiningOp<mlir::LLVM::IntToPtrOp>()) {
      value = intToPtr.getArg();
      continue;
    }
    if (auto bitcast = value.getDefiningOp<mlir::LLVM::BitcastOp>()) {
      value = bitcast.getArg();
      continue;
    }
    if (auto gep = value.getDefiningOp<mlir::LLVM::GEPOp>()) {
      value = gep.getBase();
      continue;
    }
    if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
      value = extract.getContainer();
      continue;
    }
    break;
  }
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return false;
  mlir::Operation *parent = arg.getOwner()->getParentOp();
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
  return function && parent->getNumRegions() != 0 &&
         !parent->getRegion(0).empty() &&
         arg.getOwner() == &parent->getRegion(0).front();
}

namespace container {

enum class SlotPolicy {
  Unsupported,
  NativeInteger,
  NativeBool,
  NativeFloat,
  PointerBits,
};

struct Slot {
  static SlotPolicy policy(mlir::Type type) {
    if (mlir::isa<IntType>(type))
      return SlotPolicy::NativeInteger;
    if (mlir::isa<BoolType>(type))
      return SlotPolicy::NativeBool;
    if (mlir::isa<FloatType>(type))
      return SlotPolicy::NativeFloat;
    if (mlir::isa<NoneType, StrType, ObjectType, ClassType, ExceptionType,
                  TracebackType, LocationType>(type))
      return SlotPolicy::PointerBits;
    return SlotPolicy::Unsupported;
  }

  static bool supported(mlir::Type type) {
    return policy(type) != SlotPolicy::Unsupported;
  }

  static bool packedI64Bootstrap(mlir::Type type) {
    switch (policy(type)) {
    case SlotPolicy::NativeInteger:
    case SlotPolicy::NativeBool:
    case SlotPolicy::NativeFloat:
    case SlotPolicy::PointerBits:
      return true;
    case SlotPolicy::Unsupported:
      return false;
    }
    return false;
  }

  static mlir::Type storageType(mlir::Type logicalType,
                                mlir::MLIRContext *ctx) {
    switch (policy(logicalType)) {
    case SlotPolicy::NativeInteger:
      return mlir::IntegerType::get(ctx, 64);
    case SlotPolicy::NativeBool:
      return mlir::IntegerType::get(ctx, 8);
    case SlotPolicy::NativeFloat:
      return mlir::Float64Type::get(ctx);
    case SlotPolicy::PointerBits:
      return mlir::IntegerType::get(ctx, 64);
    case SlotPolicy::Unsupported:
      return {};
    }
    return {};
  }
};

} // namespace container

inline mlir::MemRefType getListHeaderMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({kListHeaderSize},
                               mlir::IntegerType::get(ctx, 64));
}

inline mlir::MemRefType getListItemsMemRefType(mlir::Type elementType,
                                               mlir::MLIRContext *ctx) {
  mlir::Type storageType = container::Slot::storageType(elementType, ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getTupleHeaderMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({kTupleHeaderSize},
                               mlir::IntegerType::get(ctx, 64));
}

inline mlir::Type getTupleItemsStorageType(TupleType tupleType,
                                           mlir::MLIRContext *ctx) {
  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty())
    return mlir::IntegerType::get(ctx, 64);
  mlir::Type firstStorage =
      container::Slot::storageType(elementTypes.front(), ctx);
  if (!firstStorage)
    return mlir::IntegerType::get(ctx, 64);
  for (mlir::Type elementType : elementTypes.drop_front()) {
    mlir::Type storage = container::Slot::storageType(elementType, ctx);
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
  return mlir::MemRefType::get({kDictHeaderSize},
                               mlir::IntegerType::get(ctx, 64));
}

inline mlir::MemRefType getDictKeysMemRefType(DictType dictType,
                                              mlir::MLIRContext *ctx) {
  mlir::Type storageType =
      container::Slot::storageType(dictType.getKeyType(), ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getDictValuesMemRefType(DictType dictType,
                                                mlir::MLIRContext *ctx) {
  mlir::Type storageType =
      container::Slot::storageType(dictType.getValueType(), ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, storageType);
}

inline mlir::MemRefType getDictStatesMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                               mlir::IntegerType::get(ctx, 8));
}

inline bool isMemRefSlotCompatibleScalarType(mlir::Type type) {
  return container::Slot::supported(type);
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

inline bool isCompilerOwnedMemRefContainerType(mlir::Type type) {
  return isCompilerOwnedMemRefListType(type) ||
         isCompilerOwnedMemRefDictType(type) ||
         isCompilerOwnedMemRefTupleType(type);
}

} // namespace py
