#pragma once

#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/Object.h"
#include "PyDialectTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "llvm/ADT/STLExtras.h"

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

namespace lowering {

namespace attrs {

inline mlir::SmallVector<int64_t> i64Array(mlir::Operation *op,
                                           llvm::StringRef attrName) {
  mlir::SmallVector<int64_t> values;
  if (!op)
    return values;
  mlir::Attribute attr = op->getAttr(attrName);
  if (!attr)
    return values;

  if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    values.append(dense.asArrayRef().begin(), dense.asArrayRef().end());
    return values;
  }
  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute element : array)
      if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(element))
        values.push_back(integer.getInt());
    return values;
  }
  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    values.push_back(integer.getInt());
  return values;
}

} // namespace attrs

inline mlir::LogicalResult verifyNoUnrealizedCasts(mlir::ModuleOp module,
                                                   llvm::StringRef boundary) {
  llvm::SmallVector<mlir::UnrealizedConversionCastOp> deadCasts;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    if (llvm::all_of(cast->getResults(),
                     [](mlir::Value result) { return result.use_empty(); }))
      deadCasts.push_back(cast);
  });
  for (mlir::UnrealizedConversionCastOp cast : deadCasts)
    cast.erase();

  mlir::UnrealizedConversionCastOp offender = nullptr;
  module.walk([&](mlir::UnrealizedConversionCastOp cast) {
    for (mlir::Operation *user : cast.getResult(0).getUsers())
      if (mlir::isa<mlir::UnrealizedConversionCastOp>(user))
        return mlir::WalkResult::advance();
    offender = cast;
    return mlir::WalkResult::interrupt();
  });
  if (!offender)
    module.walk([&](mlir::UnrealizedConversionCastOp cast) {
      offender = cast;
      return mlir::WalkResult::interrupt();
    });
  if (!offender)
    return mlir::success();

  auto diag = offender.emitError()
              << "unrealized conversion cast reached " << boundary
              << "; operands = " << offender.getOperandTypes()
              << ", results = " << offender.getResultTypes()
              << ", op = " << *offender.getOperation();
  if (auto parentFunc = offender->getParentOfType<mlir::func::FuncOp>())
    diag << ", parent func = " << parentFunc.getName();
  if (auto parentLLVMFunc = offender->getParentOfType<mlir::LLVM::LLVMFuncOp>())
    diag << ", parent llvm func = " << parentLLVMFunc.getName();
  diag << ", operand defs = [";
  for (auto [index, operand] : llvm::enumerate(offender.getOperands())) {
    if (index != 0)
      diag << ", ";
    if (mlir::Operation *def = operand.getDefiningOp())
      diag << *def;
    else
      diag << "<block argument>";
  }
  diag << "]";
  diag << ", users = [";
  bool first = true;
  for (mlir::Operation *user : offender.getResult(0).getUsers()) {
    if (!first)
      diag << ", ";
    first = false;
    diag << *user;
  }
  diag << "]";
  return mlir::failure();
}

} // namespace lowering

inline bool isEntryBorrowedValue(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        value = cast.getOperand(0);
        continue;
      }
      if (cast->getNumOperands() > 1)
        return llvm::all_of(cast.getOperands(), [](mlir::Value operand) {
          return isEntryBorrowedValue(operand);
        });
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
    if (auto insert = value.getDefiningOp<mlir::LLVM::InsertValueOp>())
      return isEntryBorrowedValue(insert.getValue()) &&
             isEntryBorrowedValue(insert.getContainer());
    if (auto extract = value.getDefiningOp<mlir::LLVM::ExtractValueOp>()) {
      value = extract.getContainer();
      continue;
    }
    if (value.getDefiningOp<mlir::LLVM::UndefOp>())
      return true;
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
  ObjectParts,
};

struct Slot {
  static SlotPolicy policy(mlir::Type type) {
    if (mlir::isa<mlir::IntegerType>(type))
      return SlotPolicy::NativeInteger;
    if (mlir::isa<mlir::FloatType>(type))
      return SlotPolicy::NativeFloat;
    if (mlir::isa<IntType>(type))
      return SlotPolicy::NativeInteger;
    if (mlir::isa<BoolType>(type))
      return SlotPolicy::NativeBool;
    if (mlir::isa<FloatType>(type))
      return SlotPolicy::NativeFloat;
    if (mlir::isa<NoneType>(type))
      return SlotPolicy::NativeBool;
    if (mlir::isa<StrType>(type))
      return SlotPolicy::ObjectParts;
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
      return true;
    case SlotPolicy::ObjectParts:
    case SlotPolicy::Unsupported:
      return false;
    }
    return false;
  }

  static mlir::Type storageType(mlir::Type logicalType,
                                mlir::MLIRContext *ctx) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(logicalType)) {
      if (intType.getWidth() == 1)
        return mlir::IntegerType::get(ctx, 8);
      return logicalType;
    }
    if (mlir::isa<mlir::FloatType>(logicalType))
      return logicalType;
    switch (policy(logicalType)) {
    case SlotPolicy::NativeInteger:
      return mlir::IntegerType::get(ctx, 64);
    case SlotPolicy::NativeBool:
      return mlir::IntegerType::get(ctx, 8);
    case SlotPolicy::NativeFloat:
      return mlir::Float64Type::get(ctx);
    case SlotPolicy::ObjectParts: {
      llvm::SmallVector<mlir::Type, 2> parts;
      object_abi::str_abi::Parts::storageTypes(ctx, parts);
      llvm::SmallVector<mlir::MemRefType, 2> memrefs;
      for (mlir::Type part : parts) {
        auto memref = mlir::dyn_cast<mlir::MemRefType>(part);
        if (!memref)
          return {};
        memrefs.push_back(memref);
      }
      return class_layout::objectCarrierType(ctx, memrefs);
    }
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

inline mlir::MemRefType getContainerLockMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
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
    return mlir::IntegerType::get(ctx, 8);
  mlir::Type firstStorage =
      container::Slot::storageType(elementTypes.front(), ctx);
  if (!firstStorage)
    return {};
  for (mlir::Type elementType : elementTypes.drop_front()) {
    mlir::Type storage = container::Slot::storageType(elementType, ctx);
    if (!storage || storage != firstStorage)
      return {};
  }
  return firstStorage;
}

inline mlir::MemRefType getTupleItemsMemRefType(TupleType tupleType,
                                                mlir::MLIRContext *ctx) {
  mlir::Type storageType = getTupleItemsStorageType(tupleType, ctx);
  if (!storageType)
    return {};
  return mlir::MemRefType::get(
      {static_cast<int64_t>(tupleType.getElementTypes().size())}, storageType);
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
  return container::Slot::supported(type) || mlir::isa<ClassType>(type);
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
