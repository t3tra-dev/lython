#pragma once

#include "PyDialectTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

inline bool isMemRefSlotCompatibleScalarType(mlir::Type type) {
  return mlir::isa<IntType, BoolType, FloatType, StrType, ObjectType,
                   ClassType>(type);
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
