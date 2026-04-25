#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"

namespace py {

static bool isPyRuntimeBridgeType(mlir::Type type) {
  return isPyType(type) || mlir::isa<FuncType>(type) ||
         mlir::isa<TupleType>(type) || mlir::isa<ListType>(type) ||
         mlir::isa<ClassType>(type) || mlir::isa<DictType>(type) ||
         mlir::isa<ObjectType>(type);
}

PyLLVMTypeConverter::PyLLVMTypeConverter(mlir::MLIRContext *ctx)
    : mlir::LLVMTypeConverter(ctx) {
  pyObjectPtrType = mlir::LLVM::LLVMPointerType::get(ctx);

  addConversion([](mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<mlir::IntegerType, mlir::FloatType, mlir::RankedTensorType>(
            type))
      return type;
    return std::nullopt;
  });

  addConversion([this](mlir::Type type) -> std::optional<mlir::Type> {
    if (mlir::isa<ListType, DictType, TupleType>(type))
      return std::nullopt;
    if (isPyType(type) || mlir::isa<FuncType>(type) ||
        mlir::isa<TupleType>(type) || mlir::isa<ClassType>(type) ||
        mlir::isa<DictType>(type) || mlir::isa<ObjectType>(type))
      return pyObjectPtrType;
    if (mlir::isa<NoneType>(type))
      return pyObjectPtrType;
    return std::nullopt;
  });

  addConversion([ctx](ListType listType) -> std::optional<mlir::Type> {
    mlir::Type elementType = listType.getElementType();
    if (mlir::isa<IntType, BoolType, FloatType, ClassType>(elementType))
      return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                   mlir::IntegerType::get(ctx, 64));
    return std::nullopt;
  });

  addConversion([ctx](TupleType tupleType) -> std::optional<mlir::Type> {
    (void)tupleType;
    return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                 mlir::IntegerType::get(ctx, 64));
  });

  addConversion([ctx](DictType dictType) -> std::optional<mlir::Type> {
    auto isSlotType = [](mlir::Type type) {
      return mlir::isa<IntType, BoolType, FloatType, StrType, ObjectType,
                       ClassType>(type);
    };
    if (isSlotType(dictType.getKeyType()) &&
        isSlotType(dictType.getValueType()))
      return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                                   mlir::IntegerType::get(ctx, 64));
    return std::nullopt;
  });

  auto materializeBridge = [](mlir::OpBuilder &builder, mlir::Type resultType,
                              mlir::ValueRange inputs,
                              mlir::Location loc) -> mlir::Value {
    if (inputs.size() != 1)
      return {};
    mlir::Type inputType = inputs.front().getType();
    if (!isPyRuntimeBridgeType(resultType) && !isPyRuntimeBridgeType(inputType))
      return {};
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, resultType, inputs)
        .getResult(0);
  };

  addSourceMaterialization(materializeBridge);
  addTargetMaterialization(materializeBridge);
}

RuntimeAPI::RuntimeAPI(mlir::ModuleOp module,
                       mlir::ConversionPatternRewriter &rewriter,
                       const PyLLVMTypeConverter &typeConverter)
    : module(module), rewriter(rewriter),
      pyObjectPtrType(typeConverter.getPyObjectPtrType()) {}

static mlir::LLVM::LLVMFuncOp
declareRuntimeFunc(mlir::Location loc, mlir::ModuleOp module,
                   mlir::ConversionPatternRewriter &rewriter,
                   llvm::StringRef name, mlir::Type resultType,
                   llvm::ArrayRef<mlir::Type> argTypes) {
  if (auto fn = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name))
    return fn;

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto funcType =
      mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);
  return rewriter.create<mlir::LLVM::LLVMFuncOp>(loc, name, funcType);
}

mlir::LLVM::CallOp RuntimeAPI::call(mlir::Location loc, llvm::StringRef name,
                                    mlir::Type resultType,
                                    mlir::ValueRange operands) {
  mlir::SmallVector<mlir::Type> operandTypes;
  operandTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
    operandTypes.push_back(operand.getType());

  mlir::Type actualResult =
      resultType ? resultType
                 : mlir::LLVM::LLVMVoidType::get(module.getContext());
  auto callee = declareRuntimeFunc(loc, module, rewriter, name, actualResult,
                                   operandTypes);
  auto symbolRef =
      mlir::SymbolRefAttr::get(module.getContext(), callee.getName());
  bool isVoid = llvm::isa<mlir::LLVM::LLVMVoidType>(actualResult);
  llvm::SmallVector<mlir::Type, 1> resultStorage;
  if (!isVoid)
    resultStorage.push_back(actualResult);
  mlir::TypeRange results(resultStorage);
  return rewriter.create<mlir::LLVM::CallOp>(loc, results, symbolRef, operands);
}

mlir::Value RuntimeAPI::getStringLiteral(mlir::Location loc,
                                         mlir::StringAttr literal) {
  llvm::SmallString<32> symbolName("__ly_str_");
  auto hashValue = static_cast<uint64_t>(llvm::hash_value(literal.getValue()));
  symbolName += llvm::formatv("{0:X}", hashValue).str();

  mlir::LLVM::GlobalOp global =
      module.lookupSymbol<mlir::LLVM::GlobalOp>(symbolName);
  if (!global) {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto arrayType = mlir::LLVM::LLVMArrayType::get(
        rewriter.getI8Type(), literal.getValue().size() + 1);

    llvm::SmallString<32> storage(literal.getValue());
    storage.push_back('\0');
    global = rewriter.create<mlir::LLVM::GlobalOp>(
        loc, arrayType, /*isConstant=*/true, mlir::LLVM::Linkage::Internal,
        symbolName, rewriter.getStringAttr(storage));
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(module.getContext());
  mlir::Value addr = rewriter.create<mlir::LLVM::AddressOfOp>(
      loc, ptrType, global.getSymNameAttr());
  mlir::Value zero = rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getIndexAttr(0));
  return rewriter.create<mlir::LLVM::GEPOp>(
      loc, ptrType, global.getType(), addr,
      llvm::ArrayRef<mlir::Value>{zero, zero});
}

mlir::Value RuntimeAPI::getI64Constant(mlir::Location loc, std::int64_t value) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(value));
}

mlir::Value RuntimeAPI::getF64Constant(mlir::Location loc, double value) {
  return rewriter.create<mlir::LLVM::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(value));
}

} // namespace py
