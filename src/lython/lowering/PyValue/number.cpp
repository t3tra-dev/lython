#include "Common/RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"

#include <cerrno>
#include <cstdlib>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static bool formatPrimitiveAttr(Attribute attr, std::string &out) {
  if (auto intAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    out = llvm::toString(intAttr.getValue(), 10, true);
    return true;
  }
  if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
    llvm::SmallString<16> buffer;
    floatAttr.getValue().toString(buffer);
    out = buffer.str().str();
    return true;
  }
  return false;
}

static bool collectTensorLiteralElements(Value input,
                                         SmallVector<std::string> &flat) {
  if (auto fromElems = llvm::dyn_cast_or_null<tensor::FromElementsOp>(
          input.getDefiningOp())) {
    for (Value element : fromElems.getElements()) {
      auto constOp = element.getDefiningOp<arith::ConstantOp>();
      if (!constOp)
        return false;
      std::string valueStr;
      if (!formatPrimitiveAttr(constOp.getValue(), valueStr))
        return false;
      flat.push_back(valueStr);
    }
    return true;
  }

  if (auto constOp = input.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = llvm::dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      for (auto val : dense.getValues<Attribute>()) {
        std::string valueStr;
        if (!formatPrimitiveAttr(val, valueStr))
          return false;
        flat.push_back(valueStr);
      }
      return true;
    }
  }

  return false;
}

static func::FuncOp getOrInsertTensorReprFunc(ModuleOp module, StringRef name,
                                              Type memrefType,
                                              Type resultType) {
  if (auto fn = module.lookupSymbol<func::FuncOp>(name))
    return fn;

  OpBuilder builder(module.getContext());
  builder.setInsertionPointToStart(module.getBody());
  auto funcType =
      FunctionType::get(module.getContext(), {memrefType}, {resultType});
  auto func = builder.create<func::FuncOp>(module.getLoc(), name, funcType);
  func->setAttr("llvm.emit_c_interface", builder.getUnitAttr());
  func.setVisibility(SymbolTable::Visibility::Private);
  return func;
}

static std::string formatTensorLiteral(ArrayRef<int64_t> shape,
                                       ArrayRef<std::string> flat,
                                       size_t &index) {
  if (shape.empty()) {
    if (index >= flat.size())
      return "<invalid>";
    return flat[index++];
  }

  std::string result = "[";
  for (int64_t i = 0; i < shape.front(); ++i) {
    if (i > 0)
      result.append(", ");
    result.append(formatTensorLiteral(shape.drop_front(), flat, index));
  }
  result.push_back(']');
  return result;
}

struct StrConstantLowering : public OpConversionPattern<StrConstantOp> {
  StrConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<StrConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(StrConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value dataPtr = runtime.getStringLiteral(op.getLoc(), op.getValueAttr());
    Value length = runtime.getI64Constant(op.getLoc(),
                                          op.getValueAttr().getValue().size());

    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrInternStaticUtf8,
                             resultType, ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct NoneLowering : public OpConversionPattern<NoneOp> {
  NoneLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NoneOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NoneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kGetNone, resultType,
                             ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct IntConstantLowering : public OpConversionPattern<IntConstantOp> {
  IntConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<IntConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(IntConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Integer value is stored as decimal string to support arbitrary precision
    StringRef valueStr = op.getValue();

    // Hybrid approach: try to parse as int64_t for fast path
    // Max int64_t is 19 digits, so only try for shorter strings
    if (valueStr.size() <= 19) {
      char *end;
      errno = 0;
      long long parsed = std::strtoll(valueStr.data(), &end, 10);
      // Check if parsing succeeded: entire string consumed, no overflow
      if (end == valueStr.data() + valueStr.size() && errno != ERANGE) {
        // Use fast LyLong_FromI64 path
        Value value =
            runtime.getI64Constant(op.getLoc(), static_cast<int64_t>(parsed));
        auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                                 resultType, ValueRange{value});
        rewriter.replaceOp(op, call.getResults());
        return success();
      }
    }

    // Fall back to string-based creation for arbitrary precision
    Value dataPtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), valueStr));
    Value length = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(valueStr.size()));
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromString,
                             resultType, ValueRange{dataPtr, length});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct FloatConstantLowering : public OpConversionPattern<FloatConstantOp> {
  FloatConstantLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<FloatConstantOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(FloatConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    Value value =
        runtime.getF64Constant(op.getLoc(), op.getValue().convertToDouble());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                             resultType, ValueRange{value});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Lowering numeric operations to runtime calls
template <typename OpT>
struct NumericBinaryLowering : public OpConversionPattern<OpT> {
  NumericBinaryLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx,
                        llvm::StringLiteral symbol)
      : OpConversionPattern<OpT>(converter, ctx), symbol(symbol) {}

  LogicalResult
  matchAndRewrite(OpT op, typename OpConversionPattern<OpT>::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(this->getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();
    auto call = runtime.call(op.getLoc(), symbol, resultType,
                             ValueRange{adaptor.getOperands()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }

private:
  llvm::StringLiteral symbol;
};

// Type-specialized lowering for NumAddOp
struct NumAddLowering : public OpConversionPattern<NumAddOp> {
  NumAddLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumAddOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol;
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType()))
      symbol = RuntimeSymbols::kLongAdd;
    else if (isPyStrType(op.getLhs().getType()) &&
             isPyStrType(op.getRhs().getType()))
      symbol = RuntimeSymbols::kUnicodeConcat;
    else
      symbol = RuntimeSymbols::kNumberAdd;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Type-specialized lowering for NumSubOp
struct NumSubLowering : public OpConversionPattern<NumSubOp> {
  NumSubLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumSubOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumSubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // Use type-specialized function for int operands
    llvm::StringRef symbol =
        isPyIntType(op.getLhs().getType()) && isPyIntType(op.getRhs().getType())
            ? RuntimeSymbols::kLongSub
            : RuntimeSymbols::kNumberSub;
    auto call =
        runtime.call(op.getLoc(), symbol, resultType, adaptor.getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// Type-specialized lowering for NumLeOp
struct NumLeLowering : public OpConversionPattern<NumLeOp> {
  NumLeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumLeOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumLeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For <=: compare result <= 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value leZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::sle, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{leZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberLe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

// Type-specialized lowering for NumLtOp
struct NumLtLowering : public OpConversionPattern<NumLtOp> {
  NumLtLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumLtOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumLtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For <: compare result < 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value ltZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::slt, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{ltZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberLt,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

// Type-specialized lowering for NumGtOp
struct NumGtLowering : public OpConversionPattern<NumGtOp> {
  NumGtLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumGtOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumGtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For >: compare result > 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value gtZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::sgt, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{gtZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberGt,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

// Type-specialized lowering for NumGeOp
struct NumGeLowering : public OpConversionPattern<NumGeOp> {
  NumGeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumGeOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumGeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For >=: compare result >= 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value geZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::sge, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{geZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberGe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

// Type-specialized lowering for NumEqOp
struct NumEqLowering : public OpConversionPattern<NumEqOp> {
  NumEqLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumEqOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumEqOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For ==: compare result == 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value eqZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::eq, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{eqZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberEq,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};

// Type-specialized lowering for NumNeOp
struct NumNeLowering : public OpConversionPattern<NumNeOp> {
  NumNeLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<NumNeOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(NumNeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    // For int operands, use LyLong_Compare + LyBool_FromBool
    if (isPyIntType(op.getLhs().getType()) &&
        isPyIntType(op.getRhs().getType())) {
      // LyLong_Compare returns int: -1 (less), 0 (equal), 1 (greater)
      auto cmpCall = runtime.call(op.getLoc(), RuntimeSymbols::kLongCompare,
                                  rewriter.getI32Type(), adaptor.getOperands());
      // For !=: compare result != 0
      Value zero = rewriter.create<LLVM::ConstantOp>(
          op.getLoc(), rewriter.getI32Type(), rewriter.getI32IntegerAttr(0));
      Value neZero = rewriter.create<LLVM::ICmpOp>(
          op.getLoc(), LLVM::ICmpPredicate::ne, cmpCall.getResult(), zero);
      // Convert i1 to LyBool
      auto boolCall = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                                   resultType, ValueRange{neZero});
      rewriter.replaceOp(op, boolCall.getResults());
    } else {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kNumberNe,
                               resultType, adaptor.getOperands());
      rewriter.replaceOp(op, call.getResults());
    }
    return success();
  }
};
struct CastToPrimLowering : public OpConversionPattern<CastToPrimOp> {
  CastToPrimLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastToPrimOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastToPrimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = op.getResult().getType();
    Type inputType = op.getInput().getType();

    if (inputType == BoolType::get(rewriter.getContext()) &&
        resultType == rewriter.getI1Type()) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolAsBool,
                               rewriter.getI1Type(), adaptor.getInput());
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (inputType == IntType::get(rewriter.getContext())) {
      Value asI64 = runtime
                        .call(op.getLoc(), RuntimeSymbols::kLongAsI64,
                              rewriter.getI64Type(), adaptor.getInput())
                        .getResult();
      if (resultType == rewriter.getI64Type()) {
        rewriter.replaceOp(op, asI64);
        return success();
      }
      if (resultType == rewriter.getI32Type()) {
        rewriter.replaceOpWithNewOp<LLVM::TruncOp>(op, resultType, asI64);
        return success();
      }
    }

    if (inputType == FloatType::get(rewriter.getContext()) &&
        resultType == rewriter.getF64Type()) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatAsDouble,
                               rewriter.getF64Type(), adaptor.getInput());
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unsupported cast.to_prim conversion");
  }
};

struct CastFromPrimLowering : public OpConversionPattern<CastFromPrimOp> {
  CastFromPrimLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<CastFromPrimOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(CastFromPrimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    Value input = adaptor.getInput();
    Type inputType = input.getType();
    Type pyResultType = op.getResult().getType();

    if (inputType == rewriter.getI1Type() &&
        pyResultType == BoolType::get(rewriter.getContext())) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolFromBool,
                               resultType, ValueRange{input});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    // Handle integer types -> !py.int
    if (auto intType = llvm::dyn_cast<IntegerType>(inputType)) {
      unsigned width = intType.getWidth();
      Value i64Val;

      if (width < 64) {
        // Sign-extend to i64
        auto i64Type = rewriter.getI64Type();
        i64Val = rewriter.create<LLVM::SExtOp>(op.getLoc(), i64Type, input);
      } else if (width == 64) {
        i64Val = input;
      } else {
        return rewriter.notifyMatchFailure(
            op, "integer width > 64 not supported for from_prim");
      }

      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                               resultType, ValueRange{i64Val});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    // Handle float types -> !py.float
    if (llvm::isa<Float32Type>(inputType)) {
      // Extend f32 to f64
      auto f64Type = rewriter.getF64Type();
      Value f64Val =
          rewriter.create<LLVM::FPExtOp>(op.getLoc(), f64Type, input);
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, ValueRange{f64Val});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (llvm::isa<Float64Type>(inputType)) {
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kFloatFromDouble,
                               resultType, ValueRange{input});
      rewriter.replaceOp(op, call.getResults());
      return success();
    }

    if (auto tensorType = llvm::dyn_cast<RankedTensorType>(inputType)) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            op, "from_prim for tensor requires static shape");

      SmallVector<std::string> flat;
      if (!collectTensorLiteralElements(input, flat)) {
        auto elemType = tensorType.getElementType();
        auto floatType = llvm::dyn_cast<mlir::FloatType>(elemType);
        if (!floatType)
          return rewriter.notifyMatchFailure(
              op, "from_prim for tensor requires float element type");

        unsigned width = floatType.getWidth();
        StringRef funcName;
        if (width == 16) {
          funcName = "LyTensorF16_Repr";
        } else if (width == 32) {
          funcName = "LyTensorF32_Repr";
        } else if (width == 64) {
          funcName = "LyTensorF64_Repr";
        } else if (width == 128) {
          funcName = "LyTensorF128_Repr";
        } else {
          return rewriter.notifyMatchFailure(
              op, "unsupported float width for tensor repr");
        }

        auto memrefType = MemRefType::get(tensorType.getShape(), elemType);
        Value memref = rewriter.create<bufferization::ToMemrefOp>(
            op.getLoc(), memrefType, input);
        auto unrankedType = UnrankedMemRefType::get(elemType, 0);
        Value unranked =
            rewriter.create<memref::CastOp>(op.getLoc(), unrankedType, memref);

        auto func = getOrInsertTensorReprFunc(module, funcName, unrankedType,
                                              resultType);
        auto call = rewriter.create<func::CallOp>(op.getLoc(), func,
                                                  ValueRange{unranked});
        rewriter.replaceOp(op, call.getResults());
        return success();
      }

      size_t index = 0;
      std::string text =
          formatTensorLiteral(tensorType.getShape(), flat, index);

      Value dataPtr = runtime.getStringLiteral(
          op.getLoc(), StringAttr::get(rewriter.getContext(), text));
      Value length = runtime.getI64Constant(op.getLoc(),
                                            static_cast<int64_t>(text.size()));
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                               resultType, ValueRange{dataPtr, length});
      rewriter.replaceOp(op, call.getResults());

      if (Operation *def = input.getDefiningOp()) {
        if (input.use_empty() && (llvm::isa<tensor::FromElementsOp>(def) ||
                                  llvm::isa<arith::ConstantOp>(def))) {
          rewriter.eraseOp(def);
        }
      }
      return success();
    }

    return rewriter.notifyMatchFailure(op,
                                       "unsupported input type for from_prim");
  }
};

} // namespace

void populatePyNumberValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns
      .add<StrConstantLowering, NoneLowering, IntConstantLowering,
           FloatConstantLowering, NumAddLowering, NumSubLowering, NumLtLowering,
           NumLeLowering, NumGtLowering, NumGeLowering, NumEqLowering,
           NumNeLowering, CastToPrimLowering, CastFromPrimLowering>(
          typeConverter, ctx);
}

} // namespace py
