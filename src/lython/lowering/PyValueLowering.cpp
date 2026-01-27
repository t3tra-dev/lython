#include "RuntimeSupport.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cerrno>
#include <cstdlib>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static void getLocInfo(Location loc, MLIRContext *ctx, StringAttr &fileAttr,
                       std::int64_t &line, std::int64_t &col) {
  if (auto fileLoc = llvm::dyn_cast<FileLineColLoc>(loc)) {
    fileAttr = fileLoc.getFilename();
    line = static_cast<std::int64_t>(fileLoc.getLine());
    col = static_cast<std::int64_t>(fileLoc.getColumn());
    return;
  }
  if (auto nameLoc = llvm::dyn_cast<NameLoc>(loc)) {
    getLocInfo(nameLoc.getChildLoc(), ctx, fileAttr, line, col);
    return;
  }
  if (auto fused = llvm::dyn_cast<FusedLoc>(loc)) {
    for (auto subloc : fused.getLocations()) {
      if (auto subfile = llvm::dyn_cast<FileLineColLoc>(subloc)) {
        fileAttr = subfile.getFilename();
        line = static_cast<std::int64_t>(subfile.getLine());
        col = static_cast<std::int64_t>(subfile.getColumn());
        return;
      }
    }
  }
  fileAttr = StringAttr::get(ctx, "<unknown>");
  line = 0;
  col = 0;
}

static StringAttr getFuncNameAttr(func::FuncOp func, MLIRContext *ctx) {
  if (!func)
    return StringAttr::get(ctx, "<unknown>");
  StringRef name = func.getName();
  if (name == "main")
    return StringAttr::get(ctx, "<module>");
  return StringAttr::get(ctx, name);
}

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
  if (auto fromElems =
          llvm::dyn_cast_or_null<tensor::FromElementsOp>(
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
    if (auto dense =
            llvm::dyn_cast<DenseElementsAttr>(constOp.getValue())) {
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
  auto funcType = FunctionType::get(module.getContext(), {memrefType},
                                    {resultType});
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
    result.append(
        formatTensorLiteral(shape.drop_front(), flat, index));
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

    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
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
    llvm::StringRef symbol =
        isPyIntType(op.getLhs().getType()) && isPyIntType(op.getRhs().getType())
            ? RuntimeSymbols::kLongAdd
            : RuntimeSymbols::kNumberAdd;
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
    if (op.getResult().getType() != rewriter.getI1Type())
      return rewriter.notifyMatchFailure(op, "only bool casts supported");
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kBoolAsBool,
                             rewriter.getI1Type(), adaptor.getInput());
    rewriter.replaceOp(op, call.getResults());
    return success();
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
        Value memref =
            rewriter.create<bufferization::ToMemrefOp>(op.getLoc(), memrefType,
                                                       input);
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
      Value length = runtime.getI64Constant(
          op.getLoc(), static_cast<int64_t>(text.size()));
      auto call = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                               resultType, ValueRange{dataPtr, length});
      rewriter.replaceOp(op, call.getResults());

      if (Operation *def = input.getDefiningOp()) {
        if (input.use_empty() &&
            (llvm::isa<tensor::FromElementsOp>(def) ||
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

// For now, class instances are represented as dictionaries.
// ClassNewOp creates a new empty dictionary as instance storage.
struct ClassNewLowering : public OpConversionPattern<ClassNewOp> {
  ClassNewLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassNewOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
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
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kDictNew, resultType,
                             ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

// AttrGetOp lowers to dict lookup
struct AttrGetLowering : public OpConversionPattern<AttrGetOp> {
  AttrGetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrGetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrGetOp op, OpAdaptor adaptor,
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

    // Create string key for attribute name
    StringRef attrName = op.getNameAttr().getValue();
    Value namePtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), attrName));
    Value nameLen = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(attrName.size()));
    auto keyCall = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                                typeConverter->getPyObjectPtrType(),
                                ValueRange{namePtr, nameLen});

    // Lookup in dict
    auto getCall =
        runtime.call(op.getLoc(), RuntimeSymbols::kDictGetItem, resultType,
                     ValueRange{adaptor.getObject(), keyCall.getResult()});
    rewriter.replaceOp(op, getCall.getResults());
    return success();
  }
};

// AttrSetOp lowers to dict insert
struct AttrSetLowering : public OpConversionPattern<AttrSetOp> {
  AttrSetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<AttrSetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(AttrSetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    // Create string key for attribute name
    StringRef attrName = op.getNameAttr().getValue();
    Value namePtr = runtime.getStringLiteral(
        op.getLoc(), StringAttr::get(rewriter.getContext(), attrName));
    Value nameLen = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(attrName.size()));
    auto keyCall = runtime.call(op.getLoc(), RuntimeSymbols::kStrFromUtf8,
                                typeConverter->getPyObjectPtrType(),
                                ValueRange{namePtr, nameLen});

    // Insert into dict
    runtime.call(op.getLoc(), RuntimeSymbols::kDictInsert,
                 typeConverter->getPyObjectPtrType(),
                 ValueRange{adaptor.getObject(), keyCall.getResult(),
                            adaptor.getValue()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ClassOp is just a container - erase it and keep its methods at module scope
struct ClassOpLowering : public OpConversionPattern<ClassOp> {
  ClassOpLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ClassOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Just erase the class op - methods are already at module scope
    rewriter.eraseOp(op);
    return success();
  }
};

struct ExceptionNullLowering : public OpConversionPattern<ExceptionNullOp> {
  ExceptionNullLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ExceptionNullOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ExceptionNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct TracebackNullLowering : public OpConversionPattern<TracebackNullOp> {
  TracebackNullLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TracebackNullOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TracebackNullOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct LocationCurrentLowering : public OpConversionPattern<LocationCurrentOp> {
  LocationCurrentLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<LocationCurrentOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(LocationCurrentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(op, resultType);
    return success();
  }
};

struct ExceptionNewLowering : public OpConversionPattern<ExceptionNewOp> {
  ExceptionNewLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<ExceptionNewOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(ExceptionNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto call = runtime.call(
        op.getLoc(), RuntimeSymbols::kExceptionNew, resultType,
        ValueRange{adaptor.getType(), adaptor.getMessage(), adaptor.getArgs(),
                   adaptor.getCause(), adaptor.getContext(),
                   adaptor.getTraceback(), adaptor.getLocation(),
                   adaptor.getExtras()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct TryLowering : public OpConversionPattern<TryOp> {
  TryLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TryOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TryOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getTryRegion().empty())
      return rewriter.notifyMatchFailure(op, "empty try region");

    bool hasExcept = !op.getExceptRegion().empty();
    bool hasFinally = !op.getFinallyRegion().empty();

    if (!hasExcept)
      return rewriter.notifyMatchFailure(
          op, "try lowering requires except region");

    if (hasFinally && op.getNumResults() > 0)
      return rewriter.notifyMatchFailure(
          op, "finally with results is not supported yet");

    Region *parent = op->getParentRegion();
    Block *parentBlock = op->getBlock();
    auto insertIt = std::next(parent->begin(), std::distance(parent->begin(),
                                                            Region::iterator(
                                                                parentBlock)));
    Block *mergeBlock = rewriter.createBlock(parent, insertIt);
    for (Type resultType : op.getResultTypes())
      mergeBlock->addArgument(resultType, op.getLoc());

    Block *tryEntry = &op.getTryRegion().front();
    Block *exceptEntry = &op.getExceptRegion().front();
    Block *finallyEntry = hasFinally ? &op.getFinallyRegion().front() : nullptr;

    SmallVector<Block *> tryBlocks;
    SmallVector<Block *> exceptBlocks;
    SmallVector<Block *> finallyBlocks;
    for (Block &block : op.getTryRegion())
      tryBlocks.push_back(&block);
    for (Block &block : op.getExceptRegion())
      exceptBlocks.push_back(&block);
    if (hasFinally)
      for (Block &block : op.getFinallyRegion())
        finallyBlocks.push_back(&block);

    // Move operations after py.try into merge block.
    for (auto it = std::next(op->getIterator()); it != parentBlock->end();) {
      Operation *move = &*it++;
      move->moveBefore(mergeBlock, mergeBlock->end());
    }

    // Inline regions into parent before merge block.
    rewriter.inlineRegionBefore(op.getTryRegion(), *parent, mergeBlock->getIterator());
    rewriter.inlineRegionBefore(op.getExceptRegion(), *parent, mergeBlock->getIterator());
    if (hasFinally)
      rewriter.inlineRegionBefore(op.getFinallyRegion(), *parent,
                                  mergeBlock->getIterator());

    // Redirect invokes inside try region to except entry.
    for (Block *block : tryBlocks) {
      for (auto invoke : block->getOps<InvokeOp>()) {
        OpBuilder builder(invoke);
        auto excNull = builder.create<ExceptionNullOp>(
            invoke.getLoc(), ExceptionType::get(op.getContext()));
        invoke.getUnwindDestOperandsMutable().assign(excNull.getResult());
        invoke->setSuccessor(exceptEntry, 1);
      }
    }

    // Replace yields with branches.
    for (Block *block : tryBlocks) {
      if (auto yield = dyn_cast<TryYieldOp>(block->getTerminator())) {
        if (hasFinally) {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, finallyEntry,
                                                    ValueRange{});
        } else {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                    yield.getOperands());
        }
      }
    }
    for (Block *block : exceptBlocks) {
      if (auto yield = dyn_cast<ExceptYieldOp>(block->getTerminator())) {
        if (hasFinally) {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, finallyEntry,
                                                    ValueRange{});
        } else {
          rewriter.setInsertionPoint(yield);
          rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                    yield.getOperands());
        }
      }
    }
    for (Block *block : finallyBlocks) {
      if (auto yield = dyn_cast<FinallyYieldOp>(block->getTerminator())) {
        rewriter.setInsertionPoint(yield);
        rewriter.replaceOpWithNewOp<cf::BranchOp>(yield, mergeBlock,
                                                  ValueRange{});
      }
    }

    // Connect parent block to try entry.
    rewriter.setInsertionPointToEnd(parentBlock);
    rewriter.create<cf::BranchOp>(op.getLoc(), tryEntry);

    for (auto [res, arg] : llvm::zip(op.getResults(), mergeBlock->getArguments()))
      res.replaceAllUsesWith(arg);

    rewriter.eraseOp(op);
    return success();
  }
};

struct RaiseLowering : public OpConversionPattern<RaiseOp> {
  RaiseLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<RaiseOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(RaiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    StringAttr fileAttr;
    std::int64_t line = 0;
    std::int64_t col = 0;
    getLocInfo(op.getLoc(), op.getContext(), fileAttr, line, col);
    StringAttr funcAttr =
        getFuncNameAttr(op->getParentOfType<func::FuncOp>(), op.getContext());
    Value filePtr = runtime.getStringLiteral(op.getLoc(), fileAttr);
    Value funcPtr = runtime.getStringLiteral(op.getLoc(), funcAttr);
    Value lineConst = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(line)));
    Value colConst = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(static_cast<int32_t>(col)));
    runtime.call(op.getLoc(), RuntimeSymbols::kTracebackPush, Type(),
                 ValueRange{filePtr, funcPtr, lineConst, colConst});
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, Type(),
                 ValueRange{adaptor.getException()});
    rewriter.create<LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct RaiseCurrentLowering : public OpConversionPattern<RaiseCurrentOp> {
  RaiseCurrentLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<RaiseCurrentOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(RaiseCurrentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);
    Type pyObject = runtime.getPyObjectPtrType();
    auto current = runtime.call(op.getLoc(), RuntimeSymbols::kExceptionGetCurrent,
                                pyObject, ValueRange{});
    runtime.call(op.getLoc(), RuntimeSymbols::kEHThrow, Type(),
                 current.getResults());
    rewriter.create<LLVM::UnreachableOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns
      .add<StrConstantLowering, NoneLowering, IntConstantLowering,
           FloatConstantLowering, NumAddLowering, NumSubLowering, NumLtLowering,
           NumLeLowering, NumGtLowering, NumGeLowering, NumEqLowering,
           NumNeLowering, CastToPrimLowering, CastFromPrimLowering,
           ClassNewLowering, AttrGetLowering, AttrSetLowering, ClassOpLowering,
           ExceptionNullLowering, TracebackNullLowering,
           LocationCurrentLowering, ExceptionNewLowering, TryLowering,
           RaiseLowering, RaiseCurrentLowering>(
          typeConverter, ctx);
}

} // namespace py
