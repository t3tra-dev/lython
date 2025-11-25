#include "RuntimeSupport.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

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
    Value value = runtime.getI64Constant(op.getLoc(), op.getValue());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kLongFromI64,
                             resultType, ValueRange{value});
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
    llvm::StringRef symbol = isPyIntType(op.getLhs().getType()) &&
                                     isPyIntType(op.getRhs().getType())
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
    llvm::StringRef symbol = isPyIntType(op.getLhs().getType()) &&
                                     isPyIntType(op.getRhs().getType())
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

} // namespace

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<StrConstantLowering, NoneLowering, IntConstantLowering,
               FloatConstantLowering, NumAddLowering, NumSubLowering,
               NumLeLowering, CastToPrimLowering>(typeConverter, ctx);
}

} // namespace py
