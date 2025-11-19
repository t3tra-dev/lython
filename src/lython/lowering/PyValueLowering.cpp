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

} // namespace

void populatePyValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<StrConstantLowering, NoneLowering>(typeConverter, ctx);
}

} // namespace py
