#include "RuntimeSupport.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct DictEmptyLowering : public OpConversionPattern<DictEmptyOp> {
  DictEmptyLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictEmptyOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictEmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);
    Type resultType = converter->convertType(op.getResult().getType());
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kDictNew, resultType,
                             ValueRange{});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct DictInsertLowering : public OpConversionPattern<DictInsertOp> {
  DictInsertLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictInsertOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *converter);

    Type resultType = converter->convertType(op.getResult().getType());
    auto call = runtime.call(
        op.getLoc(), RuntimeSymbols::kDictInsert, resultType,
        ValueRange{adaptor.getDict(), adaptor.getKey(), adaptor.getValue()});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

} // namespace

void populatePyDictLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<DictEmptyLowering, DictInsertLowering>(typeConverter, ctx);
}

} // namespace py
