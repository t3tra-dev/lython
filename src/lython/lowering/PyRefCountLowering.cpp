#include "RuntimeSupport.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct IncRefLowering : public OpConversionPattern<IncRefOp> {
  IncRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<IncRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(IncRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kIncRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DecRefLowering : public OpConversionPattern<DecRefOp> {
  DecRefLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DecRefOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DecRefOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    runtime.call(op.getLoc(), RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
                 adaptor.getObject());
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePyRefCountLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<IncRefLowering, DecRefLowering>(typeConverter, ctx);
}

} // namespace py
