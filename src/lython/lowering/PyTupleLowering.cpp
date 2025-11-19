#include "RuntimeSupport.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

struct TupleEmptyLowering : public OpConversionPattern<TupleEmptyOp> {
  TupleEmptyLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TupleEmptyOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TupleEmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value zero = runtime.getI64Constant(op.getLoc(), 0);
    auto call = runtime.call(op.getLoc(), RuntimeSymbols::kTupleNew, resultType,
                             ValueRange{zero});
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

struct TupleCreateLowering : public OpConversionPattern<TupleCreateOp> {
  TupleCreateLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TupleCreateOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TupleCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    RuntimeAPI runtime(module, rewriter, *typeConverter);

    Type resultType = typeConverter->convertType(op.getResult().getType());
    Value count = runtime.getI64Constant(
        op.getLoc(), static_cast<int64_t>(adaptor.getElements().size()));
    Value tuple = runtime
                      .call(op.getLoc(), RuntimeSymbols::kTupleNew, resultType,
                            ValueRange{count})
                      .getResult();

    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    for (auto [index, element] : llvm::enumerate(adaptor.getElements())) {
      Value idx = runtime.getI64Constant(op.getLoc(), index);
      runtime.call(op.getLoc(), RuntimeSymbols::kTupleSetItem, voidType,
                   ValueRange{tuple, idx, element});
    }

    rewriter.replaceOp(op, tuple);
    return success();
  }
};

} // namespace

void populatePyTupleLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<TupleEmptyLowering, TupleCreateLowering>(typeConverter, ctx);
}

} // namespace py
