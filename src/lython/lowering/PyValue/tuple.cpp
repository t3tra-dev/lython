#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/TypedSlotUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static constexpr int64_t kTypedTupleHeaderSlots = 3;

struct TupleEmptyLowering : public OpConversionPattern<TupleEmptyOp> {
  TupleEmptyLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TupleEmptyOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TupleEmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto memrefType = dyn_cast_or_null<MemRefType>(resultType);
    if (!memrefType)
      return failure();
    auto tuple = rewriter.create<memref::AllocaOp>(
        op.getLoc(), memrefType,
        ValueRange{createIndexConstant(op.getLoc(), rewriter,
                                       kTypedTupleHeaderSlots)});
    for (int64_t slot = 0; slot < kTypedTupleHeaderSlots; ++slot) {
      rewriter.create<memref::StoreOp>(
          op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), tuple,
          createIndexConstant(op.getLoc(), rewriter, slot));
    }
    rewriter.replaceOp(op, tuple.getResult());
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

    Type resultType = typeConverter->convertType(op.getResult().getType());
    auto tupleType = dyn_cast<TupleType>(op.getResult().getType());
    if (!tupleType)
      return failure();
    auto memrefType = dyn_cast_or_null<MemRefType>(resultType);
    if (!memrefType)
      return failure();
    Value allocSize = createIndexConstant(
        op.getLoc(), rewriter,
        kTypedTupleHeaderSlots +
            static_cast<int64_t>(adaptor.getElements().size()));
    auto tuple = rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType,
                                                   ValueRange{allocSize});
    rewriter.create<memref::StoreOp>(
        op.getLoc(),
        createI64Constant(op.getLoc(), rewriter,
                          static_cast<int64_t>(adaptor.getElements().size())),
        tuple, createIndexConstant(op.getLoc(), rewriter, 0));
    rewriter.create<memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), tuple,
        createIndexConstant(op.getLoc(), rewriter, 1));
    rewriter.create<memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), tuple,
        createIndexConstant(op.getLoc(), rewriter, 2));

    auto elementTypes = tupleType.getElementTypes();
    for (auto [index, element] : llvm::enumerate(adaptor.getElements())) {
      FailureOr<Value> stored =
          packTypedSlot(op.getLoc(), element, elementTypes[index], module,
                        rewriter, *typeConverter);
      if (failed(stored))
        return failure();
      rewriter.create<memref::StoreOp>(
          op.getLoc(), *stored, tuple,
          createIndexConstant(op.getLoc(), rewriter,
                              kTypedTupleHeaderSlots +
                                  static_cast<int64_t>(index)));
    }

    rewriter.replaceOp(op, tuple.getResult());
    return success();
  }
};

} // namespace

void populatePyTupleValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<TupleEmptyLowering, TupleCreateLowering>(typeConverter, ctx);
}

} // namespace py
