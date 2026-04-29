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
  matchAndRewrite(TupleEmptyOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    SmallVector<Type, 2> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.size() != 2)
      return failure();
    auto headerType = dyn_cast<MemRefType>(resultTypes[0]);
    auto itemsType = dyn_cast<MemRefType>(resultTypes[1]);
    if (!headerType || !itemsType)
      return failure();
    auto header = rewriter.create<memref::AllocaOp>(op.getLoc(), headerType);
    auto items = rewriter.create<memref::AllocaOp>(
        op.getLoc(), itemsType,
        ValueRange{createIndexConstant(op.getLoc(), rewriter, 0)});
    for (int64_t slot = 0; slot < kTypedTupleHeaderSlots; ++slot) {
      rewriter.create<memref::StoreOp>(
          op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
          createIndexConstant(op.getLoc(), rewriter, slot));
    }
    rewriter.replaceOpWithMultiple(
        op, ArrayRef<ValueRange>{
                ValueRange{header.getResult(), items.getResult()}});
    return success();
  }
};

struct TupleCreateLowering : public OpConversionPattern<TupleCreateOp> {
  TupleCreateLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<TupleCreateOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(TupleCreateOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    auto tupleType = dyn_cast<TupleType>(op.getResult().getType());
    if (!tupleType)
      return failure();
    SmallVector<Type, 2> resultTypes;
    if (failed(typeConverter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
        resultTypes.size() != 2)
      return failure();
    auto headerType = dyn_cast<MemRefType>(resultTypes[0]);
    auto itemsType = dyn_cast<MemRefType>(resultTypes[1]);
    if (!headerType || !itemsType)
      return failure();
    Value allocSize = createIndexConstant(
        op.getLoc(), rewriter, static_cast<int64_t>(op.getElements().size()));
    auto header = rewriter.create<memref::AllocaOp>(op.getLoc(), headerType);
    auto items = rewriter.create<memref::AllocaOp>(op.getLoc(), itemsType,
                                                   ValueRange{allocSize});
    rewriter.create<memref::StoreOp>(
        op.getLoc(),
        createI64Constant(op.getLoc(), rewriter,
                          static_cast<int64_t>(op.getElements().size())),
        header, createIndexConstant(op.getLoc(), rewriter, 0));
    rewriter.create<memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
        createIndexConstant(op.getLoc(), rewriter, 1));
    rewriter.create<memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
        createIndexConstant(op.getLoc(), rewriter, 2));

    auto elementTypes = tupleType.getElementTypes();
    for (auto [index, element] : llvm::enumerate(adaptor.getElements())) {
      Value stored = materializeTypedContainerStorageValue(
          op.getLoc(), element.front(), elementTypes[index], module, rewriter,
          *typeConverter);
      if (!stored)
        return failure();
      rewriter.create<memref::StoreOp>(
          op.getLoc(), stored, items,
          createIndexConstant(op.getLoc(), rewriter,
                              static_cast<int64_t>(index)));
    }

    rewriter.replaceOpWithMultiple(
        op, ArrayRef<ValueRange>{
                ValueRange{header.getResult(), items.getResult()}});
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
