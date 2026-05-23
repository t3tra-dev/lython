#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

struct TupleEmptyLowering : public mlir::OpConversionPattern<TupleEmptyOp> {
  TupleEmptyLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TupleEmptyOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(TupleEmptyOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type, 2> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 2)
      return mlir::failure();
    auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
    if (!headerType || !itemsType)
      return mlir::failure();
    auto header =
        rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
    auto items = rewriter.create<mlir::memref::AllocaOp>(
        op.getLoc(), itemsType,
        mlir::ValueRange{createIndexConstant(op.getLoc(), rewriter, 0)});
    std::string descriptorGroup =
        container::descriptor::Group::make(op.getOperation(), "tuple");
    container::descriptor::Component::mark(
        header.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentHeader);
    container::descriptor::Component::mark(
        items.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentItems);
    for (int64_t slot = 0; slot < kTupleHeaderSize; ++slot) {
      rewriter.create<mlir::memref::StoreOp>(
          op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
          createIndexConstant(op.getLoc(), rewriter, slot));
    }
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{
                mlir::ValueRange{header.getResult(), items.getResult()}});
    return mlir::success();
  }
};

struct TupleCreateLowering : public mlir::OpConversionPattern<TupleCreateOp> {
  TupleCreateLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TupleCreateOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(TupleCreateOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    auto tupleType = mlir::dyn_cast<TupleType>(op.getResult().getType());
    if (!tupleType)
      return mlir::failure();
    llvm::SmallVector<mlir::Type, 2> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 2)
      return mlir::failure();
    auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
    if (!headerType || !itemsType)
      return mlir::failure();
    mlir::Value allocSize = createIndexConstant(
        op.getLoc(), rewriter, static_cast<int64_t>(op.getElements().size()));
    auto header =
        rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
    auto items = rewriter.create<mlir::memref::AllocaOp>(
        op.getLoc(), itemsType, mlir::ValueRange{allocSize});
    std::string descriptorGroup =
        container::descriptor::Group::make(op.getOperation(), "tuple");
    container::descriptor::Component::mark(
        header.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentHeader);
    container::descriptor::Component::mark(
        items.getResult(), descriptorGroup,
        ContainerSafetyAttrs::kComponentItems);
    rewriter.create<mlir::memref::StoreOp>(
        op.getLoc(),
        createI64Constant(op.getLoc(), rewriter,
                          static_cast<int64_t>(op.getElements().size())),
        header, createIndexConstant(op.getLoc(), rewriter, 0));
    rewriter.create<mlir::memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
        createIndexConstant(op.getLoc(), rewriter, 1));
    rewriter.create<mlir::memref::StoreOp>(
        op.getLoc(), createI64Constant(op.getLoc(), rewriter, 0), header,
        createIndexConstant(op.getLoc(), rewriter, 2));

    auto elementTypes = tupleType.getElementTypes();
    for (auto [index, element] : llvm::enumerate(adaptor.getElements())) {
      mlir::Value source = element.front();
      mlir::Value stored =
          Slot::storage(op.getLoc(), source, elementTypes[index], module,
                        rewriter, *typeConverter);
      if (!stored)
        return mlir::failure();
      auto store = rewriter.create<mlir::memref::StoreOp>(
          op.getLoc(), stored, items,
          createIndexConstant(op.getLoc(), rewriter,
                              static_cast<int64_t>(index)));
      Slot::markTransfer(store.getOperation());
      Slot::releaseSource(op.getLoc(), source, elementTypes[index], module,
                          rewriter, *typeConverter);
    }

    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{
                mlir::ValueRange{header.getResult(), items.getResult()}});
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::tuple::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<TupleEmptyLowering, TupleCreateLowering>(typeConverter, ctx);
}
} // namespace lowering::value::tuple::Patterns

} // namespace py
