#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "PyValue/ClassHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

namespace lowering::value::tuple::ClassSlot {

mlir::Value carrierFromValues(mlir::Location loc, mlir::ValueRange values,
                              mlir::LLVM::LLVMStructType objectType,
                              mlir::OpBuilder &builder) {
  return Slot::classCarrierFromValues(loc, values, objectType, builder);
}

mlir::Value view(mlir::Location loc, mlir::Value items, int64_t index,
                 mlir::LLVM::LLVMStructType objectType,
                 mlir::ConversionPatternRewriter &rewriter) {
  return Slot::classCarrierView(loc, items,
                                createIndexConstant(loc, rewriter, index),
                                objectType, rewriter);
}

mlir::LogicalResult copyInto(mlir::Location loc, mlir::ModuleOp module,
                             ClassType classType,
                             mlir::LLVM::LLVMStructType objectType,
                             mlir::Value destPtr, mlir::Value source,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter) {
  (void)objectType;
  if (mlir::failed(::py::lowering::value::class_::Copy::ensure(
          loc, module, classType, rewriter, typeConverter)))
    return mlir::failure();
  if (!mlir::isa<mlir::MemRefType>(destPtr.getType()))
    return mlir::failure();
  if (mlir::failed(Slot::classCarrierInitialize(loc, destPtr, classType, module,
                                                rewriter, typeConverter)))
    return mlir::failure();
  return Slot::classCarrierCopyTo(loc, destPtr, source, classType, module,
                                  rewriter);
}

} // namespace lowering::value::tuple::ClassSlot

static mlir::Value stripTupleSourceCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

static bool hasUseAfter(mlir::Operation *anchor, mlir::Value value) {
  value = stripTupleSourceCasts(value);
  for (mlir::Operation *user : value.getUsers()) {
    if (user == anchor)
      continue;
    if (user->getBlock() != anchor->getBlock())
      return true;
    if (anchor->isBeforeInBlock(user))
      return true;
  }
  return false;
}

static std::optional<int64_t> constantTupleIndex(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  int64_t index = integer.getInt();
  if (index < 0)
    return std::nullopt;
  return index;
}

static std::optional<mlir::Type> tupleElementType(TupleType tupleType,
                                                  int64_t index) {
  auto elementTypes = tupleType.getElementTypes();
  if (elementTypes.empty())
    return std::nullopt;
  if (elementTypes.size() == 1)
    return elementTypes.front();
  if (index < 0 || index >= static_cast<int64_t>(elementTypes.size()))
    return std::nullopt;
  return elementTypes[index];
}

static void consumeInlineClassSource(
    mlir::Location loc, mlir::ModuleOp module, ClassType classType,
    mlir::LLVM::LLVMStructType objectType, mlir::ValueRange loweredSource,
    mlir::Value sourceCarrier, mlir::Value logicalSource,
    mlir::Operation *owner, mlir::ConversionPatternRewriter &rewriter) {
  mlir::Value root = stripTupleSourceCasts(logicalSource);
  bool localSource = mlir::isa_and_nonnull<ClassNewOp>(root.getDefiningOp());
  bool canDestroyLocal = localSource && !hasUseAfter(owner, logicalSource) &&
                         loweredSource.size() == 1 &&
                         loweredSource.front().getType() == objectType;
  if (canDestroyLocal) {
    Slot::classRefcount(loc, loweredSource.front(), classType, module, rewriter,
                        "destroy_local", /*aggregateEffect=*/true,
                        ThreadSafetyAttrs::kPremiseOwnedToken);
    return;
  }

  if (!mlir::isa<mlir::MemRefType>(sourceCarrier.getType()))
    return;
  Slot::classCarrierRefcount(loc, sourceCarrier, classType, module, rewriter,
                             "decref", /*aggregateEffect=*/true,
                             ThreadSafetyAttrs::kPremiseOwnedToken);
}

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
    auto items =
        itemsType.hasStaticShape()
            ? rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), itemsType)
            : rewriter.create<mlir::memref::AllocaOp>(
                  op.getLoc(), itemsType,
                  mlir::ValueRange{
                      createIndexConstant(op.getLoc(), rewriter, 0)});
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
    auto header =
        rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
    auto items =
        itemsType.hasStaticShape()
            ? rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), itemsType)
            : rewriter.create<mlir::memref::AllocaOp>(
                  op.getLoc(), itemsType,
                  mlir::ValueRange{createIndexConstant(
                      op.getLoc(), rewriter,
                      static_cast<int64_t>(op.getElements().size()))});
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

    auto elementTypes = tupleType.getElementTypes();
    auto itemCarrierType = class_layout::objectCarrierType(itemsType);
    for (auto [index, element] : llvm::enumerate(adaptor.getElements())) {
      mlir::Type logicalElementType =
          elementTypes.size() == 1 ? elementTypes.front() : elementTypes[index];
      mlir::Value source = element.front();
      if (itemCarrierType) {
        auto classType = mlir::dyn_cast<ClassType>(logicalElementType);
        source = lowering::value::tuple::ClassSlot::carrierFromValues(
            op.getLoc(), element, itemCarrierType, rewriter);
        if (!source)
          return mlir::failure();
        if (classType) {
          mlir::Value destSlot = lowering::value::tuple::ClassSlot::view(
              op.getLoc(), items, static_cast<int64_t>(index), itemCarrierType,
              rewriter);
          if (!destSlot)
            return mlir::failure();
          if (mlir::failed(lowering::value::tuple::ClassSlot::copyInto(
                  op.getLoc(), module, classType, itemCarrierType, destSlot,
                  source, rewriter, *typeConverter)))
            return mlir::failure();
          consumeInlineClassSource(
              op.getLoc(), module, classType, itemCarrierType, element, source,
              op.getElements()[index], op.getOperation(), rewriter);
        } else {
          auto store = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), source, items,
              createIndexConstant(op.getLoc(), rewriter,
                                  static_cast<int64_t>(index)));
          Slot::markTransfer(store.getOperation());
        }
      } else {
        mlir::Value stored = Slot::storage(
            op.getLoc(), element, logicalElementType,
            itemsType.getElementType(), module, rewriter, *typeConverter);
        if (!stored)
          return mlir::failure();
        auto store = rewriter.create<mlir::memref::StoreOp>(
            op.getLoc(), stored, items,
            createIndexConstant(op.getLoc(), rewriter,
                                static_cast<int64_t>(index)));
        Slot::markTransfer(store.getOperation());
      }
      Slot::releaseSource(op.getLoc(), source, logicalElementType, module,
                          rewriter, *typeConverter);
    }

    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{
                mlir::ValueRange{header.getResult(), items.getResult()}});
    return mlir::success();
  }
};

struct TupleGetLowering : public mlir::OpConversionPattern<TupleGetOp> {
  TupleGetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<TupleGetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(TupleGetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto tupleType = mlir::dyn_cast<TupleType>(op.getTuple().getType());
    if (!tupleType)
      return mlir::failure();
    mlir::ValueRange tuple = adaptor.getTuple();
    if (tuple.size() != 2)
      return mlir::failure();
    mlir::Value items = tuple[kTupleItemsComponent];
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
    if (!itemsType)
      return mlir::failure();

    std::optional<int64_t> index = constantTupleIndex(op.getIndex());
    if (!index)
      return rewriter.notifyMatchFailure(op, "tuple.get requires static index");
    std::optional<mlir::Type> elementType = tupleElementType(tupleType, *index);
    if (!elementType)
      return mlir::failure();
    if (class_layout::objectCarrierType(itemsType) &&
        mlir::isa<ClassType>(*elementType))
      return rewriter.notifyMatchFailure(
          op, "tuple.get for inline class tuple elements is not implemented");

    mlir::Value loaded = rewriter.create<mlir::memref::LoadOp>(
        op.getLoc(), items, createIndexConstant(op.getLoc(), rewriter, *index));
    if (!isPyType(*elementType)) {
      if (loaded.getType() != op.getResult().getType())
        return mlir::failure();
      rewriter.replaceOp(op, loaded);
      return mlir::success();
    }

    ownership::aggregate::Slot::markLoad(loaded);
    if (mlir::failed(
            Slot::refcount(op.getLoc(), loaded, *elementType, module, rewriter,
                           *typeConverter, "incref", /*aggregateEffect=*/false,
                           ThreadSafetyAttrs::kPremiseAggregateBorrow)))
      return mlir::failure();
    if (mlir::failed(Slot::replaceBoxedStorage(op.getOperation(), loaded,
                                               *elementType, module, rewriter,
                                               *typeConverter)))
      return mlir::failure();
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::tuple::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<TupleEmptyLowering, TupleCreateLowering, TupleGetLowering>(
      typeConverter, ctx);
}
} // namespace lowering::value::tuple::Patterns

} // namespace py
