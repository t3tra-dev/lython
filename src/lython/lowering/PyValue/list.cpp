#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/TypedSlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

enum class TypedContainerStorage {
  Unsupported,
  MemRefI64Slots,
};

static TypedContainerStorage classifyTypedListStorage(ListType listType) {
  if (!listType)
    return TypedContainerStorage::Unsupported;
  if (isMemRefSlotCompatibleScalarType(listType.getElementType()))
    return TypedContainerStorage::MemRefI64Slots;
  return TypedContainerStorage::Unsupported;
}

static bool shouldUseMemRefListStorage(ListType listType) {
  return classifyTypedListStorage(listType) ==
         TypedContainerStorage::MemRefI64Slots;
}

static constexpr int64_t kTypedListHeaderSlots = 4;
static constexpr int64_t kTypedListDefaultCapacity = 64;

struct ListNewLowering : public OpConversionPattern<ListNewOp> {
  using OpConversionPattern<ListNewOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ListNewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    auto listType = dyn_cast<ListType>(op.getResult().getType());
    if (!listType)
      return failure();

    Type resultType = typeConverter->convertType(op.getResult().getType());
    if (!resultType)
      return failure();

    if (shouldUseMemRefListStorage(listType)) {
      auto memrefType = dyn_cast<MemRefType>(resultType);
      if (!memrefType)
        return failure();
      Value allocSize = createIndexConstant(op.getLoc(), rewriter,
                                            kTypedListHeaderSlots +
                                                kTypedListDefaultCapacity);
      auto list = rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType,
                                                    ValueRange{allocSize});
      Value zeroIndex = createIndexConstant(op.getLoc(), rewriter, 0);
      Value oneIndex = createIndexConstant(op.getLoc(), rewriter, 1);
      Value twoIndex = createIndexConstant(op.getLoc(), rewriter, 2);
      Value threeIndex = createIndexConstant(op.getLoc(), rewriter, 3);
      Value zero = createI64Constant(op.getLoc(), rewriter, 0);
      Value capacity =
          createI64Constant(op.getLoc(), rewriter, kTypedListDefaultCapacity);
      rewriter.create<memref::StoreOp>(op.getLoc(), zero, list, zeroIndex);
      rewriter.create<memref::StoreOp>(op.getLoc(), capacity, list, oneIndex);
      rewriter.create<memref::StoreOp>(op.getLoc(), zero, list, twoIndex);
      rewriter.create<memref::StoreOp>(op.getLoc(), zero, list, threeIndex);
      rewriter.replaceOp(op, list.getResult());
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list element type; runtime fallback disabled");
  }
};

static Value normalizeListIndex(Location loc, Value index, Type originalType,
                                ModuleOp module,
                                ConversionPatternRewriter &rewriter,
                                const PyLLVMTypeConverter &typeConverter) {
  auto i64Type = rewriter.getI64Type();
  if (index.getType() == i64Type)
    return index;
  if (isa<IntType>(originalType)) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, i64Type, ValueRange{index})
        .getResult();
  }
  if (auto intType = dyn_cast<IntegerType>(index.getType())) {
    if (intType.getWidth() < 64)
      return rewriter.create<LLVM::SExtOp>(loc, i64Type, index);
    if (intType.getWidth() > 64)
      return rewriter.create<LLVM::TruncOp>(loc, i64Type, index);
    return index;
  }
  return Value();
}

struct ListAppendLowering : public OpConversionPattern<ListAppendOp> {
  using OpConversionPattern<ListAppendOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ListAppendOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto listType = dyn_cast<ListType>(op.getList().getType());
    if (listType && shouldUseMemRefListStorage(listType)) {
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      FailureOr<Value> stored = packTypedSlot(op.getLoc(), adaptor.getValue(),
                                              listType.getElementType(), module,
                                              rewriter, *typeConverter);
      if (failed(stored))
        return failure();
      bool consumeValue =
          static_cast<bool>(op->getAttr("lython.consume_value"));
      if (!consumeValue)
        if (auto classType = dyn_cast<ClassType>(listType.getElementType()))
          emitClassSlotRefcount(op.getLoc(), *stored, classType, module,
                                rewriter, "incref");
      Value sizeIndex = createIndexConstant(op.getLoc(), rewriter, 0);
      Value sizeValue = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getList(), sizeIndex);
      Value sizeAsIndex = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), sizeValue);
      Value itemIndex = rewriter.create<arith::AddIOp>(
          op.getLoc(), sizeAsIndex,
          createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots));
      rewriter.create<memref::StoreOp>(op.getLoc(), *stored, adaptor.getList(),
                                       itemIndex);
      Value nextSize = rewriter.create<arith::AddIOp>(
          op.getLoc(), sizeValue, createI64Constant(op.getLoc(), rewriter, 1));
      rewriter.create<memref::StoreOp>(op.getLoc(), nextSize, adaptor.getList(),
                                       sizeIndex);
      rewriter.eraseOp(op);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list append; runtime fallback disabled");
  }
};

struct ListRemoveLowering : public OpConversionPattern<ListRemoveOp> {
  using OpConversionPattern<ListRemoveOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ListRemoveOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto listType = dyn_cast<ListType>(op.getList().getType());
    if (listType && shouldUseMemRefListStorage(listType)) {
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      FailureOr<Value> target = packTypedSlot(op.getLoc(), adaptor.getValue(),
                                              listType.getElementType(), module,
                                              rewriter, *typeConverter);
      if (failed(target))
        return failure();

      Value sizeIndex = createIndexConstant(op.getLoc(), rewriter, 0);
      Value sizeValue = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getList(), sizeIndex);
      Value sizeAsIndex = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), sizeValue);
      Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
      Value step = createIndexConstant(op.getLoc(), rewriter, 1);
      Value foundInit = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), false, rewriter.getI1Type());

      auto findLoop = rewriter.create<scf::ForOp>(
          op.getLoc(), lower, sizeAsIndex, step, ValueRange{foundInit, lower});
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(findLoop.getBody());
        Value iv = findLoop.getInductionVar();
        Value found = findLoop.getRegionIterArgs()[0];
        Value foundIndex = findLoop.getRegionIterArgs()[1];
        Value itemIndex = rewriter.create<arith::AddIOp>(
            op.getLoc(),
            createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots),
            iv);
        Value item = rewriter.create<memref::LoadOp>(
            op.getLoc(), adaptor.getList(), itemIndex);
        Value same = rewriter.create<arith::CmpIOp>(
            op.getLoc(), arith::CmpIPredicate::eq, item, *target);
        Value notFound = rewriter.create<arith::XOrIOp>(
            op.getLoc(), found,
            rewriter.create<arith::ConstantIntOp>(op.getLoc(), true,
                                                  rewriter.getI1Type()));
        Value match =
            rewriter.create<arith::AndIOp>(op.getLoc(), notFound, same);
        Value nextFound =
            rewriter.create<arith::OrIOp>(op.getLoc(), found, match);
        Value nextIndex = rewriter.create<arith::SelectOp>(op.getLoc(), match,
                                                           iv, foundIndex);
        rewriter.create<scf::YieldOp>(op.getLoc(),
                                      ValueRange{nextFound, nextIndex});
      }

      auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), findLoop.getResult(0),
                                             /*withElseRegion=*/false);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        if (auto classType = dyn_cast<ClassType>(listType.getElementType())) {
          Value removedIndex = rewriter.create<arith::AddIOp>(
              op.getLoc(),
              createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots),
              findLoop.getResult(1));
          Value removed = rewriter.create<memref::LoadOp>(
              op.getLoc(), adaptor.getList(), removedIndex);
          emitClassSlotRefcount(op.getLoc(), removed, classType, module,
                                rewriter, "decref");
        }
        Value shiftLower = rewriter.create<arith::AddIOp>(
            op.getLoc(), findLoop.getResult(1), step);
        auto shiftLoop = rewriter.create<scf::ForOp>(op.getLoc(), shiftLower,
                                                     sizeAsIndex, step);
        {
          OpBuilder::InsertionGuard shiftGuard(rewriter);
          rewriter.setInsertionPointToStart(shiftLoop.getBody());
          Value iv = shiftLoop.getInductionVar();
          Value srcIndex = rewriter.create<arith::AddIOp>(
              op.getLoc(),
              createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots),
              iv);
          Value dstIndex = rewriter.create<arith::AddIOp>(
              op.getLoc(),
              createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots),
              rewriter.create<arith::SubIOp>(op.getLoc(), iv, step));
          Value item = rewriter.create<memref::LoadOp>(
              op.getLoc(), adaptor.getList(), srcIndex);
          rewriter.create<memref::StoreOp>(op.getLoc(), item, adaptor.getList(),
                                           dstIndex);
        }
        Value nextSize = rewriter.create<arith::SubIOp>(
            op.getLoc(), sizeValue,
            createI64Constant(op.getLoc(), rewriter, 1));
        rewriter.create<memref::StoreOp>(op.getLoc(), nextSize,
                                         adaptor.getList(), sizeIndex);
      }
      rewriter.eraseOp(op);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list remove; runtime fallback disabled");
  }
};

struct ListGetLowering : public OpConversionPattern<ListGetOp> {
  using OpConversionPattern<ListGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ListGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto listType = dyn_cast<ListType>(op.getList().getType());
    if (listType && shouldUseMemRefListStorage(listType)) {
      Value index = normalizeListIndex(op.getLoc(), adaptor.getIndex(),
                                       op.getIndex().getType(), module,
                                       rewriter, *typeConverter);
      if (!index)
        return failure();
      Value indexAsIndex = rewriter.create<arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), index);
      Value itemIndex = rewriter.create<arith::AddIOp>(
          op.getLoc(), indexAsIndex,
          createIndexConstant(op.getLoc(), rewriter, kTypedListHeaderSlots));
      Value loaded = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getList(), itemIndex);
      if (auto classType = dyn_cast<ClassType>(listType.getElementType()))
        emitClassSlotRefcount(op.getLoc(), loaded, classType, module, rewriter,
                              "incref");
      FailureOr<Value> boxed =
          boxTypedSlot(op.getLoc(), loaded, listType.getElementType(), module,
                       rewriter, *typeConverter);
      if (failed(boxed))
        return failure();
      rewriter.replaceOp(op, *boxed);
      return success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list get; runtime fallback disabled");
  }
};

} // namespace

void populatePyListValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ListNewLowering, ListAppendLowering, ListRemoveLowering,
               ListGetLowering>(typeConverter, ctx);
}

} // namespace py
