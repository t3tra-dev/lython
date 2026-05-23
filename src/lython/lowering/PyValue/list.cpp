#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

enum class StorageKind {
  Unsupported,
  PackedI64BootstrapSlots,
};

namespace lowering::value::list::Storage {

StorageKind classify(ListType listType) {
  if (!listType)
    return StorageKind::Unsupported;
  if (container::Slot::packedI64Bootstrap(listType.getElementType()))
    return StorageKind::PackedI64BootstrapSlots;
  return StorageKind::Unsupported;
}

bool memref(ListType listType) {
  return classify(listType) == StorageKind::PackedI64BootstrapSlots;
}

} // namespace lowering::value::list::Storage

static constexpr int64_t kTypedListDefaultCapacity = 64;

namespace lowering::value::list::Abort {

void emit(mlir::Location loc, mlir::ModuleOp module,
          mlir::ConversionPatternRewriter &rewriter) {
  auto abortFn = getOrInsertLLVMFunc(
      loc, module, rewriter, "abort",
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
  rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{},
      mlir::SymbolRefAttr::get(module.getContext(), abortFn.getName()),
      mlir::ValueRange{});
}

} // namespace lowering::value::list::Abort

struct ListNewLowering : public mlir::OpConversionPattern<ListNewOp> {
  using mlir::OpConversionPattern<ListNewOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListNewOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    auto listType = mlir::dyn_cast<ListType>(op.getResult().getType());
    if (!listType)
      return mlir::failure();

    llvm::SmallVector<mlir::Type, 2> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 2)
      return mlir::failure();

    if (lowering::value::list::Storage::memref(listType)) {
      auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
      if (!headerType || !itemsType)
        return mlir::failure();
      auto header =
          rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
      mlir::Value allocSize =
          createIndexConstant(op.getLoc(), rewriter, kTypedListDefaultCapacity);
      auto items = rewriter.create<mlir::memref::AllocaOp>(
          op.getLoc(), itemsType, mlir::ValueRange{allocSize});
      std::string descriptorGroup =
          container::descriptor::Group::make(op.getOperation(), "list");
      container::descriptor::Component::mark(
          header.getResult(), descriptorGroup,
          ContainerSafetyAttrs::kComponentHeader);
      container::descriptor::Component::mark(
          items.getResult(), descriptorGroup,
          ContainerSafetyAttrs::kComponentItems);
      mlir::Value zeroIndex = createIndexConstant(op.getLoc(), rewriter, 0);
      mlir::Value oneIndex = createIndexConstant(op.getLoc(), rewriter, 1);
      mlir::Value twoIndex = createIndexConstant(op.getLoc(), rewriter, 2);
      mlir::Value threeIndex = createIndexConstant(op.getLoc(), rewriter, 3);
      mlir::Value zero = createI64Constant(op.getLoc(), rewriter, 0);
      mlir::Value capacity =
          createI64Constant(op.getLoc(), rewriter, kTypedListDefaultCapacity);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), zero, header,
                                             zeroIndex);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), capacity, header,
                                             oneIndex);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), zero, header,
                                             twoIndex);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), zero, header,
                                             threeIndex);
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{
                  mlir::ValueRange{header.getResult(), items.getResult()}});
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list element type; runtime fallback disabled");
  }
};

namespace lowering::value::list::Index {

mlir::Value normalize(mlir::Location loc, mlir::Value index,
                      mlir::Type originalType, mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  auto i64Type = rewriter.getI64Type();
  if (index.getType() == i64Type)
    return index;
  if (mlir::isa<IntType>(originalType)) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, i64Type, mlir::ValueRange{index})
        .getResult();
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(index.getType())) {
    if (intType.getWidth() < 64)
      return rewriter.create<mlir::LLVM::SExtOp>(loc, i64Type, index);
    if (intType.getWidth() > 64)
      return rewriter.create<mlir::LLVM::TruncOp>(loc, i64Type, index);
    return index;
  }
  return mlir::Value();
}

} // namespace lowering::value::list::Index

struct ListAppendLowering : public mlir::OpConversionPattern<ListAppendOp> {
  using mlir::OpConversionPattern<ListAppendOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListAppendOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto listType = mlir::dyn_cast<ListType>(op.getList().getType());
    if (listType && lowering::value::list::Storage::memref(listType)) {
      mlir::ValueRange list = adaptor.getList();
      if (list.size() != 2)
        return mlir::failure();
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      mlir::Value stored = Slot::storage(
          op.getLoc(), adaptor.getValue().front(), listType.getElementType(),
          module, rewriter, *typeConverter);
      if (!stored)
        return mlir::failure();
      bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
      llvm::StringRef valueRetainPremise =
          isEntryBorrowedValue(adaptor.getValue().front())
              ? ThreadSafetyAttrs::kPremiseEntryBorrowed
              : ThreadSafetyAttrs::kPremiseOwnedToken;
      auto emitAppendBody = [&](bool sharedAccess) {
        auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
          if (sharedAccess)
            container::access::Contract::mark(access, list[0], target);
        };
        mlir::Value sizeIndex =
            createIndexConstant(op.getLoc(), rewriter, kTypedListSizeSlot);
        mlir::Value capacityIndex =
            createIndexConstant(op.getLoc(), rewriter, 1);
        auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), list[0], sizeIndex);
        markAccess(sizeLoad.getOperation(), list[0]);
        mlir::Value sizeValue = sizeLoad;
        auto capacityLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), list[0], capacityIndex);
        markAccess(capacityLoad.getOperation(), list[0]);
        mlir::Value capacityValue = capacityLoad;
        mlir::Value hasCapacity = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), mlir::arith::CmpIPredicate::ult, sizeValue,
            capacityValue);
        auto boundsIf =
            rewriter.create<mlir::scf::IfOp>(op.getLoc(), hasCapacity,
                                             /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard boundsGuard(rewriter);
          rewriter.setInsertionPointToStart(boundsIf.thenBlock());
          if (!consumeValue)
            Slot::refcount(op.getLoc(), stored, listType.getElementType(),
                           module, rewriter, *typeConverter, "incref",
                           /*aggregateEffect=*/false, valueRetainPremise);
          mlir::Value sizeAsIndex = rewriter.create<mlir::arith::IndexCastOp>(
              op.getLoc(), rewriter.getIndexType(), sizeValue);
          auto store = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), stored, list[1], sizeAsIndex);
          markAccess(store.getOperation(), list[1]);
          Slot::markTransfer(store.getOperation());
          mlir::Value nextSize = rewriter.create<mlir::arith::AddIOp>(
              op.getLoc(), sizeValue,
              createI64Constant(op.getLoc(), rewriter, 1));
          auto sizeStore = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), nextSize, list[0], sizeIndex);
          markAccess(sizeStore.getOperation(), list[0]);
        }
        {
          mlir::OpBuilder::InsertionGuard boundsGuard(rewriter);
          rewriter.setInsertionPointToStart(boundsIf.elseBlock());
          if (consumeValue)
            Slot::refcount(op.getLoc(), stored, listType.getElementType(),
                           module, rewriter, *typeConverter, "decref");
          lowering::value::list::Abort::emit(op.getLoc(), module, rewriter);
        }
      };
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), list[0], kTypedListRefcountSlot, rewriter);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                   /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        container::Managed::lock(op.getLoc(), list[0], kTypedListLockSlot,
                                 rewriter);
        emitAppendBody(/*sharedAccess=*/true);
        container::Managed::unlock(op.getLoc(), list[0], kTypedListLockSlot,
                                   rewriter);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        emitAppendBody(/*sharedAccess=*/false);
      }
      if (consumeValue)
        Slot::releaseSource(op.getLoc(), adaptor.getValue().front(),
                            listType.getElementType(), module, rewriter,
                            *typeConverter);
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list append; runtime fallback disabled");
  }
};

struct ListRemoveLowering : public mlir::OpConversionPattern<ListRemoveOp> {
  using mlir::OpConversionPattern<ListRemoveOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListRemoveOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto listType = mlir::dyn_cast<ListType>(op.getList().getType());
    if (listType && lowering::value::list::Storage::memref(listType)) {
      mlir::ValueRange list = adaptor.getList();
      if (list.size() != 2)
        return mlir::failure();
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      mlir::Value target = Slot::storage(
          op.getLoc(), adaptor.getValue().front(), listType.getElementType(),
          module, rewriter, *typeConverter);
      if (!target)
        return mlir::failure();

      auto emitRemoveBody = [&](bool sharedAccess) {
        auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
          if (sharedAccess)
            container::access::Contract::mark(access, list[0], target);
        };
        mlir::Value sizeIndex =
            createIndexConstant(op.getLoc(), rewriter, kTypedListSizeSlot);
        auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), list[0], sizeIndex);
        markAccess(sizeLoad.getOperation(), list[0]);
        mlir::Value sizeValue = sizeLoad;
        mlir::Value sizeAsIndex = rewriter.create<mlir::arith::IndexCastOp>(
            op.getLoc(), rewriter.getIndexType(), sizeValue);
        mlir::Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
        mlir::Value step = createIndexConstant(op.getLoc(), rewriter, 1);
        mlir::Value foundInit = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), false, rewriter.getI1Type());

        auto findLoop = rewriter.create<mlir::scf::ForOp>(
            op.getLoc(), lower, sizeAsIndex, step,
            mlir::ValueRange{foundInit, lower});
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(findLoop.getBody());
          mlir::Value iv = findLoop.getInductionVar();
          mlir::Value found = findLoop.getRegionIterArgs()[0];
          mlir::Value foundIndex = findLoop.getRegionIterArgs()[1];
          auto itemLoad =
              rewriter.create<mlir::memref::LoadOp>(op.getLoc(), list[1], iv);
          markAccess(itemLoad.getOperation(), list[1]);
          mlir::Value item = itemLoad;
          mlir::Value same = rewriter.create<mlir::arith::CmpIOp>(
              op.getLoc(), mlir::arith::CmpIPredicate::eq, item, target);
          mlir::Value notFound = rewriter.create<mlir::arith::XOrIOp>(
              op.getLoc(), found,
              rewriter.create<mlir::arith::ConstantIntOp>(
                  op.getLoc(), true, rewriter.getI1Type()));
          mlir::Value match =
              rewriter.create<mlir::arith::AndIOp>(op.getLoc(), notFound, same);
          mlir::Value nextFound =
              rewriter.create<mlir::arith::OrIOp>(op.getLoc(), found, match);
          mlir::Value nextIndex = rewriter.create<mlir::arith::SelectOp>(
              op.getLoc(), match, iv, foundIndex);
          rewriter.create<mlir::scf::YieldOp>(
              op.getLoc(), mlir::ValueRange{nextFound, nextIndex});
        }

        auto ifOp =
            rewriter.create<mlir::scf::IfOp>(op.getLoc(), findLoop.getResult(0),
                                             /*withElseRegion=*/false);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          if (Slot::refcounted(listType.getElementType())) {
            auto removedLoad = rewriter.create<mlir::memref::LoadOp>(
                op.getLoc(), list[1], findLoop.getResult(1));
            markAccess(removedLoad.getOperation(), list[1]);
            mlir::Value removed = removedLoad;
            Slot::refcount(op.getLoc(), removed, listType.getElementType(),
                           module, rewriter, *typeConverter, "decref",
                           /*aggregateEffect=*/true);
          }
          mlir::Value shiftLower = rewriter.create<mlir::arith::AddIOp>(
              op.getLoc(), findLoop.getResult(1), step);
          auto shiftLoop = rewriter.create<mlir::scf::ForOp>(
              op.getLoc(), shiftLower, sizeAsIndex, step);
          {
            mlir::OpBuilder::InsertionGuard shiftGuard(rewriter);
            rewriter.setInsertionPointToStart(shiftLoop.getBody());
            mlir::Value iv = shiftLoop.getInductionVar();
            mlir::Value dstIndex =
                rewriter.create<mlir::arith::SubIOp>(op.getLoc(), iv, step);
            auto itemLoad =
                rewriter.create<mlir::memref::LoadOp>(op.getLoc(), list[1], iv);
            markAccess(itemLoad.getOperation(), list[1]);
            mlir::Value item = itemLoad;
            auto itemStore = rewriter.create<mlir::memref::StoreOp>(
                op.getLoc(), item, list[1], dstIndex);
            markAccess(itemStore.getOperation(), list[1]);
          }
          mlir::Value nextSize = rewriter.create<mlir::arith::SubIOp>(
              op.getLoc(), sizeValue,
              createI64Constant(op.getLoc(), rewriter, 1));
          auto sizeStore = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), nextSize, list[0], sizeIndex);
          markAccess(sizeStore.getOperation(), list[0]);
        }
      };
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), list[0], kTypedListRefcountSlot, rewriter);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                   /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        container::Managed::lock(op.getLoc(), list[0], kTypedListLockSlot,
                                 rewriter);
        emitRemoveBody(/*sharedAccess=*/true);
        container::Managed::unlock(op.getLoc(), list[0], kTypedListLockSlot,
                                   rewriter);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        emitRemoveBody(/*sharedAccess=*/false);
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list remove; runtime fallback disabled");
  }
};

struct ListGetLowering : public mlir::OpConversionPattern<ListGetOp> {
  using mlir::OpConversionPattern<ListGetOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(ListGetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto listType = mlir::dyn_cast<ListType>(op.getList().getType());
    if (listType && lowering::value::list::Storage::memref(listType)) {
      mlir::ValueRange list = adaptor.getList();
      if (list.size() != 2)
        return mlir::failure();
      mlir::Value index = lowering::value::list::Index::normalize(
          op.getLoc(), adaptor.getIndex().front(), op.getIndex().getType(),
          module, rewriter, *typeConverter);
      if (!index)
        return mlir::failure();
      mlir::Value indexAsIndex = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), index);
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(list[1].getType());
      if (!itemsType)
        return mlir::failure();
      auto emitLoadAndRetain = [&](llvm::StringRef retainPremise,
                                   bool sharedAccess) -> mlir::Value {
        auto load = rewriter.create<mlir::memref::LoadOp>(op.getLoc(), list[1],
                                                          indexAsIndex);
        if (sharedAccess)
          container::access::Contract::mark(load.getOperation(), list[0],
                                            list[1]);
        else
          ownership::aggregate::Slot::markLoad(load.getResult());
        mlir::Value loaded = load;
        Slot::refcount(op.getLoc(), loaded, listType.getElementType(), module,
                       rewriter, *typeConverter, "incref",
                       /*aggregateEffect=*/false, retainPremise);
        return loaded;
      };
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), list[0], kTypedListRefcountSlot, rewriter);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(
          op.getLoc(), mlir::TypeRange{itemsType.getElementType()}, isManaged,
          /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        container::Managed::lock(op.getLoc(), list[0], kTypedListLockSlot,
                                 rewriter);
        mlir::Value loaded =
            emitLoadAndRetain(ThreadSafetyAttrs::kPremiseLockedBorrow,
                              /*sharedAccess=*/true);
        container::Managed::unlock(op.getLoc(), list[0], kTypedListLockSlot,
                                   rewriter);
        rewriter.create<mlir::scf::YieldOp>(op.getLoc(), loaded);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        rewriter.create<mlir::scf::YieldOp>(
            op.getLoc(),
            emitLoadAndRetain(ThreadSafetyAttrs::kPremiseAggregateBorrow,
                              /*sharedAccess=*/false));
      }
      mlir::Value loaded = ifOp.getResult(0);
      mlir::FailureOr<mlir::Value> boxed =
          Slot::boxStorage(op.getLoc(), loaded, listType.getElementType(),
                           module, rewriter, *typeConverter);
      if (mlir::failed(boxed))
        return mlir::failure();
      rewriter.replaceOp(op, *boxed);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list get; runtime fallback disabled");
  }
};

} // namespace

namespace lowering::value::list::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ListNewLowering, ListAppendLowering, ListRemoveLowering,
               ListGetLowering>(typeConverter, ctx);
}
} // namespace lowering::value::list::Patterns

} // namespace py
