#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "PyValue/ClassHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#include <cstdint>
#include <optional>

namespace py {
namespace {

enum class StorageKind {
  Unsupported,
  TypedMemRefSlots,
};

namespace lowering::value::list::Storage {

StorageKind classify(ListType listType) {
  if (!listType)
    return StorageKind::Unsupported;
  mlir::Type elementType = listType.getElementType();
  if (container::Slot::supported(elementType) ||
      mlir::isa<ClassType>(elementType))
    return StorageKind::TypedMemRefSlots;
  return StorageKind::Unsupported;
}

bool memref(ListType listType) {
  return classify(listType) == StorageKind::TypedMemRefSlots;
}

} // namespace lowering::value::list::Storage

static constexpr int64_t kTypedListDefaultCapacity = 64;

namespace lowering::value::list::Abort {

void emit(mlir::Location loc, mlir::ModuleOp module,
          mlir::ConversionPatternRewriter &rewriter) {
  auto abortFn = getOrInsertLLVMFunc(
      loc, module, rewriter, "abort",
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {});
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{},
      mlir::SymbolRefAttr::get(module.getContext(), abortFn.getName()),
      mlir::ValueRange{});
  call->setAttr(ControlFlowContractAttrs::kNoReturn,
                mlir::UnitAttr::get(module.getContext()));
}

} // namespace lowering::value::list::Abort

namespace lowering::value::list::ClassSlot {

bool inlineStorage(ListType listType, mlir::Value items,
                   mlir::LLVM::LLVMStructType &objectType,
                   ClassType &classType) {
  classType = mlir::dyn_cast_or_null<ClassType>(listType.getElementType());
  if (!classType)
    return false;
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType)
    return false;
  objectType = class_layout::objectCarrierType(itemsType);
  return static_cast<bool>(objectType);
}

mlir::Value view(mlir::Location loc, mlir::Value items, mlir::Value index,
                 mlir::LLVM::LLVMStructType objectType,
                 mlir::OpBuilder &builder) {
  return Slot::classCarrierView(loc, items, index, objectType, builder);
}

mlir::Value carrierFromValues(mlir::Location loc, mlir::ValueRange values,
                              mlir::LLVM::LLVMStructType objectType,
                              mlir::OpBuilder &builder) {
  return Slot::classCarrierFromValues(loc, values, objectType, builder);
}

mlir::LogicalResult copyInto(mlir::Location loc, mlir::ModuleOp module,
                             ClassType classType,
                             mlir::LLVM::LLVMStructType objectType,
                             mlir::Value destPtr, mlir::ValueRange sourceValues,
                             mlir::ConversionPatternRewriter &rewriter,
                             const PyLLVMTypeConverter &typeConverter,
                             bool initializeDest = true) {
  if (mlir::failed(::py::lowering::value::class_::Copy::ensure(
          loc, module, classType, rewriter, typeConverter)))
    return mlir::failure();
  mlir::Value source =
      carrierFromValues(loc, sourceValues, objectType, rewriter);
  if (!source || !mlir::isa<mlir::MemRefType>(destPtr.getType()))
    return mlir::failure();
  if (initializeDest &&
      mlir::failed(Slot::classCarrierInitialize(loc, destPtr, classType, module,
                                                rewriter, typeConverter)))
    return mlir::failure();
  return Slot::classCarrierCopyTo(loc, destPtr, source, classType, module,
                                  rewriter);
}

mlir::LogicalResult copyIntoParts(mlir::Location loc, mlir::ModuleOp module,
                                  ClassType classType,
                                  mlir::LLVM::LLVMStructType objectType,
                                  mlir::ValueRange destValues,
                                  mlir::ValueRange sourceValues,
                                  mlir::ConversionPatternRewriter &rewriter,
                                  const PyLLVMTypeConverter &typeConverter) {
  if (mlir::failed(::py::lowering::value::class_::Copy::ensure(
          loc, module, classType, rewriter, typeConverter)))
    return mlir::failure();
  mlir::Value dest = carrierFromValues(loc, destValues, objectType, rewriter);
  mlir::Value source =
      carrierFromValues(loc, sourceValues, objectType, rewriter);
  if (!dest || !source)
    return mlir::failure();
  return Slot::classCarrierCopyObjectTo(loc, dest, source, classType, module,
                                        rewriter);
}

struct LocalCopySlot {
  mlir::Value header;
  llvm::SmallVector<mlir::Value, 8> payloadParts;
  mlir::Value view;
};

mlir::FailureOr<LocalCopySlot>
localCopySlot(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
              mlir::LLVM::LLVMStructType objectType,
              mlir::ConversionPatternRewriter &rewriter,
              const PyLLVMTypeConverter &typeConverter) {
  auto storageType = mlir::MemRefType::get({1}, objectType);
  mlir::Value storage =
      rewriter.create<mlir::memref::AllocaOp>(loc, storageType);
  mlir::Value view = Slot::classCarrierView(
      loc, storage, createIndexConstant(loc, rewriter, 0), objectType,
      rewriter);
  mlir::FailureOr<Slot::ClassCarrierParts> parts =
      Slot::classCarrierInitializeParts(loc, view, classType, module, rewriter,
                                        typeConverter,
                                        /*transferToSlot=*/false);
  if (mlir::failed(parts))
    return mlir::failure();
  return LocalCopySlot{parts->header, parts->payloadParts, view};
}

void refcount(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
              mlir::Value objectPtr, llvm::StringRef suffix,
              bool aggregateEffect, llvm::StringRef retainPremise,
              mlir::OpBuilder &builder) {
  if (mlir::isa<mlir::MemRefType>(objectPtr.getType())) {
    Slot::classCarrierRefcount(loc, objectPtr, classType, module, builder,
                               suffix, aggregateEffect, retainPremise);
    return;
  }
  Slot::classRefcount(loc, objectPtr, classType, module, builder, suffix,
                      aggregateEffect, retainPremise);
}

mlir::Value equal(mlir::Location loc, mlir::ModuleOp module,
                  ClassType classType, mlir::Value lhs,
                  mlir::LLVM::LLVMStructType objectType, mlir::Value rhs,
                  mlir::OpBuilder &builder) {
  lhs = carrierFromValues(loc, mlir::ValueRange{lhs}, objectType, builder);
  rhs = carrierFromValues(loc, mlir::ValueRange{rhs}, objectType, builder);
  if (!lhs || !rhs || rhs.getType() != lhs.getType())
    return {};
  return Slot::classCarrierEqual(loc, lhs, rhs, classType, module, builder);
}

mlir::Value stripCasts(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

bool hasUseAfter(mlir::Operation *anchor, mlir::Value value) {
  value = stripCasts(value);
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

void consumeSource(mlir::Location loc, mlir::ModuleOp module,
                   ClassType classType, mlir::LLVM::LLVMStructType objectType,
                   mlir::ValueRange loweredSource, mlir::Value logicalSource,
                   mlir::Operation *owner,
                   mlir::ConversionPatternRewriter &rewriter) {
  mlir::Value root = stripCasts(logicalSource);
  bool localSource = mlir::isa_and_nonnull<ClassNewOp>(root.getDefiningOp());
  bool canDestroyLocal = localSource && !hasUseAfter(owner, logicalSource) &&
                         loweredSource.size() == 1 &&
                         loweredSource.front().getType() == objectType;
  if (canDestroyLocal) {
    refcount(loc, module, classType, loweredSource.front(), "destroy_local",
             /*aggregateEffect=*/true, ThreadSafetyAttrs::kPremiseOwnedToken,
             rewriter);
    return;
  }

  mlir::Value source =
      carrierFromValues(loc, loweredSource, objectType, rewriter);
  if (!source)
    return;
  refcount(loc, module, classType, source, "decref",
           /*aggregateEffect=*/true, ThreadSafetyAttrs::kPremiseOwnedToken,
           rewriter);
}

} // namespace lowering::value::list::ClassSlot

namespace lowering::value::list::Local {

bool useBefore(mlir::Operation *user, mlir::Operation *anchor) {
  return user->getBlock() == anchor->getBlock() &&
         user->isBeforeInBlock(anchor);
}

bool safeUseBeforeAppend(mlir::Operation *user, ListAppendOp append) {
  if (user == append.getOperation())
    return true;
  if (user->getBlock() != append->getBlock())
    return false;
  if (!user->isBeforeInBlock(append.getOperation()))
    return true;
  return mlir::isa<ListAppendOp, ListGetOp, ListRemoveOp>(user);
}

bool freshBeforeEscape(ListAppendOp append) {
  mlir::Value list = append.getList();
  auto listNew = list.getDefiningOp<ListNewOp>();
  if (!listNew || !useBefore(listNew.getOperation(), append.getOperation()))
    return false;

  for (mlir::Operation *user : list.getUsers())
    if (!safeUseBeforeAppend(user, append))
      return false;
  return true;
}

} // namespace lowering::value::list::Local

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

    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.size() != 3)
      return mlir::failure();

    if (lowering::value::list::Storage::memref(listType)) {
      auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
      auto lockType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[2]);
      if (!headerType || !lockType || !itemsType)
        return mlir::failure();
      auto header =
          rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
      auto lock =
          rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), lockType);
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
          lock.getResult(), descriptorGroup,
          ContainerSafetyAttrs::kComponentLock);
      container::descriptor::Component::mark(
          items.getResult(), descriptorGroup,
          ContainerSafetyAttrs::kComponentItems);
      mlir::Value zeroIndex = createIndexConstant(op.getLoc(), rewriter, 0);
      mlir::Value oneIndex = createIndexConstant(op.getLoc(), rewriter, 1);
      mlir::Value twoIndex = createIndexConstant(op.getLoc(), rewriter, 2);
      mlir::Value zero = createI64Constant(op.getLoc(), rewriter, 0);
      mlir::Value capacity =
          createI64Constant(op.getLoc(), rewriter, kTypedListDefaultCapacity);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), zero, header,
                                             zeroIndex);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), capacity, header,
                                             oneIndex);
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), zero, header,
                                             twoIndex);
      auto lockZero = rewriter.create<mlir::arith::ConstantIntOp>(
          op.getLoc(), 0, rewriter.getI32Type());
      rewriter.create<mlir::memref::StoreOp>(op.getLoc(), lockZero, lock,
                                             zeroIndex);
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{
                  header.getResult(), lock.getResult(), items.getResult()}});
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list element type; typed memref lowering "
            "required");
  }
};

namespace lowering::value::list::Index {

std::optional<int64_t> i64Literal(mlir::Value originalIndex) {
  auto literal = originalIndex ? originalIndex.getDefiningOp<IntConstantOp>()
                               : IntConstantOp();
  if (!literal)
    return std::nullopt;
  int64_t value = 0;
  if (literal.getValue().getAsInteger(10, value))
    return std::nullopt;
  return value;
}

mlir::Value normalize(mlir::Location loc, mlir::Value index,
                      mlir::Value originalIndex, mlir::Type originalType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  auto i64Type = rewriter.getI64Type();
  if (index.getType() == i64Type)
    return index;
  if (mlir::isa<IntType>(originalType)) {
    if (std::optional<int64_t> literal = i64Literal(originalIndex))
      return rewriter.create<mlir::arith::ConstantIntOp>(loc, *literal, 64);
    return mlir::Value();
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

mlir::Value normalize(mlir::Location loc, mlir::ValueRange index,
                      mlir::Value originalIndex, mlir::Type originalType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  if (index.size() == 1)
    return normalize(loc, index.front(), originalIndex, originalType, module,
                     rewriter, typeConverter);
  if (mlir::isa<IntType>(originalType) && index.size() == 3) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(), index)
        .getResult();
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
      if (list.size() != 3)
        return mlir::failure();
      mlir::Value header = list[kListHeaderComponent];
      mlir::Value lock = list[kListLockComponent];
      mlir::Value items = list[kListItemsComponent];
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
      if (!itemsType)
        return mlir::failure();
      mlir::LLVM::LLVMStructType inlineCarrierType;
      ClassType inlineClassType;
      bool inlineClass = lowering::value::list::ClassSlot::inlineStorage(
          listType, items, inlineCarrierType, inlineClassType);
      mlir::Value stored =
          inlineClass
              ? mlir::Value()
              : Slot::ownedContainerStorage(op.getLoc(), adaptor.getValue(),
                                            listType.getElementType(),
                                            itemsType.getElementType(), module,
                                            rewriter, *typeConverter);
      if (!inlineClass && !stored)
        return mlir::failure();
      bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
      llvm::StringRef valueRetainPremise =
          isEntryBorrowedValue(adaptor.getValue().front())
              ? ThreadSafetyAttrs::kPremiseEntryBorrowed
              : ThreadSafetyAttrs::kPremiseOwnedToken;
      auto emitAppendBody = [&](bool sharedAccess) -> mlir::LogicalResult {
        auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
          if (sharedAccess)
            container::access::Contract::mark(access, header, target);
        };
        mlir::Value sizeIndex =
            createIndexConstant(op.getLoc(), rewriter, kTypedListSizeSlot);
        mlir::Value capacityIndex =
            createIndexConstant(op.getLoc(), rewriter, 1);
        auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), header, sizeIndex);
        markAccess(sizeLoad.getOperation(), header);
        mlir::Value sizeValue = sizeLoad;
        auto capacityLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), header, capacityIndex);
        markAccess(capacityLoad.getOperation(), header);
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
          if (!consumeValue && !inlineClass)
            if (mlir::failed(Slot::refcount(
                    op.getLoc(), stored, listType.getElementType(), module,
                    rewriter, *typeConverter, "incref",
                    /*aggregateEffect=*/false, valueRetainPremise)))
              return mlir::failure();
          mlir::Value sizeAsIndex = rewriter.create<mlir::arith::IndexCastOp>(
              op.getLoc(), rewriter.getIndexType(), sizeValue);
          if (inlineClass) {
            mlir::Value destSlot = lowering::value::list::ClassSlot::view(
                op.getLoc(), items, sizeAsIndex, inlineCarrierType, rewriter);
            if (!destSlot)
              return mlir::failure();
            if (mlir::failed(lowering::value::list::ClassSlot::copyInto(
                    op.getLoc(), module, inlineClassType, inlineCarrierType,
                    destSlot, adaptor.getValue(), rewriter, *typeConverter)))
              return mlir::failure();
          } else {
            auto store = rewriter.create<mlir::memref::StoreOp>(
                op.getLoc(), stored, items, sizeAsIndex);
            markAccess(store.getOperation(), items);
            Slot::markTransfer(store.getOperation());
          }
          mlir::Value nextSize = rewriter.create<mlir::arith::AddIOp>(
              op.getLoc(), sizeValue,
              createI64Constant(op.getLoc(), rewriter, 1));
          auto sizeStore = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), nextSize, header, sizeIndex);
          markAccess(sizeStore.getOperation(), header);
        }
        {
          mlir::OpBuilder::InsertionGuard boundsGuard(rewriter);
          rewriter.setInsertionPointToStart(boundsIf.elseBlock());
          if (!inlineClass && consumeValue)
            if (mlir::failed(Slot::refcount(
                    op.getLoc(), stored, listType.getElementType(), module,
                    rewriter, *typeConverter, "decref")))
              return mlir::failure();
          lowering::value::list::Abort::emit(op.getLoc(), module, rewriter);
        }
        return mlir::success();
      };
      if (lowering::value::list::Local::freshBeforeEscape(op)) {
        if (mlir::failed(emitAppendBody(/*sharedAccess=*/false)))
          return mlir::failure();
      } else {
        mlir::Value isManaged = container::Managed::predicate(
            op.getLoc(), header, kTypedListRefcountSlot, rewriter);
        auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                     /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          container::Managed::lock(op.getLoc(), header, lock, rewriter);
          if (mlir::failed(emitAppendBody(/*sharedAccess=*/true)))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), header, lock, rewriter);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.elseBlock());
          if (mlir::failed(emitAppendBody(/*sharedAccess=*/false)))
            return mlir::failure();
        }
      }
      if (consumeValue) {
        if (inlineClass) {
          lowering::value::list::ClassSlot::consumeSource(
              op.getLoc(), module, inlineClassType, inlineCarrierType,
              adaptor.getValue(), op.getValue(), op.getOperation(), rewriter);
        } else {
          Slot::releaseSource(op.getLoc(), adaptor.getValue().front(),
                              listType.getElementType(), module, rewriter,
                              *typeConverter);
        }
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list append; typed memref lowering required");
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
      if (list.size() != 3)
        return mlir::failure();
      mlir::Value header = list[kListHeaderComponent];
      mlir::Value lock = list[kListLockComponent];
      mlir::Value items = list[kListItemsComponent];
      auto *typeConverter =
          static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
      if (!itemsType)
        return mlir::failure();
      mlir::LLVM::LLVMStructType inlineCarrierType;
      ClassType inlineClassType;
      bool inlineClass = lowering::value::list::ClassSlot::inlineStorage(
          listType, items, inlineCarrierType, inlineClassType);
      if (inlineClass &&
          mlir::failed(::py::lowering::value::class_::Eq::ensure(
              op.getLoc(), module, inlineClassType, rewriter, *typeConverter)))
        return mlir::failure();
      mlir::Value target =
          inlineClass ? lowering::value::list::ClassSlot::carrierFromValues(
                            op.getLoc(), adaptor.getValue(), inlineCarrierType,
                            rewriter)
                      : Slot::storage(op.getLoc(), adaptor.getValue(),
                                      listType.getElementType(),
                                      itemsType.getElementType(), module,
                                      rewriter, *typeConverter);
      if (!target)
        return mlir::failure();

      auto emitRemoveBody = [&](bool sharedAccess) -> mlir::LogicalResult {
        auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
          if (sharedAccess)
            container::access::Contract::mark(access, header, target);
        };
        mlir::Value sizeIndex =
            createIndexConstant(op.getLoc(), rewriter, kTypedListSizeSlot);
        auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), header, sizeIndex);
        markAccess(sizeLoad.getOperation(), header);
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
          mlir::Value same;
          if (inlineClass) {
            mlir::Value itemSlot = lowering::value::list::ClassSlot::view(
                op.getLoc(), items, iv, inlineCarrierType, rewriter);
            if (!itemSlot)
              return mlir::failure();
            same = lowering::value::list::ClassSlot::equal(
                op.getLoc(), module, inlineClassType, itemSlot,
                inlineCarrierType, target, rewriter);
          } else {
            auto itemLoad =
                rewriter.create<mlir::memref::LoadOp>(op.getLoc(), items, iv);
            markAccess(itemLoad.getOperation(), items);
            mlir::Value item = itemLoad;
            same = rewriter.create<mlir::arith::CmpIOp>(
                op.getLoc(), mlir::arith::CmpIPredicate::eq, item, target);
          }
          if (!same)
            return mlir::failure();
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
          if (inlineClass) {
            mlir::Value removedSlot = lowering::value::list::ClassSlot::view(
                op.getLoc(), items, findLoop.getResult(1), inlineCarrierType,
                rewriter);
            if (!removedSlot)
              return mlir::failure();
            lowering::value::list::ClassSlot::refcount(
                op.getLoc(), module, inlineClassType, removedSlot, "decref",
                /*aggregateEffect=*/true,
                ThreadSafetyAttrs::kPremiseAggregateBorrow, rewriter);
          } else if (Slot::refcounted(listType.getElementType())) {
            auto removedLoad = rewriter.create<mlir::memref::LoadOp>(
                op.getLoc(), items, findLoop.getResult(1));
            markAccess(removedLoad.getOperation(), items);
            mlir::Value removed = removedLoad;
            if (mlir::failed(Slot::refcount(op.getLoc(), removed,
                                            listType.getElementType(), module,
                                            rewriter, *typeConverter, "decref",
                                            /*aggregateEffect=*/true)))
              return mlir::failure();
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
            if (inlineClass) {
              mlir::Value sourceSlot = lowering::value::list::ClassSlot::view(
                  op.getLoc(), items, iv, inlineCarrierType, rewriter);
              mlir::Value destSlot = lowering::value::list::ClassSlot::view(
                  op.getLoc(), items, dstIndex, inlineCarrierType, rewriter);
              if (!sourceSlot || !destSlot)
                return mlir::failure();
              if (mlir::failed(lowering::value::list::ClassSlot::copyInto(
                      op.getLoc(), module, inlineClassType, inlineCarrierType,
                      destSlot, mlir::ValueRange{sourceSlot}, rewriter,
                      *typeConverter)))
                return mlir::failure();
              lowering::value::list::ClassSlot::refcount(
                  op.getLoc(), module, inlineClassType, sourceSlot, "decref",
                  /*aggregateEffect=*/true,
                  ThreadSafetyAttrs::kPremiseAggregateBorrow, rewriter);
              auto zero = rewriter.create<mlir::LLVM::ZeroOp>(
                  op.getLoc(), inlineCarrierType);
              auto clearStore = rewriter.create<mlir::memref::StoreOp>(
                  op.getLoc(), zero, sourceSlot,
                  createIndexConstant(op.getLoc(), rewriter, 0));
              markAccess(clearStore.getOperation(), items);
              Slot::markTransfer(clearStore.getOperation());
            } else {
              auto itemLoad =
                  rewriter.create<mlir::memref::LoadOp>(op.getLoc(), items, iv);
              markAccess(itemLoad.getOperation(), items);
              mlir::Value item = itemLoad;
              auto itemStore = rewriter.create<mlir::memref::StoreOp>(
                  op.getLoc(), item, items, dstIndex);
              markAccess(itemStore.getOperation(), items);
              if (Slot::refcounted(listType.getElementType()))
                Slot::markTransfer(itemStore.getOperation());
            }
          }
          mlir::Value nextSize = rewriter.create<mlir::arith::SubIOp>(
              op.getLoc(), sizeValue,
              createI64Constant(op.getLoc(), rewriter, 1));
          auto sizeStore = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), nextSize, header, sizeIndex);
          markAccess(sizeStore.getOperation(), header);
        }
        return mlir::success();
      };
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), header, kTypedListRefcountSlot, rewriter);
      auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                   /*withElseRegion=*/true);
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        container::Managed::lock(op.getLoc(), header, lock, rewriter);
        if (mlir::failed(emitRemoveBody(/*sharedAccess=*/true)))
          return mlir::failure();
        container::Managed::unlock(op.getLoc(), header, lock, rewriter);
      }
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.elseBlock());
        if (mlir::failed(emitRemoveBody(/*sharedAccess=*/false)))
          return mlir::failure();
      }
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list remove; typed memref lowering required");
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
      if (list.size() != 3)
        return mlir::failure();
      mlir::Value header = list[kListHeaderComponent];
      mlir::Value lock = list[kListLockComponent];
      mlir::Value items = list[kListItemsComponent];
      mlir::Value index = lowering::value::list::Index::normalize(
          op.getLoc(), adaptor.getIndex(), op.getIndex(),
          op.getIndex().getType(), module, rewriter, *typeConverter);
      if (!index)
        return mlir::failure();
      mlir::Value indexAsIndex = rewriter.create<mlir::arith::IndexCastOp>(
          op.getLoc(), rewriter.getIndexType(), index);
      auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
      if (!itemsType)
        return mlir::failure();
      mlir::LLVM::LLVMStructType inlineCarrierType;
      ClassType inlineClassType;
      bool inlineClass = lowering::value::list::ClassSlot::inlineStorage(
          listType, items, inlineCarrierType, inlineClassType);
      lowering::value::list::ClassSlot::LocalCopySlot inlineClassResult;
      if (inlineClass) {
        mlir::FailureOr<lowering::value::list::ClassSlot::LocalCopySlot>
            localCopy = lowering::value::list::ClassSlot::localCopySlot(
                op.getLoc(), module, inlineClassType, inlineCarrierType,
                rewriter, *typeConverter);
        if (mlir::failed(localCopy))
          return mlir::failure();
        inlineClassResult = *localCopy;
        if (!inlineClassResult.header ||
            inlineClassResult.payloadParts.empty() || !inlineClassResult.view)
          return mlir::failure();
      }
      auto emitLoadAndRetain =
          [&](llvm::StringRef retainPremise,
              bool sharedAccess) -> mlir::FailureOr<mlir::Value> {
        if (inlineClass) {
          mlir::Value objectSlot = lowering::value::list::ClassSlot::view(
              op.getLoc(), items, index, inlineCarrierType, rewriter);
          if (!objectSlot)
            return mlir::failure();
          llvm::SmallVector<mlir::Value, 8> destValues{
              inlineClassResult.header};
          destValues.append(inlineClassResult.payloadParts.begin(),
                            inlineClassResult.payloadParts.end());
          if (mlir::failed(lowering::value::list::ClassSlot::copyIntoParts(
                  op.getLoc(), module, inlineClassType, inlineCarrierType,
                  mlir::ValueRange(destValues), mlir::ValueRange{objectSlot},
                  rewriter, *typeConverter)))
            return mlir::failure();
          return inlineClassResult.view;
        }
        auto load = rewriter.create<mlir::memref::LoadOp>(op.getLoc(), items,
                                                          indexAsIndex);
        if (sharedAccess)
          container::access::Contract::mark(load.getOperation(), header, items);
        else
          ownership::aggregate::Slot::markLoad(load.getResult());
        mlir::Value loaded = load;
        if (mlir::failed(
                Slot::refcount(op.getLoc(), loaded, listType.getElementType(),
                               module, rewriter, *typeConverter, "incref",
                               /*aggregateEffect=*/false, retainPremise)))
          return mlir::failure();
        return loaded;
      };
      mlir::Value isManaged = container::Managed::predicate(
          op.getLoc(), header, kTypedListRefcountSlot, rewriter);
      mlir::Value loaded;
      if (inlineClass) {
        auto ifOp = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                     /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          container::Managed::lock(op.getLoc(), header, lock, rewriter);
          if (mlir::failed(
                  emitLoadAndRetain(ThreadSafetyAttrs::kPremiseLockedBorrow,
                                    /*sharedAccess=*/true)))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), header, lock, rewriter);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.elseBlock());
          if (mlir::failed(
                  emitLoadAndRetain(ThreadSafetyAttrs::kPremiseAggregateBorrow,
                                    /*sharedAccess=*/false)))
            return mlir::failure();
        }
        loaded = inlineClassResult.view;
      } else {
        auto ifOp = rewriter.create<mlir::scf::IfOp>(
            op.getLoc(), mlir::TypeRange{itemsType.getElementType()}, isManaged,
            /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.thenBlock());
          container::Managed::lock(op.getLoc(), header, lock, rewriter);
          mlir::FailureOr<mlir::Value> loaded =
              emitLoadAndRetain(ThreadSafetyAttrs::kPremiseLockedBorrow,
                                /*sharedAccess=*/true);
          if (mlir::failed(loaded))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), header, lock, rewriter);
          rewriter.create<mlir::scf::YieldOp>(op.getLoc(), *loaded);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(ifOp.elseBlock());
          mlir::FailureOr<mlir::Value> loaded =
              emitLoadAndRetain(ThreadSafetyAttrs::kPremiseAggregateBorrow,
                                /*sharedAccess=*/false);
          if (mlir::failed(loaded))
            return mlir::failure();
          rewriter.create<mlir::scf::YieldOp>(op.getLoc(), *loaded);
        }
        loaded = ifOp.getResult(0);
      }
      if (inlineClass) {
        llvm::SmallVector<mlir::Value, 8> replacements{
            inlineClassResult.header};
        replacements.append(inlineClassResult.payloadParts.begin(),
                            inlineClassResult.payloadParts.end());
        rewriter.replaceOpWithMultiple(op, llvm::ArrayRef<mlir::ValueRange>{
                                               mlir::ValueRange(replacements)});
        return mlir::success();
      }
      if (mlir::failed(Slot::replaceBoxedStorage(
              op.getOperation(), loaded, listType.getElementType(), module,
              rewriter, *typeConverter)))
        return mlir::failure();
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed list get; typed memref lowering required");
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
