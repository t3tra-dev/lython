#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

enum class StorageKind {
  Unsupported,
  TypedMemRefs,
};

static constexpr int64_t kTypedDictDefaultCapacity = 64;

namespace lowering::value::dict::Abort {

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

} // namespace lowering::value::dict::Abort

namespace lowering::value::dict::KeyError {

void throw_(mlir::Location loc, mlir::ModuleOp module,
            mlir::ConversionPatternRewriter &rewriter,
            const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  llvm::SmallVector<mlir::Type> resultTypes;
  object_abi::exception_abi::Parts::storageTypes(module.getContext(),
                                                 resultTypes);
  llvm::SmallVector<mlir::Type, 2> unicodeTypes;
  object_abi::str_abi::Parts::storageTypes(module.getContext(), unicodeTypes);
  mlir::Value bytes = runtime.getByteLiteral(loc, rewriter.getStringAttr(""));
  mlir::Value start = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value length = runtime.getI64Constant(loc, 0);
  auto message = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                              mlir::TypeRange(unicodeTypes),
                              mlir::ValueRange{bytes, start, length});
  auto header =
      runtime.call(loc, RuntimeSymbols::kExceptionNew,
                   mlir::TypeRange{resultTypes.front()}, mlir::ValueRange{});
  llvm::SmallVector<mlir::Value, 3> parts;
  parts.push_back(header.getResult());
  parts.append(message.getResults().begin(), message.getResults().end());
  runtime.call(loc, RuntimeSymbols::kEHThrowException, mlir::Type(), parts);
}

} // namespace lowering::value::dict::KeyError

namespace lowering::value::dict::Probe {

mlir::Value slot(mlir::Location loc, mlir::OpBuilder &builder,
                 mlir::Value keySlot, mlir::Value probeIndex) {
  mlir::Value keyBits = keySlot;
  if (keyBits.getType() == builder.getF64Type())
    keyBits = builder.create<mlir::arith::BitcastOp>(loc, builder.getI64Type(),
                                                     keyBits);
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(keyBits.getType())) {
    if (intType.getWidth() < 64)
      keyBits = builder.create<mlir::arith::ExtUIOp>(loc, builder.getI64Type(),
                                                     keyBits);
    else if (intType.getWidth() > 64)
      keyBits = builder.create<mlir::arith::TruncIOp>(loc, builder.getI64Type(),
                                                      keyBits);
  }
  if (keyBits.getType() != builder.getI64Type())
    return {};
  mlir::Value probeI64 = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getI64Type(), probeIndex);
  mlir::Value capacity =
      createI64Constant(loc, builder, kTypedDictDefaultCapacity);
  mlir::Value mixed =
      builder.create<mlir::arith::AddIOp>(loc, keyBits, probeI64);
  mlir::Value slotI64 =
      builder.create<mlir::arith::RemUIOp>(loc, mixed, capacity);
  return builder.create<mlir::arith::IndexCastOp>(loc, builder.getIndexType(),
                                                  slotI64);
}

} // namespace lowering::value::dict::Probe

namespace lowering::value::dict::Storage {

StorageKind classify(DictType dictType) {
  if (!dictType)
    return StorageKind::Unsupported;
  if (!container::Slot::supported(dictType.getKeyType()) ||
      !container::Slot::supported(dictType.getValueType()))
    return StorageKind::Unsupported;
  return StorageKind::TypedMemRefs;
}

mlir::Value equal(mlir::Location loc, mlir::Value lhs, mlir::Value rhs,
                  mlir::OpBuilder &builder) {
  if (lhs.getType() != rhs.getType())
    return {};
  if (mlir::isa<mlir::FloatType>(lhs.getType()))
    return builder.create<mlir::arith::CmpFOp>(
        loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
  if (mlir::isa<mlir::IntegerType>(lhs.getType()))
    return builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, lhs, rhs);
  return {};
}

mlir::Value zero(mlir::Location loc, mlir::Type storageType,
                 mlir::OpBuilder &builder) {
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(storageType))
    return builder.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
  if (storageType == builder.getF64Type())
    return builder.create<mlir::arith::ConstantFloatOp>(loc, llvm::APFloat(0.0),
                                                        builder.getF64Type());
  return {};
}

} // namespace lowering::value::dict::Storage

struct DictEmptyLowering : public mlir::OpConversionPattern<DictEmptyOp> {
  DictEmptyLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DictEmptyOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(DictEmptyOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = mlir::dyn_cast<DictType>(op.getResult().getType())) {
      if (lowering::value::dict::Storage::classify(dictType) ==
          StorageKind::TypedMemRefs) {
        llvm::SmallVector<mlir::Type, 5> resultTypes;
        if (mlir::failed(converter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
            resultTypes.size() != 5)
          return mlir::failure();
        auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
        auto lockType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[1]);
        auto keysType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[2]);
        auto valuesType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[3]);
        auto statesType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[4]);
        if (!headerType || !lockType || !keysType || !valuesType || !statesType)
          return mlir::failure();

        auto header =
            rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), headerType);
        auto lock =
            rewriter.create<mlir::memref::AllocaOp>(op.getLoc(), lockType);
        mlir::Value capacityIndex = createIndexConstant(
            op.getLoc(), rewriter, kTypedDictDefaultCapacity);
        auto keys = rewriter.create<mlir::memref::AllocaOp>(
            op.getLoc(), keysType, mlir::ValueRange{capacityIndex});
        auto values = rewriter.create<mlir::memref::AllocaOp>(
            op.getLoc(), valuesType, mlir::ValueRange{capacityIndex});
        auto states = rewriter.create<mlir::memref::AllocaOp>(
            op.getLoc(), statesType, mlir::ValueRange{capacityIndex});
        std::string descriptorGroup =
            container::descriptor::Group::make(op.getOperation(), "dict");
        container::descriptor::Component::mark(
            header.getResult(), descriptorGroup,
            ContainerSafetyAttrs::kComponentHeader);
        container::descriptor::Component::mark(
            lock.getResult(), descriptorGroup,
            ContainerSafetyAttrs::kComponentLock);
        container::descriptor::Component::mark(
            keys.getResult(), descriptorGroup,
            ContainerSafetyAttrs::kComponentKeys);
        container::descriptor::Component::mark(
            values.getResult(), descriptorGroup,
            ContainerSafetyAttrs::kComponentValues);
        container::descriptor::Component::mark(
            states.getResult(), descriptorGroup,
            ContainerSafetyAttrs::kComponentStates);

        mlir::Value zero = createI64Constant(op.getLoc(), rewriter, 0);
        mlir::Value capacity =
            createI64Constant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
        for (int64_t slot = 0; slot < kDictHeaderSize; ++slot) {
          mlir::Value value = slot == 1 ? capacity : zero;
          rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), value, header,
              createIndexConstant(op.getLoc(), rewriter, slot));
        }
        auto lockZero = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), 0, rewriter.getI32Type());
        rewriter.create<mlir::memref::StoreOp>(
            op.getLoc(), lockZero, lock,
            createIndexConstant(op.getLoc(), rewriter,
                                kTypedContainerLockSlot));
        mlir::Value zeroState = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), 0, rewriter.getI8Type());
        for (int64_t slot = 0; slot < kTypedDictDefaultCapacity; ++slot) {
          rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), zeroState, states,
              createIndexConstant(op.getLoc(), rewriter, slot));
        }
        rewriter.replaceOpWithMultiple(
            op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange{
                    header.getResult(), lock.getResult(), keys.getResult(),
                    values.getResult(), states.getResult()}});
        return mlir::success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed dict key/value type; typed memref lowering "
            "required");
  }
};

struct DictInsertLowering : public mlir::OpConversionPattern<DictInsertOp> {
  DictInsertLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DictInsertOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(DictInsertOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = mlir::dyn_cast<DictType>(op.getDict().getType())) {
      if (lowering::value::dict::Storage::classify(dictType) ==
          StorageKind::TypedMemRefs) {
        mlir::ValueRange dict = adaptor.getDict();
        if (dict.size() != 5 || adaptor.getKey().empty() ||
            adaptor.getValue().empty())
          return mlir::failure();
        mlir::Value header = dict[kDictHeaderComponent];
        mlir::Value lock = dict[kDictLockComponent];
        mlir::Value keys = dict[kDictKeysComponent];
        mlir::Value values = dict[kDictValuesComponent];
        mlir::Value states = dict[kDictStatesComponent];
        auto keysType = mlir::dyn_cast<mlir::MemRefType>(keys.getType());
        auto valuesType = mlir::dyn_cast<mlir::MemRefType>(values.getType());
        if (!keysType || !valuesType)
          return mlir::failure();
        mlir::Value key = Slot::storage(
            op.getLoc(), adaptor.getKey(), dictType.getKeyType(),
            keysType.getElementType(), module, rewriter, *converter);
        mlir::Value value = Slot::storage(
            op.getLoc(), adaptor.getValue(), dictType.getValueType(),
            valuesType.getElementType(), module, rewriter, *converter);
        if (!key || !value)
          return mlir::failure();
        bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));

        auto emitInsertBody = [&](bool sharedAccess) -> mlir::LogicalResult {
          auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
            if (sharedAccess)
              container::access::Contract::mark(access, header, target);
          };
          mlir::Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
          mlir::Value upper = createIndexConstant(op.getLoc(), rewriter,
                                                  kTypedDictDefaultCapacity);
          mlir::Value step = createIndexConstant(op.getLoc(), rewriter, 1);
          mlir::Value doneInit = rewriter.create<mlir::arith::ConstantIntOp>(
              op.getLoc(), false, rewriter.getI1Type());
          mlir::Value insertedNewInit =
              rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), false,
                                                          rewriter.getI1Type());

          auto loop = rewriter.create<mlir::scf::ForOp>(
              op.getLoc(), lower, upper, step,
              mlir::ValueRange{doneInit, insertedNewInit});
          {
            mlir::OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(loop.getBody());
            mlir::Value iv = loop.getInductionVar();
            mlir::Value done = loop.getRegionIterArgs()[0];
            mlir::Value insertedNew = loop.getRegionIterArgs()[1];
            mlir::Value probeSlot = lowering::value::dict::Probe::slot(
                op.getLoc(), rewriter, key, iv);
            auto stateLoad = rewriter.create<mlir::memref::LoadOp>(
                op.getLoc(), states, probeSlot);
            markAccess(stateLoad.getOperation(), states);
            mlir::Value state = stateLoad;
            auto keyLoad = rewriter.create<mlir::memref::LoadOp>(
                op.getLoc(), keys, probeSlot);
            markAccess(keyLoad.getOperation(), keys);
            mlir::Value keyAt = keyLoad;
            mlir::Value zeroState = rewriter.create<mlir::arith::ConstantIntOp>(
                op.getLoc(), 0, rewriter.getI8Type());
            mlir::Value occupiedState =
                rewriter.create<mlir::arith::ConstantIntOp>(
                    op.getLoc(), 1, rewriter.getI8Type());
            mlir::Value empty = rewriter.create<mlir::arith::CmpIOp>(
                op.getLoc(), mlir::arith::CmpIPredicate::eq, state, zeroState);
            mlir::Value occupied = rewriter.create<mlir::arith::CmpIOp>(
                op.getLoc(), mlir::arith::CmpIPredicate::eq, state,
                occupiedState);
            mlir::Value sameKey = lowering::value::dict::Storage::equal(
                op.getLoc(), keyAt, key, rewriter);
            mlir::Value updateExisting = rewriter.create<mlir::arith::AndIOp>(
                op.getLoc(), occupied, sameKey);
            mlir::Value writable = rewriter.create<mlir::arith::OrIOp>(
                op.getLoc(), empty, updateExisting);
            mlir::Value notDone = rewriter.create<mlir::arith::XOrIOp>(
                op.getLoc(), done,
                rewriter.create<mlir::arith::ConstantIntOp>(
                    op.getLoc(), true, rewriter.getI1Type()));
            mlir::Value shouldWrite = rewriter.create<mlir::arith::AndIOp>(
                op.getLoc(), notDone, writable);

            auto ifOp =
                rewriter.create<mlir::scf::IfOp>(op.getLoc(), shouldWrite,
                                                 /*withElseRegion=*/false);
            {
              mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
              rewriter.setInsertionPointToStart(ifOp.thenBlock());
              if (Slot::refcounted(dictType.getKeyType())) {
                if (consumeValue) {
                  auto releaseUnusedKey = rewriter.create<mlir::scf::IfOp>(
                      op.getLoc(), updateExisting,
                      /*withElseRegion=*/false);
                  mlir::OpBuilder::InsertionGuard releaseGuard(rewriter);
                  rewriter.setInsertionPointToStart(
                      releaseUnusedKey.thenBlock());
                  if (mlir::failed(Slot::refcount(
                          op.getLoc(), key, dictType.getKeyType(), module,
                          rewriter, *converter, "decref")))
                    return mlir::failure();
                }
              }
              bool valueRefcounted = Slot::refcounted(dictType.getValueType());
              if (valueRefcounted) {
                if (!consumeValue)
                  if (mlir::failed(Slot::refcount(
                          op.getLoc(), value, dictType.getValueType(), module,
                          rewriter, *converter, "incref")))
                    return mlir::failure();
                auto releaseOld = rewriter.create<mlir::scf::IfOp>(
                    op.getLoc(), updateExisting,
                    /*withElseRegion=*/false);
                mlir::OpBuilder::InsertionGuard releaseGuard(rewriter);
                rewriter.setInsertionPointToStart(releaseOld.thenBlock());
                auto oldValueLoad = rewriter.create<mlir::memref::LoadOp>(
                    op.getLoc(), values, probeSlot);
                markAccess(oldValueLoad.getOperation(), values);
                mlir::Value oldValue = oldValueLoad;
                if (mlir::failed(Slot::refcount(op.getLoc(), oldValue,
                                                dictType.getValueType(), module,
                                                rewriter, *converter, "decref",
                                                /*aggregateEffect=*/true)))
                  return mlir::failure();
              }
              auto storeKey =
                  rewriter.create<mlir::scf::IfOp>(op.getLoc(), empty,
                                                   /*withElseRegion=*/false);
              {
                mlir::OpBuilder::InsertionGuard storeKeyGuard(rewriter);
                rewriter.setInsertionPointToStart(storeKey.thenBlock());
                bool keyRefcounted = Slot::refcounted(dictType.getKeyType());
                if (keyRefcounted && !consumeValue)
                  if (mlir::failed(Slot::refcount(
                          op.getLoc(), key, dictType.getKeyType(), module,
                          rewriter, *converter, "incref")))
                    return mlir::failure();
                auto keyStore = rewriter.create<mlir::memref::StoreOp>(
                    op.getLoc(), key, keys, probeSlot);
                markAccess(keyStore.getOperation(), keys);
                if (keyRefcounted)
                  Slot::markTransfer(keyStore.getOperation());
              }
              auto valueStore = rewriter.create<mlir::memref::StoreOp>(
                  op.getLoc(), value, values, probeSlot);
              markAccess(valueStore.getOperation(), values);
              if (valueRefcounted)
                Slot::markTransfer(valueStore.getOperation());
              auto stateStore = rewriter.create<mlir::memref::StoreOp>(
                  op.getLoc(), occupiedState, states, probeSlot);
              markAccess(stateStore.getOperation(), states);
            }

            mlir::Value nextDone = rewriter.create<mlir::arith::OrIOp>(
                op.getLoc(), done, shouldWrite);
            mlir::Value newEntry = rewriter.create<mlir::arith::AndIOp>(
                op.getLoc(), shouldWrite, empty);
            mlir::Value nextInsertedNew = rewriter.create<mlir::arith::OrIOp>(
                op.getLoc(), insertedNew, newEntry);
            rewriter.create<mlir::scf::YieldOp>(
                op.getLoc(), mlir::ValueRange{nextDone, nextInsertedNew});
          }

          mlir::Value inserted = loop.getResult(0);
          mlir::Value notInserted = rewriter.create<mlir::arith::XOrIOp>(
              op.getLoc(), inserted,
              rewriter.create<mlir::arith::ConstantIntOp>(
                  op.getLoc(), true, rewriter.getI1Type()));
          auto fullIf =
              rewriter.create<mlir::scf::IfOp>(op.getLoc(), notInserted,
                                               /*withElseRegion=*/false);
          {
            mlir::OpBuilder::InsertionGuard fullGuard(rewriter);
            rewriter.setInsertionPointToStart(fullIf.thenBlock());
            if (consumeValue) {
              if (mlir::failed(Slot::refcount(op.getLoc(), key,
                                              dictType.getKeyType(), module,
                                              rewriter, *converter, "decref")))
                return mlir::failure();
              if (mlir::failed(Slot::refcount(op.getLoc(), value,
                                              dictType.getValueType(), module,
                                              rewriter, *converter, "decref")))
                return mlir::failure();
            }
            lowering::value::dict::Abort::emit(op.getLoc(), module, rewriter);
          }

          mlir::Value sizeIndex =
              createIndexConstant(op.getLoc(), rewriter, kTypedDictSizeSlot);
          auto sizeLoad = rewriter.create<mlir::memref::LoadOp>(
              op.getLoc(), header, sizeIndex);
          markAccess(sizeLoad.getOperation(), header);
          mlir::Value size = sizeLoad;
          mlir::Value one = createI64Constant(op.getLoc(), rewriter, 1);
          mlir::Value incremented =
              rewriter.create<mlir::arith::AddIOp>(op.getLoc(), size, one);
          mlir::Value nextSize = rewriter.create<mlir::arith::SelectOp>(
              op.getLoc(), loop.getResult(1), incremented, size);
          auto sizeStore = rewriter.create<mlir::memref::StoreOp>(
              op.getLoc(), nextSize, header, sizeIndex);
          markAccess(sizeStore.getOperation(), header);
          return mlir::success();
        };

        mlir::Value isManaged = container::Managed::predicate(
            op.getLoc(), header, kTypedDictRefcountSlot, rewriter);
        auto lockIf = rewriter.create<mlir::scf::IfOp>(op.getLoc(), isManaged,
                                                       /*withElseRegion=*/true);
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockIf.thenBlock());
          container::Managed::lock(op.getLoc(), lock, rewriter);
          if (mlir::failed(emitInsertBody(/*sharedAccess=*/true)))
            return mlir::failure();
          container::Managed::unlock(op.getLoc(), lock, rewriter);
        }
        {
          mlir::OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(lockIf.elseBlock());
          if (mlir::failed(emitInsertBody(/*sharedAccess=*/false)))
            return mlir::failure();
        }
        if (consumeValue) {
          Slot::releaseSource(op.getLoc(), adaptor.getKey().front(),
                              dictType.getKeyType(), module, rewriter,
                              *converter);
          Slot::releaseSource(op.getLoc(), adaptor.getValue().front(),
                              dictType.getValueType(), module, rewriter,
                              *converter);
        }
        rewriter.eraseOp(op);
        return mlir::success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed dict insert; typed memref lowering required");
  }
};

struct DictGetLowering : public mlir::OpConversionPattern<DictGetOp> {
  DictGetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<DictGetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(DictGetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto dictType = mlir::dyn_cast<DictType>(op.getDict().getType());
    if (!dictType || lowering::value::dict::Storage::classify(dictType) !=
                         StorageKind::TypedMemRefs)
      return mlir::failure();

    mlir::ValueRange dict = adaptor.getDict();
    if (dict.size() != 5 || adaptor.getKey().empty())
      return mlir::failure();
    mlir::Value header = dict[kDictHeaderComponent];
    mlir::Value lock = dict[kDictLockComponent];
    mlir::Value keys = dict[kDictKeysComponent];
    mlir::Value values = dict[kDictValuesComponent];
    mlir::Value states = dict[kDictStatesComponent];
    auto keysType = mlir::dyn_cast<mlir::MemRefType>(keys.getType());
    auto valuesType = mlir::dyn_cast<mlir::MemRefType>(values.getType());
    if (!keysType || !valuesType)
      return mlir::failure();
    mlir::Value key =
        Slot::storage(op.getLoc(), adaptor.getKey(), dictType.getKeyType(),
                      keysType.getElementType(), module, rewriter, *converter);
    if (!key)
      return mlir::failure();

    mlir::Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
    mlir::Value upper =
        createIndexConstant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
    mlir::Value step = createIndexConstant(op.getLoc(), rewriter, 1);
    mlir::Value foundInit = rewriter.create<mlir::arith::ConstantIntOp>(
        op.getLoc(), false, rewriter.getI1Type());
    mlir::Type valueStorageType = valuesType.getElementType();
    mlir::Value resultInit = valueStorageType
                                 ? lowering::value::dict::Storage::zero(
                                       op.getLoc(), valueStorageType, rewriter)
                                 : mlir::Value{};
    if (!resultInit)
      return mlir::failure();

    auto emitLookupAndRetain = [&](llvm::StringRef retainPremise,
                                   bool sharedAccess)
        -> mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> {
      auto markAccess = [&](mlir::Operation *access, mlir::Value target) {
        if (sharedAccess)
          container::access::Contract::mark(access, header, target);
      };
      auto loop = rewriter.create<mlir::scf::ForOp>(
          op.getLoc(), lower, upper, step,
          mlir::ValueRange{foundInit, resultInit});
      {
        mlir::OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(loop.getBody());
        mlir::Value iv = loop.getInductionVar();
        mlir::Value currentFound = loop.getRegionIterArgs()[0];
        mlir::Value currentResult = loop.getRegionIterArgs()[1];

        mlir::Value probeSlot =
            lowering::value::dict::Probe::slot(op.getLoc(), rewriter, key, iv);
        auto stateLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), states, probeSlot);
        markAccess(stateLoad.getOperation(), states);
        mlir::Value state = stateLoad;
        auto keyLoad =
            rewriter.create<mlir::memref::LoadOp>(op.getLoc(), keys, probeSlot);
        markAccess(keyLoad.getOperation(), keys);
        mlir::Value keyAt = keyLoad;
        auto valueLoad = rewriter.create<mlir::memref::LoadOp>(
            op.getLoc(), values, probeSlot);
        markAccess(valueLoad.getOperation(), values);
        if (!sharedAccess)
          ownership::aggregate::Slot::markLoad(valueLoad.getResult());
        mlir::Value valueAt = valueLoad;
        mlir::Value occupiedState = rewriter.create<mlir::arith::ConstantIntOp>(
            op.getLoc(), 1, rewriter.getI8Type());
        mlir::Value occupied = rewriter.create<mlir::arith::CmpIOp>(
            op.getLoc(), mlir::arith::CmpIPredicate::eq, state, occupiedState);
        mlir::Value sameKey = lowering::value::dict::Storage::equal(
            op.getLoc(), keyAt, key, rewriter);
        mlir::Value match = rewriter.create<mlir::arith::AndIOp>(
            op.getLoc(), occupied, sameKey);
        mlir::Value nextFound = rewriter.create<mlir::arith::OrIOp>(
            op.getLoc(), currentFound, match);
        mlir::Value nextResult = rewriter.create<mlir::arith::SelectOp>(
            op.getLoc(), match, valueAt, currentResult);
        rewriter.create<mlir::scf::YieldOp>(
            op.getLoc(), mlir::ValueRange{nextFound, nextResult});
      }
      mlir::Value found = loop.getResult(0);
      mlir::Value result = loop.getResult(1);
      auto retainIfFound =
          rewriter.create<mlir::scf::IfOp>(op.getLoc(), found,
                                           /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard retainGuard(rewriter);
        rewriter.setInsertionPointToStart(retainIfFound.thenBlock());
        if (mlir::failed(
                Slot::refcount(op.getLoc(), result, dictType.getValueType(),
                               module, rewriter, *converter, "incref",
                               /*aggregateEffect=*/false, retainPremise)))
          return mlir::failure();
      }
      return std::pair<mlir::Value, mlir::Value>{found, result};
    };

    mlir::Value isManaged = container::Managed::predicate(
        op.getLoc(), header, kTypedDictRefcountSlot, rewriter);
    auto ifOp = rewriter.create<mlir::scf::IfOp>(
        op.getLoc(), mlir::TypeRange{rewriter.getI1Type(), valueStorageType},
        isManaged,
        /*withElseRegion=*/true);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.thenBlock());
      container::Managed::lock(op.getLoc(), lock, rewriter);
      mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> foundResult =
          emitLookupAndRetain(ThreadSafetyAttrs::kPremiseLockedBorrow,
                              /*sharedAccess=*/true);
      if (mlir::failed(foundResult))
        return mlir::failure();
      container::Managed::unlock(op.getLoc(), lock, rewriter);
      rewriter.create<mlir::scf::YieldOp>(
          op.getLoc(),
          mlir::ValueRange{foundResult->first, foundResult->second});
    }
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(ifOp.elseBlock());
      mlir::FailureOr<std::pair<mlir::Value, mlir::Value>> foundResult =
          emitLookupAndRetain(ThreadSafetyAttrs::kPremiseAggregateBorrow,
                              /*sharedAccess=*/false);
      if (mlir::failed(foundResult))
        return mlir::failure();
      rewriter.create<mlir::scf::YieldOp>(
          op.getLoc(),
          mlir::ValueRange{foundResult->first, foundResult->second});
    }

    mlir::Value found = ifOp.getResult(0);
    mlir::Value notFound = rewriter.create<mlir::arith::XOrIOp>(
        op.getLoc(), found,
        rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), true,
                                                    rewriter.getI1Type()));
    auto missingIf = rewriter.create<mlir::scf::IfOp>(op.getLoc(), notFound,
                                                      /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(missingIf.thenBlock());
      lowering::value::dict::KeyError::throw_(op.getLoc(), module, rewriter,
                                              *converter);
    }

    if (mlir::failed(Slot::replaceBoxedStorage(
            op.getOperation(), ifOp.getResult(1), dictType.getValueType(),
            module, rewriter, *converter)))
      return mlir::failure();
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::dict::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<DictEmptyLowering, DictInsertLowering, DictGetLowering>(
      typeConverter, ctx);
}
} // namespace lowering::value::dict::Patterns

} // namespace py
