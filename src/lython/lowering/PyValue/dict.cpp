#include "Common/LoweringUtils.h"
#include "Common/RuntimeSupport.h"
#include "Common/TypedSlotUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

enum class TypedDictStorage {
  Unsupported,
  TypedMemRefs,
};

static constexpr int64_t kTypedDictDefaultCapacity = 64;
static constexpr int64_t kTypedDictSizeSlot = 0;
static constexpr int64_t kTypedDictLockSlot = 3;
static constexpr int64_t kTypedDictRefcountSlot = 4;

static Value createProbeSlot(Location loc, OpBuilder &builder, Value keySlot,
                             Value probeIndex) {
  Value keyBits = keySlot;
  if (keyBits.getType() == builder.getF64Type())
    keyBits =
        builder.create<arith::BitcastOp>(loc, builder.getI64Type(), keyBits);
  if (auto intType = dyn_cast<IntegerType>(keyBits.getType())) {
    if (intType.getWidth() < 64)
      keyBits =
          builder.create<arith::ExtUIOp>(loc, builder.getI64Type(), keyBits);
    else if (intType.getWidth() > 64)
      keyBits =
          builder.create<arith::TruncIOp>(loc, builder.getI64Type(), keyBits);
  }
  if (keyBits.getType() != builder.getI64Type())
    return {};
  Value probeI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), probeIndex);
  Value capacity = createI64Constant(loc, builder, kTypedDictDefaultCapacity);
  Value mixed = builder.create<arith::AddIOp>(loc, keyBits, probeI64);
  Value slotI64 = builder.create<arith::RemUIOp>(loc, mixed, capacity);
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            slotI64);
}

static TypedDictStorage classifyTypedDictStorage(DictType dictType) {
  if (!dictType)
    return TypedDictStorage::Unsupported;
  if (!isTypedContainerSlotSupported(dictType.getKeyType()) ||
      !isTypedContainerSlotSupported(dictType.getValueType()))
    return TypedDictStorage::Unsupported;
  return TypedDictStorage::TypedMemRefs;
}

static Value compareDictStorageEqual(Location loc, Value lhs, Value rhs,
                                     OpBuilder &builder) {
  if (lhs.getType() != rhs.getType())
    return {};
  if (isa<FloatType>(lhs.getType()))
    return builder.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, lhs,
                                         rhs);
  if (isa<IntegerType>(lhs.getType()))
    return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs,
                                         rhs);
  return {};
}

static Value createStorageZero(Location loc, Type storageType,
                               OpBuilder &builder) {
  if (auto intType = dyn_cast<IntegerType>(storageType))
    return builder.create<arith::ConstantIntOp>(loc, 0, intType);
  if (storageType == builder.getF64Type())
    return builder.create<arith::ConstantFloatOp>(loc, llvm::APFloat(0.0),
                                                  builder.getF64Type());
  return {};
}

struct DictEmptyLowering : public OpConversionPattern<DictEmptyOp> {
  DictEmptyLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictEmptyOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictEmptyOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    (void)adaptor;
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = dyn_cast<DictType>(op.getResult().getType())) {
      if (classifyTypedDictStorage(dictType) ==
          TypedDictStorage::TypedMemRefs) {
        SmallVector<Type, 4> resultTypes;
        if (failed(converter->convertType(op.getResult().getType(),
                                          resultTypes)) ||
            resultTypes.size() != 4)
          return failure();
        auto headerType = dyn_cast<MemRefType>(resultTypes[0]);
        auto keysType = dyn_cast<MemRefType>(resultTypes[1]);
        auto valuesType = dyn_cast<MemRefType>(resultTypes[2]);
        auto statesType = dyn_cast<MemRefType>(resultTypes[3]);
        if (!headerType || !keysType || !valuesType || !statesType)
          return failure();

        auto header =
            rewriter.create<memref::AllocaOp>(op.getLoc(), headerType);
        Value capacityIndex = createIndexConstant(op.getLoc(), rewriter,
                                                  kTypedDictDefaultCapacity);
        auto keys = rewriter.create<memref::AllocaOp>(
            op.getLoc(), keysType, ValueRange{capacityIndex});
        auto values = rewriter.create<memref::AllocaOp>(
            op.getLoc(), valuesType, ValueRange{capacityIndex});
        auto states = rewriter.create<memref::AllocaOp>(
            op.getLoc(), statesType, ValueRange{capacityIndex});

        Value zero = createI64Constant(op.getLoc(), rewriter, 0);
        Value capacity =
            createI64Constant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
        for (int64_t slot = 0; slot < 5; ++slot) {
          Value value = slot == 1 ? capacity : zero;
          rewriter.create<memref::StoreOp>(
              op.getLoc(), value, header,
              createIndexConstant(op.getLoc(), rewriter, slot));
        }
        Value zeroState = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), 0, rewriter.getI8Type());
        for (int64_t slot = 0; slot < kTypedDictDefaultCapacity; ++slot) {
          rewriter.create<memref::StoreOp>(
              op.getLoc(), zeroState, states,
              createIndexConstant(op.getLoc(), rewriter, slot));
        }
        rewriter.replaceOpWithMultiple(
            op, ArrayRef<ValueRange>{
                    ValueRange{header.getResult(), keys.getResult(),
                               values.getResult(), states.getResult()}});
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed dict key/value type; runtime fallback disabled");
  }
};

struct DictInsertLowering : public OpConversionPattern<DictInsertOp> {
  DictInsertLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictInsertOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictInsertOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = dyn_cast<DictType>(op.getResult().getType())) {
      if (classifyTypedDictStorage(dictType) ==
          TypedDictStorage::TypedMemRefs) {
        ValueRange dict = adaptor.getDict();
        if (dict.size() != 4 || adaptor.getKey().empty() ||
            adaptor.getValue().empty())
          return failure();
        Value key = materializeTypedContainerStorageValue(
            op.getLoc(), adaptor.getKey().front(), dictType.getKeyType(),
            module, rewriter, *converter);
        Value value = materializeTypedContainerStorageValue(
            op.getLoc(), adaptor.getValue().front(), dictType.getValueType(),
            module, rewriter, *converter);
        if (!key || !value)
          return failure();

        Value locked;
        if (isMutableContainerArgument(op, op.getDict())) {
          locked = emitManagedContainerPredicate(
              op.getLoc(), dict[0], kTypedDictRefcountSlot, rewriter);
          auto lockIf = rewriter.create<scf::IfOp>(op.getLoc(), locked,
                                                   /*withElseRegion=*/false);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(lockIf.thenBlock());
            emitManagedContainerLockAcquire(op.getLoc(), dict[0],
                                            kTypedDictLockSlot, rewriter);
          }
        }

        Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
        Value upper = createIndexConstant(op.getLoc(), rewriter,
                                          kTypedDictDefaultCapacity);
        Value step = createIndexConstant(op.getLoc(), rewriter, 1);
        Value doneInit = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), false, rewriter.getI1Type());
        Value insertedNewInit = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), false, rewriter.getI1Type());

        auto loop =
            rewriter.create<scf::ForOp>(op.getLoc(), lower, upper, step,
                                        ValueRange{doneInit, insertedNewInit});
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(loop.getBody());
          Value iv = loop.getInductionVar();
          Value done = loop.getRegionIterArgs()[0];
          Value insertedNew = loop.getRegionIterArgs()[1];
          Value probeSlot = createProbeSlot(op.getLoc(), rewriter, key, iv);
          Value state =
              rewriter.create<memref::LoadOp>(op.getLoc(), dict[3], probeSlot);
          Value keyAt =
              rewriter.create<memref::LoadOp>(op.getLoc(), dict[1], probeSlot);
          Value zeroState = rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), 0, rewriter.getI8Type());
          Value occupiedState = rewriter.create<arith::ConstantIntOp>(
              op.getLoc(), 1, rewriter.getI8Type());
          Value empty = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, state, zeroState);
          Value occupied = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, state, occupiedState);
          Value sameKey =
              compareDictStorageEqual(op.getLoc(), keyAt, key, rewriter);
          Value updateExisting =
              rewriter.create<arith::AndIOp>(op.getLoc(), occupied, sameKey);
          Value writable =
              rewriter.create<arith::OrIOp>(op.getLoc(), empty, updateExisting);
          Value notDone = rewriter.create<arith::XOrIOp>(
              op.getLoc(), done,
              rewriter.create<arith::ConstantIntOp>(op.getLoc(), true,
                                                    rewriter.getI1Type()));
          Value shouldWrite =
              rewriter.create<arith::AndIOp>(op.getLoc(), notDone, writable);

          auto ifOp = rewriter.create<scf::IfOp>(op.getLoc(), shouldWrite,
                                                 /*withElseRegion=*/false);
          {
            OpBuilder::InsertionGuard ifGuard(rewriter);
            rewriter.setInsertionPointToStart(ifOp.thenBlock());
            if (auto keyClassType =
                    dyn_cast<ClassType>(dictType.getKeyType())) {
              auto retainKey =
                  rewriter.create<scf::IfOp>(op.getLoc(), empty,
                                             /*withElseRegion=*/false);
              OpBuilder::InsertionGuard retainGuard(rewriter);
              rewriter.setInsertionPointToStart(retainKey.thenBlock());
              emitClassSlotRefcount(op.getLoc(), key, keyClassType, module,
                                    rewriter, "incref");
            }
            if (auto valueClassType =
                    dyn_cast<ClassType>(dictType.getValueType())) {
              emitClassSlotRefcount(op.getLoc(), value, valueClassType, module,
                                    rewriter, "incref");
              auto releaseOld =
                  rewriter.create<scf::IfOp>(op.getLoc(), updateExisting,
                                             /*withElseRegion=*/false);
              OpBuilder::InsertionGuard releaseGuard(rewriter);
              rewriter.setInsertionPointToStart(releaseOld.thenBlock());
              Value oldValue = rewriter.create<memref::LoadOp>(
                  op.getLoc(), dict[2], probeSlot);
              emitClassSlotRefcount(op.getLoc(), oldValue, valueClassType,
                                    module, rewriter, "decref");
            }
            rewriter.create<memref::StoreOp>(op.getLoc(), key, dict[1],
                                             probeSlot);
            rewriter.create<memref::StoreOp>(op.getLoc(), value, dict[2],
                                             probeSlot);
            rewriter.create<memref::StoreOp>(op.getLoc(), occupiedState,
                                             dict[3], probeSlot);
          }

          Value nextDone =
              rewriter.create<arith::OrIOp>(op.getLoc(), done, shouldWrite);
          Value newEntry =
              rewriter.create<arith::AndIOp>(op.getLoc(), shouldWrite, empty);
          Value nextInsertedNew =
              rewriter.create<arith::OrIOp>(op.getLoc(), insertedNew, newEntry);
          rewriter.create<scf::YieldOp>(op.getLoc(),
                                        ValueRange{nextDone, nextInsertedNew});
        }

        Value sizeIndex =
            createIndexConstant(op.getLoc(), rewriter, kTypedDictSizeSlot);
        Value size =
            rewriter.create<memref::LoadOp>(op.getLoc(), dict[0], sizeIndex);
        Value one = createI64Constant(op.getLoc(), rewriter, 1);
        Value incremented =
            rewriter.create<arith::AddIOp>(op.getLoc(), size, one);
        Value nextSize = rewriter.create<arith::SelectOp>(
            op.getLoc(), loop.getResult(1), incremented, size);
        rewriter.create<memref::StoreOp>(op.getLoc(), nextSize, dict[0],
                                         sizeIndex);
        if (locked) {
          auto unlockIf = rewriter.create<scf::IfOp>(op.getLoc(), locked,
                                                     /*withElseRegion=*/false);
          {
            OpBuilder::InsertionGuard guard(rewriter);
            rewriter.setInsertionPointToStart(unlockIf.thenBlock());
            emitManagedContainerLockRelease(op.getLoc(), dict[0],
                                            kTypedDictLockSlot, rewriter);
          }
        }
        rewriter.replaceOpWithMultiple(
            op, ArrayRef<ValueRange>{
                    ValueRange{dict[0], dict[1], dict[2], dict[3]}});
        return success();
      }
    }
    return rewriter.notifyMatchFailure(
        op, "unsupported typed dict insert; runtime fallback disabled");
  }
};

struct DictGetLowering : public OpConversionPattern<DictGetOp> {
  DictGetLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictGetOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictGetOp op, OneToNOpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto dictType = dyn_cast<DictType>(op.getDict().getType());
    if (!dictType ||
        classifyTypedDictStorage(dictType) != TypedDictStorage::TypedMemRefs)
      return failure();

    ValueRange dict = adaptor.getDict();
    if (dict.size() != 4 || adaptor.getKey().empty())
      return failure();
    Value key = materializeTypedContainerStorageValue(
        op.getLoc(), adaptor.getKey().front(), dictType.getKeyType(), module,
        rewriter, *converter);
    if (!key)
      return failure();

    Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
    Value upper =
        createIndexConstant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
    Value step = createIndexConstant(op.getLoc(), rewriter, 1);
    Value foundInit = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), false, rewriter.getI1Type());
    Type valueStorageType = getTypedContainerElementStorageType(
        dictType.getValueType(), rewriter.getContext());
    Value resultInit =
        valueStorageType
            ? createStorageZero(op.getLoc(), valueStorageType, rewriter)
            : Value{};
    if (!resultInit)
      return failure();

    auto loop = rewriter.create<scf::ForOp>(op.getLoc(), lower, upper, step,
                                            ValueRange{foundInit, resultInit});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iv = loop.getInductionVar();
      Value currentFound = loop.getRegionIterArgs()[0];
      Value currentResult = loop.getRegionIterArgs()[1];

      Value probeSlot = createProbeSlot(op.getLoc(), rewriter, key, iv);
      Value state =
          rewriter.create<memref::LoadOp>(op.getLoc(), dict[3], probeSlot);
      Value keyAt =
          rewriter.create<memref::LoadOp>(op.getLoc(), dict[1], probeSlot);
      Value valueAt =
          rewriter.create<memref::LoadOp>(op.getLoc(), dict[2], probeSlot);
      Value occupiedState = rewriter.create<arith::ConstantIntOp>(
          op.getLoc(), 1, rewriter.getI8Type());
      Value occupied = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, state, occupiedState);
      Value sameKey =
          compareDictStorageEqual(op.getLoc(), keyAt, key, rewriter);
      Value match =
          rewriter.create<arith::AndIOp>(op.getLoc(), occupied, sameKey);
      Value nextFound =
          rewriter.create<arith::OrIOp>(op.getLoc(), currentFound, match);
      Value nextResult = rewriter.create<arith::SelectOp>(
          op.getLoc(), match, valueAt, currentResult);
      rewriter.create<scf::YieldOp>(op.getLoc(),
                                    ValueRange{nextFound, nextResult});
    }

    FailureOr<Value> boxed = boxTypedContainerStorageValue(
        op.getLoc(), loop.getResult(1), dictType.getValueType(), module,
        rewriter, *converter);
    if (failed(boxed))
      return failure();
    rewriter.replaceOp(op, *boxed);
    return success();
  }
};

} // namespace

void populatePyDictValueLoweringPatterns(PyLLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<DictEmptyLowering, DictInsertLowering, DictGetLowering>(
      typeConverter, ctx);
}

} // namespace py
