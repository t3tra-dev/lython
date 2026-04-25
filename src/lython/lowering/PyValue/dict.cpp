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
  MemRefSlots,
};

static constexpr int64_t kTypedDictHeaderSlots = 5;
static constexpr int64_t kTypedDictDefaultCapacity = 64;

static Value createProbeSlot(Location loc, OpBuilder &builder, Value keySlot,
                             Value probeIndex) {
  Value probeI64 =
      builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), probeIndex);
  Value capacity = createI64Constant(loc, builder, kTypedDictDefaultCapacity);
  Value mixed = builder.create<arith::AddIOp>(loc, keySlot, probeI64);
  Value slotI64 = builder.create<arith::RemUIOp>(loc, mixed, capacity);
  return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                            slotI64);
}

static TypedDictStorage classifyTypedDictStorage(DictType dictType) {
  if (!dictType)
    return TypedDictStorage::Unsupported;
  if (!isMemRefSlotCompatibleScalarType(dictType.getKeyType()) ||
      !isMemRefSlotCompatibleScalarType(dictType.getValueType()))
    return TypedDictStorage::Unsupported;
  return TypedDictStorage::MemRefSlots;
}

struct DictEmptyLowering : public OpConversionPattern<DictEmptyOp> {
  DictEmptyLowering(PyLLVMTypeConverter &converter, MLIRContext *ctx)
      : OpConversionPattern<DictEmptyOp>(converter, ctx) {}

  LogicalResult
  matchAndRewrite(DictEmptyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = dyn_cast<DictType>(op.getResult().getType())) {
      if (classifyTypedDictStorage(dictType) == TypedDictStorage::MemRefSlots) {
        Type resultType = converter->convertType(op.getResult().getType());
        auto memrefType = dyn_cast_or_null<MemRefType>(resultType);
        if (!memrefType)
          return failure();
        Value allocSize = createIndexConstant(
            op.getLoc(), rewriter,
            kTypedDictHeaderSlots + kTypedDictDefaultCapacity * 3);
        auto dict = rewriter.create<memref::AllocaOp>(op.getLoc(), memrefType,
                                                      ValueRange{allocSize});
        Value zero = createI64Constant(op.getLoc(), rewriter, 0);
        Value capacity =
            createI64Constant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
        for (int64_t slot = 0; slot < kTypedDictHeaderSlots; ++slot) {
          Value value = slot == 1 ? capacity : zero;
          rewriter.create<memref::StoreOp>(
              op.getLoc(), value, dict,
              createIndexConstant(op.getLoc(), rewriter, slot));
        }
        Value stateBase = createIndexConstant(
            op.getLoc(), rewriter,
            kTypedDictHeaderSlots + kTypedDictDefaultCapacity * 2);
        for (int64_t slot = 0; slot < kTypedDictDefaultCapacity; ++slot) {
          Value stateIndex = rewriter.create<arith::AddIOp>(
              op.getLoc(), stateBase,
              createIndexConstant(op.getLoc(), rewriter, slot));
          rewriter.create<memref::StoreOp>(op.getLoc(), zero, dict, stateIndex);
        }
        rewriter.replaceOp(op, dict.getResult());
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
  matchAndRewrite(DictInsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (auto dictType = dyn_cast<DictType>(op.getResult().getType())) {
      if (classifyTypedDictStorage(dictType) == TypedDictStorage::MemRefSlots) {
        FailureOr<Value> key =
            packTypedSlot(op.getLoc(), adaptor.getKey(), dictType.getKeyType(),
                          module, rewriter, *converter);
        FailureOr<Value> value = packTypedSlot(op.getLoc(), adaptor.getValue(),
                                               dictType.getValueType(), module,
                                               rewriter, *converter);
        if (failed(key) || failed(value))
          return failure();

        Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
        Value upper = createIndexConstant(op.getLoc(), rewriter,
                                          kTypedDictDefaultCapacity);
        Value step = createIndexConstant(op.getLoc(), rewriter, 1);
        Value doneInit = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), false, rewriter.getI1Type());
        Value insertedNewInit = rewriter.create<arith::ConstantIntOp>(
            op.getLoc(), false, rewriter.getI1Type());

        Value keyBase =
            createIndexConstant(op.getLoc(), rewriter, kTypedDictHeaderSlots);
        Value valueBase = createIndexConstant(op.getLoc(), rewriter,
                                              kTypedDictHeaderSlots +
                                                  kTypedDictDefaultCapacity);
        Value stateBase = createIndexConstant(
            op.getLoc(), rewriter,
            kTypedDictHeaderSlots + kTypedDictDefaultCapacity * 2);

        auto loop =
            rewriter.create<scf::ForOp>(op.getLoc(), lower, upper, step,
                                        ValueRange{doneInit, insertedNewInit});
        {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(loop.getBody());
          Value iv = loop.getInductionVar();
          Value done = loop.getRegionIterArgs()[0];
          Value insertedNew = loop.getRegionIterArgs()[1];
          Value probeSlot = createProbeSlot(op.getLoc(), rewriter, *key, iv);
          Value keyIndex =
              rewriter.create<arith::AddIOp>(op.getLoc(), keyBase, probeSlot);
          Value valueIndex =
              rewriter.create<arith::AddIOp>(op.getLoc(), valueBase, probeSlot);
          Value stateIndex =
              rewriter.create<arith::AddIOp>(op.getLoc(), stateBase, probeSlot);
          Value state = rewriter.create<memref::LoadOp>(
              op.getLoc(), adaptor.getDict(), stateIndex);
          Value keyAt = rewriter.create<memref::LoadOp>(
              op.getLoc(), adaptor.getDict(), keyIndex);
          Value empty = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, state,
              createI64Constant(op.getLoc(), rewriter, 0));
          Value occupied = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, state,
              createI64Constant(op.getLoc(), rewriter, 1));
          Value sameKey = rewriter.create<arith::CmpIOp>(
              op.getLoc(), arith::CmpIPredicate::eq, keyAt, *key);
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
            rewriter.create<memref::StoreOp>(op.getLoc(), *key,
                                             adaptor.getDict(), keyIndex);
            rewriter.create<memref::StoreOp>(op.getLoc(), *value,
                                             adaptor.getDict(), valueIndex);
            rewriter.create<memref::StoreOp>(
                op.getLoc(), createI64Constant(op.getLoc(), rewriter, 1),
                adaptor.getDict(), stateIndex);
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

        Value sizeIndex = createIndexConstant(op.getLoc(), rewriter, 0);
        Value size = rewriter.create<memref::LoadOp>(
            op.getLoc(), adaptor.getDict(), sizeIndex);
        Value one = createI64Constant(op.getLoc(), rewriter, 1);
        Value incremented =
            rewriter.create<arith::AddIOp>(op.getLoc(), size, one);
        Value nextSize = rewriter.create<arith::SelectOp>(
            op.getLoc(), loop.getResult(1), incremented, size);
        rewriter.create<memref::StoreOp>(op.getLoc(), nextSize,
                                         adaptor.getDict(), sizeIndex);
        rewriter.replaceOp(op, adaptor.getDict());
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
  matchAndRewrite(DictGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();
    auto *converter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    auto dictType = dyn_cast<DictType>(op.getDict().getType());
    if (!dictType ||
        classifyTypedDictStorage(dictType) != TypedDictStorage::MemRefSlots)
      return failure();

    FailureOr<Value> key =
        packTypedSlot(op.getLoc(), adaptor.getKey(), dictType.getKeyType(),
                      module, rewriter, *converter);
    if (failed(key))
      return failure();

    Value lower = createIndexConstant(op.getLoc(), rewriter, 0);
    Value upper =
        createIndexConstant(op.getLoc(), rewriter, kTypedDictDefaultCapacity);
    Value step = createIndexConstant(op.getLoc(), rewriter, 1);
    Value foundInit = rewriter.create<arith::ConstantIntOp>(
        op.getLoc(), false, rewriter.getI1Type());
    Value resultInit = createI64Constant(op.getLoc(), rewriter, 0);

    auto loop = rewriter.create<scf::ForOp>(op.getLoc(), lower, upper, step,
                                            ValueRange{foundInit, resultInit});
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      Value iv = loop.getInductionVar();
      Value currentFound = loop.getRegionIterArgs()[0];
      Value currentResult = loop.getRegionIterArgs()[1];

      Value keyBase =
          createIndexConstant(op.getLoc(), rewriter, kTypedDictHeaderSlots);
      Value valueBase = createIndexConstant(op.getLoc(), rewriter,
                                            kTypedDictHeaderSlots +
                                                kTypedDictDefaultCapacity);
      Value stateBase = createIndexConstant(op.getLoc(), rewriter,
                                            kTypedDictHeaderSlots +
                                                kTypedDictDefaultCapacity * 2);
      Value probeSlot = createProbeSlot(op.getLoc(), rewriter, *key, iv);
      Value keyIndex =
          rewriter.create<arith::AddIOp>(op.getLoc(), keyBase, probeSlot);
      Value valueIndex =
          rewriter.create<arith::AddIOp>(op.getLoc(), valueBase, probeSlot);
      Value stateIndex =
          rewriter.create<arith::AddIOp>(op.getLoc(), stateBase, probeSlot);
      Value state = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getDict(), stateIndex);
      Value keyAt = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getDict(), keyIndex);
      Value valueAt = rewriter.create<memref::LoadOp>(
          op.getLoc(), adaptor.getDict(), valueIndex);
      Value occupied = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, state,
          createI64Constant(op.getLoc(), rewriter, 1));
      Value sameKey = rewriter.create<arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::eq, keyAt, *key);
      Value match =
          rewriter.create<arith::AndIOp>(op.getLoc(), occupied, sameKey);
      Value nextFound =
          rewriter.create<arith::OrIOp>(op.getLoc(), currentFound, match);
      Value nextResult = rewriter.create<arith::SelectOp>(
          op.getLoc(), match, valueAt, currentResult);
      rewriter.create<scf::YieldOp>(op.getLoc(),
                                    ValueRange{nextFound, nextResult});
    }

    FailureOr<Value> boxed =
        boxTypedSlot(op.getLoc(), loop.getResult(1), dictType.getValueType(),
                     module, rewriter, *converter);
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
