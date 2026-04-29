#include "Common/TypedSlotUtils.h"

#include "Common/LoweringUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <optional>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;

namespace py {
namespace {

static bool arrayAttrContainsIndex(ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (Attribute element : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(element);
    if (intAttr && intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

static Value stripContainerAccessCasts(Value value) {
  while (true) {
    if (auto identity = value.getDefiningOp<CastIdentityOp>()) {
      value = identity.getInput();
      continue;
    }
    if (auto publish = value.getDefiningOp<PublishOp>()) {
      value = publish.getInput();
      continue;
    }
    return value;
  }
}

static bool isFixedI64MemRef(Value value, int64_t slots) {
  auto type = dyn_cast<MemRefType>(value.getType());
  return type && type.hasStaticShape() && type.getRank() == 1 &&
         type.getDimSize(0) == slots && type.getElementType().isInteger(64);
}

static unsigned getLogicalArgWidth(Block &entry, unsigned argIndex) {
  Value arg = entry.getArgument(argIndex);
  unsigned remaining = entry.getNumArguments() - argIndex;
  if (isFixedI64MemRef(arg, 5) && remaining >= 4)
    return 4;
  if ((isFixedI64MemRef(arg, 3) || isFixedI64MemRef(arg, 4)) && remaining >= 2)
    return 2;
  return 1;
}

static std::optional<unsigned>
getLogicalArgIndexFromConvertedEntryArg(BlockArgument arg) {
  Block *entry = arg.getOwner();
  if (!entry)
    return std::nullopt;
  auto parentFunc = dyn_cast_or_null<func::FuncOp>(entry->getParentOp());
  if (!parentFunc || entry != &parentFunc.getBody().front())
    return std::nullopt;

  unsigned logicalIndex = 0;
  for (unsigned physicalIndex = 0; physicalIndex < entry->getNumArguments();) {
    if (physicalIndex == arg.getArgNumber())
      return logicalIndex;
    unsigned width = getLogicalArgWidth(*entry, physicalIndex);
    if (arg.getArgNumber() < physicalIndex + width)
      return std::nullopt;
    physicalIndex += width;
    ++logicalIndex;
  }
  return std::nullopt;
}

static std::optional<unsigned> getLogicalArgIndex(Operation *op, Value value) {
  value = stripContainerAccessCasts(value);
  if (auto cast = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 0)
      return std::nullopt;
    value = cast->getOperand(0);
  }
  auto arg = dyn_cast<BlockArgument>(value);
  if (!arg)
    return std::nullopt;
  auto index = getLogicalArgIndexFromConvertedEntryArg(arg);
  if (!index)
    return std::nullopt;

  auto parentFunc = op->getParentOfType<func::FuncOp>();
  if (!parentFunc || arg.getOwner() != &parentFunc.getBody().front())
    return std::nullopt;
  return index;
}

} // namespace

FailureOr<Value> packTypedSlot(Location loc, Value value, Type logicalType,
                               ModuleOp module,
                               ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());

  if (isa<IntType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(),
              ValueRange{value})
        .getResult();
  }

  if (isa<BoolType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    if (value.getType() == rewriter.getI1Type())
      return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), value)
          .getResult();
    Value raw = runtime
                    .call(loc, RuntimeSymbols::kBoolAsBool,
                          rewriter.getI1Type(), ValueRange{value})
                    .getResult();
    return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), raw)
        .getResult();
  }

  if (isa<FloatType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    Value asDouble = value;
    if (value.getType() != rewriter.getF64Type())
      asDouble = runtime
                     .call(loc, RuntimeSymbols::kFloatAsDouble,
                           rewriter.getF64Type(), ValueRange{value})
                     .getResult();
    return rewriter
        .create<arith::BitcastOp>(loc, rewriter.getI64Type(), asDouble)
        .getResult();
  }

  if (value.getType() == rewriter.getI64Type())
    return value;

  if (isa<StrType, ObjectType, ClassType>(logicalType)) {
    Value asPtr = value;
    if (asPtr.getType() != ptrType)
      asPtr = rewriter.create<CastIdentityOp>(loc, ptrType, asPtr);
    return rewriter.create<LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), asPtr)
        .getResult();
  }

  return failure();
}

FailureOr<Value> boxTypedSlot(Location loc, Value value, Type logicalType,
                              ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);

  if (isa<IntType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kLongFromI64,
              typeConverter.getPyObjectPtrType(), ValueRange{value})
        .getResult();

  if (isa<BoolType>(logicalType)) {
    Value zero = createI64Constant(loc, rewriter, 0);
    Value asBool = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne,
                                                  value, zero);
    return runtime
        .call(loc, RuntimeSymbols::kBoolFromBool,
              typeConverter.getPyObjectPtrType(), ValueRange{asBool})
        .getResult();
  }

  if (isa<FloatType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kFloatFromDouble,
              typeConverter.getPyObjectPtrType(),
              ValueRange{rewriter.create<arith::BitcastOp>(
                  loc, rewriter.getF64Type(), value)})
        .getResult();

  if (isa<StrType, ObjectType, ClassType>(logicalType)) {
    auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
    return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, value).getResult();
  }

  return failure();
}

Value materializeTypedContainerStorageValue(
    Location loc, Value value, Type logicalType, ModuleOp module,
    ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter) {
  Type storageType =
      getTypedContainerElementStorageType(logicalType, rewriter.getContext());
  if (!storageType)
    return {};

  if (value.getType() == storageType)
    return value;

  if (isa<BoolType>(logicalType) && storageType == rewriter.getI8Type()) {
    if (auto intType = dyn_cast<IntegerType>(value.getType())) {
      if (intType.getWidth() > 8)
        return rewriter.create<arith::TruncIOp>(loc, storageType, value);
      if (intType.getWidth() < 8)
        return rewriter.create<arith::ExtUIOp>(loc, storageType, value);
    }
    RuntimeAPI runtime(module, rewriter, typeConverter);
    Value raw = runtime
                    .call(loc, RuntimeSymbols::kBoolAsBool,
                          rewriter.getI1Type(), ValueRange{value})
                    .getResult();
    return rewriter.create<arith::ExtUIOp>(loc, storageType, raw);
  }

  if (isa<FloatType>(logicalType) && storageType == rewriter.getF64Type()) {
    if (value.getType() == rewriter.getI64Type())
      return rewriter.create<arith::BitcastOp>(loc, storageType, value);
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kFloatAsDouble, storageType,
              ValueRange{value})
        .getResult();
  }

  if (isa<IntType>(logicalType) && storageType == rewriter.getI64Type()) {
    if (auto intType = dyn_cast<IntegerType>(value.getType())) {
      if (intType.getWidth() < 64)
        return rewriter.create<arith::ExtUIOp>(loc, storageType, value);
      if (intType.getWidth() > 64)
        return rewriter.create<arith::TruncIOp>(loc, storageType, value);
    }
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, storageType, ValueRange{value})
        .getResult();
  }

  FailureOr<Value> packed =
      packTypedSlot(loc, value, logicalType, module, rewriter, typeConverter);
  if (failed(packed))
    return {};
  Value result = *packed;
  if (result.getType() == storageType)
    return result;
  if (isa<BoolType>(logicalType) && storageType == rewriter.getI8Type()) {
    if (auto intType = dyn_cast<IntegerType>(result.getType())) {
      if (intType.getWidth() > 8)
        return rewriter.create<arith::TruncIOp>(loc, storageType, result);
      if (intType.getWidth() < 8)
        return rewriter.create<arith::ExtUIOp>(loc, storageType, result);
    }
  }
  if (isa<FloatType>(logicalType) && storageType == rewriter.getF64Type() &&
      result.getType() == rewriter.getI64Type())
    return rewriter.create<arith::BitcastOp>(loc, storageType, result);
  if (storageType == rewriter.getI64Type()) {
    if (auto intType = dyn_cast<IntegerType>(result.getType())) {
      if (intType.getWidth() < 64)
        return rewriter.create<arith::ExtUIOp>(loc, storageType, result);
      if (intType.getWidth() > 64)
        return rewriter.create<arith::TruncIOp>(loc, storageType, result);
    }
  }
  return {};
}

FailureOr<Value>
boxTypedContainerStorageValue(Location loc, Value value, Type logicalType,
                              ModuleOp module,
                              ConversionPatternRewriter &rewriter,
                              const PyLLVMTypeConverter &typeConverter) {
  Value slot = value;
  if (isa<BoolType>(logicalType)) {
    if (slot.getType() != rewriter.getI64Type())
      slot = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), slot);
  } else if (isa<FloatType>(logicalType)) {
    if (slot.getType() == rewriter.getF64Type())
      slot =
          rewriter.create<arith::BitcastOp>(loc, rewriter.getI64Type(), slot);
  } else if (slot.getType() != rewriter.getI64Type() &&
             isa<IntegerType>(slot.getType())) {
    auto intType = cast<IntegerType>(slot.getType());
    if (intType.getWidth() < 64)
      slot = rewriter.create<arith::ExtUIOp>(loc, rewriter.getI64Type(), slot);
    else if (intType.getWidth() > 64)
      slot = rewriter.create<arith::TruncIOp>(loc, rewriter.getI64Type(), slot);
  }
  return boxTypedSlot(loc, slot, logicalType, module, rewriter, typeConverter);
}

void emitClassSlotRefcount(Location loc, Value slot, ClassType type,
                           ModuleOp module, ConversionPatternRewriter &rewriter,
                           StringRef suffix) {
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  Value asPtr = rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, slot);
  auto helper = getOrInsertLLVMFunc(
      loc, module, rewriter, getClassHelperName(type, suffix),
      LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  auto helperRef = SymbolRefAttr::get(module.getContext(), helper.getName());
  rewriter.create<LLVM::CallOp>(loc, TypeRange{}, helperRef, ValueRange{asPtr});
}

bool isMutableContainerArgument(Operation *op, Value container) {
  auto index = getLogicalArgIndex(op, container);
  if (!index)
    return false;
  auto parentFunc = op->getParentOfType<func::FuncOp>();
  return parentFunc &&
         arrayAttrContainsIndex(
             parentFunc->getAttrOfType<ArrayAttr>("lython.mutable_args"),
             *index);
}

Value emitManagedContainerPredicate(Location loc, Value header,
                                    int64_t refcountSlot, OpBuilder &builder) {
  Value marker = builder.create<memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, refcountSlot));
  return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, marker,
                                       createI64Constant(loc, builder, 0));
}

void emitManagedContainerLockAcquire(Location loc, Value header,
                                     int64_t lockSlot, OpBuilder &builder) {
  Value lockIndex = createIndexConstant(loc, builder, lockSlot);
  auto whileOp = builder.create<scf::WhileOp>(loc, TypeRange{}, ValueRange{});
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);
    Value previous = builder.create<memref::AtomicRMWOp>(
        loc, arith::AtomicRMWKind::assign, createI64Constant(loc, builder, 1),
        header, ValueRange{lockIndex});
    Value busy =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, previous,
                                      createI64Constant(loc, builder, 0));
    builder.create<scf::ConditionOp>(loc, busy, ValueRange{});
  }
  {
    OpBuilder::InsertionGuard guard(builder);
    Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<scf::YieldOp>(loc);
  }
}

void emitManagedContainerLockRelease(Location loc, Value header,
                                     int64_t lockSlot, OpBuilder &builder) {
  builder.create<memref::AtomicRMWOp>(
      loc, arith::AtomicRMWKind::assign, createI64Constant(loc, builder, 0),
      header, ValueRange{createIndexConstant(loc, builder, lockSlot)});
}

} // namespace py
