#include "Common/SlotUtils.h"

#include "Common/Container.h"
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

namespace py {
namespace {

static bool arrayAttrContainsIndex(mlir::ArrayAttr attr, unsigned index) {
  if (!attr)
    return false;
  for (mlir::Attribute element : attr) {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(element);
    if (intAttr && intAttr.getInt() == static_cast<int64_t>(index))
      return true;
  }
  return false;
}

static mlir::Value stripContainerAccessCasts(mlir::Value value) {
  while (true) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        return value;
      value = cast.getOperand(0);
      continue;
    }
    if (auto publish = value.getDefiningOp<PublishOp>()) {
      value = publish.getInput();
      continue;
    }
    return value;
  }
}

static unsigned getLogicalArgWidth(mlir::Block &entry, unsigned argIndex) {
  mlir::Value arg = entry.getArgument(argIndex);
  unsigned remaining = entry.getNumArguments() - argIndex;
  if (auto kind = container::Kind::fromHeader(arg)) {
    unsigned width = container::Descriptor::componentCount(arg);
    if (remaining >= width)
      return width;
  }
  return 1;
}

static std::optional<unsigned>
getLogicalArgIndexFromConvertedEntryArg(mlir::BlockArgument arg) {
  mlir::Block *entry = arg.getOwner();
  if (!entry)
    return std::nullopt;
  auto parentFunc =
      mlir::dyn_cast_or_null<mlir::func::FuncOp>(entry->getParentOp());
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

static std::optional<unsigned> getLogicalArgIndex(mlir::Operation *op,
                                                  mlir::Value value) {
  value = stripContainerAccessCasts(value);
  if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() == 0)
      return std::nullopt;
    value = cast->getOperand(0);
  }
  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg)
    return std::nullopt;
  auto index = getLogicalArgIndexFromConvertedEntryArg(arg);
  if (!index)
    return std::nullopt;

  auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
  if (!parentFunc || arg.getOwner() != &parentFunc.getBody().front())
    return std::nullopt;
  return index;
}

static void copyMemRefPrefix(mlir::Location loc, mlir::Value source,
                             mlir::Value target, mlir::Value count,
                             mlir::OpBuilder &builder) {
  mlir::Value lower = createIndexConstant(loc, builder, 0);
  mlir::Value step = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, count, step);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value value = builder.create<mlir::memref::LoadOp>(loc, source, iv);
    builder.create<mlir::memref::StoreOp>(loc, value, target, iv);
  }
}

static mlir::Value loadI64HeaderSlot(mlir::Location loc, mlir::Value header,
                                     int64_t slot, mlir::OpBuilder &builder) {
  return builder.create<mlir::memref::LoadOp>(
      loc, header, createIndexConstant(loc, builder, slot));
}

static mlir::memref::StoreOp storeI64HeaderSlot(mlir::Location loc,
                                                mlir::Value header,
                                                int64_t slot, int64_t value,
                                                mlir::OpBuilder &builder) {
  auto store = builder.create<mlir::memref::StoreOp>(
      loc, createI64Constant(loc, builder, value), header,
      createIndexConstant(loc, builder, slot));
  if (auto refcountSlot = container::Refcount::slot(header)) {
    if (*refcountSlot == slot && value == 1) {
      store->setAttr(ContainerSafetyAttrs::kRefcountInit,
                     builder.getI64IntegerAttr(value));
      store->setAttr(
          ContainerSafetyAttrs::kRefcountState,
          builder.getStringAttr(ContainerSafetyAttrs::kStateManaged));
    }
  }
  return store;
}

static mlir::Value staticMemRefSize(mlir::Location loc, mlir::MemRefType type,
                                    mlir::OpBuilder &builder) {
  if (type.getRank() != 1 || mlir::ShapedType::isDynamic(type.getShape()[0]))
    return {};
  return createIndexConstant(loc, builder, type.getShape()[0]);
}

static mlir::Value
materializeClonedContainerSlot(mlir::Location loc, mlir::Value slot,
                               mlir::Type logicalType, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    ownership::aggregate::Slot::markLoad(slot);
    mlir::Value asPtr =
        rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, slot);
    std::string helperName = getClassHelperName(classType, "promote");
    auto helper = getOrInsertLLVMFunc(loc, module, rewriter, helperName,
                                      ptrType, {ptrType});
    ownership::llvm_func::Contract::apply(helper, helperName);
    auto helperRef =
        mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
    auto call = rewriter.create<mlir::LLVM::CallOp>(
        loc, mlir::TypeRange{ptrType}, helperRef, mlir::ValueRange{asPtr});
    return rewriter.create<mlir::LLVM::PtrToIntOp>(loc, rewriter.getI64Type(),
                                                   call.getResult());
  }

  Slot::refcount(loc, slot, logicalType, module, rewriter, typeConverter,
                 "incref", /*aggregateEffect=*/true,
                 ThreadSafetyAttrs::kPremiseAggregateBorrow);
  return slot;
}

static mlir::Value materializePromotedContainerSlot(
    mlir::Location loc, mlir::Value slot, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (!cloneReferenceSlots)
    return slot;
  return materializeClonedContainerSlot(loc, slot, logicalType, module,
                                        rewriter, typeConverter);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoteListDescriptor(
    mlir::Location loc, mlir::ValueRange input, ListType listType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (input.size() != 2)
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(input[0].getType());
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(input[1].getType());
  if (!headerType || !itemsType)
    return mlir::failure();
  mlir::Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return mlir::failure();

  mlir::Value managedHeader =
      rewriter.create<mlir::memref::AllocOp>(loc, headerType);
  mlir::Value size =
      loadI64HeaderSlot(loc, input[0], kTypedListSizeSlot, rewriter);
  mlir::Value capacity =
      loadI64HeaderSlot(loc, input[0], kTypedListCapacitySlot, rewriter);
  mlir::Value sizeIndex = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rewriter.getIndexType(), size);
  mlir::Value capacityIndex = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rewriter.getIndexType(), capacity);
  mlir::Value managedItems = rewriter.create<mlir::memref::AllocOp>(
      loc, itemsType, mlir::ValueRange{capacityIndex});

  std::string descriptorGroup =
      container::descriptor::Group::make(managedHeader.getDefiningOp(), "list");
  container::descriptor::Component::mark(
      managedHeader, descriptorGroup, ContainerSafetyAttrs::kComponentHeader);
  container::descriptor::Component::mark(managedItems, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentItems);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedItems, sizeIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, kTypedListRefcountSlot, 1, rewriter);

  if (Slot::refcounted(listType.getElementType()) && cloneReferenceSlots) {
    mlir::Value lower = createIndexConstant(loc, rewriter, 0);
    mlir::Value step = createIndexConstant(loc, rewriter, 1);
    auto loop = rewriter.create<mlir::scf::ForOp>(loc, lower, sizeIndex, step);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value slot = rewriter.create<mlir::memref::LoadOp>(
          loc, managedItems, loop.getInductionVar());
      mlir::Value promoted = materializePromotedContainerSlot(
          loc, slot, listType.getElementType(), module, rewriter, typeConverter,
          cloneReferenceSlots);
      auto store = rewriter.create<mlir::memref::StoreOp>(
          loc, promoted, managedItems, loop.getInductionVar());
      Slot::markTransfer(store.getOperation());
    }
  }

  return llvm::SmallVector<mlir::Value>{managedHeader, managedItems};
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoteTupleDescriptor(
    mlir::Location loc, mlir::ValueRange input, TupleType tupleType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (input.size() != 2)
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(input[0].getType());
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(input[1].getType());
  if (!headerType || !itemsType)
    return mlir::failure();
  mlir::Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return mlir::failure();

  mlir::Value managedHeader =
      rewriter.create<mlir::memref::AllocOp>(loc, headerType);
  mlir::Value size =
      loadI64HeaderSlot(loc, input[0], kTypedTupleSizeSlot, rewriter);
  mlir::Value sizeIndex = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rewriter.getIndexType(), size);
  mlir::Value managedItems = rewriter.create<mlir::memref::AllocOp>(
      loc, itemsType, mlir::ValueRange{sizeIndex});

  std::string descriptorGroup = container::descriptor::Group::make(
      managedHeader.getDefiningOp(), "tuple");
  container::descriptor::Component::mark(
      managedHeader, descriptorGroup, ContainerSafetyAttrs::kComponentHeader);
  container::descriptor::Component::mark(managedItems, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentItems);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedItems, sizeIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, kTypedTupleRefcountSlot, 1, rewriter);

  if (cloneReferenceSlots) {
    for (auto [index, elementType] :
         llvm::enumerate(tupleType.getElementTypes())) {
      if (!Slot::refcounted(elementType))
        continue;
      mlir::Value slot = rewriter.create<mlir::memref::LoadOp>(
          loc, managedItems,
          createIndexConstant(loc, rewriter, static_cast<int64_t>(index)));
      mlir::Value promoted = materializePromotedContainerSlot(
          loc, slot, elementType, module, rewriter, typeConverter,
          cloneReferenceSlots);
      auto store = rewriter.create<mlir::memref::StoreOp>(
          loc, promoted, managedItems,
          createIndexConstant(loc, rewriter, static_cast<int64_t>(index)));
      Slot::markTransfer(store.getOperation());
    }
  }

  return llvm::SmallVector<mlir::Value>{managedHeader, managedItems};
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoteDictDescriptor(
    mlir::Location loc, mlir::ValueRange input, DictType dictType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (input.size() != 4)
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(input[0].getType());
  auto keysType = mlir::dyn_cast<mlir::MemRefType>(input[1].getType());
  auto valuesType = mlir::dyn_cast<mlir::MemRefType>(input[2].getType());
  auto statesType = mlir::dyn_cast<mlir::MemRefType>(input[3].getType());
  if (!headerType || !keysType || !valuesType || !statesType)
    return mlir::failure();
  mlir::Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return mlir::failure();

  mlir::Value managedHeader =
      rewriter.create<mlir::memref::AllocOp>(loc, headerType);
  mlir::Value capacity =
      loadI64HeaderSlot(loc, input[0], kTypedDictCapacitySlot, rewriter);
  mlir::Value capacityIndex = rewriter.create<mlir::arith::IndexCastOp>(
      loc, rewriter.getIndexType(), capacity);
  mlir::Value managedKeys = rewriter.create<mlir::memref::AllocOp>(
      loc, keysType, mlir::ValueRange{capacityIndex});
  mlir::Value managedValues = rewriter.create<mlir::memref::AllocOp>(
      loc, valuesType, mlir::ValueRange{capacityIndex});
  mlir::Value managedStates = rewriter.create<mlir::memref::AllocOp>(
      loc, statesType, mlir::ValueRange{capacityIndex});

  std::string descriptorGroup =
      container::descriptor::Group::make(managedHeader.getDefiningOp(), "dict");
  container::descriptor::Component::mark(
      managedHeader, descriptorGroup, ContainerSafetyAttrs::kComponentHeader);
  container::descriptor::Component::mark(managedKeys, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentKeys);
  container::descriptor::Component::mark(
      managedValues, descriptorGroup, ContainerSafetyAttrs::kComponentValues);
  container::descriptor::Component::mark(
      managedStates, descriptorGroup, ContainerSafetyAttrs::kComponentStates);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  copyMemRefPrefix(loc, input[1], managedKeys, capacityIndex, rewriter);
  copyMemRefPrefix(loc, input[2], managedValues, capacityIndex, rewriter);
  copyMemRefPrefix(loc, input[3], managedStates, capacityIndex, rewriter);
  storeI64HeaderSlot(loc, managedHeader, kTypedDictRefcountSlot, 1, rewriter);

  bool keyRefcounted = Slot::refcounted(dictType.getKeyType());
  bool valueRefcounted = Slot::refcounted(dictType.getValueType());
  if ((keyRefcounted || valueRefcounted) && cloneReferenceSlots) {
    mlir::Value lower = createIndexConstant(loc, rewriter, 0);
    mlir::Value step = createIndexConstant(loc, rewriter, 1);
    auto loop =
        rewriter.create<mlir::scf::ForOp>(loc, lower, capacityIndex, step);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      mlir::Value state = rewriter.create<mlir::memref::LoadOp>(
          loc, managedStates, loop.getInductionVar());
      mlir::Value occupied = rewriter.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, state,
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 1,
                                                      rewriter.getI8Type()));
      auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, occupied,
                                                   /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard ifGuard(rewriter);
        rewriter.setInsertionPointToStart(ifOp.thenBlock());
        if (keyRefcounted) {
          mlir::Value key = rewriter.create<mlir::memref::LoadOp>(
              loc, managedKeys, loop.getInductionVar());
          mlir::Value promotedKey = materializePromotedContainerSlot(
              loc, key, dictType.getKeyType(), module, rewriter, typeConverter,
              cloneReferenceSlots);
          auto keyStore = rewriter.create<mlir::memref::StoreOp>(
              loc, promotedKey, managedKeys, loop.getInductionVar());
          Slot::markTransfer(keyStore.getOperation());
        }
        if (valueRefcounted) {
          mlir::Value value = rewriter.create<mlir::memref::LoadOp>(
              loc, managedValues, loop.getInductionVar());
          mlir::Value promotedValue = materializePromotedContainerSlot(
              loc, value, dictType.getValueType(), module, rewriter,
              typeConverter, cloneReferenceSlots);
          auto valueStore = rewriter.create<mlir::memref::StoreOp>(
              loc, promotedValue, managedValues, loop.getInductionVar());
          Slot::markTransfer(valueStore.getOperation());
        }
      }
    }
  }

  return llvm::SmallVector<mlir::Value>{managedHeader, managedKeys,
                                        managedValues, managedStates};
}

} // namespace

mlir::Value Slot::bridgePointer(mlir::Location loc, mlir::Value value,
                                mlir::ConversionPatternRewriter &rewriter) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  if (value.getType() == ptrType)
    return value;

  mlir::Value current = value;
  while (true) {
    if (auto cast = current.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() == 1) {
        mlir::Value operand = cast->getOperand(0);
        if (operand.getType() == ptrType)
          return operand;
        current = operand;
        continue;
      }
    }
    if (auto publish = current.getDefiningOp<PublishOp>()) {
      current = publish.getInput();
      if (current.getType() == ptrType)
        return current;
      continue;
    }
    break;
  }

  return rewriter
      .create<mlir::UnrealizedConversionCastOp>(loc, mlir::TypeRange{ptrType},
                                                mlir::ValueRange{value})
      .getResult(0);
}

mlir::FailureOr<mlir::Value>
Slot::pack(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
           mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
           const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);

  if (mlir::isa<IntType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, rewriter.getI64Type(),
              mlir::ValueRange{value})
        .getResult();
  }

  if (mlir::isa<BoolType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    if (value.getType() == rewriter.getI1Type())
      return rewriter
          .create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(), value)
          .getResult();
    mlir::Value raw = runtime
                          .call(loc, RuntimeSymbols::kBoolAsBool,
                                rewriter.getI1Type(), mlir::ValueRange{value})
                          .getResult();
    return rewriter
        .create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(), raw)
        .getResult();
  }

  if (mlir::isa<mlir::FloatType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    mlir::Value asDouble = value;
    if (value.getType() != rewriter.getF64Type())
      asDouble = runtime
                     .call(loc, RuntimeSymbols::kFloatAsDouble,
                           rewriter.getF64Type(), mlir::ValueRange{value})
                     .getResult();
    return rewriter
        .create<mlir::arith::BitcastOp>(loc, rewriter.getI64Type(), asDouble)
        .getResult();
  }

  if (value.getType() == rewriter.getI64Type())
    return value;

  if (mlir::isa<NoneType, StrType, ObjectType, ClassType, ExceptionType,
                TracebackType, LocationType>(logicalType)) {
    mlir::Value asPtr = Slot::bridgePointer(loc, value, rewriter);
    return rewriter
        .create<mlir::LLVM::PtrToIntOp>(loc, rewriter.getI64Type(), asPtr)
        .getResult();
  }

  return mlir::failure();
}

mlir::FailureOr<mlir::Value>
Slot::box(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
          mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
          const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);

  if (mlir::isa<IntType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kLongFromI64,
              typeConverter.getPyObjectPtrType(), mlir::ValueRange{value})
        .getResult();

  if (mlir::isa<BoolType>(logicalType)) {
    mlir::Value zero = createI64Constant(loc, rewriter, 0);
    mlir::Value asBool = rewriter.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, value, zero);
    return runtime
        .call(loc, RuntimeSymbols::kBoolFromBool,
              typeConverter.getPyObjectPtrType(), mlir::ValueRange{asBool})
        .getResult();
  }

  if (mlir::isa<mlir::FloatType>(logicalType))
    return runtime
        .call(loc, RuntimeSymbols::kFloatFromDouble,
              typeConverter.getPyObjectPtrType(),
              mlir::ValueRange{rewriter.create<mlir::arith::BitcastOp>(
                  loc, rewriter.getF64Type(), value)})
        .getResult();

  if (mlir::isa<NoneType, StrType, ObjectType, ClassType, ExceptionType,
                TracebackType, LocationType>(logicalType)) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
    return rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, value)
        .getResult();
  }

  return mlir::failure();
}

mlir::Value Slot::storage(mlir::Location loc, mlir::Value value,
                          mlir::Type logicalType, mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  mlir::Type storageType =
      container::Slot::storageType(logicalType, rewriter.getContext());
  if (!storageType)
    return {};

  if (value.getType() == storageType)
    return value;

  if (mlir::isa<BoolType>(logicalType) && storageType == rewriter.getI8Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intType.getWidth() > 8)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, value);
      if (intType.getWidth() < 8)
        return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, value);
    }
    RuntimeAPI runtime(module, rewriter, typeConverter);
    mlir::Value raw = runtime
                          .call(loc, RuntimeSymbols::kBoolAsBool,
                                rewriter.getI1Type(), mlir::ValueRange{value})
                          .getResult();
    return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, raw);
  }

  if (mlir::isa<mlir::FloatType>(logicalType) &&
      storageType == rewriter.getF64Type()) {
    if (value.getType() == rewriter.getI64Type())
      return rewriter.create<mlir::arith::BitcastOp>(loc, storageType, value);
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kFloatAsDouble, storageType,
              mlir::ValueRange{value})
        .getResult();
  }

  if (mlir::isa<IntType>(logicalType) && storageType == rewriter.getI64Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intType.getWidth() < 64)
        return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, value);
      if (intType.getWidth() > 64)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, value);
    }
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime
        .call(loc, RuntimeSymbols::kLongAsI64, storageType,
              mlir::ValueRange{value})
        .getResult();
  }

  mlir::FailureOr<mlir::Value> packed =
      Slot::pack(loc, value, logicalType, module, rewriter, typeConverter);
  if (mlir::failed(packed))
    return {};
  mlir::Value result = *packed;
  if (result.getType() == storageType)
    return result;
  if (mlir::isa<BoolType>(logicalType) && storageType == rewriter.getI8Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(result.getType())) {
      if (intType.getWidth() > 8)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, result);
      if (intType.getWidth() < 8)
        return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, result);
    }
  }
  if (mlir::isa<mlir::FloatType>(logicalType) &&
      storageType == rewriter.getF64Type() &&
      result.getType() == rewriter.getI64Type())
    return rewriter.create<mlir::arith::BitcastOp>(loc, storageType, result);
  if (storageType == rewriter.getI64Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(result.getType())) {
      if (intType.getWidth() < 64)
        return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, result);
      if (intType.getWidth() > 64)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, result);
    }
  }
  return {};
}

mlir::FailureOr<llvm::SmallVector<mlir::Value>> container::Descriptor::promote(
    mlir::Location loc, mlir::Type logicalType, mlir::ValueRange input,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (auto listType = mlir::dyn_cast<ListType>(logicalType))
    return promoteListDescriptor(loc, input, listType, module, rewriter,
                                 typeConverter, cloneReferenceSlots);
  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType))
    return promoteTupleDescriptor(loc, input, tupleType, module, rewriter,
                                  typeConverter, cloneReferenceSlots);
  if (auto dictType = mlir::dyn_cast<DictType>(logicalType))
    return promoteDictDescriptor(loc, input, dictType, module, rewriter,
                                 typeConverter, cloneReferenceSlots);
  return mlir::failure();
}

mlir::FailureOr<mlir::Value>
Slot::boxStorage(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
                 mlir::ModuleOp module,
                 mlir::ConversionPatternRewriter &rewriter,
                 const PyLLVMTypeConverter &typeConverter) {
  mlir::Value slot = value;
  if (mlir::isa<BoolType>(logicalType)) {
    if (slot.getType() != rewriter.getI64Type())
      slot = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(),
                                                   slot);
  } else if (mlir::isa<mlir::FloatType>(logicalType)) {
    if (slot.getType() == rewriter.getF64Type())
      slot = rewriter.create<mlir::arith::BitcastOp>(loc, rewriter.getI64Type(),
                                                     slot);
  } else if (slot.getType() != rewriter.getI64Type() &&
             mlir::isa<mlir::IntegerType>(slot.getType())) {
    auto intType = mlir::cast<mlir::IntegerType>(slot.getType());
    if (intType.getWidth() < 64)
      slot = rewriter.create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(),
                                                   slot);
    else if (intType.getWidth() > 64)
      slot = rewriter.create<mlir::arith::TruncIOp>(loc, rewriter.getI64Type(),
                                                    slot);
  }
  return Slot::box(loc, slot, logicalType, module, rewriter, typeConverter);
}

void Slot::classRefcount(mlir::Location loc, mlir::Value slot, ClassType type,
                         mlir::ModuleOp module, mlir::OpBuilder &rewriter,
                         llvm::StringRef suffix, bool aggregateEffect,
                         llvm::StringRef retainPremise) {
  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value asPtr =
      rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, slot);
  auto helper = getOrInsertLLVMFunc(
      loc, module, rewriter, getClassHelperName(type, suffix),
      mlir::LLVM::LLVMVoidType::get(rewriter.getContext()), {ptrType});
  ownership::llvm_func::Contract::apply(helper, helper.getName());
  if (suffix == "incref" || suffix == "decref") {
    mlir::OpBuilder attrBuilder(helper.getContext());
    helper->setAttr(ClassSafetyAttrs::kHelperKind,
                    attrBuilder.getStringAttr(
                        suffix == "incref" ? ClassSafetyAttrs::kKindIncref
                                           : ClassSafetyAttrs::kKindDecref));
    helper->setAttr(ClassSafetyAttrs::kHelperClass,
                    attrBuilder.getStringAttr(type.getClassName()));
  }
  auto helperRef =
      mlir::SymbolRefAttr::get(module.getContext(), helper.getName());
  auto call = rewriter.create<mlir::LLVM::CallOp>(
      loc, mlir::TypeRange{}, helperRef, mlir::ValueRange{asPtr});
  if (suffix == "incref")
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  if (aggregateEffect) {
    ownership::aggregate::Slot::markLoad(slot);
    call->setAttr(suffix == "incref"
                      ? OwnershipContractAttrs::kAggregateRetain
                      : OwnershipContractAttrs::kAggregateRelease,
                  rewriter.getUnitAttr());
  }
}

bool Slot::refcounted(mlir::Type logicalType) {
  return mlir::isa<ClassType, StrType, ObjectType, ExceptionType, TracebackType,
                   LocationType>(logicalType);
}

void Slot::markTransfer(mlir::Operation *storeLike) {
  if (!storeLike)
    return;
  ownership::aggregate::Slot::markStore(storeLike);
}

static bool isSmallIntegerConstant(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::LLVM::ConstantOp>();
  if (!constant)
    return false;
  auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!intAttr)
    return false;
  int64_t raw = intAttr.getInt();
  return raw >= -5 && raw <= 256;
}

static bool isKnownImmortalTypedStorageSource(mlir::Value source) {
  auto call = source.getDefiningOp<mlir::LLVM::CallOp>();
  if (!call)
    return false;
  auto callee = call.getCallee();
  if (!callee)
    return false;
  if (*callee == RuntimeSymbols::kBoolFromBool)
    return true;
  if (*callee == RuntimeSymbols::kStrInternStaticUtf8)
    return true;
  if (*callee == RuntimeSymbols::kLongFromI64)
    return call.getNumOperands() == 1 &&
           isSmallIntegerConstant(call.getOperand(0));
  return false;
}

void Slot::releaseSource(mlir::Location loc, mlir::Value source,
                         mlir::Type logicalType, mlir::ModuleOp module,
                         mlir::ConversionPatternRewriter &rewriter,
                         const PyLLVMTypeConverter &typeConverter) {
  if (Slot::refcounted(logicalType))
    return;
  if (mlir::isa<NoneType>(logicalType))
    return;
  if (!source || !mlir::isa<mlir::LLVM::LLVMPointerType>(source.getType()))
    return;
  if (isKnownImmortalTypedStorageSource(source))
    return;

  RuntimeAPI runtime(module, rewriter, typeConverter);
  runtime.call(loc, RuntimeSymbols::kDecRef, /*resultType=*/nullptr,
               mlir::ValueRange{source});
}

void Slot::refcount(mlir::Location loc, mlir::Value slot,
                    mlir::Type logicalType, mlir::ModuleOp module,
                    mlir::OpBuilder &rewriter,
                    const PyLLVMTypeConverter &typeConverter,
                    llvm::StringRef suffix, bool aggregateEffect,
                    llvm::StringRef retainPremise) {
  if (!Slot::refcounted(logicalType))
    return;

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    Slot::classRefcount(loc, slot, classType, module, rewriter, suffix,
                        aggregateEffect, retainPremise);
    return;
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(rewriter.getContext());
  mlir::Value asPtr = slot;
  if (asPtr.getType() != ptrType)
    asPtr = rewriter.create<mlir::LLVM::IntToPtrOp>(loc, ptrType, slot);

  llvm::StringRef symbol;
  if (suffix == "incref")
    symbol = RuntimeSymbols::kIncRef;
  else if (suffix == "decref")
    symbol = RuntimeSymbols::kDecRef;
  else
    return;

  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto call = runtime.call(loc, symbol, /*resultType=*/nullptr,
                           mlir::ValueRange{asPtr});
  if (suffix == "incref")
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  if (aggregateEffect) {
    ownership::aggregate::Slot::markLoad(slot);
    call->setAttr(suffix == "incref"
                      ? OwnershipContractAttrs::kAggregateRetain
                      : OwnershipContractAttrs::kAggregateRelease,
                  rewriter.getUnitAttr());
  }
}

void container::Elements::refcount(mlir::Location loc, mlir::Type logicalType,
                                   mlir::ValueRange descriptor,
                                   mlir::ModuleOp module,
                                   mlir::OpBuilder &builder,
                                   const PyLLVMTypeConverter &typeConverter,
                                   llvm::StringRef suffix) {
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    mlir::Type elementType = listType.getElementType();
    if (!::py::Slot::refcounted(elementType) || descriptor.size() != 2)
      return;

    mlir::Value size = builder.create<mlir::memref::LoadOp>(
        loc, descriptor[0], createIndexConstant(loc, builder, 0));
    mlir::Value upper = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), size);
    mlir::Value lower = createIndexConstant(loc, builder, 0);
    mlir::Value step = createIndexConstant(loc, builder, 1);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      mlir::Value slot = builder.create<mlir::memref::LoadOp>(
          loc, descriptor[1], loop.getInductionVar());
      ::py::Slot::refcount(loc, slot, elementType, module, builder,
                           typeConverter, suffix,
                           /*aggregateEffect=*/suffix == "decref");
    }
    return;
  }

  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    if (descriptor.size() != 2)
      return;
    for (auto [index, elementType] :
         llvm::enumerate(tupleType.getElementTypes())) {
      if (!::py::Slot::refcounted(elementType))
        continue;
      mlir::Value slot = builder.create<mlir::memref::LoadOp>(
          loc, descriptor[1],
          createIndexConstant(loc, builder, static_cast<int64_t>(index)));
      ::py::Slot::refcount(loc, slot, elementType, module, builder,
                           typeConverter, suffix,
                           /*aggregateEffect=*/suffix == "decref");
    }
    return;
  }

  if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    bool keyRefcounted = ::py::Slot::refcounted(dictType.getKeyType());
    bool valueRefcounted = ::py::Slot::refcounted(dictType.getValueType());
    if ((!keyRefcounted && !valueRefcounted) || descriptor.size() != 4)
      return;

    mlir::Value capacity = builder.create<mlir::memref::LoadOp>(
        loc, descriptor[0], createIndexConstant(loc, builder, 1));
    mlir::Value upper = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), capacity);
    mlir::Value lower = createIndexConstant(loc, builder, 0);
    mlir::Value step = createIndexConstant(loc, builder, 1);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      mlir::Value state = builder.create<mlir::memref::LoadOp>(
          loc, descriptor[3], loop.getInductionVar());
      mlir::Value occupied = builder.create<mlir::arith::CmpIOp>(
          loc, mlir::arith::CmpIPredicate::eq, state,
          builder.create<mlir::arith::ConstantIntOp>(loc, 1,
                                                     builder.getI8Type()));
      auto ifOp = builder.create<mlir::scf::IfOp>(loc, occupied,
                                                  /*withElseRegion=*/false);
      {
        mlir::OpBuilder::InsertionGuard ifGuard(builder);
        builder.setInsertionPointToStart(ifOp.thenBlock());
        if (keyRefcounted) {
          mlir::Value key = builder.create<mlir::memref::LoadOp>(
              loc, descriptor[1], loop.getInductionVar());
          ::py::Slot::refcount(loc, key, dictType.getKeyType(), module, builder,
                               typeConverter, suffix,
                               /*aggregateEffect=*/suffix == "decref");
        }
        if (valueRefcounted) {
          mlir::Value value = builder.create<mlir::memref::LoadOp>(
              loc, descriptor[2], loop.getInductionVar());
          ::py::Slot::refcount(loc, value, dictType.getValueType(), module,
                               builder, typeConverter, suffix,
                               /*aggregateEffect=*/suffix == "decref");
        }
      }
    }
  }
}

bool container::Managed::mutableArgument(mlir::Operation *op,
                                         mlir::Value value) {
  auto index = getLogicalArgIndex(op, value);
  if (!index)
    return false;
  auto parentFunc = op->getParentOfType<mlir::func::FuncOp>();
  return parentFunc &&
         arrayAttrContainsIndex(
             parentFunc->getAttrOfType<mlir::ArrayAttr>("ly.mutable_args"),
             *index);
}

mlir::Value container::Managed::predicate(mlir::Location loc,
                                          mlir::Value header,
                                          int64_t refcountSlot,
                                          mlir::OpBuilder &builder) {
  auto atomic = builder.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::addi, createI64Constant(loc, builder, 0),
      header,
      mlir::ValueRange{createIndexConstant(loc, builder, refcountSlot)});
  threadsafe::Atomic::set(atomic.getOperation(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  container::Header::markAtomicResource(atomic.getOperation(), header,
                                        refcountSlot);
  mlir::Value marker = atomic;
  return builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, marker,
      createI64Constant(loc, builder, 0));
}

void container::Managed::lock(mlir::Location loc, mlir::Value header,
                              int64_t lockSlot, mlir::OpBuilder &builder) {
  mlir::Value lockIndex = createIndexConstant(loc, builder, lockSlot);
  auto whileOp = builder.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{},
                                                    mlir::ValueRange{});
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);
    auto atomic = builder.create<mlir::memref::AtomicRMWOp>(
        loc, mlir::arith::AtomicRMWKind::assign,
        createI64Constant(loc, builder, 1), header,
        mlir::ValueRange{lockIndex});
    threadsafe::Atomic::set(atomic.getOperation(),
                            ThreadSafetyAttrs::kRoleContainerLockAcquire,
                            ThreadSafetyAttrs::kOrderingAcquire);
    container::Header::markAtomicResource(atomic.getOperation(), header,
                                          lockSlot);
    mlir::Value previous = atomic;
    mlir::Value busy = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, previous,
        createI64Constant(loc, builder, 0));
    builder.create<mlir::scf::ConditionOp>(loc, busy, mlir::ValueRange{});
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *after = builder.createBlock(&whileOp.getAfter());
    builder.setInsertionPointToStart(after);
    builder.create<mlir::scf::YieldOp>(loc);
  }
}

void container::Managed::unlock(mlir::Location loc, mlir::Value header,
                                int64_t lockSlot, mlir::OpBuilder &builder) {
  auto atomic = builder.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::assign,
      createI64Constant(loc, builder, 0), header,
      mlir::ValueRange{createIndexConstant(loc, builder, lockSlot)});
  threadsafe::Atomic::set(atomic.getOperation(),
                          ThreadSafetyAttrs::kRoleContainerLockRelease,
                          ThreadSafetyAttrs::kOrderingRelease);
  container::Header::markAtomicResource(atomic.getOperation(), header,
                                        lockSlot);
}

} // namespace py
