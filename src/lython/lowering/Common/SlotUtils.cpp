#include "Common/SlotUtils.h"

#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>
#include <string>

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

static mlir::Value descriptorAttrSource(mlir::Value value) {
  llvm::SmallPtrSet<mlir::Operation *, 8> seen;
  while (value) {
    if (auto publish = value.getDefiningOp<PublishOp>()) {
      value = publish.getInput();
      continue;
    }

    auto result = mlir::dyn_cast<mlir::OpResult>(value);
    auto cast = result
                    ? mlir::dyn_cast_or_null<mlir::UnrealizedConversionCastOp>(
                          result.getOwner())
                    : mlir::UnrealizedConversionCastOp();
    if (!cast || !seen.insert(cast.getOperation()).second)
      return value;

    unsigned resultIndex = result.getResultNumber();
    if (cast->getNumOperands() == cast->getNumResults() &&
        resultIndex < cast->getNumOperands()) {
      value = cast.getOperand(resultIndex);
      continue;
    }

    if (cast->getNumOperands() == 1) {
      mlir::Value source = cast.getOperand(0);
      if (auto sourceCast =
              source.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (resultIndex < sourceCast->getNumOperands()) {
          value = sourceCast.getOperand(resultIndex);
          continue;
        }
      }
      value = source;
      continue;
    }

    return value;
  }
  return {};
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
                             mlir::OpBuilder &builder,
                             bool objectSlotCopy = false) {
  mlir::Value lower = createIndexConstant(loc, builder, 0);
  mlir::Value step = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, count, step);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value value = builder.create<mlir::memref::LoadOp>(loc, source, iv);
    if (objectSlotCopy)
      ownership::aggregate::Slot::markLoad(value);
    auto store = builder.create<mlir::memref::StoreOp>(loc, value, target, iv);
    if (objectSlotCopy)
      Slot::markTransfer(store.getOperation());
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

static void copyAggregateSlotLoadAttrs(mlir::Value source,
                                       mlir::Operation *target) {
  mlir::Operation *sourceOp = source ? source.getDefiningOp() : nullptr;
  if (!sourceOp || !target)
    return;
  llvm::StringRef attrs[] = {OwnershipContractAttrs::kAggregateSlotLoad,
                             OwnershipContractAttrs::kAggregateSlotGroup,
                             OwnershipContractAttrs::kAggregateSlotComponent,
                             OwnershipContractAttrs::kAggregateSlotIndex};
  for (llvm::StringRef attr : attrs)
    if (mlir::Attribute value = sourceOp->getAttr(attr))
      target->setAttr(attr, value);
}

static mlir::Value loadClassObject(mlir::Location loc, mlir::Value value,
                                   mlir::LLVM::LLVMStructType objectType,
                                   mlir::OpBuilder &builder) {
  if (!value || !class_layout::isObjectCarrierType(objectType))
    return {};
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  if (value.getType() == objectType)
    return value;
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      memrefType.getElementType() != objectType)
    return {};
  auto load = builder.create<mlir::memref::LoadOp>(
      loc, value, createIndexConstant(loc, builder, 0));
  load->setAttr(ClassSafetyAttrs::kCarrierLoad, builder.getUnitAttr());
  ownership::aggregate::Slot::markLoad(load.getResult());
  return load;
}

static mlir::Value memRefFromClassPart(mlir::Location loc,
                                       mlir::Value descriptor,
                                       mlir::MemRefType memrefType,
                                       mlir::OpBuilder &builder) {
  if (!descriptor || !memrefType)
    return {};
  if (descriptor.getType() == memrefType)
    return descriptor;
  if (mlir::isa<mlir::MemRefType>(descriptor.getType()))
    return builder.create<mlir::memref::CastOp>(loc, memrefType, descriptor);
  return builder
      .create<mlir::UnrealizedConversionCastOp>(loc, memrefType,
                                                mlir::ValueRange{descriptor})
      .getResult(0);
}

static bool
appendFlattenedDescriptorPart(mlir::Location loc, mlir::Value object,
                              mlir::LLVM::LLVMStructType objectType,
                              int64_t objectPart, mlir::FunctionType fnType,
                              unsigned &inputOffset, mlir::OpBuilder &builder,
                              llvm::SmallVectorImpl<mlir::Value> &args) {
  auto descriptorType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
      class_layout::Object::descriptorType(objectType, objectPart));
  if (!descriptorType || descriptorType.isOpaque())
    return false;

  for (auto [componentIndex, componentType] :
       llvm::enumerate(descriptorType.getBody())) {
    if (auto arrayType =
            mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(componentType)) {
      for (int64_t element = 0, end = arrayType.getNumElements(); element < end;
           ++element) {
        if (inputOffset >= fnType.getNumInputs() ||
            fnType.getInput(inputOffset) != arrayType.getElementType())
          return false;
        args.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
            loc, arrayType.getElementType(), object,
            builder.getDenseI64ArrayAttr(
                {objectPart, static_cast<int64_t>(componentIndex), element})));
        ++inputOffset;
      }
      continue;
    }

    if (inputOffset >= fnType.getNumInputs() ||
        fnType.getInput(inputOffset) != componentType)
      return false;
    args.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
        loc, componentType, object,
        builder.getDenseI64ArrayAttr(
            {objectPart, static_cast<int64_t>(componentIndex)})));
    ++inputOffset;
  }
  return true;
}

static bool appendClassObjectPartsForHelper(
    mlir::Location loc, mlir::Value object,
    mlir::LLVM::LLVMStructType objectType, mlir::func::FuncOp helper,
    unsigned inputOffset, mlir::OpBuilder &builder,
    llvm::SmallVectorImpl<mlir::Value> &args, mlir::Value &objectForAttrs,
    unsigned *consumed = nullptr) {
  mlir::FunctionType fnType = helper.getFunctionType();
  int64_t objectPartCount = class_layout::Object::partCount(objectType);
  if (objectPartCount < 2 ||
      fnType.getNumInputs() < inputOffset + objectPartCount ||
      !class_layout::isObjectCarrierType(objectType))
    return false;

  if (!object || object.getType() != objectType)
    return false;
  objectForAttrs = object;

  llvm::SmallVector<mlir::MemRefType, 4> memrefTypes;
  memrefTypes.reserve(static_cast<size_t>(objectPartCount));
  for (int64_t index = 0; index < objectPartCount; ++index) {
    auto memrefType =
        mlir::dyn_cast<mlir::MemRefType>(fnType.getInput(inputOffset + index));
    if (!memrefType)
      break;
    memrefTypes.push_back(memrefType);
  }
  if (memrefTypes.size() == static_cast<size_t>(objectPartCount)) {
    for (auto [index, memrefType] : llvm::enumerate(memrefTypes)) {
      mlir::Value descriptor = class_layout::Object::descriptor(
          loc, objectType, object, static_cast<int64_t>(index), builder);
      mlir::Value memref =
          memRefFromClassPart(loc, descriptor, memrefType, builder);
      if (!memref)
        return false;
      args.push_back(memref);
    }
    if (consumed)
      *consumed = static_cast<unsigned>(objectPartCount);
    return true;
  }

  unsigned cursor = inputOffset;
  for (int64_t index : {class_layout::Object::kHeaderIndex,
                        class_layout::Object::kPayloadIndex}) {
    if (!appendFlattenedDescriptorPart(loc, object, objectType, index, fnType,
                                       cursor, builder, args))
      return false;
  }
  if (consumed)
    *consumed = cursor - inputOffset;
  return true;
}

static bool appendClassPartsForHelper(mlir::Location loc, mlir::Value slotView,
                                      mlir::LLVM::LLVMStructType objectType,
                                      mlir::func::FuncOp helper,
                                      unsigned inputOffset,
                                      mlir::OpBuilder &builder,
                                      llvm::SmallVectorImpl<mlir::Value> &args,
                                      mlir::Value &objectForAttrs,
                                      unsigned *consumed = nullptr) {
  mlir::Value object = loadClassObject(loc, slotView, objectType, builder);
  return appendClassObjectPartsForHelper(loc, object, objectType, helper,
                                         inputOffset, builder, args,
                                         objectForAttrs, consumed);
}

static bool classPartsForHelper(mlir::Location loc, mlir::Value slotView,
                                mlir::LLVM::LLVMStructType objectType,
                                mlir::func::FuncOp helper,
                                mlir::OpBuilder &builder,
                                llvm::SmallVectorImpl<mlir::Value> &args,
                                mlir::Value &objectForAttrs) {
  unsigned consumed = 0;
  if (!appendClassPartsForHelper(loc, slotView, objectType, helper,
                                 /*inputOffset=*/0, builder, args,
                                 objectForAttrs, &consumed))
    return false;
  return consumed == helper.getFunctionType().getNumInputs();
}

static bool inlineClassListStorage(ListType listType,
                                   mlir::MemRefType itemsType,
                                   ClassType &classType,
                                   mlir::LLVM::LLVMStructType &objectType) {
  classType = mlir::dyn_cast<ClassType>(listType.getElementType());
  if (!classType)
    return false;
  objectType = class_layout::objectCarrierType(itemsType);
  return static_cast<bool>(objectType);
}

static bool
objectPartsStorageTypes(mlir::Type logicalType, mlir::MLIRContext *ctx,
                        llvm::SmallVectorImpl<mlir::MemRefType> &partTypes) {
  partTypes.clear();
  llvm::SmallVector<mlir::Type, 2> storageTypes;
  if (mlir::isa<StrType>(logicalType)) {
    object_abi::str_abi::Parts::storageTypes(ctx, storageTypes);
  } else if (mlir::isa<ExceptionType>(logicalType)) {
    object_abi::exception_abi::Parts::storageTypes(ctx, storageTypes);
  } else {
    return false;
  }
  for (mlir::Type storageType : storageTypes) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(storageType);
    if (!memrefType)
      return false;
    partTypes.push_back(memrefType);
  }
  return partTypes.size() >= 2;
}

static mlir::LLVM::LLVMStructType
objectPartsCarrierType(mlir::Type logicalType, mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::MemRefType, 2> partTypes;
  if (!objectPartsStorageTypes(logicalType, ctx, partTypes))
    return {};
  return class_layout::objectCarrierType(ctx, partTypes);
}

static mlir::Value
materializeClonedContainerSlot(mlir::Location loc, mlir::Value slot,
                               mlir::Type logicalType, mlir::ModuleOp module,
                               mlir::ConversionPatternRewriter &rewriter,
                               const PyLLVMTypeConverter &typeConverter) {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    (void)classType;
    ownership::aggregate::Slot::markLoad(slot);
    return {};
  }

  if (mlir::failed(Slot::refcount(loc, slot, logicalType, module, rewriter,
                                  typeConverter, "incref",
                                  /*aggregateEffect=*/true,
                                  ThreadSafetyAttrs::kPremiseAggregateBorrow)))
    return {};
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

static mlir::Value classCarrierForHelper(mlir::Location loc, mlir::Value value,
                                         mlir::LLVM::LLVMStructType objectType,
                                         mlir::OpBuilder &builder) {
  if (!value)
    return {};
  if (value.getType() == objectType)
    return value;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
    if (memrefType.getRank() == 1 && memrefType.getElementType() == objectType)
      return value;
  }
  if (auto descriptorType =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(value.getType())) {
    if (class_layout::isDescriptorStorageType(descriptorType))
      return value;
  }
  return {};
}

static int64_t classLayoutId(ClassType classType) {
  uint64_t hash = 1469598103934665603ULL;
  for (char ch : classType.getClassName()) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  }
  return static_cast<int64_t>(hash & 0x7fffffffffffffffULL);
}

static mlir::Value
loadClassStorageFromDescriptor(mlir::Location loc, mlir::Value value,
                               mlir::LLVM::LLVMStructType objectType,
                               mlir::OpBuilder &builder) {
  if (!class_layout::isObjectCarrierType(objectType))
    return {};
  if (value && value.getType() == objectType)
    return value;
  if (!value || !objectType)
    return {};

  while (true) {
    if (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
      if (cast->getNumOperands() != 1)
        break;
      value = cast.getOperand(0);
      continue;
    }
    if (auto publish = value.getDefiningOp<PublishOp>()) {
      value = publish.getInput();
      continue;
    }
    break;
  }

  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      memrefType.getElementType() != objectType)
    return {};

  auto load = builder.create<mlir::memref::LoadOp>(
      loc, value, createIndexConstant(loc, builder, 0));
  load->setAttr(ClassSafetyAttrs::kCarrierLoad, builder.getUnitAttr());
  ownership::aggregate::Slot::markLoad(load.getResult());
  return load.getResult();
}

static mlir::Value descriptorFromMemRef(mlir::Location loc, mlir::Value memref,
                                        mlir::MemRefType memrefType,
                                        mlir::OpBuilder &builder) {
  mlir::Type descriptorType =
      class_layout::descriptorStorageType(memrefType, builder.getContext());
  if (!descriptorType)
    return {};
  if (memref.getType() == descriptorType)
    return memref;
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, descriptorType, mlir::ValueRange{memref});
  class_layout::DescriptorShape::mark(cast.getOperation(), memrefType);
  return cast.getResult(0);
}

static mlir::Value memRefView(mlir::Location loc, mlir::Value memref,
                              mlir::MemRefType targetType,
                              mlir::OpBuilder &builder) {
  if (!memref || memref.getType() == targetType)
    return memref;
  if (!mlir::isa<mlir::MemRefType>(memref.getType()))
    return {};
  return builder.create<mlir::memref::CastOp>(loc, targetType, memref);
}

static mlir::Value memRefDescriptorPresent(mlir::Location loc,
                                           mlir::Value descriptor,
                                           mlir::MemRefType memrefType,
                                           mlir::OpBuilder &builder) {
  if (!descriptor || !memrefType)
    return {};
  mlir::Value memref = descriptor;
  if (memref.getType() != memrefType) {
    if (mlir::Operation *def = descriptor.getDefiningOp())
      if (class_layout::DescriptorShape::has(def) &&
          mlir::failed(class_layout::DescriptorShape::verify(
              def, memrefType, "class carrier descriptor")))
        return {};
    auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
        loc, memrefType, mlir::ValueRange{descriptor});
    class_layout::DescriptorShape::mark(cast.getOperation(), memrefType);
    memref = cast.getResult(0);
  }
  auto metadata =
      builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, memref);
  mlir::Value base =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
          loc, metadata.getBaseBuffer());
  mlir::Value storage =
      builder.create<mlir::arith::AddIOp>(loc, base, metadata.getOffset());
  mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  return builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, storage, zero);
}

static mlir::Value memRefFromDescriptor(mlir::Location loc,
                                        mlir::Value descriptor,
                                        mlir::MemRefType memrefType,
                                        mlir::OpBuilder &builder) {
  if (!descriptor || !memrefType)
    return {};
  if (descriptor.getType() == memrefType)
    return descriptor;
  if (mlir::Operation *def = descriptor.getDefiningOp())
    if (class_layout::DescriptorShape::has(def) &&
        mlir::failed(class_layout::DescriptorShape::verify(
            def, memrefType, "class carrier descriptor")))
      return {};
  if (mlir::isa<mlir::MemRefType>(descriptor.getType()))
    return memRefView(loc, descriptor, memrefType, builder);
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, memrefType, mlir::ValueRange{descriptor});
  class_layout::DescriptorShape::mark(cast.getOperation(), memrefType);
  return cast.getResult(0);
}

static void storeHeaderSlot(mlir::Location loc, mlir::Value header,
                            int64_t slot, mlir::Value value,
                            mlir::OpBuilder &builder) {
  builder.create<mlir::memref::StoreOp>(
      loc, value, header,
      mlir::ValueRange{createIndexConstant(loc, builder, slot)});
}

static std::optional<llvm::StringRef>
descriptorStringAttr(mlir::Value value, llvm::StringRef attrName) {
  value = descriptorAttrSource(value);
  value = stripContainerAccessCasts(value);
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Operation *parent =
        arg.getOwner() ? arg.getOwner()->getParentOp() : nullptr;
    auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
    if (function && arg.getArgNumber() < function.getNumArguments()) {
      auto attr = mlir::dyn_cast_or_null<mlir::StringAttr>(
          function.getArgAttr(arg.getArgNumber(), attrName));
      if (attr)
        return attr.getValue();
    }

    auto argAttrs = parent ? parent->getAttrOfType<mlir::ArrayAttr>("arg_attrs")
                           : mlir::ArrayAttr();
    if (!argAttrs || arg.getArgNumber() >= argAttrs.size())
      return std::nullopt;
    auto dict =
        mlir::dyn_cast<mlir::DictionaryAttr>(argAttrs[arg.getArgNumber()]);
    if (!dict)
      return std::nullopt;
    auto attr = mlir::dyn_cast_or_null<mlir::StringAttr>(dict.get(attrName));
    if (!attr)
      return std::nullopt;
    return attr.getValue();
  }
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto attr = def->getAttrOfType<mlir::StringAttr>(attrName);
  if (!attr)
    return std::nullopt;
  return attr.getValue();
}

static void markLockAtomicResource(mlir::Operation *op, mlir::Value header,
                                   mlir::Value lock) {
  if (!op || !lock)
    return;
  mlir::Value source = header ? header : lock;
  auto group =
      descriptorStringAttr(source, ContainerSafetyAttrs::kDescriptorGroup);
  auto kind =
      descriptorStringAttr(source, ContainerSafetyAttrs::kDescriptorKind);
  if (!group || !kind)
    return;
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentLock,
                                  kTypedContainerLockSlot, group->str(), *kind);
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoteListDescriptor(
    mlir::Location loc, mlir::ValueRange input, ListType listType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool cloneReferenceSlots) {
  if (input.size() != 3)
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(input[0].getType());
  auto lockType = mlir::dyn_cast<mlir::MemRefType>(input[1].getType());
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(input[2].getType());
  if (!headerType || !lockType || !itemsType)
    return mlir::failure();
  mlir::Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return mlir::failure();

  mlir::Value managedHeader =
      rewriter.create<mlir::memref::AllocOp>(loc, headerType);
  mlir::Value managedLock =
      rewriter.create<mlir::memref::AllocOp>(loc, lockType);
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
  ClassType inlineClassType;
  mlir::LLVM::LLVMStructType inlineCarrierType;
  bool inlineClass = inlineClassListStorage(listType, itemsType,
                                            inlineClassType, inlineCarrierType);

  std::string descriptorGroup =
      container::descriptor::Group::make(managedHeader.getDefiningOp(), "list");
  container::descriptor::Component::mark(
      managedHeader, descriptorGroup, ContainerSafetyAttrs::kComponentHeader);
  container::descriptor::Component::mark(managedLock, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentLock);
  container::descriptor::Component::mark(managedItems, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentItems);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  auto lockZero = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, 0, rewriter.getI32Type());
  rewriter.create<mlir::memref::StoreOp>(
      loc, lockZero, managedLock,
      createIndexConstant(loc, rewriter, kTypedContainerLockSlot));
  if (!inlineClass)
    copyMemRefPrefix(loc, input[2], managedItems, sizeIndex, rewriter,
                     Slot::refcounted(listType.getElementType()) &&
                         cloneReferenceSlots);
  storeI64HeaderSlot(loc, managedHeader, kTypedListRefcountSlot, 1, rewriter);

  if (Slot::refcounted(listType.getElementType()) && cloneReferenceSlots) {
    mlir::Value lower = createIndexConstant(loc, rewriter, 0);
    mlir::Value step = createIndexConstant(loc, rewriter, 1);
    auto loop = rewriter.create<mlir::scf::ForOp>(loc, lower, sizeIndex, step);
    {
      mlir::OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(loop.getBody());
      if (inlineClass) {
        mlir::Value sourceSlot = Slot::classCarrierView(
            loc, input[2], loop.getInductionVar(), inlineCarrierType, rewriter);
        mlir::Value destSlot =
            Slot::classCarrierView(loc, managedItems, loop.getInductionVar(),
                                   inlineCarrierType, rewriter);
        if (!sourceSlot || !destSlot)
          return mlir::failure();
        if (mlir::failed(Slot::classCarrierInitialize(loc, destSlot,
                                                      inlineClassType, module,
                                                      rewriter, typeConverter)))
          return mlir::failure();
        if (mlir::failed(Slot::classCarrierCopy(
                loc, destSlot, sourceSlot, inlineClassType, module, rewriter)))
          return mlir::failure();
      } else {
        mlir::Value slot = rewriter.create<mlir::memref::LoadOp>(
            loc, managedItems, loop.getInductionVar());
        mlir::Value promoted = materializePromotedContainerSlot(
            loc, slot, listType.getElementType(), module, rewriter,
            typeConverter, cloneReferenceSlots);
        auto store = rewriter.create<mlir::memref::StoreOp>(
            loc, promoted, managedItems, loop.getInductionVar());
        Slot::markTransfer(store.getOperation());
      }
    }
  }

  return llvm::SmallVector<mlir::Value>{managedHeader, managedLock,
                                        managedItems};
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
  auto inlineCarrierType = class_layout::objectCarrierType(itemsType);

  std::string descriptorGroup = container::descriptor::Group::make(
      managedHeader.getDefiningOp(), "tuple");
  container::descriptor::Component::mark(
      managedHeader, descriptorGroup, ContainerSafetyAttrs::kComponentHeader);
  container::descriptor::Component::mark(managedItems, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentItems);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  if (!inlineCarrierType)
    copyMemRefPrefix(loc, input[1], managedItems, sizeIndex, rewriter,
                     cloneReferenceSlots &&
                         llvm::any_of(tupleType.getElementTypes(),
                                      [](mlir::Type elementType) {
                                        return Slot::refcounted(elementType);
                                      }));
  storeI64HeaderSlot(loc, managedHeader, kTypedTupleRefcountSlot, 1, rewriter);

  if (cloneReferenceSlots) {
    for (auto [index, elementType] :
         llvm::enumerate(tupleType.getElementTypes())) {
      if (!Slot::refcounted(elementType))
        continue;
      if (inlineCarrierType) {
        auto classType = mlir::dyn_cast<ClassType>(elementType);
        mlir::Value indexValue =
            createIndexConstant(loc, rewriter, static_cast<int64_t>(index));
        if (classType) {
          mlir::Value objectSlot = Slot::classCarrierView(
              loc, managedItems, indexValue, inlineCarrierType, rewriter);
          mlir::Value sourceSlot = Slot::classCarrierView(
              loc, input[1], indexValue, inlineCarrierType, rewriter);
          if (!objectSlot || !sourceSlot)
            return mlir::failure();
          if (mlir::failed(Slot::classCarrierInitialize(
                  loc, objectSlot, classType, module, rewriter, typeConverter)))
            return mlir::failure();
          if (mlir::failed(Slot::classCarrierCopy(loc, objectSlot, sourceSlot,
                                                  classType, module, rewriter)))
            return mlir::failure();
          continue;
        }

        mlir::Value slot =
            rewriter.create<mlir::memref::LoadOp>(loc, input[1], indexValue);
        mlir::Value promoted = materializePromotedContainerSlot(
            loc, slot, elementType, module, rewriter, typeConverter,
            cloneReferenceSlots);
        if (!promoted)
          return mlir::failure();
        auto store = rewriter.create<mlir::memref::StoreOp>(
            loc, promoted, managedItems, indexValue);
        Slot::markTransfer(store.getOperation());
        continue;
      }
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
  if (input.size() != 5)
    return mlir::failure();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(input[0].getType());
  auto lockType = mlir::dyn_cast<mlir::MemRefType>(input[1].getType());
  auto keysType = mlir::dyn_cast<mlir::MemRefType>(input[2].getType());
  auto valuesType = mlir::dyn_cast<mlir::MemRefType>(input[3].getType());
  auto statesType = mlir::dyn_cast<mlir::MemRefType>(input[4].getType());
  if (!headerType || !lockType || !keysType || !valuesType || !statesType)
    return mlir::failure();
  mlir::Value headerSize = staticMemRefSize(loc, headerType, rewriter);
  if (!headerSize)
    return mlir::failure();

  mlir::Value managedHeader =
      rewriter.create<mlir::memref::AllocOp>(loc, headerType);
  mlir::Value managedLock =
      rewriter.create<mlir::memref::AllocOp>(loc, lockType);
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
  container::descriptor::Component::mark(managedLock, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentLock);
  container::descriptor::Component::mark(managedKeys, descriptorGroup,
                                         ContainerSafetyAttrs::kComponentKeys);
  container::descriptor::Component::mark(
      managedValues, descriptorGroup, ContainerSafetyAttrs::kComponentValues);
  container::descriptor::Component::mark(
      managedStates, descriptorGroup, ContainerSafetyAttrs::kComponentStates);
  copyMemRefPrefix(loc, input[0], managedHeader, headerSize, rewriter);
  auto lockZero = rewriter.create<mlir::arith::ConstantIntOp>(
      loc, 0, rewriter.getI32Type());
  rewriter.create<mlir::memref::StoreOp>(
      loc, lockZero, managedLock,
      createIndexConstant(loc, rewriter, kTypedContainerLockSlot));
  copyMemRefPrefix(loc, input[2], managedKeys, capacityIndex, rewriter,
                   cloneReferenceSlots &&
                       Slot::refcounted(dictType.getKeyType()));
  copyMemRefPrefix(loc, input[3], managedValues, capacityIndex, rewriter,
                   cloneReferenceSlots &&
                       Slot::refcounted(dictType.getValueType()));
  copyMemRefPrefix(loc, input[4], managedStates, capacityIndex, rewriter);
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

  return llvm::SmallVector<mlir::Value>{managedHeader, managedLock, managedKeys,
                                        managedValues, managedStates};
}

} // namespace

mlir::MemRefType
Slot::classCarrierViewType(mlir::LLVM::LLVMStructType objectType,
                           mlir::MLIRContext *ctx) {
  if (!class_layout::isObjectCarrierType(objectType))
    return {};
  return class_layout::carrierType(objectType, ctx);
}

mlir::Value Slot::classCarrierView(mlir::Location loc, mlir::Value items,
                                   mlir::Value index,
                                   mlir::LLVM::LLVMStructType objectType,
                                   mlir::OpBuilder &builder) {
  if (!class_layout::isObjectCarrierType(objectType))
    return {};
  auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
  if (!itemsType || itemsType.getRank() != 1 ||
      itemsType.getElementType() != objectType)
    return {};
  mlir::Value indexValue = index;
  if (!indexValue.getType().isIndex()) {
    if (!mlir::isa<mlir::IntegerType>(indexValue.getType()))
      return {};
    indexValue = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), indexValue);
  }
  llvm::SmallVector<mlir::OpFoldResult, 1> offsets{indexValue};
  llvm::SmallVector<mlir::OpFoldResult, 1> sizes{builder.getIndexAttr(1)};
  llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
  return builder.create<mlir::memref::SubViewOp>(
      loc, Slot::classCarrierViewType(objectType, builder.getContext()), items,
      offsets, sizes, strides);
}

mlir::Value Slot::classCarrierFromValues(mlir::Location loc,
                                         mlir::ValueRange values,
                                         mlir::LLVM::LLVMStructType objectType,
                                         mlir::OpBuilder &builder) {
  if (class_layout::isObjectCarrierType(objectType) &&
      values.size() ==
          static_cast<size_t>(class_layout::Object::partCount(objectType))) {
    llvm::SmallVector<mlir::Value, 4> descriptors;
    for (auto [index, value] : llvm::enumerate(values)) {
      mlir::Type partType =
          class_layout::Object::descriptorType(objectType, index);
      if (!partType)
        return {};
      mlir::Value descriptor = value;
      if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType()))
        descriptor = descriptorFromMemRef(loc, value, memrefType, builder);
      if (!descriptor)
        return {};
      if (descriptor.getType() != partType)
        descriptor = builder
                         .create<mlir::UnrealizedConversionCastOp>(
                             loc, partType, mlir::ValueRange{descriptor})
                         .getResult(0);
      descriptors.push_back(descriptor);
    }
    return class_layout::Object::fromDescriptors(loc, objectType, descriptors,
                                                 builder);
  }

  if (values.size() != 1)
    return {};
  mlir::Value value = values.front();
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  if (value.getType() == objectType) {
    auto storageType = mlir::MemRefType::get({1}, objectType);
    mlir::Value storage = builder.create<mlir::memref::AllocaOp>(
        loc, storageType, mlir::ValueRange{});
    builder.create<mlir::memref::StoreOp>(loc, value, storage,
                                          createIndexConstant(loc, builder, 0));
    return Slot::classCarrierView(loc, storage,
                                  createIndexConstant(loc, builder, 0),
                                  objectType, builder);
  }
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      memrefType.getElementType() != objectType)
    return {};
  auto canonicalType =
      Slot::classCarrierViewType(objectType, builder.getContext());
  if (value.getType() == canonicalType)
    return value;
  return builder.create<mlir::memref::CastOp>(loc, canonicalType, value)
      .getResult();
}

mlir::func::CallOp Slot::classCarrierRefcount(
    mlir::Location loc, mlir::Value slotView, ClassType type,
    mlir::ModuleOp module, mlir::OpBuilder &builder, llvm::StringRef suffix,
    bool aggregateEffect, llvm::StringRef retainPremise) {
  auto slotType = mlir::dyn_cast<mlir::MemRefType>(slotView.getType());
  auto objectType = class_layout::objectCarrierType(slotView.getType());
  if (!objectType)
    return {};
  auto helper =
      module.lookupSymbol<mlir::func::FuncOp>(getClassHelperName(type, suffix));
  if (!helper)
    return {};
  llvm::SmallVector<mlir::Value, 2> callArgs;
  mlir::Value attrsSource = slotView;
  if (classPartsForHelper(loc, slotView, objectType, helper, builder, callArgs,
                          attrsSource)) {
    auto call = builder.create<mlir::func::CallOp>(loc, helper, callArgs);
    copyAggregateSlotLoadAttrs(attrsSource, call.getOperation());
    if (suffix == "incref")
      threadsafe::Retain::premise(call.getOperation(), retainPremise);
    if (suffix == "destroy_local")
      call->setAttr(OwnershipContractAttrs::kLocalDestroy,
                    builder.getUnitAttr());
    if (aggregateEffect) {
      call->setAttr(suffix == "incref"
                        ? OwnershipContractAttrs::kAggregateRetain
                        : OwnershipContractAttrs::kAggregateRelease,
                    builder.getUnitAttr());
    }
    return call;
  }
  mlir::Value object = slotView;
  mlir::Type helperArgType = helper.getFunctionType().getInput(0);
  if (auto helperCarrierType = class_layout::objectCarrierType(helperArgType)) {
    object = loadClassObject(loc, object, helperCarrierType, builder);
    if (!object)
      return {};
  } else if (object.getType() != helperArgType) {
    auto helperMemRef = mlir::dyn_cast<mlir::MemRefType>(helperArgType);
    if (!slotType || !helperMemRef ||
        slotType.getElementType() != helperMemRef.getElementType() ||
        slotType.getRank() != helperMemRef.getRank())
      return {};
    object = builder.create<mlir::memref::CastOp>(loc, helperMemRef, object);
  }
  if (aggregateEffect)
    ownership::aggregate::Slot::markLoad(object);
  auto call =
      builder.create<mlir::func::CallOp>(loc, helper, mlir::ValueRange{object});
  copyAggregateSlotLoadAttrs(object, call.getOperation());
  if (suffix == "incref")
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  if (suffix == "destroy_local")
    call->setAttr(OwnershipContractAttrs::kLocalDestroy, builder.getUnitAttr());
  if (aggregateEffect) {
    call->setAttr(suffix == "incref"
                      ? OwnershipContractAttrs::kAggregateRetain
                      : OwnershipContractAttrs::kAggregateRelease,
                  builder.getUnitAttr());
  }
  return call;
}

mlir::LogicalResult
Slot::classCarrierCopyTo(mlir::Location loc, mlir::Value destSlotView,
                         mlir::Value sourcePtr, ClassType type,
                         mlir::ModuleOp module, mlir::OpBuilder &builder) {
  auto destType = mlir::dyn_cast<mlir::MemRefType>(destSlotView.getType());
  if (!destType)
    return mlir::failure();
  auto objectType = class_layout::objectCarrierType(destType);
  if (!objectType)
    return mlir::failure();
  mlir::Value source =
      classCarrierForHelper(loc, sourcePtr, objectType, builder);
  if (!source)
    return mlir::failure();
  auto helper =
      module.lookupSymbol<mlir::func::FuncOp>(getClassHelperName(type, "copy"));
  if (!helper)
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> args;
  unsigned consumed = 0;
  mlir::Value attrsSource = destSlotView;
  if (!appendClassPartsForHelper(loc, destSlotView, objectType, helper,
                                 /*inputOffset=*/0, builder, args, attrsSource,
                                 &consumed))
    return mlir::failure();
  mlir::Value sourceAttrs = source;
  if (!appendClassPartsForHelper(loc, source, objectType, helper,
                                 /*inputOffset=*/consumed, builder, args,
                                 sourceAttrs, &consumed))
    return mlir::failure();
  if (args.size() != helper.getFunctionType().getNumInputs())
    return mlir::failure();
  builder.create<mlir::func::CallOp>(loc, helper, mlir::ValueRange{args});
  return mlir::success();
}

mlir::LogicalResult Slot::classCarrierCopyObjectTo(
    mlir::Location loc, mlir::Value destObject, mlir::Value sourcePtr,
    ClassType type, mlir::ModuleOp module, mlir::OpBuilder &builder) {
  auto objectType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(destObject.getType());
  if (!class_layout::isObjectCarrierType(objectType))
    return mlir::failure();
  mlir::Value source =
      classCarrierForHelper(loc, sourcePtr, objectType, builder);
  if (!source)
    return mlir::failure();
  auto helper =
      module.lookupSymbol<mlir::func::FuncOp>(getClassHelperName(type, "copy"));
  if (!helper)
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> args;
  unsigned consumed = 0;
  mlir::Value attrsSource = destObject;
  if (!appendClassObjectPartsForHelper(loc, destObject, objectType, helper,
                                       /*inputOffset=*/0, builder, args,
                                       attrsSource, &consumed))
    return mlir::failure();
  mlir::Value sourceAttrs = source;
  if (!appendClassPartsForHelper(loc, source, objectType, helper,
                                 /*inputOffset=*/consumed, builder, args,
                                 sourceAttrs, &consumed))
    return mlir::failure();
  if (args.size() != helper.getFunctionType().getNumInputs())
    return mlir::failure();
  builder.create<mlir::func::CallOp>(loc, helper, mlir::ValueRange{args});
  return mlir::success();
}

mlir::LogicalResult
Slot::classCarrierCopy(mlir::Location loc, mlir::Value destSlotView,
                       mlir::Value sourceSlotView, ClassType type,
                       mlir::ModuleOp module, mlir::OpBuilder &builder) {
  return Slot::classCarrierCopyTo(loc, destSlotView, sourceSlotView, type,
                                  module, builder);
}

mlir::FailureOr<Slot::ClassCarrierParts> Slot::classCarrierInitializeParts(
    mlir::Location loc, mlir::Value destSlotView, ClassType type,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, bool transferToSlot) {
  mlir::FailureOr<class_layout::Layout> layout =
      class_layout::get(module, type, typeConverter);
  if (mlir::failed(layout))
    return mlir::failure();
  if (!destSlotView ||
      destSlotView.getType() !=
          class_layout::carrierType(layout->objectType, builder.getContext()))
    return mlir::failure();

  mlir::Value header =
      builder.create<mlir::memref::AllocOp>(loc, layout->headerType);
  if (mlir::Operation *def = header.getDefiningOp()) {
    def->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                 builder.getUnitAttr());
    def->setAttr(OwnershipContractAttrs::kObjectHeader, builder.getUnitAttr());
  }

  llvm::SmallVector<mlir::Value, 8> payloadParts;
  for (mlir::MemRefType partType : class_layout::Payload::partTypes(*layout)) {
    auto ownedPartType =
        mlir::MemRefType::get(partType.getShape(), partType.getElementType());
    mlir::Value ownedPart =
        builder.create<mlir::memref::AllocOp>(loc, ownedPartType);
    if (mlir::Operation *def = ownedPart.getDefiningOp())
      def->setAttr(ClassSafetyAttrs::kPayloadPart, builder.getUnitAttr());
    mlir::Value part = memRefView(loc, ownedPart, partType, builder);
    if (!part)
      return mlir::failure();
    payloadParts.push_back(part);
  }

  auto refcountStore = builder.create<mlir::memref::StoreOp>(
      loc, createI64Constant(loc, builder, 1), header,
      mlir::ValueRange{createIndexConstant(
          loc, builder, class_layout::Header::kRefcountSlot)});
  refcountStore->setAttr(ClassSafetyAttrs::kPromoteRefcountInit,
                         builder.getUnitAttr());
  storeHeaderSlot(loc, header, class_layout::Header::kLayoutIdSlot,
                  createI64Constant(loc, builder, classLayoutId(type)),
                  builder);

  mlir::Value storage =
      class_layout::Payload::zeroStorage(loc, *layout, builder);
  if (!storage)
    return mlir::failure();
  for (auto [index, field] : llvm::enumerate(layout->fields)) {
    (void)field;
    mlir::Value value = class_layout::Payload::extractField(
        loc, *layout, storage, static_cast<int64_t>(index), builder);
    if (!value)
      return mlir::failure();
    auto parts = class_layout::Payload::decomposeField(
        loc, *layout, static_cast<int64_t>(index), value, builder);
    if (mlir::failed(parts))
      return mlir::failure();
    int64_t start = class_layout::Payload::fieldPartIndex(
        *layout, static_cast<int64_t>(index));
    int64_t count = class_layout::Payload::fieldPartCount(
        *layout, static_cast<int64_t>(index));
    if (start < 0 || parts->size() != static_cast<size_t>(count))
      return mlir::failure();
    for (auto [partOffset, part] : llvm::enumerate(*parts)) {
      int64_t payloadIndex = start + static_cast<int64_t>(partOffset);
      auto fieldStore = builder.create<mlir::memref::StoreOp>(
          loc, part, payloadParts[payloadIndex],
          mlir::ValueRange{createIndexConstant(loc, builder, 0)});
      fieldStore->setAttr(OwnershipContractAttrs::kMemRefSlotTransfer,
                          builder.getUnitAttr());
      fieldStore->setAttr(OwnershipContractAttrs::kAggregateSlotGroup,
                          builder.getStringAttr("class.payload"));
      std::string component = "field." + std::to_string(index) + ".part." +
                              std::to_string(partOffset);
      fieldStore->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                          builder.getStringAttr(component));
    }
  }
  mlir::Value lockPart =
      payloadParts[class_layout::Payload::lockPartIndex(*layout)];
  auto lockStore = builder.create<mlir::memref::StoreOp>(
      loc, builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32), lockPart,
      mlir::ValueRange{createIndexConstant(loc, builder, 0)});
  lockStore->setAttr(ClassSafetyAttrs::kPromoteLockInit, builder.getUnitAttr());

  mlir::Value headerDescriptor =
      descriptorFromMemRef(loc, header, layout->headerType, builder);
  if (!headerDescriptor)
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 8> descriptors{headerDescriptor};
  for (auto [index, part] : llvm::enumerate(payloadParts)) {
    mlir::MemRefType partType =
        class_layout::Payload::partType(*layout, static_cast<int64_t>(index));
    mlir::Value descriptor = descriptorFromMemRef(loc, part, partType, builder);
    if (!descriptor)
      return mlir::failure();
    descriptors.push_back(descriptor);
  }
  mlir::Value objectValue = class_layout::Object::fromDescriptors(
      loc, layout->objectType, descriptors, builder);
  if (!objectValue)
    return mlir::failure();
  auto objectStore = builder.create<mlir::memref::StoreOp>(
      loc, objectValue, destSlotView,
      mlir::ValueRange{createIndexConstant(loc, builder, 0)});
  if (transferToSlot) {
    objectStore->setAttr(OwnershipContractAttrs::kMemRefSlotTransfer,
                         builder.getUnitAttr());
    objectStore->setAttr(OwnershipContractAttrs::kAggregateSlotGroup,
                         builder.getStringAttr("class.object"));
    objectStore->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                         builder.getStringAttr("parts"));
  }
  return ClassCarrierParts{header, payloadParts};
}

mlir::LogicalResult
Slot::classCarrierInitialize(mlir::Location loc, mlir::Value destSlotView,
                             ClassType type, mlir::ModuleOp module,
                             mlir::OpBuilder &builder,
                             const PyLLVMTypeConverter &typeConverter) {
  if (mlir::failed(Slot::classCarrierInitializeParts(
          loc, destSlotView, type, module, builder, typeConverter)))
    return mlir::failure();
  return mlir::success();
}

mlir::Value Slot::classCarrierEqual(mlir::Location loc, mlir::Value lhs,
                                    mlir::Value rhs, ClassType type,
                                    mlir::ModuleOp module,
                                    mlir::OpBuilder &builder) {
  auto lhsType = mlir::dyn_cast<mlir::MemRefType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::MemRefType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != 1 ||
      rhsType.getRank() != 1 ||
      lhsType.getElementType() != rhsType.getElementType())
    return {};
  auto objectType = class_layout::objectCarrierType(lhsType);
  if (!objectType)
    return {};

  lhs = loadClassObject(loc, lhs, objectType, builder);
  rhs = loadClassObject(loc, rhs, objectType, builder);
  if (!lhs || !rhs)
    return {};

  std::string helperName = getClassHelperName(type, "eq");
  auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
  if (!helper || helper.getFunctionType().getNumResults() != 1)
    return {};
  llvm::SmallVector<mlir::Value, 4> args;
  unsigned consumed = 0;
  mlir::Value lhsAttrs = lhs;
  if (!appendClassPartsForHelper(loc, lhs, objectType, helper,
                                 /*inputOffset=*/0, builder, args, lhsAttrs,
                                 &consumed))
    return {};
  mlir::Value rhsAttrs = rhs;
  if (!appendClassPartsForHelper(loc, rhs, objectType, helper,
                                 /*inputOffset=*/consumed, builder, args,
                                 rhsAttrs, &consumed))
    return {};
  if (args.size() != helper.getFunctionType().getNumInputs())
    return {};
  auto call =
      builder.create<mlir::func::CallOp>(loc, helper, mlir::ValueRange{args});
  return call.getResult(0);
}

mlir::FailureOr<mlir::Value>
Slot::pack(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
           mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
           const PyLLVMTypeConverter &typeConverter) {
  RuntimeAPI runtime(module, rewriter, typeConverter);

  if (mlir::isa<IntType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intType.getWidth() < 64)
        return rewriter
            .create<mlir::arith::ExtSIOp>(loc, rewriter.getI64Type(), value)
            .getResult();
      if (intType.getWidth() > 64)
        return rewriter
            .create<mlir::arith::TruncIOp>(loc, rewriter.getI64Type(), value)
            .getResult();
    }
    return mlir::failure();
  }

  if (mlir::isa<BoolType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    if (value.getType() == rewriter.getI1Type())
      return rewriter
          .create<mlir::arith::ExtUIOp>(loc, rewriter.getI64Type(), value)
          .getResult();
    return mlir::failure();
  }

  if (mlir::isa<FloatType>(logicalType)) {
    if (value.getType() == rewriter.getI64Type())
      return value;
    mlir::Value asDouble = value;
    if (value.getType() != rewriter.getF64Type()) {
      return mlir::failure();
    }
    return rewriter
        .create<mlir::arith::BitcastOp>(loc, rewriter.getI64Type(), asDouble)
        .getResult();
  }

  return mlir::failure();
}

mlir::FailureOr<mlir::Value>
Slot::box(mlir::Location loc, mlir::Value value, mlir::Type logicalType,
          mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
          const PyLLVMTypeConverter &typeConverter) {
  (void)module;

  if (mlir::isa<IntType>(logicalType)) {
    return mlir::failure();
  }

  if (mlir::isa<BoolType>(logicalType)) {
    mlir::Type converted = typeConverter.convertType(logicalType);
    if (converted == rewriter.getI1Type()) {
      if (value.getType() == converted)
        return value;
      if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
        mlir::Value zero =
            rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
        return rewriter
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                         value, zero)
            .getResult();
      }
      return mlir::failure();
    }
    return mlir::failure();
  }

  if (mlir::isa<FloatType>(logicalType)) {
    mlir::Type converted = typeConverter.convertType(logicalType);
    if (converted == rewriter.getF64Type()) {
      if (value.getType() == converted)
        return value;
      if (value.getType() == rewriter.getI64Type())
        return rewriter.create<mlir::arith::BitcastOp>(loc, converted, value)
            .getResult();
      return mlir::failure();
    }
    return mlir::failure();
  }

  if (mlir::isa<ClassType>(logicalType)) {
    auto classType = mlir::cast<ClassType>(logicalType);
    mlir::Type converted = class_layout::carrierStorageType(
        module, classType, typeConverter, rewriter.getContext());
    if (!converted)
      return mlir::failure();
    if (value.getType() == converted)
      return value;
    if (auto convertedObject = class_layout::objectCarrierType(converted)) {
      if (mlir::Value object =
              loadClassObject(loc, value, convertedObject, rewriter))
        return object;
    }
    return mlir::failure();
  }

  return mlir::failure();
}

mlir::Value Slot::storage(mlir::Location loc, mlir::Value value,
                          mlir::Type logicalType, mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  mlir::Type storageType =
      container::Slot::storageType(logicalType, rewriter.getContext());
  return Slot::storage(loc, value, logicalType, storageType, module, rewriter,
                       typeConverter);
}

mlir::Value Slot::storage(mlir::Location loc, mlir::ValueRange values,
                          mlir::Type logicalType, mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  mlir::Type storageType =
      container::Slot::storageType(logicalType, rewriter.getContext());
  return Slot::storage(loc, values, logicalType, storageType, module, rewriter,
                       typeConverter);
}

mlir::Value Slot::storage(mlir::Location loc, mlir::Value value,
                          mlir::Type logicalType, mlir::Type storageType,
                          mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  if (!storageType)
    return {};

  if (value.getType() == storageType)
    return value;

  if (auto logicalInt = mlir::dyn_cast<mlir::IntegerType>(logicalType)) {
    auto storageInt = mlir::dyn_cast<mlir::IntegerType>(storageType);
    if (!storageInt)
      return {};
    mlir::Value scalar = value;
    if (!mlir::isa<mlir::IntegerType>(scalar.getType())) {
      return {};
    }
    auto valueInt = mlir::dyn_cast<mlir::IntegerType>(scalar.getType());
    if (!valueInt)
      return {};
    if (valueInt.getWidth() == storageInt.getWidth())
      return scalar;
    if (valueInt.getWidth() < storageInt.getWidth())
      return rewriter.create<mlir::arith::ExtSIOp>(loc, storageType, scalar);
    return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, scalar);
  }

  if (mlir::isa<mlir::FloatType>(logicalType)) {
    if (value.getType() == storageType)
      return value;
    return {};
  }

  if (mlir::isa<ClassType>(logicalType) &&
      mlir::isa<mlir::LLVM::LLVMStructType>(storageType)) {
    return loadClassStorageFromDescriptor(
        loc, value, mlir::cast<mlir::LLVM::LLVMStructType>(storageType),
        rewriter);
  }

  if (mlir::isa<BoolType>(logicalType) && storageType == rewriter.getI8Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intType.getWidth() > 8)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, value);
      if (intType.getWidth() < 8)
        return rewriter.create<mlir::arith::ExtUIOp>(loc, storageType, value);
    }
    return {};
  }

  if (mlir::isa<FloatType>(logicalType) &&
      storageType == rewriter.getF64Type()) {
    if (value.getType() == rewriter.getI64Type())
      return rewriter.create<mlir::arith::BitcastOp>(loc, storageType, value);
    return {};
  }

  if (mlir::isa<IntType>(logicalType) && storageType == rewriter.getI64Type()) {
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType())) {
      if (intType.getWidth() < 64)
        return rewriter.create<mlir::arith::ExtSIOp>(loc, storageType, value);
      if (intType.getWidth() > 64)
        return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, value);
    }
    return {};
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
  if (mlir::isa<FloatType>(logicalType) &&
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

mlir::Value Slot::storage(mlir::Location loc, mlir::ValueRange values,
                          mlir::Type logicalType, mlir::Type storageType,
                          mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  if (values.size() == 1)
    return Slot::storage(loc, values.front(), logicalType, storageType, module,
                         rewriter, typeConverter);

  if (mlir::isa<IntType>(logicalType) && storageType == rewriter.getI64Type() &&
      values.size() == 3) {
    RuntimeAPI runtime(module, rewriter, typeConverter);
    return runtime.call(loc, RuntimeSymbols::kLongAsI64, storageType, values)
        .getResult();
  }

  if (mlir::isa<StrType, ExceptionType>(logicalType)) {
    auto carrierType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storageType);
    if (carrierType && carrierType == objectPartsCarrierType(
                                          logicalType, rewriter.getContext()))
      return Slot::classCarrierFromValues(loc, values, carrierType, rewriter);
  }

  return {};
}

mlir::Value
Slot::ownedContainerStorage(mlir::Location loc, mlir::Value value,
                            mlir::Type logicalType, mlir::Type storageType,
                            mlir::ModuleOp module,
                            mlir::ConversionPatternRewriter &rewriter,
                            const PyLLVMTypeConverter &typeConverter) {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    (void)classType;
    if (storageType == rewriter.getI64Type()) {
      return {};
    }
  }
  return Slot::storage(loc, value, logicalType, storageType, module, rewriter,
                       typeConverter);
}

mlir::Value
Slot::ownedContainerStorage(mlir::Location loc, mlir::ValueRange values,
                            mlir::Type logicalType, mlir::Type storageType,
                            mlir::ModuleOp module,
                            mlir::ConversionPatternRewriter &rewriter,
                            const PyLLVMTypeConverter &typeConverter) {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    (void)classType;
    if (storageType == rewriter.getI64Type())
      return {};
  }
  return Slot::storage(loc, values, logicalType, storageType, module, rewriter,
                       typeConverter);
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
  } else if (mlir::isa<FloatType>(logicalType)) {
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

mlir::LogicalResult
Slot::replaceBoxedStorage(mlir::Operation *op, mlir::Value value,
                          mlir::Type logicalType, mlir::ModuleOp module,
                          mlir::ConversionPatternRewriter &rewriter,
                          const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, resultTypes)) ||
      resultTypes.empty())
    return mlir::failure();

  if (mlir::isa<IntType>(logicalType)) {
    llvm::SmallVector<mlir::Type, 3> longTypes;
    object_abi::long_abi::Parts::storageTypes(rewriter.getContext(), longTypes);
    if (resultTypes != longTypes)
      return mlir::failure();
    mlir::Value scalar = value;
    if (!scalar.getType().isInteger(64)) {
      auto intType = mlir::dyn_cast<mlir::IntegerType>(scalar.getType());
      if (!intType)
        return mlir::failure();
      if (intType.getWidth() < 64)
        scalar = rewriter.create<mlir::arith::ExtSIOp>(
            op->getLoc(), rewriter.getI64Type(), scalar);
      else if (intType.getWidth() > 64)
        scalar = rewriter.create<mlir::arith::TruncIOp>(
            op->getLoc(), rewriter.getI64Type(), scalar);
    }
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto call =
        runtime.call(op->getLoc(), RuntimeSymbols::kLongFromI64,
                     mlir::TypeRange(resultTypes), mlir::ValueRange{scalar});
    rewriter.replaceOpWithMultiple(
        op,
        llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(call.getResults())});
    return mlir::success();
  }

  if (mlir::isa<StrType, ExceptionType>(logicalType) &&
      resultTypes.size() > 1) {
    auto objectType = class_layout::objectCarrierType(value.getType());
    if (!objectType)
      return mlir::failure();
    if (class_layout::Object::partCount(objectType) !=
        static_cast<int64_t>(resultTypes.size()))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> parts;
    parts.reserve(resultTypes.size());
    for (auto [index, expected] : llvm::enumerate(resultTypes)) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(expected);
      if (!memrefType)
        return mlir::failure();
      mlir::Value descriptor = class_layout::Object::descriptor(
          op->getLoc(), objectType, value, static_cast<int64_t>(index),
          rewriter);
      mlir::Value part =
          memRefFromDescriptor(op->getLoc(), descriptor, memrefType, rewriter);
      if (!part)
        return mlir::failure();
      parts.push_back(part);
    }
    rewriter.replaceOpWithMultiple(
        op, llvm::ArrayRef<mlir::ValueRange>{mlir::ValueRange(parts)});
    return mlir::success();
  }

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    mlir::FailureOr<class_layout::Layout> layout =
        class_layout::get(module, classType, typeConverter);
    if (mlir::failed(layout))
      return mlir::failure();
    llvm::SmallVector<mlir::Type, 4> expectedTypes;
    class_layout::partsValueTypes(*layout, expectedTypes);
    if (resultTypes != expectedTypes ||
        class_layout::Object::partCount(layout->objectType) !=
            static_cast<int64_t>(expectedTypes.size()))
      return mlir::failure();
    auto headerType = mlir::dyn_cast<mlir::MemRefType>(resultTypes[0]);
    auto objectType = layout->objectType;
    if (!headerType || !objectType)
      return mlir::failure();
    mlir::Value object =
        loadClassObject(op->getLoc(), value, objectType, rewriter);
    if (!object)
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> parts;
    for (auto [index, expected] : llvm::enumerate(expectedTypes)) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(expected);
      if (!memrefType)
        return mlir::failure();
      mlir::Value descriptor = class_layout::Object::descriptor(
          op->getLoc(), objectType, object, static_cast<int64_t>(index),
          rewriter);
      mlir::Value memref =
          memRefFromDescriptor(op->getLoc(), descriptor, memrefType, rewriter);
      if (!memref)
        return mlir::failure();
      parts.push_back(memref);
    }
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(parts)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }

  if (auto logicalInt = mlir::dyn_cast<mlir::IntegerType>(logicalType)) {
    auto valueInt = mlir::dyn_cast<mlir::IntegerType>(value.getType());
    if (!valueInt || resultTypes.size() != 1 ||
        resultTypes.front() != logicalInt)
      return mlir::failure();
    mlir::Value replacement = value;
    if (valueInt.getWidth() < logicalInt.getWidth())
      replacement = rewriter.create<mlir::arith::ExtSIOp>(op->getLoc(),
                                                          logicalInt, value);
    else if (valueInt.getWidth() > logicalInt.getWidth())
      replacement = rewriter.create<mlir::arith::TruncIOp>(op->getLoc(),
                                                           logicalInt, value);
    rewriter.replaceOp(op, replacement);
    return mlir::success();
  }

  if (mlir::isa<mlir::FloatType>(logicalType)) {
    if (resultTypes.size() != 1 || resultTypes.front() != logicalType ||
        value.getType() != logicalType)
      return mlir::failure();
    rewriter.replaceOp(op, value);
    return mlir::success();
  }

  if (resultTypes.size() != 1)
    return mlir::failure();

  mlir::FailureOr<mlir::Value> boxed = Slot::boxStorage(
      op->getLoc(), value, logicalType, module, rewriter, typeConverter);
  if (mlir::failed(boxed))
    return mlir::failure();
  rewriter.replaceOp(op, *boxed);
  return mlir::success();
}

void Slot::classRefcount(mlir::Location loc, mlir::Value slot, ClassType type,
                         mlir::ModuleOp module, mlir::OpBuilder &rewriter,
                         llvm::StringRef suffix, bool aggregateEffect,
                         llvm::StringRef retainPremise) {
  if (auto slotType = mlir::dyn_cast<mlir::MemRefType>(slot.getType())) {
    if (class_layout::isObjectCarrierMemRefType(slotType)) {
      classCarrierRefcount(loc, slot, type, module, rewriter, suffix,
                           aggregateEffect, retainPremise);
      return;
    }
  }
  auto objectType = class_layout::objectCarrierType(slot.getType());
  bool descriptor = class_layout::isDescriptorStorageType(slot.getType());
  bool objectAggregate = class_layout::isObjectCarrierType(objectType);
  if (!descriptor && !objectAggregate)
    return;
  auto helper =
      module.lookupSymbol<mlir::func::FuncOp>(getClassHelperName(type, suffix));
  if (!helper)
    return;
  if (suffix == "incref" || suffix == "decref" || suffix == "destroy_local") {
    mlir::OpBuilder attrBuilder(helper.getContext());
    helper->setAttr(ClassSafetyAttrs::kHelperKind,
                    attrBuilder.getStringAttr([&]() -> llvm::StringRef {
                      if (suffix == "incref")
                        return ClassSafetyAttrs::kKindIncref;
                      if (suffix == "destroy_local")
                        return ClassSafetyAttrs::kKindDestroyLocal;
                      return ClassSafetyAttrs::kKindDecref;
                    }()));
    helper->setAttr(ClassSafetyAttrs::kHelperClass,
                    attrBuilder.getStringAttr(type.getClassName()));
  }
  auto emitCall = [&](mlir::OpBuilder &builder) {
    llvm::SmallVector<mlir::Value, 2> callArgs;
    mlir::Value attrsSource = slot;
    if (classPartsForHelper(loc, slot, objectType, helper, builder, callArgs,
                            attrsSource)) {
      auto call = builder.create<mlir::func::CallOp>(loc, helper, callArgs);
      copyAggregateSlotLoadAttrs(attrsSource, call.getOperation());
      if (suffix == "incref")
        threadsafe::Retain::premise(call.getOperation(), retainPremise);
      if (suffix == "destroy_local")
        call->setAttr(OwnershipContractAttrs::kLocalDestroy,
                      builder.getUnitAttr());
      if (aggregateEffect) {
        ownership::aggregate::Slot::markLoad(slot);
        call->setAttr(suffix == "incref"
                          ? OwnershipContractAttrs::kAggregateRetain
                          : OwnershipContractAttrs::kAggregateRelease,
                      builder.getUnitAttr());
      }
      return;
    }
    if (helper.getFunctionType().getNumInputs() != 1)
      return;
    mlir::Type helperArgType = helper.getFunctionType().getInput(0);
    mlir::Value object = slot;
    if (object.getType() != helperArgType)
      object = builder
                   .create<mlir::UnrealizedConversionCastOp>(
                       loc, helperArgType, mlir::ValueRange{object})
                   .getResult(0);
    auto call = builder.create<mlir::func::CallOp>(loc, helper,
                                                   mlir::ValueRange{object});
    copyAggregateSlotLoadAttrs(object, call.getOperation());
    if (suffix == "incref")
      threadsafe::Retain::premise(call.getOperation(), retainPremise);
    if (suffix == "destroy_local")
      call->setAttr(OwnershipContractAttrs::kLocalDestroy,
                    builder.getUnitAttr());
    if (aggregateEffect) {
      ownership::aggregate::Slot::markLoad(slot);
      call->setAttr(suffix == "incref"
                        ? OwnershipContractAttrs::kAggregateRetain
                        : OwnershipContractAttrs::kAggregateRelease,
                    builder.getUnitAttr());
    }
  };

  mlir::Value headerDescriptor = slot;
  if (objectAggregate)
    headerDescriptor =
        class_layout::Object::headerDescriptor(loc, objectType, slot, rewriter);
  mlir::Value hasStorage = memRefDescriptorPresent(
      loc, headerDescriptor,
      class_layout::Header::memrefType(rewriter.getContext()), rewriter);
  if (!hasStorage)
    return;
  auto ifOp = rewriter.create<mlir::scf::IfOp>(loc, hasStorage,
                                               /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    emitCall(rewriter);
  }
}

bool Slot::refcounted(mlir::Type logicalType) {
  return mlir::isa<ClassType, StrType, ExceptionType, TracebackType,
                   LocationType>(logicalType);
}

void Slot::markTransfer(mlir::Operation *storeLike) {
  if (!storeLike)
    return;
  ownership::aggregate::Slot::markStore(storeLike);
}

void Slot::releaseSource(mlir::Location loc, mlir::Value source,
                         mlir::Type logicalType, mlir::ModuleOp module,
                         mlir::ConversionPatternRewriter &rewriter,
                         const PyLLVMTypeConverter &typeConverter) {
  (void)loc;
  (void)source;
  (void)logicalType;
  (void)module;
  (void)rewriter;
  (void)typeConverter;
}

static mlir::LogicalResult refcountObjectPartsCarrier(
    mlir::Location loc, mlir::Value slot, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef suffix,
    bool aggregateEffect, llvm::StringRef retainPremise) {
  auto objectType = class_layout::objectCarrierType(slot.getType());
  if (!objectType)
    return mlir::failure();
  llvm::SmallVector<mlir::MemRefType, 2> partTypes;
  if (!objectPartsStorageTypes(logicalType, builder.getContext(), partTypes) ||
      partTypes.size() !=
          static_cast<size_t>(class_layout::Object::partCount(objectType)))
    return mlir::failure();

  mlir::Value object = loadClassObject(loc, slot, objectType, builder);
  if (!object)
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 2> parts;
  for (auto [index, partType] : llvm::enumerate(partTypes)) {
    mlir::Value descriptor = class_layout::Object::descriptor(
        loc, objectType, object, static_cast<int64_t>(index), builder);
    mlir::Value part = memRefFromClassPart(loc, descriptor, partType, builder);
    if (!part)
      return mlir::failure();
    parts.push_back(part);
  }

  RuntimeAPI runtime(module, builder, typeConverter);
  if (suffix == "incref") {
    auto call = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                             mlir::ValueRange{parts.front()});
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
    if (aggregateEffect) {
      ownership::aggregate::Slot::markLoad(object);
      copyAggregateSlotLoadAttrs(object, call.getOperation());
      call->setAttr(OwnershipContractAttrs::kAggregateRetain,
                    builder.getUnitAttr());
    }
    return mlir::success();
  }

  llvm::StringRef symbol;
  if (mlir::isa<StrType>(logicalType)) {
    symbol = RuntimeSymbols::kUnicodeDecRef;
  } else if (mlir::isa<ExceptionType>(logicalType)) {
    symbol = RuntimeSymbols::kExceptionDecRef;
  } else {
    return mlir::failure();
  }

  if (suffix != "decref") {
    mlir::emitError(loc) << "unsupported slot refcount operation: " << suffix;
    return mlir::failure();
  }
  auto call = runtime.call(loc, symbol, mlir::Type(), parts);
  if (aggregateEffect) {
    ownership::aggregate::Slot::markLoad(object);
    copyAggregateSlotLoadAttrs(object, call.getOperation());
    call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                  builder.getUnitAttr());
  }
  return mlir::success();
}

mlir::LogicalResult
Slot::refcount(mlir::Location loc, mlir::Value slot, mlir::Type logicalType,
               mlir::ModuleOp module, mlir::OpBuilder &rewriter,
               const PyLLVMTypeConverter &typeConverter, llvm::StringRef suffix,
               bool aggregateEffect, llvm::StringRef retainPremise) {
  if (!Slot::refcounted(logicalType))
    return mlir::success();

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    Slot::classRefcount(loc, slot, classType, module, rewriter, suffix,
                        aggregateEffect, retainPremise);
    return mlir::success();
  }

  if (mlir::isa<StrType, ExceptionType>(logicalType) &&
      class_layout::objectCarrierType(slot.getType()))
    return refcountObjectPartsCarrier(loc, slot, logicalType, module, rewriter,
                                      typeConverter, suffix, aggregateEffect,
                                      retainPremise);

  if (!object_abi::Type::isStorageLike(slot.getType()))
    return mlir::success();

  llvm::StringRef symbol;
  if (suffix == "incref")
    symbol = RuntimeSymbols::kIncRef;
  else if (suffix == "decref") {
    mlir::emitError(loc)
        << "slot decref requires a concrete header/payload release helper; "
           "generic Ly_DecRef is not part of the object ABI";
    return mlir::failure();
  } else {
    mlir::emitError(loc) << "unsupported slot refcount operation: " << suffix;
    return mlir::failure();
  }

  RuntimeAPI runtime(module, rewriter, typeConverter);
  auto call =
      runtime.call(loc, symbol, /*resultType=*/nullptr, mlir::ValueRange{slot});
  if (suffix == "incref")
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  if (aggregateEffect) {
    ownership::aggregate::Slot::markLoad(slot);
    call->setAttr(suffix == "incref"
                      ? OwnershipContractAttrs::kAggregateRetain
                      : OwnershipContractAttrs::kAggregateRelease,
                  rewriter.getUnitAttr());
  }
  return mlir::success();
}

mlir::LogicalResult
container::Elements::refcount(mlir::Location loc, mlir::Type logicalType,
                              mlir::ValueRange descriptor,
                              mlir::ModuleOp module, mlir::OpBuilder &builder,
                              const PyLLVMTypeConverter &typeConverter,
                              llvm::StringRef suffix, bool markAggregate) {
  bool aggregateEffect = markAggregate || suffix == "decref";
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    mlir::Type elementType = listType.getElementType();
    if (!::py::Slot::refcounted(elementType) || descriptor.size() != 3)
      return mlir::success();
    mlir::Value header = descriptor[kListHeaderComponent];
    mlir::Value items = descriptor[kListItemsComponent];
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(items.getType());
    if (!itemsType)
      return mlir::success();
    ClassType inlineClassType;
    mlir::LLVM::LLVMStructType inlineCarrierType;
    bool inlineClass = inlineClassListStorage(
        listType, itemsType, inlineClassType, inlineCarrierType);

    mlir::Value size = builder.create<mlir::memref::LoadOp>(
        loc, header, createIndexConstant(loc, builder, kTypedListSizeSlot));
    mlir::Value upper = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), size);
    mlir::Value lower = createIndexConstant(loc, builder, 0);
    mlir::Value step = createIndexConstant(loc, builder, 1);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      if (inlineClass) {
        mlir::Value objectSlot = ::py::Slot::classCarrierView(
            loc, items, loop.getInductionVar(), inlineCarrierType, builder);
        if (!objectSlot)
          return mlir::failure();
        ::py::Slot::classCarrierRefcount(
            loc, objectSlot, inlineClassType, module, builder, suffix,
            aggregateEffect, ThreadSafetyAttrs::kPremiseAggregateBorrow);
      } else {
        mlir::Value slot = builder.create<mlir::memref::LoadOp>(
            loc, items, loop.getInductionVar());
        if (mlir::failed(::py::Slot::refcount(loc, slot, elementType, module,
                                              builder, typeConverter, suffix,
                                              aggregateEffect)))
          return mlir::failure();
      }
    }
    return mlir::success();
  }

  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    if (descriptor.size() != 2)
      return mlir::success();
    auto itemsType = mlir::dyn_cast<mlir::MemRefType>(descriptor[1].getType());
    if (!itemsType)
      return mlir::success();
    auto inlineCarrierType = class_layout::objectCarrierType(itemsType);
    for (auto [index, elementType] :
         llvm::enumerate(tupleType.getElementTypes())) {
      if (!::py::Slot::refcounted(elementType))
        continue;
      if (inlineCarrierType) {
        auto classType = mlir::dyn_cast<ClassType>(elementType);
        if (classType) {
          mlir::Value objectSlot = ::py::Slot::classCarrierView(
              loc, descriptor[1],
              createIndexConstant(loc, builder, static_cast<int64_t>(index)),
              inlineCarrierType, builder);
          if (!objectSlot)
            continue;
          ::py::Slot::classCarrierRefcount(
              loc, objectSlot, classType, module, builder, suffix,
              aggregateEffect, ThreadSafetyAttrs::kPremiseAggregateBorrow);
          continue;
        }
      }
      mlir::Value slot = builder.create<mlir::memref::LoadOp>(
          loc, descriptor[1],
          createIndexConstant(loc, builder, static_cast<int64_t>(index)));
      if (mlir::failed(::py::Slot::refcount(loc, slot, elementType, module,
                                            builder, typeConverter, suffix,
                                            aggregateEffect)))
        return mlir::failure();
    }
    return mlir::success();
  }

  if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    bool keyRefcounted = ::py::Slot::refcounted(dictType.getKeyType());
    bool valueRefcounted = ::py::Slot::refcounted(dictType.getValueType());
    if ((!keyRefcounted && !valueRefcounted) || descriptor.size() != 5)
      return mlir::success();
    mlir::Value header = descriptor[kDictHeaderComponent];
    mlir::Value keys = descriptor[kDictKeysComponent];
    mlir::Value values = descriptor[kDictValuesComponent];
    mlir::Value states = descriptor[kDictStatesComponent];

    mlir::Value capacity = builder.create<mlir::memref::LoadOp>(
        loc, header, createIndexConstant(loc, builder, kTypedDictCapacitySlot));
    mlir::Value upper = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getIndexType(), capacity);
    mlir::Value lower = createIndexConstant(loc, builder, 0);
    mlir::Value step = createIndexConstant(loc, builder, 1);
    auto loop = builder.create<mlir::scf::ForOp>(loc, lower, upper, step);
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(loop.getBody());
      mlir::Value state = builder.create<mlir::memref::LoadOp>(
          loc, states, loop.getInductionVar());
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
              loc, keys, loop.getInductionVar());
          if (mlir::failed(::py::Slot::refcount(loc, key, dictType.getKeyType(),
                                                module, builder, typeConverter,
                                                suffix, aggregateEffect)))
            return mlir::failure();
        }
        if (valueRefcounted) {
          mlir::Value value = builder.create<mlir::memref::LoadOp>(
              loc, values, loop.getInductionVar());
          if (mlir::failed(::py::Slot::refcount(
                  loc, value, dictType.getValueType(), module, builder,
                  typeConverter, suffix, aggregateEffect)))
            return mlir::failure();
        }
      }
    }
    return mlir::success();
  }
  return mlir::success();
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
                              mlir::Value lock, mlir::OpBuilder &builder) {
  mlir::Value lockIndex =
      createIndexConstant(loc, builder, kTypedContainerLockSlot);
  auto whileOp = builder.create<mlir::scf::WhileOp>(loc, mlir::TypeRange{},
                                                    mlir::ValueRange{});
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Block *before = builder.createBlock(&whileOp.getBefore());
    builder.setInsertionPointToStart(before);
    auto atomic = builder.create<mlir::memref::AtomicRMWOp>(
        loc, mlir::arith::AtomicRMWKind::assign,
        builder.create<mlir::arith::ConstantIntOp>(loc, 1,
                                                   builder.getI32Type()),
        lock, mlir::ValueRange{lockIndex});
    threadsafe::Atomic::set(atomic.getOperation(),
                            ThreadSafetyAttrs::kRoleContainerLockAcquire,
                            ThreadSafetyAttrs::kOrderingAcquire);
    markLockAtomicResource(atomic.getOperation(), header, lock);
    mlir::Value previous = atomic;
    mlir::Value busy = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, previous,
        builder.create<mlir::arith::ConstantIntOp>(loc, 0,
                                                   builder.getI32Type()));
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
                                mlir::Value lock, mlir::OpBuilder &builder) {
  auto atomic = builder.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::assign,
      builder.create<mlir::arith::ConstantIntOp>(loc, 0, builder.getI32Type()),
      lock,
      mlir::ValueRange{
          createIndexConstant(loc, builder, kTypedContainerLockSlot)});
  threadsafe::Atomic::set(atomic.getOperation(),
                          ThreadSafetyAttrs::kRoleContainerLockRelease,
                          ThreadSafetyAttrs::kOrderingRelease);
  markLockAtomicResource(atomic.getOperation(), header, lock);
}

void container::Managed::lock(mlir::Location loc, mlir::Value lock,
                              mlir::OpBuilder &builder) {
  container::Managed::lock(loc, {}, lock, builder);
}

void container::Managed::unlock(mlir::Location loc, mlir::Value lock,
                                mlir::OpBuilder &builder) {
  container::Managed::unlock(loc, {}, lock, builder);
}

} // namespace py
