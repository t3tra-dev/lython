#include "Common/ClassLayout.h"
#include "Common/Container.h"
#include "Common/LoweringUtils.h"
#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "Common/SlotUtils.h"
#include "PyValue/ClassHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include <cstdint>
#include <string>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

using StaticClassFieldInfo = class_layout::FieldInfo;
using StaticClassLayout = class_layout::Layout;
static constexpr llvm::StringLiteral kDecrefIntent{"__ly_decref_intent"};

namespace class_field::Refcount {
namespace Parts {
static std::optional<size_t> count(mlir::Type logicalType) {
  if (mlir::isa<IntType>(logicalType))
    return 2;
  if (mlir::isa<StrType>(logicalType))
    return 2;
  return std::nullopt;
}

bool storage(mlir::Type logicalType, mlir::Type storageType) {
  std::optional<size_t> expectedCount = count(logicalType);
  if (!expectedCount)
    return false;
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storageType);
  if (!aggregate || aggregate.isOpaque())
    return false;
  if (aggregate.getBody().size() != *expectedCount)
    return false;
  return llvm::all_of(aggregate.getBody(), object_abi::Type::isLoweredStorage);
}

llvm::StringRef releaseSymbol(mlir::Type logicalType) {
  if (mlir::isa<IntType>(logicalType))
    return RuntimeSymbols::kLongDecRef;
  if (mlir::isa<StrType>(logicalType))
    return RuntimeSymbols::kUnicodeDecRef;
  return {};
}

llvm::SmallVector<mlir::Value, 3> parts(mlir::Location loc, mlir::Value value,
                                        mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 3> result;
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(value.getType());
  if (!aggregate || aggregate.isOpaque())
    return result;
  for (auto [index, type] : llvm::enumerate(aggregate.getBody())) {
    result.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
        loc, type, value,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)})));
  }
  return result;
}
} // namespace Parts

bool needed(mlir::Type logicalType, mlir::Type storageType,
            const PyLLVMTypeConverter &typeConverter) {
  (void)typeConverter;
  if (!isPyType(logicalType) ||
      mlir::isa<ClassType, NoneType, BoolType, FloatType>(logicalType))
    return false;
  if (Parts::storage(logicalType, storageType))
    return true;
  return object_abi::Type::isLoweredStorage(storageType);
}

bool direct(mlir::Type logicalType, mlir::Type storageType,
            const PyLLVMTypeConverter &typeConverter) {
  return mlir::isa<ClassType>(logicalType) ||
         class_field::Refcount::needed(logicalType, storageType, typeConverter);
}
} // namespace class_field::Refcount

namespace class_field::Ownership {
static constexpr llvm::StringLiteral kGroup{"class.field"};
static constexpr llvm::StringLiteral kComponent{"payload"};

void markLoad(mlir::Value value, unsigned fieldIndex) {
  ownership::aggregate::Slot::markLoad(value, kGroup, kComponent,
                                       static_cast<int64_t>(fieldIndex));
}

void markStore(mlir::Operation *op, unsigned fieldIndex) {
  ownership::aggregate::Slot::markStore(op, kGroup, kComponent,
                                        static_cast<int64_t>(fieldIndex));
}

void copyLoadTo(mlir::Value source, mlir::Operation *target) {
  if (!source || !target)
    return;
  ownership::aggregate::Slot::markLoad(source);
  mlir::Operation *sourceOp = source.getDefiningOp();
  if (!sourceOp)
    return;
  llvm::StringRef attrs[] = {OwnershipContractAttrs::kAggregateSlotLoad,
                             OwnershipContractAttrs::kAggregateSlotGroup,
                             OwnershipContractAttrs::kAggregateSlotComponent,
                             OwnershipContractAttrs::kAggregateSlotIndex};
  for (llvm::StringRef attr : attrs)
    if (mlir::Attribute value = sourceOp->getAttr(attr))
      target->setAttr(attr, value);
}
} // namespace class_field::Ownership

static bool isRetainSymbol(llvm::StringRef symbol) {
  return symbol == RuntimeSymbols::kIncRef;
}

static bool isReleaseSymbol(llvm::StringRef symbol) {
  return symbol == kDecrefIntent;
}

namespace class_element::List {
ClassType type(mlir::Type logicalType) {
  auto listType = mlir::dyn_cast<ListType>(logicalType);
  if (!listType)
    return {};
  return mlir::dyn_cast<ClassType>(listType.getElementType());
}
} // namespace class_element::List

namespace class_element::Tuple {
void slots(mlir::Type logicalType,
           llvm::SmallVectorImpl<std::pair<unsigned, ClassType>> &slots) {
  auto tupleType = mlir::dyn_cast<TupleType>(logicalType);
  if (!tupleType)
    return;
  for (auto [index, type] : llvm::enumerate(tupleType.getElementTypes()))
    if (auto classType = mlir::dyn_cast<ClassType>(type))
      slots.push_back({static_cast<unsigned>(index), classType});
}
} // namespace class_element::Tuple

namespace class_element::Dict {
std::pair<ClassType, ClassType> types(mlir::Type logicalType) {
  auto dictType = mlir::dyn_cast<DictType>(logicalType);
  if (!dictType)
    return {};
  return {mlir::dyn_cast<ClassType>(dictType.getKeyType()),
          mlir::dyn_cast<ClassType>(dictType.getValueType())};
}
} // namespace class_element::Dict

namespace class_helper::Field {
std::string name(ClassType classType, llvm::StringRef action,
                 unsigned fieldIndex) {
  return ("__ly_class_" + action + "_" + classType.getClassName() + "_" +
          std::to_string(fieldIndex))
      .str();
}
} // namespace class_helper::Field

static mlir::Value castClassPart(mlir::Location loc, mlir::Value object,
                                 mlir::Type targetType, int64_t partIndex,
                                 mlir::OpBuilder &builder) {
  auto objectType = class_layout::objectCarrierType(object.getType());
  if (!objectType)
    return {};
  mlir::Value descriptor = class_layout::Object::descriptor(
      loc, objectType, object, partIndex, builder);
  if (!descriptor)
    return {};
  if (descriptor.getType() == targetType)
    return descriptor;
  return builder
      .create<mlir::UnrealizedConversionCastOp>(loc, targetType,
                                                mlir::ValueRange{descriptor})
      .getResult(0);
}

static void rewriteAggregateHelperCalls(mlir::Location loc,
                                        mlir::ModuleOp module,
                                        llvm::StringRef name,
                                        mlir::ArrayRef<mlir::Type> inputTypes) {
  if (inputTypes.size() < 2)
    return;

  llvm::SmallVector<mlir::func::CallOp, 8> calls;
  module.walk([&](mlir::func::CallOp call) {
    if (call.getCallee() == name && call.getNumOperands() == 1)
      calls.push_back(call);
  });

  for (mlir::func::CallOp call : calls) {
    mlir::OpBuilder builder(call);
    mlir::Value object = call.getOperand(0);
    auto objectType = class_layout::objectCarrierType(object.getType());
    if (!objectType)
      continue;

    llvm::SmallVector<mlir::Value, 8> parts;
    parts.reserve(inputTypes.size());
    bool valid = true;
    for (auto [index, type] : llvm::enumerate(inputTypes)) {
      mlir::Value part = castClassPart(loc, object, type,
                                       static_cast<int64_t>(index), builder);
      if (!part) {
        valid = false;
        break;
      }
      parts.push_back(part);
    }
    if (!valid)
      continue;

    auto replacement = builder.create<mlir::func::CallOp>(
        call.getLoc(), call.getCallee(), call.getResultTypes(), parts);
    mlir::StringAttr calleeAttr = builder.getStringAttr("callee");
    for (mlir::NamedAttribute attr : call->getAttrs())
      if (attr.getName() != calleeAttr)
        replacement->setAttr(attr.getName(), attr.getValue());

    for (auto [oldResult, newResult] :
         llvm::zip(call.getResults(), replacement.getResults()))
      oldResult.replaceAllUsesWith(newResult);
    call.erase();
  }
}

static mlir::func::FuncOp
getOrInsertClassFunc(mlir::Location loc, mlir::ModuleOp module,
                     mlir::OpBuilder &builder, llvm::StringRef name,
                     llvm::ArrayRef<mlir::Type> inputTypes,
                     llvm::ArrayRef<mlir::Type> resultTypes = {}) {
  auto fnType =
      mlir::FunctionType::get(module.getContext(), inputTypes, resultTypes);
  if (auto fn = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    if (fn.getFunctionType() != fnType) {
      if (!fn.getBody().empty()) {
        fn.emitError("class helper ABI mismatch for '") << name << "'";
        return {};
      }
      mlir::function_interface_impl::setFunctionType(fn, fnType);
      rewriteAggregateHelperCalls(loc, module, name, inputTypes);
    }
    return fn;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return builder.create<mlir::func::FuncOp>(loc, name, fnType);
}

namespace class_helper::Kind {
void mark(mlir::Operation *fn, ClassType classType, llvm::StringRef kind) {
  if (!fn)
    return;
  mlir::OpBuilder builder(fn->getContext());
  fn->setAttr(ClassSafetyAttrs::kHelperKind, builder.getStringAttr(kind));
  fn->setAttr(ClassSafetyAttrs::kHelperClass,
              builder.getStringAttr(classType.getClassName()));
}

void mark(mlir::func::FuncOp fn, ClassType classType, llvm::StringRef kind) {
  mark(fn ? fn.getOperation() : nullptr, classType, kind);
}
} // namespace class_helper::Kind

namespace class_helper::Schema {
void markObjectArg(mlir::func::FuncOp fn, const StaticClassLayout &layout,
                   unsigned firstArg) {
  if (!fn)
    return;
  mlir::FunctionType type = fn.getFunctionType();
  if (firstArg >= type.getNumInputs() || firstArg >= fn.getNumArguments())
    return;
  mlir::Type headerType = type.getInput(firstArg);
  if (headerType != layout.headerType &&
      !object_abi::Header::isOwned(headerType) &&
      !object_abi::Header::isView(headerType))
    return;
  fn.setArgAttr(firstArg, OwnershipContractAttrs::kObjectHeader,
                mlir::UnitAttr::get(fn.getContext()));
}

void mark(mlir::Operation *fn, const StaticClassLayout &layout,
          const PyLLVMTypeConverter &typeConverter) {
  if (!fn)
    return;
  int64_t directRefcountFields = 0;
  int64_t containerFields = 0;
  llvm::SmallVector<int64_t, 8> directRefcountFieldIndices;
  llvm::SmallVector<int64_t, 8> containerFieldIndices;
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    if (class_field::Refcount::direct(field.logicalType, field.storageType,
                                      typeConverter)) {
      ++directRefcountFields;
      directRefcountFieldIndices.push_back(static_cast<int64_t>(index));
      continue;
    }
    if (mlir::isa<ListType, TupleType, DictType>(field.logicalType)) {
      ++containerFields;
      containerFieldIndices.push_back(static_cast<int64_t>(index));
    }
  }
  mlir::OpBuilder builder(fn->getContext());
  fn->setAttr(
      ClassSafetyAttrs::kHelperFieldCount,
      builder.getI64IntegerAttr(static_cast<int64_t>(layout.fields.size())));
  fn->setAttr(ClassSafetyAttrs::kHelperDirectRefcountFields,
              builder.getI64IntegerAttr(directRefcountFields));
  fn->setAttr(ClassSafetyAttrs::kHelperContainerFields,
              builder.getI64IntegerAttr(containerFields));
  auto makeIndexArray = [&](llvm::ArrayRef<int64_t> indices) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(indices.size());
    for (int64_t index : indices)
      attrs.push_back(builder.getI64IntegerAttr(index));
    return builder.getArrayAttr(attrs);
  };
  fn->setAttr(ClassSafetyAttrs::kHelperDirectRefcountFieldIndices,
              makeIndexArray(directRefcountFieldIndices));
  fn->setAttr(ClassSafetyAttrs::kHelperContainerFieldIndices,
              makeIndexArray(containerFieldIndices));
}

void mark(mlir::func::FuncOp fn, const StaticClassLayout &layout,
          const PyLLVMTypeConverter &typeConverter) {
  mark(fn ? fn.getOperation() : nullptr, layout, typeConverter);
  markObjectArg(fn, layout, 0);
}
} // namespace class_helper::Schema

namespace class_helper::Field {
void mark(mlir::Operation *fn, ClassType classType, llvm::StringRef kind,
          unsigned fieldIndex) {
  class_helper::Kind::mark(fn, classType, kind);
  if (!fn)
    return;
  mlir::OpBuilder builder(fn->getContext());
  fn->setAttr(ClassSafetyAttrs::kHelperFieldIndex,
              builder.getI64IntegerAttr(fieldIndex));
}

void mark(mlir::func::FuncOp fn, ClassType classType, llvm::StringRef kind,
          unsigned fieldIndex) {
  mark(fn ? fn.getOperation() : nullptr, classType, kind, fieldIndex);
}
} // namespace class_helper::Field

namespace class_object::Slot {
mlir::Value create(mlir::Location loc, mlir::MemRefType memrefType,
                   mlir::ConversionPatternRewriter &rewriter,
                   mlir::Operation *anchor) {
  auto parentFunc = anchor ? anchor->getParentOfType<mlir::func::FuncOp>()
                           : mlir::func::FuncOp();
  if (!parentFunc)
    return {};
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(&parentFunc.getBody().front());
  auto ownedType =
      mlir::MemRefType::get(memrefType.getShape(), memrefType.getElementType());
  mlir::Value storage =
      rewriter
          .create<mlir::memref::AllocaOp>(loc, ownedType, mlir::ValueRange{})
          .getResult();
  if (storage.getType() == memrefType)
    return storage;
  mlir::Value cast =
      rewriter.create<mlir::memref::CastOp>(loc, memrefType, storage);
  return cast;
}
} // namespace class_object::Slot

namespace class_object::Local {
bool slot(mlir::Value value, mlir::LLVM::LLVMStructType objectType) {
  if (!value || !objectType)
    return false;
  if (value.getType() == objectType &&
      class_layout::isObjectCarrierType(objectType))
    return true;
  return class_layout::objectCarrierType(value.getType()) == objectType;
}

mlir::Value zeroIndex(mlir::Location loc, mlir::OpBuilder &builder) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
}

mlir::Value loadObject(mlir::Location loc, mlir::Value object,
                       const StaticClassLayout &layout,
                       mlir::OpBuilder &builder) {
  if (object.getType() == layout.objectType)
    return object;
  if (!class_object::Local::slot(object, layout.objectType))
    return {};
  auto load = builder.create<mlir::memref::LoadOp>(
      loc, object,
      mlir::ValueRange{class_object::Local::zeroIndex(loc, builder)});
  load->setAttr(ClassSafetyAttrs::kCarrierLoad, builder.getUnitAttr());
  ownership::aggregate::Slot::markLoad(load.getResult());
  return load.getResult();
}

mlir::Value descriptorFromMemRef(mlir::Location loc, mlir::Value memref,
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

mlir::Value memRefFromDescriptor(mlir::Location loc, mlir::Value descriptor,
                                 mlir::MemRefType memrefType,
                                 mlir::OpBuilder &builder) {
  if (descriptor.getType() == memrefType)
    return descriptor;
  if (mlir::Operation *def = descriptor.getDefiningOp())
    if (class_layout::DescriptorShape::has(def) &&
        mlir::failed(class_layout::DescriptorShape::verify(
            def, memrefType, "class carrier descriptor")))
      return {};
  if (mlir::isa<mlir::MemRefType>(descriptor.getType()))
    return builder.create<mlir::memref::CastOp>(loc, memrefType, descriptor);
  auto cast = builder.create<mlir::UnrealizedConversionCastOp>(
      loc, memrefType, mlir::ValueRange{descriptor});
  class_layout::DescriptorShape::mark(cast.getOperation(), memrefType);
  return cast.getResult(0);
}

mlir::Value partDescriptor(mlir::Location loc, mlir::Value objectValue,
                           const StaticClassLayout &layout, int64_t index,
                           mlir::OpBuilder &builder) {
  if (objectValue.getType() != layout.objectType)
    return {};
  return class_layout::Object::descriptor(loc, layout.objectType, objectValue,
                                          index, builder);
}

mlir::Value header(mlir::Location loc, mlir::Value object,
                   const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  mlir::Value objectValue =
      class_object::Local::loadObject(loc, object, layout, builder);
  if (!objectValue)
    return {};
  mlir::Value descriptor = class_object::Local::partDescriptor(
      loc, objectValue, layout, class_layout::Object::kHeaderIndex, builder);
  if (!descriptor)
    return {};
  return class_object::Local::memRefFromDescriptor(loc, descriptor,
                                                   layout.headerType, builder);
}

mlir::Value payloadPart(mlir::Location loc, mlir::Value object,
                        const StaticClassLayout &layout, int64_t partIndex,
                        mlir::OpBuilder &builder) {
  mlir::Value objectValue =
      class_object::Local::loadObject(loc, object, layout, builder);
  if (!objectValue)
    return {};
  mlir::Value descriptor = class_object::Local::partDescriptor(
      loc, objectValue, layout, class_layout::Object::kPayloadIndex + partIndex,
      builder);
  if (!descriptor)
    return {};
  mlir::MemRefType payloadType =
      class_layout::Payload::partType(layout, partIndex);
  if (!payloadType)
    return {};
  return class_object::Local::memRefFromDescriptor(loc, descriptor, payloadType,
                                                   builder);
}

mlir::Value objectValueFromParts(mlir::Location loc, mlir::Value header,
                                 mlir::ValueRange payloadParts,
                                 const StaticClassLayout &layout,
                                 mlir::OpBuilder &builder) {
  mlir::Value headerDescriptor = class_object::Local::descriptorFromMemRef(
      loc, header, layout.headerType, builder);
  if (!headerDescriptor ||
      payloadParts.size() !=
          static_cast<size_t>(class_layout::Payload::partCount(layout)))
    return {};

  llvm::SmallVector<mlir::Value, 8> descriptors;
  descriptors.push_back(headerDescriptor);
  for (auto [index, payload] : llvm::enumerate(payloadParts)) {
    mlir::MemRefType payloadType =
        class_layout::Payload::partType(layout, static_cast<int64_t>(index));
    if (!payloadType)
      return {};
    mlir::Value descriptor = class_object::Local::descriptorFromMemRef(
        loc, payload, payloadType, builder);
    if (!descriptor)
      return {};
    descriptors.push_back(descriptor);
  }
  return class_layout::Object::fromDescriptors(loc, layout.objectType,
                                               descriptors, builder);
}

void deallocParts(mlir::Location loc, mlir::Value object,
                  const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  for (int64_t index = 0; index < class_layout::Payload::partCount(layout);
       ++index) {
    mlir::Value part =
        class_object::Local::payloadPart(loc, object, layout, index, builder);
    if (part) {
      auto dealloc = builder.create<mlir::memref::DeallocOp>(loc, part);
      dealloc->setAttr(ClassSafetyAttrs::kDeallocPart,
                       builder.getStringAttr("payload"));
    }
  }
  mlir::Value header =
      class_object::Local::header(loc, object, layout, builder);
  if (header) {
    auto dealloc = builder.create<mlir::memref::DeallocOp>(loc, header);
    dealloc->setAttr(ClassSafetyAttrs::kDeallocPart,
                     builder.getStringAttr("header"));
  }
}

mlir::Value loadField(mlir::Location loc, mlir::Value object,
                      const StaticClassLayout &layout, unsigned fieldIndex,
                      mlir::Type fieldType, mlir::OpBuilder &builder) {
  int64_t start = class_layout::Payload::fieldPartIndex(layout, fieldIndex);
  int64_t count = class_layout::Payload::fieldPartCount(layout, fieldIndex);
  if (start < 0 || count <= 0)
    return {};
  (void)fieldType;
  llvm::SmallVector<mlir::Value, 4> parts;
  parts.reserve(count);
  for (int64_t offset = 0; offset < count; ++offset) {
    mlir::Value payload = class_object::Local::payloadPart(
        loc, object, layout, start + offset, builder);
    if (!payload)
      return {};
    auto load = builder.create<mlir::memref::LoadOp>(
        loc, payload,
        mlir::ValueRange{class_object::Local::zeroIndex(loc, builder)});
    class_field::Ownership::markLoad(load.getResult(), fieldIndex);
    parts.push_back(load.getResult());
  }
  mlir::Value value = class_layout::Payload::composeField(
      loc, layout, fieldIndex, parts, builder);
  if (!value)
    return {};
  class_field::Ownership::markLoad(value, fieldIndex);
  return value;
}

mlir::Operation *storeField(mlir::Location loc, mlir::Value object,
                            const StaticClassLayout &layout,
                            unsigned fieldIndex, mlir::Value fieldValue,
                            mlir::OpBuilder &builder) {
  int64_t start = class_layout::Payload::fieldPartIndex(layout, fieldIndex);
  int64_t count = class_layout::Payload::fieldPartCount(layout, fieldIndex);
  if (start < 0 || count <= 0)
    return nullptr;
  auto parts = class_layout::Payload::decomposeField(loc, layout, fieldIndex,
                                                     fieldValue, builder);
  if (mlir::failed(parts) || parts->size() != static_cast<size_t>(count))
    return nullptr;
  mlir::Operation *lastStore = nullptr;
  for (auto [offset, part] : llvm::enumerate(*parts)) {
    mlir::Value payload = class_object::Local::payloadPart(
        loc, object, layout, start + static_cast<int64_t>(offset), builder);
    if (!payload)
      return nullptr;
    auto store = builder.create<mlir::memref::StoreOp>(
        loc, part, payload,
        mlir::ValueRange{class_object::Local::zeroIndex(loc, builder)});
    class_field::Ownership::markStore(store.getOperation(), fieldIndex);
    store->setAttr(OwnershipContractAttrs::kAggregateSlotPartIndex,
                   builder.getI64IntegerAttr(static_cast<int64_t>(offset)));
    lastStore = store.getOperation();
  }
  return lastStore;
}
} // namespace class_object::Local

namespace class_object::Identity {
static mlir::Value fromMemRef(mlir::Location loc, mlir::Value memref,
                              mlir::OpBuilder &builder) {
  auto metadata =
      builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, memref);
  mlir::Value base =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
          loc, metadata.getBaseBuffer());
  mlir::Value identity =
      builder.create<mlir::arith::AddIOp>(loc, base, metadata.getOffset());
  return builder.create<mlir::arith::IndexCastOp>(loc, builder.getI64Type(),
                                                  identity);
}

mlir::Value value(mlir::Location loc, mlir::Value object,
                  const StaticClassLayout *layout, mlir::OpBuilder &builder) {
  if (layout && class_object::Local::slot(object, layout->objectType)) {
    mlir::Value header =
        class_object::Local::header(loc, object, *layout, builder);
    if (header)
      return fromMemRef(loc, header, builder);
  }
  if (auto objectType = class_layout::objectCarrierType(object.getType())) {
    mlir::Value headerDescriptor = class_layout::Object::headerDescriptor(
        loc, objectType, object, builder);
    auto headerType = class_layout::Header::memrefType(builder.getContext());
    mlir::Value headerMemRef = headerDescriptor;
    if (headerMemRef.getType() != headerType)
      headerMemRef =
          builder
              .create<mlir::UnrealizedConversionCastOp>(
                  loc, headerType, mlir::ValueRange{headerDescriptor})
              .getResult(0);
    return fromMemRef(loc, headerMemRef, builder);
  }
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(object.getType());
  if (!memrefType || memrefType.getRank() != 1)
    return {};
  return fromMemRef(loc, object, builder);
}
} // namespace class_object::Identity

namespace class_object::Payload {
void markPart(mlir::Value value, mlir::OpBuilder &builder) {
  if (mlir::Operation *def = value ? value.getDefiningOp() : nullptr)
    def->setAttr(ClassSafetyAttrs::kPayloadPart, builder.getUnitAttr());
}

mlir::Operation *initializePart(mlir::Location loc, mlir::Value payloadPart,
                                mlir::Value value, bool managed,
                                llvm::StringRef component,
                                mlir::OpBuilder &builder) {
  auto store = builder.create<mlir::memref::StoreOp>(
      loc, value, payloadPart,
      mlir::ValueRange{class_object::Local::zeroIndex(loc, builder)});
  if (managed) {
    store->setAttr(OwnershipContractAttrs::kMemRefSlotTransfer,
                   builder.getUnitAttr());
    store->setAttr(OwnershipContractAttrs::kAggregateSlotGroup,
                   builder.getStringAttr("class.payload"));
    store->setAttr(OwnershipContractAttrs::kAggregateSlotComponent,
                   builder.getStringAttr(component));
  }
  return store.getOperation();
}

mlir::Operation *initializeLock(mlir::Location loc, mlir::Value lockPart,
                                bool managed, mlir::OpBuilder &builder) {
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);
  auto store = builder.create<mlir::memref::StoreOp>(
      loc, zero, lockPart,
      mlir::ValueRange{class_object::Local::zeroIndex(loc, builder)});
  if (managed)
    store->setAttr(ClassSafetyAttrs::kPromoteLockInit, builder.getUnitAttr());
  return store.getOperation();
}

mlir::Operation *initialize(mlir::Location loc, mlir::ValueRange payloadParts,
                            mlir::ValueRange fieldValues,
                            const StaticClassLayout &layout, bool managed,
                            mlir::OpBuilder &builder) {
  if (payloadParts.size() !=
          static_cast<size_t>(class_layout::Payload::partCount(layout)) ||
      fieldValues.size() != layout.fields.size())
    return nullptr;
  mlir::Operation *lastStore = nullptr;
  for (auto [index, value] : llvm::enumerate(fieldValues)) {
    int64_t start = class_layout::Payload::fieldPartIndex(layout, index);
    int64_t count = class_layout::Payload::fieldPartCount(layout, index);
    if (start < 0 || count <= 0)
      return nullptr;
    auto parts = class_layout::Payload::decomposeField(
        loc, layout, static_cast<int64_t>(index), value, builder);
    if (mlir::failed(parts) || parts->size() != static_cast<size_t>(count))
      return nullptr;
    for (auto [partOffset, part] : llvm::enumerate(*parts)) {
      int64_t payloadIndex = start + static_cast<int64_t>(partOffset);
      std::string component = "field." + std::to_string(index) + ".part." +
                              std::to_string(partOffset);
      lastStore = class_object::Payload::initializePart(
          loc, payloadParts[payloadIndex], part, managed, component, builder);
      if (!lastStore)
        return nullptr;
    }
  }
  mlir::Value lockPart =
      payloadParts[class_layout::Payload::lockPartIndex(layout)];
  mlir::Operation *lockStore =
      class_object::Payload::initializeLock(loc, lockPart, managed, builder);
  return lockStore ? lockStore : lastStore;
}

} // namespace class_object::Payload

namespace class_object::Header {
int64_t layoutId(ClassType classType) {
  uint64_t hash = 1469598103934665603ULL;
  for (char ch : classType.getClassName()) {
    hash ^= static_cast<unsigned char>(ch);
    hash *= 1099511628211ULL;
  }
  return static_cast<int64_t>(hash & 0x7fffffffffffffffULL);
}

mlir::Value i64(mlir::Location loc, int64_t value, mlir::OpBuilder &builder) {
  return builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(value));
}

mlir::Operation *storeSlot(mlir::Location loc, mlir::Value header, int64_t slot,
                           mlir::Value value, mlir::OpBuilder &builder) {
  mlir::Value index = builder.create<mlir::arith::ConstantIndexOp>(loc, slot);
  auto store = builder.create<mlir::memref::StoreOp>(loc, value, header,
                                                     mlir::ValueRange{index});
  return store.getOperation();
}

bool initialize(mlir::Location loc, mlir::Value header, ClassType classType,
                const StaticClassLayout &layout, int64_t refcount,
                mlir::OpBuilder &builder) {
  if (!header || header.getType() != layout.headerType)
    return false;
  mlir::Operation *refcountStore = class_object::Header::storeSlot(
      loc, header, class_layout::Header::kRefcountSlot,
      i64(loc, refcount, builder), builder);
  if (refcount == 1 && refcountStore)
    refcountStore->setAttr(ClassSafetyAttrs::kPromoteRefcountInit,
                           builder.getUnitAttr());
  class_object::Header::storeSlot(
      loc, header, class_layout::Header::kLayoutIdSlot,
      i64(loc, layoutId(classType), builder), builder);
  return true;
}

} // namespace class_object::Header

namespace class_object::Lock {
mlir::Value storage(mlir::Location loc, mlir::Value object,
                    const StaticClassLayout &layout, mlir::OpBuilder &builder);
} // namespace class_object::Lock

namespace class_object::Payload {
mlir::Value loadField(mlir::Location loc, mlir::Value object,
                      const StaticClassLayout &layout, unsigned fieldIndex,
                      mlir::OpBuilder &builder) {
  if (fieldIndex >= layout.fields.size())
    return {};
  mlir::Type fieldType = layout.fields[fieldIndex].storageType;
  if (class_object::Local::slot(object, layout.objectType))
    return class_object::Local::loadField(loc, object, layout, fieldIndex,
                                          fieldType, builder);
  return {};
}

mlir::Operation *storeField(mlir::Location loc, mlir::Value object,
                            const StaticClassLayout &layout,
                            unsigned fieldIndex, mlir::Value fieldValue,
                            mlir::OpBuilder &builder) {
  if (fieldIndex >= layout.fields.size())
    return nullptr;
  if (class_object::Local::slot(object, layout.objectType))
    return class_object::Local::storeField(loc, object, layout, fieldIndex,
                                           fieldValue, builder);
  return nullptr;
}
} // namespace class_object::Payload

namespace class_object::Atomic {
void markHeader(mlir::Operation *op, mlir::Value header,
                std::optional<int64_t> slot);
} // namespace class_object::Atomic

namespace class_object::Managed {
mlir::Value load(mlir::Location loc, mlir::Value object,
                 const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  mlir::Value header =
      class_object::Local::header(loc, object, layout, builder);
  if (!header)
    return {};
  mlir::Value zero =
      builder.create<mlir::arith::ConstantIntOp>(loc, 0, builder.getI64Type());
  mlir::Value slot = builder.create<mlir::arith::ConstantIndexOp>(
      loc, class_layout::Header::kRefcountSlot);
  auto refcount = builder.create<mlir::memref::AtomicRMWOp>(
      loc, mlir::arith::AtomicRMWKind::addi, zero, header,
      mlir::ValueRange{slot});
  threadsafe::Atomic::set(refcount.getOperation(),
                          ThreadSafetyAttrs::kRoleClassRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_object::Atomic::markHeader(refcount.getOperation(), header,
                                   class_layout::Header::kRefcountSlot);
  return builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount.getResult(), zero);
}
} // namespace class_object::Managed

namespace class_object::Refcount {} // namespace class_object::Refcount

namespace class_object::Lock {
mlir::Value storage(mlir::Location loc, mlir::Value object,
                    const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  return class_object::Local::payloadPart(
      loc, object, layout, class_layout::Payload::lockPartIndex(layout),
      builder);
}
} // namespace class_object::Lock

static RuntimeAPI::Call emitRuntimeCall(
    mlir::Location loc, mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef runtimeSymbol,
    mlir::Type resultType, mlir::ValueRange args) {
  RuntimeAPI runtime(module, builder, typeConverter);
  return runtime.call(loc, runtimeSymbol, resultType, args);
}

static RuntimeAPI::Call
emitRuntimeVoidCall(mlir::Location loc, mlir::ModuleOp module,
                    mlir::OpBuilder &builder,
                    const PyLLVMTypeConverter &typeConverter,
                    llvm::StringRef runtimeSymbol, mlir::ValueRange args) {
  return emitRuntimeCall(loc, module, builder, typeConverter, runtimeSymbol,
                         mlir::LLVM::LLVMVoidType::get(builder.getContext()),
                         args);
}

// Atomic lowering policy:
// - Prefer memref.atomic_rmw for memref-backed storage and attach ly.atomic
//   ordering metadata. MemRefToLLVM preserves that contract as LLVM ordering.
// - Use LLVM atomics only when the storage is already a raw LLVM pointer.
namespace atomic_storage::MemRef {
mlir::Value rmw(mlir::Location loc, mlir::arith::AtomicRMWKind kind,
                mlir::Value memref, mlir::Value value, mlir::ValueRange indices,
                mlir::OpBuilder &builder) {
  if (!mlir::isa<mlir::MemRefType>(memref.getType()))
    return {};
  return builder.create<mlir::memref::AtomicRMWOp>(loc, kind, value, memref,
                                                   indices);
}
} // namespace atomic_storage::MemRef

namespace atomic_storage::Integer {
mlir::Value rmw(mlir::Location loc, mlir::LLVM::AtomicBinOp llvmKind,
                mlir::arith::AtomicRMWKind memrefKind, mlir::Value storage,
                mlir::Value value, mlir::ValueRange indices,
                mlir::LLVM::AtomicOrdering ordering, mlir::OpBuilder &builder) {
  if (mlir::Value result = atomic_storage::MemRef::rmw(loc, memrefKind, storage,
                                                       value, indices, builder))
    return result;
  return builder.create<mlir::LLVM::AtomicRMWOp>(loc, llvmKind, storage, value,
                                                 ordering);
}
} // namespace atomic_storage::Integer

namespace atomic_storage::Store {
mlir::Operation *ordered(mlir::Location loc, mlir::Value storage,
                         mlir::Value value, mlir::ValueRange indices,
                         mlir::LLVM::AtomicOrdering ordering,
                         mlir::OpBuilder &builder) {
  if (mlir::isa<mlir::MemRefType>(storage.getType())) {
    mlir::Value result =
        atomic_storage::MemRef::rmw(loc, mlir::arith::AtomicRMWKind::assign,
                                    storage, value, indices, builder);
    return result.getDefiningOp();
  }
  unsigned alignment = 0;
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType()))
    alignment = std::max<unsigned>(1, intType.getWidth() / 8);
  return builder
      .create<mlir::LLVM::StoreOp>(loc, value, storage, alignment,
                                   /*isVolatile=*/false,
                                   /*isNonTemporal=*/false,
                                   /*isInvariantGroup=*/false, ordering)
      .getOperation();
}
} // namespace atomic_storage::Store

namespace class_container::Kind {
std::optional<llvm::StringRef> name(mlir::Type logicalType) {
  if (mlir::isa<ListType>(logicalType))
    return ContainerSafetyAttrs::kKindList;
  if (mlir::isa<TupleType>(logicalType))
    return ContainerSafetyAttrs::kKindTuple;
  if (mlir::isa<DictType>(logicalType))
    return ContainerSafetyAttrs::kKindDict;
  return std::nullopt;
}
} // namespace class_container::Kind

namespace class_container::Atomic {
std::string group(mlir::Value descriptor) {
  if (!descriptor)
    return {};
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(descriptor))
    return "blockarg." + std::to_string(arg.getArgNumber());
  return "value." + std::to_string(reinterpret_cast<std::uintptr_t>(
                        descriptor.getAsOpaquePointer()));
}

void markHeader(mlir::Operation *op, mlir::Value descriptor,
                std::optional<int64_t> slot, mlir::Type logicalType = {}) {
  auto kind = logicalType ? class_container::Kind::name(logicalType)
                          : std::optional<llvm::StringRef>{};
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentHeader,
                                  slot,
                                  class_container::Atomic::group(descriptor),
                                  kind ? *kind : llvm::StringRef{});
  if (op) {
    mlir::OpBuilder builder(op->getContext());
    op->setAttr(
        ThreadSafetyAttrs::kAtomicProvenance,
        builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  }
}
} // namespace class_container::Atomic

namespace class_object::Atomic {
void markHeader(mlir::Operation *op, mlir::Value header,
                std::optional<int64_t> slot) {
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentHeader,
                                  slot, class_container::Atomic::group(header),
                                  llvm::StringRef{});
  if (op) {
    mlir::OpBuilder builder(op->getContext());
    op->setAttr(
        ThreadSafetyAttrs::kAtomicProvenance,
        builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  }
}

void markLock(mlir::Operation *op, mlir::Value lock) {
  threadsafe::memref::Atomic::set(op, ContainerSafetyAttrs::kComponentLock, 0,
                                  class_container::Atomic::group(lock),
                                  llvm::StringRef{});
  if (op) {
    mlir::OpBuilder builder(op->getContext());
    op->setAttr(
        ThreadSafetyAttrs::kAtomicProvenance,
        builder.getStringAttr(ThreadSafetyAttrs::kProvenanceMemRefDescriptor));
  }
}
} // namespace class_object::Atomic

namespace class_object::Lock {
void acquire(mlir::Location loc, mlir::Value lock, mlir::ModuleOp module,
             mlir::OpBuilder &builder) {
  (void)module;
  mlir::Block *currentBlock = builder.getInsertionBlock();
  auto insertionPoint = builder.getInsertionPoint();
  mlir::Region *region = builder.getInsertionBlock()->getParent();
  mlir::Block *loopBlock = builder.createBlock(region);
  mlir::Block *acquiredBlock = builder.createBlock(region);

  builder.setInsertionPoint(currentBlock, insertionPoint);
  builder.create<mlir::cf::BranchOp>(loc, loopBlock);

  builder.setInsertionPointToStart(loopBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
  mlir::Value slot = class_object::Local::zeroIndex(loc, builder);
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::xchg, mlir::arith::AtomicRMWKind::assign,
      lock, one, mlir::ValueRange{slot}, mlir::LLVM::AtomicOrdering::acquire,
      builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassLockAcquire,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_object::Atomic::markLock(previous.getDefiningOp(), lock);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  mlir::Value acquired = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, previous, zero);
  builder.create<mlir::cf::CondBranchOp>(loc, acquired, acquiredBlock,
                                         loopBlock);

  builder.setInsertionPointToStart(acquiredBlock);
}

void release(mlir::Location loc, mlir::Value lock, mlir::ModuleOp module,
             mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
  mlir::Value slot = class_object::Local::zeroIndex(loc, builder);
  mlir::Operation *store = atomic_storage::Store::ordered(
      loc, lock, zero, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::release, builder);
  threadsafe::Atomic::set(store, ThreadSafetyAttrs::kRoleClassLockRelease,
                          ThreadSafetyAttrs::kOrderingRelease);
  class_object::Atomic::markLock(store, lock);
}
} // namespace class_object::Lock

namespace class_object::Refcount {
void inc(mlir::Location loc, mlir::Value header, mlir::ModuleOp module,
         mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
  mlir::Value slot = builder.create<mlir::arith::ConstantIndexOp>(
      loc, class_layout::Header::kRefcountSlot);
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      header, one, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic,
                          ThreadSafetyAttrs::kPremiseOwnedToken);
  class_object::Atomic::markHeader(previous.getDefiningOp(), header,
                                   class_layout::Header::kRefcountSlot);
  threadsafe::Retain::verifyOwnedToken(
      previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
}

mlir::Value decAndIsZero(mlir::Location loc, mlir::Value header,
                         mlir::ModuleOp module, mlir::OpBuilder &builder) {
  (void)module;
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
  mlir::Value negOne = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(-1));
  mlir::Value slot = builder.create<mlir::arith::ConstantIndexOp>(
      loc, class_layout::Header::kRefcountSlot);
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      header, negOne, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acq_rel, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleClassRefcountRelease,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  class_object::Atomic::markHeader(previous.getDefiningOp(), header,
                                   class_layout::Header::kRefcountSlot);
  return builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, previous, one);
}
} // namespace class_object::Refcount

namespace class_container::Refcount {
std::optional<int64_t> slot(mlir::Type logicalType) {
  return container::Refcount::slotForLogicalType(logicalType);
}
} // namespace class_container::Refcount

namespace class_container::Descriptor {
mlir::Value part(mlir::Location loc, mlir::Value descriptor, unsigned index,
                 mlir::OpBuilder &builder) {
  auto structType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  if (!structType || structType.isOpaque() ||
      index >= structType.getBody().size())
    return {};
  return builder.create<mlir::LLVM::ExtractValueOp>(
      loc, structType.getBody()[index], descriptor,
      builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
}

mlir::Value header(mlir::Location loc, mlir::Value descriptor,
                   mlir::OpBuilder &builder) {
  if (mlir::Value header =
          class_container::Descriptor::part(loc, descriptor, 0, builder))
    return header;
  return descriptor;
}

llvm::SmallVector<mlir::Value, 4> parts(mlir::Location loc,
                                        mlir::Value descriptor,
                                        mlir::Type logicalType,
                                        mlir::OpBuilder &builder) {
  llvm::SmallVector<mlir::Value, 4> descriptors;
  unsigned parts = 0;
  if (mlir::isa<TupleType>(logicalType))
    parts = 2;
  else if (mlir::isa<ListType>(logicalType))
    parts = 3;
  else if (mlir::isa<DictType>(logicalType))
    parts = 5;
  if (parts == 0)
    return descriptors;

  for (unsigned index = 0; index < parts; ++index) {
    mlir::Value part =
        class_container::Descriptor::part(loc, descriptor, index, builder);
    if (!part)
      return {};
    descriptors.push_back(part);
  }
  return descriptors;
}
} // namespace class_container::Descriptor

namespace class_container::Elements {
mlir::LogicalResult refcount(mlir::Location loc, mlir::Value descriptor,
                             mlir::Type logicalType, mlir::ModuleOp module,
                             mlir::OpBuilder &builder,
                             const PyLLVMTypeConverter &typeConverter,
                             llvm::StringRef suffix, bool markAggregate = true,
                             bool transferRetainedToSlot = false) {
  (void)transferRetainedToSlot;
  llvm::SmallVector<mlir::Value, 4> parts =
      class_container::Descriptor::parts(loc, descriptor, logicalType, builder);
  if (parts.empty())
    return mlir::success();

  llvm::SmallVector<mlir::MemRefType, 4> memrefTypes;
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    memrefTypes.push_back(getListHeaderMemRefType(builder.getContext()));
    memrefTypes.push_back(getContainerLockMemRefType(builder.getContext()));
    memrefTypes.push_back(typeConverter.getListItemsMemRefType(listType));
  } else if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    memrefTypes.push_back(getTupleHeaderMemRefType(builder.getContext()));
    memrefTypes.push_back(typeConverter.getTupleItemsMemRefType(tupleType));
  } else if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    memrefTypes.push_back(getDictHeaderMemRefType(builder.getContext()));
    memrefTypes.push_back(getContainerLockMemRefType(builder.getContext()));
    memrefTypes.push_back(typeConverter.getDictKeysMemRefType(dictType));
    memrefTypes.push_back(typeConverter.getDictValuesMemRefType(dictType));
    memrefTypes.push_back(getDictStatesMemRefType(builder.getContext()));
  }
  if (parts.size() != memrefTypes.size())
    return mlir::success();

  llvm::SmallVector<mlir::Value, 4> memrefs;
  memrefs.reserve(parts.size());
  for (auto [part, memrefType] : llvm::zip(parts, memrefTypes)) {
    if (!memrefType)
      return mlir::success();
    memrefs.push_back(builder
                          .create<mlir::UnrealizedConversionCastOp>(
                              loc, memrefType, mlir::ValueRange{part})
                          .getResult(0));
  }
  return container::Elements::refcount(loc, logicalType, memrefs, module,
                                       builder, typeConverter, suffix,
                                       markAggregate);
}
} // namespace class_container::Elements

namespace class_container::MemRef {
mlir::MemRefType headerType(mlir::Type logicalType, mlir::MLIRContext *ctx) {
  if (mlir::isa<ListType>(logicalType))
    return getListHeaderMemRefType(ctx);
  if (mlir::isa<TupleType>(logicalType))
    return getTupleHeaderMemRefType(ctx);
  if (mlir::isa<DictType>(logicalType))
    return getDictHeaderMemRefType(ctx);
  return {};
}

mlir::Value header(mlir::Location loc, mlir::Value descriptor,
                   mlir::Type logicalType, mlir::OpBuilder &builder) {
  mlir::MemRefType type =
      class_container::MemRef::headerType(logicalType, builder.getContext());
  if (!type)
    return {};
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  if (!header)
    return {};
  if (header.getType() == type)
    return header;
  if (mlir::isa<mlir::MemRefType>(header.getType()))
    return builder.create<mlir::memref::CastOp>(loc, type, header);
  return builder
      .create<mlir::UnrealizedConversionCastOp>(loc, type,
                                                mlir::ValueRange{header})
      .getResult(0);
}

mlir::Value slotIndex(mlir::Location loc, int64_t slot,
                      mlir::OpBuilder &builder) {
  return builder.create<mlir::arith::ConstantIndexOp>(loc, slot);
}

mlir::Value present(mlir::Location loc, mlir::Value descriptor,
                    mlir::Type logicalType, mlir::OpBuilder &builder) {
  mlir::Value headerMemRef =
      class_container::MemRef::header(loc, descriptor, logicalType, builder);
  if (!headerMemRef)
    return {};
  auto metadata =
      builder.create<mlir::memref::ExtractStridedMetadataOp>(loc, headerMemRef);
  mlir::Value base =
      builder.create<mlir::memref::ExtractAlignedPointerAsIndexOp>(
          loc, metadata.getBaseBuffer());
  mlir::Value storage =
      builder.create<mlir::arith::AddIOp>(loc, base, metadata.getOffset());
  mlir::Value zero = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  return builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, storage, zero);
}

llvm::SmallVector<mlir::MemRefType, 4>
types(mlir::Type logicalType, const PyLLVMTypeConverter &typeConverter,
      mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::MemRefType, 4> result;
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    result.push_back(getListHeaderMemRefType(ctx));
    result.push_back(getContainerLockMemRefType(ctx));
    result.push_back(typeConverter.getListItemsMemRefType(listType));
    return result;
  }
  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    result.push_back(getTupleHeaderMemRefType(ctx));
    result.push_back(typeConverter.getTupleItemsMemRefType(tupleType));
    return result;
  }
  if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    result.push_back(getDictHeaderMemRefType(ctx));
    result.push_back(getContainerLockMemRefType(ctx));
    result.push_back(typeConverter.getDictKeysMemRefType(dictType));
    result.push_back(typeConverter.getDictValuesMemRefType(dictType));
    result.push_back(getDictStatesMemRefType(ctx));
    return result;
  }
  return result;
}

llvm::StringRef component(mlir::Type logicalType, unsigned index) {
  if (mlir::isa<TupleType>(logicalType)) {
    if (index == 0)
      return ContainerSafetyAttrs::kComponentHeader;
    if (index == 1)
      return ContainerSafetyAttrs::kComponentItems;
    return {};
  }
  if (mlir::isa<ListType>(logicalType)) {
    switch (index) {
    case kListHeaderComponent:
      return ContainerSafetyAttrs::kComponentHeader;
    case kListLockComponent:
      return ContainerSafetyAttrs::kComponentLock;
    case kListItemsComponent:
      return ContainerSafetyAttrs::kComponentItems;
    default:
      return {};
    }
  }
  if (mlir::isa<DictType>(logicalType)) {
    switch (index) {
    case kDictHeaderComponent:
      return ContainerSafetyAttrs::kComponentHeader;
    case kDictLockComponent:
      return ContainerSafetyAttrs::kComponentLock;
    case kDictKeysComponent:
      return ContainerSafetyAttrs::kComponentKeys;
    case kDictValuesComponent:
      return ContainerSafetyAttrs::kComponentValues;
    case kDictStatesComponent:
      return ContainerSafetyAttrs::kComponentStates;
    default:
      return {};
    }
  }
  return {};
}

void dealloc(mlir::Location loc, mlir::Value descriptor, mlir::Type logicalType,
             mlir::ModuleOp module, mlir::OpBuilder &builder,
             const PyLLVMTypeConverter &typeConverter,
             llvm::StringRef deallocGroup = {}) {
  (void)module;
  std::string group = deallocGroup.str();
  if (group.empty()) {
    mlir::Value header =
        class_container::Descriptor::header(loc, descriptor, builder);
    group = class_container::Atomic::group(header);
  }
  llvm::SmallVector<mlir::Value, 4> parts =
      class_container::Descriptor::parts(loc, descriptor, logicalType, builder);
  llvm::SmallVector<mlir::MemRefType, 4> memrefTypes =
      class_container::MemRef::types(logicalType, typeConverter,
                                     builder.getContext());
  if (parts.size() != memrefTypes.size())
    return;
  for (auto [index, part] : llvm::enumerate(parts)) {
    if (!memrefTypes[index])
      return;
    mlir::Value memref =
        builder
            .create<mlir::UnrealizedConversionCastOp>(loc, memrefTypes[index],
                                                      mlir::ValueRange{part})
            .getResult(0);
    auto dealloc = builder.create<mlir::memref::DeallocOp>(loc, memref);
    if (!group.empty())
      dealloc->setAttr(ContainerSafetyAttrs::kDeallocGroup,
                       builder.getStringAttr(group));
    if (llvm::StringRef component =
            class_container::MemRef::component(logicalType, index);
        !component.empty())
      dealloc->setAttr(ContainerSafetyAttrs::kDeallocComponent,
                       builder.getStringAttr(component));
  }
}
} // namespace class_container::MemRef

namespace class_container::Field {
void retainIfManaged(mlir::Location loc, mlir::Value descriptor,
                     mlir::Type logicalType, mlir::ModuleOp module,
                     mlir::OpBuilder &builder, llvm::StringRef premise) {
  (void)module;
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *retainBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value hasHeader =
      class_container::MemRef::present(loc, descriptor, logicalType, builder);
  if (!hasHeader)
    return;
  builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                         afterBlock);

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value headerMemref =
      class_container::MemRef::header(loc, descriptor, logicalType, builder);
  if (!headerMemref)
    return;
  mlir::Value slot =
      class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, zero, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                         afterBlock);

  builder.setInsertionPointToStart(retainBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, one, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic, premise);
  if (premise == ThreadSafetyAttrs::kPremiseOwnedToken)
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
  class_field::Ownership::copyLoadTo(descriptor, previous.getDefiningOp());
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  builder.create<mlir::cf::BranchOp>(loc, afterBlock);

  builder.setInsertionPointToStart(afterBlock);
}

mlir::LogicalResult destroy(mlir::Location loc, mlir::Value descriptor,
                            mlir::Type logicalType, mlir::ModuleOp module,
                            mlir::OpBuilder &builder,
                            const PyLLVMTypeConverter &typeConverter) {
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return mlir::success();

  auto *parent = builder.getInsertionBlock()->getParent();
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *managedBlock = builder.createBlock(parent);
  mlir::Block *localBlock = builder.createBlock(parent);
  mlir::Block *freeBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value hasHeader =
      class_container::MemRef::present(loc, descriptor, logicalType, builder);
  if (!hasHeader)
    return mlir::failure();
  builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                         afterBlock);

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value headerMemref =
      class_container::MemRef::header(loc, descriptor, logicalType, builder);
  if (!headerMemref)
    return mlir::failure();
  mlir::Value slot =
      class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, zero, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  builder.create<mlir::cf::CondBranchOp>(loc, isManaged, managedBlock,
                                         localBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value negOne = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(-1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, negOne, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acq_rel, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRelease,
                          ThreadSafetyAttrs::kOrderingAcqRel);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value shouldFree = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, previous, one);
  builder.create<mlir::cf::CondBranchOp>(loc, shouldFree, freeBlock,
                                         afterBlock);

  builder.setInsertionPointToStart(localBlock);
  if (mlir::failed(class_container::Elements::refcount(
          loc, descriptor, logicalType, module, builder, typeConverter,
          "decref")))
    return mlir::failure();
  builder.create<mlir::cf::BranchOp>(loc, afterBlock);

  builder.setInsertionPointToStart(freeBlock);
  if (mlir::failed(class_container::Elements::refcount(
          loc, descriptor, logicalType, module, builder, typeConverter,
          "decref")))
    return mlir::failure();
  class_container::MemRef::dealloc(loc, descriptor, logicalType, module,
                                   builder, typeConverter,
                                   class_container::Atomic::group(header));
  builder.create<mlir::cf::BranchOp>(loc, afterBlock);

  builder.setInsertionPointToStart(afterBlock);
  return mlir::success();
}
} // namespace class_container::Field

namespace class_field::Runtime {
mlir::LogicalResult single(
    mlir::Location loc, mlir::Value fieldValue, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::OpBuilder &builder,
    const PyLLVMTypeConverter &typeConverter, llvm::StringRef runtimeSymbol,
    llvm::StringRef retainPremise = ThreadSafetyAttrs::kPremiseAggregateBorrow);

mlir::LogicalResult forEach(mlir::Location loc, mlir::Value object,
                            const StaticClassLayout &layout,
                            mlir::ModuleOp module, mlir::OpBuilder &builder,
                            const PyLLVMTypeConverter &typeConverter,
                            llvm::StringRef runtimeSymbol,
                            bool includeContainerFields = true,
                            bool includeDirectClassFields = true) {
  for (auto [index, fieldInfo] : llvm::enumerate(layout.fields)) {
    bool needsObjectRefcount = class_field::Refcount::needed(
        fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
    ClassType directClass = mlir::dyn_cast<ClassType>(fieldInfo.logicalType);
    ClassType listElementClass =
        includeContainerFields
            ? class_element::List::type(fieldInfo.logicalType)
            : ClassType{};
    llvm::SmallVector<std::pair<unsigned, ClassType>, 4> tupleElementClasses;
    if (includeContainerFields)
      class_element::Tuple::slots(fieldInfo.logicalType, tupleElementClasses);
    auto [dictKeyClass, dictValueClass] =
        includeContainerFields
            ? class_element::Dict::types(fieldInfo.logicalType)
            : std::pair<ClassType, ClassType>{};
    if (!needsObjectRefcount && (!directClass || !includeDirectClassFields) &&
        !listElementClass && tupleElementClasses.empty() && !dictKeyClass &&
        !dictValueClass)
      continue;

    mlir::Value fieldValue = class_object::Payload::loadField(
        loc, object, layout, static_cast<unsigned>(index), builder);
    if (!fieldValue)
      continue;
    class_field::Ownership::markLoad(fieldValue, static_cast<unsigned>(index));
    if (needsObjectRefcount) {
      if (mlir::failed(single(loc, fieldValue, fieldInfo.logicalType, module,
                              builder, typeConverter, runtimeSymbol,
                              ThreadSafetyAttrs::kPremiseAggregateBorrow)))
        return mlir::failure();
      continue;
    }

    llvm::StringRef suffix;
    if (runtimeSymbol == RuntimeSymbols::kIncRef) {
      suffix = "incref";
    } else if (runtimeSymbol == kDecrefIntent) {
      suffix = "decref";
    } else {
      continue;
    }
    if (directClass && includeDirectClassFields) {
      Slot::classRefcount(loc, fieldValue, directClass, module, builder, suffix,
                          /*aggregateEffect=*/true,
                          ThreadSafetyAttrs::kPremiseAggregateBorrow);
      continue;
    }
    if (listElementClass || !tupleElementClasses.empty() || dictKeyClass ||
        dictValueClass) {
      if (mlir::failed(class_container::Elements::refcount(
              loc, fieldValue, fieldInfo.logicalType, module, builder,
              typeConverter, suffix)))
        return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult single(mlir::Location loc, mlir::Value fieldValue,
                           mlir::Type logicalType, mlir::ModuleOp module,
                           mlir::OpBuilder &builder,
                           const PyLLVMTypeConverter &typeConverter,
                           llvm::StringRef runtimeSymbol,
                           llvm::StringRef retainPremise) {
  if (class_field::Refcount::Parts::storage(logicalType,
                                            fieldValue.getType())) {
    llvm::SmallVector<mlir::Value, 3> parts =
        class_field::Refcount::Parts::parts(loc, fieldValue, builder);
    if (parts.empty())
      return mlir::success();
    RuntimeAPI runtime(module, builder, typeConverter);
    RuntimeAPI::Call call;
    if (runtimeSymbol == RuntimeSymbols::kIncRef) {
      call = runtime.call(loc, RuntimeSymbols::kIncRef, mlir::Type(),
                          mlir::ValueRange{parts[0]});
      ownership::aggregate::Slot::markLoad(fieldValue);
      ownership::aggregate::Slot::markLoad(parts[0]);
      class_field::Ownership::copyLoadTo(parts[0], call.getOperation());
      call->setAttr(OwnershipContractAttrs::kAggregateRetain,
                    builder.getUnitAttr());
      threadsafe::Retain::premise(call.getOperation(), retainPremise);
      return mlir::success();
    }
    if (runtimeSymbol == kDecrefIntent) {
      llvm::StringRef releaseSymbol =
          class_field::Refcount::Parts::releaseSymbol(logicalType);
      if (releaseSymbol.empty())
        return mlir::success();
      call = runtime.call(loc, releaseSymbol, mlir::Type(),
                          mlir::ValueRange(parts));
      ownership::aggregate::Slot::markLoad(fieldValue);
      for (mlir::Value part : parts)
        ownership::aggregate::Slot::markLoad(part);
      class_field::Ownership::copyLoadTo(parts[0], call.getOperation());
      call->setAttr(OwnershipContractAttrs::kAggregateRelease,
                    builder.getUnitAttr());
      return mlir::success();
    }
    return mlir::success();
  }
  if (isReleaseSymbol(runtimeSymbol)) {
    mlir::emitError(loc)
        << "class field decref requires a concrete header/payload release "
           "helper; generic Ly_DecRef is not part of the object ABI";
    return mlir::failure();
  }
  auto call = emitRuntimeVoidCall(loc, module, builder, typeConverter,
                                  runtimeSymbol, mlir::ValueRange{fieldValue});
  if (isRetainSymbol(runtimeSymbol)) {
    ownership::aggregate::Slot::markLoad(fieldValue);
    class_field::Ownership::copyLoadTo(fieldValue, call.getOperation());
    call->setAttr(OwnershipContractAttrs::kAggregateRetain,
                  builder.getUnitAttr());
    threadsafe::Retain::premise(call.getOperation(), retainPremise);
  }
  return mlir::success();
}
} // namespace class_field::Runtime

namespace class_field::Destroy {
mlir::LogicalResult all(mlir::Location loc, mlir::Value object,
                        const StaticClassLayout &layout, mlir::ModuleOp module,
                        mlir::OpBuilder &builder,
                        const PyLLVMTypeConverter &typeConverter) {
  for (auto [index, fieldInfo] : llvm::enumerate(layout.fields)) {
    mlir::Value fieldValue = class_object::Payload::loadField(
        loc, object, layout, static_cast<unsigned>(index), builder);
    if (!fieldValue)
      continue;
    class_field::Ownership::markLoad(fieldValue, static_cast<unsigned>(index));
    if (auto directClass = mlir::dyn_cast<ClassType>(fieldInfo.logicalType)) {
      Slot::classRefcount(loc, fieldValue, directClass, module, builder,
                          "decref", /*aggregateEffect=*/true,
                          ThreadSafetyAttrs::kPremiseAggregateBorrow);
      continue;
    }
    if (class_field::Refcount::needed(fieldInfo.logicalType,
                                      fieldInfo.storageType, typeConverter)) {
      if (mlir::failed(class_field::Runtime::single(
              loc, fieldValue, fieldInfo.logicalType, module, builder,
              typeConverter, kDecrefIntent)))
        return mlir::failure();
      continue;
    }
    if (mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType)) {
      ownership::aggregate::Slot::markLoad(fieldValue);
      if (mlir::failed(class_container::Field::destroy(
              loc, fieldValue, fieldInfo.logicalType, module, builder,
              typeConverter)))
        return mlir::failure();
    }
  }
  return mlir::success();
}
} // namespace class_field::Destroy

namespace raw_memref {
mlir::Value allocRank1Descriptor(mlir::Location loc, mlir::ModuleOp module,
                                 mlir::Type elementType, int64_t elementCount,
                                 mlir::OpBuilder &builder,
                                 bool ownedRoot = true,
                                 bool objectHeader = false) {
  (void)module;
  auto memrefType = mlir::MemRefType::get({elementCount}, elementType);
  mlir::Value storage = builder.create<mlir::memref::AllocOp>(loc, memrefType);
  if (mlir::Operation *def = storage.getDefiningOp()) {
    if (ownedRoot)
      def->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                   builder.getUnitAttr());
    if (objectHeader)
      def->setAttr(OwnershipContractAttrs::kObjectHeader,
                   builder.getUnitAttr());
    else
      def->setAttr(ClassSafetyAttrs::kPayloadPart, builder.getUnitAttr());
    def->setAttr(ClassSafetyAttrs::kPromoteFreshObject, builder.getUnitAttr());
  }
  return storage;
}

} // namespace raw_memref

namespace class_helper::Parts {
llvm::SmallVector<mlir::Type, 2> types(const StaticClassLayout &layout) {
  llvm::SmallVector<mlir::Type, 2> result;
  class_layout::partsValueTypes(layout, result);
  return result;
}

mlir::Value object(mlir::Location loc, mlir::ValueRange values,
                   const StaticClassLayout &layout, mlir::OpBuilder &builder) {
  if (values.size() == 1)
    return values.front();
  int64_t payloadCount = class_layout::Payload::partCount(layout);
  if (values.size() != static_cast<size_t>(1 + payloadCount))
    return {};
  return class_object::Local::objectValueFromParts(
      loc, values[0], values.drop_front(), layout, builder);
}

mlir::Value objectFromArgs(mlir::Location loc, mlir::Block *entry,
                           unsigned firstArg, const StaticClassLayout &layout,
                           mlir::OpBuilder &builder) {
  int64_t payloadCount = class_layout::Payload::partCount(layout);
  if (!entry || entry->getNumArguments() < firstArg + 1 + payloadCount)
    return {};
  llvm::SmallVector<mlir::Value, 8> values;
  values.reserve(1 + payloadCount);
  for (int64_t index = 0; index < 1 + payloadCount; ++index)
    values.push_back(
        entry->getArgument(firstArg + static_cast<unsigned>(index)));
  return object(loc, values, layout, builder);
}

void append(mlir::Location loc, mlir::Value object,
            const StaticClassLayout &layout, mlir::OpBuilder &builder,
            llvm::SmallVectorImpl<mlir::Value> &args) {
  mlir::Value header =
      class_object::Local::header(loc, object, layout, builder);
  if (!header)
    return;
  args.push_back(header);
  for (int64_t index = 0; index < class_layout::Payload::partCount(layout);
       ++index) {
    mlir::Value part =
        class_object::Local::payloadPart(loc, object, layout, index, builder);
    if (!part)
      return;
    args.push_back(part);
  }
}

void append(mlir::Location loc, mlir::ValueRange values,
            const StaticClassLayout &layout, mlir::OpBuilder &builder,
            llvm::SmallVectorImpl<mlir::Value> &args) {
  int64_t payloadCount = class_layout::Payload::partCount(layout);
  if (values.size() == static_cast<size_t>(1 + payloadCount) &&
      payloadCount >= 1) {
    mlir::Value header = object_abi::Header::isOwned(values[0].getType())
                             ? values[0]
                             : object_abi::Type::castStorage(
                                   loc, values[0], layout.headerType, builder);
    if (!header || header.getType() != layout.headerType)
      return;
    args.push_back(header);
    for (int64_t index = 0; index < payloadCount; ++index) {
      mlir::Value part = values[static_cast<size_t>(1 + index)];
      mlir::MemRefType partType =
          class_layout::Payload::partType(layout, index);
      if (!partType)
        return;
      if (part.getType() != partType &&
          mlir::isa<mlir::MemRefType>(part.getType()))
        part = builder.create<mlir::memref::CastOp>(loc, partType, part);
      if (!part || part.getType() != partType)
        return;
      args.push_back(part);
    }
    return;
  }
  if (values.size() == 1)
    append(loc, values.front(), layout, builder, args);
}
} // namespace class_helper::Parts

static llvm::StringRef containerDescriptorKind(mlir::Type logicalType) {
  if (mlir::isa<ListType>(logicalType))
    return "list";
  if (mlir::isa<TupleType>(logicalType))
    return "tuple";
  if (mlir::isa<DictType>(logicalType))
    return "dict";
  return {};
}

static llvm::StringRef containerDescriptorComponent(mlir::Type logicalType,
                                                    unsigned index) {
  if (mlir::isa<ListType, TupleType>(logicalType)) {
    if (index == 0)
      return ContainerSafetyAttrs::kComponentHeader;
    if (index == 1)
      return ContainerSafetyAttrs::kComponentItems;
    return {};
  }
  if (mlir::isa<DictType>(logicalType)) {
    switch (index) {
    case 0:
      return ContainerSafetyAttrs::kComponentHeader;
    case 1:
      return ContainerSafetyAttrs::kComponentKeys;
    case 2:
      return ContainerSafetyAttrs::kComponentValues;
    case 3:
      return ContainerSafetyAttrs::kComponentStates;
    default:
      return {};
    }
  }
  return {};
}

namespace class_field::ABI {
llvm::SmallVector<mlir::Type, 4>
types(mlir::Operation *from, mlir::ModuleOp module,
      const StaticClassFieldInfo &fieldInfo,
      const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 4> result;
  if (auto classType = mlir::dyn_cast<ClassType>(fieldInfo.logicalType)) {
    mlir::FailureOr<StaticClassLayout> layout = class_layout::get(
        from ? from : module.getOperation(), classType, typeConverter);
    if (mlir::succeeded(layout))
      class_layout::partsValueTypes(*layout, result);
    return result;
  }

  if (mlir::succeeded(
          typeConverter.convertType(fieldInfo.logicalType, result)) &&
      !result.empty())
    return result;

  result.push_back(fieldInfo.storageType);
  return result;
}

static mlir::Value castIntegerStorage(mlir::Location loc, mlir::Value value,
                                      mlir::Type storageType,
                                      mlir::OpBuilder &builder) {
  auto source = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto target = mlir::dyn_cast<mlir::IntegerType>(storageType);
  if (!source || !target)
    return {};
  if (source.getWidth() == target.getWidth())
    return value;
  if (source.getWidth() < target.getWidth())
    return builder.create<mlir::arith::ExtSIOp>(loc, storageType, value);
  return builder.create<mlir::arith::TruncIOp>(loc, storageType, value);
}

static mlir::Value castPart(mlir::Location loc, mlir::Value value,
                            mlir::Type targetType, mlir::OpBuilder &builder) {
  if (!value)
    return {};
  if (value.getType() == targetType)
    return value;
  if (auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(targetType)) {
    if (mlir::isa<mlir::MemRefType>(value.getType()))
      return builder.create<mlir::memref::CastOp>(loc, targetMemRef, value);
    return class_object::Local::memRefFromDescriptor(loc, value, targetMemRef,
                                                     builder);
  }
  return builder
      .create<mlir::UnrealizedConversionCastOp>(loc, targetType,
                                                mlir::ValueRange{value})
      .getResult(0);
}

mlir::FailureOr<mlir::Value>
storage(mlir::Location loc, mlir::ValueRange values,
        const StaticClassFieldInfo &fieldInfo, mlir::ModuleOp module,
        mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter) {
  mlir::Type logicalType = fieldInfo.logicalType;
  mlir::Type storageType = fieldInfo.storageType;
  if (values.empty())
    return mlir::failure();

  if (values.size() == 1) {
    mlir::Value value = values.front();
    if (value.getType() == storageType)
      return value;
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
      mlir::Value descriptor = class_object::Local::descriptorFromMemRef(
          loc, value, memrefType, builder);
      if (descriptor && descriptor.getType() == storageType)
        return descriptor;
    }
  }

  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    mlir::FailureOr<StaticClassLayout> layout =
        class_layout::get(module.getOperation(), classType, typeConverter);
    if (mlir::failed(layout) || storageType != layout->objectType)
      return mlir::failure();
    mlir::Value object =
        class_helper::Parts::object(loc, values, *layout, builder);
    if (!object || object.getType() != storageType)
      return mlir::failure();
    return object;
  }

  if (mlir::isa<IntType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageType) && values.size() == 3) {
    RuntimeAPI runtime(module, builder, typeConverter);
    mlir::Value value =
        runtime
            .call(loc, RuntimeSymbols::kLongAsI64, builder.getI64Type(), values)
            .getResult();
    if (!value)
      return mlir::failure();
    mlir::Value casted = castIntegerStorage(loc, value, storageType, builder);
    if (!casted)
      return mlir::failure();
    return casted;
  }

  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storageType);
  if (!aggregate || aggregate.isOpaque() ||
      aggregate.getBody().size() != values.size())
    return mlir::failure();

  mlir::Value storageValue =
      builder.create<mlir::LLVM::UndefOp>(loc, storageType);
  for (auto [index, value] : llvm::enumerate(values)) {
    mlir::Type partType = aggregate.getBody()[index];
    mlir::Value part = value;
    if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(part.getType()))
      part = class_object::Local::descriptorFromMemRef(loc, part, memrefType,
                                                       builder);
    if (!part)
      return mlir::failure();
    if (part.getType() != partType)
      part = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     loc, partType, mlir::ValueRange{part})
                 .getResult(0);
    storageValue = builder.create<mlir::LLVM::InsertValueOp>(
        loc, storageType, storageValue, part,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
  }
  return storageValue;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value>>
parts(mlir::Location loc, mlir::Value storageValue,
      const StaticClassFieldInfo &fieldInfo, mlir::ModuleOp module,
      mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 4> abiTypes =
      types(storageValue.getDefiningOp(), module, fieldInfo, typeConverter);
  if (abiTypes.empty())
    return mlir::failure();

  if (abiTypes.size() == 1) {
    mlir::Value value = castPart(loc, storageValue, abiTypes.front(), builder);
    if (!value)
      return mlir::failure();
    return llvm::SmallVector<mlir::Value>{value};
  }

  if (auto classType = mlir::dyn_cast<ClassType>(fieldInfo.logicalType)) {
    mlir::FailureOr<StaticClassLayout> layout =
        class_layout::get(module.getOperation(), classType, typeConverter);
    if (mlir::failed(layout) || storageValue.getType() != layout->objectType)
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 2> values;
    class_helper::Parts::append(loc, storageValue, *layout, builder, values);
    if (values.size() != abiTypes.size())
      return mlir::failure();
    for (auto [index, value] : llvm::enumerate(values)) {
      mlir::Value casted = castPart(loc, value, abiTypes[index], builder);
      if (!casted)
        return mlir::failure();
      values[index] = casted;
    }
    return llvm::SmallVector<mlir::Value>(values.begin(), values.end());
  }

  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storageValue.getType());
  if (!aggregate || aggregate.isOpaque() ||
      aggregate.getBody().size() < abiTypes.size())
    return mlir::failure();

  llvm::StringRef containerKind =
      containerDescriptorKind(fieldInfo.logicalType);
  mlir::Operation *owner = storageValue.getDefiningOp();
  if (!owner && builder.getInsertionBlock())
    owner = builder.getInsertionBlock()->getParentOp();
  std::string containerGroup =
      containerKind.empty()
          ? std::string()
          : container::descriptor::Group::make(owner, containerKind);

  llvm::SmallVector<mlir::Value> values;
  values.reserve(abiTypes.size());
  for (auto [index, type] : llvm::enumerate(abiTypes)) {
    mlir::Value part = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, aggregate.getBody()[index], storageValue,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    mlir::Value value = castPart(loc, part, type, builder);
    if (!value)
      return mlir::failure();
    llvm::StringRef component =
        containerDescriptorComponent(fieldInfo.logicalType, index);
    if (!component.empty()) {
      container::descriptor::Component::mark(part, containerGroup, component);
      container::descriptor::Component::mark(value, containerGroup, component);
    }
    values.push_back(value);
  }
  return values;
}
} // namespace class_field::ABI

namespace class_helper::Retain {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 2> objectArgTypes;
  class_layout::partsValueTypes(layout, objectArgTypes);
  std::string helperName = getClassHelperName(classType, "incref");
  auto fn =
      getOrInsertClassFunc(loc, module, builder, helperName, objectArgTypes);
  ownership::effect::retain(fn.getOperation(), {0, 1});
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindIncref);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value object =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!object)
    return fn;

  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::BranchOp>(loc, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managed =
      class_object::Managed::load(loc, object, layout, builder);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, managed, managedBlock,
                                         localBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value header =
      class_object::Local::header(loc, object, layout, builder);
  if (!header)
    return fn;
  class_object::Refcount::inc(loc, header, module, builder);
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(localBlock);
  if (mlir::failed(
          class_field::Runtime::forEach(loc, object, layout, module, builder,
                                        typeConverter, RuntimeSymbols::kIncRef,
                                        /*includeContainerFields=*/false)))
    return {};
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::func::ReturnOp>(loc);
  return fn;
}
} // namespace class_helper::Retain

namespace class_helper::Release {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 2> objectArgTypes;
  class_layout::partsValueTypes(layout, objectArgTypes);
  std::string helperName = getClassHelperName(classType, "decref");
  auto fn =
      getOrInsertClassFunc(loc, module, builder, helperName, objectArgTypes);
  ownership::effect::release(fn.getOperation(), {0, 1});
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindDecref);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value object =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!object)
    return fn;

  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::BranchOp>(loc, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managed =
      class_object::Managed::load(loc, object, layout, builder);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, managed, managedBlock,
                                         localBlock);

  builder.setInsertionPointToStart(localBlock);
  if (mlir::failed(class_field::Runtime::forEach(
          loc, object, layout, module, builder, typeConverter, kDecrefIntent,
          /*includeContainerFields=*/false)))
    return {};
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value header =
      class_object::Local::header(loc, object, layout, builder);
  if (!header)
    return fn;
  mlir::Value isZero =
      class_object::Refcount::decAndIsZero(loc, header, module, builder);
  mlir::Block *destroyBlock = builder.createBlock(&fn.getBody());
  mlir::Block *managedRetBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, isZero, destroyBlock,
                                         managedRetBlock);

  builder.setInsertionPointToStart(destroyBlock);
  if (mlir::failed(class_field::Destroy::all(loc, object, layout, module,
                                             builder, typeConverter)))
    return {};
  class_object::Local::deallocParts(loc, object, layout, builder);
  // The class object carrier is a caller-owned slot/view.  A refcount release
  // owns only the header/payload pair stored in that slot.
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(managedRetBlock);
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::func::ReturnOp>(loc);
  return fn;
}
} // namespace class_helper::Release

namespace class_helper::LocalDestroy {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 2> objectArgTypes;
  class_layout::partsValueTypes(layout, objectArgTypes);
  std::string helperName = getClassHelperName(classType, "destroy_local");
  auto fn =
      getOrInsertClassFunc(loc, module, builder, helperName, objectArgTypes);
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindDestroyLocal);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value object =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!object)
    return fn;

  mlir::Block *retBlock = builder.createBlock(&fn.getBody());
  mlir::Block *destroyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::BranchOp>(loc, destroyBlock);

  builder.setInsertionPointToStart(destroyBlock);
  if (mlir::failed(class_field::Destroy::all(loc, object, layout, module,
                                             builder, typeConverter)))
    return {};
  builder.create<mlir::cf::BranchOp>(loc, retBlock);

  builder.setInsertionPointToStart(retBlock);
  builder.create<mlir::func::ReturnOp>(loc);
  return fn;
}
} // namespace class_helper::LocalDestroy

namespace class_helper::Repr {
void ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
            mlir::OpBuilder &builder,
            const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 3> reprTypes;
  if (mlir::failed(typeConverter.convertType(StrType::get(builder.getContext()),
                                             reprTypes)) ||
      reprTypes.empty())
    return;
  bool splitRepr = reprTypes.size() == 2 &&
                   object_abi::str_abi::Parts::isHeader(reprTypes[0]) &&
                   object_abi::str_abi::Parts::isBytes(reprTypes[1]);
  if (!splitRepr)
    return;
  mlir::FailureOr<StaticClassLayout> layoutOr =
      class_layout::get(module, classType, typeConverter);
  if (mlir::failed(layoutOr))
    return;
  const StaticClassLayout &layout = *layoutOr;
  llvm::SmallVector<mlir::Type, 2> objectArgTypes =
      class_helper::Parts::types(layout);

  std::string helperName = getClassHelperName(classType, "repr");
  std::string customName = (classType.getClassName() + ".__repr__").str();
  if (auto customFunc = module.lookupSymbol<mlir::func::FuncOp>(customName)) {
    if (module.lookupSymbol<mlir::func::FuncOp>(helperName))
      return;
    mlir::FunctionType customType = customFunc.getFunctionType();
    if (customType.getNumResults() != reprTypes.size() ||
        !llvm::equal(customType.getResults(), reprTypes))
      return;
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    auto funcType =
        mlir::FunctionType::get(module.getContext(), objectArgTypes, reprTypes);
    auto wrapper =
        builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
    wrapper.setVisibility(mlir::SymbolTable::Visibility::Private);
    class_helper::Schema::mark(wrapper, layout, typeConverter);
    llvm::SmallVector<mlir::Attribute> ownedResults;
    ownedResults.push_back(builder.getI64IntegerAttr(0));
    wrapper->setAttr(OwnershipContractAttrs::kOwnedResults,
                     builder.getArrayAttr(ownedResults));
    mlir::Block *entry = wrapper.addEntryBlock();
    builder.setInsertionPointToStart(entry);
    llvm::SmallVector<mlir::Value, 2> callArgs;
    if (customType.getNumInputs() == objectArgTypes.size() &&
        llvm::equal(customType.getInputs(), objectArgTypes)) {
      callArgs.append(entry->args_begin(), entry->args_end());
    } else if (customType.getNumInputs() == 1) {
      mlir::Value self =
          class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
      if (!self || customType.getInput(0) != self.getType())
        return;
      callArgs.push_back(self);
    } else {
      return;
    }
    auto call = builder.create<mlir::func::CallOp>(loc, customFunc, callArgs);
    builder.create<mlir::func::ReturnOp>(loc, call.getResults());
    return;
  }

  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(helperName)) {
    (void)existing;
    return;
  }

  mlir::OpBuilder::InsertionGuard moduleGuard(builder);
  builder.setInsertionPointToStart(module.getBody());
  auto funcType =
      mlir::FunctionType::get(module.getContext(), objectArgTypes, reprTypes);
  auto fn = builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
  class_helper::Schema::mark(fn, layout, typeConverter);
  llvm::SmallVector<mlir::Attribute> ownedResults;
  ownedResults.push_back(builder.getI64IntegerAttr(0));
  fn->setAttr(OwnershipContractAttrs::kOwnedResults,
              builder.getArrayAttr(ownedResults));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  mlir::Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  RuntimeAPI runtime(module, builder, typeConverter);

  mlir::Value self =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!self)
    return;
  mlir::Value id = class_object::Identity::value(loc, self, &layout, builder);
  if (!id)
    return;
  auto literal =
      [&](llvm::StringRef text) -> llvm::SmallVector<mlir::Value, 3> {
    mlir::Value bytes =
        runtime.getByteLiteral(loc, builder.getStringAttr(text));
    mlir::Value start = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    mlir::Value length =
        runtime.getI64Constant(loc, static_cast<int64_t>(text.size()));
    auto call = runtime.call(loc, RuntimeSymbols::kUnicodeFromBytes,
                             mlir::TypeRange(reprTypes),
                             mlir::ValueRange{bytes, start, length});
    return llvm::SmallVector<mlir::Value, 3>(call.getResults());
  };
  auto release = [&](mlir::ValueRange value) {
    runtime.call(loc, RuntimeSymbols::kUnicodeDecRef, mlir::Type(), value);
  };
  auto concatOwned =
      [&](mlir::ValueRange lhs,
          mlir::ValueRange rhs) -> llvm::SmallVector<mlir::Value, 3> {
    llvm::SmallVector<mlir::Value, 4> operands;
    operands.append(lhs.begin(), lhs.end());
    operands.append(rhs.begin(), rhs.end());
    auto call = runtime.call(loc, RuntimeSymbols::kUnicodeConcat,
                             mlir::TypeRange(reprTypes), operands);
    release(lhs);
    release(rhs);
    return llvm::SmallVector<mlir::Value, 3>(call.getResults());
  };

  llvm::SmallVector<mlir::Value, 3> open = literal("<");
  llvm::SmallVector<mlir::Value, 3> name = literal(classType.getClassName());
  llvm::SmallVector<mlir::Value, 3> withName = concatOwned(open, name);
  llvm::SmallVector<mlir::Value, 3> middle = literal(" object at ");
  llvm::SmallVector<mlir::Value, 3> withMiddle = concatOwned(withName, middle);
  auto idCall = runtime.call(loc, RuntimeSymbols::kUnicodeFromI64,
                             mlir::TypeRange(reprTypes), mlir::ValueRange{id});
  llvm::SmallVector<mlir::Value, 3> idText(idCall.getResults());
  llvm::SmallVector<mlir::Value, 3> withId = concatOwned(withMiddle, idText);
  llvm::SmallVector<mlir::Value, 3> close = literal(">");
  llvm::SmallVector<mlir::Value, 3> result = concatOwned(withId, close);
  builder.create<mlir::func::ReturnOp>(loc, result);
}
} // namespace class_helper::Repr

namespace class_helper::Eq {

bool descriptorLike(mlir::LLVM::LLVMStructType type) {
  return class_layout::isDescriptorStorageType(type);
}

mlir::Value descriptorEqual(mlir::Location loc,
                            mlir::LLVM::LLVMStructType descriptorType,
                            mlir::Value lhs, mlir::Value rhs,
                            mlir::OpBuilder &builder) {
  auto i1Type = builder.getI1Type();
  mlir::Value equal = builder.create<mlir::LLVM::ConstantOp>(
      loc, i1Type, builder.getBoolAttr(true));
  auto andEqual = [&](mlir::Value current, mlir::Value next) {
    return builder.create<mlir::LLVM::AndOp>(loc, current, next).getResult();
  };
  auto compare = [&](llvm::ArrayRef<int64_t> position) {
    mlir::Type componentType = descriptorType.getBody()[position.front()];
    if (position.size() == 2) {
      auto arrayType = mlir::cast<mlir::LLVM::LLVMArrayType>(componentType);
      componentType = arrayType.getElementType();
    }
    mlir::Value lhsPart = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, componentType, lhs, builder.getDenseI64ArrayAttr(position));
    mlir::Value rhsPart = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, componentType, rhs, builder.getDenseI64ArrayAttr(position));
    return builder.create<mlir::LLVM::ICmpOp>(
        loc, mlir::LLVM::ICmpPredicate::eq, lhsPart, rhsPart);
  };
  equal = andEqual(equal, compare({1}));
  equal = andEqual(equal, compare({2}));
  equal = andEqual(equal, compare({3, 0}));
  return equal;
}

mlir::Value partsDescriptorAggregateEqual(mlir::Location loc,
                                          mlir::LLVM::LLVMStructType type,
                                          mlir::Value lhs, mlir::Value rhs,
                                          mlir::OpBuilder &builder) {
  if (!type || type.isOpaque() || type.getBody().empty())
    return {};
  auto i1Type = builder.getI1Type();
  mlir::Value equal = builder.create<mlir::LLVM::ConstantOp>(
      loc, i1Type, builder.getBoolAttr(true));
  for (auto [index, fieldType] : llvm::enumerate(type.getBody())) {
    auto descriptorType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(fieldType);
    if (!descriptorLike(descriptorType))
      return {};
    mlir::Value lhsPart = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptorType, lhs,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    mlir::Value rhsPart = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, descriptorType, rhs,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    mlir::Value partEqual =
        descriptorEqual(loc, descriptorType, lhsPart, rhsPart, builder);
    equal = builder.create<mlir::LLVM::AndOp>(loc, equal, partEqual);
  }
  return equal;
}

mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  auto i1Type = builder.getI1Type();
  std::string helperName = getClassHelperName(classType, "eq");
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 2> objectArgTypes =
      class_helper::Parts::types(layout);
  inputTypes.append(objectArgTypes.begin(), objectArgTypes.end());
  inputTypes.append(objectArgTypes.begin(), objectArgTypes.end());
  auto fn = getOrInsertClassFunc(loc, module, builder, helperName, inputTypes,
                                 {i1Type});
  class_helper::Schema::mark(fn, layout, typeConverter);
  class_helper::Schema::markObjectArg(
      fn, layout, static_cast<unsigned>(objectArgTypes.size()));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);
  mlir::Value lhsObject =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  unsigned rhsFirstArg = static_cast<unsigned>(objectArgTypes.size());
  mlir::Value rhsObject = class_helper::Parts::objectFromArgs(
      loc, entry, rhsFirstArg, layout, builder);
  if (!lhsObject || !rhsObject)
    return fn;
  mlir::Value lhsIdentity =
      class_object::Identity::value(loc, lhsObject, &layout, builder);
  mlir::Value rhsIdentity =
      class_object::Identity::value(loc, rhsObject, &layout, builder);
  if (!lhsIdentity || !rhsIdentity)
    return fn;
  mlir::Value equal = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::eq, lhsIdentity, rhsIdentity);
  if (!layout.fields.empty()) {
    mlir::Value fieldsEqual = builder.create<mlir::LLVM::ConstantOp>(
        loc, i1Type, builder.getBoolAttr(true));
    for (auto [fieldIndex, fieldInfo] : llvm::enumerate(layout.fields)) {
      mlir::Value lhs = class_object::Payload::loadField(
          loc, lhsObject, layout, static_cast<unsigned>(fieldIndex), builder);
      mlir::Value rhs = class_object::Payload::loadField(
          loc, rhsObject, layout, static_cast<unsigned>(fieldIndex), builder);
      if (!lhs || !rhs) {
        builder.create<mlir::func::ReturnOp>(loc, equal);
        return fn;
      }
      mlir::Value fieldEqual;
      if (mlir::isa<mlir::IntegerType>(fieldInfo.storageType)) {
        fieldEqual = builder.create<mlir::LLVM::ICmpOp>(
            loc, mlir::LLVM::ICmpPredicate::eq, lhs, rhs);
      } else if (auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
                     fieldInfo.storageType);
                 descriptorLike(structType)) {
        fieldEqual = descriptorEqual(loc, structType, lhs, rhs, builder);
      } else if (auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(
                     fieldInfo.storageType)) {
        fieldEqual =
            partsDescriptorAggregateEqual(loc, structType, lhs, rhs, builder);
        if (!fieldEqual) {
          builder.create<mlir::func::ReturnOp>(loc, equal);
          return fn;
        }
      } else {
        builder.create<mlir::func::ReturnOp>(loc, equal);
        return fn;
      }
      fieldsEqual =
          builder.create<mlir::arith::AndIOp>(loc, fieldsEqual, fieldEqual);
    }
    equal = builder.create<mlir::arith::OrIOp>(loc, equal, fieldsEqual);
  }
  builder.create<mlir::func::ReturnOp>(loc, equal);
  return fn;
}
} // namespace class_helper::Eq

namespace class_container::Clone {
mlir::Value rank1Descriptor(mlir::Location loc, mlir::Value descriptor,
                            mlir::MemRefType memrefType,
                            mlir::OpBuilder &builder) {
  auto descriptorType =
      mlir::cast<mlir::LLVM::LLVMStructType>(descriptor.getType());
  mlir::Value source = builder
                           .create<mlir::UnrealizedConversionCastOp>(
                               loc, memrefType, mlir::ValueRange{descriptor})
                           .getResult(0);
  mlir::Value size = builder.create<mlir::memref::DimOp>(loc, source, 0);
  llvm::SmallVector<mlir::Value, 1> dynamicSizes;
  if (memrefType.isDynamicDim(0))
    dynamicSizes.push_back(size);
  auto clonedStorage =
      builder.create<mlir::memref::AllocOp>(loc, memrefType, dynamicSizes);

  mlir::Value lower = createIndexConstant(loc, builder, 0);
  mlir::Value step = createIndexConstant(loc, builder, 1);
  auto loop = builder.create<mlir::scf::ForOp>(loc, lower, size, step);
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(loop.getBody());
    mlir::Value iv = loop.getInductionVar();
    mlir::Value item = builder.create<mlir::memref::LoadOp>(loc, source, iv);
    auto store =
        builder.create<mlir::memref::StoreOp>(loc, item, clonedStorage, iv);
    if (mlir::isa<mlir::LLVM::LLVMStructType>(memrefType.getElementType())) {
      ownership::aggregate::Slot::markLoad(item);
      ownership::aggregate::Slot::markStore(store.getOperation());
    }
  }

  mlir::Value cloned =
      builder
          .create<mlir::UnrealizedConversionCastOp>(
              loc, descriptorType, mlir::ValueRange{clonedStorage})
          .getResult(0);
  return cloned;
}

void markOwnership(mlir::Location loc, mlir::Value headerDescriptor,
                   mlir::MemRefType headerType, int64_t markerSlot,
                   int64_t markerValue, mlir::OpBuilder &builder) {
  mlir::Value header =
      builder
          .create<mlir::UnrealizedConversionCastOp>(
              loc, headerType, mlir::ValueRange{headerDescriptor})
          .getResult(0);
  auto store = builder.create<mlir::memref::StoreOp>(
      loc, createI64Constant(loc, builder, markerValue), header,
      createIndexConstant(loc, builder, markerSlot));
  store->setAttr(ContainerSafetyAttrs::kRefcountInit,
                 builder.getI64IntegerAttr(markerValue));
  store->setAttr(ContainerSafetyAttrs::kRefcountState,
                 builder.getStringAttr(ContainerSafetyAttrs::kStateManaged));
}

mlir::Value storage(mlir::Location loc, mlir::Value storageValue,
                    mlir::Type logicalType, mlir::OpBuilder &builder,
                    const PyLLVMTypeConverter &typeConverter,
                    int64_t initialRefcount = 1) {
  auto appendDescriptor = [&](unsigned index, mlir::MemRefType memrefType,
                              mlir::Value &result) {
    auto storageStruct =
        mlir::cast<mlir::LLVM::LLVMStructType>(result.getType());
    mlir::Type partType = storageStruct.getBody()[index];
    mlir::Value descriptor = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, partType, result, builder.getDenseI64ArrayAttr({index}));
    mlir::Value cloned = class_container::Clone::rank1Descriptor(
        loc, descriptor, memrefType, builder);
    result = builder.create<mlir::LLVM::InsertValueOp>(
        loc, storageStruct, result, cloned,
        builder.getDenseI64ArrayAttr({index}));
    return cloned;
  };

  mlir::Value result = storageValue;
  if (auto listType = mlir::dyn_cast<ListType>(logicalType)) {
    auto headerType = getListHeaderMemRefType(builder.getContext());
    auto lockType = getContainerLockMemRefType(builder.getContext());
    auto itemsType = typeConverter.getListItemsMemRefType(listType);
    if (!itemsType)
      return storageValue;
    mlir::Value header = appendDescriptor(0, headerType, result);
    appendDescriptor(1, lockType, result);
    appendDescriptor(2, itemsType, result);
    class_container::Clone::markOwnership(loc, header, headerType,
                                          kTypedListRefcountSlot,
                                          initialRefcount, builder);
    return result;
  }
  if (auto tupleType = mlir::dyn_cast<TupleType>(logicalType)) {
    auto headerType = getTupleHeaderMemRefType(builder.getContext());
    mlir::Value header = appendDescriptor(0, headerType, result);
    appendDescriptor(1, typeConverter.getTupleItemsMemRefType(tupleType),
                     result);
    class_container::Clone::markOwnership(loc, header, headerType,
                                          kTypedTupleRefcountSlot,
                                          initialRefcount, builder);
    return result;
  }
  if (auto dictType = mlir::dyn_cast<DictType>(logicalType)) {
    auto headerType = getDictHeaderMemRefType(builder.getContext());
    auto lockType = getContainerLockMemRefType(builder.getContext());
    auto keysType = typeConverter.getDictKeysMemRefType(dictType);
    auto valuesType = typeConverter.getDictValuesMemRefType(dictType);
    if (!keysType || !valuesType)
      return storageValue;
    mlir::Value header = appendDescriptor(0, headerType, result);
    appendDescriptor(1, lockType, result);
    appendDescriptor(2, keysType, result);
    appendDescriptor(3, valuesType, result);
    appendDescriptor(4, getDictStatesMemRefType(builder.getContext()), result);
    class_container::Clone::markOwnership(loc, header, headerType,
                                          kTypedDictRefcountSlot,
                                          initialRefcount, builder);
    return result;
  }
  return storageValue;
}

mlir::Value retainOrCloneLocalAlias(mlir::Location loc, mlir::Value descriptor,
                                    mlir::Type logicalType, mlir::Value owner,
                                    const StaticClassLayout &layout,
                                    unsigned fieldIndex, mlir::ModuleOp module,
                                    mlir::OpBuilder &builder,
                                    const PyLLVMTypeConverter &typeConverter,
                                    llvm::StringRef premise) {
  (void)module;
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return descriptor;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *retainBlock = builder.createBlock(parent);
  mlir::Block *cloneBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  afterBlock->addArgument(descriptor.getType(), loc);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value hasHeader =
      class_container::MemRef::present(loc, descriptor, logicalType, builder);
  if (!hasHeader)
    return descriptor;
  builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                         mlir::ValueRange{}, afterBlock,
                                         mlir::ValueRange{descriptor});

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value headerMemref =
      class_container::MemRef::header(loc, descriptor, logicalType, builder);
  if (!headerMemref)
    return descriptor;
  mlir::Value slot =
      class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, zero, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                         cloneBlock);

  builder.setInsertionPointToStart(retainBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, one, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic, premise);
  if (premise == ThreadSafetyAttrs::kPremiseOwnedToken)
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  builder.create<mlir::cf::BranchOp>(loc, afterBlock,
                                     mlir::ValueRange{descriptor});

  builder.setInsertionPointToStart(cloneBlock);
  mlir::Value cloned = class_container::Clone::storage(
      loc, descriptor, logicalType, builder, typeConverter,
      /*initialRefcount=*/2);
  mlir::Operation *store = class_object::Payload::storeField(
      loc, owner, layout, fieldIndex, cloned, builder);
  if (!store)
    return descriptor;
  class_field::Ownership::markStore(store, fieldIndex);
  builder.create<mlir::cf::BranchOp>(loc, afterBlock, mlir::ValueRange{cloned});

  builder.setInsertionPointToStart(afterBlock);
  return afterBlock->getArgument(0);
}

mlir::Value retainOrCloneOwned(mlir::Location loc, mlir::Value descriptor,
                               mlir::Type logicalType, mlir::ModuleOp module,
                               mlir::OpBuilder &builder,
                               const PyLLVMTypeConverter &typeConverter,
                               llvm::StringRef premise) {
  auto refcountSlot = class_container::Refcount::slot(logicalType);
  if (!refcountSlot)
    return descriptor;

  auto *parent = builder.getInsertionBlock()->getParent();
  auto i64Type = builder.getI64Type();
  mlir::Block *currentBlock = builder.getInsertionBlock();
  mlir::Block *nonNullBlock = builder.createBlock(parent);
  mlir::Block *retainBlock = builder.createBlock(parent);
  mlir::Block *cloneBlock = builder.createBlock(parent);
  mlir::Block *afterBlock = builder.createBlock(parent);
  afterBlock->addArgument(descriptor.getType(), loc);

  builder.setInsertionPointToEnd(currentBlock);
  mlir::Value header =
      class_container::Descriptor::header(loc, descriptor, builder);
  mlir::Value hasHeader =
      class_container::MemRef::present(loc, descriptor, logicalType, builder);
  if (!hasHeader)
    return descriptor;
  builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                         mlir::ValueRange{}, afterBlock,
                                         mlir::ValueRange{descriptor});

  builder.setInsertionPointToStart(nonNullBlock);
  mlir::Value headerMemref =
      class_container::MemRef::header(loc, descriptor, logicalType, builder);
  if (!headerMemref)
    return descriptor;
  mlir::Value slot =
      class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
  mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(0));
  mlir::Value refcount = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, zero, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::acquire, builder);
  threadsafe::Atomic::set(refcount.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                          ThreadSafetyAttrs::kOrderingAcquire);
  class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
  builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                         cloneBlock);

  builder.setInsertionPointToStart(retainBlock);
  mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
      loc, i64Type, builder.getI64IntegerAttr(1));
  mlir::Value previous = atomic_storage::Integer::rmw(
      loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
      headerMemref, one, mlir::ValueRange{slot},
      mlir::LLVM::AtomicOrdering::monotonic, builder);
  threadsafe::Atomic::set(previous.getDefiningOp(),
                          ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                          ThreadSafetyAttrs::kOrderingMonotonic, premise);
  if (premise == ThreadSafetyAttrs::kPremiseOwnedToken)
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
  class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                      refcountSlot, logicalType);
  builder.create<mlir::cf::BranchOp>(loc, afterBlock,
                                     mlir::ValueRange{descriptor});

  builder.setInsertionPointToStart(cloneBlock);
  mlir::Value cloned = class_container::Clone::storage(
      loc, descriptor, logicalType, builder, typeConverter,
      /*initialRefcount=*/1);
  if (mlir::failed(class_container::Elements::refcount(
          loc, cloned, logicalType, module, builder, typeConverter, "incref",
          /*markAggregate=*/true,
          /*transferRetainedToSlot=*/true)))
    return {};
  builder.create<mlir::cf::BranchOp>(loc, afterBlock, mlir::ValueRange{cloned});

  builder.setInsertionPointToStart(afterBlock);
  return afterBlock->getArgument(0);
}
} // namespace class_container::Clone

namespace class_container::Clone {
void fields(mlir::Location loc, mlir::Value object,
            const StaticClassLayout &layout, mlir::OpBuilder &builder,
            const PyLLVMTypeConverter &typeConverter) {
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    if (!mlir::isa<ListType, TupleType, DictType>(field.logicalType))
      continue;
    mlir::Value fieldValue = class_object::Payload::loadField(
        loc, object, layout, static_cast<unsigned>(index), builder);
    if (!fieldValue)
      continue;
    mlir::Value cloned = class_container::Clone::storage(
        loc, fieldValue, field.logicalType, builder, typeConverter);
    mlir::Operation *store = class_object::Payload::storeField(
        loc, object, layout, static_cast<unsigned>(index), cloned, builder);
    if (store)
      class_field::Ownership::markStore(store, static_cast<unsigned>(index));
  }
}
} // namespace class_container::Clone

namespace class_direct::Promote {
mlir::Value value(mlir::Location loc, mlir::Value object, ClassType classType,
                  mlir::ModuleOp module, mlir::OpBuilder &builder,
                  const PyLLVMTypeConverter &typeConverter);

void fields(mlir::Location loc, mlir::Value object,
            const StaticClassLayout &layout, mlir::ModuleOp module,
            mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter);
} // namespace class_direct::Promote

namespace class_helper::Promote {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<mlir::Type, 2> objectResultTypes =
      class_helper::Parts::types(layout);
  llvm::SmallVector<mlir::Type, 2> objectArgTypes =
      class_helper::Parts::types(layout);
  std::string helperName = getClassHelperName(classType, "promote");
  auto fn = getOrInsertClassFunc(loc, module, builder, helperName,
                                 objectArgTypes, objectResultTypes);
  ownership::effect::ownedResults(fn.getOperation(), {0});
  class_helper::Kind::mark(fn, classType, ClassSafetyAttrs::kKindPromote);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  auto retainHelper = class_helper::Retain::get(loc, module, classType, layout,
                                                builder, typeConverter);
  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value inputDescriptor =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!inputDescriptor)
    return fn;
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::BranchOp>(loc, dispatchBlock);

  builder.setInsertionPointToStart(dispatchBlock);
  mlir::Value managed =
      class_object::Managed::load(loc, inputDescriptor, layout, builder);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *copyBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, managed, managedBlock, copyBlock);

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value inputHeader =
      class_object::Local::header(loc, inputDescriptor, layout, builder);
  llvm::SmallVector<mlir::Value, 8> inputParts;
  for (int64_t index = 0; index < class_layout::Payload::partCount(layout);
       ++index) {
    mlir::Value part = class_object::Local::payloadPart(loc, inputDescriptor,
                                                        layout, index, builder);
    if (!part)
      return fn;
    inputParts.push_back(part);
  }
  if (!inputHeader)
    return fn;
  llvm::SmallVector<mlir::Value, 8> retainArgs{inputHeader};
  retainArgs.append(inputParts.begin(), inputParts.end());
  auto retain =
      builder.create<mlir::func::CallOp>(loc, retainHelper, retainArgs);
  threadsafe::Retain::premise(retain.getOperation(),
                              ThreadSafetyAttrs::kPremiseEntryBorrowed);
  builder.create<mlir::func::ReturnOp>(loc, retainArgs);

  builder.setInsertionPointToStart(copyBlock);
  mlir::Value header = raw_memref::allocRank1Descriptor(
      loc, module, builder.getI64Type(), object_abi::kHeaderSlots, builder,
      /*ownedRoot=*/true, /*objectHeader=*/true);
  llvm::SmallVector<mlir::Value, 8> payloadParts;
  for (mlir::MemRefType partType : class_layout::Payload::partTypes(layout)) {
    mlir::Value descriptor = raw_memref::allocRank1Descriptor(
        loc, module, partType.getElementType(), partType.getShape()[0], builder,
        /*ownedRoot=*/false);
    mlir::Value part = class_object::Local::memRefFromDescriptor(
        loc, descriptor, partType, builder);
    if (!part)
      return fn;
    payloadParts.push_back(part);
  }
  if (!class_object::Header::initialize(loc, header, classType, layout,
                                        /*refcount=*/1, builder))
    return fn;
  llvm::SmallVector<mlir::Value, 8> fieldValues;
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    (void)field;
    mlir::Value value = class_object::Payload::loadField(
        loc, inputDescriptor, layout, static_cast<unsigned>(index), builder);
    if (!value)
      return fn;
    fieldValues.push_back(value);
  }
  mlir::Operation *payloadStore = class_object::Payload::initialize(
      loc, payloadParts, fieldValues, layout, /*managed=*/true, builder);
  if (!payloadStore)
    return fn;
  mlir::Value initializedObject = class_object::Local::objectValueFromParts(
      loc, header, payloadParts, layout, builder);
  if (!initializedObject)
    return fn;
  class_container::Clone::fields(loc, initializedObject, layout, builder,
                                 typeConverter);
  class_direct::Promote::fields(loc, initializedObject, layout, module, builder,
                                typeConverter);
  if (mlir::failed(class_field::Runtime::forEach(
          loc, initializedObject, layout, module, builder, typeConverter,
          RuntimeSymbols::kIncRef,
          /*includeContainerFields=*/true,
          /*includeDirectClassFields=*/false)))
    return {};
  llvm::SmallVector<mlir::Value, 8> resultParts{header};
  resultParts.append(payloadParts.begin(), payloadParts.end());
  builder.create<mlir::func::ReturnOp>(loc, resultParts);
  return fn;
}
} // namespace class_helper::Promote

namespace class_direct::Promote {
mlir::Value value(mlir::Location loc, mlir::Value object, ClassType classType,
                  mlir::ModuleOp module, mlir::OpBuilder &builder,
                  const PyLLVMTypeConverter &typeConverter) {
  mlir::FailureOr<StaticClassLayout> layout =
      class_layout::get(module.getOperation(), classType, typeConverter);
  if (mlir::failed(layout))
    return {};
  auto helper = class_helper::Promote::get(loc, module, classType, *layout,
                                           builder, typeConverter);
  if (!helper)
    return {};
  llvm::SmallVector<mlir::Value, 8> args;
  class_helper::Parts::append(loc, object, *layout, builder, args);
  if (args.size() !=
      static_cast<size_t>(class_layout::Object::partCount(layout->objectType)))
    return {};
  auto call =
      builder.create<mlir::func::CallOp>(loc, helper, mlir::ValueRange{args});
  return class_helper::Parts::object(loc, call.getResults(), *layout, builder);
}

void fields(mlir::Location loc, mlir::Value object,
            const StaticClassLayout &layout, mlir::ModuleOp module,
            mlir::OpBuilder &builder,
            const PyLLVMTypeConverter &typeConverter) {
  for (auto [index, field] : llvm::enumerate(layout.fields)) {
    auto directClass = mlir::dyn_cast<ClassType>(field.logicalType);
    if (!directClass)
      continue;
    mlir::Value fieldValue = class_object::Payload::loadField(
        loc, object, layout, static_cast<unsigned>(index), builder);
    if (!fieldValue)
      continue;
    class_field::Ownership::markLoad(fieldValue, static_cast<unsigned>(index));
    mlir::Value promoted =
        value(loc, fieldValue, directClass, module, builder, typeConverter);
    if (!promoted)
      continue;
    mlir::Operation *store = class_object::Payload::storeField(
        loc, object, layout, static_cast<unsigned>(index), promoted, builder);
    if (store)
      class_field::Ownership::markStore(store, static_cast<unsigned>(index));
  }
}
} // namespace class_direct::Promote

namespace class_helper::GetField {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       unsigned fieldIndex, mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto i1Type = builder.getI1Type();
  llvm::SmallVector<mlir::Type, 3> inputTypes =
      class_helper::Parts::types(layout);
  unsigned objectArgCount =
      static_cast<unsigned>(class_layout::Object::partCount(layout.objectType));
  inputTypes.push_back(i1Type);
  llvm::SmallVector<mlir::Type, 4> resultTypes = class_field::ABI::types(
      module.getOperation(), module, fieldInfo, typeConverter);
  if (resultTypes.empty())
    return {};
  std::string helperName =
      class_helper::Field::name(classType, "getfield", fieldIndex);
  auto fn = getOrInsertClassFunc(loc, module, builder, helperName, inputTypes,
                                 resultTypes);
  class_helper::Field::mark(fn, classType, ClassSafetyAttrs::kKindGetField,
                            fieldIndex);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn->setAttr(OwnershipContractAttrs::kGetFieldBorrowArg,
              builder.getI64IntegerAttr(objectArgCount));
  fn->setAttr(OwnershipContractAttrs::kGetFieldOwnedResult,
              builder.getI64IntegerAttr(0));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  bool needsRefcount = class_field::Refcount::needed(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
  ClassType directClass = mlir::dyn_cast<ClassType>(fieldInfo.logicalType);
  bool needsClassRefcount = static_cast<bool>(directClass);
  bool needsContainerRefcount =
      mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType);
  if (directClass) {
    mlir::FailureOr<StaticClassLayout> directLayout =
        class_layout::get(module.getOperation(), directClass, typeConverter);
    if (mlir::succeeded(directLayout))
      class_helper::Retain::get(loc, module, directClass, *directLayout,
                                builder, typeConverter);
  }

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!object)
    return fn;
  mlir::Value borrowLocal = entry->getArgument(objectArgCount);
  mlir::Value managed =
      class_object::Managed::load(loc, object, layout, builder);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(entry);
  builder.create<mlir::cf::CondBranchOp>(loc, managed, managedBlock,
                                         localBlock);

  builder.setInsertionPointToStart(localBlock);
  mlir::Value localValue = class_object::Payload::loadField(
      loc, object, layout, fieldIndex, builder);
  if (!localValue)
    return fn;
  if (needsRefcount || needsClassRefcount || needsContainerRefcount)
    class_field::Ownership::markLoad(localValue, fieldIndex);
  auto returnFieldValues = [&](mlir::Value value) -> bool {
    mlir::FailureOr<llvm::SmallVector<mlir::Value>> values =
        class_field::ABI::parts(loc, value, fieldInfo, module, builder,
                                typeConverter);
    if (mlir::failed(values))
      return false;
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange(*values));
    return true;
  };
  auto retainFreshContainerForReturn = [&](mlir::Value value) -> bool {
    auto refcountSlot = class_container::Refcount::slot(fieldInfo.logicalType);
    if (!refcountSlot)
      return false;
    mlir::Value header =
        class_container::Descriptor::header(loc, value, builder);
    mlir::Value headerMemref = class_container::MemRef::header(
        loc, value, fieldInfo.logicalType, builder);
    if (!headerMemref)
      return false;
    mlir::Value slot =
        class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
    mlir::Value previous = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, one, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::monotonic, builder);
    threadsafe::Atomic::set(previous.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                            ThreadSafetyAttrs::kOrderingMonotonic,
                            ThreadSafetyAttrs::kPremiseOwnedToken);
    threadsafe::Retain::verifyOwnedToken(
        previous.getDefiningOp(), ThreadSafetyAttrs::kProofClassFieldHelper);
    class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    return true;
  };
  auto returnLocalContainerAlias = [&](mlir::Value value) -> bool {
    auto refcountSlot = class_container::Refcount::slot(fieldInfo.logicalType);
    if (!refcountSlot)
      return false;

    auto *parent = builder.getInsertionBlock()->getParent();
    mlir::Block *currentBlock = builder.getInsertionBlock();
    mlir::Block *nonNullBlock = builder.createBlock(parent);
    mlir::Block *retainBlock = builder.createBlock(parent);
    mlir::Block *cloneBlock = builder.createBlock(parent);
    mlir::Block *nullBlock = builder.createBlock(parent);

    builder.setInsertionPointToEnd(currentBlock);
    mlir::Value header =
        class_container::Descriptor::header(loc, value, builder);
    mlir::Value hasHeader = class_container::MemRef::present(
        loc, value, fieldInfo.logicalType, builder);
    if (!hasHeader)
      return false;
    builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                           nullBlock);

    builder.setInsertionPointToStart(nullBlock);
    builder.create<mlir::LLVM::UnreachableOp>(loc);

    builder.setInsertionPointToStart(nonNullBlock);
    mlir::Value headerMemref = class_container::MemRef::header(
        loc, value, fieldInfo.logicalType, builder);
    if (!headerMemref)
      return false;
    mlir::Value slot =
        class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    mlir::Value refcount = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, zero, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::acquire, builder);
    threadsafe::Atomic::set(refcount.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
    class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
    builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                           cloneBlock);

    builder.setInsertionPointToStart(retainBlock);
    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
    mlir::Value previous = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, one, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::monotonic, builder);
    threadsafe::Atomic::set(previous.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                            ThreadSafetyAttrs::kOrderingMonotonic,
                            ThreadSafetyAttrs::kPremiseAggregateBorrow);
    class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    if (!returnFieldValues(value))
      return false;

    builder.setInsertionPointToStart(cloneBlock);
    mlir::Value cloned = class_container::Clone::storage(
        loc, value, fieldInfo.logicalType, builder, typeConverter,
        /*initialRefcount=*/1);
    if (!retainFreshContainerForReturn(cloned))
      return false;
    mlir::Operation *store = class_object::Payload::storeField(
        loc, object, layout, fieldIndex, cloned, builder);
    if (!store)
      return false;
    class_field::Ownership::markStore(store, fieldIndex);
    if (!returnFieldValues(cloned))
      return false;
    return true;
  };
  if (!needsRefcount && !needsClassRefcount && !needsContainerRefcount) {
    if (!returnFieldValues(localValue))
      return fn;
  } else {
    mlir::Block *localBorrowBlock = builder.createBlock(&fn.getBody());
    mlir::Block *localRetainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<mlir::cf::CondBranchOp>(loc, borrowLocal, localBorrowBlock,
                                           localRetainBlock);

    builder.setInsertionPointToStart(localBorrowBlock);
    {
      mlir::FailureOr<llvm::SmallVector<mlir::Value>> values =
          class_field::ABI::parts(loc, localValue, fieldInfo, module, builder,
                                  typeConverter);
      if (mlir::failed(values))
        return fn;
      builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange(*values));
    }

    builder.setInsertionPointToStart(localRetainBlock);
    if (needsRefcount) {
      if (mlir::failed(class_field::Runtime::single(
              loc, localValue, fieldInfo.logicalType, module, builder,
              typeConverter, RuntimeSymbols::kIncRef)))
        return {};
    } else if (needsClassRefcount) {
      Slot::classRefcount(loc, localValue, directClass, module, builder,
                          "incref", /*aggregateEffect=*/true,
                          ThreadSafetyAttrs::kPremiseAggregateBorrow);
    } else {
      if (!returnLocalContainerAlias(localValue))
        return fn;
      localValue = {};
    }
    if (localValue && !returnFieldValues(localValue))
      return fn;
  }

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value lock = class_object::Lock::storage(loc, object, layout, builder);
  if (!lock)
    return fn;
  class_object::Lock::acquire(loc, lock, module, builder);
  mlir::Value managedValue = class_object::Payload::loadField(
      loc, object, layout, fieldIndex, builder);
  if (!managedValue)
    return fn;
  if (needsRefcount || needsClassRefcount || needsContainerRefcount)
    class_field::Ownership::markLoad(managedValue, fieldIndex);
  if (needsContainerRefcount) {
    auto refcountSlot = class_container::Refcount::slot(fieldInfo.logicalType);
    if (!refcountSlot)
      return fn;

    auto *parent = builder.getInsertionBlock()->getParent();
    mlir::Block *currentBlock = builder.getInsertionBlock();
    mlir::Block *nonNullBlock = builder.createBlock(parent);
    mlir::Block *retainBlock = builder.createBlock(parent);
    mlir::Block *invalidBlock = builder.createBlock(parent);

    builder.setInsertionPointToEnd(currentBlock);
    mlir::Value header =
        class_container::Descriptor::header(loc, managedValue, builder);
    mlir::Value hasHeader = class_container::MemRef::present(
        loc, managedValue, fieldInfo.logicalType, builder);
    if (!hasHeader)
      return fn;
    builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                           invalidBlock);

    builder.setInsertionPointToStart(invalidBlock);
    builder.create<mlir::LLVM::UnreachableOp>(loc);

    builder.setInsertionPointToStart(nonNullBlock);
    mlir::Value headerMemref = class_container::MemRef::header(
        loc, managedValue, fieldInfo.logicalType, builder);
    if (!headerMemref)
      return fn;
    mlir::Value slot =
        class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(0));
    mlir::Value refcount = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, zero, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::acquire, builder);
    threadsafe::Atomic::set(refcount.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
    class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
    builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                           invalidBlock);

    builder.setInsertionPointToStart(retainBlock);
    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
    mlir::Value previous = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, one, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::monotonic, builder);
    threadsafe::Atomic::set(previous.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                            ThreadSafetyAttrs::kOrderingMonotonic,
                            ThreadSafetyAttrs::kPremiseLockedBorrow);
    class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    class_object::Lock::release(loc, lock, module, builder);
    if (!returnFieldValues(managedValue))
      return fn;
    return fn;
  }
  if (needsRefcount) {
    if (mlir::failed(class_field::Runtime::single(
            loc, managedValue, fieldInfo.logicalType, module, builder,
            typeConverter, RuntimeSymbols::kIncRef,
            ThreadSafetyAttrs::kPremiseLockedBorrow)))
      return {};
  } else if (needsClassRefcount) {
    Slot::classRefcount(loc, managedValue, directClass, module, builder,
                        "incref", /*aggregateEffect=*/true,
                        ThreadSafetyAttrs::kPremiseLockedBorrow);
  } else if (needsContainerRefcount) {
    class_container::Field::retainIfManaged(
        loc, managedValue, fieldInfo.logicalType, module, builder,
        ThreadSafetyAttrs::kPremiseLockedBorrow);
  }
  class_object::Lock::release(loc, lock, module, builder);
  mlir::FailureOr<llvm::SmallVector<mlir::Value>> values =
      class_field::ABI::parts(loc, managedValue, fieldInfo, module, builder,
                              typeConverter);
  if (mlir::failed(values))
    return fn;
  builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange(*values));
  return fn;
}
} // namespace class_helper::GetField

namespace class_helper::SetField {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       unsigned fieldIndex, mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  assert(fieldIndex < layout.fields.size() && "field index out of range");
  const StaticClassFieldInfo &fieldInfo = layout.fields[fieldIndex];
  auto i1Type = builder.getI1Type();
  llvm::SmallVector<mlir::Type, 5> inputTypes =
      class_helper::Parts::types(layout);
  llvm::SmallVector<mlir::Type, 4> valueTypes = class_field::ABI::types(
      module.getOperation(), module, fieldInfo, typeConverter);
  if (valueTypes.empty())
    return {};
  inputTypes.append(valueTypes.begin(), valueTypes.end());
  inputTypes.push_back(i1Type);
  inputTypes.push_back(i1Type);
  unsigned objectArgCount =
      static_cast<unsigned>(class_layout::Object::partCount(layout.objectType));
  std::string helperName =
      class_helper::Field::name(classType, "setfield", fieldIndex);
  auto fn = getOrInsertClassFunc(loc, module, builder, helperName, inputTypes);
  bool needsRefcount = class_field::Refcount::needed(
      fieldInfo.logicalType, fieldInfo.storageType, typeConverter);
  ClassType directClass = mlir::dyn_cast<ClassType>(fieldInfo.logicalType);
  bool needsClassRefcount = static_cast<bool>(directClass);
  bool needsContainerRefcount =
      mlir::isa<ListType, TupleType, DictType>(fieldInfo.logicalType);
  class_helper::Field::mark(fn, classType, ClassSafetyAttrs::kKindSetField,
                            fieldIndex);
  class_helper::Schema::mark(fn, layout, typeConverter);
  fn->setAttr(OwnershipContractAttrs::kSetFieldValueArg,
              builder.getI64IntegerAttr(objectArgCount));
  fn->setAttr(OwnershipContractAttrs::kSetFieldRetainArg,
              builder.getI64IntegerAttr(
                  objectArgCount + static_cast<int64_t>(valueTypes.size())));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  if (directClass) {
    mlir::FailureOr<StaticClassLayout> directLayout =
        class_layout::get(module.getOperation(), directClass, typeConverter);
    if (mlir::succeeded(directLayout)) {
      class_helper::Retain::get(loc, module, directClass, *directLayout,
                                builder, typeConverter);
      class_helper::Release::get(loc, module, directClass, *directLayout,
                                 builder, typeConverter);
    }
  }

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value object =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  if (!object)
    return fn;
  llvm::SmallVector<mlir::Value, 4> valueArgs;
  valueArgs.reserve(valueTypes.size());
  for (unsigned index = 0; index < valueTypes.size(); ++index)
    valueArgs.push_back(entry->getArgument(objectArgCount + index));
  mlir::FailureOr<mlir::Value> newValueOr =
      class_field::ABI::storage(loc, mlir::ValueRange(valueArgs), fieldInfo,
                                module, builder, typeConverter);
  if (mlir::failed(newValueOr))
    return fn;
  mlir::Value newValue = *newValueOr;
  unsigned retainArgIndex =
      objectArgCount + static_cast<unsigned>(valueTypes.size());
  mlir::Value retainNewValue = entry->getArgument(retainArgIndex);
  mlir::Value skipOldValueDrop = entry->getArgument(retainArgIndex + 1);
  mlir::Block *dispatchBlock = builder.createBlock(&fn.getBody());
  dispatchBlock->addArgument(fieldInfo.storageType, loc);
  if (needsRefcount) {
    mlir::Block *retainBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::cf::CondBranchOp>(loc, retainNewValue, retainBlock,
                                           mlir::ValueRange{}, dispatchBlock,
                                           mlir::ValueRange{newValue});
    builder.setInsertionPointToStart(retainBlock);
    mlir::Value retainedValue = newValue;
    if (needsRefcount) {
      if (mlir::failed(class_field::Runtime::single(
              loc, newValue, fieldInfo.logicalType, module, builder,
              typeConverter, RuntimeSymbols::kIncRef,
              ThreadSafetyAttrs::kPremiseEntryBorrowed)))
        return {};
    }
    builder.create<mlir::cf::BranchOp>(loc, dispatchBlock,
                                       mlir::ValueRange{retainedValue});
  } else {
    builder.setInsertionPointToEnd(entry);
    builder.create<mlir::cf::BranchOp>(loc, dispatchBlock,
                                       mlir::ValueRange{newValue});
  }

  builder.setInsertionPointToStart(dispatchBlock);
  newValue = dispatchBlock->getArgument(0);
  auto retainDirectForLocalStore = [&](mlir::Value value) -> mlir::Value {
    if (!needsClassRefcount)
      return value;
    mlir::FailureOr<StaticClassLayout> directLayout =
        class_layout::get(module.getOperation(), directClass, typeConverter);
    if (mlir::failed(directLayout))
      return value;
    auto retainHelper = class_helper::Retain::get(
        loc, module, directClass, *directLayout, builder, typeConverter);
    llvm::SmallVector<mlir::Value, 8> args;
    class_helper::Parts::append(loc, value, *directLayout, builder, args);
    if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                           directLayout->objectType)))
      return value;
    auto retain = builder.create<mlir::func::CallOp>(loc, retainHelper, args);
    ownership::aggregate::Slot::markLoad(value);
    retain->setAttr(OwnershipContractAttrs::kAggregateRetain,
                    builder.getUnitAttr());
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseEntryBorrowed);
    return value;
  };
  auto promoteDirectForManagedStore = [&](mlir::Value value) -> mlir::Value {
    if (!needsClassRefcount)
      return value;
    return class_direct::Promote::value(loc, value, directClass, module,
                                        builder, typeConverter);
  };
  auto destroyOldContainer = [&](mlir::Value oldValue) -> bool {
    if (!oldValue)
      return true;
    return mlir::succeeded(class_container::Field::destroy(
        loc, oldValue, fieldInfo.logicalType, module, builder, typeConverter));
  };
  auto storeContainerAndReturn = [&](mlir::Value storedValue,
                                     bool transferOwnership,
                                     mlir::Value lockValue, bool alreadyLocked,
                                     bool loadOldValue) -> bool {
    mlir::Value oldValue;
    if (lockValue && !alreadyLocked)
      class_object::Lock::acquire(loc, lockValue, module, builder);
    if (loadOldValue) {
      oldValue = class_object::Payload::loadField(loc, object, layout,
                                                  fieldIndex, builder);
      if (!oldValue)
        return false;
      class_field::Ownership::markLoad(oldValue, fieldIndex);
    }
    mlir::Operation *storeOp = class_object::Payload::storeField(
        loc, object, layout, fieldIndex, storedValue, builder);
    if (!storeOp)
      return false;
    if (transferOwnership)
      class_field::Ownership::markStore(storeOp, fieldIndex);
    if (lockValue)
      class_object::Lock::release(loc, lockValue, module, builder);
    if (!destroyOldContainer(oldValue))
      return false;
    builder.create<mlir::func::ReturnOp>(loc);
    return true;
  };
  auto storeContainerForField = [&](mlir::Value value, mlir::Value lockValue,
                                    bool alreadyLocked,
                                    bool loadOldValue) -> bool {
    if (!needsContainerRefcount)
      return false;
    auto refcountSlot = class_container::Refcount::slot(fieldInfo.logicalType);
    if (!refcountSlot)
      return false;

    auto *parent = builder.getInsertionBlock()->getParent();
    auto i64Type = builder.getI64Type();
    mlir::Block *currentBlock = builder.getInsertionBlock();
    mlir::Block *nonNullBlock = builder.createBlock(parent);
    mlir::Block *retainBlock = builder.createBlock(parent);
    mlir::Block *cloneBlock = builder.createBlock(parent);
    mlir::Block *nullBlock = builder.createBlock(parent);

    builder.setInsertionPointToEnd(currentBlock);
    mlir::Value header =
        class_container::Descriptor::header(loc, value, builder);
    mlir::Value hasHeader = class_container::MemRef::present(
        loc, value, fieldInfo.logicalType, builder);
    if (!hasHeader)
      return false;
    builder.create<mlir::cf::CondBranchOp>(loc, hasHeader, nonNullBlock,
                                           nullBlock);

    builder.setInsertionPointToStart(nullBlock);
    // The hasHeader=false branch carries no container ownership token, but the
    // store still crosses the class-field slot boundary. Mark the same slot
    // contract so the verifier can reason from the proven entry argument
    // instead of the physical aggregate shape.
    if (!storeContainerAndReturn(value, /*transferOwnership=*/true, lockValue,
                                 alreadyLocked, loadOldValue))
      return false;

    builder.setInsertionPointToStart(nonNullBlock);
    mlir::Value headerMemref = class_container::MemRef::header(
        loc, value, fieldInfo.logicalType, builder);
    if (!headerMemref)
      return false;
    mlir::Value slot =
        class_container::MemRef::slotIndex(loc, *refcountSlot, builder);
    mlir::Value zero = builder.create<mlir::LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(0));
    mlir::Value refcount = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, zero, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::acquire, builder);
    threadsafe::Atomic::set(refcount.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountLoad,
                            ThreadSafetyAttrs::kOrderingAcquire);
    class_container::Atomic::markHeader(refcount.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    mlir::Value isManaged = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::ne, refcount, zero);
    builder.create<mlir::cf::CondBranchOp>(loc, isManaged, retainBlock,
                                           cloneBlock);

    builder.setInsertionPointToStart(retainBlock);
    mlir::Value one = builder.create<mlir::LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(1));
    mlir::Value previous = atomic_storage::Integer::rmw(
        loc, mlir::LLVM::AtomicBinOp::add, mlir::arith::AtomicRMWKind::addi,
        headerMemref, one, mlir::ValueRange{slot},
        mlir::LLVM::AtomicOrdering::monotonic, builder);
    threadsafe::Atomic::set(previous.getDefiningOp(),
                            ThreadSafetyAttrs::kRoleContainerRefcountRetain,
                            ThreadSafetyAttrs::kOrderingMonotonic,
                            ThreadSafetyAttrs::kPremiseEntryBorrowed);
    class_container::Atomic::markHeader(previous.getDefiningOp(), header,
                                        refcountSlot, fieldInfo.logicalType);
    if (!storeContainerAndReturn(value, /*transferOwnership=*/true, lockValue,
                                 alreadyLocked, loadOldValue))
      return false;

    builder.setInsertionPointToStart(cloneBlock);
    mlir::Value cloned = class_container::Clone::storage(
        loc, value, fieldInfo.logicalType, builder, typeConverter,
        /*initialRefcount=*/1);
    if (mlir::failed(class_container::Elements::refcount(
            loc, cloned, fieldInfo.logicalType, module, builder, typeConverter,
            "incref",
            /*markAggregate=*/true,
            /*transferRetainedToSlot=*/true)))
      return false;
    if (!storeContainerAndReturn(cloned, /*transferOwnership=*/true, lockValue,
                                 alreadyLocked, loadOldValue))
      return false;
    return true;
  };
  mlir::Value managed =
      class_object::Managed::load(loc, object, layout, builder);
  mlir::Block *managedBlock = builder.createBlock(&fn.getBody());
  mlir::Block *localBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(dispatchBlock);
  builder.create<mlir::cf::CondBranchOp>(loc, managed, managedBlock,
                                         localBlock);

  builder.setInsertionPointToStart(localBlock);
  if (!needsRefcount && !needsClassRefcount && !needsContainerRefcount) {
    if (!class_object::Payload::storeField(loc, object, layout, fieldIndex,
                                           newValue, builder))
      return fn;
    builder.create<mlir::func::ReturnOp>(loc);
  } else {
    mlir::Block *localSkipOldBlock = builder.createBlock(&fn.getBody());
    mlir::Block *localLoadOldBlock = builder.createBlock(&fn.getBody());
    builder.setInsertionPointToEnd(localBlock);
    builder.create<mlir::cf::CondBranchOp>(
        loc, skipOldValueDrop, localSkipOldBlock, localLoadOldBlock);

    builder.setInsertionPointToStart(localSkipOldBlock);
    if (needsContainerRefcount) {
      if (!storeContainerForField(newValue, /*lockValue=*/{},
                                  /*alreadyLocked=*/false, false))
        return fn;
    } else {
      mlir::Value storedValue = newValue;
      storedValue = retainDirectForLocalStore(storedValue);
      mlir::Operation *storeOp = class_object::Payload::storeField(
          loc, object, layout, fieldIndex, storedValue, builder);
      if (!storeOp)
        return fn;
      class_field::Ownership::markStore(storeOp, fieldIndex);
      builder.create<mlir::func::ReturnOp>(loc);
    }

    builder.setInsertionPointToStart(localLoadOldBlock);
    if (needsContainerRefcount) {
      if (!storeContainerForField(newValue, /*lockValue=*/{},
                                  /*alreadyLocked=*/false, true))
        return fn;
    } else {
      mlir::Value storedValue = newValue;
      storedValue = retainDirectForLocalStore(storedValue);
      mlir::Value oldValue = class_object::Payload::loadField(
          loc, object, layout, fieldIndex, builder);
      if (!oldValue)
        return fn;
      class_field::Ownership::markLoad(oldValue, fieldIndex);
      mlir::Operation *storeOp = class_object::Payload::storeField(
          loc, object, layout, fieldIndex, storedValue, builder);
      if (!storeOp)
        return fn;
      class_field::Ownership::markStore(storeOp, fieldIndex);
      if (needsRefcount) {
        if (mlir::failed(class_field::Runtime::single(
                loc, oldValue, fieldInfo.logicalType, module, builder,
                typeConverter, kDecrefIntent)))
          return {};
      } else if (needsClassRefcount) {
        Slot::classRefcount(loc, oldValue, directClass, module, builder,
                            "decref",
                            /*aggregateEffect=*/true,
                            ThreadSafetyAttrs::kPremiseAggregateBorrow);
      } else {
        if (mlir::failed(class_container::Field::destroy(
                loc, oldValue, fieldInfo.logicalType, module, builder,
                typeConverter)))
          return {};
      }
      builder.create<mlir::func::ReturnOp>(loc);
    }
  }

  builder.setInsertionPointToStart(managedBlock);
  mlir::Value lock = class_object::Lock::storage(loc, object, layout, builder);
  if (!lock)
    return fn;
  if (!needsRefcount && !needsClassRefcount && !needsContainerRefcount) {
    class_object::Lock::acquire(loc, lock, module, builder);
    if (!class_object::Payload::storeField(loc, object, layout, fieldIndex,
                                           newValue, builder))
      return fn;
    class_object::Lock::release(loc, lock, module, builder);
    builder.create<mlir::func::ReturnOp>(loc);
    return fn;
  }

  mlir::Block *managedSkipOldBlock = builder.createBlock(&fn.getBody());
  mlir::Block *managedLoadOldBlock = builder.createBlock(&fn.getBody());
  builder.setInsertionPointToEnd(managedBlock);
  builder.create<mlir::cf::CondBranchOp>(
      loc, skipOldValueDrop, managedSkipOldBlock, managedLoadOldBlock);

  builder.setInsertionPointToStart(managedSkipOldBlock);
  if (needsContainerRefcount) {
    class_object::Lock::acquire(loc, lock, module, builder);
    if (!storeContainerForField(newValue, lock, /*alreadyLocked=*/true, false))
      return fn;
  } else {
    mlir::Value storedValue = newValue;
    storedValue = promoteDirectForManagedStore(storedValue);
    if (!storedValue)
      return fn;
    class_object::Lock::acquire(loc, lock, module, builder);
    mlir::Operation *storeOp = class_object::Payload::storeField(
        loc, object, layout, fieldIndex, storedValue, builder);
    if (!storeOp)
      return fn;
    class_field::Ownership::markStore(storeOp, fieldIndex);
    class_object::Lock::release(loc, lock, module, builder);
    builder.create<mlir::func::ReturnOp>(loc);
  }

  builder.setInsertionPointToStart(managedLoadOldBlock);
  if (needsContainerRefcount) {
    class_object::Lock::acquire(loc, lock, module, builder);
    if (!storeContainerForField(newValue, lock, /*alreadyLocked=*/true, true))
      return fn;
  } else {
    mlir::Value storedValue = newValue;
    storedValue = promoteDirectForManagedStore(storedValue);
    if (!storedValue)
      return fn;
    class_object::Lock::acquire(loc, lock, module, builder);
    mlir::Value oldValue = class_object::Payload::loadField(
        loc, object, layout, fieldIndex, builder);
    if (!oldValue)
      return fn;
    class_field::Ownership::markLoad(oldValue, fieldIndex);
    mlir::Operation *storeOp = class_object::Payload::storeField(
        loc, object, layout, fieldIndex, storedValue, builder);
    if (!storeOp)
      return fn;
    class_field::Ownership::markStore(storeOp, fieldIndex);
    class_object::Lock::release(loc, lock, module, builder);
    if (needsRefcount) {
      if (mlir::failed(class_field::Runtime::single(
              loc, oldValue, fieldInfo.logicalType, module, builder,
              typeConverter, kDecrefIntent)))
        return {};
    } else if (needsClassRefcount) {
      Slot::classRefcount(loc, oldValue, directClass, module, builder, "decref",
                          /*aggregateEffect=*/true,
                          ThreadSafetyAttrs::kPremiseAggregateBorrow);
    } else {
      if (mlir::failed(class_container::Field::destroy(
              loc, oldValue, fieldInfo.logicalType, module, builder,
              typeConverter)))
        return {};
    }
    builder.create<mlir::func::ReturnOp>(loc);
  }
  return fn;
}
} // namespace class_helper::SetField

namespace class_helper::Copy {
mlir::func::FuncOp get(mlir::Location loc, mlir::ModuleOp module,
                       ClassType classType, const StaticClassLayout &layout,
                       mlir::OpBuilder &builder,
                       const PyLLVMTypeConverter &typeConverter) {
  std::string helperName = getClassHelperName(classType, "copy");
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  llvm::SmallVector<mlir::Type, 2> objectArgTypes =
      class_helper::Parts::types(layout);
  inputTypes.append(objectArgTypes.begin(), objectArgTypes.end());
  inputTypes.append(objectArgTypes.begin(), objectArgTypes.end());
  auto fn = getOrInsertClassFunc(loc, module, builder, helperName, inputTypes);
  class_helper::Schema::mark(fn, layout, typeConverter);
  class_helper::Schema::markObjectArg(
      fn, layout, static_cast<unsigned>(objectArgTypes.size()));
  fn.setVisibility(mlir::SymbolTable::Visibility::Private);
  if (!fn.getBody().empty())
    return fn;

  mlir::Block *entry = fn.addEntryBlock();
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(entry);

  mlir::Value dest =
      class_helper::Parts::objectFromArgs(loc, entry, 0, layout, builder);
  unsigned srcFirstArg = static_cast<unsigned>(objectArgTypes.size());
  mlir::Value src = class_helper::Parts::objectFromArgs(loc, entry, srcFirstArg,
                                                        layout, builder);
  if (!dest || !src)
    return fn;
  mlir::Value borrowLocal = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));
  mlir::Value retainNewValue = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));
  mlir::Value skipOldDrop = builder.create<mlir::LLVM::ConstantOp>(
      loc, builder.getI1Type(), builder.getBoolAttr(true));

  for (auto [fieldIndex, fieldInfo] : llvm::enumerate(layout.fields)) {
    auto getHelper = class_helper::GetField::get(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    llvm::SmallVector<mlir::Value, 3> getArgs;
    class_helper::Parts::append(loc, src, layout, builder, getArgs);
    getArgs.push_back(borrowLocal);
    auto fieldValue = builder.create<mlir::func::CallOp>(
        loc, getHelper, mlir::ValueRange{getArgs});

    auto setHelper = class_helper::SetField::get(
        loc, module, classType, layout, fieldIndex, builder, typeConverter);
    llvm::SmallVector<mlir::Value, 8> setArgs;
    class_helper::Parts::append(loc, dest, layout, builder, setArgs);
    setArgs.append(fieldValue.getResults().begin(),
                   fieldValue.getResults().end());
    setArgs.push_back(retainNewValue);
    setArgs.push_back(skipOldDrop);
    builder.create<mlir::func::CallOp>(loc, setHelper,
                                       mlir::ValueRange{setArgs});
  }

  builder.create<mlir::func::ReturnOp>(loc);
  return fn;
}
} // namespace class_helper::Copy

static void
ensureStaticClassHelperBodies(ClassOp classOp, const StaticClassLayout &layout,
                              mlir::ModuleOp module, mlir::OpBuilder &builder,
                              const PyLLVMTypeConverter &typeConverter) {
  ClassType classType =
      ClassType::get(classOp.getContext(), classOp.getSymNameAttr().getValue());
  class_helper::Retain::get(classOp.getLoc(), module, classType, layout,
                            builder, typeConverter);
  class_helper::Release::get(classOp.getLoc(), module, classType, layout,
                             builder, typeConverter);
  class_helper::LocalDestroy::get(classOp.getLoc(), module, classType, layout,
                                  builder, typeConverter);
  class_helper::Promote::get(classOp.getLoc(), module, classType, layout,
                             builder, typeConverter);
  class_helper::Repr::ensure(classOp.getLoc(), module, classType, builder,
                             typeConverter);
  class_helper::Eq::get(classOp.getLoc(), module, classType, layout, builder,
                        typeConverter);
  class_helper::Copy::get(classOp.getLoc(), module, classType, layout, builder,
                          typeConverter);
}

static mlir::FailureOr<mlir::Value> boxStaticFieldValue(
    mlir::Location loc, mlir::Value storageValue, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool borrowExistingRef = false) {
  RuntimeAPI runtime(module, rewriter, typeConverter);
  mlir::Type convertedLogicalType = typeConverter.convertType(logicalType);
  if (mlir::isa<FloatType>(logicalType) &&
      mlir::isa<mlir::FloatType>(storageValue.getType())) {
    if (storageValue.getType() == convertedLogicalType)
      return storageValue;
    if (auto targetFloat =
            mlir::dyn_cast_or_null<mlir::FloatType>(convertedLogicalType)) {
      auto sourceFloat = mlir::cast<mlir::FloatType>(storageValue.getType());
      if (sourceFloat.getWidth() < targetFloat.getWidth())
        return rewriter
            .create<mlir::arith::ExtFOp>(loc, convertedLogicalType,
                                         storageValue)
            .getResult();
      if (sourceFloat.getWidth() > targetFloat.getWidth())
        return rewriter
            .create<mlir::arith::TruncFOp>(loc, convertedLogicalType,
                                           storageValue)
            .getResult();
    }
    return mlir::failure();
  }
  if (mlir::isa<BoolType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageValue.getType())) {
    if (storageValue.getType() == convertedLogicalType)
      return storageValue;
    if (convertedLogicalType && convertedLogicalType.isInteger(1)) {
      auto intType = mlir::cast<mlir::IntegerType>(storageValue.getType());
      mlir::Value zero =
          rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, intType);
      return rewriter
          .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                       storageValue, zero)
          .getResult();
    }
    return mlir::failure();
  }
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    if (!borrowExistingRef)
      Slot::classRefcount(loc, storageValue, classType, module, rewriter,
                          "incref", /*aggregateEffect=*/true,
                          ThreadSafetyAttrs::kPremiseAggregateBorrow);
    if (convertedLogicalType && storageValue.getType() != convertedLogicalType)
      return rewriter
          .create<mlir::UnrealizedConversionCastOp>(loc, convertedLogicalType,
                                                    storageValue)
          .getResult(0);
    return storageValue;
  }
  if (class_field::Refcount::needed(logicalType, storageValue.getType(),
                                    typeConverter)) {
    if (!borrowExistingRef) {
      auto retain =
          runtime.call(loc, RuntimeSymbols::kIncRef,
                       /*resultType=*/nullptr, mlir::ValueRange{storageValue});
      threadsafe::Retain::premise(retain.getOperation(),
                                  ThreadSafetyAttrs::kPremiseAggregateBorrow);
    }
    if (object_abi::Type::isLoweredStorage(storageValue.getType())) {
      mlir::Type convertedLogicalType = typeConverter.convertType(logicalType);
      if (convertedLogicalType &&
          storageValue.getType() != convertedLogicalType)
        return rewriter
            .create<mlir::UnrealizedConversionCastOp>(loc, convertedLogicalType,
                                                      storageValue)
            .getResult(0);
    }
    return storageValue;
  }
  if (convertedLogicalType &&
      mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
          convertedLogicalType)) {
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, convertedLogicalType,
                                                  storageValue)
        .getResult(0);
  }
  if (storageValue.getType() != convertedLogicalType) {
    mlir::emitError(loc) << "unsupported static field boxing from "
                         << storageValue.getType() << " to " << logicalType;
    return mlir::failure();
  }
  return storageValue;
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>>
extractContainerDescriptorParts(mlir::Location loc, mlir::Value storageValue,
                                mlir::Type logicalType,
                                mlir::TypeRange convertedTypes,
                                mlir::ConversionPatternRewriter &rewriter) {
  llvm::StringRef kind = containerDescriptorKind(logicalType);
  if (kind.empty())
    return mlir::failure();
  auto storageType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(storageValue.getType());
  if (!storageType || storageType.isOpaque() ||
      storageType.getBody().size() < convertedTypes.size())
    return mlir::failure();

  mlir::Operation *owner = storageValue.getDefiningOp();
  if (!owner && rewriter.getInsertionBlock())
    owner = rewriter.getInsertionBlock()->getParentOp();
  std::string group = container::descriptor::Group::make(owner, kind);
  llvm::SmallVector<mlir::Value> values;
  values.reserve(convertedTypes.size());
  for (auto [index, type] : llvm::enumerate(convertedTypes)) {
    mlir::Value storagePart = rewriter.create<mlir::LLVM::ExtractValueOp>(
        loc, storageType.getBody()[index], storageValue,
        rewriter.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    mlir::Value extracted = storagePart;
    if (storagePart.getType() != type)
      extracted =
          rewriter
              .create<mlir::UnrealizedConversionCastOp>(loc, type, storagePart)
              .getResult(0);
    llvm::StringRef component =
        containerDescriptorComponent(logicalType, index);
    if (!component.empty()) {
      container::descriptor::Component::mark(storagePart, group, component);
      container::descriptor::Component::mark(extracted, group, component);
    }
    values.push_back(extracted);
  }
  return values;
}

static void markContainerDescriptorParts(mlir::Operation *owner,
                                         mlir::ValueRange values,
                                         mlir::Type logicalType) {
  llvm::StringRef kind = containerDescriptorKind(logicalType);
  if (kind.empty())
    return;
  std::string group = container::descriptor::Group::make(owner, kind);
  for (auto [index, value] : llvm::enumerate(values)) {
    llvm::StringRef component =
        containerDescriptorComponent(logicalType, index);
    if (!component.empty())
      container::descriptor::Component::mark(value, group, component);
  }
}

static mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxStaticFieldValues(
    mlir::Location loc, mlir::Value storageValue, mlir::Type logicalType,
    mlir::ModuleOp module, mlir::ConversionPatternRewriter &rewriter,
    const PyLLVMTypeConverter &typeConverter, bool borrowExistingRef = false) {
  llvm::SmallVector<mlir::Type, 4> convertedTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return mlir::failure();
  if (mlir::isa<IntType>(logicalType) && convertedTypes.size() == 3 &&
      mlir::isa<mlir::IntegerType>(storageValue.getType())) {
    mlir::Value scalar = storageValue;
    auto intType = mlir::cast<mlir::IntegerType>(scalar.getType());
    if (intType.getWidth() < 64)
      scalar = rewriter.create<mlir::arith::ExtSIOp>(loc, rewriter.getI64Type(),
                                                     scalar);
    else if (intType.getWidth() > 64)
      scalar = rewriter.create<mlir::arith::TruncIOp>(
          loc, rewriter.getI64Type(), scalar);
    RuntimeAPI runtime(module, rewriter, typeConverter);
    auto call =
        runtime.call(loc, RuntimeSymbols::kLongFromI64,
                     mlir::TypeRange(convertedTypes), mlir::ValueRange{scalar});
    return llvm::SmallVector<mlir::Value>(call.getResults());
  }
  if (convertedTypes.size() > 1) {
    if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
      mlir::FailureOr<StaticClassLayout> layoutOr =
          class_layout::get(module.getOperation(), classType, typeConverter);
      if (mlir::failed(layoutOr) ||
          storageValue.getType() != layoutOr->objectType)
        return mlir::failure();
      if (!borrowExistingRef)
        Slot::classRefcount(loc, storageValue, classType, module, rewriter,
                            "incref", /*aggregateEffect=*/true,
                            ThreadSafetyAttrs::kPremiseAggregateBorrow);
      mlir::Value header =
          class_object::Local::header(loc, storageValue, *layoutOr, rewriter);
      if (!header ||
          convertedTypes.size() !=
              static_cast<size_t>(
                  class_layout::Object::partCount(layoutOr->objectType)) ||
          header.getType() != convertedTypes[0])
        return mlir::failure();
      llvm::SmallVector<mlir::Value> values{header};
      for (int64_t index = 0;
           index < class_layout::Payload::partCount(*layoutOr); ++index) {
        mlir::Value part = class_object::Local::payloadPart(
            loc, storageValue, *layoutOr, index, rewriter);
        if (!part ||
            part.getType() != convertedTypes[static_cast<size_t>(1 + index)])
          return mlir::failure();
        values.push_back(part);
      }
      return values;
    }
    if (auto extracted = extractContainerDescriptorParts(
            loc, storageValue, logicalType, mlir::TypeRange(convertedTypes),
            rewriter);
        mlir::succeeded(extracted))
      return *extracted;
    llvm::SmallVector<mlir::Value> values =
        typeConverter.materializeTargetConversion(
            rewriter, loc, mlir::TypeRange(convertedTypes),
            mlir::ValueRange{storageValue}, logicalType);
    if (values.empty())
      return mlir::failure();
    return values;
  }

  mlir::FailureOr<mlir::Value> boxed =
      boxStaticFieldValue(loc, storageValue, logicalType, module, rewriter,
                          typeConverter, borrowExistingRef);
  if (mlir::failed(boxed))
    return mlir::failure();
  return llvm::SmallVector<mlir::Value>{*boxed};
}

namespace attr_get::Borrow {
bool onlyUser(mlir::Operation *user) {
  return mlir::isa<AddOp, StrConcat3Op, SubOp, LeOp, LtOp, GtOp, GeOp, EqOp,
                   NeOp, ReprOp, ListAppendOp, ListRemoveOp, ListGetOp>(user);
}

template <typename AttrGetLikeOp> DecRefOp drop(AttrGetLikeOp op) {
  DecRefOp drop;
  mlir::Operation *borrowUser = nullptr;

  for (mlir::Operation *user : op.getResult().getUsers()) {
    if (auto decref = mlir::dyn_cast<DecRefOp>(user)) {
      if (drop)
        return nullptr;
      drop = decref;
      continue;
    }
    if (!attr_get::Borrow::onlyUser(user))
      return nullptr;
    if (borrowUser)
      return nullptr;
    borrowUser = user;
  }

  if (!borrowUser || !drop)
    return nullptr;
  if (borrowUser->getBlock() != drop->getBlock())
    return nullptr;
  if (!borrowUser->isBeforeInBlock(drop))
    return nullptr;
  return drop;
}
} // namespace attr_get::Borrow

static mlir::FailureOr<mlir::Value>
castIntegerFieldStorage(mlir::Location loc, mlir::Value value,
                        mlir::Type storageType,
                        mlir::ConversionPatternRewriter &rewriter) {
  auto valueInt = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  auto targetInt = mlir::dyn_cast<mlir::IntegerType>(storageType);
  if (!valueInt || !targetInt)
    return mlir::failure();
  if (valueInt.getWidth() == targetInt.getWidth())
    return value;
  if (valueInt.getWidth() < targetInt.getWidth())
    return rewriter.create<mlir::arith::ExtSIOp>(loc, storageType, value)
        .getResult();
  return rewriter.create<mlir::arith::TruncIOp>(loc, storageType, value)
      .getResult();
}

static mlir::FailureOr<mlir::Value>
unboxStaticFieldValue(mlir::Location loc, mlir::Value boxedValue,
                      mlir::Type logicalType, mlir::Type storageType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(module.getOperation(), classType, typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    auto helper = class_helper::Promote::get(loc, module, classType, *layoutOr,
                                             rewriter, typeConverter);
    if (!helper)
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> args;
    class_helper::Parts::append(loc, boxedValue, *layoutOr, rewriter, args);
    if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                           layoutOr->objectType)))
      return mlir::failure();
    auto promote = rewriter.create<mlir::func::CallOp>(loc, helper,
                                                       mlir::ValueRange{args});
    mlir::Value promoted = class_helper::Parts::object(
        loc, promote.getResults(), *layoutOr, rewriter);
    if (!promoted)
      return mlir::failure();
    if (promoted.getType() == storageType)
      return promoted;
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, storageType,
                                                  mlir::ValueRange{promoted})
        .getResult(0);
  }

  (void)module;
  (void)typeConverter;
  if (mlir::isa<IntType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageType)) {
    return castIntegerFieldStorage(loc, boxedValue, storageType, rewriter);
  }
  if (mlir::isa<FloatType>(logicalType) &&
      mlir::isa<mlir::FloatType>(storageType)) {
    if (boxedValue.getType() == storageType)
      return boxedValue;
    return mlir::failure();
  }
  if (mlir::isa<BoolType>(logicalType) &&
      mlir::isa<mlir::IntegerType>(storageType)) {
    if (auto valueInt =
            mlir::dyn_cast<mlir::IntegerType>(boxedValue.getType())) {
      auto targetInt = mlir::cast<mlir::IntegerType>(storageType);
      if (valueInt.getWidth() == targetInt.getWidth())
        return boxedValue;
      if (targetInt.getWidth() == 1) {
        mlir::Value zero =
            rewriter.create<mlir::arith::ConstantIntOp>(loc, 0, valueInt);
        return rewriter
            .create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::ne,
                                         boxedValue, zero)
            .getResult();
      }
      if (valueInt.getWidth() < targetInt.getWidth())
        return rewriter
            .create<mlir::arith::ExtUIOp>(loc, storageType, boxedValue)
            .getResult();
      return rewriter
          .create<mlir::arith::TruncIOp>(loc, storageType, boxedValue)
          .getResult();
    }
    return mlir::failure();
  }
  if (mlir::isa<mlir::MemRefType, mlir::UnrankedMemRefType>(
          boxedValue.getType())) {
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, storageType, boxedValue)
        .getResult(0);
  }
  if (boxedValue.getType() != storageType) {
    mlir::emitError(loc) << "unsupported static field unboxing from "
                         << boxedValue.getType() << " to " << storageType;
    return mlir::failure();
  }
  return boxedValue;
}

static mlir::FailureOr<mlir::Value>
unboxStaticFieldValue(mlir::Location loc, mlir::ValueRange boxedValues,
                      mlir::Type logicalType, mlir::Type storageType,
                      mlir::ModuleOp module,
                      mlir::ConversionPatternRewriter &rewriter,
                      const PyLLVMTypeConverter &typeConverter) {
  if (boxedValues.size() > 1) {
    if (auto classType = mlir::dyn_cast<ClassType>(logicalType)) {
      mlir::FailureOr<StaticClassLayout> layoutOr =
          class_layout::get(module.getOperation(), classType, typeConverter);
      if (mlir::failed(layoutOr) || storageType != layoutOr->objectType)
        return mlir::failure();
      mlir::Value object =
          class_helper::Parts::object(loc, boxedValues, *layoutOr, rewriter);
      if (!object)
        return mlir::failure();
      return object;
    }
    if (mlir::isa<IntType>(logicalType) &&
        mlir::isa<mlir::IntegerType>(storageType) && boxedValues.size() == 3) {
      RuntimeAPI runtime(module, rewriter, typeConverter);
      mlir::Value value = runtime
                              .call(loc, RuntimeSymbols::kLongAsI64,
                                    rewriter.getI64Type(), boxedValues)
                              .getResult();
      return castIntegerFieldStorage(loc, value, storageType, rewriter);
    }
    return rewriter
        .create<mlir::UnrealizedConversionCastOp>(loc, storageType, boxedValues)
        .getResult(0);
  }
  if (boxedValues.empty())
    return mlir::failure();
  return unboxStaticFieldValue(loc, boxedValues.front(), logicalType,
                               storageType, module, rewriter, typeConverter);
}

namespace class_object::Fresh {
mlir::func::FuncOp parentFunc(mlir::Value value) {
  if (mlir::Operation *def = value.getDefiningOp())
    return def->getParentOfType<mlir::func::FuncOp>();
  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    mlir::Block *owner = arg.getOwner();
    return owner ? mlir::dyn_cast_or_null<mlir::func::FuncOp>(
                       owner->getParentOp())
                 : mlir::func::FuncOp();
  }
  return {};
}

mlir::Value root(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      break;
    value = cast.getOperand(0);
  }
  return value;
}

bool isEntrySelfDescriptorCast(mlir::Value value, mlir::func::FuncOp func) {
  auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>();
  if (!cast || !func || cast->getNumOperands() == 0)
    return false;
  if (!mlir::isa<ClassType>(value.getType()))
    return false;
  mlir::Block &entry = func.getBody().front();
  for (auto [index, operand] : llvm::enumerate(cast->getOperands())) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(operand);
    if (!arg || arg.getOwner() != &entry ||
        arg.getArgNumber() != static_cast<unsigned>(index))
      return false;
  }
  return true;
}

bool zeroInitialized(mlir::Value value) {
  if (mlir::func::FuncOp func = parentFunc(value);
      func && static_cast<bool>(func->getAttr("ly.zero_initialized_self")) &&
      isEntrySelfDescriptorCast(value, func))
    return true;

  value = root(value);
  if (!mlir::isa<ClassType>(value.getType()))
    return false;

  if (mlir::isa_and_nonnull<ClassNewOp>(value.getDefiningOp()))
    return true;

  auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (!arg || arg.getArgNumber() != 0)
    return false;
  auto func = parentFunc(value);
  return func && static_cast<bool>(func->getAttr("ly.zero_initialized_self"));
}

bool skipOldDrop(mlir::Operation *op, mlir::Value object) {
  return static_cast<bool>(op->getAttr("ly.zero_init_first_store")) ||
         zeroInitialized(object);
}
} // namespace class_object::Fresh

// Static class instances lower to function-local stack slots.
struct ClassNewLowering : public mlir::OpConversionPattern<ClassNewOp> {
  ClassNewLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassNewOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassNewOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    if (classType.getClassName() == "Exception") {
      mlir::Type resultType =
          typeConverter->convertType(op.getResult().getType());
      if (!resultType)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<mlir::LLVM::ZeroOp>(op, resultType);
      return mlir::success();
    }
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();

    mlir::Value header = class_object::Slot::create(
        op.getLoc(), layoutOr->headerType, rewriter, op);
    llvm::SmallVector<mlir::Value, 8> payloadParts;
    for (mlir::MemRefType partType :
         class_layout::Payload::partTypes(*layoutOr)) {
      mlir::Value part =
          class_object::Slot::create(op.getLoc(), partType, rewriter, op);
      if (!part)
        return mlir::failure();
      class_object::Payload::markPart(part, rewriter);
      payloadParts.push_back(part);
    }
    if (!header)
      return mlir::failure();
    if (!class_object::Header::initialize(op.getLoc(), header, classType,
                                          *layoutOr, /*refcount=*/0, rewriter))
      return mlir::failure();
    mlir::Value zeroStorage =
        class_layout::Payload::zeroStorage(op.getLoc(), *layoutOr, rewriter);
    if (!zeroStorage)
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> fieldValues;
    for (auto [index, field] : llvm::enumerate(layoutOr->fields)) {
      (void)field;
      mlir::Value value = class_layout::Payload::extractField(
          op.getLoc(), *layoutOr, zeroStorage, static_cast<int64_t>(index),
          rewriter);
      if (!value)
        return mlir::failure();
      fieldValues.push_back(value);
    }
    if (!class_object::Payload::initialize(op.getLoc(), payloadParts,
                                           fieldValues, *layoutOr,
                                           /*managed=*/false, rewriter))
      return mlir::failure();
    mlir::Value objectValue = class_object::Local::objectValueFromParts(
        op.getLoc(), header, payloadParts, *layoutOr, rewriter);
    if (!objectValue)
      return mlir::failure();
    auto markOwnedLocal = [&](mlir::Value value) {
      if (mlir::Operation *def = value.getDefiningOp()) {
        def->setAttr(OwnershipContractAttrs::kOwnedLocalObject,
                     rewriter.getUnitAttr());
        def->setAttr(OwnershipContractAttrs::kObjectHeader,
                     rewriter.getUnitAttr());
      }
    };
    markOwnedLocal(header);
    llvm::SmallVector<mlir::Value> replacementValues{header};
    replacementValues.append(payloadParts.begin(), payloadParts.end());
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(replacementValues)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

struct ClassPromoteLowering : public mlir::OpConversionPattern<ClassPromoteOp> {
  ClassPromoteLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassPromoteOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassPromoteOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType());
    if (!classType)
      return mlir::failure();
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    auto helper = class_helper::Promote::get(
        op.getLoc(), module, classType, *layoutOr, rewriter, *typeConverter);
    if (!helper)
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> args;
    class_helper::Parts::append(op.getLoc(), adaptor.getInput(), *layoutOr,
                                rewriter, args);
    if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                           layoutOr->objectType)))
      return mlir::failure();
    auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), helper,
                                                    mlir::ValueRange{args});
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(call.getResults())};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

struct ClassReprLowering : public mlir::OpConversionPattern<ReprOp> {
  ClassReprLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ReprOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ReprOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto classType = mlir::dyn_cast<ClassType>(op.getInput().getType());
    if (!classType)
      return mlir::failure();
    if (adaptor.getInput().empty())
      return mlir::failure();

    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();

    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    llvm::SmallVector<mlir::Type, 3> resultTypes;
    if (mlir::failed(typeConverter->convertType(op.getResult().getType(),
                                                resultTypes)) ||
        resultTypes.empty())
      return mlir::failure();

    class_helper::Repr::ensure(op.getLoc(), module, classType, rewriter,
                               *typeConverter);
    std::string helperName = getClassHelperName(classType, "repr");
    if (auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName)) {
      mlir::FailureOr<StaticClassLayout> layoutOr =
          class_layout::get(op, classType, *typeConverter);
      if (mlir::failed(layoutOr))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 8> args;
      class_helper::Parts::append(op.getLoc(), adaptor.getInput(), *layoutOr,
                                  rewriter, args);
      if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                             layoutOr->objectType)))
        return mlir::failure();
      auto call =
          rewriter.create<mlir::func::CallOp>(op.getLoc(), helper, args);
      if (call->getNumResults() != resultTypes.size())
        return mlir::failure();
      llvm::SmallVector<mlir::ValueRange, 1> replacements{
          mlir::ValueRange(call.getResults())};
      rewriter.replaceOpWithMultiple(op, replacements);
      return mlir::success();
    }
    return mlir::failure();
  }
};

struct PublishLowering : public mlir::OpConversionPattern<PublishOp> {
  PublishLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<PublishOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(PublishOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto *pyTypeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());

    if (auto classType = mlir::dyn_cast<ClassType>(op.getResult().getType())) {
      if (adaptor.getInput().empty())
        return mlir::failure();
      mlir::FailureOr<StaticClassLayout> layoutOr =
          class_layout::get(op, classType, *pyTypeConverter);
      if (mlir::failed(layoutOr))
        return mlir::failure();
      auto helper =
          class_helper::Promote::get(op.getLoc(), module, classType, *layoutOr,
                                     rewriter, *pyTypeConverter);
      if (!helper)
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 8> args;
      class_helper::Parts::append(op.getLoc(), adaptor.getInput(), *layoutOr,
                                  rewriter, args);
      if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                             layoutOr->objectType)))
        return mlir::failure();
      auto call =
          rewriter.create<mlir::func::CallOp>(op.getLoc(), helper, args);
      llvm::SmallVector<mlir::ValueRange, 1> replacements{
          mlir::ValueRange(call.getResults())};
      rewriter.replaceOpWithMultiple(op, replacements);
      return mlir::success();
    }

    mlir::FailureOr<llvm::SmallVector<mlir::Value>> promoted = mlir::failure();
    mlir::Type resultType = op.getResult().getType();
    promoted = container::Descriptor::promote(
        op.getLoc(), resultType, adaptor.getInput(), module, rewriter,
        *pyTypeConverter, /*cloneReferenceSlots=*/true);
    if (mlir::succeeded(promoted)) {
      rewriter.replaceOpWithMultiple(
          op, llvm::ArrayRef<mlir::ValueRange>{*promoted});
      return mlir::success();
    }

    if (adaptor.getInput().size() != 1)
      return mlir::failure();
    RuntimeAPI runtime(module, rewriter, *pyTypeConverter);
    auto retain = runtime.call(op.getLoc(), RuntimeSymbols::kIncRef,
                               /*resultType=*/nullptr,
                               mlir::ValueRange{adaptor.getInput().front()});
    threadsafe::Retain::premise(retain.getOperation(),
                                ThreadSafetyAttrs::kPremiseOwnedToken);

    rewriter.replaceOp(op, adaptor.getInput().front());
    return mlir::success();
  }
};

// AttrGetOp lowers to direct field load from the class slot.
struct AttrGetLowering : public mlir::OpConversionPattern<AttrGetOp> {
  AttrGetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrGetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrGetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        class_layout::lookupField(op, classType, *typeConverter,
                                  op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    auto helper =
        class_helper::GetField::get(op.getLoc(), module, classType, *layoutOr,
                                    fieldOr->first, rewriter, *typeConverter);
    bool needsRefcount = class_field::Refcount::needed(
        fieldOr->second.logicalType, fieldOr->second.storageType,
        *typeConverter);
    bool needsClassRefcount = static_cast<bool>(
        mlir::dyn_cast<ClassType>(fieldOr->second.logicalType));
    bool needsContainerRefcount =
        mlir::isa<ListType, TupleType, DictType>(fieldOr->second.logicalType);
    DecRefOp borrowedDrop = nullptr;
    if (needsRefcount || needsClassRefcount || needsContainerRefcount)
      borrowedDrop = attr_get::Borrow::drop(op);
    mlir::Value borrowLocal = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(static_cast<bool>(borrowedDrop)));
    llvm::SmallVector<mlir::Value, 8> args;
    class_helper::Parts::append(op.getLoc(), adaptor.getObject(), *layoutOr,
                                rewriter, args);
    if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                           layoutOr->objectType)))
      return mlir::failure();
    args.push_back(borrowLocal);
    auto call = rewriter.create<mlir::func::CallOp>(op.getLoc(), helper,
                                                    mlir::ValueRange{args});
    llvm::SmallVector<mlir::Value> boxed;
    if (call.getNumResults() == 1) {
      mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxedOr =
          boxStaticFieldValues(op.getLoc(), call.getResult(0),
                               fieldOr->second.logicalType, module, rewriter,
                               *typeConverter, /*borrowExistingRef=*/true);
      if (mlir::failed(boxedOr))
        return mlir::failure();
      boxed = *boxedOr;
    } else {
      markContainerDescriptorParts(call.getOperation(), call.getResults(),
                                   fieldOr->second.logicalType);
      boxed.append(call.getResults().begin(), call.getResults().end());
    }
    if (llvm::any_of(boxed, [](mlir::Value value) { return !value; }))
      return mlir::failure();
    if (borrowedDrop)
      rewriter.eraseOp(borrowedDrop);
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(boxed)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

// AttrGetLocalOp is introduced only by the zero-cost rewrite pass after a
// local/non-shared proof. It may skip synchronization, but it must preserve the
// same owned-result contract as AttrGetOp; pair elision belongs in the
// ownership optimizer, not in lowering.
struct AttrGetLocalLowering : public mlir::OpConversionPattern<AttrGetLocalOp> {
  AttrGetLocalLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrGetLocalOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrGetLocalOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        class_layout::lookupField(op, classType, *typeConverter,
                                  op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    DecRefOp borrowedDrop = nullptr;
    bool isContainerField =
        mlir::isa<ListType, TupleType, DictType>(fieldOr->second.logicalType);
    bool borrowedLocalContainer =
        isContainerField &&
        static_cast<bool>(op->getAttr(ClassSafetyAttrs::kBorrowedLocalField));
    bool needsObjectRefcount = class_field::Refcount::needed(
        fieldOr->second.logicalType, fieldOr->second.storageType,
        *typeConverter);
    ClassType directClass =
        mlir::dyn_cast<ClassType>(fieldOr->second.logicalType);
    bool needsClassRefcount = static_cast<bool>(directClass);
    bool needsOwnershipUpdate =
        needsObjectRefcount || needsClassRefcount || isContainerField;
    if (needsObjectRefcount || needsClassRefcount || borrowedLocalContainer)
      borrowedDrop = attr_get::Borrow::drop(op);

    mlir::Value object = class_helper::Parts::object(
        op.getLoc(), adaptor.getObject(), *layoutOr, rewriter);
    if (!object)
      return mlir::failure();
    mlir::Value loaded = class_object::Local::loadField(
        op.getLoc(), object, *layoutOr, fieldOr->first,
        fieldOr->second.storageType, rewriter);
    if (!loaded)
      return mlir::failure();
    if (needsOwnershipUpdate)
      class_field::Ownership::markLoad(loaded, fieldOr->first);
    mlir::Value resultStorage = loaded;
    if (needsOwnershipUpdate && !borrowedDrop) {
      if (needsObjectRefcount) {
        if (mlir::failed(class_field::Runtime::single(
                op.getLoc(), resultStorage, fieldOr->second.logicalType, module,
                rewriter, *typeConverter, RuntimeSymbols::kIncRef,
                ThreadSafetyAttrs::kPremiseAggregateBorrow)))
          return mlir::failure();
      } else if (needsClassRefcount) {
        Slot::classRefcount(op.getLoc(), resultStorage, directClass, module,
                            rewriter, "incref", /*aggregateEffect=*/true,
                            ThreadSafetyAttrs::kPremiseAggregateBorrow);
      } else {
        ownership::aggregate::Slot::markLoad(resultStorage);
        resultStorage = class_container::Clone::retainOrCloneLocalAlias(
            op.getLoc(), resultStorage, fieldOr->second.logicalType, object,
            *layoutOr, fieldOr->first, module, rewriter, *typeConverter,
            ThreadSafetyAttrs::kPremiseAggregateBorrow);
      }
    }
    mlir::FailureOr<llvm::SmallVector<mlir::Value>> boxed =
        boxStaticFieldValues(op.getLoc(), resultStorage,
                             fieldOr->second.logicalType, module, rewriter,
                             *typeConverter,
                             needsObjectRefcount || needsClassRefcount ||
                                 static_cast<bool>(borrowedDrop));
    if (mlir::failed(boxed))
      return mlir::failure();
    if (llvm::any_of(*boxed, [](mlir::Value value) { return !value; }))
      return mlir::failure();
    if (borrowedDrop)
      rewriter.eraseOp(borrowedDrop);
    llvm::SmallVector<mlir::ValueRange, 1> replacements{
        mlir::ValueRange(*boxed)};
    rewriter.replaceOpWithMultiple(op, replacements);
    return mlir::success();
  }
};

// AttrSetOp lowers through the synchronized helper path.
struct AttrSetLowering : public mlir::OpConversionPattern<AttrSetOp> {
  AttrSetLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrSetOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrSetOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        class_layout::lookupField(op, classType, *typeConverter,
                                  op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> unboxed = unboxStaticFieldValue(
        op.getLoc(), adaptor.getValue(), fieldOr->second.logicalType,
        fieldOr->second.storageType, module, rewriter, *typeConverter);
    if (mlir::failed(unboxed))
      return mlir::failure();
    bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
    if (mlir::isa<ClassType, ListType, TupleType, DictType>(
            fieldOr->second.logicalType))
      consumeValue = false;
    bool skipOldValueLoad =
        class_object::Fresh::skipOldDrop(op, op.getObject());
    auto helper =
        class_helper::SetField::get(op.getLoc(), module, classType, *layoutOr,
                                    fieldOr->first, rewriter, *typeConverter);
    if (!helper)
      return mlir::failure();
    mlir::Value retainNewValue = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(), rewriter.getBoolAttr(!consumeValue));
    mlir::Value skipOldDrop = rewriter.create<mlir::LLVM::ConstantOp>(
        op.getLoc(), rewriter.getI1Type(),
        rewriter.getBoolAttr(skipOldValueLoad));
    llvm::SmallVector<mlir::Value, 8> args;
    class_helper::Parts::append(op.getLoc(), adaptor.getObject(), *layoutOr,
                                rewriter, args);
    if (args.size() != static_cast<size_t>(class_layout::Object::partCount(
                           layoutOr->objectType)))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Value>> valueParts =
        class_field::ABI::parts(op.getLoc(), *unboxed, fieldOr->second, module,
                                rewriter, *typeConverter);
    if (mlir::failed(valueParts))
      return mlir::failure();
    args.append(valueParts->begin(), valueParts->end());
    args.push_back(retainNewValue);
    args.push_back(skipOldDrop);
    rewriter.create<mlir::func::CallOp>(op.getLoc(), helper,
                                        mlir::ValueRange{args});
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// AttrSetLocalOp is introduced only by the zero-cost rewrite pass after a
// local/non-shared proof. It can therefore lower to direct field mutation.
struct AttrSetLocalLowering : public mlir::OpConversionPattern<AttrSetLocalOp> {
  AttrSetLocalLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<AttrSetLocalOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(AttrSetLocalOp op, OneToNOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    auto classType = mlir::dyn_cast<ClassType>(op.getObject().getType());
    if (!classType)
      return mlir::failure();
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    mlir::FailureOr<std::pair<unsigned, StaticClassFieldInfo>> fieldOr =
        class_layout::lookupField(op, classType, *typeConverter,
                                  op.getNameAttr().getValue());
    if (mlir::failed(fieldOr))
      return mlir::failure();

    mlir::FailureOr<mlir::Value> unboxed = unboxStaticFieldValue(
        op.getLoc(), adaptor.getValue(), fieldOr->second.logicalType,
        fieldOr->second.storageType, module, rewriter, *typeConverter);
    if (mlir::failed(unboxed))
      return mlir::failure();
    bool consumeValue = static_cast<bool>(op->getAttr("ly.consume_value"));
    if (mlir::isa<ClassType, ListType, TupleType, DictType>(
            fieldOr->second.logicalType))
      consumeValue = false;
    bool skipOldValueLoad =
        class_object::Fresh::skipOldDrop(op, op.getObject());

    mlir::Value oldValue;
    bool needsRefcount = class_field::Refcount::needed(
        fieldOr->second.logicalType, fieldOr->second.storageType,
        *typeConverter);
    ClassType directClass =
        mlir::dyn_cast<ClassType>(fieldOr->second.logicalType);
    bool needsContainerRefcount =
        mlir::isa<ListType, TupleType, DictType>(fieldOr->second.logicalType);
    bool needsOwnershipUpdate = needsRefcount ||
                                static_cast<bool>(directClass) ||
                                needsContainerRefcount;
    mlir::Value storedValue = *unboxed;
    mlir::Value object = class_helper::Parts::object(
        op.getLoc(), adaptor.getObject(), *layoutOr, rewriter);
    if (!object)
      return mlir::failure();
    if (needsOwnershipUpdate) {
      if (!skipOldValueLoad) {
        oldValue = class_object::Local::loadField(
            op.getLoc(), object, *layoutOr, fieldOr->first,
            fieldOr->second.storageType, rewriter);
        if (!oldValue)
          return mlir::failure();
      }
      if (!consumeValue) {
        if (needsContainerRefcount) {
          ownership::aggregate::Slot::markLoad(storedValue);
          storedValue = class_container::Clone::retainOrCloneOwned(
              op.getLoc(), storedValue, fieldOr->second.logicalType, module,
              rewriter, *typeConverter,
              isEntryBorrowedValue(storedValue)
                  ? ThreadSafetyAttrs::kPremiseEntryBorrowed
                  : ThreadSafetyAttrs::kPremiseOwnedToken);
        } else if (directClass) {
          Slot::classRefcount(op.getLoc(), storedValue, directClass, module,
                              rewriter, "incref", /*aggregateEffect=*/true,
                              isEntryBorrowedValue(storedValue)
                                  ? ThreadSafetyAttrs::kPremiseEntryBorrowed
                                  : ThreadSafetyAttrs::kPremiseOwnedToken);
        } else {
          if (mlir::failed(class_field::Runtime::single(
                  op.getLoc(), storedValue, fieldOr->second.logicalType, module,
                  rewriter, *typeConverter, RuntimeSymbols::kIncRef,
                  isEntryBorrowedValue(storedValue)
                      ? ThreadSafetyAttrs::kPremiseEntryBorrowed
                      : ThreadSafetyAttrs::kPremiseOwnedToken)))
            return mlir::failure();
        }
      }
    }
    mlir::Operation *store = class_object::Local::storeField(
        op.getLoc(), object, *layoutOr, fieldOr->first, storedValue, rewriter);
    if (!store)
      return mlir::failure();
    if (needsOwnershipUpdate)
      class_field::Ownership::markStore(store, fieldOr->first);
    if (oldValue) {
      if (directClass) {
        Slot::classRefcount(op.getLoc(), oldValue, directClass, module,
                            rewriter, "decref", /*aggregateEffect=*/true,
                            ThreadSafetyAttrs::kPremiseAggregateBorrow);
      } else if (needsRefcount) {
        if (mlir::failed(class_field::Runtime::single(
                op.getLoc(), oldValue, fieldOr->second.logicalType, module,
                rewriter, *typeConverter, kDecrefIntent)))
          return mlir::failure();
      } else if (needsContainerRefcount) {
        if (mlir::failed(class_container::Field::destroy(
                op.getLoc(), oldValue, fieldOr->second.logicalType, module,
                rewriter, *typeConverter)))
          return mlir::failure();
      }
    }
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

// ClassOp is just a container - erase it and keep its methods at module scope
struct ClassOpLowering : public mlir::OpConversionPattern<ClassOp> {
  ClassOpLowering(PyLLVMTypeConverter &converter, mlir::MLIRContext *ctx)
      : mlir::OpConversionPattern<ClassOp>(converter, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto *typeConverter =
        static_cast<const PyLLVMTypeConverter *>(getTypeConverter());
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return mlir::failure();
    ClassType classType =
        ClassType::get(op.getContext(), op.getSymNameAttr().getValue());
    mlir::FailureOr<StaticClassLayout> layoutOr =
        class_layout::get(op, classType, *typeConverter);
    if (mlir::failed(layoutOr))
      return mlir::failure();
    ensureStaticClassHelperBodies(op, *layoutOr, module, rewriter,
                                  *typeConverter);
    rewriter.eraseOp(op);
    return mlir::success();
  }
};

} // namespace

namespace lowering::value::class_::Copy {
mlir::FailureOr<mlir::func::FuncOp>
ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
       mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter) {
  std::string helperName = getClassHelperName(classType, "copy");
  if (auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName))
    return helper;

  mlir::FailureOr<StaticClassLayout> layoutOr =
      class_layout::get(module.getOperation(), classType, typeConverter);
  if (mlir::failed(layoutOr))
    return mlir::failure();
  auto helper = class_helper::Copy::get(loc, module, classType, *layoutOr,
                                        builder, typeConverter);
  if (!helper)
    return mlir::failure();
  return helper;
}
} // namespace lowering::value::class_::Copy

namespace lowering::value::class_::Eq {
mlir::FailureOr<mlir::func::FuncOp>
ensure(mlir::Location loc, mlir::ModuleOp module, ClassType classType,
       mlir::OpBuilder &builder, const PyLLVMTypeConverter &typeConverter) {
  std::string helperName = getClassHelperName(classType, "eq");
  if (auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName))
    return helper;

  mlir::FailureOr<StaticClassLayout> layoutOr =
      class_layout::get(module.getOperation(), classType, typeConverter);
  if (mlir::failed(layoutOr))
    return mlir::failure();
  auto helper = class_helper::Eq::get(loc, module, classType, *layoutOr,
                                      builder, typeConverter);
  if (!helper)
    return mlir::failure();
  return helper;
}
} // namespace lowering::value::class_::Eq

namespace lowering::value::class_::Patterns {
void populate(PyLLVMTypeConverter &typeConverter,
              mlir::RewritePatternSet &patterns) {
  auto *ctx = patterns.getContext();
  patterns.add<ClassNewLowering, ClassPromoteLowering, PublishLowering,
               ClassReprLowering, AttrGetLowering, AttrGetLocalLowering,
               AttrSetLowering, AttrSetLocalLowering, ClassOpLowering>(
      typeConverter, ctx);
}
} // namespace lowering::value::class_::Patterns

} // namespace py
