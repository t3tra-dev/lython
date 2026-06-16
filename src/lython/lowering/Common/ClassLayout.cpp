#include "Common/ClassLayout.h"

#include "Common/Object.h"
#include "Common/RuntimeSupport.h"
#include "cpp/PyTypeObject.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <utility>

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

static ClassOp lookup(mlir::Operation *from, ClassType classType) {
  if (!from || !classType)
    return nullptr;
  mlir::StringAttr nameAttr =
      mlir::StringAttr::get(from->getContext(), classType.getClassName());
  for (mlir::Operation *symbolTableOp = from; symbolTableOp;
       symbolTableOp = symbolTableOp->getParentOp()) {
    if (!symbolTableOp->hasTrait<mlir::OpTrait::SymbolTable>())
      continue;
    if (mlir::Operation *symbol =
            mlir::SymbolTable::lookupSymbolIn(symbolTableOp, nameAttr))
      if (ClassOp classOp = mlir::dyn_cast<ClassOp>(symbol))
        return classOp;
  }
  return nullptr;
}

} // namespace

namespace class_layout {

mlir::MemRefType Header::memrefType(mlir::MLIRContext *ctx) {
  return object_abi::Header::owned(ctx);
}

namespace DescriptorShape {

static int64_t staticSize(mlir::MemRefType type) {
  if (!type || type.getRank() != 1 ||
      mlir::ShapedType::isDynamic(type.getShape().front()))
    return -1;
  return type.getShape().front();
}

static int64_t elementWidth(mlir::MemRefType type) {
  if (!type)
    return -1;
  mlir::Type element = type.getElementType();
  if (auto integer = mlir::dyn_cast<mlir::IntegerType>(element))
    return integer.getWidth();
  if (auto floating = mlir::dyn_cast<mlir::FloatType>(element))
    return floating.getWidth();
  return -1;
}

static void setI64(mlir::Operation *op, llvm::StringRef name, int64_t value) {
  if (!op)
    return;
  auto type = mlir::IntegerType::get(op->getContext(), 64);
  op->setAttr(name, mlir::IntegerAttr::get(type, value));
}

static std::optional<int64_t> getI64(mlir::Operation *op,
                                     llvm::StringRef name) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  if (!attr)
    return std::nullopt;
  return attr.getInt();
}

void mark(mlir::Operation *op, mlir::MemRefType memrefType) {
  if (!op || !memrefType)
    return;
  setI64(op, ClassSafetyAttrs::kCarrierPartRank, memrefType.getRank());
  setI64(op, ClassSafetyAttrs::kCarrierPartElementWidth,
         elementWidth(memrefType));
  setI64(op, ClassSafetyAttrs::kCarrierPartStaticSize, staticSize(memrefType));
}

void copy(mlir::Operation *source, mlir::Operation *target) {
  if (!source || !target)
    return;
  for (llvm::StringRef attr : {ClassSafetyAttrs::kCarrierPartRank,
                               ClassSafetyAttrs::kCarrierPartElementWidth,
                               ClassSafetyAttrs::kCarrierPartStaticSize})
    if (mlir::Attribute value = source->getAttr(attr))
      target->setAttr(attr, value);
}

bool has(mlir::Operation *op) {
  return getI64(op, ClassSafetyAttrs::kCarrierPartRank).has_value() &&
         getI64(op, ClassSafetyAttrs::kCarrierPartElementWidth).has_value() &&
         getI64(op, ClassSafetyAttrs::kCarrierPartStaticSize).has_value();
}

bool matches(mlir::Operation *op, mlir::MemRefType memrefType) {
  auto rank = getI64(op, ClassSafetyAttrs::kCarrierPartRank);
  auto width = getI64(op, ClassSafetyAttrs::kCarrierPartElementWidth);
  auto size = getI64(op, ClassSafetyAttrs::kCarrierPartStaticSize);
  return rank && width && size && *rank == memrefType.getRank() &&
         *width == elementWidth(memrefType) && *size == staticSize(memrefType);
}

mlir::LogicalResult verify(mlir::Operation *op, mlir::MemRefType memrefType,
                           llvm::StringRef what) {
  if (!has(op))
    return op->emitOpError() << what << " lacks class carrier memref shape "
                             << "contract";
  if (matches(op, memrefType))
    return mlir::success();
  return op->emitOpError() << what << " shape contract does not match expected "
                           << memrefType;
}

} // namespace DescriptorShape

mlir::MemRefType Payload::lockMemRefType(mlir::MLIRContext *ctx) {
  return mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
}

mlir::MemRefType Payload::tableMemRefType(mlir::MLIRContext *ctx) {
  auto layout = mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic,
                                             llvm::ArrayRef<int64_t>{1});
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic},
                               object_abi::Type::loweredStorage(ctx), layout);
}

mlir::MemRefType Payload::fieldMemRefType(mlir::Type fieldStorageType,
                                          mlir::MLIRContext *ctx) {
  auto layout = mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic,
                                             llvm::ArrayRef<int64_t>{1});
  return mlir::MemRefType::get({1}, fieldStorageType, layout);
}

static mlir::LLVM::LLVMStructType
objectType(mlir::MLIRContext *ctx, llvm::ArrayRef<mlir::Type> descriptors) {
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, descriptors);
}

static void appendPayloadMetadata(mlir::MLIRContext *ctx,
                                  llvm::SmallVectorImpl<mlir::Type> &types) {
  types.push_back(descriptorStorageType(Payload::lockMemRefType(ctx), ctx));
}

static void assignStorage(Layout &layout, mlir::MLIRContext *ctx) {
  llvm::SmallVector<mlir::Type, 8> payloadTypes;
  appendPayloadMetadata(ctx, payloadTypes);
  for (const FieldInfo &field : layout.fields)
    payloadTypes.append(field.storageParts.begin(), field.storageParts.end());
  layout.headerType = Header::memrefType(ctx);
  layout.storageType = mlir::LLVM::LLVMStructType::getLiteral(
      ctx, llvm::ArrayRef<mlir::Type>(payloadTypes));
  layout.payloadPartTypes.clear();
  layout.payloadPartTypes.push_back(Payload::lockMemRefType(ctx));
  for (const FieldInfo &field : layout.fields)
    for (mlir::Type storagePart : field.storageParts)
      layout.payloadPartTypes.push_back(
          Payload::fieldMemRefType(storagePart, ctx));
  mlir::Type headerDescriptorType =
      descriptorStorageType(layout.headerType, ctx);
  mlir::Type tableDescriptorType =
      descriptorStorageType(Payload::tableMemRefType(ctx), ctx);
  llvm::SmallVector<mlir::Type, 2> descriptors{headerDescriptorType,
                                               tableDescriptorType};
  layout.objectType = objectType(ctx, descriptors);
}

int64_t Object::partCount(mlir::LLVM::LLVMStructType objectType) {
  if (!isObjectCarrierType(objectType))
    return 0;
  return static_cast<int64_t>(objectType.getBody().size());
}

mlir::Type Object::descriptorType(mlir::LLVM::LLVMStructType objectType,
                                  int64_t partIndex) {
  if (!isObjectCarrierType(objectType) || partIndex < 0 ||
      partIndex >= static_cast<int64_t>(objectType.getBody().size()))
    return {};
  return objectType.getBody()[partIndex];
}

mlir::Type Object::headerDescriptorType(mlir::LLVM::LLVMStructType objectType) {
  return Object::descriptorType(objectType, Object::kHeaderIndex);
}

mlir::Value Object::descriptor(mlir::Location loc,
                               mlir::LLVM::LLVMStructType objectType,
                               mlir::Value object, int64_t partIndex,
                               mlir::OpBuilder &builder) {
  mlir::Type partType = Object::descriptorType(objectType, partIndex);
  if (!partType || !object || object.getType() != objectType)
    return {};
  auto extract = builder.create<mlir::LLVM::ExtractValueOp>(
      loc, partType, object, builder.getDenseI64ArrayAttr({partIndex}));
  extract->setAttr(ClassSafetyAttrs::kCarrierPart,
                   builder.getI64IntegerAttr(partIndex));
  return extract;
}

mlir::Value Object::headerDescriptor(mlir::Location loc,
                                     mlir::LLVM::LLVMStructType objectType,
                                     mlir::Value object,
                                     mlir::OpBuilder &builder) {
  return Object::descriptor(loc, objectType, object, Object::kHeaderIndex,
                            builder);
}

mlir::Value Object::fromDescriptors(mlir::Location loc,
                                    mlir::LLVM::LLVMStructType objectType,
                                    mlir::ValueRange descriptors,
                                    mlir::OpBuilder &builder) {
  if (!isObjectCarrierType(objectType) ||
      descriptors.size() != static_cast<size_t>(Object::partCount(objectType)))
    return {};
  mlir::Value objectValue =
      builder.create<mlir::LLVM::UndefOp>(loc, objectType);
  for (auto [index, descriptor] : llvm::enumerate(descriptors)) {
    mlir::Type expected =
        Object::descriptorType(objectType, static_cast<int64_t>(index));
    if (!descriptor || descriptor.getType() != expected)
      return {};
    auto insert = builder.create<mlir::LLVM::InsertValueOp>(
        loc, objectType, objectValue, descriptor,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
    insert->setAttr(ClassSafetyAttrs::kCarrierPack, builder.getUnitAttr());
    insert->setAttr(ClassSafetyAttrs::kCarrierPart,
                    builder.getI64IntegerAttr(static_cast<int64_t>(index)));
    DescriptorShape::copy(descriptor.getDefiningOp(), insert.getOperation());
    if (!DescriptorShape::has(insert.getOperation()) &&
        index == Object::kHeaderIndex)
      DescriptorShape::mark(insert.getOperation(),
                            object_abi::Header::owned(builder.getContext()));
    objectValue = insert;
  }
  return objectValue;
}

llvm::ArrayRef<mlir::MemRefType> Payload::partTypes(const Layout &layout) {
  return layout.payloadPartTypes;
}

int64_t Payload::partCount(const Layout &layout) {
  return static_cast<int64_t>(layout.payloadPartTypes.size());
}

mlir::MemRefType Payload::partType(const Layout &layout, int64_t partIndex) {
  if (partIndex < 0 ||
      partIndex >= static_cast<int64_t>(layout.payloadPartTypes.size()))
    return {};
  return layout.payloadPartTypes[static_cast<size_t>(partIndex)];
}

int64_t Payload::fieldPartIndex(const Layout &layout, int64_t fieldIndex) {
  if (fieldIndex < 0 ||
      fieldIndex >= static_cast<int64_t>(layout.fields.size()))
    return -1;
  return layout.fields[static_cast<size_t>(fieldIndex)].payloadPartStart;
}

int64_t Payload::fieldPartCount(const Layout &layout, int64_t fieldIndex) {
  if (fieldIndex < 0 ||
      fieldIndex >= static_cast<int64_t>(layout.fields.size()))
    return 0;
  return layout.fields[static_cast<size_t>(fieldIndex)].payloadPartCount;
}

int64_t Payload::lockPartIndex(const Layout &layout) {
  return layout.payloadPartTypes.empty() ? -1 : 0;
}

mlir::MemRefType Payload::fieldPartType(const Layout &layout,
                                        int64_t fieldIndex) {
  if (Payload::fieldPartCount(layout, fieldIndex) != 1)
    return {};
  return Payload::partType(layout, Payload::fieldPartIndex(layout, fieldIndex));
}

mlir::LLVM::LLVMStructType Payload::storageType(const Layout &layout) {
  return layout.storageType;
}

bool Payload::isStorageValue(const Layout &layout, mlir::Value value) {
  return value && value.getType() == Payload::storageType(layout);
}

mlir::Type Payload::fieldStorageType(const Layout &layout, int64_t fieldIndex) {
  if (fieldIndex < 0 ||
      fieldIndex >= static_cast<int64_t>(layout.fields.size()))
    return {};
  return layout.fields[static_cast<size_t>(fieldIndex)].storageType;
}

mlir::Type Payload::lockStorageType(const Layout &layout) {
  int64_t index = Payload::lockPartIndex(layout);
  if (!layout.storageType || layout.storageType.isOpaque() ||
      index >= static_cast<int64_t>(layout.storageType.getBody().size()))
    return {};
  return layout.storageType.getBody()[index];
}

mlir::Value Payload::zeroStorage(mlir::Location loc, const Layout &layout,
                                 mlir::OpBuilder &builder) {
  if (!layout.storageType)
    return {};
  return builder.create<mlir::LLVM::ZeroOp>(loc, layout.storageType);
}

mlir::Value Payload::extractField(mlir::Location loc, const Layout &layout,
                                  mlir::Value storage, int64_t fieldIndex,
                                  mlir::OpBuilder &builder) {
  if (!storage || storage.getType() != layout.storageType)
    return {};
  int64_t start = Payload::fieldPartIndex(layout, fieldIndex);
  int64_t count = Payload::fieldPartCount(layout, fieldIndex);
  if (start < 0 || count <= 0)
    return {};
  llvm::SmallVector<mlir::Value, 4> parts;
  parts.reserve(count);
  for (int64_t offset = 0; offset < count; ++offset) {
    int64_t partIndex = start + offset;
    mlir::Type partType = layout.storageType.getBody()[partIndex];
    parts.push_back(builder.create<mlir::LLVM::ExtractValueOp>(
        loc, partType, storage, builder.getDenseI64ArrayAttr({partIndex})));
  }
  return Payload::composeField(loc, layout, fieldIndex, parts, builder);
}

mlir::Value Payload::insertField(mlir::Location loc, const Layout &layout,
                                 mlir::Value storage, int64_t fieldIndex,
                                 mlir::Value fieldValue,
                                 mlir::OpBuilder &builder) {
  if (!storage || !fieldValue || storage.getType() != layout.storageType)
    return {};
  auto parts =
      Payload::decomposeField(loc, layout, fieldIndex, fieldValue, builder);
  if (mlir::failed(parts))
    return {};
  int64_t start = Payload::fieldPartIndex(layout, fieldIndex);
  if (start < 0 || parts->size() != static_cast<size_t>(Payload::fieldPartCount(
                                        layout, fieldIndex)))
    return {};
  mlir::Value result = storage;
  for (auto [offset, part] : llvm::enumerate(*parts)) {
    int64_t partIndex = start + static_cast<int64_t>(offset);
    result = builder.create<mlir::LLVM::InsertValueOp>(
        loc, layout.storageType, result, part,
        builder.getDenseI64ArrayAttr({partIndex}));
  }
  return result;
}

mlir::Value Payload::composeField(mlir::Location loc, const Layout &layout,
                                  int64_t fieldIndex, mlir::ValueRange parts,
                                  mlir::OpBuilder &builder) {
  mlir::Type fieldType = Payload::fieldStorageType(layout, fieldIndex);
  int64_t count = Payload::fieldPartCount(layout, fieldIndex);
  if (!fieldType || parts.size() != static_cast<size_t>(count))
    return {};
  if (count == 1) {
    mlir::Value part = parts.front();
    if (part.getType() == fieldType)
      return part;
    return builder
        .create<mlir::UnrealizedConversionCastOp>(loc, fieldType,
                                                  mlir::ValueRange{part})
        .getResult(0);
  }
  if (auto objectType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(fieldType);
      isObjectCarrierType(objectType))
    return Object::fromDescriptors(loc, objectType, parts, builder);
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(fieldType);
  if (!aggregate || aggregate.isOpaque() ||
      aggregate.getBody().size() != parts.size())
    return {};
  mlir::Value value = builder.create<mlir::LLVM::UndefOp>(loc, fieldType);
  for (auto [index, part] : llvm::enumerate(parts)) {
    if (part.getType() != aggregate.getBody()[index])
      part = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     loc, aggregate.getBody()[index], mlir::ValueRange{part})
                 .getResult(0);
    value = builder.create<mlir::LLVM::InsertValueOp>(
        loc, fieldType, value, part,
        builder.getDenseI64ArrayAttr({static_cast<int64_t>(index)}));
  }
  return value;
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
Payload::decomposeField(mlir::Location loc, const Layout &layout,
                        int64_t fieldIndex, mlir::Value fieldValue,
                        mlir::OpBuilder &builder) {
  if (fieldIndex < 0 ||
      fieldIndex >= static_cast<int64_t>(layout.fields.size()))
    return mlir::failure();
  int64_t count = Payload::fieldPartCount(layout, fieldIndex);
  if (!fieldValue || count <= 0)
    return mlir::failure();
  const FieldInfo &field = layout.fields[static_cast<size_t>(fieldIndex)];
  if (field.storageParts.size() != static_cast<size_t>(count))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 4> parts;
  parts.reserve(count);
  if (count == 1) {
    mlir::Value part = fieldValue;
    if (part.getType() != field.storageParts.front())
      part = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     loc, field.storageParts.front(), mlir::ValueRange{part})
                 .getResult(0);
    parts.push_back(part);
    return parts;
  }
  if (auto objectType =
          mlir::dyn_cast<mlir::LLVM::LLVMStructType>(fieldValue.getType());
      isObjectCarrierType(objectType)) {
    for (int64_t index = 0; index < count; ++index) {
      mlir::Value part =
          Object::descriptor(loc, objectType, fieldValue, index, builder);
      if (!part)
        return mlir::failure();
      parts.push_back(part);
    }
    return parts;
  }
  auto aggregate =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(fieldValue.getType());
  if (!aggregate || aggregate.isOpaque() ||
      aggregate.getBody().size() != static_cast<size_t>(count))
    return mlir::failure();
  for (int64_t index = 0; index < count; ++index) {
    mlir::Value part = builder.create<mlir::LLVM::ExtractValueOp>(
        loc, aggregate.getBody()[static_cast<size_t>(index)], fieldValue,
        builder.getDenseI64ArrayAttr({index}));
    if (part.getType() != field.storageParts[static_cast<size_t>(index)])
      part = builder
                 .create<mlir::UnrealizedConversionCastOp>(
                     loc, field.storageParts[static_cast<size_t>(index)],
                     mlir::ValueRange{part})
                 .getResult(0);
    parts.push_back(part);
  }
  return parts;
}

mlir::Type descriptorStorageType(mlir::MemRefType memrefType,
                                 mlir::MLIRContext *ctx) {
  if (!memrefType || memrefType.getRank() != 1)
    return {};
  return object_abi::Type::loweredStorage(ctx);
}

bool isDescriptorStorageType(mlir::Type type) {
  if (!type)
    return false;
  return object_abi::Type::isLoweredStorage(type);
}

bool isObjectCarrierType(mlir::Type type) {
  if (!type)
    return false;
  auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type);
  if (!aggregate || aggregate.isOpaque() || aggregate.getBody().size() < 2)
    return false;
  return llvm::all_of(aggregate.getBody(), isDescriptorStorageType);
}

bool isObjectCarrierMemRefType(mlir::Type type) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  return memrefType && memrefType.getRank() == 1 &&
         isObjectCarrierType(memrefType.getElementType());
}

mlir::LLVM::LLVMStructType
objectCarrierType(mlir::MLIRContext *ctx,
                  llvm::ArrayRef<mlir::MemRefType> partTypes) {
  if (!ctx || partTypes.size() < 2)
    return {};
  llvm::SmallVector<mlir::Type, 4> descriptors;
  descriptors.reserve(partTypes.size());
  for (mlir::MemRefType partType : partTypes) {
    mlir::Type descriptor = descriptorStorageType(partType, ctx);
    if (!descriptor)
      return {};
    descriptors.push_back(descriptor);
  }
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, descriptors);
}

mlir::LLVM::LLVMStructType objectCarrierType(mlir::Type type) {
  if (!type)
    return {};
  if (auto aggregate = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(type))
    return isObjectCarrierType(aggregate) ? aggregate
                                          : mlir::LLVM::LLVMStructType();
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType || memrefType.getRank() != 1)
    return {};
  auto elementType =
      mlir::dyn_cast<mlir::LLVM::LLVMStructType>(memrefType.getElementType());
  return isObjectCarrierType(elementType) ? elementType
                                          : mlir::LLVM::LLVMStructType();
}

mlir::MemRefType carrierType(mlir::LLVM::LLVMStructType objectType,
                             mlir::MLIRContext *ctx) {
  auto layout = mlir::StridedLayoutAttr::get(ctx, mlir::ShapedType::kDynamic,
                                             llvm::ArrayRef<int64_t>{1});
  return mlir::MemRefType::get({1}, objectType, layout);
}

void partsValueTypes(const Layout &layout,
                     llvm::SmallVectorImpl<mlir::Type> &types) {
  types.push_back(layout.headerType);
  types.push_back(Payload::tableMemRefType(layout.headerType.getContext()));
}

static mlir::FailureOr<Layout>
build(mlir::Operation *from, ClassType classType,
      const PyLLVMTypeConverter &typeConverter,
      llvm::SmallVectorImpl<ClassType> &activeClasses);

static mlir::LogicalResult appendClassFields(
    mlir::Operation *from, ClassOp owner,
    llvm::SmallVectorImpl<std::pair<mlir::StringAttr, mlir::TypeAttr>> &fields,
    llvm::StringMap<unsigned> &fieldIndexes) {
  mlir::ArrayAttr fieldNamesAttr = owner.getFieldNamesAttr();
  mlir::ArrayAttr fieldTypesAttr = owner.getFieldTypesAttr();
  if (!fieldNamesAttr && !fieldTypesAttr)
    return mlir::success();
  if (!fieldNamesAttr || !fieldTypesAttr ||
      fieldNamesAttr.size() != fieldTypesAttr.size()) {
    return from->emitError("class '") << owner.getSymNameAttr().getValue()
                                      << "' has malformed static field schema";
  }

  for (auto [nameAttr, typeAttr] : llvm::zip(fieldNamesAttr, fieldTypesAttr)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto type = mlir::dyn_cast<mlir::TypeAttr>(typeAttr);
    if (!stringAttr || !type) {
      return from->emitError("class '")
             << owner.getSymNameAttr().getValue()
             << "' has malformed static field schema";
    }
    auto existing = fieldIndexes.find(stringAttr.getValue());
    if (existing != fieldIndexes.end()) {
      fields[existing->second] = {stringAttr, type};
      continue;
    }
    fieldIndexes[stringAttr.getValue()] = static_cast<unsigned>(fields.size());
    fields.push_back({stringAttr, type});
  }
  return mlir::success();
}

static mlir::FailureOr<
    llvm::SmallVector<std::pair<mlir::StringAttr, mlir::TypeAttr>, 8>>
resolvedFieldSchema(mlir::Operation *from, ClassType classType) {
  mlir::FailureOr<llvm::SmallVector<llvm::StringRef, 8>> mro =
      type_object::mroNames(from, classType.getClassName());
  if (mlir::failed(mro))
    return mlir::failure();
  llvm::SmallVector<std::pair<mlir::StringAttr, mlir::TypeAttr>, 8> fields;
  llvm::StringMap<unsigned> fieldIndexes;
  for (llvm::StringRef className : llvm::reverse(*mro)) {
    ClassOp owner = type_object::lookup(from, className);
    if (!owner) {
      from->emitError("unable to resolve class '") << className << "'";
      return mlir::failure();
    }
    if (mlir::failed(appendClassFields(from, owner, fields, fieldIndexes)))
      return mlir::failure();
  }
  return fields;
}

static mlir::Type
fieldStorageType(mlir::Operation *from, mlir::Type logicalType,
                 const PyLLVMTypeConverter &typeConverter,
                 mlir::MLIRContext *ctx,
                 llvm::SmallVectorImpl<ClassType> &activeClasses,
                 llvm::SmallVectorImpl<mlir::Type> &storageParts) {
  if (auto fieldClassType = mlir::dyn_cast<ClassType>(logicalType)) {
    mlir::FailureOr<Layout> layout =
        build(from, fieldClassType, typeConverter, activeClasses);
    if (mlir::failed(layout))
      return {};
    llvm::SmallVector<mlir::Type, 2> abiParts;
    partsValueTypes(*layout, abiParts);
    storageParts.reserve(abiParts.size());
    for (mlir::Type abiPart : abiParts) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(abiPart);
      if (!memrefType)
        return {};
      mlir::Type descriptor = descriptorStorageType(memrefType, ctx);
      if (!descriptor)
        return {};
      storageParts.push_back(descriptor);
    }
    return layout->objectType;
  }

  llvm::SmallVector<mlir::Type, 4> convertedTypes;
  if (mlir::failed(typeConverter.convertType(logicalType, convertedTypes)) ||
      convertedTypes.empty())
    return {};

  if (convertedTypes.size() == 1) {
    if (auto memrefType =
            mlir::dyn_cast<mlir::MemRefType>(convertedTypes.front())) {
      mlir::Type descriptor = descriptorStorageType(memrefType, ctx);
      if (!descriptor)
        return {};
      storageParts.push_back(descriptor);
      return descriptor;
    }
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(convertedTypes.front())) {
      from->emitError("class field '")
          << logicalType
          << "' lowered to raw pointer storage; use a typed descriptor or "
             "header/payload ABI instead";
      return {};
    }
    storageParts.push_back(convertedTypes.front());
    return convertedTypes.front();
  }

  llvm::SmallVector<mlir::Type, 4> descriptorTypes;
  descriptorTypes.reserve(convertedTypes.size());
  for (mlir::Type converted : convertedTypes) {
    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(converted);
    if (!memrefType)
      return {};
    mlir::Type descriptorType = descriptorStorageType(memrefType, ctx);
    if (!descriptorType)
      return {};
    descriptorTypes.push_back(descriptorType);
  }
  storageParts.append(descriptorTypes.begin(), descriptorTypes.end());
  return mlir::LLVM::LLVMStructType::getLiteral(ctx, descriptorTypes);
}

static mlir::FailureOr<Layout>
build(mlir::Operation *from, ClassType classType,
      const PyLLVMTypeConverter &typeConverter,
      llvm::SmallVectorImpl<ClassType> &activeClasses) {
  if (!from || !classType)
    return mlir::failure();

  if (llvm::is_contained(activeClasses, classType)) {
    from->emitError("recursive class field schema for '")
        << classType.getClassName()
        << "' cannot be represented by a finite header/payload descriptor";
    return mlir::failure();
  }

  ClassOp classOp = lookup(from, classType);
  if (!classOp) {
    from->emitError("unable to resolve class '")
        << classType.getClassName() << "'";
    return mlir::failure();
  }

  activeClasses.push_back(classType);
  auto popActiveClass =
      llvm::make_scope_exit([&]() { activeClasses.pop_back(); });

  mlir::MLIRContext *ctx = from->getContext();

  Layout layout;
  auto schema = resolvedFieldSchema(from, classType);
  if (mlir::failed(schema))
    return mlir::failure();
  if (schema->empty()) {
    assignStorage(layout, ctx);
    return layout;
  }

  int64_t nextPayloadPart = 1;
  for (auto [stringAttr, type] : *schema) {
    mlir::Type logicalType = type.getValue();
    llvm::SmallVector<mlir::Type, 4> storageParts;
    mlir::Type storageType = fieldStorageType(from, logicalType, typeConverter,
                                              ctx, activeClasses, storageParts);
    if (!storageType || storageParts.empty()) {
      from->emitError("failed to convert field type ")
          << logicalType << " in class '" << classType.getClassName() << "'";
      return mlir::failure();
    }
    FieldInfo fieldInfo;
    fieldInfo.name = stringAttr;
    fieldInfo.logicalType = logicalType;
    fieldInfo.storageType = storageType;
    fieldInfo.storageParts = std::move(storageParts);
    fieldInfo.payloadPartStart = nextPayloadPart;
    fieldInfo.payloadPartCount =
        static_cast<int64_t>(fieldInfo.storageParts.size());
    nextPayloadPart += fieldInfo.payloadPartCount;
    layout.fields.push_back(std::move(fieldInfo));
  }

  assignStorage(layout, ctx);
  return layout;
}

mlir::FailureOr<Layout> get(mlir::Operation *from, ClassType classType,
                            const PyLLVMTypeConverter &typeConverter) {
  llvm::SmallVector<ClassType, 4> activeClasses;
  return build(from, classType, typeConverter, activeClasses);
}

mlir::Type carrierStorageType(mlir::ModuleOp module, ClassType classType,
                              const PyLLVMTypeConverter &typeConverter,
                              mlir::MLIRContext *ctx) {
  (void)ctx;
  if (!module)
    return {};
  llvm::SmallVector<ClassType, 4> activeClasses;
  mlir::FailureOr<Layout> layout =
      build(module.getOperation(), classType, typeConverter, activeClasses);
  if (mlir::failed(layout))
    return {};
  return layout->objectType;
}

mlir::FailureOr<std::pair<unsigned, FieldInfo>>
lookupField(mlir::Operation *from, ClassType classType,
            const PyLLVMTypeConverter &typeConverter,
            llvm::StringRef fieldName) {
  mlir::FailureOr<Layout> layout = get(from, classType, typeConverter);
  if (mlir::failed(layout))
    return mlir::failure();

  for (auto [index, fieldInfo] : llvm::enumerate(layout->fields))
    if (fieldInfo.name.getValue() == fieldName)
      return std::make_pair(static_cast<unsigned>(index), fieldInfo);

  from->emitError("class '")
      << classType.getClassName() << "' has no field '" << fieldName << "'";
  return mlir::failure();
}

} // namespace class_layout
} // namespace py
