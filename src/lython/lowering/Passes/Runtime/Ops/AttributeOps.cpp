#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

namespace {

constexpr unsigned kPrimitiveFieldSlotBase = 4;
constexpr unsigned kPrimitiveFieldSlotLimit = 16;

void appendValueSlice(mlir::ValueRange values, unsigned begin, unsigned count,
                      llvm::SmallVectorImpl<mlir::Value> &out) {
  for (unsigned index = 0; index < count; ++index)
    out.push_back(values[begin + index]);
}

bool isMethodDescriptorKind(py::AttrGetOp op) {
  auto kind = op->getAttrOfType<mlir::StringAttr>("ly.attr.kind");
  if (!kind)
    return false;
  llvm::StringRef value = kind.getValue();
  return value == "instance" || value == "static" || value == "class" ||
         value == "classmethod";
}

std::optional<unsigned> primitiveI64FieldSlot(mlir::Type fieldType,
                                              unsigned fieldIndex) {
  if (runtimeContractName(fieldType) != "builtins.int")
    return std::nullopt;
  unsigned slot = kPrimitiveFieldSlotBase + fieldIndex;
  if (slot >= kPrimitiveFieldSlotLimit)
    return std::nullopt;
  return slot;
}

std::optional<mlir::Attribute> classStaticValue(py::ClassOp classOp,
                                                llvm::StringRef name) {
  auto names =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_names");
  auto values =
      classOp->getAttrOfType<mlir::ArrayAttr>("class_static_attr_values");
  if (!names || !values || names.size() != values.size())
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(names)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return values[index];
  }
  return std::nullopt;
}

} // namespace

std::optional<unsigned>
RuntimeBundleLowerer::classFieldIndex(py::ClassOp classOp,
                                      llvm::StringRef name) const {
  auto fieldNames = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
  if (!fieldNames)
    return std::nullopt;
  for (auto [index, attr] : llvm::enumerate(fieldNames)) {
    auto stringAttr = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (stringAttr && stringAttr.getValue() == name)
      return static_cast<unsigned>(index);
  }
  return std::nullopt;
}

mlir::FailureOr<unsigned> RuntimeBundleLowerer::classFieldValueOffset(
    mlir::Operation *op, py::ClassOp classOp, unsigned fieldIndex,
    llvm::StringRef purpose) const {
  const RuntimeValueShape *objectShape = manifest.valueShape("builtins.object");
  if (!objectShape)
    return op->emitError()
           << "runtime manifest has no builtins.object ABI shape for "
           << purpose;
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (fieldIndex >= fieldTypes.size())
    return op->emitError() << purpose << " field index " << fieldIndex
                           << " is outside " << classOp.getSymName();

  unsigned offset = static_cast<unsigned>(objectShape->valueTypes.size());
  for (unsigned index = 0; index < fieldIndex; ++index) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::classFieldStorageValueTypes(op, fieldTypes[index],
                                                          purpose);
    if (mlir::failed(valueTypes))
      return mlir::failure();
    offset += static_cast<unsigned>(valueTypes->size());
  }
  return offset;
}

bool RuntimeBundleLowerer::classFieldStoredBoxed(
    mlir::Type fieldContract) const {
  llvm::StringRef contractName = runtimeShapeContractName(fieldContract);
  return contractName == "builtins.object" || contractName == "builtins.dict";
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::classFieldStorageValueTypes(
    mlir::Operation *op, mlir::Type fieldContract,
    llvm::StringRef purpose) const {
  if (classFieldStoredBoxed(fieldContract)) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    return llvm::SmallVector<mlir::Type, 8>(objectShape->valueTypes.begin(),
                                            objectShape->valueTypes.end());
  }
  return RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldContract, purpose);
}

mlir::LogicalResult
RuntimeBundleLowerer::writeBackFieldAlias(mlir::Operation *op,
                                          const RuntimeBundle &updatedField) {
  if (!updatedField.fieldAliasOwner || updatedField.fieldAliasName.empty())
    return mlir::success();
  auto owner = valueBundles.find(updatedField.fieldAliasOwner);
  if (owner == valueBundles.end())
    return mlir::success();

  RuntimeBundle ownerBundle = owner->second;
  RuntimeBundle storedField = updatedField.withObjectOwnership(
      ownership::logicalOwnershipKind(updatedField.objectValue.contract,
                                      /*ownsObject=*/true));
  ownerBundle.fieldBundles[updatedField.fieldAliasName] =
      std::make_shared<RuntimeBundle>(storedField);
  if (ownerBundle.kind != RuntimeBundle::Kind::Object) {
    valueBundles[updatedField.fieldAliasOwner] = std::move(ownerBundle);
    return mlir::success();
  }

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(ownerBundle.objectValue.contract);
  if (!classOp)
    return op->emitError() << "field alias owner has no class schema";
  std::optional<unsigned> fieldIndex = RuntimeBundleLowerer::classFieldIndex(
      classOp, updatedField.fieldAliasName);
  if (!fieldIndex)
    return op->emitError() << "class " << classOp.getSymName()
                           << " has no field '" << updatedField.fieldAliasName
                           << "'";
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (*fieldIndex >= fieldTypes.size())
    return op->emitError() << "class field metadata is malformed for "
                           << classOp.getSymName();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldTypes[*fieldIndex],
                                                 "field alias writeback ABI");
  if (mlir::failed(fieldValueTypes))
    return mlir::failure();
  if (fieldValueTypes->size() != updatedField.physicalValues().size())
    return op->emitError() << "field alias update has "
                           << updatedField.physicalValues().size()
                           << " physical values, but field expects "
                           << fieldValueTypes->size();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "field alias writeback ABI");
  if (mlir::failed(offset))
    return mlir::failure();
  if (*offset + fieldValueTypes->size() > ownerBundle.objectValue.values.size())
    return op->emitError() << "field alias update exceeds owner payload";
  for (auto [index, replacement] :
       llvm::enumerate(updatedField.physicalValues()))
    ownerBundle.objectValue.values[*offset + index] = replacement;

  // An owned local's release must see the updated representation (a mutation
  // may have reallocated the field's storage): re-root the owned-local marker
  // over the new value set. The old marker keeps flowing as a plain identity
  // cast; the ownership attributes move so the local roots exactly once.
  if (!ownerBundle.objectValue.values.empty()) {
    mlir::Value front = ownerBundle.objectValue.values.front();
    // The owned-local marker is a PARALLEL view: the bundle may hold the raw
    // construction values while the marker cast wraps them for the release
    // machinery. Find it as the front value's marked user (or defining op).
    mlir::UnrealizedConversionCastOp oldRoot =
        front.getDefiningOp<mlir::UnrealizedConversionCastOp>();
    if (!oldRoot || !oldRoot->hasAttr(ownership::kOwnedLocalObjectAttr)) {
      oldRoot = nullptr;
      for (mlir::Operation *user : front.getUsers()) {
        auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
        if (cast && cast->hasAttr(ownership::kOwnedLocalObjectAttr) &&
            cast.getInputs().size() == ownerBundle.objectValue.values.size() &&
            cast.getInputs().front() == front) {
          oldRoot = cast;
          break;
        }
      }
    }
    if (oldRoot && oldRoot->hasAttr(ownership::kOwnedLocalObjectAttr)) {
      builder.setInsertionPoint(op);
      llvm::SmallVector<mlir::Type, 8> resultTypes;
      for (mlir::Value value : ownerBundle.objectValue.values)
        resultTypes.push_back(value.getType());
      auto rooted = mlir::UnrealizedConversionCastOp::create(
          builder, op->getLoc(), resultTypes, ownerBundle.objectValue.values);
      rooted->setAttr(ownership::kOwnedLocalObjectAttr,
                      builder.getUnitAttr());
      if (mlir::Attribute contract =
              oldRoot->getAttr(ownership::kOwnedLocalObjectContractAttr))
        rooted->setAttr(ownership::kOwnedLocalObjectContractAttr, contract);
      oldRoot->removeAttr(ownership::kOwnedLocalObjectAttr);
      oldRoot->removeAttr(ownership::kOwnedLocalObjectContractAttr);
      ownerBundle.objectValue.values.assign(rooted.getResults().begin(),
                                            rooted.getResults().end());
    }
  }

  valueBundles[updatedField.fieldAliasOwner] = std::move(ownerBundle);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrGet(py::AttrGetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";
  if (object->kind == RuntimeBundle::Kind::TypeObject) {
    if (isMethodDescriptorKind(op) &&
        RuntimeBundleLowerer::classDefinesMethod(object->instanceContract,
                                                 op.getName())) {
      RuntimeBundle result =
          RuntimeBundle::object(op.getResult().getType(), mlir::ValueRange{});
      result.boundMethodReceiver = std::make_shared<RuntimeBundle>(*object);
      result.boundMethodName = op.getName().str();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
    if (py::ClassOp classOp =
            RuntimeBundleLowerer::classForContract(object->instanceContract)) {
      if (std::optional<mlir::Attribute> staticValue =
              classStaticValue(classOp, op.getName())) {
        auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(*staticValue);
        if (!dict)
          return op.emitError() << "static class attribute metadata for '"
                                << op.getName() << "' is malformed";
        auto kind = dict.getAs<mlir::StringAttr>("kind");
        if (!kind)
          return op.emitError() << "static class attribute '" << op.getName()
                                << "' has no metadata kind";
        llvm::StringRef spelling = kind.getValue();
        llvm::StringRef defaultKind = spelling;
        if (spelling == "constant.none")
          defaultKind = "none";
        else if (spelling == "constant.bool")
          defaultKind = "bool";
        else if (spelling == "constant.int")
          defaultKind = "int";
        else if (spelling == "constant.float")
          defaultKind = "float";
        else if (spelling == "constant.str")
          defaultKind = "str";
        else
          return op.emitError()
                 << "unsupported static class attribute expression for '"
                 << op.getName() << "'";

        llvm::SmallVector<mlir::NamedAttribute, 4> attrs;
        attrs.push_back(
            builder.getNamedAttr("kind", builder.getStringAttr(defaultKind)));
        if (mlir::Attribute value = dict.get("value"))
          attrs.push_back(builder.getNamedAttr("value", value));
        mlir::DictionaryAttr defaultValue = builder.getDictionaryAttr(attrs);

        builder.setInsertionPoint(op);
        RuntimeBundle result;
        if (mlir::failed(RuntimeBundleLowerer::materializeDefaultValue(
                op, op.getResult().getType(), defaultValue, result)))
          return mlir::failure();
        valueBundles[op.getResult()] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
    }
    mlir::LogicalResult descriptorResult =
        RuntimeBundleLowerer::lowerStaticCtypesTypeFieldDescriptorGet(op,
                                                                      *object);
    if (mlir::succeeded(descriptorResult))
      return mlir::success();
    return op.emitError()
           << "attr.get type object has no static runtime attribute '"
           << op.getName() << "'";
  }
  if (object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";

  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Library)
    return RuntimeBundleLowerer::lowerStaticCtypesAttrGet(op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Module)
    return RuntimeBundleLowerer::lowerStaticCtypesModuleAttrGet(op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::FieldDescriptor)
    return RuntimeBundleLowerer::lowerStaticCtypesFieldDescriptorAttrGet(
        op, *object);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell) {
    mlir::LogicalResult fieldResult =
        RuntimeBundleLowerer::lowerStaticCtypesFieldAttrGet(op, *object);
    if (mlir::succeeded(fieldResult))
      return mlir::success();
  }
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      op.getName() == "value")
    return RuntimeBundleLowerer::lowerStaticCtypesValueAttrGet(op, *object);

  if (object->kind == RuntimeBundle::Kind::Object &&
      runtimeContractName(op.getObject().getType()) ==
          "builtins.StopIteration" &&
      op.getName() == "value") {
    if (object->physicalValues().size() < 3)
      return op.emitError()
             << "StopIteration.value requires exception message storage";
    mlir::Type stringType = runtimeContractType(context, "builtins.str");
    RuntimeBundle result = RuntimeBundle::objectWithOwnership(
        stringType,
        mlir::ValueRange{object->physicalValues()[1],
                         object->physicalValues()[2]},
        ownership::logicalOwnershipKind(stringType, /*ownsObject=*/false));
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError() << "attribute evidence "
                            << result.objectValue.contract
                            << " is not assignable to result "
                            << op.getResult().getType();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (isMethodDescriptorKind(op) &&
      RuntimeBundleLowerer::classDefinesMethod(op.getObject().getType(),
                                               op.getName())) {
    RuntimeBundle result =
        RuntimeBundle::object(op.getResult().getType(), mlir::ValueRange{});
    result.boundMethodReceiver = std::make_shared<RuntimeBundle>(*object);
    result.boundMethodName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(op.getObject().getType());
  std::optional<unsigned> fieldIndex;
  llvm::SmallVector<mlir::Type, 8> fieldTypes;
  if (classOp) {
    fieldIndex = RuntimeBundleLowerer::classFieldIndex(classOp, op.getName());
    fieldTypes = RuntimeBundleLowerer::classFieldContractTypes(classOp);
  }
  if (fieldIndex) {
    if (*fieldIndex >= fieldTypes.size())
      return op.emitError() << "class field metadata is malformed for "
                            << classOp.getSymName();
    mlir::Type fieldType = fieldTypes[*fieldIndex];
    if (std::optional<unsigned> primitiveSlot =
            primitiveI64FieldSlot(fieldType, *fieldIndex)) {
      builder.setInsertionPoint(op);
      mlir::FailureOr<mlir::Value> header =
          RuntimeBundleLowerer::objectPhysicalHeader(op, object->objectValue);
      if (mlir::failed(header))
        return mlir::failure();
      mlir::Value slotIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(), *primitiveSlot);
      mlir::Value raw =
          mlir::memref::LoadOp::create(builder, op.getLoc(), *header, slotIndex)
              .getResult();
      mlir::Value valid =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1)
              .getResult();
      RuntimeBundle result = RuntimeBundle::objectWithOwnership(
          fieldType, mlir::ValueRange{},
          ownership::logicalOwnershipKind(fieldType,
                                          /*ownsObject=*/false));
      result.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
      if (!py::isAssignableTo(result.objectValue.contract,
                              op.getResult().getType(), op))
        return op.emitError()
               << "attribute evidence " << result.objectValue.contract
               << " is not assignable to result " << op.getResult().getType();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }

  // Box-fronted container fields: the box is the source of truth (a runtime
  // mutation may have reallocated the arrays), so compile-time field evidence
  // must not short-circuit the load — always reconstruct from the box words.
  bool boxedContainerField =
      fieldIndex && *fieldIndex < fieldTypes.size() &&
      RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[*fieldIndex]) &&
      runtimeShapeContractName(fieldTypes[*fieldIndex]) != "builtins.object";

  auto fieldBundle = object->fieldBundles.find(op.getName());
  if (!boxedContainerField && fieldBundle != object->fieldBundles.end()) {
    if (!fieldBundle->second)
      return op.emitError()
             << "attribute evidence for '" << op.getName() << "' is empty";
    RuntimeBundle result = *fieldBundle->second;
    if (result.boxedObject &&
        py::isAssignableTo(result.boxedObject->objectValue.contract,
                           op.getResult().getType(), op))
      result = *result.boxedObject;
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError()
             << "attribute evidence " << result.objectValue.contract
             << " is not assignable to result " << op.getResult().getType();
    result.setObjectLogicalOwnership(/*ownsObject=*/false);
    result.fieldAliasOwner = op.getObject();
    result.fieldAliasName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (auto unionType = mlir::dyn_cast<py::UnionType>(op.getObject().getType())) {
    if (object->physicalValues().empty())
      return op.emitError() << "union attribute input has no runtime tag";

    mlir::Type commonFieldType;
    llvm::SmallVector<mlir::Type, 8> commonValueTypes;
    llvm::SmallVector<mlir::Value, 4> selectedValues;
    mlir::Value inputTag = object->physicalValues().front();

    builder.setInsertionPoint(op);
    for (auto [memberIndex, memberType] :
         llvm::enumerate(unionType.getMemberTypes())) {
      py::ClassOp memberClass =
          RuntimeBundleLowerer::classForContract(memberType);
      if (!memberClass)
        return op.emitError() << "union member " << memberType
                              << " has no class schema for attribute '"
                              << op.getName() << "'";
      std::optional<unsigned> memberFieldIndex =
          RuntimeBundleLowerer::classFieldIndex(memberClass, op.getName());
      if (!memberFieldIndex)
        return op.emitError() << "class " << memberClass.getSymName()
                              << " has no field '" << op.getName()
                              << "' for union attribute access";
      llvm::SmallVector<mlir::Type, 8> memberFieldTypes =
          RuntimeBundleLowerer::classFieldContractTypes(memberClass);
      if (*memberFieldIndex >= memberFieldTypes.size())
        return op.emitError()
               << "class field metadata is malformed for "
               << memberClass.getSymName();
      mlir::Type memberFieldType = memberFieldTypes[*memberFieldIndex];
      if (primitiveI64FieldSlot(memberFieldType, *memberFieldIndex))
        return op.emitError()
               << "primitive union field attribute access is not supported";

      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberValueTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, memberFieldType,
                                                     "union field ABI");
      if (mlir::failed(memberValueTypes))
        return mlir::failure();
      if (!commonFieldType) {
        commonFieldType = memberFieldType;
        commonValueTypes = *memberValueTypes;
      } else if (commonFieldType != memberFieldType ||
                 commonValueTypes != *memberValueTypes) {
        return op.emitError()
               << "union field '" << op.getName()
               << "' has incompatible member field types";
      }

      mlir::FailureOr<unsigned> memberOffset =
          RuntimeBundleLowerer::unionMemberValueOffset(
              op, unionType, static_cast<unsigned>(memberIndex),
              "union field member ABI");
      if (mlir::failed(memberOffset))
        return mlir::failure();
      mlir::FailureOr<unsigned> fieldOffset =
          RuntimeBundleLowerer::classFieldValueOffset(
              op, memberClass, *memberFieldIndex, "union field ABI");
      if (mlir::failed(fieldOffset))
        return mlir::failure();
      unsigned offset = *memberOffset + *fieldOffset;
      if (offset + commonValueTypes.size() > object->physicalValues().size())
        return op.emitError() << "union field ABI exceeds object payload";

      llvm::SmallVector<mlir::Value, 4> memberValues;
      appendValueSlice(object->physicalValues(), offset,
                       static_cast<unsigned>(commonValueTypes.size()),
                       memberValues);
      if (selectedValues.empty()) {
        selectedValues = memberValues;
        continue;
      }

      mlir::Value tag = mlir::arith::ConstantIntOp::create(
          builder, op.getLoc(), static_cast<std::int64_t>(memberIndex), 64);
      mlir::Value active = mlir::arith::CmpIOp::create(
          builder, op.getLoc(), mlir::arith::CmpIPredicate::eq, inputTag, tag);
      for (auto [index, memberValue] : llvm::enumerate(memberValues))
        selectedValues[index] =
            mlir::arith::SelectOp::create(builder, op.getLoc(), active,
                                          memberValue, selectedValues[index])
                .getResult();
    }

    RuntimeBundle result = RuntimeBundle::objectWithOwnership(
        commonFieldType, selectedValues,
        ownership::logicalOwnershipKind(commonFieldType,
                                        /*ownsObject=*/false));
    if (!py::isAssignableTo(result.objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError() << "attribute evidence "
                            << result.objectValue.contract
                            << " is not assignable to result "
                            << op.getResult().getType();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (!classOp)
    return op.emitError() << "attr.get object type has no class schema";
  if (!fieldIndex)
    return op.emitError() << "class " << classOp.getSymName()
                          << " has no field '" << op.getName() << "'";
  if (*fieldIndex >= fieldTypes.size())
    return op.emitError() << "class field metadata is malformed for "
                          << classOp.getSymName();

  mlir::Type fieldType = fieldTypes[*fieldIndex];
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::classFieldStorageValueTypes(op, fieldType,
                                                        "class field ABI");
  if (mlir::failed(valueTypes))
    return mlir::failure();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "class field ABI");
  if (mlir::failed(offset))
    return mlir::failure();
  if (*offset + valueTypes->size() > object->physicalValues().size())
    return op.emitError() << "class field ABI exceeds object payload";

  llvm::SmallVector<mlir::Value, 4> values;
  appendValueSlice(object->physicalValues(), *offset,
                   static_cast<unsigned>(valueTypes->size()), values);
  if (boxedContainerField) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> arrayTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldType,
                                                   "class field ABI");
    if (mlir::failed(arrayTypes))
      return mlir::failure();
    if (values.empty())
      return op.emitError() << "box-fronted field has no box slot";
    builder.setInsertionPoint(op);
    mlir::Value box = values.front();
    llvm::SmallVector<mlir::Value, 4> rebuilt;
    for (auto [index, type] : llvm::enumerate(*arrayTypes)) {
      auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
      if (!memrefType)
        return op.emitError()
               << "box-fronted field '" << op.getName()
               << "' expects memref physical values, got " << type;
      mlir::Value ptrIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(), static_cast<std::int64_t>(4 + index));
      mlir::Value sizeIndex = mlir::arith::ConstantIndexOp::create(
          builder, op.getLoc(), static_cast<std::int64_t>(9 + index));
      mlir::Value ptrWord =
          mlir::memref::LoadOp::create(builder, op.getLoc(), box, ptrIndex)
              .getResult();
      mlir::Value sizeWord =
          mlir::memref::LoadOp::create(builder, op.getLoc(), box, sizeIndex)
              .getResult();
      rebuilt.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
          builder, op.getLoc(), ptrWord, sizeWord, memrefType));
    }
    values = std::move(rebuilt);
  }
  RuntimeBundle result = RuntimeBundle::objectWithOwnership(
      fieldType, values,
      ownership::logicalOwnershipKind(fieldType,
                                      /*ownsObject=*/false));
  result.fieldAliasOwner = op.getObject();
  result.fieldAliasName = op.getName().str();
  if (!py::isAssignableTo(result.objectValue.contract, op.getResult().getType(),
                          op))
    return op.emitError() << "attribute evidence "
                          << result.objectValue.contract
                          << " is not assignable to result "
                          << op.getResult().getType();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerClassTest(py::ClassTestOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!object || object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "class.test input has no lowered object bundle";

  mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>> targetIds =
      RuntimeBundleLowerer::runtimeClassIdsForNominalTarget(op, op.getTarget());
  if (mlir::failed(targetIds))
    return mlir::failure();

  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, object->objectValue);
  if (mlir::failed(header))
    return mlir::failure();

  mlir::Location loc = op.getLoc();
  mlir::Value storage = *header;
  mlir::Type dynamicHeaderType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI64Type());
  if (storage.getType() != dynamicHeaderType)
    storage =
        mlir::memref::CastOp::create(builder, loc, dynamicHeaderType, storage)
            .getResult();

  mlir::Value classIdSlot =
      mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value actualClassId =
      mlir::memref::LoadOp::create(builder, loc, storage, classIdSlot)
          .getResult();
  mlir::Value result = mlir::arith::ConstantIntOp::create(builder, loc, 0, 1);
  for (std::int64_t targetId : *targetIds) {
    mlir::Value expected =
        mlir::arith::ConstantIntOp::create(builder, loc, targetId, 64);
    mlir::Value match = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, actualClassId, expected);
    result = mlir::arith::OrIOp::create(builder, loc, result, match);
  }

  op.getResult().replaceAllUsesWith(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrSet(py::AttrSetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
  if (object && object->kind == RuntimeBundle::Kind::TypeObject)
    return op.emitError()
           << "class static attribute mutation is not supported; declare "
              "static attributes in the class body";
  if (!object || object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.set object has no lowered runtime bundle";
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Symbol)
    return RuntimeBundleLowerer::lowerStaticCtypesAttrSet(op, *object, value);
  if (object->ctypes &&
      object->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell) {
    mlir::LogicalResult fieldResult =
        RuntimeBundleLowerer::lowerStaticCtypesFieldAttrSet(op, *object, value);
    if (mlir::succeeded(fieldResult))
      return mlir::success();
    if (op.getName() == "value")
      return RuntimeBundleLowerer::lowerStaticCtypesValueAttrSet(op, *object,
                                                                 value);
  }
  if (!value || value->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.set value has no lowered runtime bundle";

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(op.getObject().getType());
  if (!classOp)
    return op.emitError() << "attr.set object type has no class schema";
  std::optional<unsigned> fieldIndex =
      RuntimeBundleLowerer::classFieldIndex(classOp, op.getName());
  if (!fieldIndex)
    return op.emitError() << "class " << classOp.getSymName()
                          << " has no field '" << op.getName() << "'";
  llvm::SmallVector<mlir::Type, 8> fieldTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  if (*fieldIndex >= fieldTypes.size())
    return op.emitError() << "class field metadata is malformed for "
                          << classOp.getSymName();
  if (!py::isAssignableTo(value->objectValue.contract, fieldTypes[*fieldIndex],
                          op))
    return op.emitError() << "attribute value " << value->objectValue.contract
                          << " is not assignable to field "
                          << fieldTypes[*fieldIndex];

  std::optional<unsigned> primitiveSlot =
      primitiveI64FieldSlot(fieldTypes[*fieldIndex], *fieldIndex);
  if (primitiveSlot) {
    builder.setInsertionPoint(op);
    mlir::Value primitiveRawValue;
    if (value->primitiveI64 && value->primitiveI64->value) {
      primitiveRawValue = value->primitiveI64->value;
    } else {
      std::optional<RuntimeSymbol> unbox =
          manifest.primitive(value->contractName(), "unbox.i64");
      if (!unbox)
        return op.emitError() << "attribute value " << value->contractName()
                              << " has no unbox.i64 primitive for field '"
                              << op.getName() << "'";
      llvm::SmallVector<const RuntimeBundle *, 1> unboxSources{value};
      llvm::SmallVector<mlir::Value, 4> unboxOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *unbox, unboxSources,
                                                unboxOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
          op.getLoc(), *unbox, unboxOperands);
      if (unboxCall.getNumResults() != 1 ||
          !unboxCall.getResult(0).getType().isInteger(64))
        return unbox->function.emitError()
               << "unbox.i64 primitive must return one i64";
      primitiveRawValue = unboxCall.getResult(0);
    }

    mlir::FailureOr<mlir::Value> header =
        RuntimeBundleLowerer::objectPhysicalHeader(op, object->objectValue);
    if (mlir::failed(header))
      return mlir::failure();
    mlir::Value slotIndex = mlir::arith::ConstantIndexOp::create(
        builder, op.getLoc(), *primitiveSlot);
    mlir::memref::StoreOp::create(builder, op.getLoc(), primitiveRawValue,
                                  *header, slotIndex);
    erase.push_back(op);
    return mlir::success();
  }

  bool boxedField =
      RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[*fieldIndex]);
  mlir::Type slotStorageType =
      boxedField ? runtimeContractType(context, "builtins.object")
                 : fieldTypes[*fieldIndex];
  RuntimeBundle slotValue;
  bool newBoxOwnsSlot = false;
  if (boxedField && !(value->contractName() == "builtins.object" &&
                      value->physicalValues().size() == 1)) {
    mlir::FailureOr<RuntimeBundle> boxed =
        RuntimeBundleLowerer::boxRuntimeObject(op, *value,
                                               /*retainPayload=*/true);
    if (mlir::failed(boxed))
      return mlir::failure();
    slotValue = std::move(*boxed);
    newBoxOwnsSlot = true;
  } else {
    mlir::FailureOr<RuntimeBundle> storageValue =
        RuntimeBundleLowerer::materializeObjectBundleForStorage(
            op, *value, fieldTypes[*fieldIndex], "attribute value ABI");
    if (mlir::failed(storageValue))
      return mlir::failure();
    slotValue = std::move(*storageValue);
  }

  bool retainExistingObjectHandle = false;
  if (boxedField) {
    if (slotValue.contractName() == "builtins.object" &&
        slotValue.physicalValues().size() == 1) {
      retainExistingObjectHandle = !newBoxOwnsSlot;
    } else {
      mlir::FailureOr<RuntimeBundle> boxed =
          RuntimeBundleLowerer::boxRuntimeObject(op, slotValue,
                                                 /*retainPayload=*/true);
      if (mlir::failed(boxed))
        return mlir::failure();
      slotValue = std::move(*boxed);
    }
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
      RuntimeBundleLowerer::classFieldStorageValueTypes(
          op, fieldTypes[*fieldIndex], "class field ABI");
  if (mlir::failed(fieldValueTypes))
    return mlir::failure();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "class field ABI");
  if (mlir::failed(offset))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> values(object->physicalValues().begin(),
                                           object->physicalValues().end());
  if (*offset + fieldValueTypes->size() > values.size())
    return op.emitError() << "class field ABI exceeds object payload";
  llvm::SmallVector<mlir::Value, 4> oldValues;
  appendValueSlice(values, *offset,
                   static_cast<unsigned>(fieldValueTypes->size()), oldValues);
  builder.setInsertionPoint(op);
  std::string slotName = (llvm::Twine("class.") + op.getName()).str();
  bool releaseOwnedSource = false;
  if (const RuntimeBundle *source =
          RuntimeBundleLowerer::concreteObjectForOwnership(*value)) {
    releaseOwnedSource =
        source->kind == RuntimeBundle::Kind::Object &&
        source->objectValue.ownership == ownership::OwnershipKind::Own &&
        !source->physicalValues().empty();
  }
  const RuntimeBundle *oldSlotValue = nullptr;
  auto oldFieldBundle = object->fieldBundles.find(op.getName());
  if (oldFieldBundle != object->fieldBundles.end())
    oldSlotValue = oldFieldBundle->second.get();
  if (boxedField) {
    if (retainExistingObjectHandle &&
        mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, slotStorageType, slotValue.physicalValues(), slotName)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, slotStorageType, oldValues, slotName)))
      return mlir::failure();
  } else {
    if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
            op, fieldTypes[*fieldIndex], oldValues, oldSlotValue,
            fieldTypes[*fieldIndex], slotValue, slotName,
            /*releaseMissingOldObjectSlot=*/true)))
      return mlir::failure();
  }
  if (releaseOwnedSource &&
      mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
          op, *value, llvm::Twine(slotName).concat(".source").str())))
    return mlir::failure();
  for (auto [index, replacement] : llvm::enumerate(slotValue.physicalValues()))
    values[*offset + index] = replacement;
  slotValue.setObjectLogicalOwnership(/*ownsObject=*/true);

  RuntimeBundle updated;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
          op, op.getObject().getType(), values, updated,
          object->objectValue.ownership)))
    return mlir::failure();
  updated.copyEvidenceFrom(*object);
  updated.fieldBundles[op.getName()] =
      std::make_shared<RuntimeBundle>(std::move(slotValue));
  if (mlir::failed(RuntimeBundleLowerer::markOwnedLocalObjectBundle(
          op, op.getObject(), updated)))
    return mlir::failure();
  valueBundles[op.getObject()] = std::move(updated);
  erase.push_back(op);
  return mlir::success();
}
} // namespace py::lowering
