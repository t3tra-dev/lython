#include "Runtime/Core/Lowerer.h"

namespace py::runtime_lowering {

namespace {

void appendValueSlice(mlir::ValueRange values, unsigned begin, unsigned count,
                      llvm::SmallVectorImpl<mlir::Value> &out) {
  for (unsigned index = 0; index < count; ++index)
    out.push_back(values[begin + index]);
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
        RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldTypes[index],
                                                   purpose);
    if (mlir::failed(valueTypes))
      return mlir::failure();
    offset += static_cast<unsigned>(valueTypes->size());
  }
  return offset;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrGet(py::AttrGetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";
  if (object->kind == RuntimeBundle::Kind::TypeObject) {
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

  auto fieldBundle = object->fieldBundles.find(op.getName());
  if (fieldBundle != object->fieldBundles.end()) {
    if (!fieldBundle->second)
      return op.emitError()
             << "attribute evidence for '" << op.getName() << "' is empty";
    if (!py::isAssignableTo(fieldBundle->second->objectValue.contract,
                            op.getResult().getType(), op))
      return op.emitError()
             << "attribute evidence "
             << fieldBundle->second->objectValue.contract
             << " is not assignable to result " << op.getResult().getType();
    RuntimeBundle result = *fieldBundle->second;
    result.fieldAliasOwner = op.getObject();
    result.fieldAliasName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  py::ClassOp classOp =
      RuntimeBundleLowerer::classForContract(op.getObject().getType());
  if (!classOp)
    return op.emitError() << "attr.get object type has no class schema";
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

  mlir::Type fieldType = fieldTypes[*fieldIndex];
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldType,
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
  RuntimeBundle result = RuntimeBundle::object(fieldType, values);
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

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrSet(py::AttrSetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  const RuntimeBundle *value = RuntimeBundleLowerer::bundleFor(op.getValue());
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

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldTypes[*fieldIndex],
                                                 "class field ABI");
  if (mlir::failed(fieldValueTypes))
    return mlir::failure();
  if (fieldValueTypes->size() != value->physicalValues().size())
    return op.emitError() << "attribute value ABI has "
                          << value->physicalValues().size()
                          << " values, but field expects "
                          << fieldValueTypes->size();
  mlir::FailureOr<unsigned> offset =
      RuntimeBundleLowerer::classFieldValueOffset(op, classOp, *fieldIndex,
                                                  "class field ABI");
  if (mlir::failed(offset))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> values(object->physicalValues().begin(),
                                           object->physicalValues().end());
  if (*offset + fieldValueTypes->size() > values.size())
    return op.emitError() << "class field ABI exceeds object payload";
  for (auto [index, replacement] : llvm::enumerate(value->physicalValues()))
    values[*offset + index] = replacement;

  RuntimeBundle updated;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, op.getObject().getType(), values, updated)))
    return mlir::failure();
  updated.copyEvidenceFrom(*object);
  updated.fieldBundles[op.getName()] = std::make_shared<RuntimeBundle>(*value);
  valueBundles[op.getObject()] = std::move(updated);
  erase.push_back(op);
  return mlir::success();
}
} // namespace py::runtime_lowering
