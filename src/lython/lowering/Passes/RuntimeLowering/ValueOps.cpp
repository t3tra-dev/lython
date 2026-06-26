#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

std::optional<unsigned>
RuntimeBundleLowerer::findUnionMemberIndex(py::UnionType unionType,
                                           mlir::Type member) const {
  for (auto [index, candidate] : llvm::enumerate(unionType.getMemberTypes()))
    if (candidate == member)
      return static_cast<unsigned>(index);

  std::optional<unsigned> found;
  for (auto [index, candidate] : llvm::enumerate(unionType.getMemberTypes())) {
    if (!py::isAssignableTo(member, candidate) &&
        !py::isAssignableTo(candidate, member))
      continue;
    if (found)
      return std::nullopt;
    found = static_cast<unsigned>(index);
  }
  return found;
}

mlir::FailureOr<unsigned> RuntimeBundleLowerer::requireUnionMemberIndex(
    mlir::Operation *op, py::UnionType unionType, mlir::Type member,
    llvm::StringRef purpose) const {
  std::optional<unsigned> index =
      RuntimeBundleLowerer::findUnionMemberIndex(unionType, member);
  if (!index)
    return op->emitError() << purpose << " cannot map " << member
                           << " into union " << unionType;
  return *index;
}

mlir::FailureOr<unsigned> RuntimeBundleLowerer::unionMemberValueOffset(
    mlir::Operation *op, py::UnionType unionType, unsigned memberIndex,
    llvm::StringRef purpose) const {
  if (memberIndex >= unionType.getMemberTypes().size())
    return op->emitError() << purpose << " member index " << memberIndex
                           << " is outside " << unionType;

  unsigned offset = 1;
  for (unsigned index = 0; index < memberIndex; ++index) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            op, unionType.getMemberTypes()[index], purpose);
    if (mlir::failed(memberTypes))
      return mlir::failure();
    offset += static_cast<unsigned>(memberTypes->size());
  }
  return offset;
}

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

mlir::LogicalResult
RuntimeBundleLowerer::lowerStrConstant(py::StrConstantOp op) {
  if (isStaticKeywordName(op)) {
    erase.push_back(op);
    return mlir::success();
  }

  builder.setInsertionPoint(op);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
          op, op.getValue(), result)))
    return mlir::failure();
  if (objectShapeMatches(runtimeContractName(op.getResult().getType()),
                         result.physicalValues()))
    result = RuntimeBundle::object(op.getResult().getType(),
                                   result.physicalValues());
  result.literalText = op.getValue().str();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isStaticKeywordName(py::StrConstantOp op) const {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses()) {
    auto pack = mlir::dyn_cast<py::PackOp>(use.getOwner());
    if (!pack)
      return false;
    bool feedsKeywordNames = false;
    for (mlir::OpOperand &packUse : pack.getResult().getUses()) {
      auto call = mlir::dyn_cast<py::CallOp>(packUse.getOwner());
      if (call && call.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
    }
    if (!feedsKeywordNames)
      return false;
  }
  return true;
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerIntConstant(py::IntConstantOp op) {
  std::int64_t parsed = 0;
  if (op.getValue().getAsInteger(10, parsed))
    return op.emitError()
           << "integer literal is outside the currently lowered i64 path";

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value value =
      builder.create<mlir::arith::ConstantIntOp>(loc, parsed, 64).getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{value}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFloatConstant(py::FloatConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value value = builder
                          .create<mlir::arith::ConstantFloatOp>(
                              op.getLoc(), op.getValue(), builder.getF64Type())
                          .getResult();
  RuntimeBundle result;
  if (mlir::failed(initializeObjectFromRawValues(
          op, op.getResult().getType(), mlir::ValueRange{value}, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBoolConstant(py::BoolConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value bit = builder
                        .create<mlir::arith::ConstantIntOp>(
                            op.getLoc(), op.getValue() ? 1 : 0, 1)
                        .getResult();
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "builtins.bool"),
          bit)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNone(py::NoneOp op) {
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

namespace {

static void appendValueSlice(mlir::ValueRange values, unsigned begin,
                             unsigned count,
                             llvm::SmallVectorImpl<mlir::Value> &out) {
  for (unsigned index = 0; index < count; ++index)
    out.push_back(values[begin + index]);
}

static bool canDeferFunctionObjectMaterialization(py::BindingRefOp op) {
  for (mlir::OpOperand &use : op.getResult().getUses()) {
    auto call = mlir::dyn_cast<py::CallOp>(use.getOwner());
    if (!call || call.getCallable() != op.getResult())
      return false;
  }
  return true;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::appendUnionRuntimeValues(
    mlir::Operation *op, py::UnionType resultUnion, const RuntimeBundle &source,
    mlir::Type sourceType, llvm::SmallVectorImpl<mlir::Value> &values) {
  if (source.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "union source must be an object bundle";

  auto sourceUnion = mlir::dyn_cast<py::UnionType>(sourceType);
  auto appendDeadMember = [&](mlir::Type member) -> mlir::LogicalResult {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, member,
                                                   "union member ABI");
    if (mlir::failed(memberTypes))
      return mlir::failure();
    for (mlir::Type type : *memberTypes) {
      mlir::FailureOr<mlir::Value> value =
          RuntimeBundleLowerer::materializeDeadPhysicalValue(op, type);
      if (mlir::failed(value))
        return mlir::failure();
      values.push_back(*value);
    }
    return mlir::success();
  };

  auto appendMappedMember =
      [&](py::UnionType sourceUnionType, unsigned sourceIndex,
          mlir::Type resultMember) -> mlir::LogicalResult {
    mlir::Type sourceMember = sourceUnionType.getMemberTypes()[sourceIndex];
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> sourceTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, sourceMember,
                                                   "source union member ABI");
    if (mlir::failed(sourceTypes))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, resultMember,
                                                   "result union member ABI");
    if (mlir::failed(resultTypes))
      return mlir::failure();
    if (!sameTypeSequence(*sourceTypes, *resultTypes))
      return op->emitError() << "union member ABI mismatch while wrapping "
                             << sourceMember << " as " << resultMember;
    mlir::FailureOr<unsigned> offset =
        RuntimeBundleLowerer::unionMemberValueOffset(
            op, sourceUnionType, sourceIndex, "source union member ABI");
    if (mlir::failed(offset))
      return mlir::failure();
    appendValueSlice(source.physicalValues(), *offset,
                     static_cast<unsigned>(sourceTypes->size()), values);
    return mlir::success();
  };

  if (sourceUnion) {
    if (source.physicalValues().empty())
      return op->emitError() << "source union has no runtime tag";
    mlir::Value sourceTag = source.physicalValues().front();
    mlir::Value remappedTag =
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, 64)
            .getResult();
    for (auto [sourceIndex, sourceMember] :
         llvm::enumerate(sourceUnion.getMemberTypes())) {
      mlir::FailureOr<unsigned> resultIndex =
          RuntimeBundleLowerer::requireUnionMemberIndex(
              op, resultUnion, sourceMember, "union injection");
      if (mlir::failed(resultIndex))
        return mlir::failure();
      mlir::Value sourceIndexValue =
          builder
              .create<mlir::arith::ConstantIntOp>(
                  op->getLoc(), static_cast<std::int64_t>(sourceIndex), 64)
              .getResult();
      mlir::Value matches = builder.create<mlir::arith::CmpIOp>(
          op->getLoc(), mlir::arith::CmpIPredicate::eq, sourceTag,
          sourceIndexValue);
      mlir::Value resultIndexValue =
          builder
              .create<mlir::arith::ConstantIntOp>(
                  op->getLoc(), static_cast<std::int64_t>(*resultIndex), 64)
              .getResult();
      remappedTag =
          builder
              .create<mlir::arith::SelectOp>(op->getLoc(), matches,
                                             resultIndexValue, remappedTag)
              .getResult();
    }
    values.push_back(remappedTag);
    for (mlir::Type resultMember : resultUnion.getMemberTypes()) {
      std::optional<unsigned> sourceIndex =
          RuntimeBundleLowerer::findUnionMemberIndex(sourceUnion, resultMember);
      if (!sourceIndex) {
        if (mlir::failed(appendDeadMember(resultMember)))
          return mlir::failure();
        continue;
      }
      if (mlir::failed(
              appendMappedMember(sourceUnion, *sourceIndex, resultMember)))
        return mlir::failure();
    }
  } else {
    mlir::FailureOr<unsigned> activeIndex =
        RuntimeBundleLowerer::requireUnionMemberIndex(
            op, resultUnion, sourceType, "union injection");
    if (mlir::failed(activeIndex))
      return mlir::failure();
    mlir::Value tag =
        builder
            .create<mlir::arith::ConstantIntOp>(
                op->getLoc(), static_cast<std::int64_t>(*activeIndex), 64)
            .getResult();
    values.push_back(tag);
    for (auto [index, resultMember] :
         llvm::enumerate(resultUnion.getMemberTypes())) {
      if (index != *activeIndex) {
        if (mlir::failed(appendDeadMember(resultMember)))
          return mlir::failure();
        continue;
      }
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, resultMember,
                                                     "union member ABI");
      if (mlir::failed(resultTypes))
        return mlir::failure();
      if (source.physicalValues().size() != resultTypes->size())
        return op->emitError()
               << "union source has " << source.physicalValues().size()
               << " physical values, but member ABI expects "
               << resultTypes->size();
      appendValueSlice(source.physicalValues(), 0,
                       static_cast<unsigned>(resultTypes->size()), values);
    }
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerUnionWrap(py::UnionWrapOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "union.wrap input has no lowered runtime bundle";

  auto resultUnion = mlir::dyn_cast<py::UnionType>(op.getResult().getType());
  if (!resultUnion)
    return op.emitError() << "union.wrap result must be a union";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 8> values;
  if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
          op, resultUnion, *input, op.getInput().getType(), values)))
    return mlir::failure();

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, op.getResult().getType(), values, result)))
    return mlir::failure();
  result.copyEvidenceFrom(*input);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerUnionTest(py::UnionTestOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "union.test input has no lowered runtime bundle";
  auto unionType = mlir::dyn_cast<py::UnionType>(op.getInput().getType());
  if (!unionType)
    return op.emitError() << "union.test input must be a union";
  mlir::FailureOr<unsigned> testedIndex =
      RuntimeBundleLowerer::requireUnionMemberIndex(
          op, unionType, op.getMember(), "union.test");
  if (mlir::failed(testedIndex))
    return mlir::failure();
  if (input->physicalValues().empty())
    return op.emitError() << "union.test input has no runtime tag";

  builder.setInsertionPoint(op);
  mlir::Value testedTag =
      builder
          .create<mlir::arith::ConstantIntOp>(
              op.getLoc(), static_cast<std::int64_t>(*testedIndex), 64)
          .getResult();
  mlir::Value bit = builder.create<mlir::arith::CmpIOp>(
      op.getLoc(), mlir::arith::CmpIPredicate::eq,
      input->physicalValues().front(), testedTag);
  op.getResult().replaceAllUsesWith(bit);
  if (mlir::failed(assignObjectBundle(
          op, bit, runtimeContractType(context, "builtins.bool"), bit)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerUnionUnwrap(py::UnionUnwrapOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "union.unwrap input has no lowered runtime bundle";
  if (input->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "union.unwrap input must be an object bundle";
  auto inputUnion = mlir::dyn_cast<py::UnionType>(op.getInput().getType());
  if (!inputUnion)
    return op.emitError() << "union.unwrap input must be a union";
  if (input->physicalValues().empty())
    return op.emitError() << "union.unwrap input has no runtime tag";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 8> values;
  auto resultUnion = mlir::dyn_cast<py::UnionType>(op.getResult().getType());
  if (resultUnion) {
    mlir::Value inputTag = input->physicalValues().front();
    mlir::Value remappedTag =
        builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64)
            .getResult();
    for (auto [resultIndex, resultMember] :
         llvm::enumerate(resultUnion.getMemberTypes())) {
      mlir::FailureOr<unsigned> inputIndex =
          RuntimeBundleLowerer::requireUnionMemberIndex(
              op, inputUnion, resultMember, "union.unwrap");
      if (mlir::failed(inputIndex))
        return mlir::failure();
      mlir::Value inputIndexValue =
          builder
              .create<mlir::arith::ConstantIntOp>(
                  op.getLoc(), static_cast<std::int64_t>(*inputIndex), 64)
              .getResult();
      mlir::Value matches = builder.create<mlir::arith::CmpIOp>(
          op.getLoc(), mlir::arith::CmpIPredicate::eq, inputTag,
          inputIndexValue);
      mlir::Value resultIndexValue =
          builder
              .create<mlir::arith::ConstantIntOp>(
                  op.getLoc(), static_cast<std::int64_t>(resultIndex), 64)
              .getResult();
      remappedTag = builder
                        .create<mlir::arith::SelectOp>(
                            op.getLoc(), matches, resultIndexValue, remappedTag)
                        .getResult();
    }
    values.push_back(remappedTag);
    for (mlir::Type resultMember : resultUnion.getMemberTypes()) {
      mlir::FailureOr<unsigned> inputIndex =
          RuntimeBundleLowerer::requireUnionMemberIndex(
              op, inputUnion, resultMember, "union.unwrap");
      if (mlir::failed(inputIndex))
        return mlir::failure();
      mlir::Type inputMember = inputUnion.getMemberTypes()[*inputIndex];
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> inputTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, inputMember,
                                                     "source union member ABI");
      if (mlir::failed(inputTypes))
        return mlir::failure();
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, resultMember,
                                                     "result union member ABI");
      if (mlir::failed(resultTypes))
        return mlir::failure();
      if (!sameTypeSequence(*inputTypes, *resultTypes))
        return op.emitError() << "union member ABI mismatch while unwrapping "
                              << inputMember << " as " << resultMember;
      mlir::FailureOr<unsigned> offset =
          RuntimeBundleLowerer::unionMemberValueOffset(
              op, inputUnion, *inputIndex, "source union member ABI");
      if (mlir::failed(offset))
        return mlir::failure();
      appendValueSlice(input->physicalValues(), *offset,
                       static_cast<unsigned>(inputTypes->size()), values);
    }
  } else {
    mlir::FailureOr<unsigned> activeIndex =
        RuntimeBundleLowerer::requireUnionMemberIndex(
            op, inputUnion, op.getResult().getType(), "union.unwrap");
    if (mlir::failed(activeIndex))
      return mlir::failure();
    mlir::Type inputMember = inputUnion.getMemberTypes()[*activeIndex];
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> inputTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, inputMember,
                                                   "source union member ABI");
    if (mlir::failed(inputTypes))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, op.getResult().getType(),
                                                   "result union member ABI");
    if (mlir::failed(resultTypes))
      return mlir::failure();
    if (!sameTypeSequence(*inputTypes, *resultTypes))
      return op.emitError()
             << "union member ABI mismatch while unwrapping " << inputMember
             << " as " << op.getResult().getType();
    mlir::FailureOr<unsigned> offset =
        RuntimeBundleLowerer::unionMemberValueOffset(
            op, inputUnion, *activeIndex, "source union member ABI");
    if (mlir::failed(offset))
      return mlir::failure();
    appendValueSlice(input->physicalValues(), *offset,
                     static_cast<unsigned>(inputTypes->size()), values);
  }

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, op.getResult().getType(), values, result)))
    return mlir::failure();
  result.copyEvidenceFrom(*input);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerCastFromPrim(py::CastFromPrimOp op) {
  std::string expected = runtimeContractName(op.getResult().getType());
  if (expected.empty())
    return op.emitError() << "primitive cast result has no runtime contract";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 1> inputValues{op.getInput()};
  if (objectShapeMatches(expected, inputValues)) {
    if (mlir::failed(assignObjectBundle(op, op.getResult(),
                                        runtimeContractType(context, expected),
                                        inputValues)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  RuntimeBundle result;
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, op.getResult().getType(), inputValues, result,
          /*emitErrors=*/false))) {
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << "primitive cast to " << expected
                        << " has no manifest-driven runtime lowering yet";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerTypeObject(py::TypeObjectOp op) {
  valueBundles[op.getResult()] = RuntimeBundle::typeObject(
      op.getResult().getType(), op.getInstanceContract());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAttrGet(py::AttrGetOp op) {
  const RuntimeBundle *object = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!object || object->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "attr.get object has no lowered runtime bundle";

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

namespace {

static bool isCallArgumentPackUse(mlir::OpOperand &use) {
  mlir::Value value = use.get();
  if (auto call = mlir::dyn_cast<py::CallOp>(use.getOwner()))
    return call.getPosargs() == value || call.getKwnames() == value ||
           call.getKwvalues() == value;
  if (auto init = mlir::dyn_cast<py::InitOp>(use.getOwner()))
    return init.getPosargs() == value || init.getKwnames() == value ||
           init.getKwvalues() == value;
  if (auto newOp = mlir::dyn_cast<py::NewOp>(use.getOwner()))
    return newOp.getPosargs() == value || newOp.getKwnames() == value ||
           newOp.getKwvalues() == value;
  return false;
}

static bool isOnlyUsedAsCallArgumentPack(py::PackOp op) {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses())
    if (!isCallArgumentPackUse(use))
      return false;
  return true;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerPack(py::PackOp op) {
  if (isOnlyUsedAsCallArgumentPack(op)) {
    RuntimeBundle bundle =
        RuntimeBundle::aggregate(op.getResult().getType(), op.getValues());
    if (auto flags =
            op->getAttrOfType<mlir::ArrayAttr>(kPackUnpackedOperandsAttr)) {
      if (flags.size() != op.getValues().size())
        return op.emitError()
               << kPackUnpackedOperandsAttr << " size must match pack operands";
      bundle.aggregateUnpackedOperands.reserve(flags.size());
      for (mlir::Attribute flag : flags) {
        auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(flag);
        if (!boolAttr)
          return op.emitError() << kPackUnpackedOperandsAttr
                                << " must contain bool attributes";
        bundle.aggregateUnpackedOperands.push_back(boolAttr.getValue());
      }
    }
    valueBundles[op.getResult()] = std::move(bundle);
    erase.push_back(op);
    return mlir::success();
  }

  std::string contractName = runtimeContractName(op.getResult().getType());
  if (contractName.empty()) {
    valueBundles[op.getResult()] =
        RuntimeBundle::aggregate(op.getResult().getType(), op.getValues());
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<RuntimeValue, 8> elements;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> elementBundles;
  llvm::SmallVector<std::string, 8> keys;
  mlir::ValueRange values = op.getValues();
  if (contractName != "builtins.dict") {
    elements.reserve(values.size());
    elementBundles.reserve(values.size());
    for (mlir::Value value : values) {
      const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(value);
      if (!bundle || bundle->kind != RuntimeBundle::Kind::Object)
        return op.emitError() << contractName
                              << " literal element has no lowered object "
                                 "bundle";
      elements.push_back(bundle->objectValue);
      elementBundles.push_back(std::make_shared<RuntimeBundle>(*bundle));
    }
  } else {
    if (values.size() % 2 != 0)
      return op.emitError() << "dict literal pack has an odd operand count";
    elements.reserve(values.size() / 2);
    keys.reserve(values.size() / 2);
    bool allStaticStringKeys = true;
    for (unsigned index = 0, end = values.size(); index < end; index += 2) {
      std::optional<std::string> key =
          RuntimeBundleLowerer::keywordNameFromValue(values[index]);
      if (!key) {
        allStaticStringKeys = false;
        break;
      }
      const RuntimeBundle *valueBundle =
          RuntimeBundleLowerer::bundleFor(values[index + 1]);
      if (!valueBundle || valueBundle->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "dict literal value has no lowered object bundle";
      keys.push_back(std::move(*key));
      elements.push_back(valueBundle->objectValue);
    }
    if (!allStaticStringKeys) {
      keys.clear();
      elements.clear();
    }
  }

  RuntimeBundle bundle;
  std::uint64_t arity =
      contractName == "builtins.dict" ? values.size() / 2 : values.size();
  if (mlir::failed(materializeArityObject(op, op.getResult().getType(), arity,
                                          bundle, elements, keys)))
    return mlir::failure();
  if (contractName != "builtins.dict")
    bundle.sequenceElementBundles.append(elementBundles.begin(),
                                         elementBundles.end());
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBindingRef(py::BindingRefOp op) {
  std::optional<RuntimeSymbol> builtin =
      manifest.builtinCallable(op.getBinding());
  if (builtin) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(), op.getBinding());
    erase.push_back(op);
    return mlir::success();
  }

  if (auto function = module.lookupSymbol<mlir::func::FuncOp>(op.getBinding()))
    return RuntimeBundleLowerer::lowerFunctionBindingRef(op, function);

  return op.emitError() << "unresolved runtime binding '" << op.getBinding()
                        << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFunctionBindingRef(py::BindingRefOp op,
                                              mlir::func::FuncOp function) {
  auto callableType = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  if (!callableType)
    return op.emitError() << "runtime binding '" << op.getBinding()
                          << "' names a func.func without callable_type";
  if (!mlir::isa<py::CallableType>(callableType.getValue()))
    return op.emitError()
           << "runtime binding '" << op.getBinding()
           << "' names a func.func whose callable_type is not Callable";

  mlir::Type functionContract =
      runtimeContractType(context, "builtins.function");
  RuntimeBundle bundle = RuntimeBundle::object(functionContract, {});
  bundle.functionTarget = function.getSymName().str();
  if (mlir::failed(appendClosureValues(op, function, bundle)))
    return mlir::failure();

  // A direct call only needs callable evidence. Emitting builtins.function here
  // would allocate a function object on every recursive/static call even though
  // the object identity is never observed.
  if (canDeferFunctionObjectMaterialization(op)) {
    valueBundles[op.getResult()] = std::move(bundle);
    erase.push_back(op);
    return mlir::success();
  }

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("builtins.function", "__new__");
  if (!initializer)
    return op.emitError()
           << "runtime manifest has no builtins.function.__new__";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.push_back(
      builder
          .create<mlir::arith::ConstantIntOp>(
              op.getLoc(),
              RuntimeBundleLowerer::functionTargetId(function.getSymName()), 64)
          .getResult());
  for (unsigned index = 0; index < 5; ++index)
    operands.push_back(
        builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64)
            .getResult());

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, operands);
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, functionContract, call.getResults(), bundle)))
    return mlir::failure();
  bundle.functionTarget = function.getSymName().str();
  if (mlir::failed(appendClosureValues(op, function, bundle)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::appendClosureValues(
    py::BindingRefOp op, mlir::func::FuncOp function, RuntimeBundle &bundle) {
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      callableClosureTypes(function);
  if (closureTypes.size() != op.getCaptures().size())
    return op.emitError() << "binding '" << op.getBinding() << "' captures "
                          << op.getCaptures().size()
                          << " values, but target declares "
                          << closureTypes.size() << " closure inputs";

  for (auto [index, capture] : llvm::enumerate(op.getCaptures())) {
    const RuntimeBundle *captureBundle =
        RuntimeBundleLowerer::bundleFor(capture);
    if (!captureBundle || captureBundle->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "closure capture " << index
                            << " must be a lowered Python object bundle";
    if (!py::isAssignableTo(captureBundle->contract, closureTypes[index],
                            op.getOperation()))
      return op.emitError()
             << "closure capture " << index << " has type "
             << captureBundle->contract << ", expected " << closureTypes[index];
    bundle.closureValues.push_back(captureBundle->objectValue);
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue) {
  const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
  if (!inputBundle)
    return op->emitError()
           << "aliasing contract view input has no lowered runtime bundle";
  valueBundles[resultValue] = *inputBundle;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectPackedObjectSources(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> *unpackedSources) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  if (unpackedSources) {
    std::size_t reserve = unpackedSources->size();
    for (auto [index, operand] : llvm::enumerate(pack->aggregateOperands)) {
      (void)operand;
      bool unpacked = index < pack->aggregateUnpackedOperands.size() &&
                      pack->aggregateUnpackedOperands[index] != 0;
      if (!unpacked)
        continue;
      const RuntimeBundle *source =
          RuntimeBundleLowerer::bundleFor(pack->aggregateOperands[index]);
      if (source)
        reserve += source->sequenceElements.size();
    }
    unpackedSources->reserve(reserve);
  }
  for (auto [index, operand] : llvm::enumerate(pack->aggregateOperands)) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(operand);
    if (!source)
      return op->emitError()
             << label << " operand has no lowered runtime bundle";
    bool unpacked = index < pack->aggregateUnpackedOperands.size() &&
                    pack->aggregateUnpackedOperands[index] != 0;
    if (unpacked) {
      if (!unpackedSources)
        return op->emitError()
               << label << " starred operand needs bundle storage";
      if (source->kind != RuntimeBundle::Kind::Object)
        return op->emitError()
               << label << " starred operand must be a Python object bundle";
      if (!source->sequenceIndices.empty())
        return op->emitError()
               << label << " starred operand has only partial sequence "
               << "evidence";
      if (source->sequenceElements.empty())
        return op->emitError()
               << label << " starred operand needs sequence evidence";
      for (const RuntimeValue &element : source->sequenceElements) {
        unpackedSources->push_back(
            RuntimeBundle::object(element.contract, element.values));
        sources.push_back(&unpackedSources->back());
      }
      continue;
    }
    sources.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectObjectSources(
    mlir::Operation *op, mlir::ValueRange values, llvm::StringRef message,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources) const {
  sources.reserve(sources.size() + values.size());
  for (mlir::Value value : values) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(value);
    if (!source)
      return op->emitError() << message;
    sources.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::requireEmptyAggregate(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  if (!pack->aggregateOperands.empty())
    return op->emitError() << label << " lowering is not keyword-aware yet";
  return mlir::success();
}

} // namespace py::runtime_lowering
