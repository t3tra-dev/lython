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

} // namespace py::runtime_lowering
