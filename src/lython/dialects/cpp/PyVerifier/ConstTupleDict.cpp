#include "cpp/PyVerifier/Common.h"

namespace py {

mlir::LogicalResult StrConstantOp::verify() {
  if (!getValueAttr())
    return emitOpError("requires 'value' attribute");
  if (!isPyStrType(getResult().getType()))
    return emitOpError("result must be of type !py.str");
  return mlir::success();
}

mlir::LogicalResult TupleEmptyOp::verify() {
  auto tupleTy = mlir::dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a !py.tuple type");
  if (!tupleTy.getElementTypes().empty())
    return emitOpError(
        "result type must encode no element types for tuple.empty");
  return mlir::success();
}

mlir::LogicalResult TupleCreateOp::verify() {
  auto resultTy = mlir::dyn_cast<TupleType>(getResult().getType());
  if (!resultTy)
    return emitOpError("result must be a !py.tuple type");

  auto operands = getElements();
  auto elementTypes = resultTy.getElementTypes();

  if (elementTypes.empty()) {
    if (!operands.empty())
      return emitOpError("cannot populate an empty tuple with elements");
    return mlir::success();
  }

  if (elementTypes.size() == 1) {
    mlir::Type target = elementTypes.front();
    for (mlir::Value operand : operands)
      if (!isSubtypeOf(operand.getType(), target, getOperation()))
        return emitOpError("element type ")
               << operand.getType()
               << " is not compatible with tuple element type " << target;
    return mlir::success();
  }

  if (operands.size() != elementTypes.size())
    return emitOpError("number of operands must match tuple arity");

  for (auto [value, target] : llvm::zip(operands, elementTypes))
    if (!isSubtypeOf(value.getType(), target, getOperation()))
      return emitOpError("element type ")
             << value.getType() << " is not compatible with tuple element type "
             << target;

  return mlir::success();
}

mlir::LogicalResult GetItemOp::verify() {
  return verifyUnaryMethodContract(
      getOperation(), getCalleeType(), getContainer().getType(),
      getIndex().getType(), getResult().getType(), "__getitem__");
}

mlir::LogicalResult DictInsertOp::verify() {
  auto dictTy = mlir::dyn_cast<DictType>(getDict().getType());
  if (!dictTy)
    return emitOpError("dict operand must be !py.dict");

  if (!isSubtypeOf(getKey().getType(), dictTy.getKeyType(), getOperation()))
    return emitOpError("key type ")
           << getKey().getType()
           << " is not compatible with dictionary key type "
           << dictTy.getKeyType();

  if (!isSubtypeOf(getValue().getType(), dictTy.getValueType(), getOperation()))
    return emitOpError("value type ")
           << getValue().getType()
           << " is not compatible with dictionary value type "
           << dictTy.getValueType();

  return mlir::success();
}

} // namespace py
