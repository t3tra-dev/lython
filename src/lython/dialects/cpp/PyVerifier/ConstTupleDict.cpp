#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

LogicalResult StrConstantOp::verify() {
  if (!getValueAttr())
    return emitOpError("requires 'value' attribute");
  if (!isPyStrType(getResult().getType()))
    return emitOpError("result must be of type !py.str");
  return success();
}

LogicalResult TupleEmptyOp::verify() {
  auto tupleTy = dyn_cast<TupleType>(getResult().getType());
  if (!tupleTy)
    return emitOpError("result must be a !py.tuple type");
  if (!tupleTy.getElementTypes().empty())
    return emitOpError(
        "result type must encode no element types for tuple.empty");
  return success();
}

LogicalResult TupleCreateOp::verify() {
  auto resultTy = dyn_cast<TupleType>(getResult().getType());
  if (!resultTy)
    return emitOpError("result must be a !py.tuple type");

  auto operands = getElements();
  auto elementTypes = resultTy.getElementTypes();

  if (elementTypes.empty()) {
    if (!operands.empty())
      return emitOpError("cannot populate an empty tuple with elements");
    return success();
  }

  if (elementTypes.size() == 1) {
    Type target = elementTypes.front();
    for (Value operand : operands)
      if (!isSubtypeOf(operand.getType(), target))
        return emitOpError("element type ")
               << operand.getType()
               << " is not compatible with tuple element type " << target;
    return success();
  }

  if (operands.size() != elementTypes.size())
    return emitOpError("number of operands must match tuple arity");

  for (auto [value, target] : llvm::zip(operands, elementTypes))
    if (!isSubtypeOf(value.getType(), target))
      return emitOpError("element type ")
             << value.getType() << " is not compatible with tuple element type "
             << target;

  return success();
}

LogicalResult DictInsertOp::verify() {
  auto dictTy = dyn_cast<DictType>(getDict().getType());
  if (!dictTy)
    return emitOpError("dict operand must be !py.dict");

  if (getResult().getType() != getDict().getType())
    return emitOpError("result type must match dictionary operand type");

  if (!isSubtypeOf(getKey().getType(), dictTy.getKeyType()))
    return emitOpError("key type ")
           << getKey().getType()
           << " is not compatible with dictionary key type "
           << dictTy.getKeyType();

  if (!isSubtypeOf(getValue().getType(), dictTy.getValueType()))
    return emitOpError("value type ")
           << getValue().getType()
           << " is not compatible with dictionary value type "
           << dictTy.getValueType();

  return success();
}

} // namespace py
