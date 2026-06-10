#include "cpp/PyVerifier/Common.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include <optional>

namespace py {
namespace {

std::optional<int64_t> constantIndex(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}

} // namespace

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
      if (!isSubtypeOf(operand.getType(), target))
        return emitOpError("element type ")
               << operand.getType()
               << " is not compatible with tuple element type " << target;
    return mlir::success();
  }

  if (operands.size() != elementTypes.size())
    return emitOpError("number of operands must match tuple arity");

  for (auto [value, target] : llvm::zip(operands, elementTypes))
    if (!isSubtypeOf(value.getType(), target))
      return emitOpError("element type ")
             << value.getType() << " is not compatible with tuple element type "
             << target;

  return mlir::success();
}

mlir::LogicalResult TupleGetOp::verify() {
  auto tupleTy = mlir::dyn_cast<TupleType>(getTuple().getType());
  if (!tupleTy)
    return emitOpError("tuple operand must be a !py.tuple value");
  if (!getIndex().getType().isIndex())
    return emitOpError("index operand must be index type");

  auto elementTypes = tupleTy.getElementTypes();
  if (elementTypes.empty())
    return emitOpError("cannot index an empty tuple type");

  std::optional<int64_t> index = constantIndex(getIndex());
  if (!index)
    return emitOpError("tuple index must be a static index constant");
  if (*index < 0)
    return emitOpError("tuple index must be non-negative");

  if (auto create = getTuple().getDefiningOp<TupleCreateOp>()) {
    if (*index >= static_cast<int64_t>(create.getElements().size()))
      return emitOpError("tuple index is out of bounds for tuple.create");
  }
  if (getTuple().getDefiningOp<TupleEmptyOp>())
    return emitOpError("cannot index tuple.empty");

  mlir::Type expected = elementTypes.front();
  if (elementTypes.size() > 1) {
    if (*index >= static_cast<int64_t>(elementTypes.size()))
      return emitOpError("tuple index is out of bounds for tuple type");
    expected = elementTypes[*index];
  }
  if (getResult().getType() != expected)
    return emitOpError("result type ")
           << getResult().getType() << " does not match tuple element type "
           << expected;

  return mlir::success();
}

mlir::LogicalResult DictInsertOp::verify() {
  auto dictTy = mlir::dyn_cast<DictType>(getDict().getType());
  if (!dictTy)
    return emitOpError("dict operand must be !py.dict");

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

  return mlir::success();
}

} // namespace py
