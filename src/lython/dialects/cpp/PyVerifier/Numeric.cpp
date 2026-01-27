#include "cpp/PyVerifier/Common.h"

using namespace mlir;

namespace py {

LogicalResult NumAddOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  Type resultType = getResult().getType();

  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (lhsType != resultType)
    return emitOpError("result type must match operand types");

  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");

  return success();
}

LogicalResult FloatConstantOp::verify() { return success(); }

LogicalResult NumLeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

LogicalResult NumLtOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

LogicalResult NumGtOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

LogicalResult NumGeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

LogicalResult NumEqOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

LogicalResult NumNeOp::verify() {
  Type lhsType = getLhs().getType();
  Type rhsType = getRhs().getType();
  if (lhsType != rhsType)
    return emitOpError("operand types must match");
  if (!isPyIntType(lhsType) && !isPyFloatType(lhsType))
    return emitOpError("operands must be !py.int or !py.float");
  if (!isPyBoolType(getResult().getType()))
    return emitOpError("result must be !py.bool");
  return success();
}

} // namespace py
