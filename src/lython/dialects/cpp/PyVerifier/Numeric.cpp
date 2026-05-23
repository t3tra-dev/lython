#include "cpp/PyVerifier/Common.h"

namespace py {

static mlir::LogicalResult
verifyPythonOperands(mlir::Operation *op, mlir::Value lhs, mlir::Value rhs) {
  mlir::Type lhsType = lhs.getType();
  mlir::Type rhsType = rhs.getType();

  if (!isPyType(lhsType))
    return op->emitOpError("lhs must be a !py.* type");
  if (!isPyType(rhsType))
    return op->emitOpError("rhs must be a !py.* type");
  return mlir::success();
}

static mlir::LogicalResult verifyPythonResult(mlir::Operation *op,
                                              mlir::Type resultType) {
  if (!isPyType(resultType))
    return op->emitOpError("result must be a !py.* type");
  return mlir::success();
}

mlir::LogicalResult AddOp::verify() {
  if (mlir::failed(verifyPythonOperands(getOperation(), getLhs(), getRhs())))
    return mlir::failure();
  return verifyPythonResult(getOperation(), getResult().getType());
}

mlir::LogicalResult SubOp::verify() {
  if (mlir::failed(verifyPythonOperands(getOperation(), getLhs(), getRhs())))
    return mlir::failure();
  return verifyPythonResult(getOperation(), getResult().getType());
}

mlir::LogicalResult FloatConstantOp::verify() { return mlir::success(); }

static mlir::LogicalResult verifyPythonComparison(mlir::Operation *op,
                                                  mlir::Value lhs,
                                                  mlir::Value rhs,
                                                  mlir::Type resultType) {
  if (mlir::failed(verifyPythonOperands(op, lhs, rhs)))
    return mlir::failure();
  if (!isPyBoolType(resultType))
    return op->emitOpError("result must be !py.bool");
  return mlir::success();
}

mlir::LogicalResult LeOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult LtOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult GtOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult GeOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult EqOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult NeOp::verify() {
  return verifyPythonComparison(getOperation(), getLhs(), getRhs(),
                                getResult().getType());
}

mlir::LogicalResult ReprOp::verify() {
  if (!isPyType(getInput().getType()))
    return emitOpError("input must be a !py.* type");
  if (!isPyStrType(getResult().getType()))
    return emitOpError("result must be !py.str");
  return mlir::success();
}

} // namespace py
