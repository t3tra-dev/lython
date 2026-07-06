#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

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
      auto init = mlir::dyn_cast<py::InitOp>(packUse.getOwner());
      if (init && init.getKwnames() == pack.getResult()) {
        feedsKeywordNames = true;
        break;
      }
      auto newOp = mlir::dyn_cast<py::NewOp>(packUse.getOwner());
      if (newOp && newOp.getKwnames() == pack.getResult()) {
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
      mlir::arith::ConstantIntOp::create(builder, loc, parsed, 64).getResult();
  mlir::Value valid =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult();
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult().getType(), value, valid, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFloatConstant(py::FloatConstantOp op) {
  builder.setInsertionPoint(op);
  mlir::Value value =
      mlir::arith::ConstantFloatOp::create(builder, op.getLoc(),
                                           builder.getF64Type(), op.getValue())
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
  mlir::Value bit = mlir::arith::ConstantIntOp::create(builder, op.getLoc(),
                                                       op.getValue() ? 1 : 0, 1)
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

mlir::LogicalResult
RuntimeBundleLowerer::lowerCastFromPrim(py::CastFromPrimOp op) {
  std::string expected = runtimeContractName(op.getResult().getType());
  if (expected.empty())
    return op.emitError() << "primitive cast result has no runtime contract";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 1> inputValues{op.getInput()};
  if (expected == "builtins.int") {
    if (auto integer =
            mlir::dyn_cast<mlir::IntegerType>(op.getInput().getType())) {
      mlir::Value value = op.getInput();
      mlir::Type i64 = builder.getI64Type();
      if (integer.getWidth() < 64) {
        if (integer.getWidth() == 1)
          value = mlir::arith::ExtUIOp::create(builder, op.getLoc(), i64, value)
                      .getResult();
        else
          value = mlir::arith::ExtSIOp::create(builder, op.getLoc(), i64, value)
                      .getResult();
      } else if (integer.getWidth() > 64) {
        value = mlir::arith::TruncIOp::create(builder, op.getLoc(), i64, value)
                    .getResult();
      }
      mlir::Value valid =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1)
              .getResult();
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
              op, op.getResult().getType(), value, valid, result)))
        return mlir::failure();
      valueBundles[op.getResult()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
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
} // namespace py::lowering
