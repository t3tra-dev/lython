#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

mlir::LogicalResult RuntimeBundleLowerer::lowerIncRef(py::IncRefOp op) {
  builder.setInsertionPoint(op);
  const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!bundle)
    return op.emitError() << "py.incref operand has no runtime bundle";
  if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
          op.getOperation(), *bundle, "py.incref")))
    return mlir::failure();
  erase.push_back(op.getOperation());
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerDecRef(py::DecRefOp op) {
  builder.setInsertionPoint(op);
  const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!bundle)
    return op.emitError() << "py.decref operand has no runtime bundle";
  if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
          op.getOperation(), *bundle, "py.decref")))
    return mlir::failure();
  erase.push_back(op.getOperation());
  return mlir::success();
}

} // namespace py::lowering
