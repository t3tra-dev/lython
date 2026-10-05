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

// Every i64 storage lane of the value goes to a call that does nothing with
// it, which is a use the release placement sees: the reference is released
// after this point. A value with no such lane (an int's unboxed lane, None)
// owns nothing a finalizer could observe, and needs nothing.
// ⛔ Not the pointer word: the release placement does not see a use through
// an address taken out of the memref (Calls/Builtin.cpp, the touch call).
mlir::LogicalResult RuntimeBundleLowerer::lowerKeepAlive(py::KeepAliveOp op) {
  builder.setInsertionPoint(op);
  const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(op.getObject());
  if (!bundle)
    return op.emitError() << "py.keep_alive operand has no runtime bundle";
  std::optional<RuntimeSymbol> keepAlive =
      manifest.primitive("builtins.object", "keep_alive");
  if (!keepAlive)
    return op.emitError() << "runtime manifest has no keep_alive primitive";
  mlir::Type storageType = keepAlive->function.getFunctionType().getInput(0);
  for (mlir::Value lane : bundle->physicalValues()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(lane.getType());
    if (!memref || memref.getRank() != 1 ||
        !memref.getElementType().isInteger(64) ||
        !memref.getLayout().isIdentity())
      continue;
    mlir::Value storage = lane;
    if (storage.getType() != storageType)
      storage = mlir::memref::CastOp::create(builder, op.getLoc(), storageType,
                                             storage)
                    .getResult();
    // ⛔ Not createRuntimeCall: that marks the call as one a `try` catches
    // out of, and this one cannot raise (isNonRaisingRuntimeSymbol). Marked,
    // every keep-alive at a coroutine's exit became an edge into its handler.
    mlir::func::CallOp::create(builder, op.getLoc(), keepAlive->function,
                               mlir::ValueRange{storage});
  }
  erase.push_back(op.getOperation());
  return mlir::success();
}

} // namespace py::lowering
