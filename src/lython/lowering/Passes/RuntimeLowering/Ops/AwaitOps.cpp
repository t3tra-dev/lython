#include "RuntimeLowering/RuntimeBundleLowerer.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::runtime_lowering {

mlir::LogicalResult RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &awaitable,
    llvm::StringRef label) {
  const RuntimeValue *sleepResult =
      awaitable.objectEvidence.slot(kAsyncioSleepResultSlot);
  if (!sleepResult)
    return op->emitError() << label << " has no result evidence";

  if (awaitable.objectEvidence.hasFlag(kAsyncioSleepZeroDelayFlag)) {
    if (mlir::failed(RuntimeBundleLowerer::bindEvidenceObjectResult(
            op, resultValue, label, *sleepResult)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  const RuntimeValue *sleepLoop =
      awaitable.objectEvidence.slot(kAsyncioSleepLoopSlot);
  if (!sleepLoop)
    return op->emitError() << label
                           << " has no event loop evidence for timer dispatch";
  if (!awaitable.objectEvidence.hasFlag(kAsyncioSleepTimerScheduledFlag))
    return op->emitError() << label << " has no scheduled timer evidence";

  RuntimeBundle loop =
      RuntimeBundle::object(sleepLoop->contract, sleepLoop->values);
  std::optional<RuntimeSymbol> dispatchDue =
      manifest.primitive("asyncio.AbstractEventLoop", "timer.dispatch_due");
  if (!dispatchDue)
    return op->emitError()
           << "runtime manifest has no asyncio.AbstractEventLoop."
              "timer.dispatch_due primitive";

  llvm::SmallVector<const RuntimeBundle *, 1> timerSources{&loop};
  llvm::SmallVector<mlir::Value, 4> timerOperands;
  builder.setInsertionPoint(op);
  if (mlir::failed(buildRuntimeCallOperands(op, *dispatchDue, timerSources,
                                            timerOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();

  mlir::func::CallOp readyCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *dispatchDue, timerOperands);
  if (readyCall.getNumResults() != 1 ||
      !readyCall.getResult(0).getType().isInteger(1))
    return dispatchDue->function.emitError()
           << "asyncio.AbstractEventLoop.timer.dispatch_due primitive must "
              "return one i1";

  llvm::SmallVector<RuntimeValue, 1> candidates{*sleepResult};
  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, resultValue, candidates, mlir::ValueRange{readyCall.getResult(0)},
          label, "builtins.RuntimeError", "asyncio.sleep timer is not ready");
  if (mlir::failed(selected))
    return mlir::failure();

  if (awaitable.contractName() == "_asyncio.Task")
    awaitable.objectEvidence.setSlot(kFutureResultSlot, *sleepResult);

  return RuntimeBundleLowerer::bindSelectedEvidenceObjectResult(
      op, resultValue, std::move(*selected));
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAwait(py::AwaitOp op) {
  auto awaitableIt = valueBundles.find(op.getAwaitable());
  if (awaitableIt == valueBundles.end())
    return op.emitError() << "awaitable has no lowered runtime bundle";
  RuntimeBundle &awaitable = awaitableIt->second;

  if (awaitable.contractName() == "types.CoroutineType") {
    if (awaitable.objectEvidence.hasFlag(kCoroutineAwaitConsumedFlag))
      return op.emitError()
             << "cannot await an already awaited coroutine object";
    awaitable.objectEvidence.setFlag(kCoroutineAwaitConsumedFlag);
  }

  if (awaitable.contractName() == "types.CoroutineType" &&
      hasAsyncioSleepEvidence(awaitable))
    return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
        op.getOperation(), op.getResult(), awaitable, "await asyncio.sleep");

  if ((awaitable.contractName() == "_asyncio.Future" ||
       awaitable.contractName() == "_asyncio.Task") &&
      hasFutureTerminalEvidence(awaitable)) {
    if (mlir::failed(RuntimeBundleLowerer::lowerFutureResultEvidence(
            op.getOperation(), op.getResult(), awaitable,
            awaitable.contractName() == "_asyncio.Task"
                ? "await _asyncio.Task"
                : "await _asyncio.Future")))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (awaitable.contractName() == "_asyncio.Task" &&
      hasAsyncioSleepEvidence(awaitable))
    return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
        op.getOperation(), op.getResult(), awaitable,
        "await _asyncio.Task sleep");

  if (awaitable.coroutineTarget.empty())
    return op.emitError()
           << "general Awaitable iterator driving is not implemented yet";

  mlir::func::FuncOp target =
      module.lookupSymbol<mlir::func::FuncOp>(awaitable.coroutineTarget);
  if (!target)
    return op.emitError() << "coroutine body target '"
                          << awaitable.coroutineTarget << "' is not defined";

  std::optional<RuntimeSymbol> resumeBegin =
      manifest.primitive(awaitable.contractName(), "resume.begin");
  std::optional<RuntimeSymbol> resumeComplete =
      manifest.primitive(awaitable.contractName(), "resume.complete");
  if (!resumeBegin || !resumeComplete)
    return op.emitError() << "runtime manifest has no coroutine resume "
                             "primitive for "
                          << awaitable.contractName();

  llvm::SmallVector<const RuntimeBundle *, 1> coroutineSource{&awaitable};
  mlir::FunctionType functionType = target.getFunctionType();
  RuntimeSymbol targetSymbol;
  targetSymbol.function = target;
  targetSymbol.contract = "types.CoroutineType";
  targetSymbol.role = "primitive";
  targetSymbol.name = awaitable.coroutineTarget;

  context->loadDialect<mlir::async::AsyncDialect>();
  llvm::SmallVector<mlir::Type, 4> asyncResultTypes;
  for (mlir::Type resultType : functionType.getResults())
    asyncResultTypes.push_back(resultType);

  std::optional<std::int64_t> tryHandlerId = currentTryHandlerId();
  mlir::LogicalResult bodyStatus = mlir::success();
  builder.setInsertionPoint(op);
  mlir::async::ExecuteOp execute = builder.create<mlir::async::ExecuteOp>(
      op.getLoc(), asyncResultTypes, mlir::ValueRange{}, mlir::ValueRange{},
      [&](mlir::OpBuilder &bodyBuilder, mlir::Location loc, mlir::ValueRange) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.restoreInsertionPoint(bodyBuilder.saveInsertionPoint());
        if (tryHandlerId)
          tryHandlerIds[builder.getInsertionBlock()] = *tryHandlerId;

        mlir::async::CoroIdOp coroId =
            builder.create<mlir::async::CoroIdOp>(loc);
        mlir::async::CoroBeginOp coroHandle =
            builder.create<mlir::async::CoroBeginOp>(loc, coroId.getResult());
        builder.create<mlir::async::CoroSaveOp>(loc, coroHandle.getResult());

        llvm::SmallVector<mlir::Value, 4> resumeBeginOperands;
        if (mlir::failed(buildRuntimeCallOperands(
                op, *resumeBegin, coroutineSource, resumeBeginOperands,
                /*allowUnusedSources=*/false))) {
          bodyStatus = mlir::failure();
          return;
        }
        mlir::func::CallOp resumeBeginCall =
            RuntimeBundleLowerer::createRuntimeCall(loc, *resumeBegin,
                                                    resumeBeginOperands);
        if (resumeBeginCall.getNumResults() != 1 ||
            !resumeBeginCall.getResult(0).getType().isInteger(1)) {
          resumeBegin->function.emitError()
              << "coroutine resume.begin primitive must return one i1";
          bodyStatus = mlir::failure();
          return;
        }
        auto resumed = builder.create<mlir::scf::IfOp>(
            loc, functionType.getResults(), resumeBeginCall.getResult(0),
            /*withElseRegion=*/true);
        builder.setInsertionPointToStart(&resumed.getThenRegion().front());

        llvm::SmallVector<RuntimeBundle, 8> sourceBundles;
        sourceBundles.reserve(awaitable.coroutineSources.size());
        llvm::SmallVector<mlir::Value, 8> operands;
        unsigned inputIndex = 0;
        for (const RuntimeValue &sourceValue : awaitable.coroutineSources) {
          RuntimeBundle &source = sourceBundles.emplace_back(
              RuntimeBundle::object(sourceValue.contract, sourceValue.values));
          if (inputIndex >= functionType.getNumInputs()) {
            op.emitError() << "too many coroutine frame sources for "
                           << awaitable.coroutineTarget;
            bodyStatus = mlir::failure();
            return;
          }
          if (mlir::failed(appendRuntimeSource(op, targetSymbol, functionType,
                                               inputIndex, source, operands))) {
            bodyStatus = mlir::failure();
            return;
          }
        }
        if (inputIndex != functionType.getNumInputs()) {
          op.emitError() << "missing coroutine frame sources for "
                         << awaitable.coroutineTarget;
          bodyStatus = mlir::failure();
          return;
        }

        mlir::func::CallOp bodyCall = RuntimeBundleLowerer::createRuntimeCall(
            loc, targetSymbol, operands);

        llvm::SmallVector<mlir::Value, 4> resumeCompleteOperands;
        if (mlir::failed(buildRuntimeCallOperands(
                op, *resumeComplete, coroutineSource, resumeCompleteOperands,
                /*allowUnusedSources=*/false))) {
          bodyStatus = mlir::failure();
          return;
        }
        RuntimeBundleLowerer::createRuntimeCall(loc, *resumeComplete,
                                                resumeCompleteOperands);
        builder.create<mlir::scf::YieldOp>(loc, bodyCall.getResults());

        builder.setInsertionPointToStart(&resumed.getElseRegion().front());
        if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
                op.getOperation(), "builtins.RuntimeError",
                "cannot await a coroutine that is already running or "
                "completed"))) {
          bodyStatus = mlir::failure();
          return;
        }
        llvm::SmallVector<mlir::Value, 4> deadValues;
        deadValues.reserve(functionType.getNumResults());
        for (mlir::Type resultType : functionType.getResults()) {
          mlir::FailureOr<mlir::Value> dead =
              RuntimeBundleLowerer::materializeDeadPhysicalValue(
                  op.getOperation(), resultType);
          if (mlir::failed(dead)) {
            bodyStatus = mlir::failure();
            return;
          }
          deadValues.push_back(*dead);
        }
        builder.create<mlir::scf::YieldOp>(loc, deadValues);

        builder.setInsertionPointAfter(resumed);
        builder.create<mlir::async::CoroEndOp>(loc, coroHandle.getResult());
        builder.create<mlir::async::YieldOp>(loc, resumed.getResults());
      });
  execute->setAttr("ly.async.python_await", builder.getUnitAttr());
  if (mlir::failed(bodyStatus))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> awaitedValues;
  for (mlir::Value asyncValue : execute.getBodyResults()) {
    auto asyncType = mlir::cast<mlir::async::ValueType>(asyncValue.getType());
    mlir::OperationState awaitState(op.getLoc(),
                                    mlir::async::AwaitOp::getOperationName());
    awaitState.addTypes(asyncType.getValueType());
    awaitState.addOperands(asyncValue);
    auto awaitValue =
        mlir::cast<mlir::async::AwaitOp>(builder.create(awaitState));
    awaitedValues.push_back(awaitValue.getResult());
  }

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, op.getResult().getType(), awaitedValues, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
