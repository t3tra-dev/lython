#include "Runtime/Core/Lowerer.h"

namespace py::runtime_lowering {
namespace {

void clearFutureTerminalState(RuntimeBundle &future) {
  future.objectEvidence.eraseSlot(kFutureResultSlot);
  future.objectEvidence.eraseSlot(kFutureExceptionSlot);
  future.objectEvidence.eraseSlot(kFutureCancelMessageSlot);
  future.objectEvidence.eraseFlag(kFutureCancelledFlag);
}

RuntimeBundle noneLiteralBundle(mlir::MLIRContext *context) {
  return RuntimeBundle::object(py::LiteralType::get(context, "None"),
                               mlir::ValueRange{});
}

bool isStaticZeroDelay(const RuntimeBundle &delay) {
  auto literal = mlir::dyn_cast_or_null<py::LiteralType>(delay.contract);
  return literal && literal.getSpelling() == "0";
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerFutureResultEvidence(
    mlir::Operation *op, mlir::Value resultValue, const RuntimeBundle &receiver,
    llvm::StringRef label) {
  if (receiver.objectEvidence.hasFlag(kFutureCancelledFlag)) {
    RuntimeBundle messageObject;
    if (const RuntimeValue *message =
            receiver.objectEvidence.slot(kFutureCancelMessageSlot)) {
      messageObject = RuntimeBundle::object(message->contract, message->values);
    } else if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                   op, "", messageObject))) {
      return mlir::failure();
    }
    builder.setInsertionPoint(op);
    if (mlir::failed(
            RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
                op, "asyncio.CancelledError", messageObject)))
      return mlir::failure();
    mlir::FailureOr<RuntimeValue> dead = materializeDeadObjectValue(
        op, resultValue.getType(), "_asyncio.Future result cancelled");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[resultValue] =
        RuntimeBundle::object(dead->contract, dead->values);
    return mlir::success();
  }

  if (const RuntimeValue *futureException =
          receiver.objectEvidence.slot(kFutureExceptionSlot)) {
    RuntimeBundle exception = RuntimeBundle::object(futureException->contract,
                                                    futureException->values);
    std::optional<RuntimeSymbol> raise =
        manifest.primitive(exception.contractName(), "raise");
    if (!raise)
      return op->emitError() << "runtime manifest has no "
                             << exception.contractName() << ".raise primitive";
    llvm::SmallVector<const RuntimeBundle *, 1> raiseSources{&exception};
    llvm::SmallVector<mlir::Value, 8> raiseOperands;
    builder.setInsertionPoint(op);
    if (mlir::failed(emitTracebackFrame(op)))
      return mlir::failure();
    if (mlir::failed(buildRuntimeCallOperands(op, *raise, raiseSources,
                                              raiseOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *raise,
                                            raiseOperands);
    mlir::FailureOr<RuntimeValue> dead = materializeDeadObjectValue(
        op, resultValue.getType(), "_asyncio.Future result exception");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[resultValue] =
        RuntimeBundle::object(dead->contract, dead->values);
    return mlir::success();
  }

  const RuntimeValue *futureResult =
      receiver.objectEvidence.slot(kFutureResultSlot);
  if (!futureResult) {
    builder.setInsertionPoint(op);
    if (mlir::failed(emitRuntimeException(op, "builtins.RuntimeError",
                                          "Result is not set.")))
      return mlir::failure();
    mlir::FailureOr<RuntimeValue> dead = materializeDeadObjectValue(
        op, resultValue.getType(), "_asyncio.Future result pending");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[resultValue] =
        RuntimeBundle::object(dead->contract, dead->values);
    return mlir::success();
  }

  return RuntimeBundleLowerer::bindEvidenceObjectResult(op, resultValue, label,
                                                        *futureResult);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAsyncioSleepCall(py::CallOp op,
                                            const RuntimeSymbol &symbol) {
  (void)symbol;
  if (op.getNumResults() != 1)
    return op.emitError()
           << "asyncio.sleep lowering expects one coroutine result";

  llvm::SmallVector<const RuntimeBundle *, 2> positional;
  llvm::SmallVector<RuntimeBundle, 2> unpackedPositional;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "asyncio.sleep positional args",
                                              positional, &unpackedPositional)))
    return mlir::failure();
  if (positional.size() > 2)
    return op.emitError() << "asyncio.sleep accepts at most delay and result";

  const RuntimeBundle *delay = nullptr;
  const RuntimeBundle *result = nullptr;
  if (!positional.empty())
    delay = positional[0];
  if (positional.size() > 1)
    result = positional[1];

  const RuntimeBundle *kwNames =
      RuntimeBundleLowerer::bundleFor(op.getKwnames());
  const RuntimeBundle *kwValues =
      RuntimeBundleLowerer::bundleFor(op.getKwvalues());
  if (!kwNames || kwNames->kind != RuntimeBundle::Kind::Aggregate ||
      !kwValues || kwValues->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError()
           << "asyncio.sleep keyword arguments must be lowered aggregates";
  if (kwNames->aggregateOperands.size() != kwValues->aggregateOperands.size())
    return op.emitError()
           << "asyncio.sleep keyword name/value packs have different arity";

  for (auto [index, nameValue] : llvm::enumerate(kwNames->aggregateOperands)) {
    std::optional<std::string> name =
        RuntimeBundleLowerer::keywordNameFromValue(nameValue);
    if (!name)
      return op.emitError()
             << "asyncio.sleep keyword name must be a static string";
    const RuntimeBundle *value =
        RuntimeBundleLowerer::bundleFor(kwValues->aggregateOperands[index]);
    if (!value || value->kind != RuntimeBundle::Kind::Object)
      return op.emitError()
             << "asyncio.sleep keyword value must be a Python object";
    if (*name == "delay") {
      if (delay)
        return op.emitError() << "asyncio.sleep got multiple values for delay";
      delay = value;
      continue;
    }
    if (*name == "result") {
      if (result)
        return op.emitError() << "asyncio.sleep got multiple values for result";
      result = value;
      continue;
    }
    return op.emitError() << "asyncio.sleep got unexpected keyword argument '"
                          << *name << "'";
  }

  if (!delay)
    return op.emitError() << "asyncio.sleep missing required delay argument";
  if (delay->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "asyncio.sleep delay must be a Python object";

  RuntimeBundle defaultResult = noneLiteralBundle(context);
  if (!result)
    result = &defaultResult;
  if (result->kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "asyncio.sleep result must be a Python object";

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("types.CoroutineType", "__new__");
  if (!initializer)
    return op.emitError()
           << "runtime manifest has no types.CoroutineType.__new__";
  if (mlir::failed(verifySelectedRuntimeTarget(op, *initializer)))
    return mlir::failure();
  mlir::FunctionType initializerType = initializer->function.getFunctionType();
  if (initializerType.getNumInputs() != 1 ||
      !initializerType.getInput(0).isInteger(64))
    return initializer->function.emitError()
           << "types.CoroutineType.__new__ must take one i64 target id";

  builder.setInsertionPoint(op);
  mlir::Value targetId =
      builder
          .create<mlir::arith::ConstantIntOp>(
              op.getLoc(),
              RuntimeBundleLowerer::functionTargetId("asyncio.sleep"), 64)
          .getResult();
  mlir::func::CallOp newCall = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, mlir::ValueRange{targetId});

  RuntimeBundle coroutine;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, op.getResult(0).getType(), newCall, coroutine)))
    return mlir::failure();

  if (isStaticZeroDelay(*delay)) {
    coroutine.objectEvidence.setFlag(kAsyncioSleepZeroDelayFlag);
  } else {
    coroutine.objectEvidence.setFlag(kAsyncioSleepTimerPendingFlag);
    std::optional<RuntimeSymbol> getLoop =
        manifest.builtinCallable("asyncio.get_event_loop");
    if (!getLoop)
      return op.emitError()
             << "runtime manifest has no asyncio.get_event_loop builtin";
    llvm::SmallVector<mlir::Value, 1> loopOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *getLoop, {}, loopOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp loopCall = RuntimeBundleLowerer::createRuntimeCall(
        op.getLoc(), *getLoop, loopOperands);
    RuntimeBundle loop;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, runtimeContractType(context, "asyncio.AbstractEventLoop"),
            loopCall, loop)))
      return mlir::failure();

    std::optional<RuntimeSymbol> recordTimer =
        manifest.primitive("asyncio.AbstractEventLoop", "timer.record");
    if (!recordTimer)
      return op.emitError()
             << "runtime manifest has no asyncio.AbstractEventLoop."
                "timer.record primitive";
    llvm::SmallVector<const RuntimeBundle *, 1> timerSources{&loop};
    llvm::SmallVector<mlir::Value, 4> timerOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *recordTimer, timerSources,
                                              timerOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *recordTimer,
                                            timerOperands);
    coroutine.objectEvidence.setFlag(kAsyncioSleepTimerScheduledFlag);
    coroutine.objectEvidence.setSlot(kAsyncioSleepLoopSlot, loop.objectValue);
  }
  coroutine.objectEvidence.setSlot(kAsyncioSleepDelaySlot, delay->objectValue);
  coroutine.objectEvidence.setSlot(kAsyncioSleepResultSlot,
                                   result->objectValue);
  valueBundles[op.getResult(0)] = std::move(coroutine);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerFutureBoundMethod(
    py::CallOp op, RuntimeBundle &receiver, llvm::StringRef methodName) {
  llvm::SmallVector<const RuntimeBundle *, 8> sources{&receiver};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  if (methodName == "set_result") {
    if (sources.size() != 2 || !sources[1] ||
        sources[1]->kind != RuntimeBundle::Kind::Object)
      return op.emitError()
             << "_asyncio.Future.set_result requires one Python object value";
    bool wasPending = !hasFutureTerminalEvidence(receiver);
    if (mlir::failed(lowerManifestMethodResult(
            op, op.getResult(0), receiver, methodName, sources,
            /*allowUnusedSources=*/false,
            /*preferManifestObjectResult=*/true)))
      return mlir::failure();
    if (wasPending) {
      clearFutureTerminalState(receiver);
      receiver.objectEvidence.setSlot(kFutureResultSlot,
                                      sources[1]->objectValue);
    }
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "set_exception") {
    if (sources.size() != 2 || !sources[1] ||
        sources[1]->kind != RuntimeBundle::Kind::Object)
      return op.emitError()
             << "_asyncio.Future.set_exception requires one Python exception";
    bool wasPending = !hasFutureTerminalEvidence(receiver);
    if (mlir::failed(lowerManifestMethodResult(
            op, op.getResult(0), receiver, methodName, sources,
            /*allowUnusedSources=*/false,
            /*preferManifestObjectResult=*/true)))
      return mlir::failure();
    if (wasPending) {
      clearFutureTerminalState(receiver);
      receiver.objectEvidence.setSlot(kFutureExceptionSlot,
                                      sources[1]->objectValue);
    }
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "cancel") {
    if (sources.size() > 2)
      return op.emitError() << "_asyncio.Future.cancel takes at most one "
                               "cancellation message";
    if (sources.size() == 2 &&
        (!sources[1] || sources[1]->kind != RuntimeBundle::Kind::Object))
      return op.emitError()
             << "_asyncio.Future.cancel message must be a Python object value";

    llvm::SmallVector<const RuntimeBundle *, 1> runtimeSources{&receiver};
    if (mlir::failed(lowerManifestMethodResult(
            op, op.getResult(0), receiver, methodName, runtimeSources,
            /*allowUnusedSources=*/false,
            /*preferManifestObjectResult=*/true)))
      return mlir::failure();

    bool wasPending = !hasFutureTerminalEvidence(receiver);
    if (wasPending) {
      receiver.objectEvidence.setFlag(kFutureCancelledFlag);
      if (sources.size() == 2 &&
          sources[1]->contractName() != "types.NoneType") {
        if (sources[1]->contractName() != "builtins.str")
          return op.emitError()
                 << "_asyncio.Future.cancel message lowering currently "
                    "requires builtins.str or None, got "
                 << sources[1]->contractName();
        receiver.objectEvidence.setSlot(kFutureCancelMessageSlot,
                                        sources[1]->objectValue);
      }
    }
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "result") {
    if (sources.size() != 1)
      return op.emitError() << "_asyncio.Future.result takes no arguments";
    if (mlir::failed(RuntimeBundleLowerer::lowerFutureResultEvidence(
            op.getOperation(), op.getResult(0), receiver,
            "_asyncio.Future.result")))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "exception") {
    if (sources.size() != 1)
      return op.emitError() << "_asyncio.Future.exception takes no arguments";
    if (receiver.objectEvidence.hasFlag(kFutureCancelledFlag)) {
      RuntimeBundle messageObject;
      if (const RuntimeValue *message =
              receiver.objectEvidence.slot(kFutureCancelMessageSlot)) {
        messageObject =
            RuntimeBundle::object(message->contract, message->values);
      } else if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                     op, "", messageObject))) {
        return mlir::failure();
      }
      builder.setInsertionPoint(op);
      if (mlir::failed(
              RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
                  op.getOperation(), "asyncio.CancelledError", messageObject)))
        return mlir::failure();
      valueBundles[op.getResult(0)] = noneLiteralBundle(context);
      erase.push_back(op);
      return mlir::success();
    }
    const RuntimeValue *futureException =
        receiver.objectEvidence.slot(kFutureExceptionSlot);
    const RuntimeValue *futureResult =
        receiver.objectEvidence.slot(kFutureResultSlot);
    if (!futureException && !futureResult) {
      builder.setInsertionPoint(op);
      if (mlir::failed(emitRuntimeException(op.getOperation(),
                                            "builtins.RuntimeError",
                                            "Exception is not set.")))
        return mlir::failure();
      valueBundles[op.getResult(0)] = noneLiteralBundle(context);
      erase.push_back(op);
      return mlir::success();
    }
    RuntimeValue value = futureException
                             ? *futureException
                             : noneLiteralBundle(context).objectValue;
    if (mlir::failed(bindEvidenceObjectResult(
            op, op.getResult(0), "_asyncio.Future.exception", value)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(0), receiver, methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAsyncFunctionTargetCall(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("types.CoroutineType", "__new__");
  if (!initializer)
    return op.emitError()
           << "runtime manifest has no types.CoroutineType.__new__";
  if (op.getNumResults() != 1)
    return op.emitError()
           << "async function call lowering expects one coroutine result";

  mlir::FunctionType initializerType = initializer->function.getFunctionType();
  if (initializerType.getNumInputs() != 1 ||
      !initializerType.getInput(0).isInteger(64))
    return initializer->function.emitError()
           << "types.CoroutineType.__new__ must take one i64 target id";

  builder.setInsertionPoint(op);
  mlir::Value targetId =
      builder
          .create<mlir::arith::ConstantIntOp>(
              op.getLoc(), RuntimeBundleLowerer::functionTargetId(targetName),
              64)
          .getResult();
  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, mlir::ValueRange{targetId});

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, op.getResult(0).getType(), call, result)))
    return mlir::failure();
  result.coroutineTarget = target.getSymName().str();
  result.coroutineSources.reserve(sources.size());
  for (const RuntimeBundle *source : sources) {
    if (!source || source->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "async coroutine frame source for " << targetName
                            << " must be a lowered Python object bundle";
    result.coroutineSources.push_back(source->objectValue);
  }
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
