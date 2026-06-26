#include "RuntimeLowering/RuntimeLowering.h"

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

bool sameRuntimeValueIdentity(const RuntimeValue &lhs,
                              const RuntimeValue &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.values, rhs.values))
    if (left != right)
      return false;
  return true;
}

bool isStructuralSequenceObject(const RuntimeBundle &bundle) {
  return bundle.contractName() == "builtins.list" ||
         bundle.contractName() == "builtins.tuple" ||
         !bundle.sequenceElements.empty();
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

mlir::LogicalResult RuntimeBundleLowerer::lowerCall(py::CallOp op) {
  const RuntimeBundle *callable =
      RuntimeBundleLowerer::bundleFor(op.getCallable());
  if (!callable)
    return op.emitError() << "callable has no lowered runtime bundle";
  if (auto method = op->getAttrOfType<mlir::StringAttr>("ly.bound_method"))
    return RuntimeBundleLowerer::lowerBoundMethodCall(op, *callable,
                                                      method.getValue());
  if (callable->kind != RuntimeBundle::Kind::BuiltinCallable)
    return RuntimeBundleLowerer::lowerObjectCallableCall(op, *callable);
  std::optional<RuntimeSymbol> builtin =
      manifest.builtinCallable(callable->binding);
  if (!builtin)
    return op.emitError() << "runtime manifest has no builtin callable '"
                          << callable->binding << "'";
  if (builtin->builtinLowering == "method")
    return RuntimeBundleLowerer::lowerBuiltinMethodCall(op, *builtin);
  if (builtin->builtinLowering == "method_sink")
    return RuntimeBundleLowerer::lowerBuiltinMethodSinkCall(op, *builtin);
  if (builtin->builtinLowering == "direct")
    return RuntimeBundleLowerer::lowerDirectBuiltinCall(op, *builtin);
  if (builtin->builtinLowering == "asyncio_sleep")
    return RuntimeBundleLowerer::lowerAsyncioSleepCall(op, *builtin);
  return op.emitError() << "builtin callable '" << callable->binding
                        << "' has unsupported lowering strategy '"
                        << builtin->builtinLowering << "'";
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

mlir::LogicalResult RuntimeBundleLowerer::lowerBoundMethodCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  if (receiver.kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "bound method receiver must be an object bundle";
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python bound method lowering expects exactly one Python result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  auto receiverIt = valueBundles.find(op.getCallable());
  if (receiverIt != valueBundles.end() &&
      receiverIt->second.contractName() == "_asyncio.Future")
    return RuntimeBundleLowerer::lowerFutureBoundMethod(op, receiverIt->second,
                                                        methodName);

  llvm::SmallVector<const RuntimeBundle *, 8> sources{&receiver};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  if (isStructuralSequenceObject(receiver) &&
      (methodName == "append" || methodName == "remove")) {
    if (sources.size() != 2 || !sources[1] ||
        sources[1]->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "sequence " << methodName
                            << " expects one Python object argument";

    RuntimeBundle updated = receiver;
    if (methodName == "append") {
      updated.sequenceElements.push_back(sources[1]->objectValue);
      updated.sequenceElementBundles.push_back(
          std::make_shared<RuntimeBundle>(*sources[1]));
    } else {
      auto found = llvm::find_if(updated.sequenceElements, [&](const auto &it) {
        return sameRuntimeValueIdentity(it, sources[1]->objectValue);
      });
      if (found == updated.sequenceElements.end()) {
        builder.setInsertionPoint(op);
        if (mlir::failed(emitRuntimeException(op, "builtins.ValueError",
                                              "list.remove(x): x not in list")))
          return mlir::failure();
      } else {
        unsigned index = static_cast<unsigned>(
            std::distance(updated.sequenceElements.begin(), found));
        updated.sequenceElements.erase(found);
        if (index < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles.erase(
              updated.sequenceElementBundles.begin() + index);
      }
    }

    valueBundles[op.getCallable()] = updated;
    if (updated.fieldAliasOwner && !updated.fieldAliasName.empty()) {
      auto owner = valueBundles.find(updated.fieldAliasOwner);
      if (owner != valueBundles.end()) {
        RuntimeBundle ownerBundle = owner->second;
        ownerBundle.fieldBundles[updated.fieldAliasName] =
            std::make_shared<RuntimeBundle>(updated);
        valueBundles[updated.fieldAliasOwner] = std::move(ownerBundle);
      }
    }
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(0), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "__repr__" && sources.size() == 1 &&
      RuntimeBundleLowerer::needsDefaultObjectRepr(receiver)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
            op, receiver, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
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

mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectCallableCall(py::CallOp op,
                                              const RuntimeBundle &callable) {
  if (callable.kind != RuntimeBundle::Kind::Object)
    return op.emitError()
           << "callable is not an object bundle with a __call__ contract";
  if (!callable.functionTarget.empty())
    return RuntimeBundleLowerer::lowerFunctionTargetCall(op, callable);
  if (runtimeContractName(callable.contract) == "builtins.function")
    return RuntimeBundleLowerer::lowerIndirectFunctionObjectCall(op, callable);
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python callable lowering expects exactly one Python result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources{&callable};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(0), callable, "__call__", sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectFunctionTargetRuntimeSources(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    const RuntimeBundle &callable,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
    llvm::SmallVectorImpl<RuntimeBundle> &closureSources,
    llvm::SmallVectorImpl<RuntimeBundle> &argumentEvidenceSources,
    llvm::SmallVectorImpl<RuntimeBundle> &aggregateEvidenceSources) {
  if (mlir::failed(collectFunctionCallSources(op, target, targetName, sources,
                                              materializedDefaults)))
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 4> closureTypes =
      RuntimeBundleLowerer::callableClosureTypes(target);
  if (closureTypes.size() != callable.closureValues.size())
    return op.emitError() << "function target " << targetName << " requires "
                          << closureTypes.size()
                          << " closure values, but callable object carries "
                          << callable.closureValues.size();

  closureSources.reserve(callable.closureValues.size());
  for (auto [index, closureValue] : llvm::enumerate(callable.closureValues)) {
    mlir::Type expected = closureTypes[index];
    if (!py::isAssignableTo(closureValue.contract, expected, op.getOperation()))
      return op.emitError()
             << "function target " << targetName << " closure " << index
             << " has contract " << closureValue.contract << ", expected "
             << expected;
    closureSources.push_back(
        RuntimeBundle::object(closureValue.contract, closureValue.values));
    sources.push_back(&closureSources.back());
  }

  auto argumentEvidence = callableArgumentEvidenceABIs.find(targetName);
  if (argumentEvidence != callableArgumentEvidenceABIs.end() &&
      mlir::failed(RuntimeBundleLowerer::appendCallableArgumentEvidenceSources(
          op, targetName, argumentEvidence->second, sources,
          argumentEvidenceSources)))
    return mlir::failure();

  auto aggregateEvidence = callableAggregateEvidenceABIs.find(targetName);
  if (aggregateEvidence != callableAggregateEvidenceABIs.end() &&
      mlir::failed(RuntimeBundleLowerer::appendCallableAggregateEvidenceSources(
          op, targetName, aggregateEvidence->second, sources,
          aggregateEvidenceSources)))
    return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFunctionTargetCall(py::CallOp op,
                                              const RuntimeBundle &callable) {
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python callable lowering expects exactly one Python result";

  auto target =
      module.lookupSymbol<mlir::func::FuncOp>(callable.functionTarget);
  if (!target)
    return op.emitError() << "function object target '"
                          << callable.functionTarget << "' is not defined";

  builder.setInsertionPoint(op);
  llvm::SmallVector<const RuntimeBundle *, 8> sources;
  llvm::SmallVector<RuntimeBundle, 8> materializedDefaults;
  llvm::SmallVector<RuntimeBundle, 4> closureSources;
  llvm::SmallVector<RuntimeBundle, 8> argumentEvidenceSources;
  llvm::SmallVector<RuntimeBundle, 8> aggregateEvidenceSources;
  if (mlir::failed(RuntimeBundleLowerer::collectFunctionTargetRuntimeSources(
          op, target, callable.functionTarget, callable, sources,
          materializedDefaults, closureSources, argumentEvidenceSources,
          aggregateEvidenceSources)))
    return mlir::failure();

  if (target->hasAttr("ly.async.body_result"))
    return RuntimeBundleLowerer::lowerAsyncFunctionTargetCall(
        op, target, callable.functionTarget, sources);

  mlir::FailureOr<mlir::func::CallOp> call =
      RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
          op, target, callable.functionTarget, sources);
  if (mlir::failed(call))
    return mlir::failure();

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleFunctionTargetCallResult(
          op, target, callable.functionTarget, *call, sources, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::FailureOr<mlir::func::CallOp>
RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  mlir::FunctionType functionType = target.getFunctionType();
  for (mlir::Type input : functionType.getInputs())
    if (py::isPyType(input))
      return op.emitError() << "function target '" << targetName
                            << "' still has unresolved Python parameter ABI";
  for (mlir::Type result : functionType.getResults())
    if (py::isPyType(result))
      return op.emitError() << "function target '" << targetName
                            << "' still has unresolved Python result ABI";

  RuntimeSymbol targetSymbol;
  targetSymbol.function = target;
  targetSymbol.contract = "builtins.function";
  targetSymbol.role = "method";
  targetSymbol.name = targetName;

  llvm::SmallVector<mlir::Type, 8> logicalInputTypes;
  if (auto callableAttr =
          target->getAttrOfType<mlir::TypeAttr>("callable_type")) {
    if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
            callableAttr.getValue()))
      logicalInputTypes =
          RuntimeBundleLowerer::callableLogicalInputTypes(target, callable);
  }

  llvm::SmallVector<mlir::Value, 8> operands;
  unsigned inputIndex = 0;
  for (auto [sourceIndex, source] : llvm::enumerate(sources)) {
    if (inputIndex >= functionType.getNumInputs())
      return op.emitError()
             << "too many positional args for function target " << targetName;
    mlir::LogicalResult appended =
        sourceIndex < logicalInputTypes.size()
            ? RuntimeBundleLowerer::appendRuntimeSourceAs(
                  op, targetSymbol, functionType, inputIndex, *source,
                  logicalInputTypes[sourceIndex], operands)
            : RuntimeBundleLowerer::appendRuntimeSource(
                  op, targetSymbol, functionType, inputIndex, *source,
                  operands);
    if (mlir::failed(appended))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::appendPrimitiveI64EvidenceOperand(
            op, functionType, inputIndex, *source, operands)))
      return mlir::failure();
  }
  if (inputIndex != functionType.getNumInputs())
    return op.emitError() << "missing positional args for function target "
                          << targetName;

  return RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), targetSymbol,
                                                 operands);
}

mlir::LogicalResult RuntimeBundleLowerer::bundleFunctionTargetCallResult(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    mlir::func::CallOp call, llvm::ArrayRef<const RuntimeBundle *> sources,
    RuntimeBundle &result) {
  mlir::Type expectedResult = op.getResult(0).getType();
  if (runtimeContractName(expectedResult).empty()) {
    auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
    if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
            callableAttr ? callableAttr.getValue() : mlir::Type())) {
      if (callable.getResultTypes().size() == 1)
        expectedResult = callable.getResultTypes().front();
    }
  }

  auto returnedCoroutine = returnedCoroutineSummaries.find(targetName);
  auto returnedObjectEvidence =
      returnedObjectEvidenceSummaries.find(targetName);
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, expectedResult, "function target result object ABI");
  if (mlir::failed(resultTypes))
    return mlir::failure();
  if (call.getNumResults() < resultTypes->size())
    return op.emitError() << "function target '" << targetName
                          << "' returned too few values for result object ABI";

  llvm::SmallVector<mlir::Value, 4> objectValues;
  unsigned resultIndex = 0;
  auto consumePrimitiveI64Evidence =
      [&](mlir::Type contract, RuntimeBundle &bundle,
          llvm::StringRef label) -> mlir::LogicalResult {
    if (!RuntimeBundleLowerer::hasPrimitiveI64ABI(contract))
      return mlir::success();
    if (resultIndex + 2 > call.getNumResults() ||
        !call.getResult(resultIndex).getType().isInteger(64) ||
        !call.getResult(resultIndex + 1).getType().isInteger(1))
      return op.emitError()
             << "function target '" << targetName << "' returned no "
             << "primitive i64 evidence for " << label;
    bundle.primitiveI64 =
        RuntimePrimitiveI64Evidence{call.getResult(resultIndex),
                                    call.getResult(resultIndex + 1)};
    resultIndex += 2;
    return mlir::success();
  };
  for (unsigned end = static_cast<unsigned>(resultTypes->size());
       resultIndex < end; ++resultIndex)
    objectValues.push_back(call.getResult(resultIndex));
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, expectedResult, objectValues, result)))
    return mlir::failure();
  if (mlir::failed(consumePrimitiveI64Evidence(expectedResult, result,
                                              "result object")))
    return mlir::failure();

  if (returnedCoroutine != returnedCoroutineSummaries.end()) {
    if (runtimeContractName(expectedResult) != "types.CoroutineType")
      return op.emitError() << "function target '" << targetName
                            << "' has coroutine return evidence, but result "
                               "contract is "
                            << expectedResult;

    result.coroutineTarget = returnedCoroutine->second.target;
    for (mlir::Type sourceType : returnedCoroutine->second.sourceContracts) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> sourceTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(
              op, sourceType, "returned coroutine frame source ABI");
      if (mlir::failed(sourceTypes))
        return mlir::failure();
      if (resultIndex + sourceTypes->size() > call.getNumResults())
        return op.emitError()
               << "function target '" << targetName
               << "' returned too few values for coroutine frame ABI";

      llvm::SmallVector<mlir::Value, 4> sourceValues;
      for (unsigned end =
               resultIndex + static_cast<unsigned>(sourceTypes->size());
           resultIndex < end; ++resultIndex)
        sourceValues.push_back(call.getResult(resultIndex));

      RuntimeBundle sourceBundle;
      if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
              op, sourceType, sourceValues, sourceBundle)))
        return mlir::failure();
      if (mlir::failed(consumePrimitiveI64Evidence(
              sourceType, sourceBundle, "coroutine frame source")))
        return mlir::failure();
      result.coroutineSources.push_back(sourceBundle.objectValue);
    }
  }

  auto returnedValue = returnedValueSummaries.find(targetName);
  if (returnedValue != returnedValueSummaries.end() &&
      returnedValue->second.argumentIndices.size() == 1) {
    unsigned sourceIndex = returnedValue->second.argumentIndices.front();
    if (sourceIndex >= sources.size())
      return op.emitError()
             << "returned value summary for " << targetName
             << " references logical argument " << sourceIndex
             << ", but call has only " << sources.size() << " logical sources";
    if (sources[sourceIndex]->kind == RuntimeBundle::Kind::Object)
      result.copyEvidenceFrom(*sources[sourceIndex]);
  }

  auto returned = returnedCallableSummaries.find(targetName);
  if (returned != returnedCallableSummaries.end()) {
    result.callableAlternatives.clear();
    result.callableAlternatives.reserve(returned->second.alternatives.size());
    for (const ReturnedCallableAlternativeSummary &returnedAlternative :
         returned->second.alternatives) {
      RuntimeCallableAlternative alternative;
      alternative.functionTarget = returnedAlternative.target;
      for (unsigned index : returnedAlternative.captureArgumentIndices) {
        if (index >= sources.size())
          return op.emitError() << "returned callable summary for "
                                << targetName << " references logical argument "
                                << index << ", but call has only "
                                << sources.size() << " logical sources";
        if (sources[index]->kind != RuntimeBundle::Kind::Object)
          return op.emitError()
                 << "returned callable capture source must be an object bundle";
        alternative.closureValues.push_back(sources[index]->objectValue);
      }
      result.callableAlternatives.push_back(std::move(alternative));
    }
    if (result.callableAlternatives.size() == 1) {
      const RuntimeCallableAlternative &alternative =
          result.callableAlternatives.front();
      result.functionTarget = alternative.functionTarget;
      result.closureValues = alternative.closureValues;
    }
  }
  if (returnedObjectEvidence != returnedObjectEvidenceSummaries.end() &&
      compatibleRuntimeObjectEvidenceContract(
          expectedResult, returnedObjectEvidence->second.objectContract)) {
    for (llvm::StringRef flag : returnedObjectEvidence->second.flags)
      result.objectEvidence.setFlag(flag);
    for (const ReturnedObjectEvidenceSlot &slot :
         returnedObjectEvidence->second.slots) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> slotTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(
              op, slot.sourceContract, "returned object evidence slot ABI");
      if (mlir::failed(slotTypes))
        return mlir::failure();
      if (resultIndex + slotTypes->size() > call.getNumResults())
        return op.emitError()
               << "function target '" << targetName
               << "' returned too few values for object evidence ABI";

      llvm::SmallVector<mlir::Value, 4> slotValues;
      for (unsigned end =
               resultIndex + static_cast<unsigned>(slotTypes->size());
           resultIndex < end; ++resultIndex)
        slotValues.push_back(call.getResult(resultIndex));

      RuntimeBundle slotBundle;
      if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
              op, slot.sourceContract, slotValues, slotBundle)))
        return mlir::failure();
      if (mlir::failed(consumePrimitiveI64Evidence(
              slot.sourceContract, slotBundle, "object evidence slot")))
        return mlir::failure();
      result.objectEvidence.setSlot(slot.name, slotBundle.objectValue);
    }
  }
  if (resultIndex != call.getNumResults())
    return op.emitError() << "function target '" << targetName << "' returned "
                          << call.getNumResults()
                          << " physical values, but call result bundling "
                             "consumed "
                          << resultIndex;
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
