#include "RuntimeLowering/RuntimeBundleLowerer.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::runtime_lowering {

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

  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(target))
    return RuntimeBundleLowerer::lowerPrimitiveI64CloneCall(
        op, target, callable.functionTarget, sources);

  if (std::optional<std::string> cloneName =
          RuntimeBundleLowerer::primitiveI64CloneFor(callable.functionTarget)) {
    if (RuntimeBundleLowerer::allSourcesHavePrimitiveI64Evidence(sources)) {
      if (mlir::func::FuncOp clone =
              module.lookupSymbol<mlir::func::FuncOp>(*cloneName))
        return RuntimeBundleLowerer::lowerPrimitiveI64CloneFallbackCall(
            op, target, callable.functionTarget, clone, sources);
    }
  }

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

mlir::LogicalResult RuntimeBundleLowerer::bundlePrimitiveI64CloneCallResult(
    py::CallOp op, mlir::func::FuncOp target, mlir::func::CallOp call,
    RuntimeBundle &result) {
  if (call.getNumResults() != 2 || !call.getResult(0).getType().isInteger(64) ||
      !call.getResult(1).getType().isInteger(1))
    return op.emitError() << "primitive i64 callable clone '"
                          << target.getSymName() << "' must return (i64, i1)";
  return RuntimeBundleLowerer::makePrimitiveI64Bundle(
      op, op.getResult(0).getType(), call.getResult(0), call.getResult(1),
      result);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveI64CloneCall(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  mlir::FailureOr<mlir::func::CallOp> call =
      RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(op, target,
                                                          targetName, sources);
  if (mlir::failed(call))
    return mlir::failure();

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundlePrimitiveI64CloneCallResult(
          op, target, *call, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerPrimitiveI64CloneFallbackCall(
    py::CallOp op, mlir::func::FuncOp original, llvm::StringRef originalName,
    mlir::func::FuncOp clone, llvm::ArrayRef<const RuntimeBundle *> sources) {
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::emitPrimitiveI64CloneFallbackResult(
          op, original, originalName, clone, sources, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::emitPrimitiveI64CloneFallbackResult(
    py::CallOp op, mlir::func::FuncOp original, llvm::StringRef originalName,
    mlir::func::FuncOp clone, llvm::ArrayRef<const RuntimeBundle *> sources,
    RuntimeBundle &result) {
  mlir::Type resultType = op.getResult(0).getType();
  if (runtimeContractName(resultType).empty()) {
    auto callableAttr =
        original->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (callable && callable.getResultTypes().size() == 1)
      resultType = callable.getResultTypes().front();
  }
  if (runtimeContractName(resultType) != "builtins.int")
    return op.emitError()
           << "primitive i64 callable clone fallback requires builtins.int "
              "result";

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> objectTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, resultType, "primitive i64 callable clone fallback result ABI");
  if (mlir::failed(objectTypes))
    return mlir::failure();

  mlir::FailureOr<mlir::func::CallOp> cloneCall =
      RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
          op, clone, clone.getSymName(), sources);
  if (mlir::failed(cloneCall))
    return mlir::failure();
  if ((*cloneCall).getNumResults() != 2 ||
      !(*cloneCall).getResult(0).getType().isInteger(64) ||
      !(*cloneCall).getResult(1).getType().isInteger(1))
    return op.emitError() << "primitive i64 callable clone '"
                          << clone.getSymName() << "' must return (i64, i1)";

  context->loadDialect<mlir::scf::SCFDialect>();
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Type, 10> ifResultTypes;
  ifResultTypes.append(objectTypes->begin(), objectTypes->end());
  ifResultTypes.push_back(mlir::IntegerType::get(context, 64));
  ifResultTypes.push_back(mlir::IntegerType::get(context, 1));

  auto ifOp = builder.create<mlir::scf::IfOp>(loc, ifResultTypes,
                                              (*cloneCall).getResult(1),
                                              /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  RuntimeBundle fastObject;
  if (mlir::failed(RuntimeBundleLowerer::initializeObjectFromRawValues(
          op, resultType, mlir::ValueRange{(*cloneCall).getResult(0)},
          fastObject)))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 10> fastYield(
      fastObject.physicalValues().begin(), fastObject.physicalValues().end());
  if (fastYield.size() != objectTypes->size())
    return op.emitError() << "primitive i64 clone fast path produced "
                          << fastYield.size() << " object values, expected "
                          << objectTypes->size();
  fastYield.push_back((*cloneCall).getResult(0));
  fastYield.push_back((*cloneCall).getResult(1));
  builder.create<mlir::scf::YieldOp>(loc, fastYield);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  mlir::FailureOr<mlir::func::CallOp> fallbackCall =
      RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
          op, original, originalName, sources);
  if (mlir::failed(fallbackCall))
    return mlir::failure();
  if ((*fallbackCall).getNumResults() != ifResultTypes.size())
    return op.emitError() << "primitive i64 clone fallback call returned "
                          << (*fallbackCall).getNumResults()
                          << " values, expected " << ifResultTypes.size();
  for (auto [index, value] : llvm::enumerate((*fallbackCall).getResults())) {
    if (value.getType() != ifResultTypes[index])
      return op.emitError()
             << "primitive i64 clone fallback result " << index << " has type "
             << value.getType() << ", expected " << ifResultTypes[index];
  }
  builder.create<mlir::scf::YieldOp>(loc, (*fallbackCall).getResults());

  builder.setInsertionPointAfter(ifOp);
  llvm::SmallVector<mlir::Value, 8> objectValues;
  for (unsigned index = 0, end = static_cast<unsigned>(objectTypes->size());
       index < end; ++index)
    objectValues.push_back(ifOp.getResult(index));
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, resultType, objectValues, result)))
    return mlir::failure();
  result.primitiveI64 =
      RuntimePrimitiveI64Evidence{ifOp.getResult(objectTypes->size()),
                                  ifOp.getResult(objectTypes->size() + 1)};
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

  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(target)) {
    if (sources.size() != logicalInputTypes.size())
      return op.emitError() << "primitive i64 callable clone '" << targetName
                            << "' expects " << logicalInputTypes.size()
                            << " arguments, got " << sources.size();
    llvm::SmallVector<mlir::Value, 8> operands;
    unsigned inputIndex = 0;
    for (auto [sourceIndex, source] : llvm::enumerate(sources)) {
      if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(source))
        return op.emitError() << "primitive i64 callable clone '" << targetName
                              << "' argument " << sourceIndex
                              << " has no primitive i64 evidence";
      if (inputIndex + 2 > functionType.getNumInputs() ||
          !functionType.getInput(inputIndex).isInteger(64) ||
          !functionType.getInput(inputIndex + 1).isInteger(1))
        return op.emitError() << "primitive i64 callable clone '" << targetName
                              << "' has malformed ABI at input " << inputIndex;
      operands.push_back(source->primitiveI64->value);
      operands.push_back(source->primitiveI64->valid);
      inputIndex += 2;
    }
    if (inputIndex != functionType.getNumInputs())
      return op.emitError() << "primitive i64 callable clone '" << targetName
                            << "' ABI was not fully populated";
    return RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), targetSymbol,
                                                   operands);
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
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{
        call.getResult(resultIndex), call.getResult(resultIndex + 1)};
    resultIndex += 2;
    return mlir::success();
  };
  for (unsigned end = static_cast<unsigned>(resultTypes->size());
       resultIndex < end; ++resultIndex)
    objectValues.push_back(call.getResult(resultIndex));
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, expectedResult, objectValues, result)))
    return mlir::failure();
  if (mlir::failed(
          consumePrimitiveI64Evidence(expectedResult, result, "result object")))
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
      if (mlir::failed(consumePrimitiveI64Evidence(sourceType, sourceBundle,
                                                   "coroutine frame source")))
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

} // namespace py::runtime_lowering
