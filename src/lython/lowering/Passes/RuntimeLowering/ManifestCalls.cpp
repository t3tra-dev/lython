#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

mlir::LogicalResult
RuntimeBundleLowerer::verifySelectedRuntimeTarget(mlir::Operation *op,
                                                  RuntimeSymbol &symbol) {
  auto target = op->getAttrOfType<mlir::FlatSymbolRefAttr>("target");
  if (!target)
    return mlir::success();
  if (target.getValue() == symbol.function.getSymName() ||
      target.getValue() == symbol.name)
    return mlir::success();
  return op->emitError() << "resolved target @" << target.getValue()
                         << " does not match runtime manifest symbol @"
                         << symbol.function.getSymName()
                         << " or manifest member " << symbol.contract << "."
                         << symbol.name;
}

mlir::FailureOr<RuntimeSymbol> RuntimeBundleLowerer::selectManifestMethod(
    mlir::Operation *op, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources) {
  std::string receiverContract = receiver.contractName();
  if (receiverContract.empty()) {
    if (mlir::isa<py::ProtocolType>(receiver.contract))
      return op->emitError()
             << "protocol-typed receiver " << receiver.contract
             << " has no concrete runtime method evidence for " << methodName
             << "; specialize it from a concrete producer or carry explicit "
                "method evidence before runtime lowering";
    return op->emitError()
           << "runtime method receiver has no concrete contract";
  }

  llvm::ArrayRef<RuntimeSymbol> methods =
      manifest.methodCandidates(receiverContract, methodName);
  if (methods.empty())
    return op->emitError() << "runtime manifest has no " << receiverContract
                           << "." << methodName << " method";

  const RuntimeSymbol *method = nullptr;
  for (const RuntimeSymbol &candidate : methods) {
    if (!canBuildRuntimeCallOperands(candidate, sources, allowUnusedSources,
                                     /*classObject=*/nullptr))
      continue;
    if (method)
      return op->emitError() << "runtime manifest has ambiguous overloads for "
                             << receiverContract << "." << methodName;
    method = &candidate;
  }
  if (!method)
    method = &methods.front();

  RuntimeSymbol selected = *method;
  if (mlir::failed(verifySelectedRuntimeTarget(op, selected)))
    return mlir::failure();
  return selected;
}

mlir::LogicalResult RuntimeBundleLowerer::emitManifestMethodCall(
    mlir::Operation *op, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources, std::optional<EmittedRuntimeCall> &emitted) {
  mlir::FailureOr<RuntimeSymbol> selected =
      RuntimeBundleLowerer::selectManifestMethod(op, receiver, methodName,
                                                 sources, allowUnusedSources);
  if (mlir::failed(selected))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> operands;
  builder.setInsertionPoint(op);
  if (mlir::failed(buildRuntimeCallOperands(op, *selected, sources, operands,
                                            allowUnusedSources)))
    return mlir::failure();

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *selected, operands);
  emitted.emplace(EmittedRuntimeCall{*selected, call});
  return mlir::success();
}

std::string
RuntimeBundleLowerer::resultContractFor(mlir::Value resultValue,
                                        const RuntimeSymbol &symbol,
                                        bool preferManifestObjectResult) const {
  std::string contract = runtimeContractName(resultValue.getType());
  if (!symbol.resultContract.empty() &&
      (contract.empty() ||
       (preferManifestObjectResult && contract == "builtins.object")))
    return symbol.resultContract;
  return contract;
}

mlir::LogicalResult RuntimeBundleLowerer::bindRuntimeCallResult(
    mlir::Operation *op, mlir::Value resultValue,
    const EmittedRuntimeCall &emitted, bool preferManifestObjectResult,
    const RuntimeBundle *receiverEvidence) {
  std::string resultContract = RuntimeBundleLowerer::resultContractFor(
      resultValue, emitted.symbol, preferManifestObjectResult);
  if (resultContract.empty())
    return op->emitError() << "runtime method result for "
                           << emitted.symbol.contract << "."
                           << emitted.symbol.name
                           << " needs a concrete manifest result contract";

  RuntimeBundle result;
  if (mlir::failed(
          bundleRuntimeResults(op, runtimeContractType(context, resultContract),
                               emitted.call, result)))
    return mlir::failure();
  if (emitted.symbol.resultEvidence == "receiver") {
    if (!receiverEvidence)
      return op->emitError()
             << "runtime method result for " << emitted.symbol.contract << "."
             << emitted.symbol.name
             << " requires receiver evidence, but no receiver was provided";
    if (receiverEvidence->kind != RuntimeBundle::Kind::Object)
      return op->emitError()
             << "runtime method result for " << emitted.symbol.contract << "."
             << emitted.symbol.name << " requires object receiver evidence";
    result.copyEvidenceFrom(*receiverEvidence);
  }
  valueBundles[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerManifestMethodResult(
    mlir::Operation *op, mlir::Value resultValue, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources, bool preferManifestObjectResult) {
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(emitManifestMethodCall(op, receiver, methodName, sources,
                                          allowUnusedSources, emitted)))
    return mlir::failure();
  return RuntimeBundleLowerer::bindRuntimeCallResult(
      op, resultValue, *emitted, preferManifestObjectResult, &receiver);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerManifestI1MethodResult(
    mlir::Operation *op, mlir::Value resultValue, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources) {
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(emitManifestMethodCall(op, receiver, methodName, sources,
                                          allowUnusedSources, emitted)))
    return mlir::failure();
  mlir::func::CallOp call = emitted->call;
  if (call.getNumResults() != 1 || !call.getResult(0).getType().isInteger(1))
    return op->emitError() << "runtime " << methodName
                           << " method must lower to a single i1 result";
  resultValue.replaceAllUsesWith(call.getResult(0));
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerManifestVoidMethod(
    mlir::Operation *op, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources) {
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(emitManifestMethodCall(op, receiver, methodName, sources,
                                          allowUnusedSources, emitted)))
    return mlir::failure();
  if (emitted->call.getNumResults() != 0)
    return op->emitError()
           << "runtime " << methodName << " method produced "
           << emitted->call.getNumResults()
           << " physical results, but this operation expects NoneType";
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNew(py::NewOp op) {
  const RuntimeBundle *classObject =
      RuntimeBundleLowerer::bundleFor(op.getClassObject());
  if (!classObject || classObject->kind != RuntimeBundle::Kind::TypeObject)
    return op.emitError() << "new class object has no lowered type bundle";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  std::string contract = runtimeContractName(op.getInstance().getType());
  if (contract.empty())
    return op.emitError() << "new result has no concrete runtime contract";
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__new__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (*methodName == "__new__" &&
      mlir::succeeded(RuntimeBundleLowerer::bindErasedCtypesNew(op, contract)))
    return mlir::success();
  if (*methodName == "__new__" &&
      mlir::succeeded(
          RuntimeBundleLowerer::bindStaticCtypesLibraryNew(op, contract)))
    return mlir::success();
  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contract, *methodName);
  if (!initializer)
    if (py::ClassOp classOp = RuntimeBundleLowerer::classForContract(
            op.getInstance().getType())) {
      (void)classOp;
      builder.setInsertionPoint(op);
      mlir::FailureOr<RuntimeValue> value =
          RuntimeBundleLowerer::materializeDeadObjectValue(
              op, op.getInstance().getType(), "class __new__ ABI");
      if (mlir::failed(value))
        return mlir::failure();
      valueBundles[op.getInstance()] =
          RuntimeBundle::object(value->contract, value->values);
      erase.push_back(op);
      return mlir::success();
    }
  if (!initializer)
    return op.emitError() << "runtime manifest has no " << contract << "."
                          << *methodName;
  if (mlir::failed(verifySelectedRuntimeTarget(op, *initializer)))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources;
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> operands;
  builder.setInsertionPoint(op);
  if (mlir::failed(buildRuntimeCallOperands(op, *initializer, sources, operands,
                                            /*allowUnusedSources=*/true,
                                            classObject)))
    return mlir::failure();

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, operands);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, op.getInstance().getType(), call, result)))
    return mlir::failure();
  if (contract == "_asyncio.Task") {
    if (sources.empty() || !sources.front() ||
        sources.front()->kind != RuntimeBundle::Kind::Object ||
        sources.front()->contractName() != "types.CoroutineType" ||
        (sources.front()->coroutineTarget.empty() &&
         !hasAsyncioSleepEvidence(*sources.front())))
      return op.emitError()
             << "_asyncio.Task.__new__ requires a lowered coroutine object";
    result.copyEvidenceFrom(*sources.front());
    if (hasAsyncioSleepEvidence(result) &&
        result.objectEvidence.hasFlag(kAsyncioSleepZeroDelayFlag)) {
      if (const RuntimeValue *sleepResult =
              result.objectEvidence.slot(kAsyncioSleepResultSlot))
        result.objectEvidence.setSlot(kFutureResultSlot, *sleepResult);
      std::optional<RuntimeSymbol> finish =
          manifest.primitive("_asyncio.Task", "finish.request");
      if (!finish)
        return op.emitError()
               << "runtime manifest has no _asyncio.Task.finish.request";
      llvm::SmallVector<const RuntimeBundle *, 1> finishSources{&result};
      llvm::SmallVector<mlir::Value, 4> finishOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *finish, finishSources,
                                                finishOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *finish,
                                              finishOperands);
    }
  }
  valueBundles[op.getInstance()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerInit(py::InitOp op) {
  const RuntimeBundle *instance =
      RuntimeBundleLowerer::bundleFor(op.getInstance());
  if (!instance)
    return op.emitError() << "init instance has no lowered runtime bundle";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources{instance};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__init__");
  if (mlir::failed(methodName))
    return mlir::failure();

  if (*methodName == "__init__" && instance->ctypes &&
      instance->ctypes->kind == RuntimeCtypesEvidence::Kind::Library)
    return RuntimeBundleLowerer::lowerStaticCtypesLibraryInit(op, *instance,
                                                              sources);
  if (*methodName == "__init__" && instance->ctypes)
    return RuntimeBundleLowerer::lowerErasedCtypesInit(op, *instance, sources);

  if (py::ClassOp classOp =
          RuntimeBundleLowerer::classForContract(op.getInstance().getType())) {
    if (*methodName != "__init__")
      return op.emitError()
             << "class init lowering expected __init__, got " << *methodName;
    llvm::SmallVector<mlir::Type, 8> fieldTypes =
        RuntimeBundleLowerer::classFieldContractTypes(classOp);
    if (sources.size() > fieldTypes.size() + 1)
      return op.emitError()
             << "class " << classOp.getSymName() << " initializer received "
             << (sources.size() - 1) << " positional arguments for "
             << fieldTypes.size() << " fields";

    llvm::SmallVector<mlir::Value, 8> values(instance->physicalValues().begin(),
                                             instance->physicalValues().end());
    for (unsigned index = 0; index < fieldTypes.size(); ++index) {
      if (index + 1 >= sources.size())
        continue;
      const RuntimeBundle *fieldValue = sources[index + 1];
      if (!fieldValue || fieldValue->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "class initializer field source has no object bundle";
      if (!py::isAssignableTo(fieldValue->objectValue.contract,
                              fieldTypes[index], op))
        return op.emitError()
               << "class initializer argument " << index << " has type "
               << fieldValue->objectValue.contract << ", but field expects "
               << fieldTypes[index];
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldTypes[index],
                                                     "class field ABI");
      if (mlir::failed(fieldValueTypes))
        return mlir::failure();
      if (fieldValueTypes->size() != fieldValue->physicalValues().size())
        return op.emitError() << "class initializer argument " << index
                              << " has " << fieldValue->physicalValues().size()
                              << " physical values, but field ABI expects "
                              << fieldValueTypes->size();
      mlir::FailureOr<unsigned> offset =
          RuntimeBundleLowerer::classFieldValueOffset(op, classOp, index,
                                                      "class field ABI");
      if (mlir::failed(offset))
        return mlir::failure();
      if (*offset + fieldValueTypes->size() > values.size())
        return op.emitError() << "class field ABI exceeds object payload";
      for (auto [fieldOffset, replacement] :
           llvm::enumerate(fieldValue->physicalValues()))
        values[*offset + fieldOffset] = replacement;
    }

    RuntimeBundle updated;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, op.getInstance().getType(), values, updated)))
      return mlir::failure();
    updated.copyEvidenceFrom(*instance);
    auto fieldNames = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
    if (fieldNames) {
      for (unsigned index = 0; index < fieldTypes.size(); ++index) {
        if (index + 1 >= sources.size() || index >= fieldNames.size())
          continue;
        auto name = mlir::dyn_cast<mlir::StringAttr>(fieldNames[index]);
        if (!name)
          return op.emitError() << "class field metadata is malformed for "
                                << classOp.getSymName();
        updated.fieldBundles[name.getValue()] =
            std::make_shared<RuntimeBundle>(*sources[index + 1]);
      }
    }
    valueBundles[op.getInstance()] = std::move(updated);
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(emitManifestMethodCall(op, *instance, *methodName, sources,
                                          /*allowUnusedSources=*/true,
                                          emitted)))
    return mlir::failure();
  if (emitted->call.getNumResults() != 0) {
    RuntimeBundle updatedInstance;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, op.getInstance().getType(), emitted->call, updatedInstance)))
      return mlir::failure();
    valueBundles[op.getInstance()] = std::move(updatedInstance);
  }

  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
