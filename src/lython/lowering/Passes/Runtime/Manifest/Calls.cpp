#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"

namespace py::lowering {

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
  if (methods.empty()) {
    // User exception receivers share their builtin ancestor's methods
    // (__init__/__str__/...): same physical shape, subclass-specific
    // identity only in the header's class id.
    if (std::optional<std::string> ancestor =
            RuntimeBundleLowerer::exceptionAncestorContractFor(
                runtimeContractType(context, receiverContract)))
      methods = manifest.methodCandidates(*ancestor, methodName);
  }
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
  mlir::Type resultType = runtimeContractType(context, resultContract);
  if (mlir::failed(RuntimeBundleLowerer::bindRuntimeCallBundle(
          op, resultType, emitted, receiverEvidence, result)))
    return mlir::failure();
  valueBundles[resultValue] = std::move(result);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindRuntimeCallBundle(
    mlir::Operation *op, mlir::Type resultType,
    const EmittedRuntimeCall &emitted, const RuntimeBundle *receiverEvidence,
    RuntimeBundle &result) {
  std::string resultContract = runtimeContractName(resultType);
  if (resultContract.empty())
    return op->emitError() << "runtime method result for "
                           << emitted.symbol.contract << "."
                           << emitted.symbol.name
                           << " needs a concrete manifest result contract";

  std::optional<unsigned> resultEvidenceStart;
  mlir::func::CallOp call = emitted.call;
  if (emitted.symbol.resultEvidenceSlots.empty()) {
    if (mlir::failed(bundleRuntimeResults(op, resultType, call, result)))
      return mlir::failure();
  } else {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            op, resultType, "runtime method result ABI");
    if (mlir::failed(resultTypes))
      return mlir::failure();
    if (call.getNumResults() < resultTypes->size())
      return op->emitError()
             << "runtime method result for " << emitted.symbol.contract << "."
             << emitted.symbol.name
             << " returned too few values for primary result ABI";
    unsigned resultIndex = 0;
    llvm::SmallVector<mlir::Value, 4> resultValues;
    for (unsigned end = static_cast<unsigned>(resultTypes->size());
         resultIndex < end; ++resultIndex)
      resultValues.push_back(call.getResult(resultIndex));
    if (mlir::failed(bundleRuntimeResults(op, resultType, resultValues, result)))
      return mlir::failure();
    resultEvidenceStart = resultIndex;
  }
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
  if (emitted.symbol.name == "__await__" && receiverEvidence)
    result.copyEvidenceFrom(*receiverEvidence);
  if (resultEvidenceStart) {
    unsigned resultIndex = *resultEvidenceStart;
    for (const RuntimeResultEvidenceSlot &slot :
         emitted.symbol.resultEvidenceSlots) {
      mlir::Type slotType = runtimeContractType(context, slot.contract);
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> slotTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(
              op, slotType, "runtime method result evidence slot ABI");
      if (mlir::failed(slotTypes))
        return mlir::failure();
      if (resultIndex + slotTypes->size() > call.getNumResults())
        return op->emitError()
               << "runtime method result for " << emitted.symbol.contract
               << "." << emitted.symbol.name
               << " returned too few values for result evidence slot '"
               << slot.name << "'";
      llvm::SmallVector<mlir::Value, 4> slotValues;
      for (unsigned end =
               resultIndex + static_cast<unsigned>(slotTypes->size());
           resultIndex < end; ++resultIndex)
        slotValues.push_back(call.getResult(resultIndex));
      RuntimeBundle slotBundle;
      if (mlir::failed(makeObjectBundle(op, slotType, slotValues, slotBundle)))
        return mlir::failure();
      result.objectEvidence.setSlot(slot.name, slotBundle.objectValue);
    }
    if (resultIndex != call.getNumResults())
      return op->emitError()
             << "runtime method result for " << emitted.symbol.contract << "."
             << emitted.symbol.name << " returned " << call.getNumResults()
             << " physical values, but result evidence binding consumed "
             << resultIndex;
  }
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
  const RuntimeBundle *kwNames =
      RuntimeBundleLowerer::bundleFor(op.getKwnames());
  const RuntimeBundle *kwValues =
      RuntimeBundleLowerer::bundleFor(op.getKwvalues());
  if (!kwNames || kwNames->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "kw names must be a lowered aggregate bundle";
  if (!kwValues || kwValues->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "kw values must be a lowered aggregate bundle";
  if (kwNames->aggregateOperands.size() != kwValues->aggregateOperands.size())
    return op.emitError() << "new keyword name/value count mismatch";
  bool hasKeywords = !kwNames->aggregateOperands.empty();

  std::string contract = runtimeContractName(op.getInstance().getType());
  if (contract.empty())
    return op.emitError() << "new result has no concrete runtime contract";
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__new__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (*methodName == "__new__" &&
      hasKeywords &&
      (RuntimeBundleLowerer::isErasedCtypesContract(contract) ||
       RuntimeBundleLowerer::isStaticCtypesLibraryContract(contract)))
    return op.emitError() << "ctypes __new__ lowering is not keyword-aware yet";
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
      // User exception classes construct through the builtin exception
      // ancestor's initializer; the class-id argument channel stamps the
      // source class's own id into the header.
      if (std::optional<std::string> ancestor =
              RuntimeBundleLowerer::exceptionAncestorContract(classOp))
        initializer = manifest.initializer(*ancestor, *methodName);
    }
  if (!initializer)
    if (py::ClassOp classOp = RuntimeBundleLowerer::classForContract(
            op.getInstance().getType())) {
      builder.setInsertionPoint(op);
      mlir::FailureOr<RuntimeValue> value =
          RuntimeBundleLowerer::materializeClassObjectValue(
              op, classOp, op.getInstance().getType(), "class __new__ ABI");
      if (mlir::failed(value))
        return mlir::failure();
      RuntimeBundle result = RuntimeBundle::object(value->contract,
                                                   value->values);
      if (mlir::failed(RuntimeBundleLowerer::markOwnedLocalObjectBundle(
              op, op.getInstance(), result)))
        return mlir::failure();
      valueBundles[op.getInstance()] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  if (!initializer)
    return op.emitError() << "runtime manifest has no " << contract << "."
                          << *methodName;
  if (hasKeywords)
    return op.emitError()
           << "runtime manifest __new__ lowering is not keyword-aware yet";
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
      instance->ctypes->kind == RuntimeCtypesEvidence::Kind::Library) {
    if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
        mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
      return mlir::failure();
    return RuntimeBundleLowerer::lowerStaticCtypesLibraryInit(op, *instance,
                                                              sources);
  }
  if (*methodName == "__init__" && instance->ctypes) {
    if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
        mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
      return mlir::failure();
    return RuntimeBundleLowerer::lowerErasedCtypesInit(op, *instance, sources);
  }

  // Exception-backed classes skip the field-record path: their __init__ is
  // the builtin exception ancestor's (message binding), reached through the
  // generic manifest method call below.
  py::ClassOp instanceClassOp =
      RuntimeBundleLowerer::classForContract(op.getInstance().getType());
  if (py::ClassOp classOp =
          instanceClassOp &&
                  !RuntimeBundleLowerer::exceptionAncestorContract(
                      instanceClassOp)
              ? instanceClassOp
              : py::ClassOp()) {
    if (*methodName != "__init__")
      return op.emitError()
             << "class init lowering expected __init__, got " << *methodName;
    llvm::SmallVector<mlir::Type, 8> fieldTypes =
        RuntimeBundleLowerer::classFieldContractTypes(classOp);
    auto fieldNames = classOp->getAttrOfType<mlir::ArrayAttr>("field_names");
    if (!fieldNames)
      return op.emitError() << "class " << classOp.getSymName()
                            << " has no field schema";
    if (sources.size() > fieldTypes.size() + 1)
      return op.emitError()
             << "class " << classOp.getSymName() << " initializer received "
             << (sources.size() - 1) << " positional arguments for "
             << fieldTypes.size() << " fields";
    if (fieldNames.size() != fieldTypes.size())
      return op.emitError() << "class " << classOp.getSymName()
                            << " field schema is malformed";

    llvm::SmallVector<const RuntimeBundle *, 8> fieldSources(fieldTypes.size(),
                                                            nullptr);
    for (unsigned index = 1; index < sources.size(); ++index)
      fieldSources[index - 1] = sources[index];

    const RuntimeBundle *kwNames =
        RuntimeBundleLowerer::bundleFor(op.getKwnames());
    const RuntimeBundle *kwValues =
        RuntimeBundleLowerer::bundleFor(op.getKwvalues());
    if (!kwNames || kwNames->kind != RuntimeBundle::Kind::Aggregate)
      return op.emitError()
             << "class initializer kw names must be a lowered aggregate bundle";
    if (!kwValues || kwValues->kind != RuntimeBundle::Kind::Aggregate)
      return op.emitError()
             << "class initializer kw values must be a lowered aggregate "
                "bundle";
    if (kwNames->aggregateOperands.size() !=
        kwValues->aggregateOperands.size())
      return op.emitError()
             << "class initializer keyword name/value count mismatch";
    for (auto [kwIndex, nameValue] :
         llvm::enumerate(kwNames->aggregateOperands)) {
      std::optional<std::string> keyword =
          RuntimeBundleLowerer::keywordNameFromValue(nameValue);
      if (!keyword)
        return op.emitError()
               << "class initializer keyword name must be statically known";
      unsigned fieldIndex = fieldTypes.size();
      for (auto [index, fieldNameAttr] : llvm::enumerate(fieldNames)) {
        auto fieldName = mlir::dyn_cast<mlir::StringAttr>(fieldNameAttr);
        if (!fieldName)
          return op.emitError() << "class field metadata is malformed for "
                                << classOp.getSymName();
        if (fieldName.getValue() == *keyword) {
          fieldIndex = static_cast<unsigned>(index);
          break;
        }
      }
      if (fieldIndex >= fieldTypes.size())
        return op.emitError()
               << "unexpected class initializer keyword '" << *keyword << "'";
      if (fieldSources[fieldIndex])
        return op.emitError()
               << "multiple values for class initializer field '" << *keyword
               << "'";
      const RuntimeBundle *keywordValue = RuntimeBundleLowerer::bundleFor(
          kwValues->aggregateOperands[kwIndex]);
      if (!keywordValue)
        return op.emitError()
               << "class initializer keyword value has no lowered bundle";
      fieldSources[fieldIndex] = keywordValue;
    }

    llvm::SmallVector<mlir::Value, 8> values(instance->physicalValues().begin(),
                                             instance->physicalValues().end());
    llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> updatedFieldBundles;
    updatedFieldBundles.resize(fieldTypes.size());
    for (unsigned index = 0; index < fieldTypes.size(); ++index) {
      const RuntimeBundle *fieldValue = fieldSources[index];
      if (!fieldValue) {
        std::string fieldName = std::to_string(index);
        if (index < fieldNames.size()) {
          auto name = mlir::dyn_cast<mlir::StringAttr>(fieldNames[index]);
          if (!name)
            return op.emitError() << "class field metadata is malformed for "
                                  << classOp.getSymName();
          fieldName = name.getValue().str();
        }
        return op.emitError()
               << "missing required class initializer field '" << fieldName
               << "'";
      }
      if (!fieldValue || fieldValue->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "class initializer field source has no object bundle";
      if (!py::isAssignableTo(fieldValue->objectValue.contract,
                              fieldTypes[index], op))
        return op.emitError()
               << "class initializer argument " << index << " has type "
               << fieldValue->objectValue.contract << ", but field expects "
               << fieldTypes[index];
      std::string fieldName = "field";
      if (fieldNames && index < fieldNames.size()) {
        auto name = mlir::dyn_cast<mlir::StringAttr>(fieldNames[index]);
        if (!name)
          return op.emitError() << "class field metadata is malformed for "
                                << classOp.getSymName();
        fieldName = name.getValue().str();
      }
      // A header-word field's storage is a word of the instance header, and
      // this initializer used to write the placeholder LANES instead: a class
      // declared without __init__ (`class P: x: int`) read its int fields back
      // as zero, silently, because attr.get has always loaded the word.
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> storageTypes =
          RuntimeBundleLowerer::classFieldStorageValueTypes(
              op, fieldTypes[index], index, "class field ABI");
      if (mlir::failed(storageTypes))
        return mlir::failure();
      if (storageTypes->empty()) {
        if (mlir::failed(RuntimeBundleLowerer::storePrimitiveFieldSlot(
                op, *instance, *fieldValue, fieldTypes[index],
                static_cast<unsigned>(box_abi::kPointerWordBase) + index,
                fieldName)))
          return mlir::failure();
        updatedFieldBundles[index] =
            std::make_shared<RuntimeBundle>(*fieldValue);
        continue;
      }
      bool boxedField =
          RuntimeBundleLowerer::classFieldStoredBoxed(fieldTypes[index]);
      // Every box-fronted field fills the box16 the instance materialization
      // already allocated instead of boxing into a fresh slot: the box pointer
      // is the instance-lifetime stable handle that every later store writes
      // through and every other frame reads through, so it must never be
      // re-rooted after construction.
      if (boxedField) {
        std::string slotName = (llvm::Twine("class.") + fieldName).str();
        mlir::FailureOr<unsigned> offset =
            RuntimeBundleLowerer::classFieldValueOffset(op, classOp, index,
                                                        "class field ABI");
        if (mlir::failed(offset))
          return mlir::failure();
        if (*offset >= values.size())
          return op.emitError() << "class field ABI exceeds object payload";
        builder.setInsertionPoint(op);
        mlir::FailureOr<RuntimeBundle> stored =
            RuntimeBundleLowerer::storeBoxedFieldPayloadInPlace(
                op, values[*offset], *fieldValue, slotName);
        if (mlir::failed(stored))
          return mlir::failure();
        updatedFieldBundles[index] =
            std::make_shared<RuntimeBundle>(std::move(*stored));
        continue;
      }
      // Residual: a field with no single object contract to put behind a
      // handle — a union, a zero-lane contract, or an int/bool past the last
      // header word. See lowerAttrSet for why these keep the lane splice.
      RuntimeBundle slotValue;
      {
        mlir::FailureOr<RuntimeBundle> storageValue =
            RuntimeBundleLowerer::materializeObjectBundleForStorage(
                op, *fieldValue, fieldTypes[index],
                "class initializer argument ABI");
        if (mlir::failed(storageValue))
          return mlir::failure();
        slotValue = std::move(*storageValue);
      }

      llvm::SmallVector<mlir::Type, 8> *fieldValueTypes = &*storageTypes;
      mlir::FailureOr<unsigned> offset =
          RuntimeBundleLowerer::classFieldValueOffset(op, classOp, index,
                                                      "class field ABI");
      if (mlir::failed(offset))
        return mlir::failure();
      if (*offset + fieldValueTypes->size() > values.size())
        return op.emitError() << "class field ABI exceeds object payload";
      llvm::SmallVector<mlir::Value, 4> oldValues;
      oldValues.reserve(fieldValueTypes->size());
      for (unsigned fieldOffset = 0; fieldOffset < fieldValueTypes->size();
           ++fieldOffset)
        oldValues.push_back(values[*offset + fieldOffset]);
      std::string slotName = (llvm::Twine("class.") + fieldName).str();
      builder.setInsertionPoint(op);
      const RuntimeBundle *oldSlotValue = nullptr;
      if (auto oldField = instance->fieldBundles.find(fieldName);
          oldField != instance->fieldBundles.end())
        oldSlotValue = oldField->second.get();
      if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
              op, fieldTypes[index], oldValues, oldSlotValue,
              fieldTypes[index], slotValue, slotName,
              /*releaseMissingOldObjectSlot=*/false)))
        return mlir::failure();
      for (auto [fieldOffset, replacement] :
           llvm::enumerate(slotValue.physicalValues()))
        values[*offset + fieldOffset] = replacement;
      slotValue.setObjectLogicalOwnership(/*ownsObject=*/true);
      updatedFieldBundles[index] =
          std::make_shared<RuntimeBundle>(std::move(slotValue));
    }

    RuntimeBundle updated;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
            op, op.getInstance().getType(), values, updated,
            instance->objectValue.ownership)))
      return mlir::failure();
    updated.copyEvidenceFrom(*instance);
    if (fieldNames) {
      for (unsigned index = 0; index < fieldTypes.size(); ++index) {
        if (!fieldSources[index] || index >= fieldNames.size())
          continue;
        auto name = mlir::dyn_cast<mlir::StringAttr>(fieldNames[index]);
        if (!name)
          return op.emitError() << "class field metadata is malformed for "
                                << classOp.getSymName();
        if (updatedFieldBundles[index])
          updated.fieldBundles[name.getValue()] = updatedFieldBundles[index];
        else
          updated.fieldBundles[name.getValue()] =
              std::make_shared<RuntimeBundle>(*fieldSources[index]);
      }
    }
    if (mlir::failed(RuntimeBundleLowerer::markOwnedLocalObjectBundle(
            op, op.getInstance(), updated)))
      return mlir::failure();
    valueBundles[op.getInstance()] = std::move(updated);
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  // ExceptionGroup(message, [members...]): the member sequence becomes the
  // exception's extended member block (word 3); the message goes through the
  // ancestor __init__ below. The gate is static — CPython's construction-time
  // TypeError/ValueError (non-exception member, empty sequence) surface as
  // compile diagnostics here, and an unsupported second argument is rejected
  // instead of falling into the unused-source drop.
  std::string initContract = runtimeContractName(op.getInstance().getType());
  bool groupInit = *methodName == "__init__" &&
                   (initContract == "builtins.ExceptionGroup" ||
                    initContract == "builtins.BaseExceptionGroup");
  if (groupInit && sources.size() > 3)
    return op.emitError() << initContract
                          << " takes exactly two arguments (message, "
                             "exceptions)";
  if (groupInit && sources.size() == 3) {
    if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
        mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
      return mlir::failure();
    const RuntimeBundle *memberSequence = sources[2];
    if (!memberSequence)
      return op.emitError() << initContract
                            << " member sequence has no lowered bundle";
    struct GroupMember {
      mlir::Type contract;
      llvm::SmallVector<mlir::Value, 4> values;
    };
    llvm::SmallVector<GroupMember, 8> groupMembers;
    if (!memberSequence->sequenceElementBundles.empty()) {
      for (const std::shared_ptr<RuntimeBundle> &element :
           memberSequence->sequenceElementBundles) {
        if (!element)
          return op.emitError()
                 << initContract << " member has no lowered bundle";
        groupMembers.push_back(GroupMember{
            element->objectValue.contract,
            llvm::SmallVector<mlir::Value, 4>(
                element->physicalValues().begin(),
                element->physicalValues().end())});
      }
    } else {
      for (const RuntimeValue &element : memberSequence->sequenceElements)
        groupMembers.push_back(GroupMember{
            element.contract,
            llvm::SmallVector<mlir::Value, 4>(element.values.begin(),
                                              element.values.end())});
    }
    if (groupMembers.empty())
      return op.emitError()
             << initContract
             << " requires a statically-known, non-empty exception sequence "
                "as its second argument";
    mlir::Type requiredBase = runtimeContractType(
        context, initContract == "builtins.ExceptionGroup"
                     ? "builtins.Exception"
                     : "builtins.BaseException");
    mlir::Type exceptionBase =
        runtimeContractType(context, "builtins.Exception");
    bool allMembersAreExceptions = true;
    // Source exception classes do not resolve against builtin contracts
    // through py::isAssignableTo; their builtin taxonomy ancestor carries
    // the derivation instead.
    auto derivesFrom = [&](mlir::Type memberContract, mlir::Type base) {
      if (py::isAssignableTo(memberContract, base, op))
        return true;
      std::optional<std::string> ancestor =
          RuntimeBundleLowerer::exceptionAncestorContractFor(memberContract);
      return ancestor &&
             py::isAssignableTo(runtimeContractType(context, *ancestor), base,
                                op);
    };
    for (auto [memberIndex, member] : llvm::enumerate(groupMembers)) {
      bool exceptionShaped =
          member.values.size() == 3 &&
          (manifest.classId(runtimeContractName(member.contract)) ||
           RuntimeBundleLowerer::exceptionAncestorContractFor(member.contract));
      if (!exceptionShaped || !derivesFrom(member.contract, requiredBase))
        return op.emitError()
               << initContract << " member " << memberIndex << " has type "
               << member.contract << ", which is not derived from "
               << requiredBase;
      allMembersAreExceptions = allMembersAreExceptions &&
                                derivesFrom(member.contract, exceptionBase);
    }

    std::optional<EmittedRuntimeCall> messageInit;
    llvm::SmallVector<const RuntimeBundle *, 2> messageSources{instance,
                                                               sources[1]};
    if (mlir::failed(emitManifestMethodCall(op, *instance, *methodName,
                                            messageSources,
                                            /*allowUnusedSources=*/true,
                                            messageInit)))
      return mlir::failure();
    RuntimeBundle updatedInstance = *instance;
    if (messageInit->call.getNumResults() != 0 &&
        mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, op.getInstance().getType(), messageInit->call,
            updatedInstance)))
      return mlir::failure();

    std::optional<RuntimeSymbol> membersAlloc =
        manifest.primitive(initContract, "members_alloc");
    if (!membersAlloc)
      membersAlloc =
          manifest.primitive("builtins.BaseExceptionGroup", "members_alloc");
    std::optional<RuntimeSymbol> memberStore =
        manifest.primitive(initContract, "member_store");
    if (!memberStore)
      memberStore =
          manifest.primitive("builtins.BaseExceptionGroup", "member_store");
    if (!membersAlloc || !memberStore)
      return op.emitError() << "runtime manifest has no ExceptionGroup member "
                               "block primitives";

    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    auto adaptOperand = [&](mlir::Value value, mlir::Type want) -> mlir::Value {
      if (value.getType() == want)
        return value;
      return mlir::memref::CastOp::create(builder, loc, want, value)
          .getResult();
    };
    llvm::ArrayRef<mlir::Type> allocInputs =
        membersAlloc->function.getFunctionType().getInputs();
    if (allocInputs.size() != 4 ||
        updatedInstance.physicalValues().size() != 3)
      return op.emitError() << "ExceptionGroup members_alloc ABI mismatch";
    mlir::Value memberCount =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(groupMembers.size()), 64)
            .getResult();
    llvm::SmallVector<mlir::Value, 4> allocOperands;
    for (unsigned index = 0; index < 3; ++index)
      allocOperands.push_back(adaptOperand(
          updatedInstance.physicalValues()[index], allocInputs[index]));
    allocOperands.push_back(memberCount);
    mlir::func::CallOp allocCall = RuntimeBundleLowerer::createRuntimeCall(
        loc, *membersAlloc, allocOperands);
    if (allocCall.getNumResults() != 1)
      return op.emitError() << "ExceptionGroup members_alloc must return the "
                               "block word";
    mlir::Value blockWord = allocCall.getResult(0);
    llvm::ArrayRef<mlir::Type> storeInputs =
        memberStore->function.getFunctionType().getInputs();
    if (storeInputs.size() != 5)
      return op.emitError() << "ExceptionGroup member_store ABI mismatch";
    for (auto [memberIndex, member] : llvm::enumerate(groupMembers)) {
      mlir::Value slot =
          mlir::arith::ConstantIntOp::create(
              builder, loc, static_cast<std::int64_t>(memberIndex), 64)
              .getResult();
      llvm::SmallVector<mlir::Value, 8> storeOperands{blockWord, slot};
      for (unsigned index = 0; index < 3; ++index)
        storeOperands.push_back(
            adaptOperand(member.values[index], storeInputs[index + 2]));
      RuntimeBundleLowerer::createRuntimeCall(loc, *memberStore,
                                              storeOperands);
    }
    // CPython's BaseExceptionGroup.__new__ returns an ExceptionGroup when
    // every member is an Exception; the dynamic class word follows suit so
    // repr/except-matching see the derived class.
    if (initContract == "builtins.BaseExceptionGroup" &&
        allMembersAreExceptions) {
      std::optional<RuntimeSymbol> extSet =
          manifest.primitive("builtins.BaseException", "ext_set");
      if (!extSet)
        return op.emitError()
               << "runtime manifest has no BaseException ext_set primitive";
      mlir::Value classSlot =
          mlir::arith::ConstantIntOp::create(builder, loc, 2, 64).getResult();
      mlir::Value exceptionGroupId =
          mlir::arith::ConstantIntOp::create(builder, loc, 102, 64)
              .getResult();
      llvm::ArrayRef<mlir::Type> extInputs =
          extSet->function.getFunctionType().getInputs();
      if (extInputs.size() != 3)
        return op.emitError() << "BaseException ext_set ABI mismatch";
      RuntimeBundleLowerer::createRuntimeCall(
          loc, *extSet,
          {adaptOperand(updatedInstance.physicalValues()[0], extInputs[0]),
           classSlot, exceptionGroupId});
    }
    // A materialized member sequence must stay alive until every member has
    // been retained into the block — without a use here the refcount pass
    // releases the sequence (and the member references it owns) right after
    // its construction, before the stores above run.
    if (!memberSequence->physicalValues().empty() &&
        mlir::failed(pinContainerLiveness(op, *memberSequence)))
      return mlir::failure();

    valueBundles[op.getInstance()] = std::move(updatedInstance);
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  // Multi-value exception args: Exception("a", 1) fills the extended
  // payload block with the boxed values (the .args source) and renders
  // CPython's str(e) — "('a', 1)" — into the message lane. The former path
  // dropped every argument past the first through the unused-source hole.
  bool exceptionInit =
      *methodName == "__init__" &&
      (py::isAssignableTo(runtimeContractType(context, initContract),
                          runtimeContractType(context,
                                              "builtins.BaseException"),
                          op) ||
       (instanceClassOp &&
        RuntimeBundleLowerer::exceptionAncestorContract(instanceClassOp)));
  if (exceptionInit && !groupInit && sources.size() > 2) {
    if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
        mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
      return mlir::failure();
    std::optional<RuntimeSymbol> membersAlloc =
        manifest.primitive("builtins.BaseExceptionGroup", "members_alloc");
    std::optional<RuntimeSymbol> storeWords =
        manifest.primitive("builtins.BaseException", "payload_store_words");
    std::optional<RuntimeSymbol> initPayloadMessage =
        manifest.primitive("builtins.BaseException", "init_payload_message");
    if (!membersAlloc || !storeWords || !initPayloadMessage)
      return op.emitError() << "runtime manifest has no exception payload "
                               "primitives";
    if (instance->physicalValues().size() != 3)
      return op.emitError() << "multi-argument exception init requires the "
                               "3-value exception ABI";

    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    auto adaptOperand = [&](mlir::Value value, mlir::Type want) -> mlir::Value {
      if (value.getType() == want)
        return value;
      return mlir::memref::CastOp::create(builder, loc, want, value)
          .getResult();
    };
    llvm::ArrayRef<mlir::Type> allocInputs =
        membersAlloc->function.getFunctionType().getInputs();
    if (allocInputs.size() != 4)
      return op.emitError() << "exception payload alloc ABI mismatch";
    mlir::Value argCount =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(sources.size() - 1), 64)
            .getResult();
    llvm::SmallVector<mlir::Value, 4> allocOperands;
    for (unsigned index = 0; index < 3; ++index)
      allocOperands.push_back(
          adaptOperand(instance->physicalValues()[index], allocInputs[index]));
    allocOperands.push_back(argCount);
    mlir::func::CallOp allocCall = RuntimeBundleLowerer::createRuntimeCall(
        loc, *membersAlloc, allocOperands);
    if (allocCall.getNumResults() != 1)
      return op.emitError() << "exception payload alloc must return the "
                               "block word";
    mlir::Value blockWord = allocCall.getResult(0);

    for (unsigned index = 1; index < sources.size(); ++index) {
      const RuntimeBundle *argSource = sources[index];
      if (!argSource)
        return op.emitError()
               << "exception argument " << (index - 1) << " has no bundle";
      mlir::FailureOr<RuntimeBundle> payload =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, *argSource);
      if (mlir::failed(payload))
        return mlir::failure();
      if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, *payload, "exception.args")))
        return mlir::failure();
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
          RuntimeBundleLowerer::objectPayloadHandleWords(op, *payload);
      if (mlir::failed(words))
        return mlir::failure();
      builder.setInsertionPoint(op);
      mlir::Value slot = mlir::arith::ConstantIntOp::create(
                             builder, loc,
                             static_cast<std::int64_t>(index - 1), 64)
                             .getResult();
      llvm::SmallVector<mlir::Value, 18> storeOperands{blockWord, slot};
      storeOperands.append(words->begin(), words->end());
      RuntimeBundleLowerer::createRuntimeCall(loc, *storeWords, storeOperands);
      if (payload->objectValue.ownership == ownership::OwnershipKind::Own &&
          mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
              op, *payload, "exception.args.source")))
        return mlir::failure();
    }

    builder.setInsertionPoint(op);
    llvm::ArrayRef<mlir::Type> messageInputs =
        initPayloadMessage->function.getFunctionType().getInputs();
    if (messageInputs.size() != 3)
      return op.emitError() << "init_payload_message ABI mismatch";
    llvm::SmallVector<mlir::Value, 4> messageOperands;
    for (unsigned index = 0; index < 3; ++index)
      messageOperands.push_back(
          adaptOperand(instance->physicalValues()[index],
                       messageInputs[index]));
    mlir::func::CallOp messageCall = RuntimeBundleLowerer::createRuntimeCall(
        loc, *initPayloadMessage, messageOperands);
    RuntimeBundle updatedInstance;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, op.getInstance().getType(), messageCall, updatedInstance)))
      return mlir::failure();
    valueBundles[op.getInstance()] = std::move(updatedInstance);
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();
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

} // namespace py::lowering
