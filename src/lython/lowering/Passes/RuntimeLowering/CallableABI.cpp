#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

const RuntimeValueShape *
RuntimeBundleLowerer::runtimeValueShapeFor(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef purpose) const {
  std::string contract = runtimeShapeContractName(type);
  if (contract.empty()) {
    op->emitError() << purpose << " has no concrete runtime contract: " << type;
    return nullptr;
  }
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    op->emitError() << "runtime manifest has no ABI shape for " << contract
                    << " " << purpose;
  return shape;
}

py::ClassOp RuntimeBundleLowerer::classForContract(mlir::Type type) const {
  std::string contract = runtimeContractName(type);
  if (contract.empty())
    return {};
  mlir::ModuleOp mutableModule =
      const_cast<RuntimeBundleLowerer *>(this)->module;
  auto lookup = [&](llvm::StringRef name) -> py::ClassOp {
    return mlir::dyn_cast_or_null<py::ClassOp>(
        mlir::SymbolTable::lookupSymbolIn(mutableModule.getOperation(), name));
  };
  if (py::ClassOp classOp = lookup(contract))
    return classOp;

  llvm::StringRef shortName = llvm::StringRef(contract).rsplit('.').second;
  if (!shortName.empty() && shortName != contract)
    return lookup(shortName);
  return {};
}

bool RuntimeBundleLowerer::classDefinesMethod(mlir::Type type,
                                              llvm::StringRef name) const {
  py::ClassOp classOp = RuntimeBundleLowerer::classForContract(type);
  if (!classOp)
    return false;
  auto methodNames = classOp->getAttrOfType<mlir::ArrayAttr>("method_names");
  if (!methodNames)
    return false;
  for (mlir::Attribute attr : methodNames) {
    auto methodName = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (methodName && methodName.getValue() == name)
      return true;
  }
  return false;
}

llvm::SmallVector<mlir::Type, 8>
RuntimeBundleLowerer::classFieldContractTypes(py::ClassOp classOp) const {
  llvm::SmallVector<mlir::Type, 8> types;
  auto attrs = classOp->getAttrOfType<mlir::ArrayAttr>("field_contract_types");
  if (!attrs)
    attrs = classOp->getAttrOfType<mlir::ArrayAttr>("field_types");
  if (!attrs)
    return types;
  types.reserve(attrs.size());
  for (mlir::Attribute attr : attrs) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return {};
    types.push_back(typeAttr.getValue());
  }
  return types;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::runtimeValueTypesFor(mlir::Operation *op, mlir::Type type,
                                           llvm::StringRef purpose) const {
  if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
    llvm::SmallVector<mlir::Type, 8> types{mlir::IntegerType::get(context, 64)};
    for (mlir::Type member : unionType.getMemberTypes()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> memberTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, member, purpose);
      if (mlir::failed(memberTypes))
        return mlir::failure();
      types.append(memberTypes->begin(), memberTypes->end());
    }
    return types;
  }

  std::string contract = runtimeShapeContractName(type);
  if (!contract.empty()) {
    if (const RuntimeValueShape *shape = manifest.valueShape(contract))
      return llvm::SmallVector<mlir::Type, 8>(shape->valueTypes.begin(),
                                              shape->valueTypes.end());
  }

  if (py::ClassOp classOp = RuntimeBundleLowerer::classForContract(type)) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    llvm::SmallVector<mlir::Type, 8> types(objectShape->valueTypes.begin(),
                                           objectShape->valueTypes.end());
    for (mlir::Type fieldType :
         RuntimeBundleLowerer::classFieldContractTypes(classOp)) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldTypes =
          RuntimeBundleLowerer::runtimeValueTypesFor(op, fieldType, purpose);
      if (mlir::failed(fieldTypes))
        return mlir::failure();
      types.append(fieldTypes->begin(), fieldTypes->end());
    }
    return types;
  }

  if (!contract.empty()) {
    const RuntimeValueShape *objectShape =
        manifest.valueShape("builtins.object");
    if (!objectShape)
      return op->emitError()
             << "runtime manifest has no builtins.object ABI shape for "
             << purpose;
    return llvm::SmallVector<mlir::Type, 8>(objectShape->valueTypes.begin(),
                                            objectShape->valueTypes.end());
  }

  const RuntimeValueShape *shape =
      RuntimeBundleLowerer::runtimeValueShapeFor(op, type, purpose);
  if (!shape)
    return mlir::failure();
  return llvm::SmallVector<mlir::Type, 8>(shape->valueTypes.begin(),
                                          shape->valueTypes.end());
}

mlir::LogicalResult RuntimeBundleLowerer::appendRuntimeValueTypes(
    mlir::Operation *op, mlir::Type type,
    llvm::SmallVectorImpl<mlir::Type> &types) const {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, type, "callable ABI type");
  if (mlir::failed(valueTypes))
    return mlir::failure();
  types.append(valueTypes->begin(), valueTypes->end());
  return mlir::success();
}

bool RuntimeBundleLowerer::hasPrimitiveI64ABI(mlir::Type type) const {
  return runtimeContractName(type) == "builtins.int";
}

void RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(
    mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &types) const {
  if (!RuntimeBundleLowerer::hasPrimitiveI64ABI(type))
    return;
  types.push_back(mlir::IntegerType::get(context, 64));
  types.push_back(mlir::IntegerType::get(context, 1));
}

llvm::SmallVector<mlir::Type, 4>
RuntimeBundleLowerer::callableClosureTypes(mlir::func::FuncOp function) const {
  llvm::SmallVector<mlir::Type, 4> types;
  auto attr = function->getAttrOfType<mlir::ArrayAttr>("closure_types");
  if (!attr)
    return types;
  for (mlir::Attribute entry : attr) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(entry);
    if (!typeAttr)
      return {};
    types.push_back(typeAttr.getValue());
  }
  return types;
}

mlir::Type
RuntimeBundleLowerer::callableVarargValueType(mlir::func::FuncOp function,
                                              py::CallableType callable) const {
  if (auto attr =
          function->getAttrOfType<mlir::TypeAttr>(kCallableVarargValueTypeAttr))
    return attr.getValue();
  return callable.hasVararg() ? callable.getVarargType() : mlir::Type();
}

mlir::Type
RuntimeBundleLowerer::callableKwargValueType(mlir::func::FuncOp function,
                                             py::CallableType callable) const {
  if (auto attr =
          function->getAttrOfType<mlir::TypeAttr>(kCallableKwargValueTypeAttr))
    return attr.getValue();
  return callable.hasKwarg() ? callable.getKwargType() : mlir::Type();
}

llvm::SmallVector<mlir::Type, 8>
RuntimeBundleLowerer::callableLogicalInputTypes(
    mlir::func::FuncOp function, py::CallableType callable) const {
  llvm::SmallVector<mlir::Type, 8> logicalInputTypes(
      callable.getPositionalTypes().begin(),
      callable.getPositionalTypes().end());
  logicalInputTypes.append(callable.getKwOnlyTypes().begin(),
                           callable.getKwOnlyTypes().end());
  if (callable.hasVararg())
    logicalInputTypes.push_back(callableVarargValueType(function, callable));
  if (callable.hasKwarg())
    logicalInputTypes.push_back(callableKwargValueType(function, callable));
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      callableClosureTypes(function);
  logicalInputTypes.append(closureTypes.begin(), closureTypes.end());
  return logicalInputTypes;
}

mlir::LogicalResult RuntimeBundleLowerer::buildCallableProtocolArgumentABIs() {
  struct Accumulator {
    llvm::SmallVector<mlir::Type, 8> concreteTypes;
    llvm::SmallVector<bool, 8> conflicts;
  };

  llvm::StringMap<Accumulator> accumulators;
  module.walk([&](py::CallOp call) {
    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return mlir::WalkResult::advance();

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration())
      return mlir::WalkResult::advance();

    auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable)
      return mlir::WalkResult::advance();

    llvm::SmallVector<mlir::Type, 8> logicalTypes =
        callableLogicalInputTypes(target, callable);
    std::optional<llvm::SmallVector<mlir::Type, 4>> sourceTypes =
        RuntimeBundleLowerer::collectCallableArgumentSourceTypes(call,
                                                                 callable);
    if (!sourceTypes || sourceTypes->size() > logicalTypes.size())
      return mlir::WalkResult::advance();

    Accumulator &acc = accumulators[target.getSymName()];
    if (acc.concreteTypes.empty()) {
      acc.concreteTypes.resize(logicalTypes.size());
      acc.conflicts.resize(logicalTypes.size());
    }

    for (auto [index, sourceType] : llvm::enumerate(*sourceTypes)) {
      mlir::Type logicalType = logicalTypes[index];
      if (!mlir::isa<py::ProtocolType>(logicalType) ||
          !runtimeContractName(logicalType).empty())
        continue;

      std::string sourceContract = runtimeContractName(sourceType);
      if (sourceContract.empty())
        continue;

      mlir::Type &stored = acc.concreteTypes[index];
      if (!stored) {
        stored = sourceType;
        continue;
      }
      if (stored != sourceType)
        acc.conflicts[index] = true;
    }
    return mlir::WalkResult::advance();
  });

  for (auto &entry : accumulators) {
    Accumulator &acc = entry.getValue();
    for (auto [index, conflict] : llvm::enumerate(acc.conflicts))
      if (conflict)
        acc.concreteTypes[index] = {};
    if (llvm::any_of(acc.concreteTypes,
                     [](mlir::Type type) { return static_cast<bool>(type); }))
      callableProtocolArgumentABIs[entry.getKey()] =
          std::move(acc.concreteTypes);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::prepareCallableFunctionABIs() {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    auto callableType =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    if (!callableType)
      return mlir::WalkResult::advance();
    auto callable = mlir::dyn_cast<py::CallableType>(callableType.getValue());
    if (!callable) {
      function.emitError() << "callable_type must be Callable";
      result = mlir::failure();
      return mlir::WalkResult::interrupt();
    }
    llvm::SmallVector<mlir::Type, 8> logicalInputTypes =
        callableLogicalInputTypes(function, callable);
    llvm::SmallVector<mlir::Type, 8> abiInputTypes = logicalInputTypes;
    auto protocolEvidence =
        callableProtocolArgumentABIs.find(function.getSymName());
    if (protocolEvidence != callableProtocolArgumentABIs.end()) {
      llvm::SmallVector<mlir::Type, 8> &evidence = protocolEvidence->second;
      for (auto [index, type] : llvm::enumerate(evidence))
        if (index < abiInputTypes.size() && type)
          abiInputTypes[index] = type;
    }

    llvm::SmallVector<mlir::Type, 8> inputTypes;
    for (mlir::Type inputType : abiInputTypes) {
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, inputType, inputTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                            inputTypes);
    }
    const CallableArgumentEvidenceABI *argumentEvidence = nullptr;
    auto argumentEvidenceIt =
        callableArgumentEvidenceABIs.find(function.getSymName());
    if (argumentEvidenceIt != callableArgumentEvidenceABIs.end()) {
      argumentEvidence = &argumentEvidenceIt->second;
      for (const RuntimeArgumentEvidenceSet &evidenceSet :
           argumentEvidence->logicalArguments) {
        for (const RuntimeArgumentEvidence &evidence :
             evidenceSet.alternatives) {
          for (mlir::Type inputType : evidence.closureValueTypes) {
            if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                    function, inputType, inputTypes))) {
              result = mlir::failure();
              return mlir::WalkResult::interrupt();
            }
            RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                                  inputTypes);
          }
        }
      }
    }
    const CallableAggregateEvidenceABI *aggregateEvidence = nullptr;
    auto evidence = callableAggregateEvidenceABIs.find(function.getSymName());
    if (evidence != callableAggregateEvidenceABIs.end()) {
      aggregateEvidence = &evidence->second;
      for (mlir::Type inputType : aggregateEvidence->varargElementTypes) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, inputType, inputTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                              inputTypes);
      }
      for (mlir::Type inputType : aggregateEvidence->kwargValueTypes) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, inputType, inputTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(inputType,
                                                              inputTypes);
        if (aggregateEvidence->kwargIsFull)
          inputTypes.push_back(builder.getI1Type());
      }
    }

    llvm::SmallVector<mlir::Type, 8> resultTypes;
    auto returnedCoroutine =
        returnedCoroutineSummaries.find(function.getSymName());
    auto returnedObjectEvidence =
        returnedObjectEvidenceSummaries.find(function.getSymName());
    for (auto [logicalResultIndex, resultType] :
         llvm::enumerate(callable.getResultTypes())) {
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, resultType, resultTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(resultType,
                                                            resultTypes);
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          runtimeContractName(resultType) == "types.CoroutineType") {
        for (mlir::Type sourceType :
             returnedCoroutine->second.sourceContracts) {
          if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                  function, sourceType, resultTypes))) {
            result = mlir::failure();
            return mlir::WalkResult::interrupt();
          }
          RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(sourceType,
                                                                resultTypes);
        }
      }
      if (returnedObjectEvidence == returnedObjectEvidenceSummaries.end() ||
          returnedObjectEvidence->second.resultIndex != logicalResultIndex)
        continue;
      for (const ReturnedObjectEvidenceSlot &slot :
           returnedObjectEvidence->second.slots) {
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, slot.sourceContract, resultTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(
            slot.sourceContract, resultTypes);
      }
    }
    if (!function.isDeclaration()) {
      if (mlir::failed(seedCallableEntryArgumentBundles(
              function, logicalInputTypes, abiInputTypes, aggregateEvidence))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
    }
    function.setFunctionType(
        mlir::FunctionType::get(context, inputTypes, resultTypes));
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult RuntimeBundleLowerer::seedCallableEntryArgumentBundles(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes,
    mlir::ArrayRef<mlir::Type> abiTypes,
    const CallableAggregateEvidenceABI *aggregateEvidence) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "callable function entry argument count does not match "
              "callable_type";
  if (abiTypes.size() != logicalTypes.size())
    return function.emitError()
           << "callable ABI type count does not match callable_type";

  auto seedHiddenPrimitiveI64Evidence =
      [&](mlir::Type abiType, RuntimeBundle &bundle,
          mlir::Location loc) -> mlir::LogicalResult {
    if (!RuntimeBundleLowerer::hasPrimitiveI64ABI(abiType))
      return mlir::success();
    mlir::BlockArgument raw =
        entry.addArgument(mlir::IntegerType::get(context, 64), loc);
    mlir::BlockArgument valid =
        entry.addArgument(mlir::IntegerType::get(context, 1), loc);
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
    return mlir::success();
  };

  unsigned logicalArgCount = entry.getNumArguments();
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    mlir::Type abiType = abiTypes[index];
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(function, abiType,
                                                   "callable parameter ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> physicalArgs;
    for (mlir::Type physicalType : *valueTypes)
      physicalArgs.push_back(
          entry.addArgument(physicalType, logicalArg.getLoc()));

    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, abiType, physicalArgs, bundle)))
      return mlir::failure();
    if (mlir::failed(seedHiddenPrimitiveI64Evidence(abiType, bundle,
                                                   logicalArg.getLoc())))
      return mlir::failure();
    valueBundles[logicalArg] = std::move(bundle);
  }
  auto appendHiddenObject =
      [&](mlir::Type logicalType) -> mlir::FailureOr<RuntimeValue> {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            function, logicalType, "callable aggregate evidence ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 4> physicalArgs;
    for (mlir::Type physicalType : *valueTypes)
      physicalArgs.push_back(
          entry.addArgument(physicalType, function.getLoc()));

    RuntimeBundle bundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            function, logicalType, physicalArgs, bundle)))
      return mlir::failure();
    if (mlir::failed(seedHiddenPrimitiveI64Evidence(
            logicalType, bundle, function.getLoc())))
      return mlir::failure();
    return bundle.objectValue;
  };

  auto argumentEvidence =
      callableArgumentEvidenceABIs.find(function.getSymName());
  if (argumentEvidence != callableArgumentEvidenceABIs.end()) {
    for (auto [logicalIndex, evidenceSet] :
         llvm::enumerate(argumentEvidence->second.logicalArguments)) {
      if (logicalIndex >= logicalTypes.size() || evidenceSet.empty())
        continue;
      mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
      auto found = valueBundles.find(logicalArg);
      if (found == valueBundles.end() ||
          found->second.kind != RuntimeBundle::Kind::Object)
        return function.emitError()
               << "argument evidence logical argument has no object runtime "
                  "bundle";
      RuntimeBundle &bundle = found->second;
      if (evidenceSet.alternatives.size() == 1) {
        const RuntimeArgumentEvidence &evidence =
            evidenceSet.alternatives.front();
        bundle.functionTarget = evidence.functionTarget;
      }
      for (const RuntimeArgumentEvidence &evidence : evidenceSet.alternatives) {
        RuntimeCallableAlternative alternative;
        alternative.functionTarget = evidence.functionTarget;
        for (mlir::Type closureType : evidence.closureValueTypes) {
          mlir::FailureOr<RuntimeValue> closure =
              appendHiddenObject(closureType);
          if (mlir::failed(closure))
            return mlir::failure();
          alternative.closureValues.push_back(*closure);
        }
        if (evidenceSet.alternatives.size() == 1)
          bundle.closureValues = alternative.closureValues;
        bundle.callableAlternatives.push_back(std::move(alternative));
      }
    }
  }

  if (aggregateEvidence && aggregateEvidence->varargLogicalIndex) {
    unsigned logicalIndex = *aggregateEvidence->varargLogicalIndex;
    if (logicalIndex >= logicalTypes.size())
      return function.emitError()
             << "vararg aggregate evidence ABI references logical argument "
             << logicalIndex << ", but function has only "
             << logicalTypes.size() << " logical inputs";
    mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
    auto found = valueBundles.find(logicalArg);
    if (found == valueBundles.end() ||
        found->second.kind != RuntimeBundle::Kind::Object)
      return function.emitError()
             << "vararg aggregate evidence logical argument has no object "
                "runtime bundle";
    RuntimeBundle &bundle = found->second;
    bundle.sequenceIndices = aggregateEvidence->varargElementIndices;
    for (mlir::Type elementType : aggregateEvidence->varargElementTypes) {
      mlir::FailureOr<RuntimeValue> element = appendHiddenObject(elementType);
      if (mlir::failed(element))
        return mlir::failure();
      bundle.sequenceElements.push_back(*element);
    }
  }

  if (aggregateEvidence && aggregateEvidence->kwargLogicalIndex) {
    unsigned logicalIndex = *aggregateEvidence->kwargLogicalIndex;
    if (logicalIndex >= logicalTypes.size())
      return function.emitError()
             << "kwarg aggregate evidence ABI references logical argument "
             << logicalIndex << ", but function has only "
             << logicalTypes.size() << " logical inputs";
    mlir::BlockArgument logicalArg = entry.getArgument(logicalIndex);
    auto found = valueBundles.find(logicalArg);
    if (found == valueBundles.end() ||
        found->second.kind != RuntimeBundle::Kind::Object)
      return function.emitError()
             << "kwarg aggregate evidence logical argument has no object "
                "runtime bundle";
    RuntimeBundle &bundle = found->second;
    bundle.mappingKeys = aggregateEvidence->kwargKeys;
    for (mlir::Type valueType : aggregateEvidence->kwargValueTypes) {
      mlir::FailureOr<RuntimeValue> value = appendHiddenObject(valueType);
      if (mlir::failed(value))
        return mlir::failure();
      bundle.mappingValues.push_back(*value);
      if (aggregateEvidence->kwargIsFull) {
        mlir::BlockArgument present =
            entry.addArgument(builder.getI1Type(), function.getLoc());
        bundle.mappingPresent.push_back(present);
      }
    }
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

} // namespace py::runtime_lowering
