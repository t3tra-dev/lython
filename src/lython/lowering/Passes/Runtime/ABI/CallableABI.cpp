#include "Runtime/Core/Lowerer.h"

namespace py::lowering {
namespace {

bool isPrimitiveOnlyCallable(py::CallableType callable) {
  if (!callable || callable.hasVararg() || callable.hasKwarg())
    return false;
  auto isRuntimePrimitive = [](mlir::Type type) {
    return type && !py::isPyType(type);
  };
  return llvm::all_of(callable.getPositionalTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getKwOnlyTypes(), isRuntimePrimitive) &&
         llvm::all_of(callable.getResultTypes(), isRuntimePrimitive);
}

bool isCoroutineLikeResultType(mlir::Type type) {
  if (runtimeContractName(type) == "types.CoroutineType")
    return true;
  auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type);
  return protocol && protocol.getProtocolName() == "Coroutine";
}

bool isAwaitIteratorLikeResultType(mlir::Type type) {
  std::string contract = runtimeContractName(type);
  if (contract == "types.CoroutineAwaitIterator" ||
      contract == "_asyncio.FutureIter" || contract == "_asyncio.TaskIter")
    return true;
  auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type);
  return protocol && protocol.getProtocolName() == "Generator";
}

mlir::Type concreteCoroutineTypeForTarget(mlir::MLIRContext *context,
                                          mlir::func::FuncOp target) {
  auto bodyResult =
      target->getAttrOfType<mlir::TypeAttr>("ly.async.body_result");
  if (!bodyResult)
    return {};
  mlir::Type object = runtimeContractType(context, "builtins.object");
  return py::ContractType::get(context, "types.CoroutineType",
                               {object, object, bodyResult.getValue()});
}

bool hasProtocolArgumentOverride(llvm::ArrayRef<mlir::Type> types) {
  return llvm::any_of(types,
                      [](mlir::Type type) { return static_cast<bool>(type); });
}

bool sameProtocolArgumentOverrides(llvm::ArrayRef<mlir::Type> lhs,
                                   llvm::ArrayRef<mlir::Type> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto entry) {
           return std::get<0>(entry) == std::get<1>(entry);
         });
}

std::string protocolSpecializationName(llvm::StringRef originalName,
                                       unsigned ordinal) {
  return (llvm::Twine(originalName) + "__lyrt_proto_" + llvm::Twine(ordinal))
      .str();
}

} // namespace

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

std::optional<std::string>
RuntimeBundleLowerer::classMethodSymbol(py::ClassOp classOp,
                                        llvm::StringRef name) const {
  if (!classOp)
    return std::nullopt;
  auto methodNames = classOp->getAttrOfType<mlir::ArrayAttr>("method_names");
  auto methodSymbols =
      classOp->getAttrOfType<mlir::ArrayAttr>("method_symbols");
  if (!methodNames || !methodSymbols ||
      methodNames.size() != methodSymbols.size())
    return std::nullopt;
  for (auto [nameAttr, symbolAttr] : llvm::zip(methodNames, methodSymbols)) {
    auto methodName = mlir::dyn_cast<mlir::StringAttr>(nameAttr);
    auto symbol = mlir::dyn_cast<mlir::StringAttr>(symbolAttr);
    if (methodName && symbol && methodName.getValue() == name)
      return symbol.getValue().str();
  }
  return std::nullopt;
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
          RuntimeBundleLowerer::classFieldStorageValueTypes(op, fieldType,
                                                            purpose);
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

bool RuntimeBundleLowerer::isPrimitiveI64CallableClone(
    mlir::func::FuncOp function) const {
  return function && function->hasAttr(kPrimitiveI64CloneAttr);
}

bool RuntimeBundleLowerer::isCallableProtocolTemplate(
    mlir::func::FuncOp function) const {
  return function && function->hasAttr(kProtocolTemplateAttr);
}

std::optional<std::string>
RuntimeBundleLowerer::callableProtocolSpecializationFor(
    llvm::StringRef target,
    llvm::ArrayRef<const RuntimeBundle *> sources) const {
  auto found = callableProtocolSpecializations.find(target);
  if (found == callableProtocolSpecializations.end())
    return std::nullopt;

  for (const CallableProtocolSpecialization &specialization :
       found->second) {
    bool matches = true;
    for (auto [index, expected] :
         llvm::enumerate(specialization.argumentTypes)) {
      if (!expected)
        continue;
      if (index >= sources.size() || !sources[index]) {
        matches = false;
        break;
      }
      mlir::Type actual = sources[index]->contract;
      if (actual == expected)
        continue;
      if (py::isAssignableTo(actual, expected))
        continue;
      matches = false;
      break;
    }
    if (matches)
      return specialization.cloneName;
  }
  return std::nullopt;
}

mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::selectCallableProtocolSpecialization(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (std::optional<std::string> cloneName =
          RuntimeBundleLowerer::callableProtocolSpecializationFor(targetName,
                                                                  sources)) {
    if (mlir::func::FuncOp clone =
            module.lookupSymbol<mlir::func::FuncOp>(*cloneName))
      return clone;
    return op.emitError() << "protocol specialization clone @" << *cloneName
                          << " for callable target " << targetName
                          << " is not defined";
  }

  if (RuntimeBundleLowerer::isCallableProtocolTemplate(target))
    return op.emitError()
           << "protocol-typed callable target " << targetName
           << " has no static specialization for these argument contracts";
  return target;
}

std::optional<std::string>
RuntimeBundleLowerer::primitiveI64CloneFor(llvm::StringRef target) const {
  auto found = primitiveI64CallableClones.find(target);
  if (found == primitiveI64CallableClones.end())
    return std::nullopt;
  return found->second;
}

bool RuntimeBundleLowerer::isPrimitiveI64CallableEligible(
    mlir::func::FuncOp function) const {
  if (!function || function.isDeclaration() ||
      RuntimeBundleLowerer::isPrimitiveI64CallableClone(function) ||
      RuntimeBundleLowerer::isCallableProtocolTemplate(function))
    return false;
  if (!RuntimeBundleLowerer::callableClosureTypes(function).empty())
    return false;

  auto callableAttr = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable || callable.getResultTypes().size() != 1 ||
      callable.hasVararg() || callable.hasKwarg() ||
      !callable.getKwOnlyTypes().empty() ||
      llvm::any_of(callable.getPositionalDefaults(),
                   [](mlir::BoolAttr attr) { return attr && attr.getValue(); }))
    return false;
  if (runtimeContractName(callable.getResultTypes().front()) != "builtins.int")
    return false;
  if (!llvm::all_of(callable.getPositionalTypes(), [](mlir::Type type) {
        return runtimeContractName(type) == "builtins.int";
      }))
    return false;
  // A return value that is a block argument comes from a control-flow merge
  // (if/loop) and is a boxed object, which the unboxed-i64 clone return ABI
  // cannot represent. Such functions must stay on the boxed-int path.
  bool returnsBlockArgument = false;
  function.walk([&](mlir::func::ReturnOp returnOp) {
    for (mlir::Value operand : returnOp.getOperands())
      if (mlir::isa<mlir::BlockArgument>(operand))
        returnsBlockArgument = true;
  });
  return !returnsBlockArgument;
}

mlir::LogicalResult RuntimeBundleLowerer::buildPrimitiveI64CallableClones() {
  llvm::SmallVector<mlir::func::FuncOp, 8> originals;
  module.walk([&](mlir::func::FuncOp function) {
    if (RuntimeBundleLowerer::isPrimitiveI64CallableEligible(function))
      originals.push_back(function);
  });

  for (mlir::func::FuncOp original : originals) {
    std::string originalName = original.getSymName().str();
    std::string cloneName = (original.getSymName() + "__lyrt_prim_i64").str();
    if (module.lookupSymbol<mlir::func::FuncOp>(cloneName)) {
      primitiveI64CallableClones[originalName] = cloneName;
      continue;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(original);
    mlir::func::FuncOp clone = original.clone();
    clone.setSymName(cloneName);
    clone->setAttr(kPrimitiveI64CloneAttr,
                   builder.getStringAttr(original.getSymName()));
    mlir::SymbolTable::setSymbolVisibility(
        clone, mlir::SymbolTable::Visibility::Private);
    builder.insert(clone);
    primitiveI64CallableClones[originalName] = cloneName;
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::seedPrimitiveI64CallableEntryArgumentBundles(
    mlir::func::FuncOp function, mlir::ArrayRef<mlir::Type> logicalTypes) {
  if (function.isDeclaration())
    return mlir::success();
  mlir::Block &entry = function.getBody().front();
  if (entry.getNumArguments() != logicalTypes.size())
    return function.emitError()
           << "primitive i64 callable clone entry argument count does not "
              "match callable_type";

  unsigned logicalArgCount = entry.getNumArguments();
  for (auto [index, logicalType] : llvm::enumerate(logicalTypes)) {
    if (runtimeContractName(logicalType) != "builtins.int")
      return function.emitError()
             << "primitive i64 callable clone argument " << index
             << " must be builtins.int, got " << logicalType;
    mlir::BlockArgument logicalArg = entry.getArgument(index);
    mlir::BlockArgument raw = entry.addArgument(
        mlir::IntegerType::get(context, 64), logicalArg.getLoc());
    mlir::BlockArgument valid = entry.addArgument(
        mlir::IntegerType::get(context, 1), logicalArg.getLoc());

    RuntimeBundle bundle = RuntimeBundle::objectWithOwnership(
        logicalType, mlir::ValueRange{},
        ownership::logicalOwnershipKind(logicalType,
                                                /*ownsObject=*/false));
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
    valueBundles[logicalArg] = std::move(bundle);
  }
  callableLogicalEntryArgCounts.push_back(
      CallableLogicalEntryArgs{function, logicalArgCount});
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildCallableProtocolArgumentABIs() {
  struct Accumulator {
    llvm::SmallVector<llvm::SmallVector<mlir::Type, 8>, 4> specializations;
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

    llvm::SmallVector<mlir::Type, 8> argumentTypes(logicalTypes.size());
    for (auto [index, sourceType] : llvm::enumerate(*sourceTypes)) {
      mlir::Type logicalType = logicalTypes[index];
      if (!mlir::isa<py::ProtocolType>(logicalType) ||
          !runtimeContractName(logicalType).empty())
        continue;

      std::string sourceContract = runtimeContractName(sourceType);
      if (sourceContract.empty())
        continue;
      argumentTypes[index] = sourceType;
    }
    if (!hasProtocolArgumentOverride(argumentTypes))
      return mlir::WalkResult::advance();

    Accumulator &acc = accumulators[target.getSymName()];
    if (llvm::none_of(acc.specializations, [&](llvm::ArrayRef<mlir::Type> item) {
          return sameProtocolArgumentOverrides(item, argumentTypes);
        }))
      acc.specializations.push_back(std::move(argumentTypes));
    return mlir::WalkResult::advance();
  });

  for (auto &entry : accumulators) {
    mlir::func::FuncOp original =
        module.lookupSymbol<mlir::func::FuncOp>(entry.getKey());
    if (!original || original.isDeclaration())
      continue;

    llvm::SmallVector<CallableProtocolSpecialization, 4> &specializations =
        callableProtocolSpecializations[entry.getKey()];
    for (auto [ordinal, argumentTypes] :
         llvm::enumerate(entry.getValue().specializations)) {
      std::string cloneName =
          protocolSpecializationName(entry.getKey(), ordinal);
      mlir::func::FuncOp clone =
          module.lookupSymbol<mlir::func::FuncOp>(cloneName);
      if (!clone) {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(original);
        clone = original.clone();
        clone.setSymName(cloneName);
        clone->setAttr(kProtocolSpecializationAttr,
                       builder.getStringAttr(original.getSymName()));
        mlir::SymbolTable::setSymbolVisibility(
            clone, mlir::SymbolTable::Visibility::Private);
        builder.insert(clone);
      }
      callableProtocolArgumentABIs[cloneName] = argumentTypes;
      if (auto returnedValue = returnedValueSummaries.find(entry.getKey());
          returnedValue != returnedValueSummaries.end())
        returnedValueSummaries[cloneName] = returnedValue->second;
      if (auto returnedCallable =
              returnedCallableSummaries.find(entry.getKey());
          returnedCallable != returnedCallableSummaries.end())
        returnedCallableSummaries[cloneName] = returnedCallable->second;
      specializations.push_back(CallableProtocolSpecialization{
          cloneName, llvm::SmallVector<mlir::Type, 8>(argumentTypes)});
    }
    if (!specializations.empty())
      original->setAttr(kProtocolTemplateAttr, builder.getUnitAttr());
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
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      return mlir::WalkResult::advance();
    if (isPrimitiveOnlyCallable(callable))
      return mlir::WalkResult::advance();
    llvm::SmallVector<mlir::Type, 8> logicalInputTypes =
        callableLogicalInputTypes(function, callable);
    if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function)) {
      llvm::SmallVector<mlir::Type, 8> inputTypes;
      for (mlir::Type logicalType : logicalInputTypes) {
        if (runtimeContractName(logicalType) != "builtins.int") {
          function.emitError()
              << "primitive i64 callable clone parameter must be builtins.int";
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(logicalType,
                                                              inputTypes);
      }
      if (callable.getResultTypes().empty() ||
          !llvm::all_of(callable.getResultTypes(), [](mlir::Type type) {
            return runtimeContractName(type) == "builtins.int";
          })) {
        function.emitError()
            << "primitive i64 callable clone results must be builtins.int";
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      llvm::SmallVector<mlir::Type, 8> resultTypes;
      for (mlir::Type resultType : callable.getResultTypes())
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(resultType,
                                                              resultTypes);
      if (!function.isDeclaration() &&
          mlir::failed(seedPrimitiveI64CallableEntryArgumentBundles(
              function, logicalInputTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      function.setFunctionType(
          mlir::FunctionType::get(context, inputTypes, resultTypes));
      return mlir::WalkResult::advance();
    }

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
          for (mlir::Type inputType : evidence.coroutineSourceTypes) {
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
    llvm::SmallVector<std::int64_t, 4> ownedResultOffsets;
    llvm::SmallVector<mlir::Attribute, 4> ownedResultContracts;
    auto returnedCoroutine =
        returnedCoroutineSummaries.find(function.getSymName());
    auto returnedObjectEvidence =
        returnedObjectEvidenceSummaries.find(function.getSymName());
    auto returnedStaticObject =
        returnedStaticObjectSummaries.find(function.getSymName());
    for (auto [logicalResultIndex, resultType] :
         llvm::enumerate(callable.getResultTypes())) {
      mlir::Type abiResultType = resultType;
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          isCoroutineLikeResultType(resultType)) {
        if (mlir::func::FuncOp target =
                module.lookupSymbol<mlir::func::FuncOp>(
                    returnedCoroutine->second.target)) {
          if (mlir::Type concrete =
                  concreteCoroutineTypeForTarget(context, target))
            abiResultType = concrete;
        }
      }
      bool protocolPrimaryOwnsResult = false;
      if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(
              resultType))
        protocolPrimaryOwnsResult =
            runtimeShapeContractName(resultType) == "builtins.object" &&
            ((returnedStaticObject != returnedStaticObjectSummaries.end() &&
              returnedStaticObject->second.resultIndex == logicalResultIndex) ||
             protocol.getProtocolName() == "Generator");
      if (protocolPrimaryOwnsResult) {
        ownedResultOffsets.push_back(
            static_cast<std::int64_t>(resultTypes.size()));
        ownedResultContracts.push_back(builder.getStringAttr("builtins.object"));
      }
      if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
              function, abiResultType, resultTypes))) {
        result = mlir::failure();
        return mlir::WalkResult::interrupt();
      }
      RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(abiResultType,
                                                            resultTypes);
      if (returnedStaticObject != returnedStaticObjectSummaries.end() &&
          returnedStaticObject->second.resultIndex == logicalResultIndex) {
        mlir::Type objectContract =
            returnedStaticObject->second.objectContract;
        std::string objectContractName = runtimeContractName(objectContract);
        if (objectContractName.empty()) {
          result = function.emitError()
                   << "static returned object evidence has no runtime "
                      "contract: "
                   << objectContract;
          return mlir::WalkResult::interrupt();
        }
        ownedResultOffsets.push_back(
            static_cast<std::int64_t>(resultTypes.size()));
        ownedResultContracts.push_back(
            builder.getStringAttr(objectContractName));
        if (mlir::failed(RuntimeBundleLowerer::appendRuntimeValueTypes(
                function, objectContract, resultTypes))) {
          result = mlir::failure();
          return mlir::WalkResult::interrupt();
        }
        RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(objectContract,
                                                              resultTypes);
      }
      if (returnedCoroutine != returnedCoroutineSummaries.end() &&
          (isCoroutineLikeResultType(resultType) ||
           isAwaitIteratorLikeResultType(resultType))) {
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
    if (!ownedResultOffsets.empty())
      function->setAttr(
          ownership::kOwnedResultsAttr,
          mlir::DenseI64ArrayAttr::get(context, ownedResultOffsets));
    if (!ownedResultContracts.empty())
      function->setAttr(ownership::kOwnedResultContractsAttr,
                        builder.getArrayAttr(ownedResultContracts));
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
            function, abiType, physicalArgs, bundle,
            /*ownsObject=*/false)))
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
            function, logicalType, physicalArgs, bundle,
            /*ownsObject=*/false)))
      return mlir::failure();
    if (mlir::failed(seedHiddenPrimitiveI64Evidence(logicalType, bundle,
                                                    function.getLoc())))
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
        if (!evidence.functionTarget.empty())
          bundle.functionTarget = evidence.functionTarget;
        if (!evidence.coroutineTarget.empty())
          bundle.coroutineTarget = evidence.coroutineTarget;
      }
      for (const RuntimeArgumentEvidence &evidence : evidenceSet.alternatives) {
        if (!evidence.functionTarget.empty() ||
            !evidence.closureValueTypes.empty()) {
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
        if (!evidence.coroutineTarget.empty()) {
          llvm::SmallVector<RuntimeValue, 4> coroutineSources;
          llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 4>
              coroutineSourceBundles;
          for (mlir::Type sourceType : evidence.coroutineSourceTypes) {
            mlir::FailureOr<RuntimeValue> source =
                appendHiddenObject(sourceType);
            if (mlir::failed(source))
              return mlir::failure();
            coroutineSources.push_back(*source);
            coroutineSourceBundles.push_back(std::make_shared<RuntimeBundle>(
                RuntimeBundle::object(source->contract, source->values)));
          }
          if (evidenceSet.alternatives.size() == 1) {
            bundle.coroutineSources = std::move(coroutineSources);
            bundle.coroutineSourceBundles =
                std::move(coroutineSourceBundles);
          }
        }
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

} // namespace py::lowering
