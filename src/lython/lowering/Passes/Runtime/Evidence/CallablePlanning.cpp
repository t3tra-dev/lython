#include "Runtime/Core/Lowerer.h"

#include "Runtime/Evidence/Callable.h"

#include "PyCallableShape.h"
#include "PyProtocols.h"

namespace py::lowering {

namespace {

bool callableDefaultAt(py::CallableType callable, unsigned index) {
  llvm::ArrayRef<mlir::Type> positional = callable.getPositionalTypes();
  if (index < positional.size()) {
    llvm::ArrayRef<mlir::BoolAttr> defaults = callable.getPositionalDefaults();
    return index < defaults.size() && defaults[index].getValue();
  }
  llvm::ArrayRef<mlir::BoolAttr> defaults = callable.getKwOnlyDefaults();
  unsigned kwIndex = index - static_cast<unsigned>(positional.size());
  return kwIndex < defaults.size() && defaults[kwIndex].getValue();
}

llvm::SmallVector<mlir::Type, 8>
fixedCallableParameterTypes(py::CallableType callable) {
  llvm::SmallVector<mlir::Type, 8> parameterTypes(
      callable.getPositionalTypes().begin(),
      callable.getPositionalTypes().end());
  parameterTypes.append(callable.getKwOnlyTypes().begin(),
                        callable.getKwOnlyTypes().end());
  return parameterTypes;
}

bool aggregateOperandIsUnpacked(const RuntimeBundle &aggregate,
                                unsigned index) {
  return index < aggregate.aggregateUnpackedOperands.size() &&
         aggregate.aggregateUnpackedOperands[index] != 0;
}

void appendStaticUnpackedPositionalTypes(
    mlir::Type type, llvm::SmallVectorImpl<mlir::Type> &positionalTypes,
    llvm::SmallVectorImpl<mlir::Type> &actualTypes) {
  lython::callable::VarargShape<mlir::Type> shape =
      py::callableVarargShape(type);
  if (shape.valid()) {
    if (shape.kind ==
        lython::callable::VarargShape<mlir::Type>::Kind::Repeated) {
      positionalTypes.push_back(*shape.repeated);
      actualTypes.push_back(*shape.repeated);
      return;
    }
    positionalTypes.append(shape.exact.begin(), shape.exact.end());
    actualTypes.append(shape.exact.begin(), shape.exact.end());
    return;
  }
  positionalTypes.push_back(type);
  actualTypes.push_back(type);
}

bool containsStaticParameter(mlir::Type type) {
  if (!type)
    return false;
  if (mlir::isa<py::TypeVarType, py::ParamSpecType, py::TypeVarTupleType>(type))
    return true;
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type))
    return containsStaticParameter(unpack.getPackedType());
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return llvm::any_of(contract.getArguments(), containsStaticParameter);
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type))
    return llvm::any_of(protocol.getArguments(), containsStaticParameter);
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type))
    return llvm::any_of(unionType.getMemberTypes(), containsStaticParameter);
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return containsStaticParameter(typeType.getInstanceType());
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type)) {
    return llvm::any_of(callable.getPositionalTypes(),
                        containsStaticParameter) ||
           llvm::any_of(callable.getKwOnlyTypes(), containsStaticParameter) ||
           llvm::any_of(callable.getResultTypes(), containsStaticParameter) ||
           (callable.hasVararg() &&
            containsStaticParameter(callable.getVarargType())) ||
           (callable.hasKwarg() &&
            containsStaticParameter(callable.getKwargType()));
  }
  return false;
}

bool structuralProtocolArgumentAccepts(mlir::Type actual,
                                       py::ProtocolType expected) {
  return py::protocols::Table::get(*expected.getContext())
      .structurallyAccepts(actual, expected.getProtocolName(),
                           expected.getArguments());
}

bool callableArgumentAccepts(mlir::Type actual, mlir::Type expected,
                             mlir::Operation *op, bool emitErrors) {
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(expected))
    if (structuralProtocolArgumentAccepts(actual, protocol))
      return true;
  // ⛔ THE OP IS THE SYMBOL-LOOKUP ANCHOR, NOT A DIAGNOSTIC SINK. Passing it
  // only when errors were being emitted made the LATTICE depend on that flag:
  // without it a source class has no `py.class` to walk, so `f(MyErr("x"))`
  // against `def f(e: Exception)` found no plan on the silent pass and then
  // reported "arguments do not match Callable contract" on the loud one --
  // which is the same question answered two ways.
  return py::isAssignableTo(actual, expected, op);
}

std::optional<CallableArgumentPlan>
buildCallableArgumentPlan(mlir::Operation *op, py::CallableType callable,
                          llvm::ArrayRef<mlir::Type> positionalTypes,
                          llvm::ArrayRef<PlannedKeywordArgument> keywords,
                          llvm::StringRef targetName, bool emitErrors) {
  auto fail = [&](llvm::Twine message) -> std::optional<CallableArgumentPlan> {
    if (emitErrors)
      op->emitError() << message;
    return std::nullopt;
  };

  std::optional<py::CallableSignatureShape> shape =
      py::callableSignatureShape(callable);
  if (!shape)
    return fail(llvm::Twine("callable_type is malformed for function target ") +
                targetName);

  llvm::SmallVector<py::CallableKeyword, 8> typeKeywords;
  typeKeywords.reserve(keywords.size());
  for (const PlannedKeywordArgument &keyword : keywords)
    typeKeywords.push_back(py::CallableKeyword{keyword.name, keyword.type});

  auto resolution = py::resolveCallableApplicationShape(
      *shape, llvm::ArrayRef<mlir::Type>(positionalTypes),
      llvm::ArrayRef<py::CallableKeyword>(typeKeywords),
      [&](mlir::Type expected, mlir::Type actual) {
        if (containsStaticParameter(expected))
          return true;
        return callableArgumentAccepts(actual, expected, op, emitErrors);
      },
      [](const py::CallableKeyword &keyword) -> llvm::StringRef {
        return keyword.name;
      },
      [](const py::CallableKeyword &keyword) { return keyword.type; });
  if (!resolution)
    return fail(llvm::Twine("arguments do not match Callable contract for "
                            "function target ") +
                targetName);

  CallableArgumentPlan plan;
  const unsigned positionalCount =
      static_cast<unsigned>(shape->positional.size());
  const unsigned fixedCount =
      positionalCount + static_cast<unsigned>(shape->kwonly.size());
  plan.fixedActuals.resize(fixedCount);

  const unsigned directCount = std::min<unsigned>(
      static_cast<unsigned>(positionalTypes.size()), positionalCount);
  for (unsigned index = 0; index < directCount; ++index)
    plan.fixedActuals[index] = index;

  for (unsigned index = positionalCount, end = positionalTypes.size();
       index < end; ++index)
    plan.varargActuals.push_back(index);

  llvm::StringMap<unsigned> positionalNameToIndex;
  for (auto [index, name] : llvm::enumerate(shape->positionalNames)) {
    if (index < shape->positional.size() &&
        index >= shape->positionalOnlyCount && !name.empty())
      positionalNameToIndex[name] = static_cast<unsigned>(index);
  }

  llvm::StringMap<unsigned> kwonlyNameToIndex;
  for (auto [index, name] : llvm::enumerate(shape->kwonlyNames)) {
    if (index < shape->kwonly.size() && !name.empty())
      kwonlyNameToIndex[name] = positionalCount + static_cast<unsigned>(index);
  }

  for (const PlannedKeywordArgument &keyword : keywords) {
    if (keyword.name.empty()) {
      plan.kwargActuals.push_back(keyword.actualIndex);
      continue;
    }
    auto positional = positionalNameToIndex.find(keyword.name);
    if (positional != positionalNameToIndex.end()) {
      plan.fixedActuals[positional->second] = keyword.actualIndex;
      continue;
    }
    auto kwonly = kwonlyNameToIndex.find(keyword.name);
    if (kwonly != kwonlyNameToIndex.end()) {
      plan.fixedActuals[kwonly->second] = keyword.actualIndex;
      continue;
    }
    plan.kwargActuals.push_back(keyword.actualIndex);
  }

  for (unsigned index = 0; index < fixedCount; ++index) {
    if (plan.fixedActuals[index])
      continue;
    if (!callableDefaultAt(callable, index))
      return fail(
          llvm::Twine("missing required argument for function target ") +
          targetName);
    plan.defaultedFixed.push_back(index);
  }
  return plan;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::collectFunctionCallSources(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
    const RuntimeBundle *callableObject) {
  auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
  if (!callableAttr)
    return op.emitError() << "function target '" << targetName
                          << "' has no callable_type";
  auto callable = mlir::dyn_cast<py::CallableType>(callableAttr.getValue());
  if (!callable)
    return op.emitError() << "function target '" << targetName
                          << "' callable_type is not Callable";

  const RuntimeBundle *posargs =
      RuntimeBundleLowerer::bundleFor(op.getPosargs());
  if (!posargs || posargs->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "positional args must be a lowered aggregate "
                             "bundle";
  std::size_t storageReserve =
      callable.getPositionalTypes().size() + callable.getKwOnlyTypes().size() +
      (callable.hasVararg() ? 1 : 0) + (callable.hasKwarg() ? 1 : 0);
  for (auto [index, operand] : llvm::enumerate(posargs->aggregateOperands)) {
    (void)operand;
    if (!aggregateOperandIsUnpacked(*posargs, static_cast<unsigned>(index)))
      continue;
    const RuntimeBundle *source =
        RuntimeBundleLowerer::bundleFor(posargs->aggregateOperands[index]);
    if (source)
      storageReserve += source->sequenceElements.size();
  }
  materializedDefaults.reserve(storageReserve);

  llvm::SmallVector<mlir::Type, 8> positionalTypes;
  llvm::SmallVector<const RuntimeBundle *, 8> actualSources;
  // The SSA value each actual came from, parallel to `actualSources`. Null for
  // an element taken out of a starred operand, which has no operand of its own
  // -- the same convention `lowerPack` uses for a key it minted itself.
  llvm::SmallVector<mlir::Value, 8> actualSourceValues;
  positionalTypes.reserve(posargs->aggregateOperands.size());
  actualSources.reserve(posargs->aggregateOperands.size());
  actualSourceValues.reserve(posargs->aggregateOperands.size());
  for (auto [index, operand] : llvm::enumerate(posargs->aggregateOperands)) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(operand);
    if (!source)
      return op.emitError()
             << "positional arg operand has no lowered runtime bundle";
    if (aggregateOperandIsUnpacked(*posargs, static_cast<unsigned>(index))) {
      mlir::FailureOr<llvm::SmallVector<RuntimeValue, 4>> elements =
          RuntimeBundleLowerer::starredSequenceElements(
              op, *source,
              ("starred positional arg operand for function target " +
               targetName)
                  .str());
      if (mlir::failed(elements))
        return mlir::failure();
      for (const RuntimeValue &element : *elements) {
        materializedDefaults.push_back(
            RuntimeBundle::object(element.contract, element.values));
        positionalTypes.push_back(element.contract);
        actualSources.push_back(&materializedDefaults.back());
        actualSourceValues.push_back(mlir::Value{});
      }
      continue;
    }
    positionalTypes.push_back(operand.getType());
    actualSources.push_back(source);
    actualSourceValues.push_back(operand);
  }

  const RuntimeBundle *kwNames =
      RuntimeBundleLowerer::bundleFor(op.getKwnames());
  const RuntimeBundle *kwValues =
      RuntimeBundleLowerer::bundleFor(op.getKwvalues());
  if (!kwNames || kwNames->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "kw names must be a lowered aggregate bundle";
  if (!kwValues || kwValues->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "kw values must be a lowered aggregate bundle";
  if (kwNames->aggregateOperands.size() != kwValues->aggregateOperands.size())
    return op.emitError() << "kw names and kw values must have the same size";

  llvm::SmallVector<PlannedKeywordArgument, 8> keywords;
  keywords.reserve(kwNames->aggregateOperands.size());
  for (auto [index, nameValue] : llvm::enumerate(kwNames->aggregateOperands)) {
    std::optional<std::string> keyword =
        RuntimeBundleLowerer::keywordNameFromValue(nameValue);
    if (!keyword)
      return op.emitError()
             << "function keyword name must be a static string literal";

    const RuntimeBundle *source =
        RuntimeBundleLowerer::bundleFor(kwValues->aggregateOperands[index]);
    if (!source)
      return op.emitError()
             << "keyword value operand has no lowered runtime bundle";

    unsigned actualIndex = static_cast<unsigned>(actualSources.size());
    actualSources.push_back(source);
    actualSourceValues.push_back(kwValues->aggregateOperands[index]);
    keywords.push_back(
        PlannedKeywordArgument{std::move(*keyword), actualIndex,
                               kwValues->aggregateOperands[index].getType()});
  }

  std::optional<CallableArgumentPlan> plan = buildCallableArgumentPlan(
      op, callable, positionalTypes, keywords, targetName,
      /*emitErrors=*/true);
  if (!plan)
    return mlir::failure();

  llvm::SmallVector<mlir::Type, 8> parameters =
      fixedCallableParameterTypes(callable);
  for (unsigned index = 0, end = parameters.size(); index < end; ++index) {
    if (plan->fixedActuals[index]) {
      unsigned actualIndex = *plan->fixedActuals[index];
      if (actualIndex >= actualSources.size())
        return op.emitError() << "argument planner produced an invalid source "
                                 "index for function target "
                              << targetName;
      sources.push_back(actualSources[actualIndex]);
      continue;
    }
    const RuntimeBundle *source = nullptr;
    if (mlir::failed(materializeDefaultArgument(
            op, target, targetName, index, parameters[index],
            materializedDefaults, source, callableObject)))
      return mlir::failure();
    sources.push_back(source);
  }

  if (callable.hasVararg()) {
    mlir::Type varargValueType =
        RuntimeBundleLowerer::callableVarargValueType(target, callable);
    if (!varargValueType)
      return op.emitError() << "function target " << targetName
                            << " has no vararg runtime value type";
    RuntimeBundle bundle;
    llvm::SmallVector<RuntimeValue, 8> elements;
    llvm::SmallVector<const RuntimeBundle *, 8> elementBundles;
    llvm::SmallVector<mlir::Value, 8> elementSources;
    elements.reserve(plan->varargActuals.size());
    elementBundles.reserve(plan->varargActuals.size());
    elementSources.reserve(plan->varargActuals.size());
    for (unsigned actualIndex : plan->varargActuals) {
      if (actualIndex >= actualSources.size() ||
          actualSources[actualIndex]->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "*args source for function target " << targetName
               << " must be a lowered object bundle";
      elements.push_back(actualSources[actualIndex]->objectValue);
      elementBundles.push_back(actualSources[actualIndex]);
      elementSources.push_back(actualSourceValues[actualIndex]);
    }
    if (mlir::failed(materializeArityObject(
            op, varargValueType, plan->varargActuals.size(), bundle, elements,
            /*keys=*/{}, elementBundles, elementSources)))
      return mlir::failure();
    materializedDefaults.push_back(std::move(bundle));
    sources.push_back(&materializedDefaults.back());
  }
  if (callable.hasKwarg()) {
    mlir::Type kwargValueType =
        RuntimeBundleLowerer::callableKwargValueType(target, callable);
    if (!kwargValueType)
      return op.emitError() << "function target " << targetName
                            << " has no kwarg runtime value type";
    RuntimeBundle bundle;
    llvm::SmallVector<RuntimeValue, 8> values;
    llvm::SmallVector<std::string, 8> keys;
    llvm::SmallVector<const RuntimeBundle *, 8> valueBundlesForKwargs;
    llvm::SmallVector<mlir::Value, 8> valueSources;
    values.reserve(plan->kwargActuals.size());
    keys.reserve(plan->kwargActuals.size());
    valueBundlesForKwargs.reserve(plan->kwargActuals.size());
    valueSources.reserve(plan->kwargActuals.size());
    for (unsigned actualIndex : plan->kwargActuals) {
      auto keyword =
          llvm::find_if(keywords, [&](const PlannedKeywordArgument &candidate) {
            return candidate.actualIndex == actualIndex;
          });
      if (keyword == keywords.end() || keyword->name.empty())
        return op.emitError() << "**kwargs source for function target "
                              << targetName << " needs a static keyword name";
      if (actualIndex >= actualSources.size() ||
          actualSources[actualIndex]->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "**kwargs source for function target " << targetName
               << " must be a lowered object bundle";
      keys.push_back(keyword->name);
      values.push_back(actualSources[actualIndex]->objectValue);
      valueBundlesForKwargs.push_back(actualSources[actualIndex]);
      valueSources.push_back(actualSourceValues[actualIndex]);
    }
    if (mlir::failed(materializeArityObject(
            op, kwargValueType, plan->kwargActuals.size(), bundle, values, keys,
            valueBundlesForKwargs, valueSources)))
      return mlir::failure();
    materializedDefaults.push_back(std::move(bundle));
    sources.push_back(&materializedDefaults.back());
  }
  return mlir::success();
}

std::optional<StaticCallableInvocation>
RuntimeBundleLowerer::collectStaticCallableInvocation(py::CallOp op) const {
  auto posargs = op.getPosargs().getDefiningOp<py::PackOp>();
  auto kwNames = op.getKwnames().getDefiningOp<py::PackOp>();
  auto kwValues = op.getKwvalues().getDefiningOp<py::PackOp>();
  if (!posargs || !kwNames || !kwValues)
    return std::nullopt;
  if (kwNames.getValues().size() != kwValues.getValues().size())
    return std::nullopt;

  StaticCallableInvocation invocation;
  invocation.positionalTypes.reserve(posargs.getValues().size());
  invocation.actualTypes.reserve(posargs.getValues().size() +
                                 kwValues.getValues().size());
  invocation.actualValues.reserve(posargs.getValues().size() +
                                  kwValues.getValues().size());
  for (auto [index, value] : llvm::enumerate(posargs.getValues())) {
    if (callable_evidence::packOperandIsUnpacked(posargs, static_cast<unsigned>(index))) {
      std::size_t before = invocation.actualTypes.size();
      appendStaticUnpackedPositionalTypes(
          value.getType(), invocation.positionalTypes, invocation.actualTypes);
      for (std::size_t expanded = before, end = invocation.actualTypes.size();
           expanded < end; ++expanded)
        invocation.actualValues.push_back(value);
      continue;
    }
    invocation.positionalTypes.push_back(value.getType());
    invocation.actualTypes.push_back(value.getType());
    invocation.actualValues.push_back(value);
  }

  invocation.keywords.reserve(kwNames.getValues().size());
  for (auto [index, nameValue] : llvm::enumerate(kwNames.getValues())) {
    std::optional<std::string> keyword =
        RuntimeBundleLowerer::keywordNameFromValue(nameValue);
    if (!keyword)
      return std::nullopt;
    mlir::Type type = kwValues.getValues()[index].getType();
    unsigned actualIndex = static_cast<unsigned>(invocation.actualTypes.size());
    invocation.actualTypes.push_back(type);
    invocation.actualValues.push_back(kwValues.getValues()[index]);
    invocation.keywords.push_back(
        PlannedKeywordArgument{std::move(*keyword), actualIndex, type});
  }
  return invocation;
}

std::optional<CallableArgumentPlan>
RuntimeBundleLowerer::collectCallableArgumentPlan(py::CallOp op,
                                                  py::CallableType callable,
                                                  bool emitErrors) const {
  std::optional<StaticCallableInvocation> invocation =
      RuntimeBundleLowerer::collectStaticCallableInvocation(op);
  if (!invocation)
    return std::nullopt;
  return buildCallableArgumentPlan(op, callable, invocation->positionalTypes,
                                   invocation->keywords, llvm::StringRef{},
                                   emitErrors);
}

std::optional<CallableAggregateEvidenceCall>
RuntimeBundleLowerer::collectCallableAggregateEvidence(
    py::CallOp op, py::CallableType callable) const {
  std::optional<StaticCallableInvocation> invocation =
      RuntimeBundleLowerer::collectStaticCallableInvocation(op);
  if (!invocation)
    return std::nullopt;

  std::optional<CallableArgumentPlan> plan =
      RuntimeBundleLowerer::collectCallableArgumentPlan(op, callable,
                                                        /*emitErrors=*/false);
  if (!plan)
    return std::nullopt;

  CallableAggregateEvidenceCall evidence;
  auto resolvedCall =
      mlir::dyn_cast_if_present<py::CallableType>(op.getCallContract());
  if (callable.hasVararg()) {
    evidence.varargElementTypes.reserve(plan->varargActuals.size());
    for (unsigned actualIndex : plan->varargActuals) {
      if (actualIndex >= invocation->actualTypes.size())
        return std::nullopt;
      mlir::Type type = invocation->actualTypes[actualIndex];
      if (resolvedCall &&
          actualIndex < resolvedCall.getPositionalTypes().size())
        type = resolvedCall.getPositionalTypes()[actualIndex];
      evidence.varargElementTypes.push_back(type);
    }
  }
  if (callable.hasKwarg()) {
    evidence.kwargKeys.reserve(plan->kwargActuals.size());
    evidence.kwargValueTypes.reserve(plan->kwargActuals.size());
    for (unsigned actualIndex : plan->kwargActuals) {
      auto keyword = llvm::find_if(
          invocation->keywords, [&](const PlannedKeywordArgument &candidate) {
            return candidate.actualIndex == actualIndex;
          });
      if (keyword == invocation->keywords.end() || keyword->name.empty())
        return std::nullopt;
      if (actualIndex >= invocation->actualTypes.size())
        return std::nullopt;
      evidence.kwargKeys.push_back(keyword->name);
      evidence.kwargValueTypes.push_back(invocation->actualTypes[actualIndex]);
    }
  }
  return evidence;
}

std::optional<llvm::SmallVector<mlir::Type, 4>>
RuntimeBundleLowerer::collectCallableArgumentSourceTypes(
    py::CallOp op, py::CallableType callable) const {
  std::optional<StaticCallableInvocation> invocation =
      RuntimeBundleLowerer::collectStaticCallableInvocation(op);
  if (!invocation)
    return std::nullopt;

  std::optional<CallableArgumentPlan> plan =
      RuntimeBundleLowerer::collectCallableArgumentPlan(op, callable,
                                                        /*emitErrors=*/false);
  if (!plan)
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> parameters =
      fixedCallableParameterTypes(callable);
  llvm::SmallVector<mlir::Type, 4> sourceTypes;
  sourceTypes.reserve(parameters.size() + (callable.hasVararg() ? 1 : 0) +
                      (callable.hasKwarg() ? 1 : 0));
  for (auto [index, parameterType] : llvm::enumerate(parameters)) {
    if (!plan->fixedActuals[index]) {
      sourceTypes.push_back(parameterType);
      continue;
    }
    unsigned actualIndex = *plan->fixedActuals[index];
    if (actualIndex >= invocation->actualTypes.size())
      return std::nullopt;
    sourceTypes.push_back(invocation->actualTypes[actualIndex]);
  }
  if (callable.hasVararg())
    sourceTypes.push_back(callable.getVarargType());
  if (callable.hasKwarg())
    sourceTypes.push_back(callable.getKwargType());
  return sourceTypes;
}

mlir::LogicalResult RuntimeBundleLowerer::materializeDefaultArgument(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    unsigned index, mlir::Type parameterType,
    llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
    const RuntimeBundle *&source, const RuntimeBundle *callableObject) {
  auto values =
      target->getAttrOfType<mlir::ArrayAttr>(kCallableDefaultValuesAttr);
  if (!values || index >= values.size() ||
      mlir::isa<mlir::UnitAttr>(values[index]))
    return op.emitError() << "function target '" << targetName
                          << "' has default metadata mismatch at argument "
                          << index;

  // Def-statement evaluated defaults of nested defs live in the closure
  // evidence (one evaluation per enclosing execution); read the shared
  // value instead of re-evaluating.
  if (auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(values[index])) {
    if (auto kind = dict.getAs<mlir::StringAttr>("kind");
        kind && kind.getValue() == "capture") {
      auto captureIndex = dict.getAs<mlir::IntegerAttr>("value");
      if (!captureIndex)
        return op.emitError() << "capture default value has no closure index";
      if (!callableObject ||
          static_cast<std::size_t>(captureIndex.getInt()) >=
              callableObject->closureValues.size())
        return op.emitError()
               << "function target '" << targetName
               << "' default argument " << index
               << " needs the def-statement evaluated value from closure "
                  "evidence, which this call site does not carry";
      const RuntimeValue &captured =
          callableObject->closureValues[captureIndex.getInt()];
      RuntimeBundle bundle =
          RuntimeBundle::object(captured.contract, captured.values);
      bundle.setObjectLogicalOwnership(/*ownsObject=*/false);
      materializedDefaults.push_back(std::move(bundle));
      source = &materializedDefaults.back();
      return mlir::success();
    }
  }

  RuntimeBundle bundle;
  if (mlir::failed(RuntimeBundleLowerer::materializeDefaultValue(
          op, parameterType, values[index], bundle)))
    return mlir::failure();
  materializedDefaults.push_back(std::move(bundle));
  source = &materializedDefaults.back();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::materializeArityObject(
    mlir::Operation *op, mlir::Type contract, std::uint64_t arity,
    RuntimeBundle &bundle, mlir::ArrayRef<RuntimeValue> elements,
    llvm::ArrayRef<std::string> keys,
    llvm::ArrayRef<const RuntimeBundle *> elementBundles,
    llvm::ArrayRef<mlir::Value> logicalSources) {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty())
    return op->emitError() << "variadic callable aggregate has no concrete "
                              "runtime contract";

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contractName, "__new__");
  if (!initializer) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<RuntimeValue> value =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, contract, "structural aggregate object ABI");
    if (mlir::failed(value))
      return mlir::failure();
    bundle = RuntimeBundle::object(value->contract, value->values);
    bundle.sequenceCapacity =
        RuntimeBundleLowerer::collectionInitialCapacity(arity);
    bundle.sequenceElements.append(elements.begin(), elements.end());
    bundle.mappingKeys.append(keys.begin(), keys.end());
    if (!keys.empty()) {
      if (keys.size() != elements.size())
        return op->emitError()
               << "mapping evidence key/value count mismatch for "
               << contractName;
      bundle.mappingValues.append(elements.begin(), elements.end());
      mlir::Value present =
          mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1)
              .getResult();
      for (std::size_t index = 0; index < keys.size(); ++index)
        bundle.mappingPresent.push_back(present);
      bundle.sequenceElements.clear();
      bundle.mappingCapacity =
          RuntimeBundleLowerer::collectionInitialCapacity(keys.size());
    }
    return mlir::success();
  }

  mlir::FunctionType functionType = initializer->function.getFunctionType();
  if (functionType.getNumInputs() != 1 ||
      !functionType.getInput(0).isInteger(64))
    return initializer->function.emitError()
           << contractName << ".__new__ must take one i64 arity argument";

  builder.setInsertionPoint(op);
  mlir::Value length =
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(),
                                         static_cast<std::int64_t>(arity), 64)
          .getResult();
  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *initializer, mlir::ValueRange{length});
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, contract, call.getResults(), bundle)))
    return mlir::failure();
  bundle.sequenceCapacity =
      RuntimeBundleLowerer::collectionInitialCapacity(arity);
  bundle.sequenceElements.append(elements.begin(), elements.end());
  bundle.mappingKeys.append(keys.begin(), keys.end());
  // ⭐ The object exists now and is EMPTY -- `__new__` takes an arity and
  // nothing else. Until this store the elements lived only in the bundle's
  // evidence, and the hidden per-element lanes of the aggregate evidence ABI
  // were the only way a callee could reach them. That ABI is established only
  // when the callee SUBSCRIPTS the tuple or forwards it (CallableAggregate.cpp
  // scans for exactly those two), so every other consumer read uninitialized
  // memory:
  //
  //     def f(*args: int) -> None:
  //         for a in args:
  //             print(a)
  //     f(1, 2)          # printed nothing, or garbage, or aborted in Ly_IncRef
  //
  // `len(args)` was right the whole time, which is what made this look like a
  // partial feature rather than a wrong answer -- the length is the one thing
  // `__new__` was told.
  //
  // ⛔ Do NOT repair this by widening the scan in CallableAggregate.cpp. The
  // consumers to enumerate are every operation on a tuple, and `print(args)`
  // needs a real object no lane can stand in for. An object handed to a callee
  // has to describe itself; the lanes are a shortcut on top of that, not the
  // carrier.
  if (!keys.empty()) {
    if (keys.size() != elements.size())
      return op->emitError() << "mapping evidence key/value count mismatch for "
                             << contractName;
    bundle.mappingValues.append(elements.begin(), elements.end());
    mlir::Value present =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1)
            .getResult();
    for (std::size_t index = 0; index < keys.size(); ++index)
      bundle.mappingPresent.push_back(present);
    bundle.sequenceElements.clear();
    bundle.mappingCapacity =
        RuntimeBundleLowerer::collectionInitialCapacity(keys.size());
  }
  if (!elementBundles.empty()) {
    llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> shared;
    shared.reserve(elementBundles.size());
    for (const RuntimeBundle *element : elementBundles)
      shared.push_back(std::make_shared<RuntimeBundle>(*element));
    if (keys.empty()) {
      bundle.sequenceElementBundles.append(shared.begin(), shared.end());
      if (mlir::failed(RuntimeBundleLowerer::initializeSequencePayload(
              op, bundle, shared, logicalSources)))
        return mlir::failure();
    } else {
      // The **kwargs instance. The keys are the keyword names, which exist
      // only as C++ strings here, so they are materialized the way a dict
      // literal's static keys are -- with no logical source, since nothing in
      // the program produced them.
      if (keys.size() != shared.size())
        return op->emitError()
               << "mapping payload key/value count mismatch for "
               << contractName;
      llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> keyBundles;
      llvm::SmallVector<mlir::Value, 8> keySources;
      keyBundles.reserve(keys.size());
      keySources.reserve(keys.size());
      builder.setInsertionPoint(op);
      for (const std::string &key : keys) {
        RuntimeBundle materialized;
        if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                op, key, materialized)))
          return mlir::failure();
        materialized.literalText = key;
        keyBundles.push_back(
            std::make_shared<RuntimeBundle>(std::move(materialized)));
        keySources.push_back(mlir::Value{});
      }
      bundle.mappingKeyBundles.append(keyBundles.begin(), keyBundles.end());
      bundle.mappingValueBundles.append(shared.begin(), shared.end());
      if (mlir::failed(RuntimeBundleLowerer::initializeDictPayload(
              op, bundle, keyBundles, shared, keySources, logicalSources)))
        return mlir::failure();
    }
  }
  return mlir::success();
}

std::optional<std::string>
RuntimeBundleLowerer::keywordNameFromValue(mlir::Value value) const {
  if (auto str = value.getDefiningOp<py::StrConstantOp>())
    return str.getValue().str();
  return std::nullopt;
}

} // namespace py::lowering
