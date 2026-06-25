#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {

mlir::LogicalResult RuntimeBundleLowerer::lowerCall(py::CallOp op) {
  const RuntimeBundle *callable =
      RuntimeBundleLowerer::bundleFor(op.getCallable());
  if (!callable)
    return op.emitError() << "callable has no lowered runtime bundle";
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
  return op.emitError() << "builtin callable '" << callable->binding
                        << "' has unsupported lowering strategy '"
                        << builtin->builtinLowering << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectCallableCall(py::CallOp op,
                                              const RuntimeBundle &callable) {
  if (callable.kind != RuntimeBundle::Kind::Object)
    return op.emitError()
           << "callable is not an object bundle with a __call__ contract";
  if (!callable.functionTarget.empty())
    return RuntimeBundleLowerer::lowerFunctionTargetCall(op, callable);
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python callable lowering expects exactly one Python result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources{&callable};
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "positional args", sources)))
    return mlir::failure();

  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(0), callable, "__call__", sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
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
  if (mlir::failed(collectFunctionCallSources(
          op, target, callable.functionTarget, sources, materializedDefaults)))
    return mlir::failure();
  llvm::SmallVector<RuntimeBundle, 4> closureSources;
  closureSources.reserve(callable.closureValues.size());
  for (const RuntimeValue &closureValue : callable.closureValues) {
    closureSources.push_back(
        RuntimeBundle::object(closureValue.contract, closureValue.values));
    sources.push_back(&closureSources.back());
  }

  mlir::FunctionType functionType = target.getFunctionType();
  for (mlir::Type input : functionType.getInputs())
    if (py::isPyType(input))
      return op.emitError() << "function target '" << callable.functionTarget
                            << "' still has unresolved Python parameter ABI";
  for (mlir::Type result : functionType.getResults())
    if (py::isPyType(result))
      return op.emitError() << "function target '" << callable.functionTarget
                            << "' still has unresolved Python result ABI";
  RuntimeSymbol targetSymbol;
  targetSymbol.function = target;
  targetSymbol.contract = "builtins.function";
  targetSymbol.role = "method";
  targetSymbol.name = callable.functionTarget;

  llvm::SmallVector<mlir::Value, 8> operands;
  unsigned inputIndex = 0;
  for (const RuntimeBundle *source : sources) {
    if (inputIndex >= functionType.getNumInputs())
      return op.emitError() << "too many positional args for function target "
                            << callable.functionTarget;
    if (mlir::failed(appendRuntimeSource(op, targetSymbol, functionType,
                                         inputIndex, *source, operands)))
      return mlir::failure();
  }
  if (inputIndex != functionType.getNumInputs())
    return op.emitError() << "missing positional args for function target "
                          << callable.functionTarget;

  mlir::func::CallOp call = builder.create<mlir::func::CallOp>(
      op.getLoc(), target.getSymName(), functionType.getResults(), operands);
  RuntimeBundle result;
  mlir::Type expectedResult = op.getResult(0).getType();
  if (runtimeContractName(expectedResult).empty()) {
    auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
    if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
            callableAttr ? callableAttr.getValue() : mlir::Type())) {
      if (callable.getResultTypes().size() == 1)
        expectedResult = callable.getResultTypes().front();
    }
  }
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, expectedResult, call, result)))
    return mlir::failure();
  auto returned = returnedCallableSummaries.find(callable.functionTarget);
  if (returned != returnedCallableSummaries.end()) {
    result.functionTarget = returned->second.target;
    for (unsigned index : returned->second.captureArgumentIndices) {
      if (index >= sources.size())
        return op.emitError()
               << "returned callable summary for " << callable.functionTarget
               << " references logical argument " << index
               << ", but call has only " << sources.size()
               << " logical sources";
      if (sources[index]->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "returned callable capture source must be an object bundle";
      result.closureValues.push_back(sources[index]->objectValue);
    }
  }
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectFunctionCallSources(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults) {
  auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
  if (!callableAttr)
    return op.emitError() << "function target '" << targetName
                          << "' has no callable_type";
  auto callable = mlir::dyn_cast<py::CallableType>(callableAttr.getValue());
  if (!callable)
    return op.emitError() << "function target '" << targetName
                          << "' callable_type is not Callable";
  if (callable.hasVararg() || callable.hasKwarg())
    return op.emitError() << "function target '" << targetName
                          << "' has unsupported variadic parameters";

  llvm::ArrayRef<mlir::Type> positionalParameters =
      callable.getPositionalTypes();
  llvm::ArrayRef<mlir::Type> kwonlyParameters = callable.getKwOnlyTypes();
  llvm::SmallVector<mlir::Type, 8> parameters(positionalParameters.begin(),
                                              positionalParameters.end());
  parameters.append(kwonlyParameters.begin(), kwonlyParameters.end());
  llvm::SmallVector<const RuntimeBundle *, 8> ordered(parameters.size(),
                                                      nullptr);
  materializedDefaults.reserve(parameters.size());

  const RuntimeBundle *posargs =
      RuntimeBundleLowerer::bundleFor(op.getPosargs());
  if (!posargs || posargs->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "positional args must be a lowered aggregate "
                             "bundle";
  if (posargs->aggregateOperands.size() > positionalParameters.size())
    return op.emitError() << "too many positional args for function target "
                          << targetName;
  for (auto [index, operand] : llvm::enumerate(posargs->aggregateOperands)) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(operand);
    if (!source)
      return op.emitError()
             << "positional arg operand has no lowered runtime bundle";
    ordered[index] = source;
  }

  if (mlir::failed(RuntimeBundleLowerer::applyFunctionKeywordSources(
          op, callable, targetName, ordered)))
    return mlir::failure();

  auto hasDefaultAt = [&](unsigned index) {
    llvm::ArrayRef<mlir::BoolAttr> positionalDefaults =
        callable.getPositionalDefaults();
    if (index < positionalParameters.size())
      return index < positionalDefaults.size() &&
             positionalDefaults[index].getValue();
    llvm::ArrayRef<mlir::BoolAttr> kwonlyDefaults =
        callable.getKwOnlyDefaults();
    unsigned kwIndex =
        index - static_cast<unsigned>(positionalParameters.size());
    return kwIndex < kwonlyDefaults.size() &&
           kwonlyDefaults[kwIndex].getValue();
  };
  for (unsigned index = 0, end = ordered.size(); index < end; ++index) {
    if (ordered[index]) {
      sources.push_back(ordered[index]);
      continue;
    }
    if (hasDefaultAt(index)) {
      const RuntimeBundle *source = nullptr;
      if (mlir::failed(materializeDefaultArgument(
              op, target, targetName, index, parameters[index],
              materializedDefaults, source)))
        return mlir::failure();
      sources.push_back(source);
      continue;
    }
    return op.emitError() << "missing required argument " << index
                          << " for function target " << targetName;
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::materializeDefaultArgument(
    py::CallOp op, mlir::func::FuncOp target, llvm::StringRef targetName,
    unsigned index, mlir::Type parameterType,
    llvm::SmallVectorImpl<RuntimeBundle> &materializedDefaults,
    const RuntimeBundle *&source) {
  auto values =
      target->getAttrOfType<mlir::ArrayAttr>(kCallableDefaultValuesAttr);
  if (!values || index >= values.size() ||
      mlir::isa<mlir::UnitAttr>(values[index]))
    return op.emitError() << "function target '" << targetName
                          << "' has default metadata mismatch at argument "
                          << index;

  RuntimeBundle bundle;
  if (mlir::failed(RuntimeBundleLowerer::materializeDefaultValue(
          op, parameterType, values[index], bundle)))
    return mlir::failure();
  materializedDefaults.push_back(std::move(bundle));
  source = &materializedDefaults.back();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::applyFunctionKeywordSources(
    py::CallOp op, py::CallableType callable, llvm::StringRef targetName,
    llvm::MutableArrayRef<const RuntimeBundle *> ordered) const {
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
  if (kwNames->aggregateOperands.empty())
    return mlir::success();

  llvm::ArrayRef<mlir::StringAttr> positionalNames =
      callable.getPositionalNames();
  if (positionalNames.size() != callable.getPositionalTypes().size())
    return op.emitError()
           << "function target '" << targetName
           << "' has no complete positional parameter name metadata";
  llvm::ArrayRef<mlir::StringAttr> kwonlyNames = callable.getKwOnlyNames();
  if (kwonlyNames.size() != callable.getKwOnlyTypes().size())
    return op.emitError()
           << "function target '" << targetName
           << "' has no complete keyword-only parameter name metadata";

  for (auto [index, nameValue] : llvm::enumerate(kwNames->aggregateOperands)) {
    std::optional<std::string> keyword =
        RuntimeBundleLowerer::keywordNameFromValue(nameValue);
    if (!keyword)
      return op.emitError()
             << "function keyword name must be a static string literal";

    std::optional<unsigned> parameterIndex =
        RuntimeBundleLowerer::keywordParameterIndex(callable, *keyword);
    if (!parameterIndex)
      return op.emitError() << "unexpected keyword argument '" << *keyword
                            << "' for function target " << targetName;
    if (*parameterIndex < callable.getPositionalOnlyCount())
      return op.emitError()
             << "positional-only argument '" << *keyword
             << "' passed as keyword to function target " << targetName;
    if (ordered[*parameterIndex])
      return op.emitError() << "multiple values for argument '" << *keyword
                            << "' in function target " << targetName;

    const RuntimeBundle *source =
        RuntimeBundleLowerer::bundleFor(kwValues->aggregateOperands[index]);
    if (!source)
      return op.emitError()
             << "keyword value operand has no lowered runtime bundle";
    ordered[*parameterIndex] = source;
  }
  return mlir::success();
}

std::optional<std::string>
RuntimeBundleLowerer::keywordNameFromValue(mlir::Value value) const {
  if (auto str = value.getDefiningOp<py::StrConstantOp>())
    return str.getValue().str();
  return std::nullopt;
}

std::optional<unsigned>
RuntimeBundleLowerer::keywordParameterIndex(py::CallableType callable,
                                            llvm::StringRef keyword) const {
  llvm::ArrayRef<mlir::StringAttr> names = callable.getPositionalNames();
  for (auto [index, name] : llvm::enumerate(names))
    if (name && name.getValue() == keyword)
      return static_cast<unsigned>(index);
  unsigned offset = static_cast<unsigned>(callable.getPositionalTypes().size());
  for (auto [index, name] : llvm::enumerate(callable.getKwOnlyNames()))
    if (name && name.getValue() == keyword)
      return offset + static_cast<unsigned>(index);
  return std::nullopt;
}

} // namespace py::runtime_lowering
