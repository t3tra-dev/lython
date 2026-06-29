#include "RuntimeLowering/RuntimeLowering.h"

#include <functional>

namespace py::runtime_lowering {

mlir::LogicalResult RuntimeBundleLowerer::collectSingleBuiltinArgument(
    py::CallOp op, const RuntimeSymbol &symbol,
    const RuntimeBundle *&argument) const {
  const RuntimeBundle *posargs =
      RuntimeBundleLowerer::bundleFor(op.getPosargs());
  if (!posargs || posargs->kind != RuntimeBundle::Kind::Aggregate)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires packed positional arguments";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();
  if (posargs->aggregateOperands.size() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' expects exactly one positional argument";

  argument = RuntimeBundleLowerer::bundleFor(posargs->aggregateOperands[0]);
  if (!argument)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' argument has no lowered runtime bundle";
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodCall(py::CallOp op,
                                             const RuntimeSymbol &symbol) {
  if (op.getNumResults() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' method lowering must produce one result";

  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();

  if (symbol.builtinName == "repr" && symbol.builtinMethod == "__repr__" &&
      RuntimeBundleLowerer::needsDefaultObjectRepr(*argument)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
            op, *argument, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
          op, *argument, symbol.builtinMethod, sources,
          /*allowUnusedSources=*/false, emitted)))
    return mlir::failure();

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(
          bundleRuntimeResults(op, runtimeContractType(context, resultContract),
                               emitted->call, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerDirectBuiltinCall(py::CallOp op,
                                             const RuntimeSymbol &symbol) {
  if (op.getNumResults() != 1)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' direct lowering must produce one result";

  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> operands;
  if (mlir::failed(buildRuntimeCallOperands(op, symbol, sources, operands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), symbol, operands);

  std::string resultContract = runtimeContractName(op.getResult(0).getType());
  if (resultContract.empty() || resultContract == "builtins.object")
    resultContract = symbol.resultContract;
  if (resultContract.empty())
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' needs a concrete result contract";

  RuntimeBundle result;
  if (mlir::failed(bundleRuntimeResults(
          op, runtimeContractType(context, resultContract), call, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerBuiltinMethodSinkCall(py::CallOp op,
                                                 const RuntimeSymbol &symbol) {
  const RuntimeBundle *argument = nullptr;
  if (mlir::failed(collectSingleBuiltinArgument(op, symbol, argument)))
    return mlir::failure();

  RuntimeBundle printable = *argument;
  if (printable.contractName() != symbol.builtinSinkContract &&
      symbol.builtinMethod == "__repr__" &&
      symbol.builtinSinkContract == "builtins.str") {
    auto renderRepr =
        [&](auto &&self,
            const RuntimeBundle &bundle) -> std::optional<std::string> {
      if (bundle.literalText && bundle.contractName() == "builtins.str")
        return *bundle.literalText;
      if (!bundle.sequenceElementBundles.empty()) {
        std::string text = "[";
        for (auto [index, element] :
             llvm::enumerate(bundle.sequenceElementBundles)) {
          if (index)
            text += ", ";
          if (!element)
            return std::nullopt;
          std::optional<std::string> rendered = self(self, *element);
          if (!rendered)
            return std::nullopt;
          text += *rendered;
        }
        text += "]";
        return text;
      }
      if (!bundle.fieldBundles.empty() &&
          RuntimeBundleLowerer::classDefinesMethod(bundle.contract,
                                                   "__repr__")) {
        std::string contractName = bundle.contractName();
        llvm::StringRef contract(contractName);
        std::string text = contract.rsplit('.').second.str();
        if (text.empty())
          text = contract.str();
        text += "(";
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &entry : bundle.fieldBundles)
          names.push_back(entry.getKey());
        llvm::sort(names);
        for (auto [index, name] : llvm::enumerate(names)) {
          if (index)
            text += ", ";
          text += name.str();
          text += "=";
          auto field = bundle.fieldBundles.find(name);
          if (field == bundle.fieldBundles.end() || !field->second)
            return std::nullopt;
          std::optional<std::string> rendered = self(self, *field->second);
          if (!rendered)
            return std::nullopt;
          text += *rendered;
        }
        text += ")";
        return text;
      }
      return std::nullopt;
    };
    if (std::optional<std::string> rendered =
            renderRepr(renderRepr, printable)) {
      if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
              op, *rendered, printable)))
        return mlir::failure();
      printable.literalText = std::move(*rendered);
    } else {
      auto concatStrings = [&](const RuntimeBundle &lhs,
                               const RuntimeBundle &rhs,
                               RuntimeBundle &result) -> mlir::LogicalResult {
        llvm::SmallVector<const RuntimeBundle *, 2> sources{&lhs, &rhs};
        std::optional<EmittedRuntimeCall> emitted;
        if (mlir::failed(emitManifestMethodCall(op, lhs, "__add__", sources,
                                                /*allowUnusedSources=*/false,
                                                emitted)))
          return mlir::failure();
        return bundleRuntimeResults(
            op, runtimeContractType(context, "builtins.str"), emitted->call,
            result);
      };

      auto appendText = [&](RuntimeBundle &target,
                            llvm::StringRef text) -> mlir::LogicalResult {
        RuntimeBundle suffix;
        if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(op, text,
                                                                       suffix)))
          return mlir::failure();
        RuntimeBundle combined;
        if (mlir::failed(concatStrings(target, suffix, combined)))
          return mlir::failure();
        target = std::move(combined);
        return mlir::success();
      };

      auto appendBundle =
          [&](RuntimeBundle &target,
              const RuntimeBundle &suffix) -> mlir::LogicalResult {
        RuntimeBundle combined;
        if (mlir::failed(concatStrings(target, suffix, combined)))
          return mlir::failure();
        target = std::move(combined);
        return mlir::success();
      };

      std::function<mlir::LogicalResult(const RuntimeBundle &, RuntimeBundle &)>
          renderDynamicRepr =
              [&](const RuntimeBundle &bundle,
                  RuntimeBundle &result) -> mlir::LogicalResult {
        if (bundle.literalText && bundle.contractName() == "builtins.str")
          return RuntimeBundleLowerer::materializeStringObject(
              op, *bundle.literalText, result);

        if (!bundle.sequenceElementBundles.empty()) {
          if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                  op, "[", result)))
            return mlir::failure();
          for (auto [index, element] :
               llvm::enumerate(bundle.sequenceElementBundles)) {
            if (index && mlir::failed(appendText(result, ", ")))
              return mlir::failure();
            RuntimeBundle elementRepr;
            if (element) {
              if (mlir::failed(renderDynamicRepr(*element, elementRepr)))
                return mlir::failure();
            } else if (mlir::failed(
                           RuntimeBundleLowerer::materializeStringObject(
                               op, "builtins.object", elementRepr))) {
              return mlir::failure();
            }
            if (mlir::failed(appendBundle(result, elementRepr)))
              return mlir::failure();
          }
          return appendText(result, "]");
        }

        if (RuntimeBundleLowerer::needsDefaultObjectRepr(bundle))
          return RuntimeBundleLowerer::materializeDefaultObjectRepr(op, bundle,
                                                                    result);

        llvm::SmallVector<const RuntimeBundle *, 1> sources{&bundle};
        std::optional<EmittedRuntimeCall> emitted;
        if (mlir::failed(emitManifestMethodCall(
                op, bundle, symbol.builtinMethod, sources,
                /*allowUnusedSources=*/false, emitted)))
          return mlir::failure();
        return bundleRuntimeResults(
            op, runtimeContractType(context, symbol.builtinSinkContract),
            emitted->call, result);
      };

      RuntimeBundle dynamic;
      if (mlir::failed(renderDynamicRepr(printable, dynamic)))
        return mlir::failure();
      printable = std::move(dynamic);
    }
  }
  if (printable.contractName() != symbol.builtinSinkContract) {
    if (symbol.builtinMethod == "__repr__" &&
        RuntimeBundleLowerer::needsDefaultObjectRepr(*argument)) {
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *argument, printable)))
        return mlir::failure();
    } else {
      llvm::SmallVector<const RuntimeBundle *, 1> sources{argument};
      std::optional<EmittedRuntimeCall> emitted;
      if (mlir::failed(emitManifestMethodCall(
              op, *argument, symbol.builtinMethod, sources,
              /*allowUnusedSources=*/false, emitted)))
        return mlir::failure();
      if (mlir::failed(bundleRuntimeResults(
              op, runtimeContractType(context, symbol.builtinSinkContract),
              emitted->call, printable)))
        return mlir::failure();
    }
  }
  if (printable.contractName() != symbol.builtinSinkContract)
    return op.emitError() << "builtin callable '" << symbol.builtinName
                          << "' requires a " << symbol.builtinSinkContract
                          << "-compatible argument";

  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), symbol,
                                          printable.physicalValues());
  std::string resultContract =
      symbol.resultContract.empty() ? "types.NoneType" : symbol.resultContract;
  for (mlir::Value result : op.getResults()) {
    if (mlir::failed(assignObjectBundle(
            op, result, runtimeContractType(context, resultContract), {})))
      return mlir::failure();
  }
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
