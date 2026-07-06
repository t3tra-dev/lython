#include "Runtime/Core/Lowerer.h"

#include <functional>

namespace py::lowering {
namespace {

bool isCollectionMetaType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  if (memref.hasStaticShape() && memref.getDimSize(0) < 1)
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  return element && element.getWidth() == 64;
}

mlir::LogicalResult keepAliveCollectionEvidenceUse(mlir::Operation *op,
                                                   mlir::OpBuilder &builder,
                                                   const RuntimeBundle &bundle,
                                                   llvm::StringRef label) {
  if (bundle.physicalValues().size() < 2)
    return op->emitError() << label
                           << " collection has no physical length metadata";
  mlir::Value meta = bundle.physicalValues()[1];
  if (!isCollectionMetaType(meta.getType()))
    return op->emitError() << label
                           << " collection length metadata has invalid type "
                           << meta.getType();
  mlir::Value slot =
      builder.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 0);
  builder.create<mlir::memref::LoadOp>(op->getLoc(), meta, slot);
  return mlir::success();
}

} // namespace

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
  const RuntimeBundle *receiver =
      RuntimeBundleLowerer::concreteObjectForOwnership(*argument);
  if (!receiver)
    receiver = argument;

  if (symbol.builtinName == "repr" && symbol.builtinMethod == "__repr__" &&
      RuntimeBundleLowerer::needsDefaultObjectRepr(*receiver)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
            op, *receiver, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{receiver};
  std::optional<EmittedRuntimeCall> emitted;
  if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
          op, *receiver, symbol.builtinMethod, sources,
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
  const RuntimeBundle *sinkArgument =
      RuntimeBundleLowerer::concreteObjectForOwnership(*argument);
  if (!sinkArgument)
    sinkArgument = argument;

  RuntimeBundle printable = *sinkArgument;
  auto assignSinkResults = [&]() -> mlir::LogicalResult {
    std::string resultContract =
        symbol.resultContract.empty() ? "types.NoneType" : symbol.resultContract;
    for (mlir::Value result : op.getResults()) {
      if (mlir::failed(assignObjectBundle(
              op, result, runtimeContractType(context, resultContract), {})))
        return mlir::failure();
    }
    erase.push_back(op);
    return mlir::success();
  };

  if (symbol.builtinName == "print" && symbol.builtinMethod == "__repr__" &&
      symbol.builtinSinkContract == "builtins.str" &&
      printable.contractName() == "builtins.object") {
    std::optional<RuntimeSymbol> objectPrint =
        manifest.primitive("builtins.object", "print_line");
    if (!objectPrint)
      return op.emitError()
             << "runtime manifest has no builtins.object.print_line primitive";
    llvm::SmallVector<const RuntimeBundle *, 1> sources{&printable};
    llvm::SmallVector<mlir::Value, 1> operands;
    builder.setInsertionPoint(op);
    if (mlir::failed(buildRuntimeCallOperands(op, *objectPrint, sources,
                                              operands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *objectPrint,
                                            operands);
    return assignSinkResults();
  }

  if (printable.contractName() != symbol.builtinSinkContract &&
      symbol.builtinMethod == "__repr__" &&
      symbol.builtinSinkContract == "builtins.str") {
    auto renderRepr =
        [&](auto &&self,
            const RuntimeBundle &bundle) -> std::optional<std::string> {
      const RuntimeBundle *resolved =
          RuntimeBundleLowerer::concreteObjectForOwnership(bundle);
      const RuntimeBundle &view = resolved ? *resolved : bundle;
      if (view.literalText && view.contractName() == "builtins.str")
        return *view.literalText;
      if (!view.sequenceElementBundles.empty()) {
        std::string text = "[";
        for (auto [index, element] :
             llvm::enumerate(view.sequenceElementBundles)) {
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
      if (!view.fieldBundles.empty() &&
          RuntimeBundleLowerer::classDefinesMethod(view.contract,
                                                   "__repr__")) {
        std::string contractName = view.contractName();
        llvm::StringRef contract(contractName);
        std::string text = contract.rsplit('.').second.str();
        if (text.empty())
          text = contract.str();
        text += "(";
        llvm::SmallVector<llvm::StringRef, 4> names;
        for (const auto &entry : view.fieldBundles)
          names.push_back(entry.getKey());
        llvm::sort(names);
        for (auto [index, name] : llvm::enumerate(names)) {
          if (index)
            text += ", ";
          text += name.str();
          text += "=";
          auto field = view.fieldBundles.find(name);
          if (field == view.fieldBundles.end() || !field->second)
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
        const RuntimeBundle *resolved =
            RuntimeBundleLowerer::concreteObjectForOwnership(bundle);
        const RuntimeBundle &view = resolved ? *resolved : bundle;
        if (view.literalText && view.contractName() == "builtins.str")
          return RuntimeBundleLowerer::materializeStringObject(
              op, *view.literalText, result);

        if (!view.sequenceElementBundles.empty()) {
          if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                  op, "[", result)))
            return mlir::failure();
          for (auto [index, element] :
               llvm::enumerate(view.sequenceElementBundles)) {
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

        if (RuntimeBundleLowerer::needsDefaultObjectRepr(view))
          return RuntimeBundleLowerer::materializeDefaultObjectRepr(op, view,
                                                                    result);

        llvm::SmallVector<const RuntimeBundle *, 1> sources{&view};
        std::optional<EmittedRuntimeCall> emitted;
        if (mlir::failed(emitManifestMethodCall(
                op, view, symbol.builtinMethod, sources,
                /*allowUnusedSources=*/false, emitted)))
          return mlir::failure();
        return bundleRuntimeResults(
            op, runtimeContractType(context, symbol.builtinSinkContract),
            emitted->call, result);
      };

      const RuntimeBundle *dynamicKeepAlive = nullptr;
      const RuntimeBundle *resolvedPrintable =
          RuntimeBundleLowerer::concreteObjectForOwnership(printable);
      const RuntimeBundle &printableView =
          resolvedPrintable ? *resolvedPrintable : printable;
      if (!printableView.sequenceElementBundles.empty())
        dynamicKeepAlive = &printableView;

      RuntimeBundle dynamic;
      if (mlir::failed(renderDynamicRepr(printable, dynamic)))
        return mlir::failure();
      if (dynamicKeepAlive) {
        builder.setInsertionPoint(op);
        if (mlir::failed(keepAliveCollectionEvidenceUse(
                op, builder, *dynamicKeepAlive, "dynamic repr")))
          return mlir::failure();
      }
      printable = std::move(dynamic);
    }
  }
  if (printable.contractName() != symbol.builtinSinkContract) {
    if (symbol.builtinMethod == "__repr__" &&
        RuntimeBundleLowerer::needsDefaultObjectRepr(printable)) {
      RuntimeBundle rendered;
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, printable, rendered)))
        return mlir::failure();
      printable = std::move(rendered);
    } else {
      llvm::SmallVector<const RuntimeBundle *, 1> sources{&printable};
      std::optional<EmittedRuntimeCall> emitted;
      if (mlir::failed(emitManifestMethodCall(
              op, printable, symbol.builtinMethod, sources,
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
  return assignSinkResults();
}

} // namespace py::lowering
