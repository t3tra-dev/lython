#include "RuntimeLowering/RuntimeLowering.h"

namespace py::runtime_lowering {
namespace {

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

} // namespace py::runtime_lowering
