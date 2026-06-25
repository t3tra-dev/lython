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

mlir::LogicalResult RuntimeBundleLowerer::emitManifestMethodCall(
    mlir::Operation *op, const RuntimeBundle &receiver,
    llvm::StringRef methodName, llvm::ArrayRef<const RuntimeBundle *> sources,
    bool allowUnusedSources, std::optional<EmittedRuntimeCall> &emitted) {
  std::string receiverContract = receiver.contractName();
  if (receiverContract.empty())
    return op->emitError()
           << "runtime method receiver has no concrete contract";

  std::optional<RuntimeSymbol> method =
      manifest.method(receiverContract, methodName);
  if (!method)
    return op->emitError() << "runtime manifest has no " << receiverContract
                           << "." << methodName << " method";
  if (mlir::failed(verifySelectedRuntimeTarget(op, *method)))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 8> operands;
  builder.setInsertionPoint(op);
  if (mlir::failed(buildRuntimeCallOperands(op, *method, sources, operands,
                                            allowUnusedSources)))
    return mlir::failure();

  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *method, operands);
  emitted.emplace(EmittedRuntimeCall{*method, call});
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
    const EmittedRuntimeCall &emitted, bool preferManifestObjectResult) {
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
      op, resultValue, *emitted, preferManifestObjectResult);
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
  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contract, *methodName);
  if (!initializer)
    return op.emitError() << "runtime manifest has no " << contract << "."
                          << *methodName;
  if (mlir::failed(verifySelectedRuntimeTarget(op, *initializer)))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "positional args", sources)))
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
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "positional args", sources)))
    return mlir::failure();

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__init__");
  if (mlir::failed(methodName))
    return mlir::failure();

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
