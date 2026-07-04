#include "RuntimeLowering/RuntimeBundleLowerer.h"

namespace py::runtime_lowering {

mlir::LogicalResult RuntimeBundleLowerer::lowerReceiverMethodResult(
    mlir::Operation *op, mlir::Value receiverValue, mlir::Value resultValue,
    llvm::StringRef missingSubject, llvm::StringRef methodName,
    bool preferManifestObjectResult) {
  const RuntimeBundle *receiver =
      RuntimeBundleLowerer::bundleFor(receiverValue);
  if (!receiver)
    return op->emitError() << missingSubject
                           << " has no lowered runtime bundle";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{receiver};
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, *receiver, methodName, sources,
          /*allowUnusedSources=*/false, preferManifestObjectResult)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBool(py::BoolOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "bool input has no lowered runtime bundle";
  llvm::ArrayRef<mlir::Value> inputValues = input->physicalValues();
  if (input->contractName() == "builtins.bool" && inputValues.size() == 1 &&
      inputValues.front().getType().isInteger(1)) {
    op.getResult().replaceAllUsesWith(inputValues.front());
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{input};
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__bool__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(op, op.getResult(), *input,
                                               *methodName, sources,
                                               /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerLen(py::LenOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__len__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getInput(), op.getResult(), "len input", *methodName);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSetItem(py::SetItemOp op) {
  llvm::SmallVector<mlir::Value, 3> inputs{op.getContainer(), op.getIndex(),
                                           op.getValue()};
  llvm::SmallVector<const RuntimeBundle *, 3> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "setitem operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__setitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerDelItem(py::DelItemOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getIndex()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "delitem operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__delitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerContains(py::ContainsOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getItem()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "contains operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__contains__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerIter(py::IterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getIterable(),
                                                op.getResult());
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__iter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getIterable(), op.getResult(), "iter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNext(py::NextOp op) {
  const RuntimeBundle *iterator =
      RuntimeBundleLowerer::bundleFor(op.getIterator());
  if (!iterator)
    return op.emitError() << "next iterator has no lowered runtime bundle";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{iterator};
  std::optional<EmittedRuntimeCall> emitted;
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__next__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(emitManifestMethodCall(op, *iterator, *methodName, sources,
                                          /*allowUnusedSources=*/false,
                                          emitted)))
    return mlir::failure();
  if (!emitted->symbol.validResultIndex)
    return op.emitError()
           << "runtime __next__ method must declare valid_result_index";

  mlir::func::CallOp call = emitted->call;
  unsigned validIndex = *emitted->symbol.validResultIndex;
  if (validIndex >= call.getNumResults())
    return op.emitError() << "runtime __next__ valid_result_index is outside "
                             "the result list";
  mlir::Value valid = call.getResult(validIndex);
  if (!valid.getType().isInteger(1))
    return op.emitError() << "runtime __next__ valid result must be an i1";

  std::string elementContract = runtimeContractName(op.getElement().getType());
  if (elementContract.empty())
    elementContract = emitted->symbol.elementContract;
  if (elementContract.empty())
    return op.emitError() << "runtime __next__ element needs a concrete "
                             "manifest element contract";

  std::string nextContract = runtimeContractName(op.getNext().getType());
  if (nextContract.empty())
    nextContract = emitted->symbol.nextContract;
  if (nextContract.empty())
    nextContract = iterator->contractName();
  if (nextContract.empty())
    return op.emitError() << "runtime __next__ next state needs a concrete "
                             "manifest contract";

  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (unsigned index = 0; index < validIndex; ++index)
    elementValues.push_back(call.getResult(index));

  llvm::SmallVector<mlir::Value, 4> nextValues;
  for (unsigned index = validIndex + 1; index < call.getNumResults(); ++index)
    nextValues.push_back(call.getResult(index));

  op.getValid().replaceAllUsesWith(valid);
  if (mlir::failed(assignObjectBundle(
          op, op.getElement(), runtimeContractType(context, elementContract),
          elementValues)))
    return mlir::failure();
  if (mlir::failed(assignObjectBundle(
          op, op.getNext(), runtimeContractType(context, nextContract),
          nextValues)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerEnter(py::EnterOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__enter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getManager(), op.getResult(), "enter manager", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerExit(py::ExitOp op) {
  llvm::SmallVector<mlir::Value, 4> inputs{op.getManager(), op.getExcType(),
                                           op.getExcValue(), op.getTraceback()};
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "exit operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__exit__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/true,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAEnter(py::AEnterOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aenter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getManager(), op.getResult(), "aenter manager", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAExit(py::AExitOp op) {
  llvm::SmallVector<mlir::Value, 4> inputs{op.getManager(), op.getExcType(),
                                           op.getExcValue(), op.getTraceback()};
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "aexit operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aexit__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/true,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAIter(py::AIterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getAsyncIterable(),
                                                op.getResult());
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aiter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterable(), op.getResult(), "aiter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerANext(py::ANextOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__anext__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterator(), op.getAwaitable(), "anext iterator",
      *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerRound(py::RoundOp op) {
  if (op.getInputs().empty())
    return op.emitError() << "round requires at least a receiver input";

  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, op.getInputs(), "round input has no lowered runtime bundle",
          sources)))
    return mlir::failure();

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__round__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerUnarySpecial(mlir::Operation *op, mlir::Value input,
                                        llvm::StringRef methodName,
                                        mlir::Value resultValue) {
  if (methodName == "__repr__") {
    const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
    if (!inputBundle)
      return op->emitError() << "repr operand has no lowered runtime bundle";
    if (RuntimeBundleLowerer::needsDefaultObjectRepr(*inputBundle)) {
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *inputBundle, result)))
        return mlir::failure();
      valueBundles[resultValue] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, input, resultValue, "unary special method operand", methodName,
      /*preferManifestObjectResult=*/true);
}

} // namespace py::runtime_lowering
