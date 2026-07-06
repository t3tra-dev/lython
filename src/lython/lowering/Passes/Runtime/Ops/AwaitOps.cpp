#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <functional>

namespace py::lowering {
namespace {

mlir::Value stripObjectView(mlir::Value value) {
  while (value) {
    mlir::Operation *def = value.getDefiningOp();
    if (!def || def->getNumOperands() != 1 || def->getNumResults() != 1)
      return value;
    llvm::StringRef name = def->getName().getStringRef();
    if (name != "py.class.upcast" && name != "py.class.refine" &&
        name != "py.protocol.view")
      return value;
    value = def->getOperand(0);
  }
  return value;
}

std::optional<std::string> returnedSelfFieldAwait(mlir::func::FuncOp target) {
  if (!target || target.empty())
    return std::nullopt;
  std::optional<std::string> fieldName;
  bool invalid = false;
  mlir::Block &entry = target.getBody().front();
  target.walk([&](mlir::func::ReturnOp ret) {
    if (ret->getParentOfType<mlir::func::FuncOp>() != target || invalid)
      return;
    if (ret.getNumOperands() != 1) {
      invalid = true;
      return;
    }
    mlir::Value returned = stripObjectView(ret.getOperand(0));
    auto call = returned.getDefiningOp<py::CallOp>();
    if (!call || call.getNumResults() != 1) {
      invalid = true;
      return;
    }
    auto method = call->getAttrOfType<mlir::StringAttr>("ly.bound_method");
    if (!method || method.getValue() != "__await__") {
      invalid = true;
      return;
    }
    mlir::Value receiver = stripObjectView(call.getCallable());
    auto attr = receiver.getDefiningOp<py::AttrGetOp>();
    if (!attr) {
      invalid = true;
      return;
    }
    auto self = mlir::dyn_cast<mlir::BlockArgument>(attr.getObject());
    if (!self || self.getOwner() != &entry || self.getArgNumber() != 0) {
      invalid = true;
      return;
    }
    std::string candidate = attr.getName().str();
    if (!fieldName) {
      fieldName = std::move(candidate);
      return;
    }
    if (*fieldName != candidate)
      invalid = true;
  });
  if (invalid)
    return std::nullopt;
  return fieldName;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &awaitable,
    llvm::StringRef label) {
  const RuntimeValue *sleepResult =
      awaitable.objectEvidence.slot(kAsyncioSleepResultSlot);
  if (!sleepResult)
    return op->emitError() << label << " has no result evidence";

  if (awaitable.objectEvidence.hasFlag(kAsyncioSleepZeroDelayFlag)) {
    if (mlir::failed(RuntimeBundleLowerer::bindEvidenceObjectResult(
            op, resultValue, label, *sleepResult)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  const RuntimeValue *sleepLoop =
      awaitable.objectEvidence.slot(kAsyncioSleepLoopSlot);
  if (!sleepLoop)
    return op->emitError() << label
                           << " has no event loop evidence for timer dispatch";
  if (!awaitable.objectEvidence.hasFlag(kAsyncioSleepTimerScheduledFlag))
    return op->emitError() << label << " has no scheduled timer evidence";

  RuntimeBundle loop =
      RuntimeBundle::object(sleepLoop->contract, sleepLoop->values);
  std::optional<RuntimeSymbol> dispatchDue =
      manifest.primitive("asyncio.AbstractEventLoop", "timer.dispatch_due");
  if (!dispatchDue)
    return op->emitError()
           << "runtime manifest has no asyncio.AbstractEventLoop."
              "timer.dispatch_due primitive";

  llvm::SmallVector<const RuntimeBundle *, 1> timerSources{&loop};
  llvm::SmallVector<mlir::Value, 4> timerOperands;
  builder.setInsertionPoint(op);
  if (mlir::failed(buildRuntimeCallOperands(op, *dispatchDue, timerSources,
                                            timerOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();

  mlir::func::CallOp readyCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *dispatchDue, timerOperands);
  if (readyCall.getNumResults() != 1 ||
      !readyCall.getResult(0).getType().isInteger(1))
    return dispatchDue->function.emitError()
           << "asyncio.AbstractEventLoop.timer.dispatch_due primitive must "
              "return one i1";

  llvm::SmallVector<RuntimeValue, 1> candidates{*sleepResult};
  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, resultValue, candidates, mlir::ValueRange{readyCall.getResult(0)},
          label, "builtins.RuntimeError", "asyncio.sleep timer is not ready");
  if (mlir::failed(selected))
    return mlir::failure();

  if (awaitable.contractName() == "_asyncio.Task")
    awaitable.objectEvidence.setSlot(kFutureResultSlot, *sleepResult);

  return RuntimeBundleLowerer::bindSelectedEvidenceObjectResult(
      op, resultValue, std::move(*selected));
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAwait(py::AwaitOp op) {
  auto awaitableIt = valueBundles.find(op.getAwaitable());
  if (awaitableIt == valueBundles.end())
    return op.emitError() << "awaitable has no lowered runtime bundle";
  RuntimeBundle &awaitable = awaitableIt->second;

  if (awaitable.contractName() == "types.CoroutineType") {
    if (awaitable.objectEvidence.hasFlag(kCoroutineAwaitConsumedFlag))
      return op.emitError()
             << "cannot await an already awaited coroutine object";
    awaitable.objectEvidence.setFlag(kCoroutineAwaitConsumedFlag);
  }

  if (awaitable.contractName() == "types.CoroutineType" &&
      hasAsyncioSleepEvidence(awaitable))
    return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
        op.getOperation(), op.getResult(), awaitable, "await asyncio.sleep");

  if ((awaitable.contractName() == "_asyncio.Future" ||
       awaitable.contractName() == "_asyncio.Task") &&
      hasFutureTerminalEvidence(awaitable)) {
    if (mlir::failed(RuntimeBundleLowerer::lowerFutureResultEvidence(
            op.getOperation(), op.getResult(), awaitable,
            awaitable.contractName() == "_asyncio.Task"
                ? "await _asyncio.Task"
                : "await _asyncio.Future")))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (awaitable.contractName() == "_asyncio.Task" &&
      hasAsyncioSleepEvidence(awaitable))
    return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
        op.getOperation(), op.getResult(), awaitable,
        "await _asyncio.Task sleep");

  if (!awaitable.coroutineTarget.empty())
    return RuntimeBundleLowerer::lowerCoroutineObjectAwait(
        op.getOperation(), op.getResult(), awaitable, "await coroutine");

  return RuntimeBundleLowerer::lowerGeneralAwaitableIterator(op, awaitable);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerGeneralAwaitableIterator(py::AwaitOp op,
                                                    RuntimeBundle &awaitable) {
  llvm::SmallVector<const RuntimeBundle *, 1> sources{&awaitable};
  RuntimeBundle iterator;
  bool emitted = false;

  if (py::ClassOp classOp =
          RuntimeBundleLowerer::classForContract(awaitable.contract)) {
    if (std::optional<std::string> methodSymbol =
            RuntimeBundleLowerer::classMethodSymbol(classOp, "__await__")) {
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(*methodSymbol);
      if (!target)
        return op.emitError()
               << "source class method @" << *methodSymbol << " is not defined";
      if (std::optional<std::string> fieldName =
              returnedSelfFieldAwait(target)) {
        auto field = awaitable.fieldBundles.find(*fieldName);
        if (field != awaitable.fieldBundles.end() && field->second) {
          RuntimeBundle fieldAwaitable = *field->second;
          if (fieldAwaitable.boxedObject &&
              py::isAssignableTo(
                  fieldAwaitable.boxedObject->objectValue.contract,
                  fieldAwaitable.contract, op.getOperation()))
            fieldAwaitable = *fieldAwaitable.boxedObject;
          fieldAwaitable.setObjectLogicalOwnership(/*ownsObject=*/false);
          if (fieldAwaitable.contractName() == "types.CoroutineType") {
            if (fieldAwaitable.objectEvidence.hasFlag(
                    kCoroutineAwaitConsumedFlag))
              return op.emitError()
                     << "cannot await an already awaited coroutine object";
            fieldAwaitable.objectEvidence.setFlag(kCoroutineAwaitConsumedFlag);
            if (hasAsyncioSleepEvidence(fieldAwaitable))
              return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
                  op.getOperation(), op.getResult(), fieldAwaitable,
                  "await forwarded coroutine field");
            if (!fieldAwaitable.coroutineTarget.empty())
              return RuntimeBundleLowerer::lowerCoroutineObjectAwait(
                  op.getOperation(), op.getResult(), fieldAwaitable,
                  "await forwarded coroutine field");
          }
          if ((fieldAwaitable.contractName() == "_asyncio.Future" ||
               fieldAwaitable.contractName() == "_asyncio.Task") &&
              hasFutureTerminalEvidence(fieldAwaitable)) {
            if (mlir::failed(RuntimeBundleLowerer::lowerFutureResultEvidence(
                    op.getOperation(), op.getResult(), fieldAwaitable,
                    "await forwarded future field")))
              return mlir::failure();
            erase.push_back(op);
            return mlir::success();
          }
          if (fieldAwaitable.contractName() == "_asyncio.Task" &&
              hasAsyncioSleepEvidence(fieldAwaitable))
            return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
                op.getOperation(), op.getResult(), fieldAwaitable,
                "await forwarded task field");
          return RuntimeBundleLowerer::lowerGeneralAwaitableIterator(
              op, fieldAwaitable);
        }
      }
      if (target->hasAttr("ly.async.body_result"))
        return op.emitError()
               << "__await__ must return an iterator, not a coroutine";

      mlir::Type expectedResult;
      if (auto callableAttr =
              target->getAttrOfType<mlir::TypeAttr>("callable_type")) {
        if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
                callableAttr.getValue()))
          if (callable.getResultTypes().size() == 1)
            expectedResult = callable.getResultTypes().front();
      }
      if (mlir::failed(RuntimeBundleLowerer::emitSourceFunctionTargetCallResult(
              op.getOperation(), expectedResult, target, *methodSymbol, sources,
              iterator)))
        return mlir::failure();
      emitted = true;
    }
  }

  if (!emitted) {
    std::optional<EmittedRuntimeCall> emittedCall;
    if (mlir::failed(RuntimeBundleLowerer::emitManifestMethodCall(
            op.getOperation(), awaitable, "__await__", sources,
            /*allowUnusedSources=*/false, emittedCall)))
      return mlir::failure();

    mlir::Type resultType;
    if (!emittedCall->symbol.resultContract.empty())
      resultType =
          runtimeContractType(context, emittedCall->symbol.resultContract);
    else if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
                 op.getAwaitContract()))
      if (callable.getResultTypes().size() == 1)
        resultType = callable.getResultTypes().front();
    if (!resultType)
      return op.emitError() << "__await__ result needs a concrete contract";

    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op.getOperation(), resultType, emittedCall->call, iterator)))
      return mlir::failure();
    iterator.copyEvidenceFrom(awaitable);
  }

  return RuntimeBundleLowerer::lowerAwaitIteratorResult(
      op.getOperation(), op.getResult(), iterator, "await iterator");
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAwaitIteratorResult(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &iterator,
    llvm::StringRef label) {
  llvm::ArrayRef<mlir::Value> values = iterator.physicalValues();
  if (iterator.contractName() == "types.CoroutineAwaitIterator") {
    if (values.size() < 2)
      return op->emitError() << label << " has no coroutine storage evidence";
    mlir::Type object = runtimeContractType(context, "builtins.object");
    mlir::Type coroutineType =
        py::ContractType::get(context, "types.CoroutineType",
                              {object, object, resultValue.getType()});
    RuntimeBundle coroutine =
        RuntimeBundle::object(coroutineType, mlir::ValueRange{values[1]});
    coroutine.copyEvidenceFrom(iterator);
    if (coroutine.coroutineTarget.empty())
      return RuntimeBundleLowerer::lowerCoroutineStorageTargetIdAwait(
          op, resultValue, coroutine, label);
    return RuntimeBundleLowerer::lowerCoroutineObjectAwait(op, resultValue,
                                                           coroutine, label);
  }

  if (iterator.contractName() == "_asyncio.FutureIter") {
    if (values.size() < 2)
      return op->emitError() << label << " has no Future storage evidence";
    mlir::Type futureType = py::ContractType::get(context, "_asyncio.Future",
                                                  {resultValue.getType()});
    RuntimeBundle future =
        RuntimeBundle::object(futureType, mlir::ValueRange{values[1]});
    future.copyEvidenceFrom(iterator);
    if (hasFutureTerminalEvidence(future))
      return RuntimeBundleLowerer::lowerFutureResultEvidence(op, resultValue,
                                                             future, label);
  }

  if (iterator.contractName() == "_asyncio.TaskIter") {
    if (values.size() < 3)
      return op->emitError() << label << " has no Task storage evidence";
    mlir::Type taskType = py::ContractType::get(context, "_asyncio.Task",
                                                {resultValue.getType()});
    RuntimeBundle task =
        RuntimeBundle::object(taskType, mlir::ValueRange{values[1], values[2]});
    task.copyEvidenceFrom(iterator);
    if (hasFutureTerminalEvidence(task))
      return RuntimeBundleLowerer::lowerFutureResultEvidence(op, resultValue,
                                                             task, label);
    if (hasAsyncioSleepEvidence(task))
      return RuntimeBundleLowerer::lowerAsyncioSleepEvidenceAwait(
          op, resultValue, task, label);
  }

  return op->emitError()
         << "general Awaitable iterator driving is not implemented yet"
         << " (iterator_contract=" << iterator.contractName()
         << ", coroutine_sources=" << iterator.coroutineSources.size() << ")";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerCoroutineStorageTargetIdAwait(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &awaitable,
    llvm::StringRef label) {
  if (!awaitable.coroutineSources.empty())
    return op->emitError()
           << label
           << " lost static coroutine frame source evidence across an erased "
              "Awaitable boundary";
  llvm::ArrayRef<mlir::Value> coroutineValues = awaitable.physicalValues();
  if (coroutineValues.size() != 1)
    return op->emitError() << label
                           << " needs one coroutine storage value for dynamic "
                              "target dispatch";
  auto coroutineStorage =
      mlir::dyn_cast<mlir::MemRefType>(coroutineValues.front().getType());
  if (!coroutineStorage || coroutineStorage.getRank() != 1 ||
      !coroutineStorage.hasStaticShape() || coroutineStorage.getDimSize(0) < 4)
    return op->emitError() << label
                           << " coroutine storage has invalid ABI type "
                           << coroutineValues.front().getType();

  llvm::SmallVector<mlir::func::FuncOp, 8> candidates;
  llvm::SmallVector<mlir::Type, 8> expectedResults;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("ly.async.body_result") || function.isDeclaration())
      return;
    auto callableAttr =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable)
      return;
    if (!RuntimeBundleLowerer::callableLogicalInputTypes(function, callable)
             .empty())
      return;
    if (callable.getResultTypes().size() != 1 ||
        !py::isAssignableTo(callable.getResultTypes().front(),
                            resultValue.getType(), op))
      return;
    mlir::FunctionType type = function.getFunctionType();
    if (expectedResults.empty()) {
      expectedResults.append(type.getResults().begin(),
                             type.getResults().end());
    } else if (!sameTypeSequence(expectedResults, type.getResults())) {
      return;
    }
    candidates.push_back(function);
  });
  if (candidates.empty())
    return op->emitError()
           << label
           << " has no no-source async target candidate for erased coroutine";

  std::optional<RuntimeSymbol> resumeBegin =
      manifest.primitive("types.CoroutineType", "resume.begin");
  std::optional<RuntimeSymbol> resumeComplete =
      manifest.primitive("types.CoroutineType", "resume.complete");
  if (!resumeBegin || !resumeComplete)
    return op->emitError()
           << "runtime manifest has no coroutine resume primitive for "
              "types.CoroutineType";

  llvm::SmallVector<const RuntimeBundle *, 1> coroutineSource{&awaitable};
  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> resumeBeginOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *resumeBegin, coroutineSource,
                                            resumeBeginOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp resumeBeginCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *resumeBegin, resumeBeginOperands);
  if (resumeBeginCall.getNumResults() != 1 ||
      !resumeBeginCall.getResult(0).getType().isInteger(1))
    return resumeBegin->function.emitError()
           << "coroutine resume.begin primitive must return one i1";

  auto resumed = mlir::scf::IfOp::create(builder, op->getLoc(), expectedResults,
                                         resumeBeginCall.getResult(0),
                                         /*withElseRegion=*/true);

  auto emitDeadValues = [&](llvm::StringRef message)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.RuntimeError", message)))
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 4> deadValues;
    for (mlir::Type resultType : expectedResults) {
      mlir::FailureOr<mlir::Value> dead =
          RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
      if (mlir::failed(dead))
        return mlir::failure();
      deadValues.push_back(*dead);
    }
    return deadValues;
  };

  builder.setInsertionPointToStart(&resumed.getThenRegion().front());
  mlir::Value targetSlot =
      mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 3);
  mlir::Value targetId = mlir::memref::LoadOp::create(
      builder, op->getLoc(), coroutineValues.front(), targetSlot);

  std::function<mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>(unsigned)>
      emitDispatch = [&](unsigned index)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    if (index >= candidates.size())
      return emitDeadValues("unknown erased coroutine target id");
    mlir::func::FuncOp target = candidates[index];
    mlir::Value expectedId = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(),
        RuntimeBundleLowerer::functionTargetId(target.getSymName()), 64);
    mlir::Value matches = mlir::arith::CmpIOp::create(
        builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, targetId,
        expectedId);
    auto selected =
        mlir::scf::IfOp::create(builder, op->getLoc(), expectedResults, matches,
                                /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&selected.getThenRegion().front());
    RuntimeSymbol targetSymbol;
    targetSymbol.function = target;
    targetSymbol.contract = "types.CoroutineType";
    targetSymbol.role = "primitive";
    targetSymbol.name = target.getSymName();
    mlir::func::CallOp bodyCall = RuntimeBundleLowerer::createRuntimeCall(
        op->getLoc(), targetSymbol, mlir::ValueRange{});
    mlir::scf::YieldOp::create(builder, op->getLoc(), bodyCall.getResults());
    builder.setInsertionPointToStart(&selected.getElseRegion().front());
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> fallback =
        emitDispatch(index + 1);
    if (mlir::failed(fallback))
      return mlir::failure();
    mlir::scf::YieldOp::create(builder, op->getLoc(), *fallback);
    builder.setInsertionPointAfter(selected);
    llvm::SmallVector<mlir::Value, 4> results;
    results.append(selected.getResults().begin(), selected.getResults().end());
    return results;
  };
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> dispatched =
      emitDispatch(0);
  if (mlir::failed(dispatched))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> resumeCompleteOperands;
  if (mlir::failed(buildRuntimeCallOperands(
          op, *resumeComplete, coroutineSource, resumeCompleteOperands,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeComplete,
                                          resumeCompleteOperands);
  mlir::scf::YieldOp::create(builder, op->getLoc(), *dispatched);

  builder.setInsertionPointToStart(&resumed.getElseRegion().front());
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> dead = emitDeadValues(
      "cannot await a coroutine that is already running or completed");
  if (mlir::failed(dead))
    return mlir::failure();
  mlir::scf::YieldOp::create(builder, op->getLoc(), *dead);

  builder.setInsertionPointAfter(resumed);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, resultValue.getType(), resumed.getResults(), result)))
    return mlir::failure();
  valueBundles[resultValue] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerCoroutineObjectAwait(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle &awaitable,
    llvm::StringRef label) {
  if (awaitable.coroutineTarget.empty()) {
    mlir::func::FuncOp owner = op->getParentOfType<mlir::func::FuncOp>();
    return op->emitError()
           << "general Awaitable iterator driving is not implemented yet"
           << " (function="
           << (owner ? owner.getSymName() : llvm::StringRef("<unknown>"))
           << ", contract=" << awaitable.contractName()
           << ", coroutine_sources=" << awaitable.coroutineSources.size()
           << ")";
  }

  mlir::func::FuncOp target =
      module.lookupSymbol<mlir::func::FuncOp>(awaitable.coroutineTarget);
  if (!target)
    return op->emitError() << "coroutine body target '"
                           << awaitable.coroutineTarget << "' is not defined";

  std::optional<RuntimeSymbol> resumeBegin =
      manifest.primitive(awaitable.contractName(), "resume.begin");
  std::optional<RuntimeSymbol> resumeComplete =
      manifest.primitive(awaitable.contractName(), "resume.complete");
  if (!resumeBegin || !resumeComplete)
    return op->emitError() << "runtime manifest has no coroutine resume "
                              "primitive for "
                           << awaitable.contractName();

  llvm::SmallVector<const RuntimeBundle *, 1> coroutineSource{&awaitable};
  mlir::FunctionType functionType = target.getFunctionType();
  RuntimeSymbol targetSymbol;
  targetSymbol.function = target;
  targetSymbol.contract = "types.CoroutineType";
  targetSymbol.role = "primitive";
  targetSymbol.name = awaitable.coroutineTarget;

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> resumeBeginOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *resumeBegin, coroutineSource,
                                            resumeBeginOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp resumeBeginCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *resumeBegin, resumeBeginOperands);
  if (resumeBeginCall.getNumResults() != 1 ||
      !resumeBeginCall.getResult(0).getType().isInteger(1))
    return resumeBegin->function.emitError()
           << "coroutine resume.begin primitive must return one i1";

  auto resumed =
      mlir::scf::IfOp::create(builder, op->getLoc(), functionType.getResults(),
                              resumeBeginCall.getResult(0),
                              /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&resumed.getThenRegion().front());
  llvm::SmallVector<RuntimeBundle, 8> sourceBundles;
  sourceBundles.reserve(awaitable.coroutineSources.size());
  llvm::SmallVector<mlir::Value, 8> operands;
  llvm::SmallVector<mlir::Type, 8> logicalInputTypes;
  if (auto callableAttr =
          target->getAttrOfType<mlir::TypeAttr>("callable_type")) {
    if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(
            callableAttr.getValue()))
      logicalInputTypes =
          RuntimeBundleLowerer::callableLogicalInputTypes(target, callable);
  }
  unsigned inputIndex = 0;
  for (auto [sourceIndex, sourceValue] :
       llvm::enumerate(awaitable.coroutineSources)) {
    RuntimeBundle &source =
        sourceIndex < awaitable.coroutineSourceBundles.size() &&
                awaitable.coroutineSourceBundles[sourceIndex]
            ? sourceBundles.emplace_back(
                  *awaitable.coroutineSourceBundles[sourceIndex])
            : sourceBundles.emplace_back(RuntimeBundle::object(
                  sourceValue.contract, sourceValue.values));
    source.contract = sourceValue.contract;
    source.objectValue = sourceValue;
    if (inputIndex >= functionType.getNumInputs())
      return op->emitError() << "too many coroutine frame sources for "
                             << awaitable.coroutineTarget;
    mlir::LogicalResult appended =
        sourceIndex < logicalInputTypes.size()
            ? appendRuntimeSourceAs(op, targetSymbol, functionType, inputIndex,
                                    source, logicalInputTypes[sourceIndex],
                                    operands)
            : appendRuntimeSource(op, targetSymbol, functionType, inputIndex,
                                  source, operands);
    if (mlir::failed(appended))
      return mlir::failure();
    if (mlir::failed(appendPrimitiveI64EvidenceOperand(
            op, functionType, inputIndex, source, operands)))
      return mlir::failure();
  }
  if (inputIndex != functionType.getNumInputs())
    return op->emitError() << "missing coroutine frame sources for "
                           << awaitable.coroutineTarget;

  mlir::func::CallOp bodyCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), targetSymbol, operands);

  llvm::SmallVector<mlir::Value, 4> resumeCompleteOperands;
  if (mlir::failed(buildRuntimeCallOperands(
          op, *resumeComplete, coroutineSource, resumeCompleteOperands,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeComplete,
                                          resumeCompleteOperands);
  mlir::scf::YieldOp::create(builder, op->getLoc(), bodyCall.getResults());

  builder.setInsertionPointToStart(&resumed.getElseRegion().front());
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.RuntimeError",
          "cannot await a coroutine that is already running or completed")))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 4> deadValues;
  deadValues.reserve(functionType.getNumResults());
  for (mlir::Type resultType : functionType.getResults()) {
    mlir::FailureOr<mlir::Value> dead =
        RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
    if (mlir::failed(dead))
      return mlir::failure();
    deadValues.push_back(*dead);
  }
  mlir::scf::YieldOp::create(builder, op->getLoc(), deadValues);

  builder.setInsertionPointAfter(resumed);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
          op, resultValue.getType(), resumed.getResults(), result)))
    return mlir::failure();
  valueBundles[resultValue] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
