#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

static bool sameReturnedCoroutineSummary(const ReturnedCoroutineSummary &lhs,
                                         const ReturnedCoroutineSummary &rhs) {
  return lhs.target == rhs.target &&
         sameTypeSequence(lhs.sourceContracts, rhs.sourceContracts);
}

static bool isAwaitIteratorContract(llvm::StringRef contract) {
  return contract == "types.CoroutineAwaitIterator" ||
         contract == "_asyncio.FutureIter" || contract == "_asyncio.TaskIter";
}

static bool
sameReturnedObjectEvidenceSummary(const ReturnedObjectEvidenceSummary &lhs,
                                  const ReturnedObjectEvidenceSummary &rhs) {
  if (lhs.objectContract != rhs.objectContract ||
      lhs.resultIndex != rhs.resultIndex || lhs.flags != rhs.flags ||
      lhs.slots.size() != rhs.slots.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs.slots, rhs.slots))
    if (left.name != right.name || left.sourceContract != right.sourceContract)
      return false;
  return true;
}

static bool sameReturnedStaticObjectSummary(
    const ReturnedStaticObjectSummary &lhs,
    const ReturnedStaticObjectSummary &rhs) {
  return lhs.objectContract == rhs.objectContract &&
         lhs.resultIndex == rhs.resultIndex;
}

static bool emptyStaticPack(mlir::Value value) {
  auto pack = value.getDefiningOp<py::PackOp>();
  return pack && pack.getValues().empty();
}

static std::optional<llvm::SmallVector<mlir::Value, 4>>
staticPackValues(mlir::Value value) {
  auto pack = value.getDefiningOp<py::PackOp>();
  if (!pack)
    return std::nullopt;
  llvm::SmallVector<mlir::Value, 4> values;
  values.append(pack.getValues().begin(), pack.getValues().end());
  return values;
}

static std::optional<ReturnedObjectEvidenceSummary>
summarizeFutureTerminalCall(py::CallOp call, mlir::Value returnedFuture) {
  if (stripReturnedObjectView(call.getCallable()) != returnedFuture)
    return std::nullopt;
  auto method = call->getAttrOfType<mlir::StringAttr>("ly.bound_method");
  if (!method)
    return std::nullopt;
  if (!emptyStaticPack(call.getKwnames()) ||
      !emptyStaticPack(call.getKwvalues()))
    return std::nullopt;
  std::optional<llvm::SmallVector<mlir::Value, 4>> posargs =
      staticPackValues(call.getPosargs());
  if (!posargs)
    return std::nullopt;

  ReturnedObjectEvidenceSummary summary;
  summary.objectContract = returnedFuture.getType();
  llvm::StringRef methodName = method.getValue();
  if (methodName == "set_result") {
    if (posargs->size() != 1 ||
        runtimeShapeContractName((*posargs)[0].getType()).empty())
      return std::nullopt;
    summary.slots.push_back(ReturnedObjectEvidenceSlot{
        kFutureResultSlot.str(), (*posargs)[0].getType()});
    return summary;
  }

  if (methodName == "set_exception") {
    if (posargs->size() != 1 ||
        runtimeShapeContractName((*posargs)[0].getType()).empty())
      return std::nullopt;
    summary.slots.push_back(ReturnedObjectEvidenceSlot{
        kFutureExceptionSlot.str(), (*posargs)[0].getType()});
    return summary;
  }

  if (methodName == "cancel") {
    if (posargs->size() > 1)
      return std::nullopt;
    summary.flags.push_back(kFutureCancelledFlag.str());
    if (posargs->empty() ||
        runtimeContractName((*posargs)[0].getType()) == "types.NoneType")
      return summary;
    if (runtimeContractName((*posargs)[0].getType()) != "builtins.str" ||
        runtimeShapeContractName((*posargs)[0].getType()).empty())
      return std::nullopt;
    summary.slots.push_back(ReturnedObjectEvidenceSlot{
        kFutureCancelMessageSlot.str(), (*posargs)[0].getType()});
    return summary;
  }

  return std::nullopt;
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedValueSummaries() {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  module.walk([&](mlir::func::FuncOp function) {
    if (function->hasAttr("callable_type") && !function.isDeclaration())
      functions.push_back(function);
  });

  auto entryArgumentIndex = [](mlir::func::FuncOp function,
                               mlir::Value value) -> std::optional<unsigned> {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || function.empty() ||
        argument.getOwner() != &function.getBody().front())
      return std::nullopt;
    return argument.getArgNumber();
  };

  auto summarizeOperand = [&](mlir::func::FuncOp function, mlir::Value operand)
      -> std::optional<unsigned> {
    mlir::Value returned = stripReturnedObjectView(operand);
    if (std::optional<unsigned> direct =
            entryArgumentIndex(function, returned))
      return direct;

    auto call = returned.getDefiningOp<py::CallOp>();
    if (!call || call.getNumResults() != 1)
      return std::nullopt;
    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    py::CallableType callable = callableTypeOf(target);
    if (!callable)
      return std::nullopt;
    auto returnedSummary = returnedValueSummaries.find(target.getSymName());
    if (returnedSummary == returnedValueSummaries.end() ||
        returnedSummary->second.argumentIndices.size() != 1)
      return std::nullopt;

    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callable,
                                                          /*emitErrors=*/false);
    if (!invocation || !plan)
      return std::nullopt;

    unsigned logicalIndex = returnedSummary->second.argumentIndices.front();
    if (logicalIndex >= plan->fixedActuals.size() ||
        !plan->fixedActuals[logicalIndex])
      return std::nullopt;
    unsigned actualIndex = *plan->fixedActuals[logicalIndex];
    if (actualIndex >= invocation->actualValues.size())
      return std::nullopt;
    return entryArgumentIndex(
        function,
        stripReturnedObjectView(invocation->actualValues[actualIndex]));
  };

  auto summarizeFunction =
      [&](mlir::func::FuncOp function) -> std::optional<ReturnedValueSummary> {
    std::optional<ReturnedValueSummary> summary;
    bool sawReturn = false;
    bool allReturnsSummarized = true;
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      sawReturn = true;
      ReturnedValueSummary candidate;
      candidate.argumentIndices.reserve(ret.getNumOperands());
      for (mlir::Value operand : ret.getOperands()) {
        std::optional<unsigned> index = summarizeOperand(function, operand);
        if (!index) {
          allReturnsSummarized = false;
          return;
        }
        candidate.argumentIndices.push_back(*index);
      }
      if (!summary) {
        summary = std::move(candidate);
        return;
      }
      if (summary->argumentIndices != candidate.argumentIndices)
        allReturnsSummarized = false;
    });
    if (!sawReturn || !allReturnsSummarized || !summary)
      return std::nullopt;
    return summary;
  };

  runToFixpoint([&](bool &changed) {
    for (mlir::func::FuncOp function : functions) {
      std::optional<ReturnedValueSummary> summary =
          summarizeFunction(function);
      if (!summary)
        continue;
      auto found = returnedValueSummaries.find(function.getSymName());
      if (found != returnedValueSummaries.end() &&
          found->second.argumentIndices == summary->argumentIndices)
        continue;
      returnedValueSummaries[function.getSymName()] = std::move(*summary);
      changed = true;
    }
  });
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedCallableSummaries() {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return mlir::WalkResult::advance();

    ReturnedCallableSummary summary;
    bool sawReturn = false;
    bool invalid = false;
    mlir::Block &entry = function.getBody().front();
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function || invalid)
        return;
      sawReturn = true;
      if (ret.getNumOperands() != 1) {
        invalid = true;
        return;
      }
      mlir::Value returned = stripReturnedObjectView(ret.getOperand(0));
      auto binding = returned.getDefiningOp<py::BindingRefOp>();
      if (!binding) {
        invalid = true;
        return;
      }
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
      if (!target || !target->hasAttr("callable_type")) {
        invalid = true;
        return;
      }

      ReturnedCallableAlternativeSummary candidate;
      candidate.target = binding.getBinding().str();
      for (mlir::Value capture : binding.getCaptures()) {
        auto arg = mlir::dyn_cast<mlir::BlockArgument>(capture);
        if (!arg || arg.getOwner() != &entry) {
          invalid = true;
          return;
        }
        candidate.captureArgumentIndices.push_back(arg.getArgNumber());
      }

      auto sameCandidate =
          [&](const ReturnedCallableAlternativeSummary &alternative) {
            return alternative.target == candidate.target &&
                   alternative.captureArgumentIndices ==
                       candidate.captureArgumentIndices;
          };
      if (!llvm::any_of(summary.alternatives, sameCandidate))
        summary.alternatives.push_back(std::move(candidate));
    });

    if (!invalid && sawReturn && !summary.alternatives.empty())
      returnedCallableSummaries[function.getSymName()] = std::move(summary);
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult
RuntimeBundleLowerer::buildReturnedObjectEvidenceSummaries() {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return;
    functions.push_back(function);
  });

  auto singleReturnedObject =
      [&](mlir::func::FuncOp function) -> std::optional<mlir::Value> {
    std::optional<mlir::Value> returnedObject;
    bool sawReturn = false;
    bool valid = true;
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function || !valid)
        return;
      sawReturn = true;
      if (ret.getNumOperands() != 1) {
        valid = false;
        return;
      }
      mlir::Value returned =
          stripReturnedObjectView(ret.getOperand(0));
      if (runtimeContractName(returned.getType()).empty()) {
        valid = false;
        return;
      }
      if (!returnedObject) {
        returnedObject = returned;
        return;
      }
      if (*returnedObject != returned)
        valid = false;
    });
    if (!sawReturn || !valid || !returnedObject)
      return std::nullopt;
    return returnedObject;
  };

  auto summarizeNestedCall = [&](mlir::Value returnedObject)
      -> std::optional<ReturnedObjectEvidenceSummary> {
    auto call = returnedObject.getDefiningOp<py::CallOp>();
    if (!call || call.getNumResults() != 1)
      return std::nullopt;
    mlir::Value callee =
        stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;
    auto found = returnedObjectEvidenceSummaries.find(binding.getBinding());
    if (found == returnedObjectEvidenceSummaries.end())
      return std::nullopt;
    if (!compatibleRuntimeObjectEvidenceContract(returnedObject.getType(),
                                                 found->second.objectContract))
      return std::nullopt;
    ReturnedObjectEvidenceSummary summary = found->second;
    summary.objectContract = returnedObject.getType();
    return summary;
  };

  auto summarizeDirectFuture = [&](mlir::func::FuncOp function,
                                   mlir::Value returnedObject)
      -> std::optional<ReturnedObjectEvidenceSummary> {
    if (runtimeContractName(returnedObject.getType()) != "_asyncio.Future")
      return std::nullopt;
    std::optional<ReturnedObjectEvidenceSummary> summary;
    bool invalid = false;
    function.walk([&](py::CallOp call) {
      if (invalid)
        return;
      std::optional<ReturnedObjectEvidenceSummary> candidate =
          summarizeFutureTerminalCall(call, returnedObject);
      if (!candidate)
        return;
      if (summary) {
        invalid = true;
        return;
      }
      summary = std::move(candidate);
    });
    if (invalid || !summary)
      return std::nullopt;
    return summary;
  };

  auto summarizeFunction = [&](mlir::func::FuncOp function)
      -> std::optional<ReturnedObjectEvidenceSummary> {
    std::optional<mlir::Value> returnedObject = singleReturnedObject(function);
    if (!returnedObject)
      return std::nullopt;
    if (std::optional<ReturnedObjectEvidenceSummary> direct =
            summarizeDirectFuture(function, *returnedObject))
      return direct;
    return summarizeNestedCall(*returnedObject);
  };

  runToFixpoint([&](bool &changed) {
    for (mlir::func::FuncOp function : functions) {
      std::optional<ReturnedObjectEvidenceSummary> summary =
          summarizeFunction(function);
      if (!summary)
        continue;
      auto found = returnedObjectEvidenceSummaries.find(function.getSymName());
      if (found != returnedObjectEvidenceSummaries.end() &&
          sameReturnedObjectEvidenceSummary(found->second, *summary))
        continue;
      returnedObjectEvidenceSummaries[function.getSymName()] =
          std::move(*summary);
      changed = true;
    }
  });

  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedStaticObjectSummaries() {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return;
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      return;
    functions.push_back(function);
  });

  auto concreteObjectContract = [&](mlir::Value value,
                                    mlir::Type expectedResult)
      -> std::optional<mlir::Type> {
    mlir::Value stripped = stripReturnedObjectView(value);
    mlir::Type type = stripped.getType();
    if (runtimeContractName(type).empty())
      return std::nullopt;
    if (py::isAssignableTo(type, expectedResult, stripped.getDefiningOp()))
      return type;
    if (auto unionType = mlir::dyn_cast<py::UnionType>(expectedResult)) {
      for (mlir::Type member : unionType.getMemberTypes()) {
        if (py::isPyNoneType(member))
          continue;
        if (py::isAssignableTo(type, member, stripped.getDefiningOp()))
          return type;
      }
    }
    return std::nullopt;
  };

  auto unionAllowsEmptyStaticObjectEvidence = [](mlir::Type type) {
    auto unionType = mlir::dyn_cast<py::UnionType>(type);
    return unionType &&
           llvm::any_of(unionType.getMemberTypes(), [](mlir::Type member) {
             return py::isPyNoneType(member);
           });
  };

  struct StaticObjectCandidate {
    bool hasObject = false;
    mlir::Type objectContract;
  };

  auto emptyStaticObjectCandidate =
      [&](mlir::Type expectedResult) -> std::optional<StaticObjectCandidate> {
    if (!unionAllowsEmptyStaticObjectEvidence(expectedResult))
      return std::nullopt;
    return StaticObjectCandidate{};
  };

  auto directStaticObjectCandidate =
      [&](mlir::Value value,
          mlir::Type expectedResult) -> std::optional<StaticObjectCandidate> {
    mlir::Value stripped = stripReturnedObjectView(value);
    if (py::isPyNoneType(stripped.getType()))
      return emptyStaticObjectCandidate(expectedResult);
    if (auto direct = concreteObjectContract(stripped, expectedResult))
      return StaticObjectCandidate{/*hasObject=*/true, *direct};
    if (auto wrap = stripped.getDefiningOp<py::UnionWrapOp>()) {
      mlir::Value input =
          stripReturnedObjectView(wrap.getInput());
      if (py::isPyNoneType(input.getType()))
        return emptyStaticObjectCandidate(expectedResult);
      if (auto wrapped = concreteObjectContract(input, expectedResult))
        return StaticObjectCandidate{/*hasObject=*/true, *wrapped};
    }
    return std::nullopt;
  };

  auto resultSupportsStaticObjectEvidence = [](mlir::Type resultType) {
    if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(resultType))
      return runtimeContractName(resultType).empty() &&
             runtimeShapeContractName(resultType) == "builtins.object";
    auto unionType = mlir::dyn_cast<py::UnionType>(resultType);
    if (!unionType)
      return false;
    bool hasNone = false;
    bool hasObjectMember = false;
    for (mlir::Type member : unionType.getMemberTypes()) {
      if (py::isPyNoneType(member)) {
        hasNone = true;
        continue;
      }
      if (!runtimeContractName(member).empty())
        hasObjectMember = true;
    }
    return hasNone && hasObjectMember;
  };

  auto staticObjectCandidateFromSummary =
      [&](mlir::Type candidate, mlir::Type expectedResult)
      -> std::optional<StaticObjectCandidate> {
    if (py::isAssignableTo(candidate, expectedResult))
      return StaticObjectCandidate{/*hasObject=*/true, candidate};
    if (auto unionType = mlir::dyn_cast<py::UnionType>(expectedResult)) {
      for (mlir::Type member : unionType.getMemberTypes()) {
        if (!py::isPyNoneType(member) && py::isAssignableTo(candidate, member))
          return StaticObjectCandidate{/*hasObject=*/true, candidate};
      }
    }
    return std::nullopt;
  };

  auto summarizeNestedReturnedValue =
      [&](mlir::func::FuncOp function, py::CallOp call,
          mlir::func::FuncOp target, py::CallableType callable,
          mlir::Type expectedResult)
      -> std::optional<StaticObjectCandidate> {
    auto returnedValue = returnedValueSummaries.find(target.getSymName());
    if (returnedValue == returnedValueSummaries.end() ||
        returnedValue->second.argumentIndices.size() != 1)
      return std::nullopt;

    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callable,
                                                          /*emitErrors=*/false);
    if (!invocation || !plan)
      return std::nullopt;

    unsigned logicalIndex = returnedValue->second.argumentIndices.front();
    if (logicalIndex >= plan->fixedActuals.size() ||
        !plan->fixedActuals[logicalIndex])
      return std::nullopt;
    unsigned actualIndex = *plan->fixedActuals[logicalIndex];
    if (actualIndex >= invocation->actualValues.size())
      return std::nullopt;
    mlir::Value actual =
        stripReturnedObjectView(
            invocation->actualValues[actualIndex]);
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(actual))
      if (!function.empty() && argument.getOwner() == &function.getBody().front())
        return std::nullopt;
    if (auto direct = directStaticObjectCandidate(actual, expectedResult))
      return direct;
    return std::nullopt;
  };

  auto summarizeOperand = [&](mlir::func::FuncOp function, mlir::Value operand,
                              mlir::Type expectedResult)
      -> std::optional<StaticObjectCandidate> {
    mlir::Value stripped = stripReturnedObjectView(operand);
    if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(stripped))
      if (!function.empty() && argument.getOwner() == &function.getBody().front())
        return std::nullopt;
    if (auto direct = directStaticObjectCandidate(operand, expectedResult))
      return direct;

    mlir::Value returned = stripped;
    if (auto wrap = returned.getDefiningOp<py::UnionWrapOp>())
      returned = stripReturnedObjectView(wrap.getInput());
    auto call = returned.getDefiningOp<py::CallOp>();
    if (!call || call.getNumResults() != 1)
      return std::nullopt;

    if (auto method = call->getAttrOfType<mlir::StringAttr>("ly.bound_method")) {
      mlir::Value receiver =
          stripReturnedObjectView(call.getCallable());
      std::string receiverContract = runtimeContractName(receiver.getType());
      if (!receiverContract.empty()) {
        for (const RuntimeSymbol &symbol :
             manifest.methodCandidates(receiverContract, method.getValue())) {
          if (!isAwaitIteratorContract(symbol.resultContract))
            continue;
          mlir::Type objectContract =
              runtimeContractType(context, symbol.resultContract);
          if (auto protocol =
                  mlir::dyn_cast_if_present<py::ProtocolType>(expectedResult))
            if (protocol.getProtocolName() == "Generator")
              return StaticObjectCandidate{/*hasObject=*/true, objectContract};
          if (py::isAssignableTo(objectContract, expectedResult,
                                 call.getOperation()))
            return StaticObjectCandidate{/*hasObject=*/true, objectContract};
        }
      }
    }

    mlir::Value callee = stripReturnedObjectView(
        call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    py::CallableType callable = callableTypeOf(target);
    if (!callable)
      return std::nullopt;

    if (auto nested = returnedStaticObjectSummaries.find(target.getSymName());
        nested != returnedStaticObjectSummaries.end() &&
        py::isAssignableTo(nested->second.objectContract, expectedResult,
                           call.getOperation()))
      return StaticObjectCandidate{/*hasObject=*/true,
                                   nested->second.objectContract};
    if (auto nested = returnedStaticObjectSummaries.find(target.getSymName());
        nested != returnedStaticObjectSummaries.end()) {
      if (auto candidate = staticObjectCandidateFromSummary(
              nested->second.objectContract, expectedResult))
        return candidate;
    }

    return summarizeNestedReturnedValue(function, call, target, callable,
                                        expectedResult);
  };

  auto summarizeFunction =
      [&](mlir::func::FuncOp function)
      -> std::optional<ReturnedStaticObjectSummary> {
    py::CallableType callable = callableTypeOf(function);
    if (!callable || callable.getResultTypes().size() != 1)
      return std::nullopt;
    mlir::Type resultType = callable.getResultTypes().front();
    if (!resultSupportsStaticObjectEvidence(resultType))
      return std::nullopt;

    std::optional<mlir::Type> objectContract;
    bool sawReturn = false;
    bool allReturnsSummarized = true;
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      sawReturn = true;
      if (ret.getNumOperands() != 1) {
        allReturnsSummarized = false;
        return;
      }
      std::optional<StaticObjectCandidate> candidate =
          summarizeOperand(function, ret.getOperand(0), resultType);
      if (!candidate) {
        allReturnsSummarized = false;
        return;
      }
      if (!candidate->hasObject)
        return;
      if (!objectContract) {
        objectContract = candidate->objectContract;
        return;
      }
      if (*objectContract != candidate->objectContract)
        allReturnsSummarized = false;
    });

    if (!sawReturn || !allReturnsSummarized || !objectContract)
      return std::nullopt;
    ReturnedStaticObjectSummary summary;
    summary.objectContract = *objectContract;
    summary.resultIndex = 0;
    return summary;
  };

  runToFixpoint([&](bool &changed) {
    for (mlir::func::FuncOp function : functions) {
      std::optional<ReturnedStaticObjectSummary> summary =
          summarizeFunction(function);
      if (!summary)
        continue;
      auto found = returnedStaticObjectSummaries.find(function.getSymName());
      if (found != returnedStaticObjectSummaries.end() &&
          sameReturnedStaticObjectSummary(found->second, *summary))
        continue;
      returnedStaticObjectSummaries[function.getSymName()] = std::move(*summary);
      changed = true;
    }
  });

  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedCoroutineSummaries() {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return;
    functions.push_back(function);
  });

  auto appendAggregateSourceContracts =
      [&](llvm::StringRef targetName,
          llvm::SmallVectorImpl<mlir::Type> &sourceContracts) {
    auto aggregateEvidence = callableAggregateEvidenceABIs.find(targetName);
    if (aggregateEvidence == callableAggregateEvidenceABIs.end())
      return;
    sourceContracts.append(
        aggregateEvidence->second.varargElementTypes.begin(),
        aggregateEvidence->second.varargElementTypes.end());
    for (mlir::Type valueType : aggregateEvidence->second.kwargValueTypes) {
      sourceContracts.push_back(valueType);
      if (aggregateEvidence->second.kwargIsFull)
        sourceContracts.push_back(runtimeContractType(context,
                                                      "builtins.bool"));
    }
  };

  auto asyncTargetSourceContracts =
      [&](py::CallOp call, mlir::func::FuncOp target,
          llvm::ArrayRef<mlir::Type> closureSourceTypes)
      -> std::optional<llvm::SmallVector<mlir::Type, 4>> {
    py::CallableType callable = callableTypeOf(target);
    if (!callable)
      return std::nullopt;

    std::optional<llvm::SmallVector<mlir::Type, 4>> sourceContracts =
        RuntimeBundleLowerer::collectCallableArgumentSourceTypes(call,
                                                                 callable);
    if (!sourceContracts)
      return std::nullopt;

    llvm::SmallVector<mlir::Type, 4> closureTypes =
        callableClosureTypes(target);
    if (closureTypes.size() != closureSourceTypes.size())
      return std::nullopt;
    sourceContracts->append(closureSourceTypes.begin(),
                            closureSourceTypes.end());
    appendAggregateSourceContracts(target.getSymName(), *sourceContracts);
    return sourceContracts;
  };

  auto summarizeTarget =
      [&](py::CallOp call, mlir::func::FuncOp target,
          llvm::ArrayRef<mlir::Type> closureSourceTypes)
      -> std::optional<ReturnedCoroutineSummary> {
    if (!target)
      return std::nullopt;

    if (target->hasAttr("ly.async.body_result")) {
      std::optional<llvm::SmallVector<mlir::Type, 4>> sourceContracts =
          asyncTargetSourceContracts(call, target, closureSourceTypes);
      if (!sourceContracts)
        return std::nullopt;
      ReturnedCoroutineSummary candidate;
      candidate.target = target.getSymName().str();
      candidate.sourceContracts = std::move(*sourceContracts);
      return candidate;
    }

    auto nested = returnedCoroutineSummaries.find(target.getSymName());
    if (nested == returnedCoroutineSummaries.end())
      return std::nullopt;
    return nested->second;
  };

  auto summarizeCallableArgument =
      [&](py::CallOp call, mlir::BlockArgument argument)
      -> std::optional<ReturnedCoroutineSummary> {
    mlir::func::FuncOp owner = call->getParentOfType<mlir::func::FuncOp>();
    if (!owner || owner.empty() ||
        argument.getOwner() != &owner.getBody().front())
      return std::nullopt;

    auto evidence = callableArgumentEvidenceABIs.find(owner.getSymName());
    if (evidence == callableArgumentEvidenceABIs.end() ||
        argument.getArgNumber() >= evidence->second.logicalArguments.size())
      return std::nullopt;

    const RuntimeArgumentEvidenceSet &evidenceSet =
        evidence->second.logicalArguments[argument.getArgNumber()];
    if (evidenceSet.empty())
      return std::nullopt;

    std::optional<ReturnedCoroutineSummary> summary;
    bool sawCallableAlternative = false;
    for (const RuntimeArgumentEvidence &alternative :
         evidenceSet.alternatives) {
      if (alternative.functionTarget.empty())
        continue;
      sawCallableAlternative = true;
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(alternative.functionTarget);
      std::optional<ReturnedCoroutineSummary> candidate =
          summarizeTarget(call, target, alternative.closureValueTypes);
      if (!candidate)
        return std::nullopt;
      if (!summary) {
        summary = std::move(candidate);
        continue;
      }
      if (!sameReturnedCoroutineSummary(*summary, *candidate))
        return std::nullopt;
    }
    if (!sawCallableAlternative)
      return std::nullopt;
    return summary;
  };

  auto summarizeCall =
      [&](py::CallOp call) -> std::optional<ReturnedCoroutineSummary> {
    if (call.getNumResults() != 1)
      return std::nullopt;
    if (!isCoroutineLikeResultType(call.getResult(0).getType())) {
      auto method =
          call->getAttrOfType<mlir::StringAttr>("ly.bound_method");
      if (!method || method.getValue() != "__await__")
        return std::nullopt;

      mlir::Value receiver = stripReturnedObjectView(call.getCallable());
      if (auto receiverCall = receiver.getDefiningOp<py::CallOp>()) {
        if (!isCoroutineLikeResultType(receiverCall.getResult(0).getType()))
          return std::nullopt;
        mlir::Value callee = stripReturnedObjectView(receiverCall.getCallable());
        auto binding = callee.getDefiningOp<py::BindingRefOp>();
        if (!binding) {
          if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(callee))
            return summarizeCallableArgument(receiverCall, argument);
          return std::nullopt;
        }
        mlir::func::FuncOp target =
            module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
        llvm::SmallVector<mlir::Type, 4> closureSourceTypes;
        closureSourceTypes.reserve(binding.getCaptures().size());
        for (mlir::Value capture : binding.getCaptures())
          closureSourceTypes.push_back(capture.getType());
        return summarizeTarget(receiverCall, target, closureSourceTypes);
      }
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(receiver))
        return summarizeCallableArgument(call, argument);
      return std::nullopt;
    }

    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding) {
      if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(callee))
        return summarizeCallableArgument(call, argument);
      return std::nullopt;
    }
    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    llvm::SmallVector<mlir::Type, 4> closureSourceTypes;
    closureSourceTypes.reserve(binding.getCaptures().size());
    for (mlir::Value capture : binding.getCaptures())
      closureSourceTypes.push_back(capture.getType());
    return summarizeTarget(call, target, closureSourceTypes);
  };

  auto summarizeFunction = [&](mlir::func::FuncOp function)
      -> std::optional<ReturnedCoroutineSummary> {
    std::optional<ReturnedCoroutineSummary> summary;
    bool sawReturn = false;
    bool allReturnsSummarized = true;
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      sawReturn = true;
      if (ret.getNumOperands() != 1) {
        allReturnsSummarized = false;
        return;
      }

      mlir::Value returned = stripReturnedObjectView(ret.getOperand(0));
      auto call = returned.getDefiningOp<py::CallOp>();
      if (!call) {
        allReturnsSummarized = false;
        return;
      }

      std::optional<ReturnedCoroutineSummary> candidate = summarizeCall(call);
      if (!candidate) {
        allReturnsSummarized = false;
        return;
      }
      if (!summary) {
        summary = std::move(candidate);
        return;
      }
      if (!sameReturnedCoroutineSummary(*summary, *candidate))
        allReturnsSummarized = false;
    });

    if (!sawReturn || !allReturnsSummarized || !summary)
      return std::nullopt;
    return summary;
  };

  runToFixpoint([&](bool &changed) {
    for (mlir::func::FuncOp function : functions) {
      std::optional<ReturnedCoroutineSummary> summary =
          summarizeFunction(function);
      if (!summary)
        continue;
      auto found = returnedCoroutineSummaries.find(function.getSymName());
      if (found != returnedCoroutineSummaries.end() &&
          sameReturnedCoroutineSummary(found->second, *summary))
        continue;
      returnedCoroutineSummaries[function.getSymName()] = std::move(*summary);
      changed = true;
    }
  });

  return mlir::success();
}

} // namespace py::lowering
