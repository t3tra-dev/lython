#include "RuntimeLowering/RuntimeBundleLowerer.h"

namespace py::runtime_lowering {

mlir::Value RuntimeBundleLowerer::stripReturnedObjectView(mlir::Value value) {
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

static bool sameReturnedCoroutineSummary(const ReturnedCoroutineSummary &lhs,
                                         const ReturnedCoroutineSummary &rhs) {
  return lhs.target == rhs.target &&
         sameTypeSequence(lhs.sourceContracts, rhs.sourceContracts);
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

static mlir::Value stripObjectViewForSummary(mlir::Value value) {
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

static std::optional<ReturnedObjectEvidenceSummary>
summarizeFutureTerminalCall(py::CallOp call, mlir::Value returnedFuture) {
  if (stripObjectViewForSummary(call.getCallable()) != returnedFuture)
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
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return mlir::WalkResult::advance();

    std::optional<ReturnedValueSummary> summary;
    bool sawReturn = false;
    bool invalid = false;
    mlir::Block &entry = function.getBody().front();
    function.walk([&](mlir::func::ReturnOp ret) {
      if (ret->getParentOfType<mlir::func::FuncOp>() != function || invalid)
        return;

      ReturnedValueSummary candidate;
      candidate.argumentIndices.reserve(ret.getNumOperands());
      for (mlir::Value operand : ret.getOperands()) {
        mlir::Value returned = stripReturnedObjectView(operand);
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(returned);
        if (!argument || argument.getOwner() != &entry) {
          invalid = true;
          return;
        }
        candidate.argumentIndices.push_back(argument.getArgNumber());
      }

      if (!summary) {
        summary = std::move(candidate);
      } else if (summary->argumentIndices != candidate.argumentIndices) {
        invalid = true;
        return;
      }
      sawReturn = true;
    });

    if (!invalid && sawReturn && summary)
      returnedValueSummaries[function.getSymName()] = std::move(*summary);
    return mlir::WalkResult::advance();
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
          RuntimeBundleLowerer::stripReturnedObjectView(ret.getOperand(0));
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
        RuntimeBundleLowerer::stripReturnedObjectView(call.getCallable());
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

  bool changed = true;
  while (changed) {
    changed = false;
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
  }

  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::buildReturnedCoroutineSummaries() {
  llvm::SmallVector<mlir::func::FuncOp, 8> functions;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr("callable_type") || function.isDeclaration())
      return;
    functions.push_back(function);
  });

  auto directCoroutineSourceContracts =
      [&](py::CallOp call, mlir::func::FuncOp target, py::BindingRefOp binding)
      -> std::optional<llvm::SmallVector<mlir::Type, 4>> {
    auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable)
      return std::nullopt;

    std::optional<llvm::SmallVector<mlir::Type, 4>> sourceContracts =
        RuntimeBundleLowerer::collectCallableArgumentSourceTypes(call,
                                                                 callable);
    if (!sourceContracts)
      return std::nullopt;

    llvm::SmallVector<mlir::Type, 4> closureTypes =
        callableClosureTypes(target);
    if (closureTypes.size() != binding.getCaptures().size())
      return std::nullopt;
    for (mlir::Value capture : binding.getCaptures())
      sourceContracts->push_back(capture.getType());
    auto aggregateEvidence =
        callableAggregateEvidenceABIs.find(target.getSymName());
    if (aggregateEvidence != callableAggregateEvidenceABIs.end()) {
      sourceContracts->append(
          aggregateEvidence->second.varargElementTypes.begin(),
          aggregateEvidence->second.varargElementTypes.end());
      for (mlir::Type valueType : aggregateEvidence->second.kwargValueTypes) {
        sourceContracts->push_back(valueType);
        if (aggregateEvidence->second.kwargIsFull)
          sourceContracts->push_back(
              runtimeContractType(context, "builtins.bool"));
      }
    }
    return sourceContracts;
  };

  auto summarizeCall =
      [&](py::CallOp call) -> std::optional<ReturnedCoroutineSummary> {
    if (call.getNumResults() != 1)
      return std::nullopt;
    if (runtimeContractName(call.getResult(0).getType()) !=
        "types.CoroutineType")
      return std::nullopt;

    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;
    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target)
      return std::nullopt;

    if (target->hasAttr("ly.async.body_result")) {
      std::optional<llvm::SmallVector<mlir::Type, 4>> sourceContracts =
          directCoroutineSourceContracts(call, target, binding);
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

  bool changed = true;
  while (changed) {
    changed = false;
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
  }

  return mlir::success();
}

} // namespace py::runtime_lowering
