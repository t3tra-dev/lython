#include "Runtime/Evidence/Callable.h"

namespace py::lowering {

using namespace callable_evidence;

mlir::LogicalResult RuntimeBundleLowerer::buildCallableArgumentEvidenceABIs() {
  enum EvidenceKind : unsigned {
    CallableEvidence = 1u << 0,
    CoroutineEvidence = 1u << 1,
  };

  struct Slot {
    llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;
    unsigned invalidKinds = 0;
  };

  auto mergeEvidence = [&](Slot &slot,
                           RuntimeArgumentEvidence evidence,
                           unsigned requiredKinds) -> bool {
    if (evidence.empty()) {
      unsigned before = slot.invalidKinds;
      slot.invalidKinds |= requiredKinds;
      bool changed = slot.invalidKinds != before;
      return changed;
    }
    auto sameEvidence = [&](const RuntimeArgumentEvidence &candidate) {
      return candidate.functionTarget == evidence.functionTarget &&
             sameTypeSequence(candidate.closureValueTypes,
                              evidence.closureValueTypes) &&
             candidate.coroutineTarget == evidence.coroutineTarget &&
             sameTypeSequence(candidate.coroutineSourceTypes,
                              evidence.coroutineSourceTypes);
    };
    if (llvm::any_of(slot.alternatives, sameEvidence))
      return false;
    slot.alternatives.push_back(std::move(evidence));
    return true;
  };

  auto logicalSourceType = [&](mlir::func::FuncOp target,
                               py::CallableType callable,
                               const StaticCallableInvocation &invocation,
                               const CallableArgumentPlan &plan,
                               unsigned logicalIndex) -> mlir::Type {
    llvm::SmallVector<mlir::Type, 8> fixedTypes(
        callable.getPositionalTypes().begin(),
        callable.getPositionalTypes().end());
    fixedTypes.append(callable.getKwOnlyTypes().begin(),
                      callable.getKwOnlyTypes().end());
    if (logicalIndex < fixedTypes.size()) {
      if (logicalIndex >= plan.fixedActuals.size() ||
          !plan.fixedActuals[logicalIndex])
        return fixedTypes[logicalIndex];
      unsigned actualIndex = *plan.fixedActuals[logicalIndex];
      return actualIndex < invocation.actualTypes.size()
                 ? invocation.actualTypes[actualIndex]
                 : mlir::Type();
    }
    unsigned variadicIndex = static_cast<unsigned>(fixedTypes.size());
    if (callable.hasVararg() && logicalIndex == variadicIndex)
      return callableVarargValueType(target, callable);
    if (callable.hasVararg())
      ++variadicIndex;
    if (callable.hasKwarg() && logicalIndex == variadicIndex)
      return callableKwargValueType(target, callable);
    return {};
  };

  auto isCoroutineLikeType = [](mlir::Type type) {
    if (runtimeContractName(type) == "types.CoroutineType")
      return true;
    auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type);
    return protocol && protocol.getProtocolName() == "Coroutine";
  };

  auto coroutineEvidenceFromCall =
      [&](py::CallOp call) -> std::optional<RuntimeArgumentEvidence> {
    if (!call || call.getNumResults() != 1 ||
        !isCoroutineLikeType(call.getResult(0).getType()))
      return std::nullopt;

    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration() ||
        !target->hasAttr("ly.async.body_result"))
      return std::nullopt;

    py::CallableType callable = callableTypeOf(target);
    if (!callable)
      return std::nullopt;

    std::optional<llvm::SmallVector<mlir::Type, 4>> sourceTypes =
        RuntimeBundleLowerer::collectCallableArgumentSourceTypes(call,
                                                                 callable);
    if (!sourceTypes)
      return std::nullopt;

    llvm::SmallVector<mlir::Type, 4> closureTypes =
        callableClosureTypes(target);
    sourceTypes->append(closureTypes.begin(), closureTypes.end());

    RuntimeArgumentEvidence evidence;
    evidence.coroutineTarget = target.getSymName().str();
    evidence.coroutineSourceTypes = std::move(*sourceTypes);
    return evidence;
  };

  auto evidenceAlternativesFromValue = [&](mlir::Value value,
                                           unsigned requiredKinds)
      -> std::optional<llvm::SmallVector<RuntimeArgumentEvidence, 4>> {
    value = stripReturnedObjectView(value);
    llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;
    unsigned coveredKinds = 0;

    if ((requiredKinds & CallableEvidence) != 0) {
      if (auto binding = value.getDefiningOp<py::BindingRefOp>()) {
        mlir::func::FuncOp target =
            module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
        if (target && !target.isDeclaration() &&
            target->hasAttr("callable_type")) {
          RuntimeArgumentEvidence evidence;
          evidence.functionTarget = binding.getBinding().str();
          llvm::SmallVector<mlir::Type, 4> closureTypes =
              callableClosureTypes(target);
          evidence.closureValueTypes.append(closureTypes.begin(),
                                            closureTypes.end());
          alternatives.push_back(std::move(evidence));
          coveredKinds |= CallableEvidence;
        }
      }
    }

    if ((requiredKinds & CoroutineEvidence) != 0) {
      if (auto call = value.getDefiningOp<py::CallOp>()) {
        if (std::optional<RuntimeArgumentEvidence> evidence =
                coroutineEvidenceFromCall(call)) {
          alternatives.push_back(std::move(*evidence));
          coveredKinds |= CoroutineEvidence;
        }
      }
    }

    if ((coveredKinds & requiredKinds) == requiredKinds)
      return alternatives;

    if ((requiredKinds & CallableEvidence) == 0)
      return std::nullopt;

    auto call = value.getDefiningOp<py::CallOp>();
    if (!call || call.getNumResults() != 1)
      return std::nullopt;
    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return std::nullopt;
    auto returned = returnedCallableSummaries.find(binding.getBinding());
    if (returned == returnedCallableSummaries.end())
      return std::nullopt;

    mlir::func::FuncOp producer =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!producer || producer.isDeclaration())
      return std::nullopt;
    py::CallableType callable = callableTypeOf(producer);
    if (!callable)
      return std::nullopt;
    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callable,
                                                          /*emitErrors=*/false);
    if (!invocation || !plan)
      return std::nullopt;

    alternatives.clear();
    alternatives.reserve(returned->second.alternatives.size());
    for (const ReturnedCallableAlternativeSummary &returnedAlternative :
         returned->second.alternatives) {
      RuntimeArgumentEvidence evidence;
      evidence.functionTarget = returnedAlternative.target;
      for (unsigned logicalIndex : returnedAlternative.captureArgumentIndices) {
        mlir::Type type = logicalSourceType(producer, callable, *invocation,
                                            *plan, logicalIndex);
        if (!type)
          return std::nullopt;
        evidence.closureValueTypes.push_back(runtimeEvidenceType(type));
      }
      alternatives.push_back(std::move(evidence));
    }
    return alternatives;
  };

  llvm::StringMap<llvm::SmallVector<char, 8>> requirements;
  module.walk([&](mlir::func::FuncOp function) {
    py::CallableType callable = callableTypeOf(function);
    if (!callable)
      return;
    llvm::SmallVector<mlir::Type, 8> logicalTypes =
        RuntimeBundleLowerer::callableLogicalInputTypes(function, callable);
    if (logicalTypes.empty())
      return;

    mlir::Block &entry = function.getBody().front();
    llvm::SmallVector<char, 8> required(logicalTypes.size(), 0);
    function.walk([&](py::CallOp call) {
      if (call->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      mlir::Value callableValue = stripReturnedObjectView(call.getCallable());
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(callableValue);
      if (!argument || argument.getOwner() != &entry)
        return;
      unsigned index = argument.getArgNumber();
      if (index < required.size())
        required[index] |= CallableEvidence;
    });
    function.walk([&](py::AwaitOp awaitOp) {
      if (awaitOp->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      mlir::Value awaitable = stripReturnedObjectView(awaitOp.getAwaitable());
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(awaitable);
      if (!argument || argument.getOwner() != &entry)
        return;
      unsigned index = argument.getArgNumber();
      if (index < required.size())
        required[index] |= CoroutineEvidence;
    });
    if (llvm::any_of(required, [](char value) { return value != 0; }))
      requirements[function.getSymName()] = std::move(required);
  });

  auto ensureRequirement =
      [&](mlir::func::FuncOp function) -> llvm::SmallVector<char, 8> * {
    py::CallableType callable = callableTypeOf(function);
    if (!callable)
      return nullptr;
    llvm::SmallVector<mlir::Type, 8> logicalTypes =
        RuntimeBundleLowerer::callableLogicalInputTypes(function, callable);
    if (logicalTypes.empty())
      return nullptr;
    llvm::SmallVector<char, 8> &required = requirements[function.getSymName()];
    if (required.empty())
      required.resize(logicalTypes.size(), 0);
    return &required;
  };

  bool requirementChanged = true;
  while (requirementChanged) {
    requirementChanged = false;
    module.walk([&](py::CallOp call) {
      mlir::Value callee = stripReturnedObjectView(call.getCallable());
      auto binding = callee.getDefiningOp<py::BindingRefOp>();
      if (!binding)
        return mlir::WalkResult::advance();

      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
      py::CallableType callable = callableTypeOf(target);
      if (!callable)
        return mlir::WalkResult::advance();

      auto required = requirements.find(target.getSymName());
      if (required == requirements.end())
        return mlir::WalkResult::advance();

      mlir::func::FuncOp caller = call->getParentOfType<mlir::func::FuncOp>();
      llvm::SmallVector<char, 8> *callerRequired = ensureRequirement(caller);
      if (!callerRequired)
        return mlir::WalkResult::advance();
      mlir::Block &callerEntry = caller.getBody().front();

      std::optional<StaticCallableInvocation> invocation =
          RuntimeBundleLowerer::collectStaticCallableInvocation(call);
      std::optional<CallableArgumentPlan> plan =
          RuntimeBundleLowerer::collectCallableArgumentPlan(
              call, callable, /*emitErrors=*/false);
      if (!invocation || !plan)
        return mlir::WalkResult::advance();

      for (auto [logicalIndex, actualIndex] :
           llvm::enumerate(plan->fixedActuals)) {
        if (logicalIndex >= required->second.size() ||
            !required->second[logicalIndex] || !actualIndex ||
            *actualIndex >= invocation->actualValues.size())
          continue;
        mlir::Value actual =
            stripReturnedObjectView(invocation->actualValues[*actualIndex]);
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(actual);
        if (!argument || argument.getOwner() != &callerEntry)
          continue;
        unsigned callerIndex = argument.getArgNumber();
        if (callerIndex >= callerRequired->size())
          continue;
        unsigned before =
            static_cast<unsigned>((*callerRequired)[callerIndex]);
        unsigned propagated =
            before | static_cast<unsigned>(required->second[logicalIndex]);
        if (propagated == before)
          continue;
        (*callerRequired)[callerIndex] = static_cast<char>(propagated);
        requirementChanged = true;
      }
      return mlir::WalkResult::advance();
    });
  }

  llvm::StringMap<llvm::SmallVector<Slot, 8>> accumulators;
  for (auto &entry : requirements)
    accumulators[entry.getKey()].resize(entry.getValue().size());

  auto mergeEvidenceFromValue = [&](Slot &slot, mlir::Value value,
                                    mlir::func::FuncOp caller,
                                    unsigned requiredKinds) -> bool {
    value = stripReturnedObjectView(value);
    if (std::optional<llvm::SmallVector<RuntimeArgumentEvidence, 4>>
            alternatives = evidenceAlternativesFromValue(value,
                                                        requiredKinds)) {
      bool changed = false;
      for (RuntimeArgumentEvidence &evidence : *alternatives)
        changed |= mergeEvidence(slot, std::move(evidence), requiredKinds);
      return changed;
    }

    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || !caller || caller.isDeclaration() ||
        argument.getOwner() != &caller.getBody().front()) {
      unsigned before = slot.invalidKinds;
      slot.invalidKinds |= requiredKinds;
      bool changed = slot.invalidKinds != before;
      return changed;
    }

    auto source = accumulators.find(caller.getSymName());
    if (source == accumulators.end() ||
        argument.getArgNumber() >= source->second.size())
      return false;

    Slot &sourceSlot = source->second[argument.getArgNumber()];
    if ((sourceSlot.invalidKinds & requiredKinds) != 0) {
      unsigned before = slot.invalidKinds;
      slot.invalidKinds |= requiredKinds;
      bool changed = slot.invalidKinds != before;
      return changed;
    }

    bool changed = false;
    for (const RuntimeArgumentEvidence &alternative : sourceSlot.alternatives) {
      RuntimeArgumentEvidence copy = alternative;
      changed |= mergeEvidence(slot, std::move(copy), requiredKinds);
    }
    return changed;
  };

  bool evidenceChanged = true;
  while (evidenceChanged) {
    evidenceChanged = false;
    module.walk([&](py::CallOp call) {
      mlir::Value callee = stripReturnedObjectView(call.getCallable());
      auto binding = callee.getDefiningOp<py::BindingRefOp>();
      if (!binding)
        return mlir::WalkResult::advance();

      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
      py::CallableType callable = callableTypeOf(target);
      if (!callable)
        return mlir::WalkResult::advance();
      auto required = requirements.find(target.getSymName());
      if (required == requirements.end())
        return mlir::WalkResult::advance();

      std::optional<StaticCallableInvocation> invocation =
          RuntimeBundleLowerer::collectStaticCallableInvocation(call);
      if (!invocation)
        return mlir::WalkResult::advance();
      std::optional<CallableArgumentPlan> plan =
          RuntimeBundleLowerer::collectCallableArgumentPlan(
              call, callable, /*emitErrors=*/false);
      if (!plan)
        return mlir::WalkResult::advance();

      mlir::func::FuncOp caller = call->getParentOfType<mlir::func::FuncOp>();
      llvm::SmallVector<Slot, 8> &acc = accumulators[target.getSymName()];

      for (auto [index, actualIndex] : llvm::enumerate(plan->fixedActuals)) {
        if (index >= acc.size() || index >= required->second.size() ||
            !required->second[index])
          continue;
        if (!actualIndex) {
          unsigned requiredKinds =
              static_cast<unsigned>(required->second[index]);
          unsigned before = acc[index].invalidKinds;
          acc[index].invalidKinds |= requiredKinds;
          if (acc[index].invalidKinds != before)
            evidenceChanged = true;
          continue;
        }
        if (*actualIndex >= invocation->actualValues.size())
          continue;
        evidenceChanged |= mergeEvidenceFromValue(
            acc[index], invocation->actualValues[*actualIndex], caller,
            static_cast<unsigned>(required->second[index]));
      }
      return mlir::WalkResult::advance();
    });
  }

  for (auto &entry : accumulators) {
    llvm::SmallVector<Slot, 8> &acc = entry.getValue();
    CallableArgumentEvidenceABI abi;
    abi.logicalArguments.resize(acc.size());
    for (auto [index, slot] : llvm::enumerate(acc)) {
      if (slot.invalidKinds != 0 || slot.alternatives.empty())
        continue;
      abi.logicalArguments[index].alternatives = std::move(slot.alternatives);
    }
    if (!abi.empty())
      callableArgumentEvidenceABIs[entry.getKey()] = std::move(abi);
  }

  for (auto &entry : callableProtocolSpecializations) {
    auto original = callableArgumentEvidenceABIs.find(entry.getKey());
    if (original == callableArgumentEvidenceABIs.end())
      continue;
    for (const CallableProtocolSpecialization &specialization :
         entry.getValue()) {
      if (callableArgumentEvidenceABIs.count(specialization.cloneName) != 0)
        continue;
      callableArgumentEvidenceABIs[specialization.cloneName] = original->second;
    }
  }
  return mlir::success();
}
} // namespace py::lowering
