#include "Runtime/Evidence/Callable.h"

namespace py::runtime_lowering {

using namespace callable_evidence;

mlir::LogicalResult RuntimeBundleLowerer::buildCallableArgumentEvidenceABIs() {
  struct Slot {
    llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;
    bool invalid = false;
  };

  auto mergeEvidence = [&](Slot &slot,
                           RuntimeArgumentEvidence evidence) -> bool {
    if (evidence.empty()) {
      bool changed = !slot.invalid;
      slot.invalid = true;
      return changed;
    }
    auto sameEvidence = [&](const RuntimeArgumentEvidence &candidate) {
      return candidate.functionTarget == evidence.functionTarget &&
             sameTypeSequence(candidate.closureValueTypes,
                              evidence.closureValueTypes);
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

  auto evidenceAlternativesFromValue = [&](mlir::Value value)
      -> std::optional<llvm::SmallVector<RuntimeArgumentEvidence, 4>> {
    value = stripReturnedObjectView(value);
    if (auto binding = value.getDefiningOp<py::BindingRefOp>()) {
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
      if (!target || target.isDeclaration() ||
          !target->hasAttr("callable_type"))
        return std::nullopt;
      RuntimeArgumentEvidence evidence;
      evidence.functionTarget = binding.getBinding().str();
      llvm::SmallVector<mlir::Type, 4> closureTypes =
          callableClosureTypes(target);
      evidence.closureValueTypes.append(closureTypes.begin(),
                                        closureTypes.end());
      llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;
      alternatives.push_back(std::move(evidence));
      return alternatives;
    }

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
    auto callableAttr =
        producer->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable)
      return std::nullopt;
    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callable,
                                                          /*emitErrors=*/false);
    if (!invocation || !plan)
      return std::nullopt;

    llvm::SmallVector<RuntimeArgumentEvidence, 4> alternatives;
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
    auto callableAttr =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || function.isDeclaration())
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
        required[index] = 1;
    });
    if (llvm::any_of(required, [](char value) { return value != 0; }))
      requirements[function.getSymName()] = std::move(required);
  });

  auto callableTypeFor = [](mlir::func::FuncOp function) -> py::CallableType {
    if (!function || function.isDeclaration())
      return {};
    auto callableAttr =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    return mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
  };

  auto ensureRequirement =
      [&](mlir::func::FuncOp function) -> llvm::SmallVector<char, 8> * {
    py::CallableType callable = callableTypeFor(function);
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
      py::CallableType callable = callableTypeFor(target);
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
        if (callerIndex >= callerRequired->size() ||
            (*callerRequired)[callerIndex])
          continue;
        (*callerRequired)[callerIndex] = 1;
        requirementChanged = true;
      }
      return mlir::WalkResult::advance();
    });
  }

  llvm::StringMap<llvm::SmallVector<Slot, 8>> accumulators;
  for (auto &entry : requirements)
    accumulators[entry.getKey()].resize(entry.getValue().size());

  auto mergeEvidenceFromValue = [&](Slot &slot, mlir::Value value,
                                    mlir::func::FuncOp caller) -> bool {
    value = stripReturnedObjectView(value);
    if (std::optional<llvm::SmallVector<RuntimeArgumentEvidence, 4>>
            alternatives = evidenceAlternativesFromValue(value)) {
      bool changed = false;
      for (RuntimeArgumentEvidence &evidence : *alternatives)
        changed |= mergeEvidence(slot, std::move(evidence));
      return changed;
    }

    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument || !caller || caller.isDeclaration() ||
        argument.getOwner() != &caller.getBody().front()) {
      bool changed = !slot.invalid;
      slot.invalid = true;
      return changed;
    }

    auto source = accumulators.find(caller.getSymName());
    if (source == accumulators.end() ||
        argument.getArgNumber() >= source->second.size())
      return false;

    Slot &sourceSlot = source->second[argument.getArgNumber()];
    if (sourceSlot.invalid) {
      bool changed = !slot.invalid;
      slot.invalid = true;
      return changed;
    }

    bool changed = false;
    for (const RuntimeArgumentEvidence &alternative : sourceSlot.alternatives) {
      RuntimeArgumentEvidence copy = alternative;
      changed |= mergeEvidence(slot, std::move(copy));
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
      py::CallableType callable = callableTypeFor(target);
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
          if (!acc[index].invalid) {
            acc[index].invalid = true;
            evidenceChanged = true;
          }
          continue;
        }
        if (*actualIndex >= invocation->actualValues.size())
          continue;
        evidenceChanged |= mergeEvidenceFromValue(
            acc[index], invocation->actualValues[*actualIndex], caller);
      }
      return mlir::WalkResult::advance();
    });
  }

  for (auto &entry : accumulators) {
    llvm::SmallVector<Slot, 8> &acc = entry.getValue();
    CallableArgumentEvidenceABI abi;
    abi.logicalArguments.resize(acc.size());
    for (auto [index, slot] : llvm::enumerate(acc)) {
      if (slot.invalid || slot.alternatives.empty())
        continue;
      abi.logicalArguments[index].alternatives = std::move(slot.alternatives);
    }
    if (!abi.empty())
      callableArgumentEvidenceABIs[entry.getKey()] = std::move(abi);
  }
  return mlir::success();
}
} // namespace py::runtime_lowering
