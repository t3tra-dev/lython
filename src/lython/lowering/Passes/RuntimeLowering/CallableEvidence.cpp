#include "RuntimeLowering/RuntimeLowering.h"

#include "cpp/PyCallableShape.h"

namespace py::runtime_lowering {

namespace {

void sortKeywordEvidence(llvm::SmallVectorImpl<std::string> &keys,
                         llvm::SmallVectorImpl<mlir::Type> &types) {
  llvm::SmallVector<unsigned, 8> order;
  order.reserve(keys.size());
  for (unsigned index = 0, end = static_cast<unsigned>(keys.size());
       index < end; ++index)
    order.push_back(index);

  llvm::sort(order,
             [&](unsigned lhs, unsigned rhs) { return keys[lhs] < keys[rhs]; });

  llvm::SmallVector<std::string, 8> sortedKeys;
  llvm::SmallVector<mlir::Type, 8> sortedTypes;
  sortedKeys.reserve(keys.size());
  sortedTypes.reserve(types.size());
  for (unsigned index : order) {
    sortedKeys.push_back(keys[index]);
    sortedTypes.push_back(types[index]);
  }
  keys.assign(sortedKeys.begin(), sortedKeys.end());
  types.assign(sortedTypes.begin(), sortedTypes.end());
}

unsigned fixedCallableInputCount(py::CallableType callable) {
  return static_cast<unsigned>(callable.getPositionalTypes().size() +
                               callable.getKwOnlyTypes().size());
}

bool mergePrefixCompatibleTypes(llvm::SmallVectorImpl<mlir::Type> &into,
                                llvm::ArrayRef<mlir::Type> candidate) {
  unsigned common = std::min<unsigned>(into.size(), candidate.size());
  for (unsigned index = 0; index < common; ++index)
    if (into[index] != candidate[index])
      return false;
  if (candidate.size() > into.size())
    into.append(candidate.begin() + into.size(), candidate.end());
  return true;
}

bool mergeSortedKeywordEvidence(llvm::SmallVectorImpl<std::string> &keys,
                                llvm::SmallVectorImpl<mlir::Type> &types,
                                llvm::ArrayRef<std::string> candidateKeys,
                                llvm::ArrayRef<mlir::Type> candidateTypes) {
  if (candidateKeys.size() != candidateTypes.size() ||
      keys.size() != types.size())
    return false;

  llvm::StringMap<mlir::Type> merged;
  for (auto [key, type] : llvm::zip(keys, types))
    merged[key] = type;
  for (auto [key, type] : llvm::zip(candidateKeys, candidateTypes)) {
    auto inserted = merged.try_emplace(key, type);
    if (!inserted.second && inserted.first->second != type)
      return false;
  }

  llvm::SmallVector<std::string, 8> mergedKeys;
  mergedKeys.reserve(merged.size());
  for (const auto &entry : merged)
    mergedKeys.push_back(entry.getKey().str());
  llvm::sort(mergedKeys);

  llvm::SmallVector<mlir::Type, 8> mergedTypes;
  mergedTypes.reserve(mergedKeys.size());
  for (llvm::StringRef key : mergedKeys)
    mergedTypes.push_back(merged.lookup(key));
  keys.assign(mergedKeys.begin(), mergedKeys.end());
  types.assign(mergedTypes.begin(), mergedTypes.end());
  return true;
}

std::optional<std::int64_t> integerLiteralFromValue(mlir::Value value) {
  auto constant = value.getDefiningOp<py::IntConstantOp>();
  if (!constant)
    return std::nullopt;
  std::int64_t parsed = 0;
  if (constant.getValue().getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

bool packOperandIsUnpacked(py::PackOp pack, unsigned index) {
  auto flags = pack->getAttrOfType<mlir::ArrayAttr>(kPackUnpackedOperandsAttr);
  if (!flags || index >= flags.size())
    return false;
  auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(flags[index]);
  return boolAttr && boolAttr.getValue();
}

mlir::Type runtimeEvidenceType(mlir::Type type) {
  if (auto literal = mlir::dyn_cast_if_present<py::LiteralType>(type)) {
    std::string contract = runtimeContractName(literal);
    if (!contract.empty())
      return runtimeContractType(type.getContext(), contract);
  }
  if (type && mlir::isa<py::ObjectType>(type))
    return runtimeContractType(type.getContext(), "builtins.object");
  return type;
}

bool containsStaticEvidenceParameter(mlir::Type type) {
  if (!type)
    return false;
  if (mlir::isa<py::TypeVarType, py::ParamSpecType, py::TypeVarTupleType>(type))
    return true;
  if (auto unpack = mlir::dyn_cast_if_present<py::UnpackType>(type))
    return containsStaticEvidenceParameter(unpack.getPackedType());
  if (auto callable = mlir::dyn_cast_if_present<py::CallableType>(type))
    return llvm::any_of(callable.getPositionalTypes(),
                        containsStaticEvidenceParameter) ||
           llvm::any_of(callable.getKwOnlyTypes(),
                        containsStaticEvidenceParameter) ||
           llvm::any_of(callable.getResultTypes(),
                        containsStaticEvidenceParameter) ||
           (callable.hasVararg() &&
            containsStaticEvidenceParameter(callable.getVarargType())) ||
           (callable.hasKwarg() &&
            containsStaticEvidenceParameter(callable.getKwargType()));
  if (auto contract = mlir::dyn_cast_if_present<py::ContractType>(type))
    return llvm::any_of(contract.getArguments(),
                        containsStaticEvidenceParameter);
  if (auto protocol = mlir::dyn_cast_if_present<py::ProtocolType>(type))
    return llvm::any_of(protocol.getArguments(),
                        containsStaticEvidenceParameter);
  if (auto unionType = mlir::dyn_cast_if_present<py::UnionType>(type))
    return llvm::any_of(unionType.getMemberTypes(),
                        containsStaticEvidenceParameter);
  if (auto typeType = mlir::dyn_cast_if_present<py::TypeType>(type))
    return containsStaticEvidenceParameter(typeType.getInstanceType());
  if (auto tuple = mlir::dyn_cast_if_present<py::TupleType>(type))
    return llvm::any_of(tuple.getElementTypes(),
                        containsStaticEvidenceParameter);
  if (auto list = mlir::dyn_cast_if_present<py::ListType>(type))
    return containsStaticEvidenceParameter(list.getElementType());
  if (auto dict = mlir::dyn_cast_if_present<py::DictType>(type))
    return containsStaticEvidenceParameter(dict.getKeyType()) ||
           containsStaticEvidenceParameter(dict.getValueType());
  return false;
}

std::optional<llvm::SmallVector<mlir::Type, 8>>
actualVarargEvidenceTypes(llvm::ArrayRef<mlir::Type> actualTypes,
                          llvm::ArrayRef<std::int64_t> indices) {
  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(indices.size());
  for (std::int64_t rawIndex : indices) {
    std::int64_t normalized = rawIndex;
    if (normalized < 0)
      normalized += static_cast<std::int64_t>(actualTypes.size());
    if (normalized < 0 ||
        normalized >= static_cast<std::int64_t>(actualTypes.size()))
      return std::nullopt;
    types.push_back(
        runtimeEvidenceType(actualTypes[static_cast<unsigned>(normalized)]));
  }
  return types;
}

std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedVarargEvidenceTypes(py::CallableType callable,
                            llvm::ArrayRef<std::int64_t> indices,
                            unsigned actualCount) {
  if (!callable.hasVararg())
    return std::nullopt;

  lython::callable::VarargShape<mlir::Type> shape =
      py::callableVarargShape(callable.getVarargType());
  if (!shape.valid())
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(indices.size());
  if (shape.kind == lython::callable::VarargShape<mlir::Type>::Kind::Repeated) {
    for (std::int64_t rawIndex : indices) {
      std::int64_t normalized = rawIndex;
      if (normalized < 0)
        normalized += actualCount;
      if (normalized < 0 ||
          normalized >= static_cast<std::int64_t>(actualCount))
        return std::nullopt;
      types.push_back(runtimeEvidenceType(*shape.repeated));
    }
    return types;
  }

  for (std::int64_t rawIndex : indices) {
    std::int64_t normalized = rawIndex;
    if (normalized < 0)
      normalized += actualCount;
    if (normalized < 0 ||
        normalized >= static_cast<std::int64_t>(shape.exact.size()))
      return std::nullopt;
    types.push_back(
        runtimeEvidenceType(shape.exact[static_cast<unsigned>(normalized)]));
  }
  return types;
}

std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedDenseVarargEvidenceTypes(py::CallableType callable, unsigned count) {
  if (!callable.hasVararg())
    return std::nullopt;

  lython::callable::VarargShape<mlir::Type> shape =
      py::callableVarargShape(callable.getVarargType());
  if (!shape.valid())
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(count);
  if (shape.kind == lython::callable::VarargShape<mlir::Type>::Kind::Repeated) {
    for (unsigned index = 0; index < count; ++index)
      types.push_back(runtimeEvidenceType(*shape.repeated));
    return types;
  }

  if (count > shape.exact.size())
    return std::nullopt;
  for (mlir::Type type :
       llvm::ArrayRef<mlir::Type>(shape.exact).take_front(count))
    types.push_back(runtimeEvidenceType(type));
  return types;
}

std::optional<llvm::SmallVector<mlir::Type, 8>>
expectedKwargEvidenceTypes(py::CallableType callable, unsigned count) {
  if (!callable.hasKwarg())
    return std::nullopt;

  std::optional<mlir::Type> valueType =
      py::callableKwargValueType(callable.getKwargType());
  if (!valueType)
    return std::nullopt;

  llvm::SmallVector<mlir::Type, 8> types;
  types.reserve(count);
  for (unsigned index = 0; index < count; ++index)
    types.push_back(*valueType);
  return types;
}

} // namespace

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

mlir::LogicalResult RuntimeBundleLowerer::buildCallableAggregateEvidenceABIs() {
  struct Requirements {
    llvm::SmallVector<std::int64_t, 8> varargIndices;
    bool needsFullVararg = false;
    bool needsFullKwarg = false;
    llvm::SmallVector<std::string, 8> kwargKeys;

    bool needsVarargEvidence() const {
      return needsFullVararg || !varargIndices.empty();
    }

    bool needsKwargEvidence() const {
      return !kwargKeys.empty() || needsFullKwarg;
    }
  };

  struct Accumulator {
    py::CallableType callable;
    bool sawVararg = false;
    bool invalidVararg = false;
    llvm::SmallVector<mlir::Type, 8> varargElementTypes;
    bool sawKwarg = false;
    bool invalidKwarg = false;
    llvm::SmallVector<std::string, 8> kwargKeys;
    llvm::SmallVector<mlir::Type, 8> kwargValueTypes;
  };

  llvm::StringMap<Requirements> requirements;
  module.walk([&](mlir::func::FuncOp function) {
    auto callableAttr =
        function->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || function.isDeclaration() ||
        (!callable.hasVararg() && !callable.hasKwarg()))
      return;

    mlir::Block &entry = function.getBody().front();
    unsigned fixedCount = fixedCallableInputCount(callable);
    std::optional<unsigned> varargIndex;
    std::optional<unsigned> kwargIndex;
    if (callable.hasVararg())
      varargIndex = fixedCount;
    if (callable.hasKwarg())
      kwargIndex = fixedCount + (callable.hasVararg() ? 1 : 0);

    Requirements req;
    function.walk([&](py::GetItemOp getItem) {
      if (getItem->getParentOfType<mlir::func::FuncOp>() != function)
        return;

      mlir::Value container = stripReturnedObjectView(getItem.getContainer());
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(container);
      if (!argument || argument.getOwner() != &entry)
        return;

      unsigned argumentIndex = argument.getArgNumber();
      if (varargIndex && argumentIndex == *varargIndex) {
        std::optional<std::int64_t> index =
            integerLiteralFromValue(getItem.getIndex());
        if (!index) {
          req.needsFullVararg = true;
          return;
        }
        if (llvm::find(req.varargIndices, *index) == req.varargIndices.end())
          req.varargIndices.push_back(*index);
        return;
      }

      if (kwargIndex && argumentIndex == *kwargIndex) {
        std::optional<std::string> key =
            RuntimeBundleLowerer::keywordNameFromValue(getItem.getIndex());
        if (!key) {
          req.needsFullKwarg = true;
          return;
        }
        if (llvm::find(req.kwargKeys, *key) == req.kwargKeys.end())
          req.kwargKeys.push_back(std::move(*key));
      }
    });

    function.walk([&](py::CallOp call) {
      if (!varargIndex ||
          call->getParentOfType<mlir::func::FuncOp>() != function)
        return;
      auto pack = call.getPosargs().getDefiningOp<py::PackOp>();
      if (!pack)
        return;
      for (auto [index, value] : llvm::enumerate(pack.getValues())) {
        if (!packOperandIsUnpacked(pack, static_cast<unsigned>(index)))
          continue;
        value = stripReturnedObjectView(value);
        auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
        if (argument && argument.getOwner() == &entry &&
            argument.getArgNumber() == *varargIndex)
          req.needsFullVararg = true;
      }
    });

    if (!req.varargIndices.empty())
      llvm::sort(req.varargIndices);
    if (!req.kwargKeys.empty())
      llvm::sort(req.kwargKeys);
    if (req.needsVarargEvidence() || req.needsKwargEvidence())
      requirements[function.getSymName()] = std::move(req);
  });

  llvm::StringMap<Accumulator> accumulators;
  module.walk([&](py::CallOp call) {
    mlir::Value callee = stripReturnedObjectView(call.getCallable());
    auto binding = callee.getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return mlir::WalkResult::advance();

    mlir::func::FuncOp target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration())
      return mlir::WalkResult::advance();
    auto required = requirements.find(target.getSymName());
    if (required == requirements.end())
      return mlir::WalkResult::advance();

    auto callableAttr = target->getAttrOfType<mlir::TypeAttr>("callable_type");
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || (!callable.hasVararg() && !callable.hasKwarg()))
      return mlir::WalkResult::advance();

    Accumulator &acc = accumulators[target.getSymName()];
    acc.callable = callable;

    std::optional<CallableAggregateEvidenceCall> evidence =
        RuntimeBundleLowerer::collectCallableAggregateEvidence(call, callable);
    if (!evidence) {
      if (callable.hasVararg() && required->second.needsVarargEvidence())
        acc.invalidVararg = true;
      if (callable.hasKwarg() && required->second.needsKwargEvidence())
        acc.invalidKwarg = true;
      return mlir::WalkResult::advance();
    }

    if (callable.hasVararg() && required->second.needsVarargEvidence()) {
      llvm::SmallVector<mlir::Type, 8> candidate;
      llvm::SmallVector<std::int64_t, 8> requiredIndices;
      if (required->second.needsFullVararg) {
        unsigned denseCount =
            std::max<unsigned>(1, evidence->varargElementTypes.size());
        requiredIndices.reserve(denseCount);
        for (unsigned index = 0; index < denseCount; ++index)
          requiredIndices.push_back(index);
      } else {
        requiredIndices = required->second.varargIndices;
      }
      std::optional<llvm::SmallVector<mlir::Type, 8>> expectedTypes;
      if (containsStaticEvidenceParameter(callable.getVarargType())) {
        expectedTypes = actualVarargEvidenceTypes(evidence->varargElementTypes,
                                                  requiredIndices);
      } else if (required->second.needsFullVararg) {
        expectedTypes = expectedDenseVarargEvidenceTypes(
            callable, static_cast<unsigned>(requiredIndices.size()));
      } else {
        expectedTypes = expectedVarargEvidenceTypes(
            callable, requiredIndices,
            static_cast<unsigned>(evidence->varargElementTypes.size()));
      }
      if (!expectedTypes)
        acc.invalidVararg = true;
      else
        candidate = std::move(*expectedTypes);
      if (acc.invalidVararg)
        return mlir::WalkResult::advance();
      if (required->second.needsFullVararg) {
        if (!mergePrefixCompatibleTypes(acc.varargElementTypes, candidate))
          acc.invalidVararg = true;
        acc.sawVararg = true;
      } else if (!acc.sawVararg) {
        acc.varargElementTypes = std::move(candidate);
        acc.sawVararg = true;
      } else if (!sameTypeSequence(acc.varargElementTypes, candidate)) {
        acc.invalidVararg = true;
      }
    }

    if (callable.hasKwarg() && required->second.needsKwargEvidence()) {
      llvm::SmallVector<std::string, 8> candidateKeys;
      llvm::SmallVector<mlir::Type, 8> candidateTypes;
      if (required->second.needsFullKwarg) {
        candidateKeys = evidence->kwargKeys;
        std::optional<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
            expectedKwargEvidenceTypes(
                callable, static_cast<unsigned>(candidateKeys.size()));
        if (!expectedTypes) {
          acc.invalidKwarg = true;
          return mlir::WalkResult::advance();
        }
        candidateTypes = std::move(*expectedTypes);
        sortKeywordEvidence(candidateKeys, candidateTypes);
      } else {
        candidateKeys = required->second.kwargKeys;
        for (llvm::StringRef key : candidateKeys) {
          auto stored = llvm::find(evidence->kwargKeys, key);
          if (stored == evidence->kwargKeys.end()) {
            acc.invalidKwarg = true;
            break;
          }
          unsigned index =
              static_cast<unsigned>(stored - evidence->kwargKeys.begin());
          if (index >= evidence->kwargValueTypes.size()) {
            acc.invalidKwarg = true;
            break;
          }
        }
        std::optional<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
            expectedKwargEvidenceTypes(
                callable, static_cast<unsigned>(candidateKeys.size()));
        if (!expectedTypes)
          acc.invalidKwarg = true;
        else
          candidateTypes = std::move(*expectedTypes);
      }
      if (acc.invalidKwarg)
        return mlir::WalkResult::advance();
      if (required->second.needsFullKwarg) {
        if (!acc.sawKwarg) {
          acc.kwargKeys = std::move(candidateKeys);
          acc.kwargValueTypes = std::move(candidateTypes);
          acc.sawKwarg = true;
        } else if (!mergeSortedKeywordEvidence(acc.kwargKeys,
                                               acc.kwargValueTypes,
                                               candidateKeys, candidateTypes)) {
          acc.invalidKwarg = true;
        }
      } else if (!acc.sawKwarg) {
        acc.kwargKeys = std::move(candidateKeys);
        acc.kwargValueTypes = std::move(candidateTypes);
        acc.sawKwarg = true;
      } else if (acc.kwargKeys != candidateKeys ||
                 !sameTypeSequence(acc.kwargValueTypes, candidateTypes)) {
        acc.invalidKwarg = true;
      }
    }

    return mlir::WalkResult::advance();
  });

  for (auto &entry : accumulators) {
    Accumulator &acc = entry.getValue();
    CallableAggregateEvidenceABI abi;
    unsigned fixedCount = fixedCallableInputCount(acc.callable);
    if (acc.callable.hasVararg() && acc.sawVararg && !acc.invalidVararg) {
      abi.varargLogicalIndex = fixedCount;
      auto required = requirements.find(entry.getKey());
      if (required != requirements.end() && !required->second.needsFullVararg)
        abi.varargElementIndices = required->second.varargIndices;
      abi.varargElementTypes = acc.varargElementTypes;
    }
    if (acc.callable.hasKwarg() && acc.sawKwarg && !acc.invalidKwarg) {
      abi.kwargLogicalIndex = fixedCount + (acc.callable.hasVararg() ? 1 : 0);
      auto required = requirements.find(entry.getKey());
      abi.kwargIsFull =
          required != requirements.end() && required->second.needsFullKwarg;
      abi.kwargKeys = acc.kwargKeys;
      abi.kwargValueTypes = acc.kwargValueTypes;
    }
    if (abi.varargLogicalIndex || abi.kwargLogicalIndex)
      callableAggregateEvidenceABIs[entry.getKey()] = std::move(abi);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::appendCallableArgumentEvidenceSources(
    py::CallOp op, llvm::StringRef targetName,
    const CallableArgumentEvidenceABI &evidence,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> &evidenceSources) {
  std::size_t logicalSourceCount = sources.size();
  std::size_t hiddenCount = 0;
  for (const RuntimeArgumentEvidenceSet &evidenceSet :
       evidence.logicalArguments) {
    for (const RuntimeArgumentEvidence &argumentEvidence :
         evidenceSet.alternatives)
      hiddenCount += argumentEvidence.closureValueTypes.size();
  }
  evidenceSources.reserve(evidenceSources.size() + hiddenCount);
  sources.reserve(sources.size() + hiddenCount);

  auto appendEvidenceValue =
      [&](const RuntimeValue &value) -> mlir::LogicalResult {
    evidenceSources.push_back(
        RuntimeBundle::object(value.contract, value.values));
    sources.push_back(&evidenceSources.back());
    return mlir::success();
  };
  auto appendPlaceholder = [&](mlir::Type expected) -> mlir::LogicalResult {
    mlir::FailureOr<RuntimeValue> dead =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, expected, "callable argument evidence placeholder");
    if (mlir::failed(dead))
      return mlir::failure();
    return appendEvidenceValue(*dead);
  };
  auto appendClosureEvidence =
      [&](llvm::ArrayRef<RuntimeValue> values,
          llvm::ArrayRef<mlir::Type> expectedTypes) -> mlir::LogicalResult {
    if (values.size() != expectedTypes.size())
      return op.emitError()
             << "argument evidence closure count mismatch for " << targetName;
    for (auto [closureIndex, expected] : llvm::enumerate(expectedTypes)) {
      const RuntimeValue &value = values[closureIndex];
      if (!py::isAssignableTo(value.contract, expected, op.getOperation()))
        return op.emitError() << "argument evidence closure " << closureIndex
                              << " for " << targetName << " has contract "
                              << value.contract << ", expected " << expected;
      if (mlir::failed(appendEvidenceValue(value)))
        return mlir::failure();
    }
    return mlir::success();
  };
  auto findAlternative =
      [](const RuntimeBundle &source,
         llvm::StringRef target) -> const RuntimeCallableAlternative * {
    for (const RuntimeCallableAlternative &alternative :
         source.callableAlternatives)
      if (alternative.functionTarget == target)
        return &alternative;
    return nullptr;
  };

  for (auto [logicalIndex, evidenceSet] :
       llvm::enumerate(evidence.logicalArguments)) {
    if (evidenceSet.empty())
      continue;
    if (logicalIndex >= logicalSourceCount)
      return op.emitError() << "argument evidence ABI for " << targetName
                            << " references logical argument " << logicalIndex
                            << ", but call has only " << logicalSourceCount
                            << " logical sources";
    const RuntimeBundle *source = sources[logicalIndex];
    if (!source || source->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "argument evidence ABI source for " << targetName
                            << " must be an object bundle";

    for (const RuntimeArgumentEvidence &argumentEvidence :
         evidenceSet.alternatives) {
      if (argumentEvidence.empty())
        continue;

      if (argumentEvidence.functionTarget.empty()) {
        if (mlir::failed(appendClosureEvidence(
                source->closureValues, argumentEvidence.closureValueTypes)))
          return mlir::failure();
        continue;
      }

      if (source->functionTarget == argumentEvidence.functionTarget) {
        if (mlir::failed(appendClosureEvidence(
                source->closureValues, argumentEvidence.closureValueTypes)))
          return mlir::failure();
        continue;
      }

      if (const RuntimeCallableAlternative *alternative =
              findAlternative(*source, argumentEvidence.functionTarget)) {
        if (mlir::failed(
                appendClosureEvidence(alternative->closureValues,
                                      argumentEvidence.closureValueTypes)))
          return mlir::failure();
        continue;
      }

      if (source->functionTarget.empty() &&
          source->callableAlternatives.empty())
        return op.emitError() << "argument evidence source for " << targetName
                              << " has no static callable target alternative '"
                              << argumentEvidence.functionTarget << "'";

      for (mlir::Type expected : argumentEvidence.closureValueTypes)
        if (mlir::failed(appendPlaceholder(expected)))
          return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::appendCallableAggregateEvidenceSources(
    py::CallOp op, llvm::StringRef targetName,
    const CallableAggregateEvidenceABI &evidence,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> &evidenceSources) {
  auto appendObjectEvidenceSource =
      [&](const RuntimeValue &value) -> mlir::LogicalResult {
    evidenceSources.push_back(
        RuntimeBundle::object(value.contract, value.values));
    sources.push_back(&evidenceSources.back());
    return mlir::success();
  };
  auto appendPresenceEvidenceSource = [&](bool present) -> mlir::LogicalResult {
    mlir::Value bit =
        builder
            .create<mlir::arith::ConstantIntOp>(op.getLoc(), present ? 1 : 0, 1)
            .getResult();
    evidenceSources.push_back(RuntimeBundle::object(
        runtimeContractType(context, "builtins.bool"), bit));
    sources.push_back(&evidenceSources.back());
    return mlir::success();
  };

  if (evidence.varargLogicalIndex) {
    unsigned sourceIndex = *evidence.varargLogicalIndex;
    if (sourceIndex >= sources.size())
      return op.emitError()
             << "vararg evidence ABI for " << targetName
             << " references logical argument " << sourceIndex
             << ", but call has only " << sources.size() << " logical sources";
    const RuntimeBundle *vararg = sources[sourceIndex];
    if (!vararg || vararg->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "vararg evidence ABI source for " << targetName
                            << " must be an object bundle";
    if (!vararg->sequenceIndices.empty())
      return op.emitError()
             << "vararg evidence source for " << targetName
             << " is partial and cannot be re-indexed for another callable "
                "ABI yet";
    bool fullDenseVarargEvidence = evidence.varargElementIndices.empty() &&
                                   !evidence.varargElementTypes.empty();
    if (!fullDenseVarargEvidence && evidence.varargElementIndices.size() !=
                                        evidence.varargElementTypes.size())
      return op.emitError() << "vararg evidence ABI index/type count mismatch "
                               "for "
                            << targetName;
    auto findPlaceholder = [&](mlir::Type expected) -> const RuntimeValue * {
      std::string expectedContract = runtimeContractName(expected);
      for (const RuntimeValue &candidate : vararg->sequenceElements)
        if (runtimeContractName(candidate.contract) == expectedContract)
          return &candidate;
      return nullptr;
    };
    for (unsigned index = 0, end = static_cast<unsigned>(
                                 evidence.varargElementTypes.size());
         index < end; ++index) {
      mlir::Type expected = evidence.varargElementTypes[index];
      std::int64_t rawIndex = fullDenseVarargEvidence
                                  ? static_cast<std::int64_t>(index)
                                  : evidence.varargElementIndices[index];
      std::int64_t normalized = rawIndex;
      std::int64_t size =
          static_cast<std::int64_t>(vararg->sequenceElements.size());
      if (normalized < 0)
        normalized += size;
      const RuntimeValue *value = nullptr;
      if (normalized >= 0 && normalized < size) {
        value = &vararg->sequenceElements[static_cast<unsigned>(normalized)];
      } else if (fullDenseVarargEvidence) {
        // The callee checks the real tuple length before selecting a dense
        // evidence slot. Missing slots only need a physical placeholder so the
        // static ABI can remain uniform across direct call sites.
        value = findPlaceholder(expected);
        if (!value) {
          mlir::FailureOr<RuntimeValue> dead =
              RuntimeBundleLowerer::materializeDeadObjectValue(
                  op, expected, "vararg evidence placeholder");
          if (mlir::failed(dead))
            return mlir::failure();
          if (mlir::failed(appendObjectEvidenceSource(*dead)))
            return mlir::failure();
          continue;
        }
      } else {
        return op.emitError()
               << "vararg evidence ABI for " << targetName << " needs index "
               << rawIndex << ", but call source has "
               << vararg->sequenceElements.size() << " elements";
      }
      if (runtimeContractName(value->contract) != runtimeContractName(expected))
        return op.emitError() << "vararg evidence element " << index << " for "
                              << targetName << " has contract "
                              << value->contract << ", expected " << expected;
      if (mlir::failed(appendObjectEvidenceSource(*value)))
        return mlir::failure();
    }
  }

  if (evidence.kwargLogicalIndex) {
    unsigned sourceIndex = *evidence.kwargLogicalIndex;
    if (sourceIndex >= sources.size())
      return op.emitError()
             << "kwarg evidence ABI for " << targetName
             << " references logical argument " << sourceIndex
             << ", but call has only " << sources.size() << " logical sources";
    const RuntimeBundle *kwarg = sources[sourceIndex];
    if (!kwarg || kwarg->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "kwarg evidence ABI source for " << targetName
                            << " must be an object bundle";
    for (auto [index, key] : llvm::enumerate(evidence.kwargKeys)) {
      auto storedKey = llvm::find(kwarg->mappingKeys, key);
      mlir::Type expected = evidence.kwargValueTypes[index];
      bool present = storedKey != kwarg->mappingKeys.end();
      if (present) {
        unsigned sourceIndex =
            static_cast<unsigned>(storedKey - kwarg->mappingKeys.begin());
        if (sourceIndex >= kwarg->mappingValues.size())
          return op.emitError()
                 << "kwarg evidence key/value count mismatch for "
                 << targetName;
        const RuntimeValue &value = kwarg->mappingValues[sourceIndex];
        if (runtimeContractName(value.contract) !=
            runtimeContractName(expected))
          return op.emitError() << "kwarg evidence value " << index << " for "
                                << targetName << " has contract "
                                << value.contract << ", expected " << expected;
        if (mlir::failed(appendObjectEvidenceSource(value)))
          return mlir::failure();
      } else if (evidence.kwargIsFull) {
        mlir::FailureOr<RuntimeValue> dead =
            RuntimeBundleLowerer::materializeDeadObjectValue(
                op, expected, "kwarg evidence placeholder");
        if (mlir::failed(dead))
          return mlir::failure();
        if (mlir::failed(appendObjectEvidenceSource(*dead)))
          return mlir::failure();
      } else {
        return op.emitError() << "kwarg evidence source for " << targetName
                              << " has no key '" << key << "'";
      }
      if (evidence.kwargIsFull &&
          mlir::failed(appendPresenceEvidenceSource(present)))
        return mlir::failure();
    }
  }
  return mlir::success();
}

} // namespace py::runtime_lowering
