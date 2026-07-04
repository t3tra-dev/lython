#include "Runtime/Core/Lowerer.h"

namespace py::runtime_lowering {

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
