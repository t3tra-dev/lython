#include "RuntimeLowering/CallableEvidence.h"

namespace py::runtime_lowering {

using namespace callable_evidence;

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
} // namespace py::runtime_lowering
