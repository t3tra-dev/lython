#include "RuntimeLowering/RuntimeLowering.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::runtime_lowering {

namespace {

std::optional<std::int64_t> integerLiteralFromValue(mlir::Value value) {
  auto constant = value.getDefiningOp<py::IntConstantOp>();
  if (!constant)
    return std::nullopt;
  std::int64_t parsed = 0;
  if (constant.getValue().getAsInteger(10, parsed))
    return std::nullopt;
  return parsed;
}

bool sameValueTypes(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.getType() != right.getType())
      return false;
  return true;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::bindEvidenceObjectResult(
    mlir::Operation *op, mlir::Value resultValue, llvm::StringRef label,
    const RuntimeValue &value) {
  if (!py::isAssignableTo(value.contract, resultValue.getType(), op))
    return op->emitError() << label << " evidence contract " << value.contract
                           << " is not assignable to result "
                           << resultValue.getType();

  mlir::Type bundleContract = value.contract;
  std::string resultContract = runtimeContractName(resultValue.getType());
  if (!resultContract.empty() &&
      objectShapeMatches(resultContract, value.values))
    bundleContract = resultValue.getType();

  valueBundles[resultValue] =
      RuntimeBundle::object(bundleContract, value.values);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindSelectedEvidenceObjectResult(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle bundle) {
  valueBundles[resultValue] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::selectEvidenceObjectByMatch(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, mlir::ValueRange matches,
    llvm::StringRef label, llvm::StringRef missingContract,
    llvm::StringRef missingMessage) {
  context->loadDialect<mlir::scf::SCFDialect>();
  if (candidates.empty() || candidates.size() != matches.size()) {
    op->emitError() << label << " evidence match/value count mismatch";
    return mlir::failure();
  }

  const RuntimeValue &first = candidates.front();
  if (first.values.empty()) {
    op->emitError() << label << " evidence candidate has no physical values";
    return mlir::failure();
  }

  llvm::ArrayRef<mlir::Value> firstValues = first.values;
  for (auto [position, candidate] : llvm::enumerate(candidates)) {
    if (!py::isAssignableTo(candidate.contract, resultValue.getType(), op)) {
      op->emitError() << label << " evidence candidate " << position
                      << " contract " << candidate.contract
                      << " is not assignable to result "
                      << resultValue.getType();
      return mlir::failure();
    }
    if (!sameValueTypes(firstValues, candidate.values)) {
      op->emitError() << label << " evidence candidate " << position
                      << " has a different physical ABI shape";
      return mlir::failure();
    }
    if (!matches[position].getType().isInteger(1)) {
      op->emitError() << label << " evidence match " << position
                      << " must be i1";
      return mlir::failure();
    }
  }

  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Value value : firstValues)
    resultTypes.push_back(value.getType());

  auto emitChain = [&](auto &&self, unsigned position)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    auto ifOp = builder.create<mlir::scf::IfOp>(
        loc, resultTypes, matches[position], /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    builder.create<mlir::scf::YieldOp>(loc, candidates[position].values);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (position + 1 < candidates.size()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> nested =
          self(self, position + 1);
      if (mlir::failed(nested))
        return mlir::failure();
      builder.create<mlir::scf::YieldOp>(loc, *nested);
    } else {
      if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
              op, missingContract, missingMessage)))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 4> deadValues;
      deadValues.reserve(resultTypes.size());
      for (mlir::Type resultType : resultTypes) {
        mlir::FailureOr<mlir::Value> dead =
            RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
        if (mlir::failed(dead))
          return mlir::failure();
        deadValues.push_back(*dead);
      }
      builder.create<mlir::scf::YieldOp>(loc, deadValues);
    }

    builder.setInsertionPointAfter(ifOp);
    return llvm::SmallVector<mlir::Value, 4>(ifOp.getResults().begin(),
                                             ifOp.getResults().end());
  };

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> selected =
      emitChain(emitChain, 0);
  if (mlir::failed(selected))
    return mlir::failure();

  mlir::Type bundleContract = first.contract;
  std::string resultContract = runtimeContractName(resultValue.getType());
  if (!resultContract.empty() && objectShapeMatches(resultContract, *selected))
    bundleContract = resultValue.getType();
  return RuntimeBundle::object(bundleContract, *selected);
}

mlir::FailureOr<RuntimeBundle> RuntimeBundleLowerer::selectEvidenceObjectMiss(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, llvm::StringRef label,
    llvm::StringRef missingContract, llvm::StringRef missingMessage) {
  builder.setInsertionPoint(op);
  mlir::Value neverMatches =
      builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 0, 1)
          .getResult();
  llvm::SmallVector<mlir::Value, 8> matches(candidates.size(), neverMatches);
  return RuntimeBundleLowerer::selectEvidenceObjectByMatch(
      op, resultValue, candidates, matches, label, missingContract,
      missingMessage);
}

mlir::FailureOr<bool> RuntimeBundleLowerer::lowerSequenceEvidenceGetItem(
    py::GetItemOp op, const RuntimeBundle &container,
    const RuntimeBundle &index) {
  if (container.sequenceElements.empty())
    return false;

  std::optional<std::int64_t> rawIndex = integerLiteralFromValue(op.getIndex());
  if (rawIndex) {
    if (!container.sequenceIndices.empty()) {
      if (container.sequenceIndices.size() !=
          container.sequenceElements.size()) {
        op.emitError() << "sequence evidence index/value count mismatch";
        return mlir::failure();
      }
      for (auto [position, storedIndex] :
           llvm::enumerate(container.sequenceIndices)) {
        if (storedIndex != *rawIndex)
          continue;
        if (position < container.sequenceElementBundles.size() &&
            container.sequenceElementBundles[position]) {
          RuntimeBundle selected = *container.sequenceElementBundles[position];
          if (mlir::failed(bindSelectedEvidenceObjectResult(
                  op, op.getResult(), std::move(selected))))
            return mlir::failure();
          return true;
        }
        const RuntimeValue &element = container.sequenceElements[position];
        if (mlir::failed(bindEvidenceObjectResult(
                op, op.getResult(), "sequence __getitem__", element)))
          return mlir::failure();
        erase.push_back(op);
        return true;
      }
      mlir::FailureOr<RuntimeBundle> selected =
          RuntimeBundleLowerer::selectEvidenceObjectMiss(
              op, op.getResult(), container.sequenceElements,
              "sequence __getitem__", "builtins.IndexError",
              "sequence index out of range");
      if (mlir::failed(selected))
        return mlir::failure();
      if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                        std::move(*selected))))
        return mlir::failure();
      return true;
    }

    std::int64_t normalized = *rawIndex;
    std::int64_t size =
        static_cast<std::int64_t>(container.sequenceElements.size());
    if (normalized < 0)
      normalized += size;
    if (normalized < 0 || normalized >= size) {
      mlir::FailureOr<RuntimeBundle> selected =
          RuntimeBundleLowerer::selectEvidenceObjectMiss(
              op, op.getResult(), container.sequenceElements,
              "sequence __getitem__", "builtins.IndexError",
              "sequence index out of range");
      if (mlir::failed(selected))
        return mlir::failure();
      if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                        std::move(*selected))))
        return mlir::failure();
      return true;
    }

    unsigned elementIndex = static_cast<unsigned>(normalized);
    if (elementIndex < container.sequenceElementBundles.size() &&
        container.sequenceElementBundles[elementIndex]) {
      RuntimeBundle selected = *container.sequenceElementBundles[elementIndex];
      if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                        std::move(selected))))
        return mlir::failure();
      return true;
    }

    const RuntimeValue &element = container.sequenceElements[elementIndex];
    if (mlir::failed(bindEvidenceObjectResult(op, op.getResult(),
                                              "sequence __getitem__", element)))
      return mlir::failure();
    erase.push_back(op);
    return true;
  }

  if (!container.sequenceIndices.empty())
    return false;

  std::optional<RuntimeSymbol> unbox =
      manifest.primitive(index.contractName(), "unbox.i64");
  if (!unbox)
    return false;

  builder.setInsertionPoint(op);
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod) {
    op.emitError() << "sequence evidence dynamic index needs a runtime __len__";
    return mlir::failure();
  }
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp lenCall = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *lenMethod, lenOperands);
  if (lenCall.getNumResults() != 1 ||
      !lenCall.getResult(0).getType().isInteger(64)) {
    lenMethod->function.emitError()
        << "sequence __len__ evidence method must return one i64";
    return mlir::failure();
  }

  mlir::func::CallOp indexCall = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *unbox, index.physicalValues());
  if (indexCall.getNumResults() != 1 ||
      !indexCall.getResult(0).getType().isInteger(64)) {
    unbox->function.emitError()
        << "index unbox.i64 primitive must return one i64";
    return mlir::failure();
  }

  mlir::Location loc = op.getLoc();
  mlir::Value rawRuntimeIndex = indexCall.getResult(0);
  mlir::Value zero = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 64);
  mlir::Value runtimeSize = lenCall.getResult(0);
  mlir::Value isNegative = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, rawRuntimeIndex, zero);
  mlir::Value adjusted =
      builder.create<mlir::arith::AddIOp>(loc, rawRuntimeIndex, runtimeSize);
  mlir::Value normalized = builder.create<mlir::arith::SelectOp>(
      loc, isNegative, adjusted, rawRuntimeIndex);
  mlir::Value lowerOk = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::sge, normalized, zero);
  mlir::Value upperOk = builder.create<mlir::arith::CmpIOp>(
      loc, mlir::arith::CmpIPredicate::slt, normalized, runtimeSize);
  mlir::Value inRange =
      builder.create<mlir::arith::AndIOp>(loc, lowerOk, upperOk);

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(container.sequenceElements.size());
  for (unsigned position = 0,
                end = static_cast<unsigned>(container.sequenceElements.size());
       position < end; ++position) {
    mlir::Value expected = builder.create<mlir::arith::ConstantIntOp>(
        loc, static_cast<std::int64_t>(position), 64);
    mlir::Value indexMatches = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, normalized, expected);
    matches.push_back(
        builder.create<mlir::arith::AndIOp>(loc, inRange, indexMatches));
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.sequenceElements, matches,
          "sequence __getitem__", "builtins.IndexError",
          "sequence index out of range");
  if (mlir::failed(selected))
    return mlir::failure();
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(*selected))))
    return mlir::failure();
  return true;
}

mlir::FailureOr<bool>
RuntimeBundleLowerer::lowerDictEvidenceGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index) {
  if (container.contractName() != "builtins.dict" ||
      container.mappingKeys.empty())
    return false;

  if (container.mappingKeys.size() != container.mappingValues.size()) {
    op.emitError() << "dict evidence key/value count mismatch";
    return mlir::failure();
  }
  bool hasPresence = !container.mappingPresent.empty();
  if (hasPresence &&
      container.mappingPresent.size() != container.mappingKeys.size()) {
    op.emitError() << "dict evidence key/presence count mismatch";
    return mlir::failure();
  }

  std::optional<std::string> key =
      RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
  if (key) {
    for (auto [position, storedKey] : llvm::enumerate(container.mappingKeys)) {
      if (storedKey != *key)
        continue;
      const RuntimeValue &value = container.mappingValues[position];
      if (hasPresence) {
        builder.setInsertionPoint(op);
        mlir::FailureOr<RuntimeBundle> selected =
            RuntimeBundleLowerer::selectEvidenceObjectByMatch(
                op, op.getResult(), llvm::ArrayRef<RuntimeValue>(&value, 1),
                mlir::ValueRange{container.mappingPresent[position]},
                "dict __getitem__", "builtins.KeyError", "key not found");
        if (mlir::failed(selected))
          return mlir::failure();
        if (mlir::failed(bindSelectedEvidenceObjectResult(
                op, op.getResult(), std::move(*selected))))
          return mlir::failure();
        return true;
      }
      if (mlir::failed(bindEvidenceObjectResult(op, op.getResult(),
                                                "dict __getitem__", value)))
        return mlir::failure();
      erase.push_back(op);
      return true;
    }

    mlir::FailureOr<RuntimeBundle> selected =
        RuntimeBundleLowerer::selectEvidenceObjectMiss(
            op, op.getResult(), container.mappingValues, "dict __getitem__",
            "builtins.KeyError", "key not found");
    if (mlir::failed(selected))
      return mlir::failure();
    if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                      std::move(*selected))))
      return mlir::failure();
    return true;
  }

  if (index.contractName() != "builtins.str")
    return false;

  builder.setInsertionPoint(op);
  std::optional<RuntimeSymbol> eq =
      manifest.method(index.contractName(), "__eq__");
  if (!eq) {
    op.emitError() << "dict evidence dynamic key needs str.__eq__";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(container.mappingKeys.size());
  llvm::SmallVector<RuntimeBundle, 8> materializedKeys;
  materializedKeys.reserve(container.mappingKeys.size());
  for (auto [position, storedKey] : llvm::enumerate(container.mappingKeys)) {
    RuntimeBundle &keyObject = materializedKeys.emplace_back();
    if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
            op, storedKey, keyObject)))
      return mlir::failure();
    llvm::SmallVector<const RuntimeBundle *, 2> eqSources{&index, &keyObject};
    llvm::SmallVector<mlir::Value, 4> eqOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *eq, eqSources, eqOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp eqCall =
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *eq, eqOperands);
    if (eqCall.getNumResults() != 1 ||
        !eqCall.getResult(0).getType().isInteger(1)) {
      eq->function.emitError()
          << "str.__eq__ evidence method must return one i1";
      return mlir::failure();
    }
    mlir::Value match = eqCall.getResult(0);
    if (hasPresence)
      match = builder.create<mlir::arith::AndIOp>(
          op.getLoc(), match, container.mappingPresent[position]);
    matches.push_back(match);
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.mappingValues, matches,
          "dict __getitem__", "builtins.KeyError", "key not found");
  if (mlir::failed(selected))
    return mlir::failure();
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(*selected))))
    return mlir::failure();
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerGetItem(py::GetItemOp op) {
  const RuntimeBundle *container =
      RuntimeBundleLowerer::bundleFor(op.getContainer());
  const RuntimeBundle *index = RuntimeBundleLowerer::bundleFor(op.getIndex());
  if (!container || !index)
    return op.emitError() << "getitem operands need runtime bundles";

  if (container->ctypes &&
      (container->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell ||
       container->ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer)) {
    mlir::LogicalResult ctypesHandled =
        RuntimeBundleLowerer::lowerStaticCtypesGetItem(op, *container, *index);
    if (mlir::succeeded(ctypesHandled))
      return mlir::success();
  }

  mlir::FailureOr<bool> sequenceHandled =
      RuntimeBundleLowerer::lowerSequenceEvidenceGetItem(op, *container,
                                                         *index);
  if (mlir::failed(sequenceHandled))
    return mlir::failure();
  if (*sequenceHandled)
    return mlir::success();

  mlir::FailureOr<bool> dictHandled =
      RuntimeBundleLowerer::lowerDictEvidenceGetItem(op, *container, *index);
  if (mlir::failed(dictHandled))
    return mlir::failure();
  if (*dictHandled)
    return mlir::success();

  llvm::SmallVector<const RuntimeBundle *, 2> sources{container, index};
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__getitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *container, *methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
