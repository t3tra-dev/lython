#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {

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

  valueBundles[resultValue] = RuntimeBundle::objectWithOwnership(
      bundleContract, value.values,
      ownership::logicalOwnershipKind(bundleContract,
                                      /*ownsObject=*/false));
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindSelectedEvidenceObjectResult(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle bundle) {
  bundle.setObjectLogicalOwnership(/*ownsObject=*/false);
  valueBundles[resultValue] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

// An evidence-selected container element is a borrow whose provenance (the
// container) may be released before the element's last use — the container's
// liveness ends at its last direct IR use, which evidence selection does not
// create. Retaining the element through its contract's `own` primitive turns
// the borrow into a checked owned token that survives the container and is
// released by the ordinary owned-result machinery. The retain is inserted
// immediately after the element's defining ops, where the element is provably
// alive (its container has not been released yet). Contracts without an `own`
// primitive keep the borrowed binding (their uses stay tied to structurally
// live containers, e.g. instance fields).
std::optional<RuntimeValue>
RuntimeBundleLowerer::retainEvidenceElement(mlir::Operation *op,
                                            const RuntimeValue &value,
                                            bool atOperation) {
  std::string contract = runtimeContractName(value.contract);
  if (contract.empty() || value.values.empty())
    return std::nullopt;
  if (!ownership::isObjectHeaderLikeType(value.values.front().getType()))
    return std::nullopt;
  mlir::func::FuncOp retain = RuntimeBundleLowerer::findRetainFunction();
  if (!retain || retain.getFunctionType().getNumInputs() != 1)
    return std::nullopt;
  mlir::Operation *latest = nullptr;
  if (!atOperation) {
    // An inline-constructed local (its entity root is a raw alloc) is not yet
    // initialized at its defining op — the refcount word store lands later in
    // the construction sequence — so a retain placed after the def would read
    // garbage. Such evidence elements need the at-operation retain (with a
    // container pin, the caller's responsibility) or stay borrowed.
    // Slot-reconstructed and selection-merged elements (loads/casts/merges at
    // the access site) are long-initialized and retain safely after their
    // defs, independent of any container pin.
    if (mlir::isa_and_nonnull<mlir::memref::AllocOp>(
            value.values.front().getDefiningOp()))
      return std::nullopt;
    for (mlir::Value physical : value.values) {
      mlir::Operation *definition = physical.getDefiningOp();
      if (!definition)
        return std::nullopt;
      if (!latest) {
        latest = definition;
      } else if (definition->getBlock() != latest->getBlock()) {
        return std::nullopt;
      } else if (latest->isBeforeInBlock(definition)) {
        latest = definition;
      }
    }
    if (!latest)
      return std::nullopt;
  }
  // Borrow → own: one retain on the entity root, rooted for the ownership
  // machinery by the owned-local-object aggregation marker (an identity cast
  // erased at reconciliation). The contract → retain relation is static; no
  // per-contract runtime wrapper is involved.
  mlir::OpBuilder::InsertionGuard guard(builder);
  if (atOperation)
    builder.setInsertionPoint(op);
  else
    builder.setInsertionPointAfter(latest);
  mlir::Location loc = atOperation ? op->getLoc() : latest->getLoc();
  mlir::Value header = value.values.front();
  mlir::Type retainInput = retain.getFunctionType().getInput(0);
  if (header.getType() != retainInput) {
    if (mlir::memref::CastOp::areCastCompatible(header.getType(),
                                                retainInput)) {
      header = mlir::memref::CastOp::create(builder, loc, retainInput, header)
                   .getResult();
    } else {
      // Box-fronted entities (source-class instances: the whole 16-word box is
      // the entity root) hold the refcount+class prefix inside a wider static
      // shape than the retain input; retain through a rank-1 prefix view.
      auto sourceType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
      auto targetType = mlir::dyn_cast<mlir::MemRefType>(retainInput);
      if (!sourceType || !targetType || sourceType.getRank() != 1 ||
          targetType.getRank() != 1 || !sourceType.hasStaticShape() ||
          !targetType.hasStaticShape() ||
          sourceType.getElementType() != targetType.getElementType() ||
          sourceType.getDimSize(0) < targetType.getDimSize(0))
        return std::nullopt;
      llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
      llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
          builder.getIndexAttr(targetType.getDimSize(0))};
      llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
      llvm::SmallVector<int64_t, 1> resultShape{targetType.getDimSize(0)};
      auto inferredType = mlir::cast<mlir::MemRefType>(
          mlir::memref::SubViewOp::inferRankReducedResultType(
              resultShape, sourceType, offsets, sizes, strides));
      header = mlir::memref::SubViewOp::create(builder, loc, inferredType,
                                               header, offsets, sizes, strides)
                   .getResult();
      if (header.getType() != retainInput) {
        if (!mlir::memref::CastOp::areCastCompatible(header.getType(),
                                                     retainInput))
          return std::nullopt;
        header =
            mlir::memref::CastOp::create(builder, loc, retainInput, header)
                .getResult();
      }
    }
  }
  mlir::func::CallOp::create(builder, loc, retain, header);
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Value physical : value.values)
    resultTypes.push_back(physical.getType());
  auto rooted = mlir::UnrealizedConversionCastOp::create(
      builder, loc, resultTypes, value.values);
  rooted->setAttr(ownership::kOwnedLocalObjectAttr, builder.getUnitAttr());
  rooted->setAttr(ownership::kOwnedLocalObjectContractAttr,
                  builder.getStringAttr(contract));
  RuntimeValue retained = value;
  retained.values.assign(rooted.getResults().begin(),
                         rooted.getResults().end());
  return retained;
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::slotStorageShapesFor(mlir::Operation *op,
                                           mlir::Type contract,
                                           llvm::StringRef purpose) {
  std::string name = runtimeContractName(contract);
  if (std::optional<RuntimeSymbol> box = manifest.primitive(name, "box")) {
    mlir::FunctionType type = box->function.getFunctionType();
    return llvm::SmallVector<mlir::Type, 8>(type.getResults().begin(),
                                            type.getResults().end());
  }
  return RuntimeBundleLowerer::runtimeValueTypesFor(op, contract, purpose);
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::unboxSlotElementValues(mlir::Operation *op,
                                             mlir::Type contract,
                                             llvm::ArrayRef<mlir::Value> values) {
  std::string name = runtimeContractName(contract);
  std::optional<RuntimeSymbol> unbox = manifest.primitive(name, "unbox");
  if (!unbox)
    return llvm::SmallVector<mlir::Value, 4>(values.begin(), values.end());
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *unbox, values);
  return llvm::SmallVector<mlir::Value, 4>(call.getResults().begin(),
                                           call.getResults().end());
}

// The at-operation retain of an element does not itself use the container, so
// the container's release could still be placed before it; an explicit
// `__len__` use right after the retain pins the container past it (the same
// device the dynamic-index selection uses).
mlir::LogicalResult
RuntimeBundleLowerer::pinContainerLiveness(mlir::Operation *op,
                                           const RuntimeBundle &container) {
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod)
    return op->emitError()
           << "evidence element retain needs a runtime __len__ to pin "
           << container.contractName();
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *lenMethod,
                                          lenOperands);
  return mlir::success();
}

mlir::FailureOr<std::optional<RuntimeValue>>
RuntimeBundleLowerer::retainEvidenceElementWithFallback(
    mlir::Operation *op, const RuntimeValue &value,
    const RuntimeBundle *container) {
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, value);
  if (retained || !container)
    return retained;
  if (!manifest.method(container->contractName(), "__len__"))
    return retained;
  retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, value,
                                                  /*atOperation=*/true);
  if (!retained)
    return retained;
  if (mlir::failed(pinContainerLiveness(op, *container)))
    return mlir::failure();
  return retained;
}

mlir::LogicalResult RuntimeBundleLowerer::bindRetainedEvidenceValue(
    mlir::Operation *op, mlir::Value resultValue, llvm::StringRef label,
    const RuntimeValue &value, const RuntimeBundle *container) {
  mlir::FailureOr<std::optional<RuntimeValue>> retained =
      retainEvidenceElementWithFallback(op, value, container);
  if (mlir::failed(retained))
    return mlir::failure();
  if (mlir::failed(bindEvidenceObjectResult(op, resultValue, label,
                                            *retained ? **retained : value)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindRetainedEvidenceBundle(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle bundle,
    const RuntimeBundle *container) {
  RuntimeValue element{bundle.objectValue.contract,
                       llvm::SmallVector<mlir::Value, 4>(
                           bundle.physicalValues().begin(),
                           bundle.physicalValues().end()),
                       bundle.objectValue.ownership};
  mlir::FailureOr<std::optional<RuntimeValue>> retained =
      retainEvidenceElementWithFallback(op, element, container);
  if (mlir::failed(retained))
    return mlir::failure();
  if (!*retained)
    return bindSelectedEvidenceObjectResult(op, resultValue, std::move(bundle));
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      bundle.objectValue.contract, (*retained)->values,
      ownership::logicalOwnershipKind(bundle.objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(bundle);
  return bindSelectedEvidenceObjectResult(op, resultValue, std::move(rebuilt));
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::selectEvidenceObjectByMatch(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, mlir::ValueRange matches,
    llvm::StringRef label, llvm::StringRef missingContract,
    llvm::StringRef missingMessage, bool raiseOnMiss) {
  context->loadDialect<mlir::scf::SCFDialect>();
  if (candidates.empty() || candidates.size() != matches.size()) {
    op->emitError() << label << " evidence match/value count mismatch";
    return mlir::failure();
  }

  const RuntimeValue &first = candidates.front();
  if (first.values.empty()) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expected =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, first.contract,
                                                   "evidence candidate ABI");
    if (mlir::failed(expected))
      return mlir::failure();
    if (!expected->empty()) {
      op->emitError() << label << " evidence candidate has no physical values";
      return mlir::failure();
    }
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

  // A zero-result scf.if (e.g. a None candidate with no physical values) gets
  // its empty terminators auto-inserted at build time; creating manual yields
  // there would leave two terminators per block.
  bool needsYields = !resultTypes.empty();
  auto emitChain = [&](auto &&self, unsigned position)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, resultTypes, matches[position], /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (needsYields)
      mlir::scf::YieldOp::create(builder, loc, candidates[position].values);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (position + 1 < candidates.size()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> nested =
          self(self, position + 1);
      if (mlir::failed(nested))
        return mlir::failure();
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, *nested);
    } else if (raiseOnMiss) {
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
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, deadValues);
    } else {
      // A raise-free miss (e.g. iterator exhaustion) executes on every normal
      // completion; use immortal static placeholders instead of heap
      // allocations so the miss path does not leak.
      mlir::FailureOr<RuntimeValue> dead =
          RuntimeBundleLowerer::materializeDeadObjectValueImpl(
              op, first.contract, label, DeadObjectStorage::StaticNonOwning);
      if (mlir::failed(dead))
        return mlir::failure();
      if (dead->values.size() != resultTypes.size())
        return op->emitError()
               << label << " static miss placeholder ABI mismatch";
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, dead->values);
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
  return RuntimeBundle::objectWithOwnership(
      bundleContract, *selected,
      ownership::logicalOwnershipKind(bundleContract,
                                      /*ownsObject=*/false));
}

mlir::FailureOr<RuntimeBundle> RuntimeBundleLowerer::selectEvidenceObjectMiss(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, llvm::StringRef label,
    llvm::StringRef missingContract, llvm::StringRef missingMessage) {
  (void)candidates;
  builder.setInsertionPoint(op);
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, missingContract, missingMessage)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, resultValue.getType(),
                                                 label);
  if (mlir::failed(resultTypes))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> deadValues;
  deadValues.reserve(resultTypes->size());
  for (mlir::Type resultType : *resultTypes) {
    mlir::FailureOr<mlir::Value> dead =
        RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
    if (mlir::failed(dead))
      return mlir::failure();
    deadValues.push_back(*dead);
  }
  return RuntimeBundle::objectWithOwnership(
      resultValue.getType(), deadValues,
      ownership::logicalOwnershipKind(resultValue.getType(),
                                      /*ownsObject=*/false));
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
          if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                      std::move(selected),
                                                      &container)))
            return mlir::failure();
          return true;
        }
        const RuntimeValue &element = container.sequenceElements[position];
        if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                                   "sequence __getitem__",
                                                   element, &container)))
          return mlir::failure();
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
      if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                  std::move(selected),
                                                  &container)))
        return mlir::failure();
      return true;
    }

    const RuntimeValue &element = container.sequenceElements[elementIndex];
    if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                               "sequence __getitem__", element,
                                               &container)))
      return mlir::failure();
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
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value runtimeSize = lenCall.getResult(0);
  mlir::Value isNegative = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, rawRuntimeIndex, zero);
  mlir::Value adjusted =
      mlir::arith::AddIOp::create(builder, loc, rawRuntimeIndex, runtimeSize);
  mlir::Value normalized = mlir::arith::SelectOp::create(
      builder, loc, isNegative, adjusted, rawRuntimeIndex);
  mlir::Value lowerOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, normalized, zero);
  mlir::Value upperOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, normalized, runtimeSize);
  mlir::Value inRange =
      mlir::arith::AndIOp::create(builder, loc, lowerOk, upperOk);

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(container.sequenceElements.size());
  for (unsigned position = 0,
                end = static_cast<unsigned>(container.sequenceElements.size());
       position < end; ++position) {
    mlir::Value expected = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(position), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, normalized, expected);
    matches.push_back(
        mlir::arith::AndIOp::create(builder, loc, inRange, indexMatches));
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.sequenceElements, matches,
          "sequence __getitem__", "builtins.IndexError",
          "sequence index out of range");
  if (mlir::failed(selected))
    return mlir::failure();

  // The selected element is a borrow of a container slot. Retain it right
  // after the selection (per selection, so a getitem inside a loop stays
  // balanced), then pin the container's liveness past the retain with an
  // explicit `__len__` use — otherwise the container's release is placed after
  // its previous last use (the bounds check above) and would free the elements
  // before the selection reads them.
  RuntimeValue chainElement{(*selected).objectValue.contract,
                            llvm::SmallVector<mlir::Value, 4>(
                                (*selected).physicalValues().begin(),
                                (*selected).physicalValues().end()),
                            (*selected).objectValue.ownership};
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, chainElement);
  if (!retained) {
    op.emitError() << "sequence __getitem__ cannot retain evidence element "
                   << chainElement.contract << " selected by a dynamic index";
    return mlir::failure();
  }
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *lenMethod, lenOperands);
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      (*selected).objectValue.contract, retained->values,
      ownership::logicalOwnershipKind((*selected).objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(*selected);
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(rebuilt))))
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
      if (!hasPresence && position < container.mappingValueBundles.size() &&
          container.mappingValueBundles[position]) {
        RuntimeBundle selected = *container.mappingValueBundles[position];
        if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                    std::move(selected),
                                                    &container)))
          return mlir::failure();
        return true;
      }
      const RuntimeValue &value = container.mappingValues[position];
      if (hasPresence) {
        builder.setInsertionPoint(op);
        // Retain the candidate at its definition so the selected value
        // survives the container's release; fall back to the borrowed view for
        // contracts without an `own` primitive (previous behavior). The
        // presence test is a guard that raises on a missing key — the value is
        // bound directly (not routed through scf.if results) so its ownership
        // stays visible to the affine verifier.
        RuntimeValue candidate = value;
        if (std::optional<RuntimeValue> retained =
                RuntimeBundleLowerer::retainEvidenceElement(op, value))
          candidate = std::move(*retained);
        context->loadDialect<mlir::scf::SCFDialect>();
        mlir::Location loc = op.getLoc();
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
        mlir::Value missing = mlir::arith::XOrIOp::create(
            builder, loc, container.mappingPresent[position], one);
        auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                             missing, /*withElseRegion=*/false);
        builder.setInsertionPointToStart(&guard.getThenRegion().front());
        if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
                op, "builtins.KeyError", "key not found")))
          return mlir::failure();
        mlir::Block &thenBlock = guard.getThenRegion().front();
        if (thenBlock.empty() ||
            !thenBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
          mlir::scf::YieldOp::create(builder, loc);
        builder.setInsertionPointAfter(guard);
        if (mlir::failed(bindEvidenceObjectResult(
                op, op.getResult(), "dict __getitem__", candidate)))
          return mlir::failure();
        erase.push_back(op);
        return true;
      }
      if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                                 "dict __getitem__", value,
                                                 &container)))
        return mlir::failure();
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
      match = mlir::arith::AndIOp::create(builder, op.getLoc(), match,
                                          container.mappingPresent[position]);
    matches.push_back(match);
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.mappingValues, matches,
          "dict __getitem__", "builtins.KeyError", "key not found");
  if (mlir::failed(selected))
    return mlir::failure();

  // Retain the selected value per selection and pin the container's liveness
  // past the retain (see the sequence dynamic-index path for the rationale).
  RuntimeValue chainValue{(*selected).objectValue.contract,
                          llvm::SmallVector<mlir::Value, 4>(
                              (*selected).physicalValues().begin(),
                              (*selected).physicalValues().end()),
                          (*selected).objectValue.ownership};
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, chainValue);
  if (!retained) {
    op.emitError() << "dict __getitem__ cannot retain evidence value "
                   << chainValue.contract << " selected by a dynamic key";
    return mlir::failure();
  }
  std::optional<RuntimeSymbol> dictLen =
      manifest.method(container.contractName(), "__len__");
  if (!dictLen) {
    op.emitError() << "dict evidence dynamic key needs a runtime __len__ to "
                      "pin the container";
    return mlir::failure();
  }
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *dictLen, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *dictLen, lenOperands);
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      (*selected).objectValue.contract, retained->values,
      ownership::logicalOwnershipKind((*selected).objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(*selected);
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(rebuilt))))
    return mlir::failure();
  return true;
}

// Shared helpers for runtime-mode container reads: rebuild the result's
// physical values from a payload box slot and retain the element.
static mlir::Value loadContainerBoxWord(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value array,
                                        mlir::Value slotBase,
                                        std::int64_t wordIndex) {
  mlir::Value offset =
      mlir::arith::ConstantIntOp::create(builder, loc, wordIndex, 64);
  mlir::Value word =
      mlir::arith::AddIOp::create(builder, loc, slotBase, offset).getResult();
  mlir::Value index = mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), word)
                          .getResult();
  return mlir::memref::LoadOp::create(builder, loc, array, index).getResult();
}

// `xs[i]` on a runtime-mode list or tuple (identical physical layout: header,
// length, boxes): runtime bounds check (negative indices normalize),
// IndexError on miss, element rebuilt from its box words and retained
// (borrow → own).
mlir::FailureOr<bool> RuntimeBundleLowerer::lowerRuntimeSequenceGetItem(
    py::GetItemOp op, const RuntimeBundle &containerRef,
    const RuntimeBundle &indexRef) {
  // Copies: binding results below inserts into valueBundles (a DenseMap),
  // which invalidates references into it.
  RuntimeBundle container = containerRef;
  RuntimeBundle index = indexRef;
  if ((container.contractName() != "builtins.list" &&
       container.contractName() != "builtins.tuple") ||
      container.sequenceEvidenceBacked ||
      !container.sequenceElements.empty() ||
      container.physicalValues().size() < 3)
    return false;
  mlir::Type elementContract = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, elementContract,
                                                 "runtime list element");
  if (mlir::failed(shapes))
    return mlir::failure();
  for (mlir::Type shape : *shapes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
    if (!memref || memref.getRank() != 1)
      return false;
  }
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value raw;
  if (index.primitiveI64) {
    raw = index.primitiveI64->value;
  } else if (std::optional<std::int64_t> literal =
                 integerLiteralFromValue(op.getIndex())) {
    raw = mlir::arith::ConstantIntOp::create(builder, loc, *literal, 64);
  } else {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(index.contractName(), "unbox.i64");
    if (!unbox ||
        unbox->function.getNumArguments() != index.physicalValues().size())
      return false;
    mlir::func::CallOp indexCall = RuntimeBundleLowerer::createRuntimeCall(
        loc, *unbox, index.physicalValues());
    raw = indexCall.getResult(0);
  }
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value lengthSlot =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value length = mlir::memref::LoadOp::create(
                           builder, loc, container.physicalValues()[1],
                           lengthSlot)
                           .getResult();
  mlir::Value isNegative = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, raw, zero);
  mlir::Value adjusted =
      mlir::arith::AddIOp::create(builder, loc, raw, length).getResult();
  mlir::Value normalized =
      mlir::arith::SelectOp::create(builder, loc, isNegative, adjusted, raw)
          .getResult();
  mlir::Value lowerOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, normalized, zero);
  mlir::Value upperOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, normalized, length);
  mlir::Value inRange =
      mlir::arith::AndIOp::create(builder, loc, lowerOk, upperOk).getResult();
  mlir::Value outOfRange = mlir::arith::XOrIOp::create(
      builder, loc, inRange,
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult());
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       outOfRange, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    if (mlir::failed(emitRuntimeException(op, "builtins.IndexError",
                                          "sequence index out of range")))
      return mlir::failure();
  }
  builder.setInsertionPointAfter(guard);
  // Unreachable at runtime past the raise; clamp keeps the loads in bounds.
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, inRange, normalized, zero)
          .getResult();
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, 16, 64);
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
          .getResult();
  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (auto [position, shape] : llvm::enumerate(*shapes)) {
    mlir::Value pointerWord = loadContainerBoxWord(
        builder, loc, container.physicalValues()[2], base,
        4 + static_cast<std::int64_t>(position));
    mlir::Value sizeWord = loadContainerBoxWord(
        builder, loc, container.physicalValues()[2], base,
        9 + static_cast<std::int64_t>(position));
    elementValues.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, pointerWord, sizeWord,
        mlir::cast<mlir::MemRefType>(shape)));
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, elementContract,
                                                   elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  RuntimeValue element{elementContract, *canonical,
                       ownership::logicalOwnershipKind(elementContract,
                                                       /*ownsObject=*/false)};
  if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                             "runtime sequence __getitem__",
                                             element)))
    return mlir::failure();
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod) {
    op.emitError() << "runtime sequence getitem needs a runtime __len__";
    return mlir::failure();
  }
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op->getNextNode());
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);
  return true;
}

// `d[k]` on a runtime-mode dict: runtime probe (str keys), KeyError on miss,
// value rebuilt from its box words and retained (borrow → own).
mlir::FailureOr<bool> RuntimeBundleLowerer::lowerRuntimeDictGetItem(
    py::GetItemOp op, const RuntimeBundle &containerRef,
    const RuntimeBundle &indexRef) {
  // Copies: see lowerRuntimeSequenceGetItem.
  RuntimeBundle container = containerRef;
  RuntimeBundle index = indexRef;
  if (container.contractName() != "builtins.dict" ||
      container.mappingEvidenceBacked || !container.mappingKeys.empty() ||
      container.physicalValues().size() < 5)
    return false;
  if (index.contractName() != "builtins.str" ||
      index.physicalValues().size() < 2)
    return false;
  mlir::Type valueContract = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, valueContract,
                                                 "runtime dict value");
  if (mlir::failed(shapes))
    return mlir::failure();
  for (mlir::Type shape : *shapes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
    if (!memref || memref.getRank() != 1)
      return false;
  }
  std::optional<RuntimeSymbol> findSlot =
      manifest.primitive("builtins.dict", "find_slot");
  if (!findSlot) {
    op.emitError() << "runtime manifest has no dict find_slot primitive";
    return mlir::failure();
  }

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value keyBytes = index.physicalValues()[1];
  mlir::Value keyPointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                           keyBytes);
  mlir::Value keyPointer = mlir::arith::IndexCastOp::create(
                               builder, loc, builder.getI64Type(),
                               keyPointerIndex)
                               .getResult();
  mlir::Value keyDim =
      mlir::memref::DimOp::create(
          builder, loc, keyBytes,
          mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult())
          .getResult();
  mlir::Value keyLength = mlir::arith::IndexCastOp::create(
                              builder, loc, builder.getI64Type(), keyDim)
                              .getResult();
  llvm::SmallVector<mlir::Value, 8> findOperands(
      container.physicalValues().begin(), container.physicalValues().end());
  findOperands.push_back(keyPointer);
  findOperands.push_back(keyLength);
  mlir::func::CallOp findCall =
      RuntimeBundleLowerer::createRuntimeCall(loc, *findSlot, findOperands);
  mlir::Value found = findCall.getResult(0);
  // The probe consumed only raw pointer words: pin the key's liveness past
  // it with an explicit __len__ use.
  if (std::optional<RuntimeSymbol> keyLen =
          manifest.method(index.contractName(), "__len__")) {
    llvm::SmallVector<const RuntimeBundle *, 1> keySources{&index};
    llvm::SmallVector<mlir::Value, 4> keyOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *keyLen, keySources,
                                              keyOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(loc, *keyLen, keyOperands);
  }
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value missing = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, found, zero);
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       missing, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    if (mlir::failed(emitRuntimeException(op, "builtins.KeyError",
                                          "key not found")))
      return mlir::failure();
  }
  builder.setInsertionPointAfter(guard);
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, missing, zero, found)
          .getResult();
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, 16, 64);
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
          .getResult();
  llvm::SmallVector<mlir::Value, 4> resultValues;
  for (auto [position, shape] : llvm::enumerate(*shapes)) {
    mlir::Value pointerWord = loadContainerBoxWord(
        builder, loc, container.physicalValues()[3], base,
        4 + static_cast<std::int64_t>(position));
    mlir::Value sizeWord = loadContainerBoxWord(
        builder, loc, container.physicalValues()[3], base,
        9 + static_cast<std::int64_t>(position));
    resultValues.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, pointerWord, sizeWord,
        mlir::cast<mlir::MemRefType>(shape)));
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, valueContract,
                                                   resultValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  RuntimeValue value{valueContract, *canonical,
                     ownership::logicalOwnershipKind(valueContract,
                                                     /*ownsObject=*/false)};
  if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                             "runtime dict __getitem__",
                                             value)))
    return mlir::failure();
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod) {
    op.emitError() << "runtime dict getitem needs a runtime __len__";
    return mlir::failure();
  }
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op->getNextNode());
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);
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

  if (container->ctypes &&
      container->ctypes->kind == RuntimeCtypesEvidence::Kind::Library)
    return RuntimeBundleLowerer::lowerStaticCtypesLibraryGetItem(op,
                                                                 *container);

  mlir::FailureOr<bool> runtimeSequenceHandled =
      RuntimeBundleLowerer::lowerRuntimeSequenceGetItem(op, *container, *index);
  if (mlir::failed(runtimeSequenceHandled))
    return mlir::failure();
  if (*runtimeSequenceHandled)
    return mlir::success();

  mlir::FailureOr<bool> runtimeDictHandled =
      RuntimeBundleLowerer::lowerRuntimeDictGetItem(op, *container, *index);
  if (mlir::failed(runtimeDictHandled))
    return mlir::failure();
  if (*runtimeDictHandled)
    return mlir::success();

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

} // namespace py::lowering
