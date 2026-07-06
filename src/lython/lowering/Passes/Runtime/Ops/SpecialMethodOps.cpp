#include "Runtime/Core/Lowerer.h"

namespace py::lowering {
namespace {

bool isEvidenceCollection(llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple" ||
         contract == "builtins.dict";
}

bool isCollectionMetaType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  if (memref.hasStaticShape() && memref.getDimSize(0) < 1)
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  return element && element.getWidth() == 64;
}

mlir::FailureOr<mlir::Value> loadCollectionLength(mlir::Operation *op,
                                                  mlir::OpBuilder &builder,
                                                  const RuntimeBundle &bundle,
                                                  llvm::StringRef label) {
  if (bundle.physicalValues().size() < 2)
    return op->emitError() << label
                           << " collection has no physical length metadata";
  mlir::Value meta = bundle.physicalValues()[1];
  if (!isCollectionMetaType(meta.getType()))
    return op->emitError() << label
                           << " collection length metadata has invalid type "
                           << meta.getType();
  mlir::Value slot =
      builder.create<mlir::arith::ConstantIndexOp>(op->getLoc(), 0);
  return builder.create<mlir::memref::LoadOp>(op->getLoc(), meta, slot)
      .getResult();
}

mlir::LogicalResult touchCollectionEvidenceUse(mlir::Operation *op,
                                               mlir::OpBuilder &builder,
                                               const RuntimeBundle &bundle,
                                               llvm::StringRef label) {
  mlir::FailureOr<mlir::Value> length =
      loadCollectionLength(op, builder, bundle, label);
  return mlir::failed(length) ? mlir::failure() : mlir::success();
}

bool sameRuntimeValueIdentity(const RuntimeValue &lhs,
                              const RuntimeValue &rhs) {
  if (lhs.values.size() != rhs.values.size())
    return false;
  if (lhs.values.empty())
    return false;
  for (auto [left, right] : llvm::zip(lhs.values, rhs.values))
    if (left != right)
      return false;
  return true;
}

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return builder.create<mlir::arith::ConstantIntOp>(loc, value ? 1 : 0, 1)
      .getResult();
}

std::optional<mlir::Value>
knownEvidenceEquality(mlir::Operation *op, mlir::OpBuilder &builder,
                      const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  if (lhs.kind != RuntimeBundle::Kind::Object ||
      rhs.kind != RuntimeBundle::Kind::Object)
    return std::nullopt;
  mlir::Location loc = op->getLoc();
  if (lhs.literalText && rhs.literalText)
    return constantI1(builder, loc, *lhs.literalText == *rhs.literalText);
  if (lhs.contractName() == "builtins.int" &&
      rhs.contractName() == "builtins.int" && lhs.primitiveI64 &&
      rhs.primitiveI64 && lhs.primitiveI64->value &&
      rhs.primitiveI64->value && lhs.primitiveI64->valid &&
      rhs.primitiveI64->valid) {
    mlir::Value sameValue = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::eq, lhs.primitiveI64->value,
        rhs.primitiveI64->value);
    mlir::Value bothValid = builder.create<mlir::arith::AndIOp>(
        loc, lhs.primitiveI64->valid, rhs.primitiveI64->valid);
    return builder.create<mlir::arith::AndIOp>(loc, bothValid, sameValue)
        .getResult();
  }
  if (sameRuntimeValueIdentity(lhs.objectValue, rhs.objectValue))
    return constantI1(builder, loc, true);
  return std::nullopt;
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

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerReceiverMethodResult(
    mlir::Operation *op, mlir::Value receiverValue, mlir::Value resultValue,
    llvm::StringRef missingSubject, llvm::StringRef methodName,
    bool preferManifestObjectResult) {
  const RuntimeBundle *receiver =
      RuntimeBundleLowerer::bundleFor(receiverValue);
  if (!receiver)
    return op->emitError() << missingSubject
                           << " has no lowered runtime bundle";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{receiver};
  RuntimeBundle methodReceiver = *receiver;
  if (mlir::isa<py::ProtocolType>(receiverValue.getType()) &&
      mlir::isa<py::ProtocolType>(receiver->contract)) {
    methodReceiver.contract = receiverValue.getType();
    methodReceiver.objectValue.contract = receiverValue.getType();
  }
  if (py::ClassOp classOp =
          RuntimeBundleLowerer::classForContract(methodReceiver.contract)) {
    if (std::optional<std::string> methodSymbol =
            RuntimeBundleLowerer::classMethodSymbol(classOp, methodName)) {
      mlir::func::FuncOp target =
          module.lookupSymbol<mlir::func::FuncOp>(*methodSymbol);
      if (!target)
        return op->emitError() << "source class method @" << *methodSymbol
                               << " is not defined";
      if (target->hasAttr("ly.async.body_result")) {
        RuntimeBundle result;
        if (mlir::failed(
                RuntimeBundleLowerer::emitAsyncFunctionTargetCallResult(
                    op, resultValue, target, *methodSymbol, sources, result)))
          return mlir::failure();
        valueBundles[resultValue] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
    }
  }
  if (mlir::failed(lowerManifestMethodResult(
          op, resultValue, methodReceiver, methodName, sources,
          /*allowUnusedSources=*/false, preferManifestObjectResult)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBool(py::BoolOp op) {
  const RuntimeBundle *input = RuntimeBundleLowerer::bundleFor(op.getInput());
  if (!input)
    return op.emitError() << "bool input has no lowered runtime bundle";
  llvm::ArrayRef<mlir::Value> inputValues = input->physicalValues();
  if (input->contractName() == "builtins.bool" && inputValues.size() == 1 &&
      inputValues.front().getType().isInteger(1)) {
    op.getResult().replaceAllUsesWith(inputValues.front());
    erase.push_back(op);
    return mlir::success();
  }
  if (input->kind == RuntimeBundle::Kind::Object &&
      isEvidenceCollection(input->contractName())) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<mlir::Value> length =
        loadCollectionLength(op, builder, *input, "bool");
    if (mlir::failed(length))
      return mlir::failure();
    mlir::Value zero =
        builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 64);
    mlir::Value nonEmpty = builder.create<mlir::arith::CmpIOp>(
        op.getLoc(), mlir::arith::CmpIPredicate::ne, *length, zero);
    op.getResult().replaceAllUsesWith(nonEmpty);
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<const RuntimeBundle *, 1> sources{input};
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__bool__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(op, op.getResult(), *input,
                                               *methodName, sources,
                                               /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerLen(py::LenOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__len__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getInput(), op.getResult(), "len input", *methodName);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSetItem(py::SetItemOp op) {
  llvm::SmallVector<mlir::Value, 3> inputs{op.getContainer(), op.getIndex(),
                                           op.getValue()};
  llvm::SmallVector<const RuntimeBundle *, 3> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "setitem operands need runtime bundles", sources)))
    return mlir::failure();
  const RuntimeBundle &container = *sources[0];
  const RuntimeBundle &index = *sources[1];
  const RuntimeBundle &value = *sources[2];
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.list") {
    std::optional<std::int64_t> rawIndex =
        integerLiteralFromValue(op.getIndex());
    if (rawIndex) {
      builder.setInsertionPoint(op);
      if (mlir::failed(touchCollectionEvidenceUse(op, builder, container,
                                                  "list setitem")))
        return mlir::failure();
      std::int64_t size =
          static_cast<std::int64_t>(container.sequenceElementBundles.size());
      std::int64_t normalized = *rawIndex;
      if (normalized < 0)
        normalized += size;
      if (normalized < 0 || normalized >= size) {
        builder.setInsertionPoint(op);
        if (mlir::failed(emitRuntimeException(op, "builtins.IndexError",
                                              "list assignment index out of "
                                              "range")))
          return mlir::failure();
      } else {
        RuntimeBundle updated = container;
        unsigned position = static_cast<unsigned>(normalized);
        const RuntimeBundle *oldBundle = nullptr;
        if (position < updated.sequenceElementBundles.size())
          oldBundle = updated.sequenceElementBundles[position].get();
        mlir::Type oldType = value.objectValue.contract;
        mlir::ValueRange oldValues;
        if (position < updated.sequenceElements.size()) {
          oldType = updated.sequenceElements[position].contract;
          oldValues = updated.sequenceElements[position].values;
        }
        mlir::FailureOr<RuntimeBundle> payload =
            RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
        if (mlir::failed(payload))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
                op, oldType, oldValues, oldBundle,
                payload->objectValue.contract, *payload, "list.setitem")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
                op, updated, position, *payload)))
          return mlir::failure();
        RuntimeBundle stored = payload->withObjectOwnership(
            ownership::logicalOwnershipKind(
                payload->objectValue.contract, /*ownsObject=*/false));
        if (position < updated.sequenceElements.size())
          updated.sequenceElements[position] = stored.objectValue;
        if (position < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles[position] =
              std::make_shared<RuntimeBundle>(std::move(stored));
        if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op,
                                                                   updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      index.kind == RuntimeBundle::Kind::Object &&
      value.kind == RuntimeBundle::Kind::Object) {
    std::optional<std::string> key =
        RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
    if (!key && index.literalText)
      key = *index.literalText;
    if (key) {
      builder.setInsertionPoint(op);
      if (mlir::failed(touchCollectionEvidenceUse(op, builder, container,
                                                  "dict setitem")))
        return mlir::failure();
      RuntimeBundle updated = container;
      auto found = llvm::find(updated.mappingKeys, *key);
      mlir::FailureOr<RuntimeBundle> payloadKey =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
      if (mlir::failed(payloadKey))
        return mlir::failure();
      mlir::FailureOr<RuntimeBundle> payloadValue =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
      if (mlir::failed(payloadValue))
        return mlir::failure();
      if (found == updated.mappingKeys.end()) {
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadKey, "dict.setitem.key")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadValue, "dict.setitem")))
          return mlir::failure();
        unsigned position = static_cast<unsigned>(updated.mappingValues.size());
        if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
                op, updated, position, *payloadKey)))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                op, updated, position, *payloadValue)))
          return mlir::failure();
        mlir::Value meta = container.physicalValues()[1];
        mlir::Value slot =
            builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        mlir::Value current =
            builder.create<mlir::memref::LoadOp>(op.getLoc(), meta, slot);
        mlir::Value one =
            builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 64);
        mlir::Value next =
            builder.create<mlir::arith::AddIOp>(op.getLoc(), current, one);
        builder.create<mlir::memref::StoreOp>(op.getLoc(), next, meta, slot);
        RuntimeBundle storedKey = payloadKey->withObjectOwnership(
            ownership::logicalOwnershipKind(
                payloadKey->objectValue.contract, /*ownsObject=*/false));
        RuntimeBundle storedValue = payloadValue->withObjectOwnership(
            ownership::logicalOwnershipKind(
                payloadValue->objectValue.contract, /*ownsObject=*/false));
        updated.mappingKeys.push_back(*key);
        updated.mappingKeyBundles.push_back(
            std::make_shared<RuntimeBundle>(storedKey));
        updated.mappingValues.push_back(storedValue.objectValue);
        updated.mappingValueBundles.push_back(
            std::make_shared<RuntimeBundle>(std::move(storedValue)));
        if (!updated.mappingPresent.empty())
          updated.mappingPresent.push_back(constantI1(builder, op.getLoc(),
                                                      true));
      } else {
        unsigned position =
            static_cast<unsigned>(found - updated.mappingKeys.begin());
        mlir::Type oldType = value.objectValue.contract;
        mlir::ValueRange oldValues;
        if (position < updated.mappingValues.size()) {
          oldType = updated.mappingValues[position].contract;
          oldValues = updated.mappingValues[position].values;
        }
        const RuntimeBundle *oldValueBundle = nullptr;
        if (position < updated.mappingValueBundles.size())
          oldValueBundle = updated.mappingValueBundles[position].get();
        if (mlir::failed(RuntimeBundleLowerer::replaceAggregateSlot(
                op, oldType, oldValues, oldValueBundle,
                payloadValue->objectValue.contract, *payloadValue,
                "dict.setitem")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                op, updated, position, *payloadValue)))
          return mlir::failure();
        RuntimeBundle storedValue = payloadValue->withObjectOwnership(
            ownership::logicalOwnershipKind(
                payloadValue->objectValue.contract, /*ownsObject=*/false));
        if (position < updated.mappingValues.size())
          updated.mappingValues[position] = storedValue.objectValue;
        if (position < updated.mappingValueBundles.size())
          updated.mappingValueBundles[position] =
              std::make_shared<RuntimeBundle>(std::move(storedValue));
        if (position < updated.mappingPresent.size()) {
          builder.setInsertionPoint(op);
          updated.mappingPresent[position] =
              constantI1(builder, op.getLoc(), true);
        }
      }
      if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op,
                                                                 updated)))
        return mlir::failure();
      valueBundles[op.getContainer()] = std::move(updated);
      erase.push_back(op);
      return mlir::success();
    }
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__setitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerDelItem(py::DelItemOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getIndex()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "delitem operands need runtime bundles", sources)))
    return mlir::failure();
  const RuntimeBundle &container = *sources[0];
  const RuntimeBundle &index = *sources[1];
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.list") {
    std::optional<std::int64_t> rawIndex =
        integerLiteralFromValue(op.getIndex());
    if (rawIndex) {
      RuntimeBundle updated = container;
      std::int64_t size =
          static_cast<std::int64_t>(updated.sequenceElementBundles.size());
      std::int64_t normalized = *rawIndex;
      if (normalized < 0)
        normalized += size;
      builder.setInsertionPoint(op);
      if (normalized < 0 || normalized >= size) {
        if (mlir::failed(emitRuntimeException(op, "builtins.IndexError",
                                              "list assignment index out of "
                                              "range")))
          return mlir::failure();
      } else {
        mlir::FailureOr<mlir::Value> length =
            loadCollectionLength(op, builder, container, "list delitem");
        if (mlir::failed(length))
          return mlir::failure();
        mlir::Value meta = container.physicalValues()[1];
        mlir::Value slot =
            builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        mlir::Value one =
            builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 64);
        mlir::Value next =
            builder.create<mlir::arith::SubIOp>(op.getLoc(), *length, one);
        builder.create<mlir::memref::StoreOp>(op.getLoc(), next, meta, slot);
        unsigned position = static_cast<unsigned>(normalized);
        unsigned oldSize =
            static_cast<unsigned>(updated.sequenceElementBundles.size());
        if (position < updated.sequenceElementBundles.size() &&
            updated.sequenceElementBundles[position]) {
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, *updated.sequenceElementBundles[position],
                  "list.delitem")))
            return mlir::failure();
        } else if (position < updated.sequenceElements.size()) {
          const RuntimeValue &oldElement = updated.sequenceElements[position];
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, oldElement.contract, oldElement.values,
                  "list.delitem")))
            return mlir::failure();
        }
        if (position < updated.sequenceElements.size())
          updated.sequenceElements.erase(updated.sequenceElements.begin() +
                                         position);
        if (position < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles.erase(
              updated.sequenceElementBundles.begin() + position);
        for (unsigned rewrite = position,
                      end = static_cast<unsigned>(
                          updated.sequenceElementBundles.size());
             rewrite < end; ++rewrite) {
          if (updated.sequenceElementBundles[rewrite] &&
              mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
                  op, updated, rewrite,
                  *updated.sequenceElementBundles[rewrite])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearSequencePayloadElement(
                op, updated, oldSize - 1)))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op,
                                                                   updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict") {
    std::optional<std::string> key =
        RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
    if (!key && index.literalText)
      key = *index.literalText;
    if (key) {
      RuntimeBundle updated = container;
      auto found = llvm::find(updated.mappingKeys, *key);
      builder.setInsertionPoint(op);
      if (found == updated.mappingKeys.end()) {
        if (mlir::failed(
                emitRuntimeException(op, "builtins.KeyError", "key not found")))
          return mlir::failure();
      } else {
        mlir::FailureOr<mlir::Value> length =
            loadCollectionLength(op, builder, container, "dict delitem");
        if (mlir::failed(length))
          return mlir::failure();
        mlir::Value meta = container.physicalValues()[1];
        mlir::Value slot =
            builder.create<mlir::arith::ConstantIndexOp>(op.getLoc(), 0);
        mlir::Value one =
            builder.create<mlir::arith::ConstantIntOp>(op.getLoc(), 1, 64);
        mlir::Value next =
            builder.create<mlir::arith::SubIOp>(op.getLoc(), *length, one);
        builder.create<mlir::memref::StoreOp>(op.getLoc(), next, meta, slot);
        unsigned position =
            static_cast<unsigned>(found - updated.mappingKeys.begin());
        unsigned oldSize = static_cast<unsigned>(updated.mappingKeys.size());
        if (position < updated.mappingKeyBundles.size() &&
            updated.mappingKeyBundles[position]) {
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, *updated.mappingKeyBundles[position],
                  "dict.delitem.key")))
            return mlir::failure();
        }
        if (position < updated.mappingValues.size()) {
          if (position < updated.mappingValueBundles.size() &&
              updated.mappingValueBundles[position]) {
            if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                    op, *updated.mappingValueBundles[position],
                    "dict.delitem")))
              return mlir::failure();
          } else {
            const RuntimeValue &oldValue = updated.mappingValues[position];
            if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                    op, oldValue.contract, oldValue.values, "dict.delitem")))
              return mlir::failure();
          }
        }
        updated.mappingKeys.erase(updated.mappingKeys.begin() + position);
        if (position < updated.mappingKeyBundles.size())
          updated.mappingKeyBundles.erase(updated.mappingKeyBundles.begin() +
                                          position);
        if (position < updated.mappingValues.size())
          updated.mappingValues.erase(updated.mappingValues.begin() + position);
        if (position < updated.mappingValueBundles.size())
          updated.mappingValueBundles.erase(
              updated.mappingValueBundles.begin() + position);
        if (position < updated.mappingPresent.size())
          updated.mappingPresent.erase(updated.mappingPresent.begin() +
                                       position);
        for (unsigned rewrite = position,
                      end = static_cast<unsigned>(
                          updated.mappingKeyBundles.size());
             rewrite < end; ++rewrite) {
          if (rewrite >= updated.mappingValueBundles.size() ||
              !updated.mappingKeyBundles[rewrite] ||
              !updated.mappingValueBundles[rewrite])
            return op.emitError()
                   << "dict delitem needs key/value payload evidence to "
                      "compact storage";
          if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
                  op, updated, rewrite, *updated.mappingKeyBundles[rewrite])))
            return mlir::failure();
          if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
                  op, updated, rewrite,
                  *updated.mappingValueBundles[rewrite])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearDictPayloadEntry(
                op, updated, oldSize - 1)))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op,
                                                                   updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__delitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestVoidMethod(op, *sources.front(), *methodName,
                                           sources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerContains(py::ContainsOp op) {
  llvm::SmallVector<mlir::Value, 2> inputs{op.getContainer(), op.getItem()};
  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "contains operands need runtime bundles", sources)))
    return mlir::failure();
  const RuntimeBundle &container = *sources.front();
  const RuntimeBundle &item = *sources.back();
  if (container.kind == RuntimeBundle::Kind::Object &&
      (container.contractName() == "builtins.list" ||
       container.contractName() == "builtins.tuple") &&
      !container.sequenceElementBundles.empty()) {
    builder.setInsertionPoint(op);
    if (mlir::failed(touchCollectionEvidenceUse(op, builder, container,
                                                "sequence contains")))
      return mlir::failure();
    mlir::Location loc = op.getLoc();
    mlir::Value result = constantI1(builder, loc, false);
    for (const std::shared_ptr<RuntimeBundle> &element :
         container.sequenceElementBundles) {
      if (!element)
        return mlir::failure();
      std::optional<mlir::Value> equal =
          knownEvidenceEquality(op, builder, *element, item);
      if (!equal)
        return mlir::failure();
      result = builder.create<mlir::arith::OrIOp>(loc, result, *equal);
    }
    op.getResult().replaceAllUsesWith(result);
    erase.push_back(op);
    return mlir::success();
  }
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      !container.mappingKeys.empty() && item.contractName() == "builtins.str") {
    builder.setInsertionPoint(op);
    if (mlir::failed(touchCollectionEvidenceUse(op, builder, container,
                                                "dict contains")))
      return mlir::failure();
    mlir::Location loc = op.getLoc();
    mlir::Value result = constantI1(builder, loc, false);
    if (item.literalText) {
      for (auto [index, key] : llvm::enumerate(container.mappingKeys)) {
        mlir::Value match = constantI1(builder, loc, key == *item.literalText);
        if (index < container.mappingPresent.size())
          match = builder.create<mlir::arith::AndIOp>(
              loc, match, container.mappingPresent[index]);
        result = builder.create<mlir::arith::OrIOp>(loc, result, match);
      }
      op.getResult().replaceAllUsesWith(result);
      erase.push_back(op);
      return mlir::success();
    }

    std::optional<RuntimeSymbol> eq = manifest.method("builtins.str", "__eq__");
    if (!eq)
      return op.emitError() << "dict evidence contains needs str.__eq__";
    for (auto [index, key] : llvm::enumerate(container.mappingKeys)) {
      RuntimeBundle keyObject;
      if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
              op, key, keyObject)))
        return mlir::failure();
      llvm::SmallVector<const RuntimeBundle *, 2> eqSources{&item, &keyObject};
      llvm::SmallVector<mlir::Value, 4> eqOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *eq, eqSources, eqOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      mlir::func::CallOp eqCall =
          RuntimeBundleLowerer::createRuntimeCall(loc, *eq, eqOperands);
      if (eqCall.getNumResults() != 1 ||
          !eqCall.getResult(0).getType().isInteger(1))
        return eq->function.emitError()
               << "str.__eq__ evidence method must return one i1";
      mlir::Value match = eqCall.getResult(0);
      if (index < container.mappingPresent.size())
        match = builder.create<mlir::arith::AndIOp>(
            loc, match, container.mappingPresent[index]);
      result = builder.create<mlir::arith::OrIOp>(loc, result, match);
    }
    op.getResult().replaceAllUsesWith(result);
    erase.push_back(op);
    return mlir::success();
  }
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__contains__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestI1MethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerIter(py::IterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getIterable(),
                                                op.getResult());
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__iter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getIterable(), op.getResult(), "iter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNext(py::NextOp op) {
  const RuntimeBundle *iterator =
      RuntimeBundleLowerer::bundleFor(op.getIterator());
  if (!iterator)
    return op.emitError() << "next iterator has no lowered runtime bundle";

  llvm::SmallVector<const RuntimeBundle *, 1> sources{iterator};
  std::optional<EmittedRuntimeCall> emitted;
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__next__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(emitManifestMethodCall(op, *iterator, *methodName, sources,
                                          /*allowUnusedSources=*/false,
                                          emitted)))
    return mlir::failure();
  if (!emitted->symbol.validResultIndex)
    return op.emitError()
           << "runtime __next__ method must declare valid_result_index";

  mlir::func::CallOp call = emitted->call;
  unsigned validIndex = *emitted->symbol.validResultIndex;
  if (validIndex >= call.getNumResults())
    return op.emitError() << "runtime __next__ valid_result_index is outside "
                             "the result list";
  mlir::Value valid = call.getResult(validIndex);
  if (!valid.getType().isInteger(1))
    return op.emitError() << "runtime __next__ valid result must be an i1";

  std::string elementContract = runtimeContractName(op.getElement().getType());
  if (elementContract.empty())
    elementContract = emitted->symbol.elementContract;
  if (elementContract.empty())
    return op.emitError() << "runtime __next__ element needs a concrete "
                             "manifest element contract";

  std::string nextContract = runtimeContractName(op.getNext().getType());
  if (nextContract.empty())
    nextContract = emitted->symbol.nextContract;
  if (nextContract.empty())
    nextContract = iterator->contractName();
  if (nextContract.empty())
    return op.emitError() << "runtime __next__ next state needs a concrete "
                             "manifest contract";

  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (unsigned index = 0; index < validIndex; ++index)
    elementValues.push_back(call.getResult(index));

  llvm::SmallVector<mlir::Value, 4> nextValues;
  for (unsigned index = validIndex + 1; index < call.getNumResults(); ++index)
    nextValues.push_back(call.getResult(index));

  op.getValid().replaceAllUsesWith(valid);
  if (mlir::failed(assignObjectBundle(
          op, op.getElement(), runtimeContractType(context, elementContract),
          elementValues)))
    return mlir::failure();
  if (mlir::failed(assignObjectBundle(
          op, op.getNext(), runtimeContractType(context, nextContract),
          nextValues)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerEnter(py::EnterOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__enter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getManager(), op.getResult(), "enter manager", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerExit(py::ExitOp op) {
  llvm::SmallVector<mlir::Value, 4> inputs{op.getManager(), op.getExcType(),
                                           op.getExcValue(), op.getTraceback()};
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "exit operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__exit__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/true,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAEnter(py::AEnterOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aenter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getManager(), op.getResult(), "aenter manager", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAExit(py::AExitOp op) {
  llvm::SmallVector<mlir::Value, 4> inputs{op.getManager(), op.getExcType(),
                                           op.getExcValue(), op.getTraceback()};
  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  if (mlir::failed(collectObjectSources(
          op, inputs, "aexit operands need runtime bundles", sources)))
    return mlir::failure();
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aexit__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/true,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerAIter(py::AIterOp op) {
  if (op.getReturnedSelf())
    return RuntimeBundleLowerer::lowerAliasView(op, op.getAsyncIterable(),
                                                op.getResult());
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__aiter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterable(), op.getResult(), "aiter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerANext(py::ANextOp op) {
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__anext__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getAsyncIterator(), op.getAwaitable(), "anext iterator",
      *methodName,
      /*preferManifestObjectResult=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerRound(py::RoundOp op) {
  if (op.getInputs().empty())
    return op.emitError() << "round requires at least a receiver input";

  llvm::SmallVector<const RuntimeBundle *, 2> sources;
  if (mlir::failed(collectObjectSources(
          op, op.getInputs(), "round input has no lowered runtime bundle",
          sources)))
    return mlir::failure();

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__round__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *sources.front(), *methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerUnarySpecial(mlir::Operation *op, mlir::Value input,
                                        llvm::StringRef methodName,
                                        mlir::Value resultValue) {
  if (methodName == "__repr__") {
    const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
    if (!inputBundle)
      return op->emitError() << "repr operand has no lowered runtime bundle";
    if (RuntimeBundleLowerer::needsDefaultObjectRepr(*inputBundle)) {
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
              op, *inputBundle, result)))
        return mlir::failure();
      valueBundles[resultValue] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }
  }
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, input, resultValue, "unary special method operand", methodName,
      /*preferManifestObjectResult=*/true);
}

} // namespace py::lowering
