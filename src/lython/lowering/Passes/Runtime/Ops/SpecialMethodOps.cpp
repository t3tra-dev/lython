#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

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
      mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 0);
  return mlir::memref::LoadOp::create(builder, op->getLoc(), meta, slot)
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
  // Ownership rewrapping (retain markers) must not break identity: compare
  // the values underneath any identity-cast markers.
  for (auto [left, right] : llvm::zip(lhs.values, rhs.values))
    if (ownership::underlyingObjectValue(left) !=
        ownership::underlyingObjectValue(right))
      return false;
  return true;
}

mlir::Value constantI1(mlir::OpBuilder &builder, mlir::Location loc,
                       bool value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value ? 1 : 0, 1)
      .getResult();
}

std::optional<mlir::Value> knownEvidenceEquality(mlir::Operation *op,
                                                 mlir::OpBuilder &builder,
                                                 const RuntimeBundle &lhs,
                                                 const RuntimeBundle &rhs) {
  if (lhs.kind != RuntimeBundle::Kind::Object ||
      rhs.kind != RuntimeBundle::Kind::Object)
    return std::nullopt;
  mlir::Location loc = op->getLoc();
  if (lhs.literalText && rhs.literalText)
    return constantI1(builder, loc, *lhs.literalText == *rhs.literalText);
  if (lhs.contractName() == "builtins.int" &&
      rhs.contractName() == "builtins.int" && lhs.primitiveI64 &&
      rhs.primitiveI64 && lhs.primitiveI64->value && rhs.primitiveI64->value &&
      lhs.primitiveI64->valid && rhs.primitiveI64->valid) {
    mlir::Value sameValue = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, lhs.primitiveI64->value,
        rhs.primitiveI64->value);
    mlir::Value bothValid = mlir::arith::AndIOp::create(
        builder, loc, lhs.primitiveI64->valid, rhs.primitiveI64->valid);
    return mlir::arith::AndIOp::create(builder, loc, bothValid, sameValue)
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
        return op->emitError()
               << "source class method @" << *methodSymbol << " is not defined";
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
        mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
    mlir::Value nonEmpty = mlir::arith::CmpIOp::create(
        builder, op.getLoc(), mlir::arith::CmpIPredicate::ne, *length, zero);
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
        RuntimeBundle stored =
            payload->withObjectOwnership(ownership::logicalOwnershipKind(
                payload->objectValue.contract, /*ownsObject=*/false));
        if (position < updated.sequenceElements.size())
          updated.sequenceElements[position] = stored.objectValue;
        if (position < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles[position] =
              std::make_shared<RuntimeBundle>(std::move(stored));
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
          return mlir::failure();
        valueBundles[op.getContainer()] = std::move(updated);
      }
      erase.push_back(op);
      return mlir::success();
    }
  }
  bool structuralMutation =
      op->hasAttr("ly.structural_mutation") && op.getNumResults() == 1;
  bool runtimeDictInsert =
      container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.dict" &&
      !container.mappingEvidenceBacked && container.mappingKeys.empty() &&
      container.physicalValues().size() >= 5 &&
      index.kind == RuntimeBundle::Kind::Object &&
      value.kind == RuntimeBundle::Kind::Object;
  if ((structuralMutation || op.getNumResults() == 0) && runtimeDictInsert) {
    // Runtime-mode dict (loop-built): contents are only known to the
    // runtime. Insert through the runtime probe; the rebind result carries
    // the (possibly reallocated) representation.
    if (index.contractName() != "builtins.str")
      return op.emitError()
             << "runtime-mode dict assignment requires str keys, got "
             << index.objectValue.contract;
    std::optional<RuntimeSymbol> setItemBox =
        manifest.primitive("builtins.dict", "setitem_box");
    if (!setItemBox)
      return op.emitError()
             << "runtime manifest has no dict setitem_box primitive";

    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payloadKey =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
    if (mlir::failed(payloadKey))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> payloadValue =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, value);
    if (mlir::failed(payloadValue))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadKey, "dict.setitem.key")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadValue, "dict.setitem")))
      return mlir::failure();

    auto transientBox =
        [&](const RuntimeBundle &bundle) -> mlir::FailureOr<mlir::Value> {
      auto boxType = mlir::MemRefType::get({16}, builder.getI64Type());
      mlir::Value box =
          mlir::memref::AllocaOp::create(builder, loc, boxType).getResult();
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
          RuntimeBundleLowerer::objectPayloadHandleWords(op, bundle,
                                                         /*ownsPayload=*/true);
      if (mlir::failed(words))
        return mlir::failure();
      for (auto [wordIndex, word] : llvm::enumerate(*words)) {
        mlir::Value slot = mlir::arith::ConstantIndexOp::create(
            builder, loc, static_cast<std::int64_t>(wordIndex));
        mlir::memref::StoreOp::create(builder, loc, word, box, slot);
      }
      return box;
    };
    mlir::FailureOr<mlir::Value> keyBox = transientBox(*payloadKey);
    if (mlir::failed(keyBox))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> valueBox = transientBox(*payloadValue);
    if (mlir::failed(valueBox))
      return mlir::failure();

    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(*keyBox);
    operands.push_back(*valueBox);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *setItemBox, operands);

    RuntimeBundle updated;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, container.objectValue.contract, call.getResults(), updated)))
      return mlir::failure();
    updated.fieldAliasOwner = container.fieldAliasOwner;
    updated.fieldAliasName = container.fieldAliasName;
    if (structuralMutation) {
      valueBundles[op.getResult(0)] = std::move(updated);
    } else {
      // Non-rebind form (a box-fronted field container, `self.data[k] = v`):
      // the call transferred the container's old storage token, and the
      // holder absorbs the fresh one — retain-into-slot + end the local
      // token, the same marker pair a field store emits. The (possibly
      // reallocated) representation then writes back through the FIELD BOX
      // words (the box pointer is a stable SSA value in the owner bundle),
      // so reads in other branches/iterations observe the mutation without
      // any cross-block SSA rewrite.
      if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, updated, "dict.setitem.writeback")))
        return mlir::failure();
      if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
              op, updated, "dict.setitem.writeback.source")))
        return mlir::failure();
      mlir::Value fieldBox;
      if (updated.fieldAliasOwner && !updated.fieldAliasName.empty()) {
        auto owner = valueBundles.find(updated.fieldAliasOwner);
        if (owner != valueBundles.end() &&
            owner->second.kind == RuntimeBundle::Kind::Object) {
          if (py::ClassOp ownerClass = RuntimeBundleLowerer::classForContract(
                  owner->second.objectValue.contract)) {
            std::optional<unsigned> ownerFieldIndex =
                RuntimeBundleLowerer::classFieldIndex(ownerClass,
                                                      updated.fieldAliasName);
            if (ownerFieldIndex) {
              mlir::FailureOr<unsigned> fieldOffset =
                  RuntimeBundleLowerer::classFieldValueOffset(
                      op, ownerClass, *ownerFieldIndex,
                      "dict field writeback ABI");
              if (mlir::failed(fieldOffset))
                return mlir::failure();
              if (*fieldOffset < owner->second.physicalValues().size())
                fieldBox = owner->second.physicalValues()[*fieldOffset];
            }
          }
        }
      }
      if (!fieldBox)
        return op.emitError()
               << "in-place dict assignment requires a box-fronted field "
                  "container";
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
          RuntimeBundleLowerer::objectPayloadHandleWords(op, updated,
                                                         /*ownsPayload=*/true);
      if (mlir::failed(words))
        return mlir::failure();
      // Words 0/1 (refcount, class id) are invariant under mutation; refresh
      // the payload pointer and every (ptr, size) pair.
      for (unsigned wordIndex = 2; wordIndex < words->size(); ++wordIndex) {
        mlir::Value slot = mlir::arith::ConstantIndexOp::create(
            builder, loc, static_cast<std::int64_t>(wordIndex));
        mlir::memref::StoreOp::create(builder, loc, (*words)[wordIndex],
                                      fieldBox, slot);
      }
      valueBundles[op.getContainer()] = std::move(updated);
    }
    erase.push_back(op);
    return mlir::success();
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
            mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
        mlir::Value current =
            mlir::memref::LoadOp::create(builder, op.getLoc(), meta, slot);
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 64);
        mlir::Value next =
            mlir::arith::AddIOp::create(builder, op.getLoc(), current, one);
        mlir::memref::StoreOp::create(builder, op.getLoc(), next, meta, slot);
        RuntimeBundle storedKey =
            payloadKey->withObjectOwnership(ownership::logicalOwnershipKind(
                payloadKey->objectValue.contract, /*ownsObject=*/false));
        RuntimeBundle storedValue =
            payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
                payloadValue->objectValue.contract, /*ownsObject=*/false));
        updated.mappingKeys.push_back(*key);
        updated.mappingKeyBundles.push_back(
            std::make_shared<RuntimeBundle>(storedKey));
        updated.mappingValues.push_back(storedValue.objectValue);
        updated.mappingValueBundles.push_back(
            std::make_shared<RuntimeBundle>(std::move(storedValue)));
        if (!updated.mappingPresent.empty())
          updated.mappingPresent.push_back(
              constantI1(builder, op.getLoc(), true));
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
        RuntimeBundle storedValue =
            payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
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
      if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
        return mlir::failure();
      // The structural-mutation rebind form: downstream uses read the RESULT
      // SSA value (the emitter rebound the local to it), so the updated
      // evidence must be visible under that name too.
      if (op.getNumResults() == 1)
        valueBundles[op.getResult(0)] = updated;
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
            mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 64);
        mlir::Value next =
            mlir::arith::SubIOp::create(builder, op.getLoc(), *length, one);
        mlir::memref::StoreOp::create(builder, op.getLoc(), next, meta, slot);
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
                  op, oldElement.contract, oldElement.values, "list.delitem")))
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
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
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
            mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 64);
        mlir::Value next =
            mlir::arith::SubIOp::create(builder, op.getLoc(), *length, one);
        mlir::memref::StoreOp::create(builder, op.getLoc(), next, meta, slot);
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
        for (unsigned rewrite = position, end = static_cast<unsigned>(
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
                  op, updated, rewrite, *updated.mappingValueBundles[rewrite])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearDictPayloadEntry(
                op, updated, oldSize - 1)))
          return mlir::failure();
        if (mlir::failed(
                RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
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
  const RuntimeBundle container = *sources.front();
  const RuntimeBundle item = *sources.back();
  if (container.kind == RuntimeBundle::Kind::Object &&
      container.contractName() == "builtins.set" &&
      container.physicalValues().size() >= 3 &&
      item.kind == RuntimeBundle::Kind::Object) {
    // Runtime set membership: probe with a BORROWED transient element box,
    // then pin both the container and the probed item past the call (the box
    // holds raw pointer words the liveness cannot see).
    std::string elementContract = item.contractName();
    if (elementContract != "builtins.int" && elementContract != "builtins.str")
      return op.emitError()
             << "set membership currently requires int or str evidence, got "
             << item.objectValue.contract;
    std::optional<RuntimeSymbol> containsBox =
        manifest.primitive("builtins.set", "contains_box");
    if (!containsBox)
      return op.emitError()
             << "runtime manifest has no set contains_box primitive";
    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, item);
    if (mlir::failed(payload))
      return mlir::failure();
    auto boxType = mlir::MemRefType::get({16}, builder.getI64Type());
    mlir::Value box =
        mlir::memref::AllocaOp::create(builder, loc, boxType).getResult();
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
        RuntimeBundleLowerer::objectPayloadHandleWords(op, *payload,
                                                       /*ownsPayload=*/false);
    if (mlir::failed(words))
      return mlir::failure();
    for (auto [wordIndex, word] : llvm::enumerate(*words)) {
      mlir::Value slot = mlir::arith::ConstantIndexOp::create(
          builder, loc, static_cast<std::int64_t>(wordIndex));
      mlir::memref::StoreOp::create(builder, loc, word, box, slot);
    }
    llvm::SmallVector<mlir::Value, 6> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(box);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *containsBox, operands);
    auto pinObject = [&](const RuntimeBundle &object,
                         llvm::StringRef pinMethod) -> mlir::LogicalResult {
      std::optional<RuntimeSymbol> method =
          manifest.method(object.contractName(), pinMethod);
      if (!method)
        return op.emitError() << "set membership needs " << object.contractName()
                              << "." << pinMethod << " to pin its operand";
      llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&object};
      llvm::SmallVector<mlir::Value, 4> pinOperands;
      if (mlir::failed(buildRuntimeCallOperands(op, *method, pinSources,
                                                pinOperands,
                                                /*allowUnusedSources=*/false)))
        return mlir::failure();
      RuntimeBundleLowerer::createRuntimeCall(loc, *method, pinOperands);
      return mlir::success();
    };
    if (mlir::failed(pinObject(container, "__len__")))
      return mlir::failure();
    if (mlir::failed(pinObject(
            *payload,
            elementContract == "builtins.str" ? "__len__" : "__int__")))
      return mlir::failure();
    op.getResult().replaceAllUsesWith(call.getResult(0));
    erase.push_back(op);
    return mlir::success();
  }
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
      result = mlir::arith::OrIOp::create(builder, loc, result, *equal);
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
          match = mlir::arith::AndIOp::create(builder, loc, match,
                                              container.mappingPresent[index]);
        result = mlir::arith::OrIOp::create(builder, loc, result, match);
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
        match = mlir::arith::AndIOp::create(builder, loc, match,
                                            container.mappingPresent[index]);
      result = mlir::arith::OrIOp::create(builder, loc, result, match);
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

  // Statically evidenced list iteration: there is no runtime `list.__iter__`
  // object; iterate the compile-time element evidence through a hoisted
  // position cell instead. The cell is alloca'd once per function (so nested
  // re-creation of the iterator reuses the slot) and reset to zero here.
  if (const RuntimeBundle *iterable =
          RuntimeBundleLowerer::bundleFor(op.getIterable())) {
    bool evidenceListIterable = iterable->contractName() == "builtins.list" &&
                                !iterable->sequenceElements.empty() &&
                                iterable->sequenceIndices.empty() &&
                                !iterable->evidenceIteratorCell;
    // Runtime-mode list (no compile-time element evidence): iterate the
    // runtime payload through the same hoisted position cell; `py.next`
    // rebuilds each element from its payload box words.
    bool runtimeListIterable = iterable->contractName() == "builtins.list" &&
                               !iterable->sequenceEvidenceBacked &&
                               iterable->sequenceElements.empty() &&
                               !iterable->evidenceIteratorCell &&
                               iterable->physicalValues().size() >= 3;
    // Runtime-mode dict key iteration: the key boxes live in the keys array
    // at the same physical positions the list uses for its items (meta at
    // [1], boxes at [2]), so the runtime-list next path applies verbatim.
    bool runtimeDictIterable = iterable->contractName() == "builtins.dict" &&
                               !iterable->mappingEvidenceBacked &&
                               iterable->mappingKeys.empty() &&
                               iterable->sequenceElements.empty() &&
                               !iterable->evidenceIteratorCell &&
                               iterable->physicalValues().size() >= 5;
    // Runtime sets share the list's physical layout exactly (meta at [1],
    // boxed slots at [2]), so the runtime-list next path applies verbatim.
    bool runtimeSetIterable = iterable->contractName() == "builtins.set" &&
                              iterable->sequenceElements.empty() &&
                              !iterable->evidenceIteratorCell &&
                              iterable->physicalValues().size() >= 3;
    if (evidenceListIterable || runtimeListIterable || runtimeDictIterable ||
        runtimeSetIterable) {
      mlir::func::FuncOp function = op->getParentOfType<mlir::func::FuncOp>();
      if (!function)
        return op.emitError() << "list iteration requires a function context";
      mlir::Value cell;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&function.getBody().front());
        cell = mlir::memref::AllocaOp::create(
                   builder, op.getLoc(),
                   mlir::MemRefType::get({1}, builder.getI64Type()))
                   .getResult();
      }
      builder.setInsertionPoint(op);
      mlir::Value zero =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
      mlir::Value slot =
          mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
      mlir::memref::StoreOp::create(builder, op.getLoc(), zero, cell, slot);
      RuntimeBundle iterator = *iterable;
      iterator.evidenceIteratorCell = cell;
      valueBundles[op.getResult()] = std::move(iterator);
      erase.push_back(op);
      return mlir::success();
    }
  }

  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__iter__");
  if (mlir::failed(methodName))
    return mlir::failure();
  return RuntimeBundleLowerer::lowerReceiverMethodResult(
      op, op.getIterable(), op.getResult(), "iter iterable", *methodName,
      /*preferManifestObjectResult=*/true);
}

// `py.next` over a statically evidenced list iterator: bounds-check the
// position cell, select the element from the compile-time evidence, advance
// the cell, and pin the list's liveness with an explicit `__len__` use so the
// borrowed element stays valid throughout the loop.
mlir::LogicalResult
RuntimeBundleLowerer::lowerListEvidenceNext(py::NextOp op,
                                            RuntimeBundle iterator) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value cell = iterator.evidenceIteratorCell;
  mlir::Value slot = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value position =
      mlir::memref::LoadOp::create(builder, loc, cell, slot).getResult();
  mlir::Value size = mlir::arith::ConstantIntOp::create(
      builder, loc, static_cast<std::int64_t>(iterator.sequenceElements.size()),
      64);
  mlir::Value valid = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, position, size);
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value advanced =
      mlir::arith::AddIOp::create(builder, loc, position, one);
  mlir::memref::StoreOp::create(builder, loc, advanced, cell, slot);

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(iterator.sequenceElements.size());
  for (unsigned index = 0, end = iterator.sequenceElements.size(); index < end;
       ++index) {
    mlir::Value expected = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(index), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, position, expected);
    matches.push_back(
        mlir::arith::AndIOp::create(builder, loc, valid, indexMatches));
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getElement(), iterator.sequenceElements, matches,
          "list iteration", "builtins.IndexError", "list iteration exhausted",
          /*raiseOnMiss=*/false);
  if (mlir::failed(selected))
    return mlir::failure();

  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(iterator.contractName(), "__len__");
  if (!lenMethod)
    return op.emitError()
           << "list iteration needs a runtime __len__ to pin the container";
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&iterator};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);

  op.getValid().replaceAllUsesWith(valid);
  valueBundles[op.getElement()] = std::move(*selected);
  valueBundles[op.getNext()] = iterator;
  erase.push_back(op);
  return mlir::success();
}

// `py.next` over a runtime-mode list iterator: bounds-check the position
// cell against the runtime length, rebuild the element's physical values from
// its payload box words (immortal dead placeholder words are selected on the
// exhausted branch, so the unconditional retain below is a no-op there),
// retain the element via its contract's `own` primitive, advance the cell,
// and pin the list's liveness with an explicit `__len__` use.
// Rebuild a rank-1 memref from a payload box pointer/size word pair. The
// contract → physical shape relation is static, so the descriptor is
// assembled inline (llvm.insertvalue chain reconciled with the memref world
// by the standard unrealized-cast materialization). Borrow-only: the result
// aliases storage owned by the boxed element.
mlir::Value RuntimeBundleLowerer::memrefFromBoxWords(mlir::OpBuilder &builder,
                                                     mlir::Location loc,
                                                     mlir::Value pointerWord,
                                                     mlir::Value sizeWord,
                                                     mlir::MemRefType type) {
  mlir::MLIRContext *context = builder.getContext();
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  auto arrayType = mlir::LLVM::LLVMArrayType::get(i64, 1);
  auto descriptorType = mlir::LLVM::LLVMStructType::getLiteral(
      context, {ptrType, ptrType, i64, arrayType, arrayType});
  mlir::Value pointer =
      mlir::LLVM::IntToPtrOp::create(builder, loc, ptrType, pointerWord);
  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult();
  mlir::Value one =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 64).getResult();
  mlir::Value size =
      type.hasStaticShape()
          ? mlir::arith::ConstantIntOp::create(builder, loc,
                                               type.getDimSize(0), 64)
                .getResult()
          : sizeWord;
  mlir::Value descriptor =
      mlir::LLVM::UndefOp::create(builder, loc, descriptorType);
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, pointer, llvm::ArrayRef<std::int64_t>{1});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, zero, llvm::ArrayRef<std::int64_t>{2});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, size, llvm::ArrayRef<std::int64_t>{3, 0});
  descriptor = mlir::LLVM::InsertValueOp::create(
      builder, loc, descriptor, one, llvm::ArrayRef<std::int64_t>{4, 0});
  return mlir::UnrealizedConversionCastOp::create(builder, loc,
                                                  mlir::TypeRange{type},
                                                  descriptor)
      .getResult(0);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerListRuntimeNext(py::NextOp op,
                                           RuntimeBundle iterator) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value cell = iterator.evidenceIteratorCell;
  if (iterator.physicalValues().size() < 3)
    return op.emitError() << "runtime list iterator has no physical payload";

  mlir::Type elementContract = op.getElement().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> elementShapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, elementContract,
                                                 "runtime list element");
  if (mlir::failed(elementShapes))
    return mlir::failure();
  for (mlir::Type shape : *elementShapes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
    if (!memref || memref.getRank() != 1)
      return op.emitError()
             << "iteration over a runtime-mode list of " << elementContract
             << " requires rank-1 memref physical values, got " << shape;
  }

  mlir::Value slot = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value position =
      mlir::memref::LoadOp::create(builder, loc, cell, slot).getResult();
  mlir::Value meta = iterator.physicalValues()[1];
  mlir::Value length =
      mlir::memref::LoadOp::create(builder, loc, meta, slot).getResult();
  mlir::Value valid = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, position, length);
  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
  mlir::Value advanced =
      mlir::arith::AddIOp::create(builder, loc, position, one);
  mlir::memref::StoreOp::create(builder, loc, advanced, cell, slot);

  // Exhausted-branch placeholder: the payload always has at least one
  // allocated slot, so clamping keeps the loads in bounds; the loaded words
  // are replaced by the immortal placeholder's words before use.
  mlir::Value zero64 =
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, valid, position, zero64)
          .getResult();

  mlir::FailureOr<RuntimeValue> dead =
      RuntimeBundleLowerer::materializeNonOwningDeadObjectValue(
          op.getOperation(), elementContract,
          "exhausted runtime list element");
  if (mlir::failed(dead))
    return mlir::failure();
  // Box-stored contracts keep boxed placeholders too, so the word-select
  // machinery below stays shape-uniform.
  if (std::optional<RuntimeSymbol> box = manifest.primitive(
          runtimeContractName(elementContract), "box")) {
    mlir::func::CallOp boxed =
        RuntimeBundleLowerer::createRuntimeCall(loc, *box, dead->values);
    dead->values.assign(boxed.getResults().begin(), boxed.getResults().end());
  }
  if (elementShapes->size() != dead->values.size())
    return op.emitError()
           << "dead placeholder for " << elementContract
           << " does not match the contract's physical value count";

  mlir::Value items = iterator.physicalValues()[2];
  constexpr std::int64_t kWordsPerSlot = 16;
  constexpr std::int64_t kPointerBase = 4;
  constexpr std::int64_t kSizeBase = 9;
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, kWordsPerSlot, 64);
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
          .getResult();
  auto loadBoxWord = [&](std::int64_t wordIndex) -> mlir::Value {
    mlir::Value offset = mlir::arith::ConstantIntOp::create(
        builder, loc, wordIndex, 64);
    mlir::Value word =
        mlir::arith::AddIOp::create(builder, loc, base, offset).getResult();
    mlir::Value index = mlir::arith::IndexCastOp::create(
                            builder, loc, builder.getIndexType(), word)
                            .getResult();
    return mlir::memref::LoadOp::create(builder, loc, items, index)
        .getResult();
  };
  auto pointerWord = [&](mlir::Value value) -> mlir::Value {
    mlir::Value pointerIndex =
        mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                             value);
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getI64Type(), pointerIndex)
        .getResult();
  };
  auto sizeWord = [&](mlir::Value value) -> mlir::Value {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
    if (!memref || memref.getRank() != 1)
      return zero64;
    if (memref.hasStaticShape())
      return mlir::arith::ConstantIntOp::create(builder, loc,
                                                memref.getDimSize(0), 64);
    mlir::Value dim =
        mlir::memref::DimOp::create(builder, loc, value, 0).getResult();
    return mlir::arith::IndexCastOp::create(builder, loc,
                                            builder.getI64Type(), dim)
        .getResult();
  };

  llvm::SmallVector<mlir::Value, 4> elementValues;
  for (auto [index, deadValue] : llvm::enumerate(dead->values)) {
    mlir::Value realPointer =
        loadBoxWord(kPointerBase + static_cast<std::int64_t>(index));
    mlir::Value realSize =
        loadBoxWord(kSizeBase + static_cast<std::int64_t>(index));
    mlir::Value pointer =
        mlir::arith::SelectOp::create(builder, loc, valid, realPointer,
                                      pointerWord(deadValue))
            .getResult();
    mlir::Value size = mlir::arith::SelectOp::create(builder, loc, valid,
                                                     realSize,
                                                     sizeWord(deadValue))
                           .getResult();
    elementValues.push_back(memrefFromBoxWords(
        builder, loc, pointer, size,
        mlir::cast<mlir::MemRefType>((*elementShapes)[index])));
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, elementContract,
                                                   elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  bool valueSemantics = canonical->size() != elementValues.size() ||
                        !llvm::equal(*canonical, elementValues);
  RuntimeValue element{elementContract, *canonical,
                       ownership::logicalOwnershipKind(elementContract,
                                                       /*ownsObject=*/false)};
  if (valueSemantics) {
    // Unboxed elements are copied values; no ownership to root.
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element", element)))
      return mlir::failure();
  } else {
    std::optional<RuntimeValue> retained =
        RuntimeBundleLowerer::retainEvidenceElement(op, element);
    if (!retained)
      return op.emitError()
             << "iteration over a runtime-mode list of " << elementContract
             << " needs an own primitive in the runtime manifest";
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element",
                                              *retained)))
      return mlir::failure();
  }

  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(iterator.contractName(), "__len__");
  if (!lenMethod)
    return op.emitError()
           << "list iteration needs a runtime __len__ to pin the container";
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&iterator};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(op);
  RuntimeBundleLowerer::createRuntimeCall(loc, *lenMethod, lenOperands);

  op.getValid().replaceAllUsesWith(valid);
  valueBundles[op.getNext()] = iterator;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerNext(py::NextOp op) {
  const RuntimeBundle *iterator =
      RuntimeBundleLowerer::bundleFor(op.getIterator());
  if (!iterator)
    return op.emitError() << "next iterator has no lowered runtime bundle";
  if (iterator->evidenceIteratorCell) {
    if (iterator->sequenceElements.empty() && !iterator->sequenceEvidenceBacked)
      return RuntimeBundleLowerer::lowerListRuntimeNext(op, *iterator);
    return RuntimeBundleLowerer::lowerListEvidenceNext(op, *iterator);
  }
  if (iterator->contractName() == "types.GeneratorType" &&
      !iterator->generatorTarget.empty())
    return RuntimeBundleLowerer::lowerSourceGeneratorNext(op, *iterator);

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

mlir::FailureOr<RuntimePrimitiveI64Evidence>
RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
    mlir::Operation *op, mlir::Value value,
    llvm::ArrayRef<const RuntimeBundle *> frameSources,
    llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> &memo,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence) {
  auto cached = memo.find(value);
  if (cached != memo.end())
    return cached->second;

  if (runtimeContractName(value.getType()) != "builtins.int")
    return op->emitError()
           << "source generator next lowering currently supports only "
              "builtins.int yielded expressions";

  mlir::Operation *def = value.getDefiningOp();
  if (!def) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument)
      return op->emitError()
             << "source generator next lowering expected a defining op or "
                "frame argument for yielded int value";
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parent);
    if (!function || function.isDeclaration() ||
        argument.getOwner() != &function.getBody().front())
      return op->emitError()
             << "source generator next lowering does not support non-entry "
                "block argument frame values yet";
    unsigned index = argument.getArgNumber();
    if (index >= frameSources.size())
      return op->emitError() << "source generator frame argument " << index
                             << " has no captured source bundle";
    const RuntimeBundle *source = frameSources[index];
    if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(source))
      return op->emitError()
             << "source generator frame argument " << index
             << " currently requires primitive i64 int evidence";
    RuntimePrimitiveI64Evidence evidence = *source->primitiveI64;
    memo.insert({value, evidence});
    return evidence;
  }

  if (auto constant = mlir::dyn_cast<py::IntConstantOp>(def)) {
    std::int64_t parsed = 0;
    if (constant.getValue().getAsInteger(10, parsed))
      return constant.emitError()
             << "integer literal is outside the currently lowered i64 path";
    RuntimePrimitiveI64Evidence evidence{
        mlir::arith::ConstantIntOp::create(builder, constant.getLoc(), parsed,
                                           64)
            .getResult(),
        mlir::arith::ConstantIntOp::create(builder, constant.getLoc(), 1, 1)
            .getResult()};
    memo.insert({value, evidence});
    return evidence;
  }

  if (auto yield = mlir::dyn_cast<py::YieldValueOp>(def)) {
    if (yield.getSent() != value)
      return yield.emitError()
             << "source generator next lowering cannot materialize "
                "non-sent yield result";
    if (sentI64Evidence) {
      memo.insert({value, *sentI64Evidence});
      return *sentI64Evidence;
    }
    RuntimePrimitiveI64Evidence invalidEvidence{
        mlir::arith::ConstantIntOp::create(builder, yield.getLoc(), 0, 64)
            .getResult(),
        mlir::arith::ConstantIntOp::create(builder, yield.getLoc(), 0, 1)
            .getResult()};
    memo.insert({value, invalidEvidence});
    return invalidEvidence;
  }

  auto materializeBinary = [&](mlir::Value lhsValue, mlir::Value rhsValue,
                               mlir::FlatSymbolRefAttr targetAttr,
                               llvm::StringRef expectedName)
      -> mlir::FailureOr<RuntimePrimitiveI64Evidence> {
    mlir::FailureOr<llvm::StringRef> methodName =
        RuntimeBundleLowerer::requireMethodTarget(def, targetAttr,
                                                  expectedName);
    if (mlir::failed(methodName))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> lhs =
        RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
            op, lhsValue, frameSources, memo, sentI64Evidence);
    if (mlir::failed(lhs))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> rhs =
        RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
            op, rhsValue, frameSources, memo, sentI64Evidence);
    if (mlir::failed(rhs))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> result =
        RuntimeBundleLowerer::emitPrimitiveI64ArithmeticEvidence(
            def, *methodName, *lhs, *rhs);
    if (mlir::failed(result))
      return mlir::failure();
    memo.insert({value, *result});
    return *result;
  };

  if (auto add = mlir::dyn_cast<py::AddOp>(def))
    return materializeBinary(add.getLhs(), add.getRhs(), add.getTargetAttr(),
                             add.getMethodName());
  if (auto sub = mlir::dyn_cast<py::SubOp>(def))
    return materializeBinary(sub.getLhs(), sub.getRhs(), sub.getTargetAttr(),
                             sub.getMethodName());
  if (auto mul = mlir::dyn_cast<py::MulOp>(def))
    return materializeBinary(mul.getLhs(), mul.getRhs(), mul.getTargetAttr(),
                             mul.getMethodName());

  return def->emitError()
         << "source generator next lowering cannot materialize yielded int "
            "value produced by "
         << def->getName();
}

mlir::FailureOr<RuntimeBundleLowerer::SourceGeneratorResumeResult>
RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
    mlir::Operation *op, mlir::Type elementType, const RuntimeBundle &iterator,
    bool useCurrentInsertionPoint,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence) {
  mlir::func::FuncOp target =
      module.lookupSymbol<mlir::func::FuncOp>(iterator.generatorTarget);
  if (!target)
    return op->emitError() << "source generator target '"
                           << iterator.generatorTarget << "' is not defined";
  llvm::SmallVector<const RuntimeBundle *, 8> frameSources;
  frameSources.reserve(iterator.generatorSourceBundles.size());
  for (const std::shared_ptr<RuntimeBundle> &source :
       iterator.generatorSourceBundles) {
    if (!source)
      return op->emitError()
             << "source generator frame source bundle is missing";
    frameSources.push_back(source.get());
  }

  mlir::Block *entryBlock = nullptr;
  if (!target.isDeclaration())
    entryBlock = &target.getBody().front();
  if (!entryBlock)
    return op->emitError()
           << "source generator target must have a visible body";

  llvm::SmallVector<py::YieldValueOp, 4> yields;
  llvm::SmallVector<py::YieldFromOp, 4> yieldFroms;
  target.walk([&](py::YieldValueOp yield) { yields.push_back(yield); });
  target.walk(
      [&](py::YieldFromOp yieldFrom) { yieldFroms.push_back(yieldFrom); });

  if (runtimeContractName(elementType) != "builtins.int")
    return op->emitError()
           << "source generator next lowering currently supports int yields";
  const RuntimeBundle *delegatedSource = nullptr;
  const RuntimeBundle *delegatedIndexedIterable = nullptr;
  const RuntimeBundle *delegatedManifestIterator = nullptr;
  llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> delegatedIndexedElements;
  std::optional<RuntimeSymbol> delegatedManifestNext;
  mlir::func::FuncOp delegatedInlineTarget;
  llvm::SmallVector<mlir::Value, 4> delegatedInlineFrameSourceValues;
  llvm::SmallVector<mlir::Operation *, 8> delegatedExpressionOps;
  struct SourceYieldPlan {
    mlir::Value value;
    bool usesDelegatedFrameSources = false;
    llvm::SmallVector<mlir::Value, 4> delegatedFrameSourceValues;
  };
  if (!yieldFroms.empty()) {
    if (!yields.empty())
      return op->emitError()
             << "source generator yield from lowering does not yet support "
                "mixing direct yield and delegated yield points";
    if (yieldFroms.size() != 1)
      return op->emitError()
             << "source generator yield from lowering currently supports "
                "exactly one delegated source";

    py::YieldFromOp yieldFrom = yieldFroms.front();
    if (yieldFrom->getBlock() != entryBlock)
      return yieldFrom.emitError()
             << "source generator yield from lowering currently supports "
                "only straight-line delegation";
    auto elementAssignable = [&](mlir::Type sourceType) -> mlir::LogicalResult {
      if (sourceType && !py::isAssignableTo(sourceType, elementType, op))
        return yieldFrom.emitError()
               << "delegated iterable yields " << sourceType << ", expected "
               << elementType;
      return mlir::success();
    };

    auto rememberDelegatedExpression = [&](mlir::Operation *exprOp) {
      if (exprOp && !llvm::is_contained(delegatedExpressionOps, exprOp))
        delegatedExpressionOps.push_back(exprOp);
    };

    auto directFrameSourceFor =
        [&](mlir::Value value,
            llvm::StringRef label) -> mlir::FailureOr<const RuntimeBundle *> {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (!argument || argument.getOwner() != entryBlock)
        return yieldFrom.emitError()
               << label
               << " currently supports only direct frame source values";
      unsigned index = argument.getArgNumber();
      if (index >= frameSources.size())
        return yieldFrom.emitError() << label << " frame source " << index
                                     << " has no captured source bundle";
      const RuntimeBundle *source = frameSources[index];
      if (!source)
        return yieldFrom.emitError() << label << " frame source is missing";
      return source;
    };

    auto rememberCallPacks = [&](py::PackOp posargs, py::PackOp kwnames,
                                 py::PackOp kwvalues) {
      rememberDelegatedExpression(posargs);
      rememberDelegatedExpression(kwnames);
      rememberDelegatedExpression(kwvalues);
      for (py::PackOp pack : {posargs, kwnames, kwvalues}) {
        if (!pack)
          continue;
        for (mlir::Value value : pack.getValues())
          rememberDelegatedExpression(value.getDefiningOp());
      }
    };

    mlir::Value sourceValue = yieldFrom.getSource();
    if (mlir::isa<mlir::BlockArgument>(sourceValue)) {
      mlir::FailureOr<const RuntimeBundle *> source =
          directFrameSourceFor(sourceValue, "source generator yield from");
      if (mlir::failed(source))
        return mlir::failure();
      delegatedSource = *source;
    } else if (auto call = sourceValue.getDefiningOp<py::CallOp>()) {
      auto posargs = call.getPosargs().getDefiningOp<py::PackOp>();
      auto kwnames = call.getKwnames().getDefiningOp<py::PackOp>();
      auto kwvalues = call.getKwvalues().getDefiningOp<py::PackOp>();
      if (auto methodAttr =
              call->getAttrOfType<mlir::StringAttr>("ly.bound_method")) {
        llvm::StringRef methodName = methodAttr.getValue();
        if (methodName == "keys" || methodName == "values") {
          if (!posargs || !kwnames || !kwvalues)
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires static argument packs";
          if (!posargs.getValues().empty() || !kwnames.getValues().empty() ||
              !kwvalues.getValues().empty())
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "supports only zero-argument view calls";
          mlir::FailureOr<const RuntimeBundle *> receiver =
              directFrameSourceFor(call.getCallable(),
                                   "source generator yield from dict view");
          if (mlir::failed(receiver))
            return mlir::failure();
          if ((*receiver)->contractName() != "builtins.dict")
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires a dict frame source";
          auto viewContract = mlir::dyn_cast_if_present<py::ContractType>(
              call.getResult(0).getType());
          llvm::StringRef expectedView = methodName == "keys"
                                             ? "builtins.dict_keys"
                                             : "builtins.dict_values";
          if (!viewContract || viewContract.getContractName() != expectedView ||
              viewContract.getArguments().size() < 2)
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires a typed dict view result";
          mlir::Type projectedElement =
              methodName == "keys" ? viewContract.getArguments().front()
                                   : viewContract.getArguments()[1];
          if (mlir::failed(elementAssignable(projectedElement)))
            return mlir::failure();
          delegatedIndexedIterable = *receiver;
          delegatedIndexedElements = methodName == "keys"
                                         ? (*receiver)->mappingKeyBundles
                                         : (*receiver)->mappingValueBundles;
          if (delegatedIndexedElements.empty())
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires static dict element evidence";
          for (const std::shared_ptr<RuntimeBundle> &element :
               delegatedIndexedElements) {
            if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
              return yieldFrom.emitError()
                     << "source generator yield from dict view delegation "
                        "currently requires primitive int element evidence";
          }
          rememberCallPacks(posargs, kwnames, kwvalues);
          rememberDelegatedExpression(call);
        } else if (methodName == "items") {
          return yieldFrom.emitError()
                 << "source generator yield from dict item view delegation "
                    "currently requires tuple yield lowering";
        } else {
          return yieldFrom.emitError()
                 << "source generator yield from bound method delegation "
                    "currently supports only dict keys/values views";
        }
      } else {
        auto binding = call.getCallable().getDefiningOp<py::BindingRefOp>();
        if (!binding)
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "static source generator binding";
        mlir::func::FuncOp callTarget =
            module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
        if (!callTarget || !callTarget->hasAttr("ly.generator.body_result"))
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "source generator target";
        auto callableAttr =
            callTarget->getAttrOfType<mlir::TypeAttr>("callable_type");
        auto callableType = mlir::dyn_cast_if_present<py::CallableType>(
            callableAttr ? callableAttr.getValue() : mlir::Type());
        if (!callableType || callableType.hasVararg() ||
            callableType.hasKwarg() || !callableType.getKwOnlyTypes().empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "static positional callable signature";
        std::optional<StaticCallableInvocation> invocation =
            RuntimeBundleLowerer::collectStaticCallableInvocation(call);
        if (!invocation)
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "requires static argument packs";
        std::optional<CallableArgumentPlan> argumentPlan =
            RuntimeBundleLowerer::collectCallableArgumentPlan(
                call, callableType, /*emitErrors=*/true);
        if (!argumentPlan)
          return mlir::failure();
        if (!argumentPlan->defaultedFixed.empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "requires explicit frame arguments";
        if (!argumentPlan->varargActuals.empty() ||
            !argumentPlan->kwargActuals.empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "supports only fixed frame arguments";
        llvm::ArrayRef<mlir::Type> positionalTypes =
            callableType.getPositionalTypes();
        if (argumentPlan->fixedActuals.size() != positionalTypes.size())
          return yieldFrom.emitError() << "source generator yield from call "
                                          "delegation argument count "
                                          "does not match target frame";
        for (auto [index, inputType] : llvm::enumerate(positionalTypes)) {
          std::optional<unsigned> actualIndex =
              argumentPlan->fixedActuals[index];
          if (!actualIndex || *actualIndex >= invocation->actualValues.size())
            return yieldFrom.emitError()
                   << "source generator yield from call delegation argument "
                      "planner produced an invalid frame source";
          mlir::Value argValue = invocation->actualValues[*actualIndex];
          if (!py::isAssignableTo(argValue.getType(), inputType, op))
            return yieldFrom.emitError()
                   << "source generator yield from call argument "
                   << argValue.getType() << " is not assignable to frame input "
                   << inputType;
          delegatedInlineFrameSourceValues.push_back(argValue);
        }
        llvm::SmallVector<mlir::Type, 4> closureTypes =
            RuntimeBundleLowerer::callableClosureTypes(callTarget);
        if (closureTypes.size() != binding.getCaptures().size())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation binding "
                    "captures "
                 << binding.getCaptures().size()
                 << " values, but delegated target declares "
                 << closureTypes.size() << " closure inputs";
        for (auto [index, capture] : llvm::enumerate(binding.getCaptures())) {
          mlir::Type expected = closureTypes[index];
          if (!py::isAssignableTo(capture.getType(), expected, op))
            return yieldFrom.emitError()
                   << "source generator yield from call delegation capture "
                   << index << " has type " << capture.getType()
                   << ", expected " << expected;
          delegatedInlineFrameSourceValues.push_back(capture);
        }
        if (auto generatorContract =
                mlir::dyn_cast_if_present<py::ContractType>(
                    call.getResult(0).getType())) {
          llvm::ArrayRef<mlir::Type> args = generatorContract.getArguments();
          if (!args.empty() && mlir::failed(elementAssignable(args.front())))
            return mlir::failure();
        }
        delegatedInlineTarget = callTarget;
        rememberDelegatedExpression(binding);
        rememberCallPacks(posargs, kwnames, kwvalues);
        rememberDelegatedExpression(call);
      }
    } else {
      return yieldFrom.emitError()
             << "source generator yield from currently supports direct frame "
                "source or source generator call delegation";
    }

    if (delegatedSource &&
        delegatedSource->contractName() == "types.GeneratorType" &&
        !delegatedSource->generatorTarget.empty()) {
      if (auto generatorContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedSource->contract)) {
        llvm::ArrayRef<mlir::Type> args = generatorContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
    } else if (delegatedSource &&
               (delegatedSource->contractName() == "builtins.list" ||
                delegatedSource->contractName() == "builtins.tuple") &&
               !delegatedSource->sequenceElementBundles.empty()) {
      delegatedIndexedIterable = delegatedSource;
      delegatedIndexedElements = delegatedSource->sequenceElementBundles;
      delegatedSource = nullptr;
      if (auto sequenceContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedIndexedIterable->contract)) {
        llvm::ArrayRef<mlir::Type> args = sequenceContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
      for (const std::shared_ptr<RuntimeBundle> &element :
           delegatedIndexedElements) {
        if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
          return yieldFrom.emitError()
                 << "source generator yield from static sequence currently "
                    "requires primitive int element evidence";
      }
    } else if (delegatedSource &&
               delegatedSource->contractName() == "builtins.dict" &&
               !delegatedSource->mappingKeyBundles.empty()) {
      delegatedIndexedIterable = delegatedSource;
      delegatedIndexedElements = delegatedSource->mappingKeyBundles;
      delegatedSource = nullptr;
      if (auto dictContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedIndexedIterable->contract)) {
        llvm::ArrayRef<mlir::Type> args = dictContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
      for (const std::shared_ptr<RuntimeBundle> &key :
           delegatedIndexedElements) {
        if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(key.get()))
          return yieldFrom.emitError()
                 << "source generator yield from static dict currently "
                    "requires primitive int key evidence";
      }
    } else if (delegatedSource) {
      llvm::SmallVector<const RuntimeBundle *, 1> nextSources{delegatedSource};
      mlir::FailureOr<RuntimeSymbol> next =
          RuntimeBundleLowerer::selectManifestMethod(
              yieldFrom, *delegatedSource, "__next__", nextSources,
              /*allowUnusedSources=*/false);
      if (mlir::failed(next))
        return mlir::failure();
      if (!next->validResultIndex)
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ valid_result_index evidence";
      if (next->nextEvidence != "receiver")
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ ly.runtime.next_evidence = \"receiver\"";
      if (next->elementContract.empty())
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ element_contract evidence";
      if (mlir::failed(elementAssignable(
              runtimeContractType(context, next->elementContract))))
        return mlir::failure();
      if (next->elementContract != "builtins.int")
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator currently "
                  "requires builtins.int element evidence";
      delegatedManifestIterator = delegatedSource;
      delegatedManifestNext = *next;
      delegatedSource = nullptr;
    } else if (!delegatedIndexedIterable && !delegatedInlineTarget &&
               !delegatedManifestIterator) {
      return yieldFrom.emitError()
             << "source generator yield from currently supports only source "
                "generator, static sequence, static dict key, or self-mutating "
                "manifest iterator values";
    }
    if (delegatedSource) {
      auto generatorContract = mlir::dyn_cast_if_present<py::ContractType>(
          delegatedSource->contract);
      llvm::ArrayRef<mlir::Type> args = generatorContract
                                            ? generatorContract.getArguments()
                                            : llvm::ArrayRef<mlir::Type>();
      if (!args.empty() && !py::isAssignableTo(args.front(), elementType, op))
        return yieldFrom.emitError()
               << "delegated generator yields " << args.front() << ", expected "
               << elementType;
    }
  } else if (yields.empty()) {
    return op->emitError()
           << "source generator next lowering currently requires at least one "
              "yield";
  }

  llvm::SmallPtrSet<mlir::Operation *, 8> allowedDelegationOps;
  for (mlir::Operation *exprOp : delegatedExpressionOps)
    allowedDelegationOps.insert(exprOp);
  for (mlir::Operation &bodyOp : *entryBlock) {
    if (allowedDelegationOps.contains(&bodyOp))
      continue;
    if (mlir::isa<py::IntConstantOp, py::YieldValueOp, py::YieldFromOp,
                  py::NoneOp, py::AddOp, py::SubOp, py::MulOp,
                  mlir::func::ReturnOp>(bodyOp))
      continue;
    return bodyOp.emitError()
           << "source generator next lowering currently supports only "
              "straight-line pure int yield bodies";
  }

  auto collectStraightLineIntYieldValues = [&](mlir::func::FuncOp yieldTarget,
                                               llvm::StringRef label)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    mlir::Block *yieldEntry = nullptr;
    if (!yieldTarget.isDeclaration())
      yieldEntry = &yieldTarget.getBody().front();
    if (!yieldEntry)
      return op->emitError() << label << " target must have a visible body";

    llvm::SmallVector<py::YieldValueOp, 4> sourceYields;
    yieldTarget.walk(
        [&](py::YieldValueOp yield) { sourceYields.push_back(yield); });
    if (sourceYields.empty())
      return op->emitError()
             << label << " currently requires at least one yield";

    for (mlir::Operation &bodyOp : *yieldEntry) {
      if (mlir::isa<py::IntConstantOp, py::YieldValueOp, py::NoneOp, py::AddOp,
                    py::SubOp, py::MulOp, mlir::func::ReturnOp>(bodyOp))
        continue;
      return bodyOp.emitError()
             << label
             << " currently supports only straight-line pure int yield bodies";
    }

    llvm::SmallVector<mlir::Value, 4> values;
    values.reserve(sourceYields.size());
    for (py::YieldValueOp yield : sourceYields) {
      if (yield->getBlock() != yieldEntry)
        return yield.emitError()
               << label
               << " currently supports only straight-line yield points";
      if (!yield.getSent().use_empty())
        return yield.emitError()
               << label << " does not support sent values yet";
      if (runtimeContractName(yield.getValue().getType()) != "builtins.int")
        return yield.emitError() << label << " currently supports int yields";
      values.push_back(yield.getValue());
    }
    return values;
  };

  llvm::SmallVector<SourceYieldPlan, 4> yieldPlans;
  if (delegatedInlineTarget) {
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> inlineYields =
        collectStraightLineIntYieldValues(
            delegatedInlineTarget,
            "source generator yield from call delegation");
    if (mlir::failed(inlineYields))
      return mlir::failure();
    yieldPlans.reserve(inlineYields->size());
    for (mlir::Value yieldValue : *inlineYields) {
      SourceYieldPlan plan;
      plan.value = yieldValue;
      plan.usesDelegatedFrameSources = true;
      plan.delegatedFrameSourceValues = delegatedInlineFrameSourceValues;
      yieldPlans.push_back(std::move(plan));
    }
  } else {
    yieldPlans.reserve(yields.size());
    for (py::YieldValueOp yield : yields) {
      if (yield->getBlock() != entryBlock)
        return yield.emitError()
               << "source generator next lowering currently supports only "
                  "straight-line yield points";
      if (!yield.getSent().use_empty() &&
          runtimeContractName(yield.getSent().getType()) != "builtins.int")
        return yield.emitError()
               << "source generator next lowering currently supports only int "
                  "sent values";
      if (runtimeContractName(yield.getValue().getType()) != "builtins.int")
        return yield.emitError() << "source generator next lowering currently "
                                    "supports int yields";
      SourceYieldPlan plan;
      plan.value = yield.getValue();
      yieldPlans.push_back(std::move(plan));
    }
  }

  std::optional<RuntimeSymbol> resumeBegin =
      manifest.primitive("types.GeneratorType", "resume.begin");
  std::optional<RuntimeSymbol> resumeSuspend =
      manifest.primitive("types.GeneratorType", "resume.suspend");
  std::optional<RuntimeSymbol> resumeComplete =
      manifest.primitive("types.GeneratorType", "resume.complete");
  if (!resumeBegin || !resumeSuspend || !resumeComplete)
    return op->emitError()
           << "runtime manifest has no generator resume primitive for "
              "types.GeneratorType";

  llvm::SmallVector<const RuntimeBundle *, 1> generatorSource{&iterator};
  if (!useCurrentInsertionPoint)
    builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> resumeBeginOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *resumeBegin, generatorSource,
                                            resumeBeginOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp resumeBeginCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *resumeBegin, resumeBeginOperands);
  if (resumeBeginCall.getNumResults() != 1 ||
      !resumeBeginCall.getResult(0).getType().isInteger(1))
    return resumeBegin->function.emitError()
           << "generator resume.begin primitive must return one i1";

  llvm::ArrayRef<mlir::Value> generatorValues = iterator.physicalValues();
  if (generatorValues.size() != 1)
    return op->emitError() << "types.GeneratorType bundle must contain one "
                              "storage value";
  mlir::Value storage = generatorValues.front();
  auto storageType = mlir::dyn_cast<mlir::MemRefType>(storage.getType());
  if (!storageType || storageType.getRank() != 1 ||
      storageType.getElementType() != builder.getI64Type())
    return op->emitError() << "types.GeneratorType storage has invalid type "
                           << storage.getType();

  mlir::Value resumeSlot =
      mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 4);
  mlir::Value resumeIndex =
      mlir::memref::LoadOp::create(builder, op->getLoc(), storage, resumeSlot);
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1);
  llvm::SmallVector<mlir::Type, 3> resultTypes{
      builder.getI64Type(), builder.getI1Type(), builder.getI1Type()};

  auto emitSuspendAndYieldEvidence =
      [&](mlir::Value yieldedValue, mlir::Value yieldedValid,
          unsigned nextResumeIndexValue) -> mlir::LogicalResult {
    mlir::Value nextResumeIndex = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(), static_cast<std::int64_t>(nextResumeIndexValue),
        64);
    llvm::SmallVector<mlir::Value, 4> suspendOperands;
    unsigned suspendInputIndex = 0;
    if (mlir::failed(appendRuntimeSource(
            op, *resumeSuspend, resumeSuspend->function.getFunctionType(),
            suspendInputIndex, iterator, suspendOperands)))
      return mlir::failure();
    if (suspendInputIndex != 1 ||
        resumeSuspend->function.getFunctionType().getNumInputs() != 2 ||
        !resumeSuspend->function.getFunctionType().getInput(1).isInteger(64))
      return resumeSuspend->function.emitError()
             << "generator resume.suspend primitive must take storage and one "
                "i64 resume index";
    suspendOperands.push_back(nextResumeIndex);
    RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeSuspend,
                                            suspendOperands);
    mlir::scf::YieldOp::create(
        builder, op->getLoc(),
        mlir::ValueRange{yieldedValue, yieldedValid, trueValue});
    return mlir::success();
  };

  auto emitInvalidYield = [&]() -> mlir::LogicalResult {
    mlir::Value zeroValue =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 0, 64);
    mlir::Value falseValue =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 0, 1);
    mlir::scf::YieldOp::create(
        builder, op->getLoc(),
        mlir::ValueRange{zeroValue, falseValue, falseValue});
    return mlir::success();
  };

  auto emitSuspendAndYield =
      [&](const SourceYieldPlan &yieldPlan,
          unsigned nextResumeIndexValue) -> mlir::LogicalResult {
    llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> memo;
    mlir::FailureOr<RuntimePrimitiveI64Evidence> yieldedEvidence =
        mlir::failure();
    if (yieldPlan.usesDelegatedFrameSources) {
      llvm::SmallVector<RuntimeBundle, 4> delegatedFrameStorage;
      delegatedFrameStorage.reserve(
          yieldPlan.delegatedFrameSourceValues.size());
      llvm::SmallVector<const RuntimeBundle *, 4> delegatedFrameSources;
      delegatedFrameSources.reserve(
          yieldPlan.delegatedFrameSourceValues.size());
      for (mlir::Value sourceValue : yieldPlan.delegatedFrameSourceValues) {
        if (runtimeContractName(sourceValue.getType()) != "builtins.int") {
          delegatedFrameSources.push_back(nullptr);
          continue;
        }
        mlir::FailureOr<RuntimePrimitiveI64Evidence> sourceEvidence =
            RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
                op, sourceValue, frameSources, memo, sentI64Evidence);
        if (mlir::failed(sourceEvidence))
          return mlir::failure();
        RuntimeBundle sourceBundle;
        if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
                op, sourceValue.getType(), sourceEvidence->value,
                sourceEvidence->valid, sourceBundle)))
          return mlir::failure();
        delegatedFrameStorage.push_back(std::move(sourceBundle));
        delegatedFrameSources.push_back(&delegatedFrameStorage.back());
      }
      llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> delegatedMemo;
      yieldedEvidence =
          RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
              op, yieldPlan.value, delegatedFrameSources, delegatedMemo);
    } else {
      yieldedEvidence =
          RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
              op, yieldPlan.value, frameSources, memo, sentI64Evidence);
    }
    if (mlir::failed(yieldedEvidence))
      return mlir::failure();
    return emitSuspendAndYieldEvidence(
        yieldedEvidence->value, yieldedEvidence->valid, nextResumeIndexValue);
  };

  auto emitCompleteAndInvalidYield = [&]() -> mlir::LogicalResult {
    auto completeIf = mlir::scf::IfOp::create(builder, op->getLoc(),
                                              resumeBeginCall.getResult(0),
                                              /*withElseRegion=*/false);
    mlir::Block &completeThen = completeIf.getThenRegion().front();
    builder.setInsertionPoint(completeThen.getTerminator());
    llvm::SmallVector<mlir::Value, 4> completeOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *resumeComplete,
                                              generatorSource, completeOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeComplete,
                                            completeOperands);

    builder.setInsertionPointAfter(completeIf);
    return emitInvalidYield();
  };

  auto emitManifestIteratorNextI64Evidence =
      [&]() -> mlir::FailureOr<RuntimePrimitiveI64Evidence> {
    if (!delegatedManifestIterator || !delegatedManifestNext)
      return op->emitError()
             << "source generator yield from manifest iterator state is "
                "missing";

    const RuntimeSymbol &next = *delegatedManifestNext;
    llvm::SmallVector<const RuntimeBundle *, 1> nextSources{
        delegatedManifestIterator};
    llvm::SmallVector<mlir::Value, 8> operands;
    if (mlir::failed(buildRuntimeCallOperands(op, next, nextSources, operands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), next, operands);
    unsigned validIndex = *next.validResultIndex;
    if (validIndex >= call.getNumResults())
      return op->emitError()
             << "runtime __next__ valid_result_index is outside the result "
                "list";
    mlir::Value valid = call.getResult(validIndex);
    if (!valid.getType().isInteger(1))
      return op->emitError() << "runtime __next__ valid result must be an i1";

    llvm::SmallVector<mlir::Value, 4> elementValues;
    for (unsigned index = 0; index < validIndex; ++index)
      elementValues.push_back(call.getResult(index));
    mlir::Type intType = runtimeContractType(context, "builtins.int");
    RuntimeBundle element = RuntimeBundle::object(intType, elementValues);

    std::optional<RuntimeSymbol> unbox =
        manifest.primitive("builtins.int", "unbox.i64");
    if (!unbox)
      return op->emitError()
             << "runtime manifest has no builtins.int.unbox.i64 primitive";
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op->getLoc(), *unbox, element.physicalValues());
    if (unboxCall.getNumResults() != 1 ||
        !unboxCall.getResult(0).getType().isInteger(64))
      return unbox->function.emitError()
             << "builtins.int.unbox.i64 primitive must return one i64";

    llvm::SmallVector<mlir::Value, 4> nextValues;
    for (unsigned index = validIndex + 1; index < call.getNumResults();
         ++index)
      nextValues.push_back(call.getResult(index));
    RuntimeBundle nextState = RuntimeBundle::object(
        runtimeContractType(context, next.nextContract), nextValues);
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, nextState, "source generator yield from next state")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, element, "source generator yield from yielded element")))
      return mlir::failure();

    return RuntimePrimitiveI64Evidence{unboxCall.getResult(0), valid};
  };

  if (delegatedSource) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    auto delegatedResumeInfo =
        generatorResumeClones.find(delegatedSource->generatorTarget);
    mlir::FailureOr<SourceGeneratorResumeResult> delegatedOr =
        delegatedResumeInfo != generatorResumeClones.end()
            ? RuntimeBundleLowerer::emitStateMachineGeneratorResume(
                  op, *delegatedSource, delegatedResumeInfo->second,
                  /*useCurrentInsertionPoint=*/true)
            : RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
                  op, elementType, *delegatedSource,
                  /*useCurrentInsertionPoint=*/true);
    if (mlir::failed(delegatedOr))
      return mlir::failure();
    SourceGeneratorResumeResult delegated = *delegatedOr;
    auto hasValue = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                            delegated.hasValue,
                                            /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&hasValue.getThenRegion().front());
    if (mlir::failed(emitSuspendAndYieldEvidence(delegated.value,
                                                 delegated.valid,
                                                 /*nextResumeIndexValue=*/0)))
      return mlir::failure();

    builder.setInsertionPointToStart(&hasValue.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasValue);
    mlir::scf::YieldOp::create(builder, op->getLoc(), hasValue.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  if (delegatedIndexedIterable) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    mlir::FailureOr<mlir::Value> runtimeLength = loadCollectionLength(
        op, builder, *delegatedIndexedIterable, "source generator yield from");
    if (mlir::failed(runtimeLength))
      return mlir::failure();
    mlir::Location loc = op->getLoc();
    mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
    mlir::Value staticLength = mlir::arith::ConstantIntOp::create(
        builder, loc,
        static_cast<std::int64_t>(delegatedIndexedElements.size()), 64);
    mlir::Value nonNegative = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sge, resumeIndex, zero);
    mlir::Value belowRuntimeLength = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::slt, resumeIndex,
        *runtimeLength);
    mlir::Value belowStaticLength = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::slt, resumeIndex,
        staticLength);
    mlir::Value inRuntimeRange = mlir::arith::AndIOp::create(
        builder, loc, nonNegative, belowRuntimeLength);
    mlir::Value canYield = mlir::arith::AndIOp::create(
        builder, loc, inRuntimeRange, belowStaticLength);
    auto hasElement =
        mlir::scf::IfOp::create(builder, loc, resultTypes, canYield,
                                /*withElseRegion=*/true);

    auto emitSequenceYieldDispatch =
        [&](auto &&self,
            unsigned elementIndex) -> mlir::FailureOr<mlir::scf::IfOp> {
      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, loc, static_cast<std::int64_t>(elementIndex), 64);
      mlir::Value indexMatches = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, resumeIndex, expected);
      auto match =
          mlir::scf::IfOp::create(builder, loc, resultTypes, indexMatches,
                                  /*withElseRegion=*/true);

      builder.setInsertionPointToStart(&match.getThenRegion().front());
      const std::shared_ptr<RuntimeBundle> &element =
          delegatedIndexedElements[elementIndex];
      if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
        return match.emitError()
               << "source generator yield from static indexed iterable "
                  "currently requires primitive int element evidence";
      if (mlir::failed(emitSuspendAndYieldEvidence(element->primitiveI64->value,
                                                   element->primitiveI64->valid,
                                                   elementIndex + 1)))
        return mlir::failure();

      builder.setInsertionPointToStart(&match.getElseRegion().front());
      if (elementIndex + 1 < delegatedIndexedElements.size()) {
        mlir::FailureOr<mlir::scf::IfOp> nested = self(self, elementIndex + 1);
        if (mlir::failed(nested))
          return mlir::failure();
        builder.setInsertionPointAfter(*nested);
        mlir::scf::YieldOp::create(builder, loc, nested->getResults());
      } else if (mlir::failed(emitCompleteAndInvalidYield())) {
        return mlir::failure();
      }

      builder.setInsertionPointAfter(match);
      return match;
    };

    builder.setInsertionPointToStart(&hasElement.getThenRegion().front());
    mlir::FailureOr<mlir::scf::IfOp> selected =
        emitSequenceYieldDispatch(emitSequenceYieldDispatch, 0);
    if (mlir::failed(selected))
      return mlir::failure();
    builder.setInsertionPointAfter(*selected);
    mlir::scf::YieldOp::create(builder, loc, selected->getResults());

    builder.setInsertionPointToStart(&hasElement.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasElement);
    mlir::scf::YieldOp::create(builder, loc, hasElement.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  if (delegatedManifestIterator) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    mlir::FailureOr<RuntimePrimitiveI64Evidence> next =
        emitManifestIteratorNextI64Evidence();
    if (mlir::failed(next))
      return mlir::failure();
    auto hasValue = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                            next->valid,
                                            /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&hasValue.getThenRegion().front());
    if (mlir::failed(emitSuspendAndYieldEvidence(
            next->value, next->valid, /*nextResumeIndexValue=*/0)))
      return mlir::failure();

    builder.setInsertionPointToStart(&hasValue.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasValue);
    mlir::scf::YieldOp::create(builder, op->getLoc(), hasValue.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  auto emitYieldDispatch =
      [&](auto &&self,
          unsigned yieldIndex) -> mlir::FailureOr<mlir::scf::IfOp> {
    mlir::Value indexValue = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(), static_cast<std::int64_t>(yieldIndex), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, resumeIndex,
        indexValue);
    mlir::Value canYield = mlir::arith::AndIOp::create(
        builder, op->getLoc(), resumeBeginCall.getResult(0), indexMatches);
    auto branch =
        mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes, canYield,
                                /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    if (mlir::failed(
            emitSuspendAndYield(yieldPlans[yieldIndex], yieldIndex + 1)))
      return mlir::failure();

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (yieldIndex + 1 < yieldPlans.size()) {
      mlir::FailureOr<mlir::scf::IfOp> nested = self(self, yieldIndex + 1);
      if (mlir::failed(nested))
        return mlir::failure();
      builder.setInsertionPointAfter(*nested);
      mlir::scf::YieldOp::create(builder, op->getLoc(), nested->getResults());
    } else if (mlir::failed(emitCompleteAndInvalidYield())) {
      return mlir::failure();
    }

    builder.setInsertionPointAfter(branch);
    return branch;
  };

  mlir::FailureOr<mlir::scf::IfOp> yieldedOr =
      emitYieldDispatch(emitYieldDispatch, 0);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  mlir::scf::IfOp yielded = *yieldedOr;

  builder.setInsertionPointAfter(yielded);
  return SourceGeneratorResumeResult{yielded.getResult(0), yielded.getResult(1),
                                     yielded.getResult(2)};
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerSourceGeneratorNext(py::NextOp op,
                                               const RuntimeBundle &iterator) {
  auto resumeInfo = generatorResumeClones.find(iterator.generatorTarget);
  mlir::FailureOr<SourceGeneratorResumeResult> yieldedOr =
      resumeInfo != generatorResumeClones.end()
          ? RuntimeBundleLowerer::emitStateMachineGeneratorResume(
                op.getOperation(), iterator, resumeInfo->second)
          : RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
                op.getOperation(), op.getElement().getType(), iterator);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  SourceGeneratorResumeResult yielded = *yieldedOr;

  if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(
          op->getBlock()->getTerminator())) {
    if (llvm::is_contained(condition.getArgs(), op.getElement())) {
      auto whileOp = op->getParentOfType<mlir::scf::WhileOp>();
      if (!whileOp || condition->getParentOp() != whileOp)
        return op.emitError()
               << "source generator for-loop lowering expected enclosing "
                  "scf.while";
      if (!whileOp.getInits().empty())
        return op.emitError()
               << "source generator for-loop lowering does not support "
                  "loop-carried iterator values yet";
      if (whileOp.getResults().size() != 1 || condition.getArgs().size() != 1 ||
          condition.getArgs().front() != op.getElement() ||
          whileOp.getAfter().front().getNumArguments() != 1)
        return op.emitError()
               << "source generator for-loop lowering expected one yielded "
                  "loop value";
      if (!whileOp.getResult(0).use_empty())
        return op.emitError()
               << "source generator for-loop lowering does not support users "
                  "of the scf.while result yet";

      mlir::BlockArgument bodyElement =
          whileOp.getAfter().front().getArgument(0);
      whileOp.getResult(0).setType(builder.getI64Type());
      bodyElement.setType(builder.getI64Type());
      condition.getConditionMutable().set(yielded.hasValue);
      condition.getArgsMutable().assign(mlir::ValueRange{yielded.value});

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(&whileOp.getAfter().front());
      mlir::Value bodyValid =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
      RuntimeBundle bodyElementBundle;
      if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
              op, op.getElement().getType(), bodyElement, bodyValid,
              bodyElementBundle)))
        return mlir::failure();
      valueBundles[bodyElement] = std::move(bodyElementBundle);
    }
  }
  if (mlir::Block *block = op->getBlock()) {
    for (mlir::Operation &candidate : *block) {
      for (mlir::OpOperand &operand : candidate.getOpOperands()) {
        if (operand.get() == op.getValid())
          operand.set(yielded.hasValue);
      }
    }
  }
  llvm::SmallVector<mlir::OpOperand *, 4> validUses;
  for (mlir::OpOperand &use : op.getValid().getUses())
    validUses.push_back(&use);
  for (mlir::OpOperand *use : validUses)
    use->set(yielded.hasValue);

  RuntimeBundle element;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getElement().getType(), yielded.value, yielded.valid,
          element)))
    return mlir::failure();
  valueBundles[op.getElement()] = std::move(element);

  RuntimeBundle next = iterator;
  next.setObjectLogicalOwnership(/*ownsObject=*/false);
  valueBundles[op.getNext()] = std::move(next);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorSend(
    py::CallOp op, const RuntimeBundle &receiver,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError() << "source generator send expects exactly one value";
  if (op.getNumResults() != 1 ||
      runtimeContractName(op.getResult(0).getType()) != "builtins.int")
    return op.emitError()
           << "source generator send currently supports int yield results";

  std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence;
  bool releaseIgnoredSentObject = false;
  if (runtimeContractName(sources[1]->contract) == "builtins.int") {
    if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]))
      return op.emitError()
             << "source generator send(int) requires primitive int evidence";
    sentI64Evidence = *sources[1]->primitiveI64;
  } else if (runtimeContractName(sources[1]->contract) != "types.NoneType") {
    releaseIgnoredSentObject = true;
  }

  auto sendResumeInfo = generatorResumeClones.find(receiver.generatorTarget);
  mlir::FailureOr<SourceGeneratorResumeResult> yieldedOr =
      sendResumeInfo != generatorResumeClones.end()
          ? RuntimeBundleLowerer::emitStateMachineGeneratorResume(
                op.getOperation(), receiver, sendResumeInfo->second)
          : RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
                op.getOperation(), op.getResult(0).getType(), receiver,
                /*useCurrentInsertionPoint=*/false, sentI64Evidence);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  SourceGeneratorResumeResult yielded = *yieldedOr;

  if (releaseIgnoredSentObject &&
      mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
          op, *sources[1], "source generator send ignored value")))
    return mlir::failure();

  mlir::Location loc = op.getLoc();
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  mlir::Value exhausted =
      mlir::arith::XOrIOp::create(builder, loc, yielded.hasValue, trueValue);
  auto stopIf = mlir::scf::IfOp::create(builder, loc, exhausted,
                                        /*withElseRegion=*/false);
  builder.setInsertionPoint(stopIf.getThenRegion().front().getTerminator());
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.StopIteration", "generator exhausted")))
    return mlir::failure();
  builder.setInsertionPointAfter(stopIf);

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult(0).getType(), yielded.value, yielded.valid, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorThrow(
    py::CallOp op, const RuntimeBundle &receiver,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError()
           << "source generator throw expects exactly one exception value";
  if (op.getNumResults() != 1 ||
      runtimeContractName(op.getResult(0).getType()) != "builtins.int")
    return op.emitError()
           << "source generator throw currently supports int yield results";
  const RuntimeBundle &exception = *sources[1];
  if (!manifest.primitive(exception.contractName(), "raise"))
    return op.emitError() << "source generator throw exception type "
                          << exception.contractName()
                          << " has no raise primitive";

  llvm::SmallVector<const RuntimeBundle *, 1> closeSources{&receiver};
  if (mlir::failed(lowerManifestVoidMethod(op, receiver, "close", closeSources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::emitRaiseExceptionBundle(
          op.getOperation(), exception, /*discardCurrentException=*/true)))
    return mlir::failure();

  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
  mlir::Value invalid =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 1);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult(0).getType(), zero, invalid, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
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
