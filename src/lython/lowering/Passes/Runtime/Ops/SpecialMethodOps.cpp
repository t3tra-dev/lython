#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/CollectionPayload.h"
#include "Runtime/ABI/BoxLayout.h"
#include "Runtime/Evidence/Callable.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace py::lowering {
namespace {

using callable_evidence::integerLiteralFromValue;
using collection_abi::loadCollectionLength;
using collection_abi::touchCollectionEvidenceUse;

bool isEvidenceCollection(llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple" ||
         contract == "builtins.dict";
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

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::pinProbeOperandLiveness(
    mlir::Operation *op, const RuntimeBundle &payload) {
  if (payload.objectValue.ownership == ownership::OwnershipKind::Own)
    return RuntimeBundleLowerer::releaseAggregateSlot(op, payload,
                                                      "probe.operand");
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(payload);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return mlir::success();
  for (llvm::StringRef candidate :
       {llvm::StringRef("__len__"), llvm::StringRef("__hash__"),
        llvm::StringRef("__int__"), llvm::StringRef("__bool__")}) {
    for (RuntimeSymbol symbol :
         manifest.methodCandidates(concrete->contractName(), candidate)) {
      mlir::FunctionType type = symbol.function.getFunctionType();
      llvm::ArrayRef<mlir::Value> physicals = concrete->physicalValues();
      if (type.getNumInputs() != physicals.size())
        continue;
      bool matches = true;
      for (auto [input, physical] : llvm::zip(type.getInputs(), physicals))
        if (physical.getType() != input) {
          matches = false;
          break;
        }
      if (!matches)
        continue;
      RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), symbol,
          llvm::SmallVector<mlir::Value, 4>(physicals.begin(),
                                            physicals.end()));
      return mlir::success();
    }
  }
  // No conforming neutral use: borrowed payloads reaching here are entry
  // arguments or rooted evidence, both of which outlive the statement.
  return mlir::success();
}

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
    // the (possibly reallocated) representation. Keys may be any hashable
    // class — the probe hashes the transient box and raises TypeError for
    // unhashable ones.
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
      mlir::MemRefType boxType = box_abi::boxWordsType(builder);
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
      container.contractName() == "builtins.dict" &&
      !container.mappingEvidenceBacked && container.mappingKeys.empty() &&
      container.physicalValues().size() >= 5 &&
      index.kind == RuntimeBundle::Kind::Object) {
    // Runtime-mode dict delete: hash probe with a BORROWED transient key
    // box; the runtime raises KeyError (with the key's repr) on a miss and
    // compacts the dense entries in place, so the container's SSA
    // representation is unchanged.
    std::optional<RuntimeSymbol> delItemBox =
        manifest.primitive("builtins.dict", "delitem_box");
    if (!delItemBox)
      return op.emitError()
             << "runtime manifest has no dict delitem_box primitive";
    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::MemRefType boxType = box_abi::boxWordsType(builder);
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
    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(box);
    RuntimeBundleLowerer::createRuntimeCall(loc, *delItemBox, operands);
    if (mlir::failed(RuntimeBundleLowerer::pinProbeOperandLiveness(op,
                                                                   *payload)))
      return mlir::failure();
    for (mlir::Value result : op->getResults())
      valueBundles[result] = container;
    erase.push_back(op);
    return mlir::success();
  }
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
  bool runtimeSetProbe = container.kind == RuntimeBundle::Kind::Object &&
                         container.contractName() == "builtins.set" &&
                         container.physicalValues().size() >= 3;
  bool runtimeDictProbe = container.kind == RuntimeBundle::Kind::Object &&
                          container.contractName() == "builtins.dict" &&
                          !container.mappingEvidenceBacked &&
                          container.mappingKeys.empty() &&
                          container.physicalValues().size() >= 5;
  // Membership probe with a BORROWED transient element box (hash-based for
  // set/dict, identity-or-equality scan for list/tuple), then pin both the
  // container and the probed item past the call (the box holds raw pointer
  // words the liveness cannot see).
  auto emitBoxProbe = [&]() -> mlir::LogicalResult {
    std::optional<RuntimeSymbol> containsBox =
        manifest.primitive(container.contractName(), "contains_box");
    if (!containsBox)
      return op.emitError() << "runtime manifest has no "
                            << container.contractName()
                            << " contains_box primitive";
    builder.setInsertionPoint(op);
    mlir::Location loc = op.getLoc();
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, item);
    if (mlir::failed(payload))
      return mlir::failure();
    mlir::MemRefType boxType = box_abi::boxWordsType(builder);
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
    llvm::SmallVector<mlir::Value, 8> operands(
        container.physicalValues().begin(), container.physicalValues().end());
    operands.push_back(box);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(loc, *containsBox, operands);
    if (mlir::failed(RuntimeBundleLowerer::pinProbeOperandLiveness(op,
                                                                   *payload)))
      return mlir::failure();
    auto pinObject = [&](const RuntimeBundle &object,
                         llvm::StringRef pinMethod) -> mlir::LogicalResult {
      std::optional<RuntimeSymbol> method =
          manifest.method(object.contractName(), pinMethod);
      if (!method)
        return op.emitError() << "membership probe needs "
                              << object.contractName() << "." << pinMethod
                              << " to pin its operand";
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
    op.getResult().replaceAllUsesWith(call.getResult(0));
    erase.push_back(op);
    return mlir::success();
  };
  if ((runtimeSetProbe || runtimeDictProbe) &&
      item.kind == RuntimeBundle::Kind::Object)
    return emitBoxProbe();
  bool sequenceContainer = container.kind == RuntimeBundle::Kind::Object &&
                           (container.contractName() == "builtins.list" ||
                            container.contractName() == "builtins.tuple");
  if (sequenceContainer && !container.sequenceElementBundles.empty()) {
    // Constant-fold evidence membership when every element compares
    // statically; otherwise probe the published payload at runtime.
    bool allKnown = true;
    {
      mlir::OpBuilder::InsertionGuard probeGuard(builder);
      builder.setInsertionPoint(op);
      mlir::Block *scratch = new mlir::Block();
      builder.setInsertionPointToStart(scratch);
      for (const std::shared_ptr<RuntimeBundle> &element :
           container.sequenceElementBundles) {
        if (!element ||
            !knownEvidenceEquality(op, builder, *element, item)) {
          allKnown = false;
          break;
        }
      }
      scratch->dropAllReferences();
      delete scratch;
    }
    if (allKnown) {
      builder.setInsertionPoint(op);
      if (mlir::failed(touchCollectionEvidenceUse(op, builder, container,
                                                  "sequence contains")))
        return mlir::failure();
      mlir::Location loc = op.getLoc();
      mlir::Value result = constantI1(builder, loc, false);
      for (const std::shared_ptr<RuntimeBundle> &element :
           container.sequenceElementBundles) {
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
  }
  if (sequenceContainer && container.physicalValues().size() >= 3 &&
      item.kind == RuntimeBundle::Kind::Object)
    return emitBoxProbe();
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
      // dict/set iterators additionally remember the container's size at
      // creation (cell word 1): CPython raises RuntimeError when the size
      // changes during iteration, while list iteration legally re-checks the
      // live length each step.
      bool guardsMutation = runtimeDictIterable || runtimeSetIterable;
      mlir::Value cell;
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(&function.getBody().front());
        cell = mlir::memref::AllocaOp::create(
                   builder, op.getLoc(),
                   mlir::MemRefType::get({guardsMutation ? 2 : 1},
                                         builder.getI64Type()))
                   .getResult();
      }
      builder.setInsertionPoint(op);
      mlir::Value zero =
          mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
      mlir::Value slot =
          mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 0);
      mlir::memref::StoreOp::create(builder, op.getLoc(), zero, cell, slot);
      if (guardsMutation) {
        mlir::Value meta = iterable->physicalValues()[1];
        mlir::Value initial =
            mlir::memref::LoadOp::create(builder, op.getLoc(), meta, slot)
                .getResult();
        mlir::Value initialSlot =
            mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 1);
        mlir::memref::StoreOp::create(builder, op.getLoc(), initial, cell,
                                      initialSlot);
      }
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
  // dict/set iteration: raise RuntimeError when the container's size changed
  // since the iterator was created (CPython's mutation-during-iteration
  // guard; the size at creation sits in cell word 1).
  bool guardsMutation = iterator.contractName() == "builtins.dict" ||
                        iterator.contractName() == "builtins.set";
  if (guardsMutation) {
    mlir::Value initialSlot =
        mlir::arith::ConstantIndexOp::create(builder, loc, 1);
    mlir::Value initial =
        mlir::memref::LoadOp::create(builder, loc, cell, initialSlot)
            .getResult();
    mlir::Value changed = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::ne, length, initial);
    auto changedGuard = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{}, changed, /*withElseRegion=*/false);
    {
      mlir::OpBuilder::InsertionGuard insertionGuard(builder);
      builder.setInsertionPointToStart(&changedGuard.getThenRegion().front());
      llvm::StringRef message =
          iterator.contractName() == "builtins.dict"
              ? "dictionary changed size during iteration"
              : "Set changed size during iteration";
      if (mlir::failed(
              emitRuntimeException(op, "builtins.RuntimeError", message)))
        return mlir::failure();
    }
    builder.setInsertionPointAfter(changedGuard);
  }
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

  if (runtimeContractName(elementContract) == "builtins.object") {
    // Erased read lane: box the slot's canonical payload handle; the
    // exhausted branch yields the None handle inside the primitive, so no
    // dead placeholder machinery is needed.
    std::optional<RuntimeSymbol> fromSlot =
        manifest.primitive("builtins.object", "from_slot");
    if (!fromSlot)
      return op.emitError()
             << "runtime manifest has no object from_slot primitive";
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *fromSlot,
        mlir::ValueRange{iterator.physicalValues()[2], safe, valid});
    RuntimeValue element{elementContract,
                         {boxed.getResult(0)},
                         ownership::logicalOwnershipKind(elementContract,
                                                         /*ownsObject=*/false)};
    if (mlir::failed(bindEvidenceObjectResult(op, op.getElement(),
                                              "runtime list element", element)))
      return mlir::failure();
    std::optional<RuntimeSymbol> lenPin =
        manifest.method(iterator.contractName(), "__len__");
    if (!lenPin)
      return op.emitError()
             << "list iteration needs a runtime __len__ to pin the container";
    llvm::SmallVector<const RuntimeBundle *, 1> pinSources{&iterator};
    llvm::SmallVector<mlir::Value, 4> pinOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *lenPin, pinSources,
                                              pinOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    builder.setInsertionPoint(op);
    RuntimeBundleLowerer::createRuntimeCall(loc, *lenPin, pinOperands);
    op.getValid().replaceAllUsesWith(valid);
    valueBundles[op.getNext()] = iterator;
    erase.push_back(op);
    return mlir::success();
  }

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
  using box_abi::kPointerWordBase;
  using box_abi::kSizeWordBase;
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, box_abi::kWordsPerBox,
                                         64);
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
        loadBoxWord(kPointerWordBase + static_cast<std::int64_t>(index));
    mlir::Value realSize =
        loadBoxWord(kSizeWordBase + static_cast<std::int64_t>(index));
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
