#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"

#include <algorithm>

namespace py::lowering {
namespace {

bool isSequenceCollection(llvm::StringRef contract) {
  return contract == "builtins.list" || contract == "builtins.tuple";
}

bool isI64Payload(mlir::Value value) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  return memref && memref.getRank() == 1 &&
         mlir::isa<mlir::IntegerType>(memref.getElementType()) &&
         mlir::cast<mlir::IntegerType>(memref.getElementType()).getWidth() ==
             64;
}

mlir::Value constantI64(mlir::OpBuilder &builder, mlir::Location loc,
                        std::int64_t value) {
  return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
      .getResult();
}

mlir::Value constantIndex(mlir::OpBuilder &builder, mlir::Location loc,
                          unsigned value) {
  return mlir::arith::ConstantIndexOp::create(builder, loc, value).getResult();
}

constexpr unsigned kPayloadHandleWords =
    static_cast<unsigned>(box_abi::kWordsPerBox);
constexpr unsigned kPayloadValuePointerWords =
    static_cast<unsigned>(box_abi::kPointerWordCount);
constexpr unsigned kPayloadValuePointerBase =
    static_cast<unsigned>(box_abi::kPointerWordBase);
constexpr unsigned kPayloadValueSizeBase =
    static_cast<unsigned>(box_abi::kSizeWordBase);
constexpr unsigned kPayloadOwnedFlagSlot =
    static_cast<unsigned>(box_abi::kOwnedFlagWord);
constexpr std::uint64_t kMinimumCollectionCapacity = 64;

std::uint64_t growCapacity(std::uint64_t current, std::uint64_t required) {
  std::uint64_t capacity =
      std::max<std::uint64_t>(current, kMinimumCollectionCapacity);
  while (capacity < required)
    capacity *= 2;
  return capacity;
}

mlir::Value pointerWordForPhysicalValue(mlir::OpBuilder &builder,
                                        mlir::Location loc, mlir::Value value,
                                        mlir::Value zero) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memref || memref.getRank() != 1)
    return zero;
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc, value);
  return mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                          pointerIndex)
      .getResult();
}

mlir::Value sizeWordForPhysicalValue(mlir::OpBuilder &builder,
                                     mlir::Location loc, mlir::Value value,
                                     mlir::Value zero) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!memref || memref.getRank() != 1)
    return zero;
  if (memref.hasStaticShape())
    return constantI64(builder, loc, memref.getDimSize(0));
  mlir::Value dim = mlir::memref::DimOp::create(builder, loc, value, 0);
  return mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                          dim)
      .getResult();
}

mlir::LogicalResult storePayloadWord(mlir::Operation *op,
                                     mlir::OpBuilder &builder,
                                     mlir::Value payload, unsigned index,
                                     mlir::Value word, llvm::StringRef label) {
  if (!payload || !isI64Payload(payload))
    return op->emitError() << label << " payload has invalid type "
                           << (payload ? payload.getType() : mlir::Type());
  builder.setInsertionPoint(op);
  mlir::Value slot = constantIndex(builder, op->getLoc(), index);
  mlir::memref::StoreOp::create(builder, op->getLoc(), word, payload, slot);
  return mlir::success();
}

mlir::LogicalResult storePayloadHandle(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       mlir::Value payload, unsigned index,
                                       llvm::ArrayRef<mlir::Value> words,
                                       llvm::StringRef label) {
  if (words.size() != kPayloadHandleWords)
    return op->emitError() << label << " payload handle must have "
                           << kPayloadHandleWords << " words";
  unsigned base = index * kPayloadHandleWords;
  for (auto [offset, word] : llvm::enumerate(words))
    if (mlir::failed(
            storePayloadWord(op, builder, payload, base + offset, word, label)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult storePayloadHandleAt(mlir::Operation *op,
                                         mlir::OpBuilder &builder,
                                         mlir::Value payload,
                                         mlir::Value logicalIndex,
                                         llvm::ArrayRef<mlir::Value> words,
                                         llvm::StringRef label) {
  if (!payload || !isI64Payload(payload))
    return op->emitError() << label << " payload has invalid type "
                           << (payload ? payload.getType() : mlir::Type());
  if (words.size() != kPayloadHandleWords)
    return op->emitError() << label << " payload handle must have "
                           << kPayloadHandleWords << " words";
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value wordsPerSlot =
      constantI64(builder, loc, static_cast<std::int64_t>(kPayloadHandleWords));
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, logicalIndex, wordsPerSlot)
          .getResult();
  mlir::Value baseIndex = mlir::arith::IndexCastOp::create(
                              builder, loc, builder.getIndexType(), base)
                              .getResult();
  for (auto [offset, word] : llvm::enumerate(words)) {
    mlir::Value slot =
        mlir::arith::AddIOp::create(
            builder, loc, baseIndex,
            constantIndex(builder, loc, static_cast<unsigned>(offset)))
            .getResult();
    mlir::memref::StoreOp::create(builder, loc, word, payload, slot);
  }
  return mlir::success();
}

mlir::LogicalResult clearPayloadHandle(mlir::Operation *op,
                                       mlir::OpBuilder &builder,
                                       mlir::Value payload, unsigned index,
                                       llvm::StringRef label) {
  mlir::Value zero = constantI64(builder, op->getLoc(), 0);
  llvm::SmallVector<mlir::Value, 4> words(kPayloadHandleWords, zero);
  return storePayloadHandle(op, builder, payload, index, words, label);
}

} // namespace

std::uint64_t
RuntimeBundleLowerer::collectionInitialCapacity(std::uint64_t arity) const {
  return std::max<std::uint64_t>(arity, kMinimumCollectionCapacity);
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::objectPayloadHandleWords(mlir::Operation *op,
                                               const RuntimeBundle &value,
                                               bool ownsPayload) {
  builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value zero = constantI64(builder, loc, 0);
  auto emptyHandle = [&]() {
    return llvm::SmallVector<mlir::Value, 4>(kPayloadHandleWords, zero);
  };

  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(value);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "collection payload element is not an object";
  if (concrete->contractName() == "types.NoneType")
    return emptyHandle();
  // Slots hold CANONICAL payload handles (word 1 = payload class, words 4+
  // = the payload's own memrefs) so hash/eq/repr dispatch reads them
  // uniformly. An opaque erased `object` (no tracked concrete payload)
  // would store a handle-of-box indirection those dispatchers cannot
  // distinguish; reject it loudly rather than mis-execute.
  if (concrete->contractName() == "builtins.object")
    return op->emitError()
           << "a type-erased `object` value cannot be stored in a runtime "
              "container slot yet; give the container a concrete element "
              "type annotation";
  if (concrete->physicalValues().empty())
    return op->emitError()
           << "collection payload element " << concrete->contract
           << " has no physical object handle; materialize it before storing";

  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectPhysicalHeader(op, concrete->objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  mlir::Value classSlot = constantIndex(builder, loc, 1);
  mlir::Value payloadClass =
      mlir::memref::LoadOp::create(builder, loc, *header, classSlot)
          .getResult();
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                           *header);
  mlir::Value payloadPointer =
      mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                       pointerIndex)
          .getResult();
  mlir::Value refcount = constantI64(builder, loc, 1);
  mlir::Value valueCount =
      constantI64(builder, loc,
                  std::min<unsigned>(concrete->physicalValues().size(),
                                     kPayloadValuePointerWords));
  mlir::Value owned = constantI64(builder, loc, ownsPayload ? 1 : 0);
  llvm::SmallVector<mlir::Value, 4> words(kPayloadHandleWords, zero);
  words[0] = refcount;
  words[1] = payloadClass;
  words[2] = payloadPointer;
  words[3] = valueCount;
  for (auto [index, physical] : llvm::enumerate(
           concrete->physicalValues().take_front(kPayloadValuePointerWords))) {
    words[kPayloadValuePointerBase + index] =
        pointerWordForPhysicalValue(builder, loc, physical, zero);
    words[kPayloadValueSizeBase + index] =
        sizeWordForPhysicalValue(builder, loc, physical, zero);
  }
  words[kPayloadOwnedFlagSlot] = owned;
  return words;
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::materializePayloadObjectBundle(
    mlir::Operation *op, const RuntimeBundle &value) {
  const RuntimeBundle *concrete =
      RuntimeBundleLowerer::concreteObjectForOwnership(value);
  if (!concrete || concrete->kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "collection payload requires an object bundle";
  if (concrete->contractName() == "types.NoneType")
    return *concrete;
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*concrete)) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<RuntimeValue> object =
        RuntimeBundleLowerer::materializePrimitiveI64Object(op, *concrete);
    if (mlir::failed(object))
      return mlir::failure();
    RuntimeBundle materialized =
        RuntimeBundle::object(concrete->objectValue.contract, object->values);
    materialized.copyEvidenceFrom(*concrete);
    return materialized;
  }
  // A contract with a `box` primitive stores its boxed form (e.g. bool: the
  // canonical i1 boxes to an immortal singleton header) — the slot layout
  // requires a header-fronted value group.
  if (!concrete->physicalValues().empty() &&
      !ownership::isObjectHeaderLikeType(
          concrete->physicalValues().front().getType())) {
    if (std::optional<RuntimeSymbol> box =
            manifest.primitive(concrete->contractName(), "box")) {
      builder.setInsertionPoint(op);
      mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
          op->getLoc(), *box, concrete->physicalValues());
      RuntimeBundle materialized = RuntimeBundle::object(
          concrete->objectValue.contract, call.getResults());
      materialized.copyEvidenceFrom(*concrete);
      return materialized;
    }
  }
  if (concrete->physicalValues().empty())
    return op->emitError() << "collection payload element "
                           << concrete->contract
                           << " has no physical object handle";
  return *concrete;
}

mlir::LogicalResult RuntimeBundleLowerer::ensureSequencePayloadCapacity(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    llvm::StringRef label) {
  if (container.sequenceCapacity && index < container.sequenceCapacity)
    return mlir::success();
  if (container.sequenceCapacity == 0 && index < kMinimumCollectionCapacity)
    return mlir::success();

  std::optional<RuntimeSymbol> ensure =
      manifest.primitive(container.contractName(), "ensure_capacity");
  if (!ensure)
    return op->emitError() << label
                           << " payload capacity is exceeded, but the runtime "
                           << "manifest has no ensure_capacity primitive";

  builder.setInsertionPoint(op);
  mlir::Value required =
      constantI64(builder, op->getLoc(), static_cast<std::int64_t>(index) + 1);
  llvm::SmallVector<mlir::Value, 4> operands(container.physicalValues().begin(),
                                             container.physicalValues().end());
  operands.push_back(required);
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *ensure, operands);
  RuntimeBundle updated;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, container.objectValue.contract, call.getResults(), updated)))
    return mlir::failure();
  updated.copyEvidenceFrom(container);
  std::uint64_t oldCapacity = container.sequenceCapacity
                                  ? container.sequenceCapacity
                                  : kMinimumCollectionCapacity;
  updated.sequenceCapacity =
      growCapacity(oldCapacity, static_cast<std::uint64_t>(index) + 1);
  container = std::move(updated);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::ensureDictPayloadCapacity(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (container.mappingCapacity && index < container.mappingCapacity)
    return mlir::success();
  if (container.mappingCapacity == 0 && index < kMinimumCollectionCapacity)
    return mlir::success();

  std::optional<RuntimeSymbol> ensure =
      manifest.primitive("builtins.dict", "ensure_capacity");
  if (!ensure)
    return op->emitError()
           << "dict payload capacity is exceeded, but the runtime manifest "
              "has no ensure_capacity primitive";

  builder.setInsertionPoint(op);
  mlir::Value required =
      constantI64(builder, op->getLoc(), static_cast<std::int64_t>(index) + 1);
  llvm::SmallVector<mlir::Value, 6> operands(container.physicalValues().begin(),
                                             container.physicalValues().end());
  operands.push_back(required);
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *ensure, operands);
  RuntimeBundle updated;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, container.objectValue.contract, call.getResults(), updated)))
    return mlir::failure();
  updated.copyEvidenceFrom(container);
  std::uint64_t oldCapacity = container.mappingCapacity
                                  ? container.mappingCapacity
                                  : kMinimumCollectionCapacity;
  updated.mappingCapacity =
      growCapacity(oldCapacity, static_cast<std::uint64_t>(index) + 1);
  container = std::move(updated);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::initializeSequencePayload(
    mlir::Operation *op, RuntimeBundle &container,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> elements) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  if (container.physicalValues().size() < 3)
    return op->emitError() << container.contractName()
                           << " has no physical item payload";
  container.sequenceCapacity =
      RuntimeBundleLowerer::collectionInitialCapacity(elements.size());
  container.sequenceEvidenceBacked = true;
  for (auto [index, element] : llvm::enumerate(elements)) {
    if (!element)
      continue;
    mlir::FailureOr<RuntimeBundle> payload =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, *element);
    if (mlir::failed(payload))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payload, "sequence.literal")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
            op, container, static_cast<unsigned>(index), *payload)))
      return mlir::failure();
    if (payload->objectValue.ownership == ownership::OwnershipKind::Own &&
        mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, *payload, "sequence.literal.source")))
      return mlir::failure();
    RuntimeBundle stored = payload->withObjectOwnership(
        ownership::logicalOwnershipKind(payload->objectValue.contract,
                                        /*ownsObject=*/false));
    if (index < container.sequenceElementBundles.size())
      container.sequenceElementBundles[index] =
          std::make_shared<RuntimeBundle>(stored);
    if (index < container.sequenceElements.size())
      container.sequenceElements[index] = stored.objectValue;
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::storeSequencePayloadElement(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &element) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  if (container.physicalValues().size() < 3)
    return op->emitError() << container.contractName()
                           << " has no physical item payload";
  if (mlir::failed(RuntimeBundleLowerer::ensureSequencePayloadCapacity(
          op, container, index, container.contractName())))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, element);
  if (mlir::failed(words))
    return mlir::failure();
  return storePayloadHandle(op, builder, container.physicalValues()[2], index,
                            *words, container.contractName());
}

mlir::LogicalResult RuntimeBundleLowerer::storeSequencePayloadElementAt(
    mlir::Operation *op, RuntimeBundle &container, mlir::Value logicalIndex,
    const RuntimeBundle &element) {
  if (!isSequenceCollection(container.contractName()))
    return op->emitError() << container.contractName()
                           << " is not a sequence collection";
  if (container.physicalValues().size() < 3)
    return op->emitError() << container.contractName()
                           << " has no physical item payload";
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, element);
  if (mlir::failed(words))
    return mlir::failure();
  return storePayloadHandleAt(op, builder, container.physicalValues()[2],
                              logicalIndex, *words, container.contractName());
}

mlir::LogicalResult RuntimeBundleLowerer::clearSequencePayloadElement(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (!isSequenceCollection(container.contractName()))
    return mlir::success();
  if (container.physicalValues().size() < 3)
    return op->emitError() << container.contractName()
                           << " has no physical item payload";
  if (container.sequenceCapacity && index >= container.sequenceCapacity)
    return op->emitError() << container.contractName()
                           << " payload clear index exceeds capacity";
  return clearPayloadHandle(op, builder, container.physicalValues()[2], index,
                            container.contractName());
}

mlir::LogicalResult RuntimeBundleLowerer::initializeDictPayload(
    mlir::Operation *op, RuntimeBundle &container,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> keys,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> values) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (keys.size() != values.size())
    return op->emitError() << "dict payload key/value count mismatch";
  container.mappingCapacity =
      RuntimeBundleLowerer::collectionInitialCapacity(keys.size());
  container.mappingEvidenceBacked = true;
  for (auto [index, key] : llvm::enumerate(keys)) {
    if (!key || !values[index])
      return op->emitError() << "dict payload entry has no object evidence";
    mlir::FailureOr<RuntimeBundle> payloadKey =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op, *key);
    if (mlir::failed(payloadKey))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> payloadValue =
        RuntimeBundleLowerer::materializePayloadObjectBundle(op,
                                                             *values[index]);
    if (mlir::failed(payloadValue))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadKey, "dict.literal.key")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *payloadValue, "dict.literal.value")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::storeDictKeyPayload(
            op, container, static_cast<unsigned>(index), *payloadKey)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::storeDictValuePayload(
            op, container, static_cast<unsigned>(index), *payloadValue)))
      return mlir::failure();
    if (payloadKey->objectValue.ownership == ownership::OwnershipKind::Own &&
        mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, *payloadKey, "dict.literal.key.source")))
      return mlir::failure();
    if (payloadValue->objectValue.ownership == ownership::OwnershipKind::Own &&
        mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, *payloadValue, "dict.literal.value.source")))
      return mlir::failure();
    RuntimeBundle storedKey = payloadKey->withObjectOwnership(
        ownership::logicalOwnershipKind(payloadKey->objectValue.contract,
                                        /*ownsObject=*/false));
    RuntimeBundle storedValue =
        payloadValue->withObjectOwnership(ownership::logicalOwnershipKind(
            payloadValue->objectValue.contract, /*ownsObject=*/false));
    if (index < container.mappingKeyBundles.size())
      container.mappingKeyBundles[index] =
          std::make_shared<RuntimeBundle>(storedKey);
    if (index < container.mappingValueBundles.size())
      container.mappingValueBundles[index] =
          std::make_shared<RuntimeBundle>(storedValue);
    if (index < container.mappingValues.size())
      container.mappingValues[index] = storedValue.objectValue;
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::storeDictKeyPayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &key) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.physicalValues().size() < 5)
    return op->emitError() << "dict has no physical payload arrays";
  if (mlir::failed(RuntimeBundleLowerer::ensureDictPayloadCapacity(
          op, container, index)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, key);
  if (mlir::failed(words))
    return mlir::failure();
  return storePayloadHandle(op, builder, container.physicalValues()[2], index,
                            *words, "dict keys");
}

mlir::LogicalResult RuntimeBundleLowerer::storeDictValuePayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index,
    const RuntimeBundle &value) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.physicalValues().size() < 5)
    return op->emitError() << "dict has no physical payload arrays";
  if (mlir::failed(RuntimeBundleLowerer::ensureDictPayloadCapacity(
          op, container, index)))
    return mlir::failure();
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, value);
  if (mlir::failed(words))
    return mlir::failure();
  if (mlir::failed(storePayloadHandle(op, builder,
                                      container.physicalValues()[3], index,
                                      *words, "dict values")))
    return mlir::failure();
  return storePayloadWord(op, builder, container.physicalValues()[4], index,
                          constantI64(builder, op->getLoc(), 1),
                          "dict present");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictKeyPayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.physicalValues().size() < 5)
    return op->emitError() << "dict has no physical payload arrays";
  if (container.mappingCapacity && index >= container.mappingCapacity)
    return op->emitError() << "dict payload clear index exceeds capacity";
  return clearPayloadHandle(op, builder, container.physicalValues()[2], index,
                            "dict keys");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictValuePayload(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (container.contractName() != "builtins.dict")
    return mlir::success();
  if (container.physicalValues().size() < 5)
    return op->emitError() << "dict has no physical payload arrays";
  if (container.mappingCapacity && index >= container.mappingCapacity)
    return op->emitError() << "dict payload clear index exceeds capacity";
  mlir::Value zero = constantI64(builder, op->getLoc(), 0);
  if (mlir::failed(clearPayloadHandle(
          op, builder, container.physicalValues()[3], index, "dict values")))
    return mlir::failure();
  return storePayloadWord(op, builder, container.physicalValues()[4], index,
                          zero, "dict present");
}

mlir::LogicalResult RuntimeBundleLowerer::clearDictPayloadEntry(
    mlir::Operation *op, RuntimeBundle &container, unsigned index) {
  if (mlir::failed(
          RuntimeBundleLowerer::clearDictKeyPayload(op, container, index)))
    return mlir::failure();
  return RuntimeBundleLowerer::clearDictValuePayload(op, container, index);
}

} // namespace py::lowering
