#include "Runtime/Ctypes/Internal.h"

namespace py::runtime_lowering {

using namespace ctypes;

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesCall(py::CallOp op,
                                            const RuntimeBundle &callable) {
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "ctypes arguments", sources, &unpackedSources)))
    return mlir::failure();

  if (callable.binding == "ctypes.sizeof" ||
      callable.binding == "ctypes.alignment") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError()
             << callable.binding << " expects one static ctypes argument";
    std::string contract = ctypesContractFromBundle(*sources.front());
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    std::optional<CtypesLayout> layout = ctypesStaticLayoutForType(
        module, ctypesTypeFromBundle(*sources.front()), facts);
    if (!layout)
      layout = ctypesStaticLayout(module, contract, facts);
    if (!layout)
      return op.emitError()
             << callable.binding << "(" << contract
             << ") requires complete TargetPlatformFacts before lowering ("
             << targetFactsLabel(facts) << ")";
    std::int64_t value = callable.binding == "ctypes.sizeof"
                             ? static_cast<std::int64_t>(layout->size)
                             : static_cast<std::int64_t>(layout->align);
    builder.setInsertionPoint(op);
    mlir::Value scalar = constantI64(builder, op.getLoc(), value);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), scalar, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromAddressTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError()
             << "ctypes.from_address expects one integer address";
    const RuntimeBundle *address = sources.front();
    if (!address || !address->primitiveI64 || !address->primitiveI64->value ||
        !address->primitiveI64->valid ||
        !isKnownTrue(address->primitiveI64->valid))
      return op.emitError()
             << "ctypes.from_address requires a statically available pointer "
                "integer address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_address requires TargetPlatformFacts before "
                "lowering";
    builder.setInsertionPoint(op);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::ExternalAddress;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::External;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.addressValue =
        coerceNativeInteger(builder, op.getLoc(), address->primitiveI64->value,
                            nativePointerIntegerType(builder, facts));
    evidence.addressValid = constantI1(builder, op.getLoc(), true);
    evidence.storageAddressValue = evidence.addressValue;
    evidence.storageAddressValid = evidence.addressValid;
    if (std::optional<CtypesLayout> layout =
            ctypesStaticLayout(module, *target, facts))
      attachCtypesBufferEvidence(builder, op.getLoc(), result, evidence,
                                 *layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromBufferTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError() << "ctypes.from_buffer expects a buffer object and "
                               "optional offset";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->buffer || !source->buffer->addressValue ||
        !source->buffer->addressValid ||
        !isKnownTrue(source->buffer->addressValid))
      return op.emitError()
             << "ctypes.from_buffer(" << *target
             << ") requires statically available buffer address evidence";
    if (!source->buffer->readable || !source->buffer->writable ||
        !source->buffer->cContiguous)
      return op.emitError() << "ctypes.from_buffer(" << *target
                            << ") requires a writable C-contiguous buffer";

    std::int64_t offset = 0;
    if (sources.size() == 2) {
      const RuntimeBundle *offsetSource = sources[1];
      if (!offsetSource || !offsetSource->primitiveI64 ||
          !offsetSource->primitiveI64->value ||
          !offsetSource->primitiveI64->valid ||
          !isKnownTrue(offsetSource->primitiveI64->valid))
        return op.emitError()
               << "ctypes.from_buffer offset requires primitive integer "
                  "evidence";
      std::optional<std::int64_t> staticOffset =
          knownI64Constant(offsetSource->primitiveI64->value);
      if (!staticOffset)
        return op.emitError()
               << "ctypes.from_buffer offset must be statically known";
      if (*staticOffset < 0)
        return op.emitError() << "ctypes.from_buffer offset must be >= 0";
      offset = *staticOffset;
    }

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_buffer requires TargetPlatformFacts before "
                "lowering";
    std::optional<CtypesLayout> targetLayout =
        ctypesStaticLayout(module, *target, facts);
    if (!targetLayout)
      return op.emitError()
             << "ctypes.from_buffer(" << *target << ") has no ABI layout for "
             << targetFactsLabel(facts);
    if (!source->buffer->byteLength || !source->buffer->byteLengthValid ||
        !isKnownTrue(source->buffer->byteLengthValid))
      return op.emitError()
             << "ctypes.from_buffer requires statically available buffer "
                "length evidence";
    std::optional<std::int64_t> sourceLength =
        knownI64Constant(source->buffer->byteLength);
    if (!sourceLength)
      return op.emitError()
             << "ctypes.from_buffer buffer length must be statically known";
    if (offset > *sourceLength ||
        static_cast<std::uint64_t>(*sourceLength - offset) < targetLayout->size)
      return op.emitError() << "ctypes.from_buffer(" << *target
                            << ") exceeds the statically proven buffer size";

    builder.setInsertionPoint(op);
    mlir::Value address = addressWithOffset(
        builder, op.getLoc(), source->buffer->addressValue, offset, facts);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::BufferView;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.addressValue = address;
    evidence.addressValid = valid;
    evidence.storageAddressValue = address;
    evidence.storageAddressValid = valid;
    keepAliveSource(evidence, *source);
    evidence.keepAlive.append(source->buffer->keepAlive.begin(),
                              source->buffer->keepAlive.end());
    result.ctypes = std::move(evidence);
    result.buffer = makeCtypesBufferEvidence(
        builder, op.getLoc(), *targetLayout, address, valid, *source,
        /*writable=*/true);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (std::optional<llvm::StringRef> target =
          ctypesFromBufferCopyTarget(callable.binding)) {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError()
             << "ctypes.from_buffer_copy expects a buffer object and optional "
                "offset";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->buffer || !source->buffer->addressValue ||
        !source->buffer->addressValid ||
        !isKnownTrue(source->buffer->addressValid))
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") requires statically available buffer address evidence";
    if (!source->buffer->readable || !source->buffer->cContiguous)
      return op.emitError() << "ctypes.from_buffer_copy(" << *target
                            << ") requires a readable C-contiguous buffer";

    std::int64_t offset = 0;
    if (sources.size() == 2) {
      const RuntimeBundle *offsetSource = sources[1];
      if (!offsetSource || !offsetSource->primitiveI64 ||
          !offsetSource->primitiveI64->value ||
          !offsetSource->primitiveI64->valid ||
          !isKnownTrue(offsetSource->primitiveI64->valid))
        return op.emitError()
               << "ctypes.from_buffer_copy offset requires primitive integer "
                  "evidence";
      std::optional<std::int64_t> staticOffset =
          knownI64Constant(offsetSource->primitiveI64->value);
      if (!staticOffset)
        return op.emitError()
               << "ctypes.from_buffer_copy offset must be statically known";
      if (*staticOffset < 0)
        return op.emitError() << "ctypes.from_buffer_copy offset must be >= 0";
      offset = *staticOffset;
    }

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.from_buffer_copy requires TargetPlatformFacts before "
                "lowering";
    std::optional<CtypesLayout> targetLayout =
        ctypesStaticLayout(module, *target, facts);
    if (!targetLayout)
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") has no ABI layout for " << targetFactsLabel(facts);
    if (!isIntegerScalarLayout(*targetLayout) &&
        !isPointerScalarLayout(*targetLayout))
      return op.emitError()
             << "ctypes.from_buffer_copy(" << *target
             << ") currently requires an integer or pointer scalar layout";
    if (!source->buffer->byteLength || !source->buffer->byteLengthValid ||
        !isKnownTrue(source->buffer->byteLengthValid))
      return op.emitError()
             << "ctypes.from_buffer_copy requires statically available buffer "
                "length evidence";
    std::optional<std::int64_t> sourceLength =
        knownI64Constant(source->buffer->byteLength);
    if (!sourceLength)
      return op.emitError() << "ctypes.from_buffer_copy buffer length must be "
                               "statically known";
    if (offset > *sourceLength ||
        static_cast<std::uint64_t>(*sourceLength - offset) < targetLayout->size)
      return op.emitError() << "ctypes.from_buffer_copy(" << *target
                            << ") exceeds the statically proven buffer size";

    builder.setInsertionPoint(op);
    mlir::Value sourceAddress = addressWithOffset(
        builder, op.getLoc(), source->buffer->addressValue, offset, facts);
    mlir::IntegerType nativeType = nativeIntegerType(builder, *targetLayout);
    mlir::Value nativeValue = loadNativeIntegerFromAddress(
        builder, op.getLoc(), sourceAddress, nativeType, facts);
    mlir::Value ownedAddress =
        addressOfNativeCellAlloca(builder, op.getLoc(), nativeValue, facts);
    mlir::Value valid = constantI1(builder, op.getLoc(), true);

    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.ctypeName = target->str();
    evidence.ctype = ctypesContractType(context, *target);
    evidence.scalarValue =
        widenNativeInteger(builder, op.getLoc(), nativeValue, *targetLayout);
    evidence.scalarValid = valid;
    evidence.addressValue = ownedAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = ownedAddress;
    evidence.storageAddressValid = valid;
    evidence.ownsNativeStorage = true;
    attachCtypesBufferEvidence(builder, op.getLoc(), result, evidence,
                               *targetLayout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.addressof") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError() << "ctypes.addressof expects one ctypes object";
    const RuntimeBundle *source = sources.front();
    if (!source || !source->ctypes)
      return op.emitError()
             << "ctypes.addressof requires materialized _CData evidence";
    mlir::Value storageAddress = cdataStorageAddress(*source->ctypes);
    mlir::Value storageValid = cdataStorageAddressValid(*source->ctypes);
    if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
      return op.emitError()
             << "ctypes.addressof requires a materialized _CData storage "
                "address";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.addressof requires TargetPlatformFacts before lowering";
    builder.setInsertionPoint(op);
    mlir::Value raw =
        coerceNativeInteger(builder, op.getLoc(), storageAddress,
                            nativePointerIntegerType(builder, facts));
    mlir::Value i64 = raw;
    auto rawType = mlir::cast<mlir::IntegerType>(raw.getType());
    if (rawType.getWidth() < 64)
      i64 = builder
                .create<mlir::arith::ExtUIOp>(op.getLoc(), builder.getI64Type(),
                                              raw)
                .getResult();
    else if (rawType.getWidth() > 64)
      i64 = builder
                .create<mlir::arith::TruncIOp>(op.getLoc(),
                                               builder.getI64Type(), raw)
                .getResult();
    mlir::Value valid = constantI1(builder, op.getLoc(), true);
    RuntimeBundle result;
    if (mlir::failed(makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"), i64, valid,
            result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.cast") {
    if (op.getNumResults() != 1 || sources.size() != 2)
      return op.emitError() << "ctypes.cast expects a source and ctypes type";
    const RuntimeBundle *source = sources[0];
    const RuntimeBundle *target = sources[1];
    if (!source || !target)
      return op.emitError() << "ctypes.cast arguments have no evidence";
    std::optional<std::string> targetContract = ctypesTypeObjectName(*target);
    if (!targetContract)
      return op.emitError()
             << "ctypes.cast target must be a static ctypes type object";

    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    if (!facts)
      return op.emitError()
             << "ctypes.cast requires TargetPlatformFacts before lowering";
    builder.setInsertionPoint(op);
    std::optional<mlir::Value> address =
        extractPointerAddressInteger(op, builder, *source, facts);
    if (!address)
      return op.emitError()
             << "ctypes.cast source requires None, primitive pointer integer, "
                "c_void_p, or pointer evidence with a concrete external "
                "address ("
             << describeNativeArgumentSource(*source) << ")";

    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.ctypeName = *targetContract;
    evidence.ctype = ctypesTypeFromBundle(*target);
    evidence.provenance = RuntimeCtypesEvidence::Provenance::Cast;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::External;
    evidence.addressValue = *address;
    evidence.addressValid = constantI1(builder, op.getLoc(), true);
    keepAliveSource(evidence, *source);
    if (isCtypesVoidPointer(*targetContract)) {
      evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
      evidence.scalarValue = *address;
      evidence.scalarValid = evidence.addressValid;
    } else if (isCtypesPointerContract(*targetContract)) {
      evidence.kind = RuntimeCtypesEvidence::Kind::Pointer;
      evidence.pointeeType = *targetContract;
    } else {
      return op.emitError() << "ctypes.cast target " << *targetContract
                            << " is not a supported pointer-like ctypes type";
    }

    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.byref" ||
      callable.binding == "ctypes.pointer") {
    if (op.getNumResults() != 1 || sources.empty() || sources.size() > 2)
      return op.emitError() << callable.binding
                            << " expects a ctypes object and optional offset";
    const RuntimeBundle *pointee = sources.front();
    if (!pointee || !pointee->ctypes ||
        pointee->ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
      return op.emitError()
             << callable.binding << " requires an erased ctypes cell";
    if (sources.size() == 2) {
      const RuntimeBundle *offset = sources[1];
      if (!offset || !offset->primitiveI64)
        return op.emitError()
               << callable.binding << " offset requires primitive integer "
               << "evidence";
      // Offset provenance is a separate proof term. Until it exists, only the
      // default zero-offset form is accepted on the erased path.
      return op.emitError()
             << callable.binding << " erased offset is not implemented yet";
    }

    builder.setInsertionPoint(op);
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Pointer;
    evidence.ctypeName = callable.binding == "ctypes.pointer"
                             ? "_ctypes._Pointer"
                             : pointee->ctypes->ctypeName;
    if (callable.binding == "ctypes.pointer") {
      mlir::Type pointeeType =
          pointee->ctypes->ctype
              ? pointee->ctypes->ctype
              : ctypesContractType(context, pointee->ctypes->ctypeName);
      evidence.ctype =
          py::ContractType::get(context, "_ctypes._Pointer", {pointeeType});
    } else {
      evidence.ctype = pointee->ctypes->ctype ? pointee->ctypes->ctype
                                              : op.getResult(0).getType();
    }
    evidence.provenance =
        callable.binding == "ctypes.byref"
            ? RuntimeCtypesEvidence::Provenance::CallRegionBorrow
            : RuntimeCtypesEvidence::Provenance::NativeCell;
    evidence.lifetime = callable.binding == "ctypes.byref"
                            ? RuntimeCtypesEvidence::Lifetime::CallRegion
                            : RuntimeCtypesEvidence::Lifetime::Owner;
    evidence.pointeeType = pointee->ctypes->ctypeName;
    evidence.pointeeScalarValue = pointee->ctypes->scalarValue;
    evidence.pointeeScalarValid = pointee->ctypes->scalarValid;
    mlir::Value pointeeAddress = cdataStorageAddress(*pointee->ctypes);
    mlir::Value pointeeAddressValid =
        cdataStorageAddressValid(*pointee->ctypes);
    if (!pointeeAddress || !pointeeAddressValid ||
        !isKnownTrue(pointeeAddressValid))
      return op.emitError()
             << callable.binding
             << " requires a materialized pointee _CData storage address";
    evidence.addressValue = pointeeAddress;
    evidence.addressValid = pointeeAddressValid;
    keepAliveSource(evidence, *pointee);
    evidence.callRegionBorrow = callable.binding == "ctypes.byref";
    if (evidence.callRegionBorrow) {
      evidence.provenance = RuntimeCtypesEvidence::Provenance::CallRegionBorrow;
      evidence.lifetime = RuntimeCtypesEvidence::Lifetime::CallRegion;
    } else {
      std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
      if (!facts)
        return op.emitError()
               << "ctypes.pointer requires TargetPlatformFacts before "
                  "materializing pointer storage";
      std::optional<CtypesLayout> pointerLayout =
          ctypesLayout("_ctypes._Pointer", facts);
      if (!pointerLayout)
        return op.emitError()
               << "ctypes.pointer has no pointer object layout for "
               << targetFactsLabel(facts);
      mlir::Value pointerInteger =
          coerceNativeInteger(builder, op.getLoc(), pointeeAddress,
                              nativePointerIntegerType(builder, facts));
      evidence.storageAddressValue = addressOfNativeCellAlloca(
          builder, op.getLoc(), pointerInteger, facts);
      evidence.storageAddressValid = constantI1(builder, op.getLoc(), true);
      evidence.ownsNativeStorage = true;
      result.buffer = makeCtypesBufferEvidence(
          builder, op.getLoc(), *pointerLayout, evidence.storageAddressValue,
          evidence.storageAddressValid, *pointee,
          /*writable=*/true);
    }
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (callable.binding == "ctypes.POINTER") {
    if (op.getNumResults() != 1 || sources.size() != 1)
      return op.emitError() << "ctypes.POINTER expects one static ctypes type";
    const RuntimeBundle *pointee = sources.front();
    if (!pointee)
      return op.emitError() << "ctypes.POINTER argument has no evidence";
    std::optional<std::string> pointeeName = ctypesTypeObjectName(*pointee);
    if (!pointeeName)
      return op.emitError()
             << "ctypes.POINTER expects a static ctypes type object";
    mlir::Type pointeeType = ctypesTypeFromBundle(*pointee);
    if (!pointeeType)
      pointeeType = ctypesContractType(context, *pointeeName);
    mlir::Type pointerType =
        py::ContractType::get(context, "_ctypes._Pointer", {pointeeType});
    valueBundles[op.getResult(0)] =
        RuntimeBundle::typeObject(op.getResult(0).getType(), pointerType);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << callable.binding
                        << " has no erased ctypes lowering yet";
}

} // namespace py::runtime_lowering
