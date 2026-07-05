#include "Runtime/Core/Lowerer.h"

namespace py::lowering {
namespace {

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

std::optional<std::int64_t> constantI64Value(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

bool primitiveI64EvidenceEqual(const RuntimeBundle &lhs,
                               const RuntimeBundle &rhs) {
  if (lhs.contractName() != "builtins.int" ||
      rhs.contractName() != "builtins.int" || !lhs.primitiveI64 ||
      !rhs.primitiveI64 || !lhs.primitiveI64->value ||
      !rhs.primitiveI64->value)
    return false;
  if (lhs.primitiveI64->value == rhs.primitiveI64->value)
    return true;
  std::optional<std::int64_t> lhsValue =
      constantI64Value(lhs.primitiveI64->value);
  std::optional<std::int64_t> rhsValue =
      constantI64Value(rhs.primitiveI64->value);
  return lhsValue && rhsValue && *lhsValue == *rhsValue;
}

bool evidenceValueEqual(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  if (lhs.kind != RuntimeBundle::Kind::Object ||
      rhs.kind != RuntimeBundle::Kind::Object)
    return false;
  if (lhs.literalText && rhs.literalText)
    return *lhs.literalText == *rhs.literalText;
  if (primitiveI64EvidenceEqual(lhs, rhs))
    return true;
  return sameRuntimeValueIdentity(lhs.objectValue, rhs.objectValue);
}

bool isMutableStructuralListObject(const RuntimeBundle &bundle) {
  return bundle.contractName() == "builtins.list";
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

bool hasSingleMemrefI64Storage(llvm::ArrayRef<mlir::Value> values,
                               std::int64_t size) {
  if (values.size() != 1)
    return false;
  auto memref = mlir::dyn_cast<mlir::MemRefType>(values.front().getType());
  if (!memref || memref.getRank() != 1 || !memref.hasStaticShape() ||
      memref.getDimSize(0) != size)
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  return element && element.getWidth() == 64;
}

mlir::LogicalResult adjustListRuntimeLength(mlir::Operation *op,
                                            mlir::OpBuilder &builder,
                                            const RuntimeBundle &receiver,
                                            std::int64_t delta) {
  if (receiver.physicalValues().size() < 2)
    return op->emitError()
           << "list runtime object has no physical length metadata";

  mlir::Value meta = receiver.physicalValues()[1];
  if (!isCollectionMetaType(meta.getType()))
    return op->emitError() << "list runtime length metadata has invalid type "
                           << meta.getType();

  mlir::Location loc = op->getLoc();
  mlir::Value slot = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
  mlir::Value current = builder.create<mlir::memref::LoadOp>(loc, meta, slot);
  mlir::Value one = builder.create<mlir::arith::ConstantIntOp>(loc, 1, 64);
  mlir::Value next =
      delta >= 0
          ? builder.create<mlir::arith::AddIOp>(loc, current, one).getResult()
          : builder.create<mlir::arith::SubIOp>(loc, current, one).getResult();
  builder.create<mlir::memref::StoreOp>(loc, next, meta, slot);
  return mlir::success();
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerCall(py::CallOp op) {
  const RuntimeBundle *callable =
      RuntimeBundleLowerer::bundleFor(op.getCallable());
  if (!callable)
    return op.emitError() << "callable has no lowered runtime bundle";
  if (auto method = op->getAttrOfType<mlir::StringAttr>("ly.bound_method"))
    return RuntimeBundleLowerer::lowerBoundMethodCall(op, *callable,
                                                      method.getValue());
  if (callable->boundMethodReceiver && !callable->boundMethodName.empty())
    return RuntimeBundleLowerer::lowerBoundMethodCall(
        op, *callable->boundMethodReceiver, callable->boundMethodName);
  if (callable->kind == RuntimeBundle::Kind::TypeObject)
    return RuntimeBundleLowerer::lowerStaticCtypesTypeObjectCall(op, *callable);
  if (callable->kind != RuntimeBundle::Kind::BuiltinCallable)
    return RuntimeBundleLowerer::lowerObjectCallableCall(op, *callable);
  if (RuntimeBundleLowerer::isStaticCtypesCallable(callable->binding))
    return RuntimeBundleLowerer::lowerStaticCtypesCall(op, *callable);
  std::optional<RuntimeSymbol> builtin =
      manifest.builtinCallable(callable->binding);
  if (!builtin)
    return op.emitError() << "runtime manifest has no builtin callable '"
                          << callable->binding << "'";
  if (builtin->builtinLowering == "method")
    return RuntimeBundleLowerer::lowerBuiltinMethodCall(op, *builtin);
  if (builtin->builtinLowering == "method_sink")
    return RuntimeBundleLowerer::lowerBuiltinMethodSinkCall(op, *builtin);
  if (builtin->builtinLowering == "direct")
    return RuntimeBundleLowerer::lowerDirectBuiltinCall(op, *builtin);
  if (builtin->builtinLowering == "asyncio_sleep")
    return RuntimeBundleLowerer::lowerAsyncioSleepCall(op, *builtin);
  return op.emitError() << "builtin callable '" << callable->binding
                        << "' has unsupported lowering strategy '"
                        << builtin->builtinLowering << "'";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBoundMethodCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  if (receiver.kind == RuntimeBundle::Kind::TypeObject)
    return RuntimeBundleLowerer::lowerStaticCtypesTypeObjectMethodCall(
        op, receiver, methodName);
  if (receiver.kind != RuntimeBundle::Kind::Object)
    return op.emitError() << "bound method receiver must be an object bundle";
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python bound method lowering expects exactly one Python result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  auto receiverIt = valueBundles.find(op.getCallable());
  if (receiverIt != valueBundles.end() &&
      receiverIt->second.contractName() == "_asyncio.Future")
    return RuntimeBundleLowerer::lowerFutureBoundMethod(op, receiverIt->second,
                                                        methodName);
  if (receiver.ctypes &&
      receiver.ctypes->kind == RuntimeCtypesEvidence::Kind::Module)
    return RuntimeBundleLowerer::lowerStaticCtypesModuleCall(op, receiver,
                                                             methodName);

  llvm::SmallVector<const RuntimeBundle *, 8> sources{&receiver};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  if (isMutableStructuralListObject(receiver) &&
      (methodName == "append" || methodName == "remove")) {
    if (sources.size() != 2 || !sources[1] ||
        sources[1]->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "sequence " << methodName
                            << " expects one Python object argument";

    RuntimeBundle updated = receiver;
    if (methodName == "append") {
      builder.setInsertionPoint(op);
      if (mlir::failed(adjustListRuntimeLength(op, builder, receiver, +1)))
        return mlir::failure();
      mlir::FailureOr<RuntimeBundle> payload =
          RuntimeBundleLowerer::materializePayloadObjectBundle(op,
                                                               *sources[1]);
      if (mlir::failed(payload))
        return mlir::failure();
      if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
              op, *payload, "list.append")))
        return mlir::failure();
      if (mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
              op, updated,
              static_cast<unsigned>(updated.sequenceElementBundles.size()),
              *payload)))
        return mlir::failure();
      RuntimeBundle stored = payload->withObjectOwnership(
          ownership::logicalOwnershipKind(payload->objectValue.contract,
                                                  /*ownsObject=*/true));
      updated.sequenceElements.push_back(stored.objectValue);
      updated.sequenceElementBundles.push_back(
          std::make_shared<RuntimeBundle>(std::move(stored)));
    } else {
      std::optional<unsigned> foundIndex;
      for (auto [index, element] : llvm::enumerate(updated.sequenceElements)) {
        if (index < updated.sequenceElementBundles.size() &&
            updated.sequenceElementBundles[index]) {
          if (evidenceValueEqual(*updated.sequenceElementBundles[index],
                                 *sources[1])) {
            foundIndex = static_cast<unsigned>(index);
            break;
          }
          continue;
        }
        if (sameRuntimeValueIdentity(element, sources[1]->objectValue)) {
          foundIndex = static_cast<unsigned>(index);
          break;
        }
      }
      if (!foundIndex) {
        builder.setInsertionPoint(op);
        if (mlir::failed(emitRuntimeException(op, "builtins.ValueError",
                                              "list.remove(x): x not in list")))
          return mlir::failure();
      } else {
        unsigned index = *foundIndex;
        builder.setInsertionPoint(op);
        if (mlir::failed(adjustListRuntimeLength(op, builder, receiver, -1)))
          return mlir::failure();
        if (index < updated.sequenceElementBundles.size() &&
            updated.sequenceElementBundles[index]) {
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, *updated.sequenceElementBundles[index], "list.remove")))
            return mlir::failure();
        } else if (index < updated.sequenceElements.size()) {
          const RuntimeValue &oldElement = updated.sequenceElements[index];
          if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
                  op, oldElement.contract, oldElement.values, "list.remove")))
            return mlir::failure();
        }
        unsigned oldSize =
            static_cast<unsigned>(updated.sequenceElementBundles.size());
        updated.sequenceElements.erase(updated.sequenceElements.begin() +
                                       index);
        if (index < updated.sequenceElementBundles.size())
          updated.sequenceElementBundles.erase(
              updated.sequenceElementBundles.begin() + index);
        for (unsigned position = index,
                      end = static_cast<unsigned>(
                          updated.sequenceElementBundles.size());
             position < end; ++position) {
          if (updated.sequenceElementBundles[position] &&
              mlir::failed(RuntimeBundleLowerer::storeSequencePayloadElement(
                  op, updated, position,
                  *updated.sequenceElementBundles[position])))
            return mlir::failure();
        }
        if (oldSize > 0 &&
            mlir::failed(RuntimeBundleLowerer::clearSequencePayloadElement(
                op, updated, oldSize - 1)))
          return mlir::failure();
      }
    }

    if (mlir::failed(RuntimeBundleLowerer::writeBackFieldAlias(op, updated)))
      return mlir::failure();
    valueBundles[op.getCallable()] = updated;
    if (mlir::failed(assignObjectBundle(
            op, op.getResult(0), runtimeContractType(context, "types.NoneType"),
            mlir::ValueRange{})))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "__repr__" && sources.size() == 1 &&
      RuntimeBundleLowerer::needsDefaultObjectRepr(receiver)) {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeDefaultObjectRepr(
            op, receiver, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "__await__" &&
      mlir::isa<py::ProtocolType>(receiver.contract) &&
      hasSingleMemrefI64Storage(receiver.physicalValues(), 5)) {
    mlir::Type payload = runtimeContractType(context, "builtins.object");
    if (auto protocol = mlir::dyn_cast<py::ProtocolType>(receiver.contract))
      if (protocol.getProtocolName() == "Awaitable" &&
          protocol.getArguments().size() == 1)
        payload = protocol.getArguments().front();
    mlir::Type object = runtimeContractType(context, "builtins.object");
    mlir::Type coroutineType =
        py::ContractType::get(context, "types.CoroutineType",
                              {object, object, payload});
    RuntimeBundle concrete = receiver;
    concrete.contract = coroutineType;
    concrete.objectValue.contract = coroutineType;
    llvm::SmallVector<const RuntimeBundle *, 1> coroutineSources{&concrete};
    if (mlir::failed(lowerManifestMethodResult(
            op, op.getResult(0), concrete, methodName, coroutineSources,
            /*allowUnusedSources=*/false,
            /*preferManifestObjectResult=*/true)))
      return mlir::failure();
    erase.push_back(op);
    return mlir::success();
  }

  if (methodName == "__await__" &&
      mlir::isa<py::ProtocolType>(receiver.contract)) {
    builder.setInsertionPoint(op);
    mlir::FailureOr<RuntimeValue> dead =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, op.getResult(0).getType(), "erased Awaitable.__await__");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[op.getResult(0)] =
        RuntimeBundle::object(dead->contract, dead->values);
    erase.push_back(op);
    return mlir::success();
  }

  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(0), receiver, methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerObjectCallableCall(py::CallOp op,
                                              const RuntimeBundle &callable) {
  if (callable.kind != RuntimeBundle::Kind::Object)
    return op.emitError()
           << "callable is not an object bundle with a __call__ contract";
  if (callable.ctypes &&
      callable.ctypes->kind == RuntimeCtypesEvidence::Kind::Symbol)
    return RuntimeBundleLowerer::lowerStaticCtypesNativeCall(op, callable);
  if (!callable.functionTarget.empty())
    return RuntimeBundleLowerer::lowerFunctionTargetCall(op, callable);
  if (runtimeContractName(callable.contract) == "builtins.function")
    return RuntimeBundleLowerer::lowerIndirectFunctionObjectCall(op, callable);
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python callable lowering expects exactly one Python result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 8> sources{&callable};
  llvm::SmallVector<RuntimeBundle, 8> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(
          op, op.getPosargs(), "positional args", sources, &unpackedSources)))
    return mlir::failure();

  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(0), callable, "__call__", sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
