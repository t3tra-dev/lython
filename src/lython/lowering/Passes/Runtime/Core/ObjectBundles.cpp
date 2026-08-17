#include "Runtime/Core/Lowerer.h"

#include "Runtime/Core/OwnedLocalMarker.h"

namespace py::lowering {

mlir::LogicalResult RuntimeBundleLowerer::validateObjectShape(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values) const {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, contract,
                                                 "runtime bundle");
  if (mlir::failed(expectedTypes))
    return mlir::failure();
  if (expectedTypes->size() != values.size())
    return op->emitError() << "runtime bundle for " << contract << " has "
                           << values.size() << " values, but ABI expects "
                           << expectedTypes->size();
  for (auto [index, value] : llvm::enumerate(values)) {
    mlir::Type expected = (*expectedTypes)[index];
    if (value.getType() != expected)
      return op->emitError() << "runtime bundle value " << index << " for "
                             << contract << " has type " << value.getType()
                             << ", but ABI expects " << expected;
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::makeObjectBundle(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle, bool ownsObject) const {
  return RuntimeBundleLowerer::makeObjectBundleWithOwnership(
      op, contract, values, bundle,
      ownership::logicalOwnershipKind(contract, ownsObject));
}

mlir::LogicalResult RuntimeBundleLowerer::makeObjectBundleWithOwnership(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle, ownership::OwnershipKind ownership) const {
  if (mlir::failed(validateObjectShape(op, contract, values)))
    return mlir::failure();
  bundle = RuntimeBundle::objectWithOwnership(contract, values, ownership);
  return mlir::success();
}

bool RuntimeBundleLowerer::ownedLocalObjectMarkerFollowsExpansion(
    mlir::Value logicalValue) const {
  return logicalValue && logicalValue.getDefiningOp<py::NewOp>() != nullptr;
}

mlir::LogicalResult
RuntimeBundleLowerer::markOwnedLocalObjectBundle(mlir::Operation *op,
                                                 mlir::Value logicalValue,
                                                 const RuntimeBundle &bundle) {
  if (!RuntimeBundleLowerer::ownedLocalObjectMarkerFollowsExpansion(
          logicalValue))
    return mlir::success();
  if (bundle.kind != RuntimeBundle::Kind::Object ||
      bundle.objectValue.values.empty())
    return mlir::success();

  std::string contractName = bundle.contractName();
  bool ownsObject =
      bundle.objectValue.ownership == ownership::OwnershipKind::Own;
  if (py::ClassOp classOp =
          RuntimeBundleLowerer::classForContract(bundle.objectValue.contract)) {
    ownsObject = true;
    if (contractName.empty())
      contractName = classOp.getSymName().str();
  }
  if (!ownsObject || contractName.empty())
    return mlir::success();

  if (auto existing = ownedLocalObjectMarkers.find(logicalValue);
      existing != ownedLocalObjectMarkers.end()) {
    if (!existing->second->use_empty())
      return existing->second->emitError()
             << "owned local object marker unexpectedly has users";
    existing->second->erase();
    ownedLocalObjectMarkers.erase(existing);
  }

  builder.setInsertionPoint(op);
  mlir::UnrealizedConversionCastOp marker = mintOwnedLocalMarker(
      builder, op->getLoc(), bundle.objectValue.values, contractName);
  ownedLocalObjectMarkers[logicalValue] = marker;
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::makePrimitiveI64Bundle(
    mlir::Operation *op, mlir::Type contract, mlir::Value value,
    mlir::Value valid, RuntimeBundle &bundle) const {
  if (runtimeContractName(contract) != "builtins.int" ||
      !value.getType().isInteger(64) || !valid.getType().isInteger(1))
    return op->emitError()
           << "primitive i64 bundle requires builtins.int, i64 value, and i1 "
              "valid flag";
  bundle = RuntimeBundle::object(contract, mlir::ValueRange{});
  bundle.primitiveI64 = RuntimePrimitiveI64Evidence{value, valid};
  return mlir::success();
}

void RuntimeBundleLowerer::seedPrimitiveI64Evidence(mlir::Operation *op,
                                                    mlir::Type contract,
                                                    mlir::ValueRange rawValues,
                                                    RuntimeBundle &bundle) {
  if (runtimeContractName(contract) != "builtins.int" ||
      rawValues.size() != 1 || !rawValues.front().getType().isInteger(64))
    return;
  mlir::Value valid =
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1)
          .getResult();
  bundle.primitiveI64 = RuntimePrimitiveI64Evidence{rawValues.front(), valid};
}

bool RuntimeBundleLowerer::hasLazyPrimitiveI64Object(
    const RuntimeBundle &bundle) const {
  return bundle.kind == RuntimeBundle::Kind::Object &&
         bundle.contractName() == "builtins.int" &&
         bundle.physicalValues().empty() && bundle.primitiveI64 &&
         bundle.primitiveI64->value &&
         bundle.primitiveI64->value.getType().isInteger(64) &&
         bundle.primitiveI64->valid &&
         bundle.primitiveI64->valid.getType().isInteger(1);
}

bool RuntimeBundleLowerer::canMaterializePrimitiveI64Object(
    const RuntimeBundle &bundle) const {
  return RuntimeBundleLowerer::hasLazyPrimitiveI64Object(bundle);
}

bool RuntimeBundleLowerer::hasPrimitiveI64Evidence(
    const RuntimeBundle *bundle) const {
  return bundle && bundle->kind == RuntimeBundle::Kind::Object &&
         bundle->contractName() == "builtins.int" && bundle->primitiveI64 &&
         bundle->primitiveI64->value &&
         bundle->primitiveI64->value.getType().isInteger(64) &&
         bundle->primitiveI64->valid &&
         bundle->primitiveI64->valid.getType().isInteger(1);
}

bool RuntimeBundleLowerer::allSourcesHavePrimitiveI64Evidence(
    llvm::ArrayRef<const RuntimeBundle *> sources) const {
  return llvm::all_of(sources, [&](const RuntimeBundle *source) {
    return RuntimeBundleLowerer::hasPrimitiveI64Evidence(source);
  });
}

mlir::FailureOr<RuntimeValue>
RuntimeBundleLowerer::materializePrimitiveI64Object(
    mlir::Operation *op, const RuntimeBundle &bundle) {
  builder.setInsertionPoint(op);
  return RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
      op, bundle);
}

mlir::FailureOr<RuntimeValue>
RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
    mlir::Operation *op, const RuntimeBundle &bundle) {
  if (!RuntimeBundleLowerer::canMaterializePrimitiveI64Object(bundle))
    return op->emitError()
           << "bundle has no materializable primitive i64 object";
  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("builtins.int", "__new__");
  if (!initializer)
    return op->emitError() << "runtime manifest has no builtins.int.__new__";
  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *initializer, mlir::ValueRange{bundle.primitiveI64->value});
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> objectTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, bundle.objectValue.contract, "materialized primitive i64 object");
  if (mlir::failed(objectTypes))
    return mlir::failure();
  if (call.getNumResults() < objectTypes->size())
    return op->emitError()
           << "builtins.int.__new__ returned too few object ABI values";
  llvm::SmallVector<mlir::Value, 4> objectValues;
  objectValues.reserve(objectTypes->size());
  for (unsigned index = 0, end = static_cast<unsigned>(objectTypes->size());
       index < end; ++index)
    objectValues.push_back(call.getResult(index));
  return RuntimeValue::object(bundle.objectValue.contract, objectValues);
}

mlir::FailureOr<RuntimeValue>
RuntimeBundleLowerer::materializeObjectEvidenceValue(
    mlir::Operation *op, const RuntimeBundle &bundle,
    llvm::StringRef purpose) {
  if (bundle.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << purpose << " requires an object bundle";
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(bundle))
    return RuntimeBundleLowerer::materializePrimitiveI64Object(op, bundle);

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expected =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, bundle.objectValue.contract,
                                                 purpose);
  if (mlir::failed(expected))
    return mlir::failure();
  if (bundle.objectValue.values.size() != expected->size())
    return op->emitError() << purpose << " has "
                           << bundle.objectValue.values.size()
                           << " physical values, but contract expects "
                           << expected->size();
  return bundle.objectValue;
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::materializeObjectBundleForStorage(
    mlir::Operation *op, const RuntimeBundle &bundleRef,
    mlir::Type storageContract, llvm::StringRef purpose) {
  // Copies: this function inserts into `valueBundles` and then keeps reading its
  // operand bundles, and the caller's arguments are references INTO that
  // DenseMap -- an insertion that rehashes moves the entry and every later read
  // is freed memory. Found as a live defect on `lowerBoundMethodCall`'s receiver
  // (see CallableOps.cpp); these are the rest of the same audit. Neither of
  // these keys is ever rewritten here, so the copy changes nothing else.
  RuntimeBundle bundle = bundleRef;
  if (bundle.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << purpose << " requires an object bundle";

  RuntimeBundle result = bundle;
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(result)) {
    mlir::FailureOr<RuntimeValue> materialized =
        RuntimeBundleLowerer::materializePrimitiveI64Object(op, result);
    if (mlir::failed(materialized))
      return mlir::failure();
    result.objectValue = *materialized;
  }

  // A member arriving where a UNION slot is expected is injected, which is the
  // conversion `union.wrap` performs and the reason a store into
  // `self.v: Optional[str]` used to be refused with "attribute value ABI has 2
  // values, but storage expects 3": the slot is a tag plus the widest member's
  // lanes, and a str arrives as itself.
  //
  // ⭐ The active member has to be recorded, and that is what the earlier
  // attempt at this widening was missing. Whoever hands the slot its reference
  // asks the bundle which member holds the token; a union assembled here with
  // no answer left the store handing over a reference it could not name, and
  // the program advanced from this check to "released owned resource from
  // @LyLong_FromI64 is used after release" -- read at the time as the widening
  // being insufficient. It was the same omission `lowerUnionWrap` had.
  if (auto storageUnion = mlir::dyn_cast<py::UnionType>(storageContract);
      storageUnion && !mlir::isa<py::UnionType>(result.objectValue.contract)) {
    builder.setInsertionPoint(op);
    llvm::SmallVector<mlir::Value, 8> widened;
    RuntimeBundle lanesSource;
    if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
            op, storageUnion, result, result.objectValue.contract, widened,
            &lanesSource)))
      return mlir::failure();
    RuntimeBundle injected;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, storageContract, widened, injected)))
      return mlir::failure();
    injected.copyEvidenceFrom(result);
    injected.unionActiveMember =
        std::make_shared<RuntimeBundle>(std::move(lanesSource));
    result = std::move(injected);
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, storageContract, purpose);
  if (mlir::failed(expectedTypes))
    return mlir::failure();
  if (expectedTypes->size() != result.physicalValues().size())
    return op->emitError() << purpose << " has "
                           << result.physicalValues().size()
                           << " values, but storage expects "
                           << expectedTypes->size();
  return result;
}

bool RuntimeBundleLowerer::objectShapeMatches(llvm::StringRef contract,
                                              mlir::ValueRange values) const {
  const RuntimeValueShape *shape = manifest.valueShape(contract);
  if (!shape)
    return false;
  if (shape->valueTypes.size() != values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value.getType() != shape->valueTypes[index])
      return false;
  return true;
}

bool RuntimeBundleLowerer::rawValuesMatchRuntimeInputs(
    const RuntimeSymbol &symbol, mlir::ValueRange values) const {
  mlir::func::FuncOp function = symbol.function;
  mlir::FunctionType functionType = function.getFunctionType();
  if (functionType.getNumInputs() != values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(values))
    if (value.getType() != functionType.getInput(index))
      return false;
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::initializeObjectFromRawValues(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle, bool emitErrors) {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty()) {
    if (emitErrors)
      return op->emitError()
             << "runtime initializer target has no concrete contract";
    return mlir::failure();
  }

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer(contractName, "__new__");
  if (!initializer) {
    if (emitErrors)
      return op->emitError()
             << "runtime manifest has no " << contractName << ".__new__";
    return mlir::failure();
  }
  if (!rawValuesMatchRuntimeInputs(*initializer, values)) {
    if (emitErrors)
      return op->emitError() << "runtime initializer " << contractName
                             << ".__new__ cannot accept raw input values "
                             << describeValueTypes(values);
    return mlir::failure();
  }

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *initializer, values);
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, contract, call.getResults(), bundle)))
    return mlir::failure();
  RuntimeBundleLowerer::seedPrimitiveI64Evidence(op, contract, values, bundle);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bundleRawObjectValues(
    mlir::Operation *op, mlir::Type contract, mlir::ValueRange values,
    RuntimeBundle &bundle) {
  std::string contractName = runtimeContractName(contract);
  if (contractName.empty())
    return op->emitError()
           << "default argument has no concrete runtime contract";

  mlir::Type concreteContract = runtimeContractType(context, contractName);
  if (contractName == "builtins.int" && values.size() == 1 &&
      values.front().getType().isInteger(64)) {
    mlir::Value valid =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1)
            .getResult();
    return RuntimeBundleLowerer::makePrimitiveI64Bundle(
        op, concreteContract, values.front(), valid, bundle);
  }
  if (objectShapeMatches(contractName, values))
    return RuntimeBundleLowerer::makeObjectBundle(op, concreteContract, values,
                                                  bundle);
  if (mlir::succeeded(initializeObjectFromRawValues(
          op, concreteContract, values, bundle, /*emitErrors=*/false)))
    return mlir::success();
  return RuntimeBundleLowerer::makeObjectBundle(op, concreteContract, values,
                                                bundle);
}

mlir::LogicalResult RuntimeBundleLowerer::materializeDefaultValue(
    mlir::Operation *op, mlir::Type parameterType, mlir::Attribute attr,
    RuntimeBundle &bundle) {
  auto dict = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(attr);
  if (!dict)
    return op->emitError() << "callable default value metadata is malformed";
  auto kind = dict.getAs<mlir::StringAttr>("kind");
  if (!kind)
    return op->emitError() << "callable default value has no kind";

  mlir::Location loc = op->getLoc();
  llvm::StringRef spelling = kind.getValue();
  if (spelling == "none") {
    // A None default's payload is always the NoneType singleton even when
    // the parameter is an optional union (the union has no contract name of
    // its own for bundleRawObjectValues to resolve).
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, runtimeContractType(context, "types.NoneType"), mlir::ValueRange{},
        bundle);
  }
  if (spelling == "bool") {
    auto value = dict.getAs<mlir::BoolAttr>("value");
    if (!value)
      return op->emitError() << "bool default value has no payload";
    mlir::Value bit =
        mlir::arith::ConstantIntOp::create(builder, loc, value.getValue(), 1)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType, bit,
                                                       bundle);
  }
  if (spelling == "int") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "int default value has no payload";
    std::int64_t parsed = 0;
    if (value.getValue().getAsInteger(10, parsed))
      return op->emitError()
             << "integer default value is outside the lowered i64 path";
    mlir::Value integer =
        mlir::arith::ConstantIntOp::create(builder, loc, parsed, 64)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType,
                                                       integer, bundle);
  }
  if (spelling == "float") {
    auto value = dict.getAs<mlir::FloatAttr>("value");
    if (!value)
      return op->emitError() << "float default value has no payload";
    mlir::Value number =
        mlir::arith::ConstantFloatOp::create(builder, loc, builder.getF64Type(),
                                             value.getValue())
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType,
                                                       number, bundle);
  }
  if (spelling == "str") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "str default value has no payload";
    mlir::Value bytes =
        RuntimeBundleLowerer::materializeByteBuffer(loc, value.getValue());
    mlir::Value start =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    mlir::Value length =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(value.getValue().size()),
            64)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{bytes, start, length}, bundle);
  }
  if (spelling == "bytes") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "bytes default value has no payload";
    mlir::Value bytes =
        RuntimeBundleLowerer::materializeByteBuffer(loc, value.getValue());
    mlir::Value start =
        mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    mlir::Value length =
        mlir::arith::ConstantIntOp::create(
            builder, loc, static_cast<std::int64_t>(value.getValue().size()),
            64)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{bytes, start, length}, bundle);
  }
  if (spelling == "global") {
    // R6 definition-time defaults: the value was evaluated once when
    // __main__ reached the def statement and parked in a module-lifetime
    // object-global cell; the omitted-argument call site reads (and
    // retains) the shared value instead of re-evaluating.
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "global default value has no cell name";
    llvm::StringRef cell = value.getValue();
    std::string contractName = runtimeContractName(parameterType);
    if (contractName == "builtins.int") {
      // ⛔ A default cell is NOT a module global, even though both are
      // py.global.get/set: this population is never marked `ly.global.boxed`,
      // so an int default stays in the native word cell and is read back as
      // the primitive lane. Boxing it along with the module globals made
      // `default_once` print 0 for a default that must evaluate once.
      mlir::Value raw = RuntimeBundleLowerer::loadNativeGlobalWord(op, cell);
      mlir::Value valid =
          mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
      return RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"), raw, valid,
          bundle);
    }
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, parameterType,
                                                   "default cell ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();
    llvm::SmallVector<mlir::Value, 8> values;
    if (mlir::failed(RuntimeBundleLowerer::loadObjectGlobalValues(
            op, cell, *valueTypes, values)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, parameterType, values, "default.cell")))
      return mlir::failure();
    return RuntimeBundleLowerer::makeObjectBundleWithOwnership(
        op, parameterType, values, bundle, ownership::OwnershipKind::Own);
  }
  if (spelling == "provider") {
    // Expression defaults (user-type constructors etc.) evaluate through a
    // zero-argument provider function synthesized at the def site.
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (!value)
      return op->emitError() << "provider default value has no symbol";
    auto provider =
        module.lookupSymbol<mlir::func::FuncOp>(value.getValue());
    if (!provider)
      return op->emitError() << "default provider '" << value.getValue()
                             << "' is missing from the module";
    auto callOp = mlir::dyn_cast<py::CallOp>(op);
    if (!callOp)
      return op->emitError()
             << "expression default requires a direct call site";
    mlir::FailureOr<mlir::func::CallOp> call =
        RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
            callOp, provider, value.getValue(), {});
    if (mlir::failed(call))
      return mlir::failure();
    mlir::Type resultType = parameterType;
    if (runtimeContractName(resultType).empty()) {
      if (py::CallableType callable = callableTypeOf(provider))
        if (callable.getResultTypes().size() == 1)
          resultType = callable.getResultTypes().front();
    }
    if (RuntimeBundleLowerer::hasPrimitiveI64ABI(resultType) &&
        call->getNumResults() == 2 &&
        call->getResult(0).getType().isInteger(64) &&
        call->getResult(1).getType().isInteger(1))
      return RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, runtimeContractName(resultType)),
          call->getResult(0), call->getResult(1), bundle);
    // The provider returns through the ordinary function-target ABI, which
    // may append primitive-i64 evidence lanes after the object values (the
    // int hybrid): consume it exactly like any other call result instead of
    // assuming the raw result list IS the object shape.
    return RuntimeBundleLowerer::consumeFunctionTargetCallResult(
        op, value.getValue(), *call, resultType, {},
        /*applyReturnedSummaries=*/false, "default provider result ABI",
        bundle);
  }
  if (spelling == "unsupported") {
    auto value = dict.getAs<mlir::StringAttr>("value");
    if (value)
      return op->emitError()
             << "unsupported callable default expression " << value;
    return op->emitError() << "unsupported callable default expression";
  }
  return op->emitError() << "unknown callable default value kind '" << spelling
                         << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::assignObjectBundle(mlir::Operation *op, mlir::Value value,
                                         mlir::Type contract,
                                         mlir::ValueRange values) {
  RuntimeBundle bundle;
  if (mlir::failed(makeObjectBundle(op, contract, values, bundle)))
    return mlir::failure();
  valueBundles[value] = std::move(bundle);
  return mlir::success();
}

mlir::FailureOr<llvm::StringRef>
RuntimeBundleLowerer::requireMethodTarget(mlir::Operation *op,
                                          mlir::FlatSymbolRefAttr target,
                                          llvm::StringRef expectedName) const {
  if (target)
    return target.getValue();
  return op->emitError() << "resolved special-method op for " << expectedName
                         << " has no manifest method target";
}

} // namespace py::lowering
