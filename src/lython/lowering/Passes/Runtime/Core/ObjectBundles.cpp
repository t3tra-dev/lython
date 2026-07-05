#include "Runtime/Core/Lowerer.h"

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

mlir::LogicalResult
RuntimeBundleLowerer::makeObjectBundle(mlir::Operation *op, mlir::Type contract,
                                       mlir::ValueRange values,
                                       RuntimeBundle &bundle,
                                       bool ownsObject) const {
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
      builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 1, 1)
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

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::materializeObjectBundleForStorage(
    mlir::Operation *op, const RuntimeBundle &bundle, mlir::Type storageContract,
    llvm::StringRef purpose) {
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
        builder.create<mlir::arith::ConstantIntOp>(op->getLoc(), 1, 1)
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
  if (spelling == "none")
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{}, bundle);
  if (spelling == "bool") {
    auto value = dict.getAs<mlir::BoolAttr>("value");
    if (!value)
      return op->emitError() << "bool default value has no payload";
    mlir::Value bit =
        builder.create<mlir::arith::ConstantIntOp>(loc, value.getValue(), 1)
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
        builder.create<mlir::arith::ConstantIntOp>(loc, parsed, 64).getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(op, parameterType,
                                                       integer, bundle);
  }
  if (spelling == "float") {
    auto value = dict.getAs<mlir::FloatAttr>("value");
    if (!value)
      return op->emitError() << "float default value has no payload";
    mlir::Value number = builder
                             .create<mlir::arith::ConstantFloatOp>(
                                 loc, value.getValue(), builder.getF64Type())
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
        builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
    mlir::Value length =
        builder
            .create<mlir::arith::ConstantIntOp>(
                loc, static_cast<std::int64_t>(value.getValue().size()), 64)
            .getResult();
    return RuntimeBundleLowerer::bundleRawObjectValues(
        op, parameterType, mlir::ValueRange{bytes, start, length}, bundle);
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
