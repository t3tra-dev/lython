#include "Runtime/Core/Lowerer.h"

namespace py::runtime_lowering {

namespace {

bool compatibleMemRefView(mlir::Type source, mlir::Type target) {
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source);
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(target);
  return sourceMemRef && targetMemRef &&
         sourceMemRef.getShape() == targetMemRef.getShape() &&
         sourceMemRef.getElementType() == targetMemRef.getElementType() &&
         sourceMemRef.getMemorySpace() == targetMemRef.getMemorySpace();
}

std::string defaultReprTypeName(llvm::StringRef contract) {
  if (contract == "builtins.object")
    return "object";
  if (contract.starts_with("builtins."))
    return contract.rsplit('.').second.str();
  if (!contract.contains('.')) {
    std::string qualified = "__main__.";
    qualified += contract.str();
    return qualified;
  }
  return contract.str();
}

} // namespace

RuntimeValue RuntimeValue::object(mlir::Type contract,
                                  mlir::ValueRange values) {
  RuntimeValue value;
  value.contract = contract;
  value.values.append(values.begin(), values.end());
  return value;
}

std::string RuntimeValue::contractName() const {
  return runtimeContractName(contract);
}

const RuntimeValue *RuntimeObjectEvidence::slot(llvm::StringRef name) const {
  auto found = slots.find(name);
  if (found == slots.end())
    return nullptr;
  return &found->second;
}

void RuntimeObjectEvidence::setSlot(llvm::StringRef name,
                                    const RuntimeValue &value) {
  slots[name] = value;
}

void RuntimeObjectEvidence::eraseSlot(llvm::StringRef name) {
  slots.erase(name);
}

bool RuntimeObjectEvidence::hasFlag(llvm::StringRef name) const {
  return flags.contains(name);
}

void RuntimeObjectEvidence::setFlag(llvm::StringRef name) {
  flags.insert(name);
}

void RuntimeObjectEvidence::eraseFlag(llvm::StringRef name) {
  flags.erase(name);
}

RuntimeBundle RuntimeBundle::object(mlir::Type contract,
                                    mlir::ValueRange values) {
  RuntimeBundle bundle;
  bundle.kind = Kind::Object;
  bundle.contract = contract;
  bundle.objectValue = RuntimeValue::object(contract, values);
  return bundle;
}

RuntimeBundle RuntimeBundle::aggregate(mlir::Type contract,
                                       mlir::ValueRange operands) {
  RuntimeBundle bundle;
  bundle.kind = Kind::Aggregate;
  bundle.contract = contract;
  bundle.aggregateOperands.append(operands.begin(), operands.end());
  return bundle;
}

RuntimeBundle RuntimeBundle::builtinCallable(mlir::Type contract,
                                             llvm::StringRef binding) {
  RuntimeBundle bundle;
  bundle.kind = Kind::BuiltinCallable;
  bundle.contract = contract;
  bundle.binding = binding.str();
  return bundle;
}

RuntimeBundle RuntimeBundle::typeObject(mlir::Type typeContract,
                                        mlir::Type instanceContract) {
  RuntimeBundle bundle;
  bundle.kind = Kind::TypeObject;
  bundle.contract = typeContract;
  bundle.instanceContract = instanceContract;
  return bundle;
}

void RuntimeBundle::copyEvidenceFrom(const RuntimeBundle &source) {
  fieldAliasOwner = source.fieldAliasOwner;
  fieldAliasName = source.fieldAliasName;
  binding = source.binding;
  literalText = source.literalText;
  functionTarget = source.functionTarget;
  closureValues = source.closureValues;
  callableAlternatives = source.callableAlternatives;
  coroutineTarget = source.coroutineTarget;
  coroutineSources = source.coroutineSources;
  primitiveI64 = source.primitiveI64;
  buffer = source.buffer;
  ctypes = source.ctypes;
  objectEvidence = source.objectEvidence;
  fieldBundles = source.fieldBundles;
  sequenceElementBundles = source.sequenceElementBundles;
  sequenceElements = source.sequenceElements;
  sequenceIndices = source.sequenceIndices;
  mappingKeys = source.mappingKeys;
  mappingValues = source.mappingValues;
  mappingPresent = source.mappingPresent;
}

llvm::ArrayRef<mlir::Value> RuntimeBundle::physicalValues() const {
  return objectValue.values;
}

std::string RuntimeBundle::contractName() const {
  if (kind == Kind::Object)
    return objectValue.contractName();
  return runtimeContractName(contract);
}

std::string RuntimeBundle::instanceContractName() const {
  return runtimeContractName(instanceContract);
}

bool RuntimeBundleLowerer::isBuiltinsObjectHeaderType(mlir::Type type) const {
  const RuntimeValueShape *shape = manifest.valueShape("builtins.object");
  return shape && shape->valueTypes.size() == 1 &&
         shape->valueTypes.front() == type;
}

bool RuntimeBundleLowerer::isErasedObjectStorageType(mlir::Type type) const {
  auto memRef = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memRef || memRef.getRank() != 1 ||
      memRef.getDimSize(0) != mlir::ShapedType::kDynamic)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerType>(memRef.getElementType());
  return integer && integer.getWidth() == 64;
}

mlir::FailureOr<mlir::Value> RuntimeBundleLowerer::erasedObjectStorageView(
    mlir::Operation *op, const RuntimeValue &value, mlir::Type targetType) {
  if (value.values.empty())
    return op->emitError() << value.contractName()
                           << " runtime object has no physical storage value";
  mlir::Value storage = value.values.front();
  if (storage.getType() == targetType)
    return storage;

  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(storage.getType());
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(targetType);
  if (!sourceMemRef || !targetMemRef || sourceMemRef.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceMemRef.getElementType() != targetMemRef.getElementType() ||
      sourceMemRef.getMemorySpace() != targetMemRef.getMemorySpace())
    return op->emitError() << value.contractName() << " physical storage "
                           << storage.getType()
                           << " cannot be erased to runtime object storage "
                           << targetType;
  if (targetMemRef.getDimSize(0) != mlir::ShapedType::kDynamic)
    return op->emitError() << "erased runtime object storage target must be "
                              "dynamically sized, got "
                           << targetType;
  return builder
      .create<mlir::memref::CastOp>(op->getLoc(), targetMemRef, storage)
      .getResult();
}

mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::objectHeaderView(mlir::Operation *op,
                                       const RuntimeValue &value) {
  const RuntimeValueShape *shape = manifest.valueShape("builtins.object");
  if (!shape)
    return op->emitError()
           << "runtime manifest has no ABI shape for builtins.object";
  if (shape->valueTypes.size() != 1)
    return op->emitError()
           << "builtins.object ABI must expose exactly one header value";
  if (value.values.empty())
    return op->emitError() << value.contractName()
                           << " runtime object has no physical header value";

  mlir::Type headerType = shape->valueTypes.front();
  mlir::Value header = value.values.front();
  if (header.getType() == headerType)
    return header;
  if (compatibleMemRefView(header.getType(), headerType))
    return builder
        .create<mlir::memref::CastOp>(op->getLoc(), headerType, header)
        .getResult();

  return op->emitError() << value.contractName() << " physical header "
                         << header.getType()
                         << " cannot be viewed as builtins.object header "
                         << headerType;
}

mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::materializeDeadPhysicalValue(mlir::Operation *op,
                                                   mlir::Type type) {
  mlir::Location loc = op->getLoc();
  if (mlir::isa<mlir::IndexType>(type))
    return builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  if (mlir::isa<mlir::IntegerType>(type))
    return builder
        .create<mlir::arith::ConstantOp>(loc, type,
                                         builder.getIntegerAttr(type, 0))
        .getResult();
  if (mlir::isa<mlir::FloatType>(type))
    return builder
        .create<mlir::arith::ConstantOp>(loc, type,
                                         builder.getFloatAttr(type, 0.0))
        .getResult();

  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type)) {
    llvm::SmallVector<std::int64_t, 4> shape(memrefType.getShape().begin(),
                                             memrefType.getShape().end());
    mlir::MemRefType allocType = mlir::MemRefType::get(
        shape, memrefType.getElementType(), mlir::MemRefLayoutAttrInterface{},
        memrefType.getMemorySpace());
    llvm::SmallVector<mlir::Value, 4> dynamicSizes;
    for (std::int64_t extent : shape)
      if (extent == mlir::ShapedType::kDynamic)
        dynamicSizes.push_back(
            builder.create<mlir::arith::ConstantIndexOp>(loc, 1));
    mlir::Value allocated =
        builder.create<mlir::memref::AllocOp>(loc, allocType, dynamicSizes)
            .getResult();
    if (allocated.getType() == type)
      return allocated;
    return builder.create<mlir::memref::CastOp>(loc, type, allocated)
        .getResult();
  }

  return op->emitError() << "cannot materialize a dead runtime placeholder of "
                            "physical type "
                         << type;
}

mlir::FailureOr<RuntimeValue> RuntimeBundleLowerer::materializeDeadObjectValue(
    mlir::Operation *op, mlir::Type contract, llvm::StringRef purpose) {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, contract, purpose);
  if (mlir::failed(valueTypes))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> values;
  values.reserve(valueTypes->size());
  for (mlir::Type valueType : *valueTypes) {
    mlir::FailureOr<mlir::Value> value =
        RuntimeBundleLowerer::materializeDeadPhysicalValue(op, valueType);
    if (mlir::failed(value))
      return mlir::failure();
    values.push_back(*value);
  }
  return RuntimeValue::object(contract, values);
}

mlir::LogicalResult RuntimeBundleLowerer::materializeStringObject(
    mlir::Operation *op, llvm::StringRef text, RuntimeBundle &bundle) {
  mlir::Location loc = op->getLoc();
  mlir::Value bytes = RuntimeBundleLowerer::materializeByteBuffer(loc, text);
  mlir::Value start =
      builder.create<mlir::arith::ConstantIndexOp>(loc, 0).getResult();
  mlir::Value length = builder
                           .create<mlir::arith::ConstantIntOp>(
                               loc, static_cast<std::int64_t>(text.size()), 64)
                           .getResult();
  return RuntimeBundleLowerer::initializeObjectFromRawValues(
      op, runtimeContractType(context, "builtins.str"),
      mlir::ValueRange{bytes, start, length}, bundle);
}

bool RuntimeBundleLowerer::needsDefaultObjectRepr(
    const RuntimeBundle &object) const {
  return object.kind == RuntimeBundle::Kind::Object &&
         manifest.methodCandidates(object.contractName(), "__repr__").empty();
}

mlir::LogicalResult RuntimeBundleLowerer::materializeDefaultObjectRepr(
    mlir::Operation *op, const RuntimeBundle &object, RuntimeBundle &bundle) {
  if (object.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "default object repr requires an object bundle";

  std::optional<RuntimeSymbol> primitive =
      manifest.primitive("builtins.object", "default_repr");
  if (!primitive)
    return op->emitError()
           << "runtime manifest has no builtins.object default_repr primitive";

  builder.setInsertionPoint(op);
  mlir::FailureOr<mlir::Value> header =
      RuntimeBundleLowerer::objectHeaderView(op, object.objectValue);
  if (mlir::failed(header))
    return mlir::failure();

  std::string prefix = "<";
  prefix += defaultReprTypeName(object.contractName());
  prefix += " object at 0x";
  mlir::Value prefixBytes =
      RuntimeBundleLowerer::materializeByteBuffer(op->getLoc(), prefix);
  mlir::Value prefixLength =
      builder
          .create<mlir::arith::ConstantIntOp>(
              op->getLoc(), static_cast<std::int64_t>(prefix.size()), 64)
          .getResult();

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *primitive,
      mlir::ValueRange{*header, prefixBytes, prefixLength});
  return RuntimeBundleLowerer::bundleRuntimeResults(
      op, runtimeContractType(context, "builtins.str"), call, bundle);
}

} // namespace py::runtime_lowering
