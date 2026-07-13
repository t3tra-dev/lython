#include "Ownership.h"
#include "Runtime/Core/Lowerer.h"

#include "PyProtocols.h"
#include "PyTypeObject.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include <cctype>
#include <limits>

namespace py::lowering {

namespace {

namespace own = py::ownership;
constexpr llvm::StringLiteral kReleaseStorageToZeroName{
    "LyObject_ReleaseStorageToZero"};
constexpr unsigned kPrimitiveFieldSlotBase = 4;
constexpr unsigned kPrimitiveFieldSlotLimit = 16;

std::string sanitizeSymbolComponent(llvm::StringRef text) {
  std::string result;
  result.reserve(text.size());
  for (char ch : text) {
    unsigned char byte = static_cast<unsigned char>(ch);
    result.push_back(std::isalnum(byte) ? ch : '_');
  }
  return result;
}

std::string typeSymbolComponent(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream os(text);
  type.print(os);
  return sanitizeSymbolComponent(os.str());
}

mlir::MemRefType concreteDeadMemRefType(mlir::MemRefType type) {
  llvm::SmallVector<std::int64_t, 4> shape(type.getShape().begin(),
                                           type.getShape().end());
  for (std::int64_t &extent : shape)
    if (extent == mlir::ShapedType::kDynamic)
      extent = 1;
  return mlir::MemRefType::get(shape, type.getElementType(),
                               mlir::MemRefLayoutAttrInterface{},
                               type.getMemorySpace());
}

mlir::Attribute deadMemRefInitialValue(mlir::OpBuilder &builder,
                                       mlir::MemRefType type,
                                       bool objectHeader) {
  auto tensorType =
      mlir::RankedTensorType::get(type.getShape(), type.getElementType());
  mlir::Type elementType = type.getElementType();
  if (objectHeader) {
    auto integer = mlir::dyn_cast<mlir::IntegerType>(elementType);
    if (type.getRank() == 1 && type.hasStaticShape() &&
        type.getDimSize(0) >= 2 && integer && integer.getWidth() == 64) {
      llvm::SmallVector<mlir::Attribute, 16> values;
      values.reserve(static_cast<unsigned>(type.getDimSize(0)));
      values.push_back(builder.getIntegerAttr(
          integer, std::numeric_limits<std::int64_t>::max()));
      values.push_back(builder.getIntegerAttr(integer, 0));
      for (std::int64_t index = 2; index < type.getDimSize(0); ++index)
        values.push_back(builder.getIntegerAttr(integer, 0));
      return mlir::DenseElementsAttr::get(tensorType, values);
    }
  }

  mlir::Attribute zero = builder.getZeroAttr(elementType);
  if (!zero)
    return {};
  return mlir::DenseElementsAttr::get(tensorType, zero);
}

bool compatibleMemRefView(mlir::Type source, mlir::Type target) {
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source);
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(target);
  return sourceMemRef && targetMemRef &&
         sourceMemRef.getShape() == targetMemRef.getShape() &&
         sourceMemRef.getElementType() == targetMemRef.getElementType() &&
         sourceMemRef.getMemorySpace() == targetMemRef.getMemorySpace();
}

mlir::LogicalResult zeroInitializeMemRef(mlir::OpBuilder &builder,
                                         mlir::Location loc,
                                         mlir::Value memref) {
  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(memref.getType());
  if (!memrefType || memrefType.getRank() != 1)
    return mlir::failure();

  auto zeroAttr = mlir::dyn_cast_if_present<mlir::TypedAttr>(
      builder.getZeroAttr(memrefType.getElementType()));
  if (!zeroAttr)
    return mlir::failure();

  mlir::Value zeroIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value upper =
      memrefType.hasStaticShape()
          ? mlir::arith::ConstantIndexOp::create(builder, loc,
                                                 memrefType.getDimSize(0))
                .getResult()
          : mlir::memref::DimOp::create(builder, loc, memref, zeroIndex)
                .getResult();
  mlir::Value step = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value zeroValue = mlir::arith::ConstantOp::create(
      builder, loc, memrefType.getElementType(), zeroAttr);

  mlir::scf::ForOp loop =
      mlir::scf::ForOp::create(builder, loc, zeroIndex, upper, step);
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(loop.getBody());
  mlir::memref::StoreOp::create(builder, loc, zeroValue, memref,
                                loop.getInductionVar());
  return mlir::success();
}

bool isRankOneI64MemRef(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  return element && element.getWidth() == 64;
}

mlir::FailureOr<mlir::Value> viewRankOneI64Prefix(mlir::Operation *op,
                                                  mlir::OpBuilder &builder,
                                                  mlir::Value source,
                                                  mlir::Type targetType,
                                                  llvm::StringRef label) {
  if (source.getType() == targetType)
    return source;
  if (compatibleMemRefView(source.getType(), targetType))
    return mlir::memref::CastOp::create(builder, op->getLoc(), targetType,
                                        source)
        .getResult();

  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(source.getType());
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(targetType);
  if (!sourceMemRef || !targetMemRef || sourceMemRef.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceMemRef.getElementType() != targetMemRef.getElementType() ||
      sourceMemRef.getMemorySpace() != targetMemRef.getMemorySpace())
    return op->emitError() << label << " " << source.getType()
                           << " cannot be viewed as " << targetType;

  if (sourceMemRef.hasStaticShape() && targetMemRef.hasStaticShape() &&
      sourceMemRef.getDimSize(0) >= targetMemRef.getDimSize(0)) {
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
        builder.getIndexAttr(targetMemRef.getDimSize(0))};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
    llvm::SmallVector<int64_t, 1> resultShape{targetMemRef.getDimSize(0)};
    auto inferredType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultShape, sourceMemRef, offsets, sizes, strides));
    mlir::Value view =
        mlir::memref::SubViewOp::create(builder, op->getLoc(), inferredType,
                                        source, offsets, sizes, strides)
            .getResult();
    if (view.getType() == targetMemRef)
      return view;
    return mlir::memref::CastOp::create(builder, op->getLoc(), targetMemRef,
                                        view)
        .getResult();
  }

  return op->emitError() << label << " " << source.getType()
                         << " cannot be viewed as " << targetType;
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

bool declaredRuntimeContractMatchesClass(mlir::ModuleOp module,
                                         llvm::StringRef className) {
  auto contracts =
      module->getAttrOfType<mlir::ArrayAttr>(kManifestContractsAttr);
  if (!contracts)
    return false;
  for (mlir::Attribute attr : contracts) {
    auto contract = mlir::dyn_cast<mlir::StringAttr>(attr);
    if (!contract)
      continue;
    llvm::StringRef value = contract.getValue();
    if (value == className)
      return true;
    llvm::StringRef shortName = value.rsplit('.').second;
    if (!shortName.empty() && shortName == className)
      return true;
  }
  return false;
}

void appendClassContractCandidate(llvm::SmallVectorImpl<std::string> &out,
                                  llvm::StringRef candidate) {
  if (candidate.empty())
    return;
  if (llvm::any_of(
          out, [&](llvm::StringRef existing) { return existing == candidate; }))
    return;
  out.push_back(candidate.str());
}

llvm::SmallVector<std::string, 8>
classContractCandidates(llvm::StringRef className) {
  llvm::SmallVector<std::string, 8> candidates;
  appendClassContractCandidate(candidates, className);
  if (className.contains('.'))
    return candidates;
  appendClassContractCandidate(candidates,
                               (llvm::Twine("builtins.") + className).str());
  appendClassContractCandidate(candidates,
                               (llvm::Twine("types.") + className).str());
  appendClassContractCandidate(candidates,
                               (llvm::Twine("_asyncio.") + className).str());
  appendClassContractCandidate(candidates,
                               (llvm::Twine("asyncio.") + className).str());
  appendClassContractCandidate(candidates,
                               (llvm::Twine("contextlib.") + className).str());
  return candidates;
}

std::optional<std::string> nominalClassSymbolName(mlir::Operation *op,
                                                  mlir::Type type) {
  std::string name = runtimeContractName(type);
  if (name.empty())
    return std::nullopt;
  if (py::type_object::lookup(op, name))
    return name;
  llvm::StringRef shortName = llvm::StringRef(name).rsplit('.').second;
  if (!shortName.empty() && shortName != name &&
      py::type_object::lookup(op, shortName))
    return shortName.str();
  return name;
}

bool moduleHasDeallocatorForContract(mlir::ModuleOp module,
                                     llvm::StringRef contract) {
  bool found = false;
  module.walk([&](mlir::func::FuncOp function) {
    if (found || !function->hasAttr(kManifestDeallocatorAttr))
      return;
    auto contractAttr =
        function->getAttrOfType<mlir::StringAttr>(kManifestContractAttr);
    found = contractAttr && contractAttr.getValue() == contract;
  });
  return found;
}

std::string sourceClassDeallocatorName(llvm::StringRef className) {
  std::string name = "__ly_dealloc_";
  for (char ch : className) {
    unsigned char byte = static_cast<unsigned char>(ch);
    name.push_back(std::isalnum(byte) ? ch : '_');
  }
  return name;
}

own::OwnershipKind
normalizeRuntimeValueOwnership(mlir::Type contract,
                               own::OwnershipKind requested) {
  if (requested == own::OwnershipKind::Borrow)
    return own::logicalOwnershipKind(contract, /*ownsObject=*/false);
  if (requested == own::OwnershipKind::Own)
    return own::logicalOwnershipKind(contract, /*ownsObject=*/true);
  return own::logicalOwnershipKind(contract, /*ownsObject=*/requested ==
                                                 own::OwnershipKind::Own);
}

mlir::func::FuncOp findObjectStorageReleaseToZero(mlir::ModuleOp module) {
  mlir::func::FuncOp releaseToZero =
      module.lookupSymbol<mlir::func::FuncOp>(kReleaseStorageToZeroName);
  if (!releaseToZero)
    return {};
  mlir::FunctionType type = releaseToZero.getFunctionType();
  if (type.getNumInputs() != 1 || type.getNumResults() != 1 ||
      !type.getResult(0).isInteger(1) || !isRankOneI64MemRef(type.getInput(0)))
    return {};
  return releaseToZero;
}

mlir::FailureOr<mlir::Value> storageForReleaseToZero(mlir::Operation *op,
                                                     mlir::OpBuilder &builder,
                                                     mlir::Value storage,
                                                     mlir::Type targetType) {
  if (storage.getType() == targetType)
    return storage;
  auto sourceMemRef = mlir::dyn_cast<mlir::MemRefType>(storage.getType());
  auto targetMemRef = mlir::dyn_cast<mlir::MemRefType>(targetType);
  if (!sourceMemRef || !targetMemRef || sourceMemRef.getRank() != 1 ||
      targetMemRef.getRank() != 1 ||
      sourceMemRef.getElementType() != targetMemRef.getElementType() ||
      sourceMemRef.getMemorySpace() != targetMemRef.getMemorySpace())
    return op->emitError() << "source class storage " << storage.getType()
                           << " cannot be released through " << targetType;
  return mlir::memref::CastOp::create(builder, op->getLoc(), targetType,
                                      storage)
      .getResult();
}

llvm::SmallVector<mlir::Value, 4>
entryArgumentSlice(mlir::Block &entry, unsigned offset, unsigned count) {
  llvm::SmallVector<mlir::Value, 4> values;
  values.reserve(count);
  for (unsigned index = 0; index < count; ++index)
    values.push_back(entry.getArgument(offset + index));
  return values;
}

void initializeObjectHeader(mlir::OpBuilder &builder, mlir::Location loc,
                            mlir::Value header, std::int64_t refcount,
                            std::int64_t classId) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  if (!memref || memref.getRank() != 1)
    return;
  auto element = mlir::dyn_cast<mlir::IntegerType>(memref.getElementType());
  if (!element || element.getWidth() != 64)
    return;
  if (memref.hasStaticShape() && memref.getDimSize(0) < 2)
    return;

  mlir::Value zeroIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::Value oneIndex = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
  mlir::Value refcountValue =
      mlir::arith::ConstantIntOp::create(builder, loc, refcount, 64);
  mlir::Value classIdValue =
      mlir::arith::ConstantIntOp::create(builder, loc, classId, 64);
  mlir::memref::StoreOp::create(builder, loc, refcountValue, header, zeroIndex);
  mlir::memref::StoreOp::create(builder, loc, classIdValue, header, oneIndex);
  if (!memref.hasStaticShape() || memref.getDimSize(0) >= 4) {
    mlir::Value zeroValue =
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
    mlir::Value payloadHeaderIndex =
        mlir::arith::ConstantIndexOp::create(builder, loc, 2);
    mlir::Value payloadClassIndex =
        mlir::arith::ConstantIndexOp::create(builder, loc, 3);
    mlir::memref::StoreOp::create(builder, loc, zeroValue, header,
                                  payloadHeaderIndex);
    mlir::memref::StoreOp::create(builder, loc, zeroValue, header,
                                  payloadClassIndex);
  }
}

} // namespace

std::optional<std::int64_t>
RuntimeBundleLowerer::runtimeClassIdForClass(py::ClassOp classOp) const {
  if (!classOp)
    return std::nullopt;

  llvm::StringRef className = classOp.getSymName();
  for (const std::string &candidate : classContractCandidates(className))
    if (std::optional<std::int64_t> classId = manifest.classId(candidate))
      return classId;

  if (auto attr =
          classOp->getAttrOfType<mlir::IntegerAttr>(kManifestClassIdAttr))
    return attr.getValue().getSExtValue();

  constexpr std::int64_t kSourceClassIdBase = 1LL << 32;
  mlir::ModuleOp mutableModule =
      const_cast<RuntimeBundleLowerer *>(this)->module;
  std::optional<std::int64_t> result;
  std::int64_t ordinal = 0;
  mutableModule.walk([&](py::ClassOp current) {
    if (result)
      return;
    bool hasDeclaredRuntimeId = current->getAttrOfType<mlir::IntegerAttr>(
                                    kManifestClassIdAttr) != nullptr;
    for (const std::string &candidate :
         classContractCandidates(current.getSymName())) {
      if (manifest.classId(candidate)) {
        hasDeclaredRuntimeId = true;
        break;
      }
    }

    if (current.getOperation() == classOp.getOperation()) {
      result = kSourceClassIdBase + ordinal;
      return;
    }
    if (!hasDeclaredRuntimeId)
      ++ordinal;
  });
  return result;
}

std::optional<std::int64_t>
RuntimeBundleLowerer::runtimeClassIdForContract(mlir::Type type) const {
  std::string contract = runtimeContractName(type);
  if (!contract.empty())
    if (std::optional<std::int64_t> classId = manifest.classId(contract))
      return classId;
  if (py::ClassOp classOp = RuntimeBundleLowerer::classForContract(type))
    return RuntimeBundleLowerer::runtimeClassIdForClass(classOp);
  return std::nullopt;
}

mlir::FailureOr<llvm::SmallVector<std::int64_t, 8>>
RuntimeBundleLowerer::runtimeClassIdsForNominalTarget(
    mlir::Operation *op, mlir::Type targetType) const {
  std::optional<std::string> targetName =
      nominalClassSymbolName(op, targetType);
  if (!targetName)
    return op->emitError() << "class test target has no nominal class: "
                           << targetType;
  if (!py::type_object::lookup(op, *targetName))
    return op->emitError() << "class test target has no class schema: "
                           << *targetName;

  llvm::SmallVector<std::int64_t, 8> ids;
  auto appendId = [&](std::int64_t id) {
    if (!llvm::is_contained(ids, id))
      ids.push_back(id);
  };

  if (std::optional<std::int64_t> direct =
          RuntimeBundleLowerer::runtimeClassIdForContract(targetType))
    appendId(*direct);

  mlir::LogicalResult status = mlir::success();
  mlir::ModuleOp mutableModule =
      const_cast<RuntimeBundleLowerer *>(this)->module;
  mutableModule.walk([&](py::ClassOp classOp) {
    if (mlir::failed(status))
      return;
    llvm::StringRef derivedName = classOp.getSymName();
    bool matches = derivedName == *targetName;
    if (!matches) {
      mlir::FailureOr<bool> subclass =
          py::type_object::isSubclassOf(op, derivedName, *targetName);
      if (mlir::failed(subclass)) {
        status = mlir::failure();
        return;
      }
      matches = *subclass;
    }
    if (!matches)
      return;

    std::optional<std::int64_t> classId =
        RuntimeBundleLowerer::runtimeClassIdForClass(classOp);
    if (!classId) {
      op->emitError() << "class schema '" << classOp.getSymName()
                      << "' has no runtime class id";
      status = mlir::failure();
      return;
    }
    appendId(*classId);
  });

  if (mlir::failed(status))
    return mlir::failure();
  if (ids.empty())
    return op->emitError() << "class test target has no runtime class ids: "
                           << *targetName;
  return ids;
}

RuntimeValue RuntimeValue::object(mlir::Type contract, mlir::ValueRange values,
                                  bool ownsObject) {
  return RuntimeValue::objectWithOwnership(
      contract, values, own::logicalOwnershipKind(contract, ownsObject));
}

RuntimeValue RuntimeValue::objectWithOwnership(mlir::Type contract,
                                               mlir::ValueRange values,
                                               own::OwnershipKind ownership) {
  RuntimeValue value;
  value.contract = contract;
  value.values.append(values.begin(), values.end());
  value.ownership = normalizeRuntimeValueOwnership(contract, ownership);
  return value;
}

std::string RuntimeValue::contractName() const {
  return runtimeContractName(contract);
}

RuntimeValue RuntimeValue::withOwnership(own::OwnershipKind ownership) const {
  RuntimeValue value = *this;
  value.ownership = normalizeRuntimeValueOwnership(contract, ownership);
  return value;
}

RuntimeValue RuntimeValue::withLogicalOwnership(bool ownsObject) const {
  return withOwnership(own::logicalOwnershipKind(contract, ownsObject));
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
  return RuntimeBundle::objectWithOwnership(
      contract, values, own::logicalOwnershipKind(contract, true));
}

RuntimeBundle RuntimeBundle::objectWithOwnership(mlir::Type contract,
                                                 mlir::ValueRange values,
                                                 own::OwnershipKind ownership) {
  RuntimeBundle bundle;
  bundle.kind = Kind::Object;
  bundle.contract = contract;
  bundle.objectValue =
      RuntimeValue::objectWithOwnership(contract, values, ownership);
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
  boundMethodReceiver = source.boundMethodReceiver;
  boundMethodName = source.boundMethodName;
  coroutineTarget = source.coroutineTarget;
  coroutineSources = source.coroutineSources;
  coroutineSourceBundles = source.coroutineSourceBundles;
  generatorTarget = source.generatorTarget;
  generatorSources = source.generatorSources;
  generatorSourceBundles = source.generatorSourceBundles;
  primitiveI64 = source.primitiveI64;
  buffer = source.buffer;
  ctypes = source.ctypes;
  objectEvidence = source.objectEvidence;
  fieldBundles = source.fieldBundles;
  boxedObject = source.boxedObject;
  sequenceElementBundles = source.sequenceElementBundles;
  sequenceElements = source.sequenceElements;
  sequenceIndices = source.sequenceIndices;
  sequenceCapacity = source.sequenceCapacity;
  sequenceEvidenceBacked = source.sequenceEvidenceBacked;
  mappingKeys = source.mappingKeys;
  mappingKeyBundles = source.mappingKeyBundles;
  mappingValues = source.mappingValues;
  mappingValueBundles = source.mappingValueBundles;
  mappingPresent = source.mappingPresent;
  mappingCapacity = source.mappingCapacity;
  mappingEvidenceBacked = source.mappingEvidenceBacked;
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

void RuntimeBundle::setObjectOwnership(own::OwnershipKind ownership) {
  if (kind != Kind::Object)
    return;
  objectValue = objectValue.withOwnership(ownership);
}

void RuntimeBundle::setObjectLogicalOwnership(bool ownsObject) {
  setObjectOwnership(
      own::logicalOwnershipKind(objectValue.contract, ownsObject));
}

RuntimeBundle
RuntimeBundle::withObjectOwnership(own::OwnershipKind ownership) const {
  RuntimeBundle bundle = *this;
  bundle.setObjectOwnership(ownership);
  return bundle;
}

bool RuntimeBundleLowerer::isBuiltinsObjectHandleType(mlir::Type type) const {
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

bool RuntimeBundleLowerer::isBuiltinsObjectContract(mlir::Type type) const {
  return runtimeContractName(type) == "builtins.object";
}

const RuntimeBundle *RuntimeBundleLowerer::concreteObjectForOwnership(
    const RuntimeBundle &bundle) const {
  const RuntimeBundle *current = &bundle;
  for (unsigned depth = 0; depth < 8; ++depth) {
    if (!current || current->kind != RuntimeBundle::Kind::Object)
      return current;
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(current->contract))
      return current;
    if (!current->boxedObject)
      return current;
    current = current->boxedObject.get();
  }
  return current;
}

mlir::FailureOr<RuntimeBundle> RuntimeBundleLowerer::boxRuntimeObject(
    mlir::Operation *op, const RuntimeBundle &source, bool retainPayload) {
  builder.setInsertionPoint(op);
  return RuntimeBundleLowerer::boxRuntimeObjectAtCurrentInsertion(
      op, source, retainPayload);
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::boxRuntimeObjectAtCurrentInsertion(
    mlir::Operation *op, const RuntimeBundle &source, bool retainPayload) {
  if (source.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << "only runtime object bundles can be boxed";
  if (RuntimeBundleLowerer::isBuiltinsObjectContract(source.contract) &&
      source.boxedObject)
    return source;

  RuntimeBundle concrete = source;
  if (RuntimeBundleLowerer::hasLazyPrimitiveI64Object(concrete)) {
    mlir::FailureOr<RuntimeValue> materialized =
        RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
            op, concrete);
    if (mlir::failed(materialized))
      return mlir::failure();
    concrete.objectValue = *materialized;
  }

  mlir::Location loc = op->getLoc();
  auto boxType = mlir::MemRefType::get({16}, builder.getI64Type());
  mlir::Value box =
      mlir::memref::AllocOp::create(builder, loc, boxType).getResult();
  box.getDefiningOp()->setAttr(own::kObjectHeaderAttr, builder.getUnitAttr());

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
      RuntimeBundleLowerer::objectPayloadHandleWords(op, concrete,
                                                     retainPayload);
  if (mlir::failed(words))
    return mlir::failure();
  for (auto [index, word] : llvm::enumerate(*words)) {
    mlir::Value slot = mlir::arith::ConstantIndexOp::create(
        builder, loc, static_cast<std::int64_t>(index));
    mlir::memref::StoreOp::create(builder, loc, word, box, slot);
  }
  if (retainPayload && mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                           op, concrete, "boxed.object.payload")))
    return mlir::failure();
  concrete.setObjectLogicalOwnership(retainPayload);

  RuntimeBundle boxed = RuntimeBundle::object(
      runtimeContractType(context, "builtins.object"), mlir::ValueRange{box});
  boxed.boxedObject = std::make_shared<RuntimeBundle>(std::move(concrete));
  return boxed;
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
  return mlir::memref::CastOp::create(builder, op->getLoc(), targetMemRef,
                                      storage)
      .getResult();
}

mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::objectPhysicalHeader(mlir::Operation *op,
                                           const RuntimeValue &value) {
  if (value.values.empty())
    return op->emitError() << value.contractName()
                           << " runtime object has no physical header value";

  mlir::Value header = value.values.front();
  auto headerType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  if (!headerType || !isRankOneI64MemRef(header.getType()))
    return op->emitError() << value.contractName()
                           << " runtime object header has invalid type "
                           << header.getType();
  if (headerType.hasStaticShape() && headerType.getDimSize(0) < 2)
    return op->emitError() << value.contractName()
                           << " runtime object header must expose refcount "
                              "and class-id slots, got "
                           << header.getType();
  return header;
}

mlir::FailureOr<mlir::Value>
RuntimeBundleLowerer::materializeDeadPhysicalValue(mlir::Operation *op,
                                                   mlir::Type type) {
  mlir::Location loc = op->getLoc();
  if (mlir::isa<mlir::IndexType>(type))
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  if (mlir::isa<mlir::IntegerType>(type))
    return mlir::arith::ConstantOp::create(builder, loc, type,
                                           builder.getIntegerAttr(type, 0))
        .getResult();
  if (mlir::isa<mlir::FloatType>(type))
    return mlir::arith::ConstantOp::create(builder, loc, type,
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
            mlir::arith::ConstantIndexOp::create(builder, loc, 1));
    mlir::Value allocated =
        mlir::memref::AllocOp::create(builder, loc, allocType, dynamicSizes)
            .getResult();
    if (mlir::failed(zeroInitializeMemRef(builder, loc, allocated)))
      return mlir::failure();
    if (allocated.getType() == type)
      return allocated;
    return mlir::memref::CastOp::create(builder, loc, type, allocated)
        .getResult();
  }

  return op->emitError() << "cannot materialize a dead runtime placeholder of "
                            "physical type "
                         << type;
}

static mlir::FailureOr<mlir::Value> materializeStaticDeadPhysicalValue(
    mlir::ModuleOp module, mlir::OpBuilder &builder, mlir::Operation *op,
    mlir::Type type, bool objectHeader) {
  mlir::Location loc = op->getLoc();
  if (mlir::isa<mlir::IndexType>(type))
    return mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  if (mlir::isa<mlir::IntegerType>(type))
    return mlir::arith::ConstantOp::create(builder, loc, type,
                                           builder.getIntegerAttr(type, 0))
        .getResult();
  if (mlir::isa<mlir::FloatType>(type))
    return mlir::arith::ConstantOp::create(builder, loc, type,
                                           builder.getFloatAttr(type, 0.0))
        .getResult();

  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memrefType)
    return op->emitError()
           << "cannot materialize a static dead runtime placeholder of "
              "physical type "
           << type;

  mlir::MemRefType globalType = concreteDeadMemRefType(memrefType);
  std::string name =
      (objectHeader ? "__ly_dead_header_" : "__ly_dead_payload_") +
      typeSymbolComponent(globalType);
  if (!module.lookupSymbol<mlir::memref::GlobalOp>(name)) {
    mlir::Attribute initialValue =
        deadMemRefInitialValue(builder, globalType, objectHeader);
    if (!initialValue)
      return op->emitError()
             << "cannot build zero initializer for static dead placeholder "
             << globalType;

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    mlir::memref::GlobalOp::create(builder, loc, name,
                                   builder.getStringAttr("private"), globalType,
                                   initialValue,
                                   /*constant=*/true, /*alignment=*/nullptr);
  }

  mlir::Value global =
      mlir::memref::GetGlobalOp::create(builder, loc, globalType, name)
          .getResult();
  if (global.getType() == type)
    return global;
  return mlir::memref::CastOp::create(builder, loc, type, global).getResult();
}

mlir::FailureOr<RuntimeValue>
RuntimeBundleLowerer::materializeDeadObjectValueImpl(
    mlir::Operation *op, mlir::Type contract, llvm::StringRef purpose,
    DeadObjectStorage storage) {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, contract, purpose);
  if (mlir::failed(valueTypes))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> values;
  values.reserve(valueTypes->size());
  for (auto [index, valueType] : llvm::enumerate(*valueTypes)) {
    mlir::FailureOr<mlir::Value> value =
        storage == DeadObjectStorage::StaticNonOwning
            ? materializeStaticDeadPhysicalValue(module, builder, op, valueType,
                                                 /*objectHeader=*/index == 0)
            : RuntimeBundleLowerer::materializeDeadPhysicalValue(op, valueType);
    if (mlir::failed(value))
      return mlir::failure();
    values.push_back(*value);
  }

  if (!values.empty() && storage == DeadObjectStorage::OwningHeap)
    initializeObjectHeader(builder, op->getLoc(), values.front(),
                           /*refcount=*/1, /*classId=*/0);
  return RuntimeValue::objectWithOwnership(
      contract, values,
      storage == DeadObjectStorage::StaticNonOwning
          ? own::OwnershipKind::Immortal
          : own::logicalOwnershipKind(contract, /*ownsObject=*/true));
}

mlir::FailureOr<RuntimeValue> RuntimeBundleLowerer::materializeDeadObjectValue(
    mlir::Operation *op, mlir::Type contract, llvm::StringRef purpose) {
  return RuntimeBundleLowerer::materializeDeadObjectValueImpl(
      op, contract, purpose, DeadObjectStorage::OwningHeap);
}

mlir::FailureOr<RuntimeValue>
RuntimeBundleLowerer::materializeNonOwningDeadObjectValue(
    mlir::Operation *op, mlir::Type contract, llvm::StringRef purpose) {
  return RuntimeBundleLowerer::materializeDeadObjectValueImpl(
      op, contract, purpose, DeadObjectStorage::StaticNonOwning);
}

mlir::FailureOr<RuntimeValue> RuntimeBundleLowerer::materializeClassObjectValue(
    mlir::Operation *op, py::ClassOp classOp, mlir::Type contract,
    llvm::StringRef purpose) {
  const RuntimeValueShape *objectShape = manifest.valueShape("builtins.object");
  if (!objectShape || objectShape->valueTypes.size() != 1)
    return op->emitError()
           << "runtime manifest has no single-value builtins.object ABI shape "
              "for "
           << purpose;

  auto headerType =
      mlir::dyn_cast<mlir::MemRefType>(objectShape->valueTypes.front());
  if (!headerType || headerType.getRank() != 1 ||
      !headerType.hasStaticShape() || headerType.getDimSize(0) < 2)
    return op->emitError() << "builtins.object ABI handle shape is invalid for "
                           << purpose << ": "
                           << objectShape->valueTypes.front();

  mlir::Location loc = op->getLoc();
  llvm::SmallVector<std::int64_t, 1> shape(headerType.getShape().begin(),
                                           headerType.getShape().end());
  mlir::MemRefType allocType = mlir::MemRefType::get(
      shape, headerType.getElementType(), mlir::MemRefLayoutAttrInterface{},
      headerType.getMemorySpace());
  mlir::Value allocated =
      mlir::memref::AllocOp::create(builder, loc, allocType).getResult();
  mlir::Value header = allocated;
  if (allocated.getType() != headerType)
    header = mlir::memref::CastOp::create(builder, loc, headerType, allocated)
                 .getResult();

  std::optional<std::int64_t> classId =
      RuntimeBundleLowerer::runtimeClassIdForClass(classOp);
  if (!classId)
    return op->emitError() << "class " << classOp.getSymName()
                           << " has no runtime class id for " << purpose;
  initializeObjectHeader(builder, loc, header, /*refcount=*/1,
                         /*classId=*/*classId);

  llvm::SmallVector<mlir::Type, 8> fieldContractTypes =
      RuntimeBundleLowerer::classFieldContractTypes(classOp);
  mlir::Value zeroI64 = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  for (auto [fieldIndex, fieldType] : llvm::enumerate(fieldContractTypes)) {
    if (runtimeContractName(fieldType) != "builtins.int")
      continue;
    unsigned slot = kPrimitiveFieldSlotBase + static_cast<unsigned>(fieldIndex);
    if (slot >= kPrimitiveFieldSlotLimit)
      continue;
    mlir::Value slotIndex =
        mlir::arith::ConstantIndexOp::create(builder, loc, slot);
    mlir::memref::StoreOp::create(builder, loc, zeroI64, header, slotIndex);
  }

  llvm::SmallVector<mlir::Value, 8> values{header};
  for (mlir::Type fieldType : fieldContractTypes) {
    // Box-fronted fields store a single box16 slot; materialize the dead
    // placeholder in the STORAGE shape, not the contract's array shape.
    mlir::Type storageType =
        RuntimeBundleLowerer::classFieldStoredBoxed(fieldType)
            ? runtimeContractType(context, "builtins.object")
            : fieldType;
    mlir::FailureOr<RuntimeValue> fieldValue =
        RuntimeBundleLowerer::materializeDeadObjectValue(op, storageType,
                                                         purpose);
    if (mlir::failed(fieldValue))
      return mlir::failure();
    values.append(fieldValue->values.begin(), fieldValue->values.end());
  }

  if (headerType.hasStaticShape() && headerType.getDimSize(0) >= 3) {
    mlir::Value valueCountSlot =
        mlir::arith::ConstantIndexOp::create(builder, loc, 2);
    mlir::Value valueCount = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(values.size()), 64);
    mlir::memref::StoreOp::create(builder, loc, valueCount, header,
                                  valueCountSlot);
  }

  return RuntimeValue::object(contract, values);
}

mlir::LogicalResult RuntimeBundleLowerer::synthesizeSourceClassDeallocators() {
  struct Plan {
    py::ClassOp classOp;
    mlir::func::FuncOp function;
    std::string contract;
    llvm::SmallVector<mlir::Type, 8> fieldTypes;
    llvm::SmallVector<unsigned, 8> fieldOffsets;
  };

  const RuntimeValueShape *objectShape = manifest.valueShape("builtins.object");
  if (!objectShape || objectShape->valueTypes.size() != 1)
    return module.emitError()
           << "runtime manifest has no single-value builtins.object ABI shape "
              "for source class deallocators";
  unsigned objectHeaderValues =
      static_cast<unsigned>(objectShape->valueTypes.size());

  llvm::SmallVector<py::ClassOp, 8> classes;
  module.walk([&](py::ClassOp classOp) { classes.push_back(classOp); });

  llvm::SmallVector<Plan, 8> plans;
  for (py::ClassOp classOp : classes) {
    std::string contract = classOp.getSymName().str();
    if (contract.empty())
      continue;
    if (declaredRuntimeContractMatchesClass(module, contract))
      continue;
    if (moduleHasDeallocatorForContract(module, contract))
      continue;

    mlir::Type contractType = runtimeContractType(context, contract);
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> valueTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(
            classOp, contractType, "source class deallocator ABI");
    if (mlir::failed(valueTypes))
      return mlir::failure();
    if (valueTypes->empty())
      return classOp.emitError()
             << "source class deallocator ABI has no object header";

    llvm::SmallVector<mlir::Type, 8> fieldTypes =
        RuntimeBundleLowerer::classFieldContractTypes(classOp);
    llvm::SmallVector<unsigned, 8> fieldOffsets;
    unsigned offset = objectHeaderValues;
    fieldOffsets.reserve(fieldTypes.size());
    for (mlir::Type fieldType : fieldTypes) {
      fieldOffsets.push_back(offset);
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
          RuntimeBundleLowerer::classFieldStorageValueTypes(
              classOp, fieldType, "source class field deallocator ABI");
      if (mlir::failed(fieldValueTypes))
        return mlir::failure();
      offset += static_cast<unsigned>(fieldValueTypes->size());
    }
    if (offset != valueTypes->size())
      return classOp.emitError()
             << "source class deallocator ABI field layout for " << contract
             << " does not match class object ABI";

    std::string baseName = sourceClassDeallocatorName(contract);
    std::string functionName = baseName;
    for (unsigned suffix = 1;
         module.lookupSymbol<mlir::func::FuncOp>(functionName); ++suffix)
      functionName = baseName + "_" + std::to_string(suffix);

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto functionType = builder.getFunctionType(*valueTypes, {});
    mlir::func::FuncOp function = mlir::func::FuncOp::create(
        builder, module.getLoc(), functionName, functionType);
    function.setPrivate();
    function->setAttr(kManifestContractAttr, builder.getStringAttr(contract));
    function->setAttr(kManifestDeallocatorAttr, builder.getUnitAttr());
    function->setAttr(own::kReleaseArgsAttr,
                      mlir::DenseI64ArrayAttr::get(context, {0}));
    function.setArgAttr(0, own::kObjectHeaderAttr, builder.getUnitAttr());

    plans.push_back(Plan{classOp, function, std::move(contract),
                         std::move(fieldTypes), std::move(fieldOffsets)});
  }

  if (plans.empty())
    return mlir::success();

  mlir::func::FuncOp releaseToZero = findObjectStorageReleaseToZero(module);
  if (!releaseToZero)
    return module.emitError() << "source class deallocators require a runtime "
                                 "LyObject_ReleaseStorageToZero primitive";

  llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators =
      own::collectRuntimeDeallocators(module);

  for (Plan &plan : plans) {
    mlir::Block *entry = plan.function.addEntryBlock();
    mlir::Block *deallocBlock = plan.function.addBlock();
    mlir::Block *doneBlock = plan.function.addBlock();
    mlir::Location loc = plan.function.getLoc();

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(entry);
    mlir::FailureOr<mlir::Value> storage =
        storageForReleaseToZero(plan.function, builder, entry->getArgument(0),
                                releaseToZero.getFunctionType().getInput(0));
    if (mlir::failed(storage))
      return mlir::failure();
    mlir::func::CallOp releaseHeader = mlir::func::CallOp::create(
        builder, loc, releaseToZero, mlir::ValueRange{*storage});
    mlir::cf::CondBranchOp::create(builder, loc, releaseHeader.getResult(0),
                                   deallocBlock, doneBlock);

    builder.setInsertionPointToStart(deallocBlock);
    for (auto [index, fieldType] : llvm::enumerate(plan.fieldTypes)) {
      mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> fieldValueTypes =
          RuntimeBundleLowerer::classFieldStorageValueTypes(
              plan.function, fieldType, "source class field deallocator ABI");
      if (mlir::failed(fieldValueTypes))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 4> fieldValues =
          entryArgumentSlice(*entry, plan.fieldOffsets[index],
                             static_cast<unsigned>(fieldValueTypes->size()));
      // Box-fronted fields release through the boxed route (the release hook
      // dispatches the box's class id to the manifest deallocator); the box
      // itself is the authoritative view of a possibly-reallocated container.
      mlir::Type releaseType =
          RuntimeBundleLowerer::classFieldStoredBoxed(fieldType)
              ? runtimeContractType(context, "builtins.object")
              : fieldType;
      if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
              plan.function, releaseType, fieldValues, "class.field",
              deallocators, /*depth=*/0)))
        return mlir::failure();
    }

    mlir::memref::DeallocOp::create(builder, loc, entry->getArgument(0));
    mlir::cf::BranchOp::create(builder, loc, doneBlock);

    builder.setInsertionPointToStart(doneBlock);
    mlir::func::ReturnOp::create(builder, loc);
  }

  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::materializeStringObject(
    mlir::Operation *op, llvm::StringRef text, RuntimeBundle &bundle) {
  mlir::Location loc = op->getLoc();
  mlir::Value bytes = RuntimeBundleLowerer::materializeByteBuffer(loc, text);
  mlir::Value start =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value length =
      mlir::arith::ConstantIntOp::create(
          builder, loc, static_cast<std::int64_t>(text.size()), 64)
          .getResult();
  return RuntimeBundleLowerer::initializeObjectFromRawValues(
      op, runtimeContractType(context, "builtins.str"),
      mlir::ValueRange{bytes, start, length}, bundle);
}

mlir::LogicalResult RuntimeBundleLowerer::materializeBytesObject(
    mlir::Operation *op, llvm::StringRef data, RuntimeBundle &bundle) {
  mlir::Location loc = op->getLoc();
  mlir::Value bytes = RuntimeBundleLowerer::materializeByteBuffer(loc, data);
  mlir::Value start =
      mlir::arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  mlir::Value length =
      mlir::arith::ConstantIntOp::create(
          builder, loc, static_cast<std::int64_t>(data.size()), 64)
          .getResult();
  return RuntimeBundleLowerer::initializeObjectFromRawValues(
      op, runtimeContractType(context, "builtins.bytes"),
      mlir::ValueRange{bytes, start, length}, bundle);
}

bool RuntimeBundleLowerer::needsDefaultObjectRepr(
    const RuntimeBundle &object) const {
  // Runtime-mode sequences must not fall back to the address-based default
  // repr (CPython prints their contents); rejecting the manifest lookup below
  // keeps the failure explicit until a runtime sequence repr exists.
  std::string contract = object.contractName();
  if ((contract == "builtins.list" || contract == "builtins.tuple") &&
      !object.sequenceEvidenceBacked && object.sequenceElementBundles.empty())
    return false;
  if (contract == "builtins.dict" && !object.mappingEvidenceBacked &&
      object.mappingKeys.empty())
    return false;
  if (contract == "builtins.set")
    return false;
  return object.kind == RuntimeBundle::Kind::Object &&
         manifest.methodCandidates(contract, "__repr__").empty();
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
      RuntimeBundleLowerer::objectPhysicalHeader(op, object.objectValue);
  if (mlir::failed(header))
    return mlir::failure();
  mlir::FunctionType primitiveType = primitive->function.getFunctionType();
  if (primitiveType.getNumInputs() < 1)
    return primitive->function.emitError()
           << "builtins.object default_repr primitive must accept an object "
              "header";
  mlir::FailureOr<mlir::Value> headerView = viewRankOneI64Prefix(
      op, builder, *header, primitiveType.getInput(0), "default repr header");
  if (mlir::failed(headerView))
    return mlir::failure();

  std::string prefix = "<";
  prefix += defaultReprTypeName(object.contractName());
  prefix += " object at 0x";
  mlir::Value prefixBytes =
      RuntimeBundleLowerer::materializeByteBuffer(op->getLoc(), prefix);
  mlir::Value prefixLength =
      mlir::arith::ConstantIntOp::create(
          builder, op->getLoc(), static_cast<std::int64_t>(prefix.size()), 64)
          .getResult();

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *primitive,
      mlir::ValueRange{*headerView, prefixBytes, prefixLength});
  return RuntimeBundleLowerer::bundleRuntimeResults(
      op, runtimeContractType(context, "builtins.str"), call, bundle);
}

// Per-program boxed-slot release hook. The native slot dispatcher
// (release_payload_slot_ptr in RuntimeSupportBuilder) tries this FIRST: each
// merged contract's manifest deallocator is the single implementation of
// decref, child releases, and the block free. The hook is generated here —
// not in the always-loaded native library — because runtime object modules
// are merged per contract on demand, so only the lowering knows which
// deallocators exist (including generated source-class deallocators).
mlir::LogicalResult RuntimeBundleLowerer::generateBoxedMethodHook(
    llvm::StringRef hookName,
    llvm::function_ref<bool(mlir::func::FuncOp)> selects,
    mlir::TypeRange calleeResultTypes, bool shareExceptionSubclasses,
    llvm::StringRef sourceClassMethodName) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(hookName)) {
    // A definition already exists (idempotent); an external declaration (from a
    // merged manifest caller) is replaced by the generated body below.
    if (!existing.isExternal())
      return mlir::success();
    existing.erase();
  }

  struct HookEntry {
    std::int64_t classId;
    mlir::func::FuncOp callee;
  };
  llvm::SmallVector<HookEntry, 16> entries;
  llvm::SmallDenseSet<std::int64_t, 16> seenIds;
  // The callee's arguments are reconstructed uniformly from the box word
  // layout (slot words (4+i, 9+i) hold physical value i), so every selected
  // function must take only rank-1 memrefs and share the hook's callee result
  // shape (no per-type special-casing).
  auto conforms = [&](mlir::func::FuncOp function) {
    mlir::FunctionType type = function.getFunctionType();
    if (type.getNumResults() != calleeResultTypes.size() ||
        type.getNumInputs() == 0 || type.getNumInputs() > 5)
      return false;
    for (auto [have, want] : llvm::zip(type.getResults(), calleeResultTypes))
      if (have != want)
        return false;
    for (mlir::Type input : type.getInputs()) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(input);
      if (!memref || memref.getRank() != 1)
        return false;
    }
    return true;
  };
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isExternal() || !selects(function))
      return;
    auto contractAttr = function->getAttrOfType<mlir::StringAttr>(
        contracts::kManifestContractAttr);
    if (!contractAttr)
      return;
    std::optional<std::int64_t> classId =
        manifest.classId(contractAttr.getValue());
    if (!classId) {
      py::ClassOp classOp = RuntimeBundleLowerer::classForContract(
          runtimeContractType(context, contractAttr.getValue()));
      if (classOp)
        classId = RuntimeBundleLowerer::runtimeClassIdForClass(classOp);
    }
    // Conformance decides before the id is consumed: a non-conforming
    // candidate (e.g. bool's primitive-i1 __repr__) must not shadow a
    // conforming boxed one for the same class.
    if (!classId || !conforms(function) || !seenIds.insert(*classId).second)
      return;
    entries.push_back(HookEntry{*classId, function});
  });

  // Compiled source-class methods share the physical-value slot convention
  // (their physicals are (self box, field views...)), so they join the same
  // dispatch when their signature conforms.
  if (!sourceClassMethodName.empty()) {
    module.walk([&](py::ClassOp classOp) {
      std::optional<std::string> symbol =
          RuntimeBundleLowerer::classMethodSymbol(classOp,
                                                  sourceClassMethodName);
      if (!symbol)
        return;
      auto function = module.lookupSymbol<mlir::func::FuncOp>(*symbol);
      if (!function || function.isExternal() || !conforms(function))
        return;
      std::optional<std::int64_t> classId =
          RuntimeBundleLowerer::runtimeClassIdForClass(classOp);
      if (!classId || !seenIds.insert(*classId).second)
        return;
      entries.push_back(HookEntry{*classId, function});
    });
  }

  // Exception subclasses share BaseException's shape and (for release) its
  // deallocator but carry their own class ids; without these entries a boxed
  // subclass would miss the hook. Only used where subclasses share one callee.
  if (shareExceptionSubclasses) {
    mlir::func::FuncOp baseCallee;
    for (const HookEntry &hookEntry : entries) {
      auto contractAttr = hookEntry.callee->getAttrOfType<mlir::StringAttr>(
          contracts::kManifestContractAttr);
      if (contractAttr && contractAttr.getValue() == "builtins.BaseException") {
        baseCallee = hookEntry.callee;
        break;
      }
    }
    if (baseCallee) {
      const py::protocols::Table &table = py::protocols::Table::get(*context);
      module.walk([&](mlir::func::FuncOp function) {
        auto contractAttr = function->getAttrOfType<mlir::StringAttr>(
            contracts::kManifestContractAttr);
        auto classIdAttr = function->getAttrOfType<mlir::IntegerAttr>(
            contracts::kManifestClassIdAttr);
        if (!contractAttr || !classIdAttr ||
            !seenIds.insert(classIdAttr.getInt()).second)
          return;
        if (!table.isManifestSubclassOf(
                runtimeContractType(context, contractAttr.getValue()),
                "builtins.BaseException"))
          return;
        entries.push_back(HookEntry{classIdAttr.getInt(), baseCallee});
      });
    }
  }

  mlir::OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  auto ptrType = mlir::LLVM::LLVMPointerType::get(context);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();
  mlir::Location loc = module.getLoc();
  llvm::SmallVector<mlir::Type, 4> hookResultTypes(calleeResultTypes.begin(),
                                                   calleeResultTypes.end());
  hookResultTypes.push_back(i1);
  auto hook = mlir::func::FuncOp::create(
      builder, loc, hookName,
      builder.getFunctionType({ptrType, i64}, hookResultTypes));

  mlir::Block *entry = hook.addEntryBlock();
  mlir::Value slot = entry->getArgument(0);
  mlir::Value classValue = entry->getArgument(1);

  mlir::Block *miss = hook.addBlock();
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(miss);
    llvm::SmallVector<mlir::Value, 4> missResults;
    for (mlir::Type resultType : calleeResultTypes)
      missResults.push_back(
          mlir::ub::PoisonOp::create(builder, loc, resultType, nullptr));
    missResults.push_back(
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 1));
    mlir::func::ReturnOp::create(builder, loc, missResults);
  }

  auto loadWord = [&](mlir::OpBuilder &b, std::int64_t index) -> mlir::Value {
    mlir::Value gep = mlir::LLVM::GEPOp::create(
        b, loc, ptrType, i64, slot,
        llvm::ArrayRef<mlir::LLVM::GEPArg>{
            mlir::LLVM::GEPArg(static_cast<std::int32_t>(index))});
    return mlir::LLVM::LoadOp::create(b, loc, i64, gep).getResult();
  };

  mlir::Block *check = entry;
  for (const HookEntry &hookEntry : entries) {
    mlir::Block *handle = hook.addBlock();
    mlir::Block *next = hook.addBlock();
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToEnd(check);
      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, loc, hookEntry.classId, 64);
      mlir::Value matches = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, classValue, expected);
      mlir::cf::CondBranchOp::create(builder, loc, matches, handle,
                                     mlir::ValueRange{}, next,
                                     mlir::ValueRange{});
    }
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(handle);
      llvm::SmallVector<mlir::Value, 6> operands;
      mlir::func::FuncOp callee = hookEntry.callee;
      mlir::FunctionType type = callee.getFunctionType();
      for (auto [index, input] : llvm::enumerate(type.getInputs())) {
        auto memref = mlir::cast<mlir::MemRefType>(input);
        mlir::Value pointerWord =
            loadWord(builder, 4 + static_cast<std::int64_t>(index));
        mlir::Value sizeWord =
            loadWord(builder, 9 + static_cast<std::int64_t>(index));
        operands.push_back(RuntimeBundleLowerer::memrefFromBoxWords(
            builder, loc, pointerWord, sizeWord, memref));
      }
      mlir::func::CallOp call =
          mlir::func::CallOp::create(builder, loc, callee, operands);
      llvm::SmallVector<mlir::Value, 4> hitResults(call.getResults().begin(),
                                                   call.getResults().end());
      hitResults.push_back(
          mlir::arith::ConstantIntOp::create(builder, loc, 1, 1));
      mlir::func::ReturnOp::create(builder, loc, hitResults);
    }
    check = next;
  }
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(check);
    mlir::cf::BranchOp::create(builder, loc, miss);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::generateBoxedReleaseHook() {
  // Release is one instance of the uniform boxed-method dispatch: class id ->
  // the manifest deallocator (which returns void). Exception subclasses share
  // BaseException's deallocator.
  return generateBoxedMethodHook(
      "__ly_release_boxed_by_contract",
      [](mlir::func::FuncOp function) {
        return function->hasAttr(contracts::kManifestDeallocatorAttr);
      },
      /*calleeResultTypes=*/{}, /*shareExceptionSubclasses=*/true);
}

mlir::LogicalResult RuntimeBundleLowerer::generateBoxedReprHook() {
  // repr instance: class id -> the manifest `__repr__` (returns an owned str,
  // as (header, bytes) memrefs). Each type carries its own __repr__, so no
  // subclass sharing. Non-conforming __repr__ (bool's i1 receiver) are skipped
  // by the memref-input predicate — that is a manifest-conformance gap for that
  // type, not a special case here.
  // Only generate the hook when a merged manifest __repr__ (a container's) has
  // referenced it — otherwise it would be dead weight (and an unused public
  // hook is not eliminated). Its presence as an external declaration signals
  // the need.
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(
          "__ly_repr_boxed_by_contract");
      !existing || !existing.isExternal())
    return mlir::success();
  mlir::Type i64 = mlir::IntegerType::get(context, 64);
  mlir::Type i8 = mlir::IntegerType::get(context, 8);
  auto strHeader = mlir::MemRefType::get({2}, i64);
  auto strBytes =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, i8);
  if (mlir::failed(generateBoxedMethodHook(
          "__ly_repr_boxed_by_contract",
          [](mlir::func::FuncOp function) {
            auto method = function->getAttrOfType<mlir::StringAttr>(
                contracts::kManifestMethodAttr);
            return method && method.getValue() == "__repr__";
          },
          {strHeader, strBytes}, /*shareExceptionSubclasses=*/false,
          /*sourceClassMethodName=*/"__repr__")))
    return mlir::failure();
  // The hook forwards the manifest __repr__'s owned str result (header at 0).
  if (auto hook = module.lookupSymbol<mlir::func::FuncOp>(
          "__ly_repr_boxed_by_contract"))
    hook->setAttr("ly.ownership.owned_results",
                  mlir::OpBuilder(context).getI64ArrayAttr({0}));
  return mlir::success();
}

} // namespace py::lowering
