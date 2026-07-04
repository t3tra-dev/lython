#include "Runtime/Ctypes/Internal.h"

namespace py::runtime_lowering {

using namespace ctypes;

bool RuntimeBundleLowerer::isStaticCtypesBinding(
    llvm::StringRef binding) const {
  if (RuntimeBundleLowerer::isStaticCtypesModuleBinding(binding) ||
      RuntimeBundleLowerer::isStaticCtypesCallable(binding))
    return true;
  return ctypesQualifiedNameContract(binding).has_value();
}

bool RuntimeBundleLowerer::isStaticCtypesModuleBinding(
    llvm::StringRef binding) const {
  return binding == "ctypes" || binding == "_ctypes" ||
         binding == "ctypes.wintypes";
}

bool RuntimeBundleLowerer::isStaticCtypesCallable(
    llvm::StringRef binding) const {
  if (ctypesFromAddressTarget(binding) || ctypesFromBufferTarget(binding) ||
      ctypesFromBufferCopyTarget(binding))
    return true;
  return llvm::StringSwitch<bool>(binding)
      .Cases("ctypes.sizeof", "ctypes.alignment", "ctypes.byref",
             "ctypes.pointer", "ctypes.POINTER", "ctypes.cast",
             "ctypes.addressof", true)
      .Default(false);
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesBindingRef(py::BindingRefOp op) {
  if (RuntimeBundleLowerer::isStaticCtypesModuleBinding(op.getBinding()))
    return RuntimeBundleLowerer::lowerStaticCtypesModuleBindingRef(op);
  if (RuntimeBundleLowerer::isStaticCtypesCallable(op.getBinding())) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(), op.getBinding());
    erase.push_back(op);
    return mlir::success();
  }
  if (std::optional<std::string> contract =
          ctypesQualifiedNameContract(op.getBinding())) {
    valueBundles[op.getResult()] = RuntimeBundle::typeObject(
        op.getResult().getType(), ctypesContractType(context, *contract));
    erase.push_back(op);
    return mlir::success();
  }
  return mlir::failure();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesModuleBindingRef(py::BindingRefOp op) {
  valueBundles[op.getResult()] =
      makeCtypesModuleBundle(op.getResult().getType(), op.getBinding());
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesModuleAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Module)
    return mlir::failure();
  llvm::StringRef moduleName = object.ctypes->ctypeName;
  if (moduleName == "ctypes" && op.getName() == "wintypes") {
    valueBundles[op.getResult()] =
        makeCtypesModuleBundle(op.getResult().getType(), "ctypes.wintypes");
    erase.push_back(op);
    return mlir::success();
  }
  if (std::optional<std::string> contract =
          ctypesModuleAttrContract(moduleName, op.getName())) {
    valueBundles[op.getResult()] = RuntimeBundle::typeObject(
        op.getResult().getType(), ctypesContractType(context, *contract));
    erase.push_back(op);
    return mlir::success();
  }
  if (moduleName == "ctypes" && isStaticCtypesFunctionName(op.getName())) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(),
        (llvm::Twine("ctypes.") + op.getName()).str());
    erase.push_back(op);
    return mlir::success();
  }
  return op.emitError() << "ctypes module '" << moduleName
                        << "' has no static attribute '" << op.getName() << "'";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesValueAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();
  if (!object.ctypes->scalarValue || !object.ctypes->scalarValid)
    return op.emitError() << "ctypes value attribute requires scalar evidence";
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"),
          object.ctypes->scalarValue, object.ctypes->scalarValid, result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesFieldDescriptorAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::FieldDescriptor)
    return mlir::failure();

  if (op.getName() == "offset" || op.getName() == "size") {
    std::int64_t value =
        op.getName() == "offset"
            ? static_cast<std::int64_t>(object.ctypes->fieldOffset)
            : static_cast<std::int64_t>(object.ctypes->fieldSize);
    RuntimeBundle result;
    builder.setInsertionPoint(op);
    if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
            op, runtimeContractType(context, "builtins.int"),
            constantI64(builder, op.getLoc(), value),
            constantI1(builder, op.getLoc(), true), result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (op.getName() == "name") {
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
            op, object.ctypes->fieldName, result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  return op.emitError() << "ctypes CField descriptor has no static attribute '"
                        << op.getName() << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesTypeFieldDescriptorGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (object.kind != RuntimeBundle::Kind::TypeObject)
    return mlir::failure();
  std::string contract = object.instanceContractName();
  py::ClassOp classOp = lookupClassForContract(module, contract);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();

  RuntimeBundle result =
      RuntimeBundle::object(ctypesContractType(context, "_ctypes.CField"), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::FieldDescriptor;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = "_ctypes.CField";
  evidence.ctype = ctypesContractType(context, "_ctypes.CField");
  evidence.fieldName = field->name;
  evidence.fieldType = field->contract;
  evidence.fieldOffset = field->offset;
  evidence.fieldSize = field->layout.size;
  result.ctypes = std::move(evidence);
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesFieldAttrGet(
    py::AttrGetOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  py::ClassOp classOp =
      lookupClassForContract(module, object.ctypes->ctypeName);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();

  mlir::Value storageAddress = cdataStorageAddress(*object.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*object.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError() << "ctypes field '" << op.getName()
                          << "' requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value fieldAddress =
      addressWithOffset(builder, op.getLoc(), storageAddress,
                        static_cast<std::int64_t>(field->offset), facts);
  mlir::FailureOr<RuntimeBundle> result = materializeCtypesPythonReadResult(
      op, builder, module, field->type, field->contract, field->layout,
      fieldAddress, storageValid, object);
  if (mlir::failed(result))
    return mlir::failure();
  if ((*result).ctypes) {
    (*result).fieldAliasOwner = op.getObject();
    (*result).fieldAliasName = op.getName().str();
  }
  valueBundles[op.getResult()] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesFieldAttrSet(
    py::AttrSetOp op, const RuntimeBundle &object, const RuntimeBundle *value) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  py::ClassOp classOp =
      lookupClassForContract(module, object.ctypes->ctypeName);
  if (!classOp)
    return mlir::failure();
  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesAggregateLayout> aggregate =
      ctypesAggregateLayout(module, classOp, facts, 0);
  if (!aggregate)
    return mlir::failure();

  auto field =
      llvm::find_if(aggregate->fields, [&](const CtypesFieldLayout &it) {
        return it.name == op.getName();
      });
  if (field == aggregate->fields.end())
    return mlir::failure();
  if (!value)
    return op.emitError() << "ctypes field assignment has no value evidence";

  mlir::Value storageAddress = cdataStorageAddress(*object.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*object.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError() << "ctypes field '" << op.getName()
                          << "' requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value fieldAddress =
      addressWithOffset(builder, op.getLoc(), storageAddress,
                        static_cast<std::int64_t>(field->offset), facts);
  if (mlir::failed(storeCtypesValueToAddress(op, builder, module, fieldAddress,
                                             field->type, field->contract,
                                             field->layout, *value, facts)))
    return mlir::failure();

  valueBundles[op.getObject()] = object;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesGetItem(py::GetItemOp op,
                                               const RuntimeBundle &container,
                                               const RuntimeBundle &index) {
  if (!container.ctypes ||
      container.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  mlir::Type containerType =
      container.ctypes->ctype ? container.ctypes->ctype : container.contract;
  std::optional<CtypesArrayType> array =
      ctypesArrayType(module, containerType, facts);
  if (!array)
    return mlir::failure();
  if (!index.primitiveI64 || !index.primitiveI64->value ||
      !index.primitiveI64->valid || !isKnownTrue(index.primitiveI64->valid))
    return op.emitError()
           << "ctypes array indexing requires primitive integer evidence";
  std::optional<std::int64_t> rawIndex =
      knownI64Constant(index.primitiveI64->value);
  if (!rawIndex)
    return op.emitError()
           << "ctypes array indexing currently requires a statically known "
              "integer index";

  std::int64_t normalized = *rawIndex;
  std::int64_t count = static_cast<std::int64_t>(array->count);
  if (normalized < 0)
    normalized += count;
  if (normalized < 0 || normalized >= count) {
    builder.setInsertionPoint(op);
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.IndexError", "ctypes array index out of range")))
      return mlir::failure();
    mlir::FailureOr<RuntimeValue> dead =
        RuntimeBundleLowerer::materializeDeadObjectValue(
            op, op.getResult().getType(), "ctypes array index miss");
    if (mlir::failed(dead))
      return mlir::failure();
    valueBundles[op.getResult()] =
        RuntimeBundle::object(dead->contract, dead->values);
    erase.push_back(op);
    return mlir::success();
  }

  mlir::Value storageAddress = cdataStorageAddress(*container.ctypes);
  mlir::Value storageValid = cdataStorageAddressValid(*container.ctypes);
  if (!storageAddress || !storageValid || !isKnownTrue(storageValid))
    return op.emitError()
           << "ctypes array indexing requires materialized _CData storage";

  builder.setInsertionPoint(op);
  mlir::Value elementAddress = addressWithOffset(
      builder, op.getLoc(), storageAddress,
      static_cast<std::int64_t>(static_cast<std::uint64_t>(normalized) *
                                array->elementLayout.size),
      facts);
  mlir::FailureOr<RuntimeBundle> result = materializeCtypesPythonReadResult(
      op, builder, module, array->elementType, array->elementContract,
      array->elementLayout, elementAddress, storageValid, container);
  if (mlir::failed(result))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

bool RuntimeBundleLowerer::isErasedCtypesContract(
    llvm::StringRef contract) const {
  return isFixedOrTargetDependentCtypesScalar(contract);
}

bool RuntimeBundleLowerer::isStaticCtypesLibraryContract(
    llvm::StringRef contract) const {
  return contract == "ctypes.CDLL" || contract == "ctypes.WinDLL";
}

mlir::LogicalResult
RuntimeBundleLowerer::bindErasedCtypesNew(py::NewOp op,
                                          llvm::StringRef contract) {
  bool scalar = RuntimeBundleLowerer::isErasedCtypesContract(contract);
  py::ClassOp classOp =
      scalar
          ? py::ClassOp{}
          : RuntimeBundleLowerer::classForContract(op.getInstance().getType());
  bool aggregate = false;
  if (classOp) {
    std::optional<std::string> kind = ctypesAggregateKind(module, classOp);
    aggregate = kind && (*kind == "struct" || *kind == "union");
  }
  if (!scalar && !aggregate)
    return mlir::failure();

  RuntimeBundle bundle = RuntimeBundle::object(op.getInstance().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = contract.str();
  evidence.ctype = op.getInstance().getType();
  evidence.ownsNativeStorage = true;
  bundle.ctypes = std::move(evidence);
  valueBundles[op.getInstance()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::bindStaticCtypesLibraryNew(py::NewOp op,
                                                 llvm::StringRef contract) {
  if (!RuntimeBundleLowerer::isStaticCtypesLibraryContract(contract))
    return mlir::failure();
  RuntimeBundle bundle = RuntimeBundle::object(op.getInstance().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Library;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = contract.str();
  evidence.ctype = op.getInstance().getType();
  evidence.abi = ctypesLibraryABI(contract);
  bundle.ctypes = std::move(evidence);
  valueBundles[op.getInstance()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerErasedCtypesInit(
    py::InitOp op, const RuntimeBundle &instance,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (!instance.ctypes ||
      instance.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();
  if (sources.empty() || sources.front() != &instance)
    return op.emitError() << "ctypes initializer source evidence mismatch";

  mlir::FailureOr<RuntimeBundle> updated = materializeCtypesCell(
      op, builder, module,
      instance.ctypes->ctype ? instance.ctypes->ctype
                             : op.getInstance().getType(),
      instance.ctypes->ctypeName,
      llvm::ArrayRef<const RuntimeBundle *>(sources).drop_front());
  if (mlir::failed(updated))
    return mlir::failure();
  valueBundles[op.getInstance()] = std::move(*updated);
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesLibraryInit(
    py::InitOp op, const RuntimeBundle &instance,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (!instance.ctypes ||
      instance.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();
  if (sources.empty() || sources.front() != &instance)
    return op.emitError() << "ctypes library initializer source evidence "
                          << "mismatch";

  mlir::FailureOr<RuntimeBundle> updated = materializeCtypesLibrary(
      op, module,
      instance.ctypes->ctype ? instance.ctypes->ctype
                             : op.getInstance().getType(),
      instance.ctypes->ctypeName, sources.drop_front());
  if (mlir::failed(updated))
    return mlir::failure();
  valueBundles[op.getInstance()] = std::move(*updated);
  if (mlir::failed(assignObjectBundle(
          op, op.getResult(), runtimeContractType(context, "types.NoneType"),
          mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesTypeObjectCall(
    py::CallOp op, const RuntimeBundle &callable) {
  if (op.getNumResults() != 1)
    return op.emitError() << "ctypes type object call expects one result";
  if (mlir::failed(requireEmptyAggregate(op, op.getKwnames(), "kw names")) ||
      mlir::failed(requireEmptyAggregate(op, op.getKwvalues(), "kw values")))
    return mlir::failure();

  std::optional<std::string> contract = ctypesTypeObjectName(callable);
  if (!contract)
    return mlir::failure();

  llvm::SmallVector<const RuntimeBundle *, 4> sources;
  llvm::SmallVector<RuntimeBundle, 4> unpackedSources;
  if (mlir::failed(collectPackedObjectSources(op, op.getPosargs(),
                                              "ctypes constructor arguments",
                                              sources, &unpackedSources)))
    return mlir::failure();

  mlir::Type ctype = callable.instanceContract
                         ? callable.instanceContract
                         : ctypesContractType(context, *contract);
  mlir::FailureOr<RuntimeBundle> result =
      RuntimeBundleLowerer::isStaticCtypesLibraryContract(*contract)
          ? materializeCtypesLibrary(op, module, ctype, *contract, sources)
          : materializeCtypesCell(op, builder, module, ctype, *contract,
                                  sources);
  if (mlir::failed(result))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(*result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesModuleCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  if (!receiver.ctypes ||
      receiver.ctypes->kind != RuntimeCtypesEvidence::Kind::Module)
    return mlir::failure();
  llvm::StringRef moduleName = receiver.ctypes->ctypeName;
  if (moduleName == "ctypes" && isStaticCtypesFunctionName(methodName)) {
    RuntimeBundle callable = RuntimeBundle::builtinCallable(
        op.getCallable().getType(),
        (llvm::Twine("ctypes.") + methodName).str());
    return RuntimeBundleLowerer::lowerStaticCtypesCall(op, callable);
  }
  if (std::optional<std::string> contract =
          ctypesModuleAttrContract(moduleName, methodName)) {
    RuntimeBundle typeObject = RuntimeBundle::typeObject(
        op.getCallable().getType(), ctypesContractType(context, *contract));
    return RuntimeBundleLowerer::lowerStaticCtypesTypeObjectCall(op,
                                                                 typeObject);
  }
  return op.emitError() << "ctypes module '" << moduleName
                        << "' has no static callable attribute '" << methodName
                        << "'";
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesTypeObjectMethodCall(
    py::CallOp op, const RuntimeBundle &receiver, llvm::StringRef methodName) {
  std::optional<std::string> contract = ctypesTypeObjectName(receiver);
  if (!contract)
    return mlir::failure();
  if (methodName != "from_address" && methodName != "from_buffer" &&
      methodName != "from_buffer_copy")
    return mlir::failure();
  RuntimeBundle callable = RuntimeBundle::builtinCallable(
      op.getCallable().getType(),
      (llvm::Twine("ctypes.") + methodName + ":" + *contract).str());
  return RuntimeBundleLowerer::lowerStaticCtypesCall(op, callable);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesArrayTypeMul(
    mlir::Operation *op, const RuntimeBundle &lhs, const RuntimeBundle &rhs,
    mlir::Value resultValue) {
  const RuntimeBundle *typeObject = nullptr;
  const RuntimeBundle *countSource = nullptr;
  if (ctypesTypeObjectName(lhs) && rhs.primitiveI64) {
    typeObject = &lhs;
    countSource = &rhs;
  } else if (ctypesTypeObjectName(rhs) && lhs.primitiveI64) {
    typeObject = &rhs;
    countSource = &lhs;
  } else {
    return mlir::failure();
  }
  std::optional<std::int64_t> count =
      knownI64Constant(countSource->primitiveI64->value);
  if (!count || *count < 0)
    return mlir::failure();

  mlir::Type element = typeObject->instanceContract;
  if (!element)
    return mlir::failure();
  mlir::Type arrayType = py::ContractType::get(
      context, "_ctypes.Array",
      {element, py::LiteralType::get(context, std::to_string(*count))});
  valueBundles[resultValue] =
      RuntimeBundle::typeObject(resultValue.getType(), arrayType);
  return mlir::success();
}

} // namespace py::runtime_lowering
