#include "Runtime/Ctypes/Internal.h"

namespace py::lowering {

using namespace ctypes;

bool RuntimeBundleLowerer::isStaticCtypesBinding(
    llvm::StringRef binding) const {
  if (RuntimeBundleLowerer::isStaticCtypesModuleBinding(binding) ||
      RuntimeBundleLowerer::isStaticCtypesCallable(binding))
    return true;
  return ctypesQualifiedNameContract(*context, binding).has_value();
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
  return isStaticCtypesFunctionName(*context, binding);
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
          ctypesQualifiedNameContract(*context, op.getBinding())) {
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
          ctypesModuleAttrContract(*context, moduleName, op.getName())) {
    valueBundles[op.getResult()] = RuntimeBundle::typeObject(
        op.getResult().getType(), ctypesContractType(context, *contract));
    erase.push_back(op);
    return mlir::success();
  }
  if (moduleName == "ctypes" &&
      isStaticCtypesFunctionName(*context, op.getName())) {
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

  mlir::Value scalar = object.ctypes->scalarValue;
  mlir::Value valid = object.ctypes->scalarValid;
  // A cell with no in-SSA scalar (e.g. `from_address(addr)`) reads its value
  // from native storage. This load is async-signal-safe.
  if (!scalar || !valid) {
    mlir::Value address = cdataStorageAddress(*object.ctypes);
    mlir::Value addressValid = cdataStorageAddressValid(*object.ctypes);
    if (!address || !addressValid || !isKnownTrue(addressValid))
      return op.emitError() << "ctypes value attribute requires scalar or "
                               "materialized storage evidence";
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    std::optional<CtypesLayout> layout = ctypesStaticLayoutForType(
        module, object.ctypes->ctype, facts);
    if (!layout)
      layout = ctypesStaticLayout(module, object.ctypes->ctypeName, facts);
    if (!layout || !(isIntegerScalarLayout(*layout) ||
                     isPointerScalarLayout(*layout)))
      return op.emitError() << "ctypes value read requires a scalar integer "
                               "layout for "
                            << object.ctypes->ctypeName;
    builder.setInsertionPoint(op);
    mlir::Value loaded = loadNativeIntegerFromAddress(
        builder, op.getLoc(), address, nativeIntegerType(builder, *layout),
        facts);
    scalar = widenNativeInteger(builder, op.getLoc(), loaded, *layout);
    valid = constantI1(builder, op.getLoc(), true);
  }
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, runtimeContractType(context, "builtins.int"), scalar, valid,
          result)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

// `cell.value = v`: scalar cells carry their value both as SSA evidence
// (scalarValue) and, when materialized, in native storage -- update both so
// later reads and byref/native-call uses agree.
mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesValueAttrSet(
    py::AttrSetOp op, const RuntimeBundle &object, const RuntimeBundle *value) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Cell)
    return mlir::failure();
  if (!value)
    return op.emitError() << "ctypes value assignment has no value evidence";

  builder.setInsertionPoint(op);
  RuntimeCtypesEvidence evidence = *object.ctypes;
  if (value->primitiveI64 && value->primitiveI64->value &&
      value->primitiveI64->valid &&
      isKnownTrue(value->primitiveI64->valid)) {
    evidence.scalarValue = value->primitiveI64->value;
    evidence.scalarValid = value->primitiveI64->valid;
  } else if (value->ctypes && value->ctypes->scalarValue &&
             value->ctypes->scalarValid &&
             isKnownTrue(value->ctypes->scalarValid)) {
    evidence.scalarValue = value->ctypes->scalarValue;
    evidence.scalarValid = value->ctypes->scalarValid;
  } else {
    // Runtime-computed ints (dynamic fits-i64 flag) normalize through the
    // manifest unbox primitive.
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(value->contractName(), "unbox.i64");
    if (!unbox ||
        unbox->function.getNumArguments() != value->physicalValues().size())
      return op.emitError()
             << "ctypes value assignment requires primitive integer evidence";
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op.getLoc(), *unbox, value->physicalValues());
    evidence.scalarValue = unboxCall.getResult(0);
    evidence.scalarValid = constantI1(builder, op.getLoc(), true);
  }

  mlir::Value storageAddress = cdataStorageAddress(evidence);
  mlir::Value storageValid = cdataStorageAddressValid(evidence);
  if (storageAddress && storageValid && isKnownTrue(storageValid)) {
    std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
    std::optional<CtypesLayout> layout =
        ctypesStaticLayoutForType(module, evidence.ctype, facts);
    if (!layout)
      layout = ctypesStaticLayout(module, evidence.ctypeName, facts);
    if (!layout)
      return op.emitError() << "ctypes value assignment requires a static "
                               "layout for "
                            << evidence.ctypeName;
    // The normalized scalar evidence (known-valid i64) is what stores; the
    // original bundle may only carry a dynamic fits-i64 flag.
    RuntimeBundle scalarSource =
        RuntimeBundle::object(evidence.ctype ? evidence.ctype
                                             : op.getValue().getType(),
                              {});
    RuntimeCtypesEvidence scalarEvidence;
    scalarEvidence.kind = RuntimeCtypesEvidence::Kind::Cell;
    scalarEvidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
    scalarEvidence.ctypeName = evidence.ctypeName;
    scalarEvidence.ctype = evidence.ctype;
    scalarEvidence.scalarValue = evidence.scalarValue;
    scalarEvidence.scalarValid = constantI1(builder, op.getLoc(), true);
    scalarSource.ctypes = std::move(scalarEvidence);
    if (mlir::failed(storeCtypesValueToAddress(
            op, builder, module, storageAddress, evidence.ctype,
            evidence.ctypeName, *layout, scalarSource, facts)))
      return mlir::failure();
  }

  RuntimeBundle updated = object;
  updated.ctypes = std::move(evidence);
  valueBundles[op.getObject()] = std::move(updated);
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
  // A scalar/pointer field assigned a RUNTIME int (dynamic fits-i64 flag)
  // needs a known-valid primitive: normalize via unbox so the store accepts
  // it (extractNativeInteger/PointerArgument require statically-valid scalars).
  RuntimeBundle normalized = *value;
  if ((isIntegerScalarLayout(field->layout) ||
       isPointerScalarLayout(field->layout)) &&
      value->contractName() == "builtins.int" &&
      !(value->primitiveI64 && value->primitiveI64->valid &&
        isKnownTrue(value->primitiveI64->valid))) {
    mlir::Value raw;
    if (value->primitiveI64 && value->primitiveI64->value) {
      raw = value->primitiveI64->value;
    } else if (std::optional<RuntimeSymbol> unbox = manifest.primitive(
                   value->contractName(), "unbox.i64")) {
      raw = RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *unbox,
                                                    value->physicalValues())
                .getResult(0);
    }
    if (raw) {
      normalized.primitiveI64 =
          RuntimePrimitiveI64Evidence{raw, constantI1(builder, op.getLoc(),
                                                      true)};
    }
  }
  if (mlir::failed(storeCtypesValueToAddress(op, builder, module, fieldAddress,
                                             field->type, field->contract,
                                             field->layout, normalized, facts)))
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

  // Materialize the zero-initialized storage HERE: classes without a source
  // __init__ (the common Structure-subclass shape) never emit py.init, so
  // the cell must be usable straight from construction. A following
  // lowerErasedCtypesInit simply re-materializes with its arguments.
  builder.setInsertionPoint(op);
  mlir::FailureOr<RuntimeBundle> materialized = materializeCtypesCell(
      op, builder, module, op.getInstance().getType(), contract, {});
  if (mlir::failed(materialized))
    return mlir::failure();
  valueBundles[op.getInstance()] = std::move(*materialized);
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

// `HANDLER(f)` where HANDLER came from ctypes.CFUNCTYPE: wrap the compiled
// function in a C-ABI callback. The thunk itself can only exist at the LLVM
// layer (a func.func has no takeable address before conversion), so this
// step records the request as an ADDRESS PLACEHOLDER: a declared
// `() -> i64` function carrying the native signature and the primitive-ABI
// clone target as attributes; the callback-thunk materialization phase
// (after convert-to-llvm) generates the thunk and fills the body with
// addressof + ptrtoint.
mlir::LogicalResult RuntimeBundleLowerer::lowerCtypesCallbackConstruction(
    py::CallOp op, const RuntimeBundle &callable,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (op.getNumResults() != 1 || sources.size() != 1 || !sources.front())
    return op.emitError()
           << "ctypes callback construction expects one function argument";
  const RuntimeBundle &target = *sources.front();

  // CALL-THROUGH-ADDRESS: `PROTO(addr)` where addr is a runtime integer builds
  // a CFuncPtr whose target is a foreign function at that address (the inverse
  // of the callback thunk). Calling it emits an INDIRECT native call through
  // the address (see lowerStaticCtypesNativeCall). This is how signal-safe
  // code invokes a pre-resolved libc function pointer.
  if (target.functionTarget.empty() && target.primitiveI64 &&
      target.primitiveI64->value && target.primitiveI64->valid &&
      isKnownTrue(target.primitiveI64->valid)) {
    RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Symbol;
    evidence.provenance = RuntimeCtypesEvidence::Provenance::ExternalAddress;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
    evidence.ctypeName = "_ctypes.CFuncPtr";
    evidence.ctype = ctypesContractType(context, "_ctypes.CFuncPtr");
    evidence.argTypes = callable.ctypes->argTypes;
    evidence.resultType = callable.ctypes->resultType;
    evidence.addressValue = target.primitiveI64->value;
    evidence.addressValid = target.primitiveI64->valid;
    result.ctypes = std::move(evidence);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  if (target.functionTarget.empty())
    return op.emitError()
           << "ctypes callback requires a direct reference to a module-level "
              "function or a runtime integer address";
  std::optional<std::string> clone =
      RuntimeBundleLowerer::primitiveI64CloneFor(target.functionTarget);
  if (!clone)
    return op.emitError()
           << "ctypes callback target '" << target.functionTarget
           << "' must take only int parameters and return int so its "
              "primitive ABI clone exists";

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  auto nativeCode =
      [&](llvm::StringRef contractName) -> std::optional<std::string> {
    std::optional<CtypesLayout> layout =
        ctypesStaticLayout(module, contractName, facts);
    if (!layout)
      return std::nullopt;
    if (isPointerScalarLayout(*layout))
      return std::string("p");
    if (layout->kind == CtypesLayout::ABIKind::SignedInteger)
      return "s" + std::to_string(layout->size * 8);
    if (layout->kind == CtypesLayout::ABIKind::UnsignedInteger)
      return "u" + std::to_string(layout->size * 8);
    return std::nullopt;
  };

  llvm::SmallVector<mlir::Attribute, 4> argCodes;
  for (const std::string &argContract : callable.ctypes->argTypes) {
    std::optional<std::string> code = nativeCode(argContract);
    if (!code)
      return op.emitError() << "ctypes callback argument type " << argContract
                            << " has no scalar native ABI";
    argCodes.push_back(builder.getStringAttr(*code));
  }
  std::string resultCode = "void";
  if (callable.ctypes->resultType &&
      *callable.ctypes->resultType != "types.NoneType") {
    std::optional<std::string> code = nativeCode(*callable.ctypes->resultType);
    if (!code)
      return op.emitError() << "ctypes callback result type "
                            << *callable.ctypes->resultType
                            << " has no scalar native ABI";
    resultCode = *code;
  }

  // The clone takes one logical int per callback parameter.
  if (mlir::func::FuncOp cloneFunction =
          module.lookupSymbol<mlir::func::FuncOp>(*clone)) {
    py::CallableType callableType = callableTypeOf(cloneFunction);
    if (!callableType ||
        callableType.getPositionalTypes().size() != argCodes.size())
      return op.emitError()
             << "ctypes callback signature has " << argCodes.size()
             << " parameters, but '" << target.functionTarget << "' takes "
             << (callableType ? callableType.getPositionalTypes().size() : 0);
  }

  std::string suffix = std::to_string(callbackThunkCounter++);
  std::string addressSymbol = "__ly_callback_address_" + suffix;
  std::string thunkSymbol = "__ly_callback_thunk_" + suffix;
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(module.getBody());
    auto addressFunction = mlir::func::FuncOp::create(
        builder, op.getLoc(), addressSymbol,
        builder.getFunctionType({}, {builder.getI64Type()}));
    addressFunction.setPrivate();
    addressFunction->setAttr("ly.callback.thunk",
                             builder.getStringAttr(thunkSymbol));
    // A SymbolRef (not a plain string) so symbol DCE sees the clone as used.
    addressFunction->setAttr(
        "ly.callback.target",
        mlir::FlatSymbolRefAttr::get(builder.getContext(), *clone));
    addressFunction->setAttr("ly.callback.args",
                             builder.getArrayAttr(argCodes));
    addressFunction->setAttr("ly.callback.result",
                             builder.getStringAttr(resultCode));
  }

  builder.setInsertionPoint(op);
  auto addressCall = mlir::func::CallOp::create(
      builder, op.getLoc(), addressSymbol,
      mlir::TypeRange{builder.getI64Type()}, mlir::ValueRange{});

  RuntimeBundle result = RuntimeBundle::object(op.getResult(0).getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::CallbackThunk;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = "_ctypes.CFuncPtr";
  evidence.ctype = ctypesContractType(context, "_ctypes.CFuncPtr");
  evidence.argTypes = callable.ctypes->argTypes;
  evidence.resultType = callable.ctypes->resultType;
  evidence.scalarValue = addressCall.getResult(0);
  evidence.scalarValid = constantI1(builder, op.getLoc(), true);
  result.ctypes = std::move(evidence);
  valueBundles[op.getResult(0)] = std::move(result);
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

  if (*contract == "_ctypes.CFuncPtr" && callable.ctypes &&
      callable.ctypes->provenance ==
          RuntimeCtypesEvidence::Provenance::CallbackThunk)
    return RuntimeBundleLowerer::lowerCtypesCallbackConstruction(op, callable,
                                                                 sources);

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
  if (moduleName == "ctypes" &&
      isStaticCtypesFunctionName(*context, methodName)) {
    RuntimeBundle callable = RuntimeBundle::builtinCallable(
        op.getCallable().getType(),
        (llvm::Twine("ctypes.") + methodName).str());
    return RuntimeBundleLowerer::lowerStaticCtypesCall(op, callable);
  }
  if (std::optional<std::string> contract =
          ctypesModuleAttrContract(*context, moduleName, methodName)) {
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

} // namespace py::lowering
