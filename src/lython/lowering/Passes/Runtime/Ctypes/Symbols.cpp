#include "Runtime/Ctypes/Internal.h"

namespace py::runtime_lowering {

using namespace ctypes;

mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesAttrGet(py::AttrGetOp op,
                                               const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();

  auto existing = object.fieldBundles.find(op.getName());
  if (existing != object.fieldBundles.end()) {
    if (!existing->second)
      return op.emitError()
             << "ctypes symbol evidence for '" << op.getName() << "' is empty";
    RuntimeBundle result = *existing->second;
    result.fieldAliasOwner = op.getObject();
    result.fieldAliasName = op.getName().str();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  RuntimeBundle result = RuntimeBundle::object(op.getResult().getType(), {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Symbol;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = "_ctypes.CFuncPtr";
  evidence.ctype = op.getResult().getType();
  evidence.libraryName = object.ctypes->libraryName;
  evidence.abi = object.ctypes->abi;
  evidence.processLibrary = object.ctypes->processLibrary;
  evidence.symbolName = op.getName().str();
  result.ctypes = std::move(evidence);
  result.fieldAliasOwner = op.getObject();
  result.fieldAliasName = op.getName().str();
  valueBundles[op.getResult()] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesAttrSet(
    py::AttrSetOp op, const RuntimeBundle &object, const RuntimeBundle *value) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Symbol)
    return mlir::failure();
  if (!value)
    return op.emitError() << "ctypes symbol attribute value has no evidence";

  RuntimeBundle updated = object;
  RuntimeCtypesEvidence evidence = *object.ctypes;
  llvm::StringRef name = op.getName();
  if (name == "argtypes") {
    if (!isStaticSequenceBundle(*value))
      return op.emitError()
             << "ctypes argtypes must be a static list or tuple of ctypes "
             << "type objects";
    evidence.argTypes.clear();
    evidence.argTypes.reserve(value->sequenceElementBundles.size());
    for (auto [index, element] :
         llvm::enumerate(value->sequenceElementBundles)) {
      if (!element)
        return op.emitError()
               << "ctypes argtypes element " << index << " has no evidence";
      std::optional<std::string> ctype = ctypesTypeObjectName(*element);
      if (!ctype)
        return op.emitError() << "ctypes argtypes element " << index
                              << " must be a ctypes type object";
      evidence.argTypes.push_back(std::move(*ctype));
    }
  } else if (name == "restype") {
    if (isNoneBundle(*value)) {
      evidence.resultType = std::string("types.NoneType");
    } else {
      std::optional<std::string> ctype = ctypesTypeObjectName(*value);
      if (!ctype)
        return op.emitError()
               << "ctypes restype must be a ctypes type object or None";
      evidence.resultType = std::move(*ctype);
    }
  } else {
    return op.emitError() << "ctypes symbol attribute '" << name
                          << "' is not supported on the static path";
  }

  updated.ctypes = std::move(evidence);
  valueBundles[op.getObject()] = updated;
  if (updated.fieldAliasOwner && !updated.fieldAliasName.empty()) {
    auto owner = valueBundles.find(updated.fieldAliasOwner);
    if (owner != valueBundles.end()) {
      RuntimeBundle ownerBundle = owner->second;
      ownerBundle.fieldBundles[updated.fieldAliasName] =
          std::make_shared<RuntimeBundle>(updated);
      valueBundles[updated.fieldAliasOwner] = std::move(ownerBundle);
    }
  }
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::runtime_lowering
