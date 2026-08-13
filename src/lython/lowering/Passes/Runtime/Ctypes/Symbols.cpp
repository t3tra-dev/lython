#include "Runtime/Ctypes/Internal.h"

namespace py::lowering {

using namespace ctypes;

// A library symbol, reached by either spelling. `lib["write"]` and
// `lib.write` name the same thing, and an attribute set on one has to be
// visible through the other, so the evidence and the alias bookkeeping are
// one body -- the two callers below supply only where the NAME and the
// aliased container come from.
mlir::LogicalResult RuntimeBundleLowerer::bindStaticCtypesLibrarySymbol(
    mlir::Operation *op, const RuntimeBundle &library, llvm::StringRef symbol,
    mlir::Value aliasOwner, mlir::Value result) {
  RuntimeBundle bundle;
  // A symbol whose attributes were already set (`fn.restype = ...`) has its
  // updated evidence parked on the library; take that over a fresh one.
  auto existing = library.fieldBundles.find(symbol);
  if (existing != library.fieldBundles.end()) {
    if (!existing->second)
      return op->emitError()
             << "ctypes symbol evidence for '" << symbol << "' is empty";
    bundle = *existing->second;
  } else {
    bundle = RuntimeBundle::object(result.getType(), {});
    RuntimeCtypesEvidence evidence;
    evidence.kind = RuntimeCtypesEvidence::Kind::Symbol;
    evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
    evidence.ctypeName = "_ctypes.CFuncPtr";
    evidence.ctype = result.getType();
    evidence.libraryName = library.ctypes->libraryName;
    evidence.abi = library.ctypes->abi;
    evidence.processLibrary = library.ctypes->processLibrary;
    evidence.symbolName = symbol.str();
    bundle.ctypes = std::move(evidence);
  }
  bundle.fieldAliasOwner = aliasOwner;
  bundle.fieldAliasName = symbol.str();
  valueBundles[result] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

// `lib["symbol"]`: the subscript spelling, keyed by a static string index.
mlir::LogicalResult RuntimeBundleLowerer::lowerStaticCtypesLibraryGetItem(
    py::GetItemOp op, const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();
  std::optional<std::string> symbol =
      RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
  if (!symbol)
    return op.emitError() << "ctypes library subscript requires a static "
                             "string symbol name";
  return RuntimeBundleLowerer::bindStaticCtypesLibrarySymbol(
      op, object, *symbol, op.getContainer(), op.getResult());
}

// `lib.symbol`: the attribute spelling.
mlir::LogicalResult
RuntimeBundleLowerer::lowerStaticCtypesAttrGet(py::AttrGetOp op,
                                               const RuntimeBundle &object) {
  if (!object.ctypes ||
      object.ctypes->kind != RuntimeCtypesEvidence::Kind::Library)
    return mlir::failure();
  return RuntimeBundleLowerer::bindStaticCtypesLibrarySymbol(
      op, object, op.getName(), op.getObject(), op.getResult());
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

} // namespace py::lowering
