#include "Runtime/Ctypes/Internal.h"
#include "llvm/TargetParser/Triple.h"

namespace py::lowering::ctypes {

mlir::FailureOr<RuntimeBundle>
materializeCtypesCell(mlir::Operation *op, mlir::OpBuilder &builder,
                      mlir::ModuleOp module, mlir::Type ctype,
                      llvm::StringRef ctypeName,
                      llvm::ArrayRef<const RuntimeBundle *> sources) {
  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Cell;
  evidence.provenance = RuntimeCtypesEvidence::Provenance::NativeCell;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Owner;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.ownsNativeStorage = true;

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  std::optional<CtypesArrayType> array = ctypesArrayType(module, ctype, facts);
  if (array) {
    if (!facts)
      return op->emitError() << evidence.ctypeName
                             << " requires TargetPlatformFacts before lowering";
    if (sources.size() > array->count)
      return op->emitError() << evidence.ctypeName << " initializer received "
                             << sources.size() << " positional values for "
                             << array->count << " ctypes array elements";

    builder.setInsertionPoint(op);
    mlir::Value storageAddress = addressOfZeroedNativeBytesAlloca(
        builder, op->getLoc(), array->layout.size, facts);
    mlir::Value valid = constantI1(builder, op->getLoc(), true);
    evidence.addressValue = storageAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = storageAddress;
    evidence.storageAddressValid = valid;

    for (auto [index, source] : llvm::enumerate(sources)) {
      if (!source)
        return op->emitError() << "ctypes array initializer argument " << index
                               << " has no evidence";
      mlir::Value elementAddress = addressWithOffset(
          builder, op->getLoc(), storageAddress,
          static_cast<std::int64_t>(index * array->elementLayout.size), facts);
      if (mlir::failed(storeCtypesValueToAddress(
              op, builder, module, elementAddress, array->elementType,
              array->elementContract, array->elementLayout, *source, facts)))
        return mlir::failure();
    }

    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence,
                               array->layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    return result;
  }

  py::ClassOp aggregateClass = lookupClassForContract(module, ctypeName);
  std::optional<CtypesAggregateLayout> aggregateLayout;
  if (aggregateClass)
    aggregateLayout = ctypesAggregateLayout(module, aggregateClass, facts, 0);
  if (aggregateLayout) {
    if (!facts)
      return op->emitError() << evidence.ctypeName
                             << " requires TargetPlatformFacts before lowering";
    if (sources.size() > aggregateLayout->fields.size())
      return op->emitError()
             << evidence.ctypeName << " initializer received " << sources.size()
             << " positional field values for "
             << aggregateLayout->fields.size() << " ctypes fields";

    builder.setInsertionPoint(op);
    mlir::Value storageAddress = addressOfZeroedNativeBytesAlloca(
        builder, op->getLoc(), aggregateLayout->layout.size, facts);
    mlir::Value valid = constantI1(builder, op->getLoc(), true);
    evidence.addressValue = storageAddress;
    evidence.addressValid = valid;
    evidence.storageAddressValue = storageAddress;
    evidence.storageAddressValid = valid;
    evidence.ownsNativeStorage = true;

    for (auto [index, source] : llvm::enumerate(sources)) {
      const CtypesFieldLayout &field = aggregateLayout->fields[index];
      if (!source)
        return op->emitError() << "ctypes aggregate initializer argument "
                               << index << " has no evidence";
      mlir::Value fieldAddress =
          addressWithOffset(builder, op->getLoc(), storageAddress,
                            static_cast<std::int64_t>(field.offset), facts);
      if (mlir::failed(storeCtypesValueToAddress(
              op, builder, module, fieldAddress, field.type, field.contract,
              field.layout, *source, facts)))
        return mlir::failure();
    }

    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence,
                               aggregateLayout->layout, /*writable=*/true);
    result.ctypes = std::move(evidence);
    return result;
  }

  if (!isCtypesIntegralLike(evidence.ctypeName))
    return op->emitError() << evidence.ctypeName
                           << " erased initializer is not implemented yet";
  if (sources.size() > 1)
    return op->emitError()
           << evidence.ctypeName
           << " erased initializer supports at most one value argument";

  builder.setInsertionPoint(op);
  if (sources.empty()) {
    evidence.scalarValue = constantI64(builder, op->getLoc(), 0);
    evidence.scalarValid = constantI1(builder, op->getLoc(), true);
  } else {
    const RuntimeBundle *source = sources.front();
    if (!source)
      return op->emitError() << "ctypes initializer argument has no evidence";
    if (source->primitiveI64) {
      evidence.scalarValue = source->primitiveI64->value;
      evidence.scalarValid = source->primitiveI64->valid;
    } else if (isCtypesVoidPointer(evidence.ctypeName) &&
               isNoneBundle(*source)) {
      evidence.scalarValue = constantI64(builder, op->getLoc(), 0);
      evidence.scalarValid = constantI1(builder, op->getLoc(), true);
    } else if (source->ctypes && source->ctypes->scalarValue &&
               source->ctypes->scalarValid) {
      evidence.scalarValue = source->ctypes->scalarValue;
      evidence.scalarValid = source->ctypes->scalarValid;
    } else {
      return op->emitError()
             << evidence.ctypeName
             << " erased initializer requires primitive integer evidence";
    }
  }

  std::optional<CtypesLayout> layout =
      ctypesStaticLayoutForType(module, ctype, facts);
  if (!layout)
    layout = ctypesStaticLayout(module, evidence.ctypeName, facts);
  if (layout && evidence.scalarValue && evidence.scalarValid &&
      isKnownTrue(evidence.scalarValid) &&
      (isIntegerScalarLayout(*layout) || isPointerScalarLayout(*layout))) {
    mlir::IntegerType nativeType = nativeIntegerType(builder, *layout);
    mlir::Value nativeValue = coerceNativeInteger(
        builder, op->getLoc(), evidence.scalarValue, nativeType);
    evidence.addressValue =
        addressOfNativeCellAlloca(builder, op->getLoc(), nativeValue, facts);
    evidence.addressValid = constantI1(builder, op->getLoc(), true);
    evidence.storageAddressValue = evidence.addressValue;
    evidence.storageAddressValid = evidence.addressValid;
    attachCtypesBufferEvidence(builder, op->getLoc(), result, evidence, *layout,
                               /*writable=*/true);
  }

  result.ctypes = std::move(evidence);
  return result;
}

mlir::FailureOr<RuntimeBundle>
materializeCtypesLibrary(mlir::Operation *op, mlir::ModuleOp module,
                         mlir::Type ctype, llvm::StringRef ctypeName,
                         llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() > 1)
    return op->emitError()
           << ctypeName << " static initializer supports at most one library "
           << "name";

  std::optional<TargetPlatformFacts> facts = targetPlatformFacts(module);
  if (!facts)
    return op->emitError() << ctypeName
                           << " requires TargetPlatformFacts before lowering";
  llvm::Triple triple(facts->triple);
  if (ctypeName == "ctypes.WinDLL" && !triple.isOSWindows())
    return op->emitError()
           << "ctypes.WinDLL is only supported for Windows targets, got "
           << facts->triple;

  RuntimeBundle result = RuntimeBundle::object(ctype, {});
  RuntimeCtypesEvidence evidence;
  evidence.kind = RuntimeCtypesEvidence::Kind::Library;
  evidence.lifetime = RuntimeCtypesEvidence::Lifetime::Static;
  evidence.ctypeName = ctypeName.str();
  evidence.ctype = ctype;
  evidence.abi = ctypesLibraryABI(ctypeName);
  if (sources.empty() || isNoneBundle(*sources.front())) {
    evidence.processLibrary = true;
    evidence.libraryName.clear();
  } else if (sources.front() && sources.front()->literalText) {
    evidence.processLibrary = false;
    evidence.libraryName = *sources.front()->literalText;
  } else {
    return op->emitError()
           << ctypeName
           << " requires a literal library name or None on the static path";
  }
  result.ctypes = std::move(evidence);
  return result;
}

std::optional<mlir::Value>
stackPointerForBorrowedScalar(mlir::Operation *op, mlir::OpBuilder &builder,
                              const RuntimeCtypesEvidence &evidence,
                              const std::optional<TargetPlatformFacts> &facts) {
  if (!evidence.callRegionBorrow || evidence.pointeeType.empty() ||
      !evidence.pointeeScalarValue || !evidence.pointeeScalarValid ||
      !isKnownTrue(evidence.pointeeScalarValid))
    return std::nullopt;
  std::optional<CtypesLayout> pointeeLayout =
      ctypesLayout(evidence.pointeeType, facts);
  if (!pointeeLayout || !isIntegerScalarLayout(*pointeeLayout))
    return std::nullopt;

  mlir::Location loc = op->getLoc();
  mlir::IntegerType nativeType = nativeIntegerType(builder, *pointeeLayout);
  mlir::Value nativeValue = coerceNativeInteger(
      builder, loc, evidence.pointeeScalarValue, nativeType);
  auto bufferType = mlir::MemRefType::get({1}, nativeType);
  mlir::Value buffer = mlir::memref::AllocaOp::create(builder, loc, bufferType);
  mlir::Value zero = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::memref::StoreOp::create(builder, loc, nativeValue, buffer,
                                mlir::ValueRange{zero});
  mlir::Value pointerIndex =
      mlir::memref::ExtractAlignedPointerAsIndexOp::create(builder, loc,
                                                           buffer);
  mlir::Value pointerInteger = mlir::arith::IndexCastOp::create(
      builder, loc, nativePointerIntegerType(builder, facts), pointerIndex);
  return mlir::LLVM::IntToPtrOp::create(
      builder, loc, nativePointerType(builder.getContext()), pointerInteger);
}

std::optional<mlir::Value>
extractNativePointerArgument(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts) {
  if (isNoneBundle(source)) {
    mlir::Value zero = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(), nativePointerIntegerType(builder, facts), 0);
    return mlir::LLVM::IntToPtrOp::create(
        builder, op->getLoc(), nativePointerType(builder.getContext()), zero);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer) {
    if (source.ctypes->storageAddressValue &&
        source.ctypes->storageAddressValid &&
        isKnownTrue(source.ctypes->storageAddressValid)) {
      mlir::IntegerType pointerInteger =
          nativePointerIntegerType(builder, facts);
      mlir::Value loaded = loadNativeIntegerFromAddress(
          builder, op->getLoc(), source.ctypes->storageAddressValue,
          pointerInteger, facts);
      return integerToNativePointer(builder, op->getLoc(), loaded, facts);
    }
    if (source.ctypes->addressValue && source.ctypes->addressValid &&
        isKnownTrue(source.ctypes->addressValid))
      return integerToNativePointer(builder, op->getLoc(),
                                    source.ctypes->addressValue, facts);
    return stackPointerForBorrowedScalar(op, builder, *source.ctypes, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    mlir::Value loaded = loadNativeIntegerFromAddress(
        builder, op->getLoc(), source.ctypes->storageAddressValue,
        pointerInteger, facts);
    return integerToNativePointer(builder, op->getLoc(), loaded, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return integerToNativePointer(builder, op->getLoc(),
                                  source.ctypes->scalarValue, facts);
  // A callback thunk address (CFuncPtr built by CFUNCTYPE(...)(f)) passes as
  // any pointer-typed foreign argument.
  if (source.ctypes &&
      source.ctypes->provenance ==
          RuntimeCtypesEvidence::Provenance::CallbackThunk &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return integerToNativePointer(builder, op->getLoc(),
                                  source.ctypes->scalarValue, facts);
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid))
    return integerToNativePointer(builder, op->getLoc(),
                                  source.primitiveI64->value, facts);
  return std::nullopt;
}

std::optional<mlir::Value>
extractPointerAddressInteger(mlir::Operation *op, mlir::OpBuilder &builder,
                             const RuntimeBundle &source,
                             const std::optional<TargetPlatformFacts> &facts) {
  if (isNoneBundle(source))
    return constantI64(builder, op->getLoc(), 0);
  // A callback thunk / call-through address CFuncPtr holds its function
  // address as scalar evidence; casting it to a pointer yields that address.
  if (source.ctypes &&
      (source.ctypes->provenance ==
           RuntimeCtypesEvidence::Provenance::CallbackThunk ||
       source.ctypes->provenance ==
           RuntimeCtypesEvidence::Provenance::ExternalAddress) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->scalarValue,
                               nativePointerIntegerType(builder, facts));
  if (source.ctypes &&
      (source.ctypes->provenance ==
           RuntimeCtypesEvidence::Provenance::CallbackThunk ||
       source.ctypes->provenance ==
           RuntimeCtypesEvidence::Provenance::ExternalAddress) &&
      source.ctypes->addressValue && source.ctypes->addressValid &&
      isKnownTrue(source.ctypes->addressValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->addressValue,
                               nativePointerIntegerType(builder, facts));
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        pointerInteger, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Cell &&
      isCtypesVoidPointer(source.ctypes->ctypeName) &&
      source.ctypes->scalarValue && source.ctypes->scalarValid &&
      isKnownTrue(source.ctypes->scalarValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->scalarValue,
                               nativePointerIntegerType(builder, facts));
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer &&
      source.ctypes->storageAddressValue &&
      source.ctypes->storageAddressValid &&
      isKnownTrue(source.ctypes->storageAddressValid)) {
    mlir::IntegerType pointerInteger = nativePointerIntegerType(builder, facts);
    return loadNativeIntegerFromAddress(builder, op->getLoc(),
                                        source.ctypes->storageAddressValue,
                                        pointerInteger, facts);
  }
  if (source.ctypes &&
      source.ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer &&
      source.ctypes->addressValue && source.ctypes->addressValid &&
      isKnownTrue(source.ctypes->addressValid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.ctypes->addressValue,
                               nativePointerIntegerType(builder, facts));
  if (source.primitiveI64 && source.primitiveI64->value &&
      source.primitiveI64->valid && isKnownTrue(source.primitiveI64->valid))
    return coerceNativeInteger(builder, op->getLoc(),
                               source.primitiveI64->value,
                               nativePointerIntegerType(builder, facts));
  return std::nullopt;
}

mlir::FailureOr<mlir::func::FuncOp> getOrCreateNativeDeclaration(
    mlir::Operation *op, mlir::ModuleOp module, mlir::OpBuilder &builder,
    llvm::StringRef name, mlir::FunctionType type,
    llvm::ArrayRef<std::string> argTypes, llvm::StringRef resultType,
    llvm::StringRef abi, bool processLibrary,
    const TargetPlatformFacts &facts) {
  auto nativeStringAttrMatches = [](mlir::Operation *decl,
                                    llvm::StringRef attrName,
                                    llvm::StringRef expected) {
    auto attr = decl->getAttrOfType<mlir::StringAttr>(attrName);
    return attr && attr.getValue() == expected;
  };
  auto nativeBoolAttrMatches = [](mlir::Operation *decl,
                                  llvm::StringRef attrName, bool expected) {
    auto attr = decl->getAttrOfType<mlir::BoolAttr>(attrName);
    return attr && attr.getValue() == expected;
  };
  auto nativeIntAttrMatches = [](mlir::Operation *decl,
                                 llvm::StringRef attrName,
                                 std::uint64_t expected) {
    auto attr = decl->getAttrOfType<mlir::IntegerAttr>(attrName);
    return attr && attr.getInt() >= 0 &&
           static_cast<std::uint64_t>(attr.getInt()) == expected;
  };
  auto nativeArgTypesMatch = [](mlir::Operation *decl,
                                llvm::ArrayRef<std::string> expected) {
    auto attr =
        decl->getAttrOfType<mlir::ArrayAttr>(py::native::kNativeArgTypesAttr);
    if (!attr || attr.size() != expected.size())
      return false;
    for (auto [index, raw] : llvm::enumerate(attr)) {
      auto value = mlir::dyn_cast<mlir::StringAttr>(raw);
      if (!value || value.getValue() != expected[index])
        return false;
    }
    return true;
  };

  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    if (existing.getFunctionType() != type)
      return op->emitError() << "native symbol '" << name
                             << "' was already declared with incompatible type";
    if (!nativeStringAttrMatches(existing, py::native::kNativeSymbolAttr,
                                 name) ||
        !nativeArgTypesMatch(existing, argTypes) ||
        !nativeStringAttrMatches(existing, py::native::kNativeResultTypeAttr,
                                 resultType) ||
        !nativeStringAttrMatches(existing, py::native::kNativeABIAttr, abi) ||
        !nativeBoolAttrMatches(existing, py::native::kNativeProcessLibraryAttr,
                               processLibrary) ||
        !nativeStringAttrMatches(existing, py::native::kNativeTargetTripleAttr,
                                 facts.triple) ||
        !nativeIntAttrMatches(existing,
                              py::native::kNativeTargetPointerWidthAttr,
                              facts.pointerWidth) ||
        !nativeIntAttrMatches(existing, py::native::kNativeTargetCLongWidthAttr,
                              facts.cLongWidth))
      return op->emitError()
             << "native symbol '" << name
             << "' was already declared with incompatible static ABI contract";
    return existing;
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto function =
      mlir::func::FuncOp::create(builder, module.getLoc(), name, type);
  function.setPrivate();
  function->setAttr(py::native::kNativeSymbolAttr, builder.getStringAttr(name));
  llvm::SmallVector<mlir::Attribute, 4> args;
  args.reserve(argTypes.size());
  for (llvm::StringRef argType : argTypes)
    args.push_back(builder.getStringAttr(argType));
  function->setAttr(py::native::kNativeArgTypesAttr,
                    builder.getArrayAttr(args));
  function->setAttr(py::native::kNativeResultTypeAttr,
                    builder.getStringAttr(resultType));
  function->setAttr(py::native::kNativeABIAttr, builder.getStringAttr(abi));
  function->setAttr(py::native::kNativeProcessLibraryAttr,
                    builder.getBoolAttr(processLibrary));
  function->setAttr(py::native::kNativeTargetTripleAttr,
                    builder.getStringAttr(facts.triple));
  function->setAttr(py::native::kNativeTargetPointerWidthAttr,
                    builder.getI64IntegerAttr(facts.pointerWidth));
  function->setAttr(py::native::kNativeTargetCLongWidthAttr,
                    builder.getI64IntegerAttr(facts.cLongWidth));
  return function;
}

} // namespace py::lowering::ctypes
