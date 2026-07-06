#include "Runtime/Core/Lowerer.h"

namespace py::lowering {

namespace {

static bool canDeferFunctionObjectMaterialization(py::BindingRefOp op) {
  for (mlir::OpOperand &use : op.getResult().getUses()) {
    auto call = mlir::dyn_cast<py::CallOp>(use.getOwner());
    if (!call || call.getCallable() != op.getResult())
      return false;
  }
  return true;
}

static bool isCallArgumentPackUse(mlir::OpOperand &use) {
  mlir::Value value = use.get();
  if (auto call = mlir::dyn_cast<py::CallOp>(use.getOwner()))
    return call.getPosargs() == value || call.getKwnames() == value ||
           call.getKwvalues() == value;
  if (auto init = mlir::dyn_cast<py::InitOp>(use.getOwner()))
    return init.getPosargs() == value || init.getKwnames() == value ||
           init.getKwvalues() == value;
  if (auto newOp = mlir::dyn_cast<py::NewOp>(use.getOwner()))
    return newOp.getPosargs() == value || newOp.getKwnames() == value ||
           newOp.getKwvalues() == value;
  return false;
}

static bool isStaticMetadataSequenceUse(mlir::OpOperand &use) {
  mlir::Value value = use.get();
  if (auto attrSet = mlir::dyn_cast<py::AttrSetOp>(use.getOwner()))
    return attrSet.getValue() == value;
  return false;
}

static bool isOnlyUsedAsCallArgumentPack(py::PackOp op) {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses())
    if (!isCallArgumentPackUse(use))
      return false;
  return true;
}

static bool isOnlyUsedAsStaticMetadataSequence(py::PackOp op) {
  if (op.getResult().use_empty())
    return false;
  for (mlir::OpOperand &use : op.getResult().getUses())
    if (!isStaticMetadataSequenceUse(use))
      return false;
  return true;
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::lowerPack(py::PackOp op) {
  if (isOnlyUsedAsCallArgumentPack(op)) {
    RuntimeBundle bundle =
        RuntimeBundle::aggregate(op.getResult().getType(), op.getValues());
    if (auto flags =
            op->getAttrOfType<mlir::ArrayAttr>(kPackUnpackedOperandsAttr)) {
      if (flags.size() != op.getValues().size())
        return op.emitError()
               << kPackUnpackedOperandsAttr << " size must match pack operands";
      bundle.aggregateUnpackedOperands.reserve(flags.size());
      for (mlir::Attribute flag : flags) {
        auto boolAttr = mlir::dyn_cast<mlir::BoolAttr>(flag);
        if (!boolAttr)
          return op.emitError() << kPackUnpackedOperandsAttr
                                << " must contain bool attributes";
        bundle.aggregateUnpackedOperands.push_back(boolAttr.getValue());
      }
    }
    valueBundles[op.getResult()] = std::move(bundle);
    erase.push_back(op);
    return mlir::success();
  }

  std::string contractName = runtimeContractName(op.getResult().getType());
  if (contractName.empty()) {
    valueBundles[op.getResult()] =
        RuntimeBundle::aggregate(op.getResult().getType(), op.getValues());
    erase.push_back(op);
    return mlir::success();
  }

  llvm::SmallVector<RuntimeValue, 8> elements;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> elementBundles;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> dictKeyBundles;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> dictValueBundles;
  llvm::SmallVector<std::string, 8> keys;
  mlir::ValueRange values = op.getValues();
  if (contractName != "builtins.dict") {
    elements.reserve(values.size());
    elementBundles.reserve(values.size());
    bool allElementsObject = true;
    for (mlir::Value value : values) {
      const RuntimeBundle *bundle = RuntimeBundleLowerer::bundleFor(value);
      if (!bundle)
        return op.emitError()
               << contractName << " literal element has no lowered bundle";
      if (bundle->kind == RuntimeBundle::Kind::Object) {
        elements.push_back(bundle->objectValue);
      } else {
        allElementsObject = false;
      }
      elementBundles.push_back(std::make_shared<RuntimeBundle>(*bundle));
    }
    if (!allElementsObject) {
      if (!isOnlyUsedAsStaticMetadataSequence(op))
        return op.emitError()
               << contractName
               << " literal with non-object elements can only be used as "
                  "static metadata evidence";
      RuntimeBundle bundle =
          RuntimeBundle::object(op.getResult().getType(), {});
      bundle.sequenceElementBundles.append(elementBundles.begin(),
                                           elementBundles.end());
      valueBundles[op.getResult()] = std::move(bundle);
      erase.push_back(op);
      return mlir::success();
    }
  } else {
    if (values.size() % 2 != 0)
      return op.emitError() << "dict literal pack has an odd operand count";
    elements.reserve(values.size() / 2);
    keys.reserve(values.size() / 2);
    dictKeyBundles.reserve(values.size() / 2);
    dictValueBundles.reserve(values.size() / 2);
    bool allStaticStringKeys = true;
    for (unsigned index = 0, end = values.size(); index < end; index += 2) {
      std::optional<std::string> key =
          RuntimeBundleLowerer::keywordNameFromValue(values[index]);
      const RuntimeBundle *keyBundle =
          RuntimeBundleLowerer::bundleFor(values[index]);
      if (!keyBundle && key) {
        builder.setInsertionPoint(op);
        RuntimeBundle materializedKey;
        if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
                op, *key, materializedKey)))
          return mlir::failure();
        materializedKey.literalText = *key;
        dictKeyBundles.push_back(
            std::make_shared<RuntimeBundle>(std::move(materializedKey)));
      } else {
        if (!keyBundle || keyBundle->kind != RuntimeBundle::Kind::Object)
          return op.emitError()
                 << "dict literal key has no lowered object bundle";
        dictKeyBundles.push_back(std::make_shared<RuntimeBundle>(*keyBundle));
      }
      if (!key)
        allStaticStringKeys = false;

      const RuntimeBundle *valueBundle =
          RuntimeBundleLowerer::bundleFor(values[index + 1]);
      if (!valueBundle || valueBundle->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "dict literal value has no lowered object bundle";
      dictValueBundles.push_back(std::make_shared<RuntimeBundle>(*valueBundle));
      if (key) {
        keys.push_back(*key);
        elements.push_back(valueBundle->objectValue);
      }
    }
    if (!allStaticStringKeys) {
      keys.clear();
      elements.clear();
    }
  }

  RuntimeBundle bundle;
  std::uint64_t arity =
      contractName == "builtins.dict" ? values.size() / 2 : values.size();
  if (mlir::failed(materializeArityObject(op, op.getResult().getType(), arity,
                                          bundle, elements, keys)))
    return mlir::failure();
  if (contractName != "builtins.dict")
    bundle.sequenceElementBundles.append(elementBundles.begin(),
                                         elementBundles.end());
  if (contractName != "builtins.dict" &&
      mlir::failed(RuntimeBundleLowerer::initializeSequencePayload(
          op, bundle, bundle.sequenceElementBundles)))
    return mlir::failure();
  if (contractName == "builtins.dict") {
    bundle.mappingKeyBundles.append(dictKeyBundles.begin(),
                                    dictKeyBundles.end());
    bundle.mappingValueBundles.append(dictValueBundles.begin(),
                                      dictValueBundles.end());
    if (!dictKeyBundles.empty() &&
        mlir::failed(RuntimeBundleLowerer::initializeDictPayload(
            op, bundle, dictKeyBundles, dictValueBundles)))
      return mlir::failure();
  }
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBindingRef(py::BindingRefOp op) {
  if (RuntimeBundleLowerer::isStaticCtypesBinding(op.getBinding()))
    return RuntimeBundleLowerer::lowerStaticCtypesBindingRef(op);

  std::optional<RuntimeSymbol> builtin =
      manifest.builtinCallable(op.getBinding());
  if (builtin) {
    valueBundles[op.getResult()] = RuntimeBundle::builtinCallable(
        op.getResult().getType(), op.getBinding());
    erase.push_back(op);
    return mlir::success();
  }

  if (auto function = module.lookupSymbol<mlir::func::FuncOp>(op.getBinding()))
    return RuntimeBundleLowerer::lowerFunctionBindingRef(op, function);

  return op.emitError() << "unresolved runtime binding '" << op.getBinding()
                        << "'";
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerFunctionBindingRef(py::BindingRefOp op,
                                              mlir::func::FuncOp function) {
  mlir::func::FuncOp targetFunction = function;
  if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(
          op->getParentOfType<mlir::func::FuncOp>())) {
    if (std::optional<std::string> cloneName =
            RuntimeBundleLowerer::primitiveI64CloneFor(function.getSymName())) {
      if (mlir::func::FuncOp clone =
              module.lookupSymbol<mlir::func::FuncOp>(*cloneName))
        targetFunction = clone;
    }
  }

  auto callableType = function->getAttrOfType<mlir::TypeAttr>("callable_type");
  if (!callableType)
    return op.emitError() << "runtime binding '" << op.getBinding()
                          << "' names a func.func without callable_type";
  if (!mlir::isa<py::CallableType>(callableType.getValue()))
    return op.emitError()
           << "runtime binding '" << op.getBinding()
           << "' names a func.func whose callable_type is not Callable";

  mlir::Type functionContract =
      runtimeContractType(context, "builtins.function");
  RuntimeBundle bundle = RuntimeBundle::object(functionContract, {});
  bundle.functionTarget = targetFunction.getSymName().str();
  if (mlir::failed(appendClosureValues(op, targetFunction, bundle)))
    return mlir::failure();

  // A direct call only needs callable evidence. Emitting builtins.function here
  // would allocate a function object on every recursive/static call even though
  // the object identity is never observed.
  if (canDeferFunctionObjectMaterialization(op)) {
    valueBundles[op.getResult()] = std::move(bundle);
    erase.push_back(op);
    return mlir::success();
  }

  if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
    return op.emitError()
           << "protocol-typed function '" << op.getBinding()
           << "' must be called from statically known concrete arguments; "
              "materializing it as a runtime function object is not part of "
              "the static callable ABI";

  std::optional<RuntimeSymbol> initializer =
      manifest.initializer("builtins.function", "__new__");
  if (!initializer)
    return op.emitError()
           << "runtime manifest has no builtins.function.__new__";

  builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 6> operands;
  operands.push_back(
      mlir::arith::ConstantIntOp::create(
          builder, op.getLoc(),
          RuntimeBundleLowerer::functionTargetId(targetFunction.getSymName()),
          64)
          .getResult());
  for (unsigned index = 0; index < 5; ++index)
    operands.push_back(
        mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
            .getResult());

  mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *initializer, operands);
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, functionContract, call.getResults(), bundle)))
    return mlir::failure();
  bundle.functionTarget = targetFunction.getSymName().str();
  if (mlir::failed(appendClosureValues(op, targetFunction, bundle)))
    return mlir::failure();
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::appendClosureValues(
    py::BindingRefOp op, mlir::func::FuncOp function, RuntimeBundle &bundle) {
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      callableClosureTypes(function);
  if (closureTypes.size() != op.getCaptures().size())
    return op.emitError() << "binding '" << op.getBinding() << "' captures "
                          << op.getCaptures().size()
                          << " values, but target declares "
                          << closureTypes.size() << " closure inputs";

  for (auto [index, capture] : llvm::enumerate(op.getCaptures())) {
    const RuntimeBundle *captureBundle =
        RuntimeBundleLowerer::bundleFor(capture);
    if (!captureBundle || captureBundle->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "closure capture " << index
                            << " must be a lowered Python object bundle";
    if (!py::isAssignableTo(captureBundle->contract, closureTypes[index],
                            op.getOperation()))
      return op.emitError()
             << "closure capture " << index << " has type "
             << captureBundle->contract << ", expected " << closureTypes[index];
    bundle.closureValues.push_back(captureBundle->objectValue);
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue) {
  const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
  if (!inputBundle)
    return op->emitError()
           << "aliasing contract view input has no lowered runtime bundle";
  valueBundles[resultValue] = *inputBundle;
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectPackedObjectSources(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> *unpackedSources) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  if (unpackedSources) {
    std::size_t reserve = unpackedSources->size();
    for (auto [index, operand] : llvm::enumerate(pack->aggregateOperands)) {
      (void)operand;
      bool unpacked = index < pack->aggregateUnpackedOperands.size() &&
                      pack->aggregateUnpackedOperands[index] != 0;
      if (!unpacked)
        continue;
      const RuntimeBundle *source =
          RuntimeBundleLowerer::bundleFor(pack->aggregateOperands[index]);
      if (source)
        reserve += source->sequenceElements.size();
    }
    unpackedSources->reserve(reserve);
  }
  for (auto [index, operand] : llvm::enumerate(pack->aggregateOperands)) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(operand);
    if (!source)
      return op->emitError()
             << label << " operand has no lowered runtime bundle";
    bool unpacked = index < pack->aggregateUnpackedOperands.size() &&
                    pack->aggregateUnpackedOperands[index] != 0;
    if (unpacked) {
      if (!unpackedSources)
        return op->emitError()
               << label << " starred operand needs bundle storage";
      if (source->kind != RuntimeBundle::Kind::Object)
        return op->emitError()
               << label << " starred operand must be a Python object bundle";
      if (!source->sequenceIndices.empty())
        return op->emitError()
               << label << " starred operand has only partial sequence "
               << "evidence";
      if (source->sequenceElements.empty())
        return op->emitError()
               << label << " starred operand needs sequence evidence";
      for (const RuntimeValue &element : source->sequenceElements) {
        unpackedSources->push_back(
            RuntimeBundle::object(element.contract, element.values));
        sources.push_back(&unpackedSources->back());
      }
      continue;
    }
    sources.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::collectObjectSources(
    mlir::Operation *op, mlir::ValueRange values, llvm::StringRef message,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources) const {
  sources.reserve(sources.size() + values.size());
  for (mlir::Value value : values) {
    const RuntimeBundle *source = RuntimeBundleLowerer::bundleFor(value);
    if (!source)
      return op->emitError() << message;
    sources.push_back(source);
  }
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::requireEmptyAggregate(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label) const {
  const RuntimeBundle *pack = RuntimeBundleLowerer::bundleFor(packValue);
  if (!pack || pack->kind != RuntimeBundle::Kind::Aggregate)
    return op->emitError() << label << " must be a lowered aggregate bundle";
  if (!pack->aggregateOperands.empty())
    return op->emitError() << label << " lowering is not keyword-aware yet";
  return mlir::success();
}
} // namespace py::lowering
