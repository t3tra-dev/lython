#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/BoxLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cstddef>

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

  // ⭐ Mark every mutable container this literal ABSORBS. It now has a second
  // holder that can mutate it, so its own mutations may no longer take the
  // evidence arm; the evidence itself stays, because a READ through it is
  // still right and dropping it aborts at runtime (measured three ways, see
  // `sharedWithHolder`).
  auto markAbsorbedContainers = [&](mlir::ValueRange values) {
    for (mlir::Value value : values) {
      auto found = valueBundles.find(value);
      if (found == valueBundles.end())
        continue;
      RuntimeBundle &element = found->second;
      if (element.kind != RuntimeBundle::Kind::Object ||
          !RuntimeBundleLowerer::isMutableContainerContractName(
              element.contractName()))
        continue;
      element.sharedWithHolder = true;
    }
  };
  markAbsorbedContainers(op.getValues());

  llvm::SmallVector<RuntimeValue, 8> elements;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> elementBundles;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> dictKeyBundles;
  llvm::SmallVector<std::shared_ptr<RuntimeBundle>, 8> dictValueBundles;
  llvm::SmallVector<std::string, 8> keys;
  // The SSA value each key/value entry came from, parallel to
  // dictKeyBundles/dictValueBundles. Null marks an entry the lowering minted
  // itself (a materialized static key), which has no other user by
  // construction.
  llvm::SmallVector<mlir::Value, 8> logicalKeySources;
  llvm::SmallVector<mlir::Value, 8> logicalValueSources;
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
        logicalKeySources.push_back(mlir::Value{});
      } else {
        if (!keyBundle || keyBundle->kind != RuntimeBundle::Kind::Object)
          return op.emitError()
                 << "dict literal key has no lowered object bundle";
        dictKeyBundles.push_back(std::make_shared<RuntimeBundle>(*keyBundle));
        logicalKeySources.push_back(values[index]);
      }
      if (!key)
        allStaticStringKeys = false;

      const RuntimeBundle *valueBundle =
          RuntimeBundleLowerer::bundleFor(values[index + 1]);
      if (!valueBundle || valueBundle->kind != RuntimeBundle::Kind::Object)
        return op.emitError()
               << "dict literal value has no lowered object bundle";
      dictValueBundles.push_back(std::make_shared<RuntimeBundle>(*valueBundle));
      logicalValueSources.push_back(values[index + 1]);
      if (key) {
        keys.push_back(*key);
        elements.push_back(valueBundle->objectValue);
      }
    }
    if (!allStaticStringKeys) {
      keys.clear();
      elements.clear();
      // Non-static keys (user-class instances, runtime strings) have no
      // compile-time evidence identity, so the literal builds the SAME
      // runtime probe dict the incremental `d = {}; d[k] = v` path uses:
      // LyDict_New once, then one setitem_box insert per entry (user
      // __hash__/__eq__ dispatch and duplicate-key last-wins live in the
      // probe). Parking the entries as evidence instead leaves the bundle
      // in a dead zone no getitem/setitem dispatch accepts.
      std::optional<RuntimeSymbol> setItemBox =
          manifest.primitive("builtins.dict", "setitem_box");
      if (!setItemBox)
        return op.emitError()
               << "runtime manifest has no dict setitem_box primitive";
      // Arity 0: LyDict_New's argument is the LIVE entry count (the evidence
      // path fills its slots directly), not a capacity hint -- the probe
      // inserts below grow the count entry by entry, exactly like the
      // incremental path starting from `{}`.
      RuntimeBundle bundle;
      if (mlir::failed(materializeArityObject(op, op.getResult().getType(),
                                              /*arity=*/0, bundle, {}, {})))
        return mlir::failure();
      mlir::Location loc = op.getLoc();
      for (auto [keyBundle, valueBundle] :
           llvm::zip(dictKeyBundles, dictValueBundles)) {
        mlir::FailureOr<RuntimeBundle> payloadKey =
            RuntimeBundleLowerer::materializePayloadObjectBundle(op,
                                                                 *keyBundle);
        if (mlir::failed(payloadKey))
          return mlir::failure();
        mlir::FailureOr<RuntimeBundle> payloadValue =
            RuntimeBundleLowerer::materializePayloadObjectBundle(
                op, *valueBundle);
        if (mlir::failed(payloadValue))
          return mlir::failure();
        mlir::Block *retainBlock = builder.getInsertionBlock();
        mlir::Operation *retainAnchor = insertionAnchor(builder);
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadKey, "dict.literal.key")))
          return mlir::failure();
        if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
                op, *payloadValue, "dict.literal")))
          return mlir::failure();
        // The static-key lowering charges its slot retains to the container
        // (Core/CollectionPayload.cpp); this one never did, so `{i: 1}` handed
        // the affine walk retains with no parent to park them under.
        chargeSlotRetainsToParent(builder, retainBlock, retainAnchor, bundle);
        auto transientBox =
            [&](const RuntimeBundle &entry) -> mlir::FailureOr<mlir::Value> {
          return RuntimeBundleLowerer::transientPayloadBox(op, entry,
                                                           /*ownsPayload=*/true);
        };
        mlir::FailureOr<mlir::Value> keyBox = transientBox(*payloadKey);
        if (mlir::failed(keyBox))
          return mlir::failure();
        mlir::FailureOr<mlir::Value> valueBox = transientBox(*payloadValue);
        if (mlir::failed(valueBox))
          return mlir::failure();
        llvm::SmallVector<mlir::Value, 8> operands(
            bundle.physicalValues().begin(), bundle.physicalValues().end());
        operands.push_back(*keyBox);
        operands.push_back(*valueBox);
        mlir::func::CallOp call =
            RuntimeBundleLowerer::createRuntimeCall(loc, *setItemBox,
                                                    operands);
        RuntimeBundle updated;
        if (mlir::failed(RuntimeBundleLowerer::rebindMutatedContainer(
                op, bundle, call.getResults(), updated)))
          return mlir::failure();
        bundle = std::move(updated);
      }
      valueBundles[op.getResult()] = std::move(bundle);
      erase.push_back(op);
      return mlir::success();
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
  llvm::SmallVector<mlir::Value, 8> logicalElementSources(values.begin(),
                                                          values.end());
  if (contractName != "builtins.dict" &&
      mlir::failed(RuntimeBundleLowerer::initializeSequencePayload(
          op, bundle, bundle.sequenceElementBundles, logicalElementSources)))
    return mlir::failure();
  if (contractName == "builtins.dict") {
    bundle.mappingKeyBundles.append(dictKeyBundles.begin(),
                                    dictKeyBundles.end());
    bundle.mappingValueBundles.append(dictValueBundles.begin(),
                                      dictValueBundles.end());
    if (!dictKeyBundles.empty() &&
        mlir::failed(RuntimeBundleLowerer::initializeDictPayload(
            op, bundle, dictKeyBundles, dictValueBundles, logicalKeySources,
            logicalValueSources)))
      return mlir::failure();
  }
  valueBundles[op.getResult()] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerBindingRef(py::BindingRefOp op) {
  if (RuntimeBundleLowerer::isStaticCtypesBinding(op.getBinding()))
    return RuntimeBundleLowerer::lowerStaticCtypesBindingRef(op);

  // sys.argv is a runtime module attribute, not a folded constant or a
  // callable: each reference materializes the list[str] through the manifest
  // accessor (mutation through the temporary is rejected upstream by the
  // structural-mutation receiver check, so the fresh list cannot silently
  // drop writes).
  if (op.getBinding() == "sys.argv") {
    std::optional<RuntimeSymbol> accessor =
        manifest.primitive("builtins.list", "sys_argv");
    if (!accessor)
      return op.emitError()
             << "runtime manifest has no builtins.list sys_argv primitive";
    builder.setInsertionPoint(op);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *accessor, {});
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op, op.getResult().getType(), call, result)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

  // sys.stdout/stderr resolve to the immortal _io.TextIOWrapper singletons
  // through their manifest accessors; the borrowed handle needs no refcount
  // traffic.
  if (op.getBinding() == "sys.stdout" || op.getBinding() == "sys.stderr") {
    std::optional<RuntimeSymbol> accessor = manifest.primitive(
        "_io.TextIOWrapper",
        op.getBinding() == "sys.stdout" ? "sys_stdout" : "sys_stderr");
    if (!accessor)
      return op.emitError() << "runtime manifest has no _io.TextIOWrapper "
                            << "accessor for " << op.getBinding();
    builder.setInsertionPoint(op);
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *accessor, {});
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
            op.getOperation(), op.getResult().getType(), call.getResults(),
            result, ownership::OwnershipKind::Borrow)))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }

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

  // ⭐ AN OBJECT'S IDENTITY IS THE SOURCE FUNCTION'S, never the unboxed
  // clone's. The retarget above is for a DIRECT call from inside the unboxed
  // lane, where calling the clone is the whole point; here a real function
  // object is being built, and its target id is what every call site compares
  // against -- and a call site dispatches on the SOURCE symbol. A generator
  // that yields a function was the shape that showed it, because a generator
  // body IS a clone:
  //
  //     def five() -> int: return 5
  //     def gen(): yield five
  //     for f in gen(): print(f())   # TypeError: callable target is not
  //                                  # available -- the object carried the
  //                                  # clone's id, the dispatch compared the
  //                                  # source's
  //
  // The str- and float-returning spellings worked: neither has a clone. The
  // unboxed lane is not lost -- the call site still reaches the clone through
  // emitPrimitiveI64CloneFallbackResult, which is keyed on the source symbol.
  if (targetFunction != function) {
    targetFunction = function;
    bundle = RuntimeBundle::object(functionContract, {});
    bundle.functionTarget = targetFunction.getSymName().str();
    if (mlir::failed(appendClosureValues(op, targetFunction, bundle)))
      return mlir::failure();
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
  // ⭐ THE CAPTURES GO ON THE OBJECT. A function VALUE that crosses a boundary
  // the compiler cannot see through keeps its target id and, until this,
  // nothing else -- so the candidate walk at the call site dropped every
  // capture-taking function and the dispatch answered "callable target is not
  // available". CPython puts them in `__closure__`, which is the word this
  // fills.
  //
  // ⛔ BOXED, one per capture. A box is the one representation that holds any
  // contract in a single slot, and the read side already knows how to take a
  // value back out of one.
  RuntimeBundle captures = bundle;
  if (mlir::failed(appendClosureValues(op, targetFunction, captures)))
    return mlir::failure();
  mlir::Value closureWord =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
          .getResult();
  if (!captures.closureValues.empty()) {
    mlir::FailureOr<mlir::Value> stored =
        RuntimeBundleLowerer::materializeClosureStore(op,
                                                      captures.closureValues);
    if (mlir::failed(stored))
      return mlir::failure();
    closureWord = *stored;
  }

  llvm::SmallVector<mlir::Value, 6> operands;
  operands.push_back(
      mlir::arith::ConstantIntOp::create(
          builder, op.getLoc(),
          RuntimeBundleLowerer::functionTargetId(targetFunction.getSymName()),
          64)
          .getResult());
  auto zero = [&] {
    return mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
        .getResult();
  };
  operands.push_back(zero()); // defaults
  operands.push_back(zero()); // kwdefaults
  operands.push_back(closureWord);
  operands.push_back(zero()); // annotations
  operands.push_back(zero()); // module

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

// The closure store for one function object: a block of boxes, one per
// capture, each holding an owned reference the object drops when it dies.
mlir::FailureOr<mlir::Value> RuntimeBundleLowerer::materializeClosureStore(
    mlir::Operation *op, llvm::ArrayRef<RuntimeValue> captures) {
  std::optional<RuntimeSymbol> makeStore =
      manifest.primitive("builtins.function", "closure_new");
  std::optional<RuntimeSymbol> slotOf =
      manifest.primitive("builtins.function", "closure_slot");
  if (!makeStore || !slotOf)
    return op->emitError() << "runtime manifest has no builtins.function "
                              "closure_new / closure_slot primitive";
  mlir::Location loc = op->getLoc();
  mlir::Value count = mlir::arith::ConstantIntOp::create(
                          builder, loc,
                          static_cast<std::int64_t>(captures.size()), 64)
                          .getResult();
  mlir::Value block =
      RuntimeBundleLowerer::createRuntimeCall(loc, *makeStore,
                                              mlir::ValueRange{count})
          .getResult(0);
  // ⛔ THE WORDS ARE WRITTEN IN PLACE, not copied out of a box the frame would
  // then own. Allocating a box here and copying it left an owned
  // `memref<5xi64>` the refcount phases had to release, and the program
  // aborted on the release it could not place.
  for (auto [index, capture] : llvm::enumerate(captures)) {
    // A capture with no runtime values has nothing to write; its slot stays
    // zeroed and the read side asks the contract for lanes it does not have.
    if (capture.values.empty())
      continue;
    RuntimeBundle one = RuntimeBundle::object(capture.contract, capture.values);
    mlir::FailureOr<RuntimeBundle> normalized =
        RuntimeBundleLowerer::normalizeBoxSource(op, one);
    if (mlir::failed(normalized))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> words =
        RuntimeBundleLowerer::objectPayloadHandleWords(op, *normalized,
                                                       /*retainPayload=*/true);
    if (mlir::failed(words))
      return mlir::failure();
    mlir::Value slotWord =
        RuntimeBundleLowerer::createRuntimeCall(
            loc, *slotOf,
            mlir::ValueRange{
                block,
                mlir::arith::ConstantIntOp::create(
                    builder, loc, static_cast<std::int64_t>(index), 64)
                    .getResult()})
            .getResult(0);
    mlir::Value slot = RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, slotWord,
        mlir::arith::ConstantIntOp::create(builder, loc,
                                           box_abi::kWordsPerBox, 64)
            .getResult(),
        box_abi::boxWordsType(builder));
    for (auto [word, value] : llvm::enumerate(*words))
      mlir::memref::StoreOp::create(
          builder, loc, value, slot,
          mlir::arith::ConstantIndexOp::create(
              builder, loc, static_cast<std::int64_t>(word))
              .getResult());
    if (mlir::failed(RuntimeBundleLowerer::retainAggregateSlot(
            op, *normalized, "closure.capture")))
      return mlir::failure();
  }
  return block;
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
    // ⭐ A `type[X]` CAPTURE CARRIES NO RUNTIME VALUE, the same way a `type[X]`
    // ARGUMENT occupies no ABI input: which class it is, is in the type, and
    // the callee's own parameter type rebuilds it. Its bundle is a
    // `TypeObject`, so the object-bundle test refused
    //
    //     def go() -> None:
    //         cls = A
    //         def build(n: int) -> A:
    //             return cls(n)
    //
    // with "closure capture 0 must be a lowered Python object bundle" -- while
    // the same class passed as an ARGUMENT, held in a FIELD or taken as a
    // DEFAULT all work.
    //
    // ⛔ The slot is still allocated and still counted: dropping it would shift
    // every later capture's `closure_slot` index, and the two sides of that
    // index are in different files. It stays zeroed, and the read asks the
    // contract for its lanes, which for a `type[X]` are none.
    if (captureBundle && captureBundle->kind == RuntimeBundle::Kind::TypeObject &&
        index < closureTypes.size() &&
        mlir::isa<py::TypeType>(closureTypes[index])) {
      bundle.closureValues.push_back(
          RuntimeValue::object(closureTypes[index], mlir::ValueRange{}));
      continue;
    }
    if (!captureBundle || captureBundle->kind != RuntimeBundle::Kind::Object)
      return op.emitError() << "closure capture " << index
                            << " must be a lowered Python object bundle";
    // ⭐ EVERY SIGNATURE IS ONE RUNTIME CONTRACT, so a captured FUNCTION reads
    // back as the erased `builtins.function` and no particular Callable
    // accepts it -- a nested def that captures a sibling nested def was
    // refused here for a program whose types agree:
    //
    //     def outer(n: int) -> int:
    //         def helper(k: int) -> int: return k + 1
    //         def rec(k: int) -> int: return helper(k)
    //         return rec(n)
    //
    // ⛔ Why NOT teach isAssignableTo that builtins.function accepts a
    // Callable: that gives up the static contract everywhere it is declared.
    // The closure input's declared type is the promise about which signature,
    // and it is the type this lane already carries.
    auto erasedFunctionCapture = [&] {
      auto contract =
          mlir::dyn_cast_if_present<py::ContractType>(captureBundle->contract);
      return contract && contract.getArguments().empty() &&
             contract.getContractName() == "builtins.function" &&
             mlir::isa<py::CallableType>(closureTypes[index]);
    };
    if (!py::isAssignableTo(captureBundle->contract, closureTypes[index],
                            op.getOperation()) &&
        !erasedFunctionCapture())
      return op.emitError()
             << "closure capture " << index << " has type "
             << captureBundle->contract << ", expected " << closureTypes[index];
    // A lazy unboxed int has no physical object values; the RuntimeValue we
    // park in closureValues cannot carry the raw/valid evidence pair, so the
    // capture must be boxed here (once, at the binding) rather than at every
    // downstream call site.
    if (captureBundle->physicalValues().empty() &&
        RuntimeBundleLowerer::hasLazyPrimitiveI64Object(*captureBundle)) {
      builder.setInsertionPoint(op);
      mlir::FailureOr<RuntimeValue> materialized =
          RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
              op.getOperation(), *captureBundle);
      if (mlir::failed(materialized))
        return mlir::failure();
      bundle.closureValues.push_back(*materialized);
      continue;
    }
    bundle.closureValues.push_back(captureBundle->objectValue);
  }
  // The nested function may mutate a captured container on any later call;
  // from here on the enclosing scope's compile-time contents evidence for
  // those captures is no longer authoritative.
  for (mlir::Value capture : op.getCaptures())
    demoteMutableContainerEvidenceFor(capture);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerAliasView(mlir::Operation *op, mlir::Value input,
                                     mlir::Value resultValue) {
  const RuntimeBundle *inputBundle = RuntimeBundleLowerer::bundleFor(input);
  if (!inputBundle)
    return op->emitError()
           << "aliasing contract view input has no lowered runtime bundle";

  if (inputBundle->kind == RuntimeBundle::Kind::Object &&
      mlir::isa<py::ContractType>(resultValue.getType())) {
    if (inputBundle->boxedObject &&
        py::isAssignableTo(inputBundle->boxedObject->objectValue.contract,
                           resultValue.getType(), op)) {
      // Copy before inserting: operator[] can rehash valueBundles and
      // invalidate inputBundle (which points into it).
      RuntimeBundle boxed = *inputBundle->boxedObject;
      valueBundles[resultValue] = std::move(boxed);
      erase.push_back(op);
      return mlir::success();
    }

    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expectedTypes =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, resultValue.getType(),
                                                   "alias view ABI");
    if (mlir::failed(expectedTypes))
      return mlir::failure();

    // ⛔ AN `object` VALUE IS A BOX, so refining one to a class is an UNBOX and
    // not an alias. Its single physical value is the box header, whose word 2
    // is the entity -- the class's own storage. Aliasing instead handed the
    // class's lanes the BOX, and the first field read then loaded word 0 of the
    // entity, which is the refcount: `isinstance(o, A)` followed by `o.n`
    // printed 1 for every A, silently, because a live object's refcount is 1.
    //
    // Why here and not at the `boxedObject` fast path above: that path answers
    // when the lowering still REMEMBERS the concrete object behind the handle,
    // which it does not when the value arrived as a parameter -- the only case
    // where the class test is doing real work.
    if (runtimeContractName(inputBundle->objectValue.contract) ==
            "builtins.object" &&
        runtimeContractName(resultValue.getType()) != "builtins.object" &&
        inputBundle->physicalValues().size() == 1) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();
      mlir::Value header = inputBundle->physicalValues().front();
      mlir::Type dynamicHeader = mlir::MemRefType::get(
          {mlir::ShapedType::kDynamic}, builder.getI64Type());
      if (header.getType() != dynamicHeader)
        header = mlir::memref::CastOp::create(builder, loc, dynamicHeader,
                                              header)
                     .getResult();
      mlir::Value entityIndex = mlir::arith::ConstantIndexOp::create(
          builder, loc, box_abi::kEntityWord);
      mlir::Value entityWord =
          mlir::memref::LoadOp::create(builder, loc, header, entityIndex)
              .getResult();
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> lanes =
          RuntimeBundleLowerer::lanesFromBoxEntity(
              builder, loc, entityWord, *expectedTypes,
              runtimeContractName(resultValue.getType()), op);
      if (mlir::failed(lanes))
        return mlir::failure();
      RuntimeBundle result;
      if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
              op, resultValue.getType(), *lanes, result)))
        return mlir::failure();
      // The box still owns the entity; this is a view of it for as long as the
      // box is live, which the narrowed branch is by construction.
      result.setObjectLogicalOwnership(/*ownsObject=*/false);
      valueBundles[resultValue] = std::move(result);
      erase.push_back(op);
      return mlir::success();
    }

    if (expectedTypes->size() <= inputBundle->physicalValues().size()) {
      bool prefixMatches = true;
      for (auto [index, expected] : llvm::enumerate(*expectedTypes)) {
        if (inputBundle->physicalValues()[index].getType() == expected)
          continue;
        prefixMatches = false;
        break;
      }
      if (prefixMatches) {
        llvm::SmallVector<mlir::Value, 4> values;
        values.append(inputBundle->physicalValues().begin(),
                      inputBundle->physicalValues().begin() +
                          expectedTypes->size());
        RuntimeBundle result = RuntimeBundle::objectWithOwnership(
            resultValue.getType(), values, inputBundle->objectValue.ownership);
        result.copyEvidenceFrom(*inputBundle);
        if (!result.boxedObject &&
            inputBundle->objectValue.contract != resultValue.getType() &&
            py::isAssignableTo(inputBundle->objectValue.contract,
                               resultValue.getType(), op)) {
          RuntimeBundle concrete = *inputBundle;
          concrete.setObjectLogicalOwnership(/*ownsObject=*/false);
          result.boxedObject =
              std::make_shared<RuntimeBundle>(std::move(concrete));
        }
        valueBundles[resultValue] = std::move(result);
        erase.push_back(op);
        return mlir::success();
      }
    }
  }

  // Copy before inserting: operator[] can rehash valueBundles and invalidate
  // inputBundle (which points into it).
  RuntimeBundle aliased = *inputBundle;
  valueBundles[resultValue] = std::move(aliased);
  erase.push_back(op);
  return mlir::success();
}

// ⭐ A STARRED ARGUMENT IS AN ARITY, AND THE TYPE CARRIES IT. Sequence
// evidence -- the elements a tuple LITERAL was built from -- is the cheap
// answer and does not survive a binding, so `add(*ys)` worked for
// `add(*(1, 2))` and was refused for the same tuple through a name. The
// static type of a positional tuple says how many members there are, which is
// all the expansion needs; the elements themselves come out of the payload the
// way any other container read does.
//
// ⛔ THE LENGTH IS CHECKED even though the type asserts it. A tuple that
// reaches here through a cast or a manifest result whose declared arity is
// wrong would otherwise read past its payload, which is a wild read and not a
// wrong answer. `tuple[T]` -- the arity-erased spelling, which is what
// `tuple[T, ...]` becomes -- has no arity to check against and is still
// refused.
mlir::FailureOr<llvm::SmallVector<RuntimeValue, 4>>
RuntimeBundleLowerer::starredSequenceElements(mlir::Operation *op,
                                              const RuntimeBundle &source,
                                              llvm::StringRef label) {
  if (source.kind != RuntimeBundle::Kind::Object)
    return op->emitError() << label << " must be a lowered object bundle";
  if (!source.sequenceIndices.empty())
    return op->emitError() << label << " has only partial sequence evidence";
  if (!source.sequenceElements.empty())
    return llvm::SmallVector<RuntimeValue, 4>(source.sequenceElements.begin(),
                                              source.sequenceElements.end());

  auto tuple =
      mlir::dyn_cast_if_present<py::ContractType>(source.objectValue.contract);
  if (!tuple || tuple.getContractName() != "builtins.tuple" ||
      tuple.getArguments().size() < 2)
    return op->emitError()
           << label
           << " needs sequence evidence or a tuple type that says how many "
              "members it has";
  if (!RuntimeBundleLowerer::containerHasRuntimePayload(source))
    return op->emitError() << label << " has neither elements nor a payload";

  llvm::ArrayRef<mlir::Type> members = tuple.getArguments();
  mlir::Location loc = op->getLoc();
  mlir::FailureOr<mlir::Value> lengthOr =
      RuntimeBundleLowerer::loadContainerLength(op, source, label);
  if (mlir::failed(lengthOr))
    return mlir::failure();
  mlir::Value declared = mlir::arith::ConstantIntOp::create(
      builder, loc, static_cast<std::int64_t>(members.size()), 64);
  mlir::Value mismatched = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, *lengthOr, declared);
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       mismatched, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.TypeError",
            "starred call argument does not have the number of members its "
            "type declares")))
      return mlir::failure();
  }
  builder.setInsertionPointAfter(guard);

  mlir::Value one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  llvm::SmallVector<RuntimeValue, 4> elements;
  elements.reserve(members.size());
  for (auto [index, member] : llvm::enumerate(members)) {
    mlir::Value slot = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(index), 64);
    bool arrivesOwned = false;
    mlir::FailureOr<RuntimeValue> element =
        RuntimeBundleLowerer::payloadElementAt(op, source, slot, one, member,
                                               label, arrivesOwned);
    if (mlir::failed(element))
      return mlir::failure();
    elements.push_back(*element);
  }
  return elements;
}

mlir::LogicalResult RuntimeBundleLowerer::collectPackedObjectSources(
    mlir::Operation *op, mlir::Value packValue, llvm::StringRef label,
    llvm::SmallVectorImpl<const RuntimeBundle *> &sources,
    llvm::SmallVectorImpl<RuntimeBundle> *unpackedSources) {
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
      mlir::FailureOr<llvm::SmallVector<RuntimeValue, 4>> elements =
          RuntimeBundleLowerer::starredSequenceElements(
              op, *source, (label + " starred operand").str());
      if (mlir::failed(elements))
        return mlir::failure();
      for (const RuntimeValue &element : *elements) {
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
