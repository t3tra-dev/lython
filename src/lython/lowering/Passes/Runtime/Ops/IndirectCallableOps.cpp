#include "Runtime/Core/Lowerer.h"

#include "Common/RuntimeSupport.h"
#include "Runtime/ABI/BoxLayout.h"

#include "mlir/IR/IRMapping.h"

#include <cstddef>
#include <cstdlib>
#include <string>

namespace py::lowering {
namespace {

const RuntimeCallableAlternative *
findCallableAlternative(const RuntimeBundle &callable, llvm::StringRef target) {
  for (const RuntimeCallableAlternative &alternative :
       callable.callableAlternatives)
    if (alternative.functionTarget == target)
      return &alternative;
  return nullptr;
}

} // namespace

llvm::SmallVector<mlir::func::FuncOp, 8>
RuntimeBundleLowerer::collectIndirectCallableTargets(
    py::CallOp op, const RuntimeBundle &callableBundle) {
  llvm::SmallVector<mlir::func::FuncOp, 8> targets;

  py::CallableType expected =
      py::getCallableContract(op.getCallable().getType());
  if (!expected)
    expected = py::getCallableContract(op.getCallContract());
  if (!expected)
    return targets;

  auto visitCandidate = [&](mlir::func::FuncOp function) {
    if (function.isDeclaration() || !function->hasAttr("callable_type"))
      return;
    if (RuntimeBundleLowerer::isCallableProtocolTemplate(function))
      return;
    // Nor is a dispatcher: nothing makes a function object of one.
    if (RuntimeBundleLowerer::isIndirectCallDispatcher(function))
      return;
    // ⭐ A primitive-i64 clone is not a callable VALUE. It is an internal
    // specialization that takes its int arguments unboxed and returns its int
    // result the same way, reachable only from a site that knows to speculate
    // on it -- and it carries a copy of the original's `callable_type`, so it
    // matches here exactly as well as the original does.
    //
    // `Holder(make).call()`, with `_f: Callable[[], int]`, collected
    // `make__lyrt_prim_i64` and then consumed its result through the object
    // ABI: "function target 'make__lyrt_prim_i64' returned too few values for
    // result object ABI". The speculation path is keyed on the ORIGINAL's name,
    // so once the clone is the target nothing recognises it as one.
    if (RuntimeBundleLowerer::isPrimitiveI64CallableClone(function))
      return;
    // ⛔ Nor is a generator's or a coroutine's body: its `callable_type`
    // returns what the body returns, and calling the function makes the
    // generator, which an arm calling the body would not. A
    // `Callable[[Task], None]` matched the body of `async def _drive(self)`.
    if (function->hasAttr("ly.generator.body_result"))
      return;

    py::CallableType callable = callableTypeOf(function);
    if (!callable || callable.getResultTypes().size() != 1)
      return;

    llvm::StringRef functionName = function.getSymName();
    if (!callableBundle.functionTarget.empty() &&
        callableBundle.functionTarget != functionName)
      return;
    if (!callableBundle.callableAlternatives.empty() &&
        !findCallableAlternative(callableBundle, functionName))
      return;

    // Asked once per pair of types: the answer reads the two types and the
    // module's class hierarchy, which the lowering does not change, and the
    // same function types meet the same expected type at every call site.
    auto [assignable, fresh] =
        callableAssignable.try_emplace({callable, expected}, false);
    if (fresh)
      assignable->second =
          py::isAssignableTo(callable, expected, op.getOperation());
    if (!assignable->second)
      return;
    if (!RuntimeBundleLowerer::collectCallableArgumentPlan(
            op, callable, /*emitErrors=*/false))
      return;

    llvm::SmallVector<mlir::Type, 4> closureTypes =
        RuntimeBundleLowerer::callableClosureTypes(function);
    llvm::ArrayRef<RuntimeValue> closureValues = callableBundle.closureValues;
    if (!callableBundle.callableAlternatives.empty()) {
      const RuntimeCallableAlternative *alternative =
          findCallableAlternative(callableBundle, function.getSymName());
      if (!alternative)
        return;
      closureValues = alternative->closureValues;
    }
    // ⭐ A RUNTIME FUNCTION VALUE CARRIES ITS CAPTURES ON THE OBJECT, so the
    // absence of compile-time closure evidence is no longer a reason to drop
    // the candidate: the arm reads them back out of the object's closure store.
    // Requiring the arity to match here is what left a closure in a list or a
    // dict with no candidate at all -- "callable target is not available".
    if (closureTypes.size() != closureValues.size()) {
      if (!closureValues.empty())
        return;
      if (runtimeContractName(callableBundle.objectValue.contract) !=
          "builtins.function")
        return;
      targets.push_back(function);
      return;
    }
    for (auto [closureValue, closureType] :
         llvm::zip(closureValues, closureTypes)) {
      if (!py::isAssignableTo(closureValue.contract, closureType,
                              op.getOperation()))
        return;
    }

    targets.push_back(function);
  };
  // Every function, in module order, without entering their bodies: no
  // function holds another, so this meets the functions module.walk did, in
  // the same order.
  // ⛔ Not module.walk: it visits every operation of every body, once per
  // indirect call.
  module->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (auto function = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      visitCandidate(function);
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  return targets;
}

mlir::LogicalResult RuntimeBundleLowerer::appendBundlePhysicalOperands(
    mlir::Operation *op, const RuntimeBundle &bundle,
    llvm::ArrayRef<mlir::Type> expectedTypes,
    llvm::SmallVectorImpl<mlir::Value> &operands) {
  llvm::ArrayRef<mlir::Value> values = bundle.physicalValues();
  std::optional<RuntimeValue> materializedObject;
  if (values.empty() &&
      RuntimeBundleLowerer::hasLazyPrimitiveI64Object(bundle)) {
    mlir::FailureOr<RuntimeValue> value =
        RuntimeBundleLowerer::materializePrimitiveI64ObjectAtCurrentInsertion(
            op, bundle);
    if (mlir::failed(value))
      return mlir::failure();
    materializedObject = std::move(*value);
    values = materializedObject->values;
  }
  if (values.size() == expectedTypes.size()) {
    bool exact = true;
    for (auto [value, expected] : llvm::zip(values, expectedTypes)) {
      if (value.getType() != expected) {
        exact = false;
        break;
      }
    }
    if (exact) {
      operands.append(values.begin(), values.end());
      return mlir::success();
    }
  }

  if (expectedTypes.size() == 1 && bundle.kind == RuntimeBundle::Kind::Object &&
      isBuiltinsObjectHandleType(expectedTypes.front())) {
    if (!RuntimeBundleLowerer::isBuiltinsObjectContract(bundle.contract)) {
      mlir::FailureOr<RuntimeBundle> boxed =
          RuntimeBundleLowerer::boxRuntimeObjectAtCurrentInsertion(
              op, bundle, /*retainPayload=*/true);
      if (mlir::failed(boxed))
        return mlir::failure();
      llvm::ArrayRef<mlir::Value> values = boxed->physicalValues();
      if (values.size() == 1 &&
          values.front().getType() == expectedTypes.front()) {
        operands.push_back(values.front());
        return mlir::success();
      }
      return op->emitError()
             << "boxed indirect callable result for " << bundle.contractName()
             << " does not match expected object ABI "
             << describeTypeSequence(expectedTypes);
    }
    if (values.empty())
      return op->emitError() << "builtins.object result has no boxed handle";
    if (values.front().getType() != expectedTypes.front())
      return op->emitError()
             << "builtins.object result handle " << values.front().getType()
             << " does not match expected ABI "
             << describeTypeSequence(expectedTypes);
    operands.push_back(values.front());
    return mlir::success();
  }

  return op->emitError() << "cannot adapt runtime bundle "
                         << bundle.contractName() << " with physical values "
                         << describeValueTypes(values) << " to expected ABI "
                         << describeTypeSequence(expectedTypes);
}

// ⭐ THE CAPTURES, READ BACK OFF THE OBJECT. The function value's word 5 is
// the closure store the binding wrote: one box per capture, in the target's
// declared order. Each box hands its lanes back the same way a boxed container
// element does.
//
// ⛔ BORROWED. The store owns the captures for as long as the object lives, and
// the object lives across the call that is being lowered -- so the lanes are a
// view, not a transfer, and nothing here releases them.
mlir::FailureOr<llvm::SmallVector<RuntimeValue, 4>>
RuntimeBundleLowerer::closureValuesFromFunctionObject(
    mlir::Operation *op, const RuntimeBundle &callable,
    mlir::func::FuncOp target) {
  llvm::SmallVector<RuntimeValue, 4> values;
  llvm::SmallVector<mlir::Type, 4> closureTypes =
      RuntimeBundleLowerer::callableClosureTypes(target);
  if (closureTypes.empty())
    return values;
  std::optional<RuntimeSymbol> slotOf =
      manifest.primitive("builtins.function", "closure_slot");
  if (!slotOf)
    return op->emitError()
           << "runtime manifest has no builtins.function closure_slot";
  llvm::ArrayRef<mlir::Value> physical = callable.physicalValues();
  if (physical.size() != 1)
    return op->emitError()
           << "a function object is one handle, got " << physical.size();
  mlir::Location loc = op->getLoc();
  mlir::Value block =
      mlir::memref::LoadOp::create(
          builder, loc, physical.front(),
          mlir::arith::ConstantIndexOp::create(builder, loc, 5).getResult())
          .getResult();
  for (auto [index, closureType] : llvm::enumerate(closureTypes)) {
    // ⭐ THE SLOT'S SHAPE, NOT THE VALUE'S. `materializeClosureStore` writes
    // each capture through `normalizeBoxSource`, so what the slot holds is
    // whatever the contract's `box` primitive returns -- and reading it back as
    // the value's own lanes is the same question asked with the other
    // spelling. `builtins.bool` is the one contract where the two differ (its
    // value is an i1 and its box is a singleton header), and a generator that
    // captured one was refused: "builtins.bool has no statically sized entity
    // lane to rebuild a box from, got 'i1'".
    //
    // ⛔ Why NOT teach `lanesFromBoxEntity` to rebuild a scalar lane: the box
    // slot does not hold the i1 at all, so there would be nothing to rebuild
    // from. The pair below is the one a container element already uses.
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> laneTypes =
        RuntimeBundleLowerer::slotStorageShapesFor(op, closureType,
                                                   "closure capture ABI");
    if (mlir::failed(laneTypes))
      return mlir::failure();
    mlir::Value slotWord =
        RuntimeBundleLowerer::createRuntimeCall(
            loc, *slotOf,
            mlir::ValueRange{
                block, mlir::arith::ConstantIntOp::create(
                           builder, loc, static_cast<std::int64_t>(index), 64)
                           .getResult()})
            .getResult(0);
    mlir::Value slot = RuntimeBundleLowerer::memrefFromBoxWords(
        builder, loc, slotWord,
        mlir::arith::ConstantIntOp::create(builder, loc,
                                           box_abi::kWordsPerBox, 64)
            .getResult(),
        box_abi::slotWordsType(builder));
    mlir::Value entityWord =
        mlir::memref::LoadOp::create(
            builder, loc, slot,
            mlir::arith::ConstantIndexOp::create(builder, loc,
                                                 box_abi::kEntityWord)
                .getResult())
            .getResult();
    // ⭐ A UNION CAPTURE IS ONE BOX, and its lanes are rebuilt from the box the
    // way a container element's are. `slotStorageShapesFor` answers a union
    // with the VALUE's own lanes -- a union has no contract name to find a
    // `box` primitive under -- so lane 0 was the i64 TAG and the rebuild said
    // " has no statically sized entity lane to rebuild a box from, got 'i64'"
    // for a closure that captures an `Optional[int]`.
    if (auto captureUnion = mlir::dyn_cast<py::UnionType>(closureType)) {
      mlir::Value classWord =
          mlir::memref::LoadOp::create(
              builder, loc, slot,
              mlir::arith::ConstantIndexOp::create(builder, loc, 1).getResult())
              .getResult();
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> unionValues =
          RuntimeBundleLowerer::unionValuesFromBoxWords(op, captureUnion,
                                                        classWord, entityWord);
      if (mlir::failed(unionValues))
        return mlir::failure();
      values.push_back(RuntimeValue::objectWithOwnership(
          closureType,
          mlir::ValueRange{llvm::ArrayRef<mlir::Value>(unionValues->begin(),
                                                        unionValues->end())},
          ownership::OwnershipKind::Borrow));
      continue;
    }
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> lanes =
        RuntimeBundleLowerer::lanesFromBoxEntity(
            builder, loc, entityWord, *laneTypes,
            runtimeContractName(closureType), op, /*ownedRead=*/true);
    if (mlir::failed(lanes))
      return mlir::failure();
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> unboxed =
        RuntimeBundleLowerer::unboxSlotElementValues(op, closureType, *lanes);
    if (mlir::failed(unboxed))
      return mlir::failure();
    values.push_back(RuntimeValue::objectWithOwnership(
        closureType, mlir::ValueRange{*unboxed},
        ownership::OwnershipKind::Borrow));
  }
  return values;
}

bool RuntimeBundleLowerer::isIndirectCallDispatcher(
    mlir::func::FuncOp function) const {
  return function && function->hasAttr(kIndirectCallDispatcherAttr);
}

// An indirect call whose target only the function object knows dispatches
// over every function of a matching type in the program. Written out at the
// call site, that is a test and an argument-adapting arm per candidate, at
// every such call: a program with C of these calls and F candidates carried
// C x F arms, every one of them lowered, verified and compiled, and the call
// compared target ids one candidate at a time.
//
// So a call that needs no more than its argument types to dispatch goes
// through a dispatcher instead: a function per call shape (callable type,
// argument types, result type), taking the function object and the
// arguments, whose body is that same call -- written out once, as a switch on
// the target id -- and whose ABI is a Python function's, owned result and
// unboxed int lanes included. Its body lowers by the same code, from a call
// with nothing known about the callable, which is what this call had.
//
// ⛔ Not for a call that knows more: one with closure evidence or a named
// target, a candidate that takes argument or aggregate evidence the call
// site supplies, or keyword arguments -- the dispatcher's parameters carry
// types and nothing else. Those keep their arms where they are.
// ⛔ Not under kMinimumDispatcherTargets candidates. A few arms grow neither
// the program nor its compile, and the extra call is not free: at four
// candidates the dispatcher ran 1.5% slower than the arms, at 128 3.5% faster.
// ⛔ Not from a generator's body or resume clone, or a primitive-i64 clone:
// the state machine and the clone re-read those bodies, and no case has shown
// that a call to a function made mid-lowering survives that. They keep the
// arms, which do.
mlir::FailureOr<mlir::func::FuncOp> RuntimeBundleLowerer::indirectCallDispatcher(
    py::CallOp op, const RuntimeBundle &callable,
    llvm::ArrayRef<mlir::func::FuncOp> targets) {
  constexpr std::size_t kMinimumDispatcherTargets = 4;
  // LYTHON_ABLATE_INDIRECT_DISPATCHERS=1 writes every dispatch out at its
  // call again, for comparing the two with one binary.
  static const bool ablated =
      std::getenv("LYTHON_ABLATE_INDIRECT_DISPATCHERS") != nullptr;
  if (ablated || targets.size() < kMinimumDispatcherTargets ||
      !callable.functionTarget.empty() ||
      !callable.callableAlternatives.empty() ||
      !callable.closureValues.empty() || op.getNumResults() != 1)
    return mlir::func::FuncOp();
  auto enclosing = op->getParentOfType<mlir::func::FuncOp>();
  if (!enclosing || RuntimeBundleLowerer::isIndirectCallDispatcher(enclosing) ||
      RuntimeBundleLowerer::isPrimitiveI64CallableClone(enclosing) ||
      enclosing->hasAttr("ly.generator.resume") ||
      enclosing->hasAttr("ly.generator.body_result") ||
      RuntimeBundleLowerer::isCallableProtocolTemplate(enclosing))
    return mlir::func::FuncOp();
  auto posargs = op.getPosargs().getDefiningOp<py::PackOp>();
  auto kwnames = op.getKwnames().getDefiningOp<py::PackOp>();
  auto kwvalues = op.getKwvalues().getDefiningOp<py::PackOp>();
  if (!posargs || !kwnames || !kwvalues || !kwnames.getValues().empty() ||
      !kwvalues.getValues().empty())
    return mlir::func::FuncOp();
  for (mlir::func::FuncOp target : targets)
    if (callableArgumentEvidenceABIs.count(target.getSymName()) ||
        callableAggregateEvidenceABIs.count(target.getSymName()))
      return mlir::func::FuncOp();

  llvm::SmallVector<mlir::Type, 8> inputs{op.getCallable().getType()};
  for (mlir::Value argument : posargs.getValues())
    inputs.push_back(argument.getType());
  mlir::Type resultType = op.getResult(0).getType();
  std::string key;
  {
    llvm::raw_string_ostream stream(key);
    stream << op.getCallContract();
    for (mlir::Type input : inputs)
      stream << "|" << input;
    stream << "->" << resultType;
  }
  if (auto found = indirectCallDispatchers.find(key);
      found != indirectCallDispatchers.end())
    return found->second;

  mlir::OpBuilder::InsertionGuard guard(builder);
  // The call's own location: it makes the dispatcher a function of the
  // program, whose calls unwind through cleanups like any other's. Its frame
  // is the one thing it does not add (EH.cpp, emitTracebackFrame).
  mlir::Location loc = op.getLoc();
  builder.setInsertionPointToEnd(module.getBody());
  std::string name = (kIndirectCallDispatcherPrefix +
                      llvm::Twine(indirectCallDispatchers.size()))
                         .str();
  auto dispatcher = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, {resultType}));
  dispatcher.setPrivate();
  llvm::SmallVector<mlir::StringAttr, 8> names;
  llvm::SmallVector<mlir::BoolAttr, 8> defaults;
  llvm::SmallVector<mlir::Attribute, 8> defaultValues;
  for (unsigned index = 0; index < inputs.size(); ++index) {
    names.push_back(builder.getStringAttr("a" + std::to_string(index)));
    defaults.push_back(builder.getBoolAttr(false));
    defaultValues.push_back(builder.getUnitAttr());
  }
  py::CallableType callableType = py::CallableType::get(
      context, inputs, {}, mlir::Type(), mlir::Type(), {resultType}, names, {},
      defaults, {});
  dispatcher->setAttr("callable_type", mlir::TypeAttr::get(callableType));
  dispatcher->setAttr("callable_default_values",
                      builder.getArrayAttr(defaultValues));
  dispatcher->setAttr(kIndirectCallDispatcherAttr, builder.getUnitAttr());
  indirectCallDispatchers[key] = dispatcher;

  mlir::Block *entry = dispatcher.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  mlir::IRMapping mapping;
  mapping.map(op.getCallable(), entry->getArgument(0));
  for (auto [index, argument] : llvm::enumerate(posargs.getValues()))
    mapping.map(argument, entry->getArgument(index + 1));
  llvm::SmallVector<mlir::Operation *, 4> body;
  body.push_back(builder.clone(*posargs.getOperation(), mapping));
  body.push_back(builder.clone(*kwnames.getOperation(), mapping));
  body.push_back(builder.clone(*kwvalues.getOperation(), mapping));
  body.push_back(builder.clone(*op.getOperation(), mapping));
  mlir::func::ReturnOp::create(builder, loc, body.back()->getResult(0));

  llvm::ArrayRef<ownership::RuntimeDeallocator> deallocators =
      *settledDeallocators;
  auto memberOwnsALane = [&](mlir::Type member) {
    if (py::isPyNoneType(member))
      return false;
    std::string contract = runtimeContractName(member);
    return !contract.empty() &&
           llvm::any_of(deallocators,
                        [&](const ownership::RuntimeDeallocator &candidate) {
                          return candidate.contractName == contract;
                        });
  };
  if (mlir::failed(RuntimeBundleLowerer::prepareCallableFunctionABI(
          dispatcher, callableType,
          RuntimeBundleLowerer::callableLogicalInputTypes(dispatcher,
                                                          callableType),
          memberOwnsALane)))
    return mlir::failure();
  for (mlir::Operation *inner : body) {
    if (mlir::failed(ensureOperationOperandBundles(inner)) ||
        mlir::failed(lowerPyOp(inner)))
      return mlir::failure();
  }
  return dispatcher;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerIndirectFunctionObjectCall(
    py::CallOp op, const RuntimeBundle &callableRef) {
  // ⛔ A COPY, because the dispatch below WRITES `valueBundles` -- once per
  // arm, plus the result binding -- and the argument is a reference into that
  // map. A rehash left it pointing at freed storage, and the read that
  // followed reported a function object with zero handles for a bundle that
  // had one; it needed three callers and two function-valued globals to grow
  // the map far enough, which is why every smaller spelling of the same
  // program was fine.
  RuntimeBundle callable = callableRef;
  if (op.getNumResults() != 1)
    return op.emitError()
           << "Python callable lowering expects exactly one Python result";

  llvm::SmallVector<mlir::func::FuncOp, 8> targets =
      RuntimeBundleLowerer::collectIndirectCallableTargets(op, callable);

  // ⭐ ONE CANDIDATE IS NOT ONE ANSWER unless the callable's target is known
  // STATICALLY. The candidate walk keeps only functions whose closure arity
  // matches the evidence in hand, and a runtime function value carries no
  // closure evidence -- so every wrapper is filtered out and a plain function
  // can be left standing alone. Calling it directly is then a guess:
  //
  //     f = double(base)      # each returns a wrapper that calls its fn
  //     f = add_one(f)
  //     print(f(5))           # 6 -- base(5) + 1, where CPython prints 11
  //
  // The wrapper's `fn(n)` devirtualized to `call @base`, the only zero-closure
  // candidate. With the target unknown the dispatch below is what must run: it
  // compares the object's target id and raises on a miss, so the same program
  // now says "callable target is not available" instead of answering.
  //
  // ⛔ The fast path stays for a callable whose target the evidence names --
  // that is the ordinary static call, and it is not a guess.
  if (targets.size() == 1 && !callable.functionTarget.empty()) {
    mlir::func::FuncOp target = targets.front();
    llvm::StringRef targetName = target.getSymName();
    RuntimeBundle selectedCallable = callable;
    selectedCallable.functionTarget = targetName.str();
    if (!callable.callableAlternatives.empty()) {
      const RuntimeCallableAlternative *alternative =
          findCallableAlternative(callable, targetName);
      if (!alternative)
        return op.emitError() << "indirect callable has no closure evidence "
                                 "alternative for "
                              << targetName;
      selectedCallable.closureValues = alternative->closureValues;
    }

    builder.setInsertionPoint(op);
    llvm::SmallVector<const RuntimeBundle *, 8> sources;
    llvm::SmallVector<RuntimeBundle, 8> materializedDefaults;
    llvm::SmallVector<RuntimeBundle, 4> closureSources;
    llvm::SmallVector<RuntimeBundle, 8> argumentEvidenceSources;
    llvm::SmallVector<RuntimeBundle, 8> aggregateEvidenceSources;
    if (mlir::failed(RuntimeBundleLowerer::collectFunctionTargetRuntimeSources(
            op, target, targetName, selectedCallable, sources,
            materializedDefaults, closureSources, argumentEvidenceSources,
            aggregateEvidenceSources)))
      return mlir::failure();
    RuntimeBundle result;
    bool usedPrimitiveClone = false;
    if (std::optional<std::string> cloneName =
                   RuntimeBundleLowerer::primitiveI64CloneFor(targetName)) {
      if (RuntimeBundleLowerer::allSourcesHavePrimitiveI64Evidence(sources)) {
        if (mlir::func::FuncOp clone =
                module.lookupSymbol<mlir::func::FuncOp>(*cloneName)) {
          if (mlir::failed(
                  RuntimeBundleLowerer::emitPrimitiveI64CloneFallbackResult(
                      op, target, targetName, clone, sources, result)))
            return mlir::failure();
          usedPrimitiveClone = true;
        }
      }
    }
    if (!usedPrimitiveClone) {
      mlir::FailureOr<mlir::func::CallOp> call =
          RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
              op, target, targetName, sources);
      if (mlir::failed(call))
        return mlir::failure();

      if (mlir::failed(RuntimeBundleLowerer::bundleFunctionTargetCallResult(
              op, target, targetName, *call, sources, result)))
        return mlir::failure();
    }

    valueBundles[op.getResult(0)] = std::move(result);
    // See lowerFunctionTargetCall: the callee may mutate borrowed container
    // arguments in place.
    demoteMutableContainerArgumentEvidence(op);
    erase.push_back(op);
    return mlir::success();
  }

  mlir::FailureOr<mlir::func::FuncOp> dispatcher =
      RuntimeBundleLowerer::indirectCallDispatcher(op, callable, targets);
  if (mlir::failed(dispatcher))
    return mlir::failure();
  if (*dispatcher) {
    builder.setInsertionPoint(op);
    llvm::StringRef dispatcherName = dispatcher->getSymName();
    llvm::SmallVector<const RuntimeBundle *, 8> sources{&callable};
    for (mlir::Value argument :
         op.getPosargs().getDefiningOp<py::PackOp>().getValues()) {
      auto bundle = valueBundles.find(argument);
      if (bundle == valueBundles.end())
        return op.emitError() << "indirect call argument has no runtime value";
      sources.push_back(&bundle->second);
    }
    mlir::FailureOr<mlir::func::CallOp> call =
        RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
            op, *dispatcher, dispatcherName, sources);
    if (mlir::failed(call))
      return mlir::failure();
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::bundleFunctionTargetCallResult(
            op, *dispatcher, dispatcherName, *call, sources, result)))
      return mlir::failure();
    valueBundles[op.getResult(0)] = std::move(result);
    demoteMutableContainerArgumentEvidence(op);
    erase.push_back(op);
    return mlir::success();
  }

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, op.getResult(0).getType(), "indirect callable result ABI");
  if (mlir::failed(resultTypes))
    return mlir::failure();
  llvm::SmallVector<mlir::Type, 8> continuationTypes(resultTypes->begin(),
                                                     resultTypes->end());
  RuntimeBundleLowerer::appendPrimitiveI64EvidenceTypes(
      op.getResult(0).getType(), continuationTypes);

  builder.setInsertionPoint(op);
  mlir::MemRefType storageType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI64Type());
  mlir::FailureOr<mlir::Value> storage =
      RuntimeBundleLowerer::erasedObjectStorageView(op, callable.objectValue,
                                                    storageType);
  if (mlir::failed(storage))
    return mlir::failure();
  mlir::Value targetSlot =
      mlir::arith::ConstantIndexOp::create(builder, op.getLoc(), 2).getResult();
  mlir::Value targetId =
      mlir::memref::LoadOp::create(builder, op.getLoc(), *storage, targetSlot)
          .getResult();

  mlir::Operation *operation = op.getOperation();
  mlir::Block *entry = operation->getBlock();
  mlir::Region *region = entry->getParent();
  mlir::Block *continuation = entry->splitBlock(operation->getIterator());

  llvm::SmallVector<mlir::Value, 4> continuationArgs;
  continuationArgs.reserve(continuationTypes.size());
  for (mlir::Type type : continuationTypes) {
    mlir::BlockArgument arg = continuation->addArgument(type, op.getLoc());
    continuationArgs.push_back(arg);
  }

  llvm::ArrayRef<mlir::Value> objectContinuationArgs(continuationArgs.data(),
                                                     resultTypes->size());
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, op.getResult(0).getType(), objectContinuationArgs, result)))
    return mlir::failure();
  if (RuntimeBundleLowerer::hasPrimitiveI64ABI(op.getResult(0).getType())) {
    unsigned offset = static_cast<unsigned>(resultTypes->size());
    if (offset + 2 > continuationArgs.size())
      return op.emitError()
             << "indirect callable int result continuation is missing "
                "primitive evidence";
    result.primitiveI64 = RuntimePrimitiveI64Evidence{
        continuationArgs[offset], continuationArgs[offset + 1]};
  }
  valueBundles[op.getResult(0)] = std::move(result);

  llvm::SmallVector<mlir::Block *, 8> targetBlocks;
  targetBlocks.reserve(targets.size());
  for (mlir::func::FuncOp ignored : targets) {
    (void)ignored;
    targetBlocks.push_back(
        builder.createBlock(region, continuation->getIterator()));
  }
  llvm::SmallVector<mlir::Block *, 8> testBlocks;
  if (targets.size() > 1) {
    testBlocks.reserve(targets.size() - 1);
    for (std::size_t index = 1, end = targets.size(); index < end; ++index)
      testBlocks.push_back(
          builder.createBlock(region, continuation->getIterator()));
  }
  mlir::Block *defaultBlock =
      builder.createBlock(region, continuation->getIterator());

  // ⭐ THE DISPATCH'S BLOCKS ARE STILL INSIDE THE TRY. `createRuntimeCall`
  // emits the call-site marker when `currentTryHandlerId()` answers, and that
  // answer is keyed on the INSERTION BLOCK -- so every call this dispatch puts
  // in a block it just created was emitted unguarded, and the exception walked
  // straight out of the `try` that was written around it:
  //
  //     def go(op: Callable[[int], int], v: int) -> int:
  //         try:
  //             return op(v)
  //         except ValueError:
  //             return -1
  //     print(go(f.run, 3), go(f.run, -1))   # ValueError escapes
  //
  // Silent, and it takes TWO call sites: with one candidate target the fast
  // path emits the call in the ORIGINAL block, which still carries the id, so
  // every smaller spelling of the same program catches. A free function is the
  // same shape and was fine for the same reason -- the two sites share one
  // target.
  //
  // ⛔ The CONTINUATION inherits it too, and not only the arms: the rest of the
  // try body lives there, so a second guarded call after this one would lose
  // its marker the same way.
  {
    std::optional<std::int64_t> guardingHandlerId;
    for (mlir::Block *block = entry; block;) {
      auto found = tryHandlerIds.find(block);
      if (found != tryHandlerIds.end()) {
        guardingHandlerId = found->second;
        break;
      }
      mlir::Operation *parent = block->getParentOp();
      block = parent ? parent->getBlock() : nullptr;
    }
    if (guardingHandlerId) {
      tryHandlerIds.try_emplace(continuation, *guardingHandlerId);
      for (mlir::Block *block : targetBlocks)
        tryHandlerIds.try_emplace(block, *guardingHandlerId);
      for (mlir::Block *block : testBlocks)
        tryHandlerIds.try_emplace(block, *guardingHandlerId);
      tryHandlerIds.try_emplace(defaultBlock, *guardingHandlerId);
    }
  }

  for (auto [index, target] : llvm::enumerate(targets)) {
    llvm::StringRef targetName = target.getSymName();
    builder.setInsertionPointToStart(targetBlocks[index]);

    RuntimeBundle selectedCallable = callable;
    selectedCallable.functionTarget = targetName.str();
    if (!callable.callableAlternatives.empty()) {
      const RuntimeCallableAlternative *alternative =
          findCallableAlternative(callable, targetName);
      if (!alternative)
        return op.emitError() << "indirect callable has no closure evidence "
                                 "alternative for "
                              << targetName;
      selectedCallable.closureValues = alternative->closureValues;
    }
    if (selectedCallable.closureValues.empty()) {
      builder.setInsertionPointToEnd(targetBlocks[index]);
      mlir::FailureOr<llvm::SmallVector<RuntimeValue, 4>> fromObject =
          RuntimeBundleLowerer::closureValuesFromFunctionObject(op, callable,
                                                               target);
      if (mlir::failed(fromObject))
        return mlir::failure();
      selectedCallable.closureValues.assign(fromObject->begin(),
                                            fromObject->end());
    }
    llvm::SmallVector<const RuntimeBundle *, 8> sources;
    llvm::SmallVector<RuntimeBundle, 8> materializedDefaults;
    llvm::SmallVector<RuntimeBundle, 4> closureSources;
    llvm::SmallVector<RuntimeBundle, 8> argumentEvidenceSources;
    llvm::SmallVector<RuntimeBundle, 8> aggregateEvidenceSources;
    if (mlir::failed(RuntimeBundleLowerer::collectFunctionTargetRuntimeSources(
            op, target, targetName, selectedCallable, sources,
            materializedDefaults, closureSources, argumentEvidenceSources,
            aggregateEvidenceSources)))
      return mlir::failure();
    RuntimeBundle targetResult;
    bool usedPrimitiveClone = false;
    if (std::optional<std::string> cloneName =
                   RuntimeBundleLowerer::primitiveI64CloneFor(targetName)) {
      if (RuntimeBundleLowerer::allSourcesHavePrimitiveI64Evidence(sources)) {
        if (mlir::func::FuncOp clone =
                module.lookupSymbol<mlir::func::FuncOp>(*cloneName)) {
          if (mlir::failed(
                  RuntimeBundleLowerer::emitPrimitiveI64CloneFallbackResult(
                      op, target, targetName, clone, sources, targetResult)))
            return mlir::failure();
          usedPrimitiveClone = true;
        }
      }
    }
    if (!usedPrimitiveClone) {
      mlir::FailureOr<mlir::func::CallOp> call =
          RuntimeBundleLowerer::emitFunctionTargetRuntimeCall(
              op, target, targetName, sources);
      if (mlir::failed(call))
        return mlir::failure();

      if (mlir::failed(RuntimeBundleLowerer::bundleFunctionTargetCallResult(
              op, target, targetName, *call, sources, targetResult)))
        return mlir::failure();
    }
    llvm::SmallVector<mlir::Value, 4> branchOperands;
    if (mlir::failed(RuntimeBundleLowerer::appendBundlePhysicalOperands(
            op, targetResult, *resultTypes, branchOperands)))
      return mlir::failure();
    if (RuntimeBundleLowerer::hasPrimitiveI64ABI(op.getResult(0).getType())) {
      if (targetResult.primitiveI64) {
        branchOperands.push_back(targetResult.primitiveI64->value);
        branchOperands.push_back(targetResult.primitiveI64->valid);
      } else {
        branchOperands.push_back(
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
                .getResult());
        branchOperands.push_back(
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 1)
                .getResult());
      }
    }
    mlir::cf::BranchOp::create(builder, op.getLoc(), continuation,
                               branchOperands);
  }

  builder.setInsertionPointToStart(defaultBlock);
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.TypeError", "callable target is not available")))
    return mlir::failure();
  // ⛔ The immortal placeholder and not fresh storage: this edge follows a
  // raise and never reaches the merge, but the merge still balances it, and a
  // zeroed allocation has no header a retain can be written against -- a
  // union result was refused ("the header prefix cannot be spelled") for
  // `f(1)` on a captured `Callable[[int], Box | None]`.
  mlir::FailureOr<RuntimeValue> dead = materializeMergeableDeadObjectValue(
      op, op.getResult(0).getType(), "indirect callable dispatch miss");
  if (mlir::failed(dead))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 8> deadValues(dead->values.begin(),
                                               dead->values.end());
  if (RuntimeBundleLowerer::hasPrimitiveI64ABI(op.getResult(0).getType())) {
    deadValues.push_back(
        mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64)
            .getResult());
    deadValues.push_back(
        mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 1)
            .getResult());
  }
  mlir::cf::BranchOp::create(builder, op.getLoc(), continuation, deadValues);

  auto enclosingFunction = op->getParentOfType<mlir::func::FuncOp>();
  if (targets.empty()) {
    builder.setInsertionPointToEnd(entry);
    mlir::cf::BranchOp::create(builder, op.getLoc(), defaultBlock);
  } else if (RuntimeBundleLowerer::isIndirectCallDispatcher(enclosingFunction)) {
    // A dispatcher's arms are many by construction: one switch on the id,
    // which the target ids -- dense, numbered as functions are first named --
    // let LLVM turn into a table.
    // ⛔ Not the chain below: a call would compare ids one candidate at a time.
    builder.setInsertionPointToEnd(entry);
    llvm::SmallVector<llvm::APInt, 8> caseValues;
    llvm::SmallVector<mlir::ValueRange, 8> caseOperands(targets.size(),
                                                         mlir::ValueRange{});
    for (mlir::func::FuncOp target : targets)
      caseValues.push_back(llvm::APInt(
          64, static_cast<std::uint64_t>(RuntimeBundleLowerer::functionTargetId(
                  target.getSymName())),
          /*isSigned=*/true));
    mlir::cf::SwitchOp::create(builder, op.getLoc(), targetId, defaultBlock,
                               mlir::ValueRange{}, caseValues, targetBlocks,
                               caseOperands);
    for (mlir::Block *unused : testBlocks)
      unused->erase();
  } else {
    mlir::Block *testBlock = entry;
    for (auto [index, target] : llvm::enumerate(targets)) {
      builder.setInsertionPointToEnd(testBlock);
      mlir::Value expectedId =
          mlir::arith::ConstantIntOp::create(
              builder, op.getLoc(),
              RuntimeBundleLowerer::functionTargetId(target.getSymName()), 64)
              .getResult();
      mlir::Value matches = mlir::arith::CmpIOp::create(
          builder, op.getLoc(), mlir::arith::CmpIPredicate::eq, targetId,
          expectedId);
      mlir::Block *nextBlock =
          index + 1 < targets.size() ? testBlocks[index] : defaultBlock;
      mlir::cf::CondBranchOp::create(builder, op.getLoc(), matches,
                                     targetBlocks[index], mlir::ValueRange{},
                                     nextBlock, mlir::ValueRange{});
      if (index + 1 < targets.size())
        testBlock = nextBlock;
    }
  }

  demoteMutableContainerArgumentEvidence(op);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
