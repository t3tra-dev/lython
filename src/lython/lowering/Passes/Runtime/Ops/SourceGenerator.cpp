#include "Runtime/Core/Lowerer.h"

#include "Runtime/ABI/CollectionPayload.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"

// Source generators resumed by inline re-dispatch: each __next__/send/throw
// re-emits the generator body as an scf.if resume-index chain instead of
// calling a state-machine clone (that path lives in GeneratorStateMachine.cpp
// and is chosen via generatorResumeClones in lowerSourceGeneratorNext).

namespace py::lowering {

mlir::FailureOr<RuntimePrimitiveI64Evidence>
RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
    mlir::Operation *op, mlir::Value value,
    llvm::ArrayRef<const RuntimeBundle *> frameSources,
    llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> &memo,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence) {
  auto cached = memo.find(value);
  if (cached != memo.end())
    return cached->second;

  if (runtimeContractName(value.getType()) != "builtins.int")
    return op->emitError()
           << "source generator next lowering currently supports only "
              "builtins.int yielded expressions";

  mlir::Operation *def = value.getDefiningOp();
  if (!def) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (!argument)
      return op->emitError()
             << "source generator next lowering expected a defining op or "
                "frame argument for yielded int value";
    mlir::Operation *parent = argument.getOwner()->getParentOp();
    auto function = mlir::dyn_cast_or_null<mlir::func::FuncOp>(parent);
    if (!function || function.isDeclaration() ||
        argument.getOwner() != &function.getBody().front())
      return op->emitError()
             << "source generator next lowering does not support non-entry "
                "block argument frame values yet";
    unsigned index = argument.getArgNumber();
    if (index >= frameSources.size())
      return op->emitError() << "source generator frame argument " << index
                             << " has no captured source bundle";
    const RuntimeBundle *source = frameSources[index];
    if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(source))
      return op->emitError()
             << "source generator frame argument " << index
             << " currently requires primitive i64 int evidence";
    RuntimePrimitiveI64Evidence evidence = *source->primitiveI64;
    memo.insert({value, evidence});
    return evidence;
  }

  if (auto constant = mlir::dyn_cast<py::IntConstantOp>(def)) {
    std::int64_t parsed = 0;
    if (constant.getValue().getAsInteger(10, parsed))
      return constant.emitError()
             << "integer literal is outside the currently lowered i64 path";
    RuntimePrimitiveI64Evidence evidence{
        mlir::arith::ConstantIntOp::create(builder, constant.getLoc(), parsed,
                                           64)
            .getResult(),
        mlir::arith::ConstantIntOp::create(builder, constant.getLoc(), 1, 1)
            .getResult()};
    memo.insert({value, evidence});
    return evidence;
  }

  if (auto yield = mlir::dyn_cast<py::YieldValueOp>(def)) {
    if (yield.getSent() != value)
      return yield.emitError()
             << "source generator next lowering cannot materialize "
                "non-sent yield result";
    if (sentI64Evidence) {
      memo.insert({value, *sentI64Evidence});
      return *sentI64Evidence;
    }
    RuntimePrimitiveI64Evidence invalidEvidence{
        mlir::arith::ConstantIntOp::create(builder, yield.getLoc(), 0, 64)
            .getResult(),
        mlir::arith::ConstantIntOp::create(builder, yield.getLoc(), 0, 1)
            .getResult()};
    memo.insert({value, invalidEvidence});
    return invalidEvidence;
  }

  auto materializeBinary = [&](mlir::Value lhsValue, mlir::Value rhsValue,
                               mlir::FlatSymbolRefAttr targetAttr,
                               llvm::StringRef expectedName)
      -> mlir::FailureOr<RuntimePrimitiveI64Evidence> {
    mlir::FailureOr<llvm::StringRef> methodName =
        RuntimeBundleLowerer::requireMethodTarget(def, targetAttr,
                                                  expectedName);
    if (mlir::failed(methodName))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> lhs =
        RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
            op, lhsValue, frameSources, memo, sentI64Evidence);
    if (mlir::failed(lhs))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> rhs =
        RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
            op, rhsValue, frameSources, memo, sentI64Evidence);
    if (mlir::failed(rhs))
      return mlir::failure();
    mlir::FailureOr<RuntimePrimitiveI64Evidence> result =
        RuntimeBundleLowerer::emitPrimitiveI64ArithmeticEvidence(
            def, *methodName, *lhs, *rhs);
    if (mlir::failed(result))
      return mlir::failure();
    memo.insert({value, *result});
    return *result;
  };

  if (auto add = mlir::dyn_cast<py::AddOp>(def))
    return materializeBinary(add.getLhs(), add.getRhs(), add.getTargetAttr(),
                             add.getMethodName());
  if (auto sub = mlir::dyn_cast<py::SubOp>(def))
    return materializeBinary(sub.getLhs(), sub.getRhs(), sub.getTargetAttr(),
                             sub.getMethodName());
  if (auto mul = mlir::dyn_cast<py::MulOp>(def))
    return materializeBinary(mul.getLhs(), mul.getRhs(), mul.getTargetAttr(),
                             mul.getMethodName());

  return def->emitError()
         << "source generator next lowering cannot materialize yielded int "
            "value produced by "
         << def->getName();
}

mlir::FailureOr<RuntimeBundleLowerer::SourceGeneratorResumeResult>
RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
    mlir::Operation *op, mlir::Type elementType, const RuntimeBundle &iterator,
    bool useCurrentInsertionPoint,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence) {
  mlir::func::FuncOp target =
      module.lookupSymbol<mlir::func::FuncOp>(iterator.generatorTarget);
  if (!target)
    return op->emitError() << "source generator target '"
                           << iterator.generatorTarget << "' is not defined";
  llvm::SmallVector<const RuntimeBundle *, 8> frameSources;
  frameSources.reserve(iterator.generatorSourceBundles.size());
  for (const std::shared_ptr<RuntimeBundle> &source :
       iterator.generatorSourceBundles) {
    if (!source)
      return op->emitError()
             << "source generator frame source bundle is missing";
    frameSources.push_back(source.get());
  }

  mlir::Block *entryBlock = nullptr;
  if (!target.isDeclaration())
    entryBlock = &target.getBody().front();
  if (!entryBlock)
    return op->emitError()
           << "source generator target must have a visible body";

  llvm::SmallVector<py::YieldValueOp, 4> yields;
  llvm::SmallVector<py::YieldFromOp, 4> yieldFroms;
  target.walk([&](py::YieldValueOp yield) { yields.push_back(yield); });
  target.walk(
      [&](py::YieldFromOp yieldFrom) { yieldFroms.push_back(yieldFrom); });

  if (runtimeContractName(elementType) != "builtins.int")
    return op->emitError()
           << "source generator next lowering currently supports int yields";
  const RuntimeBundle *delegatedSource = nullptr;
  const RuntimeBundle *delegatedIndexedIterable = nullptr;
  const RuntimeBundle *delegatedManifestIterator = nullptr;
  llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> delegatedIndexedElements;
  std::optional<RuntimeSymbol> delegatedManifestNext;
  mlir::func::FuncOp delegatedInlineTarget;
  llvm::SmallVector<mlir::Value, 4> delegatedInlineFrameSourceValues;
  llvm::SmallVector<mlir::Operation *, 8> delegatedExpressionOps;
  struct SourceYieldPlan {
    mlir::Value value;
    bool usesDelegatedFrameSources = false;
    llvm::SmallVector<mlir::Value, 4> delegatedFrameSourceValues;
  };
  if (!yieldFroms.empty()) {
    if (!yields.empty())
      return op->emitError()
             << "source generator yield from lowering does not yet support "
                "mixing direct yield and delegated yield points";
    if (yieldFroms.size() != 1)
      return op->emitError()
             << "source generator yield from lowering currently supports "
                "exactly one delegated source";

    py::YieldFromOp yieldFrom = yieldFroms.front();
    if (yieldFrom->getBlock() != entryBlock)
      return yieldFrom.emitError()
             << "source generator yield from lowering currently supports "
                "only straight-line delegation";
    auto elementAssignable = [&](mlir::Type sourceType) -> mlir::LogicalResult {
      if (sourceType && !py::isAssignableTo(sourceType, elementType, op))
        return yieldFrom.emitError()
               << "delegated iterable yields " << sourceType << ", expected "
               << elementType;
      return mlir::success();
    };

    auto rememberDelegatedExpression = [&](mlir::Operation *exprOp) {
      if (exprOp && !llvm::is_contained(delegatedExpressionOps, exprOp))
        delegatedExpressionOps.push_back(exprOp);
    };

    auto directFrameSourceFor =
        [&](mlir::Value value,
            llvm::StringRef label) -> mlir::FailureOr<const RuntimeBundle *> {
      auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
      if (!argument || argument.getOwner() != entryBlock)
        return yieldFrom.emitError()
               << label
               << " currently supports only direct frame source values";
      unsigned index = argument.getArgNumber();
      if (index >= frameSources.size())
        return yieldFrom.emitError() << label << " frame source " << index
                                     << " has no captured source bundle";
      const RuntimeBundle *source = frameSources[index];
      if (!source)
        return yieldFrom.emitError() << label << " frame source is missing";
      return source;
    };

    auto rememberCallPacks = [&](py::PackOp posargs, py::PackOp kwnames,
                                 py::PackOp kwvalues) {
      rememberDelegatedExpression(posargs);
      rememberDelegatedExpression(kwnames);
      rememberDelegatedExpression(kwvalues);
      for (py::PackOp pack : {posargs, kwnames, kwvalues}) {
        if (!pack)
          continue;
        for (mlir::Value value : pack.getValues())
          rememberDelegatedExpression(value.getDefiningOp());
      }
    };

    mlir::Value sourceValue = yieldFrom.getSource();
    if (mlir::isa<mlir::BlockArgument>(sourceValue)) {
      mlir::FailureOr<const RuntimeBundle *> source =
          directFrameSourceFor(sourceValue, "source generator yield from");
      if (mlir::failed(source))
        return mlir::failure();
      delegatedSource = *source;
    } else if (auto call = sourceValue.getDefiningOp<py::CallOp>()) {
      auto posargs = call.getPosargs().getDefiningOp<py::PackOp>();
      auto kwnames = call.getKwnames().getDefiningOp<py::PackOp>();
      auto kwvalues = call.getKwvalues().getDefiningOp<py::PackOp>();
      if (auto methodAttr =
              call->getAttrOfType<mlir::StringAttr>("ly.bound_method")) {
        llvm::StringRef methodName = methodAttr.getValue();
        if (methodName == "keys" || methodName == "values") {
          if (!posargs || !kwnames || !kwvalues)
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires static argument packs";
          if (!posargs.getValues().empty() || !kwnames.getValues().empty() ||
              !kwvalues.getValues().empty())
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "supports only zero-argument view calls";
          mlir::FailureOr<const RuntimeBundle *> receiver =
              directFrameSourceFor(call.getCallable(),
                                   "source generator yield from dict view");
          if (mlir::failed(receiver))
            return mlir::failure();
          if ((*receiver)->contractName() != "builtins.dict")
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires a dict frame source";
          auto viewContract = mlir::dyn_cast_if_present<py::ContractType>(
              call.getResult(0).getType());
          llvm::StringRef expectedView = methodName == "keys"
                                             ? "builtins.dict_keys"
                                             : "builtins.dict_values";
          if (!viewContract || viewContract.getContractName() != expectedView ||
              viewContract.getArguments().size() < 2)
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires a typed dict view result";
          mlir::Type projectedElement =
              methodName == "keys" ? viewContract.getArguments().front()
                                   : viewContract.getArguments()[1];
          if (mlir::failed(elementAssignable(projectedElement)))
            return mlir::failure();
          delegatedIndexedIterable = *receiver;
          delegatedIndexedElements = methodName == "keys"
                                         ? (*receiver)->mappingKeyBundles
                                         : (*receiver)->mappingValueBundles;
          if (delegatedIndexedElements.empty())
            return yieldFrom.emitError()
                   << "source generator yield from dict view delegation "
                      "requires static dict element evidence";
          for (const std::shared_ptr<RuntimeBundle> &element :
               delegatedIndexedElements) {
            if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
              return yieldFrom.emitError()
                     << "source generator yield from dict view delegation "
                        "currently requires primitive int element evidence";
          }
          rememberCallPacks(posargs, kwnames, kwvalues);
          rememberDelegatedExpression(call);
        } else if (methodName == "items") {
          return yieldFrom.emitError()
                 << "source generator yield from dict item view delegation "
                    "currently requires tuple yield lowering";
        } else {
          return yieldFrom.emitError()
                 << "source generator yield from bound method delegation "
                    "currently supports only dict keys/values views";
        }
      } else {
        auto binding = call.getCallable().getDefiningOp<py::BindingRefOp>();
        if (!binding)
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "static source generator binding";
        mlir::func::FuncOp callTarget =
            module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
        if (!callTarget || !callTarget->hasAttr("ly.generator.body_result"))
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "source generator target";
        py::CallableType callableType = callableTypeOf(callTarget);
        if (!callableType || callableType.hasVararg() ||
            callableType.hasKwarg() || !callableType.getKwOnlyTypes().empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation requires a "
                    "static positional callable signature";
        std::optional<StaticCallableInvocation> invocation =
            RuntimeBundleLowerer::collectStaticCallableInvocation(call);
        if (!invocation)
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "requires static argument packs";
        std::optional<CallableArgumentPlan> argumentPlan =
            RuntimeBundleLowerer::collectCallableArgumentPlan(
                call, callableType, /*emitErrors=*/true);
        if (!argumentPlan)
          return mlir::failure();
        if (!argumentPlan->defaultedFixed.empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "requires explicit frame arguments";
        if (!argumentPlan->varargActuals.empty() ||
            !argumentPlan->kwargActuals.empty())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation currently "
                    "supports only fixed frame arguments";
        llvm::ArrayRef<mlir::Type> positionalTypes =
            callableType.getPositionalTypes();
        if (argumentPlan->fixedActuals.size() != positionalTypes.size())
          return yieldFrom.emitError() << "source generator yield from call "
                                          "delegation argument count "
                                          "does not match target frame";
        for (auto [index, inputType] : llvm::enumerate(positionalTypes)) {
          std::optional<unsigned> actualIndex =
              argumentPlan->fixedActuals[index];
          if (!actualIndex || *actualIndex >= invocation->actualValues.size())
            return yieldFrom.emitError()
                   << "source generator yield from call delegation argument "
                      "planner produced an invalid frame source";
          mlir::Value argValue = invocation->actualValues[*actualIndex];
          if (!py::isAssignableTo(argValue.getType(), inputType, op))
            return yieldFrom.emitError()
                   << "source generator yield from call argument "
                   << argValue.getType() << " is not assignable to frame input "
                   << inputType;
          delegatedInlineFrameSourceValues.push_back(argValue);
        }
        llvm::SmallVector<mlir::Type, 4> closureTypes =
            RuntimeBundleLowerer::callableClosureTypes(callTarget);
        if (closureTypes.size() != binding.getCaptures().size())
          return yieldFrom.emitError()
                 << "source generator yield from call delegation binding "
                    "captures "
                 << binding.getCaptures().size()
                 << " values, but delegated target declares "
                 << closureTypes.size() << " closure inputs";
        for (auto [index, capture] : llvm::enumerate(binding.getCaptures())) {
          mlir::Type expected = closureTypes[index];
          if (!py::isAssignableTo(capture.getType(), expected, op))
            return yieldFrom.emitError()
                   << "source generator yield from call delegation capture "
                   << index << " has type " << capture.getType()
                   << ", expected " << expected;
          delegatedInlineFrameSourceValues.push_back(capture);
        }
        if (auto generatorContract =
                mlir::dyn_cast_if_present<py::ContractType>(
                    call.getResult(0).getType())) {
          llvm::ArrayRef<mlir::Type> args = generatorContract.getArguments();
          if (!args.empty() && mlir::failed(elementAssignable(args.front())))
            return mlir::failure();
        }
        delegatedInlineTarget = callTarget;
        rememberDelegatedExpression(binding);
        rememberCallPacks(posargs, kwnames, kwvalues);
        rememberDelegatedExpression(call);
      }
    } else {
      return yieldFrom.emitError()
             << "source generator yield from currently supports direct frame "
                "source or source generator call delegation";
    }

    if (delegatedSource &&
        delegatedSource->contractName() == "types.GeneratorType" &&
        !delegatedSource->generatorTarget.empty()) {
      if (auto generatorContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedSource->contract)) {
        llvm::ArrayRef<mlir::Type> args = generatorContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
    } else if (delegatedSource &&
               (delegatedSource->contractName() == "builtins.list" ||
                delegatedSource->contractName() == "builtins.tuple") &&
               !delegatedSource->sequenceElementBundles.empty()) {
      delegatedIndexedIterable = delegatedSource;
      delegatedIndexedElements = delegatedSource->sequenceElementBundles;
      delegatedSource = nullptr;
      if (auto sequenceContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedIndexedIterable->contract)) {
        llvm::ArrayRef<mlir::Type> args = sequenceContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
      for (const std::shared_ptr<RuntimeBundle> &element :
           delegatedIndexedElements) {
        if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
          return yieldFrom.emitError()
                 << "source generator yield from static sequence currently "
                    "requires primitive int element evidence";
      }
    } else if (delegatedSource &&
               delegatedSource->contractName() == "builtins.dict" &&
               !delegatedSource->mappingKeyBundles.empty()) {
      delegatedIndexedIterable = delegatedSource;
      delegatedIndexedElements = delegatedSource->mappingKeyBundles;
      delegatedSource = nullptr;
      if (auto dictContract = mlir::dyn_cast_if_present<py::ContractType>(
              delegatedIndexedIterable->contract)) {
        llvm::ArrayRef<mlir::Type> args = dictContract.getArguments();
        if (!args.empty() && mlir::failed(elementAssignable(args.front())))
          return mlir::failure();
      }
      for (const std::shared_ptr<RuntimeBundle> &key :
           delegatedIndexedElements) {
        if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(key.get()))
          return yieldFrom.emitError()
                 << "source generator yield from static dict currently "
                    "requires primitive int key evidence";
      }
    } else if (delegatedSource) {
      llvm::SmallVector<const RuntimeBundle *, 1> nextSources{delegatedSource};
      mlir::FailureOr<RuntimeSymbol> next =
          RuntimeBundleLowerer::selectManifestMethod(
              yieldFrom, *delegatedSource, "__next__", nextSources,
              /*allowUnusedSources=*/false);
      if (mlir::failed(next))
        return mlir::failure();
      if (!next->validResultIndex)
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ valid_result_index evidence";
      if (next->nextEvidence != "receiver")
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ ly.runtime.next_evidence = \"receiver\"";
      if (next->elementContract.empty())
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator requires "
                  "__next__ element_contract evidence";
      if (mlir::failed(elementAssignable(
              runtimeContractType(context, next->elementContract))))
        return mlir::failure();
      if (next->elementContract != "builtins.int")
        return yieldFrom.emitError()
               << "source generator yield from manifest iterator currently "
                  "requires builtins.int element evidence";
      delegatedManifestIterator = delegatedSource;
      delegatedManifestNext = *next;
      delegatedSource = nullptr;
    } else if (!delegatedIndexedIterable && !delegatedInlineTarget &&
               !delegatedManifestIterator) {
      return yieldFrom.emitError()
             << "source generator yield from currently supports only source "
                "generator, static sequence, static dict key, or self-mutating "
                "manifest iterator values";
    }
    if (delegatedSource) {
      auto generatorContract = mlir::dyn_cast_if_present<py::ContractType>(
          delegatedSource->contract);
      llvm::ArrayRef<mlir::Type> args = generatorContract
                                            ? generatorContract.getArguments()
                                            : llvm::ArrayRef<mlir::Type>();
      if (!args.empty() && !py::isAssignableTo(args.front(), elementType, op))
        return yieldFrom.emitError()
               << "delegated generator yields " << args.front() << ", expected "
               << elementType;
    }
  } else if (yields.empty()) {
    return op->emitError()
           << "source generator next lowering currently requires at least one "
              "yield";
  }

  llvm::SmallPtrSet<mlir::Operation *, 8> allowedDelegationOps;
  for (mlir::Operation *exprOp : delegatedExpressionOps)
    allowedDelegationOps.insert(exprOp);
  for (mlir::Operation &bodyOp : *entryBlock) {
    if (allowedDelegationOps.contains(&bodyOp))
      continue;
    if (mlir::isa<py::IntConstantOp, py::YieldValueOp, py::YieldFromOp,
                  py::NoneOp, py::AddOp, py::SubOp, py::MulOp,
                  mlir::func::ReturnOp>(bodyOp))
      continue;
    return bodyOp.emitError()
           << "source generator next lowering currently supports only "
              "straight-line pure int yield bodies";
  }

  auto collectStraightLineIntYieldValues = [&](mlir::func::FuncOp yieldTarget,
                                               llvm::StringRef label)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    mlir::Block *yieldEntry = nullptr;
    if (!yieldTarget.isDeclaration())
      yieldEntry = &yieldTarget.getBody().front();
    if (!yieldEntry)
      return op->emitError() << label << " target must have a visible body";

    llvm::SmallVector<py::YieldValueOp, 4> sourceYields;
    yieldTarget.walk(
        [&](py::YieldValueOp yield) { sourceYields.push_back(yield); });
    if (sourceYields.empty())
      return op->emitError()
             << label << " currently requires at least one yield";

    for (mlir::Operation &bodyOp : *yieldEntry) {
      if (mlir::isa<py::IntConstantOp, py::YieldValueOp, py::NoneOp, py::AddOp,
                    py::SubOp, py::MulOp, mlir::func::ReturnOp>(bodyOp))
        continue;
      return bodyOp.emitError()
             << label
             << " currently supports only straight-line pure int yield bodies";
    }

    llvm::SmallVector<mlir::Value, 4> values;
    values.reserve(sourceYields.size());
    for (py::YieldValueOp yield : sourceYields) {
      if (yield->getBlock() != yieldEntry)
        return yield.emitError()
               << label
               << " currently supports only straight-line yield points";
      if (!yield.getSent().use_empty())
        return yield.emitError()
               << label << " does not support sent values yet";
      if (runtimeContractName(yield.getValue().getType()) != "builtins.int")
        return yield.emitError() << label << " currently supports int yields";
      values.push_back(yield.getValue());
    }
    return values;
  };

  llvm::SmallVector<SourceYieldPlan, 4> yieldPlans;
  if (delegatedInlineTarget) {
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> inlineYields =
        collectStraightLineIntYieldValues(
            delegatedInlineTarget,
            "source generator yield from call delegation");
    if (mlir::failed(inlineYields))
      return mlir::failure();
    yieldPlans.reserve(inlineYields->size());
    for (mlir::Value yieldValue : *inlineYields) {
      SourceYieldPlan plan;
      plan.value = yieldValue;
      plan.usesDelegatedFrameSources = true;
      plan.delegatedFrameSourceValues = delegatedInlineFrameSourceValues;
      yieldPlans.push_back(std::move(plan));
    }
  } else {
    yieldPlans.reserve(yields.size());
    for (py::YieldValueOp yield : yields) {
      if (yield->getBlock() != entryBlock)
        return yield.emitError()
               << "source generator next lowering currently supports only "
                  "straight-line yield points";
      if (!yield.getSent().use_empty() &&
          runtimeContractName(yield.getSent().getType()) != "builtins.int")
        return yield.emitError()
               << "source generator next lowering currently supports only int "
                  "sent values";
      if (runtimeContractName(yield.getValue().getType()) != "builtins.int")
        return yield.emitError() << "source generator next lowering currently "
                                    "supports int yields";
      SourceYieldPlan plan;
      plan.value = yield.getValue();
      yieldPlans.push_back(std::move(plan));
    }
  }

  std::optional<RuntimeSymbol> resumeBegin =
      manifest.primitive("types.GeneratorType", "resume.begin");
  std::optional<RuntimeSymbol> resumeSuspend =
      manifest.primitive("types.GeneratorType", "resume.suspend");
  std::optional<RuntimeSymbol> resumeComplete =
      manifest.primitive("types.GeneratorType", "resume.complete");
  if (!resumeBegin || !resumeSuspend || !resumeComplete)
    return op->emitError()
           << "runtime manifest has no generator resume primitive for "
              "types.GeneratorType";

  llvm::SmallVector<const RuntimeBundle *, 1> generatorSource{&iterator};
  if (!useCurrentInsertionPoint)
    builder.setInsertionPoint(op);
  llvm::SmallVector<mlir::Value, 4> resumeBeginOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *resumeBegin, generatorSource,
                                            resumeBeginOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp resumeBeginCall = RuntimeBundleLowerer::createRuntimeCall(
      op->getLoc(), *resumeBegin, resumeBeginOperands);
  if (resumeBeginCall.getNumResults() != 1 ||
      !resumeBeginCall.getResult(0).getType().isInteger(1))
    return resumeBegin->function.emitError()
           << "generator resume.begin primitive must return one i1";

  llvm::ArrayRef<mlir::Value> generatorValues = iterator.physicalValues();
  if (generatorValues.size() != 1)
    return op->emitError() << "types.GeneratorType bundle must contain one "
                              "storage value";
  mlir::Value storage = generatorValues.front();
  auto storageType = mlir::dyn_cast<mlir::MemRefType>(storage.getType());
  if (!storageType || storageType.getRank() != 1 ||
      storageType.getElementType() != builder.getI64Type())
    return op->emitError() << "types.GeneratorType storage has invalid type "
                           << storage.getType();

  mlir::Value resumeSlot =
      mlir::arith::ConstantIndexOp::create(builder, op->getLoc(), 4);
  mlir::Value resumeIndex =
      mlir::memref::LoadOp::create(builder, op->getLoc(), storage, resumeSlot);
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 1, 1);
  llvm::SmallVector<mlir::Type, 3> resultTypes{
      builder.getI64Type(), builder.getI1Type(), builder.getI1Type()};

  auto emitSuspendAndYieldEvidence =
      [&](mlir::Value yieldedValue, mlir::Value yieldedValid,
          unsigned nextResumeIndexValue) -> mlir::LogicalResult {
    mlir::Value nextResumeIndex = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(), static_cast<std::int64_t>(nextResumeIndexValue),
        64);
    llvm::SmallVector<mlir::Value, 4> suspendOperands;
    unsigned suspendInputIndex = 0;
    if (mlir::failed(appendRuntimeSource(
            op, *resumeSuspend, resumeSuspend->function.getFunctionType(),
            suspendInputIndex, iterator, suspendOperands)))
      return mlir::failure();
    if (suspendInputIndex != 1 ||
        resumeSuspend->function.getFunctionType().getNumInputs() != 2 ||
        !resumeSuspend->function.getFunctionType().getInput(1).isInteger(64))
      return resumeSuspend->function.emitError()
             << "generator resume.suspend primitive must take storage and one "
                "i64 resume index";
    suspendOperands.push_back(nextResumeIndex);
    RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeSuspend,
                                            suspendOperands);
    mlir::scf::YieldOp::create(
        builder, op->getLoc(),
        mlir::ValueRange{yieldedValue, yieldedValid, trueValue});
    return mlir::success();
  };

  auto emitInvalidYield = [&]() -> mlir::LogicalResult {
    mlir::Value zeroValue =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 0, 64);
    mlir::Value falseValue =
        mlir::arith::ConstantIntOp::create(builder, op->getLoc(), 0, 1);
    mlir::scf::YieldOp::create(
        builder, op->getLoc(),
        mlir::ValueRange{zeroValue, falseValue, falseValue});
    return mlir::success();
  };

  auto emitSuspendAndYield =
      [&](const SourceYieldPlan &yieldPlan,
          unsigned nextResumeIndexValue) -> mlir::LogicalResult {
    llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> memo;
    mlir::FailureOr<RuntimePrimitiveI64Evidence> yieldedEvidence =
        mlir::failure();
    if (yieldPlan.usesDelegatedFrameSources) {
      llvm::SmallVector<RuntimeBundle, 4> delegatedFrameStorage;
      delegatedFrameStorage.reserve(
          yieldPlan.delegatedFrameSourceValues.size());
      llvm::SmallVector<const RuntimeBundle *, 4> delegatedFrameSources;
      delegatedFrameSources.reserve(
          yieldPlan.delegatedFrameSourceValues.size());
      for (mlir::Value sourceValue : yieldPlan.delegatedFrameSourceValues) {
        if (runtimeContractName(sourceValue.getType()) != "builtins.int") {
          delegatedFrameSources.push_back(nullptr);
          continue;
        }
        mlir::FailureOr<RuntimePrimitiveI64Evidence> sourceEvidence =
            RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
                op, sourceValue, frameSources, memo, sentI64Evidence);
        if (mlir::failed(sourceEvidence))
          return mlir::failure();
        RuntimeBundle sourceBundle;
        if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
                op, sourceValue.getType(), sourceEvidence->value,
                sourceEvidence->valid, sourceBundle)))
          return mlir::failure();
        delegatedFrameStorage.push_back(std::move(sourceBundle));
        delegatedFrameSources.push_back(&delegatedFrameStorage.back());
      }
      llvm::DenseMap<mlir::Value, RuntimePrimitiveI64Evidence> delegatedMemo;
      yieldedEvidence =
          RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
              op, yieldPlan.value, delegatedFrameSources, delegatedMemo);
    } else {
      yieldedEvidence =
          RuntimeBundleLowerer::materializeSourceGeneratorI64Value(
              op, yieldPlan.value, frameSources, memo, sentI64Evidence);
    }
    if (mlir::failed(yieldedEvidence))
      return mlir::failure();
    return emitSuspendAndYieldEvidence(
        yieldedEvidence->value, yieldedEvidence->valid, nextResumeIndexValue);
  };

  auto emitCompleteAndInvalidYield = [&]() -> mlir::LogicalResult {
    auto completeIf = mlir::scf::IfOp::create(builder, op->getLoc(),
                                              resumeBeginCall.getResult(0),
                                              /*withElseRegion=*/false);
    mlir::Block &completeThen = completeIf.getThenRegion().front();
    builder.setInsertionPoint(completeThen.getTerminator());
    llvm::SmallVector<mlir::Value, 4> completeOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *resumeComplete,
                                              generatorSource, completeOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *resumeComplete,
                                            completeOperands);

    builder.setInsertionPointAfter(completeIf);
    return emitInvalidYield();
  };

  auto emitManifestIteratorNextI64Evidence =
      [&]() -> mlir::FailureOr<RuntimePrimitiveI64Evidence> {
    if (!delegatedManifestIterator || !delegatedManifestNext)
      return op->emitError()
             << "source generator yield from manifest iterator state is "
                "missing";

    const RuntimeSymbol &next = *delegatedManifestNext;
    llvm::SmallVector<const RuntimeBundle *, 1> nextSources{
        delegatedManifestIterator};
    llvm::SmallVector<mlir::Value, 8> operands;
    if (mlir::failed(buildRuntimeCallOperands(op, next, nextSources, operands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp call =
        RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), next, operands);
    unsigned validIndex = *next.validResultIndex;
    if (validIndex >= call.getNumResults())
      return op->emitError()
             << "runtime __next__ valid_result_index is outside the result "
                "list";
    mlir::Value valid = call.getResult(validIndex);
    if (!valid.getType().isInteger(1))
      return op->emitError() << "runtime __next__ valid result must be an i1";

    llvm::SmallVector<mlir::Value, 4> elementValues;
    for (unsigned index = 0; index < validIndex; ++index)
      elementValues.push_back(call.getResult(index));
    mlir::Type intType = runtimeContractType(context, "builtins.int");
    RuntimeBundle element = RuntimeBundle::object(intType, elementValues);

    std::optional<RuntimeSymbol> unbox =
        manifest.primitive("builtins.int", "unbox.i64");
    if (!unbox)
      return op->emitError()
             << "runtime manifest has no builtins.int.unbox.i64 primitive";
    mlir::func::CallOp unboxCall = RuntimeBundleLowerer::createRuntimeCall(
        op->getLoc(), *unbox, element.physicalValues());
    if (unboxCall.getNumResults() != 1 ||
        !unboxCall.getResult(0).getType().isInteger(64))
      return unbox->function.emitError()
             << "builtins.int.unbox.i64 primitive must return one i64";

    llvm::SmallVector<mlir::Value, 4> nextValues;
    for (unsigned index = validIndex + 1; index < call.getNumResults();
         ++index)
      nextValues.push_back(call.getResult(index));
    RuntimeBundle nextState = RuntimeBundle::object(
        runtimeContractType(context, next.nextContract), nextValues);
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, nextState, "source generator yield from next state")))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, element, "source generator yield from yielded element")))
      return mlir::failure();

    return RuntimePrimitiveI64Evidence{unboxCall.getResult(0), valid};
  };

  if (delegatedSource) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    auto delegatedResumeInfo =
        generatorResumeClones.find(delegatedSource->generatorTarget);
    mlir::FailureOr<SourceGeneratorResumeResult> delegatedOr =
        delegatedResumeInfo != generatorResumeClones.end()
            ? RuntimeBundleLowerer::emitStateMachineGeneratorResume(
                  op, *delegatedSource, delegatedResumeInfo->second,
                  /*useCurrentInsertionPoint=*/true)
            : RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
                  op, elementType, *delegatedSource,
                  /*useCurrentInsertionPoint=*/true);
    if (mlir::failed(delegatedOr))
      return mlir::failure();
    SourceGeneratorResumeResult delegated = *delegatedOr;
    auto hasValue = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                            delegated.hasValue,
                                            /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&hasValue.getThenRegion().front());
    if (mlir::failed(emitSuspendAndYieldEvidence(delegated.value,
                                                 delegated.valid,
                                                 /*nextResumeIndexValue=*/0)))
      return mlir::failure();

    builder.setInsertionPointToStart(&hasValue.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasValue);
    mlir::scf::YieldOp::create(builder, op->getLoc(), hasValue.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  if (delegatedIndexedIterable) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    mlir::FailureOr<mlir::Value> runtimeLength =
        collection_abi::loadCollectionLength(op, builder,
                                             *delegatedIndexedIterable,
                                             "source generator yield from");
    if (mlir::failed(runtimeLength))
      return mlir::failure();
    mlir::Location loc = op->getLoc();
    mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
    mlir::Value staticLength = mlir::arith::ConstantIntOp::create(
        builder, loc,
        static_cast<std::int64_t>(delegatedIndexedElements.size()), 64);
    mlir::Value nonNegative = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::sge, resumeIndex, zero);
    mlir::Value belowRuntimeLength = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::slt, resumeIndex,
        *runtimeLength);
    mlir::Value belowStaticLength = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::slt, resumeIndex,
        staticLength);
    mlir::Value inRuntimeRange = mlir::arith::AndIOp::create(
        builder, loc, nonNegative, belowRuntimeLength);
    mlir::Value canYield = mlir::arith::AndIOp::create(
        builder, loc, inRuntimeRange, belowStaticLength);
    auto hasElement =
        mlir::scf::IfOp::create(builder, loc, resultTypes, canYield,
                                /*withElseRegion=*/true);

    auto emitSequenceYieldDispatch =
        [&](auto &&self,
            unsigned elementIndex) -> mlir::FailureOr<mlir::scf::IfOp> {
      mlir::Value expected = mlir::arith::ConstantIntOp::create(
          builder, loc, static_cast<std::int64_t>(elementIndex), 64);
      mlir::Value indexMatches = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::eq, resumeIndex, expected);
      auto match =
          mlir::scf::IfOp::create(builder, loc, resultTypes, indexMatches,
                                  /*withElseRegion=*/true);

      builder.setInsertionPointToStart(&match.getThenRegion().front());
      const std::shared_ptr<RuntimeBundle> &element =
          delegatedIndexedElements[elementIndex];
      if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(element.get()))
        return match.emitError()
               << "source generator yield from static indexed iterable "
                  "currently requires primitive int element evidence";
      if (mlir::failed(emitSuspendAndYieldEvidence(element->primitiveI64->value,
                                                   element->primitiveI64->valid,
                                                   elementIndex + 1)))
        return mlir::failure();

      builder.setInsertionPointToStart(&match.getElseRegion().front());
      if (elementIndex + 1 < delegatedIndexedElements.size()) {
        mlir::FailureOr<mlir::scf::IfOp> nested = self(self, elementIndex + 1);
        if (mlir::failed(nested))
          return mlir::failure();
        builder.setInsertionPointAfter(*nested);
        mlir::scf::YieldOp::create(builder, loc, nested->getResults());
      } else if (mlir::failed(emitCompleteAndInvalidYield())) {
        return mlir::failure();
      }

      builder.setInsertionPointAfter(match);
      return match;
    };

    builder.setInsertionPointToStart(&hasElement.getThenRegion().front());
    mlir::FailureOr<mlir::scf::IfOp> selected =
        emitSequenceYieldDispatch(emitSequenceYieldDispatch, 0);
    if (mlir::failed(selected))
      return mlir::failure();
    builder.setInsertionPointAfter(*selected);
    mlir::scf::YieldOp::create(builder, loc, selected->getResults());

    builder.setInsertionPointToStart(&hasElement.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasElement);
    mlir::scf::YieldOp::create(builder, loc, hasElement.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  if (delegatedManifestIterator) {
    auto branch = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                          resumeBeginCall.getResult(0),
                                          /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    mlir::FailureOr<RuntimePrimitiveI64Evidence> next =
        emitManifestIteratorNextI64Evidence();
    if (mlir::failed(next))
      return mlir::failure();
    auto hasValue = mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes,
                                            next->valid,
                                            /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&hasValue.getThenRegion().front());
    if (mlir::failed(emitSuspendAndYieldEvidence(
            next->value, next->valid, /*nextResumeIndexValue=*/0)))
      return mlir::failure();

    builder.setInsertionPointToStart(&hasValue.getElseRegion().front());
    if (mlir::failed(emitCompleteAndInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(hasValue);
    mlir::scf::YieldOp::create(builder, op->getLoc(), hasValue.getResults());

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (mlir::failed(emitInvalidYield()))
      return mlir::failure();

    builder.setInsertionPointAfter(branch);
    return SourceGeneratorResumeResult{branch.getResult(0), branch.getResult(1),
                                       branch.getResult(2)};
  }

  auto emitYieldDispatch =
      [&](auto &&self,
          unsigned yieldIndex) -> mlir::FailureOr<mlir::scf::IfOp> {
    mlir::Value indexValue = mlir::arith::ConstantIntOp::create(
        builder, op->getLoc(), static_cast<std::int64_t>(yieldIndex), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, op->getLoc(), mlir::arith::CmpIPredicate::eq, resumeIndex,
        indexValue);
    mlir::Value canYield = mlir::arith::AndIOp::create(
        builder, op->getLoc(), resumeBeginCall.getResult(0), indexMatches);
    auto branch =
        mlir::scf::IfOp::create(builder, op->getLoc(), resultTypes, canYield,
                                /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&branch.getThenRegion().front());
    if (mlir::failed(
            emitSuspendAndYield(yieldPlans[yieldIndex], yieldIndex + 1)))
      return mlir::failure();

    builder.setInsertionPointToStart(&branch.getElseRegion().front());
    if (yieldIndex + 1 < yieldPlans.size()) {
      mlir::FailureOr<mlir::scf::IfOp> nested = self(self, yieldIndex + 1);
      if (mlir::failed(nested))
        return mlir::failure();
      builder.setInsertionPointAfter(*nested);
      mlir::scf::YieldOp::create(builder, op->getLoc(), nested->getResults());
    } else if (mlir::failed(emitCompleteAndInvalidYield())) {
      return mlir::failure();
    }

    builder.setInsertionPointAfter(branch);
    return branch;
  };

  mlir::FailureOr<mlir::scf::IfOp> yieldedOr =
      emitYieldDispatch(emitYieldDispatch, 0);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  mlir::scf::IfOp yielded = *yieldedOr;

  builder.setInsertionPointAfter(yielded);
  return SourceGeneratorResumeResult{yielded.getResult(0), yielded.getResult(1),
                                     yielded.getResult(2)};
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerSourceGeneratorNext(py::NextOp op,
                                               const RuntimeBundle &iterator) {
  auto resumeInfo = generatorResumeClones.find(iterator.generatorTarget);
  mlir::FailureOr<SourceGeneratorResumeResult> yieldedOr =
      resumeInfo != generatorResumeClones.end()
          ? RuntimeBundleLowerer::emitStateMachineGeneratorResume(
                op.getOperation(), iterator, resumeInfo->second)
          : RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
                op.getOperation(), op.getElement().getType(), iterator);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  SourceGeneratorResumeResult yielded = *yieldedOr;

  // The object-lane element ABI: the yielded value's physical span (plus the
  // trailing evidence pair for int). Legacy pure-pair resumes synthesize the
  // same shape from (value, valid) so one binding path serves both.
  llvm::SmallVector<mlir::Value, 6> elementValues = yielded.lanePhysicals;
  bool laneElement = !elementValues.empty();

  auto bindElementBundle =
      [&](mlir::ValueRange values,
          RuntimeBundle &bundle) -> mlir::LogicalResult {
    if (!laneElement)
      return RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getElement().getType(), values[0], values[1], bundle);
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op.getOperation(), op.getElement().getType(), values, bundle)))
      return mlir::failure();
    bundle.setObjectLogicalOwnership(/*ownsObject=*/true);
    return mlir::success();
  };
  if (!laneElement) {
    elementValues.push_back(yielded.value);
    elementValues.push_back(yielded.valid);
  }

  if (auto condition = mlir::dyn_cast<mlir::scf::ConditionOp>(
          op->getBlock()->getTerminator())) {
    if (llvm::is_contained(condition.getArgs(), op.getElement())) {
      auto whileOp = op->getParentOfType<mlir::scf::WhileOp>();
      if (!whileOp || condition->getParentOp() != whileOp)
        return op.emitError()
               << "source generator for-loop lowering expected enclosing "
                  "scf.while";
      if (!whileOp.getInits().empty())
        return op.emitError()
               << "source generator for-loop lowering does not support "
                  "loop-carried iterator values yet";
      if (whileOp.getResults().size() != 1 || condition.getArgs().size() != 1 ||
          condition.getArgs().front() != op.getElement() ||
          whileOp.getAfter().front().getNumArguments() != 1)
        return op.emitError()
               << "source generator for-loop lowering expected one yielded "
                  "loop value";
      if (!whileOp.getResult(0).use_empty())
        return op.emitError()
               << "source generator for-loop lowering does not support users "
                  "of the scf.while result yet";

      // The loop carries the element's physical span instead of the single
      // logical value. scf.while pins its result arity at construction, so
      // the widened loop is a fresh op that adopts the original regions; the
      // original results are unused (checked above), so nothing re-targets.
      llvm::SmallVector<mlir::Type, 6> carriedTypes;
      carriedTypes.reserve(elementValues.size());
      for (mlir::Value value : elementValues)
        carriedTypes.push_back(value.getType());

      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(whileOp);
      auto widened = mlir::scf::WhileOp::create(
          builder, whileOp.getLoc(), carriedTypes, mlir::ValueRange{});
      widened.getBefore().takeBody(whileOp.getBefore());
      widened.getAfter().takeBody(whileOp.getAfter());
      whileOp.erase();

      condition.getConditionMutable().set(yielded.hasValue);
      condition.getArgsMutable().assign(elementValues);

      mlir::Block &after = widened.getAfter().front();
      mlir::BlockArgument bodyElement = after.getArgument(0);
      bodyElement.setType(carriedTypes.front());
      llvm::SmallVector<mlir::Value, 6> bodyValues{bodyElement};
      for (unsigned index = 1; index < carriedTypes.size(); ++index)
        bodyValues.push_back(
            after.addArgument(carriedTypes[index], op.getLoc()));

      builder.setInsertionPointToStart(&after);
      RuntimeBundle bodyElementBundle;
      if (laneElement) {
        if (mlir::failed(bindElementBundle(bodyValues, bodyElementBundle)))
          return mlir::failure();
      } else {
        mlir::Value bodyValid =
            mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
        if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
                op, op.getElement().getType(), bodyValues[0], bodyValid,
                bodyElementBundle)))
          return mlir::failure();
      }
      valueBundles[bodyElement] = std::move(bodyElementBundle);
    }
  }
  if (mlir::Block *block = op->getBlock()) {
    for (mlir::Operation &candidate : *block) {
      for (mlir::OpOperand &operand : candidate.getOpOperands()) {
        if (operand.get() == op.getValid())
          operand.set(yielded.hasValue);
      }
    }
  }
  llvm::SmallVector<mlir::OpOperand *, 4> validUses;
  for (mlir::OpOperand &use : op.getValid().getUses())
    validUses.push_back(&use);
  for (mlir::OpOperand *use : validUses)
    use->set(yielded.hasValue);

  builder.setInsertionPoint(op);
  RuntimeBundle element;
  if (laneElement) {
    if (mlir::failed(bindElementBundle(elementValues, element)))
      return mlir::failure();
  } else if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
                 op, op.getElement().getType(), yielded.value, yielded.valid,
                 element))) {
    return mlir::failure();
  }
  valueBundles[op.getElement()] = std::move(element);

  RuntimeBundle next = iterator;
  next.setObjectLogicalOwnership(/*ownsObject=*/false);
  valueBundles[op.getNext()] = std::move(next);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorSend(
    py::CallOp op, const RuntimeBundle &receiver,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError() << "source generator send expects exactly one value";

  std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence;
  bool releaseIgnoredSentObject = false;
  if (runtimeContractName(sources[1]->contract) == "builtins.int") {
    if (!RuntimeBundleLowerer::hasPrimitiveI64Evidence(sources[1]))
      return op.emitError()
             << "source generator send(int) requires primitive int evidence";
    sentI64Evidence = *sources[1]->primitiveI64;
  } else if (runtimeContractName(sources[1]->contract) != "types.NoneType") {
    releaseIgnoredSentObject = true;
  }

  if (releaseIgnoredSentObject &&
      mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
          op, *sources[1], "source generator send ignored value")))
    return mlir::failure();

  return RuntimeBundleLowerer::lowerSourceGeneratorAdvance(op, receiver,
                                                           sentI64Evidence);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorDunderNext(
    py::CallOp op, const RuntimeBundle &receiver,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 1)
    return op.emitError() << "generator __next__ expects no arguments";
  return RuntimeBundleLowerer::lowerSourceGeneratorAdvance(op, receiver,
                                                           std::nullopt);
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorAdvance(
    py::CallOp op, const RuntimeBundle &receiver,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence) {
  if (op.getNumResults() != 1)
    return op.emitError() << "source generator resume expects one result";

  auto sendResumeInfo = generatorResumeClones.find(receiver.generatorTarget);
  if (sendResumeInfo != generatorResumeClones.end()) {
    // The synthesized advance driver raises StopIteration itself, so the
    // unwind originates at a call — the shape the ownership inserter models
    // when it places compensating releases at catch entries (an scf.if that
    // raises after the caller's releases would double-release).
    mlir::FailureOr<SourceGeneratorResumeResult> yieldedOr =
        RuntimeBundleLowerer::emitStateMachineGeneratorResume(
            op.getOperation(), receiver, sendResumeInfo->second,
            /*useCurrentInsertionPoint=*/false, sentI64Evidence,
            /*raiseWhenExhausted=*/true);
    if (mlir::failed(yieldedOr))
      return mlir::failure();
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::bundleRuntimeResults(
            op.getOperation(), op.getResult(0).getType(),
            mlir::ValueRange(yieldedOr->lanePhysicals), result)))
      return mlir::failure();
    result.setObjectLogicalOwnership(/*ownsObject=*/true);
    valueBundles[op.getResult(0)] = std::move(result);
    erase.push_back(op);
    return mlir::success();
  }
  if (runtimeContractName(op.getResult(0).getType()) != "builtins.int")
    return op.emitError()
           << "source generator resume currently supports int yield results";

  mlir::FailureOr<SourceGeneratorResumeResult> yieldedOr =
      RuntimeBundleLowerer::emitSourceGeneratorResumeDispatch(
          op.getOperation(), op.getResult(0).getType(), receiver,
          /*useCurrentInsertionPoint=*/false, sentI64Evidence);
  if (mlir::failed(yieldedOr))
    return mlir::failure();
  SourceGeneratorResumeResult yielded = *yieldedOr;

  mlir::Location loc = op.getLoc();
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  mlir::Value exhausted =
      mlir::arith::XOrIOp::create(builder, loc, yielded.hasValue, trueValue);
  auto stopIf = mlir::scf::IfOp::create(builder, loc, exhausted,
                                        /*withElseRegion=*/false);
  builder.setInsertionPoint(stopIf.getThenRegion().front().getTerminator());
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.StopIteration", "generator exhausted")))
    return mlir::failure();
  builder.setInsertionPointAfter(stopIf);

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult(0).getType(), yielded.value, yielded.valid, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerSourceGeneratorThrow(
    py::CallOp op, const RuntimeBundle &receiver,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError()
           << "source generator throw expects exactly one exception value";
  if (op.getNumResults() != 1 ||
      runtimeContractName(op.getResult(0).getType()) != "builtins.int")
    return op.emitError()
           << "source generator throw currently supports int yield results";
  const RuntimeBundle &exception = *sources[1];
  if (!manifest.primitive(exception.contractName(), "raise"))
    return op.emitError() << "source generator throw exception type "
                          << exception.contractName()
                          << " has no raise primitive";

  auto throwResumeInfo = generatorResumeClones.find(receiver.generatorTarget);
  if (throwResumeInfo != generatorResumeClones.end())
    return RuntimeBundleLowerer::lowerStateMachineGeneratorThrow(
        op, receiver, throwResumeInfo->second, sources);

  // Inline-dispatch generators have straight-line bodies without handlers,
  // so the exception can never be caught inside the body: closing the
  // generator and raising at the call site is observably CPython's throw().
  // The raise must carry the throw call site's try handler marker, so the
  // builder has to sit at the op's block (not wherever the previous lowering
  // left it).
  builder.setInsertionPoint(op);
  llvm::SmallVector<const RuntimeBundle *, 1> closeSources{&receiver};
  if (mlir::failed(lowerManifestVoidMethod(op, receiver, "close", closeSources,
                                           /*allowUnusedSources=*/false)))
    return mlir::failure();
  if (mlir::failed(RuntimeBundleLowerer::emitRaiseExceptionBundle(
          op.getOperation(), exception)))
    return mlir::failure();

  mlir::Value zero =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 64);
  mlir::Value invalid =
      mlir::arith::ConstantIntOp::create(builder, op.getLoc(), 0, 1);
  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult(0).getType(), zero, invalid, result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
