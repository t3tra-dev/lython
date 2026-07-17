#include "Runtime/Core/Lowerer.h"

#include "ExceptionTaxonomy.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"

// Loop-body generators: transform an int-pure generator body into a
// primitive-i64 RESUME state machine (pure SSA — no in-body memory):
//
//   @body(args: int...)                     [py CFG with py.yield_value]
//     ==>
//   @body__lyrt_gen_resume(args..., state: int, sent: int, inject: int,
//                          lives: int x K)
//       -> (state': int, has: int, value: int,
//           ret: int, hasret: int, lives': int x K)
//
// Every yield splits its block; the continuation's externally-defined values
// become block arguments (the live set, all builtins.int in this tier), the
// yield becomes `return (k, 1, value, 0, 0, lives...)`, and a raw-i64
// dispatch chain in the entry branches on the state argument. The `sent`
// lane carries the send() value into the resumed continuation (the yield
// expression result aliases the entry argument), and the `inject` lane makes
// the continuation rethrow the EH TLS exception at the suspension point —
// which is how throw() and close() (GeneratorExit) reach the body's
// handlers. `ret`/`hasret` carry the generator's return value out so the
// exhaustion StopIteration can expose it.
//
// The generator object stores state (word 4) and the live slots (words 8..)
// inline. All resumes go through synthesized driver functions instead of
// inline call-site expansion:
//
//   @body...__step(gen, argpairs..., sent, sentvalid, inject)
//       -> (has, value, valid, ret, retvalid)
//   @body...__advance = step + StopIteration(value) on exhaustion
//   @body...__throw   = TLS-stage exception, resume with inject=1
//   @body...__close   = GeneratorExit injection + swallow + closed state
//
// A separate function keeps the multi-block EH CFG legal at call sites that
// sit inside single-block SCF regions (for-loop conditions), and it makes
// the unwind edge originate at a call — which is the shape the ownership
// inserter models when it places compensating releases at catch entries.

namespace py::lowering {

namespace {

constexpr unsigned kGeneratorFrameSlotBase = 8;
constexpr unsigned kGeneratorFrameSlotLimit = 24;
constexpr llvm::StringLiteral kGeneratorBodyResultAttr{
    "ly.generator.body_result"};
constexpr llvm::StringLiteral kGeneratorPublicResultAttr{
    "ly.generator.public_result"};

// Backward liveness over a function region: liveIn(B) = values used in B or
// live into a successor, minus values B itself defines (op results and block
// arguments). Deterministic (SetVector, block/op order).
llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>>
computeLiveIns(mlir::Region &region) {
  llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>> liveIns;
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Block &blockRef : llvm::reverse(region.getBlocks())) {
      mlir::Block *block = &blockRef;
      llvm::SetVector<mlir::Value> live;
      for (mlir::Block *successor : block->getSuccessors())
        for (mlir::Value value : liveIns[successor])
          live.insert(value);
      for (mlir::Operation &op : llvm::reverse(*block)) {
        for (mlir::Value result : op.getResults())
          live.remove(result);
        for (mlir::Value operand : op.getOperands())
          live.insert(operand);
      }
      for (mlir::BlockArgument argument : block->getArguments())
        live.remove(argument);
      llvm::SetVector<mlir::Value> &slot = liveIns[block];
      if (live.size() != slot.size() ||
          !llvm::all_of(live,
                        [&](mlir::Value v) { return slot.contains(v); })) {
        slot = std::move(live);
        changed = true;
      }
    }
  }
  return liveIns;
}

// Values live immediately AFTER `yield`, excluding the function's entry
// block arguments (real arguments are re-passed on every resume; everything
// else must ride in the frame) and the yield's own sent result (it does not
// exist at suspend time — the resume's sent lane materializes it).
llvm::SetVector<mlir::Value> liveAfterYield(
    py::YieldValueOp yield,
    llvm::DenseMap<mlir::Block *, llvm::SetVector<mlir::Value>> &liveIns,
    mlir::Block *entry) {
  mlir::Block *block = yield->getBlock();
  llvm::SetVector<mlir::Value> live;
  for (mlir::Block *successor : block->getSuccessors())
    for (mlir::Value value : liveIns[successor])
      live.insert(value);
  for (mlir::Operation &op : llvm::reverse(*block)) {
    if (&op == yield.getOperation())
      break;
    for (mlir::Value result : op.getResults())
      live.remove(result);
    for (mlir::Value operand : op.getOperands())
      live.insert(operand);
  }
  live.remove(yield.getSent());
  llvm::SetVector<mlir::Value> filtered;
  for (mlir::Value value : live) {
    auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
    if (argument && argument.getOwner() == entry)
      continue;
    filtered.insert(value);
  }
  return filtered;
}

bool isIntContract(mlir::Type type) {
  return runtimeContractName(type) == "builtins.int";
}

bool isNoneLike(mlir::Type type) {
  if (runtimeContractName(type) == "types.NoneType")
    return true;
  auto literal = mlir::dyn_cast<py::LiteralType>(type);
  return literal && literal.getSpelling() == "None";
}

mlir::MemRefType generatorStorageType(mlir::OpBuilder &builder) {
  return mlir::MemRefType::get({kGeneratorFrameSlotLimit},
                               builder.getI64Type());
}

std::int64_t exceptionClassId(llvm::StringRef name) {
  const py::exceptions::BuiltinExceptionInfo *info =
      py::exceptions::findByName(name);
  return info ? info->classId : 0;
}

// Number of logical extra lanes before the frame slots: state, sent, inject.
constexpr unsigned kResumeControlLanes = 3;
// Number of logical result lanes before the frame slots:
// state', has, value, ret, hasret.
constexpr unsigned kResumeResultLanes = 5;

} // namespace

// Phase 1 (before prepareCallableFunctionABIs): decide eligibility, compute
// the frame width K, and create the resume clone with its extended logical
// signature so the primitive-i64 ABI machinery seeds it like any clone.
mlir::LogicalResult RuntimeBundleLowerer::buildGeneratorResumeCloneSignatures() {
  llvm::SmallVector<mlir::func::FuncOp, 4> bodies;
  module.walk([&](mlir::func::FuncOp fn) {
    if (fn->hasAttr(kGeneratorBodyResultAttr) && !fn.isDeclaration())
      bodies.push_back(fn);
  });

  for (mlir::func::FuncOp body : bodies) {
    bool hasYield = false;
    bool hasYieldFrom = false;
    bool unsupported = false;
    body.walk([&](mlir::Operation *op) {
      if (mlir::isa<py::YieldValueOp>(op)) {
        hasYield = true;
        return;
      }
      if (mlir::isa<py::YieldFromOp>(op)) {
        // Statically-bound `yield from inner(...)` is inlined into the
        // clone below; other delegation shapes fall back to the legacy
        // inline dispatch.
        hasYieldFrom = true;
        return;
      }
      // py.try is fine: the clone's tries are flattened to CFG below,
      // before the yield split, so suspension points inside try/finally
      // stay inside their (by then block-registered) handler scopes.
      if (op != body.getOperation() && op->getNumRegions() != 0 &&
          !mlir::isa<py::TryOp>(op))
        unsupported = true;
    });
    if (unsupported || (!hasYield && !hasYieldFrom))
      continue;
    // Straight-line int bodies also take the state machine: the inline
    // re-dispatch raises its exhaustion StopIteration inside an scf.if that
    // falls through, and an unwind that starts after the caller's releases
    // double-frees at the catch entry's compensating release. Ineligible
    // bodies (non-int surface, non-static delegation) still fall back to it.

    auto callableAttr =
        body->getAttrOfType<mlir::TypeAttr>(ownership::kCallableTypeAttr);
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || callable.hasVararg() || callable.hasKwarg())
      continue;
    if (!llvm::all_of(callable.getPositionalTypes(), isIntContract))
      continue;

    std::string cloneName = (body.getSymName() + "__lyrt_gen_resume").str();
    mlir::func::FuncOp clone;
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(body);
      clone = body.clone();
      clone.setSymName(cloneName);
      clone->removeAttr(kGeneratorBodyResultAttr);
      clone->removeAttr(kGeneratorPublicResultAttr);
      clone->setAttr(kPrimitiveI64CloneAttr,
                     builder.getStringAttr(body.getSymName()));
      // The affine-ownership verifier keys its generator-frame rule
      // (suspension boundaries carry NonObject lanes only) on this marker.
      clone->setAttr("ly.generator.resume", builder.getUnitAttr());
      mlir::SymbolTable::setSymbolVisibility(
          clone, mlir::SymbolTable::Visibility::Private);
      builder.insert(clone);
    }

    if (hasYieldFrom) {
      mlir::FailureOr<bool> inlined =
          RuntimeBundleLowerer::inlineDelegatedYieldFroms(clone);
      if (mlir::failed(inlined))
        return mlir::failure();
      if (!*inlined) {
        clone.erase();
        continue;
      }
    }

    // Eligibility on the (possibly delegation-inlined) clone: yields, sent
    // values and returns all ride primitive int lanes in this tier.
    bool eligible = true;
    llvm::SmallVector<py::YieldValueOp, 8> yields;
    clone.walk([&](mlir::Operation *op) {
      if (auto yield = mlir::dyn_cast<py::YieldValueOp>(op)) {
        yields.push_back(yield);
        if (!isIntContract(yield.getValue().getType()))
          eligible = false;
        if (!yield.getSent().use_empty() &&
            !isIntContract(yield.getSent().getType()))
          eligible = false;
        return;
      }
      if (mlir::isa<py::YieldFromOp>(op) ||
          (op != clone.getOperation() && op->getNumRegions() != 0 &&
           !mlir::isa<py::TryOp>(op)))
        eligible = false;
      // The return value rides the int ret lane; anything else would be
      // silently dropped from StopIteration.value, so reject it here and
      // let the inline path emit its diagnostic.
      if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
        for (mlir::Value operand : ret.getOperands())
          if (!isIntContract(operand.getType()) &&
              !isNoneLike(operand.getType()))
            eligible = false;
    });
    if (!eligible || yields.empty()) {
      clone.erase();
      continue;
    }

    // Flatten the clone's structured tries before anything looks at the
    // CFG: the yield split can only place suspends in function-level blocks
    // (func.return is illegal inside a py.try region), and the flattening
    // registers the try blocks' handler ids — which the split continuations
    // inherit, so injected exceptions unwind into the body's handlers.
    {
      llvm::SmallVector<py::TryOp, 4> tries;
      clone.walk([&](py::TryOp tryOp) { tries.push_back(tryOp); });
      for (py::TryOp tryOp : llvm::reverse(tries))
        if (mlir::failed(lowerTry(tryOp)))
          return mlir::failure();
    }

    // Frame width: max live-across-yield count, from backward liveness on
    // the flattened clone CFG (phase 2 recomputes on the same clone; the
    // sent alias inserted there replaces the yield result one-for-one, so
    // widths agree).
    unsigned frameWidth = 0;
    bool livesEligible = true;
    {
      llvm::SmallVector<py::YieldValueOp, 8> cloneYields;
      clone.walk(
          [&](py::YieldValueOp yield) { cloneYields.push_back(yield); });
      mlir::Block &cloneEntry = clone.getBody().front();
      auto liveIns = computeLiveIns(clone.getBody());
      for (py::YieldValueOp yield : cloneYields) {
        llvm::SetVector<mlir::Value> lives =
            liveAfterYield(yield, liveIns, &cloneEntry);
        for (mlir::Value live : lives)
          if (!isIntContract(live.getType()))
            livesEligible = false;
        frameWidth =
            std::max(frameWidth, static_cast<unsigned>(lives.size()));
      }
    }
    if (!livesEligible ||
        kGeneratorFrameSlotBase + frameWidth > kGeneratorFrameSlotLimit) {
      clone.erase();
      continue;
    }

    mlir::Type intContract = runtimeContractType(context, "builtins.int");
    llvm::SmallVector<mlir::Type, 8> argTypes(
        callable.getPositionalTypes().begin(),
        callable.getPositionalTypes().end());
    for (unsigned lane = 0; lane < kResumeControlLanes; ++lane)
      argTypes.push_back(intContract); // state, sent, inject
    for (unsigned slot = 0; slot < frameWidth; ++slot)
      argTypes.push_back(intContract);
    llvm::SmallVector<mlir::Type, 8> resultTypes;
    for (unsigned lane = 0; lane < kResumeResultLanes; ++lane)
      resultTypes.push_back(intContract); // state', has, value, ret, hasret
    for (unsigned slot = 0; slot < frameWidth; ++slot)
      resultTypes.push_back(intContract);

    auto newCallable = py::CallableType::get(
        context, argTypes, /*kwonly=*/{}, /*varargType=*/mlir::Type(),
        /*kwargsType=*/mlir::Type(), resultTypes);
    clone->setAttr(ownership::kCallableTypeAttr,
                   mlir::TypeAttr::get(newCallable));
    clone.setFunctionType(
        mlir::FunctionType::get(context, argTypes, resultTypes));
    mlir::Block &cloneEntry = clone.getBody().front();
    for (unsigned extra = 0; extra < kResumeControlLanes + frameWidth; ++extra)
      cloneEntry.addArgument(intContract, clone.getLoc());

    GeneratorResumeInfo info;
    info.cloneName = cloneName;
    info.frameWidth = frameWidth;
    info.argumentCount = static_cast<unsigned>(
        callable.getPositionalTypes().size());
    generatorResumeClones[body.getSymName().str()] = info;
  }
  return mlir::success();
}

// PEP 380 delegation by frame merging: a statically-bound
// `yield from inner(...)` clones inner's py-level body into the outer resume
// clone. The inner yields become outer suspension points, so send() values
// flow to whichever yield is active, throw()/close() injections unwind
// through inner handlers first, and the delegate's return value is simply
// the SSA value flowing into the yield-from result — no StopIteration
// transport is involved. Runs to fixpoint so an inlined body's own
// delegations inline too; the budget bounds (mutually) recursive delegation,
// which has no static expansion.
mlir::FailureOr<bool>
RuntimeBundleLowerer::inlineDelegatedYieldFroms(mlir::func::FuncOp clone) {
  for (unsigned round = 0; round < 64; ++round) {
    py::YieldFromOp yieldFrom;
    clone.walk([&](py::YieldFromOp op) {
      yieldFrom = op;
      return mlir::WalkResult::interrupt();
    });
    if (!yieldFrom)
      return true;

    auto call = yieldFrom.getSource().getDefiningOp<py::CallOp>();
    if (!call)
      return false;
    auto binding = call.getCallable().getDefiningOp<py::BindingRefOp>();
    if (!binding)
      return false;
    auto target =
        module.lookupSymbol<mlir::func::FuncOp>(binding.getBinding());
    if (!target || target.isDeclaration() ||
        !target->hasAttr(kGeneratorBodyResultAttr))
      return false;
    py::CallableType callableType = callableTypeOf(target);
    if (!callableType || callableType.hasVararg() || callableType.hasKwarg() ||
        !callableType.getKwOnlyTypes().empty())
      return false;
    std::optional<StaticCallableInvocation> invocation =
        RuntimeBundleLowerer::collectStaticCallableInvocation(call);
    if (!invocation)
      return false;
    std::optional<CallableArgumentPlan> plan =
        RuntimeBundleLowerer::collectCallableArgumentPlan(call, callableType,
                                                          /*emitErrors=*/false);
    if (!plan || !plan->defaultedFixed.empty() ||
        !plan->varargActuals.empty() || !plan->kwargActuals.empty())
      return false;
    llvm::ArrayRef<mlir::Type> positionalTypes =
        callableType.getPositionalTypes();
    if (plan->fixedActuals.size() != positionalTypes.size())
      return false;
    llvm::SmallVector<mlir::Value, 8> mappedArgs;
    for (std::optional<unsigned> actualIndex : plan->fixedActuals) {
      if (!actualIndex || *actualIndex >= invocation->actualValues.size())
        return false;
      mappedArgs.push_back(invocation->actualValues[*actualIndex]);
    }
    llvm::SmallVector<mlir::Type, 4> closureTypes =
        RuntimeBundleLowerer::callableClosureTypes(target);
    if (closureTypes.size() != binding.getCaptures().size())
      return false;
    for (mlir::Value capture : binding.getCaptures())
      mappedArgs.push_back(capture);
    mlir::Block &targetEntry = target.getBody().front();
    if (targetEntry.getNumArguments() != mappedArgs.size())
      return false;

    mlir::Value completion = yieldFrom.getResult();
    bool completionUsed = !completion.use_empty();
    bool completionIsInt = isIntContract(completion.getType());
    if (completionUsed && !completionIsInt && !isNoneLike(completion.getType()))
      return false;
    if (completionUsed && completionIsInt) {
      bool everyReturnCarriesInt = true;
      target.walk([&](mlir::func::ReturnOp ret) {
        if (llvm::none_of(ret.getOperands(), [](mlir::Value v) {
              return isIntContract(v.getType());
            }))
          everyReturnCarriesInt = false;
      });
      if (!everyReturnCarriesInt)
        return false;
    }
    for (auto [index, value] : llvm::enumerate(mappedArgs)) {
      mlir::Type expected = targetEntry.getArgument(index).getType();
      if (value.getType() != expected &&
          !py::isAssignableTo(value.getType(), expected, call))
        return false;
    }

    mlir::Location loc = yieldFrom.getLoc();
    mlir::Block *block = yieldFrom->getBlock();
    mlir::Block *after = block->splitBlock(
        std::next(mlir::Block::iterator(yieldFrom.getOperation())));
    // Coerce actuals whose type is narrower than the inner parameter.
    {
      mlir::OpBuilder b(yieldFrom);
      for (auto [index, value] : llvm::enumerate(mappedArgs)) {
        mlir::Type expected = targetEntry.getArgument(index).getType();
        if (value.getType() != expected)
          mappedArgs[index] =
              py::ClassUpcastOp::create(b, loc, expected, value).getResult();
      }
    }
    if (completionUsed) {
      if (completionIsInt) {
        mlir::BlockArgument arg =
            after->addArgument(completion.getType(), loc);
        completion.replaceAllUsesWith(arg);
      } else {
        mlir::OpBuilder b = mlir::OpBuilder::atBlockBegin(after);
        mlir::Value none =
            py::NoneOp::create(b, loc,
                               py::LiteralType::get(context, "None"))
                .getResult();
        if (none.getType() != completion.getType())
          none = py::ClassUpcastOp::create(b, loc, completion.getType(), none)
                     .getResult();
        completion.replaceAllUsesWith(none);
      }
    }

    mlir::IRMapping mapping;
    target.getBody().cloneInto(&clone.getBody(), after->getIterator(),
                               mapping);
    mlir::Block *innerEntry = mapping.lookup(&targetEntry);
    for (auto [index, value] : llvm::enumerate(mappedArgs))
      innerEntry->getArgument(static_cast<unsigned>(index))
          .replaceAllUsesWith(value);
    innerEntry->eraseArguments(0, innerEntry->getNumArguments());

    llvm::SmallVector<mlir::func::ReturnOp, 4> innerReturns;
    for (mlir::Block &targetBlock : target.getBody())
      if (mlir::Block *mapped = mapping.lookupOrNull(&targetBlock))
        for (mlir::Operation &op : *mapped)
          if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
            innerReturns.push_back(ret);
    for (mlir::func::ReturnOp ret : innerReturns) {
      mlir::OpBuilder b(ret);
      llvm::SmallVector<mlir::Value, 1> operands;
      if (completionUsed && completionIsInt) {
        mlir::Value returned;
        for (mlir::Value operand : ret.getOperands())
          if (isIntContract(operand.getType()))
            returned = operand;
        operands.push_back(returned);
      }
      mlir::cf::BranchOp::create(b, ret.getLoc(), after, operands);
      ret.erase();
    }

    {
      mlir::OpBuilder b(yieldFrom);
      mlir::cf::BranchOp::create(b, loc, innerEntry);
    }
    yieldFrom.erase();
    if (llvm::all_of(call->getResults(),
                     [](mlir::Value v) { return v.use_empty(); })) {
      llvm::SmallVector<mlir::Operation *, 4> packs;
      for (mlir::Value operand :
           {call.getPosargs(), call.getKwnames(), call.getKwvalues()})
        if (mlir::Operation *pack = operand.getDefiningOp())
          packs.push_back(pack);
      call.erase();
      for (mlir::Operation *pack : packs)
        if (pack->use_empty())
          pack->erase();
      if (binding->use_empty())
        binding->erase();
    }
  }
  return clone.emitError()
         << "yield from delegation exceeded the static inlining budget "
            "(recursive delegation has no static expansion)";
}

// Phase 2 (after ABI seeding): CFG surgery on the seeded clone.
mlir::LogicalResult RuntimeBundleLowerer::buildGeneratorResumeBodies() {
  for (auto &entry : generatorResumeClones) {
    GeneratorResumeInfo &info = entry.second;
    mlir::func::FuncOp clone =
        module.lookupSymbol<mlir::func::FuncOp>(info.cloneName);
    if (!clone)
      return module.emitError()
             << "generator resume clone " << info.cloneName << " is missing";
    unsigned argumentCount = info.argumentCount;
    unsigned frameWidth = info.frameWidth;
    unsigned logicalCount = argumentCount + kResumeControlLanes + frameWidth;
    mlir::Block &entryBlock = clone.getBody().front();
    if (entryBlock.getNumArguments() != logicalCount + 2 * logicalCount)
      return clone.emitError()
             << "generator resume clone entry was not seeded as a "
                "primitive-i64 callable";
    mlir::Location loc = clone.getLoc();
    mlir::Type intContract = runtimeContractType(context, "builtins.int");

    auto rawOf = [&](unsigned logicalIndex) {
      return entryBlock.getArgument(logicalCount + 2 * logicalIndex);
    };
    auto validOf = [&](unsigned logicalIndex) {
      return entryBlock.getArgument(logicalCount + 2 * logicalIndex + 1);
    };
    unsigned stateIndex = argumentCount;
    unsigned sentIndex = argumentCount + 1;
    unsigned injectIndex = argumentCount + 2;
    mlir::Value rawState = rawOf(stateIndex);

    // Logical int constant with primitive evidence: an unbound py constant
    // would need the full py lowering chain; instead reuse the entry's own
    // logical/raw pairing trick — materialize via py.int.constant and let
    // the walk lower it (it runs after this transform).
    auto intLiteral = [&](mlir::OpBuilder &b,
                          std::int64_t value) -> mlir::Value {
      std::string text = std::to_string(value);
      mlir::Type literalType = py::LiteralType::get(context, text);
      mlir::Value literal =
          py::IntConstantOp::create(b, loc, literalType,
                                    b.getStringAttr(text))
              .getResult();
      return py::ClassUpcastOp::create(b, loc, intContract, literal)
          .getResult();
    };

    // Collect yields in deterministic order and the original returns.
    llvm::SmallVector<py::YieldValueOp, 8> yields;
    llvm::SmallVector<mlir::func::ReturnOp, 4> originalReturns;
    clone.walk([&](mlir::Operation *op) {
      if (auto yield = mlir::dyn_cast<py::YieldValueOp>(op))
        yields.push_back(yield);
      else if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op))
        originalReturns.push_back(ret);
    });
    std::int64_t doneState = static_cast<std::int64_t>(yields.size()) + 1;

    // Sent values: the yield expression result does not exist at suspend
    // time — it is the NEXT resume's sent lane. Materialize an alias op so
    // the value participates in liveness like any other int (a send() value
    // held across a later yield must ride the frame), and swap its operand
    // to the entry sent argument after the split.
    llvm::DenseMap<mlir::Operation *, py::ClassUpcastOp> sentAliases;
    for (py::YieldValueOp yield : yields) {
      if (yield.getSent().use_empty())
        continue;
      mlir::OpBuilder b(context);
      b.setInsertionPointAfter(yield);
      auto alias = py::ClassUpcastOp::create(b, loc, yield.getSent().getType(),
                                             yield.getSent());
      yield.getSent().replaceAllUsesExcept(alias.getResult(), alias);
      sentAliases[yield.getOperation()] = alias;
    }

    // Exhausted / falling-off-the-end returns. A `return X` return threads X
    // through the ret lane so the exhaustion StopIteration can carry it.
    for (mlir::func::ReturnOp ret : originalReturns) {
      mlir::OpBuilder b(ret);
      mlir::Value returnedValue;
      for (mlir::Value operand : ret.getOperands())
        if (isIntContract(operand.getType()))
          returnedValue = operand;
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(intLiteral(b, doneState));
      operands.push_back(intLiteral(b, 0)); // has
      operands.push_back(intLiteral(b, 0)); // value
      operands.push_back(returnedValue ? returnedValue : intLiteral(b, 0));
      operands.push_back(intLiteral(b, returnedValue ? 1 : 0));
      for (unsigned slot = 0; slot < frameWidth; ++slot)
        operands.push_back(intLiteral(b, 0));
      mlir::func::ReturnOp::create(b, ret.getLoc(), operands);
      ret.erase();
    }

    // Live sets from backward liveness on the still-unsplit body, so the
    // frame covers uses reached through successor blocks (joins, loop
    // headers), not just the block suffix.
    auto liveIns = computeLiveIns(clone.getBody());

    // Split at each yield; the yield's live-across set becomes the
    // continuation's logical block arguments and the suspend payload.
    struct Continuation {
      mlir::Block *block = nullptr;
      unsigned liveCount = 0;
    };
    llvm::SmallVector<Continuation, 8> continuations;
    llvm::DenseSet<mlir::Block *> continuationBlocks;
    llvm::DenseMap<std::pair<mlir::Block *, mlir::Value>, mlir::BlockArgument>
        blockValueArguments;
    for (auto [index, yield] : llvm::enumerate(yields)) {
      std::int64_t state = static_cast<std::int64_t>(index) + 1;
      mlir::Block *block = yield->getBlock();
      llvm::SetVector<mlir::Value> lives =
          liveAfterYield(yield, liveIns, &entryBlock);
      if (lives.size() > frameWidth)
        return yield.emitError()
               << "generator resume live set exceeds the computed frame";
      mlir::Block *cont = block->splitBlock(
          std::next(mlir::Block::iterator(yield.getOperation())));
      // A suspension point inside a (flattened) try keeps its handler scope:
      // the continuation inherits the block's handler id so the injected
      // rethrow and the continuation's calls unwind into the body handlers.
      {
        auto handlerIt = tryHandlerIds.find(block);
        if (handlerIt != tryHandlerIds.end())
          tryHandlerIds.try_emplace(cont, handlerIt->second);
      }
      llvm::SmallVector<mlir::Value, 8> liveValues(lives.begin(), lives.end());
      for (mlir::Value live : liveValues) {
        mlir::BlockArgument arg = cont->addArgument(live.getType(), loc);
        blockValueArguments[{cont, live}] = arg;
        live.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
          return use.getOwner()->getBlock() == cont;
        });
        // Register with the logical block-argument machinery so the standard
        // post-lowering passes drop the logical operands/arguments (the raw
        // evidence pairs are the runtime ABI).
        if (controlFlowLogicalBlockArgumentSet.insert(arg).second)
          controlFlowLogicalBlockArguments.push_back(
              ControlFlowLogicalBlockArgumentABI{arg});
      }
      // The resumed continuation reads the CURRENT resume's sent lane: the
      // alias op moved into the continuation with the split; wire it to the
      // entry sent argument (visible everywhere — entry dominates).
      auto aliasIt = sentAliases.find(yield.getOperation());
      if (aliasIt != sentAliases.end())
        aliasIt->second->setOperand(0, entryBlock.getArgument(sentIndex));
      // The suspend: return (state, 1, value, 0, 0, lives...). External live
      // operands are rethreaded by the normalization pass below.
      {
        mlir::OpBuilder b = mlir::OpBuilder::atBlockEnd(block);
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.push_back(intLiteral(b, state));
        operands.push_back(intLiteral(b, 1));
        operands.push_back(yield.getValue());
        operands.push_back(intLiteral(b, 0));
        operands.push_back(intLiteral(b, 0));
        for (mlir::Value live : liveValues)
          operands.push_back(live);
        for (unsigned slot = static_cast<unsigned>(liveValues.size());
             slot < frameWidth; ++slot)
          operands.push_back(intLiteral(b, 0));
        mlir::func::ReturnOp::create(b, loc, operands);
      }
      // Continuation entry guards, in order: (1) an injected exception
      // (throw/close) rethrows the TLS current exception at the suspension
      // point, inside the body's handler scope; (2) resuming a sent-value
      // continuation via next() would materialize None in an int slot, so
      // it is rejected at this boundary (CPython raises TypeError when the
      // None reaches the int operation; there is no None-in-int runtime
      // representation here, so the boundary is the earliest sound point).
      {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(cont);
        mlir::Value zero =
            mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
        mlir::Value injected = mlir::arith::CmpIOp::create(
            builder, loc, mlir::arith::CmpIPredicate::ne, rawOf(injectIndex),
            zero);
        auto injectIf = mlir::scf::IfOp::create(builder, loc, injected,
                                                /*withElseRegion=*/false);
        builder.setInsertionPoint(
            injectIf.getThenRegion().front().getTerminator());
        emitTryCallSiteMarkerIfNeeded(loc);
        mlir::func::CallOp::create(builder, loc,
                                   getOrCreateRethrowCurrent(module, builder),
                                   mlir::ValueRange{});
        builder.setInsertionPointAfter(injectIf);
        if (aliasIt != sentAliases.end()) {
          mlir::Value trueValue =
              mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
          mlir::Value sentMissing = mlir::arith::XOrIOp::create(
              builder, loc, validOf(sentIndex), trueValue);
          auto sentIf = mlir::scf::IfOp::create(builder, loc, sentMissing,
                                                /*withElseRegion=*/false);
          builder.setInsertionPoint(
              sentIf.getThenRegion().front().getTerminator());
          if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
                  clone.getOperation(), "builtins.TypeError",
                  "generator expects a builtins.int sent value; resume it "
                  "with send(), not next()")))
            return mlir::failure();
        }
      }
      yield.erase();
      continuationBlocks.insert(cont);
      continuations.push_back(
          Continuation{cont, static_cast<unsigned>(liveValues.size())});
    }

    // Detach the original first block from the entry so the entry can become
    // the dispatch. Entry-defined ops move with it; only entry ARGUMENTS
    // remain visible everywhere.
    mlir::Block *originalStart =
        entryBlock.splitBlock(entryBlock.begin());

    // Normalization: resume dispatch adds CFG edges into the middle of the
    // body, so plain dominance no longer holds. Make every cross-block value
    // an explicit block argument (iterated to fixpoint); afterwards each
    // block only reads its own arguments, its own ops, and entry arguments.
    // The liveness-derived continuation arguments are closed under this
    // rewrite, so continuations never grow here.
    bool normalizing = true;
    while (normalizing) {
      normalizing = false;
      for (mlir::Block &blockRef : clone.getBody()) {
        mlir::Block *block = &blockRef;
        if (block == &entryBlock)
          continue;
        llvm::SetVector<mlir::Value> external;
        for (mlir::Operation &op : *block)
          for (mlir::Value operand : op.getOperands()) {
            if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
              if (argument.getOwner() == block ||
                  argument.getOwner() == &entryBlock)
                continue;
            } else if (operand.getDefiningOp()->getBlock() == block) {
              continue;
            }
            external.insert(operand);
          }
        if (external.empty())
          continue;
        if (continuationBlocks.contains(block))
          return clone.emitError()
                 << "generator resume continuation live closure violated";
        normalizing = true;
        for (mlir::Value value : external) {
          mlir::BlockArgument arg = block->addArgument(value.getType(), loc);
          blockValueArguments[{block, value}] = arg;
          value.replaceUsesWithIf(arg, [&](mlir::OpOperand &use) {
            return use.getOwner()->getBlock() == block;
          });
          for (mlir::Block *pred : block->getPredecessors()) {
            mlir::BlockArgument threaded =
                blockValueArguments.lookup({pred, value});
            mlir::Value passed = threaded ? mlir::Value(threaded) : value;
            mlir::Operation *terminator = pred->getTerminator();
            if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
              branch.getDestOperandsMutable().append(passed);
            } else if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(
                           terminator)) {
              if (cond.getTrueDest() == block)
                cond.getTrueDestOperandsMutable().append(passed);
              if (cond.getFalseDest() == block)
                cond.getFalseDestOperandsMutable().append(passed);
            } else {
              return terminator->emitError()
                     << "unsupported terminator in generator resume "
                        "normalization";
            }
          }
        }
      }
    }

    // Raw evidence pairs for the logical continuation arguments, and bundles
    // so the py ops in the continuations lower on the primitive lane.
    for (Continuation &continuation : continuations) {
      mlir::Block *cont = continuation.block;
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        mlir::BlockArgument raw = cont->addArgument(builder.getI64Type(), loc);
        mlir::BlockArgument valid = cont->addArgument(builder.getI1Type(), loc);
        RuntimeBundle bundle = RuntimeBundle::objectWithOwnership(
            intContract, mlir::ValueRange{},
            ownership::logicalOwnershipKind(intContract,
                                            /*ownsObject=*/false));
        bundle.primitiveI64 = RuntimePrimitiveI64Evidence{raw, valid};
        valueBundles[cont->getArgument(position)] = std::move(bundle);
      }
    }

    // Entry dispatch: chain raw-state comparisons in the (now empty) entry.
    // State 0 falls through to the original body.
    mlir::OpBuilder b = mlir::OpBuilder::atBlockEnd(&entryBlock);
    mlir::Block *current = &entryBlock;
    for (auto [index, continuation] : llvm::enumerate(continuations)) {
      std::int64_t state = static_cast<std::int64_t>(index) + 1;
      mlir::Block *next = clone.addBlock();
      b.setInsertionPointToEnd(current);
      mlir::Value expected =
          mlir::arith::ConstantIntOp::create(b, loc, state, 64);
      mlir::Value matches = mlir::arith::CmpIOp::create(
          b, loc, mlir::arith::CmpIPredicate::eq, rawState, expected);
      llvm::SmallVector<mlir::Value, 8> destOperands;
      for (unsigned position = 0; position < continuation.liveCount;
           ++position)
        destOperands.push_back(entryBlock.getArgument(
            argumentCount + kResumeControlLanes + position));
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        destOperands.push_back(
            rawOf(argumentCount + kResumeControlLanes + position));
        destOperands.push_back(
            validOf(argumentCount + kResumeControlLanes + position));
      }
      mlir::cf::CondBranchOp::create(b, loc, matches, continuation.block,
                                     destOperands, next, mlir::ValueRange{});
      current = next;
    }
    // Default: state 0 → run from the start; anything else is exhausted.
    {
      b.setInsertionPointToEnd(current);
      mlir::Value zero = mlir::arith::ConstantIntOp::create(b, loc, 0, 64);
      mlir::Value fresh = mlir::arith::CmpIOp::create(
          b, loc, mlir::arith::CmpIPredicate::eq, rawState, zero);
      mlir::Block *exhausted = clone.addBlock();
      mlir::cf::CondBranchOp::create(b, loc, fresh, originalStart,
                                     mlir::ValueRange{}, exhausted,
                                     mlir::ValueRange{});
      b.setInsertionPointToEnd(exhausted);
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(intLiteral(b, doneState));
      for (unsigned lane = 1; lane < kResumeResultLanes; ++lane)
        operands.push_back(intLiteral(b, 0));
      for (unsigned slot = 0; slot < frameWidth; ++slot)
        operands.push_back(intLiteral(b, 0));
      mlir::func::ReturnOp::create(b, loc, operands);
    }
  }
  return mlir::success();
}

// step: resume the generator once. Loads state + live slots from the
// generator object, calls the resume clone through the primitive ABI, stores
// them back, and keeps the lifecycle word (word 2) coherent. When the body
// escapes with an exception the generator is marked exhausted first
// (resuming after a raise must report exhaustion, not re-run the body), and
// an escaping StopIteration becomes RuntimeError (PEP 479).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.stepName.empty()) {
    if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(info.stepName))
      return existing;
  }
  mlir::func::FuncOp clone =
      module.lookupSymbol<mlir::func::FuncOp>(info.cloneName);
  if (!clone)
    return op->emitError() << "generator resume clone " << info.cloneName
                           << " is not defined";
  mlir::Location loc = clone.getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  inputs.push_back(i64); // sent
  inputs.push_back(i1);  // sent valid
  inputs.push_back(i64); // inject
  llvm::SmallVector<mlir::Type, 5> results{i1, i64, i1, i64, i1};

  std::string name = info.cloneName + "__step";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, results));
  function.setPrivate();
  info.stepName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  mlir::Value generator = entry->getArgument(0);
  unsigned sentValueIndex = 1 + 2 * info.argumentCount;
  builder.setInsertionPointToStart(entry);
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };

  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value state =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(4))
          .getResult();
  // Lifecycle gate (word 2: 3 = exhausted, 4 = closed): a finished generator
  // must not run body code. Feeding the clone an out-of-range state routes
  // it to the pure exhausted return, so no branch is needed.
  mlir::Value finished = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, lifecycle, i64Const(3));
  state = mlir::arith::SelectOp::create(builder, loc, finished, i64Const(-1),
                                        state)
              .getResult();
  // send(non-None) before the first resume never reaches body code in
  // CPython either; reject it here where the state is known.
  mlir::Value created = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  mlir::Value sentValid = entry->getArgument(sentValueIndex + 1);
  mlir::Value inject = entry->getArgument(sentValueIndex + 2);
  mlir::Value noInject = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, inject, i64Const(0));
  mlir::Value freshSend =
      mlir::arith::AndIOp::create(builder, loc, created, sentValid);
  freshSend = mlir::arith::AndIOp::create(builder, loc, freshSend, noInject);
  {
    auto freshIf = mlir::scf::IfOp::create(builder, loc, freshSend,
                                           /*withElseRegion=*/false);
    mlir::OpBuilder::InsertionGuard freshGuard(builder);
    builder.setInsertionPoint(freshIf.getThenRegion().front().getTerminator());
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            clone.getOperation(), "builtins.TypeError",
            "can't send non-None value to a just-started generator")))
      return mlir::failure();
  }

  std::int64_t handlerId = nextTryHandlerId++;
  mlir::Block *callBlock = builder.createBlock(&body);
  mlir::Block *catchBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(handlerId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0), catchBlock,
                                 mlir::ValueRange{}, callBlock,
                                 mlir::ValueRange{});

  // Normal resume.
  builder.setInsertionPointToEnd(callBlock);
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  llvm::SmallVector<mlir::Value, 24> operands;
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    operands.push_back(entry->getArgument(1 + 2 * index));
    operands.push_back(entry->getArgument(2 + 2 * index));
  }
  operands.push_back(state);
  operands.push_back(trueValue);
  operands.push_back(entry->getArgument(sentValueIndex));
  operands.push_back(sentValid);
  operands.push_back(inject);
  operands.push_back(trueValue);
  for (unsigned slot = 0; slot < info.frameWidth; ++slot) {
    mlir::Value live = mlir::memref::LoadOp::create(
                           builder, loc, generator,
                           slotIndex(kGeneratorFrameSlotBase + slot))
                           .getResult();
    operands.push_back(live);
    operands.push_back(trueValue);
  }
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, clone, operands);
  unsigned expectedResults = 2 * (kResumeResultLanes + info.frameWidth);
  if (call.getNumResults() != expectedResults)
    return op->emitError() << "generator resume clone ABI mismatch";
  mlir::Value newState = call.getResult(0);
  mlir::Value hasRaw = call.getResult(2);
  mlir::Value value = call.getResult(4);
  mlir::Value valueValid = call.getResult(5);
  mlir::Value returnedValue = call.getResult(6);
  mlir::Value returnedRaw = call.getResult(8);
  mlir::memref::StoreOp::create(builder, loc, newState, generator,
                                slotIndex(4));
  for (unsigned slot = 0; slot < info.frameWidth; ++slot)
    mlir::memref::StoreOp::create(
        builder, loc, call.getResult(2 * kResumeResultLanes + 2 * slot),
        generator, slotIndex(kGeneratorFrameSlotBase + slot));
  mlir::Value hasValue = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, hasRaw, i64Const(0));
  mlir::Value hasReturned = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, returnedRaw, i64Const(0));
  // Suspended while values flow, exhausted when the body finishes; a
  // finished generator keeps its previous (exhausted/closed) state.
  mlir::Value doneLifecycle =
      mlir::arith::SelectOp::create(builder, loc, finished, lifecycle,
                                    i64Const(3))
          .getResult();
  mlir::Value nextLifecycle =
      mlir::arith::SelectOp::create(builder, loc, hasValue, i64Const(2),
                                    doneLifecycle)
          .getResult();
  mlir::memref::StoreOp::create(builder, loc, nextLifecycle, generator,
                                slotIndex(2));
  mlir::func::ReturnOp::create(
      builder, loc,
      mlir::ValueRange{hasValue, value, valueValid, returnedValue,
                       hasReturned});

  // Body escaped with an exception: the generator is finished for good, and
  // PEP 479 turns an escaping StopIteration into RuntimeError. Everything
  // else propagates to the resumer.
  builder.setInsertionPointToEnd(catchBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::memref::StoreOp::create(builder, loc, i64Const(-1), generator,
                                slotIndex(4));
  mlir::memref::StoreOp::create(builder, loc, i64Const(3), generator,
                                slotIndex(2));
  auto currentMatches = getOrCreatePrivateFunction(
      module, builder, "LyEH_CurrentExceptionMatches",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
  auto isStopIteration = mlir::func::CallOp::create(
      builder, loc, currentMatches,
      mlir::ValueRange{i64Const(exceptionClassId("StopIteration"))});
  mlir::Block *pep479Block = builder.createBlock(&body);
  mlir::Block *rethrowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(catchBlock);
  mlir::cf::CondBranchOp::create(builder, loc, isStopIteration.getResult(0),
                                 pep479Block, mlir::ValueRange{}, rethrowBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(pep479Block);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          clone.getOperation(), "builtins.RuntimeError",
          "generator raised StopIteration")))
    return mlir::failure();
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(pep479Block);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// advance: step + the next()/send() exhaustion protocol — StopIteration
// carrying str(return value) as its message (CPython's str(StopIteration(v))
// is str(v); the typed .value attribute is not represented yet).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.advanceName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.advanceName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> step =
      RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(step))
    return mlir::failure();
  mlir::Location loc = step->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  std::string name = info.cloneName + "__advance";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name,
      builder.getFunctionType(step->getFunctionType().getInputs(),
                              {i64, i1}));
  function.setPrivate();
  info.advanceName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  builder.setInsertionPointToStart(entry);
  mlir::func::CallOp call = mlir::func::CallOp::create(
      builder, loc, *step, entry->getArguments());
  mlir::Block *yieldedBlock = builder.createBlock(&body);
  mlir::Block *stoppedBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  mlir::cf::CondBranchOp::create(builder, loc, call.getResult(0), yieldedBlock,
                                 mlir::ValueRange{}, stoppedBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(yieldedBlock);
  mlir::func::ReturnOp::create(
      builder, loc, mlir::ValueRange{call.getResult(1), call.getResult(2)});

  builder.setInsertionPointToEnd(stoppedBlock);
  mlir::Block *valueBlock = builder.createBlock(&body);
  mlir::Block *plainBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(stoppedBlock);
  mlir::cf::CondBranchOp::create(builder, loc, call.getResult(4), valueBlock,
                                 mlir::ValueRange{}, plainBlock,
                                 mlir::ValueRange{});

  // return X → StopIteration whose message is str(X). Exhaustion can be
  // observed while another exception is the pending current one (next()
  // inside an except handler); the single-token TLS slot requires the same
  // discard-before-raise that py.raise lowering performs.
  builder.setInsertionPointToEnd(valueBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  {
    std::optional<RuntimeSymbol> intNew =
        manifest.initializer("builtins.int", "__new__");
    std::optional<RuntimeSymbol> intStr =
        manifest.method("builtins.int", "__str__");
    if (!intNew || !intStr)
      return op->emitError() << "runtime manifest cannot render the generator "
                                "return value (builtins.int __new__/__str__)";
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *intNew, mlir::ValueRange{call.getResult(3)});
    mlir::func::CallOp text = RuntimeBundleLowerer::createRuntimeCall(
        loc, *intStr, boxed.getResults());
    // The boxed int is only a rendering temporary; drop it through the
    // manifest deallocator rather than teaching this synthesized body the
    // elision machinery.
    RuntimeBundle boxedBundle;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, "builtins.int"),
            boxed.getResults(), boxedBundle)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::releaseAggregateSlot(
            op, boxedBundle, "generator return value rendering")))
      return mlir::failure();
    RuntimeBundle message;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, runtimeContractType(context, "builtins.str"),
            text.getResults(), message)))
      return mlir::failure();
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
            op, "builtins.StopIteration", message)))
      return mlir::failure();
  }
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(valueBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(plainBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, "builtins.StopIteration", "")))
    return mlir::failure();
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(plainBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// throw: stage the exception in the EH TLS slot (raise + immediate catch),
// then resume with inject=1 so the continuation rethrows it at the
// suspension point, inside the body's handler scope. A created/finished
// generator never runs body code: the exception propagates directly (and a
// never-started generator closes).
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorThrowFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.throwName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.throwName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> advance =
      RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(op, info);
  if (mlir::failed(advance))
    return mlir::failure();
  mlir::Location loc = advance->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();
  auto headerType = mlir::MemRefType::get({3}, i64);
  auto messageType = mlir::MemRefType::get({2}, i64);
  auto bytesType =
      mlir::MemRefType::get({mlir::ShapedType::kDynamic}, builder.getI8Type());

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  unsigned headerIndex = 1 + 2 * info.argumentCount;
  inputs.push_back(headerType);
  inputs.push_back(messageType);
  inputs.push_back(bytesType);

  std::string name = info.cloneName + "__throw";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, {i64, i1}));
  function.setPrivate();
  // The caller hands the exception over for good — like the *_Raise
  // primitives (the TLS slot takes it).
  function->setAttr(
      "ly.ownership.transfer_args",
      builder.getI64ArrayAttr({static_cast<std::int64_t>(headerIndex),
                               static_cast<std::int64_t>(headerIndex + 1)}));
  function.setArgAttr(headerIndex, "ly.ownership.object_header",
                      builder.getUnitAttr());
  function.setArgAttr(headerIndex + 1, "ly.ownership.object_header",
                      builder.getUnitAttr());
  info.throwName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  builder.setInsertionPointToStart(entry);
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  auto throwException = getOrCreatePrivateFunction(
      module, builder, "LyEH_ThrowException",
      builder.getFunctionType({headerType, messageType, bytesType}, {}));

  std::int64_t handlerId = nextTryHandlerId++;
  mlir::Block *raiseBlock = builder.createBlock(&body);
  mlir::Block *caughtBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(entry);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(handlerId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0),
                                 caughtBlock, mlir::ValueRange{}, raiseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::func::CallOp::create(
      builder, loc, throwException,
      mlir::ValueRange{entry->getArgument(headerIndex),
                       entry->getArgument(headerIndex + 1),
                       entry->getArgument(headerIndex + 2)});
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(caughtBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(handlerId)});
  mlir::Value generator = entry->getArgument(0);
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };
  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value suspended = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(2));
  mlir::Block *injectBlock = builder.createBlock(&body);
  mlir::Block *directBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(caughtBlock);
  mlir::cf::CondBranchOp::create(builder, loc, suspended, injectBlock,
                                 mlir::ValueRange{}, directBlock,
                                 mlir::ValueRange{});

  builder.setInsertionPointToEnd(injectBlock);
  llvm::SmallVector<mlir::Value, 16> advanceOperands;
  advanceOperands.push_back(generator);
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    advanceOperands.push_back(entry->getArgument(1 + 2 * index));
    advanceOperands.push_back(entry->getArgument(2 + 2 * index));
  }
  advanceOperands.push_back(i64Const(0));
  advanceOperands.push_back(constantI1(builder, loc, false));
  advanceOperands.push_back(i64Const(1)); // inject
  mlir::func::CallOp resumed =
      mlir::func::CallOp::create(builder, loc, *advance, advanceOperands);
  mlir::func::ReturnOp::create(builder, loc, resumed.getResults());

  // Not suspended: the body never sees the exception. CPython closes a
  // just-started generator and raises the thrown exception as-is.
  builder.setInsertionPointToEnd(directBlock);
  mlir::Value wasCreated = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  auto closeIf = mlir::scf::IfOp::create(builder, loc, wasCreated,
                                         /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard closeGuard(builder);
    builder.setInsertionPoint(closeIf.getThenRegion().front().getTerminator());
    mlir::memref::StoreOp::create(builder, loc, i64Const(4), generator,
                                  slotIndex(2));
  }
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(directBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  return function;
}

// close: PEP 342 close(). GeneratorExit is injected at the suspension point
// (so finally blocks run); the generator swallowing it (returning) or the
// exception coming back out both close the generator, while yielding again
// is the "generator ignored GeneratorExit" RuntimeError.
mlir::FailureOr<mlir::func::FuncOp>
RuntimeBundleLowerer::getOrCreateGeneratorCloseFunction(
    mlir::Operation *op, GeneratorResumeInfo &info) {
  if (!info.closeName.empty()) {
    if (auto existing =
            module.lookupSymbol<mlir::func::FuncOp>(info.closeName))
      return existing;
  }
  mlir::FailureOr<mlir::func::FuncOp> step =
      RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(step))
    return mlir::failure();
  std::optional<RuntimeSymbol> exitNew =
      manifest.initializer("builtins.GeneratorExit", "__new__");
  std::optional<RuntimeSymbol> exitRaise =
      manifest.primitive("builtins.GeneratorExit", "raise");
  if (!exitNew || !exitRaise)
    return op->emitError()
           << "runtime manifest has no builtins.GeneratorExit support";
  mlir::Location loc = step->getLoc();
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Type i64 = builder.getI64Type();
  mlir::Type i1 = builder.getI1Type();

  llvm::SmallVector<mlir::Type, 16> inputs;
  inputs.push_back(generatorStorageType(builder));
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    inputs.push_back(i64);
    inputs.push_back(i1);
  }
  std::string name = info.cloneName + "__close";
  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputs, {}));
  function.setPrivate();
  info.closeName = name;

  mlir::Block *entry = function.addEntryBlock();
  mlir::Region &body = function.getBody();
  mlir::Value generator = entry->getArgument(0);
  builder.setInsertionPointToStart(entry);
  auto i64Const = [&](std::int64_t value) {
    return mlir::arith::ConstantIntOp::create(builder, loc, value, 64)
        .getResult();
  };
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };

  mlir::Block *checkFinished = builder.createBlock(&body);
  mlir::Block *injectBlock = builder.createBlock(&body);
  mlir::Block *markClosed = builder.createBlock(&body);
  mlir::Block *returnBlock = builder.createBlock(&body);

  builder.setInsertionPointToEnd(entry);
  mlir::Value lifecycle =
      mlir::memref::LoadOp::create(builder, loc, generator, slotIndex(2))
          .getResult();
  mlir::Value created = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::eq, lifecycle, i64Const(0));
  mlir::cf::CondBranchOp::create(builder, loc, created, markClosed,
                                 mlir::ValueRange{}, checkFinished,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(checkFinished);
  mlir::Value finished = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, lifecycle, i64Const(3));
  mlir::cf::CondBranchOp::create(builder, loc, finished, returnBlock,
                                 mlir::ValueRange{}, injectBlock,
                                 mlir::ValueRange{});

  // Stage GeneratorExit in the TLS slot (raise + immediate catch), resume
  // with inject=1, and interpret the outcome.
  builder.setInsertionPointToEnd(injectBlock);
  std::int64_t stageId = nextTryHandlerId++;
  std::int64_t resumeId = nextTryHandlerId++;
  mlir::Block *raiseBlock = builder.createBlock(&body);
  mlir::Block *stagedBlock = builder.createBlock(&body);
  mlir::Block *resumeBlock = builder.createBlock(&body);
  mlir::Block *swallowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(injectBlock);
  auto anchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(stageId)});
  mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0),
                                 stagedBlock, mlir::ValueRange{}, raiseBlock,
                                 mlir::ValueRange{});
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::func::CallOp exitObject = RuntimeBundleLowerer::createRuntimeCall(
      loc, *exitNew,
      mlir::ValueRange{i64Const(exceptionClassId("GeneratorExit"))});
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(stageId)});
  RuntimeBundleLowerer::createRuntimeCall(loc, *exitRaise,
                                          exitObject.getResults());
  mlir::Block *deadBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(raiseBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);
  builder.setInsertionPointToEnd(deadBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock);

  builder.setInsertionPointToEnd(stagedBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(stageId)});
  auto resumeAnchor = mlir::func::CallOp::create(
      builder, loc, getOrCreateTryCatchAnchor(),
      mlir::ValueRange{i64Const(resumeId)});
  mlir::cf::CondBranchOp::create(builder, loc, resumeAnchor.getResult(0),
                                 swallowBlock, mlir::ValueRange{}, resumeBlock,
                                 mlir::ValueRange{});

  builder.setInsertionPointToEnd(resumeBlock);
  llvm::SmallVector<mlir::Value, 16> stepOperands;
  stepOperands.push_back(generator);
  for (unsigned index = 0; index < info.argumentCount; ++index) {
    stepOperands.push_back(entry->getArgument(1 + 2 * index));
    stepOperands.push_back(entry->getArgument(2 + 2 * index));
  }
  stepOperands.push_back(i64Const(0));
  stepOperands.push_back(constantI1(builder, loc, false));
  stepOperands.push_back(i64Const(1)); // inject
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCallSiteMarker(),
                             mlir::ValueRange{i64Const(resumeId)});
  mlir::func::CallOp resumed =
      mlir::func::CallOp::create(builder, loc, *step, stepOperands);
  auto ignoredIf = mlir::scf::IfOp::create(builder, loc, resumed.getResult(0),
                                           /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard ignoredGuard(builder);
    builder.setInsertionPoint(
        ignoredIf.getThenRegion().front().getTerminator());
    // The staged GeneratorExit may still be the pending current exception
    // (the body can be suspended inside its own handler); the EH TLS slot is
    // single-token, so raising without discarding would trap.
    mlir::func::CallOp::create(
        builder, loc, getOrCreateDiscardCurrentException(module, builder),
        mlir::ValueRange{});
    if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
            op, "builtins.RuntimeError", "generator ignored GeneratorExit")))
      return mlir::failure();
  }
  mlir::cf::BranchOp::create(builder, loc, markClosed);

  // The injected GeneratorExit came back out: that is the normal close path.
  // Anything else (body exceptions, PEP 479 RuntimeError) propagates.
  builder.setInsertionPointToEnd(swallowBlock);
  mlir::func::CallOp::create(builder, loc, getOrCreateTryCatchMarker(),
                             mlir::ValueRange{i64Const(resumeId)});
  auto currentMatches = getOrCreatePrivateFunction(
      module, builder, "LyEH_CurrentExceptionMatches",
      builder.getFunctionType({builder.getI64Type()}, {builder.getI1Type()}));
  auto isExit = mlir::func::CallOp::create(
      builder, loc, currentMatches,
      mlir::ValueRange{i64Const(exceptionClassId("GeneratorExit"))});
  mlir::Block *discardBlock = builder.createBlock(&body);
  mlir::Block *rethrowBlock = builder.createBlock(&body);
  builder.setInsertionPointToEnd(swallowBlock);
  mlir::cf::CondBranchOp::create(builder, loc, isExit.getResult(0),
                                 discardBlock, mlir::ValueRange{},
                                 rethrowBlock, mlir::ValueRange{});
  builder.setInsertionPointToEnd(discardBlock);
  mlir::func::CallOp::create(
      builder, loc, getOrCreateDiscardCurrentException(module, builder),
      mlir::ValueRange{});
  mlir::cf::BranchOp::create(builder, loc, markClosed);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::func::CallOp::create(builder, loc,
                             getOrCreateRethrowCurrent(module, builder),
                             mlir::ValueRange{});
  mlir::Block *deadBlock2 = builder.createBlock(&body);
  builder.setInsertionPointToEnd(rethrowBlock);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);
  builder.setInsertionPointToEnd(deadBlock2);
  mlir::cf::BranchOp::create(builder, loc, deadBlock2);

  builder.setInsertionPointToEnd(markClosed);
  mlir::memref::StoreOp::create(builder, loc, i64Const(4), generator,
                                slotIndex(2));
  mlir::cf::BranchOp::create(builder, loc, returnBlock);
  builder.setInsertionPointToEnd(returnBlock);
  mlir::func::ReturnOp::create(builder, loc);
  return function;
}

// Resume over a state-machine generator: collect the generator storage and
// argument lanes and call the synthesized step/advance driver.
mlir::FailureOr<RuntimeBundleLowerer::SourceGeneratorResumeResult>
RuntimeBundleLowerer::emitStateMachineGeneratorResume(
    mlir::Operation *op, const RuntimeBundle &iterator,
    GeneratorResumeInfo &info, bool useCurrentInsertionPoint,
    std::optional<RuntimePrimitiveI64Evidence> sentI64Evidence,
    bool raiseWhenExhausted) {
  if (iterator.physicalValues().empty())
    return op->emitError() << "generator object has no physical storage";
  mlir::Value generator = iterator.physicalValues().front();
  if (iterator.generatorSourceBundles.size() != info.argumentCount)
    return op->emitError()
           << "generator frame source count does not match the resume clone";

  mlir::FailureOr<mlir::func::FuncOp> driver =
      raiseWhenExhausted
          ? RuntimeBundleLowerer::getOrCreateGeneratorAdvanceFunction(op, info)
          : RuntimeBundleLowerer::getOrCreateGeneratorStepFunction(op, info);
  if (mlir::failed(driver))
    return mlir::failure();

  if (!useCurrentInsertionPoint)
    builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(generator);
  for (const std::shared_ptr<RuntimeBundle> &source :
       iterator.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op->emitError() << "state-machine generator frame sources must "
                                "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  if (sentI64Evidence) {
    operands.push_back(sentI64Evidence->value);
    operands.push_back(sentI64Evidence->valid);
  } else {
    operands.push_back(
        mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult());
    operands.push_back(constantI1(builder, loc, false));
  }
  operands.push_back(
      mlir::arith::ConstantIntOp::create(builder, loc, 0, 64).getResult());
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, *driver, operands);
  if (raiseWhenExhausted)
    return SourceGeneratorResumeResult{call.getResult(0), call.getResult(1),
                                       constantI1(builder, loc, true)};
  return SourceGeneratorResumeResult{call.getResult(1), call.getResult(2),
                                     call.getResult(0)};
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStateMachineGeneratorThrow(
    py::CallOp op, const RuntimeBundle &receiver, GeneratorResumeInfo &info,
    llvm::ArrayRef<const RuntimeBundle *> sources) {
  if (sources.size() != 2 || !sources[1])
    return op.emitError()
           << "generator throw expects exactly one exception value";
  if (op.getNumResults() != 1 ||
      runtimeContractName(op.getResult(0).getType()) != "builtins.int")
    return op.emitError()
           << "generator throw currently supports int yield results";
  const RuntimeBundle &exception = *sources[1];
  if (exception.physicalValues().size() != 3)
    return op.emitError()
           << "generator throw currently requires a builtin exception "
              "instance (header, message, bytes), got "
           << exception.contractName();
  if (receiver.physicalValues().empty())
    return op.emitError() << "generator object has no physical storage";

  mlir::FailureOr<mlir::func::FuncOp> throwFn =
      RuntimeBundleLowerer::getOrCreateGeneratorThrowFunction(
          op.getOperation(), info);
  if (mlir::failed(throwFn))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  if (mlir::failed(emitTracebackFrame(op.getOperation())))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(receiver.physicalValues().front());
  if (receiver.generatorSourceBundles.size() != info.argumentCount)
    return op.emitError()
           << "generator frame source count does not match the resume clone";
  for (const std::shared_ptr<RuntimeBundle> &source :
       receiver.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op.emitError() << "state-machine generator frame sources must "
                               "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  for (mlir::Value value : exception.physicalValues())
    operands.push_back(value);
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, *throwFn, operands);

  RuntimeBundle result;
  if (mlir::failed(RuntimeBundleLowerer::makePrimitiveI64Bundle(
          op, op.getResult(0).getType(), call.getResult(0), call.getResult(1),
          result)))
    return mlir::failure();
  valueBundles[op.getResult(0)] = std::move(result);
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerStateMachineGeneratorClose(
    py::CallOp op, const RuntimeBundle &receiver, GeneratorResumeInfo &info) {
  if (op.getNumResults() != 1)
    return op.emitError() << "generator close expects one (None) result";
  if (receiver.physicalValues().empty())
    return op.emitError() << "generator object has no physical storage";
  mlir::FailureOr<mlir::func::FuncOp> closeFn =
      RuntimeBundleLowerer::getOrCreateGeneratorCloseFunction(op.getOperation(),
                                                              info);
  if (mlir::failed(closeFn))
    return mlir::failure();

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  llvm::SmallVector<mlir::Value, 16> operands;
  operands.push_back(receiver.physicalValues().front());
  if (receiver.generatorSourceBundles.size() != info.argumentCount)
    return op.emitError()
           << "generator frame source count does not match the resume clone";
  for (const std::shared_ptr<RuntimeBundle> &source :
       receiver.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op.emitError() << "state-machine generator frame sources must "
                               "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  emitTryCallSiteMarkerIfNeeded(loc);
  mlir::func::CallOp::create(builder, loc, *closeFn, operands);
  if (mlir::failed(assignObjectBundle(
          op.getOperation(), op.getResult(0),
          runtimeContractType(context, "types.NoneType"), mlir::ValueRange{})))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
