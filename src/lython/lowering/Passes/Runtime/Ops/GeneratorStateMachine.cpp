#include "Runtime/Core/Lowerer.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

// Loop-body generators: transform an int-pure generator body into a
// primitive-i64 RESUME state machine (pure SSA — no in-body memory):
//
//   @body(args: int...)                     [py CFG with py.yield_value]
//     ==>
//   @body__lyrt_gen_resume(args..., state: int, lives: int x K)
//       -> (state': int, has: int, value: int, lives': int x K)
//
// Every yield splits its block; the continuation's externally-defined values
// become block arguments (the live set, all builtins.int in this tier), the
// yield becomes `return (k, 1, value, lives...)`, and a raw-i64 dispatch
// chain in the entry branches on the state argument. The generator object
// stores state (word 4) and the live slots (words 8..) inline; `__next__`
// loads them, calls the resume clone through the primitive ABI, and stores
// them back. Reuses: the primitive-i64 clone entry/return ABI, primitive
// evidence, and the unit generator block — no runtime wrappers.

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
// else must ride in the frame).
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
    llvm::SmallVector<py::YieldValueOp, 8> yields;
    bool unsupported = false;
    body.walk([&](mlir::Operation *op) {
      if (auto yield = mlir::dyn_cast<py::YieldValueOp>(op)) {
        yields.push_back(yield);
        return;
      }
      if (mlir::isa<py::YieldFromOp>(op) ||
          (op != body.getOperation() && op->getNumRegions() != 0))
        unsupported = true;
    });
    if (unsupported || yields.empty())
      continue;
    mlir::Block &entry = body.getBody().front();
    // Straight-line bodies keep the existing inline dispatch.
    if (llvm::all_of(yields, [&](py::YieldValueOp yield) {
          return yield->getBlock() == &entry;
        }))
      continue;

    auto callableAttr =
        body->getAttrOfType<mlir::TypeAttr>(ownership::kCallableTypeAttr);
    auto callable = mlir::dyn_cast_if_present<py::CallableType>(
        callableAttr ? callableAttr.getValue() : mlir::Type());
    if (!callable || callable.hasVararg() || callable.hasKwarg())
      continue;
    bool eligible = llvm::all_of(callable.getPositionalTypes(), isIntContract);
    for (py::YieldValueOp yield : yields) {
      if (!isIntContract(yield.getValue().getType()))
        eligible = false;
      if (!yield.getSent().use_empty())
        eligible = false;
    }
    if (!eligible)
      continue;

    // Frame width: max live-across-yield count, from backward liveness on
    // the whole body CFG (matches phase 2 exactly).
    unsigned frameWidth = 0;
    bool livesEligible = true;
    {
      auto liveIns = computeLiveIns(body.getBody());
      for (py::YieldValueOp yield : yields) {
        llvm::SetVector<mlir::Value> lives =
            liveAfterYield(yield, liveIns, &entry);
        for (mlir::Value live : lives)
          if (!isIntContract(live.getType()))
            livesEligible = false;
        frameWidth =
            std::max(frameWidth, static_cast<unsigned>(lives.size()));
      }
    }
    if (!livesEligible ||
        kGeneratorFrameSlotBase + frameWidth > kGeneratorFrameSlotLimit)
      continue;

    mlir::Type intContract = runtimeContractType(context, "builtins.int");
    llvm::SmallVector<mlir::Type, 8> argTypes(
        callable.getPositionalTypes().begin(),
        callable.getPositionalTypes().end());
    argTypes.push_back(intContract); // state
    for (unsigned slot = 0; slot < frameWidth; ++slot)
      argTypes.push_back(intContract);
    llvm::SmallVector<mlir::Type, 8> resultTypes;
    resultTypes.push_back(intContract); // state'
    resultTypes.push_back(intContract); // has-value
    resultTypes.push_back(intContract); // yielded value
    for (unsigned slot = 0; slot < frameWidth; ++slot)
      resultTypes.push_back(intContract);

    std::string cloneName = (body.getSymName() + "__lyrt_gen_resume").str();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(body);
    mlir::func::FuncOp clone = body.clone();
    clone.setSymName(cloneName);
    clone->removeAttr(kGeneratorBodyResultAttr);
    clone->removeAttr(kGeneratorPublicResultAttr);
    clone->setAttr(kPrimitiveI64CloneAttr,
                   builder.getStringAttr(body.getSymName()));
    auto newCallable = py::CallableType::get(
        context, argTypes, /*kwonly=*/{}, /*varargType=*/mlir::Type(),
        /*kwargsType=*/mlir::Type(), resultTypes);
    clone->setAttr(ownership::kCallableTypeAttr,
                   mlir::TypeAttr::get(newCallable));
    clone.setFunctionType(
        mlir::FunctionType::get(context, argTypes, resultTypes));
    mlir::Block &cloneEntry = clone.getBody().front();
    for (unsigned extra = 0; extra < 1 + frameWidth; ++extra)
      cloneEntry.addArgument(intContract, clone.getLoc());
    mlir::SymbolTable::setSymbolVisibility(
        clone, mlir::SymbolTable::Visibility::Private);
    builder.insert(clone);

    GeneratorResumeInfo info;
    info.cloneName = cloneName;
    info.frameWidth = frameWidth;
    info.argumentCount = static_cast<unsigned>(
        callable.getPositionalTypes().size());
    generatorResumeClones[body.getSymName().str()] = info;
  }
  return mlir::success();
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
    unsigned logicalCount = argumentCount + 1 + frameWidth;
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
    mlir::Value rawState = rawOf(argumentCount);

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

    // Exhausted / falling-off-the-end returns.
    for (mlir::func::ReturnOp ret : originalReturns) {
      mlir::OpBuilder b(ret);
      llvm::SmallVector<mlir::Value, 8> operands;
      operands.push_back(intLiteral(b, doneState));
      operands.push_back(intLiteral(b, 0));
      operands.push_back(intLiteral(b, 0));
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
      // The suspend: return (state, 1, value, lives...). External live
      // operands are rethreaded by the normalization pass below.
      {
        mlir::OpBuilder b = mlir::OpBuilder::atBlockEnd(block);
        llvm::SmallVector<mlir::Value, 8> operands;
        operands.push_back(intLiteral(b, state));
        operands.push_back(intLiteral(b, 1));
        operands.push_back(yield.getValue());
        for (mlir::Value live : liveValues)
          operands.push_back(live);
        for (unsigned slot = static_cast<unsigned>(liveValues.size());
             slot < frameWidth; ++slot)
          operands.push_back(intLiteral(b, 0));
        mlir::func::ReturnOp::create(b, loc, operands);
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
        destOperands.push_back(
            entryBlock.getArgument(argumentCount + 1 + position));
      for (unsigned position = 0; position < continuation.liveCount;
           ++position) {
        destOperands.push_back(rawOf(argumentCount + 1 + position));
        destOperands.push_back(validOf(argumentCount + 1 + position));
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
      operands.push_back(intLiteral(b, 0));
      operands.push_back(intLiteral(b, 0));
      for (unsigned slot = 0; slot < frameWidth; ++slot)
        operands.push_back(intLiteral(b, 0));
      mlir::func::ReturnOp::create(b, loc, operands);
    }
  }
  return mlir::success();
}

// __next__ over a state-machine generator: load state + live slots from the
// generator block, call the resume clone through the primitive ABI, store
// them back, and hand the (value, has-value) pair to the shared next tail.
mlir::FailureOr<RuntimeBundleLowerer::SourceGeneratorResumeResult>
RuntimeBundleLowerer::emitStateMachineGeneratorResume(
    mlir::Operation *op, const RuntimeBundle &iterator,
    const GeneratorResumeInfo &info, bool useCurrentInsertionPoint) {
  mlir::func::FuncOp clone =
      module.lookupSymbol<mlir::func::FuncOp>(info.cloneName);
  if (!clone)
    return op->emitError() << "generator resume clone " << info.cloneName
                           << " is not defined";
  if (iterator.physicalValues().empty())
    return op->emitError() << "generator object has no physical storage";
  mlir::Value generator = iterator.physicalValues().front();

  if (!useCurrentInsertionPoint)
    builder.setInsertionPoint(op);
  mlir::Location loc = op->getLoc();
  mlir::Value trueValue =
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
  auto slotIndex = [&](std::int64_t slot) {
    return mlir::arith::ConstantIndexOp::create(builder, loc, slot)
        .getResult();
  };
  mlir::Value state = mlir::memref::LoadOp::create(builder, loc, generator,
                                                   slotIndex(4))
                          .getResult();
  // Lifecycle gate (word 2: 3 = exhausted, 4 = closed): a finished generator
  // must not run body code. Feeding the clone an out-of-range state routes it
  // to the pure exhausted return, so no branch is needed.
  mlir::Value lifecycle = mlir::memref::LoadOp::create(builder, loc, generator,
                                                       slotIndex(2))
                              .getResult();
  mlir::Value exhaustedLifecycle =
      mlir::arith::ConstantIntOp::create(builder, loc, 3, 64);
  mlir::Value finished = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, lifecycle,
      exhaustedLifecycle);
  mlir::Value sentinel =
      mlir::arith::ConstantIntOp::create(builder, loc, -1, 64);
  state = mlir::arith::SelectOp::create(builder, loc, finished, sentinel,
                                        state)
              .getResult();

  llvm::SmallVector<mlir::Value, 16> operands;
  if (iterator.generatorSourceBundles.size() != info.argumentCount)
    return op->emitError()
           << "generator frame source count does not match the resume clone";
  for (const std::shared_ptr<RuntimeBundle> &source :
       iterator.generatorSourceBundles) {
    if (!source || !source->primitiveI64)
      return op->emitError() << "state-machine generator frame sources must "
                                "carry primitive int evidence";
    operands.push_back(source->primitiveI64->value);
    operands.push_back(source->primitiveI64->valid);
  }
  operands.push_back(state);
  operands.push_back(trueValue);
  for (unsigned slot = 0; slot < info.frameWidth; ++slot) {
    mlir::Value live = mlir::memref::LoadOp::create(
                           builder, loc, generator,
                           slotIndex(kGeneratorFrameSlotBase + slot))
                           .getResult();
    operands.push_back(live);
    operands.push_back(trueValue);
  }

  mlir::func::CallOp call =
      mlir::func::CallOp::create(builder, loc, clone, operands);
  unsigned expectedResults = 2 * (3 + info.frameWidth);
  if (call.getNumResults() != expectedResults)
    return op->emitError() << "generator resume clone ABI mismatch";
  mlir::Value newState = call.getResult(0);
  mlir::Value hasRaw = call.getResult(2);
  mlir::Value value = call.getResult(4);
  mlir::Value valueValid = call.getResult(5);
  mlir::memref::StoreOp::create(builder, loc, newState, generator,
                                slotIndex(4));
  for (unsigned slot = 0; slot < info.frameWidth; ++slot)
    mlir::memref::StoreOp::create(builder, loc,
                                  call.getResult(6 + 2 * slot), generator,
                                  slotIndex(kGeneratorFrameSlotBase + slot));
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value hasValue = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::ne, hasRaw, zero);
  // Keep the lifecycle word coherent: suspended while values flow, exhausted
  // when the body finishes; a closed generator keeps its closed state.
  mlir::Value suspendedLifecycle =
      mlir::arith::ConstantIntOp::create(builder, loc, 2, 64);
  mlir::Value doneLifecycle =
      mlir::arith::SelectOp::create(builder, loc, finished, lifecycle,
                                    exhaustedLifecycle)
          .getResult();
  mlir::Value nextLifecycle =
      mlir::arith::SelectOp::create(builder, loc, hasValue,
                                    suspendedLifecycle, doneLifecycle)
          .getResult();
  mlir::memref::StoreOp::create(builder, loc, nextLifecycle, generator,
                                slotIndex(2));
  return SourceGeneratorResumeResult{value, valueValid, hasValue};
}

} // namespace py::lowering
