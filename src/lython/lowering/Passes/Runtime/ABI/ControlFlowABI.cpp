#include "Runtime/Core/Lowerer.h"

#include <functional>

namespace py::lowering {
namespace {

bool hasRuntimeControlFlowABI(mlir::Type type) {
  if (mlir::isa<py::UnionType>(type))
    return true;
  return !runtimeShapeContractName(type).empty();
}

void insertValues(llvm::SmallVectorImpl<mlir::Value> &values, unsigned index,
                  mlir::ValueRange inserted) {
  values.insert(values.begin() + index, inserted.begin(), inserted.end());
}

void eraseValue(llvm::SmallVectorImpl<mlir::Value> &values, unsigned index) {
  values.erase(values.begin() + index);
}

bool samePhysicalIdentity(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  llvm::ArrayRef<mlir::Value> left = lhs.physicalValues();
  llvm::ArrayRef<mlir::Value> right = rhs.physicalValues();
  if (left.size() != right.size())
    return false;
  for (auto [l, r] : llvm::zip(left, right))
    if (l != r)
      return false;
  return true;
}

bool samePrimitiveI64EvidenceIdentity(const RuntimeBundle &lhs,
                                      const RuntimeBundle &rhs) {
  if (!lhs.primitiveI64 && !rhs.primitiveI64)
    return true;
  if (!lhs.primitiveI64 || !rhs.primitiveI64)
    return false;
  return lhs.primitiveI64->value == rhs.primitiveI64->value &&
         lhs.primitiveI64->valid == rhs.primitiveI64->valid;
}

bool sameControlFlowEvidenceIdentity(const RuntimeBundle &lhs,
                                     const RuntimeBundle &rhs) {
  return samePhysicalIdentity(lhs, rhs) &&
         samePrimitiveI64EvidenceIdentity(lhs, rhs);
}

} // namespace

mlir::LogicalResult RuntimeBundleLowerer::ensureValueBundle(mlir::Operation *op,
                                                            mlir::Value value) {
  if (valueBundles.find(value) != valueBundles.end())
    return mlir::success();

  auto argument = mlir::dyn_cast<mlir::BlockArgument>(value);
  if (argument) {
    if (!hasRuntimeControlFlowABI(argument.getType()))
      return mlir::success();
    return RuntimeBundleLowerer::lowerControlFlowBlockArgument(op, argument);
  }

  mlir::Operation *definition = value.getDefiningOp();
  // The canonicalizer folds branch diamonds over already-computed values into
  // arith.select on the logical type: lower it as a borrow of both sides.
  if (auto select = mlir::dyn_cast_if_present<mlir::arith::SelectOp>(definition))
    if (hasRuntimeControlFlowABI(value.getType()))
      return RuntimeBundleLowerer::lowerRuntimeValueSelect(select);
  if (!definition || !definition->getDialect() ||
      definition->getDialect()->getNamespace() != "py")
    return mlir::success();
  if (llvm::is_contained(erase, definition))
    return mlir::success();
  if (mlir::failed(
          RuntimeBundleLowerer::ensureOperationOperandBundles(definition)))
    return mlir::failure();
  if (valueBundles.find(value) != valueBundles.end() ||
      llvm::is_contained(erase, definition))
    return mlir::success();
  return RuntimeBundleLowerer::lowerPyOp(definition);
}

mlir::LogicalResult
RuntimeBundleLowerer::ensureOperationOperandBundles(mlir::Operation *op) {
  for (mlir::Value operand : op->getOperands())
    if (mlir::failed(RuntimeBundleLowerer::ensureValueBundle(op, operand)))
      return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::lowerControlFlowBlockArgument(
    mlir::Operation *op, mlir::BlockArgument argument) {
  if (valueBundles.find(argument) != valueBundles.end())
    return mlir::success();
  if (controlFlowBlockArgumentsInProgress.contains(argument))
    return op->emitError()
           << "cyclic Python control-flow block argument ABI is not "
              "implemented yet";
  if (!hasRuntimeControlFlowABI(argument.getType()))
    return mlir::success();

  // Inside primitive-i64 clones, int-typed merges stay in the primitive
  // lane: the block argument ABI is the (i64, valid) evidence pair, keeping
  // loop-carried ints unboxed (the boxed expansion would sever the evidence
  // and drag the whole loop onto the boxed path).
  auto enclosing = argument.getOwner()->getParentOp();
  bool primitiveIntLane =
      mlir::isa_and_nonnull<mlir::func::FuncOp>(enclosing) &&
      RuntimeBundleLowerer::isPrimitiveI64CallableClone(
          mlir::cast<mlir::func::FuncOp>(enclosing)) &&
      runtimeContractName(argument.getType()) == "builtins.int";

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> physicalTypes;
  if (primitiveIntLane) {
    llvm::SmallVector<mlir::Type, 8> pairTypes;
    pairTypes.push_back(mlir::IntegerType::get(context, 64));
    pairTypes.push_back(mlir::IntegerType::get(context, 1));
    physicalTypes = std::move(pairTypes);
  } else {
    physicalTypes = RuntimeBundleLowerer::runtimeValueTypesFor(
        op, argument.getType(), "control-flow block argument ABI");
  }
  if (mlir::failed(physicalTypes))
    return mlir::failure();

  controlFlowBlockArgumentsInProgress.insert(argument);

  mlir::Block *block = argument.getOwner();
  unsigned logicalIndex = argument.getArgNumber();
  llvm::SmallVector<mlir::Value, 8> physicalArguments;
  physicalArguments.reserve(physicalTypes->size());
  for (auto [offset, type] : llvm::enumerate(*physicalTypes)) {
    mlir::BlockArgument physical =
        block->insertArgument(logicalIndex + 1 + static_cast<unsigned>(offset),
                              type, argument.getLoc());
    physicalArguments.push_back(physical);
  }

  RuntimeBundle provisionalBundle;
  if (primitiveIntLane) {
    provisionalBundle = RuntimeBundle::objectWithOwnership(
        argument.getType(), mlir::ValueRange{},
        ownership::logicalOwnershipKind(argument.getType(),
                                        /*ownsObject=*/false));
    provisionalBundle.primitiveI64 = RuntimePrimitiveI64Evidence{
        physicalArguments[0], physicalArguments[1]};
  } else if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
                 op, argument.getType(), physicalArguments,
                 provisionalBundle))) {
    controlFlowBlockArgumentsInProgress.erase(argument);
    return mlir::failure();
  }
  valueBundles[argument] = std::move(provisionalBundle);

  // Deduplicate: a cond_br with BOTH successors == `block` (a canonicalized
  // empty-arm conditional) lists its predecessor twice, but one rewrite below
  // already handles both edges — visiting it again would corrupt the rebuilt
  // operand lists.
  llvm::SmallVector<mlir::Block *, 8> predecessors;
  {
    llvm::SmallPtrSet<mlir::Block *, 8> seenPredecessors;
    for (mlir::Block *predecessor : block->getPredecessors())
      if (seenPredecessors.insert(predecessor).second)
        predecessors.push_back(predecessor);
  }
  // Bundles are copied by VALUE: nested block-argument lowering inserts into
  // valueBundles, and a rehash would dangle any held pointer.
  llvm::SmallVector<RuntimeBundle, 4> sourceBundles;

  auto appendPhysicalBranchOperands =
      [&](mlir::Operation *anchor, mlir::Value logicalSource,
          llvm::SmallVectorImpl<mlir::Value> &destOperands)
      -> mlir::LogicalResult {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(anchor);
    if (mlir::failed(
            RuntimeBundleLowerer::ensureValueBundle(anchor, logicalSource)))
      return mlir::failure();
    const RuntimeBundle *source =
        RuntimeBundleLowerer::bundleFor(logicalSource);
    if (!source)
      return anchor->emitError()
             << "control-flow branch operand has no lowered runtime bundle";

    llvm::SmallVector<mlir::Value, 8> physicalOperands;
    if (primitiveIntLane) {
      if (source->primitiveI64) {
        physicalOperands.push_back(source->primitiveI64->value);
        physicalOperands.push_back(source->primitiveI64->valid);
      } else {
        // Boxed slow-path result rejoining the primitive lane: unbox.
        std::optional<RuntimeSymbol> unbox =
            manifest.primitive("builtins.int", "unbox.i64");
        if (!unbox ||
            unbox->function.getNumArguments() != source->physicalValues().size())
          return anchor->emitError()
                 << "primitive int merge source has neither evidence nor an "
                    "unboxable representation";
        mlir::func::CallOp call = RuntimeBundleLowerer::createRuntimeCall(
            anchor->getLoc(), *unbox, source->physicalValues());
        physicalOperands.push_back(call.getResult(0));
        physicalOperands.push_back(
            mlir::arith::ConstantIntOp::create(builder, anchor->getLoc(), 1, 1)
                .getResult());
      }
    } else if (auto unionType =
                   mlir::dyn_cast<py::UnionType>(argument.getType())) {
      if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
              anchor, unionType, *source, logicalSource.getType(),
              physicalOperands)))
        return mlir::failure();
    } else if (mlir::failed(RuntimeBundleLowerer::appendBundlePhysicalOperands(
                   anchor, *source, *physicalTypes, physicalOperands))) {
      return mlir::failure();
    }

    destOperands.append(physicalOperands.begin(), physicalOperands.end());
    sourceBundles.push_back(*source);
    return mlir::success();
  };

  auto rewriteBranchOperands =
      [&](mlir::Operation *terminator, mlir::Block *dest,
          mlir::ValueRange oldOperands,
          llvm::SmallVectorImpl<mlir::Value> &newOperands)
      -> mlir::LogicalResult {
    newOperands.append(oldOperands.begin(), oldOperands.end());
    if (dest != block)
      return mlir::success();
    if (logicalIndex >= newOperands.size())
      return op->emitError()
             << "control-flow predecessor operand list is shorter than the "
                "destination block argument list";

    llvm::SmallVector<mlir::Value, 8> physicalOperands;
    if (mlir::failed(appendPhysicalBranchOperands(
            terminator, newOperands[logicalIndex], physicalOperands)))
      return mlir::failure();
    insertValues(newOperands, logicalIndex + 1, physicalOperands);
    return mlir::success();
  };

  for (mlir::Block *predecessor : predecessors) {
    mlir::Operation *terminator = predecessor->getTerminator();
    if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
      llvm::SmallVector<mlir::Value, 8> operands;
      if (mlir::failed(rewriteBranchOperands(terminator, branch.getDest(),
                                             branch.getDestOperands(),
                                             operands))) {
        controlFlowBlockArgumentsInProgress.erase(argument);
        return mlir::failure();
      }
      builder.setInsertionPoint(branch);
      mlir::cf::BranchOp::create(builder, branch.getLoc(), branch.getDest(),
                                 operands);
      branch.erase();
      continue;
    }

    if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      llvm::SmallVector<mlir::Value, 8> trueOperands;
      llvm::SmallVector<mlir::Value, 8> falseOperands;
      if (mlir::failed(rewriteBranchOperands(terminator, cond.getTrueDest(),
                                             cond.getTrueDestOperands(),
                                             trueOperands)) ||
          mlir::failed(rewriteBranchOperands(terminator, cond.getFalseDest(),
                                             cond.getFalseDestOperands(),
                                             falseOperands))) {
        controlFlowBlockArgumentsInProgress.erase(argument);
        return mlir::failure();
      }
      builder.setInsertionPoint(cond);
      mlir::cf::CondBranchOp::create(
          builder, cond.getLoc(), cond.getCondition(), cond.getTrueDest(),
          trueOperands, cond.getFalseDest(), falseOperands);
      cond.erase();
      continue;
    }

    controlFlowBlockArgumentsInProgress.erase(argument);
    return op->emitError()
           << "Python control-flow block argument lowering only supports cf.br "
              "and cf.cond_br predecessors";
  }

  if (!sourceBundles.empty() &&
      llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
        return sameControlFlowEvidenceIdentity(sourceBundles.front(),
                                               candidate);
      }))
    valueBundles[argument].copyEvidenceFrom(sourceBundles.front());

  if (controlFlowLogicalBlockArgumentSet.insert(argument).second)
    controlFlowLogicalBlockArguments.push_back(
        ControlFlowLogicalBlockArgumentABI{argument});
  controlFlowBlockArgumentsInProgress.erase(argument);
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::dropControlFlowLogicalBranchOperands() {
  auto dropOperand = [&](mlir::Block *dest, mlir::ValueRange oldOperands,
                         unsigned index,
                         llvm::SmallVectorImpl<mlir::Value> &newOperands)
      -> mlir::LogicalResult {
    newOperands.append(oldOperands.begin(), oldOperands.end());
    if (index >= newOperands.size())
      return dest->getParentOp()->emitError()
             << "control-flow logical branch operand index is outside the "
                "predecessor operand list";
    eraseValue(newOperands, index);
    return mlir::success();
  };

  llvm::SmallVector<mlir::BlockArgument, 16> arguments;
  arguments.reserve(controlFlowLogicalBlockArguments.size());
  for (ControlFlowLogicalBlockArgumentABI abi :
       controlFlowLogicalBlockArguments)
    arguments.push_back(abi.argument);
  llvm::sort(arguments, [](mlir::BlockArgument lhs, mlir::BlockArgument rhs) {
    if (lhs.getOwner() != rhs.getOwner())
      return std::less<mlir::Block *>()(lhs.getOwner(), rhs.getOwner());
    return lhs.getArgNumber() > rhs.getArgNumber();
  });

  for (mlir::BlockArgument argument : arguments) {
    mlir::Block *block = argument.getOwner();
    unsigned logicalIndex = argument.getArgNumber();
    llvm::SmallVector<mlir::Block *, 8> predecessors;
    {
      // Deduplicate dual-edge predecessors (see the expansion loop above).
      llvm::SmallPtrSet<mlir::Block *, 8> seenPredecessors;
      for (mlir::Block *predecessor : block->getPredecessors())
        if (seenPredecessors.insert(predecessor).second)
          predecessors.push_back(predecessor);
    }
    for (mlir::Block *predecessor : predecessors) {
      mlir::Operation *terminator = predecessor->getTerminator();
      if (auto branch = mlir::dyn_cast<mlir::cf::BranchOp>(terminator)) {
        if (branch.getDest() != block)
          continue;
        llvm::SmallVector<mlir::Value, 8> operands;
        if (mlir::failed(dropOperand(block, branch.getDestOperands(),
                                     logicalIndex, operands)))
          return mlir::failure();
        builder.setInsertionPoint(branch);
        mlir::cf::BranchOp::create(builder, branch.getLoc(), branch.getDest(),
                                   operands);
        branch.erase();
        continue;
      }

      if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
        llvm::SmallVector<mlir::Value, 8> trueOperands;
        llvm::SmallVector<mlir::Value, 8> falseOperands;
        if (cond.getTrueDest() == block &&
            mlir::failed(dropOperand(block, cond.getTrueDestOperands(),
                                     logicalIndex, trueOperands)))
          return mlir::failure();
        if (cond.getTrueDest() != block)
          trueOperands.append(cond.getTrueDestOperands().begin(),
                              cond.getTrueDestOperands().end());
        if (cond.getFalseDest() == block &&
            mlir::failed(dropOperand(block, cond.getFalseDestOperands(),
                                     logicalIndex, falseOperands)))
          return mlir::failure();
        if (cond.getFalseDest() != block)
          falseOperands.append(cond.getFalseDestOperands().begin(),
                               cond.getFalseDestOperands().end());

        builder.setInsertionPoint(cond);
        mlir::cf::CondBranchOp::create(
            builder, cond.getLoc(), cond.getCondition(), cond.getTrueDest(),
            trueOperands, cond.getFalseDest(), falseOperands);
        cond.erase();
        continue;
      }

      return terminator->emitError()
             << "cannot drop Python logical branch operand from unsupported "
                "terminator";
    }
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::eraseControlFlowLogicalBlockArguments() {
  llvm::SmallVector<mlir::BlockArgument, 16> arguments;
  arguments.reserve(controlFlowLogicalBlockArguments.size());
  for (ControlFlowLogicalBlockArgumentABI abi :
       controlFlowLogicalBlockArguments)
    arguments.push_back(abi.argument);
  llvm::sort(arguments, [](mlir::BlockArgument lhs, mlir::BlockArgument rhs) {
    if (lhs.getOwner() != rhs.getOwner())
      return std::less<mlir::Block *>()(lhs.getOwner(), rhs.getOwner());
    return lhs.getArgNumber() > rhs.getArgNumber();
  });

  for (mlir::BlockArgument argument : arguments) {
    if (!argument.use_empty())
      return argument.getOwner()->getParentOp()->emitError()
             << "control-flow logical block argument still has users after "
                "runtime lowering";
    argument.getOwner()->eraseArgument(argument.getArgNumber());
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::lowerRuntimeValueSelect(mlir::arith::SelectOp select) {
  mlir::Value result = select.getResult();
  if (valueBundles.find(result) != valueBundles.end())
    return mlir::success();
  if (mlir::isa<py::UnionType>(result.getType()))
    return select.emitError()
           << "select over a union-typed Python value is not supported yet";

  if (mlir::failed(RuntimeBundleLowerer::ensureValueBundle(
          select, select.getTrueValue())) ||
      mlir::failed(RuntimeBundleLowerer::ensureValueBundle(
          select, select.getFalseValue())))
    return mlir::failure();
  const RuntimeBundle *truePtr =
      RuntimeBundleLowerer::bundleFor(select.getTrueValue());
  const RuntimeBundle *falsePtr =
      RuntimeBundleLowerer::bundleFor(select.getFalseValue());
  if (!truePtr || !falsePtr)
    return select.emitError()
           << "select operand has no lowered runtime bundle";
  // Copies: binding the result below inserts into valueBundles and a rehash
  // would dangle the lookups.
  RuntimeBundle trueBundle = *truePtr;
  RuntimeBundle falseBundle = *falsePtr;

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> physicalTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(select, result.getType(),
                                                 "runtime value select");
  if (mlir::failed(physicalTypes))
    return mlir::failure();

  builder.setInsertionPoint(select);
  mlir::Location loc = select.getLoc();
  llvm::SmallVector<mlir::Value, 8> trueValues;
  llvm::SmallVector<mlir::Value, 8> falseValues;
  if (mlir::failed(RuntimeBundleLowerer::appendBundlePhysicalOperands(
          select, trueBundle, *physicalTypes, trueValues)) ||
      mlir::failed(RuntimeBundleLowerer::appendBundlePhysicalOperands(
          select, falseBundle, *physicalTypes, falseValues)))
    return mlir::failure();
  if (trueValues.size() != falseValues.size())
    return select.emitError()
           << "select operands lower to mismatched physical spans";

  llvm::SmallVector<mlir::Value, 8> picked;
  picked.reserve(trueValues.size());
  for (auto [trueValue, falseValue] : llvm::zip(trueValues, falseValues))
    picked.push_back(mlir::arith::SelectOp::create(
                         builder, loc, select.getCondition(), trueValue,
                         falseValue)
                         .getResult());

  RuntimeBundle bundle = RuntimeBundle::objectWithOwnership(
      result.getType(), picked,
      ownership::logicalOwnershipKind(result.getType(),
                                      /*ownsObject=*/false));
  if (trueBundle.primitiveI64 && falseBundle.primitiveI64) {
    mlir::Value value = mlir::arith::SelectOp::create(
                            builder, loc, select.getCondition(),
                            trueBundle.primitiveI64->value,
                            falseBundle.primitiveI64->value)
                            .getResult();
    mlir::Value valid = mlir::arith::SelectOp::create(
                            builder, loc, select.getCondition(),
                            trueBundle.primitiveI64->valid,
                            falseBundle.primitiveI64->valid)
                            .getResult();
    bundle.primitiveI64 = RuntimePrimitiveI64Evidence{value, valid};
  }
  valueBundles[result] = std::move(bundle);
  erase.push_back(select);
  return mlir::success();
}

} // namespace py::lowering
