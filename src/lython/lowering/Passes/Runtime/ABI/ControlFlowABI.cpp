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
  if (!definition || !definition->getDialect() ||
      definition->getDialect()->getNamespace() != "py")
    return mlir::success();
  if (llvm::is_contained(erase, definition))
    return mlir::success();
  if (mlir::failed(RuntimeBundleLowerer::ensureOperationOperandBundles(
          definition)))
    return mlir::failure();
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

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> physicalTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(
          op, argument.getType(), "control-flow block argument ABI");
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
  if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
          op, argument.getType(), physicalArguments, provisionalBundle))) {
    controlFlowBlockArgumentsInProgress.erase(argument);
    return mlir::failure();
  }
  valueBundles[argument] = std::move(provisionalBundle);

  llvm::SmallVector<mlir::Block *, 8> predecessors(block->pred_begin(),
                                                   block->pred_end());
  llvm::SmallVector<const RuntimeBundle *, 4> sourceBundles;

  auto appendPhysicalBranchOperands =
      [&](mlir::Operation *anchor, mlir::Value logicalSource,
          llvm::SmallVectorImpl<mlir::Value> &destOperands)
      -> mlir::LogicalResult {
    if (mlir::failed(
            RuntimeBundleLowerer::ensureValueBundle(anchor, logicalSource)))
      return mlir::failure();
    const RuntimeBundle *source =
        RuntimeBundleLowerer::bundleFor(logicalSource);
    if (!source)
      return anchor->emitError()
             << "control-flow branch operand has no lowered runtime bundle";

    llvm::SmallVector<mlir::Value, 8> physicalOperands;
    if (auto unionType = mlir::dyn_cast<py::UnionType>(argument.getType())) {
      if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
              anchor, unionType, *source, logicalSource.getType(),
              physicalOperands)))
        return mlir::failure();
    } else if (mlir::failed(RuntimeBundleLowerer::appendBundlePhysicalOperands(
                   anchor, *source, *physicalTypes, physicalOperands))) {
      return mlir::failure();
    }

    destOperands.append(physicalOperands.begin(), physicalOperands.end());
    sourceBundles.push_back(source);
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
      if (mlir::failed(rewriteBranchOperands(
              terminator, branch.getDest(), branch.getDestOperands(),
              operands))) {
        controlFlowBlockArgumentsInProgress.erase(argument);
        return mlir::failure();
      }
      builder.setInsertionPoint(branch);
      builder.create<mlir::cf::BranchOp>(branch.getLoc(), branch.getDest(),
                                         operands);
      branch.erase();
      continue;
    }

    if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator)) {
      llvm::SmallVector<mlir::Value, 8> trueOperands;
      llvm::SmallVector<mlir::Value, 8> falseOperands;
      if (mlir::failed(rewriteBranchOperands(
              terminator, cond.getTrueDest(), cond.getTrueDestOperands(),
              trueOperands)) ||
          mlir::failed(rewriteBranchOperands(terminator, cond.getFalseDest(),
                                             cond.getFalseDestOperands(),
                                             falseOperands))) {
        controlFlowBlockArgumentsInProgress.erase(argument);
        return mlir::failure();
      }
      builder.setInsertionPoint(cond);
      builder.create<mlir::cf::CondBranchOp>(
          cond.getLoc(), cond.getCondition(), cond.getTrueDest(), trueOperands,
          cond.getFalseDest(), falseOperands);
      cond.erase();
      continue;
    }

    controlFlowBlockArgumentsInProgress.erase(argument);
    return op->emitError()
           << "Python control-flow block argument lowering only supports cf.br "
              "and cf.cond_br predecessors";
  }

  if (!sourceBundles.empty() &&
      llvm::all_of(sourceBundles, [&](const RuntimeBundle *candidate) {
        return candidate &&
               samePhysicalIdentity(*sourceBundles.front(), *candidate);
      }))
    valueBundles[argument].copyEvidenceFrom(*sourceBundles.front());

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
    llvm::SmallVector<mlir::Block *, 8> predecessors(block->pred_begin(),
                                                     block->pred_end());
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
        builder.create<mlir::cf::BranchOp>(branch.getLoc(), branch.getDest(),
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
        builder.create<mlir::cf::CondBranchOp>(
            cond.getLoc(), cond.getCondition(), cond.getTrueDest(),
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

} // namespace py::lowering
