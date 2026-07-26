#include "Runtime/Core/Lowerer.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include <functional>
#include <optional>

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

bool sameRuntimeValueIdentityList(llvm::ArrayRef<RuntimeValue> lhs,
                                  llvm::ArrayRef<RuntimeValue> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs)) {
    if (l.contract != r.contract || l.ownership != r.ownership ||
        l.values.size() != r.values.size())
      return false;
    for (auto [lv, rv] : llvm::zip(l.values, r.values))
      if (lv != rv)
        return false;
  }
  return true;
}

bool sameBundleIdentityShallow(const std::shared_ptr<RuntimeBundle> &lhs,
                               const std::shared_ptr<RuntimeBundle> &rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs->literalText != rhs->literalText)
    return false;
  if (lhs->primitiveI64.has_value() != rhs->primitiveI64.has_value())
    return false;
  if (lhs->primitiveI64 &&
      (lhs->primitiveI64->value != rhs->primitiveI64->value ||
       lhs->primitiveI64->valid != rhs->primitiveI64->valid))
    return false;
  return lhs->objectValue.contract == rhs->objectValue.contract &&
         sameRuntimeValueIdentityList({lhs->objectValue}, {rhs->objectValue});
}

bool sameBundleIdentityList(
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> lhs,
    llvm::ArrayRef<std::shared_ptr<RuntimeBundle>> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [l, r] : llvm::zip(lhs, rhs))
    if (!sameBundleIdentityShallow(l, r))
      return false;
  return true;
}

bool sameSequenceEvidence(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  return lhs.sequenceEvidenceBacked == rhs.sequenceEvidenceBacked &&
         lhs.sequenceIndices == rhs.sequenceIndices &&
         sameRuntimeValueIdentityList(lhs.sequenceElements,
                                      rhs.sequenceElements) &&
         sameBundleIdentityList(lhs.sequenceElementBundles,
                                rhs.sequenceElementBundles);
}

bool sameMappingEvidence(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  return lhs.mappingEvidenceBacked == rhs.mappingEvidenceBacked &&
         lhs.mappingKeys == rhs.mappingKeys &&
         lhs.mappingPresent == rhs.mappingPresent &&
         sameRuntimeValueIdentityList(lhs.mappingValues, rhs.mappingValues) &&
         sameBundleIdentityList(lhs.mappingKeyBundles, rhs.mappingKeyBundles) &&
         sameBundleIdentityList(lhs.mappingValueBundles,
                                rhs.mappingValueBundles);
}

bool sameFieldEvidence(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  if (lhs.fieldBundles.size() != rhs.fieldBundles.size())
    return false;
  for (const auto &entry : lhs.fieldBundles) {
    auto other = rhs.fieldBundles.find(entry.getKey());
    if (other == rhs.fieldBundles.end() ||
        !sameBundleIdentityShallow(entry.getValue(), other->getValue()))
      return false;
  }
  return sameBundleIdentityShallow(lhs.boxedObject, rhs.boxedObject);
}

bool sameObjectEvidence(const RuntimeBundle &lhs, const RuntimeBundle &rhs) {
  if (lhs.objectEvidence.slots.size() != rhs.objectEvidence.slots.size())
    return false;
  for (const auto &entry : lhs.objectEvidence.slots) {
    const RuntimeValue *other = rhs.objectEvidence.slot(entry.getKey());
    if (!other ||
        !sameRuntimeValueIdentityList({entry.getValue()}, {*other}))
      return false;
  }
  if (lhs.objectEvidence.flags.size() != rhs.objectEvidence.flags.size())
    return false;
  for (const auto &flag : lhs.objectEvidence.flags)
    if (!rhs.objectEvidence.hasFlag(flag.getKey()))
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
  // Seed the INTERIOR-VIEW relation before the edges are spliced, not after.
  // Splicing re-enters the lowerer, and for a loop the back edge's operand is
  // produced by an op INSIDE the body — so that op is lowered against this
  // provisional bundle. A mutation there has to know whether it is holding a
  // view of someone's slot, and learning it from the post-splice merge is one
  // step too late: the growth was already emitted with the local rebound and
  // the slot left naming the freed array.
  //
  // Only already-lowered predecessors can be consulted (the back edge is by
  // definition not one yet), which is enough: the entry edge into a loop
  // carries the relation, and every arm that carries one must agree, or the
  // merged value is a view of two different slots and belongs to neither.
  {
    unsigned index = argument.getArgNumber();
    mlir::Value owner;
    std::string name;
    bool conflict = false;
    for (mlir::Block *predecessor : block->getPredecessors()) {
      auto branch =
          mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        continue;
      for (unsigned successor = 0, end = branch->getNumSuccessors();
           successor < end; ++successor) {
        if (branch->getSuccessor(successor) != block)
          continue;
        mlir::SuccessorOperands operands =
            branch.getSuccessorOperands(successor);
        if (operands.getProducedOperandCount() != 0 || index >= operands.size())
          continue;
        auto found = valueBundles.find(operands[index]);
        if (found == valueBundles.end() || !found->second.fieldAliasOwner ||
            found->second.fieldAliasName.empty())
          continue;
        if (owner && (owner != found->second.fieldAliasOwner ||
                      name != found->second.fieldAliasName)) {
          conflict = true;
          continue;
        }
        owner = found->second.fieldAliasOwner;
        name = found->second.fieldAliasName;
      }
    }
    if (owner && !conflict) {
      provisionalBundle.fieldAliasOwner = owner;
      provisionalBundle.fieldAliasName = name;
    }
  }
  valueBundles[argument] = std::move(provisionalBundle);

  // Bundles are copied by VALUE: nested block-argument lowering inserts into
  // valueBundles, and a rehash would dangle any held pointer.
  llvm::SmallVector<RuntimeBundle, 4> sourceBundles;

  auto appendPhysicalBranchOperands =
      [&](mlir::Block *predecessor, mlir::Value logicalSource,
          llvm::SmallVectorImpl<mlir::Value> &destOperands)
      -> mlir::LogicalResult {
    mlir::OpBuilder::InsertionGuard guard(builder);
    // The anchor is re-fetched from the predecessor around the bundle
    // computation instead of held across it: the computation can replace this
    // terminator (see the edge loop below), leaving a handle taken before the
    // call — and an insertion point aimed at it — pointing at freed memory.
    builder.setInsertionPoint(predecessor->getTerminator());
    if (mlir::failed(RuntimeBundleLowerer::ensureValueBundle(
            predecessor->getTerminator(), logicalSource)))
      return mlir::failure();
    mlir::Operation *anchor = predecessor->getTerminator();
    builder.setInsertionPoint(anchor);
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

  // Incoming edges are re-derived from the block before every single splice,
  // and each edge is updated by MUTATING the terminator's successor operands.
  // Both are needed because computing an edge's physical operands re-enters
  // the lowerer: the operand may itself be an unlowered block argument whose
  // own predecessor set contains THIS block (which rebuilds a terminator), and
  // a may-raise operand pulls in unwind insertion (which SPLITS a predecessor,
  // so the block that branches here afterwards is a different one). Anything
  // captured before that — a terminator handle, a predecessor pointer, an
  // operand list, an argument index — is stale, which is why nothing is.
  //
  // An edge counts as done once our expansion sits right behind the logical
  // operand it forwards: the logical operand is never replaced, only followed.
  // Physical operands are materialized PER EDGE (never shared between two
  // edges forwarding the same value) because a value created at one
  // predecessor's terminator need not dominate another's.
  llvm::SmallPtrSet<mlir::Value, 8> splicedExpansions;
  struct PendingEdge {
    mlir::Block *predecessor = nullptr;
    unsigned successor = 0;
    mlir::Value logicalSource;
  };
  auto findPendingEdge =
      [&](std::optional<mlir::Value> onlySource) -> std::optional<PendingEdge> {
    unsigned index = argument.getArgNumber();
    for (mlir::Block *predecessor : block->getPredecessors()) {
      auto branch =
          mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
      if (!branch)
        continue;
      for (unsigned successor = 0, end = branch->getNumSuccessors();
           successor < end; ++successor) {
        if (branch->getSuccessor(successor) != block)
          continue;
        mlir::SuccessorOperands operands =
            branch.getSuccessorOperands(successor);
        if (operands.getProducedOperandCount() != 0 || index >= operands.size())
          continue;
        mlir::Value logicalSource = operands[index];
        if (onlySource && logicalSource != *onlySource)
          continue;
        // Membership alone is conclusive: the set holds only values THIS
        // expansion spliced, and those are physically typed, so a logical
        // operand can never collide with one.
        if (index + 1 < operands.size() &&
            splicedExpansions.contains(operands[index + 1]))
          continue;
        return PendingEdge{predecessor, successor, logicalSource};
      }
    }
    return std::nullopt;
  };

  // Each round expands exactly one edge, then re-derives from the block: the
  // bound only stops a pathological non-converging graph from looping forever.
  unsigned rounds = 0;
  const unsigned maxRounds = 4096;
  while (std::optional<PendingEdge> pending = findPendingEdge(std::nullopt)) {
    if (++rounds > maxRounds) {
      controlFlowBlockArgumentsInProgress.erase(argument);
      return op->emitError()
             << "control-flow block argument expansion did not converge";
    }
    llvm::SmallVector<mlir::Value, 8> physicalOperands;
    if (mlir::failed(appendPhysicalBranchOperands(
            pending->predecessor, pending->logicalSource, physicalOperands))) {
      controlFlowBlockArgumentsInProgress.erase(argument);
      return mlir::failure();
    }
    if (physicalOperands.empty())
      break; // Nothing to forward for this argument at all.

    // Re-locate the edge: the expansion above may have rebuilt this
    // terminator or split this predecessor.
    std::optional<PendingEdge> target = findPendingEdge(pending->logicalSource);
    if (!target)
      continue;
    if (target->predecessor != pending->predecessor)
      continue; // Moved to another block: re-expand there, where it dominates.
    auto branch = mlir::cast<mlir::BranchOpInterface>(
        target->predecessor->getTerminator());
    mlir::SuccessorOperands operands =
        branch.getSuccessorOperands(target->successor);
    unsigned index = argument.getArgNumber();
    llvm::SmallVector<mlir::Value, 8> updated;
    updated.reserve(operands.size() + physicalOperands.size());
    for (unsigned position = 0, end = operands.size(); position < end;
         ++position)
      updated.push_back(operands[position]);
    insertValues(updated, index + 1, physicalOperands);
    operands.getMutableForwardedOperands().assign(updated);
    splicedExpansions.insert(physicalOperands.front());
  }

  if (!sourceBundles.empty() &&
      llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
        return sameControlFlowEvidenceIdentity(sourceBundles.front(),
                                               candidate);
      })) {
    RuntimeBundle &merged = valueBundles[argument];
    merged.copyEvidenceFrom(sourceBundles.front());
    // Same physical identity does not imply same compile-time knowledge: an
    // arm may have recorded element/field evidence whose SSA values the other
    // arm never defines. Keeping the first arm's version would answer that
    // arm's contents on every path (silent mis-execution) or reference
    // non-dominating values. The physical payload carries the shared truth,
    // so evidence groups that disagree between the arms are dropped and later
    // uses read through the runtime instead.
    auto allSourcesAgree = [&](auto same) {
      return llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
        return same(sourceBundles.front(), candidate);
      });
    };
    if (!allSourcesAgree(sameSequenceEvidence)) {
      merged.sequenceEvidenceBacked = false;
      merged.sequenceElements.clear();
      merged.sequenceElementBundles.clear();
      merged.sequenceIndices.clear();
    }
    if (!allSourcesAgree(sameMappingEvidence)) {
      merged.mappingEvidenceBacked = false;
      merged.mappingKeys.clear();
      merged.mappingKeyBundles.clear();
      merged.mappingValues.clear();
      merged.mappingValueBundles.clear();
      merged.mappingPresent.clear();
    }
    if (!allSourcesAgree(sameFieldEvidence)) {
      merged.fieldBundles.clear();
      merged.boxedObject.reset();
    }
    if (!allSourcesAgree(sameObjectEvidence))
      merged.objectEvidence = RuntimeObjectEvidence{};
  }

  // The INTERIOR-VIEW relation survives a merge even when the lane identities
  // do not, which the block above cannot express: it copies evidence only when
  // every arm forwards the same SSA values, and a loop's back edge forwards the
  // reallocated ones by construction. But "which storage names this entity" is
  // a statement about the owner value, not about the lanes — a preheader view
  // of `n.f` and a grown view of `n.f` are views of the same slot. Dropping it
  // left a loop-carried field alias with no way to publish a reallocation: the
  // local saw the new array and the field kept the freed one, and the growth
  // path could not even refuse, because it could no longer tell a loop-carried
  // FIELD view from a loop-carried local.
  if (!sourceBundles.empty()) {
    const RuntimeBundle &first = sourceBundles.front();
    if (first.fieldAliasOwner && !first.fieldAliasName.empty() &&
        llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
          return candidate.fieldAliasOwner == first.fieldAliasOwner &&
                 candidate.fieldAliasName == first.fieldAliasName;
        })) {
      RuntimeBundle &merged = valueBundles[argument];
      merged.fieldAliasOwner = first.fieldAliasOwner;
      merged.fieldAliasName = first.fieldAliasName;
    }
  }

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
