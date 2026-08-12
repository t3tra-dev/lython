#include "Runtime/Core/Lowerer.h"

#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/Support/Process.h"

#include <functional>
#include <optional>

namespace py::lowering {
namespace {

// TEMPORARY INSTRUMENT (family E attribution). Prints, per block, the argument
// arity and every predecessor edge's forwarded-operand count, so a mismatch can
// be attributed to a STEP of lowerModule rather than to the pass as a whole.
bool cfArityTraceEnabled() {
  static const bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_TRACE_CF_ARITY");
    return value && !value->empty() && *value != "0";
  }();
  return enabled;
}

void traceControlFlowArity(mlir::ModuleOp module, llvm::StringRef stage) {
  if (!cfArityTraceEnabled())
    return;
  module.walk([&](mlir::func::FuncOp function) {
    if (function.isDeclaration())
      return;
    unsigned blockIndex = 0;
    for (mlir::Block &block : function.getBody()) {
      unsigned index = blockIndex++;
      if (block.getNumArguments() == 0)
        continue;
      llvm::errs() << "[cf-arity:" << stage << "] @" << function.getName()
                   << " ^bb" << index << " args=" << block.getNumArguments()
                   << " types=[";
      llvm::interleaveComma(block.getArgumentTypes(), llvm::errs());
      llvm::errs() << "]";
      for (mlir::Block *predecessor : block.getPredecessors()) {
        auto branch =
            mlir::dyn_cast<mlir::BranchOpInterface>(predecessor->getTerminator());
        if (!branch)
          continue;
        for (unsigned s = 0, e = branch->getNumSuccessors(); s < e; ++s) {
          if (branch->getSuccessor(s) != &block)
            continue;
          llvm::errs() << " edge(" << branch->getName() << ",#" << s
                       << ")=" << branch.getSuccessorOperands(s).size();
        }
      }
      llvm::errs() << "\n";
    }
  });
}

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

  // ⚠️ THE INDEX SPACES DIVERGE HERE, and this is where the split has to happen.
  //
  // The edge work below indexes a predecessor's successor-operand list with
  // `argument.getArgNumber()`, i.e. with a BLOCK-argument index. Those two agree
  // only while every block argument before this one is forwarded on that edge --
  // and the physical arguments inserted just above are NOT forwarded yet.
  //
  // Splicing re-enters the lowerer (see `appendPhysicalBranchOperands`), so a
  // sibling logical argument of THIS block can arrive here while an outer
  // expansion still has its own physical arguments unforwarded. Measured on
  //
  //     xs: list[int] = [0]
  //     for i in range(n):
  //         try: total += 1
  //         except ValueError: total += 2
  //         xs = xs + [i]
  //         total += len(xs)
  //
  // where the int accumulator's back-edge operand reads `len(xs)`, so expanding
  // the int reaches the list. The list's block index was 4 against an operand
  // list of 2, `findPendingEdge` read that as "no edge carries this argument",
  // the loop ran ZERO rounds, and the function returned success() having left a
  // `memref<9xi64>` block argument that no branch forwards:
  // `branch has 3 operands for successor #0, but target block has 4`.
  //
  // Why NOT treat `index >= operands.size()` as an error there instead: it is a
  // legitimate answer for an edge that genuinely does not carry the argument.
  // The bug is the question, not the answer.
  //
  // LYTHON_ABLATE_CF_EXPANSION_DEFERRAL=1 restores the one-shot behaviour, so the
  // arity failure and its repair come from ONE binary. Why an ablation switch and
  // not two builds: a separate `before` build re-proves the build rather than the
  // change, and this repair is meant to alter generated IR only where an edge was
  // previously left unforwarded.
  static const bool ablateDeferral = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_CF_EXPANSION_DEFERRAL");
    return value && !value->empty() && *value != "0";
  }();
  // ⚠️ THE DIRECTION MATTERS, and deferring on any in-flight sibling is WRONG --
  // measured, by this repair breaking a program that worked:
  //
  //     s: str = "a"        # block argument 0
  //     for i in range(n):  # `total` is block argument 1
  //         try: total += 1
  //         except ValueError: total += 2
  //         s = s + "b"
  //         total += len(s)
  //
  // compiled and printed 12 before, and failed with `branch has 2 operands for
  // successor #0, but target block has 5` once every in-flight sibling deferred.
  //
  // Here the OUTER expansion is `total` at index 1 and the nested one is `s` at
  // index 0. Splicing `s` immediately inserts its operands AHEAD of `total`'s
  // logical operand on every edge, which shifts the edge index by exactly what
  // inserting `s`'s block arguments shifted the block index -- so the outer's
  // indices stay aligned and it proceeds. Deferring `s` shifts only the block
  // side, and the OUTER is the one that then silently forwards nothing.
  //
  // So the condition is not "a sibling is in flight" but "a sibling in flight
  // sits BEFORE me", which is exactly when its unforwarded physical arguments are
  // counted by my block index and missing from my edge index. A sibling after me
  // contributes nothing to either.
  bool precedingSiblingInFlight = false;
  for (mlir::BlockArgument sibling : block->getArguments())
    if (sibling != argument &&
        sibling.getArgNumber() < argument.getArgNumber() &&
        controlFlowBlockArgumentsInProgress.contains(sibling))
      precedingSiblingInFlight = true;
  bool siblingInFlight = precedingSiblingInFlight && !ablateDeferral;

  if (controlFlowLogicalBlockArgumentSet.insert(argument).second)
    controlFlowLogicalBlockArguments.push_back(
        ControlFlowLogicalBlockArgumentABI{argument});

  if (cfArityTraceEnabled())
    llvm::errs() << "[cf-begin] arg#" << argument.getArgNumber() << " type "
                 << argument.getType()
                 << (siblingInFlight ? " DEFERRED" : " immediate") << "\n";

  if (siblingInFlight) {
    controlFlowDeferredExpansions.push_back(ControlFlowDeferredExpansion{
        argument, *physicalTypes, primitiveIntLane, op});
    controlFlowBlockArgumentsInProgress.erase(argument);
    return mlir::success();
  }

  mlir::LogicalResult spliced =
      RuntimeBundleLowerer::spliceControlFlowBlockArgumentEdges(
          op, argument, *physicalTypes, primitiveIntLane);
  controlFlowBlockArgumentsInProgress.erase(argument);
  if (mlir::failed(spliced))
    return mlir::failure();
  return RuntimeBundleLowerer::drainDeferredControlFlowExpansions();
}

// The edge half of one logical block argument's expansion: materialize each
// incoming edge's physical operands and splice them in behind the logical one,
// then reconcile the arms' compile-time evidence.
//
// Separate from the half above because it can be DEFERRED -- see the note at the
// split. The caller owns `controlFlowBlockArgumentsInProgress` for `argument`
// across this call, so nothing here inserts or erases it.
mlir::LogicalResult RuntimeBundleLowerer::spliceControlFlowBlockArgumentEdges(
    mlir::Operation *op, mlir::BlockArgument argument,
    llvm::ArrayRef<mlir::Type> physicalTypes, bool primitiveIntLane) {
  mlir::Block *block = argument.getOwner();

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
                   anchor, *source, physicalTypes, physicalOperands))) {
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
        // ⛔ KNOWN DEFECT, measured: the edge operand is taken at the block
    // argument's OWN index, which holds only while the block and its edges
    // are expanded in lockstep. A generator whose for-loop precedes a yield
    //
    //     def f(n: int) -> Iterator[int]:
    //         total = 0
    //         for i in range(n):
    //             total = total + i
    //         yield total
    //
    // reaches here for a logical `builtins.int` argument and finds a
    // `memref<5xi64>` (the range iterator's physical) at that index --
    // "control-flow branch operand has no lowered runtime bundle", because a
    // physical value has no bundle to find. The resume clone's continuation
    // blocks get their arguments from the state machine, not from this
    // expansion, so the two sides are not appended in the same order. The
    // same shape with the yield INSIDE the loop, and the same loop in a
    // non-generator function, are both fine.
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
  unsigned spliced = 0;
  auto traceExit = [&](const char *why) {
    if (!cfArityTraceEnabled())
      return;
    llvm::errs() << "[cf-expand] arg#" << argument.getArgNumber() << " type "
                 << argument.getType() << " physTypes=" << physicalTypes.size()
                 << " rounds=" << rounds << " spliced=" << spliced << " exit="
                 << why << "\n";
  };
  while (std::optional<PendingEdge> pending = findPendingEdge(std::nullopt)) {
    if (++rounds > maxRounds) {
      return op->emitError()
             << "control-flow block argument expansion did not converge";
    }
    llvm::SmallVector<mlir::Value, 8> physicalOperands;
    if (mlir::failed(appendPhysicalBranchOperands(
            pending->predecessor, pending->logicalSource, physicalOperands))) {
      traceExit("append-failed");
      return mlir::failure();
    }
    if (physicalOperands.empty()) {
      traceExit("empty-physical-operands");
      break; // Nothing to forward for this argument at all.
    }

    // Re-locate the edge: the expansion above may have rebuilt this
    // terminator or split this predecessor.
    std::optional<PendingEdge> target = findPendingEdge(pending->logicalSource);
    if (!target) {
      traceExit("edge-vanished-continue");
      continue;
    }
    if (target->predecessor != pending->predecessor) {
      traceExit("edge-moved-continue");
      continue; // Moved to another block: re-expand there, where it dominates.
    }
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
    ++spliced;
  }
  traceExit("loop-drained");

  if (!sourceBundles.empty() &&
      llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
        return sameControlFlowEvidenceIdentity(sourceBundles.front(),
                                               candidate);
      })) {
    RuntimeBundle &merged = valueBundles[argument];
    // ⭐ The block argument keeps its OWN primitive lane. copyEvidenceFrom
    // assigns the source's (`primitiveI64 = source.primitiveI64`), and the
    // source's lane is an SSA pair defined in a PREDECESSOR -- for a merge
    // it does not even dominate here. Inside a primitive-i64 clone the lane
    // is the whole carrier for an int, so importing an absent one erased it:
    // a generator's `for i in range(n): yield i` reached the yield with an
    // int bundle holding neither physicals nor a lane ("generator int yield
    // lane has neither physical values nor primitive evidence"), and so did
    // any for-loop preceding a yield in the same body.
    //
    // ⛔ Why NOT drop the import entirely: a destination with no lane of its
    // own has nothing to lose, and that is where the transfer was doing its
    // work. Only a lane the ABI already created for this argument is
    // protected.
    std::optional<RuntimePrimitiveI64Evidence> ownLane = merged.primitiveI64;
    merged.copyEvidenceFrom(sourceBundles.front());
    if (ownLane)
      merged.primitiveI64 = ownLane;
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

  // Same relation, from the arms as they were actually spliced. The pre-splice
  // seed can only read predecessors that were already lowered, so this catches
  // the arms that were not -- and it is skipped when the seed already found the
  // relation, which is the case that matters for a loop.
  //
  // Neither of the two belongs in the evidence merge above: that one copies only
  // when every arm forwards the SAME SSA values, and a back edge forwards the
  // reallocated ones by construction. "Which storage names this entity" is a
  // statement about the owner value, not about the lanes.
  if (RuntimeBundle &merged = valueBundles[argument];
      !merged.fieldAliasOwner && !sourceBundles.empty()) {
    const RuntimeBundle &first = sourceBundles.front();
    if (first.fieldAliasOwner && !first.fieldAliasName.empty() &&
        llvm::all_of(sourceBundles, [&](const RuntimeBundle &candidate) {
          return candidate.fieldAliasOwner == first.fieldAliasOwner &&
                 candidate.fieldAliasName == first.fieldAliasName;
        })) {
      merged.fieldAliasOwner = first.fieldAliasOwner;
      merged.fieldAliasName = first.fieldAliasName;
    }
  }

  return mlir::success();
}

// Finish every expansion that deferred its edges because a sibling of the same
// block was mid-expansion, LOWEST ARGUMENT NUMBER FIRST.
//
// The order is the correctness argument, not a preference. `getArgNumber()` is an
// index into the BLOCK's argument list and it is used as an index into a
// predecessor's SUCCESSOR-OPERAND list; those two agree exactly when every block
// argument before this one is already forwarded on that edge. Ascending order
// establishes that, because the only arguments still missing from an edge are the
// deferred ones and they are drained in the order the block lists them.
//
// Why NOT compute the operand index instead of ordering the work: an exact
// computation needs, per edge, which expansions have already spliced it -- so it
// needs per-edge state that the one-shot local `splicedExpansions` set does not
// carry across a re-entrant call. Ordering makes the same fact true by
// construction and states it in one sentence.
//
// Why NOT pre-lower the siblings before starting an expansion, which removes the
// re-entrancy instead of tolerating it: a sibling's own edge computation can reach
// back to this argument, and `controlFlowBlockArgumentsInProgress` turns that into
// "cyclic Python control-flow block argument ABI is not implemented yet" -- a
// refusal of a program that works today.
mlir::LogicalResult RuntimeBundleLowerer::drainDeferredControlFlowExpansions() {
  while (!controlFlowDeferredExpansions.empty()) {
    unsigned best = 0;
    for (unsigned index = 1, end = controlFlowDeferredExpansions.size();
         index < end; ++index) {
      const ControlFlowDeferredExpansion &candidate =
          controlFlowDeferredExpansions[index];
      const ControlFlowDeferredExpansion &incumbent =
          controlFlowDeferredExpansions[best];
      if (candidate.argument.getOwner() == incumbent.argument.getOwner() &&
          candidate.argument.getArgNumber() <
              incumbent.argument.getArgNumber())
        best = index;
    }
    ControlFlowDeferredExpansion pending = controlFlowDeferredExpansions[best];
    controlFlowDeferredExpansions.erase(controlFlowDeferredExpansions.begin() +
                                        best);
    if (cfArityTraceEnabled())
      llvm::errs() << "[cf-drain] arg#" << pending.argument.getArgNumber()
                   << " type " << pending.argument.getType() << ", "
                   << controlFlowDeferredExpansions.size() << " still deferred\n";
    controlFlowBlockArgumentsInProgress.insert(pending.argument);
    mlir::LogicalResult finished =
        RuntimeBundleLowerer::spliceControlFlowBlockArgumentEdges(
            pending.op, pending.argument, pending.physicalTypes,
            pending.primitiveIntLane);
    controlFlowBlockArgumentsInProgress.erase(pending.argument);
    if (mlir::failed(finished))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::dropControlFlowLogicalBranchOperands() {
  traceControlFlowArity(module, "before-drop-logical-operands");
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
  traceControlFlowArity(module, "after-drop-logical-operands");
  return mlir::success();
}

mlir::LogicalResult
RuntimeBundleLowerer::eraseControlFlowLogicalBlockArguments() {
  traceControlFlowArity(module, "before-erase-logical-args");
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
  traceControlFlowArity(module, "after-erase-logical-args");
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
