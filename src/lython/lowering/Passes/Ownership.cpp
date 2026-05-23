#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

namespace py {
namespace {

using AliasAnalysis = OwnershipAliasAnalysis;

namespace root {

bool immortal(mlir::Value root, const AliasAnalysis &aliases) {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  aliases.collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    mlir::Operation *def = member.getDefiningOp();
    if (def && isPyOwnershipImmortalOp(def))
      return true;
  }
  return false;
}

bool aggregateBorrow(mlir::Value root, const AliasAnalysis &aliases) {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  aliases.collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    mlir::Operation *def = member.getDefiningOp();
    if (!def)
      continue;
    if (def->hasAttr(OwnershipContractAttrs::kAggregateSlotLoad))
      return true;
    auto role =
        def->getAttrOfType<mlir::StringAttr>(ThreadSafetyAttrs::kAtomicRole);
    if (role && role.getValue() == ThreadSafetyAttrs::kRoleAsyncExceptionLoad)
      return true;
  }
  return false;
}

bool same(mlir::Value lhs, mlir::Value rhs, const AliasAnalysis &aliases) {
  return aliases.getRoot(lhs) == aliases.getRoot(rhs);
}

} // namespace root

namespace operation {

bool refcount(mlir::Operation *op) { return mlir::isa<IncRefOp, DecRefOp>(op); }

bool mentionsRoot(mlir::Operation *op, mlir::Value root,
                  const AliasAnalysis &aliases) {
  for (mlir::Value operand : op->getOperands())
    if (isPyOwnershipTrackedType(operand.getType()) &&
        root::same(operand, root, aliases))
      return true;
  for (mlir::Value result : op->getResults())
    if (isPyOwnershipTrackedType(result.getType()) &&
        root::same(result, root, aliases))
      return true;
  return false;
}

bool hasOwnershipEffect(mlir::Operation *op) {
  if (createsPyOwnedResult(op) || mlir::isa<IncRefOp, DecRefOp>(op))
    return true;
  for (mlir::Value operand : op->getOperands())
    if (consumesPyOwnedOperand(op, operand))
      return true;
  return false;
}

} // namespace operation

namespace pair_elision {

bool canMoveAcross(mlir::Operation *op, mlir::Value root,
                   const AliasAnalysis &aliases) {
  if (operation::refcount(op))
    return false;
  if (operation::mentionsRoot(op, root, aliases))
    return false;
  if (op->hasTrait<mlir::OpTrait::IsTerminator>())
    return false;
  return true;
}

DecRefOp pairedDecRef(IncRefOp inc, const AliasAnalysis &aliases) {
  mlir::Value object = inc.getObject();
  mlir::Operation *cursor = inc->getNextNode();
  while (cursor) {
    if (auto dec = mlir::dyn_cast<DecRefOp>(cursor)) {
      if (root::same(dec.getObject(), object, aliases))
        return dec;
      return nullptr;
    }
    if (!canMoveAcross(cursor, object, aliases))
      return nullptr;
    cursor = cursor->getNextNode();
  }
  return nullptr;
}

bool run(mlir::Operation *funcLike, mlir::Region &body) {
  if (body.empty())
    return false;

  AliasAnalysis aliases(body, isPyOwnershipTrackedType,
                        isPyOwnershipIdentityTransform);
  llvm::SmallVector<std::pair<IncRefOp, DecRefOp>, 16> pairs;

  funcLike->walk([&](IncRefOp inc) {
    if (auto dec = pairedDecRef(inc, aliases))
      pairs.push_back({inc, dec});
  });

  if (pairs.empty())
    return false;

  llvm::SmallPtrSet<mlir::Operation *, 32> erased;
  for (auto [inc, dec] : pairs) {
    if (erased.contains(inc.getOperation()) ||
        erased.contains(dec.getOperation()))
      continue;
    erased.insert(inc.getOperation());
    erased.insert(dec.getOperation());
    inc.erase();
    dec.erase();
  }
  return true;
}

} // namespace pair_elision

namespace ownership_state {

struct State {
  llvm::DenseMap<mlir::Value, int64_t> balance;

  void normalize() {
    llvm::SmallVector<mlir::Value> zeros;
    for (auto [root, count] : balance)
      if (count == 0)
        zeros.push_back(root);
    for (mlir::Value root : zeros)
      balance.erase(root);
  }
};

bool trackedThroughAlias(mlir::Value value, const AliasAnalysis &aliases) {
  if (isPyOwnershipTrackedType(value.getType()))
    return true;
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  aliases.collectAliases(value, aliasSet);
  return llvm::any_of(aliasSet, [](mlir::Value alias) {
    return isPyOwnershipTrackedType(alias.getType());
  });
}

mlir::LogicalResult add(State &state, mlir::Value value, int64_t delta,
                        mlir::Operation *op, const AliasAnalysis &aliases) {
  if (!trackedThroughAlias(value, aliases))
    return mlir::success();
  mlir::Value rootValue = aliases.getRoot(value);
  if (root::immortal(rootValue, aliases))
    return mlir::success();
  int64_t &current = state.balance[rootValue];
  current += delta;
  if (current < 0)
    return op->emitOpError("ownership balance became negative for value ")
           << value;
  if (current == 0)
    state.balance.erase(rootValue);
  return mlir::success();
}

bool hasToken(const State &state, mlir::Value value,
              const AliasAnalysis &aliases) {
  if (!trackedThroughAlias(value, aliases))
    return true;
  mlir::Value root = aliases.getRoot(value);
  return state.balance.contains(root);
}

bool entryBorrowed(mlir::Value value, mlir::Block &entry,
                   const AliasAnalysis &aliases) {
  if (!isPyOwnershipTrackedType(value.getType()))
    return false;
  mlir::Value root = aliases.getRoot(value);
  for (mlir::BlockArgument arg : entry.getArguments())
    if (aliases.getRoot(arg) == root)
      return true;
  return false;
}

} // namespace ownership_state

namespace inc_ref {

mlir::LogicalResult verifyPremise(const ownership_state::State &state,
                                  IncRefOp inc, mlir::Block &entry,
                                  const AliasAnalysis &aliases,
                                  bool entryArgsBorrowed) {
  mlir::Value object = inc.getObject();
  mlir::Value rootValue = aliases.getRoot(object);
  if (root::immortal(rootValue, aliases))
    return mlir::success();
  if (ownership_state::hasToken(state, object, aliases)) {
    threadsafe::Retain::verifyOwnedToken(inc.getOperation());
    return mlir::success();
  }
  if (root::aggregateBorrow(rootValue, aliases)) {
    threadsafe::Retain::premise(inc.getOperation(),
                                ThreadSafetyAttrs::kPremiseAggregateBorrow);
    return mlir::success();
  }
  if (entryArgsBorrowed &&
      ownership_state::entryBorrowed(object, entry, aliases)) {
    threadsafe::Retain::premise(inc.getOperation(),
                                ThreadSafetyAttrs::kPremiseEntryBorrowed);
    return mlir::success();
  }
  return inc->emitOpError(
             "incref requires an owned token or entry-borrowed lifetime for ")
         << object;
}

} // namespace inc_ref

namespace successor_arg {

bool produced(mlir::Operation &op, mlir::Value value) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(&op);
  if (!branch)
    return false;
  for (unsigned successorIndex = 0, e = branch->getNumSuccessors();
       successorIndex != e; ++successorIndex) {
    mlir::SuccessorOperands operands =
        branch.getSuccessorOperands(successorIndex);
    for (unsigned i = 0, operandCount = operands.size(); i != operandCount; ++i)
      if (operands.isOperandProduced(i) && operands[i] == value)
        return true;
  }
  return false;
}

} // namespace successor_arg

namespace borrow {

mlir::LogicalResult verifyUse(const ownership_state::State &state,
                              mlir::Operation &op, mlir::Value value,
                              mlir::Block &entry, const AliasAnalysis &aliases,
                              bool entryArgsBorrowed) {
  if (!isPyOwnershipTrackedType(value.getType()))
    return mlir::success();
  mlir::Value rootValue = aliases.getRoot(value);
  if (root::immortal(rootValue, aliases))
    return mlir::success();
  if (ownership_state::hasToken(state, value, aliases))
    return mlir::success();
  if (entryArgsBorrowed &&
      ownership_state::entryBorrowed(value, entry, aliases))
    return mlir::success();
  return op.emitOpError("borrow use lacks a live ownership token or explicit "
                        "entry-borrowed lifetime for ")
         << value;
}

mlir::LogicalResult verifyUses(const ownership_state::State &state,
                               mlir::Operation &op, mlir::Block &entry,
                               const AliasAnalysis &aliases,
                               bool entryArgsBorrowed) {
  for (mlir::Value operand : op.getOperands()) {
    if (mlir::isa<IncRefOp, DecRefOp>(&op))
      continue;
    if (consumesPyOwnedOperand(&op, operand))
      continue;
    if (successor_arg::produced(op, operand))
      continue;
    if (mlir::failed(
            verifyUse(state, op, operand, entry, aliases, entryArgsBorrowed)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace borrow

namespace function_like {
mlir::LogicalResult verify(mlir::Operation *funcLike, mlir::Region &body,
                           bool entryArgsBorrowed);
} // namespace function_like

namespace nested_effect {

mlir::LogicalResult verifyAbsent(mlir::Operation &op) {
  if (op.getNumRegions() == 0)
    return mlir::success();

  if (mlir::isa<TryOp>(&op)) {
    for (mlir::Region &region : op.getRegions()) {
      if (!region.empty() &&
          mlir::failed(function_like::verify(&op, region,
                                             /*entryArgsBorrowed=*/false)))
        return mlir::failure();
    }
    return mlir::success();
  }

  mlir::Operation *bad = nullptr;
  for (mlir::Region &region : op.getRegions()) {
    region.walk([&](mlir::Operation *nested) {
      if (bad || nested == &op)
        return;
      if (operation::hasOwnershipEffect(nested))
        bad = nested;
    });
  }
  if (!bad)
    return mlir::success();
  return bad->emitOpError("nested ownership effect is not represented in the "
                          "flat CFG ownership proof");
}

} // namespace nested_effect

namespace effect {

mlir::LogicalResult apply(mlir::Operation &op, ownership_state::State &state,
                          mlir::Block &entry, const AliasAnalysis &aliases,
                          bool entryArgsBorrowed) {
  if (mlir::failed(
          borrow::verifyUses(state, op, entry, aliases, entryArgsBorrowed)))
    return mlir::failure();

  if (createsPyOwnedResult(&op))
    for (mlir::Value result : op.getResults())
      if (mlir::failed(ownership_state::add(state, result, +1, &op, aliases)))
        return mlir::failure();

  if (auto inc = mlir::dyn_cast<IncRefOp>(&op))
    if (mlir::failed(inc_ref::verifyPremise(state, inc, entry, aliases,
                                            entryArgsBorrowed)))
      return mlir::failure();

  if (auto inc = mlir::dyn_cast<IncRefOp>(&op))
    if (mlir::failed(
            ownership_state::add(state, inc.getObject(), +1, &op, aliases)))
      return mlir::failure();

  if (auto dec = mlir::dyn_cast<DecRefOp>(&op))
    if (mlir::failed(
            ownership_state::add(state, dec.getObject(), -1, &op, aliases)))
      return mlir::failure();

  for (mlir::Value operand : op.getOperands()) {
    if (mlir::isa<IncRefOp, DecRefOp>(&op))
      continue;
    if (!consumesPyOwnedOperand(&op, operand))
      continue;
    if (mlir::failed(ownership_state::add(state, operand, -1, &op, aliases)))
      return mlir::failure();
  }

  return mlir::success();
}

} // namespace effect

namespace ownership_state {

bool equal(const State &lhs, const State &rhs) {
  if (lhs.balance.size() != rhs.balance.size())
    return false;
  for (auto [root, count] : lhs.balance) {
    auto it = rhs.balance.find(root);
    if (it == rhs.balance.end() || it->second != count)
      return false;
  }
  return true;
}

mlir::LogicalResult mismatch(mlir::Operation *funcLike, mlir::Region &body,
                             mlir::Block *block, const State &existing,
                             const State &incoming) {
  unsigned blockIndex = 0;
  for (mlir::Block &candidate : body) {
    if (&candidate == block)
      break;
    ++blockIndex;
  }
  mlir::InFlightDiagnostic diag =
      funcLike->emitOpError(
          "ownership balance differs across predecessors for block ")
      << blockIndex;
  diag << "; existing={";
  bool first = true;
  for (auto [root, count] : existing.balance) {
    if (!first)
      diag << ", ";
    first = false;
    diag << root << ":" << count;
  }
  diag << "}, incoming={";
  first = true;
  for (auto [root, count] : incoming.balance) {
    if (!first)
      diag << ", ";
    first = false;
    diag << root << ":" << count;
  }
  diag << "}";
  return mlir::failure();
}

mlir::LogicalResult verifyExit(mlir::Operation *terminator, const State &state,
                               const AliasAnalysis &aliases) {
  for (auto [root, count] : state.balance) {
    if (count == 0)
      continue;
    return terminator->emitOpError(
               "ownership balance is not closed on exit; remaining tokens = ")
           << count << " for value " << root;
  }
  return mlir::success();
}

} // namespace ownership_state

namespace successor_arg {

mlir::LogicalResult addProducedOwnership(mlir::Operation *terminator,
                                         unsigned successorIndex,
                                         ownership_state::State &state,
                                         const AliasAnalysis &aliases) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return mlir::success();

  mlir::Block *successor = branch->getSuccessor(successorIndex);
  mlir::SuccessorOperands operands =
      branch.getSuccessorOperands(successorIndex);
  unsigned count =
      std::min<unsigned>(operands.size(), successor->getNumArguments());
  for (unsigned i = 0; i != count; ++i) {
    if (!operands.isOperandProduced(i))
      continue;
    mlir::BlockArgument argument = successor->getArgument(i);
    if (mlir::failed(
            ownership_state::add(state, argument, +1, terminator, aliases)))
      return mlir::failure();
  }
  return mlir::success();
}

} // namespace successor_arg

namespace function_like {

mlir::LogicalResult verify(mlir::Operation *funcLike, mlir::Region &body,
                           bool entryArgsBorrowed) {
  if (body.empty())
    return mlir::success();

  AliasAnalysis aliases(body, isPyOwnershipTrackedType,
                        isPyOwnershipIdentityTransform);

  llvm::DenseMap<mlir::Block *, ownership_state::State> entryStates;
  llvm::SmallVector<mlir::Block *, 16> worklist;
  llvm::SmallPtrSet<mlir::Block *, 16> queued;

  mlir::Block *entry = &body.front();
  ownership_state::State initialState;
  if (!entryArgsBorrowed) {
    for (mlir::BlockArgument arg : entry->getArguments())
      if (mlir::failed(
              ownership_state::add(initialState, arg, +1, funcLike, aliases)))
        return mlir::failure();
  }
  entryStates.try_emplace(entry, initialState);
  worklist.push_back(entry);
  queued.insert(entry);

  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    queued.erase(block);

    auto entryIt = entryStates.find(block);
    if (entryIt == entryStates.end())
      continue;
    ownership_state::State state = entryIt->second;

    for (mlir::Operation &op : *block) {
      if (mlir::failed(nested_effect::verifyAbsent(op)))
        return mlir::failure();
      if (mlir::failed(
              effect::apply(op, state, *entry, aliases, entryArgsBorrowed)))
        return mlir::failure();
    }

    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      return block->front().emitOpError("block has no terminator");

    if (terminator->getNumSuccessors() == 0) {
      if (mlir::failed(ownership_state::verifyExit(terminator, state, aliases)))
        return mlir::failure();
      continue;
    }

    for (auto [successorIndex, successor] :
         llvm::enumerate(block->getSuccessors())) {
      ownership_state::State edgeState = state;
      if (mlir::failed(successor_arg::addProducedOwnership(
              terminator, successorIndex, edgeState, aliases)))
        return mlir::failure();

      auto [it, inserted] = entryStates.try_emplace(successor, edgeState);
      if (!inserted) {
        if (!ownership_state::equal(it->second, edgeState))
          return ownership_state::mismatch(funcLike, body, successor,
                                           it->second, edgeState);
        continue;
      }
      if (queued.insert(successor).second)
        worklist.push_back(successor);
    }
  }

  return mlir::success();
}

} // namespace function_like

struct RefCountPairElisionPass
    : public mlir::PassWrapper<RefCountPairElisionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountPairElisionPass)

  llvm::StringRef getArgument() const override {
    return "py-refcount-pair-elision";
  }

  llvm::StringRef getDescription() const override {
    return "Erase only proven no-op py.incref/py.decref ownership pairs";
  }

  void runOnOperation() override {
    bool changed = false;
    getOperation().walk([&](FuncOp func) {
      changed |= pair_elision::run(func.getOperation(), func.getBody());
    });
    getOperation().walk([&](mlir::async::FuncOp func) {
      changed |= pair_elision::run(func.getOperation(), func.getBody());
    });
    (void)changed;
  }
};

struct OwnershipVerifierPass
    : public mlir::PassWrapper<OwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipVerifierPass)

  llvm::StringRef getArgument() const override { return "py-ownership-verify"; }

  llvm::StringRef getDescription() const override {
    return "Verify conservative quantitative ownership balance for Py values";
  }

  void runOnOperation() override {
    if (mlir::failed(verifyOwnership(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountPairElisionPass() {
  return std::make_unique<RefCountPairElisionPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass() {
  return std::make_unique<OwnershipVerifierPass>();
}

mlir::LogicalResult verifyOwnership(mlir::ModuleOp module) {
  bool failedAny = false;
  module.walk([&](FuncOp func) {
    if (mlir::failed(function_like::verify(func.getOperation(), func.getBody(),
                                           /*entryArgsBorrowed=*/true)))
      failedAny = true;
  });
  module.walk([&](mlir::async::FuncOp func) {
    if (mlir::failed(function_like::verify(func.getOperation(), func.getBody(),
                                           /*entryArgsBorrowed=*/true)))
      failedAny = true;
  });
  return mlir::failure(failedAny);
}

} // namespace py
