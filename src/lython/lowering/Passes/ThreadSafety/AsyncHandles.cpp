#include "Verifier.h"

namespace py::threadsafe {

namespace entry {

static bool blockArgument(mlir::BlockArgument arg) {
  mlir::Operation *parent = arg.getOwner()->getParentOp();
  auto function = mlir::dyn_cast_or_null<mlir::FunctionOpInterface>(parent);
  return function && parent->getNumRegions() != 0 &&
         !parent->getRegion(0).empty() &&
         arg.getOwner() == &parent->getRegion(0).front();
}

} // namespace entry

namespace handle {

static bool carrier(mlir::Value value) {
  value = pointer::stripCasts(value);
  if (!value || !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
    return false;

  if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    if (!entry::blockArgument(arg))
      return true;
    mlir::Operation *parent = arg.getOwner()->getParentOp();
    return function_arg::hasAttr(arg, AsyncSafetyAttrs::kRuntimeHandle) ||
           py::async_runtime::Entry::isFunction(parent);
  }

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->hasAttr(AsyncSafetyAttrs::kRuntimeHandle))
    return true;
  return py::async_runtime::Handle::ownedResult(def);
}

static mlir::Value root(mlir::Value value) {
  if (!value)
    return {};
  return pointer::stripCasts(value);
}

struct HandleState {
  ::llvm::DenseMap<mlir::Value, int64_t> balance;
  ::llvm::DenseMap<mlir::Value, int64_t> consumedEntry;
  ::llvm::DenseMap<mlir::Value, int64_t> consumedGenerated;

  bool empty() const {
    return balance.empty() && consumedEntry.empty() &&
           consumedGenerated.empty();
  }
};

static bool mapEquals(const ::llvm::DenseMap<mlir::Value, int64_t> &lhs,
                      const ::llvm::DenseMap<mlir::Value, int64_t> &rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [root, count] : lhs) {
    auto it = rhs.find(root);
    if (it == rhs.end() || it->second != count)
      return false;
  }
  return true;
}

static bool statesEqual(const HandleState &lhs, const HandleState &rhs,
                        bool compareConsumedEntry) {
  if (!mapEquals(lhs.balance, rhs.balance))
    return false;
  return !compareConsumedEntry ||
         mapEquals(lhs.consumedEntry, rhs.consumedEntry);
}

static bool mergeMayConsumedGenerated(HandleState &existing,
                                      const HandleState &incoming) {
  bool changed = false;
  for (auto [root, count] : incoming.consumedGenerated) {
    int64_t &slot = existing.consumedGenerated[root];
    int64_t merged = std::max(slot, count);
    if (slot == merged)
      continue;
    slot = merged;
    changed = true;
  }
  return changed;
}

static bool entryRoot(mlir::Value value, mlir::Block &entry) {
  value = root(value);
  if (!value)
    return false;
  for (mlir::BlockArgument arg : entry.getArguments())
    if (carrier(arg) && root(arg) == value)
      return true;
  return false;
}

static void eraseIfZero(::llvm::DenseMap<mlir::Value, int64_t> &map,
                        mlir::Value root) {
  auto it = map.find(root);
  if (it != map.end() && it->second == 0)
    map.erase(it);
}

static mlir::LogicalResult add(HandleState &state, mlir::Value value,
                               int64_t delta, mlir::Operation *op,
                               mlir::Block &entry) {
  if (!carrier(value) || delta == 0)
    return mlir::success();
  mlir::Value root = handle::root(value);
  if (delta > 0) {
    state.balance[root] += delta;
    return mlir::success();
  }

  int64_t required = -delta;
  int64_t current = state.balance.lookup(root);
  if (current >= required) {
    state.balance[root] = current - required;
    if (state.balance.lookup(root) == 0 && value.getDefiningOp())
      state.consumedGenerated[root] = 1;
    eraseIfZero(state.balance, root);
    return mlir::success();
  }

  int64_t deficit = required - current;
  if (value.getDefiningOp() && deficit == 1 &&
      state.consumedGenerated.lookup(root) == 0) {
    state.balance.erase(root);
    state.consumedGenerated[root] = 1;
    return mlir::success();
  }

  if (entryRoot(root, entry) && state.consumedEntry.lookup(root) == 0 &&
      deficit == 1) {
    state.balance.erase(root);
    state.consumedEntry[root] = 1;
    return mlir::success();
  }

  return op->emitOpError("MLIR async runtime handle refcount becomes negative "
                         "for ")
         << value;
}

static mlir::LogicalResult addOwnedResult(HandleState &state, mlir::Value value,
                                          mlir::Operation *op) {
  if (!value || !mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType()))
    return mlir::success();
  mlir::Value root = handle::root(value);
  if (!root)
    return op->emitOpError("MLIR async runtime handle result has no root");
  ++state.balance[root];
  return mlir::success();
}

static bool live(const HandleState &state, mlir::Value value,
                 mlir::Block &entry) {
  if (!carrier(value))
    return true;
  mlir::Value root = handle::root(value);
  if (state.balance.lookup(root) > 0)
    return true;
  return entryRoot(root, entry) && state.consumedEntry.lookup(root) == 0;
}

namespace borrow {

static mlir::LogicalResult verify(mlir::Operation *op, mlir::Value value,
                                  const HandleState &state,
                                  mlir::Block &entry) {
  if (live(state, value, entry))
    return mlir::success();
  mlir::InFlightDiagnostic diag = op->emitOpError(
      "MLIR async runtime handle borrow lacks a live refcounted handle for ");
  diag << value;
  diag << "; live roots=[";
  ::llvm::interleaveComma(state.balance, diag, [&](auto item) {
    diag << item.first << ":" << item.second;
  });
  diag << "], consumed-generated=[";
  ::llvm::interleaveComma(state.consumedGenerated, diag, [&](auto item) {
    diag << item.first << ":" << item.second;
  });
  diag << "]";
  diag << "; op=" << *op;
  return diag;
}

} // namespace borrow

static mlir::LogicalResult applyCall(mlir::Operation *op, HandleState &state,
                                     mlir::Block &entry) {
  if (auto delta = attrs::i64(op, AsyncSafetyAttrs::kRuntimeRefcountDelta)) {
    if (op->getNumOperands() != 2)
      return op->emitOpError("async runtime refcount effect must have handle "
                             "and count operands");
    if (mlir::failed(borrow::verify(op, op->getOperand(0), state, entry)))
      return mlir::failure();
    auto count = constant::anyInt(op->getOperand(1));
    if (!count || *count <= 0)
      return op->emitOpError("async runtime refcount count must be a positive "
                             "integer constant");
    return handle::add(state, op->getOperand(0), (*delta) * (*count), op,
                       entry);
  }
  for (unsigned index : py::async_runtime::Handle::borrowedOperands(op))
    if (index < op->getNumOperands())
      if (mlir::failed(borrow::verify(op, op->getOperand(index), state, entry)))
        return mlir::failure();

  for (unsigned index : py::async_runtime::Handle::transferredOperands(op)) {
    if (index >= op->getNumOperands())
      continue;
    if (mlir::failed(handle::add(state, op->getOperand(index), -1, op, entry)))
      return mlir::failure();
  }

  if (!py::async_runtime::Handle::ownedResult(op))
    return mlir::success();

  for (mlir::Value result : op->getResults())
    if (mlir::failed(handle::addOwnedResult(state, result, op)))
      return mlir::failure();
  return mlir::success();
}

static void moveRoot(::llvm::DenseMap<mlir::Value, int64_t> &map,
                     mlir::Value from, mlir::Value to) {
  from = root(from);
  to = root(to);
  if (!from || !to || from == to)
    return;
  auto it = map.find(from);
  if (it == map.end())
    return;
  map[to] += it->second;
  map.erase(it);
  eraseIfZero(map, to);
}

static void remapSuccessor(mlir::Operation *terminator, unsigned successorIndex,
                           HandleState &state) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return;
  mlir::Block *successor = branch->getSuccessor(successorIndex);
  mlir::SuccessorOperands operands =
      branch.getSuccessorOperands(successorIndex);
  unsigned count =
      std::min<unsigned>(operands.size(), successor->getNumArguments());
  for (unsigned i = 0; i != count; ++i) {
    if (operands.isOperandProduced(i))
      continue;
    mlir::Value operand = operands[i];
    mlir::BlockArgument argument = successor->getArgument(i);
    if (!handle::carrier(operand) || !handle::carrier(argument))
      continue;
    handle::moveRoot(state.balance, operand, argument);
    handle::moveRoot(state.consumedEntry, operand, argument);
    handle::moveRoot(state.consumedGenerated, operand, argument);
  }
}

} // namespace handle

namespace frame {

static void
collectTransfers(mlir::Block *block,
                 ::llvm::SmallPtrSetImpl<mlir::Block *> &visited,
                 mlir::SmallVectorImpl<mlir::Value> &transferredRoots) {
  if (!block)
    return;
  if (!visited.insert(block).second)
    return;
  for (mlir::Operation &op : *block) {
    if (!op.hasAttr(OwnershipContractAttrs::kFrameTransfer))
      continue;
    auto delta = attrs::i64(&op, AsyncSafetyAttrs::kRuntimeRefcountDelta);
    if (!delta || *delta >= 0 || op.getNumOperands() < 1)
      continue;
    mlir::Value root = handle::root(op.getOperand(0));
    if (!::llvm::is_contained(transferredRoots, root))
      transferredRoots.push_back(root);
  }

  mlir::Operation *terminator = block->getTerminator();
  if (!terminator)
    return;
  for (mlir::Block *successor : terminator->getSuccessors())
    collectTransfers(successor, visited, transferredRoots);
}

static void dropOnSuspendDefault(mlir::Operation *terminator,
                                 mlir::Block *successor,
                                 handle::HandleState &state) {
  auto switchOp = mlir::dyn_cast<mlir::LLVM::SwitchOp>(terminator);
  if (!switchOp || successor != switchOp.getDefaultDestination() ||
      !isCoroSuspendStatus(switchOp.getValue()))
    return;

  auto cleanupIndex = findCoroSuspendCleanupSuccessorIndex(switchOp);
  if (!cleanupIndex)
    return;

  mlir::SmallVector<mlir::Value, 4> transferredRoots;
  ::llvm::SmallPtrSet<mlir::Block *, 8> visited;
  collectTransfers(switchOp->getSuccessor(*cleanupIndex), visited,
                   transferredRoots);
  for (mlir::Value root : transferredRoots)
    state.balance.erase(root);
}

} // namespace frame

static mlir::LogicalResult stateMismatch(mlir::Operation *funcLike,
                                         mlir::Block *block,
                                         const handle::HandleState &existing,
                                         const handle::HandleState &incoming) {
  mlir::InFlightDiagnostic diag = funcLike->emitOpError(
      "MLIR async runtime handle balance differs across predecessors");
  if (block) {
    std::string blockName;
    ::llvm::raw_string_ostream os(blockName);
    block->printAsOperand(os, /*printType=*/false);
    diag << " for block " << os.str();
  }
  diag << "; existing refs=" << existing.balance.size()
       << ", incoming refs=" << incoming.balance.size()
       << ", existing consumed-entry=" << existing.consumedEntry.size()
       << ", incoming consumed-entry=" << incoming.consumedEntry.size()
       << ", existing consumed-generated=" << existing.consumedGenerated.size()
       << ", incoming consumed-generated=" << incoming.consumedGenerated.size();
  diag << "; existing roots=[";
  ::llvm::interleaveComma(existing.balance, diag, [&](auto entry) {
    diag << entry.first << ":" << entry.second;
  });
  diag << "], incoming roots=[";
  ::llvm::interleaveComma(incoming.balance, diag, [&](auto entry) {
    diag << entry.first << ":" << entry.second;
  });
  diag << "]";
  return mlir::failure();
}

namespace exit {

static mlir::LogicalResult verify(mlir::Operation *terminator,
                                  const handle::HandleState &state) {
  if (control::noReturn(terminator))
    return mlir::success();

  ::llvm::DenseMap<mlir::Value, int64_t> returned;
  if (mlir::isa<mlir::func::ReturnOp, mlir::LLVM::ReturnOp>(terminator))
    for (mlir::Value operand : terminator->getOperands())
      if (handle::carrier(operand))
        ++returned[handle::root(operand)];

  for (auto [root, count] : state.balance) {
    int64_t transferred = returned.lookup(root);
    if (count == transferred)
      continue;
    return terminator->emitOpError(
               "MLIR async runtime handle balance is not closed on exit; "
               "remaining refs=")
           << count << ", returned refs=" << transferred << " for " << root;
  }
  return mlir::success();
}

} // namespace exit

mlir::LogicalResult
verifier::async_runtime::Handles::balance(mlir::Operation *funcLike,
                                          mlir::Region &body) {
  if (body.empty())
    return mlir::success();
  bool compareConsumedEntry = !hasPresplitCoroutinePassthrough(funcLike);

  ::llvm::DenseMap<mlir::Block *, handle::HandleState> entryStates;
  mlir::SmallVector<mlir::Block *, 16> worklist;
  ::llvm::SmallPtrSet<mlir::Block *, 16> queued;

  mlir::Block *entry = &body.front();
  entryStates.try_emplace(entry);
  worklist.push_back(entry);
  queued.insert(entry);

  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    queued.erase(block);

    auto entryIt = entryStates.find(block);
    if (entryIt == entryStates.end())
      continue;
    handle::HandleState state = entryIt->second;

    for (mlir::Operation &op : *block)
      if (mlir::failed(handle::applyCall(&op, state, *entry)))
        return mlir::failure();

    mlir::Operation *terminator = block->getTerminator();
    if (!terminator)
      return block->front().emitOpError("block has no terminator");

    if (terminator->getNumSuccessors() == 0) {
      if (mlir::failed(exit::verify(terminator, state)))
        return mlir::failure();
      continue;
    }

    for (unsigned successorIndex = 0, e = terminator->getNumSuccessors();
         successorIndex != e; ++successorIndex) {
      mlir::Block *successor = terminator->getSuccessor(successorIndex);
      handle::HandleState edgeState = state;
      handle::remapSuccessor(terminator, successorIndex, edgeState);
      frame::dropOnSuspendDefault(terminator, successor, edgeState);
      auto [it, inserted] = entryStates.try_emplace(successor, edgeState);
      if (!inserted) {
        if (!handle::statesEqual(it->second, edgeState, compareConsumedEntry))
          return stateMismatch(funcLike, successor, it->second, edgeState);
        if (handle::mergeMayConsumedGenerated(it->second, edgeState))
          if (queued.insert(successor).second)
            worklist.push_back(successor);
        continue;
      }
      if (queued.insert(successor).second)
        worklist.push_back(successor);
    }
  }
  return mlir::success();
}

} // namespace py::threadsafe
