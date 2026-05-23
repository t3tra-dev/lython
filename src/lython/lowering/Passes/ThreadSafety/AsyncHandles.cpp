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
  auto callee = getDirectCallee(def);
  if (!callee)
    return false;
  mlir::Operation *calleeOp = lookupCallableSymbol(def, *callee);
  return py::async_runtime::Handle::ownedResult(*callee, calleeOp);
}

static mlir::Value root(mlir::Value value) {
  if (!value)
    return {};
  return pointer::stripCasts(value);
}

struct HandleState {
  ::llvm::DenseMap<mlir::Value, int64_t> balance;
  ::llvm::DenseMap<mlir::Value, int64_t> consumedEntry;

  bool empty() const { return balance.empty() && consumedEntry.empty(); }
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
    eraseIfZero(state.balance, root);
    return mlir::success();
  }

  int64_t deficit = required - current;
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
  return op->emitOpError("MLIR async runtime handle borrow lacks a live "
                         "refcounted handle for ")
         << value;
}

} // namespace borrow

static mlir::LogicalResult applyCall(mlir::Operation *op, HandleState &state,
                                     mlir::Block &entry) {
  auto callee = getDirectCallee(op);
  if (!callee)
    return mlir::success();
  mlir::Operation *calleeOp = lookupCallableSymbol(op, *callee);
  for (unsigned index : py::async_runtime::Handle::borrowedOperands(*callee))
    if (index < op->getNumOperands())
      if (mlir::failed(borrow::verify(op, op->getOperand(index), state, entry)))
        return mlir::failure();

  if (py::async_runtime::Callee::refcount(*callee)) {
    if (op->getNumOperands() != 2)
      return mlir::success();
    auto count = constant::anyInt(op->getOperand(1));
    if (!count || *count <= 0)
      return mlir::success();
    int64_t delta = *callee == "mlirAsyncRuntimeAddRef" ? *count : -*count;
    return handle::add(state, op->getOperand(0), delta, op, entry);
  }

  if (py::async_runtime::Handle::transferToExecute(*callee) &&
      op->getNumOperands() > 0) {
    if (mlir::failed(handle::add(state, op->getOperand(0), -1, op, entry)))
      return mlir::failure();
  }

  if (!py::async_runtime::Handle::ownedResult(*callee, calleeOp))
    return mlir::success();

  for (mlir::Value result : op->getResults())
    if (handle::carrier(result))
      if (mlir::failed(handle::add(state, result, +1, op, entry)))
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
    auto callee = getDirectCallee(&op);
    if (!callee || *callee != "mlirAsyncRuntimeDropRef" ||
        op.getNumOperands() < 1)
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
       << ", incoming consumed-entry=" << incoming.consumedEntry.size();
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

    for (auto [successorIndex, successor] :
         ::llvm::enumerate(block->getSuccessors())) {
      handle::HandleState edgeState = state;
      handle::remapSuccessor(terminator, successorIndex, edgeState);
      frame::dropOnSuspendDefault(terminator, successor, edgeState);
      auto [it, inserted] = entryStates.try_emplace(successor, edgeState);
      if (!inserted) {
        if (!handle::statesEqual(it->second, edgeState, compareConsumedEntry))
          return stateMismatch(funcLike, successor, it->second, edgeState);
        continue;
      }
      if (queued.insert(successor).second)
        worklist.push_back(successor);
    }
  }
  return mlir::success();
}

} // namespace py::threadsafe
