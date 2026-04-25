#include "Optimizer/Utils.h"

using namespace mlir;

namespace py::optimizer {

/// Clean up dead tuple operations whose only users are DecRefOps.
/// When removing a TupleCreateOp, we must add DecRefs for its elements
/// since the tuple's destructor would have handled them.
/// Returns true if any operations were erased.
bool cleanupDeadTuples(ModuleOp module) {
  SmallVector<Operation *> toErase;
  SmallVector<std::pair<TupleCreateOp, SmallVector<Operation *>>> tupleCreates;

  module.walk([&](Operation *tupleOp) {
    if (auto tupleEmpty = dyn_cast<TupleEmptyOp>(tupleOp)) {
      Value result = tupleEmpty->getResult(0);
      SmallVector<Operation *> decrefsToErase;
      bool canErase = true;

      for (Operation *user : result.getUsers()) {
        if (auto decref = dyn_cast<DecRefOp>(user)) {
          decrefsToErase.push_back(decref);
        } else {
          canErase = false;
          break;
        }
      }

      if (canErase && !decrefsToErase.empty()) {
        for (Operation *decref : decrefsToErase)
          toErase.push_back(decref);
        toErase.push_back(tupleOp);
      }
    } else if (auto tupleCreate = dyn_cast<TupleCreateOp>(tupleOp)) {
      Value result = tupleCreate->getResult(0);
      SmallVector<Operation *> decrefsToErase;
      bool canErase = true;

      for (Operation *user : result.getUsers()) {
        if (auto decref = dyn_cast<DecRefOp>(user)) {
          decrefsToErase.push_back(decref);
        } else {
          canErase = false;
          break;
        }
      }

      if (canErase && !decrefsToErase.empty())
        tupleCreates.push_back({tupleCreate, std::move(decrefsToErase)});
    }
  });

  for (auto &[tupleCreate, decrefs] : tupleCreates) {
    SmallVector<Value> elements(tupleCreate.getElements());
    llvm::SmallDenseSet<Value, 8> elementSet(elements.begin(), elements.end());

    for (Operation *prev = tupleCreate->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      auto incref = dyn_cast<IncRefOp>(prev);
      if (!incref)
        break;
      if (elementSet.contains(incref.getObject()))
        toErase.push_back(incref);
    }

    if (!decrefs.empty() && !elements.empty()) {
      OpBuilder builder(decrefs.front());
      for (Value element : elements) {
        Value root = stripIdentityCasts(element);
        if (Operation *defOp = root.getDefiningOp())
          if (isa<NoneOp, FuncObjectOp, TupleEmptyOp>(defOp))
            continue;

        bool hasOtherUsers = false;
        for (Operation *user : element.getUsers()) {
          if (user == tupleCreate.getOperation())
            continue;
          if (isa<CastIdentityOp>(user))
            continue;
          hasOtherUsers = true;
          break;
        }

        if (!hasOtherUsers) {
          if (auto arg = dyn_cast<BlockArgument>(root)) {
            auto *owner = arg.getOwner();
            auto *parent = owner ? owner->getParentOp() : nullptr;
            if (auto pyFunc = dyn_cast_or_null<FuncOp>(parent))
              if (owner == &pyFunc.getBody().front())
                continue;
            if (auto loweredFunc = dyn_cast_or_null<func::FuncOp>(parent))
              if (owner == &loweredFunc.getBody().front())
                continue;
          }
          builder.create<DecRefOp>(tupleCreate.getLoc(), element);
        }
      }
    }

    for (Operation *decref : decrefs)
      toErase.push_back(decref);
    toErase.push_back(tupleCreate.getOperation());
  }

  for (Operation *op : toErase)
    op->erase();

  return !toErase.empty();
}

/// Remove unused TupleEmptyOps.
void removeUnusedTupleEmpties(ModuleOp module) {
  SmallVector<TupleEmptyOp> toErase;
  module.walk([&](TupleEmptyOp op) {
    if (op.getResult().use_empty())
      toErase.push_back(op);
  });
  for (auto op : toErase)
    op->erase();
}

void runContainerPreLoweringOptimizations(ModuleOp module) {
  cleanupDeadTuples(module);
  removeUnusedTupleEmpties(module);
  markConsumedListAppendValues(module);
}

} // namespace py::optimizer
