#include "Optimizer/Utils.h"

#include "Passes/OwnershipAnalysis.h"

namespace py::optimizer {

namespace {

mlir::Value tupleDropValue(mlir::Value element) { return element; }

bool isPureConstantRoot(mlir::Operation *op) {
  return mlir::isa_and_nonnull<NoneOp, FuncObjectOp, TupleEmptyOp,
                               StrConstantOp, IntConstantOp, FloatConstantOp>(
      op);
}

bool usedOnlyByDeadTupleElement(mlir::Value root, mlir::Value element,
                                TupleCreateOp tupleCreate) {
  if (!root || !element || !tupleCreate)
    return false;

  mlir::Operation *elementProducer = element.getDefiningOp();
  for (mlir::Operation *user : element.getUsers())
    if (user != tupleCreate.getOperation() && !mlir::isa<DecRefOp>(user))
      return false;

  for (mlir::Operation *user : root.getUsers()) {
    if (user == tupleCreate.getOperation() || user == elementProducer ||
        mlir::isa<DecRefOp>(user))
      continue;
    return false;
  }
  return true;
}

} // namespace

/// Clean up dead tuple operations whose only users are DecRefOps.
/// When removing a TupleCreateOp, we must add DecRefs for its elements
/// since the tuple's destructor would have handled them.
/// Returns true if any operations were erased.
bool container::cleanupDead(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> toErase;
  llvm::SmallVector<
      std::pair<TupleCreateOp, llvm::SmallVector<mlir::Operation *>>>
      tupleCreates;

  module.walk([&](mlir::Operation *tupleOp) {
    if (auto tupleEmpty = mlir::dyn_cast<TupleEmptyOp>(tupleOp)) {
      mlir::Value result = tupleEmpty->getResult(0);
      llvm::SmallVector<mlir::Operation *> decrefsToErase;
      bool canErase = true;

      for (mlir::Operation *user : result.getUsers()) {
        if (auto decref = mlir::dyn_cast<DecRefOp>(user)) {
          decrefsToErase.push_back(decref);
        } else {
          canErase = false;
          break;
        }
      }

      if (canErase && !decrefsToErase.empty()) {
        for (mlir::Operation *decref : decrefsToErase)
          toErase.push_back(decref);
        toErase.push_back(tupleOp);
      }
    } else if (auto tupleCreate = mlir::dyn_cast<TupleCreateOp>(tupleOp)) {
      if (tupleCreate->hasAttr("ly.async.gather_tuple"))
        return;
      mlir::Value result = tupleCreate->getResult(0);
      llvm::SmallVector<mlir::Operation *> decrefsToErase;
      bool canErase = true;

      for (mlir::Operation *user : result.getUsers()) {
        if (auto decref = mlir::dyn_cast<DecRefOp>(user)) {
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
    llvm::SmallVector<mlir::Value> elements(tupleCreate.getElements());
    llvm::SmallDenseSet<mlir::Value, 8> elementSet(elements.begin(),
                                                   elements.end());
    llvm::SmallDenseSet<mlir::Value, 8> retainedElements;

    for (mlir::Operation *prev = tupleCreate->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      auto incref = mlir::dyn_cast<IncRefOp>(prev);
      if (!incref)
        break;
      if (elementSet.contains(incref.getObject())) {
        toErase.push_back(incref);
        retainedElements.insert(incref.getObject());
      }
    }

    if (!decrefs.empty() && !elements.empty()) {
      mlir::OpBuilder builder(decrefs.front());
      for (mlir::Value element : elements) {
        mlir::Value dropValue = element;
        if (!consumesPyOwnedOperand(tupleCreate.getOperation(), element))
          continue;
        if (retainedElements.contains(element) ||
            retainedElements.contains(dropValue) ||
            retainedElements.contains(tupleDropValue(element)))
          continue;
        if (mlir::isa<ClassType>(dropValue.getType()) &&
            class_state::local(dropValue))
          continue;
        mlir::Value root = value::stripCasts(dropValue);
        if (mlir::Operation *defOp = root.getDefiningOp())
          if (isPureConstantRoot(defOp) &&
              usedOnlyByDeadTupleElement(root, element, tupleCreate))
            continue;

        if (auto arg = mlir::dyn_cast<mlir::BlockArgument>(root)) {
          auto *owner = arg.getOwner();
          auto *parent = owner ? owner->getParentOp() : nullptr;
          if (auto pyFunc = mlir::dyn_cast_or_null<FuncOp>(parent))
            if (owner == &pyFunc.getBody().front())
              continue;
          if (auto loweredFunc =
                  mlir::dyn_cast_or_null<mlir::func::FuncOp>(parent))
            if (owner == &loweredFunc.getBody().front())
              continue;
        }
        builder.create<DecRefOp>(tupleCreate.getLoc(), dropValue);
      }
    }

    for (mlir::Operation *decref : decrefs)
      toErase.push_back(decref);
    toErase.push_back(tupleCreate.getOperation());
  }

  for (mlir::Operation *op : toErase)
    op->erase();

  return !toErase.empty();
}

/// Remove unused TupleEmptyOps.
void container::removeEmptyTuples(mlir::ModuleOp module) {
  llvm::SmallVector<TupleEmptyOp> toErase;
  module.walk([&](TupleEmptyOp op) {
    if (op.getResult().use_empty())
      toErase.push_back(op);
  });
  for (auto op : toErase)
    op.erase();
}

void pipeline::containerPre(mlir::ModuleOp module) {
  container::cleanupDead(module);
  container::removeEmptyTuples(module);
  consume::listAppendValues(module);
}

} // namespace py::optimizer
