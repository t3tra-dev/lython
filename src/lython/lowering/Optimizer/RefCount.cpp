#include "Optimizer/Utils.h"

namespace py::optimizer {

static bool isAfter(mlir::Operation *candidate, mlir::Operation *anchor) {
  return candidate->getBlock() == anchor->getBlock() &&
         anchor->isBeforeInBlock(candidate);
}

static void
collectSameBlockUsersAfter(mlir::Value value, mlir::Operation *anchor,
                           llvm::SmallPtrSetImpl<mlir::Value> &seenValues,
                           llvm::SmallPtrSetImpl<mlir::Operation *> &seenOps,
                           llvm::SmallVectorImpl<mlir::Operation *> &users) {
  if (!seenValues.insert(value).second)
    return;

  for (mlir::Operation *user : value.getUsers()) {
    if (user->getBlock() != anchor->getBlock())
      continue;
    if (isAfter(user, anchor) && seenOps.insert(user).second)
      users.push_back(user);
    for (mlir::Value result : user->getResults())
      collectSameBlockUsersAfter(result, anchor, seenValues, seenOps, users);
  }
}

void refcount::sinkClassDecrefs(mlir::ModuleOp module) {
  llvm::SmallVector<std::pair<DecRefOp, mlir::Operation *>> moves;

  module.walk([&](DecRefOp decref) {
    mlir::Value object = value::stripCasts(decref.getObject());
    if (!mlir::isa<ClassType>(object.getType()))
      return;

    mlir::Operation *latest = decref.getOperation();
    for (mlir::Operation *cursor = decref->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      mlir::Value attrObject;
      mlir::Value attrResult;
      if (auto attrGet = mlir::dyn_cast<AttrGetOp>(cursor)) {
        attrObject = attrGet.getObject();
        attrResult = attrGet.getResult();
      } else if (auto attrGet = mlir::dyn_cast<AttrGetLocalOp>(cursor)) {
        attrObject = attrGet.getObject();
        attrResult = attrGet.getResult();
      } else {
        continue;
      }
      if (value::stripCasts(attrObject) != object)
        continue;

      llvm::SmallPtrSet<mlir::Value, 16> seenValues;
      llvm::SmallPtrSet<mlir::Operation *, 16> seenOps;
      llvm::SmallVector<mlir::Operation *, 16> users;
      collectSameBlockUsersAfter(attrResult, decref.getOperation(), seenValues,
                                 seenOps, users);
      for (mlir::Operation *user : users) {
        if (mlir::isa<DecRefOp>(user) && user->getOperand(0) == object)
          continue;
        if (latest == decref.getOperation() || latest->isBeforeInBlock(user))
          latest = user;
      }
    }

    if (latest != decref.getOperation() &&
        !latest->hasTrait<mlir::OpTrait::IsTerminator>())
      moves.push_back({decref, latest});
  });

  for (auto [decref, latest] : moves)
    decref->moveAfter(latest);
}

void refcount::markFinalLocal(mlir::ModuleOp module) {
  module.walk([&](DecRefOp decref) {
    mlir::Value object = value::stripCasts(decref.getObject());
    if (!mlir::isa<ClassType>(object.getType()))
      return;
    if (!object.getDefiningOp<ClassNewOp>())
      return;

    for (mlir::Operation *user : object.getUsers()) {
      if (user == decref.getOperation())
        continue;
      if (user->getBlock() == decref->getBlock() &&
          decref->isBeforeInBlock(user))
        return;
    }

    decref->setAttr("ly.final_local_class_decref",
                    mlir::UnitAttr::get(module.getContext()));
  });
}

void pipeline::refcountPost(mlir::ModuleOp module) {
  (void)module;
  // Do not remove DecRef calls by SSA-value identity here. Multiple ownership
  // tokens may legitimately refer to the same pointer, and collapsing them to a
  // set breaks the quantitative ownership invariant.
}

} // namespace py::optimizer
