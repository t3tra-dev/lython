#include "Optimizer/Utils.h"

using namespace mlir;

namespace py::optimizer {

static bool isAfter(Operation *candidate, Operation *anchor) {
  return candidate->getBlock() == anchor->getBlock() &&
         anchor->isBeforeInBlock(candidate);
}

static void
collectSameBlockUsersAfter(Value value, Operation *anchor,
                           llvm::SmallPtrSetImpl<Value> &seenValues,
                           llvm::SmallPtrSetImpl<Operation *> &seenOps,
                           SmallVectorImpl<Operation *> &users) {
  if (!seenValues.insert(value).second)
    return;

  for (Operation *user : value.getUsers()) {
    if (user->getBlock() != anchor->getBlock())
      continue;
    if (isAfter(user, anchor) && seenOps.insert(user).second)
      users.push_back(user);
    for (Value result : user->getResults())
      collectSameBlockUsersAfter(result, anchor, seenValues, seenOps, users);
  }
}

void sinkClassDecrefsPastBorrowedAttrUses(ModuleOp module) {
  SmallVector<std::pair<DecRefOp, Operation *>> moves;

  module.walk([&](DecRefOp decref) {
    Value object = stripIdentityCasts(decref.getObject());
    if (!isa<ClassType>(object.getType()))
      return;

    Operation *latest = decref.getOperation();
    for (Operation *cursor = decref->getPrevNode(); cursor;
         cursor = cursor->getPrevNode()) {
      auto attrGet = dyn_cast<AttrGetOp>(cursor);
      if (!attrGet)
        continue;
      if (stripIdentityCasts(attrGet.getObject()) != object)
        continue;

      llvm::SmallPtrSet<Value, 16> seenValues;
      llvm::SmallPtrSet<Operation *, 16> seenOps;
      SmallVector<Operation *, 16> users;
      collectSameBlockUsersAfter(attrGet.getResult(), decref.getOperation(),
                                 seenValues, seenOps, users);
      for (Operation *user : users) {
        if (isa<DecRefOp>(user) && user->getOperand(0) == object)
          continue;
        if (latest == decref.getOperation() || latest->isBeforeInBlock(user))
          latest = user;
      }
    }

    if (latest != decref.getOperation() &&
        !latest->hasTrait<OpTrait::IsTerminator>())
      moves.push_back({decref, latest});
  });

  for (auto [decref, latest] : moves)
    decref->moveAfter(latest);
}

void markFinalLocalClassDecrefs(ModuleOp module) {
  module.walk([&](DecRefOp decref) {
    Value object = stripIdentityCasts(decref.getObject());
    if (!isa<ClassType>(object.getType()))
      return;
    if (!object.getDefiningOp<ClassNewOp>())
      return;

    for (Operation *user : object.getUsers()) {
      if (user == decref.getOperation())
        continue;
      if (user->getBlock() == decref->getBlock() &&
          decref->isBeforeInBlock(user))
        return;
    }

    decref->setAttr("lython.final_local_class_decref",
                    UnitAttr::get(module.getContext()));
  });
}

/// Remove duplicate Ly_DecRef calls on the same SSA value within a block
/// unless an intervening Ly_IncRef recreates ownership on that value.
void removeDuplicateDecRefs(ModuleOp module) {
  auto processBlocks = [&](auto funcLike) {
    SmallVector<LLVM::CallOp> toErase;
    for (Block &block : funcLike.getBody().getBlocks()) {
      llvm::SmallDenseSet<Value, 16> decrefSeen;
      for (Operation &op : block) {
        auto call = dyn_cast<LLVM::CallOp>(&op);
        if (!call || call.getNumOperands() != 1)
          continue;
        Value operand = call.getOperand(0);
        if (isCallTo(call, RuntimeSymbols::kIncRef)) {
          decrefSeen.erase(operand);
          continue;
        }
        if (!isCallTo(call, RuntimeSymbols::kDecRef))
          continue;
        if (decrefSeen.contains(operand)) {
          toErase.push_back(call);
          continue;
        }
        decrefSeen.insert(operand);
      }
    }

    for (auto call : toErase)
      call->erase();
  };

  module.walk([&](func::FuncOp func) {
    if (func.isExternal())
      return;
    processBlocks(func);
  });

  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal())
      return;
    processBlocks(func);
  });
}

/// Some borrowed-entry return retains inserted during py.return lowering can be
/// dropped by later rewrites for !py.func paths. Repair the final LLVM IR by
/// ensuring a direct `llvm.return %argN` on an entry block pointer argument has
/// a matching `Ly_IncRef(%argN)` immediately before it.
void repairMissingDirectArgReturnIncRefs(ModuleOp module) {
  module.walk([&](func::FuncOp func) {
    if (func.isExternal() || func.getBody().empty())
      return;

    Block &entry = func.getBody().front();

    for (Block &block : func.getBody()) {
      auto ret = dyn_cast<func::ReturnOp>(block.getTerminator());
      if (!ret || ret.getNumOperands() != 1)
        continue;

      Value returned = ret.getOperand(0);
      auto arg = dyn_cast<BlockArgument>(returned);
      if (!arg || arg.getOwner() != &entry)
        continue;
      if (!isa<LLVM::LLVMPointerType>(returned.getType()))
        continue;

      bool hasImmediateIncRef = false;
      if (Operation *prev = ret->getPrevNode()) {
        if (auto call = dyn_cast<LLVM::CallOp>(prev)) {
          if (isCallTo(call, RuntimeSymbols::kIncRef) &&
              call.getNumOperands() == 1 && call.getOperand(0) == returned)
            hasImmediateIncRef = true;
        }
      }
      if (hasImmediateIncRef)
        continue;

      OpBuilder builder(ret);
      auto increfFunc = getOrCreateRuntimeFunc(
          module, RuntimeSymbols::kIncRef,
          LLVM::LLVMVoidType::get(module.getContext()), ValueRange{returned});
      builder.create<LLVM::CallOp>(ret.getLoc(), increfFunc,
                                   ValueRange{returned});
    }
  });

  module.walk([&](LLVM::LLVMFuncOp func) {
    if (func.isExternal() || func.isDeclaration())
      return;
    if (func.getBody().empty())
      return;

    Block &entry = func.getBody().front();

    for (Block &block : func.getBody()) {
      auto ret = dyn_cast<LLVM::ReturnOp>(block.getTerminator());
      if (!ret || ret.getNumOperands() != 1)
        continue;

      Value returned = ret.getOperand(0);
      auto arg = dyn_cast<BlockArgument>(returned);
      if (!arg || arg.getOwner() != &entry)
        continue;
      if (!isa<LLVM::LLVMPointerType>(returned.getType()))
        continue;

      bool hasImmediateIncRef = false;
      if (Operation *prev = ret->getPrevNode()) {
        if (auto call = dyn_cast<LLVM::CallOp>(prev)) {
          if (isCallTo(call, RuntimeSymbols::kIncRef) &&
              call.getNumOperands() == 1 && call.getOperand(0) == returned)
            hasImmediateIncRef = true;
        }
      }
      if (hasImmediateIncRef)
        continue;

      OpBuilder builder(ret);
      auto increfFunc = getOrCreateRuntimeFunc(
          module, RuntimeSymbols::kIncRef,
          LLVM::LLVMVoidType::get(module.getContext()), ValueRange{returned});
      builder.create<LLVM::CallOp>(ret.getLoc(), increfFunc,
                                   ValueRange{returned});
    }
  });
}

void runRefcountPostLoweringOptimizations(ModuleOp module) {
  repairMissingDirectArgReturnIncRefs(module);
  removeDuplicateDecRefs(module);
}

} // namespace py::optimizer
