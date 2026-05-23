#include "Common/RuntimeSupport.h"
#include "Passes/OwnershipAnalysis.h"

#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "PyDialect.h.inc"

#define GET_OP_CLASSES
#include "PyOps.h.inc"
#undef GET_OP_CLASSES

#define DEBUG_TYPE "py-refcount-insertion"

namespace py {
namespace {

static mlir::LogicalResult
insertDecRefOnSuccessorEdge(mlir::OpBuilder &builder, mlir::Value value,
                            mlir::Operation *terminator,
                            unsigned successorIndex) {
  mlir::Block *source = terminator->getBlock();
  mlir::Block *successor = terminator->getSuccessor(successorIndex);

  if (successor->getUniquePredecessor() == source) {
    builder.setInsertionPointToStart(successor);
    builder.create<DecRefOp>(value.getLoc(), value);
    return mlir::success();
  }

  mlir::Block *refCountBlock = new mlir::Block();
  for (mlir::BlockArgument arg : successor->getArguments())
    refCountBlock->addArgument(arg.getType(), arg.getLoc());
  refCountBlock->insertBefore(successor);

  builder.setInsertionPointToStart(refCountBlock);
  builder.create<DecRefOp>(value.getLoc(), value);
  builder.create<mlir::cf::BranchOp>(value.getLoc(), successor,
                                     refCountBlock->getArguments());
  terminator->setSuccessor(refCountBlock, successorIndex);
  return mlir::success();
}

using AliasAnalysis = OwnershipAliasAnalysis;

static bool hasImmortalProducer(mlir::Value root,
                                const AliasAnalysis &aliases) {
  llvm::SmallVector<mlir::Value, 4> aliasSet;
  aliases.collectAliases(root, aliasSet);
  for (mlir::Value member : aliasSet) {
    mlir::Operation *def = member.getDefiningOp();
    if (def && isPyOwnershipImmortalOp(def))
      return true;
  }
  return false;
}

struct RefCountInsertionPass
    : public mlir::PassWrapper<RefCountInsertionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountInsertionPass)

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();

    module.walk([&](FuncOp func) {
      if (mlir::failed(processFunctionLike(func.getOperation(), func.getBody(),
                                           /*entryArgsBorrowed=*/true)))
        signalPassFailure();
    });
    module.walk([&](mlir::async::FuncOp func) {
      if (mlir::failed(processFunctionLike(func.getOperation(), func.getBody(),
                                           /*entryArgsBorrowed=*/true)))
        signalPassFailure();
    });
  }

private:
  mlir::LogicalResult processFunctionLike(mlir::Operation *funcLike,
                                          mlir::Region &body,
                                          bool entryArgsBorrowed) {
    if (body.empty())
      return mlir::success();

    AliasAnalysis aliases(body, isPyOwnershipTrackedType,
                          isPyOwnershipIdentityTransform);
    mlir::Liveness liveness(funcLike);
    llvm::DenseSet<mlir::Value> processedRoots;

    auto processOwnedRoot = [&](mlir::Value root) -> mlir::LogicalResult {
      if (hasImmortalProducer(root, aliases))
        return mlir::success();
      if (processedRoots.insert(root).second)
        return addRefCountingForAliasSet(root, aliases, liveness);
      return mlir::success();
    };

    // Function entry block arguments are borrowed references and must not be
    // decremented by the callee. Exception/try region entry arguments are
    // ownership transfers and are handled by the same affine token model as
    // ordinary owned producers.
    mlir::Block &entryBlock = body.front();
    for (mlir::BlockArgument arg : entryBlock.getArguments()) {
      if (!isPyOwnershipTrackedType(arg.getType()))
        continue;
      mlir::Value root = aliases.getRoot(arg);
      if (entryArgsBorrowed) {
        if (hasImmortalProducer(root, aliases))
          continue;
        processedRoots.insert(root);
        addRetainsForBorrowedConsumes(root, aliases);
        continue;
      }
      if (mlir::failed(processOwnedRoot(root)))
        return mlir::failure();
    }

    for (mlir::Block &block : body) {
      // Skip entry block args since they were handled above according to the
      // region entry ownership convention.
      if (&block == &entryBlock)
        continue;

      for (mlir::BlockArgument arg : block.getArguments()) {
        if (!isPyOwnershipTrackedType(arg.getType()))
          continue;

        mlir::Value root = aliases.getRoot(arg);
        if (mlir::failed(processOwnedRoot(root)))
          return mlir::failure();
      }
    }

    for (mlir::Block &block : body) {
      for (mlir::Operation &operation : block) {
        mlir::Operation *op = &operation;
        if (!createsPyOwnedResult(op))
          continue;

        for (mlir::Value result : op->getResults()) {
          if (!isPyOwnershipTrackedType(result.getType()))
            continue;

          mlir::Value root = aliases.getRoot(result);
          if (mlir::failed(processOwnedRoot(root)))
            return mlir::failure();
        }
      }
    }

    for (mlir::Block &block : body) {
      for (mlir::Operation &operation : block) {
        auto tryOp = mlir::dyn_cast<TryOp>(operation);
        if (!tryOp)
          continue;
        for (mlir::Region &region : tryOp->getRegions()) {
          if (mlir::failed(processFunctionLike(tryOp.getOperation(), region,
                                               /*entryArgsBorrowed=*/false)))
            return mlir::failure();
        }
      }
    }

    return mlir::success();
  }

  void addRetainsForBorrowedConsumes(mlir::Value root, AliasAnalysis &aliases) {
    mlir::OpBuilder builder(root.getContext());

    llvm::SmallVector<mlir::Value, 4> aliasSet;
    aliases.collectAliases(root, aliasSet);

    for (mlir::Value alias : aliasSet) {
      for (mlir::OpOperand &use : alias.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (isPyOwnershipIdentityTransform(user))
          continue;
        if (!consumesPyOwnedOperand(user, alias))
          continue;

        builder.setInsertionPoint(user);
        builder.create<IncRefOp>(alias.getLoc(), alias);
      }
    }
  }

  mlir::LogicalResult addRefCountingForAliasSet(mlir::Value root,
                                                AliasAnalysis &aliases,
                                                mlir::Liveness &liveness) {
    mlir::OpBuilder builder(root.getContext());

    llvm::SmallVector<mlir::Value, 4> aliasSet;
    aliases.collectAliases(root, aliasSet);

    llvm::SmallPtrSet<mlir::Operation *, 8> allUsers;
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 4>>
        consumingOperandsByUser;
    for (mlir::Value alias : aliasSet) {
      for (mlir::Operation *user : alias.getUsers()) {
        if (!isPyOwnershipIdentityTransform(user))
          allUsers.insert(user);
      }
      for (mlir::OpOperand &use : alias.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (isPyOwnershipIdentityTransform(user))
          continue;
        if (!consumesPyOwnedOperand(user, alias))
          continue;
        consumingOperandsByUser[user].push_back(alias);
      }
    }

    if (allUsers.empty()) {
      if (mlir::Operation *defOp = root.getDefiningOp()) {
        builder.setInsertionPointAfter(defOp);
      } else {
        builder.setInsertionPointToStart(root.getParentBlock());
      }
      builder.create<DecRefOp>(root.getLoc(), root);
      return mlir::success();
    }

    mlir::Region *definingRegion = root.getParentRegion();

    llvm::DenseMap<mlir::Block *, mlir::Operation *> lastUserInBlock;
    for (mlir::Operation *user : allUsers) {
      mlir::Block *userBlock = user->getBlock();
      mlir::Block *ancestor =
          definingRegion->findAncestorBlockInRegion(*userBlock);
      if (!ancestor)
        continue;

      mlir::Operation *ancestorOp = ancestor->findAncestorOpInBlock(*user);
      if (!ancestorOp)
        continue;

      auto it = lastUserInBlock.find(ancestor);
      if (it == lastUserInBlock.end() ||
          it->second->isBeforeInBlock(ancestorOp)) {
        lastUserInBlock[ancestor] = ancestorOp;
      }
    }

    for (auto &[block, lastUser] : lastUserInBlock) {
      const mlir::LivenessBlockInfo *blockLiveness =
          liveness.getLiveness(block);
      if (!blockLiveness)
        continue;

      bool anyLiveOut = false;
      for (mlir::Value alias : aliasSet) {
        if (blockLiveness->isLiveOut(alias)) {
          anyLiveOut = true;
          break;
        }
      }

      if (anyLiveOut)
        continue; // mlir::Value doesn't die in this block

      mlir::Value usedAlias = root;
      for (mlir::Value alias : aliasSet) {
        for (mlir::Value operand : lastUser->getOperands()) {
          if (operand == alias) {
            usedAlias = alias;
            break;
          }
        }
      }

      // Apply Affine SSA rules
      if (consumesPyOwnedOperand(lastUser, usedAlias)) {
        // Last use + Consume = Move Semantics (no action needed)
      } else {
        // Last use + Borrow = Need to drop after use
        if (lastUser->getNumSuccessors() == 0) {
          builder.setInsertionPointAfter(lastUser);
          builder.create<DecRefOp>(root.getLoc(), usedAlias);
        } else {
          for (unsigned i = 0; i < lastUser->getNumSuccessors(); ++i) {
            if (mlir::failed(insertDecRefOnSuccessorEdge(builder, usedAlias,
                                                         lastUser, i)))
              return mlir::failure();
          }
        }
      }
    }

    for (mlir::Value alias : aliasSet) {
      for (mlir::OpOperand &use : alias.getUses()) {
        mlir::Operation *user = use.getOwner();

        if (isPyOwnershipIdentityTransform(user))
          continue;

        if (!consumesPyOwnedOperand(user, alias))
          continue;

        mlir::Block *userBlock = user->getBlock();
        const mlir::LivenessBlockInfo *blockLiveness =
            liveness.getLiveness(userBlock);
        if (!blockLiveness)
          continue;

        bool anyLiveOut = false;
        for (mlir::Value a : aliasSet) {
          if (blockLiveness->isLiveOut(a)) {
            anyLiveOut = true;
            break;
          }
        }

        bool isLastInBlock = (lastUserInBlock.count(userBlock) &&
                              lastUserInBlock[userBlock] == user);

        if (anyLiveOut || !isLastInBlock) {
          builder.setInsertionPoint(user);
          builder.create<IncRefOp>(alias.getLoc(), alias);
        }
      }
    }

    for (auto &[user, consumedOperands] : consumingOperandsByUser) {
      if (consumedOperands.size() <= 1)
        continue;

      mlir::Block *userBlock = user->getBlock();
      const mlir::LivenessBlockInfo *blockLiveness =
          liveness.getLiveness(userBlock);
      if (!blockLiveness)
        continue;

      bool anyLiveOut = false;
      for (mlir::Value alias : aliasSet) {
        if (blockLiveness->isLiveOut(alias)) {
          anyLiveOut = true;
          break;
        }
      }

      bool isLastInBlock = lastUserInBlock.count(userBlock) &&
                           lastUserInBlock[userBlock] == user;
      if (anyLiveOut || !isLastInBlock)
        continue;

      // The last consuming operation may move the existing token once. Any
      // additional consuming occurrence of the same ownership root needs a
      // matching retain so the affine token count remains non-negative.
      builder.setInsertionPoint(user);
      for (size_t i = 1, e = consumedOperands.size(); i < e; ++i)
        builder.create<IncRefOp>(consumedOperands[i].getLoc(),
                                 consumedOperands[i]);
    }

    if (mlir::failed(addDropRefInDivergentSuccessors(root, aliasSet, liveness)))
      return mlir::failure();

    return mlir::success();
  }

  mlir::LogicalResult
  addDropRefInDivergentSuccessors(mlir::Value root,
                                  llvm::ArrayRef<mlir::Value> aliasSet,
                                  mlir::Liveness &liveness) {
    using BlockSet = llvm::SmallPtrSet<mlir::Block *, 4>;
    mlir::OpBuilder builder(root.getContext());

    mlir::Region *definingRegion = root.getParentRegion();

    llvm::SmallDenseMap<mlir::Block *, BlockSet> divergentBlocks;

    for (mlir::Block &block : definingRegion->getBlocks()) {
      const mlir::LivenessBlockInfo *blockLiveness =
          liveness.getLiveness(&block);
      if (!blockLiveness)
        continue;

      bool anyLiveOut = false;
      for (mlir::Value alias : aliasSet) {
        if (blockLiveness->isLiveOut(alias)) {
          anyLiveOut = true;
          break;
        }
      }
      if (!anyLiveOut)
        continue;

      BlockSet liveInSuccessors;
      BlockSet noLiveInSuccessors;

      for (mlir::Block *succ : block.getSuccessors()) {
        const mlir::LivenessBlockInfo *succLiveness =
            liveness.getLiveness(succ);
        bool anyLiveIn = false;
        if (succLiveness) {
          for (mlir::Value alias : aliasSet) {
            if (succLiveness->isLiveIn(alias)) {
              anyLiveIn = true;
              break;
            }
          }
        }

        if (anyLiveIn)
          liveInSuccessors.insert(succ);
        else
          noLiveInSuccessors.insert(succ);
      }

      if (!liveInSuccessors.empty() && !noLiveInSuccessors.empty())
        divergentBlocks.try_emplace(&block, std::move(noLiveInSuccessors));
    }

    for (auto &[block, noLiveSuccessors] : divergentBlocks) {
      mlir::Operation *terminator = block->getTerminator();

      for (mlir::Block *successor : noLiveSuccessors) {
        for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
          if (terminator->getSuccessor(i) != successor)
            continue;
          if (mlir::failed(
                  insertDecRefOnSuccessorEdge(builder, root, terminator, i)))
            return mlir::failure();
        }
      }
    }

    return mlir::success();
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass() {
  return std::make_unique<RefCountInsertionPass>();
}

} // namespace py
