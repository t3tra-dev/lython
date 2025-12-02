#include "RuntimeSupport.h"

#include "mlir/Analysis/Liveness.h"
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

using namespace mlir;

namespace py {
namespace {

static bool needsRefCount(Type type) {
  if (isa<FuncSignatureType>(type) || isa<PrimFuncType>(type))
    return false;
  return isPyType(type);
}

static bool isImmortal(Operation *op) {
  // None is a singleton
  if (isa<NoneOp>(op))
    return true;
  // TODO: Handle dynamically created closures differently
  if (isa<FuncObjectOp>(op))
    return true;
  // Empty tuple is an immortal singleton (Ly_GetEmptyTuple)
  if (isa<TupleEmptyOp>(op))
    return true;
  return false;
}

static bool isIdentityTransform(Operation *op) {
  return isa<UpcastOp, CastIdentityOp>(op);
}

static bool createsNewRef(Operation *op) {
  if (isImmortal(op))
    return false;

  if (isIdentityTransform(op))
    return false;

  if (isa<TupleCreateOp, TupleEmptyOp, DictEmptyOp, DictInsertOp, StrConstantOp,
          IntConstantOp, FloatConstantOp, NumAddOp, NumSubOp, NumLeOp,
          MakeFunctionOp, CastFromPrimOp>(op))
    return true;

  if (isa<CallOp, CallVectorOp, NativeCallOp>(op))
    return true;

  return false;
}

static bool doesConsumeOperand(Operation *op, Value operand) {
  if (isa<ReturnOp>(op))
    return true;

  if (isa<TupleCreateOp>(op))
    return true;

  return false;
}

class AliasAnalysis {
public:
  AliasAnalysis(FuncOp func) {
    func.walk([&](Operation *op) {
      for (Value result : op->getResults()) {
        parent[result] = result;
      }
      for (Value operand : op->getOperands()) {
        if (!parent.count(operand))
          parent[operand] = operand;
      }
    });

    for (Block &block : func.getBody()) {
      for (BlockArgument arg : block.getArguments()) {
        parent[arg] = arg;
      }
    }

    func.walk([&](Operation *op) {
      if (!isIdentityTransform(op))
        return;

      Value input = op->getOperand(0);
      Value output = op->getResult(0);
      unionSets(input, output);
    });

    for (auto &kv : parent) {
      Value root = find(kv.first);
      members[root].push_back(kv.first);
    }
  }

  Value getRoot(Value v) const { return find(v); }

  void collectAliases(Value v, llvm::SmallVectorImpl<Value> &aliases) const {
    Value root = find(v);
    auto it = members.find(root);
    if (it != members.end()) {
      aliases.append(it->second.begin(), it->second.end());
    } else {
      aliases.push_back(v);
    }
  }

private:
  Value find(Value v) const {
    auto it = parent.find(v);
    if (it == parent.end())
      return v;
    if (it->second == v)
      return v;
    Value root = find(it->second);
    parent[v] = root;
    return root;
  }

  void unionSets(Value a, Value b) {
    Value rootA = find(a);
    Value rootB = find(b);
    if (rootA != rootB) {
      parent[rootB] = rootA;
    }
  }

  mutable llvm::DenseMap<Value, Value> parent;
  llvm::DenseMap<Value, llvm::SmallVector<Value, 4>> members;
};

struct RefCountInsertionPass
    : public PassWrapper<RefCountInsertionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountInsertionPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();

    module.walk([&](FuncOp func) {
      if (failed(processFunction(func)))
        signalPassFailure();
    });
  }

private:
  LogicalResult processFunction(FuncOp func) {
    if (func.getBody().empty())
      return success();

    AliasAnalysis aliases(func);
    Liveness liveness(func);
    llvm::DenseSet<Value> processedRoots;

    // Skip function entry block arguments - they are borrowed references
    // and should not be decremented by the callee. The caller owns them.
    Block &entryBlock = func.getBody().front();
    for (BlockArgument arg : entryBlock.getArguments()) {
      if (needsRefCount(arg.getType())) {
        Value root = aliases.getRoot(arg);
        processedRoots.insert(root);
      }
    }

    for (Block &block : func.getBody()) {
      // Skip entry block args since they're borrowed refs (handled above)
      if (&block == &entryBlock)
        continue;

      for (BlockArgument arg : block.getArguments()) {
        if (!needsRefCount(arg.getType()))
          continue;

        Value root = aliases.getRoot(arg);
        if (processedRoots.insert(root).second) {
          if (failed(addRefCountingForAliasSet(root, aliases, liveness)))
            return failure();
        }
      }
    }

    func.walk([&](Operation *op) {
      if (!createsNewRef(op))
        return;

      for (Value result : op->getResults()) {
        if (!needsRefCount(result.getType()))
          continue;

        Value root = aliases.getRoot(result);
        if (processedRoots.insert(root).second) {
          if (failed(addRefCountingForAliasSet(root, aliases, liveness)))
            signalPassFailure();
        }
      }
    });

    return success();
  }

  LogicalResult addRefCountingForAliasSet(Value root, AliasAnalysis &aliases,
                                          Liveness &liveness) {
    OpBuilder builder(root.getContext());

    llvm::SmallVector<Value, 4> aliasSet;
    aliases.collectAliases(root, aliasSet);

    llvm::SmallPtrSet<Operation *, 8> allUsers;
    for (Value alias : aliasSet) {
      for (Operation *user : alias.getUsers()) {
        if (!isIdentityTransform(user))
          allUsers.insert(user);
      }
    }

    if (allUsers.empty()) {
      if (Operation *defOp = root.getDefiningOp()) {
        builder.setInsertionPointAfter(defOp);
      } else {
        builder.setInsertionPointToStart(root.getParentBlock());
      }
      builder.create<DecRefOp>(root.getLoc(), root);
      return success();
    }

    Region *definingRegion = root.getParentRegion();

    llvm::DenseMap<Block *, Operation *> lastUserInBlock;
    for (Operation *user : allUsers) {
      Block *userBlock = user->getBlock();
      Block *ancestor = definingRegion->findAncestorBlockInRegion(*userBlock);
      if (!ancestor)
        continue;

      Operation *ancestorOp = ancestor->findAncestorOpInBlock(*user);
      if (!ancestorOp)
        continue;

      auto it = lastUserInBlock.find(ancestor);
      if (it == lastUserInBlock.end() ||
          it->second->isBeforeInBlock(ancestorOp)) {
        lastUserInBlock[ancestor] = ancestorOp;
      }
    }

    for (auto &[block, lastUser] : lastUserInBlock) {
      const LivenessBlockInfo *blockLiveness = liveness.getLiveness(block);
      if (!blockLiveness)
        continue;

      bool anyLiveOut = false;
      for (Value alias : aliasSet) {
        if (blockLiveness->isLiveOut(alias)) {
          anyLiveOut = true;
          break;
        }
      }

      if (anyLiveOut)
        continue; // Value doesn't die in this block

      Value usedAlias = root;
      for (Value alias : aliasSet) {
        for (Value operand : lastUser->getOperands()) {
          if (operand == alias) {
            usedAlias = alias;
            break;
          }
        }
      }

      // Apply Affine SSA rules
      if (doesConsumeOperand(lastUser, usedAlias)) {
        // Last use + Consume = Move Semantics (no action needed)
      } else {
        // Last use + Borrow = Need to drop after use
        builder.setInsertionPointAfter(lastUser);
        builder.create<DecRefOp>(root.getLoc(), usedAlias);
      }
    }

    for (Value alias : aliasSet) {
      for (OpOperand &use : alias.getUses()) {
        Operation *user = use.getOwner();

        if (isIdentityTransform(user))
          continue;

        if (!doesConsumeOperand(user, alias))
          continue;

        Block *userBlock = user->getBlock();
        const LivenessBlockInfo *blockLiveness =
            liveness.getLiveness(userBlock);
        if (!blockLiveness)
          continue;

        bool anyLiveOut = false;
        for (Value a : aliasSet) {
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

    if (failed(addDropRefInDivergentSuccessors(root, aliasSet, liveness)))
      return failure();

    return success();
  }

  LogicalResult addDropRefInDivergentSuccessors(Value root,
                                                llvm::ArrayRef<Value> aliasSet,
                                                Liveness &liveness) {
    using BlockSet = llvm::SmallPtrSet<Block *, 4>;
    OpBuilder builder(root.getContext());

    Region *definingRegion = root.getParentRegion();

    llvm::SmallDenseMap<Block *, BlockSet> divergentBlocks;

    for (Block &block : definingRegion->getBlocks()) {
      const LivenessBlockInfo *blockLiveness = liveness.getLiveness(&block);
      if (!blockLiveness)
        continue;

      bool anyLiveOut = false;
      for (Value alias : aliasSet) {
        if (blockLiveness->isLiveOut(alias)) {
          anyLiveOut = true;
          break;
        }
      }
      if (!anyLiveOut)
        continue;

      BlockSet liveInSuccessors;
      BlockSet noLiveInSuccessors;

      for (Block *succ : block.getSuccessors()) {
        const LivenessBlockInfo *succLiveness = liveness.getLiveness(succ);
        bool anyLiveIn = false;
        if (succLiveness) {
          for (Value alias : aliasSet) {
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
      Operation *terminator = block->getTerminator();

      for (Block *successor : noLiveSuccessors) {
        if (successor->getUniquePredecessor() == block) {
          builder.setInsertionPointToStart(successor);
          builder.create<DecRefOp>(root.getLoc(), root);
        } else {
          Block *refCountBlock = new Block();
          refCountBlock->insertBefore(successor);

          builder.setInsertionPointToStart(refCountBlock);
          builder.create<DecRefOp>(root.getLoc(), root);
          builder.create<cf::BranchOp>(root.getLoc(), successor);

          for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
            if (terminator->getSuccessor(i) == successor)
              terminator->setSuccessor(refCountBlock, i);
          }
        }
      }
    }

    return success();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createRefCountInsertionPass() {
  return std::make_unique<RefCountInsertionPass>();
}

} // namespace py
