#include "Ownership.h"
#include "Common/Instrumentation.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"
#include "Runtime/Model/Contracts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

#include <cstdint>
#include <memory>
#include <optional>

namespace py::lowering {
namespace {

namespace own = py::ownership;

std::optional<std::string> callableResultContractAtOffset(
    mlir::func::FuncOp function, unsigned resultOffset,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators) {
  auto callableAttr =
      function->getAttrOfType<mlir::TypeAttr>(own::kCallableTypeAttr);
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable)
    return std::nullopt;

  unsigned offset = 0;
  for (mlir::Type resultType : callable.getResultTypes()) {
    std::string contract = runtimeContractName(resultType);
    if (contract.empty())
      return std::nullopt;
    const own::RuntimeDeallocator *deallocator = nullptr;
    for (const own::RuntimeDeallocator &candidate : deallocators) {
      if (candidate.contractName == contract) {
        deallocator = &candidate;
        break;
      }
    }
    if (!deallocator)
      return std::nullopt;
    if (offset == resultOffset)
      return contract;
    offset += static_cast<unsigned>(deallocator->inputTypes.size());
  }
  return std::nullopt;
}

mlir::func::FuncOp findRetainFunction(mlir::ModuleOp module) {
  mlir::func::FuncOp retained;
  module.walk([&](mlir::func::FuncOp function) {
    auto primitive =
        function->getAttrOfType<mlir::StringAttr>(kManifestPrimitiveAttr);
    if (!primitive || primitive.getValue() != "retain")
      return;
    retained = function;
  });
  return retained;
}

struct CachedFuncContract {
  mlir::func::FuncOp function;
  own::FunctionContract contract;
};

class FuncContractCache {
public:
  explicit FuncContractCache(mlir::ModuleOp module) {
    module.walk([&](mlir::func::FuncOp function) {
      functions.insert({function.getSymName(), function});
    });
  }

  mlir::FailureOr<const CachedFuncContract *> lookup(llvm::StringRef name) {
    auto cached = contracts.find(name);
    if (cached != contracts.end())
      return &cached->second;

    auto function = functions.find(name);
    if (function == functions.end())
      return static_cast<const CachedFuncContract *>(nullptr);

    auto contract = own::readFunctionContract(function->second);
    if (mlir::failed(contract))
      return mlir::failure();

    CachedFuncContract entry{function->second, *contract};
    auto inserted = contracts.insert({name, std::move(entry)});
    return &inserted.first->second;
  }

  mlir::FailureOr<const CachedFuncContract *>
  lookup(mlir::func::FuncOp function) {
    if (!function)
      return static_cast<const CachedFuncContract *>(nullptr);
    return lookup(function.getSymName());
  }

private:
  llvm::StringMap<mlir::func::FuncOp> functions;
  llvm::StringMap<CachedFuncContract> contracts;
};

mlir::FailureOr<mlir::Value> buildRetainHeaderView(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value header,
                                                   mlir::Type retainInputType) {
  if (header.getType() == retainInputType)
    return header;

  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  auto targetType = mlir::dyn_cast<mlir::MemRefType>(retainInputType);
  if (!sourceType || !targetType)
    return mlir::failure();
  if (sourceType.getRank() != 1 || targetType.getRank() != 1)
    return mlir::failure();
  if (sourceType.getElementType() != targetType.getElementType())
    return mlir::failure();

  if (sourceType.getDimSize(0) == targetType.getDimSize(0))
    return mlir::memref::CastOp::create(builder, loc, retainInputType, header)
        .getResult();

  if (sourceType.hasStaticShape() && targetType.hasStaticShape() &&
      sourceType.getDimSize(0) >= targetType.getDimSize(0)) {
    llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
    llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
        builder.getIndexAttr(targetType.getDimSize(0))};
    llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
    llvm::SmallVector<int64_t, 1> resultShape{targetType.getDimSize(0)};
    auto inferredType = mlir::cast<mlir::MemRefType>(
        mlir::memref::SubViewOp::inferRankReducedResultType(
            resultShape, sourceType, offsets, sizes, strides));
    mlir::Value view =
        mlir::memref::SubViewOp::create(builder, loc, inferredType, header,
                                        offsets, sizes, strides)
            .getResult();
    if (view.getType() == targetType)
      return view;
    return mlir::memref::CastOp::create(builder, loc, targetType, view)
        .getResult();
  }

  return mlir::failure();
}

mlir::LogicalResult insertRetain(mlir::func::FuncOp retain,
                                 mlir::func::ReturnOp returnOp,
                                 mlir::Value header) {
  if (!retain)
    return returnOp.emitError()
           << "borrowed object return requires a runtime retain primitive";
  if (retain.getFunctionType().getNumInputs() != 1)
    return retain.emitError()
           << "runtime retain primitive must accept one object header";

  mlir::OpBuilder builder(returnOp);
  mlir::FailureOr<mlir::Value> headerView = buildRetainHeaderView(
      builder, returnOp.getLoc(), header, retain.getFunctionType().getInput(0));
  if (mlir::failed(headerView))
    return returnOp.emitError()
           << "cannot build object retain view for borrowed return";

  mlir::func::CallOp::create(builder, returnOp.getLoc(), retain, *headerView);
  return mlir::success();
}

bool valueAliasesEntryArgument(mlir::Value value, mlir::Block &entry,
                               own::AliasAnalysis &aliases) {
  for (mlir::BlockArgument argument : entry.getArguments())
    if (aliases.same(value, argument))
      return true;
  return false;
}

bool valueDerivedFromEntryArgument(mlir::Value value, mlir::Block &entry,
                                   own::AliasAnalysis &aliases,
                                   unsigned depth = 0) {
  if (!value || depth > 8)
    return false;
  if (valueAliasesEntryArgument(value, entry, aliases))
    return true;

  auto select = value.getDefiningOp<mlir::arith::SelectOp>();
  if (!select)
    return false;
  return valueDerivedFromEntryArgument(select.getTrueValue(), entry, aliases,
                                       depth + 1) &&
         valueDerivedFromEntryArgument(select.getFalseValue(), entry, aliases,
                                       depth + 1);
}

bool valueGroupDerivedFromEntryArguments(mlir::func::FuncOp function,
                                         llvm::ArrayRef<mlir::Value> group,
                                         own::AliasAnalysis &aliases) {
  if (function.empty() || group.empty())
    return false;
  mlir::Block &entry = function.front();
  for (mlir::Value value : group)
    if (!valueDerivedFromEntryArgument(value, entry, aliases))
      return false;
  return true;
}

mlir::LogicalResult insertBorrowedReturnRetains(
    mlir::ModuleOp module, mlir::func::FuncOp retain,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    if (mlir::failed(result))
      return;
    if (!own::functionUsesOwnedReturnABI(function))
      return;

    function.walk([&](mlir::func::ReturnOp returnOp) {
      unsigned offset = 0;
      while (offset < returnOp.getNumOperands()) {
        std::optional<std::string> logicalContract =
            callableResultContractAtOffset(function, offset, deallocators);
        const own::RuntimeDeallocator *deallocator =
            logicalContract
                ? own::findDeallocatorForValueGroup(returnOp.getOperands(),
                                                    offset, deallocators,
                                                    *logicalContract)
                : own::findDeallocatorForValueGroup(returnOp.getOperands(),
                                                    offset, deallocators);
        if (!deallocator) {
          ++offset;
          continue;
        }

        llvm::SmallVector<mlir::Value, 4> group = own::valueSlice(
            returnOp.getOperands(), offset,
            static_cast<unsigned>(deallocator->inputTypes.size()));
        if (own::valueGroupEqualsEntryArgumentGroup(function, group) ||
            valueGroupDerivedFromEntryArguments(function, group, aliases)) {
          if (mlir::failed(insertRetain(retain, returnOp, group.front()))) {
            result = mlir::failure();
            return;
          }
        }
        offset += static_cast<unsigned>(deallocator->inputTypes.size());
      }
    });
  });
  return result;
}

bool isOwnershipConsumingUse(FuncContractCache &contracts,
                             mlir::OpOperand &use) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner());
  if (!call)
    return false;
  auto cached = contracts.lookup(call.getCallee());
  return mlir::succeeded(cached) && *cached &&
         (*cached)->contract.consumesArg(
             static_cast<unsigned>(use.getOperandNumber()));
}

bool groupMatchesOperands(mlir::OperandRange operands, unsigned offset,
                          llvm::ArrayRef<mlir::Value> group,
                          own::AliasAnalysis &aliases) {
  if (offset + group.size() > operands.size())
    return false;
  for (auto [index, value] : llvm::enumerate(group)) {
    if (!aliases.same(operands[offset + index], value))
      return false;
  }
  return true;
}

bool callConsumesGroup(FuncContractCache &contracts, mlir::func::CallOp call,
                       llvm::ArrayRef<mlir::Value> group,
                       own::AliasAnalysis &aliases) {
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  for (unsigned offset : (*cached)->contract.releaseArgs.values)
    if (groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : (*cached)->contract.transferArgs.values)
    if (groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

bool callConsumesTrackedHeader(FuncContractCache &contracts,
                               mlir::func::CallOp call,
                               llvm::ArrayRef<mlir::Value> group,
                               own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  auto consumesHeaderAt = [&](unsigned offset) {
    return offset < call.getNumOperands() &&
           aliases.same(call.getOperand(offset), group.front());
  };
  for (unsigned offset : (*cached)->contract.releaseArgs.values)
    if (consumesHeaderAt(offset) &&
        !groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : (*cached)->contract.transferArgs.values)
    if (consumesHeaderAt(offset) &&
        !groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

bool ownershipConsumingUseInvalidatesGroup(FuncContractCache &contracts,
                                           mlir::OpOperand &use,
                                           llvm::ArrayRef<mlir::Value> group,
                                           own::AliasAnalysis &aliases) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner());
  if (!call)
    return false;
  return callConsumesGroup(contracts, call, group, aliases) ||
         callConsumesTrackedHeader(contracts, call, group, aliases);
}

mlir::Operation *latestUserInBlock(mlir::Operation *lhs, mlir::Operation *rhs) {
  if (!lhs)
    return rhs;
  return lhs->isBeforeInBlock(rhs) ? rhs : lhs;
}

bool returnTransfersGroup(FuncContractCache &contracts,
                          mlir::func::FuncOp function,
                          mlir::func::ReturnOp returnOp,
                          llvm::ArrayRef<mlir::Value> group,
                          llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                          own::AliasAnalysis &aliases) {
  auto cached = contracts.lookup(function);
  if (mlir::succeeded(cached) && *cached) {
    for (unsigned offset : (*cached)->contract.ownedResults.values)
      if (own::groupMatchesValues(returnOp.getOperands(), offset, group,
                                  aliases))
        return true;
  }

  if (!own::functionUsesOwnedReturnABI(function)) {
    return false;
  }

  std::optional<llvm::SmallVector<own::OwnedReturnRange, 4>> ranges =
      own::callableOwnedReturnRanges(function, returnOp.getOperands(),
                                     deallocators);
  if (!ranges) {
    for (unsigned offset = 0;
         offset + group.size() <= returnOp.getNumOperands(); ++offset)
      if (own::groupMatchesValues(returnOp.getOperands(), offset, group,
                                  aliases))
        return true;
    return false;
  }
  for (const own::OwnedReturnRange &range : *ranges)
    if (own::groupMatchesOwnedReturnRange(returnOp.getOperands(), range, group,
                                          deallocators, aliases))
      return true;
  return false;
}

mlir::Operation *ancestorInBlock(mlir::Operation *op, mlir::Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op && op->getBlock() == block ? op : nullptr;
}

// The ancestor operation whose block belongs directly to `region` (the op
// itself when already top-level there); null when `op` is not nested inside
// the region at all.
mlir::Operation *ancestorInRegion(mlir::Operation *op, mlir::Region *region) {
  while (op && op->getBlock() && op->getBlock()->getParent() != region)
    op = op->getParentOp();
  return op && op->getBlock() && op->getBlock()->getParent() == region
             ? op
             : nullptr;
}

llvm::SmallVector<mlir::Value, 4> remapGroupThroughValueMapping(
    mlir::ValueRange sources, mlir::ValueRange targets,
    llvm::ArrayRef<mlir::Value> group, own::AliasAnalysis &aliases,
    llvm::SmallVectorImpl<bool> *mappedMask = nullptr) {
  llvm::SmallVector<mlir::Value, 4> mapped(group.begin(), group.end());
  if (mappedMask) {
    mappedMask->clear();
    mappedMask->append(group.size(), false);
  }

  unsigned count = std::min<unsigned>(sources.size(), targets.size());
  for (auto [groupIndex, value] : llvm::enumerate(group)) {
    for (unsigned index = 0; index < count; ++index) {
      if (!sources[index] || !targets[index] ||
          !aliases.same(sources[index], value))
        continue;
      mapped[groupIndex] = targets[index];
      if (mappedMask)
        (*mappedMask)[groupIndex] = true;
      break;
    }
  }
  return mapped;
}

std::optional<llvm::SmallVector<mlir::Value, 4>>
mapRegionTerminatorGroupToParentResults(mlir::Operation *terminator,
                                        llvm::ArrayRef<mlir::Value> group,
                                        own::AliasAnalysis &aliases) {
  if (!terminator->hasTrait<mlir::OpTrait::IsTerminator>())
    return std::nullopt;
  mlir::Region *region = terminator->getParentRegion();
  mlir::Operation *owner = region ? region->getParentOp() : nullptr;
  if (!owner || mlir::isa<mlir::func::FuncOp>(owner) ||
      owner->getNumResults() == 0)
    return std::nullopt;

  llvm::SmallVector<bool, 4> mappedMask;
  llvm::SmallVector<mlir::Value, 4> mapped = remapGroupThroughValueMapping(
      terminator->getOperands(), owner->getResults(), group, aliases,
      &mappedMask);
  if (!llvm::all_of(mappedMask, [](bool mapped) { return mapped; }))
    return std::nullopt;
  return mapped;
}

bool branchForwardsGroupToBlockArgument(mlir::Operation *terminator,
                                        llvm::ArrayRef<mlir::Value> group,
                                        own::AliasAnalysis &aliases) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return false;

  for (unsigned successorIndex = 0,
                successorCount = terminator->getNumSuccessors();
       successorIndex < successorCount; ++successorIndex) {
    mlir::Block *successor = terminator->getSuccessor(successorIndex);
    if (!successor || successor->getNumArguments() == 0)
      continue;

    mlir::SuccessorOperands operands =
        branch.getSuccessorOperands(successorIndex);
    unsigned argumentCount =
        std::min<unsigned>(successor->getNumArguments(), operands.size());
    for (unsigned argumentIndex = 0; argumentIndex < argumentCount;
         ++argumentIndex) {
      mlir::Value forwarded = operands[argumentIndex];
      if (!forwarded)
        continue;
      for (mlir::Value value : group)
        if (aliases.same(forwarded, value))
          return true;
    }
  }
  return false;
}

bool sameExactGroup(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left != right)
      return false;
  return true;
}

bool usePrecedesOwnerInBlock(mlir::Operation *owner, mlir::Operation *user,
                             mlir::Block *ownerBlock) {
  mlir::Operation *blockUser = ancestorInBlock(user, ownerBlock);
  return blockUser && blockUser != owner && blockUser->isBeforeInBlock(owner);
}

struct ReleaseInsertion {
  mlir::Operation *after = nullptr;
  mlir::Operation *before = nullptr;
  llvm::SmallVector<mlir::Value, 4> group;
};

mlir::Block *releaseInsertionBlock(const ReleaseInsertion &release) {
  if (release.before)
    return release.before->getBlock();
  return release.after ? release.after->getBlock() : nullptr;
}

std::optional<ReleaseInsertion>
mergeReleaseInsertion(std::optional<ReleaseInsertion> current,
                      ReleaseInsertion next) {
  if (!current)
    return next;
  if (!sameExactGroup(current->group, next.group))
    return std::nullopt;
  if (releaseInsertionBlock(*current) != releaseInsertionBlock(next))
    return std::nullopt;
  if (current->before || next.before) {
    if (current->before && next.before && current->before != next.before)
      return std::nullopt;
    if (!current->before) {
      current->before = next.before;
      current->after = nullptr;
    }
    return current;
  }
  current->after = latestUserInBlock(current->after, next.after);
  return current;
}

std::optional<ReleaseInsertion>
findReleaseInsertion(FuncContractCache &contracts, mlir::Operation *owner,
                     llvm::ArrayRef<mlir::Value> group,
                     llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                     own::AliasAnalysis &aliases, unsigned depth = 0,
                     llvm::ArrayRef<mlir::Value> views = {}) {
  if (!owner || group.empty() || depth > 16)
    return std::nullopt;
  mlir::Block *block = owner->getBlock();
  if (!block)
    return std::nullopt;

  // Box-word reconstructions (borrowed memref views assembled from the
  // entity's box words) pin liveness exactly like canonical-shape views. The
  // loads typically read the raw storage value, so walk alias equivalents.
  llvm::SmallVector<mlir::Value, 8> pinnedViews(views.begin(), views.end());
  {
    llvm::SmallVector<mlir::Value, 8> groupEquivalents;
    for (mlir::Value result : group) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.aliasesOf(result, equivalentValues);
      if (equivalentValues.empty())
        equivalentValues.push_back(result);
      groupEquivalents.append(equivalentValues.begin(),
                              equivalentValues.end());
    }
    own::collectBoxWordDerivedViews(groupEquivalents, pinnedViews);
  }

  mlir::Operation *lastUser = nullptr;
  // Interior views pin the entity. Terminator uses are judged by the main
  // group walk (region forwards recurse with the views mapped alongside);
  // everything else contributes plain liveness.
  for (mlir::Value view : pinnedViews) {
    for (mlir::OpOperand &use : view.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == owner)
        continue;
      if (usePrecedesOwnerInBlock(owner, user, block))
        continue;
      mlir::Operation *blockUser = ancestorInBlock(user, block);
      if (!blockUser)
        return std::nullopt;
      if (blockUser == owner ||
          blockUser->hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      lastUser = latestUserInBlock(lastUser, blockUser);
    }
  }
  std::optional<ReleaseInsertion> forwardedRelease;
  std::optional<ReleaseInsertion> terminalRelease;
  for (mlir::Value result : group) {
    llvm::SmallVector<mlir::Value, 8> equivalentValues;
    aliases.aliasesOf(result, equivalentValues);
    if (equivalentValues.empty())
      equivalentValues.push_back(result);

    for (mlir::Value equivalent : equivalentValues) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (user == owner)
          continue;
        if (usePrecedesOwnerInBlock(owner, user, block))
          continue;
        if (ownershipConsumingUseInvalidatesGroup(contracts, use, group,
                                                  aliases))
          return std::nullopt;

        if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(user)) {
          mlir::func::FuncOp function =
              returnOp->getParentOfType<mlir::func::FuncOp>();
          if (function && returnTransfersGroup(contracts, function, returnOp,
                                               group, deallocators, aliases))
            return std::nullopt;
          mlir::Operation *blockUser = ancestorInBlock(user, block);
          if (!blockUser)
            return std::nullopt;
          ReleaseInsertion release;
          release.before = blockUser;
          release.group.append(group.begin(), group.end());
          terminalRelease =
              mergeReleaseInsertion(std::move(terminalRelease), release);
          if (!terminalRelease)
            return std::nullopt;
          continue;
        }

        if (user->hasTrait<mlir::OpTrait::IsTerminator>()) {
          if (std::optional<llvm::SmallVector<mlir::Value, 4>> mapped =
                  mapRegionTerminatorGroupToParentResults(user, group,
                                                          aliases)) {
            mlir::Operation *regionOwner =
                user->getParentRegion() ? user->getParentRegion()->getParentOp()
                                        : nullptr;
            llvm::SmallVector<mlir::Value, 4> mappedViews;
            if (!views.empty() && regionOwner) {
              llvm::SmallVector<bool, 4> viewMask;
              mappedViews = remapGroupThroughValueMapping(
                  user->getOperands(), regionOwner->getResults(), views,
                  aliases, &viewMask);
              llvm::SmallVector<mlir::Value, 4> escaped;
              for (auto [index, isMapped] : llvm::enumerate(viewMask))
                if (isMapped)
                  escaped.push_back(mappedViews[index]);
              mappedViews = std::move(escaped);
            }
            std::optional<ReleaseInsertion> release =
                findReleaseInsertion(contracts, regionOwner, *mapped,
                                     deallocators, aliases, depth + 1,
                                     mappedViews);
            if (!release)
              return std::nullopt;
            forwardedRelease =
                mergeReleaseInsertion(std::move(forwardedRelease), *release);
            if (!forwardedRelease)
              return std::nullopt;
            continue;
          }
          if (branchForwardsGroupToBlockArgument(user, group, aliases))
            return std::nullopt;
          return std::nullopt;
        }

        mlir::Operation *blockUser = ancestorInBlock(user, block);
        if (!blockUser)
          return std::nullopt;
        if (blockUser == owner)
          continue;
        if (blockUser->hasTrait<mlir::OpTrait::IsTerminator>())
          return std::nullopt;
        lastUser = latestUserInBlock(lastUser, blockUser);
      }
    }
  }

  if (forwardedRelease)
    return forwardedRelease;
  if (terminalRelease)
    return terminalRelease;

  ReleaseInsertion release;
  release.after = lastUser ? lastUser : owner;
  release.group.append(group.begin(), group.end());
  return release;
}

bool groupContainsOperand(mlir::Operation *op,
                          llvm::ArrayRef<mlir::Value> group,
                          own::AliasAnalysis &aliases) {
  for (mlir::Value operand : op->getOperands())
    for (mlir::Value value : group)
      if (aliases.same(operand, value))
        return true;
  return false;
}

struct ConditionalReleaseBlocks {
  mlir::Block *active = nullptr;
  mlir::Block *inactive = nullptr;
  mlir::Operation *branch = nullptr;
  unsigned activeSuccessor = 0;
  unsigned inactiveSuccessor = 0;
};

std::optional<ConditionalReleaseBlocks>
classifyConditionalBranch(mlir::Operation *op,
                          const own::OwnershipCondition &condition) {
  auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(op);
  if (!branch)
    return std::nullopt;

  std::optional<own::OwnershipConditionBranch> classified =
      own::classifyOwnershipConditionBranch(op, condition);
  if (!classified)
    return std::nullopt;

  ConditionalReleaseBlocks blocks;
  blocks.branch = branch.getOperation();
  blocks.active = classified->activeSuccessor == 0 ? branch.getTrueDest()
                                                   : branch.getFalseDest();
  blocks.inactive = classified->inactiveSuccessor == 0 ? branch.getTrueDest()
                                                       : branch.getFalseDest();
  blocks.activeSuccessor = classified->activeSuccessor;
  blocks.inactiveSuccessor = classified->inactiveSuccessor;
  return blocks;
}

std::optional<ConditionalReleaseBlocks> findConditionalBranchAfterOwner(
    mlir::Operation *owner, llvm::ArrayRef<mlir::Value> group,
    const own::OwnershipCondition &condition, own::AliasAnalysis &aliases) {
  for (mlir::Operation *op = owner->getNextNode(); op; op = op->getNextNode()) {
    if (std::optional<ConditionalReleaseBlocks> blocks =
            classifyConditionalBranch(op, condition))
      return blocks;
    if (groupContainsOperand(op, group, aliases))
      return std::nullopt;
    if (op->hasTrait<mlir::OpTrait::IsTerminator>())
      return std::nullopt;
  }
  return std::nullopt;
}

std::optional<mlir::Operation *> findLastConditionalUserInActiveBlock(
    FuncContractCache &contracts, mlir::Operation *owner,
    llvm::ArrayRef<mlir::Value> group, const ConditionalReleaseBlocks &blocks,
    own::AliasAnalysis &aliases) {
  mlir::Operation *lastUser = nullptr;
  for (mlir::Value value : group) {
    for (mlir::OpOperand &use : value.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == owner || user == blocks.branch)
        continue;
      if (isOwnershipConsumingUse(contracts, use))
        return std::nullopt;

      mlir::Operation *activeUser = ancestorInBlock(user, blocks.active);
      if (activeUser) {
        if (activeUser->hasTrait<mlir::OpTrait::IsTerminator>())
          return std::nullopt;
        lastUser = latestUserInBlock(lastUser, activeUser);
        continue;
      }

      if (ancestorInBlock(user, blocks.inactive))
        return std::nullopt;
      return std::nullopt;
    }
  }
  return lastUser;
}

mlir::LogicalResult
insertReleaseOnActiveEdge(mlir::func::CallOp call,
                          const own::ResourceGroup &group,
                          ConditionalReleaseBlocks &blocks) {
  auto branch = mlir::cast<mlir::cf::CondBranchOp>(blocks.branch);
  if ((blocks.activeSuccessor == 0 && !branch.getTrueDestOperands().empty()) ||
      (blocks.activeSuccessor == 1 && !branch.getFalseDestOperands().empty()))
    return mlir::success();

  mlir::OpBuilder builder(call.getContext());
  mlir::Block *releaseBlock = builder.createBlock(blocks.active->getParent(),
                                                  blocks.active->getIterator());
  builder.setInsertionPointToStart(releaseBlock);
  mlir::func::CallOp::create(builder, call.getLoc(),
                             group.deallocator->function, group.values);
  mlir::cf::BranchOp::create(builder, call.getLoc(), blocks.active);

  llvm::SmallVector<mlir::Value, 4> trueOperands(
      branch.getTrueDestOperands().begin(), branch.getTrueDestOperands().end());
  llvm::SmallVector<mlir::Value, 4> falseOperands(
      branch.getFalseDestOperands().begin(),
      branch.getFalseDestOperands().end());
  builder.setInsertionPoint(branch);
  if (blocks.activeSuccessor == 0) {
    mlir::cf::CondBranchOp::create(
        builder, branch.getLoc(), branch.getCondition(), releaseBlock,
        mlir::ValueRange{}, branch.getFalseDest(), falseOperands);
  } else {
    mlir::cf::CondBranchOp::create(
        builder, branch.getLoc(), branch.getCondition(), branch.getTrueDest(),
        trueOperands, releaseBlock, mlir::ValueRange{});
  }
  branch.erase();
  return mlir::success();
}

mlir::FailureOr<bool> insertConditionalOwnedResultRelease(
    FuncContractCache &contracts, mlir::func::CallOp call,
    const own::ResourceGroup &group, own::AliasAnalysis &aliases) {
  if (!group.condition)
    return false;

  std::optional<ConditionalReleaseBlocks> blocks =
      findConditionalBranchAfterOwner(call, group.values, *group.condition,
                                      aliases);
  if (!blocks)
    return false;

  std::optional<mlir::Operation *> lastUser =
      findLastConditionalUserInActiveBlock(contracts, call, group.values,
                                           *blocks, aliases);
  if (!lastUser)
    return false;

  mlir::OpBuilder builder(call);
  if (*lastUser) {
    builder.setInsertionPointAfter(*lastUser);
  } else if (llvm::hasSingleElement(blocks->active->getPredecessors())) {
    builder.setInsertionPointToStart(blocks->active);
  } else {
    if (mlir::failed(insertReleaseOnActiveEdge(call, group, *blocks)))
      return mlir::failure();
    return true;
  }
  mlir::func::CallOp::create(builder, call.getLoc(),
                             group.deallocator->function, group.values);
  return true;
}

// Can `start` reach `target` by following successors without ever entering
// `avoid`? Used to reject loop-invariant values from the per-successor release
// path: if a using successor can re-reach itself without passing back through
// the value's defining block, the value is NOT re-defined on that cycle and a
// release after the use would be a use-after-release next iteration.
bool blockReachesAvoiding(mlir::Block *start, mlir::Block *target,
                          mlir::Block *avoid) {
  llvm::SmallVector<mlir::Block *, 8> worklist(start->succ_begin(),
                                               start->succ_end());
  llvm::SmallPtrSet<mlir::Block *, 8> visited;
  while (!worklist.empty()) {
    mlir::Block *block = worklist.pop_back_val();
    if (block == avoid)
      continue;
    if (block == target)
      return true;
    if (!visited.insert(block).second)
      continue;
    worklist.append(block->succ_begin(), block->succ_end());
  }
  return false;
}

// Release an unconditionally-owned group whose uses are confined to the
// immediate successors of the block that defines it. This is the
// loop-produced-value pattern: e.g. the `py.next` element is defined in the
// loop-header block and consumed only in the loop body, so it is dead on the
// back-edge and on the non-consuming (loop-exit) successor edge. The value must
// therefore be released after its last use in each using successor and at the
// entry of each non-using successor. This conservative version only handles
// non-using successors that have a single predecessor (so releasing at the
// successor entry needs no edge split); anything else is left to the caller and
// re-checked by the affine ownership verifier. Returns true when handled.
bool insertImmediateSuccessorReleases(FuncContractCache &contracts,
                                      mlir::func::CallOp call,
                                      const own::ResourceGroup &group,
                                      own::AliasAnalysis &aliases) {
  if (!group.deallocator || group.condition)
    return false;
  mlir::Block *defBlock = call->getBlock();
  if (!defBlock)
    return false;
  mlir::Operation *terminator = defBlock->getTerminator();
  if (!terminator || terminator->getNumSuccessors() == 0)
    return false;
  if (!mlir::isa<mlir::cf::CondBranchOp>(terminator) &&
      !mlir::isa<mlir::cf::BranchOp>(terminator))
    return false;
  if (branchForwardsGroupToBlockArgument(terminator, group.values, aliases))
    return false;

  llvm::SmallVector<mlir::Block *, 2> successors;
  for (mlir::Block *successor : terminator->getSuccessors())
    if (!llvm::is_contained(successors, successor))
      successors.push_back(successor);

  llvm::SmallDenseMap<mlir::Block *, mlir::Operation *, 2> lastUser;
  for (mlir::Block *successor : successors)
    lastUser.try_emplace(successor, nullptr);

  for (mlir::Value result : group.values) {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.aliasesOf(result, equivalents);
    if (equivalents.empty())
      equivalents.push_back(result);
    for (mlir::Value equivalent : equivalents) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (user == call.getOperation() || user == terminator)
          continue;
        if (ownershipConsumingUseInvalidatesGroup(contracts, use, group.values,
                                                  aliases))
          return false;
        if (mlir::isa<mlir::func::ReturnOp>(user) ||
            user->hasTrait<mlir::OpTrait::IsTerminator>())
          return false;
        mlir::Block *owningSuccessor = nullptr;
        for (mlir::Block *successor : successors)
          if (ancestorInBlock(user, successor)) {
            owningSuccessor = successor;
            break;
          }
        if (!owningSuccessor)
          return false;
        mlir::Operation *successorUser = ancestorInBlock(user, owningSuccessor);
        lastUser[owningSuccessor] =
            latestUserInBlock(lastUser[owningSuccessor], successorUser);
      }
    }
  }

  bool anyUse = false;
  for (mlir::Block *successor : successors)
    if (lastUser[successor])
      anyUse = true;
  if (!anyUse)
    return false;

  // A per-successor release is only valid when the value is re-defined on every
  // cycle that reaches it, i.e. no successor can reach itself while bypassing
  // the defining block. A loop-invariant value defined before the loop (e.g. a
  // stateful iterator threaded through the header) would otherwise be released
  // inside the loop and used again on the next iteration.
  for (mlir::Block *successor : successors)
    if (blockReachesAvoiding(successor, successor, defBlock))
      return false;

  // Non-using successors must be releasable at their entry without an edge
  // split, i.e. reached only from the defining block.
  for (mlir::Block *successor : successors)
    if (!lastUser[successor] &&
        !llvm::hasSingleElement(successor->getPredecessors()))
      return false;

  for (mlir::Block *successor : successors) {
    mlir::OpBuilder builder(call.getContext());
    if (mlir::Operation *last = lastUser[successor])
      builder.setInsertionPointAfter(last);
    else
      builder.setInsertionPointToStart(successor);
    mlir::func::CallOp::create(builder, call.getLoc(),
                               group.deallocator->function, group.values);
  }
  return true;
}

// Release `group` on the CFG edge from `terminator` to its successor
// #succIndex. For cf.br the release is emitted before the branch (the branch's
// only edge); for cf.cond_br the edge is split with a dedicated release block.
// Returns false for unsupported terminators.
bool releaseOnTerminatorEdge(mlir::Operation *terminator, unsigned succIndex,
                             const own::ResourceGroup &group,
                             mlir::Location loc) {
  mlir::OpBuilder builder(terminator);
  if (mlir::isa<mlir::cf::BranchOp>(terminator)) {
    builder.setInsertionPoint(terminator);
    mlir::func::CallOp::create(builder, loc, group.deallocator->function,
                               group.values);
    return true;
  }
  auto condbr = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator);
  if (!condbr)
    return false;
  mlir::Block *successor = condbr->getSuccessor(succIndex);
  llvm::SmallVector<mlir::Value, 4> trueOps(
      condbr.getTrueDestOperands().begin(), condbr.getTrueDestOperands().end());
  llvm::SmallVector<mlir::Value, 4> falseOps(
      condbr.getFalseDestOperands().begin(),
      condbr.getFalseDestOperands().end());
  mlir::Block *releaseBlock = builder.createBlock(successor->getParent(),
                                                  successor->getIterator());
  builder.setInsertionPointToStart(releaseBlock);
  mlir::func::CallOp::create(builder, loc, group.deallocator->function,
                             group.values);
  mlir::cf::BranchOp::create(builder, loc, successor,
                             succIndex == 0 ? trueOps : falseOps);
  builder.setInsertionPoint(condbr);
  if (succIndex == 0)
    mlir::cf::CondBranchOp::create(builder, condbr.getLoc(),
                                   condbr.getCondition(), releaseBlock,
                                   mlir::ValueRange{}, condbr.getFalseDest(),
                                   falseOps);
  else
    mlir::cf::CondBranchOp::create(builder, condbr.getLoc(),
                                   condbr.getCondition(), condbr.getTrueDest(),
                                   trueOps, releaseBlock, mlir::ValueRange{});
  condbr.erase();
  return true;
}

// General liveness-based release for an unconditionally-owned call-result group
// whose uses span the CFG (e.g. a loop element consumed across a body with
// continue/break/nested control flow). Computes single-def liveness (the value
// originates at `call` and is redefined on every entry to its defining block,
// so it is never live-in of that block from a back-edge), then releases the
// value exactly once on every path: after its last use in a block where it
// dies, or on the edges into successors where it becomes dead. Bails (leaving
// the caller/verifier to handle) on consuming/return/terminator/nested-region
// uses or unsupported edge terminators, so it never introduces unsafety.
std::optional<llvm::SmallVector<mlir::Value, 4>>
forwardedBlockArgGroup(mlir::Operation *terminator,
                       llvm::ArrayRef<mlir::Value> group,
                       own::AliasAnalysis &aliases);

// Core: release an unconditionally-owned `group` by single-def liveness. The
// value originates at `selfOp` (a call) or, when `selfOp` is null, at the entry
// of `defBlock` (a block argument). Releases the value where it dies. Returns
// true if handled; bails safely otherwise.
bool releaseOwnedGroupByLiveness(
    FuncContractCache &contracts, mlir::Operation *selfOp,
    mlir::Block *defBlock, mlir::Location loc, const own::ResourceGroup &group,
    own::AliasAnalysis &aliases, bool consumeIsDeath = false,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators = {}) {
  if (!group.deallocator || group.condition)
    return false;
  if (!defBlock)
    return false;
  mlir::Region *region = defBlock->getParent();
  if (!region)
    return false;
  // A branch that forwards every group value back into its OWN block-argument
  // position (a loop continue edge) transfers the token identically into the
  // next iteration: it is neither a use nor a death.
  auto isIdentitySelfForward = [&](mlir::Operation *user) {
    if (!consumeIsDeath)
      return false;
    std::optional<llvm::SmallVector<mlir::Value, 4>> forwarded =
        forwardedBlockArgGroup(user, group.values, aliases);
    if (!forwarded || forwarded->size() != group.values.size())
      return false;
    for (auto [index, destination] : llvm::enumerate(*forwarded))
      if (destination != group.values[index])
        return false;
    return true;
  };

  // Does `terminator` transfer the whole group into successor #succIndex's
  // block arguments? On such an edge the affine token leaves with the forward
  // (the destination argument group owns it); the value must not also be
  // released there. Other edges of the same terminator keep the token.
  auto forwardsGroupToSuccessor = [&](mlir::Operation *terminator,
                                      unsigned succIndex) {
    auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
    if (!branch)
      return false;
    mlir::SuccessorOperands operands = branch.getSuccessorOperands(succIndex);
    for (mlir::Value value : group.values) {
      bool found = false;
      for (unsigned index = 0, end = operands.size(); index < end; ++index)
        if (operands[index] && aliases.same(operands[index], value)) {
          found = true;
          break;
        }
      if (!found)
        return false;
    }
    return true;
  };
  auto forwardsGroupAnywhere = [&](mlir::Operation *terminator) {
    for (unsigned index = 0, end = terminator->getNumSuccessors(); index < end;
         ++index)
      if (forwardsGroupToSuccessor(terminator, index))
        return true;
    return false;
  };

  llvm::DenseMap<mlir::Block *, mlir::Operation *> lastUse;
  // Blocks that already release the group via a consuming use (e.g. the
  // emitter's loop back-edge decref-on-replace). The value dies there and must
  // NOT be released again; other dead paths still need a release.
  llvm::DenseSet<mlir::Block *> consumedBlocks;
  // Interior views (canonical-shape tail beyond the release interface) pin
  // the entity: every use is a plain liveness contribution. Box-word
  // reconstructions (borrowed memref views assembled from the entity's box
  // words) pin the same way.
  llvm::SmallVector<mlir::Value, 8> pinnedViews(group.views.begin(),
                                                group.views.end());
  {
    llvm::SmallVector<mlir::Value, 8> groupEquivalents;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.aliasesOf(result, equivalentValues);
      if (equivalentValues.empty())
        equivalentValues.push_back(result);
      groupEquivalents.append(equivalentValues.begin(),
                              equivalentValues.end());
    }
    own::collectBoxWordDerivedViews(groupEquivalents, pinnedViews);
  }
  // Nested-region uses may pin liveness at their top-level ancestor — but
  // only when the group has NO consuming calls: this walk is order-blind
  // within blocks, so extending liveness past a mid-block consume would place
  // a second (double-freeing) release downstream. Groups with consuming uses
  // keep the conservative nested-use bail.
  bool groupHasConsumingCall = false;
  for (mlir::Value result : group.values) {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.aliasesOf(result, equivalents);
    if (equivalents.empty())
      equivalents.push_back(result);
    for (mlir::Value equivalent : equivalents) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        if (use.getOwner() == selfOp)
          continue;
        if (ownershipConsumingUseInvalidatesGroup(contracts, use, group.values,
                                                  aliases)) {
          groupHasConsumingCall = true;
          break;
        }
      }
      if (groupHasConsumingCall)
        break;
    }
    if (groupHasConsumingCall)
      break;
  }

  for (mlir::Value view : pinnedViews) {
    for (mlir::OpOperand &use : view.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == selfOp)
        continue;
      // A use nested inside a region op (e.g. the boxed lane of a prim/boxed
      // scf.if dispatch) pins liveness at its top-level ancestor. Nested
      // terminators forward the view out through a region result; views only
      // pin, so treat that like a top-level terminator use (ignored).
      if (user->hasTrait<mlir::OpTrait::IsTerminator>())
        continue;
      mlir::Operation *blockUser =
          groupHasConsumingCall && user->getBlock()->getParent() != region
              ? nullptr
              : ancestorInRegion(user, region);
      if (!blockUser)
        return false;
      if (blockUser->hasTrait<mlir::OpTrait::IsTerminator>())
        continue; // view forwards ride along with the token's edges
      lastUse[blockUser->getBlock()] =
          latestUserInBlock(lastUse[blockUser->getBlock()], blockUser);
    }
  }
  for (mlir::Value result : group.values) {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.aliasesOf(result, equivalents);
    if (equivalents.empty())
      equivalents.push_back(result);
    for (mlir::Value equivalent : equivalents)
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (user == selfOp)
          continue;
        if (ownershipConsumingUseInvalidatesGroup(contracts, use, group.values,
                                                  aliases)) {
          if (!consumeIsDeath)
            return false;
          if (user->getBlock()->getParent() != region)
            return false;
          consumedBlocks.insert(user->getBlock());
          lastUse[user->getBlock()] =
              latestUserInBlock(lastUse[user->getBlock()], user);
          continue;
        }
        if (user->hasTrait<mlir::OpTrait::IsTerminator>() &&
            isIdentitySelfForward(user))
          continue;
        if (user->hasTrait<mlir::OpTrait::IsTerminator>() &&
            !mlir::isa<mlir::func::ReturnOp>(user) &&
            user->getBlock()->getParent() == region &&
            forwardsGroupAnywhere(user)) {
          // Whole-group transfer into a successor's block arguments: the token
          // leaves on the forwarding edge (no release there), but the value is
          // live up to this terminator; non-forwarding edges of the same
          // terminator keep the token and are handled by the edge scan below.
          consumedBlocks.insert(user->getBlock());
          lastUse[user->getBlock()] =
              latestUserInBlock(lastUse[user->getBlock()], user);
          continue;
        }
        if (auto returnOp = mlir::dyn_cast<mlir::func::ReturnOp>(user)) {
          // An owned return transfers the token: a consuming death on that
          // path; the other paths still get their releases below.
          auto function = returnOp->getParentOfType<mlir::func::FuncOp>();
          if (consumeIsDeath && !deallocators.empty() && function &&
              returnTransfersGroup(contracts, function, returnOp, group.values,
                                   deallocators, aliases)) {
            consumedBlocks.insert(user->getBlock());
            lastUse[user->getBlock()] =
                latestUserInBlock(lastUse[user->getBlock()], user);
            continue;
          }
          return false;
        }
        if (user->hasTrait<mlir::OpTrait::IsTerminator>())
          return false;
        if (user->getBlock()->getParent() != region) {
          // A plain use nested inside a region op pins liveness at its
          // top-level ancestor. Nested TERMINATORS (scf.yield etc.) keep the
          // conservative bail: they forward the token out through a region
          // result under a new name this walk cannot track, so a release at
          // the ancestor would be premature.
          mlir::Operation *blockUser =
              (user->hasTrait<mlir::OpTrait::IsTerminator>() ||
               groupHasConsumingCall)
                  ? nullptr
                  : ancestorInRegion(user, region);
          if (!blockUser ||
              blockUser->hasTrait<mlir::OpTrait::IsTerminator>())
            return false;
          lastUse[blockUser->getBlock()] =
              latestUserInBlock(lastUse[blockUser->getBlock()], blockUser);
          continue;
        }
        lastUse[user->getBlock()] =
            latestUserInBlock(lastUse[user->getBlock()], user);
      }
  }
  // A call root with no uses is handled by findReleaseInsertion; only bail for
  // it here. A block-argument root with no uses dies at its defining block and
  // is released there (below).
  if (lastUse.empty() && selfOp)
    return false;

  // Single-def backward liveness. liveIn[defBlock] is forced false: the value
  // originates at `call` and any back-edge into defBlock re-defines it.
  llvm::DenseMap<mlir::Block *, char> liveIn, liveOut;
  for (mlir::Block &block : *region) {
    liveIn[&block] = 0;
    liveOut[&block] = 0;
  }
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
      exceptionEdges = own::collectExceptionEdges(*region);
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Block &block : llvm::reverse(*region)) {
      char out = 0;
      for (mlir::Block *successor : block.getSuccessors())
        if (liveIn[successor]) {
          out = 1;
          break;
        }
      if (!out)
        if (auto found = exceptionEdges.find(&block);
            found != exceptionEdges.end())
          for (mlir::Block *successor : found->second)
            if (liveIn[successor]) {
              out = 1;
              break;
            }
      char in = (&block == defBlock)
                    ? 0
                    : ((lastUse.count(&block) || out) ? 1 : 0);
      if (out != liveOut[&block]) {
        liveOut[&block] = out;
        changed = true;
      }
      if (in != liveIn[&block]) {
        liveIn[&block] = in;
        changed = true;
      }
    }
  }

  auto blockIsLive = [&](mlir::Block *block) {
    return block == defBlock || liveIn[block] || lastUse.count(block);
  };

  llvm::SmallVector<std::pair<mlir::Operation *, unsigned>, 8> edgeReleases;
  llvm::SmallVector<mlir::Operation *, 8> afterUseReleases;
  llvm::SmallVector<mlir::Block *, 8> beforeTermReleases;
  llvm::SmallVector<mlir::Block *, 4> atStartReleases;
  for (mlir::Block &blockRef : *region) {
    mlir::Block *block = &blockRef;
    if (!blockIsLive(block))
      continue;
    if (!liveOut[block]) {
      // The value already died here via a consuming use (e.g. back-edge
      // decref-on-replace); do not release again.
      if (consumedBlocks.count(block)) {
        // When the consuming use is a TERMINATOR forwarding the group into a
        // successor's block arguments on only SOME edges (`cond_br %c,
        // ^replaced, ^merge(%group...)`), the token leaves only along the
        // forwarding edges — the remaining edges exit the block with the
        // token still owned and the values dead, so they need edge releases
        // (the replaced lane of a conditional reassignment).
        mlir::Operation *terminator = block->getTerminator();
        auto it = lastUse.find(block);
        if (!groupHasConsumingCall && it != lastUse.end() &&
            it->second == terminator && forwardsGroupAnywhere(terminator)) {
          for (unsigned index = 0, end = terminator->getNumSuccessors();
               index < end; ++index) {
            if (forwardsGroupToSuccessor(terminator, index))
              continue;
            mlir::Block *successor = terminator->getSuccessor(index);
            if (liveIn[successor])
              continue;
            if (successor == defBlock && isIdentitySelfForward(terminator))
              continue;
            if (!successor->empty() &&
                isIdentitySelfForward(successor->getTerminator()))
              continue;
            edgeReleases.push_back({terminator, index});
          }
        }
        continue;
      }
      auto it = lastUse.find(block);
      if (it != lastUse.end())
        afterUseReleases.push_back(it->second);
      else if (block == defBlock && selfOp)
        afterUseReleases.push_back(selfOp);
      else if (block == defBlock)
        atStartReleases.push_back(block); // block-arg root, dies unused in defBlock
      else
        beforeTermReleases.push_back(block);
    } else {
      mlir::Operation *terminator = block->getTerminator();
      for (unsigned index = 0, end = terminator->getNumSuccessors();
           index < end; ++index) {
        if (!liveIn[terminator->getSuccessor(index)]) {
          // A continue edge identity-forwards the token into the next
          // iteration's incarnation of the same block arguments: a transfer,
          // not a death.
          if (terminator->getSuccessor(index) == defBlock &&
              isIdentitySelfForward(terminator))
            continue;
          // A whole-group forward into this successor's arguments transfers
          // the token to the destination argument group: not a death either.
          if (forwardsGroupToSuccessor(terminator, index))
            continue;
          // The successor itself may only pass the token back into the next
          // iteration of the defining block's arguments (the non-mutating arm
          // of a conditional structural mutation): identity self-forwards are
          // invisible to the liveness above, but the token survives through
          // them — not a death.
          mlir::Block *successor = terminator->getSuccessor(index);
          if (!successor->empty() &&
              isIdentitySelfForward(successor->getTerminator()))
            continue;
          edgeReleases.push_back({terminator, index});
        }
      }
    }
  }

  // Validate every edge release targets a terminator we can split before we
  // mutate anything (a cond_br has at most one dead successor, since liveOut
  // implies a live successor, so no terminator is split twice).
  for (auto &edge : edgeReleases)
    if (!mlir::isa<mlir::cf::BranchOp, mlir::cf::CondBranchOp>(edge.first))
      return false;

  // A release placed before a later marked call site of the SAME block would
  // double-free on unwind: the handler (live on the exception edge) performs
  // its own release, but the try-path one has already run. Delay the release
  // past the block's last marker so an unwind from any marked call in the
  // block reaches the handler with the token intact.
  auto delayPastCallSiteMarkers =
      [&](mlir::Operation *insertAfter) -> mlir::Operation * {
    mlir::Block *block = insertAfter->getBlock();
    if (!exceptionEdges.count(block))
      return insertAfter;
    mlir::Operation *last = insertAfter;
    for (mlir::Operation *op = insertAfter->getNextNode(); op;
         op = op->getNextNode())
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
        if (call.getCallee() == "LyEH_TryCallSiteMarker")
          last = op;
    return last;
  };

  for (mlir::Operation *afterOp : afterUseReleases) {
    mlir::Operation *anchor = delayPastCallSiteMarkers(afterOp);
    mlir::OpBuilder builder(anchor);
    builder.setInsertionPointAfter(anchor);
    mlir::func::CallOp::create(builder, loc, group.deallocator->function,
                               group.values);
  }
  for (mlir::Block *block : beforeTermReleases) {
    mlir::OpBuilder builder(block->getTerminator());
    mlir::func::CallOp::create(builder, loc, group.deallocator->function,
                               group.values);
  }
  for (mlir::Block *block : atStartReleases) {
    mlir::OpBuilder builder(&block->front());
    mlir::func::CallOp::create(builder, loc, group.deallocator->function,
                               group.values);
  }
  for (auto &edge : edgeReleases)
    if (!releaseOnTerminatorEdge(edge.first, edge.second, group, loc))
      return false;
  return true;
}

// Wrapper: liveness-based release for an owned call-result group.
bool insertOwnedValueReleasesByLiveness(FuncContractCache &contracts,
                                        mlir::func::CallOp call,
                                        const own::ResourceGroup &group,
                                        own::AliasAnalysis &aliases) {
  return releaseOwnedGroupByLiveness(contracts, call.getOperation(),
                                     call->getBlock(), call.getLoc(), group,
                                     aliases);
}

// If `terminator` forwards every value of `group` to arguments of a single
// successor block, return that successor's argument group (in `group` order).
std::optional<llvm::SmallVector<mlir::Value, 4>>
forwardedBlockArgGroup(mlir::Operation *terminator,
                       llvm::ArrayRef<mlir::Value> group,
                       own::AliasAnalysis &aliases) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return std::nullopt;
  mlir::Block *destBlock = nullptr;
  llvm::SmallVector<mlir::Value, 4> destArgs(group.size());
  for (unsigned s = 0, e = terminator->getNumSuccessors(); s < e; ++s) {
    mlir::Block *successor = terminator->getSuccessor(s);
    mlir::SuccessorOperands ops = branch.getSuccessorOperands(s);
    unsigned n = std::min<unsigned>(successor->getNumArguments(), ops.size());
    for (unsigned a = 0; a < n; ++a)
      for (unsigned j = 0; j < group.size(); ++j)
        if (ops[a] && aliases.same(ops[a], group[j])) {
          if (destBlock && destBlock != successor)
            return std::nullopt; // group split across successors
          destBlock = successor;
          destArgs[j] = successor->getArgument(a);
        }
  }
  if (!destBlock)
    return std::nullopt;
  for (mlir::Value arg : destArgs)
    if (!arg)
      return std::nullopt; // not every group value forwarded
  return destArgs;
}

// Release owned values TRANSFERRED into merge/loop block arguments (e.g.
// `if c: y=a else: y=b; print(y)`, or a loop-carried accumulator's final
// value). The refcount pass sees the source owned call-result group forward to
// a block argument and bails on releasing the source; the destination block
// argument then carries the ownership. A destination block-arg group shares the
// source's deallocator (same transferred value), so no separate metadata is
// needed. Ownership is propagated through block-arg->block-arg forwards (loop
// headers) by fixpoint, and only groups where EVERY predecessor forwards an
// owned value survive a soundness fixpoint (so no borrowed value is released).
// `releaseOwnedGroupByLiveness` then releases each group where it dies and bails
// on forwarded/returned args (transfers), so loop-header args are left alone and
// only dead after-loop / merge args are released.
mlir::LogicalResult insertOwnedBlockArgumentReleases(
    mlir::ModuleOp module, FuncContractCache &contracts,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  llvm::DenseSet<mlir::Value> ownedValues;
  module.walk([&](mlir::func::CallOp call) {
    mlir::func::FuncOp fn = call->getParentOfType<mlir::func::FuncOp>();
    if (!fn || own::isRuntimeManifestFunction(fn))
      return;
    for (const own::ResourceGroup &g :
         own::collectOwnedCallResultGroups(module, call, deallocators)) {
      if (!g.deallocator || g.condition)
        continue;
      for (mlir::Value v : g.values)
        ownedValues.insert(v);
      // The token also lives in every region-merge result the group maps
      // through (e.g. the int fast/slow scf.if): those parent results are
      // what function-level branches forward.
      mlir::Block *callBlock = call->getBlock();
      llvm::SmallVector<mlir::Value, 4> values(g.values.begin(),
                                               g.values.end());
      while (callBlock && callBlock->getTerminator() &&
             !mlir::isa<mlir::func::FuncOp>(callBlock->getParentOp())) {
        auto mapped = mapRegionTerminatorGroupToParentResults(
            callBlock->getTerminator(), values, aliases);
        if (!mapped)
          break;
        values = std::move(*mapped);
        for (mlir::Value v : values)
          ownedValues.insert(v);
        callBlock = callBlock->getParentOp()->getBlock();
      }
    }
  });

  struct Candidate {
    llvm::SmallVector<mlir::Value, 4> args;
    llvm::SmallVector<mlir::Value, 4> views;
    const own::RuntimeDeallocator *deallocator;
  };
  llvm::MapVector<mlir::Value, Candidate> candidates;

  // Map a group's interior views across the same forwarding edge. Views that
  // do not forward make the candidate unsafe to release (their uses would be
  // invisible to the liveness): callers must drop it (leak-safe) rather than
  // risk a premature release.
  auto forwardedViews =
      [&](mlir::Operation *terminator, llvm::ArrayRef<mlir::Value> views)
      -> std::optional<llvm::SmallVector<mlir::Value, 4>> {
    if (views.empty())
      return llvm::SmallVector<mlir::Value, 4>{};
    return forwardedBlockArgGroup(terminator, views, aliases);
  };

  // Seed: owned call-result groups forwarded to block-arg groups. A group
  // born inside a region merge (e.g. the int fast/slow scf.if) first maps
  // through its region terminator(s) to the parent op's results; the
  // function-level branch then decides the forward.
  module.walk([&](mlir::func::CallOp call) {
    mlir::func::FuncOp fn = call->getParentOfType<mlir::func::FuncOp>();
    if (!fn || own::isRuntimeManifestFunction(fn))
      return;
    for (const own::ResourceGroup &g :
         own::collectOwnedCallResultGroups(module, call, deallocators)) {
      if (!g.deallocator || g.condition)
        continue;
      mlir::Block *callBlock = call->getBlock();
      llvm::SmallVector<mlir::Value, 4> values(g.values.begin(),
                                               g.values.end());
      llvm::SmallVector<mlir::Value, 4> views(g.views.begin(), g.views.end());
      bool escaped = false;
      while (callBlock && callBlock->getTerminator() &&
             !mlir::isa<mlir::func::FuncOp>(callBlock->getParentOp())) {
        mlir::Operation *terminator = callBlock->getTerminator();
        auto mappedValues = mapRegionTerminatorGroupToParentResults(
            terminator, values, aliases);
        if (!mappedValues) {
          escaped = true;
          break;
        }
        if (!views.empty()) {
          auto mappedViews = mapRegionTerminatorGroupToParentResults(
              terminator, views, aliases);
          if (!mappedViews) {
            escaped = true;
            break;
          }
          views = std::move(*mappedViews);
        }
        values = std::move(*mappedValues);
        callBlock = callBlock->getParentOp()->getBlock();
      }
      if (escaped || !callBlock || !callBlock->getTerminator())
        continue;
      if (auto dest = forwardedBlockArgGroup(callBlock->getTerminator(),
                                             values, aliases)) {
        auto destViews = forwardedViews(callBlock->getTerminator(), views);
        if (!destViews)
          continue;
        candidates.insert(
            {dest->front(), Candidate{*dest, *destViews, g.deallocator}});
      }
    }
  });

  // Propagate ownership through block-arg -> block-arg forwards (loop headers,
  // chained merges) until no new destination group is discovered.
  bool changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<Candidate, 8> snapshot;
    for (auto &entry : candidates)
      snapshot.push_back(entry.second);
    for (Candidate &candidate : snapshot) {
      auto firstArg = mlir::dyn_cast<mlir::BlockArgument>(candidate.args.front());
      if (!firstArg || !firstArg.getOwner()->getTerminator())
        continue;
      if (auto dest = forwardedBlockArgGroup(firstArg.getOwner()->getTerminator(),
                                             candidate.args, aliases)) {
        auto destViews = forwardedViews(firstArg.getOwner()->getTerminator(),
                                        candidate.views);
        if (destViews && !candidates.count(dest->front())) {
          candidates.insert({dest->front(), Candidate{*dest, *destViews,
                                                      candidate.deallocator}});
          changed = true;
        }
      }
    }
  }

  // Soundness: every incoming edge must deliver a TOKEN to the destination
  // group. An edge whose incoming values die at the terminator (an owned
  // local forwarded at its last use) transfers its token. An edge whose
  // incoming values stay live past the branch (a loop-carried local, an
  // entry argument) only lends a borrow — releasing the merge argument would
  // over-release it. Those edges get an explicit retain (borrow → own via
  // the checked-retain premise: the SSA operand is provably alive at the
  // terminator), so every edge transfers uniformly. Candidates with no
  // owned-transfer edge at all are plain borrow merges: dropped.
  auto isOwnedIncoming = [&](mlir::Value v) {
    if (ownedValues.count(v))
      return true;
    if (mlir::isa<mlir::BlockArgument>(v))
      for (auto &entry : candidates)
        for (mlir::Value arg : entry.second.args)
          if (arg == v)
            return true;
    return false;
  };
  // An incoming value dies on the edge pred→dest when it is not LIVE-IN at
  // dest under standard backward liveness (upward-exposed uses; a loop
  // body's uses of its own new incarnation do not keep the previous
  // iteration's value alive across the back edge). The forwarded block
  // argument replaces the value for all merged uses.
  llvm::DenseMap<mlir::Operation *,
                 llvm::DenseMap<mlir::Block *, llvm::DenseSet<mlir::Value>>>
      functionLiveIns;
  auto functionLevelBlock = [](mlir::Value v) -> mlir::Block * {
    mlir::Block *block = v.getParentBlock();
    while (block && block->getParentOp() &&
           !mlir::isa<mlir::func::FuncOp>(block->getParentOp()))
      block = block->getParentOp()->getBlock();
    return block;
  };
  auto liveInsFor = [&](mlir::func::FuncOp fn)
      -> llvm::DenseMap<mlir::Block *, llvm::DenseSet<mlir::Value>> & {
    auto existing = functionLiveIns.find(fn.getOperation());
    if (existing != functionLiveIns.end())
      return existing->second;
    auto &liveIns = functionLiveIns[fn.getOperation()];
    llvm::DenseMap<mlir::Block *, llvm::DenseSet<mlir::Value>> blockUses;
    for (mlir::Block &block : fn.getBody()) {
      llvm::DenseSet<mlir::Value> &uses = blockUses[&block];
      for (mlir::Operation &op : block)
        op.walk([&](mlir::Operation *inner) {
          for (mlir::Value operand : inner->getOperands())
            if (functionLevelBlock(operand) != &block)
              uses.insert(operand);
        });
    }
    llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
        exceptionEdges = own::collectExceptionEdges(fn.getBody());
    bool converged = false;
    while (!converged) {
      converged = true;
      for (mlir::Block &block : llvm::reverse(fn.getBody())) {
        llvm::DenseSet<mlir::Value> live = blockUses[&block];
        for (mlir::Block *successor : block.getSuccessors())
          for (mlir::Value v : liveIns[successor])
            if (functionLevelBlock(v) != &block)
              live.insert(v);
        if (auto found = exceptionEdges.find(&block);
            found != exceptionEdges.end())
          for (mlir::Block *successor : found->second)
            for (mlir::Value v : liveIns[successor])
              if (functionLevelBlock(v) != &block)
                live.insert(v);
        llvm::DenseSet<mlir::Value> &slot = liveIns[&block];
        if (live.size() != slot.size()) {
          slot = std::move(live);
          converged = false;
        }
      }
    }
    return liveIns;
  };
  auto diesOnEdge = [&](mlir::Value v, mlir::Operation *terminator,
                        mlir::Block *dest) {
    auto fn = terminator->getParentOfType<mlir::func::FuncOp>();
    if (!fn)
      return false;
    auto &liveIns = liveInsFor(fn);
    auto found = liveIns.find(dest);
    return found == liveIns.end() || !found->second.contains(v);
  };
  mlir::func::FuncOp retainFunction =
      module.lookupSymbol<mlir::func::FuncOp>("Ly_IncRef");
  struct EdgeRetain {
    mlir::Operation *terminator;
    unsigned successorIndex = 0;
    mlir::Value header;
  };
  llvm::SmallVector<EdgeRetain, 8> edgeRetains;
  changed = true;
  while (changed) {
    changed = false;
    llvm::SmallVector<mlir::Value, 8> toRemove;
    edgeRetains.clear();
    for (auto &entry : candidates) {
      Candidate &candidate = entry.second;
      auto firstArg = mlir::cast<mlir::BlockArgument>(candidate.args.front());
      mlir::Block *destBlock = firstArg.getOwner();
      llvm::SmallVector<unsigned, 4> argIndices;
      for (mlir::Value arg : candidate.args)
        argIndices.push_back(mlir::cast<mlir::BlockArgument>(arg).getArgNumber());
      bool sound = true;
      bool anyTransfer = false;
      llvm::SmallVector<EdgeRetain, 4> retains;
      for (mlir::Block *pred : destBlock->getPredecessors()) {
        auto branch =
            mlir::dyn_cast<mlir::BranchOpInterface>(pred->getTerminator());
        if (!branch) {
          sound = false;
          break;
        }
        unsigned edgesToDest = 0;
        for (unsigned s = 0, e = pred->getTerminator()->getNumSuccessors();
             s < e && sound; ++s) {
          if (pred->getTerminator()->getSuccessor(s) != destBlock)
            continue;
          if (++edgesToDest > 1) {
            // Two edges into the merge from one terminator cannot both be
            // retained (only one is taken at runtime).
            sound = false;
            break;
          }
          mlir::SuccessorOperands ops = branch.getSuccessorOperands(s);
          bool transfers = true;
          mlir::Value header;
          for (mlir::Value arg : candidate.args) {
            unsigned idx = mlir::cast<mlir::BlockArgument>(arg).getArgNumber();
            mlir::Value incoming = idx < ops.size() ? ops[idx] : mlir::Value();
            if (!incoming) {
              sound = false;
              break;
            }
            if (!header)
              header = incoming;
            if (!isOwnedIncoming(incoming) ||
                !diesOnEdge(incoming, pred->getTerminator(), destBlock))
              transfers = false;
          }
          if (!sound)
            break;
          if (transfers) {
            anyTransfer = true;
          } else if (retainFunction && header &&
                     ownership::isObjectHeaderLikeType(header.getType())) {
            retains.push_back(EdgeRetain{pred->getTerminator(), s, header});
          } else {
            sound = false;
          }
        }
        if (!sound)
          break;
      }
      if (!sound || !anyTransfer)
        toRemove.push_back(entry.first);
      else
        edgeRetains.append(retains.begin(), retains.end());
    }
    for (mlir::Value key : toRemove) {
      candidates.erase(key);
      changed = true;
    }
  }
  // Several retains can target the same terminator (multiple candidate
  // groups on one edge, or both edges of one cond_br): splitting erases and
  // recreates the cond_br, so resolve each retain's terminator through the
  // replacement map, and anchor same-edge retains at the shared edge block.
  llvm::DenseMap<mlir::Operation *, mlir::Operation *> replacedConds;
  llvm::DenseMap<std::pair<mlir::Operation *, unsigned>, mlir::Operation *>
      edgeAnchors;
  for (const EdgeRetain &retain : edgeRetains) {
    // A retain placed before a multi-successor terminator would execute on
    // the paths that do NOT take the borrow edge (an unreleased over-retain):
    // split the edge and put the retain in a dedicated edge block.
    mlir::Operation *anchor = retain.terminator;
    auto anchorKey = std::make_pair(retain.terminator, retain.successorIndex);
    if (auto existing = edgeAnchors.lookup(anchorKey)) {
      anchor = existing;
    } else {
      if (auto replaced = replacedConds.lookup(retain.terminator))
        anchor = replaced;
      if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(anchor)) {
        bool trueEdge = retain.successorIndex == 0;
        mlir::Block *dest =
            trueEdge ? cond.getTrueDest() : cond.getFalseDest();
        mlir::ValueRange operands = trueEdge ? cond.getTrueDestOperands()
                                             : cond.getFalseDestOperands();
        auto *edge = new mlir::Block;
        dest->getParent()->getBlocks().insert(dest->getIterator(), edge);
        mlir::OpBuilder edgeBuilder(edge, edge->begin());
        auto edgeBranch = mlir::cf::BranchOp::create(edgeBuilder,
                                                     cond.getLoc(), dest,
                                                     operands);
        mlir::OpBuilder condBuilder(cond);
        auto newCond = mlir::cf::CondBranchOp::create(
            condBuilder, cond.getLoc(), cond.getCondition(),
            trueEdge ? edge : cond.getTrueDest(),
            trueEdge ? mlir::ValueRange{} : cond.getTrueDestOperands(),
            trueEdge ? cond.getFalseDest() : edge,
            trueEdge ? cond.getFalseDestOperands() : mlir::ValueRange{});
        cond.erase();
        replacedConds[retain.terminator] = newCond.getOperation();
        edgeAnchors[anchorKey] = edgeBranch.getOperation();
        anchor = edgeBranch;
      }
    }
    mlir::OpBuilder builder(anchor);
    // The lend must precede any release in the same block: on identity merge
    // edges the retained value IS the value the block's decref-on-replace
    // releases, and with the retain after the release a refcount of one dips
    // to zero mid-block — the release frees the object and the retain then
    // reads freed memory. Insert at the earliest point after the header's
    // definition instead of just before the terminator.
    mlir::Block *anchorBlock = anchor->getBlock();
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(retain.header)) {
      if (blockArg.getOwner() == anchorBlock)
        builder.setInsertionPointToStart(anchorBlock);
    } else if (mlir::Operation *definition = retain.header.getDefiningOp()) {
      if (definition->getBlock() == anchorBlock)
        builder.setInsertionPointAfter(definition);
    }
    mlir::Value header = retain.header;
    mlir::Type retainInput = retainFunction.getFunctionType().getInput(0);
    if (header.getType() != retainInput) {
      if (!mlir::memref::CastOp::areCastCompatible(header.getType(),
                                                   retainInput))
        continue;
      header = mlir::memref::CastOp::create(builder, anchor->getLoc(),
                                            retainInput, header)
                   .getResult();
    }
    auto call = mlir::func::CallOp::create(builder, anchor->getLoc(),
                                           retainFunction, header);
    call->setAttr("ly.ownership.aggregate_retain",
                  builder.getStringAttr(own::kBlockArgMergeBorrowLabel));
  }

  for (auto &entry : candidates) {
    Candidate &candidate = entry.second;
    auto firstArg = mlir::cast<mlir::BlockArgument>(candidate.args.front());
    own::ResourceGroup destGroup;
    destGroup.values.assign(candidate.args.begin(), candidate.args.end());
    destGroup.views.assign(candidate.views.begin(), candidate.views.end());
    destGroup.deallocator = candidate.deallocator;
    releaseOwnedGroupByLiveness(contracts, /*selfOp=*/nullptr,
                                firstArg.getOwner(), firstArg.getLoc(),
                                destGroup, aliases, /*consumeIsDeath=*/true,
                                deallocators);
  }
  return mlir::success();
}

mlir::LogicalResult
insertOwnedResultReleases(mlir::ModuleOp module, mlir::func::CallOp call,
                          FuncContractCache &contracts,
                          llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                          own::AliasAnalysis &aliases) {
  if (call.getNumResults() == 0)
    return mlir::success();

  for (own::ResourceGroup group :
       own::collectOwnedCallResultGroups(module, call, deallocators)) {
    if (!group.deallocator)
      continue;

    if (group.condition) {
      mlir::FailureOr<bool> inserted = insertConditionalOwnedResultRelease(
          contracts, call, group, aliases);
      if (mlir::failed(inserted))
        return mlir::failure();
      if (*inserted)
        continue;
    }

    std::optional<ReleaseInsertion> release =
        findReleaseInsertion(contracts, call, group.values, deallocators,
                             aliases, /*depth=*/0, group.views);
    if (release) {
      mlir::OpBuilder builder(call);
      if (release->before)
        builder.setInsertionPoint(release->before);
      else
        builder.setInsertionPointAfter(release->after);
      mlir::func::CallOp::create(builder, call.getLoc(),
                                 group.deallocator->function, release->group);
      continue;
    }

    if (insertImmediateSuccessorReleases(contracts, call, group, aliases))
      continue;

    if (insertOwnedValueReleasesByLiveness(contracts, call, group, aliases))
      continue;

    mlir::func::FuncOp function = call->getParentOfType<mlir::func::FuncOp>();
    if (!function)
      continue;

    bool canReleaseAtExits = true;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.aliasesOf(result, equivalentValues);
      if (equivalentValues.empty())
        equivalentValues.push_back(result);
      for (mlir::Value equivalent : equivalentValues) {
        for (mlir::OpOperand &use : equivalent.getUses()) {
          mlir::Operation *user = use.getOwner();
          if (user == call.getOperation())
            continue;
          if (user->getParentOfType<mlir::func::FuncOp>() != function ||
              ownershipConsumingUseInvalidatesGroup(contracts, use,
                                                    group.values, aliases) ||
              mlir::isa<mlir::func::ReturnOp>(user) ||
              branchForwardsGroupToBlockArgument(user, group.values, aliases)) {
            canReleaseAtExits = false;
            break;
          }
        }
        if (!canReleaseAtExits)
          break;
      }
      if (!canReleaseAtExits)
        break;
    }
    if (!canReleaseAtExits)
      continue;

    mlir::DominanceInfo dominance(function);
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp returnOp) {
      if (dominance.dominates(call.getOperation(), returnOp.getOperation()))
        returns.push_back(returnOp);
    });
    for (mlir::func::ReturnOp returnOp : returns) {
      mlir::OpBuilder builder(returnOp);
      mlir::func::CallOp::create(builder, returnOp.getLoc(),
                                 group.deallocator->function, group.values);
    }
  }
  return mlir::success();
}

mlir::LogicalResult insertOwnedLocalObjectReleases(
    mlir::ModuleOp module, mlir::Operation *op, FuncContractCache &contracts,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  for (own::ResourceGroup group :
       own::collectOwnedLocalObjectGroups(op, deallocators)) {
    if (!group.deallocator)
      continue;

    std::optional<ReleaseInsertion> release =
        findReleaseInsertion(contracts, op, group.values, deallocators,
                             aliases, /*depth=*/0, group.views);
    if (release) {
      mlir::OpBuilder builder(op);
      if (release->before)
        builder.setInsertionPoint(release->before);
      else
        builder.setInsertionPointAfter(release->after);
      mlir::func::CallOp::create(builder, op->getLoc(),
                                 group.deallocator->function, release->group);
      continue;
    }

    // Straight-line placement failed (e.g. the entity crosses blocks inside a
    // loop body): fall back to CFG liveness, mirroring the owned-call-result
    // path.
    if (releaseOwnedGroupByLiveness(contracts, op, op->getBlock(),
                                    op->getLoc(), group, aliases,
                                    /*consumeIsDeath=*/true))
      continue;

    mlir::func::FuncOp function = op->getParentOfType<mlir::func::FuncOp>();
    if (!function)
      continue;

    bool canReleaseAtExits = true;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.aliasesOf(result, equivalentValues);
      if (equivalentValues.empty())
        equivalentValues.push_back(result);
      for (mlir::Value equivalent : equivalentValues) {
        for (mlir::OpOperand &use : equivalent.getUses()) {
          mlir::Operation *user = use.getOwner();
          if (user == op)
            continue;
          if (user->getParentOfType<mlir::func::FuncOp>() != function ||
              ownershipConsumingUseInvalidatesGroup(contracts, use,
                                                    group.values, aliases) ||
              mlir::isa<mlir::func::ReturnOp>(user) ||
              branchForwardsGroupToBlockArgument(user, group.values, aliases)) {
            canReleaseAtExits = false;
            break;
          }
        }
        if (!canReleaseAtExits)
          break;
      }
      if (!canReleaseAtExits)
        break;
    }
    if (!canReleaseAtExits)
      continue;

    mlir::DominanceInfo dominance(function);
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp returnOp) {
      if (dominance.dominates(op, returnOp.getOperation()))
        returns.push_back(returnOp);
    });
    for (mlir::func::ReturnOp returnOp : returns) {
      mlir::OpBuilder builder(returnOp);
      mlir::func::CallOp::create(builder, returnOp.getLoc(),
                                 group.deallocator->function, group.values);
    }
  }
  return mlir::success();
}

class RefCountInsertionPass
    : public mlir::PassWrapper<RefCountInsertionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RefCountInsertionPass)

  llvm::StringRef getArgument() const final {
    return "lython-refcount-insertion";
  }
  llvm::StringRef getDescription() const final {
    return "insert manifest-driven releases for runtime-owned call results";
  }

  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators;
    {
      py::PerfScope perf("refcount-insertion.collect-deallocators");
      deallocators = own::collectRuntimeDeallocators(module);
    }
    if (deallocators.empty())
      return;
    {
      // Split dual-edge cond_br terminators (both successors == one block,
      // a canonicalized empty-arm conditional) through fresh edge blocks:
      // the per-edge token classification below needs a place to insert
      // edge-specific retains/releases, and a shared terminator has none.
      py::PerfScope perf("refcount-insertion.split-dual-edges");
      llvm::SmallVector<mlir::cf::CondBranchOp, 8> dualEdges;
      module.walk([&](mlir::cf::CondBranchOp cond) {
        if (cond.getTrueDest() == cond.getFalseDest())
          dualEdges.push_back(cond);
      });
      for (mlir::cf::CondBranchOp cond : dualEdges) {
        mlir::Block *dest = cond.getTrueDest();
        mlir::Region *region = dest->getParent();
        mlir::OpBuilder builder(cond);
        auto makeEdgeBlock = [&](mlir::ValueRange operands) {
          auto *edge = new mlir::Block;
          region->getBlocks().insert(dest->getIterator(), edge);
          mlir::OpBuilder edgeBuilder(edge, edge->begin());
          mlir::cf::BranchOp::create(edgeBuilder, cond.getLoc(), dest,
                                     operands);
          return edge;
        };
        mlir::Block *trueEdge = makeEdgeBlock(cond.getTrueDestOperands());
        mlir::Block *falseEdge = makeEdgeBlock(cond.getFalseDestOperands());
        mlir::cf::CondBranchOp::create(builder, cond.getLoc(),
                                       cond.getCondition(), trueEdge,
                                       mlir::ValueRange{}, falseEdge,
                                       mlir::ValueRange{});
        cond.erase();
      }
    }
    own::AliasAnalysis aliases;
    {
      py::PerfScope perf("refcount-insertion.alias-analysis");
      aliases.build(module);
    }
    FuncContractCache contracts(module);

    mlir::func::FuncOp retain = findRetainFunction(module);
    {
      py::PerfScope perf("refcount-insertion.borrowed-return-retains");
      if (mlir::failed(
              insertBorrowedReturnRetains(module, retain, deallocators,
                                          aliases))) {
        signalPassFailure();
        return;
      }
    }

    llvm::SmallVector<mlir::func::CallOp, 32> calls;
    {
      py::PerfScope perf("refcount-insertion.collect-calls");
      module.walk([&](mlir::func::FuncOp function) {
        if (own::isRuntimeManifestFunction(function))
          return;
        function.walk([&](mlir::func::CallOp call) { calls.push_back(call); });
      });
    }

    {
      py::PerfScope perf("refcount-insertion.owned-result-releases");
      for (mlir::func::CallOp call : calls) {
        if (mlir::failed(insertOwnedResultReleases(module, call, contracts,
                                                   deallocators, aliases))) {
          signalPassFailure();
          return;
        }
      }
    }

    llvm::SmallVector<mlir::Operation *, 16> localObjects;
    {
      py::PerfScope perf("refcount-insertion.collect-local-objects");
      module.walk([&](mlir::Operation *op) {
        if (op->hasAttr(own::kOwnedLocalObjectContractAttr))
          localObjects.push_back(op);
      });
    }
    {
      py::PerfScope perf("refcount-insertion.local-object-releases");
      for (mlir::Operation *op : localObjects) {
        if (mlir::failed(insertOwnedLocalObjectReleases(
                module, op, contracts, deallocators, aliases))) {
          signalPassFailure();
          return;
        }
      }
    }

    {
      py::PerfScope perf("refcount-insertion.block-argument-releases");
      if (mlir::failed(insertOwnedBlockArgumentReleases(module, contracts,
                                                        deallocators, aliases)))
        signalPassFailure();
    }
  }
};

} // namespace
} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createRefCountInsertionPass() {
  return std::make_unique<lowering::RefCountInsertionPass>();
}

} // namespace py
