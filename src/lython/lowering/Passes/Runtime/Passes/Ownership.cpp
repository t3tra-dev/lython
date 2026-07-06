#include "Ownership.h"
#include "Common/Instrumentation.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"
#include "Runtime/Model/Contracts.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

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

mlir::LogicalResult insertBorrowedReturnRetains(
    mlir::ModuleOp module, mlir::func::FuncOp retain,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators) {
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
        if (own::valueGroupEqualsEntryArgumentGroup(function, group)) {
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

bool groupMatchesValues(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<mlir::Value> group,
                        own::AliasAnalysis &aliases) {
  if (offset + group.size() > values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(group)) {
    if (!aliases.same(values[offset + index], value))
      return false;
  }
  return true;
}

std::string logicalReturnObjectContract(mlir::Type type) {
  std::string contract = runtimeContractName(type);
  if (!contract.empty())
    return contract;
  if (mlir::isa<py::ProtocolType>(type))
    return "builtins.object";
  return "";
}

bool isNoneLikeType(mlir::Type type) { return py::isPyNoneType(type); }

std::optional<unsigned>
logicalReturnValueCount(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                        mlir::Type type) {
  if (isNoneLikeType(type))
    return 0;

  if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
    if (offset >= values.size() || !values[offset].getType().isInteger(64))
      return std::nullopt;
    unsigned size = 1;
    for (mlir::Type member : unionType.getMemberTypes()) {
      std::optional<unsigned> memberSize =
          logicalReturnValueCount(values, offset + size, deallocators, member);
      if (!memberSize)
        return std::nullopt;
      size += *memberSize;
    }
    return size;
  }

  std::string contract = logicalReturnObjectContract(type);
  if (contract.empty())
    return std::nullopt;
  const own::RuntimeDeallocator *deallocator =
      own::findDeallocatorForValueGroup(values, offset, deallocators, contract);
  if (!deallocator)
    return std::nullopt;
  return static_cast<unsigned>(deallocator->inputTypes.size());
}

unsigned skipPrimitiveReturnEvidence(mlir::ValueRange values, unsigned offset,
                                     mlir::Type type) {
  if (runtimeContractName(type) != "builtins.int")
    return offset;
  if (offset + 2 > values.size() || !values[offset].getType().isInteger(64) ||
      !values[offset + 1].getType().isInteger(1))
    return offset;
  return offset + 2;
}

struct OwnedReturnRange {
  unsigned offset = 0;
  unsigned size = 0;
  mlir::Type type;
};

std::optional<llvm::SmallVector<OwnedReturnRange, 4>> callableOwnedReturnRanges(
    mlir::func::FuncOp function, mlir::ValueRange values,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators) {
  auto callableAttr =
      function->getAttrOfType<mlir::TypeAttr>(own::kCallableTypeAttr);
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable)
    return std::nullopt;

  llvm::SmallVector<OwnedReturnRange, 4> ranges;
  unsigned offset = 0;
  for (mlir::Type resultType : callable.getResultTypes()) {
    std::optional<unsigned> size =
        logicalReturnValueCount(values, offset, deallocators, resultType);
    if (!size)
      return std::nullopt;
    if (*size > 0)
      ranges.push_back(OwnedReturnRange{offset, *size, resultType});
    offset += *size;
    offset = skipPrimitiveReturnEvidence(values, offset, resultType);
  }
  return ranges;
}

bool groupMatchesOwnedReturnRange(
    mlir::ValueRange values, const OwnedReturnRange &range,
    llvm::ArrayRef<mlir::Value> group,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;

  auto matchesLogicalValue = [&](auto &&self, unsigned offset,
                                 mlir::Type type) -> bool {
    if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
      if (group.size() == range.size &&
          groupMatchesValues(values, range.offset, group, aliases))
        return true;
      if (offset >= values.size() || !values[offset].getType().isInteger(64))
        return false;
      unsigned memberOffset = offset + 1;
      for (mlir::Type member : unionType.getMemberTypes()) {
        std::optional<unsigned> memberSize =
            logicalReturnValueCount(values, memberOffset, deallocators, member);
        if (!memberSize)
          return false;
        if (*memberSize > 0 && self(self, memberOffset, member))
          return true;
        memberOffset += *memberSize;
      }
      return false;
    }

    std::optional<unsigned> size =
        logicalReturnValueCount(values, offset, deallocators, type);
    return size && group.size() == *size &&
           groupMatchesValues(values, offset, group, aliases);
  };

  return matchesLogicalValue(matchesLogicalValue, range.offset, range.type);
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
      if (groupMatchesValues(returnOp.getOperands(), offset, group, aliases))
        return true;
  }

  if (!own::functionUsesOwnedReturnABI(function)) {
    return false;
  }

  std::optional<llvm::SmallVector<OwnedReturnRange, 4>> ranges =
      callableOwnedReturnRanges(function, returnOp.getOperands(), deallocators);
  if (!ranges) {
    for (unsigned offset = 0;
         offset + group.size() <= returnOp.getNumOperands(); ++offset)
      if (groupMatchesValues(returnOp.getOperands(), offset, group, aliases))
        return true;
    return false;
  }
  for (const OwnedReturnRange &range : *ranges)
    if (groupMatchesOwnedReturnRange(returnOp.getOperands(), range, group,
                                     deallocators, aliases))
      return true;
  return false;
}

mlir::Operation *ancestorInBlock(mlir::Operation *op, mlir::Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op && op->getBlock() == block ? op : nullptr;
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
                     own::AliasAnalysis &aliases, unsigned depth = 0) {
  if (!owner || group.empty() || depth > 16)
    return std::nullopt;
  mlir::Block *block = owner->getBlock();
  if (!block)
    return std::nullopt;

  mlir::Operation *lastUser = nullptr;
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
            std::optional<ReleaseInsertion> release =
                findReleaseInsertion(contracts, regionOwner, *mapped,
                                     deallocators, aliases, depth + 1);
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

mlir::LogicalResult insertConditionalOwnedResultRelease(
    FuncContractCache &contracts, mlir::func::CallOp call,
    const own::ResourceGroup &group, own::AliasAnalysis &aliases) {
  if (!group.condition)
    return mlir::failure();

  std::optional<ConditionalReleaseBlocks> blocks =
      findConditionalBranchAfterOwner(call, group.values, *group.condition,
                                      aliases);
  if (!blocks)
    return mlir::success();

  std::optional<mlir::Operation *> lastUser =
      findLastConditionalUserInActiveBlock(contracts, call, group.values,
                                           *blocks, aliases);
  if (!lastUser)
    return mlir::success();

  mlir::OpBuilder builder(call);
  if (*lastUser) {
    builder.setInsertionPointAfter(*lastUser);
  } else if (llvm::hasSingleElement(blocks->active->getPredecessors())) {
    builder.setInsertionPointToStart(blocks->active);
  } else {
    return insertReleaseOnActiveEdge(call, group, *blocks);
  }
  mlir::func::CallOp::create(builder, call.getLoc(),
                             group.deallocator->function, group.values);
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
      if (mlir::failed(insertConditionalOwnedResultRelease(contracts, call,
                                                           group, aliases)))
        return mlir::failure();
      continue;
    }

    std::optional<ReleaseInsertion> release = findReleaseInsertion(
        contracts, call, group.values, deallocators, aliases);
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

    std::optional<ReleaseInsertion> release = findReleaseInsertion(
        contracts, op, group.values, deallocators, aliases);
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
              insertBorrowedReturnRetains(module, retain, deallocators))) {
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
