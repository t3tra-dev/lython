#include "runtime/Detail.h"

#include "Contracts.h"
#include "Ownership.h"

#include "PyDialectTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace py::lowering {
namespace {

namespace own = py::ownership;
namespace contracts = py::contracts;

bool groupMatchesOperands(mlir::ValueRange operands, unsigned offset,
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

std::string logicalReturnObjectContract(mlir::Type type) {
  std::string contract = contracts::runtimeContractName(type);
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
  if (contracts::runtimeContractName(type) != "builtins.int")
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

std::optional<llvm::SmallVector<OwnedReturnRange, 4>>
callableOwnedReturnRanges(
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

bool groupMatchesOwnedReturnRange(mlir::ValueRange values,
                                  const OwnedReturnRange &range,
                                  llvm::ArrayRef<mlir::Value> group,
                                  llvm::ArrayRef<own::RuntimeDeallocator>
                                      deallocators,
                                  own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;

  auto matchesLogicalValue = [&](auto &&self, unsigned offset, mlir::Type type)
      -> bool {
    if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
      if (group.size() == range.size &&
          groupMatchesOperands(values, range.offset, group, aliases))
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
           groupMatchesOperands(values, offset, group, aliases);
  };

  return matchesLogicalValue(matchesLogicalValue, range.offset, range.type);
}

bool returnTransfersGroup(mlir::func::FuncOp function, mlir::func::ReturnOp ret,
                          llvm::ArrayRef<mlir::Value> group,
                          llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                          own::AliasAnalysis &aliases) {
  if (!own::functionUsesOwnedReturnABI(function)) {
    auto contract = own::readFunctionContract(function);
    if (mlir::failed(contract))
      return false;
    bool anyOwned = false;
    for (unsigned offset : contract->ownedResults.values) {
      if (groupMatchesOperands(ret.getOperands(), offset, group, aliases))
        anyOwned = true;
    }
    return anyOwned;
  }

  std::optional<llvm::SmallVector<OwnedReturnRange, 4>> ranges =
      callableOwnedReturnRanges(function, ret.getOperands(), deallocators);
  if (!ranges) {
    for (unsigned offset = 0; offset + group.size() <= ret.getNumOperands();
         ++offset) {
      if (groupMatchesOperands(ret.getOperands(), offset, group, aliases))
        return true;
    }
    return false;
  }
  for (const OwnedReturnRange &range : *ranges)
    if (groupMatchesOwnedReturnRange(ret.getOperands(), range, group,
                                     deallocators, aliases))
      return true;
  return false;
}

bool returnCarriesGroupInsideOwnedAggregate(
    mlir::func::FuncOp function, mlir::func::ReturnOp ret,
    llvm::ArrayRef<mlir::Value> group,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  if (!own::functionUsesOwnedReturnABI(function) || group.empty())
    return false;

  std::optional<llvm::SmallVector<OwnedReturnRange, 4>> ranges =
      callableOwnedReturnRanges(function, ret.getOperands(), deallocators);
  if (!ranges)
    return false;

  for (const OwnedReturnRange &range : *ranges) {
    if (group.size() >= range.size)
      continue;
    unsigned end = range.offset + range.size - static_cast<unsigned>(group.size());
    for (unsigned offset = range.offset; offset <= end; ++offset)
      if (groupMatchesOperands(ret.getOperands(), offset, group, aliases))
        return true;
  }
  return false;
}

struct TrackedResource {
  mlir::func::FuncOp function;
  mlir::Operation *producer = nullptr;
  std::string producerLabel;
  unsigned resultOffset = 0;
  llvm::SmallVector<mlir::Value, 4> group;
  std::optional<own::OwnershipCondition> condition;
};

std::string describeOwnershipProducer(mlir::Operation *op) {
  if (!op)
    return "<unknown>";
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
    return (llvm::Twine("@") + call.getCallee()).str();
  return op->getName().getStringRef().str();
}

enum class AffineTokenState { Owned, Released, Conditional };

struct AffinePathState {
  mlir::Block *block = nullptr;
  mlir::Operation *start = nullptr;
  AffineTokenState token = AffineTokenState::Owned;
  unsigned retained = 0;
  llvm::SmallVector<mlir::Value, 4> group;
};

struct BorrowedEntryResource {
  mlir::func::FuncOp function;
  unsigned logicalIndex = 0;
  unsigned inputOffset = 0;
  llvm::SmallVector<mlir::Value, 4> group;
};

struct BorrowedPathState {
  mlir::Block *block = nullptr;
  mlir::Operation *start = nullptr;
  unsigned retained = 0;
  llvm::SmallVector<mlir::Value, 4> group;
};

bool sameValueGroup(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip_equal(lhs, rhs))
    if (left != right)
      return false;
  return true;
}

bool samePathState(const AffinePathState &lhs, const AffinePathState &rhs) {
  return lhs.block == rhs.block && lhs.start == rhs.start &&
         lhs.token == rhs.token && lhs.retained == rhs.retained &&
         sameValueGroup(lhs.group, rhs.group);
}

bool containsPathState(llvm::ArrayRef<AffinePathState> states,
                       const AffinePathState &candidate) {
  return llvm::any_of(states, [&](const AffinePathState &state) {
    return samePathState(state, candidate);
  });
}

bool sameBorrowedPathState(const BorrowedPathState &lhs,
                           const BorrowedPathState &rhs) {
  return lhs.block == rhs.block && lhs.start == rhs.start &&
         lhs.retained == rhs.retained && sameValueGroup(lhs.group, rhs.group);
}

bool containsBorrowedPathState(llvm::ArrayRef<BorrowedPathState> states,
                               const BorrowedPathState &candidate) {
  return llvm::any_of(states, [&](const BorrowedPathState &state) {
    return sameBorrowedPathState(state, candidate);
  });
}

bool pathReenteredBeforeTrackedDefinition(const AffinePathState &state) {
  if (!state.start)
    return false;
  for (mlir::Value value : state.group) {
    mlir::Operation *definition = value ? value.getDefiningOp() : nullptr;
    if (!definition || definition->getBlock() != state.block)
      continue;
    if (state.start == definition || state.start->isBeforeInBlock(definition))
      return true;
  }
  return false;
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

bool regionContainsGroupUse(mlir::Region &region,
                            llvm::ArrayRef<mlir::Value> group,
                            own::AliasAnalysis &aliases) {
  bool found = false;
  region.walk([&](mlir::Operation *op) {
    if (found)
      return;
    if (groupContainsOperand(op, group, aliases))
      found = true;
  });
  return found;
}

bool valueDefinedInsideRegion(mlir::Value value, mlir::Region *region) {
  if (!value || !region)
    return false;
  mlir::Region *parent = value.getParentRegion();
  return parent && region->isAncestor(parent);
}

bool groupHasValueDefinedInsideRegion(llvm::ArrayRef<mlir::Value> group,
                                      mlir::Region *region) {
  return llvm::any_of(group, [&](mlir::Value value) {
    return valueDefinedInsideRegion(value, region);
  });
}

bool callConsumesGroup(mlir::ModuleOp module, mlir::func::CallOp call,
                       llvm::ArrayRef<mlir::Value> group,
                       own::AliasAnalysis &aliases) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return false;
  auto contract = own::readFunctionContract(callee);
  if (mlir::failed(contract))
    return false;
  for (unsigned offset : contract->releaseArgs.values)
    if (groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : contract->transferArgs.values)
    if (groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

std::optional<llvm::SmallVector<mlir::Value, 4>>
callTransfersGroupToOwnedResult(mlir::ModuleOp module, mlir::func::CallOp call,
                                llvm::ArrayRef<mlir::Value> group,
                                own::AliasAnalysis &aliases) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return std::nullopt;
  auto contract = own::readFunctionContract(callee);
  if (mlir::failed(contract))
    return std::nullopt;

  bool transfers = false;
  for (unsigned offset : contract->transferArgs.values) {
    if (groupMatchesOperands(call.getOperands(), offset, group, aliases)) {
      transfers = true;
      break;
    }
  }
  if (!transfers)
    return std::nullopt;

  for (unsigned offset : contract->ownedResults.values) {
    if (offset + group.size() > call.getNumResults())
      continue;
    bool typesMatch = true;
    for (unsigned index = 0; index < group.size(); ++index) {
      if (call.getResult(offset + index).getType() != group[index].getType()) {
        typesMatch = false;
        break;
      }
    }
    if (!typesMatch)
      continue;
    llvm::SmallVector<mlir::Value, 4> replacement;
    replacement.reserve(group.size());
    for (unsigned index = 0; index < group.size(); ++index)
      replacement.push_back(call.getResult(offset + index));
    return replacement;
  }
  return std::nullopt;
}

bool callRetainsGroup(mlir::ModuleOp module, mlir::func::CallOp call,
                      llvm::ArrayRef<mlir::Value> group,
                      own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return false;
  auto contract = own::readFunctionContract(callee);
  if (mlir::failed(contract))
    return false;
  for (unsigned offset : contract->retainArgs.values) {
    if (offset >= call.getNumOperands())
      continue;
    if (aliases.same(call.getOperand(offset), group.front()))
      return true;
  }
  return false;
}

bool callPartiallyConsumesGroup(mlir::ModuleOp module, mlir::func::CallOp call,
                                llvm::ArrayRef<mlir::Value> group,
                                own::AliasAnalysis &aliases) {
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return false;
  auto contract = own::readFunctionContract(callee);
  if (mlir::failed(contract))
    return false;

  auto consumesTrackedHeaderAt = [&](unsigned index) {
    return !group.empty() && index < call.getNumOperands() &&
           aliases.same(call.getOperand(index), group.front());
  };
  for (unsigned offset : contract->releaseArgs.values)
    if (consumesTrackedHeaderAt(offset) &&
        !groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : contract->transferArgs.values)
    if (consumesTrackedHeaderAt(offset) &&
        !groupMatchesOperands(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

bool returnConsumesGroup(mlir::func::FuncOp function, mlir::func::ReturnOp ret,
                         llvm::ArrayRef<mlir::Value> group,
                         llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                         own::AliasAnalysis &aliases) {
  return returnTransfersGroup(function, ret, group, deallocators, aliases);
}

llvm::SmallVector<mlir::Value, 4>
remapGroupForSuccessor(mlir::Operation *terminator, unsigned successorIndex,
                       mlir::Block *successor,
                       llvm::ArrayRef<mlir::Value> group,
                       own::AliasAnalysis &aliases,
                       llvm::SmallVectorImpl<bool> *mappedMask = nullptr) {
  llvm::SmallVector<mlir::Value, 4> mapped(group.begin(), group.end());
  if (mappedMask) {
    mappedMask->clear();
    mappedMask->append(group.size(), false);
  }
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return mapped;

  mlir::SuccessorOperands operands =
      branch.getSuccessorOperands(successorIndex);
  unsigned argumentCount =
      std::min<unsigned>(successor->getNumArguments(), operands.size());
  for (auto [groupIndex, value] : llvm::enumerate(group)) {
    for (unsigned argumentIndex = 0; argumentIndex < argumentCount;
         ++argumentIndex) {
      mlir::Value forwarded = operands[argumentIndex];
      if (!forwarded || !aliases.same(forwarded, value))
        continue;
      mapped[groupIndex] = successor->getArgument(argumentIndex);
      if (mappedMask)
        (*mappedMask)[groupIndex] = true;
      break;
    }
  }
  return mapped;
}

bool groupContainsArgumentFromBlock(llvm::ArrayRef<mlir::Value> group,
                                    mlir::Block *block) {
  return llvm::any_of(group, [&](mlir::Value value) {
    auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(value);
    return argument && argument.getOwner() == block;
  });
}

llvm::SmallVector<mlir::Type, 8>
callableLogicalInputTypes(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Type, 8> types;
  auto callableAttr =
      function->getAttrOfType<mlir::TypeAttr>(own::kCallableTypeAttr);
  auto callable =
      mlir::dyn_cast_if_present<py::CallableType>(
          callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable)
    return types;
  types.append(callable.getPositionalTypes().begin(),
               callable.getPositionalTypes().end());
  types.append(callable.getKwOnlyTypes().begin(),
               callable.getKwOnlyTypes().end());
  if (callable.hasVararg())
    types.push_back(callable.getVarargType());
  if (callable.hasKwarg())
    types.push_back(callable.getKwargType());

  auto closureTypes = function->getAttrOfType<mlir::ArrayAttr>("closure_types");
  if (!closureTypes)
    return types;
  for (mlir::Attribute attr : closureTypes) {
    auto typeAttr = mlir::dyn_cast<mlir::TypeAttr>(attr);
    if (!typeAttr)
      return types;
    types.push_back(typeAttr.getValue());
  }
  return types;
}

bool logicalTypeHasPrimitiveI64Evidence(mlir::Type type) {
  return contracts::runtimeContractName(type) == "builtins.int";
}

void skipPrimitiveI64Evidence(mlir::Block &entry, unsigned &offset) {
  if (offset + 2 > entry.getNumArguments())
    return;
  if (!entry.getArgument(offset).getType().isInteger(64) ||
      !entry.getArgument(offset + 1).getType().isInteger(1))
    return;
  offset += 2;
}

llvm::SmallVector<BorrowedEntryResource, 8> collectBorrowedEntryResources(
    mlir::func::FuncOp function,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators) {
  llvm::SmallVector<BorrowedEntryResource, 8> resources;
  if (!function || function.isDeclaration() || function.empty() ||
      own::isRuntimeManifestFunction(function))
    return resources;

  llvm::SmallVector<mlir::Type, 8> logicalTypes =
      callableLogicalInputTypes(function);
  if (logicalTypes.empty())
    return resources;

  auto contract = own::readFunctionContract(function);
  if (mlir::failed(contract))
    return resources;

  mlir::Block &entry = function.front();
  unsigned offset = 0;
  for (auto [logicalIndex, logicalType] : llvm::enumerate(logicalTypes)) {
    if (offset >= entry.getNumArguments())
      break;

    unsigned groupOffset = offset;
    llvm::SmallVector<mlir::Value, 4> group;
    std::string contractName = contracts::runtimeContractName(logicalType);
    if (!contractName.empty()) {
      if (const own::RuntimeDeallocator *deallocator =
              own::findDeallocatorForValueGroup(entry.getArguments(), offset,
                                                deallocators, contractName)) {
        unsigned size = static_cast<unsigned>(deallocator->inputTypes.size());
        group = own::valueSlice(entry.getArguments(), offset, size);
        offset += size;
      } else if (own::isObjectHeaderLikeType(
                     entry.getArgument(offset).getType())) {
        group.push_back(entry.getArgument(offset));
        ++offset;
      } else {
        ++offset;
      }
      if (logicalTypeHasPrimitiveI64Evidence(logicalType))
        skipPrimitiveI64Evidence(entry, offset);
    } else {
      ++offset;
    }

    own::OwnershipKind ownership =
        own::logicalOwnershipKind(logicalType, /*ownsObject=*/false);
    if (group.empty() || ownership != own::OwnershipKind::Borrow)
      continue;
    if (contract->consumesArg(groupOffset))
      continue;

    BorrowedEntryResource resource;
    resource.function = function;
    resource.logicalIndex = static_cast<unsigned>(logicalIndex);
    resource.inputOffset = groupOffset;
    resource.group = std::move(group);
    resources.push_back(std::move(resource));
  }
  return resources;
}

mlir::Operation *firstOperation(mlir::Block *block) {
  if (!block || block->empty())
    return nullptr;
  return &block->front();
}

mlir::Operation *firstOperation(mlir::Region *region) {
  if (!region || region->empty())
    return nullptr;
  return firstOperation(&region->front());
}

llvm::SmallVector<mlir::Attribute, 8>
unknownOperandConstants(mlir::Operation *op) {
  return llvm::SmallVector<mlir::Attribute, 8>(op->getNumOperands(),
                                               mlir::Attribute());
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

void enqueueRegionSuccessor(mlir::Operation *owner, mlir::RegionSuccessor succ,
                            AffinePathState state,
                            llvm::SmallVectorImpl<AffinePathState> &worklist) {
  if (succ.isParent()) {
    state.block = owner->getBlock();
    state.start = owner->getNextNode();
  } else {
    state.block =
        succ.getSuccessor()->empty() ? nullptr : &succ.getSuccessor()->front();
    state.start = firstOperation(succ.getSuccessor());
  }
  worklist.push_back(std::move(state));
}

bool enqueueRegionEntryPaths(mlir::Operation *op, AffinePathState state,
                             own::AliasAnalysis &aliases,
                             llvm::SmallVectorImpl<AffinePathState> &worklist) {
  auto branch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op);
  if (!branch)
    return false;

  llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
  llvm::SmallVector<mlir::Attribute, 8> operandConstants =
      unknownOperandConstants(op);
  branch.getEntrySuccessorRegions(operandConstants, successors);
  if (successors.empty())
    return false;

  bool handled = false;
  bool hasNoUseRegionPath = false;
  for (mlir::RegionSuccessor successor : successors) {
    if (successor.isParent()) {
      AffinePathState next = state;
      mlir::OperandRange sources =
          branch.getEntrySuccessorOperands(mlir::RegionBranchPoint(successor));
      next.group = remapGroupThroughValueMapping(
          sources, successor.getSuccessorInputs(), state.group, aliases);
      enqueueRegionSuccessor(op, successor, std::move(next), worklist);
      handled = true;
      continue;
    }

    mlir::Region *region = successor.getSuccessor();
    if (!regionContainsGroupUse(*region, state.group, aliases)) {
      hasNoUseRegionPath = true;
      continue;
    }

    AffinePathState next = state;
    mlir::OperandRange sources =
        branch.getEntrySuccessorOperands(mlir::RegionBranchPoint(successor));
    next.group = remapGroupThroughValueMapping(
        sources, successor.getSuccessorInputs(), state.group, aliases);
    enqueueRegionSuccessor(op, successor, std::move(next), worklist);
    handled = true;
  }

  if (hasNoUseRegionPath) {
    AffinePathState next = state;
    next.block = op->getBlock();
    next.start = op->getNextNode();
    worklist.push_back(std::move(next));
    handled = true;
  }

  return handled;
}

void enqueueBorrowedRegionSuccessor(
    mlir::Operation *owner, mlir::RegionSuccessor succ, BorrowedPathState state,
    llvm::SmallVectorImpl<BorrowedPathState> &worklist) {
  if (succ.isParent()) {
    state.block = owner->getBlock();
    state.start = owner->getNextNode();
  } else {
    state.block =
        succ.getSuccessor()->empty() ? nullptr : &succ.getSuccessor()->front();
    state.start = firstOperation(succ.getSuccessor());
  }
  worklist.push_back(std::move(state));
}

bool enqueueBorrowedRegionEntryPaths(
    mlir::Operation *op, BorrowedPathState state, own::AliasAnalysis &aliases,
    llvm::SmallVectorImpl<BorrowedPathState> &worklist) {
  auto branch = mlir::dyn_cast<mlir::RegionBranchOpInterface>(op);
  if (!branch)
    return false;

  llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
  llvm::SmallVector<mlir::Attribute, 8> operandConstants =
      unknownOperandConstants(op);
  branch.getEntrySuccessorRegions(operandConstants, successors);
  if (successors.empty())
    return false;

  bool handled = false;
  bool hasNoUseRegionPath = false;
  for (mlir::RegionSuccessor successor : successors) {
    if (successor.isParent()) {
      BorrowedPathState next = state;
      mlir::OperandRange sources =
          branch.getEntrySuccessorOperands(mlir::RegionBranchPoint(successor));
      next.group = remapGroupThroughValueMapping(
          sources, successor.getSuccessorInputs(), state.group, aliases);
      enqueueBorrowedRegionSuccessor(op, successor, std::move(next), worklist);
      handled = true;
      continue;
    }

    mlir::Region *region = successor.getSuccessor();
    if (!regionContainsGroupUse(*region, state.group, aliases)) {
      hasNoUseRegionPath = true;
      continue;
    }

    BorrowedPathState next = state;
    mlir::OperandRange sources =
        branch.getEntrySuccessorOperands(mlir::RegionBranchPoint(successor));
    next.group = remapGroupThroughValueMapping(
        sources, successor.getSuccessorInputs(), state.group, aliases);
    enqueueBorrowedRegionSuccessor(op, successor, std::move(next), worklist);
    handled = true;
  }

  if (hasNoUseRegionPath) {
    BorrowedPathState next = state;
    next.block = op->getBlock();
    next.start = op->getNextNode();
    worklist.push_back(std::move(next));
    handled = true;
  }

  return handled;
}

mlir::LogicalResult
handleRegionTerminator(mlir::Operation *terminator, TrackedResource &resource,
                       AffinePathState state, own::AliasAnalysis &aliases,
                       llvm::SmallVectorImpl<AffinePathState> &worklist) {
  auto regionTerminator =
      mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(terminator);
  mlir::Operation *owner = terminator->getParentOp();
  mlir::Region *currentRegion = terminator->getParentRegion();
  if (!regionTerminator || !owner || !currentRegion)
    return mlir::failure();

  if (state.token == AffineTokenState::Released) {
    if (groupContainsOperand(terminator, state.group, aliases) &&
        state.retained == 0)
      return terminator->emitError()
             << "released owned resource from " << resource.producerLabel
             << " is used by region terminator";

    if (!groupHasValueDefinedInsideRegion(state.group, currentRegion)) {
      llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
      regionTerminator.getSuccessorRegions(unknownOperandConstants(terminator),
                                           successors);
      for (mlir::RegionSuccessor successor : successors)
        enqueueRegionSuccessor(owner, successor, state, worklist);
    }
    return mlir::success();
  }

  llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
  regionTerminator.getSuccessorRegions(unknownOperandConstants(terminator),
                                       successors);
  if (successors.empty())
    return terminator->emitError()
           << "owned resource from " << resource.producerLabel
           << " reaches region exit without a successor";

  bool localGroup =
      groupHasValueDefinedInsideRegion(state.group, currentRegion);
  bool enqueued = false;
  for (mlir::RegionSuccessor successor : successors) {
    llvm::SmallVector<bool, 4> mappedMask;
    llvm::SmallVector<mlir::Value, 4> mappedGroup =
        remapGroupThroughValueMapping(terminator->getOperands(),
                                      successor.getSuccessorInputs(),
                                      state.group, aliases, &mappedMask);
    bool fullyMapped =
        llvm::all_of(mappedMask, [](bool mapped) { return mapped; });

    if (localGroup && !fullyMapped)
      continue;

    AffinePathState next = state;
    next.group = std::move(mappedGroup);
    enqueueRegionSuccessor(owner, successor, std::move(next), worklist);
    enqueued = true;
  }

  if (!enqueued) {
    if (localGroup)
      return terminator->emitError()
             << "owned resource from " << resource.producerLabel
             << " result " << resource.resultOffset
             << " is produced inside a region but not yielded to any "
                "successor";
    AffinePathState next = state;
    next.block = owner->getBlock();
    next.start = owner->getNextNode();
    worklist.push_back(std::move(next));
  }

  return mlir::success();
}

mlir::LogicalResult
handleGenericRegionReturn(mlir::Operation *terminator,
                          TrackedResource &resource, AffinePathState state,
                          own::AliasAnalysis &aliases,
                          llvm::SmallVectorImpl<AffinePathState> &worklist) {
  mlir::Operation *owner = terminator->getParentOp();
  mlir::Region *currentRegion = terminator->getParentRegion();
  if (!owner || !currentRegion)
    return mlir::failure();

  bool localGroup =
      groupHasValueDefinedInsideRegion(state.group, currentRegion);
  if (state.token == AffineTokenState::Released) {
    if (groupContainsOperand(terminator, state.group, aliases) &&
        state.retained == 0)
      return terminator->emitError()
             << "released owned resource from " << resource.producerLabel
             << " is used by region terminator";
    if (!localGroup) {
      AffinePathState next = state;
      next.block = owner->getBlock();
      next.start = owner->getNextNode();
      worklist.push_back(std::move(next));
    }
    return mlir::success();
  }

  llvm::SmallVector<bool, 4> mappedMask;
  llvm::SmallVector<mlir::Value, 4> mappedGroup = remapGroupThroughValueMapping(
      terminator->getOperands(), owner->getResults(), state.group, aliases,
      &mappedMask);
  bool fullyMapped =
      llvm::all_of(mappedMask, [](bool mapped) { return mapped; });

  if (localGroup && !fullyMapped)
    return terminator->emitError()
           << "owned resource from " << resource.producerLabel
           << " result " << resource.resultOffset
           << " is produced inside a region but not yielded to the parent "
              "operation";

  AffinePathState next = state;
  next.block = owner->getBlock();
  next.start = owner->getNextNode();
  if (fullyMapped)
    next.group = std::move(mappedGroup);
  worklist.push_back(std::move(next));
  return mlir::success();
}

mlir::LogicalResult handleBorrowedRegionTerminator(
    mlir::Operation *terminator, BorrowedEntryResource &resource,
    BorrowedPathState state, own::AliasAnalysis &aliases,
    llvm::SmallVectorImpl<BorrowedPathState> &worklist) {
  auto regionTerminator =
      mlir::dyn_cast<mlir::RegionBranchTerminatorOpInterface>(terminator);
  mlir::Operation *owner = terminator->getParentOp();
  mlir::Region *currentRegion = terminator->getParentRegion();
  if (!regionTerminator || !owner || !currentRegion)
    return mlir::failure();

  llvm::SmallVector<mlir::RegionSuccessor, 4> successors;
  regionTerminator.getSuccessorRegions(unknownOperandConstants(terminator),
                                       successors);
  if (successors.empty()) {
    if (state.retained != 0)
      return terminator->emitError()
             << "borrowed entry argument " << resource.logicalIndex << " of @"
             << resource.function.getSymName()
             << " retains ownership but reaches a region exit without release "
                "or transfer";
    return mlir::success();
  }

  bool localGroup =
      groupHasValueDefinedInsideRegion(state.group, currentRegion);
  bool enqueued = false;
  for (mlir::RegionSuccessor successor : successors) {
    llvm::SmallVector<bool, 4> mappedMask;
    llvm::SmallVector<mlir::Value, 4> mappedGroup =
        remapGroupThroughValueMapping(terminator->getOperands(),
                                      successor.getSuccessorInputs(),
                                      state.group, aliases, &mappedMask);
    bool fullyMapped =
        llvm::all_of(mappedMask, [](bool mapped) { return mapped; });

    if (localGroup && !fullyMapped)
      continue;

    BorrowedPathState next = state;
    next.group = std::move(mappedGroup);
    enqueueBorrowedRegionSuccessor(owner, successor, std::move(next),
                                   worklist);
    enqueued = true;
  }

  if (!enqueued) {
    BorrowedPathState next = state;
    next.block = owner->getBlock();
    next.start = owner->getNextNode();
    worklist.push_back(std::move(next));
  }

  return mlir::success();
}

mlir::LogicalResult handleBorrowedGenericRegionReturn(
    mlir::Operation *terminator, BorrowedEntryResource &resource,
    BorrowedPathState state, own::AliasAnalysis &aliases,
    llvm::SmallVectorImpl<BorrowedPathState> &worklist) {
  mlir::Operation *owner = terminator->getParentOp();
  mlir::Region *currentRegion = terminator->getParentRegion();
  if (!owner || !currentRegion)
    return mlir::failure();

  bool localGroup =
      groupHasValueDefinedInsideRegion(state.group, currentRegion);
  llvm::SmallVector<bool, 4> mappedMask;
  llvm::SmallVector<mlir::Value, 4> mappedGroup = remapGroupThroughValueMapping(
      terminator->getOperands(), owner->getResults(), state.group, aliases,
      &mappedMask);
  bool fullyMapped =
      llvm::all_of(mappedMask, [](bool mapped) { return mapped; });

  if (localGroup && !fullyMapped) {
    if (state.retained != 0)
      return terminator->emitError()
             << "borrowed entry argument " << resource.logicalIndex << " of @"
             << resource.function.getSymName()
             << " retains ownership inside a region but is not yielded to the "
                "parent operation";
    return mlir::success();
  }

  BorrowedPathState next = state;
  next.block = owner->getBlock();
  next.start = owner->getNextNode();
  if (fullyMapped)
    next.group = std::move(mappedGroup);
  worklist.push_back(std::move(next));
  return mlir::success();
}

mlir::LogicalResult verifyBorrowedEntryOnCFGPaths(
    mlir::ModuleOp module, BorrowedEntryResource &resource,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  llvm::SmallVector<BorrowedPathState, 16> worklist;
  llvm::SmallVector<BorrowedPathState, 32> visited;
  mlir::Block &entry = resource.function.front();
  worklist.push_back(BorrowedPathState{&entry, firstOperation(&entry),
                                       /*retained=*/0, resource.group});

  constexpr unsigned kMaxBorrowedStates = 20000;
  constexpr unsigned kMaxRetainedBalance = 64;
  while (!worklist.empty()) {
    BorrowedPathState state = worklist.pop_back_val();
    if (containsBorrowedPathState(visited, state))
      continue;
    visited.push_back(state);
    if (visited.size() > kMaxBorrowedStates)
      return resource.function.emitError()
             << "borrowed entry ownership CFG exploration exceeded "
             << kMaxBorrowedStates << " states";

    mlir::Operation *op = state.start;
    while (op) {
      if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        bool consumes =
            returnConsumesGroup(resource.function, ret, state.group,
                                deallocators, aliases);
        if (consumes) {
          if (state.retained == 0)
            return ret.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName()
                   << " is returned as owned without a dominating retain";
          if (state.retained != 1)
            return ret.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName()
                   << " is returned with " << state.retained
                   << " retained ownership tokens; exactly one may be "
                      "transferred";
          break;
        }
        if (state.retained != 0)
          return ret.emitError()
                 << "borrowed entry argument " << resource.logicalIndex
                 << " of @" << resource.function.getSymName()
                 << " reaches function exit with " << state.retained
                 << " retained ownership token(s)";
        break;
      }

      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        if (callPartiallyConsumesGroup(module, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of borrowed "
                    "entry argument "
                 << resource.logicalIndex << " of @"
                 << resource.function.getSymName();

        if (callConsumesGroup(module, call, state.group, aliases)) {
          if (state.retained == 0)
            return call.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName()
                   << " is released or transferred without a prior retain";
          --state.retained;
          if (std::optional<llvm::SmallVector<mlir::Value, 4>> replacement =
                  callTransfersGroupToOwnedResult(module, call, state.group,
                                                  aliases))
            state.group = std::move(*replacement);
        }

        if (callRetainsGroup(module, call, state.group, aliases)) {
          if (call->hasAttr(own::kAggregateRetainAttr)) {
            op = op->getNextNode();
            continue;
          }
          if (state.retained >= kMaxRetainedBalance)
            return call.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName()
                   << " retain balance exceeded " << kMaxRetainedBalance;
          ++state.retained;
        }
      }

      if (op->getNumRegions() != 0 &&
          enqueueBorrowedRegionEntryPaths(op, state, aliases, worklist)) {
        op = nullptr;
        break;
      }

      if (op->hasTrait<mlir::OpTrait::IsTerminator>())
        break;
      op = op->getNextNode();
    }

    if (!op)
      continue;
    if (mlir::isa<mlir::func::ReturnOp>(op))
      continue;

    if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
      if (mlir::failed(handleBorrowedGenericRegionReturn(
              op, resource, std::move(state), aliases, worklist)))
        return mlir::failure();
      continue;
    }

    if (mlir::isa<mlir::RegionBranchTerminatorOpInterface>(op)) {
      if (mlir::failed(handleBorrowedRegionTerminator(
              op, resource, std::move(state), aliases, worklist)))
        return mlir::failure();
      continue;
    }

    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::Operation *owner = op->getParentRegion()
                                   ? op->getParentRegion()->getParentOp()
                                   : nullptr;
      if (owner && !mlir::isa<mlir::func::FuncOp>(owner)) {
        if (mlir::failed(handleBorrowedGenericRegionReturn(
                op, resource, std::move(state), aliases, worklist)))
          return mlir::failure();
        continue;
      }
    }

    unsigned successors = op->getNumSuccessors();
    if (successors == 0) {
      if (state.retained != 0)
        return op->emitError()
               << "borrowed entry argument " << resource.logicalIndex
               << " of @" << resource.function.getSymName()
               << " reaches a CFG exit with " << state.retained
               << " retained ownership token(s)";
      continue;
    }

    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      BorrowedPathState next = state;
      next.block = successor;
      next.start = firstOperation(successor);
      next.group =
          remapGroupForSuccessor(op, index, successor, state.group, aliases);
      worklist.push_back(std::move(next));
    }
  }

  return mlir::success();
}

mlir::LogicalResult verifyResourceOnCFGPaths(mlir::ModuleOp module,
                                             TrackedResource &resource,
                                             llvm::ArrayRef<own::RuntimeDeallocator>
                                                 deallocators,
                                             own::AliasAnalysis &aliases) {
  llvm::SmallVector<AffinePathState, 16> worklist;
  llvm::SmallVector<AffinePathState, 32> visited;
  AffineTokenState initialToken = resource.condition
                                      ? AffineTokenState::Conditional
                                      : AffineTokenState::Owned;
  worklist.push_back(AffinePathState{resource.producer->getBlock(),
                                     resource.producer->getNextNode(),
                                     initialToken, /*retained=*/0,
                                     resource.group});

  constexpr unsigned kMaxAffineStates = 20000;
  while (!worklist.empty()) {
    AffinePathState state = worklist.pop_back_val();
    if (containsPathState(visited, state))
      continue;
    visited.push_back(state);
    if (visited.size() > kMaxAffineStates)
      return resource.producer->emitError()
             << "ownership CFG exploration exceeded " << kMaxAffineStates
             << " states";

    if (pathReenteredBeforeTrackedDefinition(state)) {
      if (state.token == AffineTokenState::Released)
        continue;
      if (state.token == AffineTokenState::Owned)
        return state.start->emitError()
               << "owned resource from " << resource.producerLabel
               << " result " << resource.resultOffset
               << " reaches the next loop iteration without release, "
                  "transfer, or owned return";
      return state.start->emitError()
             << "conditionally owned resource from "
             << resource.producerLabel << " result " << resource.resultOffset
             << " reaches the next loop iteration without tag-conditioned "
                "release, transfer, or owned return";
    }

    mlir::Operation *op = state.start;
    while (op) {
      if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        bool consumes =
            returnConsumesGroup(resource.function, ret, state.group,
                                deallocators, aliases);
        bool uses = groupContainsOperand(op, state.group, aliases);
        if (state.token == AffineTokenState::Owned) {
          if (!consumes)
            return ret.emitError()
                   << "owned resource from " << resource.producerLabel
                   << " result " << resource.resultOffset
                   << " reaches function exit without release, transfer, or "
                      "owned return";
          if (state.retained != 0)
            return ret.emitError()
                   << "owned resource from " << resource.producerLabel
                   << " result " << resource.resultOffset << " is returned with "
                   << state.retained
                   << " additional retained ownership token(s)";
          break;
        }
        if (state.token == AffineTokenState::Conditional) {
          if (consumes)
            break;
          if (uses)
            return ret.emitError()
                   << "conditionally owned resource from "
                   << resource.producerLabel << " result "
                   << resource.resultOffset
                   << " is returned before its union tag proves the payload "
                      "active";
          return ret.emitError()
                 << "conditionally owned resource from "
                 << resource.producerLabel << " result "
                 << resource.resultOffset
                 << " reaches function exit without tag-conditioned release, "
                    "transfer, or owned return";
        }
        if (uses) {
          if (state.retained > 0 &&
              returnCarriesGroupInsideOwnedAggregate(
                  resource.function, ret, state.group, deallocators, aliases))
            break;
          return ret.emitError() << "released owned resource from "
                                 << resource.producerLabel
                                 << " is used by function return";
        }
        break;
      }

      if (state.token == AffineTokenState::Conditional) {
        if (resource.condition) {
          if (std::optional<own::OwnershipConditionBranch> branch =
                  own::classifyOwnershipConditionBranch(
                      op, *resource.condition)) {
            for (auto [successorIndex, nextToken] :
                 {std::pair<unsigned, AffineTokenState>{
                      branch->activeSuccessor, AffineTokenState::Owned},
                  std::pair<unsigned, AffineTokenState>{
                      branch->inactiveSuccessor, AffineTokenState::Released}}) {
              mlir::Block *successor = op->getSuccessor(successorIndex);
              llvm::SmallVector<mlir::Value, 4> mappedGroup =
                  remapGroupForSuccessor(op, successorIndex, successor,
                                         state.group, aliases);
              worklist.push_back(
                  AffinePathState{successor, firstOperation(successor),
                                  nextToken, state.retained,
                                  std::move(mappedGroup)});
            }
            op = nullptr;
            break;
          }
        }
        if (!op->hasTrait<mlir::OpTrait::IsTerminator>() &&
            groupContainsOperand(op, state.group, aliases))
          return op->emitError()
                 << "conditionally owned resource from "
                 << resource.producerLabel << " result "
                 << resource.resultOffset
                 << " is used before its union tag proves the payload active";
      }

      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        bool consumes = callConsumesGroup(module, call, state.group, aliases);
        bool retains = callRetainsGroup(module, call, state.group, aliases);
        if (callPartiallyConsumesGroup(module, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of owned "
                    "resource group produced by "
                 << resource.producerLabel << " result "
                 << resource.resultOffset;

        if (state.token == AffineTokenState::Released) {
          if (consumes) {
            if (state.retained == 0)
              return call.emitError()
                     << "owned resource from " << resource.producerLabel
                     << " result " << resource.resultOffset
                     << " is released or transferred more than once on one CFG "
                        "path";
            --state.retained;
            op = op->getNextNode();
            continue;
          }
          if (groupContainsOperand(op, state.group, aliases) &&
              state.retained == 0)
            return call.emitError()
                   << "released owned resource from "
                   << resource.producerLabel << " is used after release";
          if (retains)
            ++state.retained;
        } else if (state.token == AffineTokenState::Owned && consumes) {
          if (std::optional<llvm::SmallVector<mlir::Value, 4>> replacement =
                  callTransfersGroupToOwnedResult(module, call, state.group,
                                                  aliases)) {
            state.group = std::move(*replacement);
          } else {
            state.token = AffineTokenState::Released;
          }
        }
        if (state.token == AffineTokenState::Owned && retains)
          ++state.retained;
      } else if (state.token == AffineTokenState::Released &&
                 groupContainsOperand(op, state.group, aliases) &&
                 state.retained == 0) {
        return op->emitError()
               << "released owned resource from " << resource.producerLabel
               << " is used after release";
      }

      if (op->getNumRegions() != 0 &&
          enqueueRegionEntryPaths(op, state, aliases, worklist)) {
        op = nullptr;
        break;
      }

      if (op->hasTrait<mlir::OpTrait::IsTerminator>())
        break;
      op = op->getNextNode();
    }

    if (!op)
      continue;
    if (mlir::isa<mlir::func::ReturnOp>(op))
      continue;

    if (op->hasTrait<mlir::OpTrait::ReturnLike>()) {
      if (mlir::failed(handleGenericRegionReturn(op, resource, std::move(state),
                                                 aliases, worklist)))
        return mlir::failure();
      continue;
    }

    if (mlir::isa<mlir::RegionBranchTerminatorOpInterface>(op)) {
      if (mlir::failed(handleRegionTerminator(op, resource, std::move(state),
                                              aliases, worklist)))
        return mlir::failure();
      continue;
    }

    if (op->hasTrait<mlir::OpTrait::IsTerminator>()) {
      mlir::Operation *owner = op->getParentRegion()
                                   ? op->getParentRegion()->getParentOp()
                                   : nullptr;
      if (owner && !mlir::isa<mlir::func::FuncOp>(owner)) {
        if (mlir::failed(handleGenericRegionReturn(
                op, resource, std::move(state), aliases, worklist)))
          return mlir::failure();
        continue;
      }
    }

    unsigned successors = op->getNumSuccessors();
    if (successors == 0) {
      if (state.token == AffineTokenState::Owned)
        return op->emitError()
               << "owned resource from " << resource.producerLabel
               << " result " << resource.resultOffset
               << " reaches a CFG exit without release, transfer, or owned "
                  "return";
      if (state.token == AffineTokenState::Conditional)
        return op->emitError()
               << "conditionally owned resource from "
               << resource.producerLabel << " result "
               << resource.resultOffset
               << " reaches a CFG exit without tag-conditioned release, "
                  "transfer, or owned return";
      continue;
    }

    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      llvm::SmallVector<bool, 4> mappedMask;
      llvm::SmallVector<mlir::Value, 4> mappedGroup = remapGroupForSuccessor(
          op, index, successor, state.group, aliases, &mappedMask);
      bool fullyMapped =
          llvm::all_of(mappedMask, [](bool mapped) { return mapped; });
      if (!fullyMapped && state.token == AffineTokenState::Released &&
          groupContainsArgumentFromBlock(state.group, successor))
        continue;
      worklist.push_back(AffinePathState{successor, firstOperation(successor),
                                         state.token, state.retained,
                                         std::move(mappedGroup)});
    }
  }

  return mlir::success();
}

void appendTrackedResource(
    llvm::SmallVectorImpl<TrackedResource> &resources,
    mlir::func::FuncOp function, mlir::Operation *producer, unsigned offset,
    llvm::SmallVector<mlir::Value, 4> group,
    std::optional<own::OwnershipCondition> condition = std::nullopt) {
  TrackedResource resource;
  resource.function = function;
  resource.producer = producer;
  resource.producerLabel = describeOwnershipProducer(producer);
  resource.resultOffset = offset;
  resource.group = std::move(group);
  resource.condition = condition;
  resources.push_back(std::move(resource));
}

llvm::SmallVector<TrackedResource, 16>
collectTrackedResources(mlir::ModuleOp module, mlir::func::FuncOp function,
                        llvm::ArrayRef<own::RuntimeDeallocator> deallocators) {
  llvm::SmallVector<TrackedResource, 16> resources;

  function.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(own::kOwnedLocalObjectAttr))
      return;
    for (own::ResourceGroup group :
         own::collectOwnedLocalObjectGroups(op, deallocators)) {
      appendTrackedResource(resources, function, op, group.offset,
                            std::move(group.values), group.condition);
      return;
    }
    if (!op->hasAttr(own::kOwnedLocalObjectContractAttr) &&
        !mlir::isa<mlir::func::CallOp>(op) && op->getNumResults() != 0 &&
        own::isObjectHeaderLikeType(op->getResult(0).getType())) {
      llvm::SmallVector<mlir::Value, 4> group;
      group.push_back(op->getResult(0));
      appendTrackedResource(resources, function, op, /*offset=*/0,
                            std::move(group));
    }
  });

  function.walk([&](mlir::func::CallOp call) {
    for (own::ResourceGroup group :
         own::collectOwnedCallResultGroups(module, call, deallocators)) {
      appendTrackedResource(resources, function, call.getOperation(),
                            group.offset, std::move(group.values),
                            group.condition);
    }
  });

  return resources;
}

mlir::LogicalResult verifyFunctionAffineOwnership(
    mlir::ModuleOp module, mlir::func::FuncOp function,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  llvm::SmallVector<TrackedResource, 16> resources =
      collectTrackedResources(module, function, deallocators);
  for (TrackedResource &resource : resources)
    if (mlir::failed(
            verifyResourceOnCFGPaths(module, resource, deallocators, aliases)))
      return mlir::failure();

  llvm::SmallVector<BorrowedEntryResource, 8> borrowedEntryResources =
      collectBorrowedEntryResources(function, deallocators);
  for (BorrowedEntryResource &resource : borrowedEntryResources)
    if (mlir::failed(verifyBorrowedEntryOnCFGPaths(module, resource,
                                                   deallocators, aliases)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult verifyPathSensitiveAffineOwnership(
    mlir::ModuleOp module, llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  return walkVerify<mlir::func::FuncOp>(
      module, [&](mlir::func::FuncOp function) {
        return verifyFunctionAffineOwnership(module, function, deallocators,
                                             aliases);
      });
}

mlir::LogicalResult verifyFuncCallOwnershipContractsImpl(mlir::ModuleOp module) {
  own::AliasAnalysis aliases;
  aliases.build(module);
  llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators =
      own::collectRuntimeDeallocators(module);
  if (deallocators.empty())
    return mlir::success();
  return verifyPathSensitiveAffineOwnership(module, deallocators, aliases);
}

} // namespace

mlir::LogicalResult verifyFuncCallOwnershipContracts(mlir::ModuleOp module) {
  return verifyFuncCallOwnershipContractsImpl(module);
}

} // namespace py::lowering
