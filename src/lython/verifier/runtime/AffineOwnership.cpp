#include "runtime/Detail.h"

#include "Common/Instrumentation.h"

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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
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

using own::CachedFuncContract;
using own::FuncContractCache;
using own::ancestorInBlock;
using own::callConsumesGroup;
using own::groupContainsOperand;
using own::remapGroupThroughValueMapping;
using own::returnTransfersGroup;
using own::sameValueGroup;


bool returnCarriesGroupInsideOwnedAggregate(
    mlir::func::FuncOp function, mlir::func::ReturnOp ret,
    llvm::ArrayRef<mlir::Value> group,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;

  auto contract = own::readFunctionContract(function);
  if (mlir::succeeded(contract)) {
    for (auto [contractIndex, offset] :
         llvm::enumerate(contract->ownedResults.values)) {
      llvm::StringRef contractName;
      if (contractIndex < contract->ownedResultContracts.size())
        contractName = contract->ownedResultContracts[contractIndex];
      const own::RuntimeDeallocator *deallocator =
          contractName.empty()
              ? own::findDeallocatorForValueGroup(ret.getOperands(), offset,
                                                  deallocators)
              : own::findDeallocatorForValueGroup(ret.getOperands(), offset,
                                                  deallocators, contractName);
      if (!deallocator || group.size() >= deallocator->shapeTypes.size())
        continue;
      unsigned end = offset +
                     static_cast<unsigned>(deallocator->shapeTypes.size()) -
                     static_cast<unsigned>(group.size());
      for (unsigned candidate = offset; candidate <= end; ++candidate)
        if (own::groupMatchesValues(ret.getOperands(), candidate, group, aliases))
          return true;
    }
  }

  if (!own::functionUsesOwnedReturnABI(function))
    return false;

  std::optional<llvm::SmallVector<own::OwnedReturnRange, 4>> ranges =
      own::callableOwnedReturnRanges(function, ret.getOperands(), deallocators);
  if (!ranges)
    return false;

  for (const own::OwnedReturnRange &range : *ranges) {
    if (group.size() >= range.size)
      continue;
    unsigned end =
        range.offset + range.size - static_cast<unsigned>(group.size());
    for (unsigned offset = range.offset; offset <= end; ++offset)
      if (own::groupMatchesValues(ret.getOperands(), offset, group, aliases))
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
  // Interior views of the same entity (canonical-shape tail beyond the
  // release interface): their uses are entity uses, never release operands.
  llvm::SmallVector<mlir::Value, 4> views;
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

// Handler entries by try id: exceptional successors are resolved PER MARKER
// (this marker's id -> its handler), not per block. A block may carry
// markers of several ids (nested tries, per-call-site cleanup handlers), and
// a block-level edge set would pair one marker's token state with another
// marker's handler -- a path that cannot happen at runtime and mis-verifies.
using ExceptionHandlerMap = llvm::DenseMap<std::int64_t, mlir::Block *>;

mlir::Block *markerHandlerEntry(const ExceptionHandlerMap &handlers,
                                mlir::func::CallOp marker) {
  // Only markers in the function's top-level region model an exceptional
  // edge here. Markers nested in single-block regions (scf.if arms etc.)
  // cannot host the anchor/cond_br cleanup wiring the refcount-insertion
  // pass emits, so modeling their edges without matching cleanup would
  // reject every nested slow path; they stay outside the model, as they
  // were for the block-level edge map (known residual, not a new one).
  mlir::Region *region = marker->getParentRegion();
  if (!region || !region->getParentOp() ||
      !mlir::isa<mlir::func::FuncOp>(region->getParentOp()))
    return nullptr;
  std::optional<std::int64_t> id = own::exceptionMarkerId(marker);
  if (!id)
    return nullptr;
  return handlers.lookup(*id);
}

struct AffinePathState {
  mlir::Block *block = nullptr;
  mlir::Operation *start = nullptr;
  AffineTokenState token = AffineTokenState::Owned;
  unsigned retained = 0;
  llvm::SmallVector<mlir::Value, 4> group;
  // Values whose token was moved by a transferring call on this path: the
  // stale names carry no token, so releasing them again is a double free.
  // Entries naming a block argument are dropped when the path re-enters that
  // block (a new iteration redefines the argument).
  llvm::SmallVector<mlir::Value, 4> stale;
  // The group's names from before CFG renames on this path (block-argument
  // forwards; multiple generations — nested merges rename several times per
  // iteration). Releases through pre-rename names pair with outstanding
  // borrow-edge retains (`borrowed`). Entries naming a block argument drop
  // when the path re-enters that block.
  llvm::SmallVector<mlir::Value, 4> previous;
  // Outstanding block-arg-merge-borrow retains (identity merge edges lend the
  // merge argument a token; the paired release targets the pre-merge name).
  unsigned borrowed = 0;
  // Path entered through an exceptional (unwind) edge. The affine invariant
  // holds on these paths like any other (rfc/stdlib-semantics.md R2: unwind
  // cleanup is inserted, no leak is accepted); the flag only disambiguates
  // path-state dedup.
  bool exceptional = false;
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
  // Path entered through an exceptional (unwind) edge. Retain balance is
  // required on these paths like any other (rfc/stdlib-semantics.md R2).
  bool exceptional = false;
};

bool samePathState(const AffinePathState &lhs, const AffinePathState &rhs) {
  // `stale` and `previous` are deliberately NOT compared: they only refine
  // detections, and including them lets path-dependent sets defeat the
  // visited dedup (nested loops explode). Skipping a state that differs only
  // there can only miss a detection, never accept unsound IR.
  return lhs.block == rhs.block && lhs.start == rhs.start &&
         lhs.token == rhs.token && lhs.retained == rhs.retained &&
         lhs.borrowed == rhs.borrowed &&
         lhs.exceptional == rhs.exceptional &&
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
         lhs.retained == rhs.retained &&
         lhs.exceptional == rhs.exceptional &&
         sameValueGroup(lhs.group, rhs.group);
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

std::optional<llvm::SmallVector<mlir::Value, 4>>
callTransfersGroupToOwnedResult(FuncContractCache &contracts,
                                mlir::func::CallOp call,
                                llvm::ArrayRef<mlir::Value> group,
                                own::AliasAnalysis &aliases) {
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return std::nullopt;
  const own::FunctionContract &contract = (*cached)->contract;

  bool transfers = false;
  for (unsigned offset : contract.transferArgs.values) {
    if (own::groupMatchesValues(call.getOperands(), offset, group, aliases)) {
      transfers = true;
      break;
    }
  }
  if (!transfers)
    return std::nullopt;

  for (unsigned offset : contract.ownedResults.values) {
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

using own::callPartiallyConsumesGroup;
using own::callRetainsGroup;
using own::isBlockArgMergeBorrowRetain;

bool returnConsumesGroup(FuncContractCache &contracts,
                         mlir::func::FuncOp function, mlir::func::ReturnOp ret,
                         llvm::ArrayRef<mlir::Value> group,
                         llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                         own::AliasAnalysis &aliases) {
  return returnTransfersGroup(contracts, function, ret, group, deallocators,
                              aliases);
}

// A release/transfer operand naming a value whose token was already moved by
// an earlier transferring call on this path is a double free: the stale name
// no longer carries a token. Stale values that alias the CURRENT group are
// skipped (the live token legitimately covers them).
mlir::Value callConsumesStaleValue(FuncContractCache &contracts,
                                   mlir::func::CallOp call,
                                   llvm::ArrayRef<mlir::Value> stale,
                                   llvm::ArrayRef<mlir::Value> group,
                                   own::AliasAnalysis &aliases) {
  if (stale.empty())
    return {};
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return {};
  const own::FunctionContract &contract = (*cached)->contract;
  auto checkOffsets = [&](llvm::ArrayRef<unsigned> offsets) -> mlir::Value {
    for (unsigned offset : offsets) {
      if (offset >= call.getNumOperands())
        continue;
      mlir::Value operand = call.getOperand(offset);
      for (mlir::Value value : stale) {
        if (!aliases.same(operand, value))
          continue;
        bool aliasesCurrent = llvm::any_of(group, [&](mlir::Value live) {
          return aliases.same(value, live);
        });
        if (!aliasesCurrent)
          return value;
      }
    }
    return {};
  };
  if (mlir::Value hit = checkOffsets(contract.releaseArgs.values))
    return hit;
  return checkOffsets(contract.transferArgs.values);
}

llvm::SmallVector<mlir::Type, 8>
callableLogicalInputTypes(mlir::func::FuncOp function);

bool callCarriesGroupInsideUnionArgument(
    FuncContractCache &contracts, mlir::func::CallOp call,
    llvm::ArrayRef<mlir::Value> group,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  if (group.empty())
    return false;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;

  llvm::SmallVector<mlir::Type, 8> logicalTypes =
      callableLogicalInputTypes((*cached)->function);
  unsigned offset = 0;
  for (mlir::Type logicalType : logicalTypes) {
    std::optional<unsigned> size =
        logicalReturnValueCount(call.getOperands(), offset, deallocators,
                                logicalType);
    if (!size)
      return false;
    if (auto unionType = mlir::dyn_cast<py::UnionType>(logicalType)) {
      unsigned memberOffset = offset + 1;
      for (mlir::Type member : unionType.getMemberTypes()) {
        std::optional<unsigned> memberSize = logicalReturnValueCount(
            call.getOperands(), memberOffset, deallocators, member);
        if (!memberSize)
          return false;
        if (*memberSize > 0 &&
            own::groupMatchesValues(call.getOperands(), memberOffset, group,
                                 aliases))
          return true;
        memberOffset += *memberSize;
      }
    }
    offset += *size;
    offset = own::skipPrimitiveReturnEvidence(call.getOperands(), offset,
                                              logicalType);
  }
  return false;
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

bool isSingleBlockStraightLineFunction(mlir::func::FuncOp function) {
  if (!function || function.isDeclaration() ||
      !llvm::hasSingleElement(function.getBlocks()))
    return false;
  for (mlir::Operation &op : function.front()) {
    if (op.getNumRegions() != 0 || op.getNumSuccessors() != 0)
      return false;
    if (op.hasTrait<mlir::OpTrait::IsTerminator>() &&
        !mlir::isa<mlir::func::ReturnOp>(op))
      return false;
  }
  return true;
}

mlir::func::ReturnOp straightLineReturnOp(mlir::func::FuncOp function) {
  if (!function || function.empty())
    return {};
  return mlir::dyn_cast<mlir::func::ReturnOp>(function.front().getTerminator());
}

std::optional<mlir::LogicalResult>
verifyStraightLineResource(FuncContractCache &contracts,
                           TrackedResource &resource,
                           llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                           own::AliasAnalysis &aliases) {
  if (resource.condition || !resource.producer)
    return std::nullopt;
  mlir::Block *block = resource.producer->getBlock();
  if (!block || block->getParentOp() != resource.function)
    return std::nullopt;

  llvm::SmallVector<mlir::Operation *, 16> users;
  llvm::SmallPtrSet<mlir::Operation *, 16> seen;
  llvm::SmallVector<mlir::Value, 8> trackedValues(resource.group.begin(),
                                                  resource.group.end());
  trackedValues.append(resource.views.begin(), resource.views.end());
  for (mlir::Value value : trackedValues) {
    llvm::SmallVector<mlir::Value, 8> equivalentValues;
    aliases.aliasesOf(value, equivalentValues);
    if (equivalentValues.empty())
      equivalentValues.push_back(value);
    for (mlir::Value equivalent : equivalentValues) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user =
            ancestorInBlock(use.getOwner(), resource.producer->getBlock());
        if (!user)
          return std::nullopt;
        if (user == resource.producer)
          continue;
        if (!resource.producer->isBeforeInBlock(user))
          continue;
        if (seen.insert(user).second)
          users.push_back(user);
      }
    }
  }
  llvm::sort(users, [](mlir::Operation *lhs, mlir::Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  AffineTokenState token = AffineTokenState::Owned;
  unsigned retained = 0;
  llvm::SmallVector<mlir::Value, 4> group = resource.group;
  for (mlir::Operation *op : users) {
    if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
      bool consumes = returnConsumesGroup(contracts, resource.function, ret, group,
                                          deallocators, aliases);
      bool uses = groupContainsOperand(op, group, aliases) ||
                  groupContainsOperand(op, resource.views, aliases);
      if (token == AffineTokenState::Owned) {
        if (!consumes)
          return ret.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset
                 << " reaches function exit without release, transfer, or "
                    "owned return";
        if (retained != 0)
          return ret.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset << " is returned with "
                 << retained << " additional retained ownership token(s)";
        return mlir::success();
      }
      if (uses) {
        if (retained > 0 &&
            returnCarriesGroupInsideOwnedAggregate(
                resource.function, ret, group, deallocators, aliases))
          return mlir::success();
        return ret.emitError()
               << "released owned resource from " << resource.producerLabel
               << " is used by function return";
      }
      return mlir::success();
    }

    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
      bool consumes = callConsumesGroup(contracts, call, group, aliases);
      bool retains = callRetainsGroup(contracts, call, group, aliases);
      if (callPartiallyConsumesGroup(contracts, call, group, aliases))
        return call.emitError()
               << "ownership-consuming call only consumes part of owned "
                  "resource group produced by "
               << resource.producerLabel << " result " << resource.resultOffset;

      if (token == AffineTokenState::Released) {
        if (consumes) {
          if (retained == 0)
            return call.emitError()
                   << "owned resource from " << resource.producerLabel
                   << " result " << resource.resultOffset
                   << " is released or transferred more than once on one CFG "
                      "path";
          --retained;
          continue;
        }
        if ((groupContainsOperand(op, group, aliases) ||
             groupContainsOperand(op, resource.views, aliases)) &&
            retained == 0)
          return call.emitError()
                 << "released owned resource from " << resource.producerLabel
                 << " is used after release (by call to '" << call.getCallee()
                 << "')";
        if (retains)
          ++retained;
        continue;
      }

      if (consumes) {
        if (std::optional<llvm::SmallVector<mlir::Value, 4>> replacement =
                callTransfersGroupToOwnedResult(contracts, call, group,
                                                aliases)) {
          // Transferring to a fresh result changes the tracked use set; the
          // general CFG verifier already handles that case.
          return std::nullopt;
        }
        token = AffineTokenState::Released;
      }
      if (token == AffineTokenState::Owned && retains)
        ++retained;
      continue;
    }

    if (token == AffineTokenState::Released &&
        (groupContainsOperand(op, group, aliases) ||
         groupContainsOperand(op, resource.views, aliases)) &&
        retained == 0)
      return op->emitError()
             << "released owned resource from " << resource.producerLabel
             << " is used after release (by '" << op->getName() << "')";
  }

  if (token == AffineTokenState::Released)
    return mlir::success();

  mlir::func::ReturnOp ret = straightLineReturnOp(resource.function);
  mlir::Operation *errorSite = ret ? ret.getOperation() : resource.producer;
  return errorSite->emitError()
         << "owned resource from " << resource.producerLabel << " result "
         << resource.resultOffset
         << " reaches function exit without release, transfer, or owned return";
}

llvm::SmallVector<mlir::Type, 8>
callableLogicalInputTypes(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Type, 8> types;
  auto callableAttr =
      function->getAttrOfType<mlir::TypeAttr>(own::kCallableTypeAttr);
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
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
        group = own::valueSlice(
            entry.getArguments(), offset,
            static_cast<unsigned>(deallocator->inputTypes.size()));
        offset += static_cast<unsigned>(deallocator->shapeTypes.size());
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
      mlir::OperandRange sources = branch.getEntrySuccessorOperands(successor);
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
    mlir::OperandRange sources = branch.getEntrySuccessorOperands(successor);
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
      mlir::OperandRange sources = branch.getEntrySuccessorOperands(successor);
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
    mlir::OperandRange sources = branch.getEntrySuccessorOperands(successor);
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
    // A group defined outside this region keeps its identity across the
    // region exit: the yield forwards a view of the token, and releases
    // still target the outer values (mirrors the Released-state handling).
    if (!localGroup) {
      enqueueRegionSuccessor(owner, successor, state, worklist);
      enqueued = true;
      continue;
    }

    llvm::SmallVector<bool, 4> mappedMask;
    llvm::SmallVector<mlir::Value, 4> mappedGroup =
        remapGroupThroughValueMapping(terminator->getOperands(),
                                      successor.getSuccessorInputs(),
                                      state.group, aliases, &mappedMask);
    bool fullyMapped =
        llvm::all_of(mappedMask, [](bool mapped) { return mapped; });

    if (!fullyMapped)
      continue;

    AffinePathState next = state;
    next.group = std::move(mappedGroup);
    enqueueRegionSuccessor(owner, successor, std::move(next), worklist);
    enqueued = true;
  }

  if (!enqueued) {
    if (localGroup)
      return terminator->emitError()
             << "owned resource from " << resource.producerLabel << " result "
             << resource.resultOffset
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

  // A non-local group keeps its identity across the region exit (the yield
  // only forwards a view of the token); releases target the outer values.
  if (!localGroup) {
    AffinePathState next = state;
    next.block = owner->getBlock();
    next.start = owner->getNextNode();
    worklist.push_back(std::move(next));
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
           << "owned resource from " << resource.producerLabel << " result "
           << resource.resultOffset
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
    enqueueBorrowedRegionSuccessor(owner, successor, std::move(next), worklist);
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
    FuncContractCache &contracts, BorrowedEntryResource &resource,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases, const ExceptionHandlerMap &handlerEntries) {
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
        bool consumes = returnConsumesGroup(contracts, resource.function, ret, state.group,
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
        if (call.getCallee() == "LyEH_TryCallSiteMarker") {
          if (mlir::Block *handler = markerHandlerEntry(handlerEntries, call)) {
            BorrowedPathState next = state;
            // The unwind transfer happens DURING the guarded call: its
            // consume effect applies on the exceptional edge.
            if (mlir::func::CallOp guarded = own::guardedCallAfterMarker(op))
              if (callConsumesGroup(contracts, guarded, next.group, aliases) &&
                  next.retained > 0)
                --next.retained;
            next.block = handler;
            next.start = firstOperation(handler);
            next.exceptional = true;
            worklist.push_back(std::move(next));
          }
          op = op->getNextNode();
          continue;
        }
        if (callPartiallyConsumesGroup(contracts, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of borrowed "
                    "entry argument "
                 << resource.logicalIndex << " of @"
                 << resource.function.getSymName();

        if (callConsumesGroup(contracts, call, state.group, aliases)) {
          if (state.retained == 0)
            return call.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName()
                   << " is released or transferred without a prior retain";
          --state.retained;
          if (std::optional<llvm::SmallVector<mlir::Value, 4>> replacement =
                  callTransfersGroupToOwnedResult(contracts, call, state.group,
                                                  aliases))
            state.group = std::move(*replacement);
        }

        if (callRetainsGroup(contracts, call, state.group, aliases)) {
          // Slot-absorption retains (field stores etc.) park the token in the
          // holder and are invisible to this walk — EXCEPT merge-borrow
          // retains: an identity merge edge lends the merge argument a token
          // that a loop-edge decref later returns through the pre-merge name
          // (e.g. `local = borrowed_arg` then `local = local - 1` in a loop),
          // so it must count toward the retained balance.
          if (call->hasAttr(own::kAggregateRetainAttr) &&
              !isBlockArgMergeBorrowRetain(call)) {
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

        auto raiseCandidate = contracts.lookup(call.getCallee());
        if (mlir::succeeded(raiseCandidate) && *raiseCandidate &&
            own::isRaisePrimitiveFunction((*raiseCandidate)->function)) {
          // A raise primitive never returns; the syntactic continuation is
          // dead code, so walking it would verify a path that cannot run.
          op = nullptr;
          break;
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
               << "borrowed entry argument " << resource.logicalIndex << " of @"
               << resource.function.getSymName() << " reaches a CFG exit with "
               << state.retained << " retained ownership token(s)";
      continue;
    }

    // Mirror the runtime unwind state on the anchor's virtual true edge
    // (see the affine walk above).
    mlir::func::CallOp anchorGuarded = own::anchorTrueEdgeGuardedCall(op);
    bool anchorEdgeConsumes =
        anchorGuarded &&
        callConsumesGroup(contracts, anchorGuarded, state.group, aliases);
    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      BorrowedPathState next = state;
      if (anchorEdgeConsumes && index == 0 && next.retained > 0)
        --next.retained;
      next.block = successor;
      next.start = firstOperation(successor);
      llvm::SmallVector<bool, 4> mappedMask;
      next.group = remapGroupForSuccessor(op, index, successor, state.group,
                                          aliases, &mappedMask);
      // A group name that is a block argument of the successor but was NOT
      // forwarded on this edge is REDEFINED by the edge (a loop back edge
      // rebinding the merge argument to a fresh value): the borrowed token's
      // names die here, so tracking ends on this path. Without this, the
      // stale name matches the next iteration's merge-lane release and
      // reports a false double-consume.
      bool groupRedefined = false;
      for (auto [groupIndex, value] : llvm::enumerate(next.group)) {
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value);
        if (blockArg && blockArg.getOwner() == successor &&
            !mappedMask[groupIndex]) {
          groupRedefined = true;
          break;
        }
      }
      if (groupRedefined)
        continue;
      worklist.push_back(std::move(next));
    }
  }

  return mlir::success();
}

mlir::LogicalResult
verifyResourceOnCFGPaths(FuncContractCache &contracts,
                         TrackedResource &resource,
                         llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                         own::AliasAnalysis &aliases,
                         const ExceptionHandlerMap &handlerEntries) {
  llvm::SmallVector<AffinePathState, 16> worklist;
  llvm::SmallVector<AffinePathState, 32> visited;
  AffineTokenState initialToken = resource.condition
                                      ? AffineTokenState::Conditional
                                      : AffineTokenState::Owned;
  AffinePathState initial;
  initial.block = resource.producer->getBlock();
  initial.start = resource.producer->getNextNode();
  initial.token = initialToken;
  initial.retained = 0;
  initial.group = resource.group;
  worklist.push_back(std::move(initial));

  constexpr unsigned kMaxAffineStates = 20000;
  while (!worklist.empty()) {
    AffinePathState state = worklist.pop_back_val();
    if (containsPathState(visited, state))
      continue;
    visited.push_back(state);
    if (visited.size() > kMaxAffineStates)
      return resource.producer->emitError()
             << "ownership CFG exploration exceeded " << kMaxAffineStates
             << " states (last: retained=" << state.retained
             << " borrowed=" << state.borrowed << " prev=" << state.previous.size()
             << " stale=" << state.stale.size() << " group=" << state.group.size()
             << " token=" << static_cast<int>(state.token) << ")";

    if (pathReenteredBeforeTrackedDefinition(state)) {
      if (state.token == AffineTokenState::Released)
        continue;
      if (state.token == AffineTokenState::Owned) {
        return state.start->emitError()
               << "owned resource from " << resource.producerLabel << " result "
               << resource.resultOffset
               << " reaches the next loop iteration without release, "
                  "transfer, or owned return";
      }
      return state.start->emitError()
             << "conditionally owned resource from " << resource.producerLabel
             << " result " << resource.resultOffset
             << " reaches the next loop iteration without tag-conditioned "
                "release, transfer, or owned return";
    }

    mlir::Operation *op = state.start;
    while (op) {
      if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
        bool consumes = returnConsumesGroup(contracts, resource.function, ret, state.group,
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
                   << " result " << resource.resultOffset
                   << " is returned with " << state.retained
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
          return ret.emitError()
                 << "released owned resource from " << resource.producerLabel
                 << " is used by function return";
        }
        break;
      }

      if (state.token == AffineTokenState::Conditional) {
        if (resource.condition) {
          if (std::optional<own::OwnershipConditionBranch> branch =
                  own::classifyOwnershipConditionBranch(op,
                                                        *resource.condition)) {
            for (auto [successorIndex, nextToken] :
                 {std::pair<unsigned, AffineTokenState>{
                      branch->activeSuccessor, AffineTokenState::Owned},
                  std::pair<unsigned, AffineTokenState>{
                      branch->inactiveSuccessor, AffineTokenState::Released}}) {
              mlir::Block *successor = op->getSuccessor(successorIndex);
              llvm::SmallVector<mlir::Value, 4> mappedGroup =
                  remapGroupForSuccessor(op, successorIndex, successor,
                                         state.group, aliases);
              worklist.push_back(AffinePathState{
                  successor, firstOperation(successor), nextToken,
                  state.retained, std::move(mappedGroup),
                  /*stale=*/{}, /*previous=*/{}, /*borrowed=*/0,
                  state.exceptional});
            }
            op = nullptr;
            break;
          }
        }
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
          if (callConsumesGroup(contracts, call, state.group, aliases)) {
            state.token = AffineTokenState::Released;
            op = op->getNextNode();
            continue;
          }
          if (callCarriesGroupInsideUnionArgument(
                  contracts, call, state.group, deallocators, aliases)) {
            op = op->getNextNode();
            continue;
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
        if (call.getCallee() == "LyEH_TryCallSiteMarker") {
          // Exceptional edge: the marked call site may unwind to ITS
          // handler entry with the token state as of this point, except
          // that the unwind happens DURING the guarded call -- consume
          // effects of the guarded call apply on the edge, and its results
          // never materialize (a transfer is a plain release here).
          if (mlir::Block *handler = markerHandlerEntry(handlerEntries, call)) {
            AffinePathState next = state;
            if (mlir::func::CallOp guarded = own::guardedCallAfterMarker(op)) {
              if (callConsumesGroup(contracts, guarded, next.group, aliases)) {
                if (next.token == AffineTokenState::Owned ||
                    next.token == AffineTokenState::Conditional)
                  next.token = AffineTokenState::Released;
                else if (next.retained > 0)
                  --next.retained;
              }
            }
            next.block = handler;
            next.start = firstOperation(handler);
            next.exceptional = true;
            worklist.push_back(std::move(next));
          }
          op = op->getNextNode();
          continue;
        }
        if (callConsumesStaleValue(contracts, call, state.stale, state.group,
                                   aliases))
          return call.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset
                 << " is released through a value already consumed by an "
                    "ownership transfer";
        bool consumes =
            callConsumesGroup(contracts, call, state.group, aliases);
        // A release through PRE-RENAME names of the current group cancels an
        // outstanding borrow-edge retain (identity merge edge): the token
        // continues under the current names.
        if (!consumes && state.borrowed > 0 &&
            callConsumesStaleValue(contracts, call, state.previous,
                                   state.group, aliases)) {
          --state.borrowed;
          op = op->getNextNode();
          continue;
        }
        bool retains = callRetainsGroup(contracts, call, state.group, aliases);
        if (callPartiallyConsumesGroup(contracts, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of owned "
                    "resource group produced by "
                 << resource.producerLabel << " result "
                 << resource.resultOffset;

        if (state.token == AffineTokenState::Released) {
          if (consumes) {
            if (state.retained == 0 && resource.condition) {
              op = op->getNextNode();
              continue;
            }
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
          if ((groupContainsOperand(op, state.group, aliases) ||
               groupContainsOperand(op, resource.views, aliases)) &&
              state.retained == 0)
            return call.emitError()
                   << "released owned resource from " << resource.producerLabel
                   << " is used after release (by call to '"
                   << call.getCallee() << "')";
          if (retains)
            ++state.retained;
        } else if (state.token == AffineTokenState::Owned && consumes) {
          if (std::optional<llvm::SmallVector<mlir::Value, 4>> replacement =
                  callTransfersGroupToOwnedResult(contracts, call, state.group,
                                                  aliases)) {
            state.stale.append(state.group.begin(), state.group.end());
            state.group = std::move(*replacement);
          } else {
            state.token = AffineTokenState::Released;
          }
        }
        if (state.token == AffineTokenState::Owned && retains) {
          if (isBlockArgMergeBorrowRetain(call))
            ++state.borrowed;
          else
            ++state.retained;
        }
        auto raiseCandidate = contracts.lookup(call.getCallee());
        if (mlir::succeeded(raiseCandidate) && *raiseCandidate &&
            own::isRaisePrimitiveFunction((*raiseCandidate)->function)) {
          // An unguarded raise primitive (no preceding call-site marker
          // wiring it to an in-function handler) unwinds OUT of the
          // function: a token still owned here escapes with no path left to
          // release it. A guarded raise reaches its handler through the
          // marker edge enqueued above.
          if (state.token == AffineTokenState::Owned &&
              !own::precedingTryCallSiteMarker(call))
            return call.emitError()
                   << "owned resource from " << resource.producerLabel
                   << " result " << resource.resultOffset
                   << " is still owned when '" << call.getCallee()
                   << "' unwinds out of the function; the exception path "
                      "must release, transfer, or return it";
          // A raise primitive never returns; the syntactic continuation is
          // dead code, so walking it would verify a path that cannot run.
          op = nullptr;
          break;
        }
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
               << "owned resource from " << resource.producerLabel << " result "
               << resource.resultOffset
               << " reaches a CFG exit without release, transfer, or owned "
                  "return";
      if (state.token == AffineTokenState::Conditional)
        return op->emitError()
               << "conditionally owned resource from " << resource.producerLabel
               << " result " << resource.resultOffset
               << " reaches a CFG exit without tag-conditioned release, "
                  "transfer, or owned return";
      continue;
    }

    // An anchor cond_br's true edge is the virtual spelling of "the guarded
    // call unwound": apply the guarded call's consume effect there so the
    // virtual path carries the same token state the runtime unwind does.
    mlir::func::CallOp anchorGuarded = own::anchorTrueEdgeGuardedCall(op);
    bool anchorEdgeConsumes =
        anchorGuarded &&
        callConsumesGroup(contracts, anchorGuarded, state.group, aliases);
    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      AffineTokenState nextToken = state.token;
      unsigned nextRetained = state.retained;
      if (anchorEdgeConsumes && index == 0) {
        if (nextToken == AffineTokenState::Owned ||
            nextToken == AffineTokenState::Conditional)
          nextToken = AffineTokenState::Released;
        else if (nextRetained > 0)
          --nextRetained;
      }
      llvm::SmallVector<bool, 4> mappedMask;
      llvm::SmallVector<mlir::Value, 4> mappedGroup = remapGroupForSuccessor(
          op, index, successor, state.group, aliases, &mappedMask);
      bool fullyMapped =
          llvm::all_of(mappedMask, [](bool mapped) { return mapped; });
      if (!fullyMapped && nextToken == AffineTokenState::Released &&
          groupContainsArgumentFromBlock(state.group, successor))
        continue;
      // Entering the successor redefines its block arguments: stale/previous
      // entries naming them refer to the PREVIOUS iteration's token and drop.
      llvm::SmallVector<mlir::Value, 4> mappedStale;
      for (mlir::Value value : state.stale) {
        auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(value);
        if (!argument || argument.getOwner() != successor)
          mappedStale.push_back(value);
      }
      bool renamed = llvm::any_of(mappedMask, [](bool m) { return m; });
      llvm::SmallVector<mlir::Value, 4> mappedPrevious;
      auto keepPreviousName = [&](mlir::Value value) {
        auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(value);
        if (argument && argument.getOwner() == successor)
          return; // re-entering the block redefines this name
        if (!llvm::is_contained(mappedPrevious, value))
          mappedPrevious.push_back(value);
      };
      for (mlir::Value value : state.previous)
        keepPreviousName(value);
      if (renamed)
        for (mlir::Value value : state.group)
          keepPreviousName(value);
      worklist.push_back(AffinePathState{successor, firstOperation(successor),
                                         nextToken, nextRetained,
                                         std::move(mappedGroup),
                                         std::move(mappedStale),
                                         std::move(mappedPrevious),
                                         state.borrowed, state.exceptional});
    }
  }

  return mlir::success();
}

void appendTrackedResource(
    llvm::SmallVectorImpl<TrackedResource> &resources,
    mlir::func::FuncOp function, mlir::Operation *producer, unsigned offset,
    llvm::SmallVector<mlir::Value, 4> group,
    std::optional<own::OwnershipCondition> condition = std::nullopt,
    llvm::SmallVector<mlir::Value, 4> views = {}) {
  TrackedResource resource;
  resource.function = function;
  resource.producer = producer;
  resource.producerLabel = describeOwnershipProducer(producer);
  resource.resultOffset = offset;
  resource.group = std::move(group);
  resource.views = std::move(views);
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
                            std::move(group.values), group.condition,
                            std::move(group.views));
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
                            group.condition, std::move(group.views));
    }
  });

  return resources;
}

mlir::LogicalResult verifyFunctionAffineOwnership(
    mlir::ModuleOp module, mlir::func::FuncOp function,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases, FuncContractCache &contracts) {
  llvm::SmallVector<TrackedResource, 16> resources =
      collectTrackedResources(module, function, deallocators);
  llvm::SmallVector<BorrowedEntryResource, 8> borrowedEntryResources =
      collectBorrowedEntryResources(function, deallocators);

  if (isSingleBlockStraightLineFunction(function) &&
      borrowedEntryResources.empty()) {
    bool allResourcesHandled = true;
    for (TrackedResource &resource : resources) {
      std::optional<mlir::LogicalResult> result = verifyStraightLineResource(
          contracts, resource, deallocators, aliases);
      if (!result) {
        allResourcesHandled = false;
        break;
      }
      if (mlir::failed(*result))
        return mlir::failure();
    }
    if (allResourcesHandled)
      return mlir::success();
  }

  ExceptionHandlerMap handlerEntries =
      function.isDeclaration()
          ? ExceptionHandlerMap()
          : own::collectExceptionHandlerEntries(function.getBody());

  for (TrackedResource &resource : resources)
    if (mlir::failed(verifyResourceOnCFGPaths(contracts, resource, deallocators,
                                              aliases, handlerEntries)))
      return mlir::failure();

  for (BorrowedEntryResource &resource : borrowedEntryResources)
    if (mlir::failed(verifyBorrowedEntryOnCFGPaths(
            contracts, resource, deallocators, aliases, handlerEntries)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult verifyPathSensitiveAffineOwnership(
    mlir::ModuleOp module, llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  FuncContractCache contracts(module);
  return walkVerify<mlir::func::FuncOp>(
      module, [&](mlir::func::FuncOp function) {
        if (own::isRuntimeManifestFunction(function))
          return mlir::success();
        return verifyFunctionAffineOwnership(module, function, deallocators,
                                             aliases, contracts);
      });
}

mlir::LogicalResult
verifyFuncCallOwnershipContractsImpl(mlir::ModuleOp module) {
  own::AliasAnalysis aliases;
  {
    py::PerfScope perf("func-call-ownership.alias-analysis");
    aliases.build(module);
  }
  llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators;
  {
    py::PerfScope perf("func-call-ownership.collect-deallocators");
    deallocators = own::collectRuntimeDeallocators(module);
  }
  if (deallocators.empty())
    return mlir::success();
  py::PerfScope perf("func-call-ownership.path-sensitive");
  return verifyPathSensitiveAffineOwnership(module, deallocators, aliases);
}

} // namespace

mlir::LogicalResult verifyFuncCallOwnershipContracts(mlir::ModuleOp module) {
  return verifyFuncCallOwnershipContractsImpl(module);
}

} // namespace py::lowering
