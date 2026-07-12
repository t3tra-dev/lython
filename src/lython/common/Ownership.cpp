#include "Ownership.h"

#include "PyDialectTypes.h"
#include "Contracts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"

#include <algorithm>
#include <cstdint>

namespace py::ownership {

namespace contracts = py::contracts;

bool IndexSet::contains(unsigned index) const {
  return llvm::is_contained(values, index);
}

bool FunctionContract::hasAnyOwnershipAttr() const {
  return !ownedResults.empty() || !borrowedResults.empty() ||
         !retainArgs.empty() || !releaseArgs.empty() || !transferArgs.empty() ||
         objectReleaseToZero;
}

bool FunctionContract::consumesArg(unsigned index) const {
  return releaseArgs.contains(index) || transferArgs.contains(index);
}

static mlir::LogicalResult appendIndex(mlir::Operation *op,
                                       llvm::StringRef attrName,
                                       std::optional<unsigned> upperBound,
                                       int64_t raw, IndexSet &indices) {
  if (raw < 0)
    return op->emitError() << attrName << " contains negative index " << raw;
  unsigned index = static_cast<unsigned>(raw);
  if (upperBound && index >= *upperBound)
    return op->emitError() << attrName << " index " << index
                           << " is out of range [0, " << *upperBound << ")";
  if (indices.contains(index))
    return op->emitError() << attrName << " contains duplicate index " << index;
  indices.values.push_back(index);
  return mlir::success();
}

mlir::FailureOr<IndexSet>
parseIndexSetAttr(mlir::Operation *op, llvm::StringRef attrName,
                  std::optional<unsigned> upperBound) {
  IndexSet indices;
  mlir::Attribute attr = op->getAttr(attrName);
  if (!attr)
    return indices;

  if (auto dense = mlir::dyn_cast<mlir::DenseI64ArrayAttr>(attr)) {
    for (int64_t raw : dense.asArrayRef())
      if (mlir::failed(appendIndex(op, attrName, upperBound, raw, indices)))
        return mlir::failure();
    return indices;
  }

  if (auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(attr)) {
    if (!dense.getType().hasRank() || dense.getType().getRank() != 1)
      return op->emitError() << attrName << " must be a one-dimensional index "
                             << "list";
    for (mlir::APInt value : dense.getValues<mlir::APInt>())
      if (mlir::failed(appendIndex(op, attrName, upperBound,
                                   value.getSExtValue(), indices)))
        return mlir::failure();
    return indices;
  }

  if (auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr)) {
    for (mlir::Attribute element : array) {
      auto integer = mlir::dyn_cast<mlir::IntegerAttr>(element);
      if (!integer)
        return op->emitError() << attrName << " must contain integer indices";
      if (mlir::failed(
              appendIndex(op, attrName, upperBound, integer.getInt(), indices)))
        return mlir::failure();
    }
    return indices;
  }

  if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(attr)) {
    if (mlir::failed(
            appendIndex(op, attrName, upperBound, integer.getInt(), indices)))
      return mlir::failure();
    return indices;
  }

  return op->emitError() << attrName
                         << " must be an integer array ownership contract";
}

mlir::FailureOr<FunctionContract>
readFunctionContract(mlir::func::FuncOp function) {
  FunctionContract contract;
  unsigned numInputs = function.getFunctionType().getNumInputs();
  unsigned numResults = function.getFunctionType().getNumResults();

  auto owned = parseIndexSetAttr(function, kOwnedResultsAttr, numResults);
  if (mlir::failed(owned))
    return mlir::failure();
  contract.ownedResults = *owned;

  if (mlir::Attribute attr = function->getAttr(kOwnedResultContractsAttr)) {
    auto array = mlir::dyn_cast<mlir::ArrayAttr>(attr);
    if (!array)
      return function.emitError()
             << kOwnedResultContractsAttr << " must be a string array";
    if (array.size() != contract.ownedResults.values.size())
      return function.emitError()
             << kOwnedResultContractsAttr << " must have one entry per "
             << kOwnedResultsAttr << " result";
    contract.ownedResultContracts.reserve(array.size());
    for (mlir::Attribute element : array) {
      auto string = mlir::dyn_cast<mlir::StringAttr>(element);
      if (!string)
        return function.emitError()
               << kOwnedResultContractsAttr << " must contain strings";
      contract.ownedResultContracts.push_back(string.getValue().str());
    }
  }

  auto borrowed = parseIndexSetAttr(function, kBorrowedResultsAttr, numResults);
  if (mlir::failed(borrowed))
    return mlir::failure();
  contract.borrowedResults = *borrowed;

  auto retained = parseIndexSetAttr(function, kRetainArgsAttr, numInputs);
  if (mlir::failed(retained))
    return mlir::failure();
  contract.retainArgs = *retained;

  auto released = parseIndexSetAttr(function, kReleaseArgsAttr, numInputs);
  if (mlir::failed(released))
    return mlir::failure();
  contract.releaseArgs = *released;

  auto transferred = parseIndexSetAttr(function, kTransferArgsAttr, numInputs);
  if (mlir::failed(transferred))
    return mlir::failure();
  contract.transferArgs = *transferred;

  contract.objectReleaseToZero = function->hasAttr(kObjectReleaseToZeroAttr);

  for (unsigned index : contract.releaseArgs.values) {
    if (contract.transferArgs.contains(index))
      return function.emitError()
             << "argument " << index
             << " cannot be both release_args and transfer_args";
  }

  return contract;
}

static mlir::FailureOr<std::string>
parseAggregateOwnershipSlot(mlir::Operation *op, llvm::StringRef attrName,
                            mlir::Attribute attr) {
  if (mlir::isa<mlir::UnitAttr>(attr))
    return std::string();
  if (auto string = mlir::dyn_cast<mlir::StringAttr>(attr))
    return string.getValue().str();
  return op->emitError() << attrName
                         << " must be a unit or string aggregate slot marker";
}

mlir::FailureOr<std::optional<AggregateOwnershipMarker>>
readAggregateOwnershipMarker(mlir::Operation *op) {
  mlir::Attribute retain = op->getAttr(kAggregateRetainAttr);
  mlir::Attribute release = op->getAttr(kAggregateReleaseAttr);
  if (!retain && !release)
    return std::optional<AggregateOwnershipMarker>();
  if (retain && release)
    return op->emitError() << "operation cannot declare both "
                           << kAggregateRetainAttr << " and "
                           << kAggregateReleaseAttr;

  AggregateOwnershipMarker marker;
  if (retain) {
    marker.action = AggregateOwnershipAction::Retain;
    mlir::FailureOr<std::string> slot =
        parseAggregateOwnershipSlot(op, kAggregateRetainAttr, retain);
    if (mlir::failed(slot))
      return mlir::failure();
    marker.slot = std::move(*slot);
  } else {
    marker.action = AggregateOwnershipAction::Release;
    mlir::FailureOr<std::string> slot =
        parseAggregateOwnershipSlot(op, kAggregateReleaseAttr, release);
    if (mlir::failed(slot))
      return mlir::failure();
    marker.slot = std::move(*slot);
  }
  return std::optional<AggregateOwnershipMarker>(std::move(marker));
}

bool isRuntimeManifestFunction(mlir::func::FuncOp function) {
  return function && (function->hasAttr(contracts::kManifestContractAttr) ||
                      function->hasAttr(contracts::kManifestPrimitiveAttr) ||
                      function->hasAttr(contracts::kManifestMethodAttr) ||
                      function->hasAttr(contracts::kManifestInitializerAttr) ||
                      function->hasAttr(contracts::kManifestBuiltinAttr) ||
                      function->hasAttr(contracts::kManifestShapeAttr) ||
                      function->hasAttr(contracts::kManifestDeallocatorAttr));
}

bool functionUsesOwnedReturnABI(mlir::func::FuncOp function) {
  if (!function || function.isExternal() || isRuntimeManifestFunction(function))
    return false;
  return function->hasAttr(kCallableTypeAttr) ||
         function.getSymName() == "__main__";
}

bool functionOwnsResultAt(mlir::func::FuncOp function, unsigned resultIndex) {
  auto contract = readFunctionContract(function);
  if (mlir::failed(contract))
    return false;
  return contract->ownedResults.contains(resultIndex);
}

bool functionConsumesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex) {
  auto contract = readFunctionContract(function);
  if (mlir::failed(contract))
    return false;
  return contract->consumesArg(operandIndex);
}

bool functionReleasesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex) {
  auto contract = readFunctionContract(function);
  if (mlir::failed(contract))
    return false;
  return contract->releaseArgs.contains(operandIndex);
}

bool functionRetainsOperandAt(mlir::func::FuncOp function,
                              unsigned operandIndex) {
  auto contract = readFunctionContract(function);
  if (mlir::failed(contract))
    return false;
  return contract->retainArgs.contains(operandIndex);
}

llvm::SmallVector<RuntimeDeallocator, 8>
collectRuntimeDeallocators(mlir::ModuleOp module) {
  llvm::SmallVector<RuntimeDeallocator, 8> deallocators;
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr(contracts::kManifestDeallocatorAttr))
      return;
    auto contract = readFunctionContract(function);
    if (mlir::failed(contract))
      return;
    RuntimeDeallocator deallocator;
    deallocator.function = function;
    deallocator.contract = *contract;
    if (auto contractAttr = function->getAttrOfType<mlir::StringAttr>(
            contracts::kManifestContractAttr))
      deallocator.contractName = contractAttr.getValue().str();
    deallocator.inputTypes.append(
        function.getFunctionType().getInputs().begin(),
        function.getFunctionType().getInputs().end());
    deallocators.push_back(std::move(deallocator));
  });
  // Canonical shapes: ly.runtime.shape declarations extend the release
  // interface with the entity's interior-view types.
  module.walk([&](mlir::func::FuncOp function) {
    if (!function->hasAttr(contracts::kManifestShapeAttr))
      return;
    auto contractAttr = function->getAttrOfType<mlir::StringAttr>(
        contracts::kManifestContractAttr);
    if (!contractAttr)
      return;
    for (RuntimeDeallocator &deallocator : deallocators) {
      if (deallocator.contractName != contractAttr.getValue())
        continue;
      deallocator.shapeTypes.assign(
          function.getFunctionType().getResults().begin(),
          function.getFunctionType().getResults().end());
    }
  });
  for (RuntimeDeallocator &deallocator : deallocators)
    if (deallocator.shapeTypes.empty())
      deallocator.shapeTypes = deallocator.inputTypes;
  return deallocators;
}

bool valueRangeMatchesTypes(mlir::ValueRange values, unsigned offset,
                            llvm::ArrayRef<mlir::Type> types) {
  if (offset + types.size() > values.size())
    return false;
  for (auto [index, type] : llvm::enumerate(types)) {
    if (values[offset + index].getType() != type)
      return false;
  }
  return true;
}

static bool isIntegerType(mlir::Type type, unsigned width) {
  auto integer = mlir::dyn_cast<mlir::IntegerType>(type);
  return integer && integer.getWidth() == width;
}

bool isObjectHeaderLikeType(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1)
    return false;
  if (!isIntegerType(memref.getElementType(), 64))
    return false;
  return memref.isDynamicDim(0) || memref.getDimSize(0) >= 2;
}

mlir::Value underlyingObjectValue(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
    if (cast.getInputs().size() != cast.getOutputs().size())
      break;
    unsigned index = mlir::cast<mlir::OpResult>(value).getResultNumber();
    mlir::Value input = cast.getInputs()[index];
    if (input.getType() != value.getType())
      break;
    value = input;
  }
  return value;
}

const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  // Release interfaces are entity-root prefixes, so several contracts share
  // the same inputTypes; disambiguate by the longest canonical-shape match
  // (the interior-view tail differs per contract).
  const RuntimeDeallocator *matched = nullptr;
  bool ambiguous = false;
  auto shapeMatch = [&](const RuntimeDeallocator &deallocator) -> unsigned {
    if (deallocator.shapeTypes.size() <= deallocator.inputTypes.size())
      return 0;
    return valueRangeMatchesTypes(values, offset, deallocator.shapeTypes)
               ? static_cast<unsigned>(deallocator.shapeTypes.size())
               : 0;
  };
  unsigned matchedShape = 0;
  for (const RuntimeDeallocator &deallocator : deallocators) {
    if (!valueRangeMatchesTypes(values, offset, deallocator.inputTypes))
      continue;
    unsigned shape = shapeMatch(deallocator);
    if (!matched || deallocator.inputTypes.size() > matched->inputTypes.size() ||
        (deallocator.inputTypes.size() == matched->inputTypes.size() &&
         shape > matchedShape)) {
      matched = &deallocator;
      matchedShape = shape;
      ambiguous = false;
      continue;
    }
    if (deallocator.inputTypes.size() == matched->inputTypes.size() &&
        shape == matchedShape)
      ambiguous = true;
  }
  if (ambiguous)
    return nullptr;
  return matched;
}

const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             llvm::StringRef contractName) {
  if (contractName.empty())
    return findDeallocatorForValueGroup(values, offset, deallocators);

  const RuntimeDeallocator *matched = nullptr;
  bool ambiguous = false;
  for (const RuntimeDeallocator &deallocator : deallocators) {
    if (deallocator.contractName != contractName)
      continue;
    if (!valueRangeMatchesTypes(values, offset, deallocator.inputTypes))
      continue;
    if (!matched ||
        deallocator.inputTypes.size() > matched->inputTypes.size()) {
      matched = &deallocator;
      ambiguous = false;
      continue;
    }
    if (deallocator.inputTypes.size() == matched->inputTypes.size())
      ambiguous = true;
  }
  if (matched)
    return ambiguous ? nullptr : matched;

  return findDeallocatorForValueGroup(values, offset, deallocators);
}

llvm::SmallVector<mlir::Value, 4> valueSlice(mlir::ValueRange values,
                                             unsigned offset, unsigned size) {
  llvm::SmallVector<mlir::Value, 4> slice;
  slice.reserve(size);
  for (unsigned index = 0; index < size; ++index)
    slice.push_back(values[offset + index]);
  return slice;
}

// Extend a group with the entity's interior views: the canonical-shape tail
// beyond the release interface. Their uses pin the entity's liveness; they
// are never release operands.
static void appendEntityViews(ResourceGroup &group, mlir::ValueRange values,
                              unsigned offset) {
  if (!group.deallocator)
    return;
  unsigned tokenSize =
      static_cast<unsigned>(group.deallocator->inputTypes.size());
  llvm::ArrayRef<mlir::Type> shape = group.deallocator->shapeTypes;
  if (shape.size() <= tokenSize)
    return;
  llvm::ArrayRef<mlir::Type> tail = shape.drop_front(tokenSize);
  if (!valueRangeMatchesTypes(values, offset + tokenSize, tail))
    return;
  group.views = valueSlice(values, offset + tokenSize,
                           static_cast<unsigned>(tail.size()));
}

void collectBoxWordDerivedViews(llvm::ArrayRef<mlir::Value> groupValues,
                                llvm::SmallVectorImpl<mlir::Value> &views) {
  llvm::SmallDenseSet<mlir::Value, 8> known(views.begin(), views.end());
  llvm::SmallVector<mlir::Value, 8> worklist;
  for (mlir::Value value : groupValues)
    for (mlir::Operation *user : value.getUsers())
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user))
        if (load.getMemRef() == value)
          worklist.push_back(load.getResult());
  // Follow the descriptor-assembly chain only (matched by op name so the
  // common layer stays free of an LLVM-dialect dependency); any other use of
  // a loaded word (arithmetic, comparisons) terminates the walk.
  llvm::SmallDenseSet<mlir::Value, 16> visited;
  while (!worklist.empty()) {
    mlir::Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    for (mlir::Operation *user : current.getUsers()) {
      llvm::StringRef opName = user->getName().getStringRef();
      if (opName == "llvm.inttoptr" || opName == "llvm.insertvalue") {
        if (user->getNumResults() == 1)
          worklist.push_back(user->getResult(0));
        continue;
      }
      if (auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user))
        for (mlir::Value result : cast.getResults())
          if (mlir::isa<mlir::MemRefType>(result.getType()) &&
              known.insert(result).second)
            views.push_back(result);
    }
  }
}

bool valueGroupEqualsEntryArgumentGroup(mlir::func::FuncOp function,
                                        llvm::ArrayRef<mlir::Value> group) {
  if (function.empty() || group.empty())
    return false;
  mlir::Block &entry = function.front();
  if (entry.getNumArguments() < group.size())
    return false;

  for (unsigned start = 0; start + group.size() <= entry.getNumArguments();
       ++start) {
    bool matches = true;
    for (auto [index, value] : llvm::enumerate(group)) {
      if (value != entry.getArgument(start + index)) {
        matches = false;
        break;
      }
    }
    if (matches)
      return true;
  }
  return false;
}

bool callResultGroupIsOwned(mlir::func::FuncOp callee, unsigned resultIndex) {
  return functionOwnsResultAt(callee, resultIndex) ||
         (resultIndex == 0 && functionUsesOwnedReturnABI(callee));
}

llvm::StringRef ownershipKindName(OwnershipKind kind) {
  switch (kind) {
  case OwnershipKind::NonObject:
    return "NonObject";
  case OwnershipKind::Borrow:
    return "Borrow";
  case OwnershipKind::Own:
    return "Own";
  case OwnershipKind::Immortal:
    return "Immortal";
  }
  return "Unknown";
}

bool ownershipKindCarriesObjectResource(OwnershipKind kind) {
  return kind == OwnershipKind::Borrow || kind == OwnershipKind::Own;
}

OwnershipKind logicalOwnershipKind(mlir::Type logicalType, bool ownsObject) {
  std::string contractName = contracts::runtimeContractName(logicalType);
  if (contractName.empty())
    return OwnershipKind::NonObject;
  if (contractName == "types.NoneType" || contractName == "builtins.bool")
    return OwnershipKind::Immortal;
  return ownsObject ? OwnershipKind::Own : OwnershipKind::Borrow;
}

static std::optional<OwnershipCondition>
optionalUnionPayloadCondition(mlir::func::FuncOp callee,
                              mlir::func::CallOp call, unsigned groupOffset) {
  if (!callee || groupOffset != 1 || call.getNumResults() < 2)
    return std::nullopt;

  auto callableAttr = callee->getAttrOfType<mlir::TypeAttr>(kCallableTypeAttr);
  if (!callableAttr)
    return std::nullopt;
  auto callable =
      mlir::dyn_cast_if_present<py::CallableType>(callableAttr.getValue());
  if (!callable || callable.getResultTypes().size() != 1)
    return std::nullopt;

  auto unionType =
      mlir::dyn_cast_if_present<py::UnionType>(callable.getResultTypes()[0]);
  if (!unionType)
    return std::nullopt;

  llvm::ArrayRef<mlir::Type> members = unionType.getMemberTypes();
  auto isNoneLike = [](mlir::Type type) {
    return py::isPyNoneType(type);
  };
  if (members.size() != 2 ||
      (!isNoneLike(members[0]) && !isNoneLike(members[1])))
    return std::nullopt;

  unsigned payloadIndex = isNoneLike(members[0]) ? 1 : 0;
  return OwnershipCondition{call.getResult(0),
                            static_cast<std::int64_t>(payloadIndex),
                            static_cast<unsigned>(members.size())};
}

static bool isNoneLikeType(mlir::Type type) {
  return py::isPyNoneType(type);
}

static llvm::SmallVector<ResourceGroup, 4>
collectContractOwnedResultGroups(mlir::func::FuncOp callee,
                                 mlir::func::CallOp call,
                                 llvm::ArrayRef<RuntimeDeallocator>
                                     deallocators) {
  llvm::SmallVector<ResourceGroup, 4> groups;
  auto contract = readFunctionContract(callee);
  if (mlir::failed(contract) || contract->ownedResults.empty())
    return groups;

  auto contractAttr =
      callee->getAttrOfType<mlir::StringAttr>(contracts::kManifestContractAttr);
  if (!contractAttr)
    return groups;

  for (unsigned offset : contract->ownedResults.values) {
    const RuntimeDeallocator *deallocator = findDeallocatorForValueGroup(
        call.getResults(), offset, deallocators, contractAttr.getValue());
    if (!deallocator)
      continue;
    ResourceGroup group;
    group.offset = offset;
    group.deallocator = deallocator;
    group.values =
        valueSlice(call.getResults(), offset,
                   static_cast<unsigned>(deallocator->inputTypes.size()));
    appendEntityViews(group, call.getResults(), offset);
    groups.push_back(std::move(group));
  }
  return groups;
}

static std::optional<unsigned>
deallocatorValueCountForType(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             mlir::Type type) {
  if (isNoneLikeType(type))
    return 0;
  std::string contract = contracts::runtimeContractName(type);
  const RuntimeDeallocator *deallocator =
      findDeallocatorForValueGroup(values, offset, deallocators, contract);
  if (!deallocator)
    return std::nullopt;
  // Physical value span of the contract: the canonical shape, not the
  // (possibly narrower) release interface.
  return static_cast<unsigned>(deallocator->shapeTypes.size());
}

static void
collectTypedResourceGroups(mlir::Type type, mlir::ValueRange values,
                           llvm::ArrayRef<RuntimeDeallocator> deallocators,
                           unsigned baseOffset,
                           llvm::SmallVectorImpl<ResourceGroup> &groups) {
  if (values.empty())
    return;

  std::string contract = contracts::runtimeContractName(type);
  if (const RuntimeDeallocator *deallocator =
          findDeallocatorForValueGroup(values, 0, deallocators, contract)) {
    if (deallocator->inputTypes.size() == values.size() ||
        deallocator->shapeTypes.size() == values.size()) {
      ResourceGroup group;
      group.offset = baseOffset;
      group.deallocator = deallocator;
      group.values = valueSlice(
          values, 0, static_cast<unsigned>(deallocator->inputTypes.size()));
      appendEntityViews(group, values, 0);
      groups.push_back(std::move(group));
      return;
    }
  }

  auto unionType = mlir::dyn_cast<py::UnionType>(type);
  if (!unionType || values.size() < 1)
    return;

  unsigned offset = 1;
  for (auto [memberIndex, member] :
       llvm::enumerate(unionType.getMemberTypes())) {
    std::optional<unsigned> memberSize =
        deallocatorValueCountForType(values, offset, deallocators, member);
    if (!memberSize) {
      if (isNoneLikeType(member))
        memberSize = 0;
      else
        return;
    }
    if (offset + *memberSize > values.size())
      return;
    if (*memberSize > 0) {
      llvm::SmallVector<ResourceGroup, 4> memberGroups;
      collectTypedResourceGroups(member, values.slice(offset, *memberSize),
                                 deallocators, baseOffset + offset,
                                 memberGroups);
      for (ResourceGroup &group : memberGroups) {
        group.condition = OwnershipCondition{
            values.front(), static_cast<std::int64_t>(memberIndex),
            static_cast<unsigned>(unionType.getMemberTypes().size())};
        groups.push_back(std::move(group));
      }
    }
    offset += *memberSize;
  }
}

bool groupMatchesValues(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<mlir::Value> group,
                        AliasAnalysis &aliases) {
  if (offset + group.size() > values.size())
    return false;
  for (auto [index, value] : llvm::enumerate(group)) {
    if (!aliases.same(values[offset + index], value))
      return false;
  }
  return true;
}

static std::string logicalReturnObjectContract(mlir::Type type) {
  std::string contract = contracts::runtimeContractName(type);
  if (!contract.empty())
    return contract;
  if (mlir::isa<py::ProtocolType>(type))
    return "builtins.object";
  return "";
}

std::optional<unsigned>
logicalReturnValueCount(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<RuntimeDeallocator> deallocators,
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
  const RuntimeDeallocator *deallocator =
      findDeallocatorForValueGroup(values, offset, deallocators, contract);
  if (!deallocator)
    return std::nullopt;
  // Physical span = canonical shape (the release interface may be narrower).
  return static_cast<unsigned>(deallocator->shapeTypes.size());
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

std::optional<llvm::SmallVector<OwnedReturnRange, 4>>
callableOwnedReturnRanges(mlir::func::FuncOp function, mlir::ValueRange values,
                          llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  auto callableAttr =
      function->getAttrOfType<mlir::TypeAttr>(kCallableTypeAttr);
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
    llvm::ArrayRef<RuntimeDeallocator> deallocators, AliasAnalysis &aliases) {
  if (group.empty())
    return false;

  // The group is a release interface: a non-empty PREFIX of the logical
  // value span (usually just the entity root).
  auto matchesLogicalValue = [&](auto &&self, unsigned offset,
                                 mlir::Type type) -> bool {
    if (auto unionType = mlir::dyn_cast<py::UnionType>(type)) {
      if (group.size() <= range.size &&
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
    return size && group.size() <= *size &&
           groupMatchesValues(values, offset, group, aliases);
  };

  return matchesLogicalValue(matchesLogicalValue, range.offset, range.type);
}

llvm::SmallVector<ResourceGroup, 8>
collectRuntimeResourceGroups(mlir::ValueRange values,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  llvm::SmallVector<ResourceGroup, 8> groups;
  unsigned offset = 0;
  while (offset < values.size()) {
    const RuntimeDeallocator *deallocator =
        findDeallocatorForValueGroup(values, offset, deallocators);
    if (!deallocator) {
      ++offset;
      continue;
    }
    unsigned size = static_cast<unsigned>(deallocator->inputTypes.size());
    ResourceGroup group;
    group.offset = offset;
    group.deallocator = deallocator;
    group.values = valueSlice(values, offset, size);
    appendEntityViews(group, values, offset);
    unsigned span = size + static_cast<unsigned>(group.views.size());
    groups.push_back(std::move(group));
    offset += span;
  }
  return groups;
}

llvm::SmallVector<ResourceGroup, 4>
collectOwnedLocalObjectGroups(mlir::Operation *op,
                              llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  llvm::SmallVector<ResourceGroup, 4> groups;
  if (!op || !op->hasAttr(kOwnedLocalObjectAttr))
    return groups;

  auto contractAttr =
      op->getAttrOfType<mlir::StringAttr>(kOwnedLocalObjectContractAttr);
  if (!contractAttr || op->getNumResults() == 0)
    return groups;

  const RuntimeDeallocator *deallocator = findDeallocatorForValueGroup(
      op->getResults(), 0, deallocators, contractAttr.getValue());
  if (!deallocator || (deallocator->inputTypes.size() != op->getNumResults() &&
                       deallocator->shapeTypes.size() != op->getNumResults()))
    return groups;

  ResourceGroup group;
  group.offset = 0;
  group.deallocator = deallocator;
  group.values = valueSlice(
      op->getResults(), 0,
      static_cast<unsigned>(deallocator->inputTypes.size()));
  appendEntityViews(group, op->getResults(), 0);
  groups.push_back(std::move(group));
  return groups;
}

static bool resourceGroupStartsAt(llvm::ArrayRef<ResourceGroup> groups,
                                  unsigned offset) {
  return llvm::any_of(groups, [&](const ResourceGroup &group) {
    return group.offset == offset;
  });
}

static void appendUnresolvedOwnedResultRoot(
    mlir::func::CallOp call, unsigned offset,
    llvm::SmallVectorImpl<ResourceGroup> &groups) {
  if (offset >= call.getNumResults() || resourceGroupStartsAt(groups, offset))
    return;
  mlir::Value result = call.getResult(offset);
  if (!isObjectHeaderLikeType(result.getType()))
    return;
  ResourceGroup group;
  group.offset = offset;
  group.values.push_back(result);
  groups.push_back(std::move(group));
}

static std::optional<unsigned>
logicalPayloadOffsetCoveredByStaticEvidence(mlir::func::FuncOp callee,
                                            llvm::StringRef evidenceContract) {
  if (!callee || evidenceContract.empty())
    return std::nullopt;

  auto callableAttr = callee->getAttrOfType<mlir::TypeAttr>(kCallableTypeAttr);
  auto callable = mlir::dyn_cast_if_present<py::CallableType>(
      callableAttr ? callableAttr.getValue() : mlir::Type());
  if (!callable || callable.getResultTypes().size() != 1)
    return std::nullopt;

  mlir::Type resultType = callable.getResultTypes().front();
  std::string resultContract = contracts::runtimeContractName(resultType);
  if (!resultContract.empty())
    return resultContract == evidenceContract ? std::optional<unsigned>(0)
                                              : std::nullopt;

  auto unionType = mlir::dyn_cast<py::UnionType>(resultType);
  if (!unionType || unionType.getMemberTypes().size() != 2)
    return std::nullopt;

  std::optional<mlir::Type> payloadType;
  for (mlir::Type member : unionType.getMemberTypes()) {
    if (py::isPyNoneType(member))
      continue;
    if (payloadType)
      return std::nullopt;
    payloadType = member;
  }
  if (!payloadType)
    return std::nullopt;

  return contracts::runtimeContractName(*payloadType).empty()
             ? std::nullopt
             : std::optional<unsigned>(1);
}

static llvm::SmallSet<unsigned, 4> staticEvidenceCoveredLogicalOffsets(
    mlir::func::FuncOp callee, const FunctionContract &contract) {
  llvm::SmallSet<unsigned, 4> covered;
  for (auto [contractIndex, offset] :
       llvm::enumerate(contract.ownedResults.values)) {
    if (contractIndex >= contract.ownedResultContracts.size())
      continue;
    std::optional<unsigned> logicalOffset =
        logicalPayloadOffsetCoveredByStaticEvidence(
            callee, contract.ownedResultContracts[contractIndex]);
    if (logicalOffset && offset > *logicalOffset)
      covered.insert(*logicalOffset);
  }
  return covered;
}

llvm::SmallVector<ResourceGroup, 8>
collectOwnedCallResultGroups(mlir::ModuleOp module, mlir::func::CallOp call,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators) {
  llvm::SmallVector<ResourceGroup, 8> ownedGroups;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee || call.getNumResults() == 0)
    return ownedGroups;

  mlir::FailureOr<FunctionContract> functionContract =
      readFunctionContract(callee);
  llvm::SmallSet<unsigned, 4> staticEvidenceCoveredOffsets;
  if (mlir::succeeded(functionContract))
    staticEvidenceCoveredOffsets =
        staticEvidenceCoveredLogicalOffsets(callee, *functionContract);

  llvm::SmallVector<ResourceGroup, 4> contractGroups =
      collectContractOwnedResultGroups(callee, call, deallocators);
  if (!contractGroups.empty()) {
    ownedGroups.append(std::make_move_iterator(contractGroups.begin()),
                       std::make_move_iterator(contractGroups.end()));
    return ownedGroups;
  }

  if (auto callableAttr =
          callee->getAttrOfType<mlir::TypeAttr>(kCallableTypeAttr)) {
    auto callable =
        mlir::dyn_cast_if_present<py::CallableType>(callableAttr.getValue());
    if (callable && callable.getResultTypes().size() == 1) {
      llvm::SmallVector<ResourceGroup, 8> typedGroups;
      collectTypedResourceGroups(callable.getResultTypes().front(),
                                 call.getResults(), deallocators,
                                 /*baseOffset=*/0, typedGroups);
      if (!typedGroups.empty()) {
        for (ResourceGroup &group : typedGroups) {
          if (staticEvidenceCoveredOffsets.contains(group.offset))
            continue;
          if (callResultGroupIsOwned(callee, group.offset) || group.condition) {
            ownedGroups.push_back(std::move(group));
          }
        }
      }
    }
  }

  if (mlir::succeeded(functionContract)) {
    for (auto [contractIndex, offset] :
         llvm::enumerate(functionContract->ownedResults.values)) {
      if (resourceGroupStartsAt(ownedGroups, offset))
        continue;
      llvm::StringRef contractName;
      if (contractIndex < functionContract->ownedResultContracts.size())
        contractName = functionContract->ownedResultContracts[contractIndex];
      const RuntimeDeallocator *deallocator =
          contractName.empty()
              ? findDeallocatorForValueGroup(call.getResults(), offset,
                                             deallocators)
              : findDeallocatorForValueGroup(call.getResults(), offset,
                                             deallocators, contractName);
      if (!deallocator)
        continue;
      ResourceGroup group;
      group.offset = offset;
      group.deallocator = deallocator;
      group.values = valueSlice(
          call.getResults(), offset,
          static_cast<unsigned>(deallocator->inputTypes.size()));
      appendEntityViews(group, call.getResults(), offset);
      ownedGroups.push_back(std::move(group));
    }
  }

  for (ResourceGroup group :
       collectRuntimeResourceGroups(call.getResults(), deallocators)) {
    if (resourceGroupStartsAt(ownedGroups, group.offset))
      continue;
    if (staticEvidenceCoveredOffsets.contains(group.offset))
      continue;
    if (callResultGroupIsOwned(callee, group.offset)) {
      ownedGroups.push_back(std::move(group));
      continue;
    }

    if (std::optional<OwnershipCondition> condition =
            optionalUnionPayloadCondition(callee, call, group.offset)) {
      group.condition = *condition;
      ownedGroups.push_back(std::move(group));
    }
  }

  if (mlir::succeeded(functionContract)) {
    for (unsigned offset : functionContract->ownedResults.values)
      appendUnresolvedOwnedResultRoot(call, offset, ownedGroups);
  }
  if (functionUsesOwnedReturnABI(callee))
    appendUnresolvedOwnedResultRoot(call, /*offset=*/0, ownedGroups);

  return ownedGroups;
}

static std::optional<std::int64_t> constantIntValue(mlir::Value value) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getValue().getSExtValue();
}

static bool isConstant(mlir::Value value, std::int64_t expected) {
  auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue());
  if (!integer)
    return false;
  if (integer.getValue().getSExtValue() == expected)
    return true;
  return expected >= 0 && integer.getValue().getZExtValue() ==
                              static_cast<std::uint64_t>(expected);
}

static std::optional<std::int64_t>
comparedTagConstant(mlir::arith::CmpIOp cmp, mlir::Value tag) {
  if (cmp.getLhs() == tag)
    return constantIntValue(cmp.getRhs());
  if (cmp.getRhs() == tag)
    return constantIntValue(cmp.getLhs());
  return std::nullopt;
}

std::optional<bool>
conditionTrueMeansActive(mlir::Value condition,
                         const OwnershipCondition &ownershipCondition) {
  bool inverted = false;
  if (auto xori = condition.getDefiningOp<mlir::arith::XOrIOp>()) {
    if (isConstant(xori.getOperand(0), 1)) {
      condition = xori.getOperand(1);
      inverted = true;
    } else if (isConstant(xori.getOperand(1), 1)) {
      condition = xori.getOperand(0);
      inverted = true;
    }
  }

  auto cmp = condition.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;
  if (cmp.getPredicate() != mlir::arith::CmpIPredicate::eq &&
      cmp.getPredicate() != mlir::arith::CmpIPredicate::ne)
    return std::nullopt;

  std::optional<std::int64_t> matchedTag =
      comparedTagConstant(cmp, ownershipCondition.tag);
  if (!matchedTag)
    return std::nullopt;

  std::optional<bool> activeWhenEqual;
  if (*matchedTag == ownershipCondition.activeTag)
    activeWhenEqual = true;
  else if (ownershipCondition.memberCount == 2)
    activeWhenEqual = false;
  if (!activeWhenEqual)
    return std::nullopt;

  bool trueMeansActive = cmp.getPredicate() == mlir::arith::CmpIPredicate::eq
                             ? *activeWhenEqual
                             : !*activeWhenEqual;
  return inverted ? !trueMeansActive : trueMeansActive;
}

std::optional<OwnershipConditionBranch>
classifyOwnershipConditionBranch(mlir::Operation *op,
                                 const OwnershipCondition &condition) {
  auto branch = mlir::dyn_cast<mlir::cf::CondBranchOp>(op);
  if (!branch)
    return std::nullopt;

  std::optional<bool> trueMeansActive =
      conditionTrueMeansActive(branch.getCondition(), condition);
  if (!trueMeansActive)
    return std::nullopt;

  if (*trueMeansActive)
    return OwnershipConditionBranch{/*activeSuccessor=*/0,
                                    /*inactiveSuccessor=*/1};
  return OwnershipConditionBranch{/*activeSuccessor=*/1,
                                  /*inactiveSuccessor=*/0};
}

void AliasAnalysis::track(mlir::Value value) {
  if (value && !parent.contains(value)) {
    parent[value] = value;
    invalidateAliasBuckets();
  }
}

mlir::Value AliasAnalysis::find(mlir::Value value) {
  track(value);
  mlir::Value root = parent[value];
  if (root == value)
    return root;
  root = find(root);
  parent[value] = root;
  return root;
}

void AliasAnalysis::unionValues(mlir::Value lhs, mlir::Value rhs) {
  if (!lhs || !rhs)
    return;
  mlir::Value lhsRoot = find(lhs);
  mlir::Value rhsRoot = find(rhs);
  if (lhsRoot != rhsRoot) {
    parent[rhsRoot] = lhsRoot;
    invalidateAliasBuckets();
  }
}

bool AliasAnalysis::same(mlir::Value lhs, mlir::Value rhs) {
  return lhs && rhs && find(lhs) == find(rhs);
}

void AliasAnalysis::aliasesOf(mlir::Value value,
                              llvm::SmallVectorImpl<mlir::Value> &aliases) {
  if (!value)
    return;
  mlir::Value root = find(value);
  if (aliasBucketsDirty)
    rebuildAliasBuckets();
  auto found = aliasBuckets.find(root);
  if (found == aliasBuckets.end())
    return;
  aliases.append(found->second.begin(), found->second.end());
}

void AliasAnalysis::invalidateAliasBuckets() {
  aliasBuckets.clear();
  aliasBucketsDirty = true;
}

void AliasAnalysis::rebuildAliasBuckets() {
  aliasBuckets.clear();
  for (auto &entry : parent) {
    mlir::Value root = find(entry.first);
    aliasBuckets[root].push_back(entry.first);
  }
  aliasBucketsDirty = false;
}

static bool isOwnershipIdentityOp(mlir::Operation *op) {
  llvm::StringRef name = op->getName().getStringRef();
  return name == "py.union.wrap" || name == "py.union.unwrap" ||
         name == "py.class.upcast" || name == "py.class.refine" ||
         name == "py.protocol.view" || name == "memref.cast" ||
         name == "memref.subview" ||
         name == "builtin.unrealized_conversion_cast";
}

static void unionStaticEvidenceCallResultAliases(AliasAnalysis &aliases,
                                                 mlir::func::CallOp call) {
  mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return;
  mlir::func::FuncOp callee =
      module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return;

  mlir::FailureOr<FunctionContract> contract = readFunctionContract(callee);
  if (mlir::failed(contract))
    return;

  for (auto [contractIndex, staticOffset] :
       llvm::enumerate(contract->ownedResults.values)) {
    if (contractIndex >= contract->ownedResultContracts.size())
      continue;
    std::optional<unsigned> logicalOffset =
        logicalPayloadOffsetCoveredByStaticEvidence(
            callee, contract->ownedResultContracts[contractIndex]);
    if (!logicalOffset || staticOffset <= *logicalOffset ||
        staticOffset >= call.getNumResults())
      continue;

    unsigned logicalCount = staticOffset - *logicalOffset;
    unsigned staticCount = call.getNumResults() - staticOffset;
    unsigned count = std::min(logicalCount, staticCount);
    for (unsigned index = 0; index < count; ++index) {
      aliases.unionValues(call.getResult(*logicalOffset + index),
                          call.getResult(staticOffset + index));
    }
  }
}

void AliasAnalysis::build(mlir::Operation *root) {
  root->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      track(operand);
    for (mlir::Value result : op->getResults())
      track(result);
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
      unionStaticEvidenceCallResultAliases(*this, call);
    if (auto subview = mlir::dyn_cast<mlir::memref::SubViewOp>(op)) {
      unionValues(subview.getResult(), subview.getSource());
      return;
    }
    if (auto select = mlir::dyn_cast<mlir::arith::SelectOp>(op)) {
      // A select result is a borrow of whichever operand was picked: alias it
      // with BOTH so the sources stay live for as long as the result does.
      unionValues(select.getResult(), select.getTrueValue());
      unionValues(select.getResult(), select.getFalseValue());
      return;
    }
    if (!isOwnershipIdentityOp(op) ||
        op->getNumOperands() != op->getNumResults())
      return;
    for (auto [result, operand] : llvm::zip(op->getResults(), op->getOperands()))
      unionValues(result, operand);
  });
}

// Exceptional successor edges for the setjmp-style EH model: the anchor
// cond_br (`%c = call @LyEH_TryCatchAnchor(id); cf.cond_br %c, ^handler,
// ^try`) presents the try and handler paths as EXCLUSIVE, but at runtime the
// handler runs AFTER the try executed up to a raising call site. Any block
// containing `LyEH_TryCallSiteMarker(id)` may therefore transfer control to
// the handler entry of `id`; liveness that ignores these edges releases
// values on the try path that the handler still uses (use-after-free).
llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
collectExceptionEdges(mlir::Region &region) {
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>> edges;
  llvm::DenseMap<std::int64_t, mlir::Block *> handlerEntries;
  auto markerId = [](mlir::func::CallOp call) -> std::optional<std::int64_t> {
    if (call.getNumOperands() != 1)
      return std::nullopt;
    auto constant =
        call.getOperand(0).getDefiningOp<mlir::arith::ConstantIntOp>();
    if (!constant)
      return std::nullopt;
    return constant.value();
  };
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
      if (!call || call.getCallee() != "LyEH_TryCatchAnchor")
        continue;
      std::optional<std::int64_t> id = markerId(call);
      auto cond =
          mlir::dyn_cast<mlir::cf::CondBranchOp>(block.getTerminator());
      if (!id || !cond || cond.getCondition() != call.getResult(0))
        continue;
      handlerEntries[*id] = cond.getTrueDest();
    }
  }
  if (handlerEntries.empty())
    return edges;
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
      if (!call || call.getCallee() != "LyEH_TryCallSiteMarker")
        continue;
      std::optional<std::int64_t> id = markerId(call);
      if (!id)
        continue;
      auto found = handlerEntries.find(*id);
      if (found == handlerEntries.end())
        continue;
      auto &list = edges[&block];
      if (!llvm::is_contained(list, found->second))
        list.push_back(found->second);
    }
  }
  return edges;
}

} // namespace py::ownership
