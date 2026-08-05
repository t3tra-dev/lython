#include "Ownership.h"

#include "Common/PythonSourceRange.h"

#include "PyDialectTypes.h"
#include "Contracts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <optional>

namespace py::ownership {

namespace contracts = py::contracts;

// Deallocator-lookup census, enabled by LYTHON_DEALLOC_CENSUS=1.
//
// Why NOT key the counters on `values[offset]`: an arity-1 key cannot name an
// arity-3 tie.  The four container contracts led with `memref<2xi64>`, so a
// census keyed on the leading value folded a 3-type tie into the width-2 bucket
// and reported "all 119 ambiguous exits were on memref<2xi64>" when 63 of them
// were the container tie.  The key here is the whole tied inputTypes list.
//
// Why NOT a compile-time flag: the ablation has to run on ONE binary, so that a
// with/without comparison cannot be confounded by a rebuild.
namespace {

struct DeallocCensus {
  // Keyed by the printed inputTypes list of the tied release interface.
  llvm::StringMap<uint64_t> ambiguous;   // Ownership.cpp:419, contract-less exit
  llvm::StringMap<uint64_t> resolved;    // contract-aware overload succeeded
  llvm::StringMap<uint64_t> unresolved;  // callee whose owned result found none
  uint64_t emptyName = 0;                // :429, no contract name at the call
  uint64_t fallback = 0;                 // :450, name present but no type match
  uint64_t contractAwareAmbiguous = 0;   // named overload tied (same contract)
  // Times a reader that HAS a callee fell to the contract-less overload because
  // ownedResultContractName() returned nothing.  This -- not `emptyName` -- is
  // where GAP 1 surfaces: the contract-aware overload is never entered at all,
  // so its `contractName.empty()` guard never runs.
  uint64_t declaredNameAbsent = 0;

  static bool enabled() {
    static const bool on = [] {
      const char *value = std::getenv("LYTHON_DEALLOC_CENSUS");
      return value && value[0] == '1';
    }();
    return on;
  }

  ~DeallocCensus() {
    if (!enabled())
      return;
    llvm::errs() << "[DEALLOC] empty_name=" << emptyName
                 << " fallback_450=" << fallback
                 << " contract_aware_ambiguous=" << contractAwareAmbiguous
                 << " declared_name_absent=" << declaredNameAbsent << "\n";
    for (auto &entry : ambiguous)
      llvm::errs() << "[DEALLOC] ambiguous " << entry.getKey() << " = "
                   << entry.getValue() << "\n";
    for (auto &entry : resolved)
      llvm::errs() << "[DEALLOC] resolved " << entry.getKey() << " = "
                   << entry.getValue() << "\n";
    for (auto &entry : unresolved)
      llvm::errs() << "[DEALLOC] unresolved_callee " << entry.getKey() << " = "
                   << entry.getValue() << "\n";
  }
};

DeallocCensus &census() {
  static DeallocCensus instance;
  return instance;
}

// Which collector asked.  An ambiguous exit is only actionable by declaring a
// contract name if the asking collector has a callee to read the name from;
// `collectRuntimeResourceGroups` scans a bare value range and has none.
const char *g_origin = "other";

struct OriginScope {
  const char *previous;
  explicit OriginScope(const char *name) : previous(g_origin) {
    g_origin = name;
  }
  ~OriginScope() { g_origin = previous; }
};

std::string typeListKey(llvm::ArrayRef<mlir::Type> types) {
  std::string out;
  llvm::raw_string_ostream os(out);
  os << "(";
  llvm::interleaveComma(types, os);
  os << ")";
  return out;
}

} // namespace

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

bool canSpellHeaderPrefix(mlir::Type from, mlir::Type to) {
  if (from == to)
    return true;
  if (mlir::memref::CastOp::areCastCompatible(from, to))
    return true;
  auto source = mlir::dyn_cast<mlir::MemRefType>(from);
  auto target = mlir::dyn_cast<mlir::MemRefType>(to);
  return source && target && source.getRank() == 1 && target.getRank() == 1 &&
         source.hasStaticShape() && target.hasStaticShape() &&
         source.getElementType() == target.getElementType() &&
         source.getDimSize(0) >= target.getDimSize(0);
}

mlir::Value spellHeaderPrefix(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value header, mlir::Type target) {
  if (header.getType() == target)
    return header;
  if (mlir::memref::CastOp::areCastCompatible(header.getType(), target))
    return mlir::memref::CastOp::create(builder, loc, target, header)
        .getResult();
  // A handle WIDER than the retain/release interface holds the refcount+class
  // prefix inside its own static shape: a box-fronted class instance (the whole
  // 16-word box is the entity root), and every contract whose interior state
  // lives behind the handle rather than in lanes beside it. Take the rank-1
  // prefix rather than declining, which is what the caller used to do.
  auto sourceType = mlir::dyn_cast<mlir::MemRefType>(header.getType());
  auto targetType = mlir::dyn_cast<mlir::MemRefType>(target);
  if (!canSpellHeaderPrefix(header.getType(), target) || !sourceType ||
      !targetType)
    return {};
  llvm::SmallVector<mlir::OpFoldResult, 1> offsets{builder.getIndexAttr(0)};
  llvm::SmallVector<mlir::OpFoldResult, 1> sizes{
      builder.getIndexAttr(targetType.getDimSize(0))};
  llvm::SmallVector<mlir::OpFoldResult, 1> strides{builder.getIndexAttr(1)};
  llvm::SmallVector<int64_t, 1> resultShape{targetType.getDimSize(0)};
  auto inferredType = mlir::cast<mlir::MemRefType>(
      mlir::memref::SubViewOp::inferRankReducedResultType(
          resultShape, sourceType, offsets, sizes, strides));
  mlir::Value prefix = mlir::memref::SubViewOp::create(
                           builder, loc, inferredType, header, offsets, sizes,
                           strides)
                           .getResult();
  if (prefix.getType() == target)
    return prefix;
  if (!mlir::memref::CastOp::areCastCompatible(prefix.getType(), target))
    return {};
  return mlir::memref::CastOp::create(builder, loc, target, prefix).getResult();
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
  if (ambiguous) {
    if (DeallocCensus::enabled() && matched)
      ++census().ambiguous[std::string(g_origin) + " " +
                           typeListKey(matched->inputTypes)];
    return nullptr;
  }
  return matched;
}

const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             llvm::StringRef contractName) {
  if (contractName.empty()) {
    if (DeallocCensus::enabled())
      ++census().emptyName;
    return findDeallocatorForValueGroup(values, offset, deallocators);
  }

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
  if (matched) {
    if (DeallocCensus::enabled()) {
      if (ambiguous)
        ++census().contractAwareAmbiguous;
      else
        ++census().resolved[contractName];
    }
    return ambiguous ? nullptr : matched;
  }

  if (DeallocCensus::enabled())
    ++census().fallback;
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

// A call to a manifest primitive declared as returning interior words of the
// entity its operands reach (see kManifestInteriorWordAttr).
static bool isInteriorWordCall(mlir::Operation *op) {
  auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
  if (!call)
    return false;
  auto callee =
      mlir::SymbolTable::lookupNearestSymbolFrom<mlir::func::FuncOp>(
          call, call.getCalleeAttr());
  return callee && callee->hasAttr(contracts::kManifestInteriorWordAttr);
}

void collectBoxWordDerivedViews(llvm::ArrayRef<mlir::Value> groupValues,
                                llvm::SmallVectorImpl<mlir::Value> &views) {
  llvm::SmallDenseSet<mlir::Value, 8> known(views.begin(), views.end());
  llvm::SmallVector<mlir::Value, 8> worklist;
  auto seedInteriorResults = [&](mlir::Operation *user) {
    if (!isInteriorWordCall(user))
      return false;
    for (mlir::Value result : user->getResults()) {
      worklist.push_back(result);
      // The word itself pins too: a primitive that stores into or releases a
      // slot of the block consumes only the word, so without it the entity's
      // last use would look like the call that produced the word.
      if (known.insert(result).second)
        views.push_back(result);
    }
    return true;
  };
  for (mlir::Value value : groupValues)
    for (mlir::Operation *user : value.getUsers()) {
      if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(user)) {
        if (load.getMemRef() == value)
          worklist.push_back(load.getResult());
        continue;
      }
      seedInteriorResults(user);
    }
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
      if (seedInteriorResults(user))
        continue;
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

  OriginScope originScope("siteA/contractOwnedResultGroups");
  for (auto [contractIndex, offset] :
       llvm::enumerate(contract->ownedResults.values)) {
    // The declaration is the authority on WHICH entity the owned result is; the
    // receiver contract only says whose method produced it.  For a method
    // returning a different entity -- `LyLong_Repr` is a `builtins.int` method
    // whose owned result is a `builtins.str` -- the receiver name selects the
    // receiver's deallocator, and it is accepted because the release interfaces
    // of the width-2 group are type-identical.  Measured on the tree before this
    // change: 24 owned results, `LyLong_Repr` among them, were released through
    // another contract's deallocator.
    //
    // Why NOT leave this to the sibling collector, which already reads the
    // declared name: it runs only for offsets no group covers yet
    // (`resourceGroupStartsAt`), and this loop has already claimed the offset
    // under the receiver's name.  Declaring `ly.runtime.result_contract` was
    // therefore MEASURED to be inert here -- byte-identical llvm-translation IR
    // across 10 programs -- so the attribute could not fix mechanism (C) while
    // the receiver name was preferred.
    //
    // Why NOT consult the declared name only when the receiver's deallocator
    // fails to type-match: that keeps identity dependent on a width coincidence,
    // which is the defect rather than a guard against it.
    llvm::StringRef declaredName = ownedResultContractName(
        callee, *contract, static_cast<unsigned>(contractIndex));
    const RuntimeDeallocator *deallocator = findDeallocatorForValueGroup(
        call.getResults(), offset, deallocators,
        declaredName.empty() ? contractAttr.getValue() : declaredName);
    if (!deallocator)
      continue;
    ResourceGroup group;
    group.offset = offset;
    group.deallocator = deallocator;
    group.values =
        valueSlice(call.getResults(), offset,
                   static_cast<unsigned>(deallocator->inputTypes.size()));
    group.root = entityRootOf(group.values);
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
      group.root = entityRootOf(group.values);
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
  OriginScope originScope("scan/collectRuntimeResourceGroups");
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
    group.root = entityRootOf(group.values);
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
  // The marker's result 0, normalized past the identity cast: a re-root
  // republishes the SAME head through a fresh cast, so the normalized root is
  // the one name that survives it.
  group.root = entityRootOf(group.values);
  appendEntityViews(group, op->getResults(), 0);
  groups.push_back(std::move(group));
  return groups;
}

// TWO OWNED TOKENS CAN SHARE ONE OBJECT, AND ONLY ONE OF THEM IS SPELLED BY THE
// NAME A GIVEN RELEASE USES. An owned-local marker that a retain mints (an
// evidence-selected container element: retain the borrowed element, then root it
// through an identity cast) names a token the object did not have before. Every
// other name for that object -- the literal's element source, the local binding
// it came from -- carries a DIFFERENT token with its own release. Reading one of
// those releases as this token's death leaked one element object per execution
// of every container literal whose element is read back (`ys = [99]; total +=
// ys[0]`, 64 B/iteration, unbounded); answering the other way for the OTHER kind
// of marker double-frees.
//
// The other kind is the re-root, which republishes an already-owned head with no
// retain at all and leaves its own results unused -- every real use, including
// the consuming one, is on the pre-marker name. So the two kinds need opposite
// answers about the head's other releases, and one attribute covers both.
//
// Why the retain must be the marker's IMMEDIATE predecessor and not merely
// present before it: two markers on one head would each find the other's retain
// and both claim a token. Adjacency is what the producer emits (the
// header-spelling ops precede the retain call), so the tight form costs nothing.
// If a producer ever stops emitting them adjacently, this goes quiet and the
// behaviour falls back to the leak -- the safe direction, since the other one is
// a double free.
// `LYTHON_ABLATE_OWNED_LOCAL_TOKEN_SPLIT=1` restores the shipped answer (no
// marker roots a token of its own), so an A/B runs off ONE binary -- a rebuild
// of the same source does not reproduce byte for byte, so "the shas differ"
// never establishes that two arms differ. Its failure mode is the leak, so it is
// for bisecting a regression to this rule, never for production.
bool ownedLocalTokenSplitAblated() {
  static const bool ablated = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_OWNED_LOCAL_TOKEN_SPLIT");
    return value && !value->empty() && *value != "0";
  }();
  return ablated;
}

bool perReferenceReleaseLabels() {
  static const bool ablated = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_REFERENCE_RELEASE");
    return value && !value->empty() && *value != "0";
  }();
  return !ablated;
}

bool unwindTracksMintedTokensSeparately() {
  static const bool ablated = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_UNWIND_MINTED_TOKENS");
    return value && !value->empty() && *value != "0";
  }();
  return !ablated;
}

bool ownedLocalMarkerIsRetainRooted(mlir::Operation *marker,
                                    AliasAnalysis &aliases) {
  if (ownedLocalTokenSplitAblated())
    return false;
  if (!marker || !marker->hasAttr(kOwnedLocalObjectAttr) ||
      marker->getNumOperands() == 0)
    return false;
  auto call = mlir::dyn_cast_or_null<mlir::func::CallOp>(marker->getPrevNode());
  if (!call || call.getNumOperands() != 1 ||
      call->hasAttr(kAggregateRetainAttr))
    return false;
  auto module = marker->getParentOfType<mlir::ModuleOp>();
  if (!module)
    return false;
  auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  if (!callee)
    return false;
  auto primitive =
      callee->getAttrOfType<mlir::StringAttr>(contracts::kManifestPrimitiveAttr);
  if (!primitive || primitive.getValue() != "retain")
    return false;
  return aliases.same(call.getOperand(0), marker->getOperand(0));
}

llvm::StringRef ownedResultContractName(mlir::func::FuncOp function,
                                        const FunctionContract &contract,
                                        unsigned contractIndex) {
  if (!function)
    return {};
  if (contractIndex < contract.ownedResultContracts.size()) {
    llvm::StringRef declared = contract.ownedResultContracts[contractIndex];
    if (!declared.empty())
      return declared;
  }
  // Type-only matching cannot tell physical twins apart (str and bytes share the
  // (header, byte payload) shape), so a declared result contract names the
  // entity before the structural fallback runs.
  //
  // Why NOT also consult `ly.runtime.element_contract` / `next_contract` here:
  // those name a DIFFERENT result of the same call (the yielded element and the
  // advanced iterator), so feeding either one to an arbitrary `contractIndex`
  // would answer a question about result 0 with the name of result 4. Only 3
  // declarations name their owned results solely through that pair
  // (`LyRangeIterator_Next`, `LyUnicodeStrIterator_Next`, `LyCounter_Next`), and
  // supplying `owned_result_contracts` for the first of them was MEASURED not to
  // change its behaviour -- so the missing channel is not what those three need,
  // and guessing an offset mapping here would be an unmeasured change.
  if (auto resultContract = function->getAttrOfType<mlir::StringAttr>(
          contracts::kManifestResultContractAttr))
    return resultContract.getValue();
  return {};
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
  group.root = entityRootOf(group.values);
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
    llvm::StringRef contractName = ownedResultContractName(
        callee, contract, static_cast<unsigned>(contractIndex));
    if (contractName.empty())
      continue;
    std::optional<unsigned> logicalOffset =
        logicalPayloadOffsetCoveredByStaticEvidence(callee, contractName);
    if (logicalOffset && offset > *logicalOffset)
      covered.insert(*logicalOffset);
  }
  return covered;
}

llvm::SmallVector<ResourceGroup, 8>
collectOwnedCallResultGroups(mlir::ModuleOp module, mlir::func::CallOp call,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             mlir::SymbolTable *symbols) {
  llvm::SmallVector<ResourceGroup, 8> ownedGroups;
  // `symbols`, when given, is a table over the SAME module: it resolves the
  // callee in constant time instead of walking the module's symbol list, which
  // is what made a per-call-op sweep O(calls x symbols).
  mlir::func::FuncOp callee =
      symbols ? symbols->lookup<mlir::func::FuncOp>(call.getCallee())
              : module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
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
      // Contract-declared owned lanes (returned-closure captures, static
      // object evidence) trail the primary result; the typed matcher needs
      // the primary result's exact span or it finds nothing and the primary
      // object degrades to an unresolvable root.
      mlir::ValueRange typedValues = call.getResults();
      if (mlir::succeeded(functionContract) &&
          !functionContract->ownedResults.empty()) {
        unsigned firstContractOwned = *llvm::min_element(
            functionContract->ownedResults.values);
        if (firstContractOwned <= typedValues.size())
          typedValues = typedValues.take_front(firstContractOwned);
      }
      llvm::SmallVector<ResourceGroup, 8> typedGroups;
      collectTypedResourceGroups(callable.getResultTypes().front(),
                                 typedValues, deallocators,
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
      llvm::StringRef contractName = ownedResultContractName(
          callee, *functionContract, static_cast<unsigned>(contractIndex));
      OriginScope originScope("sibling/callResultGroups");
      if (DeallocCensus::enabled() && contractName.empty())
        ++census().declaredNameAbsent;
      const RuntimeDeallocator *deallocator =
          contractName.empty()
              ? findDeallocatorForValueGroup(call.getResults(), offset,
                                             deallocators)
              : findDeallocatorForValueGroup(call.getResults(), offset,
                                             deallocators, contractName);
      if (!deallocator) {
        // Name the callees whose owned result no lookup can resolve, as a SET.
        // A count says how many groups went missing; only the names say which
        // declaration would close them.
        if (DeallocCensus::enabled())
          ++census().unresolved[(callee ? callee.getName() : "<indirect>")];
        continue;
      }
      ResourceGroup group;
      group.offset = offset;
      group.deallocator = deallocator;
      group.values = valueSlice(
          call.getResults(), offset,
          static_cast<unsigned>(deallocator->inputTypes.size()));
      group.root = entityRootOf(group.values);
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

void AliasAnalysis::namesOf(mlir::Value value,
                           llvm::SmallVectorImpl<mlir::Value> &names) {
  if (!value)
    return;
  mlir::Value root = find(value);
  if (aliasBucketsDirty)
    rebuildAliasBuckets();
  auto found = aliasBuckets.find(root);
  if (found == aliasBuckets.end()) {
    names.push_back(value); // untracked: it is still a name for itself
    return;
  }
  names.append(found->second.begin(), found->second.end());
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
                                                 mlir::func::CallOp call,
                                                 mlir::SymbolTable *symbols) {
  mlir::func::FuncOp callee;
  if (symbols) {
    callee = symbols->lookup<mlir::func::FuncOp>(call.getCallee());
  } else {
    mlir::ModuleOp module = call->getParentOfType<mlir::ModuleOp>();
    if (!module)
      return;
    callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
  }
  if (!callee)
    return;

  mlir::FailureOr<FunctionContract> contract = readFunctionContract(callee);
  if (mlir::failed(contract))
    return;

  for (auto [contractIndex, staticOffset] :
       llvm::enumerate(contract->ownedResults.values)) {
    llvm::StringRef contractName = ownedResultContractName(
        callee, *contract, static_cast<unsigned>(contractIndex));
    if (contractName.empty())
      continue;
    std::optional<unsigned> logicalOffset =
        logicalPayloadOffsetCoveredByStaticEvidence(callee, contractName);
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
  // Resolving each call's callee through `Operation::lookupSymbol` walks the
  // module's symbol list, so the per-call static-evidence union below cost
  // O(calls x symbols) -- the term that exploded once an imported stdlib
  // module's symbols joined the module. One symbol table answers the same
  // question (immediate symbol children of `root`) in constant time.
  std::optional<mlir::SymbolTable> symbols;
  if (root->hasTrait<mlir::OpTrait::SymbolTable>())
    symbols.emplace(root);
  root->walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands())
      track(operand);
    for (mlir::Value result : op->getResults())
      track(result);
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
      unionStaticEvidenceCallResultAliases(
          *this, call, symbols ? &*symbols : nullptr);
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
std::optional<std::int64_t> exceptionMarkerId(mlir::func::CallOp call) {
  if (call.getNumOperands() != 1)
    return std::nullopt;
  auto constant =
      call.getOperand(0).getDefiningOp<mlir::arith::ConstantIntOp>();
  if (!constant)
    return std::nullopt;
  return constant.value();
}

llvm::DenseMap<std::int64_t, mlir::Block *>
collectExceptionHandlerEntries(mlir::Region &region) {
  // Resolution is by `LyEH_TryCatchMarker(id)` -- the block the final LLVM
  // EH phase wires each unwinding invoke to -- and NOT by the anchor
  // cond_br's true successor: release insertion may split blocks on the
  // anchor's (never-taken-at-runtime) true edge, after which the anchor
  // successor and the runtime catch target are different blocks.
  llvm::DenseMap<std::int64_t, mlir::Block *> handlerEntries;
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
      if (!call || call.getCallee() != "LyEH_TryCatchMarker")
        continue;
      if (std::optional<std::int64_t> id = exceptionMarkerId(call))
        handlerEntries[*id] = &block;
    }
  }
  return handlerEntries;
}

llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
collectExceptionEdges(mlir::Region &region) {
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>> edges;
  llvm::DenseMap<std::int64_t, mlir::Block *> handlerEntries =
      collectExceptionHandlerEntries(region);
  if (handlerEntries.empty())
    return edges;
  for (mlir::Block &block : region) {
    for (mlir::Operation &op : block) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
      if (!call || call.getCallee() != "LyEH_TryCallSiteMarker")
        continue;
      std::optional<std::int64_t> id = exceptionMarkerId(call);
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

mlir::func::CallOp guardedCallAfterMarker(mlir::Operation *marker) {
  if (!marker)
    return {};
  for (mlir::Operation *op = marker->getNextNode(); op;
       op = op->getNextNode()) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!call) {
      // Mirror the final EH phase's pairing scan: a side-effecting
      // non-call between marker and call breaks the pairing there, so a
      // pairing claimed across one here would model an edge that never
      // materializes.
      if (op->hasTrait<mlir::OpTrait::IsTerminator>() ||
          !mlir::isMemoryEffectFree(op))
        return {};
      continue;
    }
    llvm::StringRef callee = call.getCallee();
    if (callee == "LyEH_TryCallSiteMarker")
      continue; // a later marker takes over the pairing
    if (callee == "LyEH_TryCatchAnchor" || callee == "LyEH_TryCatchMarker")
      return {};
    // Non-unwinding release helpers and the raise path's traceback frame sit
    // between a marker and its guarded call once ownership insertion has
    // scheduled pre-raise releases; the final EH phase's pairing scan skips
    // them, so the mirror must too — otherwise the guarded raise is analyzed
    // as guarding a DecRef, the raised exception loses its consumed-by-raise
    // exemption, and the unwind cleanup releases the exception the handler
    // is about to read.
    if (callee == "Ly_IncRef" || callee == "Ly_DecRef" ||
        callee.ends_with("_DecRef") ||
        callee == "LyObject_ReleaseStorageToZero" ||
        callee.starts_with("__ly_dealloc_") ||
        callee.starts_with("__ly_unwind_cleanup_") ||
        callee.starts_with("LyTraceback_"))
      continue;
    return call;
  }
  return {};
}

mlir::func::CallOp precedingTryCallSiteMarker(mlir::Operation *call) {
  if (!call)
    return {};
  for (mlir::Operation *op = call->getPrevNode(); op; op = op->getPrevNode()) {
    auto candidate = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!candidate) {
      if (!mlir::isMemoryEffectFree(op))
        return {};
      continue;
    }
    if (candidate.getCallee() == "LyEH_TryCallSiteMarker")
      return candidate;
    return {};
  }
  return {};
}

mlir::func::CallOp anchorTrueEdgeGuardedCall(mlir::Operation *terminator) {
  auto cond = mlir::dyn_cast_if_present<mlir::cf::CondBranchOp>(terminator);
  if (!cond)
    return {};
  auto anchor = cond.getCondition().getDefiningOp<mlir::func::CallOp>();
  if (!anchor || anchor.getCallee() != "LyEH_TryCatchAnchor")
    return {};
  std::optional<std::int64_t> id = exceptionMarkerId(anchor);
  if (!id)
    return {};
  for (mlir::Operation &op : *cond.getFalseDest()) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
    if (!call)
      continue;
    if (call.getCallee() != "LyEH_TryCallSiteMarker")
      continue;
    std::optional<std::int64_t> markerId = exceptionMarkerId(call);
    if (markerId && *markerId == *id)
      return guardedCallAfterMarker(call);
    return {};
  }
  return {};
}

bool isRaisePrimitiveFunction(mlir::func::FuncOp function) {
  if (!function)
    return false;
  auto primitive =
      function->getAttrOfType<mlir::StringAttr>(contracts::kManifestPrimitiveAttr);
  return primitive && primitive.getValue() == "raise";
}

bool isRaiseLikeFunction(mlir::func::FuncOp function) {
  if (!function)
    return false;
  // Not folded into isRaisePrimitiveFunction: these two are lowering-created
  // support declarations without a manifest contract, so the manifest
  // attribute can never carry the fact -- and stamping the attribute onto
  // them would make manifest audits report primitives no manifest declares.
  llvm::StringRef name = function.getName();
  if (name == "LyEH_RethrowCurrent" || name == "LyEH_ThrowException" ||
      name == "LyEH_StarRethrowResidual" ||
      name == "LyEH_StarRethrowSoleCollected" ||
      name == "LyEH_StarThrowCombined")
    return true;
  return isRaisePrimitiveFunction(function);
}

bool mayRaisePythonException(mlir::func::FuncOp function) {
  if (!function)
    return false;
  if (isRaiseLikeFunction(function))
    return true;
  llvm::StringRef name = function.getName();
  // Mirror of the final EH phase's non-raising runtime set (Cleanup/EH.cpp):
  // EH bookkeeping, refcount maintenance, and traceback writes never throw, so
  // classifying them as may-raise here would demand cleanup edges the EH phase
  // never materializes.
  //
  // FIRST, not last. The generated deallocators and unwind-cleanup thunks are
  // written from a Python location, so the source test below reaches them; they
  // are release compositions, and asking for a cleanup around the call that IS
  // the cleanup makes the affine verifier refuse 42 programs that compile.
  if (name == "LyEH_BeginCatch" || name == "LyEH_ClassIdMatches" ||
      name == "LyEH_CurrentExceptionClassId" ||
      name == "LyEH_CurrentExceptionMatches" ||
      name == "LyEH_DiscardCurrentExceptionIfMatches" ||
      name == "LyEH_DiscardCurrentException" ||
      name == "LyEH_TryCallSiteMarker" || name == "LyEH_TryCatchMarker" ||
      name == "LyEH_TryCatchAnchor" || name.starts_with("LyTraceback_"))
    return false;
  if (name == "Ly_IncRef" || name == "Ly_DecRef" || name.ends_with("_DecRef") ||
      name == "LyObject_ReleaseStorageToZero" ||
      name.starts_with("__ly_dealloc_") ||
      name.starts_with("__ly_unwind_cleanup_"))
    return false;
  // Python-level callables can always raise (any call inside them can).
  if (function->hasAttr(kCallableTypeAttr) && !function.isDeclaration())
    return true;
  // AND ANYTHING ELSE COMPILED FROM PYTHON SOURCE, whatever it is named. This
  // is the first clause of `mayPropagatePythonException` in Cleanup/EH.cpp,
  // which this predicate is the mirror of, and it was the clause missing here.
  //
  // The name test below cannot stand in for it: the generator state machine
  // emits `g__lyrt_gen_resume__step` from a Python `def`, and the EH phase
  // invoke-converts calls to it -- the unwind edge is in the IR -- while this
  // predicate said the callee could not raise, so no ownership cleanup was
  // attached to that edge. `for v in g()` where `g` raises then unwound past
  // the only place that would have released the 512-byte generator frame.
  if (!function.isDeclaration() &&
      findPythonSourceLoc(function.getLoc()).has_value())
    return true;
  return name.starts_with("Ly");
}

bool returnTransfersGroup(FuncContractCache &contracts,
                          mlir::func::FuncOp function,
                          mlir::func::ReturnOp returnOp,
                          llvm::ArrayRef<mlir::Value> group,
                          llvm::ArrayRef<RuntimeDeallocator> deallocators,
                          AliasAnalysis &aliases) {
  auto cached = contracts.lookup(function);
  if (mlir::succeeded(cached) && *cached) {
    for (unsigned offset : (*cached)->contract.ownedResults.values)
      if (groupMatchesValues(returnOp.getOperands(), offset, group, aliases))
        return true;
  }

  // Without the owned-return ABI the contract loop above is the only
  // transfer surface; re-scanning the contract here (as one caller
  // historically did) can never match again and just reads as a divergence.
  if (!functionUsesOwnedReturnABI(function))
    return false;

  std::optional<llvm::SmallVector<OwnedReturnRange, 4>> ranges =
      callableOwnedReturnRanges(function, returnOp.getOperands(),
                                deallocators);
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

bool callConsumesGroup(FuncContractCache &contracts, mlir::func::CallOp call,
                       llvm::ArrayRef<mlir::Value> group,
                       AliasAnalysis &aliases) {
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  for (unsigned offset : (*cached)->contract.releaseArgs.values)
    if (groupMatchesValues(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : (*cached)->contract.transferArgs.values)
    if (groupMatchesValues(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

bool callRetainsGroup(FuncContractCache &contracts, mlir::func::CallOp call,
                      llvm::ArrayRef<mlir::Value> group,
                      AliasAnalysis &aliases) {
  if (group.empty())
    return false;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  for (unsigned offset : (*cached)->contract.retainArgs.values) {
    if (offset >= call.getNumOperands())
      continue;
    if (aliases.same(call.getOperand(offset), group.front()))
      return true;
  }
  return false;
}

bool callPartiallyConsumesGroup(FuncContractCache &contracts,
                                mlir::func::CallOp call,
                                llvm::ArrayRef<mlir::Value> group,
                                AliasAnalysis &aliases) {
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return false;
  const FunctionContract &contract = (*cached)->contract;

  auto consumesTrackedHeaderAt = [&](unsigned index) {
    return !group.empty() && index < call.getNumOperands() &&
           aliases.same(call.getOperand(index), group.front());
  };
  for (unsigned offset : contract.releaseArgs.values)
    if (consumesTrackedHeaderAt(offset) &&
        !groupMatchesValues(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : contract.transferArgs.values)
    if (consumesTrackedHeaderAt(offset) &&
        !groupMatchesValues(call.getOperands(), offset, group, aliases))
      return true;
  return false;
}

std::optional<llvm::SmallVector<mlir::Value, 4>>
callReRootsGroupLanes(FuncContractCache &contracts, mlir::func::CallOp call,
                      llvm::ArrayRef<mlir::Value> group,
                      AliasAnalysis &aliases) {
  if (group.size() < 2)
    return std::nullopt;
  auto cached = contracts.lookup(call.getCallee());
  if (mlir::failed(cached) || !*cached)
    return std::nullopt;
  const FunctionContract &contract = (*cached)->contract;
  if (contract.ownedResults.empty())
    return std::nullopt;

  auto substituteFrom = [&](unsigned operandOffset)
      -> std::optional<llvm::SmallVector<mlir::Value, 4>> {
    if (operandOffset >= call.getNumOperands())
      return std::nullopt;
    // Which lane does the consumed operand name? Lane 0 is excluded: the root
    // IS the entity, so consuming it is a consume, never a re-root.
    unsigned laneIndex = 0;
    for (unsigned index = 1; index < group.size(); ++index) {
      if (aliases.same(call.getOperand(operandOffset), group[index])) {
        laneIndex = index;
        break;
      }
    }
    if (laneIndex == 0)
      return std::nullopt;

    // How far does the sub-range reach? The operands must go on naming the
    // group's following lanes in order: a primitive that took only part of a
    // payload is not handing back a replacement for the whole of it.
    unsigned span = 1;
    while (laneIndex + span < group.size() &&
           operandOffset + span < call.getNumOperands() &&
           aliases.same(call.getOperand(operandOffset + span),
                        group[laneIndex + span]))
      ++span;

    for (unsigned resultOffset : contract.ownedResults.values) {
      if (resultOffset + span > call.getNumResults())
        continue;
      bool typesMatch = true;
      for (unsigned index = 0; index < span; ++index)
        if (call.getResult(resultOffset + index).getType() !=
            group[laneIndex + index].getType())
          typesMatch = false;
      if (!typesMatch)
        continue;
      llvm::SmallVector<mlir::Value, 4> substituted(group.begin(), group.end());
      for (unsigned index = 0; index < span; ++index)
        substituted[laneIndex + index] = call.getResult(resultOffset + index);
      return substituted;
    }
    return std::nullopt;
  };

  for (unsigned offset : contract.transferArgs.values)
    if (auto substituted = substituteFrom(offset))
      return substituted;
  for (unsigned offset : contract.releaseArgs.values)
    if (auto substituted = substituteFrom(offset))
      return substituted;
  return std::nullopt;
}

void advanceGroupLanesThroughReRoots(FuncContractCache &contracts,
                                     mlir::func::FuncOp function,
                                     ResourceGroup &group,
                                     AliasAnalysis &aliases) {
  if (group.values.size() < 2 || !function || function.isDeclaration())
    return;
  // Escape hatch for A/B measurement only (both the insertion pass and the
  // verifier read it, so the two stay in agreement either way). Not a
  // supported configuration: with the advance off, a payload re-root loses the
  // entity again.
  static const bool disabled =
      std::getenv("LYTHON_OWNERSHIP_NO_LANE_ADVANCE") != nullptr;
  if (disabled)
    return;
  std::optional<mlir::DominanceInfo> dominance;

  // Bounded: each step replaces at least one lane with a value defined later,
  // so a fixpoint exists, but the cap keeps a pathological alias cycle from
  // spinning here.
  constexpr unsigned kMaxReRootSteps = 64;
  for (unsigned step = 0; step < kMaxReRootSteps; ++step) {
    mlir::func::CallOp reRoot;
    llvm::SmallVector<mlir::Value, 4> advanced;
    for (mlir::Value lane : llvm::drop_begin(group.values)) {
      for (mlir::Operation *user : lane.getUsers()) {
        auto call = mlir::dyn_cast<mlir::func::CallOp>(user);
        if (!call)
          continue;
        std::optional<llvm::SmallVector<mlir::Value, 4>> substituted =
            callReRootsGroupLanes(contracts, call, group.values, aliases);
        if (!substituted)
          continue;
        reRoot = call;
        advanced = std::move(*substituted);
        break;
      }
      if (reRoot)
        break;
    }
    if (!reRoot)
      return;

    // Every remaining use of the entity must be ordered against the re-root,
    // or the substituted lanes are not in scope where the release goes. This
    // is what keeps a rebind inside one arm of a branch out of the model
    // instead of producing a module the MLIR verifier rejects.
    if (!dominance)
      dominance.emplace(function);
    bool dominatesAllUses = true;
    for (mlir::Value lane : group.values)
      for (mlir::Operation *user : lane.getUsers())
        if (user != reRoot.getOperation() &&
            !dominance->properlyDominates(reRoot.getOperation(), user) &&
            !dominance->properlyDominates(user, reRoot.getOperation()))
          dominatesAllUses = false;
    if (!dominatesAllUses)
      return;

    group.values = std::move(advanced);
  }
}

bool isBlockArgMergeBorrowRetain(mlir::func::CallOp call) {
  auto label = call->getAttrOfType<mlir::StringAttr>(kAggregateRetainAttr);
  return label && label.getValue() == kBlockArgMergeBorrowLabel;
}

bool groupContainsOperand(mlir::Operation *op,
                          llvm::ArrayRef<mlir::Value> group,
                          AliasAnalysis &aliases) {
  for (mlir::Value operand : op->getOperands())
    for (mlir::Value value : group)
      if (aliases.same(operand, value))
        return true;
  return false;
}

llvm::SmallVector<mlir::Value, 4> remapGroupThroughValueMapping(
    mlir::ValueRange sources, mlir::ValueRange targets,
    llvm::ArrayRef<mlir::Value> group, AliasAnalysis &aliases,
    llvm::SmallVectorImpl<bool> *mappedMask) {
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

mlir::Operation *ancestorInBlock(mlir::Operation *op, mlir::Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op && op->getBlock() == block ? op : nullptr;
}

bool sameValueGroup(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left != right)
      return false;
  return true;
}

mlir::Value entityRootOf(llvm::ArrayRef<mlir::Value> group) {
  if (group.empty())
    return {};
  return underlyingObjectValue(group.front());
}

bool sameEntityRoot(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  mlir::Value left = entityRootOf(lhs);
  mlir::Value right = entityRootOf(rhs);
  // An empty group has no entity, so it is not the same entity as anything --
  // including another empty group. Comparing them equal would let a group the
  // caller failed to resolve absorb every other group at a dedup site.
  return left && right && left == right;
}

llvm::hash_code entityRootHash(llvm::ArrayRef<mlir::Value> group) {
  return llvm::hash_value(entityRootOf(group).getAsOpaquePointer());
}

void reportEntityRootParity(llvm::StringRef site,
                            llvm::ArrayRef<mlir::Value> lhs,
                            llvm::ArrayRef<mlir::Value> rhs) {
  enum class Mode { Off, Log, Abort };
  static const Mode mode = [] {
    const char *setting = std::getenv("LYTHON_OWNERSHIP_ROOT_PARITY");
    if (!setting || !*setting || llvm::StringRef(setting) == "0")
      return Mode::Off;
    return llvm::StringRef(setting) == "abort" ? Mode::Abort : Mode::Log;
  }();
  if (mode == Mode::Off)
    return;

  bool byRoot = sameEntityRoot(lhs, rhs);
  bool byLanes = sameValueGroup(lhs, rhs);
  if (byRoot == byLanes)
    return;

  llvm::errs() << "lython: ownership root parity divergence at " << site
               << ": same-root=" << byRoot << " same-lanes=" << byLanes
               << " (" << lhs.size() << " vs " << rhs.size() << " lanes)\n";
  if (mode == Mode::Abort)
    llvm::report_fatal_error("ownership root parity divergence");
}

} // namespace py::ownership
