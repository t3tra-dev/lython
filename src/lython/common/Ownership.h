#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>
#include <string>

namespace py::ownership {

inline constexpr llvm::StringLiteral kOwnedResultsAttr{
    "ly.ownership.owned_results"};
inline constexpr llvm::StringLiteral kOwnedResultContractsAttr{
    "ly.ownership.owned_result_contracts"};
inline constexpr llvm::StringLiteral kBorrowedResultsAttr{
    "ly.ownership.borrowed_results"};
inline constexpr llvm::StringLiteral kRetainArgsAttr{
    "ly.ownership.retain_args"};
inline constexpr llvm::StringLiteral kReleaseArgsAttr{
    "ly.ownership.release_args"};
inline constexpr llvm::StringLiteral kTransferArgsAttr{
    "ly.ownership.transfer_args"};
inline constexpr llvm::StringLiteral kObjectHeaderAttr{
    "ly.ownership.object_header"};
inline constexpr llvm::StringLiteral kOwnedLocalObjectAttr{
    "ly.ownership.owned_local_object"};
inline constexpr llvm::StringLiteral kOwnedLocalObjectContractAttr{
    "ly.ownership.owned_local_object_contract"};
inline constexpr llvm::StringLiteral kObjectReleaseToZeroAttr{
    "ly.ownership.object_release_to_zero"};
inline constexpr llvm::StringLiteral kAggregateRetainAttr{
    "ly.ownership.aggregate_retain"};
inline constexpr llvm::StringLiteral kAggregateReleaseAttr{
    "ly.ownership.aggregate_release"};
// aggregate_retain label for the borrow-edge retains inserted at block-arg
// merges (identity edges of replacement/mutation merges): the retain lends
// the merge argument a token and is cancelled by the paired decref of the
// pre-merge name (loop back-edge decref-on-replace or exit release).
// Exceptional successor edges for the setjmp-style EH model: blocks that
// contain `LyEH_TryCallSiteMarker(id)` may transfer control to the handler
// entry (the true successor of the `LyEH_TryCatchAnchor(id)` cond_br). Any
// liveness or path walk that follows only CFG successors mis-models values
// the handler still uses after a partial try execution.
llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
collectExceptionEdges(mlir::Region &region);

// Handler-entry blocks by try id: the block containing
// `LyEH_TryCatchMarker(id)`, exactly the catch target the final LLVM EH
// phase wires unwinding invokes to. Shared by exception-edge collection,
// the unwind-cleanup insertion, and the affine verifier so all three agree
// on where an unwinding call site transfers control.
llvm::DenseMap<std::int64_t, mlir::Block *>
collectExceptionHandlerEntries(mlir::Region &region);

// The i64 constant id of an EH marker/anchor call, when it has one.
std::optional<std::int64_t> exceptionMarkerId(mlir::func::CallOp call);

// The call whose unwind a `LyEH_TryCallSiteMarker` guards: the first
// following non-marker call in the block (mirrors the marker/invoke pairing
// the final LLVM EH phase performs). Null when the block ends, another
// EH anchor/catch marker intervenes, or no call follows.
mlir::func::CallOp guardedCallAfterMarker(mlir::Operation *marker);

// The `LyEH_TryCallSiteMarker` pairing with `call`, i.e. the preceding
// marker with no other call in between. Null for unguarded calls: their
// unwind leaves the function instead of reaching an in-function handler.
mlir::func::CallOp precedingTryCallSiteMarker(mlir::Operation *call);

// True for runtime raise primitives (manifest `ly.runtime.primitive =
// "raise"` contract): calling one transfers control out of the function
// unless a preceding try call-site marker wires it to a local handler.
bool isRaisePrimitiveFunction(mlir::func::FuncOp function);

// Raise primitives PLUS the contract-less lowering support raises
// (`LyEH_RethrowCurrent`, `LyEH_ThrowException`): every call that never
// returns and always unwinds. Ownership walks must treat these uniformly --
// modeling only manifest raise primitives left rethrows (finally re-raise,
// bare `raise`) outside the unwind-cleanup model entirely.
bool isRaiseLikeFunction(mlir::func::FuncOp function);

// May a call to `function` unwind with a Python exception? Func-dialect
// mirror of the final EH phase's classification (Cleanup/EH.cpp): raise-like
// calls, Python-level callables, and runtime `Ly*` entry points minus the
// known non-raising EH/refcount/traceback bookkeeping. Used to model the
// unwind-out edge of calls in frames WITHOUT a local handler; an unmarked
// may-raise call in such a frame exits the function with every held token.
bool mayRaisePythonException(mlir::func::FuncOp function);

// For `%c = call @LyEH_TryCatchAnchor(id); cf.cond_br %c, ^handler, ^tail`
// where ^tail leads with a same-id call-site marker: the marker's guarded
// call. The anchor's true edge is never taken at runtime -- control reaches
// ^handler only by unwinding OUT OF the guarded call -- so a path walk of
// the true edge must apply the guarded call's consume effects to mirror the
// state the real unwind edge carries. Null when the pattern does not match.
mlir::func::CallOp anchorTrueEdgeGuardedCall(mlir::Operation *terminator);

inline constexpr llvm::StringLiteral kBlockArgMergeBorrowLabel{
    "block-arg-merge-borrow"};

inline constexpr llvm::StringLiteral kAtomicRoleAttr{"ly.atomic.role"};
inline constexpr llvm::StringLiteral kAtomicOrderingAttr{"ly.atomic.ordering"};
inline constexpr llvm::StringLiteral kAtomicRetainPremiseAttr{
    "ly.atomic.retain_premise"};

inline constexpr llvm::StringLiteral kCallableTypeAttr{"callable_type"};

// Module marker for the build-time pre-lowering of runtime-internal Python
// modules (runtime/lib/*.py compiled by RuntimePyLowering). Their artifacts
// are linked into the final LLVM module AFTER the EH phase that wires and
// erases `LyEH_Try*` markers, so no marker-based unwind cleanup can be
// materialized for them: insertion skips the wiring and the verifier skips
// the may-raise unwind-exit model in such modules (documented residual --
// leftover markers are rejected loudly at packaging instead).
inline constexpr llvm::StringLiteral kRuntimeInternalLoweringAttr{
    "ly.lowering.runtime_internal"};

struct IndexSet {
  llvm::SmallVector<unsigned, 8> values;

  bool empty() const { return values.empty(); }
  bool contains(unsigned index) const;
};

struct FunctionContract {
  IndexSet ownedResults;
  llvm::SmallVector<std::string, 8> ownedResultContracts;
  IndexSet borrowedResults;
  IndexSet retainArgs;
  IndexSet releaseArgs;
  IndexSet transferArgs;
  bool objectReleaseToZero = false;

  bool hasAnyOwnershipAttr() const;
  bool consumesArg(unsigned index) const;
};

enum class AggregateOwnershipAction { Retain, Release };

struct AggregateOwnershipMarker {
  AggregateOwnershipAction action = AggregateOwnershipAction::Retain;
  std::string slot;
};

mlir::FailureOr<IndexSet>
parseIndexSetAttr(mlir::Operation *op, llvm::StringRef attrName,
                  std::optional<unsigned> upperBound = std::nullopt);

mlir::FailureOr<FunctionContract>
readFunctionContract(mlir::func::FuncOp function);
mlir::FailureOr<std::optional<AggregateOwnershipMarker>>
readAggregateOwnershipMarker(mlir::Operation *op);

bool isRuntimeManifestFunction(mlir::func::FuncOp function);
bool functionUsesOwnedReturnABI(mlir::func::FuncOp function);
bool functionOwnsResultAt(mlir::func::FuncOp function, unsigned resultIndex);
bool functionConsumesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex);
bool functionReleasesOperandAt(mlir::func::FuncOp function,
                               unsigned operandIndex);
bool functionRetainsOperandAt(mlir::func::FuncOp function,
                              unsigned operandIndex);

struct RuntimeDeallocator {
  mlir::func::FuncOp function;
  std::string contractName;
  llvm::SmallVector<mlir::Type, 4> inputTypes;
  // Canonical value shape of the contract (from its ly.runtime.shape
  // declaration when present). The release interface (inputTypes) is a
  // prefix of this — usually just the entity root — while the remaining
  // values are interior views whose USES still pin the entity's liveness.
  llvm::SmallVector<mlir::Type, 4> shapeTypes;
  FunctionContract contract;
};

llvm::SmallVector<RuntimeDeallocator, 8>
collectRuntimeDeallocators(mlir::ModuleOp module);

bool valueRangeMatchesTypes(mlir::ValueRange values, unsigned offset,
                            llvm::ArrayRef<mlir::Type> types);
bool isObjectHeaderLikeType(mlir::Type type);
// Strips identity-shaped unrealized-cast markers (owned-local-object rooting
// and similar value-group markers keep types and arity) so SSA-identity
// comparisons see the underlying value regardless of ownership rewrapping.
mlir::Value underlyingObjectValue(mlir::Value value);
const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);
const RuntimeDeallocator *
findDeallocatorForValueGroup(mlir::ValueRange values, unsigned offset,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             llvm::StringRef contractName);
llvm::SmallVector<mlir::Value, 4> valueSlice(mlir::ValueRange values,
                                             unsigned offset, unsigned size);
bool valueGroupEqualsEntryArgumentGroup(mlir::func::FuncOp function,
                                        llvm::ArrayRef<mlir::Value> group);
bool callResultGroupIsOwned(mlir::func::FuncOp callee, unsigned resultIndex);

enum class OwnershipKind { NonObject, Borrow, Own, Immortal };

llvm::StringRef ownershipKindName(OwnershipKind kind);
bool ownershipKindCarriesObjectResource(OwnershipKind kind);
OwnershipKind logicalOwnershipKind(mlir::Type logicalType, bool ownsObject);

struct OwnershipCondition {
  mlir::Value tag;
  std::int64_t activeTag = -1;
  unsigned memberCount = 0;
};

struct OwnershipConditionBranch {
  unsigned activeSuccessor = 0;
  unsigned inactiveSuccessor = 0;
};

std::optional<bool>
conditionTrueMeansActive(mlir::Value condition,
                         const OwnershipCondition &ownershipCondition);
std::optional<OwnershipConditionBranch>
classifyOwnershipConditionBranch(mlir::Operation *op,
                                 const OwnershipCondition &condition);

struct ResourceGroup {
  unsigned offset = 0;
  OwnershipKind ownership = OwnershipKind::Own;
  llvm::SmallVector<mlir::Value, 4> values;
  // Interior views of the same entity (the canonical-shape tail beyond the
  // release interface). Uses of these keep the entity live; they are not
  // release operands.
  llvm::SmallVector<mlir::Value, 4> views;
  const RuntimeDeallocator *deallocator = nullptr;
  std::optional<OwnershipCondition> condition;
};

class AliasAnalysis;

// Owned-return ABI walking, shared by the refcount-insertion pass and the
// affine verifier (one implementation: a divergence between the two caused
// real bugs).
struct OwnedReturnRange {
  unsigned offset = 0;
  unsigned size = 0;
  mlir::Type type;
};

bool groupMatchesValues(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<mlir::Value> group,
                        AliasAnalysis &aliases);
std::optional<unsigned>
logicalReturnValueCount(mlir::ValueRange values, unsigned offset,
                        llvm::ArrayRef<RuntimeDeallocator> deallocators,
                        mlir::Type type);
unsigned skipPrimitiveReturnEvidence(mlir::ValueRange values, unsigned offset,
                                     mlir::Type type);
std::optional<llvm::SmallVector<OwnedReturnRange, 4>>
callableOwnedReturnRanges(mlir::func::FuncOp function, mlir::ValueRange values,
                          llvm::ArrayRef<RuntimeDeallocator> deallocators);
bool groupMatchesOwnedReturnRange(
    mlir::ValueRange values, const OwnedReturnRange &range,
    llvm::ArrayRef<mlir::Value> group,
    llvm::ArrayRef<RuntimeDeallocator> deallocators, AliasAnalysis &aliases);

// Box-word reconstructions are borrowed interior views: a memref descriptor
// assembled from an entity's box words (memref.load -> llvm.inttoptr ->
// llvm.insertvalue... -> unrealized cast to memref) aliases the entity's
// storage without any direct SSA use of the entity past the load. Collect
// those derived view values so release placement pins the entity until the
// views' last use (a release between the load and the consuming call would
// be a use-after-free).
void collectBoxWordDerivedViews(llvm::ArrayRef<mlir::Value> groupValues,
                                llvm::SmallVectorImpl<mlir::Value> &views);

llvm::SmallVector<ResourceGroup, 8>
collectRuntimeResourceGroups(mlir::ValueRange values,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);
llvm::SmallVector<ResourceGroup, 4>
collectOwnedLocalObjectGroups(mlir::Operation *op,
                              llvm::ArrayRef<RuntimeDeallocator> deallocators);
llvm::SmallVector<ResourceGroup, 8>
collectOwnedCallResultGroups(mlir::ModuleOp module, mlir::func::CallOp call,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);

class AliasAnalysis {
public:
  void build(mlir::Operation *root);
  void track(mlir::Value value);
  mlir::Value find(mlir::Value value);
  bool same(mlir::Value lhs, mlir::Value rhs);
  void unionValues(mlir::Value lhs, mlir::Value rhs);
  void aliasesOf(mlir::Value value,
                 llvm::SmallVectorImpl<mlir::Value> &aliases);

private:
  void invalidateAliasBuckets();
  void rebuildAliasBuckets();

  llvm::DenseMap<mlir::Value, mlir::Value> parent;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Value, 8>> aliasBuckets;
  bool aliasBucketsDirty = true;
};

// Group predicates shared by release insertion (lowering) and the affine
// ownership verifier — like the owned-return walking above, these must have
// exactly one implementation or insert/verify silently drift apart.

struct CachedFuncContract {
  mlir::func::FuncOp function;
  FunctionContract contract;
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

    auto contract = readFunctionContract(function->second);
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

bool returnTransfersGroup(FuncContractCache &contracts,
                          mlir::func::FuncOp function,
                          mlir::func::ReturnOp returnOp,
                          llvm::ArrayRef<mlir::Value> group,
                          llvm::ArrayRef<RuntimeDeallocator> deallocators,
                          AliasAnalysis &aliases);
bool callConsumesGroup(FuncContractCache &contracts, mlir::func::CallOp call,
                       llvm::ArrayRef<mlir::Value> group,
                       AliasAnalysis &aliases);
bool callRetainsGroup(FuncContractCache &contracts, mlir::func::CallOp call,
                      llvm::ArrayRef<mlir::Value> group,
                      AliasAnalysis &aliases);
bool callPartiallyConsumesGroup(FuncContractCache &contracts,
                                mlir::func::CallOp call,
                                llvm::ArrayRef<mlir::Value> group,
                                AliasAnalysis &aliases);
// Identity merge edges lend the merge argument a token via a retain labeled
// kBlockArgMergeBorrowLabel; the paired release targets the pre-merge name.
bool isBlockArgMergeBorrowRetain(mlir::func::CallOp call);
bool groupContainsOperand(mlir::Operation *op,
                          llvm::ArrayRef<mlir::Value> group,
                          AliasAnalysis &aliases);
llvm::SmallVector<mlir::Value, 4> remapGroupThroughValueMapping(
    mlir::ValueRange sources, mlir::ValueRange targets,
    llvm::ArrayRef<mlir::Value> group, AliasAnalysis &aliases,
    llvm::SmallVectorImpl<bool> *mappedMask = nullptr);
mlir::Operation *ancestorInBlock(mlir::Operation *op, mlir::Block *block);
bool sameValueGroup(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs);

} // namespace py::ownership
