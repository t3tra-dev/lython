#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
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

// WHICH REFERENCE THIS RELEASE DISCHARGES: the one its own operands name, and no
// other on the same entity.
//
// The same "an entity is not a resource" gap as `unwindTracksMintedTokensSeparately`
// below, one step further in. `aggregate_release` lets a minted token disown the
// CONTAINER's discharge, which is the only foreign release the walks could name;
// a plain release stayed unattributable, so whichever reference was walked second
// read the first one's release as its own and placed none. One discharge, two
// references, an entity stuck at refcount one -- `for ch in s: ys = [ch]; out =
// out + ys[0]` leaks its exhausted-iteration element exactly that way.
//
// The refcount-insertion pass knows the answer: it chose the operands from the
// reference it was placing for. This attribute is it saying so, and it is a
// CLAIM ABOUT THE OPERANDS rather than an opaque tag -- which is what lets the
// placer and the affine verifier read it with one rule: a labelled release none
// of whose operands is one of my names discharges someone else's reference.
// Insert and verify must not drift here, or the proof is void
// (rfc/memory-safety-proof.md).
//
// Why NOT widen who may skip `kAggregateReleaseAttr` instead: that label makes a
// DIFFERENT claim, and it is foreign only to a retain-MINTED token. A
// TRANSFERRED source has it as its only death -- a module global's initializer
// hands its construction straight to the global, so the global's teardown
// release IS that construction's discharge -- and letting an owned call result
// skip one double-freed every enum member in cross_enum_generic_handler.
//
// ⭐ NOT SUBSUMED BY `own::ReferenceMap`, measured 2026-08-05. The map can name
// the reference an operand denotes, but this label says something stronger: THE
// PLACER CHOSE THESE OPERANDS FROM THE REFERENCE IT WAS PLACING FOR. An
// emitter-written decref is unlabelled and its operand's reference is nameable
// all the same, and it may discharge something else. Dropping the label and
// going by the map alone makes
// `loop_iterator_element_into_container_literal` unbuildable.
//
// Only placements for a reference of their OWN carry it. A marker that
// republishes a reference the frame already holds has no increment to discharge,
// and labelling its release would make the holder disown it and place a second.
// An unlabelled release still reads as "could be mine", the pre-existing answer
// and the safe direction.
//
// LYTHON_ABLATE_REFERENCE_RELEASE=1 removes the label and every reader of it.
// Failure direction: the leaks above, never a double free.
inline constexpr llvm::StringLiteral kReferenceReleaseAttr{
    "ly.ownership.reference_release"};
bool perReferenceReleaseLabels();

// Does the unwind-cleanup analysis separate a retain-MINTED owned-local token
// from the reference it was minted on, instead of treating one entity as one
// resource?
//
// `LYTHON_ABLATE_UNWIND_MINTED_TOKENS=1` restores the entity-wide reading. One
// binary, two arms: a rebuild of the same source does not reproduce byte for
// byte, so "the shas differ" never establishes that two arms differ.
//
// Failure direction, RE-MEASURED 2026-08-05 because the original claim went
// stale: the ablated arm now REFUSES `cross_float_range_contracts_fields`
// rather than leaking its 80 B. `own::ReferenceMap` is not behind this switch,
// so the verifier still sees the reference the ablated analysis stops writing a
// cleanup for, and says so. `dict_iteration_views`, which this switch was
// introduced against, no longer moves at all -- three later rules prevent that
// leak independently. A refusal is the safer direction, but the arm is for
// bisecting a regression to this rule, never for production.
bool unwindTracksMintedTokensSeparately();
// The `parent` half of the kernel's `aggregate(parent, path)` resource, spelled
// so an ownership walk can name it. `kAggregateIdAttr` is an i64 identity on the
// op that PRODUCES a container; `kAggregateParentAttr` carries the same number
// on each slot-absorption retain that stored into that container, so the walk
// can charge the retain to the holder and discharge it when the holder is
// released. Ids are allocated from `kAggregateIdNextAttr` on the enclosing
// function and are unique within it.
//
// Why NOT reuse the slot label already on the retain (`"builtins.str:
// sequence.literal"`): that names the `path`, and `path` alone cannot answer
// `aggregate(parent, path)` -- two containers built at the same source position
// on two loop iterations share it. The whole point of the pair is that the
// obligation belongs to a specific parent, and the kernel's PathIsHeap corollary
// is that the effect is observed THROUGH that parent.
//
// Why NOT an SSA operand on the retain call instead of an attribute: the retain
// primitive's arity is fixed by the runtime manifest, and widening it would
// change a published contract to carry information only the verifier reads.
inline constexpr llvm::StringLiteral kAggregateIdAttr{
    "ly.ownership.aggregate_id"};
inline constexpr llvm::StringLiteral kAggregateParentAttr{
    "ly.ownership.aggregate_parent"};
inline constexpr llvm::StringLiteral kAggregateIdNextAttr{
    "ly.ownership.aggregate_id_next"};
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

// The contract name of `contract.ownedResults.values[contractIndex]`, or empty
// when the declaration names none. `contractIndex` indexes the OWNED-RESULT
// list, not the result list.
//
// One function rather than the precedence spelled at each reader: a declaration
// that yielded one name to the insertion pass and another to a verifier is the
// defect this replaces (`ABI/HandleWidthRegistry.h`, GAP 2 -- the insertion path
// had the `ly.runtime.result_contract` fallback and three helpers did not, so the
// same `func.func` produced two different answers depending on who asked).
// Adding the fallback at each site again would restore the divergence the next
// time a channel is added; the precedence has to have exactly one definition.
llvm::StringRef ownedResultContractName(mlir::func::FuncOp function,
                                        const FunctionContract &contract,
                                        unsigned contractIndex);

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
// Can a value of type `from` be spelled as the object-header interface type
// `to` (Ly_IncRef's input, a deallocator's input)? `spellHeaderPrefix` emits
// that spelling and returns null exactly when this returns false.
//
// The two are a pair on purpose. Asking "is this castable" and then emitting
// the cast are the same question, and a caller that asks the narrow version
// (memref.cast alone) accepts a candidate whose retain it then cannot write --
// which drops a retain the soundness argument already counted on and
// over-releases. A handle wider than the interface is the ordinary case for any
// contract holding its interior state behind the handle, so the prefix view is
// part of the answer, not a fallback.
bool canSpellHeaderPrefix(mlir::Type from, mlir::Type to);
mlir::Value spellHeaderPrefix(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value header, mlir::Type target);
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
  // The entity's root: the ONE stable SSA name this resource is tracked by.
  // Each producer names it -- a manifest allocation names its result, an
  // owned-local-object marker names result 0, a call names the head of its
  // owned result range -- and it is what group identity compares.
  mlir::Value root;
  // The entity's CURRENT physical lanes (the release interface the
  // deallocator takes), derived from `root`. Not the identity: a payload
  // re-root (field rebind, container growth, realloc) replaces lanes while
  // leaving the root alone, and a key that included them would lose the
  // entity exactly there.
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
// be a use-after-free). Primitives the manifest marks kManifestInteriorWordAttr
// reach the same storage through a call rather than a load, and their results
// join the walk on the same footing.
void collectBoxWordDerivedViews(llvm::ArrayRef<mlir::Value> groupValues,
                                llvm::SmallVectorImpl<mlir::Value> &views);

llvm::SmallVector<ResourceGroup, 8>
collectRuntimeResourceGroups(mlir::ValueRange values,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators);
llvm::SmallVector<ResourceGroup, 4>
collectOwnedLocalObjectGroups(mlir::Operation *op,
                              llvm::ArrayRef<RuntimeDeallocator> deallocators);
class AliasAnalysis;
// Does this owned-local marker root a token a retain just MINTED, as opposed to
// republishing one the head already had? The two kinds sit behind one attribute
// and need opposite answers about the head's other releases, so both the
// release placer and the affine verifier ask this one predicate -- if insertion
// and verification disagreed about which token a release discharges, the proof
// would be void (rfc/memory-safety-proof.md).
bool ownedLocalMarkerIsRetainRooted(mlir::Operation *marker,
                                    AliasAnalysis &aliases);
// `symbols`, when non-null, must be a symbol table over `module`; it only
// short-circuits the callee lookup (a module-symbol-list walk otherwise) for
// callers that sweep every call op and can build the table once.
llvm::SmallVector<ResourceGroup, 8>
collectOwnedCallResultGroups(mlir::ModuleOp module, mlir::func::CallOp call,
                             llvm::ArrayRef<RuntimeDeallocator> deallocators,
                             mlir::SymbolTable *symbols = nullptr);

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
// A mutation primitive that RE-ROOTS a payload sub-range of `group` rather
// than consuming the entity: its `transfer_args` names a lane at index >= 1
// (never the root) and its `owned_results` hands back replacements of the same
// types. That is how the current ABI spells in-place mutation -- "consume this
// container and return another one" (`LyList_EnsureCapacity`,
// `LyDict_SetItemBox`, ...) -- so a field or item sub-range mutated this way
// moves to new SSA names while the entity it belongs to is unchanged.
//
// Returns `group` with the sub-range replaced by the call's owned results;
// nullopt when the call is not such a re-root. The root (lane 0) is never
// substituted: a call that consumes THAT is consuming the entity, which the
// consume predicates above already decide.
std::optional<llvm::SmallVector<mlir::Value, 4>>
callReRootsGroupLanes(FuncContractCache &contracts, mlir::func::CallOp call,
                      llvm::ArrayRef<mlir::Value> group,
                      AliasAnalysis &aliases);
// Advance `group.values` from the birth expansion to the entity's CURRENT
// lanes by following those re-roots forward from the producer. The root is
// never touched, so the entity's identity is unchanged; only the lanes the
// deallocator will be handed, and the uses that decide where it is called,
// move to the post-mutation names.
//
// Bails (leaving the lanes alone) whenever a re-root does not dominate every
// remaining use of the entity: replacing a lane with a value defined in only
// one arm would make the release operand fail MLIR's dominance check, and an
// invalid module is worse than the conservative placement.
//
// `function` is only used to build dominance, and only once a candidate
// re-root has actually been found: DominanceInfo construction is linear in the
// function, so building it for every group would put back the quadratic term
// the path-sensitive rework removed (docs/ownership-perf.md). It is also not
// cached across groups on purpose -- release insertion may split blocks, which
// invalidates the tree.
void advanceGroupLanesThroughReRoots(FuncContractCache &contracts,
                                     mlir::func::FuncOp function,
                                     ResourceGroup &group,
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

// The entity root a lane list is rooted at, normalized through the
// identity-cast markers. Every producer lays an entity's lanes out with the
// release-interface head first and every deallocator names operand 0 as the
// released resource, so lane 0 is the root; stripping identity casts makes it
// survive a re-root, which republishes the SAME head through a fresh cast.
mlir::Value entityRootOf(llvm::ArrayRef<mlir::Value> group);
// Group identity. Two lane lists name the same entity iff they share a root,
// regardless of whether their payload lanes still agree.
bool sameEntityRoot(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs);
// Hash matching `sameEntityRoot`, for keying visited-state maps by entity.
llvm::hash_code entityRootHash(llvm::ArrayRef<mlir::Value> group);
// Migration probe: report where the entity-root key and the old full-lane key
// disagree. Set LYTHON_OWNERSHIP_ROOT_PARITY=1 to log each divergence, or
// =abort to stop at the first one.
//
// Why not a plain assert: a divergence is EXPECTED from the moment any
// producer re-roots payload lanes, which is precisely the case the migration
// exists to fix, so an unconditional assert would abort the suite on the
// inputs that matter instead of letting them be surveyed.
void reportEntityRootParity(llvm::StringRef site,
                            llvm::ArrayRef<mlir::Value> lhs,
                            llvm::ArrayRef<mlir::Value> rhs);

} // namespace py::ownership
