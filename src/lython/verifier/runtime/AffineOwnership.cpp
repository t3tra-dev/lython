#include "runtime/Detail.h"

#include "Common/Instrumentation.h"
#include "Common/PythonSourceRange.h"

#include "Contracts.h"
#include "Ownership.h"
#include "Reference.h"

#include "PyDialectTypes.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Process.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <memory>
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

// Env-gated trace of the path walk's STALE set: which name a rejected
// release/transfer named, and where that name is defined.
//
// Why NOT read it off the diagnostic instead: the diagnostic names the PRODUCER
// of the tracked resource, which for a construction inside a loop body is the
// same symbol on every iteration. It cannot separate "this program really
// consumes a moved token" from "the walk carried a stale name across a back edge
// past the op that rebinds it", and those two want opposite repairs.
bool ownershipStaleTraceEnabled() {
  static const bool enabled =
      std::getenv("LYTHON_OWNERSHIP_TRACE_STALE") != nullptr;
  return enabled;
}

// LYTHON_OWNERSHIP_TRACE_PATH: print the CFG path that reached a rejected
// double consume, as `^bbN>` ordinals matching a LYTHON_IR_DUMP listing.
//
// Why NOT read it off the diagnostic: the message names the producer and the
// releasing call, and both are the same symbols on every path through a loop.
// It cannot distinguish "this program releases twice" from "two cleanup
// handlers chained and each released once", which want opposite repairs -- and
// that distinction is what this trace decided for the re-raise-in-a-loop
// rejection (`^bb26>^bb37>^bb35`: a cleanup handler entered from another
// cleanup handler).
bool ownershipPathTraceEnabled() {
  static const bool enabled =
      std::getenv("LYTHON_OWNERSHIP_TRACE_PATH") != nullptr;
  return enabled;
}

// A/B escape hatch for the rebind rule below, in the same spirit as
// LYTHON_OWNERSHIP_NO_LANE_ADVANCE: it lets `redcheck.py --sentinel` take its
// "before" side from the SAME binary as its "after" side, so a GREEN verdict
// cannot come from a build pointed at the wrong tree. Not a supported
// configuration -- with it set, an ordinary rebind inside a loop is refused.
bool staleRebindDropDisabled() {
  static const bool disabled =
      std::getenv("LYTHON_ABLATE_STALE_REBIND") != nullptr;
  return disabled;
}


// LYTHON_ABLATE_RELEASED_DOMINANCE=1 restores the use-after-release rule that
// fired on any op mentioning the group's ALIAS class, whether or not the
// producer dominates it. Same purpose as the hatch above: it lets a sentinel and
// an IR differential take their "before" side from this binary. Not a supported
// configuration -- with it set, an owned local object built inside a loop body is
// refused, because the next iteration's construction of a FRESH instance reads as
// a use of the released one.
bool releasedUseDominanceDisabled() {
  static const bool disabled =
      std::getenv("LYTHON_ABLATE_RELEASED_DOMINANCE") != nullptr;
  return disabled;
}

// `^bbN` as the IR printer numbers it, so a trace line can be matched against a
// LYTHON_IR_DUMP listing by eye.
unsigned blockOrdinal(mlir::Block *block) {
  if (!block || !block->getParent())
    return 0;
  unsigned index = 0;
  for (mlir::Block &candidate : *block->getParent()) {
    if (&candidate == block)
      break;
    ++index;
  }
  return index;
}

void traceStaleValue(llvm::StringRef event, mlir::Value value,
                     mlir::Block *pathBlock) {
  if (!ownershipStaleTraceEnabled())
    return;
  llvm::errs() << "[ownership-stale] " << event << ": ";
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value))
    llvm::errs() << "block argument " << argument.getArgNumber() << " of ^bb"
                 << blockOrdinal(argument.getOwner());
  else if (mlir::Operation *definition = value.getDefiningOp()) {
    llvm::errs() << definition->getName();
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(definition))
      llvm::errs() << " @" << call.getCallee();
    llvm::errs() << " in ^bb" << blockOrdinal(definition->getBlock());
  } else {
    llvm::errs() << "<unknown>";
  }
  llvm::errs() << ", path at ^bb" << blockOrdinal(pathBlock) << "\n";
}

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
      // The shared precedence, not a local copy of it: when this read had only
      // `owned_result_contracts` and the insertion path also had the
      // `ly.runtime.result_contract` fallback, a declaration carrying just the
      // latter produced a deallocator for the inserter and `nullptr` here. Both
      // callers of this predicate turn `false` into a diagnostic, so the
      // divergence cost a REFUSAL of a program the inserter had balanced -- the
      // incomplete side, not a silently accepted double release.
      llvm::StringRef contractName = own::ownedResultContractName(
          function, *contract, static_cast<unsigned>(contractIndex));
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
  // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
  llvm::SmallVector<mlir::Value, 4> values;
  // Does this resource's producer MINT a token (a retain-rooted owned-local
  // marker) rather than republish one the head already had? If so the object
  // carries a second token under other names, whose releases this walk must not
  // count (`callReleasesForeignAggregate`). False for every other resource, which
  // leaves those walks unchanged.
  // Is this resource an increment of its OWN, so a release NAMING another
  // reference on the same entity cannot be its death? True for a retain-minted
  // marker and for an owned call result (one reference per owned result); false
  // for a marker that republishes a reference the frame already holds, which
  // shares that one and must keep reading its release as its own.
  //
  // Only `own::kReferenceReleaseAttr` is read through this. The aggregate label
  // keeps the narrower minted-only gate -- see
  // `callReleasesForeignAggregate`.
  bool ownsReference = false;
  // Does some call release this group under one of ITS OWN names? If it does,
  // an aliasing aggregate release is redundant credit and must not also be read
  // as this token's death; if it does not, that aggregate release is the only
  // release the token has and dropping it would report a leak that is not there.
  bool hasOwnNamedRelease = false;
  // Which increment this resource is the obligation for, when the analysis can
  // name one (`own::ReferenceMap`). The placer asks the same map about the same
  // groups; insert and verify disagreeing about which release discharges which
  // reference is what voids the proof.
  own::Reference reference;
  // Interior views of the same entity (canonical-shape tail beyond the
  // release interface): their uses are entity uses, never release operands.
  llvm::SmallVector<mlir::Value, 4> views;
  std::optional<own::OwnershipCondition> condition;
};

// IS THIS CONSUMING CALL SOMEBODY ELSE'S DISCHARGE RATHER THAN THIS TOKEN'S?
//
// TWO LABELS, TWO CLAIMS, AND THE GATES DIFFER.
//
// `aggregate_release` marks the discharge of an `aggregate(parent, path)` -- a
// slot's or a literal source's token, owned by the container
// (rfc/memory-safety-proof.md, Aggregates). It is foreign only to a
// retain-MINTED token: a TRANSFERRED source has it as its only death, since a
// module global's initializer hands its construction straight to the global and
// the global's teardown release IS that discharge. Letting an owned call result
// skip one double-freed every enum member in `cross_enum_generic_handler`.
//
// `own::kReferenceReleaseAttr` says "this discharges exactly the reference my
// operands name" -- a statement about the release rather than about the reader
// -- so any resource holding an increment of its own may act on it.
//
// Why a LABEL at all rather than the reference alone (measured, both directions
// on one day):
//
//   skip only labelled foreign releases -> 492/492
//   skip every foreign-named release    -> 20 refusals, and with the wider
//                                          placer rule that pairs with it,
//                                          `dict_methods_complete` aborted with
//                                          `Ly_DecRef observed non-positive
//                                          refcount`
//
// A BARE release names some local token and neither pass can tell which, so the
// only safe reading of one is "this token".
//
// The placer splits the two labels the same way and asks the same
// `own::ReferenceMap` about the names. A disagreement here would void the proof.

// Is this operand NOT one of the walked resource's names?
//
// The same condition the placer's `isNotOwnName` carries, and the same contract
// on "no claim": where `own::ReferenceMap` can name the value it answers by
// REFERENCE -- so a CAST of one of our names is recognised as ours, which the
// containment test below cannot -- and where it cannot, the answer is the
// previous reading, which HERE is containment. Insert and verify must agree
// about which release discharges which reference or the proof is void, so this
// is deliberately the same shape as the placer's.
bool operandIsNotOurs(
    const own::ReferenceMap &references,
    std::initializer_list<llvm::ArrayRef<mlir::Value>> ourNames,
    mlir::Value operand, own::Reference mine) {
  // BOTH sides have to be nameable for the map to answer. With `mine` null the
  // comparison says nothing, and returning "not ours" for every named operand is
  // how this first read `LyLong_FromI64`'s result as somebody else's in
  // stackguard_support.
  if (mine)
    if (own::Reference denoted = references.of(operand))
      return denoted != mine;
  for (llvm::ArrayRef<mlir::Value> names : ourNames)
    if (llvm::is_contained(names, operand))
      return false;
  return true;
}

bool callReleasesForeignAggregate(
    const own::ReferenceMap &references, bool hasOwnNamedRelease,
    bool ownsReference, own::Reference mine,
    std::initializer_list<llvm::ArrayRef<mlir::Value>> ourNames,
    mlir::func::CallOp call) {
  // `isMinted` rather than a bool recomputed per resource from
  // `getPrevNode()`: the same fact, from the map the placer asks.
  bool foreignAggregate = hasOwnNamedRelease && references.isMinted(mine) &&
                          call->hasAttr(own::kAggregateReleaseAttr);
  bool foreignReference = ownsReference &&
                          own::perReferenceReleaseLabels() &&
                          call->hasAttr(own::kReferenceReleaseAttr);
  if (!foreignAggregate && !foreignReference)
    return false;
  for (mlir::Value operand : call.getOperands())
    if (!operandIsNotOurs(references, ourNames, operand, mine))
      return false;
  return true;
}

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

// Only calls in the function's top-level region are modeled as unwind exits:
// markers/calls nested in single-block regions (scf.if arms etc.) cannot
// host the anchor/cond_br cleanup wiring the refcount-insertion pass emits,
// so their edges stay outside the model (known residual, see
// rfc/stdlib-semantics.md Wave 0 hand-off).
bool callInFunctionTopLevelRegion(mlir::Operation *op) {
  mlir::Region *region = op->getParentRegion();
  return region && region->getParentOp() &&
         mlir::isa<mlir::func::FuncOp>(region->getParentOp());
}

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
  // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
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
  // The resource's aliasing views (static-evidence lanes of the same object),
  // carried PER PATH because they are renamed across CFG edges exactly like
  // the group. Consulting the resource's fixed view list instead reported a
  // use of the CURRENT iteration's value as a use of the PREVIOUS iteration's
  // released token, once the group had been renamed onto a loop block
  // argument (`while True: i += 1 ... break; print(i)`).
  llvm::SmallVector<mlir::Value, 4> views;
  // Outstanding block-arg-merge-borrow retains (identity merge edges lend the
  // merge argument a token; the paired release targets the pre-merge name),
  // held as the RETAIN OPS and not as a count.
  //
  // ⭐ Bounded by construction, and it has to be: this is part of the
  // visited-state key, so anything that can grow once per trip round a loop
  // stops the fixpoint from ever closing. A count did exactly that. The
  // borrow at a loop's back edge retains a DERIVED view of the carried value
  // (`memref.cast` of a `memref.subview` of the frame), rebuilt as a fresh
  // SSA value every trip, so `callConsumesStaleValue` -- which cancels a
  // borrow by matching a release against the PRE-RENAME name -- can never
  // match it: there is no pre-rename name for a value that is recomputed.
  // Measured on a nested loop in a generator: 39 increments, 0 decrements,
  // all from ONE retain op, and `borrowed` reached 285,714 with the state cap
  // raised to 4,000,000.
  //
  // Re-executing the same static retain is the walk going round again, not a
  // second outstanding borrow, so the same op is recorded once. Distinct ops
  // still count separately, and the only question ever asked of this is
  // "is any borrow outstanding" -- see the single reader below.
  llvm::SmallVector<mlir::Operation *, 4> borrowedRetains;
  // Path entered through an exceptional (unwind) edge. The affine invariant
  // holds on these paths like any other (rfc/stdlib-semantics.md R2: unwind
  // cleanup is inserted, no leak is accepted); the flag only disambiguates
  // path-state dedup.
  bool exceptional = false;
  // Block ordinals visited on the way here, kept only when
  // LYTHON_OWNERSHIP_TRACE_PATH is set.
  //
  // Why a whole path and not the failing op: this walk's diagnostics name the
  // PRODUCER of a tracked resource, and a double-consume report has to be read
  // against the CFG to tell "the program releases twice" from "the walk crossed
  // an edge that renamed nothing". The re-raise-in-a-loop rejection fixed in
  // this file's sibling pass was attributed from one trail line: the path
  // entered a cleanup handler from ANOTHER cleanup handler, which is the shape
  // no per-op message can show.
  llvm::SmallVector<unsigned, 16> trail;
  // Slot-absorption retains that are currently PARKED IN A CONTAINER, listed by
  // the container's aggregate identity (own::kAggregateIdAttr) in the order
  // they were charged. This is the kernel's `aggregate(parent, path)` with the
  // question answered by `parent`: the token is still outstanding -- so it keeps
  // a read-back of the element legal -- but the obligation to discharge it
  // belongs to the holder, and releasing the holder discharges it here.
  //
  // Why NOT keep counting these in `retained`. `retained` is part of the
  // visited-state key and a container built inside a loop charges one retain per
  // iteration, so the key increases forever and the fixpoint never closes --
  // shipped, and it refused `s = "abc"` / `for k in range(3): t = (s,)` with
  // `ownership CFG exploration exceeded 20000 states`. Attributing the retain to
  // the holder bounds it instead: the same loop charges and discharges the same
  // identity every trip, so the second iteration reaches a state already visited.
  //
  // Why NOT drop the retain instead: the two walks track different resource
  // kinds and
  // the borrowed walk's exemption is not transferable. Reading an element back
  // out (`total += ys[0]`) hands the reader a token derived from the slot, and
  // this walk needs the retain to justify the reader's later release; dropping it
  // refused 39 golden cases that compile today.
  llvm::SmallVector<std::int64_t, 2> slotParents;
  // Of `retained`, the ones a slot-absorption retain parked in a container
  // whose identity this walk could not name. They belong to the holder, so the
  // owned-return rule must not read them as tokens the return failed to spend
  // -- the named ones are already out of `retained` and in `slotParents`.
  // Last on purpose: the positional aggregate initializers above stay valid.
  unsigned parkedUnnamed = 0;
  // The parked-without-a-name retains this path has already charged, by call
  // op. A retain whose holder cannot be named has no discharge this walk can
  // see -- the container's release names the CONTAINER -- so it stays in
  // `retained` for the whole path; charging it AGAIN each time the walk goes
  // round a cycle that contains it is what made `retained` climb without
  // bound.
  //
  // ⛔ NOT a claim that the retain runs once. It runs once per iteration and
  // so does the container's release; this walk has no iteration count for
  // either, and every other recurring charge it tracks is already deduped the
  // same way -- `slotParents` by holder identity, `borrowedRetains` by op. The
  // shape that needed it: a resource produced OUTSIDE a cycle and parked
  // INSIDE it, which is every `for i in ...: for j in ...: out.append(i)`.
  llvm::SmallVector<mlir::Operation *, 2> parkedOps;
};

struct BorrowedEntryResource {
  mlir::func::FuncOp function;
  unsigned logicalIndex = 0;
  unsigned inputOffset = 0;
  // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
  llvm::SmallVector<mlir::Value, 4> values;
};

struct BorrowedPathState {
  mlir::Block *block = nullptr;
  mlir::Operation *start = nullptr;
  unsigned retained = 0;
  // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
  llvm::SmallVector<mlir::Value, 4> group;
  // ⭐ Names the group HELD EARLIER on this path, most recent last. A
  // merge-borrow lend is taken on the pre-merge name and returned on it too
  // (the retain branch below already says so), while the group has moved on to
  // the merge argument by the time the release is reached -- so without these
  // the release is invisible and the balance climbs one lend per rename.
  //
  // Deliberately NOT part of the visited key, for the same reason
  // `AffinePathState::previous` is not: a path-dependent set in the key lets
  // nested loops defeat the dedup. Missing a state can only miss a detection.
  llvm::SmallVector<llvm::SmallVector<mlir::Value, 4>, 4> previousGroups;
  // Path entered through an exceptional (unwind) edge. Retain balance is
  // required on these paths like any other (rfc/stdlib-semantics.md R2).
  bool exceptional = false;
};

// How many pre-merge namings one path keeps. A loop renames the group once per
// iteration and the walk explores several, so this is bounded rather than
// complete: the oldest naming is dropped, which can only lose a release
// crediting (a missed detection), never invent one.
constexpr unsigned kMaxBorrowedPreviousGroups = 4;

bool samePathState(const AffinePathState &lhs, const AffinePathState &rhs) {
  // `stale`, `previous` and `views` are deliberately NOT compared: they only
  // refine detections, and including them lets path-dependent sets defeat the
  // visited dedup (nested loops explode). Skipping a state that differs only
  // there can only miss a detection, never accept unsound IR.
  // Groups are compared by ENTITY ROOT, not by lane list: a state whose
  // payload lanes were re-rooted since the last visit is the same entity at
  // the same program point, so treating it as new only re-walks a path
  // already covered. Coarsening the relation can merge states and therefore
  // miss a detection; it can never accept unsound IR (same reasoning as the
  // deliberately excluded fields below).
  own::reportEntityRootParity("samePathState", lhs.group, rhs.group);
  return lhs.block == rhs.block && lhs.start == rhs.start &&
         lhs.token == rhs.token && lhs.retained == rhs.retained &&
         lhs.parkedUnnamed == rhs.parkedUnnamed &&
         lhs.parkedOps == rhs.parkedOps &&
         lhs.borrowedRetains == rhs.borrowedRetains &&
         lhs.exceptional == rhs.exceptional &&
         lhs.slotParents == rhs.slotParents &&
         own::sameEntityRoot(lhs.group, rhs.group);
}

// Tokens that keep a RELEASED group alive on this path: the walk's own retains
// plus the ones parked in a container (`slotParents`). Every rule that asks
// "may this released group still be named here" must ask this and not
// `state.retained` alone -- a slot-absorption retain no longer lands in
// `retained`, and reading only `retained` at two of the region-terminator rules
// refused 23 golden cases that compile today.
unsigned outstandingTokens(const AffinePathState &state) {
  return state.retained + static_cast<unsigned>(state.slotParents.size());
}

bool sameBorrowedPathState(const BorrowedPathState &lhs,
                           const BorrowedPathState &rhs) {
  own::reportEntityRootParity("sameBorrowedPathState", lhs.group, rhs.group);
  return lhs.block == rhs.block && lhs.start == rhs.start &&
         lhs.retained == rhs.retained &&
         lhs.exceptional == rhs.exceptional &&
         own::sameEntityRoot(lhs.group, rhs.group);
}

// Hash over exactly the fields the `same*PathState` relations compare, so two
// states that hash apart are guaranteed distinct under those relations.
std::size_t dedupBucket(llvm::hash_code code) {
  // DenseMap reserves its two highest keys as the empty/tombstone markers, so
  // a raw hash may not be used as a key.
  return static_cast<std::size_t>(code) & ~(static_cast<std::size_t>(3) << 62);
}

std::size_t pathStateDedupKey(const AffinePathState &state) {
  llvm::hash_code code = llvm::hash_combine(
      state.block, state.start, static_cast<int>(state.token), state.retained,
      state.parkedUnnamed, state.exceptional,
      llvm::hash_combine_range(state.parkedOps.begin(), state.parkedOps.end()),
      llvm::hash_combine_range(state.borrowedRetains.begin(),
                               state.borrowedRetains.end()),
      llvm::hash_combine_range(state.slotParents.begin(),
                               state.slotParents.end()));
  // Hashing the whole lane list would break the equal-implies-same-hash
  // contract now that equality is by root: two equal states would land in
  // different buckets and the dedup would silently stop deduping.
  code = llvm::hash_combine(code, own::entityRootHash(state.group));
  return dedupBucket(code);
}

std::size_t borrowedStateDedupKey(const BorrowedPathState &state) {
  llvm::hash_code code = llvm::hash_combine(state.block, state.start,
                                            state.retained, state.exceptional);
  code = llvm::hash_combine(code, own::entityRootHash(state.group));
  return dedupBucket(code);
}

// Visited-state stores. Membership is decided by the same `samePathState` /
// `sameBorrowedPathState` predicates as before; only the candidate lookup
// changed. Why not the flat vector these replace: membership was a linear
// rescan of every state already visited, which made ONE resource's walk
// quadratic in its own state count -- up to ~2e8 state comparisons before the
// 20000-state cap fires -- and that scan, not the CFG traversal, was where the
// path-sensitive phase spent its time on a module with an imported stdlib.
// The states stay in one flat, insertion-ordered vector and the buckets hold
// indices into it, rather than the states themselves: a path state carries four
// small value vectors, so storing it inside a hash bucket pays the map's
// capacity slack on all of that.
class VisitedAffineStates {
public:
  bool contains(const AffinePathState &candidate) const {
    auto bucket = buckets.find(pathStateDedupKey(candidate));
    if (bucket == buckets.end())
      return false;
    return llvm::any_of(bucket->second, [&](unsigned index) {
      return samePathState(states[index], candidate);
    });
  }

  void insert(const AffinePathState &state) {
    buckets[pathStateDedupKey(state)].push_back(
        static_cast<unsigned>(states.size()));
    states.push_back(state);
  }

  unsigned size() const { return static_cast<unsigned>(states.size()); }

private:
  llvm::SmallVector<AffinePathState, 32> states;
  llvm::DenseMap<std::size_t, llvm::SmallVector<unsigned, 1>> buckets;
};

class VisitedBorrowedStates {
public:
  bool contains(const BorrowedPathState &candidate) const {
    auto bucket = buckets.find(borrowedStateDedupKey(candidate));
    if (bucket == buckets.end())
      return false;
    return llvm::any_of(bucket->second, [&](unsigned index) {
      return sameBorrowedPathState(states[index], candidate);
    });
  }

  void insert(const BorrowedPathState &state) {
    buckets[borrowedStateDedupKey(state)].push_back(
        static_cast<unsigned>(states.size()));
    states.push_back(state);
  }

  unsigned size() const { return static_cast<unsigned>(states.size()); }

private:
  llvm::SmallVector<BorrowedPathState, 32> states;
  llvm::DenseMap<std::size_t, llvm::SmallVector<unsigned, 1>> buckets;
};

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

// Per-function memo for the questions the path walk asks once per visited path
// state. Every answer here is a pure function of IR that this verifier only
// reads, so memoizing changes no judgment -- it changes the walk's cost from
// (states x ops x contract lookups) to (states x tracked names).
//
// Why the memo lives here and not next to the `own::` helpers it wraps: the
// refcount-insertion pass calls the same helpers while it REWRITES marker ids
// and adds cleanup blocks, so a cache owned by the helper would answer a later
// phase from an earlier phase's IR. A cache scoped to one function's
// verification cannot outlive the IR it describes.
class OwnershipWalkCache {
public:
  OwnershipWalkCache(mlir::func::FuncOp function, FuncContractCache &contracts,
                     own::AliasAnalysis &aliases,
                     const ExceptionHandlerMap &handlerEntries)
      : function(function), contracts(contracts), aliases(aliases),
        handlerEntries(handlerEntries) {
    if (function.isDeclaration())
      return;
    // "Which ops have a direct operand in this alias class" is the inverse of
    // the operand scan every group predicate performs. Building the inverse
    // once per function turns each of those predicates into a short lookup.
    //
    // Why a vector and not a set: there is one entry per alias class in the
    // whole function, and a hash set's bookkeeping plus inline slots cost more
    // than double a small vector's per entry -- at module scale that is
    // hundreds of megabytes for lists that almost always hold one or two ops.
    // Duplicates cannot accumulate without a membership test either: an op's
    // operands are visited consecutively, so a repeat is always the last entry.
    function.walk([&](mlir::Operation *op) {
      for (mlir::Value operand : op->getOperands()) {
        if (!operand)
          continue;
        auto &users = classUsers[aliases.find(operand)];
        if (users.empty() || users.back() != op)
          users.push_back(op);
      }
    });
  }

  // The shared precondition of `groupContainsOperand`, `groupMatchesValues`,
  // `callConsumesGroup`, `callRetainsGroup`, `callPartiallyConsumesGroup` and
  // `callConsumesStaleValue`: each of them can only answer `true` when some
  // operand of `op` aliases one of the tracked names. An EMPTY group reports as
  // mentioning, because `groupMatchesValues` matches an empty group vacuously
  // and filtering on it would answer differently from the predicate.
  bool mentionsTracked(mlir::Operation *op, const AffinePathState &state) {
    if (state.group.empty())
      return true;
    return mentionsAny(op, state.group) || mentionsAny(op, state.views) ||
           mentionsAny(op, state.stale) || mentionsAny(op, state.previous);
  }

  bool mentionsTracked(mlir::Operation *op, const BorrowedPathState &state) {
    if (state.group.empty())
      return true;
    return mentionsAny(op, state.group);
  }

  // Is there an op in `region` (nested regions included) with an operand
  // aliasing a value of `group`? Answered from the operand inverse instead of
  // walking the region: the region walk ran once per path state that reached
  // the region-carrying op.
  bool regionMentionsGroup(mlir::Region &region,
                           llvm::ArrayRef<mlir::Value> group) {
    for (mlir::Value value : group) {
      if (!value)
        continue;
      auto users = classUsers.find(aliases.find(value));
      if (users == classUsers.end())
        continue;
      for (mlir::Operation *user : users->second)
        if (region.isAncestor(user->getParentRegion()))
          return true;
    }
    return false;
  }

  mlir::Block *markerHandler(mlir::func::CallOp marker) {
    auto [entry, inserted] =
        markerHandlers.try_emplace(marker.getOperation(), nullptr);
    if (inserted)
      entry->second = markerHandlerEntry(handlerEntries, marker);
    return entry->second;
  }

  mlir::func::CallOp guardedCallAfterMarker(mlir::Operation *marker) {
    auto [entry, inserted] =
        markerGuarded.try_emplace(marker, mlir::func::CallOp());
    if (inserted)
      entry->second = own::guardedCallAfterMarker(marker);
    return entry->second;
  }

  mlir::func::CallOp anchorTrueEdgeGuardedCall(mlir::Operation *terminator) {
    auto [entry, inserted] =
        anchorGuarded.try_emplace(terminator, mlir::func::CallOp());
    if (inserted)
      entry->second = own::anchorTrueEdgeGuardedCall(terminator);
    return entry->second;
  }

  bool guardedByCallSiteMarker(mlir::Operation *call) {
    auto [entry, inserted] = guardedCalls.try_emplace(call, false);
    if (inserted)
      entry->second = static_cast<bool>(own::precedingTryCallSiteMarker(call));
    return entry->second;
  }

  struct RaiseFacts {
    bool raiseLike = false;
    bool mayRaise = false;
  };

  const RaiseFacts &raiseFacts(mlir::func::CallOp call) {
    auto [entry, inserted] =
        raiseFactsByCall.try_emplace(call.getOperation(), RaiseFacts{});
    if (!inserted)
      return entry->second;
    auto cached = contracts.lookup(call.getCallee());
    mlir::func::FuncOp callee = mlir::succeeded(cached) && *cached
                                    ? (*cached)->function
                                    : mlir::func::FuncOp();
    entry->second.raiseLike = own::isRaiseLikeFunction(callee);
    entry->second.mayRaise = own::mayRaisePythonException(callee);
    return entry->second;
  }

  // The aggregate identity a slot-absorption retain charged its token to, or
  // nothing when the retain carries no parent link or the link cannot be
  // resolved to exactly one producer in this function.
  //
  // Why "exactly one": a clone or an inline can republish an id, and an
  // ambiguous parent cannot answer `aggregate(parent, path)`. Reporting nothing
  // then falls back to counting the retain in `retained`, which is what shipped
  // -- conservative in the direction that keeps a read-back legal.
  std::optional<std::int64_t> slotRetainParent(mlir::func::CallOp call) {
    auto attr = call->getAttrOfType<mlir::IntegerAttr>(
        own::kAggregateParentAttr);
    if (!attr)
      return std::nullopt;
    buildAggregateParents();
    auto entry = aggregateParents.find(attr.getInt());
    if (entry == aggregateParents.end() || entry->second.empty())
      return std::nullopt;
    return attr.getInt();
  }

  // `aliases.same(value, candidate)` for some candidate in a fixed candidate
  // list, with the candidates reduced to their alias roots up front: the query
  // is asked per path state and the candidate list is function-sized.
  bool aliasesAnyRoot(llvm::ArrayRef<mlir::Value> values,
                      const llvm::DenseSet<mlir::Value> &roots) {
    if (roots.empty())
      return false;
    return llvm::any_of(values, [&](mlir::Value value) {
      return value && roots.contains(aliases.find(value));
    });
  }

  // Does `producer` dominate `op`?
  //
  // The question this answers for the walk: an op that mentions the tracked
  // group's ALIAS class is not necessarily a use of THIS instance of the
  // resource. The group's own SSA values can only be read where their definition
  // dominates, but the alias class also holds the storage the producer derived
  // them from -- for an owned local object, the `memref.alloc` the constructing
  // block wrote before the marker cast. When that block sits on a loop back edge,
  // the next iteration's writes to it mention the class while building a FRESH
  // instance, and the released one is not what they name.
  //
  // Why DominanceInfo and not "is the op after the producer in program order":
  // the two differ exactly on the shape at issue (a back edge reaches earlier
  // blocks), and dominance is the property SSA already guarantees for every
  // genuine use of a tracked value.
  //
  // Built lazily: functions whose resources are all released in their producing
  // block never ask.
  bool producerDominates(mlir::Operation *producer, mlir::Operation *op) {
    if (!producer || !op)
      return true;
    if (!dominance)
      dominance = std::make_unique<mlir::DominanceInfo>(function);
    return dominance->dominates(producer, op);
  }

  // Can control flow from `state`'s position still reach an op that mentions
  // one of its tracked names? Over-approximated on purpose: block granularity
  // (a mention anywhere in a block counts, including before the walk's
  // position), block-level exception edges rather than per-marker ones, and
  // nested-region ops attributed to their enclosing top-level block. Every
  // approximation errs towards "yes", so a `false` answer is a proof that no
  // remaining op can name the resource.
  bool anyMentionReachable(const AffinePathState &state) {
    mlir::Block *from = topLevelBlock(state.block);
    if (!from)
      return true;
    std::optional<unsigned> index = blockIndexOf(from);
    if (!index)
      return true;
    for (llvm::ArrayRef<mlir::Value> names :
         {llvm::ArrayRef<mlir::Value>(state.group),
          llvm::ArrayRef<mlir::Value>(state.views),
          llvm::ArrayRef<mlir::Value>(state.stale),
          llvm::ArrayRef<mlir::Value>(state.previous)})
      for (mlir::Value value : names) {
        if (!value)
          continue;
        if (mentionReachers(aliases.find(value))[*index])
          return true;
      }
    return false;
  }

private:
  // Container producers by aggregate identity, with the container's release
  // interface as the lane list (lane 0 is the entity root, which is what every
  // deallocator names as operand 0). An identity claimed by more than one
  // producer is erased rather than resolved: an ambiguous `parent` cannot
  // discharge anything, and the caller falls back to shipped behaviour.
  void buildAggregateParents() {
    if (aggregateParentsBuilt)
      return;
    aggregateParentsBuilt = true;
    if (function.isDeclaration())
      return;
    llvm::DenseSet<std::int64_t> ambiguous;
    function.walk([&](mlir::Operation *op) {
      auto attr = op->getAttrOfType<mlir::IntegerAttr>(own::kAggregateIdAttr);
      if (!attr)
        return;
      std::int64_t identity = attr.getInt();
      if (ambiguous.contains(identity))
        return;
      llvm::SmallVector<mlir::Value, 4> lanes(op->getResults());
      if (lanes.empty())
        return;
      auto [entry, inserted] = aggregateParents.try_emplace(identity, lanes);
      if (!inserted) {
        ambiguous.insert(identity);
        aggregateParents.erase(entry);
      }
    });
  }

  // The block of `function`'s own region that contains `block` (itself, when
  // `block` already belongs to that region).
  mlir::Block *topLevelBlock(mlir::Block *block) {
    while (block) {
      mlir::Region *region = block->getParent();
      if (!region)
        return nullptr;
      mlir::Operation *parent = region->getParentOp();
      if (!parent)
        return nullptr;
      if (parent == function.getOperation())
        return block;
      block = parent->getBlock();
    }
    return nullptr;
  }

  // Blocks from which a mention of `root`'s alias class is reachable, as a bit
  // per top-level block: the backward closure of the mentioning blocks over the
  // same augmented edge set the walk follows (CFG successors plus exception
  // edges). Why a bit vector and not a block-pointer set: one is cached per
  // tracked name generation, and a pointer set costs a word per reachable
  // block, which on a function with thousands of resources is hundreds of
  // megabytes of cache for a one-bit answer.
  const llvm::BitVector &mentionReachers(mlir::Value root) {
    auto cached = reachers.find(root);
    if (cached != reachers.end())
      return cached->second;

    buildBlockGraph();
    llvm::BitVector reaching(blockOrder.size());
    llvm::SmallVector<unsigned, 16> worklist;
    auto enqueue = [&](std::optional<unsigned> index) {
      if (index && !reaching.test(*index)) {
        reaching.set(*index);
        worklist.push_back(*index);
      }
    };
    if (auto users = classUsers.find(root); users != classUsers.end())
      for (mlir::Operation *user : users->second)
        enqueue(blockIndexOf(topLevelBlock(user->getBlock())));
    while (!worklist.empty()) {
      unsigned index = worklist.pop_back_val();
      for (unsigned predecessor : predecessors[index])
        enqueue(predecessor);
    }
    return reachers.insert({root, std::move(reaching)}).first->second;
  }

  std::optional<unsigned> blockIndexOf(mlir::Block *block) {
    if (!block)
      return std::nullopt;
    buildBlockGraph();
    auto found = blockIndex.find(block);
    if (found == blockIndex.end())
      return std::nullopt;
    return found->second;
  }

  void buildBlockGraph() {
    if (blockGraphBuilt)
      return;
    blockGraphBuilt = true;
    if (function.isDeclaration())
      return;
    for (mlir::Block &block : function.getBody()) {
      blockIndex.insert({&block, static_cast<unsigned>(blockOrder.size())});
      blockOrder.push_back(&block);
    }
    predecessors.resize(blockOrder.size());
    llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
        exceptionEdges = own::collectExceptionEdges(function.getBody());
    auto addEdge = [&](mlir::Block *source, mlir::Block *target) {
      auto sourceIndex = blockIndex.find(source);
      auto targetIndex = blockIndex.find(target);
      if (sourceIndex == blockIndex.end() || targetIndex == blockIndex.end())
        return;
      predecessors[targetIndex->second].push_back(sourceIndex->second);
    };
    for (mlir::Block &source : function.getBody()) {
      for (mlir::Block *successor : source.getSuccessors())
        addEdge(&source, successor);
      if (auto found = exceptionEdges.find(&source);
          found != exceptionEdges.end())
        for (mlir::Block *successor : found->second)
          addEdge(&source, successor);
    }
  }

  bool mentionsAny(mlir::Operation *op, llvm::ArrayRef<mlir::Value> values) {
    for (mlir::Value value : values) {
      if (!value)
        continue;
      auto users = classUsers.find(aliases.find(value));
      if (users != classUsers.end() && llvm::is_contained(users->second, op))
        return true;
    }
    return false;
  }

  mlir::func::FuncOp function;
  FuncContractCache &contracts;
  own::AliasAnalysis &aliases;
  const ExceptionHandlerMap &handlerEntries;
  llvm::DenseMap<mlir::Value, llvm::SmallVector<mlir::Operation *, 1>>
      classUsers;
  // Top-level blocks, indexed: the mention-reachability closure is a bit per
  // block, so it needs a dense numbering rather than pointers.
  llvm::SmallVector<mlir::Block *, 16> blockOrder;
  llvm::DenseMap<mlir::Block *, unsigned> blockIndex;
  llvm::SmallVector<llvm::SmallVector<unsigned, 2>, 16> predecessors;
  bool blockGraphBuilt = false;
  llvm::DenseMap<mlir::Value, llvm::BitVector> reachers;
  llvm::DenseMap<mlir::Operation *, mlir::Block *> markerHandlers;
  llvm::DenseMap<mlir::Operation *, mlir::func::CallOp> markerGuarded;
  llvm::DenseMap<mlir::Operation *, mlir::func::CallOp> anchorGuarded;
  llvm::DenseMap<mlir::Operation *, bool> guardedCalls;
  llvm::DenseMap<mlir::Operation *, RaiseFacts> raiseFactsByCall;
  llvm::DenseMap<std::int64_t, llvm::SmallVector<mlir::Value, 4>>
      aggregateParents;
  bool aggregateParentsBuilt = false;
  std::unique_ptr<mlir::DominanceInfo> dominance;
};

// WHERE THE VIRTUAL UNWIND EDGE IS.
//
// An anchor cond_br's TRUE edge is the spelling of "the guarded call unwound":
// at runtime it is only ever taken by an unwind, so a path down it must carry
// the state the runtime unwind would -- the guarded call's consume applied, and
// the path marked exceptional, exactly like the marker edges.
//
// Why one helper for two walks: this is ONE fact about the IR, and it was
// written out once per walk. A new anchor spelling taught to one copy would
// silently not exist in the other, and the two walks would then disagree about
// which edge an exception takes -- in a verifier whose entire judgment is about
// what happens on that edge. The two DIFFER in what they do with the answer
// (one has a token to release, the other only a retain counter), which is why
// this returns the fact and not the effect.
struct AnchorTrueEdge {
  bool isVirtualUnwind = false; // successor 0 is the anchor's unwind edge
  bool consumesGroup = false;   // and the call it guards consumes the group
};

template <typename State>
AnchorTrueEdge anchorTrueEdgeOf(OwnershipWalkCache &walk,
                                FuncContractCache &contracts,
                                mlir::Operation *terminator, const State &state,
                                own::AliasAnalysis &aliases) {
  AnchorTrueEdge edge;
  if (auto cond = mlir::dyn_cast<mlir::cf::CondBranchOp>(terminator))
    if (auto anchorCall =
            cond.getCondition().getDefiningOp<mlir::func::CallOp>())
      edge.isVirtualUnwind = anchorCall.getCallee() == "LyEH_TryCatchAnchor";
  mlir::func::CallOp guarded = walk.anchorTrueEdgeGuardedCall(terminator);
  edge.consumesGroup =
      guarded && walk.mentionsTracked(guarded, state) &&
      callConsumesGroup(contracts, guarded, state.group, aliases);
  return edge;
}

// Alias roots of `values`. `aliases.same(v, c)` holding for some c in `values`
// is exactly `aliases.find(v)` being one of these roots.
llvm::DenseSet<mlir::Value> aliasRootsOf(llvm::ArrayRef<mlir::Value> values,
                                         own::AliasAnalysis &aliases) {
  llvm::DenseSet<mlir::Value> roots;
  for (mlir::Value value : values)
    if (value)
      roots.insert(aliases.find(value));
  return roots;
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

  // One entity crosses one edge under ONE rename: find where the ROOT lands
  // and take the lanes from there. Mapping each lane independently made an
  // entity's survival across an edge depend on all N renames succeeding, so a
  // single lane the branch did not forward under its own name ended the
  // entity's tracking.
  auto forwardedArgumentFor = [&](mlir::Value value) -> int {
    for (unsigned argumentIndex = 0; argumentIndex < argumentCount;
         ++argumentIndex) {
      mlir::Value forwarded = operands[argumentIndex];
      if (forwarded && aliases.same(forwarded, value))
        return static_cast<int>(argumentIndex);
    }
    return -1;
  };

  int rootArgument = group.empty() ? -1 : forwardedArgumentFor(group.front());
  if (rootArgument >= 0 &&
      static_cast<unsigned>(rootArgument) + group.size() <=
          successor->getNumArguments()) {
    bool laneTypesMatch = true;
    for (auto [groupIndex, value] : llvm::enumerate(group))
      if (successor->getArgument(rootArgument + groupIndex).getType() !=
          value.getType())
        laneTypesMatch = false;
    if (laneTypesMatch) {
      for (auto [groupIndex, value] : llvm::enumerate(group)) {
        mapped[groupIndex] =
            successor->getArgument(rootArgument + groupIndex);
        if (mappedMask)
          (*mappedMask)[groupIndex] = true;
      }
      return mapped;
    }
  }

  // The lanes are not laid out contiguously behind the root on this edge
  // (unrelated arguments interleaved): fall back to per-lane renaming, which
  // is what the whole walk did before the root became authoritative.
  for (auto [groupIndex, value] : llvm::enumerate(group)) {
    int argumentIndex = forwardedArgumentFor(value);
    if (argumentIndex < 0)
      continue;
    mapped[groupIndex] = successor->getArgument(argumentIndex);
    if (mappedMask)
      (*mappedMask)[groupIndex] = true;
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
                           own::AliasAnalysis &aliases,
                           const own::ReferenceMap &references) {
  if (resource.condition || !resource.producer)
    return std::nullopt;
  mlir::Block *block = resource.producer->getBlock();
  if (!block || block->getParentOp() != resource.function)
    return std::nullopt;

  llvm::SmallVector<mlir::Operation *, 16> users;
  llvm::SmallPtrSet<mlir::Operation *, 16> seen;
  llvm::SmallVector<mlir::Value, 8> trackedValues(resource.values.begin(),
                                                  resource.values.end());
  trackedValues.append(resource.views.begin(), resource.views.end());
  for (mlir::Value value : trackedValues) {
    llvm::SmallVector<mlir::Value, 8> equivalentValues;
    aliases.namesOf(value, equivalentValues);
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
  // Of those, the ones PARKED IN A CONTAINER by a slot-absorption retain. They
  // keep the object alive exactly like a plain retain -- so every rule that
  // asks "is anything still holding this" keeps counting them -- but they
  // belong to the holder, not to this frame, so the owned-return rule below
  // must not read them as tokens the return failed to spend. The CFG walk
  // splits the same two pools (`slotParents` vs `retained`); this walk had one.
  // A retain that hands the token to a container rather than to this frame.
  auto slotAbsorption = [](mlir::func::CallOp call) {
    return call->hasAttr(own::kAggregateRetainAttr) &&
           !isBlockArgMergeBorrowRetain(call);
  };
  unsigned parked = 0;
  llvm::SmallVector<mlir::Value, 4> group = resource.values;
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
        // ⭐ Storing a value into a container and returning the same value is
        // balanced: the container holds its own reference and the return
        // transfers the frame's.
        //
        //     def f(n: int, memo: dict[int, int]) -> int:
        //         v = n * 2
        //         memo[n] = v
        //         return v
        //
        // was refused -- "returned with 1 additional retained ownership
        // token" -- because the slot's retain was counted against the return.
        // A memoized fib is the shape this was found on.
        if (retained > parked)
          return ret.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset << " is returned with "
                 << (retained - parked)
                 << " additional retained ownership token(s)";
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
      if (consumes &&
          callReleasesForeignAggregate(
              references, resource.hasOwnNamedRelease,
              resource.ownsReference && resource.hasOwnNamedRelease,
              resource.reference, {group, resource.views}, call))
        continue;
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
          if (parked > retained)
            parked = retained;
          continue;
        }
        if ((groupContainsOperand(op, group, aliases) ||
             groupContainsOperand(op, resource.views, aliases)) &&
            retained == 0)
          return call.emitError()
                 << "released owned resource from " << resource.producerLabel
                 << " is used after release (by call to '" << call.getCallee()
                 << "')";
        if (retains) {
          ++retained;
          if (slotAbsorption(call))
            ++parked;
        }
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
      if (token == AffineTokenState::Owned && retains) {
        ++retained;
        if (slotAbsorption(call))
          ++parked;
      }
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
    // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
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
    resource.values = std::move(group);
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

// Where a region-branch successor continues, for either walk: the parent
// continuation is the op's own next node, a region entry is that region's
// first block. The two walks differ in what they CARRY, not in where the edge
// goes, so the state type is the only parameter.
template <typename State>
void enqueueRegionSuccessor(mlir::Operation *owner, mlir::RegionSuccessor succ,
                            State state,
                            llvm::SmallVectorImpl<State> &worklist) {
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

// The owned and the borrowed walk enter regions the same way: remap the group
// through the branch's entry operands, and keep the fallthrough alive when a
// region never mentions the group. Only the carried state differs.
template <typename State>
bool enqueueRegionEntryPaths(mlir::Operation *op, State state,
                             own::AliasAnalysis &aliases,
                             OwnershipWalkCache &walk,
                             llvm::SmallVectorImpl<State> &worklist) {
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
    if (!successor.isParent() &&
        !walk.regionMentionsGroup(*successor.getSuccessor(), state.group)) {
      hasNoUseRegionPath = true;
      continue;
    }

    State next = state;
    mlir::OperandRange sources = branch.getEntrySuccessorOperands(successor);
    next.group = remapGroupThroughValueMapping(
        sources, successor.getSuccessorInputs(), state.group, aliases);
    enqueueRegionSuccessor(op, successor, std::move(next), worklist);
    handled = true;
  }

  if (hasNoUseRegionPath) {
    State next = state;
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
        outstandingTokens(state) == 0)
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
        outstandingTokens(state) == 0)
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
    enqueueRegionSuccessor(owner, successor, std::move(next), worklist);
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
    own::AliasAnalysis &aliases, OwnershipWalkCache &walk) {
  llvm::SmallVector<BorrowedPathState, 16> worklist;
  VisitedBorrowedStates visited;
  mlir::Block &entry = resource.function.front();
  worklist.push_back(BorrowedPathState{&entry, firstOperation(&entry),
                                       /*retained=*/0, resource.values});

  constexpr unsigned kMaxBorrowedStates = 20000;
  constexpr unsigned kMaxRetainedBalance = 64;
  while (!worklist.empty()) {
    BorrowedPathState state = worklist.pop_back_val();
    if (visited.contains(state))
      continue;
    visited.insert(state);
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
          if (mlir::Block *handler = walk.markerHandler(call)) {
            BorrowedPathState next = state;
            // The unwind transfer happens DURING the guarded call: its
            // consume effect applies on the exceptional edge. Release
            // helpers scheduled between the marker and the guarded call (a
            // raise statement's dying locals) run BEFORE any unwind, so
            // their consume effects apply on the edge too.
            mlir::func::CallOp guarded = walk.guardedCallAfterMarker(op);
            for (mlir::Operation *between = op->getNextNode();
                 between && guarded && between != guarded.getOperation();
                 between = between->getNextNode())
              if (auto releaseCall =
                      mlir::dyn_cast<mlir::func::CallOp>(between))
                if (callConsumesGroup(contracts, releaseCall, next.group,
                                      aliases) &&
                    next.retained > 0)
                  --next.retained;
            if (guarded)
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
        // No operand of this call aliases the tracked group: every predicate
        // below needs one, so they all answer `false` without being asked.
        bool mentionsTracked = walk.mentionsTracked(op, state);
        if (mentionsTracked &&
            callPartiallyConsumesGroup(contracts, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of borrowed "
                    "entry argument "
                 << resource.logicalIndex << " of @"
                 << resource.function.getSymName();

        if (mentionsTracked &&
            callConsumesGroup(contracts, call, state.group, aliases)) {
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
        } else if (state.retained > 0) {
          // ⭐ THE MERGE-BORROW LEND COMES BACK UNDER THE PRE-MERGE NAME. The
          // retain branch below has always said so -- "a token that a loop-edge
          // decref later returns through the pre-merge name" -- and counted the
          // lend; nothing credited the return, because the group had been
          // remapped to the merge argument by then. One lend per rename
          // survived, so a NESTED loop over a borrowed parameter
          //
          //     def f(n: int) -> int:
          //         i = 0
          //         while i < 2:
          //             while n >= 10:
          //                 n -= 10
          //             i += 1
          //         return n
          //
          // was refused with "returned with 2 retained ownership tokens", and
          // the `-> str` spelling with "reaches function exit with 1". Both are
          // sound: with the verifier off they print CPython's answer, and 2000
          // calls measure at net 0 allocations / 0 B.
          //
          // ⛔ Why crediting here and not exempting the exit check, which was
          // tried first: the balance climbs on the way round, so the walk hits
          // "retain balance exceeded 64" before it reaches any return. The
          // count has to stop being wrong, not stop being read.
          for (const llvm::SmallVector<mlir::Value, 4> &earlier :
               llvm::reverse(state.previousGroups)) {
            if (!callConsumesGroup(contracts, call, earlier, aliases))
              continue;
            --state.retained;
            break;
          }
        }

        if (mentionsTracked &&
            callRetainsGroup(contracts, call, state.group, aliases)) {
          // Slot-absorption retains (field stores etc.) park the token in the
          // holder and are invisible to this walk — EXCEPT merge-borrow
          // retains: an identity merge edge lends the merge argument a token
          // that a loop-edge decref later returns through the pre-merge name
          // (e.g. `local = borrowed_arg` then `local = local - 1` in a loop),
          // so it must count toward the retained balance.
          //
          // ⭐ AND EXCEPT `py.incref`, for the same reason: it parks nothing.
          // The emitter writes it where a local has to stop borrowing and
          // start owning (`cur: "Node | None" = head`), so the frame gains the
          // reference and the frame's own release discharges it. Skipping it
          // made the accounting one-sided -- the retain uncounted, the paired
          // release counted -- and the shape that reads a linked structure
          //
          //     cur: "Node | None" = head
          //     while cur is not None:
          //         cur = cur.nxt
          //
          // was refused with "released or transferred without a prior retain"
          // over a retain standing three ops above the release.
          if (call->hasAttr(own::kAggregateRetainAttr) &&
              !isBlockArgMergeBorrowRetain(call) &&
              !own::isEmitterIncrefRetain(call)) {
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

        if (walk.raiseFacts(call).raiseLike) {
          // An unguarded raise exits the function: a retained token held
          // here has no remaining release path (a guarded raise reaches its
          // handler through the marker edge enqueued above, which carries
          // the retained balance for the handler-side checks). Only checked
          // while the group still carries the ENTRY names on a normal path:
          // once a merge edge renamed the group, the balance includes
          // block-arg-merge lends whose paired release targets a pre-merge
          // name -- a state the insertion pass cannot discharge, so
          // rejecting it would hard-error plain loop-reassignment code
          // (documented residual).
          if (state.retained != 0 && !state.exceptional &&
              own::sameEntityRoot(state.group, resource.values) &&
              !walk.guardedByCallSiteMarker(call))
            return call.emitError()
                   << "borrowed entry argument " << resource.logicalIndex
                   << " of @" << resource.function.getSymName() << " holds "
                   << state.retained
                   << " retained ownership token(s) when '" << call.getCallee()
                   << "' unwinds out of the function";
          // A raise never returns; the syntactic continuation is dead code,
          // so walking it would verify a path that cannot run.
          op = nullptr;
          break;
        }
      }

      if (op->getNumRegions() != 0 &&
          enqueueRegionEntryPaths(op, state, aliases, walk, worklist)) {
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

    AnchorTrueEdge anchor =
        anchorTrueEdgeOf(walk, contracts, op, state, aliases);
    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      BorrowedPathState next = state;
      if (anchor.isVirtualUnwind && index == 0)
        next.exceptional = true;
      if (anchor.consumesGroup && index == 0 && next.retained > 0)
        --next.retained;
      next.block = successor;
      next.start = firstOperation(successor);
      llvm::SmallVector<bool, 4> mappedMask;
      next.group = remapGroupForSuccessor(op, index, successor, state.group,
                                          aliases, &mappedMask);
      // A rename: remember what the group was called, so the lend taken on
      // that name can be credited when it is returned there.
      if (next.group != state.group) {
        next.previousGroups.push_back(state.group);
        if (next.previousGroups.size() > kMaxBorrowedPreviousGroups)
          next.previousGroups.erase(next.previousGroups.begin());
      }
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

// Values whose ownership balance an unwind exit cannot classify as plainly
// held-or-consumed: operands of retain calls (an extra live token) and of
// block-arg-merge borrow lends (a token whose paired release targets a
// pre-merge name on another path). The unwind-cleanup insertion skips groups
// with such uses (collectUnwindGroupSites in Passes/Ownership.cpp), so the
// unwind-exit obligations are not imposed on them here either: insertion and
// verification must agree, or every skipped group becomes a spurious hard
// error on the very cleanup handlers the insertion did wire. Slot-absorption
// aggregate retains stay out of the set -- the holder owns that token and the
// group's own balance is unchanged.
llvm::SmallVector<mlir::Value, 8>
collectUnwindAmbiguousRetainOperands(mlir::func::FuncOp function,
                                     FuncContractCache &contracts) {
  llvm::SmallVector<mlir::Value, 8> values;
  if (function.isDeclaration())
    return values;
  function.walk([&](mlir::func::CallOp call) {
    auto cached = contracts.lookup(call.getCallee());
    if (mlir::failed(cached) || !*cached)
      return;
    if ((*cached)->contract.retainArgs.empty())
      return;
    if (call->hasAttr(own::kAggregateRetainAttr) &&
        !isBlockArgMergeBorrowRetain(call))
      return;
    for (unsigned offset : (*cached)->contract.retainArgs.values)
      if (offset < call.getNumOperands())
        values.push_back(call.getOperand(offset));
  });
  return values;
}


mlir::LogicalResult verifyResourceOnCFGPaths(
    FuncContractCache &contracts, TrackedResource &resource,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases, const own::ReferenceMap &references,
    OwnershipWalkCache &walk,
    bool modelMayRaiseUnwindExits,
    const llvm::DenseSet<mlir::Value> &ambiguousRetainRoots) {
  llvm::SmallVector<AffinePathState, 16> worklist;
  VisitedAffineStates visited;
  AffineTokenState initialToken = resource.condition
                                      ? AffineTokenState::Conditional
                                      : AffineTokenState::Owned;
  AffinePathState initial;
  initial.block = resource.producer->getBlock();
  initial.start = resource.producer->getNextNode();
  initial.token = initialToken;
  initial.retained = 0;
  initial.group = resource.values;
  initial.views = resource.views;
  worklist.push_back(std::move(initial));

  // ⛔ KNOWN DEFECT reached through this cap, and the cap is the symptom rather
  // than the cause. A CONDITIONAL REBIND together with an APPEND in the same
  // loop grows `retained` by one per explored trip, so no state ever repeats
  // and the walk runs to the cap instead of converging:
  //
  //     xs = [3, 1, 4]
  //     best = xs[0]
  //     out: list[int] = []
  //     for v in xs:
  //         if v > best:
  //             best = v
  //         out.append(v)      # cap fires at retained=2858
  //
  // Reduced both ways: drop the `if` and it compiles, drop the append and it
  // compiles, and the count is the same (2858, 2859) whether the list has one
  // element or three -- so this is a runaway walk, not a size the cap is too
  // small for. It is the conditional-rebind family: the retained reference and
  // the lent one are ONE SSA value, so no release can discharge one without
  // discharging the other, and each trip adds an obligation the walk cannot
  // spend. Separating them needs the terminator to carry a second operand.
  //
  // Reporting it as an exploration limit is the honest thing this walk can say
  // today -- it does not know the difference between "too big" and "diverging"
  // -- but a reader arriving here from the message should know it is the
  // second one.
  constexpr unsigned kMaxAffineStates = 20000;
  while (!worklist.empty()) {
    AffinePathState state = worklist.pop_back_val();
    if (visited.contains(state))
      continue;
    visited.insert(state);
    if (ownershipPathTraceEnabled())
      state.trail.push_back(blockOrdinal(state.block));
    // Outstanding tokens of every provenance: this walk's own retains plus the
    // ones parked in containers (`state.slotParents`). Both keep a released
    // group alive and both can pay for a second consume, so every rule that
    // used to read `state.retained` alone reads this instead.
    //
    // Why NOT leave the parked ones out of these rules: splitting the pools
    // without rejoining them here is precisely how the deleted skip experiment
    // produced `released or transferred more than once` on three golden cases
    // and `used after release` on thirty-six more.
    auto outstanding = [&state] { return outstandingTokens(state); };
    // Spend a plain retain first, a parked one only when no plain one is left.
    // Why that order: a plain retain has no other discharge on this path, while
    // a parked one still has the holder's release ahead of it.
    auto spendOutstanding = [&state] {
      if (state.retained > 0) {
        --state.retained;
        if (state.parkedUnnamed > state.retained)
          state.parkedUnnamed = state.retained;
      } else if (!state.slotParents.empty())
        state.slotParents.pop_back();
    };
    // ⚠️ THIS EXIT IS NOT A SAFE-SIDE FAILURE, and reading it as one cost a day
    // (2026-07-28). Bailing here says nothing about the judgements downstream of
    // where the walk stopped: "the verifier passed" and "the verifier REACHED
    // that check" are different claims, and only the second licenses a
    // conclusion. Measured instance, since repaired: a slot retain inside a loop
    // made `state.retained` -- part of the visited-state key -- increase every
    // iteration, so the fixpoint never closed and this fired; making it converge
    // then reported a REAL `used after release` in the same family of programs,
    // invisible the whole time it had been firing. The state explosion was a
    // cover, not a diagnostic.
    //
    // So a rise in this diagnostic must be investigated as a possible masked
    // finding, and any claim of the form "the affine verifier is green on X"
    // requires that X did not hit this cap.
    // tests/probe/seqlit_slot_retain_in_loop_str.py is the program that did.
    if (visited.size() > kMaxAffineStates)
      return resource.producer->emitError()
             << "ownership CFG exploration exceeded " << kMaxAffineStates
             << " states (last: retained=" << state.retained
             << " parked=" << state.slotParents.size()
             << " borrowed=" << state.borrowedRetains.size()
             << " prev=" << state.previous.size()
             << " stale=" << state.stale.size() << " group=" << state.group.size()
             << " token=" << static_cast<int>(state.token) << ")";

    // A released token carries no remaining obligation: every exit rule for it
    // (function return, region exit, CFG exit, loop re-entry) succeeds, and
    // every diagnostic still reachable on the path -- double release, use after
    // release, partial consume, release through a transferred name -- requires
    // an op naming one of the tracked values. So once no such op is reachable,
    // the rest of this path is proven diagnostic-free and exploring it only
    // spends states. This is the term that made the walk quadratic in function
    // size: a resource released next to its producer still walked every
    // remaining op of the function, once per resource.
    //
    // Why not the same prune for Owned/Conditional: those owe a release on
    // every exit, so their obligation is discharged by ops the walk has yet to
    // see -- silence downstream is exactly the leak they must report.
    if (state.token == AffineTokenState::Released &&
        !walk.anyMentionReachable(state))
      continue;

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
      // Executing an op REBINDS its results. A stale name among them was the
      // previous execution's moved token; this execution's value is a fresh
      // resource carrying a token of its own, so the name stops being stale
      // here.
      //
      // The forwarding edge already drops stale names that are block ARGUMENTS
      // of the block being entered, for exactly this reason. But naming one kind
      // of rebinding is not the same as asking which rebindings happen: a
      // construction inside a loop BODY is rebound by an op, so its stale entry
      // used to ride the back edge and then match the very call that had made it
      // stale. `cur = ValueError(...)` inside a loop was refused on that path
      // while the inserted releases were in fact balanced -- measured: correct
      // output, flat RSS across 400k iterations, clean under libgmalloc in both
      // guard orders.
      //
      // Why NOT drop at the edge by asking which ops the successor can reach:
      // reachability is not execution. A name whose defining op is merely
      // reachable has not been rebound yet, and dropping it there would lose the
      // straight-line double-consume this set exists to catch. Dropping at the
      // op asks the same question where it is decidable.
      if (!state.stale.empty() && op->getNumResults() != 0 &&
          !staleRebindDropDisabled())
        llvm::erase_if(state.stale, [&](mlir::Value value) {
          if (value.getDefiningOp() != op)
            return false;
          traceStaleValue("rebound", value, state.block);
          return true;
        });
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
          // ⭐ `parkedUnnamed`, not zero: storing a value into a container
          // and returning the same value is balanced -- the container holds
          // its own reference and the return transfers the frame's.
          //
          //     def f(n: int, memo: dict[int, int]) -> int:
          //         v = n * 2
          //         memo[n] = v
          //         return v
          //
          // A memoized fib is the shape this was found on. The named parked
          // tokens never reach `retained`; only the ones whose holder this
          // walk could not name fall through to it, and those are the ones
          // subtracted here.
          if (state.retained > state.parkedUnnamed)
            return ret.emitError()
                   << "owned resource from " << resource.producerLabel
                   << " result " << resource.resultOffset
                   << " is returned with "
                   << (state.retained - state.parkedUnnamed)
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
          // `outstanding()`, not `state.retained`: returning the element inside
          // the very container that absorbed it is the shape whose only
          // outstanding token is a parked one.
          if (outstanding() > 0 &&
              returnCarriesGroupInsideOwnedAggregate(
                  resource.function, ret, state.group, deallocators, aliases))
            break;
          // ⛔ Same dominance guard as the use-after-release arms: a return the
          // producer does not dominate is naming a value this resource's
          // release never touched -- the loop-carried argument an element
          // token was minted from, on the exit edge of the loop.
          if (!releasedUseDominanceDisabled() &&
              !walk.producerDominates(resource.producer, ret))
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
              llvm::SmallVector<mlir::Value, 4> mappedViews =
                  remapGroupForSuccessor(op, successorIndex, successor,
                                         state.views, aliases);
              AffinePathState next{
                  successor, firstOperation(successor), nextToken,
                  state.retained, std::move(mappedGroup),
                  /*stale=*/{}, /*previous=*/{}, std::move(mappedViews),
                  /*borrowedRetains=*/{}, state.exceptional};
              // Parked slot retains follow the path, not the group's names:
              // the container that holds them is unaffected by a union-tag
              // branch on the element.
              next.slotParents = state.slotParents;
              next.parkedUnnamed = state.parkedUnnamed;
              next.parkedOps = state.parkedOps;
              worklist.push_back(std::move(next));
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

      // No operand of this op aliases any tracked name: every group / stale /
      // previous / views predicate in the rest of this iteration needs one, so
      // they all answer `false` without being asked. The unwind-exit checks
      // below are NOT skipped -- they are properties of the callee, not of the
      // operands -- so the pruning removes work, never a check.
      bool mentionsTracked = walk.mentionsTracked(op, state);

      // A release written under the other token's name is invisible here
      // (`callReleasesForeignAggregate`).
      if (mentionsTracked &&
          (references.isMinted(resource.reference) ||
           resource.ownsReference)) {
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op))
          if (callReleasesForeignAggregate(
                  references, resource.hasOwnNamedRelease,
                  resource.ownsReference && resource.hasOwnNamedRelease,
                  resource.reference,
                  {state.group, state.views, state.previous, state.stale},
                  call) &&
              callConsumesGroup(contracts, call, state.group, aliases)) {
            op = op->getNextNode();
            continue;
          }
      }

      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        if (call.getCallee() == "LyEH_TryCallSiteMarker") {
          // Exceptional edge: the marked call site may unwind to ITS
          // handler entry with the token state as of this point, except
          // that the unwind happens DURING the guarded call -- consume
          // effects of the guarded call apply on the edge, and its results
          // never materialize (a transfer is a plain release here).
          if (mlir::Block *handler = walk.markerHandler(call)) {
            AffinePathState next = state;
            auto applyEdgeConsume = [&](mlir::func::CallOp consumer) {
              if (!callConsumesGroup(contracts, consumer, next.group, aliases))
                return;
              if (next.token == AffineTokenState::Owned ||
                  next.token == AffineTokenState::Conditional)
                next.token = AffineTokenState::Released;
              else if (next.retained > 0)
                --next.retained;
              else if (!next.slotParents.empty())
                next.slotParents.pop_back();
            };
            // Release helpers scheduled between the marker and the guarded
            // call (a raise statement's dying locals) run BEFORE any unwind,
            // so their consume effects apply on the exceptional edge just
            // like the guarded call's own transfer.
            mlir::func::CallOp guarded = walk.guardedCallAfterMarker(op);
            for (mlir::Operation *between = op->getNextNode();
                 between && guarded && between != guarded.getOperation();
                 between = between->getNextNode())
              if (auto releaseCall =
                      mlir::dyn_cast<mlir::func::CallOp>(between))
                applyEdgeConsume(releaseCall);
            if (guarded)
              applyEdgeConsume(guarded);
            next.block = handler;
            next.start = firstOperation(handler);
            next.exceptional = true;
            worklist.push_back(std::move(next));
          }
          op = op->getNextNode();
          continue;
        }
        if (mlir::Value consumedStale =
                mentionsTracked
                    ? callConsumesStaleValue(contracts, call, state.stale,
                                             state.group, aliases)
                    : mlir::Value()) {
          traceStaleValue("consumed", consumedStale, state.block);
          return call.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset
                 << " is released through a value already consumed by an "
                    "ownership transfer";
        }
        bool consumes = mentionsTracked &&
                        callConsumesGroup(contracts, call, state.group, aliases);
        // A release through PRE-RENAME names of the current group cancels an
        // outstanding borrow-edge retain (identity merge edge): the token
        // continues under the current names.
        if (!consumes && !state.borrowedRetains.empty() && mentionsTracked &&
            callConsumesStaleValue(contracts, call, state.previous,
                                   state.group, aliases)) {
          state.borrowedRetains.pop_back();
          op = op->getNextNode();
          continue;
        }
        // With NO outstanding borrow, a release through a pre-rename name is the
        // release OF THIS TOKEN. The rename happens whenever a branch forwards
        // the group into a block argument, so a loop-invariant object entering a
        // loop header is renamed for the rest of the walk -- including down the
        // exception edge into an `except` block, whose release is written under
        // the pre-loop name. Without this the walk reports a leak at the return
        // for a token that is demonstrably released.
        //
        // Why NOT leave it to the pad release that used to keep this quiet: that
        // release was the defect. It freed the object while the handler in the
        // same function still read it (SIGSEGV in `Ly_IncRef`), and the walk
        // stayed silent only because the freeing happened under the name it was
        // tracking. A verifier that needs a double free to accept a program is
        // reporting the wrong side of the judgment.
        //
        // Why NOT weaken this to "ignore the release": marking the token
        // Released keeps the AFFINE property checkable -- a second release, under
        // either name, is now a double consume this same walk reports, which is
        // how the normal-path over-release in this family stays caught.
        bool retains = mentionsTracked &&
                       callRetainsGroup(contracts, call, state.group, aliases);
        // A slot-absorption retain (`aggregate_retain`: an element/field store)
        // parks the token in the HOLDER. `aggregate(parent, path)` is a resource
        // of `parent` (rfc/memory-safety-proof.md, Aggregates), so the token is
        // charged to the container's identity instead of to this walk's free
        // counter, and the container's release discharges it below.
        //
        // ⛔ Why NOT skip these the way the borrowed-entry walk does (the
        // slot-retain-skipping experiment, measured
        // 2026-07-28 and deleted here): the two walks track different resource
        // kinds and the asymmetry is not a licence to copy the exemption.
        // Reading an element BACK out of a container (`total += ys[0]`) hands
        // the reader a token derived from the slot, and this walk needs the
        // retain to justify the reader's later release. Skipping it refused
        // programs that compile today --
        //
        //     for i in range(3, 6):      ys = [i]; total += ys[0]   -> refused
        //     for i in ... for j in ...: ys = [7]; total += ys[0]   -> refused
        //
        // -- 39 golden cases in all, and combined with the CollectionPayload
        // source-move predicate it turned a refusal into a SILENT WRONG ANSWER
        // for `v = ys[0]` in a nested loop, the one direction this family may
        // never move in. Charging the token to the parent keeps it outstanding
        // (so the read-back stays legal) while bounding the state key (so the
        // fixpoint closes).
        //
        // Why NOT charge it when the parent link is missing or ambiguous: an
        // unnamed parent can never be released by name either, so the charge
        // would never be discharged and would only cost states. Those fall
        // through to `retained`, which is exactly what shipped.
        std::optional<std::int64_t> slotParent;
        bool slotAbsorptionRetain = retains &&
                                    call->hasAttr(own::kAggregateRetainAttr) &&
                                    !isBlockArgMergeBorrowRetain(call);
        bool parkedWithoutAName = false;
        if (slotAbsorptionRetain) {
          slotParent = walk.slotRetainParent(call);
          slotAbsorptionRetain = slotParent.has_value();
          parkedWithoutAName = !slotAbsorptionRetain;
        }
        // The DISCHARGE. Checked for every call, not only ones mentioning the
        // tracked group: the container's release names the CONTAINER, and a
        // walk that only looks at ops naming the element never sees it. That
        // blindness is what let `state.retained` grow without bound.
        if (mentionsTracked &&
            callPartiallyConsumesGroup(contracts, call, state.group, aliases))
          return call.emitError()
                 << "ownership-consuming call only consumes part of owned "
                    "resource group produced by "
                 << resource.producerLabel << " result "
                 << resource.resultOffset;

        if (state.token == AffineTokenState::Released) {
          // ⛔ A CONSUME THE PRODUCER DOES NOT DOMINATE IS NOT THIS TOKEN'S,
          // the same reading the use arms take. The shape it answers is the
          // loop-carried variable an element token was minted from:
          //
          //     while i < n:
          //         p = [i]
          //         total = total + p[0]
          //         i = i + 1
          //
          // the loop's own release of `i` on the EXIT edge names the block
          // argument, which aliases the marker; read as this token's second
          // discharge it reported "released or transferred more than once" for
          // a program whose refcounts balance (checked with the leak gate at
          // 200k iterations: net 0).
          if (consumes && !releasedUseDominanceDisabled() &&
              !walk.producerDominates(resource.producer, op))
            consumes = false;
          if (consumes) {
            if (outstanding() == 0 && resource.condition) {
              op = op->getNextNode();
              continue;
            }
            if (outstanding() == 0) {
              if (ownershipPathTraceEnabled()) {
                llvm::errs() << "[ownership-path] double consume of "
                             << resource.producerLabel << " (produced in ^bb"
                             << blockOrdinal(resource.producer->getBlock())
                             << ") by " << call.getCallee() << " in ^bb"
                             << blockOrdinal(state.block)
                             << ", borrowed=" << state.borrowedRetains.size()
                             << " exceptional=" << state.exceptional
                             << ", path=";
                for (unsigned ordinal : state.trail)
                  llvm::errs() << "^bb" << ordinal << ">";
                llvm::errs() << "\n";
              }
              return call.emitError()
                     << "owned resource from " << resource.producerLabel
                     << " result " << resource.resultOffset
                     << " is released or transferred more than once on one CFG "
                        "path";
            }
            spendOutstanding();
            op = op->getNextNode();
            continue;
          }
          // ⛔ The dominance guard the non-call arm below already carries,
          // and for the same reason: a back edge reaches blocks the producer
          // does not dominate, and a mention there names the NEXT iteration's
          // value, not the released one.
          //
          //     while i < n:
          //         p = [i]
          //         i = p[0] + 1
          //
          // `p[0]` folds to `i` itself, so the element token's marker is an
          // identity cast of the loop-carried argument and aliases it. Its
          // release at the bottom of the body was then read as still in force
          // at the top, where `i < n` mentions the argument -- a refusal of a
          // program whose refcounts balance.
          if (mentionsTracked &&
              (groupContainsOperand(op, state.group, aliases) ||
               groupContainsOperand(op, state.views, aliases)) &&
              outstanding() == 0 &&
              (releasedUseDominanceDisabled() ||
               walk.producerDominates(resource.producer, op)))
            return call.emitError()
                   << "released owned resource from " << resource.producerLabel
                   << " is used after release (by call to '"
                   << call.getCallee() << "')";
          if (retains) {
            // A retain seen after the token was already released is a
            // resurrection, and it is the only thing that makes a later use
            // legal. Charging it to a parent instead would let the holder's
            // release cancel it, so it stays a plain retain here even when it
            // carries a parent link.
            //
            // ⛔ Except a merge borrow, which the Owned arm below already
            // counts separately for the same reason: it is the loop edge's
            // own pin, not a new reference, and it recurs every iteration.
            // Counting it here made `state.retained` -- part of the visited
            // key -- climb without bound, so the fixpoint never closed:
            //
            //     lo: int = 0
            //     for x in [4, -2, 9]:
            //         if x < lo:
            //             lo = x
            //
            // reached retained=1818 and hit the 20000-state cap. The comment
            // on that cap says a rise in it must be read as a possible masked
            // finding rather than a limit; this is the same shape it names,
            // and it was masking one: with the walk converging, the program
            // compiles and LEAKS 52 B whenever the rebound name is the loop
            // ELEMENT (`lo = x`), while an accumulator that never takes the
            // element (`n = n + 1`) is clean. That is why this golden is not
            // in the leak gate -- the value it pins is right, and the leak it
            // exposes is separate and open.
            //
            // Localised, for whoever takes it: `refcount-insertion` emits one
            // retain MORE for the conditional shape than for the
            // unconditional one (4 vs 3 on a two-element str list) while both
            // emit five releases. The extra one carries no ownership
            // attribute and sits immediately before an
            // `aggregate_release = "...:py.decref"` of a different value, so
            // the pair reads as balanced locally while the merge borrow it
            // was meant to answer keeps its reference. int lists leak the
            // same way from three elements up; two-element int lists are
            // clean, which is why the smallest reproducer is a str list.
            //
            // Traced to the emission, for whoever takes it. The loop body ends
            // with three ops on the same entity:
            //
            //     %56 = select %cond, %51, %42          ; the next iteration's
            //     Ly_IncRef(%56) {block-arg-merge-borrow}
            //     Ly_IncRef(%42)                         ; edge retain
            //     LyUnicode_DecRef(%42) {..:py.decref}
            //     cf.br ^bb4(%56, ...)                   ; %42 is ^bb4's arg
            //
            // The last two cancel, so the block looks balanced; the merge
            // borrow on `%56` is what has no discharge. The edge retain is
            // added because `emitterLaneIncrefInBlock` looks for a retain
            // labelled `":py.incref"` to credit as the transfer, and the one
            // standing there carries no label -- this pass emitted it, not the
            // emitter. So the credit search misses a retain that is already
            // paying, and the borrow is charged a second time.
            // A merge borrow after the release is the loop edge re-pinning a
            // name this path has already given up: it is neither a
            // resurrection nor an outstanding borrow to discharge, because
            // the release that would cancel it (`callConsumesStaleValue`
            // above) is behind us on this path. Counting it in EITHER
            // component makes that component climb once per iteration, and
            // both are part of the visited key -- retained=1818 first, then
            // borrowed=1818 when only the retain side was excluded.
            if (!isBlockArgMergeBorrowRetain(call))
              ++state.retained;
          }
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
          if (isBlockArgMergeBorrowRetain(call)) {
            if (!llvm::is_contained(state.borrowedRetains, call.getOperation()))
              state.borrowedRetains.push_back(call.getOperation());
          } else if (slotAbsorptionRetain) {
            if (!llvm::is_contained(state.slotParents, *slotParent))
              state.slotParents.push_back(*slotParent);
          }
          else if (parkedWithoutAName) {
            // Parked in a holder this walk cannot name. It stays in
            // `retained` -- it really does keep the object alive, and every
            // rule that asks that question must keep seeing it -- but it is
            // the holder's, so the owned-return rule subtracts it.
            //
            // Charged once per PATH, not once per traversal: see `parkedOps`.
            if (!llvm::is_contained(state.parkedOps, call.getOperation())) {
              state.parkedOps.push_back(call.getOperation());
              ++state.retained;
              ++state.parkedUnnamed;
            }
          } else {
            ++state.retained;
          }
        }
        const OwnershipWalkCache::RaiseFacts &raise = walk.raiseFacts(call);
        if (raise.raiseLike) {
          // An unguarded raise (no preceding call-site marker wiring it to
          // an in-function handler) unwinds OUT of the function: a token
          // still owned here escapes with no path left to release it. A
          // guarded raise reaches its handler through the marker edge
          // enqueued above. Exceptional paths are exempt: a handler-miss
          // rethrow can be reached from marker edges on BOTH sides of the
          // token's normal-path release, and the handler's match path may
          // still use the token -- a state no single statically-placed
          // release can discharge (documented residual, matching what the
          // insertion pass skips via its handler-use check).
          if (state.token == AffineTokenState::Owned && !state.exceptional &&
              !resource.condition && !walk.guardedByCallSiteMarker(call) &&
              !walk.aliasesAnyRoot(state.group, ambiguousRetainRoots))
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
        // An unguarded MAY-RAISE call in a frame without a local handler
        // also exits the function when it unwinds, with every held token.
        // Guarded calls reach their handler through the marker edge; a call
        // that consumes the group consumes it on the unwind edge too (the
        // transfer happens DURING the call); calls nested inside
        // single-block regions stay outside the model (the insertion pass
        // cannot wire cleanup there either -- known residual, never a
        // silent acceptance of NEW leak shapes).
        if (modelMayRaiseUnwindExits &&
            state.token == AffineTokenState::Owned && !consumes &&
            !resource.condition && raise.mayRaise &&
            callInFunctionTopLevelRegion(call) &&
            !walk.guardedByCallSiteMarker(call) &&
            !walk.aliasesAnyRoot(state.group, ambiguousRetainRoots))
        {
          // The other half of LYTHON_TRACE_UNWIND_HOLD (see the insertion pass):
          // this prints the group the verifier believes is held where the pass
          // placed no cleanup.
          static const bool traceHold =
              std::getenv("LYTHON_TRACE_UNWIND_HOLD") != nullptr;
          if (traceHold)
            // The RESOURCE's own head value, not just the walk's current
            // group: the insertion pass prints the group it tracks, and the
            // walk remaps its group as it goes, so without this the two lines
            // the instrument exists to pair up cannot be matched to each other.
            llvm::errs() << "[verify] holds " << resource.producerLabel
                         << " res0 "
                         << (resource.values.empty()
                                 ? static_cast<const void *>(nullptr)
                                 : resource.values.front().getAsOpaquePointer())
                         << " at " << call.getCallee() << " op "
                         << static_cast<const void *>(call.getOperation())
                         << " group0 "
                         << (state.group.empty()
                                 ? static_cast<const void *>(nullptr)
                                 : state.group.front().getAsOpaquePointer())
                         << "\n";
          return call.emitError()
                 << "owned resource from " << resource.producerLabel
                 << " result " << resource.resultOffset
                 << " is still owned when a call to '" << call.getCallee()
                 << "' may unwind out of the function; the unwind path "
                    "must release, transfer, or return it";
        }
      } else if (state.token == AffineTokenState::Released && mentionsTracked &&
                 groupContainsOperand(op, state.group, aliases) &&
                 outstanding() == 0 &&
                 (releasedUseDominanceDisabled() ||
                  walk.producerDominates(resource.producer, op))) {
        return op->emitError()
               << "released owned resource from " << resource.producerLabel
               << " is used after release";
      }

      if (op->getNumRegions() != 0 &&
          enqueueRegionEntryPaths(op, state, aliases, walk, worklist)) {
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

    AnchorTrueEdge anchor =
        anchorTrueEdgeOf(walk, contracts, op, state, aliases);
    for (unsigned index = 0; index < successors; ++index) {
      mlir::Block *successor = op->getSuccessor(index);
      AffineTokenState nextToken = state.token;
      unsigned nextRetained = state.retained;
      llvm::SmallVector<std::int64_t, 2> nextSlotParents = state.slotParents;
      bool nextExceptional =
          state.exceptional || (anchor.isVirtualUnwind && index == 0);
      if (anchor.consumesGroup && index == 0) {
        if (nextToken == AffineTokenState::Owned ||
            nextToken == AffineTokenState::Conditional)
          nextToken = AffineTokenState::Released;
        else if (nextRetained > 0)
          --nextRetained;
        else if (!nextSlotParents.empty())
          nextSlotParents.pop_back();
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
      // Views name the same object as the group, so they follow the same
      // rename: a view left under its pre-edge name would be read as the
      // previous iteration's token on the next trip round a loop. A view the
      // edge does not forward has no name in the successor once the group was
      // renamed, so it drops rather than lingering as a stale alias.
      llvm::SmallVector<bool, 4> viewMask;
      llvm::SmallVector<mlir::Value, 4> remappedViews = remapGroupForSuccessor(
          op, index, successor, state.views, aliases, &viewMask);
      llvm::SmallVector<mlir::Value, 4> mappedViews;
      for (auto [viewIndex, view] : llvm::enumerate(remappedViews)) {
        if (renamed && viewIndex < viewMask.size() && !viewMask[viewIndex])
          continue;
        auto argument = mlir::dyn_cast_if_present<mlir::BlockArgument>(view);
        if (argument && argument.getOwner() == successor &&
            (viewIndex >= viewMask.size() || !viewMask[viewIndex]))
          continue;
        mappedViews.push_back(view);
      }
      AffinePathState next{successor,   firstOperation(successor),
                           nextToken,   nextRetained,
                           std::move(mappedGroup),
                           std::move(mappedStale),
                           std::move(mappedPrevious),
                           std::move(mappedViews),
                           state.borrowedRetains,
                           nextExceptional};
      // Parked slot retains are keyed by the HOLDER's identity, so unlike
      // group/stale/previous/views they need no rename across the edge: the
      // container is the same allocation on both sides of it.
      next.slotParents = std::move(nextSlotParents);
      // Same reasoning as slotParents: the holder is the same allocation on
      // both sides of the edge, so the charge crosses it unrenamed.
      next.parkedUnnamed = std::min(state.parkedUnnamed, nextRetained);
      // Follows `parkedUnnamed`: the ops it names are the charges that number
      // counts, so an edge that keeps the charge must keep its provenance or
      // the same retain would be charged again on the far side.
      next.parkedOps = state.parkedOps;
      next.trail = state.trail;
      worklist.push_back(std::move(next));
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
  resource.values = std::move(group);
  resource.views = std::move(views);
  resource.condition = condition;
  resources.push_back(std::move(resource));
}

llvm::SmallVector<TrackedResource, 16>
collectTrackedResources(mlir::ModuleOp module, mlir::SymbolTable &symbols,
                        mlir::func::FuncOp function,
                        llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                        own::AliasAnalysis &aliases,
                        FuncContractCache &contracts,
                        const own::ReferenceMap &references) {
  llvm::SmallVector<TrackedResource, 16> resources;
  // ⛔ FROM THE BIRTH LANES, not the advanced ones. A payload re-root replaces
  // a group's lanes while leaving the obligation alone, so asking the map about
  // an advanced lane names the MUTATION's reference instead of this one's --
  // `cross_except_star_views` fails on exactly that. The reference is a property
  // of the producer, so it is taken from the producer.
  auto noteReference = [&](TrackedResource &resource, mlir::Value birth) {
    resource.reference = references.of(birth);
  };
  // Same lane advance the insertion pass performs, from the same shared
  // helper: insertion releases the CURRENT lanes, so a verifier still tracking
  // the birth lanes would read that release as naming another entity and
  // report the resource as never released. "If insertion and verification
  // disagree on roots, the proof is void" (rfc/memory-safety-proof.md) --
  // which is exactly why the advance lives in common/ and not in either pass.
  auto advance = [&](own::ResourceGroup &group) {
    own::advanceGroupLanesThroughReRoots(contracts, function, group, aliases);
  };
  // See `TrackedResource::hasOwnNamedRelease`: without a release of its own the
  // foreign one is the only release this resource has, and disowning it would
  // report a leak that is not there.
  auto noteOwnNamedRelease = [](TrackedResource &resource,
                                const own::RuntimeDeallocator *deallocator) {
    if (!resource.ownsReference || !deallocator)
      return;
    llvm::StringRef name = mlir::func::FuncOp(deallocator->function).getName();
    for (mlir::Value value : resource.values)
      for (mlir::OpOperand &use : value.getUses())
        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(use.getOwner()))
          if (call.getCallee() == name)
            resource.hasOwnNamedRelease = true;
  };

  function.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(own::kOwnedLocalObjectAttr))
      return;
    for (own::ResourceGroup group :
         own::collectOwnedLocalObjectGroups(op, deallocators)) {
      mlir::Value birth = group.values.empty() ? mlir::Value{}
                                               : group.values.front();
      advance(group);
      appendTrackedResource(resources, function, op, group.offset,
                            std::move(group.values), group.condition,
                            std::move(group.views));
      // Whether this marker mints is `isMinted(reference)` now, so the only
      // thing left to record is that a marker CAN own one at all.
      resources.back().ownsReference =
          own::ownedLocalMarkerIsRetainRooted(op, aliases);
      noteReference(resources.back(), birth);
      noteOwnNamedRelease(resources.back(), group.deallocator);
      return;
    }
    if (!op->hasAttr(own::kOwnedLocalObjectContractAttr) &&
        !mlir::isa<mlir::func::CallOp>(op) && op->getNumResults() != 0 &&
        own::isObjectHeaderLikeType(op->getResult(0).getType())) {
      // Named `values` like `own::ResourceGroup` and `UnwindTrackedGroup`: three
  // structs calling one thing by two names is how they read as three
  // different models of a resource when they are one.
  llvm::SmallVector<mlir::Value, 4> group;
      group.push_back(op->getResult(0));
      appendTrackedResource(resources, function, op, /*offset=*/0,
                            std::move(group));
    }
  });

  function.walk([&](mlir::func::CallOp call) {
    for (own::ResourceGroup group :
         own::collectOwnedCallResultGroups(module, call, deallocators,
                                           &symbols)) {
      mlir::Value birth = group.values.empty() ? mlir::Value{}
                                               : group.values.front();
      advance(group);
      appendTrackedResource(resources, function, call.getOperation(),
                            group.offset, std::move(group.values),
                            group.condition, std::move(group.views));
      resources.back().ownsReference = own::perReferenceReleaseLabels();
      noteReference(resources.back(), birth);
      noteOwnNamedRelease(resources.back(), group.deallocator);
    }
  });

  return resources;
}

mlir::LogicalResult verifyFunctionAffineOwnership(
    mlir::ModuleOp module, mlir::SymbolTable &symbols,
    mlir::func::FuncOp function,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases, FuncContractCache &contracts,
    const own::ReferenceMap &references) {
  // The may-raise unwind-exit obligation only applies where the insertion
  // pass can pair it with marker wiring the final EH phase materializes:
  // functions with a Python source location, outside runtime-internal
  // pre-lowered modules (whose artifacts link after the EH phase). Anything
  // else is a documented residual, not a silent acceptance elsewhere.
  bool modelMayRaiseUnwindExits =
      !module->hasAttr(own::kRuntimeInternalLoweringAttr) &&
      findPythonSourceLoc(function.getLoc()).has_value();
  llvm::SmallVector<TrackedResource, 16> resources = collectTrackedResources(
      module, symbols, function, deallocators, aliases, contracts,
      references);
  llvm::SmallVector<BorrowedEntryResource, 8> borrowedEntryResources =
      collectBorrowedEntryResources(function, deallocators);

  if (isSingleBlockStraightLineFunction(function) &&
      borrowedEntryResources.empty()) {
    bool allResourcesHandled = true;
    for (TrackedResource &resource : resources) {
      std::optional<mlir::LogicalResult> result = verifyStraightLineResource(
          contracts, resource, deallocators, aliases, references);
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

  llvm::DenseSet<mlir::Value> ambiguousRetainRoots = aliasRootsOf(
      collectUnwindAmbiguousRetainOperands(function, contracts), aliases);
  // One cache for every resource of this function: the per-op facts it holds
  // are the same for all of them, and a per-resource cache would rebuild the
  // operand inverse once per resource (the very quadratic term being removed).
  OwnershipWalkCache walk(function, contracts, aliases, handlerEntries);
  for (TrackedResource &resource : resources)
    if (mlir::failed(verifyResourceOnCFGPaths(
            contracts, resource, deallocators, aliases, references, walk,
            modelMayRaiseUnwindExits, ambiguousRetainRoots)))
      return mlir::failure();

  for (BorrowedEntryResource &resource : borrowedEntryResources)
    if (mlir::failed(verifyBorrowedEntryOnCFGPaths(contracts, resource,
                                                   deallocators, aliases, walk)))
      return mlir::failure();

  return mlir::success();
}

mlir::LogicalResult verifyPathSensitiveAffineOwnership(
    mlir::ModuleOp module, llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases) {
  FuncContractCache contracts(module);
  // The other analysis, beside the one it is not: `aliases` answers "same
  // entity", this answers "same reference". The placer builds the same map from
  // the same facts -- if insertion and verification disagreed about which
  // reference a release discharges, the proof would be void
  // (rfc/memory-safety-proof.md).
  const own::ReferenceMap references(contracts, aliases);
  // One table for the whole module walk: resolving callees per call op through
  // the module's symbol list is what made resource collection O(calls x
  // symbols).
  mlir::SymbolTable symbols(module);
  return walkVerify<mlir::func::FuncOp>(
      module, [&](mlir::func::FuncOp function) {
        if (own::isRuntimeManifestFunction(function))
          return mlir::success();
        return verifyFunctionAffineOwnership(module, symbols, function,
                                             deallocators, aliases, contracts,
                                             references);
      });
}

// Generator frames (rfc/stdlib-semantics.md R3): an Own(rho) crossing a
// yield is absorbed by the generator object (frame lanes) or transferred to
// the resumer (value lanes); both are ownership effects and must cross the
// suspension boundary through materialized contracts, never as raw words.
// The judgment this rule pins (rfc/memory-safety-proof.md, Boundary
// Contracts): a resource may be introduced across the resume boundary only
// by a verifier-visible lane contract. Concretely, a resume clone's
// signature may contain non-NonObject lanes only when
//   (a) the clone materializes `ly.generator.suspend_lanes` describing each
//       object-family result range (contract, begin, size), and
//   (b) every such range's begin offset is declared in the clone's
//       ly.ownership.owned_results contract — the transfer(rho, resumer)
//       effect the refcount inserter and this verifier both consume.
// Anything else would be an Own smuggled across a suspension with no phase
// inserting its release, so it is rejected here rather than silently
// leaked. Generator ARGUMENT lanes are the one borrow exception:
// `ly.generator.borrowed_args` ranges cross without a transfer anchor
// because the frame absorbed their ownership at creation (the creation site
// retains each object argument into the storage words and the drop
// finalizer releases it), so the resume-time span is a borrow against a
// frame-owned rho — no resource is introduced at this boundary.
mlir::LogicalResult verifyGeneratorResumeFrames(mlir::ModuleOp module) {
  return walkVerify<mlir::func::FuncOp>(
      module, [&](mlir::func::FuncOp function) {
        if (!function->hasAttr("ly.generator.resume"))
          return mlir::success();
        auto isNonObjectLane = [](mlir::Type type) {
          return type.isInteger(64) || type.isInteger(1) || type.isIndex();
        };
        mlir::FailureOr<own::FunctionContract> contract =
            own::readFunctionContract(function);
        if (mlir::failed(contract))
          return mlir::failure();

        // Lane contract reader: each object-family range must name its
        // contract and be anchored by the given ownership index set (the
        // transfer effect the refcount inserter consumes on the same side).
        auto collectLanes =
            [&](llvm::StringRef attrName, const own::IndexSet *anchors,
                llvm::StringRef anchorLabel,
                llvm::SmallVectorImpl<std::pair<std::int64_t, std::int64_t>>
                    &ranges) -> mlir::LogicalResult {
          auto lanes = function->getAttrOfType<mlir::ArrayAttr>(attrName);
          if (!lanes)
            return mlir::success();
          for (mlir::Attribute entry : lanes) {
            auto dict = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
            if (!dict) {
              function.emitError() << "generator lane contract entry in "
                                   << attrName << " is not a dictionary";
              return mlir::failure();
            }
            auto laneContract = dict.getAs<mlir::StringAttr>("contract");
            auto begin = dict.getAs<mlir::IntegerAttr>("begin");
            auto size = dict.getAs<mlir::IntegerAttr>("size");
            if (!laneContract || laneContract.getValue().empty() || !begin ||
                !size) {
              function.emitError() << "generator lane contract entry in "
                                   << attrName
                                   << " is missing contract/begin/size";
              return mlir::failure();
            }
            ranges.push_back(
                {begin.getInt(), begin.getInt() + size.getInt()});
            if (anchors && size.getInt() > 0 &&
                laneContract.getValue() != "types.NoneType" &&
                !anchors->contains(static_cast<unsigned>(begin.getInt()))) {
              function.emitError()
                  << "generator lane at " << attrName << " offset "
                  << begin.getInt() << " is not covered by the "
                  << anchorLabel << " ownership contract";
              return mlir::failure();
            }
          }
          return mlir::success();
        };
        auto checkCoverage =
            [&](llvm::ArrayRef<mlir::Type> types,
                llvm::ArrayRef<std::pair<std::int64_t, std::int64_t>> ranges,
                llvm::StringRef label) -> mlir::LogicalResult {
          for (auto [index, laneType] : llvm::enumerate(types)) {
            if (isNonObjectLane(laneType))
              continue;
            std::int64_t position = static_cast<std::int64_t>(index);
            bool covered = llvm::any_of(
                ranges, [&](std::pair<std::int64_t, std::int64_t> range) {
                  return position >= range.first && position < range.second;
                });
            if (!covered) {
              function.emitError()
                  << "generator resume " << label << " type " << laneType
                  << " is not a NonObject lane; an owned value may not "
                     "cross a suspension boundary without a generator "
                     "frame release contract";
              return mlir::failure();
            }
          }
          return mlir::success();
        };

        mlir::FunctionType type = function.getFunctionType();
        // Resume arguments: frame lanes transfer INTO the clone
        // (transfer_args); suspend results transfer OUT (owned_results).
        llvm::SmallVector<std::pair<std::int64_t, std::int64_t>, 4> argRanges;
        if (mlir::failed(collectLanes("ly.generator.resume_args",
                                      &contract->transferArgs, "transfer-args",
                                      argRanges)))
          return mlir::failure();
        // Borrowed argument lanes: no anchor — the frame owns the resource
        // (retained at creation, released by the drop finalizer), and the
        // resume span is a borrow against it.
        if (mlir::failed(collectLanes("ly.generator.borrowed_args",
                                      /*anchors=*/nullptr, "borrowed-args",
                                      argRanges)))
          return mlir::failure();
        if (mlir::failed(
                checkCoverage(type.getInputs(), argRanges, "frame argument")))
          return mlir::failure();
        llvm::SmallVector<std::pair<std::int64_t, std::int64_t>, 4>
            resultRanges;
        if (mlir::failed(collectLanes("ly.generator.suspend_lanes",
                                      &contract->ownedResults, "owned-results",
                                      resultRanges)))
          return mlir::failure();
        return checkCoverage(type.getResults(), resultRanges,
                             "suspend result");
      });
}

mlir::LogicalResult
verifyFuncCallOwnershipContractsImpl(mlir::ModuleOp module) {
  {
    py::PerfScope perf("func-call-ownership.generator-frames");
    if (mlir::failed(verifyGeneratorResumeFrames(module)))
      return mlir::failure();
  }
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
