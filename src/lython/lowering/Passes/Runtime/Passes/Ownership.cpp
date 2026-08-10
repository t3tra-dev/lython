#include "Ownership.h"
#include "Reference.h"
#include "Common/Instrumentation.h"
#include "Common/PythonSourceRange.h"
#include "Common/RuntimeSupport.h"
#include "PyDialectTypes.h"
#include "Runtime/ABI/EntityHeaderPrefix.h"
#include "Runtime/Model/Contracts.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Process.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>

namespace py::lowering {
namespace {

namespace own = py::ownership;

// `LYTHON_OWNERSHIP_TRACE_TRANSFERS=1` reports, per owned group, the
// branch-level transfers this pass found and whether each destination argument
// group survived the soundness fixpoint.
//
// Why this is worth keeping rather than deleting with the investigation that
// needed it: a contract whose mutation primitives stop consuming their receiver
// changes NOTHING about the diagnostics -- it changes which branches forward a
// group -- so the symptom of getting that wrong is a group that silently never
// becomes a candidate, and the failure surfaces two phases later as a verifier
// complaint naming the producer. Six of those cost a day to trace back to a
// zero here. `seeded=0` says it in one line. Same reason
// LYTHON_OWNERSHIP_ROOT_PARITY is env-gated rather than absent.
bool ownershipTransferTraceEnabled() {
  static bool enabled = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_OWNERSHIP_TRACE_TRANSFERS");
    return value && !value->empty() && *value != "0";
  }();
  return enabled;
}

// LYTHON_ABLATE_UNWIND_DEATH_DELAY=1 restores the placement that wrote a
// group's normal-path death release before a later call site of the same try
// whose unwind reaches a handler-entry release of that group (see
// `delayPastUnwindingCallSites`).
//
// Why an ablation switch rather than nothing: the repair is a MOVE of a release
// within one block, so "the fix is in" is not visible in any counter -- only in
// the IR. This hatch lets the golden sentinel (`redcheck.py --sentinel`) and the
// LLVM-IR differential take their "before" side from the SAME binary as their
// "after" side, which is the only way to attribute an IR change to this
// placement rather than to a rebuild.
//
// Why setting it is safe to ship: it makes the compiler emit the release EARLIER,
// which is the shape the affine verifier rejects. A mistake with this variable
// set therefore costs a refusal, never a silently shipped double free.
bool unwindDeathDelayEnabled() {
  static bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_ABLATE_UNWIND_DEATH_DELAY");
    return !(value && !value->empty() && *value != "0");
  }();
  return enabled;
}


// LYTHON_OWNERSHIP_TRACE_PLACEMENT=1 names, for every owned call-result group,
// which of the four placement strategies in `insertOwnedResultReleases` took it
// -- including the strategy "none", which is the one that matters: a group no
// strategy claims gets NO release, and the affine verifier then reports it as
// reaching function exit unreleased, naming the producing call rather than the
// placement hole.
//
// Why a trace and not a counter: the question this answers is not "how many"
// but "which strategy declined, for which group", and two variants of one
// program that differ only in whether the element is consumed in the body take
// different strategies. A count cannot distinguish them.
//
// Why NOT infer the strategy from the emitted IR instead: the absence of a
// release is exactly what several DIFFERENT strategies produce on decline, so
// reading the output cannot say which one was asked. Printing what was examined
// rather than what was concluded is the only form that separates them.
bool ownershipPlacementTraceEnabled() {
  static bool enabled = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_OWNERSHIP_TRACE_PLACEMENT");
    return value && !value->empty() && *value != "0";
  }();
  return enabled;
}

void tracePlacement(llvm::StringRef strategy, mlir::func::CallOp call,
                    const own::ResourceGroup &group) {
  if (!ownershipPlacementTraceEnabled())
    return;
  llvm::errs() << "[ownership-placement] strategy=" << strategy
               << " callee=" << call.getCallee() << " offset=" << group.offset
               << " lanes=" << group.values.size()
               << " conditional=" << (group.condition ? 1 : 0) << "\n";
}

// LYTHON_ABLATE_OWNERSHIP_SYMBOL_TABLE=1 restores the callee resolution this
// file used before the symbol table below was threaded through it: each
// `collectOwnedCallResultGroups` re-resolved `call.getCallee()` with
// `ModuleOp::lookupSymbol`, which scans the module's symbol list.
//
// Why an ablation switch rather than nothing: this change removes WORK and must
// not change a single instruction, and "the IR is identical" is only evidence
// when both sides come from one binary -- a separate `before` build re-proves
// the build, not the change. The two arms differ in which lookup answers, so if
// the table's scope (immediate symbol children of the module) ever disagreed
// with `lookupSymbol`'s, the differential would show it as an IR change instead
// of hiding it as a silently different callee.
//
// Why setting it is safe to ship: it is strictly slower and otherwise identical,
// so a mistake with this variable set costs compile time, never a wrong release.
bool ownershipSymbolTableDisabled() {
  static bool disabled = [] {
    auto value =
        llvm::sys::Process::GetEnv("LYTHON_ABLATE_OWNERSHIP_SYMBOL_TABLE");
    return value && !value->empty() && *value != "0";
  }();
  return disabled;
}

// The SHAPE of the callee-resolution work, printed beside the phase's duration
// under LYTHON_PERF: `symbols` is the factor the old code paid on every one of
// `calls` resolutions, so the pair states the product that was being formed.
// Both numbers vary with the input, which is what makes them a measurement of
// the input rather than of the compiler.
// Immediate symbol children of the module -- the list `ModuleOp::lookupSymbol`
// scans. Counted, not estimated, because the whole claim about this phase is
// that this number is a FACTOR, and a factor asserted from the source rather
// than read off the input is how "#markers x #groups" was mis-attributed once
// already.
std::uint64_t moduleSymbolCount(mlir::ModuleOp module) {
  std::uint64_t symbols = 0;
  for (mlir::Operation &op : module.getBodyRegion().front())
    if (mlir::isa<mlir::SymbolOpInterface>(&op))
      ++symbols;
  return symbols;
}

// Why the ablation state is printed and not just assumed: the hatch below changes
// only how fast a callee is resolved, so its effect is invisible in the IR BY
// DESIGN -- which makes "both arms produced identical IR" consistent with "the
// ablation never fired" as well as with "the change is inert". An A/B whose arms
// cannot be told apart is not an A/B. This line is what distinguishes them.
void reportOwnershipWorkShape(llvm::StringRef scope, std::uint64_t symbols,
                              std::uint64_t calls) {
  static const bool on = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_PERF");
    return value && (*value == "1" ||
                     llvm::StringRef(*value).equals_insensitive("true") ||
                     llvm::StringRef(*value).equals_insensitive("yes") ||
                     llvm::StringRef(*value).equals_insensitive("on"));
  }();
  if (!on)
    return;
  llvm::errs() << "[LYTHON_PERF] " << scope << " module_symbols=" << symbols
               << " callee_resolutions=" << calls
               << " product=" << (symbols * calls) << " symbol_table="
               << (ownershipSymbolTableDisabled() ? "ABLATED" : "on") << "\n";
}

// A borrow edge is admitted on the promise that a retain balances the
// destination group's release. When the retain cannot be spelled, breaking that
// promise is SILENT -- silent to the final verifier too, which counts the
// retains and releases that are PRESENT: a dropped retain leaves each group's
// own arithmetic balanced, because the retain belongs to the argument
// reconciling two groups rather than to either of them
// (rfc/memory-safety-proof.md, third failure shape).
//
// So it is refused. The kernel's rule is that an operation whose premises
// cannot be met has NO STEP -- a branch whose ownership obligation cannot be
// discharged is a program the model does not run, not one it runs
// approximately.
//
// This was tried once BEFORE the root cause was found and it refused three
// working programs; the history is on `borrowEdgeRetainIsSpellable` below. Two
// repairs made the path unreachable and both left the emitted code byte-
// identical over 324 golden cases and examples (retain 31006, release 168445,
// zero files differing):
//
//   `ABI/RuntimeABI.cpp`  the boxing path now RECORDS the ownership it takes,
//                         so a merge fed by a box is seen as the move it is
//                         rather than as an edge needing a retain.
//   `ABI/EntityHeaderPrefix.h`  an ownership marker now answers
//                         "is the prefix stored here?" directly, instead of
//                         being walked through to the raw storage underneath.
//
// What reaching this diagnostic means for a future program: the merge really
// does need a retain (the operand outlives the branch), and the value is not
// complete at its own definition and does not say otherwise with a marker.
// The fix is to mark it where it becomes complete, not to relax this.
mlir::LogicalResult reportUnspellableBorrowEdgeRetain(mlir::Value header,
                                                      mlir::Operation *anchor) {
  return anchor->emitError()
         << "ownership: this block-argument merge needs a retain on the edge "
            "and the header prefix cannot be spelled at the point the retain "
            "must go (header type "
         << header.getType() << ", "
         << (mlir::isa<mlir::BlockArgument>(header) ? "block argument"
                                                    : "op result")
         << "). The retain has to precede any release in the same block, so it "
            "is placed at the header's definition -- and there the entity's "
            "refcount/class words are not yet stored. An entity that is "
            "complete at its definition says so with "
         << own::kOwnedLocalObjectAttr
         << "; this one does not, so emitting the branch would leave the "
            "merged value's release unbalanced.";
}

using own::CachedFuncContract;
using own::FuncContractCache;
using own::ancestorInBlock;
using own::callConsumesGroup;
using own::callPartiallyConsumesGroup;
using own::callRetainsGroup;
using own::groupContainsOperand;
using own::isBlockArgMergeBorrowRetain;
using own::remapGroupThroughValueMapping;
using own::returnTransfersGroup;

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

// Take a reference immediately before `anchor`. Two callers: a borrowed value
// on its way out through a return, and a transfer whose source is read after
// the call -- the same operation, so the same body.
mlir::LogicalResult insertRetain(mlir::func::FuncOp retain,
                                 mlir::Operation *anchor, mlir::Value header) {
  if (!retain)
    return anchor->emitError()
           << "taking a reference here requires a runtime retain primitive";
  if (retain.getFunctionType().getNumInputs() != 1)
    return retain.emitError()
           << "runtime retain primitive must accept one object header";

  mlir::OpBuilder builder(anchor);
  mlir::FailureOr<mlir::Value> headerView = buildRetainHeaderView(
      builder, anchor->getLoc(), header, retain.getFunctionType().getInput(0));
  if (mlir::failed(headerView))
    return anchor->emitError() << "cannot build object retain view";

  mlir::func::CallOp::create(builder, anchor->getLoc(), retain, *headerView);
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
          if (mlir::failed(insertRetain(retain, returnOp.getOperation(), group.front()))) {
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
        !own::groupMatchesValues(call.getOperands(), offset, group, aliases))
      return true;
  for (unsigned offset : (*cached)->contract.transferArgs.values)
    if (consumesHeaderAt(offset) &&
        !own::groupMatchesValues(call.getOperands(), offset, group, aliases))
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

// AN AGGREGATE RELEASE CANNOT BE A LOCAL TOKEN'S DEATH. `aggregate_release`
// marks the discharge of an `aggregate(parent, path)` resource -- a slot's or a
// literal source's token, owned by the container (rfc/memory-safety-proof.md,
// Aggregates). When the walked group is a retain-rooted owned-local marker
// (own::ownedLocalMarkerIsRetainRooted) the object carries BOTH that token and
// the one the retain minted, and they alias, so counting the aggregate release
// as this token's death leaves the mint with no release at all: one element
// object leaked per execution of every container literal whose element is read
// back (`ys = [99]; total += ys[0]`, 64 B/iteration, unbounded).
//
// Why NOT treat every foreign-named consume that way (measured, 2026-07-28): a
// BARE release names some local token and this walk cannot tell which, so the
// only safe reading is "this one". Skipping bare ones too aborted
// `golden.cases.dict_methods_complete` with `Ly_DecRef observed non-positive
// refcount` -- shapes where the mint's release is already placed under the
// head's advanced lanes got a second one. The narrow rule's failure direction is
// a bounded leak on paths where a bare release is the only competing consume;
// the wide rule's is a double free.
bool consumeIsAggregateRelease(mlir::OpOperand &use) {
  return use.getOwner()->hasAttr(own::kAggregateReleaseAttr) ||
         (own::perReferenceReleaseLabels() &&
          use.getOwner()->hasAttr(own::kReferenceReleaseAttr));
}

// Emit `group`'s release at the builder's insertion point.
//
// `ownsReference` is the claim `own::kReferenceReleaseAttr` records: true when
// the group holds an increment of its own, so this release discharges exactly
// the reference its operands name.
//
// Not stamped when the label is ablated: an attribute is still IR, and two
// otherwise identical release calls that differ by one stop being
// interchangeable downstream -- with every reader off, the stamp ALONE changed
// what `loop_iterator_element_into_container_literal` compiled to, which would
// have made the A/B measure the wrong thing.
mlir::func::CallOp emitGroupRelease(mlir::OpBuilder &builder, mlir::Location loc,
                                    const own::ResourceGroup &group,
                                    mlir::ValueRange values,
                                    bool ownsReference) {
  auto call = mlir::func::CallOp::create(builder, loc,
                                         group.deallocator->function, values);
  if (ownsReference && own::perReferenceReleaseLabels())
    call->setAttr(own::kReferenceReleaseAttr, builder.getUnitAttr());
  return call;
}

// IS THIS NAME NOT ONE OF THE WALKED GROUP'S OWN?
//
// One condition of the death test, split out so it can be replaced -- and
// measured -- on its own. The test around it is unchanged: a consume is another
// reference's discharge when it carries `aggregate_release`, the walked group
// mints, AND the name it was reached through is not the group's.
//
// The map answers "not mine" BY REFERENCE where it can name the value, which the
// SSA containment test it replaces could not: a cast of one of my names is a
// different spelling of the same reference, and containment called it another's.
//
// ⛔ WHERE THE MAP CANNOT NAME THE VALUE, THE ANSWER IS THE OLD TEST, not
// `false`. "No claim" means the caller keeps its previous reading, and the
// previous reading HERE was containment -- the liveness walk's previous reading
// was the opposite, which is why that site and this one fall back differently.
// Reading no-claim as "mine" fails six cases; as containment, none.
bool isNotOwnName(const own::ReferenceMap &references,
                  llvm::ArrayRef<mlir::Value> group, mlir::Value equivalent,
                  own::Reference mine) {
  if (own::Reference denoted = references.of(equivalent))
    return !mine || denoted != mine;
  return !llvm::is_contained(group, equivalent);
}

// IS THIS CONSUME SOMEBODY ELSE'S DISCHARGE?
//
// The composition all four walks want, spelled once. Three independent
// properties, each its own predicate:
//
//   1. this group holds an increment of its own, so a release that is not its
//      own is possible at all (`isMinted`);
//   2. the release names the CONTAINER's obligation rather than a local token
//      (`consumeIsAggregateRelease`);
//   3. the name it was reached through is not this group's (`isNotOwnName`).
//
// Named after all four sites had been migrated one property at a time and
// measured, not before: an earlier attempt to introduce this wrapper up front
// hid which property owned which failure and cost six red cases
// (rfc/test-suite-debt.md).
bool consumeIsAnotherReferencesDischarge(const own::ReferenceMap &references,
                                         mlir::OpOperand &use,
                                         llvm::ArrayRef<mlir::Value> group,
                                         mlir::Value equivalent,
                                         own::Reference mine) {
  return references.isMinted(mine) && consumeIsAggregateRelease(use) &&
         isNotOwnName(references, group, equivalent, mine);
}


// IS THIS NAME ANOTHER REFERENCE'S, SO THAT USES UNDER IT ARE NOT OUR LIVENESS?
//
// `("k", 5) in h.items()` is why the question is asked at all: the element's own
// increment is discharged by the literal's `sequence.literal.source` release,
// but the token a later retain minted on the same object keeps being USED
// afterwards, and reaching those uses through the alias relation made the
// element look live past its own death -- so it got a second release, which the
// token's missing one silently paid for.
//
// TWO properties, composed here and named separately because they are
// independent and both were measured:
//
//   1. the name denotes a DIFFERENT reference (`own::ReferenceMap`). What stood
//      here compared the defining op against a set of the module's minted
//      markers -- the same question asked of one shape only, since a cast of a
//      marker's result denotes the marker's reference and a set of ops cannot
//      say so.
//   2. that reference was MINTED rather than received
//      (`own::ReferenceMap::isMinted`). A minted one is an increment taken on
//      top of whatever else holds the entity, so it keeps the entity alive
//      independently and uses under its names are safely somebody else's. A
//      RECEIVED one may be the very reference that just died, and dropping the
//      pins under it releases early: two leak-gate members regress the moment
//      this condition goes.
//
// Composed, not folded. Collapsing the ownership walks' predicates into one
// question was tried and measured: it does not converge, because at least this
// pair and the `aggregate_release` label are orthogonal and each is load-bearing
// somewhere (rfc/test-suite-debt.md).
bool namesAnotherMintedReference(const own::ReferenceMap &references,
                                 mlir::Value name, own::Reference mine) {
  // The rule this predicate carries is the one LYTHON_ABLATE_REFERENCE_RELEASE
  // exists to A/B, so it stays behind that switch even though it no longer reads
  // the label: an ablation that stops covering half of what it names is worse
  // than none, because the arm it produces is neither behaviour.
  if (!own::perReferenceReleaseLabels())
    return false;
  own::Reference denoted = references.of(name);
  return mine && denoted && denoted != mine && references.isMinted(denoted);
}


// The buffer a retain argument was spelled from: retains take a header PREFIX,
// so the call operand is a cast or subview of the entity root, not the root.
mlir::Value retainSpellingRoot(mlir::Value value) {
  value = own::underlyingObjectValue(value);
  // Bounded: each hop moves to the view's source; the cap keeps malformed IR
  // from spinning here.
  for (unsigned step = 0; step < 8; ++step) {
    mlir::Operation *definition = value.getDefiningOp();
    if (auto cast = mlir::dyn_cast_or_null<mlir::memref::CastOp>(definition)) {
      value = cast.getSource();
      continue;
    }
    if (auto view = mlir::dyn_cast_or_null<mlir::memref::SubViewOp>(definition)) {
      value = view.getSource();
      continue;
    }
    break;
  }
  return own::underlyingObjectValue(value);
}

// ⭐ A `py.incref` the EMITTER already placed in this block for this header --
// its loop-edge token ledger paying for a lane the edge is about to fill.
//
// `t = base` inside a loop acquires a token for `t`'s lane, and the emitter
// knows it must: base's creation token stays claimed by the pre-loop local that
// still binds it, so the ledger mints one (EmitterLoops.cpp, "must not steal
// base's token"). The borrow-edge rule below then reaches the same conclusion
// from the other side -- the incoming value is owned and does not die on the
// edge -- and lends a SECOND token. One release, two retains: the entity leaked
// once per loop, 41 B for `loop_alias_carry`'s one-character str and 52 B three
// times over in `stdlib_json_build`, whose `_object_keys` dedupe is the same
// `j = m` shape hidden behind a heap int.
//
// The evidence is consumed, not just observed: `credited` holds the increfs
// already spent on another lane of the same edge, so two lanes acquiring the
// same header still get one token each.
mlir::func::CallOp
emitterLaneIncrefInBlock(mlir::Block *block, mlir::Value header,
                         mlir::func::FuncOp retainFunction,
                         const llvm::DenseSet<mlir::Operation *> &credited) {
  if (!block || !header || !retainFunction)
    return nullptr;
  mlir::Value root = retainSpellingRoot(header);
  if (!root)
    return nullptr;
  for (mlir::Operation &op : *block) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
    if (!call || call.getCallee() != retainFunction.getSymName() ||
        call.getNumOperands() != 1 || credited.contains(call.getOperation()))
      continue;
    auto label = call->getAttrOfType<mlir::StringAttr>(own::kAggregateRetainAttr);
    if (!label || !label.getValue().ends_with(":py.incref"))
      continue;
    if (retainSpellingRoot(call.getOperand(0)) == root)
      return call;
  }
  return nullptr;
}

// Does `block` LEND this group's entity to a merge argument -- i.e. does it
// contain the `block-arg-merge-borrow` retain that pays for the destination
// argument group's token?
//
// The lend is what makes a forwarding edge NOT a transfer. Without it the
// destination argument group inherits this group's token and the source is
// consumed on the edge; with it the destination has an increment of its own and
// the source keeps hers, so both are live past the merge and both need a
// cleanup on an unwinding edge.
bool blockLendsGroupToMergeArgument(mlir::Block *block,
                                    llvm::ArrayRef<mlir::Value> group,
                                    own::AliasAnalysis &aliases) {
  if (!block || group.empty())
    return false;
  for (mlir::Operation &op : *block) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
    if (!call || !own::isBlockArgMergeBorrowRetain(call) ||
        call.getNumOperands() != 1)
      continue;
    if (aliases.same(retainSpellingRoot(call.getOperand(0)), group.front()))
      return true;
  }
  return false;
}

mlir::Operation *latestUserInBlock(mlir::Operation *lhs, mlir::Operation *rhs) {
  if (!lhs)
    return rhs;
  return lhs->isBeforeInBlock(rhs) ? rhs : lhs;
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


// Does this terminator TRANSFER the group's ownership into a successor's block
// argument?
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
  // Same entity, not same lane list: two release sites for one entity merge
  // even when a re-root between them replaced the payload lanes.
  own::reportEntityRootParity("mergeReleaseInsertion", current->group,
                              next.group);
  if (!own::sameEntityRoot(current->group, next.group))
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

// An AGGREGATE release reached through a name that is not this group's does not
// end it -- it discharges the container's -- and counts only as liveness here.
// Bare releases end it either way. Whether the group may make that claim at all
// is `isMinted(mine)`: only a retain-minted token carries an increment on top of
// the container's, so only it can have a release that is not this one.
std::optional<ReleaseInsertion>
findReleaseInsertion(FuncContractCache &contracts, mlir::Operation *owner,
                     llvm::ArrayRef<mlir::Value> group,
                     llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                     own::AliasAnalysis &aliases,
                     const own::ReferenceMap &references, own::Reference mine,
                     unsigned depth = 0,
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
      aliases.namesOf(result, equivalentValues);
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
    aliases.namesOf(result, equivalentValues);

    for (mlir::Value equivalent : equivalentValues) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (user == owner)
          continue;
        if (usePrecedesOwnerInBlock(owner, user, block))
          continue;
        if (!consumeIsAnotherReferencesDischarge(references, use, group,
                                                equivalent, mine) &&
            ownershipConsumingUseInvalidatesGroup(contracts, use, group,
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
                                     deallocators, aliases, references, mine,
                                     depth + 1, mappedViews);
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

// Ids of the try scopes whose handler entry is `block`, read off the block's own
// `LyEH_TryCatchMarker(id)` calls.
//
// Why not `own::collectExceptionHandlerEntries`, which answers the same question
// for a whole region: this runs once per owned group during release placement,
// and the only blocks asked about are a terminator's own successors. A region
// walk per group is how the sibling unwind phase became the most expensive step
// in lowering.
void collectHandlerEntryIds(mlir::Block *block,
                            llvm::SmallDenseSet<std::int64_t, 2> &ids) {
  for (mlir::Operation &op : *block) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
    if (!call || call.getCallee() != "LyEH_TryCatchMarker")
      continue;
    if (std::optional<std::int64_t> id = own::exceptionMarkerId(call))
      ids.insert(*id);
  }
}

// The last op in `insertAfter`'s block after which a normal-path death release
// may be written, given that the handler entry of every id in `handlerIds`
// releases the same group unconditionally on entry.
//
// A release written before a `LyEH_TryCallSiteMarker(id)`-guarded call frees the
// token on the try path, and the unwind out of that very call then reaches the
// handler's entry release and frees it again. `groupTokenAtPoint` cannot rescue
// this later: it correctly reports the token NOT held at the later call site, so
// the unwind phase adds no cleanup there and the entry release stands alone.
// Moving the death past the last such call site keeps the token held on every
// unwind edge into the handler, which is the condition that makes an entry
// release sound in the first place.
//
// Why past the guarded CALL and not past its marker: the marker only arms the
// call. A release between the two is still before the call that unwinds, so the
// double free simply moves one op later -- which is why the marker-only form of
// this delay (`releaseOwnedGroupByLiveness` below, before this shared helper
// existed) did not repair the shape.
//
// Why NOT drop the handler's entry release instead and let the unwind phase
// place per-call-site cleanups: it would work for this shape, but the entry
// release is also what `groupUsedOnHandlerPath` reads as "the handler side owns
// this token", so removing it moves the obligation onto a phase that can decline
// it (`deferMarkerWiring`) and turns accepted programs into leaks. Delaying a
// release can only ever extend a live range.
//
// `handlerIds` empty means no successor takes an entry release, and nothing
// moves -- which is the common case and keeps this off every other program's IR.
mlir::Operation *
delayPastUnwindingCallSites(mlir::Operation *insertAfter,
                            const llvm::SmallDenseSet<std::int64_t, 2> *handlerIds) {
  if (handlerIds && handlerIds->empty())
    return insertAfter;
  mlir::Operation *last = insertAfter;
  for (mlir::Operation *op = insertAfter->getNextNode(); op;
       op = op->getNextNode()) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
    if (!call || call.getCallee() != "LyEH_TryCallSiteMarker")
      continue;
    if (handlerIds) {
      std::optional<std::int64_t> id = own::exceptionMarkerId(call);
      if (!id || !handlerIds->contains(*id))
        continue;
    }
    mlir::func::CallOp guarded = own::guardedCallAfterMarker(op);
    last = guarded ? guarded.getOperation() : op;
  }
  return last;
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
                                      own::AliasAnalysis &aliases,
                                      bool ownsReference) {
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
    aliases.namesOf(result, equivalents);
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

  // The per-successor scheme treats each successor as a terminal region for
  // the value: a successor that can re-reach ANOTHER successor (without the
  // re-defining pass through defBlock) merges back into a path that still
  // uses or re-releases the value -- an entry release in a non-using
  // successor would then run before a downstream use (the inlined-__eq__
  // diamond feeding an f-string concat), and an after-last-use release in a
  // using successor would double-release on the rejoined path. Bail to the
  // general liveness placement, which models the merge correctly.
  for (mlir::Block *from : successors)
    for (mlir::Block *to : successors)
      if (from != to && blockReachesAvoiding(from, to, defBlock))
        return false;

  // Non-using successors must be releasable at their entry without an edge
  // split, i.e. reached only from the defining block.
  for (mlir::Block *successor : successors)
    if (!lastUser[successor] &&
        !llvm::hasSingleElement(successor->getPredecessors()))
      return false;

  // A non-using successor that is a try's HANDLER entry is not reached by the
  // CFG edge alone: every `LyEH_TryCallSiteMarker(id)`-guarded call in the
  // sibling (try-path) successor transfers control there at runtime. Its entry
  // release therefore also runs for unwinds that left the try path AFTER this
  // group died there, so the try-path death has to wait for the last of them.
  llvm::SmallDenseSet<std::int64_t, 2> entryReleaseHandlerIds;
  for (mlir::Block *successor : successors)
    if (!lastUser[successor])
      collectHandlerEntryIds(successor, entryReleaseHandlerIds);
  const llvm::SmallDenseSet<std::int64_t, 2> *unwindIds =
      unwindDeathDelayEnabled() ? &entryReleaseHandlerIds : nullptr;

  for (mlir::Block *successor : successors) {
    mlir::OpBuilder builder(call.getContext());
    if (mlir::Operation *last = lastUser[successor])
      builder.setInsertionPointAfter(
          unwindIds ? delayPastUnwindingCallSites(last, unwindIds) : last);
    else
      builder.setInsertionPointToStart(successor);
    emitGroupRelease(builder, call.getLoc(), group, group.values,
                     ownsReference);
  }
  return true;
}

// Release `group` on the CFG edge from `terminator` to its successor
// #succIndex. For cf.br the release is emitted before the branch (the branch's
// only edge); for cf.cond_br the edge is split with a dedicated release block.
// Returns false for unsupported terminators.
bool releaseOnTerminatorEdge(mlir::Operation *terminator, unsigned succIndex,
                             const own::ResourceGroup &group, mlir::Location loc,
                             bool ownsReference) {
  mlir::OpBuilder builder(terminator);
  if (mlir::isa<mlir::cf::BranchOp>(terminator)) {
    builder.setInsertionPoint(terminator);
    emitGroupRelease(builder, loc, group, group.values, ownsReference);
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
  emitGroupRelease(builder, loc, group, group.values, ownsReference);
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
    own::AliasAnalysis &aliases,
    const own::ReferenceMap &references, bool ownsReference,
    bool consumeIsDeath = false,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators = {}) {
  if (!group.deallocator || group.condition)
    return false;
  // The reference this walk is placing for, asked once. Only the liveness
  // exclusion reads it (`namesAnotherMintedReference`); the death test below
  // still goes through the `aggregate_release` label, which is a separate
  // property and stays separate.
  const own::Reference mine =
      group.values.empty() ? own::Reference{}
                           : references.of(group.values.front());
  auto consumingUseEndsThisToken = [&](mlir::OpOperand &use,
                                       mlir::Value equivalent) {
    if (consumeIsAnotherReferencesDischarge(references, use, group.values,
                                            equivalent, mine))
      return false;
    return ownershipConsumingUseInvalidatesGroup(contracts, use, group.values,
                                                 aliases);
  };
  if (!defBlock)
    return false;
  mlir::Region *region = defBlock->getParent();
  if (!region)
    return false;
  // A branch that forwards every group value back into its OWN block-argument
  // position (a loop continue edge) transfers the token identically into the
  // next iteration: it is neither a use nor a death.
  //
  // "Not a use" must not be read as "not live". The self-forward is the last
  // thing that touches the value on that path, so if it does not keep the value
  // live up to the branch, the block holding the last ORDINARY use looks like
  // the end of the value's life and gets a release there -- which the back edge
  // then walks straight past. That stayed hidden only while every container
  // mutation CONSUMED its receiver: the consuming call marked the block instead.
  // A void in-place mutation, which is what interior-behind-the-handle makes
  // possible, removes that marker and exposes the gap.
  llvm::SmallPtrSet<mlir::Block *, 4> identityForwardBlocks;
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
  // A field rebinding re-roots an owned-local marker mid-function, so alias
  // equivalents (the shared object header) have uses that PRECEDE this
  // group's definition. Those uses belong to the group's predecessor
  // incarnation: counting them as liveness would anchor releases where the
  // group values do not exist yet (and double-release the live path).
  std::optional<mlir::DominanceInfo> dominance;
  auto usePrecedesDefinition = [&](mlir::Operation *user) {
    if (!selfOp || user == selfOp)
      return false;
    if (!dominance)
      dominance.emplace(selfOp->getParentOfType<mlir::func::FuncOp>());
    return dominance->properlyDominates(user, selfOp);
  };
  // Interior views (canonical-shape tail beyond the release interface) pin
  // the entity: every use is a plain liveness contribution. Box-word
  // reconstructions (borrowed memref views assembled from the entity's box
  // words) pin the same way.
  // Nested-region uses may pin liveness at their top-level ancestor — but
  // only when the group has NO consuming calls: this walk is order-blind
  // within blocks, so extending liveness past a mid-block consume would place
  // a second (double-freeing) release downstream. Groups with consuming uses
  // keep the conservative nested-use bail.
  bool groupHasConsumingCall = false;
  // THE TOKEN IS GONE AFTER ITS CONSUME, and a use past one is somebody else's.
  //
  // A container literal takes a slot reference and releases the SOURCE's, so
  // reading the element back afterwards -- `by_name = {"red": Color.RED}` then
  // printing `by_name["red"]` -- loads the payload through a handle the frame no
  // longer has a reference to. Those loads are safe: the container's slot is
  // what keeps the object alive. They are not this group's liveness, and
  // counting them extended a dead token's range and placed a SECOND release
  // after its own discharge.
  //
  // Dominance, not block membership: a consume in a branch does not end the
  // token on the sibling branch or past the merge, and only the uses a consume
  // actually reaches are the ones it has already paid for.
  //
  // Failure direction if this ever drops a use it should have kept: one release
  // fewer, which is a leak. The behaviour it replaces was one release more.
  // ⭐ The consume that is NOT this value's death, and therefore needs the
  // folded retain put back.
  //
  // A `transfer_args` contract is CPython's borrow-and-incref with the pair
  // folded away: the callee stores a reference, the caller drops its own, and
  // when the call is the last use both cancel. Sound exactly then. Where the
  // caller reads the value afterwards -- `e = ValueError(msg); print(msg)`,
  // valid Python -- the fold has to come undone, or the read is refused as a
  // use after release.
  //
  // ⛔ Why NOT decline the whole arm and let another one handle it, which was
  // the first attempt: no other arm claims these groups. The trace showed
  // `none.consumed-or-forwarded` and nothing placed a release at all --
  // 52-156 B in `leak.dict_literal_source_move_frequency`,
  // `leak.sequence_literal_source_move_frequency` and
  // `leak.loop_iterator_element_into_container_literal`. The arm that keeps
  // the group is the one that has to take the reference.
  llvm::SmallVector<mlir::Operation *, 4> unfoldRetainBefore;
  auto readFollowsConsumeInBlock = [&](mlir::Operation *consume) {
    for (mlir::Value name : group.values) {
      llvm::SmallVector<mlir::Value, 8> names;
      aliases.namesOf(name, names);
      for (mlir::Value candidate : names)
        for (mlir::OpOperand &use : candidate.getUses()) {
          mlir::Operation *reader = use.getOwner();
          if (reader == selfOp || reader == consume ||
              reader->getBlock() != consume->getBlock() ||
              !consume->isBeforeInBlock(reader))
            continue;
          if (consumingUseEndsThisToken(use, candidate))
            continue;
          // ⛔ KNOWN DEFECT: a TERMINATOR forwarding the value counts as a
          // later use here, and for a loop back-edge that is wrong -- the
          // branch hands the value over, the destination argument gets its own
          // token from the merge-borrow retain on that edge, and the retain
          // this unfold then inserts has nothing to discharge it:
          //
          //     lo: int = 0
          //     for x in [4, -2, 9]:
          //         if x < lo:
          //             lo = x
          //
          // leaks 52 B, and the str form 81 B. The emitted block reads
          // balanced because the inserted retain sits directly before an
          // unrelated `py.decref` of the same name.
          //
          // Why NOT skip terminators outright, which is what the shape
          // suggests: measured, and both reproducers then fail to compile.
          // Some terminators ARE the later use this unfold exists for -- a
          // return of the still-owned value among them.
          //
          // Nor is it enough to skip only the branches that forward the value
          // to a merge argument (`branchForwardsGroupToBlockArgument`, which
          // answers exactly that question): measured too, and the programs
          // then fail with "released owned resource ... is used after
          // release". So the retain is NOT simply spurious -- something reads
          // the value after the consume and needs it alive.
          //
          // What the leak scales with says where the imbalance is. It is not
          // per iteration: a two- and a three-element list both leak 81 B and
          // a four-element one leaks 122 B, which tracks the number of times
          // the CONDITION held and the element was rebound. Each of those
          // takes a retain in the loop body, and the block after the loop
          // releases the merged value once. The missing release is on the
          // rebinding path, not at the loop exit.
          //
          // Pairing a release with this retain was then tried at both places
          // it can go, and neither works: before the forwarding branch, and
          // immediately after the consume. Both give "released owned resource
          // ... is used after release", because the value is READ between the
          // consume and the branch -- which is the reason the retain is there.
          // So the reference this unfold takes is genuinely live to the end of
          // the block and genuinely handed on, and the discharge has to be at
          // the merge argument's own death rather than anywhere in this block.
          // That is the destination group's business, not this one's.
          //
          // And that was tried too. The destination's release is placed with
          // `ownsReference=false` on the reasoning that a merge argument's
          // token is LENT (see the comment at that call). Flipping it to
          // `true` when a predecessor carries an unlabelled retain -- the
          // unfold's signature -- changes the label and not the count: the
          // block still emits four `LyUnicode_DecRef`s and still leaks 81 B.
          // `releaseOwnedGroupByLiveness` decides WHERE a group dies; it does
          // not add a discharge for a second reference the group holds. So the
          // fourth attempt has to give the merge argument a second death, not
          // relabel its first one.
          //
          // That was the fourth attempt, and it overshoots: emitting an extra
          // release beside the existing one, at the function's returns, gives
          // "released or transferred more than once on one CFG path".
          // Together with the third -- one release, 81 B short -- that
          // brackets the answer: on the paths this reaches, one discharge is
          // too few and two are too many, so the count is PATH-DEPENDENT. The
          // rebinding path needs the second and the fall-through path does
          // not, which is exactly the shape `releaseOwnedGroupByLiveness`
          // computes for a group and cannot express for two references of one
          // group. The repair is a second GROUP for the unfolded reference,
          // tracked and released on its own, not a second release for this
          // one.
          return true;
        }
    }
    return false;
  };

  llvm::SmallVector<mlir::Operation *, 4> consumeSites;
  llvm::SmallVector<mlir::Operation *, 4> consumingOps;
  for (mlir::Value result : group.values) {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.namesOf(result, equivalents);
    for (mlir::Value equivalent : equivalents)
      for (mlir::OpOperand &use : equivalent.getUses()) {
        if (use.getOwner() == selfOp)
          continue;
        if (consumingUseEndsThisToken(use, equivalent)) {
          if (!llvm::is_contained(consumingOps, use.getOwner()))
            consumingOps.push_back(use.getOwner());
        }
      }
  }
  // One reference in hand, one taken away per consume, one more needed if
  // anything still reads the value after the last of them. So the count to
  // reach is `consumes + (reads after the last ? 1 : 0)`, and the retains to
  // insert are that minus the one already held -- which is a retain before
  // every consume except the last, plus one before the last when reads follow
  // it. Writing it as the general count rather than as the single-consume case
  // is what makes `a = ValueError(msg); b = ValueError(msg)` fall out instead
  // of needing its own arm.
  {
    bool sameBlock = true;
    for (mlir::Operation *op : consumingOps)
      if (op->getBlock() != consumingOps.front()->getBlock())
        sameBlock = false;
    if (!sameBlock) {
      // ⛔ Ordering across blocks is a reachability question, not
      // `isBeforeInBlock`. Keep the old reading -- every consume a death --
      // so an unhandled shape stays a refusal rather than becoming a guess.
      for (mlir::Operation *op : consumingOps) {
        groupHasConsumingCall = true;
        consumeSites.push_back(op);
      }
    } else if (!consumingOps.empty()) {
      llvm::sort(consumingOps, [](mlir::Operation *a, mlir::Operation *b) {
        return a->isBeforeInBlock(b);
      });
      mlir::Operation *last = consumingOps.back();
      for (unsigned index = 0; index + 1 < consumingOps.size(); ++index)
        unfoldRetainBefore.push_back(consumingOps[index]);
      if (readFollowsConsumeInBlock(last)) {
        unfoldRetainBefore.push_back(last);
      } else {
        groupHasConsumingCall = true;
        consumeSites.push_back(last);
      }
    }
  }
  auto useFollowsAConsume = [&](mlir::Operation *user) {
    if (consumeSites.empty() || !selfOp)
      return false;
    if (!dominance)
      dominance.emplace(selfOp->getParentOfType<mlir::func::FuncOp>());
    for (mlir::Operation *consume : consumeSites)
      if (consume != user && dominance->properlyDominates(consume, user))
        return true;
    return false;
  };

  llvm::SmallVector<mlir::Value, 8> pinnedViews(group.views.begin(),
                                                group.views.end());
  {
    llvm::SmallVector<mlir::Value, 8> groupEquivalents;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.namesOf(result, equivalentValues);
      for (mlir::Value equivalent : equivalentValues) {
        // A box-word view assembled from ANOTHER minted reference's names pins
        // that reference, not this one -- same rule as the use walk below, and
        // it has to be applied here too because these pins never reach it. Two
        // mints on one entity (a global read the lowerer marks twice) left the
        // first one's release riding on loads of the second one's payload
        // handle, so it outlived its own `dict.literal.value.source` discharge
        // and got a second release: three increments against four decrements
        // once the second mint gained the release it was owed.
        if (groupHasConsumingCall &&
            namesAnotherMintedReference(references, equivalent, mine))
          continue;
        groupEquivalents.push_back(equivalent);
      }
    }
    own::collectBoxWordDerivedViews(groupEquivalents, pinnedViews);
  }
  for (mlir::Value view : pinnedViews) {
    for (mlir::OpOperand &use : view.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (user == selfOp)
        continue;
      if (usePrecedesDefinition(user))
        continue;
      // Past this group's own consume the pin belongs to whoever still holds a
      // reference -- the container's slot, for an element read back out of a
      // literal (`useFollowsAConsume`). Box-word views need the test as much as
      // the plain uses do, and more: a reconstruction's uses never reach the use
      // walk, so filtering only there left the pin in place and the dead token
      // got its second release anyway.
      if (useFollowsAConsume(user))
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
    aliases.namesOf(result, equivalents);
    for (mlir::Value equivalent : equivalents) {
      // Only PAST OUR OWN DEATH, and only as a LIVENESS PIN. With no consume of
      // ours there is nothing to be past; with one, these are exactly the uses
      // that made this reference look live after its own discharge
      // (`namesAnotherMintedReference`).
      //
      // Why not `continue` over the whole name: every bail in the loop below is
      // a safety condition (a consume, a transferring return, a nested-region
      // forward), and skipping the name skips those too -- which moved
      // placements into and out of nested regions and double-freed
      // `cross_enum_generic_handler`. Dropping only the pin is the smallest
      // form of "these uses are not mine".
      const bool pinsAnotherReference =
          groupHasConsumingCall &&
          namesAnotherMintedReference(references, equivalent, mine);
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (user == selfOp)
          continue;
        if (usePrecedesDefinition(user))
          continue;
        if (consumingUseEndsThisToken(use, equivalent)) {
          if (!consumeIsDeath)
            return false;
          if (user->getBlock()->getParent() != region)
            return false;
          if (llvm::is_contained(unfoldRetainBefore, user)) {
            lastUse[user->getBlock()] =
                latestUserInBlock(lastUse[user->getBlock()], user);
            continue;
          }
          consumedBlocks.insert(user->getBlock());
          lastUse[user->getBlock()] =
              latestUserInBlock(lastUse[user->getBlock()], user);
          continue;
        }
        if (user->hasTrait<mlir::OpTrait::IsTerminator>() &&
            isIdentitySelfForward(user)) {
          identityForwardBlocks.insert(user->getBlock());
          lastUse[user->getBlock()] =
              latestUserInBlock(lastUse[user->getBlock()], user);
          continue;
        }
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
        if (pinsAnotherReference || useFollowsAConsume(user))
          continue;
        lastUse[user->getBlock()] =
            latestUserInBlock(lastUse[user->getBlock()], user);
      }
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
  // Which outgoing edges of `terminator` the group dies on. Four ways it does
  // NOT die, and the caller must not care which arm it is asking from: the two
  // arms below reached this question by different routes -- a block whose
  // consume forwards on only some edges, and a block that is simply live-out --
  // and wrote the same four guards in two different orders. Nothing made them
  // stay the same order, or stay four.
  auto collectEdgeDeaths = [&](mlir::Operation *terminator) {
    for (unsigned index = 0, end = terminator->getNumSuccessors(); index < end;
         ++index) {
      mlir::Block *successor = terminator->getSuccessor(index);
      if (liveIn[successor])
        continue; // still needed there
      if (forwardsGroupToSuccessor(terminator, index))
        continue; // the token leaves with the forward
      // A continue edge identity-forwards the token into the next iteration's
      // incarnation of the same block arguments: a transfer, not a death.
      if (successor == defBlock && isIdentitySelfForward(terminator))
        continue;
      // Or the successor only passes it back into that next iteration (the
      // non-mutating arm of a conditional structural mutation). Identity
      // self-forwards are invisible to the liveness above, but the token
      // survives through them.
      if (!successor->empty() &&
          isIdentitySelfForward(successor->getTerminator()))
        continue;
      edgeReleases.push_back({terminator, index});
    }
  };
  llvm::SmallVector<mlir::Operation *, 8> afterUseReleases;
  llvm::SmallVector<mlir::Block *, 8> beforeTermReleases;
  llvm::SmallVector<mlir::Block *, 4> atStartReleases;
  for (mlir::Block &blockRef : *region) {
    mlir::Block *block = &blockRef;
    if (!blockIsLive(block))
      continue;
    // The token leaves this block on the self-forward, into the same argument
    // group it came from. A release here would free what the next iteration
    // reads; the loop header's argument group carries the obligation and its
    // own liveness scan places the release where the loop exits.
    if (identityForwardBlocks.count(block))
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
            it->second == terminator && forwardsGroupAnywhere(terminator))
          collectEdgeDeaths(terminator);
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
      collectEdgeDeaths(block->getTerminator());
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
  // past the last marked CALL in the block so an unwind from any marked call in
  // the block reaches the handler with the token intact.
  //
  // Why the guarded call and not the marker, which is where this stopped before:
  // the marker only arms the call, so a release written between the two is still
  // ahead of the unwinding call and the double free just moves one op later --
  // measured on the shape `a = ...; try: raise ValueError(a)`, where the last
  // marker in the block guards the raise itself.
  auto delayPastCallSiteMarkers =
      [&](mlir::Operation *insertAfter) -> mlir::Operation * {
    if (!unwindDeathDelayEnabled()) {
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
    }
    if (!exceptionEdges.count(insertAfter->getBlock()))
      return insertAfter;
    return delayPastUnwindingCallSites(insertAfter, /*handlerIds=*/nullptr);
  };

  for (mlir::Operation *afterOp : afterUseReleases) {
    mlir::Operation *anchor = delayPastCallSiteMarkers(afterOp);
    mlir::OpBuilder builder(anchor);
    builder.setInsertionPointAfter(anchor);
    emitGroupRelease(builder, loc, group, group.values, ownsReference);
  }
  for (mlir::Block *block : beforeTermReleases) {
    mlir::OpBuilder builder(block->getTerminator());
    emitGroupRelease(builder, loc, group, group.values, ownsReference);
  }
  for (mlir::Block *block : atStartReleases) {
    mlir::OpBuilder builder(&block->front());
    emitGroupRelease(builder, loc, group, group.values, ownsReference);
  }
  for (auto &edge : edgeReleases)
    if (!releaseOnTerminatorEdge(edge.first, edge.second, group, loc,
                                 ownsReference))
      return false;
  for (mlir::Operation *consume : unfoldRetainBefore) {
    mlir::ModuleOp module = consume->getParentOfType<mlir::ModuleOp>();
    if (mlir::failed(insertRetain(
            module ? module.lookupSymbol<mlir::func::FuncOp>("Ly_IncRef")
                   : mlir::func::FuncOp(),
            consume, group.values.front())))
      return false;
  }
  return true;
}

// May this edge's borrow→own retain be written through a rank-1 PREFIX view of a
// handle wider than Ly_IncRef's input, rather than declined?
//
// Spellability alone is not the question, which is why this is not
// `canSpellHeaderPrefix`. The retain's anchor is "the earliest point after the
// header's definition", chosen so the lend precedes any same-block
// decref-on-replace. For a freshly ALLOCATED box that point is before the boxing
// sequence has stored the refcount word, so a retain written there reads
// uninitialised memory and the runtime's own guard fires (`Ly_IncRef observed
// non-positive refcount`) -- measured, on dict_key_mutation and
// cross_container_box_fronted_fields.
//
// A block ARGUMENT has no such window: it is bound on entry, so the anchor at
// the block's start is already past its initialisation. That is exactly the
// header a loop-carried handle presents on the edge into a LATER loop's header,
// which is the case completing the destination-group search creates, and it is
// the case whose retain must exist -- without it the destination group's release
// and the source group's release both run.
//
// Why NOT move the anchor past the initialising stores instead: the anchor also
// has to stay ahead of the block's release, so the safe window is bounded on both
// sides by facts about a boxing sequence this pass does not model. Naming the
// case it can prove is honest; guessing the window is how a retain lands on the
// wrong side of a decref.
//
// Why NOT decline the EDGE (sound = false) when this returns false: declining it
// removes the destination's release too, and that is a measured regression, not a
// conservative choice.
//
// ⛔ MEASURED FALSE (2026-07-28), and it is the reason to read the rest of this
// comment carefully rather than trust its conclusion. This paragraph used to also
// say that a candidate accepted on a wide non-argument header "is balanced anyway,
// because nothing else releases the source", and cited that as why the dropped
// retain costs at most a bounded leak. Something else DOES release the source once
// `builtins.list` became one lane:
//
//     def run(n: int) -> int:
//         total = 0
//         for i in range(n):
//             xs: list[int] = [i]
//             ys: list[int] = xs if i % 2 == 0 else [i, i]
//             total += len(ys)
//             total += len(xs)
//         return total
//
// aborts with `Ly_DecRef observed non-positive refcount` (exit 134) while CPython
// prints 6. The site was `header type memref<9xi64>, op result` -- the merge
// argument reconciling the two incoming list groups, declined here because the
// header is an op result rather than a block argument.
//
// So the residual on this branch is NOT a bounded leak. It is a shipped
// over-release: it survives `--release`, and the affine verifier cannot see it
// because the dropped lend belongs to the reconciling argument, so each group's
// own arithmetic still balances (rfc/memory-safety-proof.md, third failure shape).
// The recorded residual was one class too weak.
//
// ✅ REPAIRED (2026-07-28) by `entity_header::prefixIsInitializedAtDefinition`,
// and the paragraph this replaces was WRONG about why. It said the
// `isa<BlockArgument>` test is a proxy for "the handle's first two words are the
// refcount/class prefix", a per-contract LAYOUT fact, and that "a one-lane `list`
// handle satisfies it and a 16-word payload box does not".
//
// **A payload box does satisfy it.** `objectPayloadHandleWords`
// (Core/CollectionPayload.cpp) writes `words[0] = refcount` and
// `words[1] = payloadClass`. Both populations carry the prefix, so no layout
// predicate can separate them, and neither can the widths -- `builtins.object`
// and the payload box are both 16 (ABI/HandleWidthRegistry.h).
//
// What separates them is PROVENANCE, keyed over the four programs that pin the
// two behaviours (one widened site each):
//
//     the retain that must exist ... memref<9xi64>  by func.call
//     the three that must not ...... memref<16xi64> by memref.alloc
//
// A call result is an entity its callee finished. A `memref.alloc` result is raw
// storage whose prefix `boxRuntimeObject` stores in the ops AFTER the alloc, so
// the anchor is inside the initialisation window -- the boxing-window reason
// above, now stated as the fact that decides rather than as an exception to a
// layout rule.
//
// So the three cases are NOT a cost of this repair. They depend on a different
// invariant -- "the anchor may sit at the header's definition" -- which is false
// for raw storage, and the predicate is exactly the test for it. Measured on one
// binary: shipped behaviour breaks the merge case and keeps the three; the naive
// widening fixes the merge case and breaks the three (rc=134, `Ly_IncRef observed
// non-positive refcount`); the predicate gets all four right, ctest 466/466 both
// arms.
//
// The whole-corpus footprint, so the next reader does not have to re-derive it:
// over 287 golden cases the naive widening reaches FOUR sites, three of them the
// `memref.alloc` boxes above. The predicate newly retains exactly one --
// `dict_iteration_views`, a `memref<8xi64>` call result -- and the
// llvm-translation differential there is `Ly_IncRef` 106 -> 107 with EVERY
// release symbol unchanged, which is the intended shape: the lend that was
// missing, against a release that was already there and already unbalanced.
//
// Why the twelve-fixed/three-broken figure recorded here is NOT reproducible as
// test counts on this tree: it was taken on the float/complex/range branch at
// 444/456. On the merged tree the suite is green in the shipped arm, because no
// case covers the merge shape -- the reason `tests/golden/cases/
// list_merge_arg_loop_release.py` was added with the repair.
//
// And why a green affine verifier is NOT evidence that this predicate is
// unnecessary: a dropped lend belongs to the argument reconciling two groups, so
// each group's own retain/release arithmetic still balances and nothing is
// reported. It is observable only once a loop REACHES its release, i.e. only
// when the loop completes. Before this predicate existed, a nested loop
// accumulating a one-lane float printed 0.0 where CPython prints 9.0 -- silent,
// and surviving --release. tests/golden/cases/scalar_loop_carried_mutate.py
// pins that shape.
bool borrowEdgeRetainIsSpellable(mlir::Value header,
                                 mlir::func::FuncOp retainFunction) {
  // Why NO ablation switch for this one, unlike the rest of this file: both other
  // readings of this predicate are known to produce a wrong RELEASE -- the shipped
  // one over-releases the merge case, the naive one over-retains three boxing
  // cases. `LYTHON_ABLATE_OWNERSHIP_SYMBOL_TABLE` is safe to ship because a
  // mistake with it set costs compile time; a switch here would cost memory
  // safety. Both arms were measured on one build and the switches removed.
  return entity_header::prefixIsInitializedAtDefinition(header) &&
         own::canSpellHeaderPrefix(
             header.getType(), retainFunction.getFunctionType().getInput(0));
}

// Wrapper: liveness-based release for an owned call-result group.
// An owned CALL-RESULT group whose token is consumed somewhere in the function
// dies at that consume, exactly as an owned local or a block-argument root does
// -- those two entry points have always passed `consumeIsDeath=true`. Passing
// `false` here made a single consuming use anywhere abandon the whole liveness
// placement, so every OTHER edge out of the defining block got no release at
// all: `for i in range(n): ys = [i]` has its element consumed by the sequence
// literal's own transfer pair on the body edge, and the loop-EXIT edge was then
// left holding a live token. `LyRangeIterator_Next` allocates its element
// unconditionally (`LyLong_FromI64` runs on the exhausted path too, with a zero
// value), so that token is real and the affine verifier was right to refuse.
//
// Why NOT model the element as conditionally owned off `ly.runtime.valid_result_index`
// (the reading this defect was handed over with): that attribute marks which
// result says the element is MEANINGFUL, not which says it is OWNED. The
// manifest allocates on both paths, so a conditional group would leave the
// exhausted path's element unreleased -- trading a loud refusal for a silent
// leak. The three declarations carrying the attribute were checked; all three
// allocate unconditionally.
//
// Why NOT widen `insertImmediateSuccessorReleases` instead: it requires every
// use to sit in an IMMEDIATE successor of the defining block, and the loop body
// here is a successor-of-successor (the try anchor block sits between). It
// declines this shape for that reason as well as for the consume.
//
// LYTHON_ABLATE_CONSUME_IS_DEATH_CALL_RESULTS=1 restores the old `false`. Its
// failure mode is the refusal this change removes -- a rejected valid program,
// never a released-twice one -- which is the only direction an ablation switch
// over release placement may fail in.
bool insertOwnedValueReleasesByLiveness(
    FuncContractCache &contracts, mlir::func::CallOp call,
    const own::ResourceGroup &group, own::AliasAnalysis &aliases,
    const own::ReferenceMap &references,
    bool ownsReference) {
  static bool consumeIsDeathDisabled = [] {
    auto value = llvm::sys::Process::GetEnv(
        "LYTHON_ABLATE_CONSUME_IS_DEATH_CALL_RESULTS");
    return value && !value->empty() && *value != "0";
  }();
  return releaseOwnedGroupByLiveness(contracts, call.getOperation(),
                                     call->getBlock(), call.getLoc(), group,
                                     aliases, references, ownsReference,
                                     /*consumeIsDeath=*/!consumeIsDeathDisabled);
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
// `insertReleases=false` re-derives the owned block-argument merge groups
// WITHOUT mutating: the post-cleanup unwind re-pass needs the same group set
// for its held-token analysis, but the normal-path releases and borrow-edge
// retains were already placed by the main pass -- inserting them twice would
// double-release/over-retain.
mlir::LogicalResult insertOwnedBlockArgumentReleases(
    mlir::ModuleOp module, FuncContractCache &contracts,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases,
    const own::ReferenceMap &references,
    mlir::SymbolTable *symbols,
    llvm::SmallVectorImpl<own::ResourceGroup> *unwindGroups,
    bool insertReleases = true) {
  std::uint64_t calleeResolutions = 0;
  llvm::DenseSet<mlir::Value> ownedValues;
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

  // Every top-level terminator of `fn` that mentions the group -- i.e. every
  // candidate forwarding edge, wherever the emitter put it.
  //
  // Why not just the group's own block: the verifier RENAMES a tracked group
  // onto the destination argument on every forwarding edge, so the release it
  // will accept afterwards is one spelled with the destination's name. Reading
  // the forward only off the defining block (or, for an argument group, off its
  // own block) finds the entry a merge emits where the value is born, but not
  // the entry into a LATER loop whose header a dominating block feeds -- the
  // shape a void in-place mutation leaves behind, because nothing re-defines
  // the receiver to move the forward next to it. That destination group was
  // never created, so no pass released the name the verifier had renamed onto,
  // and the token reached the function exit still owned.
  auto forwardingTerminators = [&](mlir::func::FuncOp fn,
                                   llvm::ArrayRef<mlir::Value> values) {
    llvm::SmallVector<mlir::Operation *, 4> terminators;
    if (values.empty())
      return terminators;
    mlir::Region *body = &fn.getBody();
    llvm::SmallPtrSet<mlir::Operation *, 4> seen;
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.namesOf(values.front(), equivalents);
    for (mlir::Value equivalent : equivalents)
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (!user->hasTrait<mlir::OpTrait::IsTerminator>() ||
            user->getBlock()->getParent() != body)
          continue;
        if (seen.insert(user).second)
          terminators.push_back(user);
      }
    return terminators;
  };

  // What one owned group contributes before any edge can be classified: its
  // values are OWNED, and every function-level branch that forwards them seeds
  // a destination candidate. Returns none when the group never reaches a
  // function-level branch -- nothing is seeded and nothing is worth tracing.
  //
  // Why one body: an owned call result and an owned-local marker differ in
  // where the group comes from and in NOTHING this does with it. Written twice
  // over two sets, the four copies drifted -- the ownedValues pair mapped no
  // views and had no escape flag, so a group the candidate pair abandoned was
  // still contributing owned lanes, and only the region-nesting depth decided
  // whether that mattered.
  struct Seeded {
    unsigned candidates = 0;
    std::size_t lanes = 0;
    std::size_t views = 0;
  };
  auto seedGroup = [&](mlir::func::FuncOp fn, const own::ResourceGroup &g,
                       mlir::Block *from) -> std::optional<Seeded> {
    llvm::SmallVector<mlir::Value, 4> values(g.values.begin(), g.values.end());
    llvm::SmallVector<mlir::Value, 4> views(g.views.begin(), g.views.end());
    for (mlir::Value v : values)
      ownedValues.insert(v);
    // A group born inside a region merge (the int fast/slow scf.if) reaches a
    // function-level branch only through the parent op's results, so walk out
    // one region at a time. Every level's results hold the token too.
    bool escaped = false;
    while (from && from->getTerminator() &&
           !mlir::isa<mlir::func::FuncOp>(from->getParentOp())) {
      mlir::Operation *terminator = from->getTerminator();
      auto mappedValues =
          mapRegionTerminatorGroupToParentResults(terminator, values, aliases);
      if (!mappedValues) {
        escaped = true;
        break;
      }
      values = std::move(*mappedValues);
      for (mlir::Value v : values)
        ownedValues.insert(v);
      // Views that stop mapping cost the CANDIDATE, not the ownership: their
      // uses would be invisible to the liveness, so no release may be placed,
      // but the lanes are still owned and the walk keeps recording them.
      if (!views.empty() && !escaped) {
        auto mappedViews =
            mapRegionTerminatorGroupToParentResults(terminator, views, aliases);
        if (!mappedViews)
          escaped = true;
        else
          views = std::move(*mappedViews);
      }
      from = from->getParentOp()->getBlock();
    }
    if (escaped || !from || !from->getTerminator())
      return std::nullopt;

    Seeded seeded{0, values.size(), views.size()};
    for (mlir::Operation *terminator : forwardingTerminators(fn, values)) {
      auto dest = forwardedBlockArgGroup(terminator, values, aliases);
      if (!dest)
        continue;
      auto destViews = forwardedViews(terminator, views);
      if (!destViews)
        continue;
      candidates.insert(
          {dest->front(), Candidate{*dest, *destViews, g.deallocator}});
      ++seeded.candidates;
    }
    return seeded;
  };

  module.walk([&](mlir::func::CallOp call) {
    mlir::func::FuncOp fn = call->getParentOfType<mlir::func::FuncOp>();
    if (!fn || own::isRuntimeManifestFunction(fn))
      return;
    ++calleeResolutions;
    for (const own::ResourceGroup &g : own::collectOwnedCallResultGroups(
             module, call, deallocators, symbols)) {
      if (!g.deallocator || g.condition)
        continue;
      std::optional<Seeded> seeded = seedGroup(fn, g, call->getBlock());
      if (seeded && ownershipTransferTraceEnabled())
        llvm::errs() << "[ownership-transfers] call-result group of "
                     << call.getCallee() << " in @" << fn.getName()
                     << ": lanes=" << seeded->lanes
                     << " views=" << seeded->views
                     << " seeded=" << seeded->candidates << "\n";
    }
  });

  // Owned LOCAL OBJECT groups (the `ly.ownership.owned_local_object` cast
  // markers: dataclass/user-class instances) transfer into merge arguments
  // exactly like owned call results (`p = P(1,2) if c else P(3,4)`) -- omitting
  // them classifies every incoming merge edge as a borrow and drops the
  // destination group, leaking the merged object on the normal path and leaving
  // its unwind token state Unknown.
  module.walk([&](mlir::Operation *op) {
    if (!op->hasAttr(own::kOwnedLocalObjectAttr))
      return;
    mlir::func::FuncOp fn = op->getParentOfType<mlir::func::FuncOp>();
    if (!fn || own::isRuntimeManifestFunction(fn))
      return;
    for (const own::ResourceGroup &g :
         own::collectOwnedLocalObjectGroups(op, deallocators)) {
      if (!g.deallocator || g.condition)
        continue;
      // What is specific to this source: the branch forwards the marker's
      // OPERANDS (the raw allocs), not its results, so the operands are owned
      // alongside the group values. Alias equivalents BEYOND the marker's own
      // operands (select results etc.) stay out -- they may equally alias
      // values that were only borrowed.
      if (op->getNumOperands() == op->getNumResults())
        for (mlir::Value operand : op->getOperands())
          ownedValues.insert(operand);
      seedGroup(fn, g, op->getBlock());
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
      auto fn = mlir::dyn_cast<mlir::func::FuncOp>(
          firstArg.getOwner()->getParentOp());
      if (!fn)
        continue; // a candidate group always lives in the function's own region
      for (mlir::Operation *terminator :
           forwardingTerminators(fn, candidate.args)) {
        auto dest = forwardedBlockArgGroup(terminator, candidate.args, aliases);
        if (!dest)
          continue;
        auto destViews = forwardedViews(terminator, candidate.views);
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
  // Every borrow-edge retain this emits was classified once over the whole
  // corpus (2026-07-30, 299 of them) and none was surplus. What that census
  // settled, and what it therefore rules out doing here:
  //
  // Why NOT move the retains to an explicit `py.incref` in the emitter, where
  // a dup belongs: `ABI/ControlFlowABI.cpp` creates the merge argument before
  // the back edge exists, so liveness -- the question that decides move versus
  // dup -- cannot be asked there. 183 of the 200 outlives-the-branch retains
  // are block-arg to block-arg, i.e. exactly the loop-carried case that needs
  // the back edge.
  //
  // Why NOT teach `isOwnedIncoming` that function parameters are owned: 70 of
  // the 299 are entry block arguments and every one is genuinely BORROWED --
  // not one carries `transfer_args`, `release_args` or `retain_args` at its
  // index. The shape resembles the boxing defect, where ownership was real and
  // merely unrecorded and recording it changed no output; here the answers are
  // opposite, and calling them owned would drop retains that are load-bearing.
  //
  // Why NOT extend `borrowEdgeRetainIsSpellable`'s initialisation predicate to
  // globals or region yields: of the 299, 278 reach `Ly_IncRef`'s input type by
  // a direct cast and never consult it, 21 by a prefix subview, and 0 are
  // refused. Every `memref.get_global`, `scf.if` and `memref.alloc` header
  // takes the cast path, so any extension would be written against no evidence.
  //
  // What is open is not a defect but a REPRESENTATION gap, recorded in
  // `proof/`: the kernel cannot say "the emitted code disagrees with the ghost
  // state", which is the shape every one of these bugs had.
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
    // Per pass over the candidate set: an emitter incref may pay for only one
    // lane, and the set is rebuilt because the candidates are.
    llvm::DenseSet<mlir::Operation *> creditedIncrefs;
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
          mlir::func::CallOp lane =
              transfers ? nullptr
                        : emitterLaneIncrefInBlock(pred, header, retainFunction,
                                                   creditedIncrefs);
          if (transfers || lane) {
            // A lane the emitter's ledger already paid for is a transfer: the
            // token it minted belongs to the destination argument.
            if (lane)
              creditedIncrefs.insert(lane.getOperation());
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
      if (ownershipTransferTraceEnabled())
        llvm::errs() << "[ownership-transfers] destination arg group in @"
                     << destBlock->getParentOp()
                            ->getAttrOfType<mlir::StringAttr>(
                                mlir::SymbolTable::getSymbolAttrName())
                            .getValue()
                     << ": lanes=" << candidate.args.size()
                     << " views=" << candidate.views.size()
                     << " sound=" << sound << " anyTransfer=" << anyTransfer
                     << " borrowEdges=" << retains.size() << "\n";
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
  for (const EdgeRetain &retain :
       insertReleases ? edgeRetains : llvm::SmallVector<EdgeRetain, 8>()) {
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
    mlir::Type retainInput = retainFunction.getFunctionType().getInput(0);
    mlir::Value header = retain.header;
    if (header.getType() != retainInput) {
      if (mlir::memref::CastOp::areCastCompatible(header.getType(),
                                                  retainInput)) {
        header = mlir::memref::CastOp::create(builder, anchor->getLoc(),
                                              retainInput, header)
                     .getResult();
      } else if (borrowEdgeRetainIsSpellable(retain.header, retainFunction)) {
        header = own::spellHeaderPrefix(builder, anchor->getLoc(), header,
                                        retainInput);
        if (!header)
          return reportUnspellableBorrowEdgeRetain(retain.header, anchor);
      } else {
        return reportUnspellableBorrowEdgeRetain(retain.header, anchor);
      }
    }
    auto call = mlir::func::CallOp::create(builder, anchor->getLoc(),
                                           retainFunction, header);
    call->setAttr("ly.ownership.aggregate_retain",
                  builder.getStringAttr(own::kBlockArgMergeBorrowLabel));
  }

  std::uint64_t candidatesSeen = 0;
  for (auto &entry : candidates) {
    Candidate &candidate = entry.second;
    ++candidatesSeen;
    auto firstArg = mlir::cast<mlir::BlockArgument>(candidate.args.front());
    own::ResourceGroup destGroup;
    destGroup.values.assign(candidate.args.begin(), candidate.args.end());
    destGroup.views.assign(candidate.views.begin(), candidate.views.end());
    destGroup.root = own::entityRootOf(destGroup.values);
    destGroup.deallocator = candidate.deallocator;
    if (unwindGroups)
      unwindGroups->push_back(destGroup);
    if (insertReleases)
      // Not labelled: a merge argument's token is LENT by the borrow-edge
      // retains above and cancelled by the pre-merge name's decref, so its
      // release is one half of a pair rather than the discharge of an increment
      // of its own. Claiming otherwise would let the other half disown it.
      releaseOwnedGroupByLiveness(contracts, /*selfOp=*/nullptr,
                                  firstArg.getOwner(), firstArg.getLoc(),
                                  destGroup, aliases, references,
                                  /*ownsReference=*/false,
                                  /*consumeIsDeath=*/true, deallocators);
  }
  if (ownershipTransferTraceEnabled())
    llvm::errs() << "[ownership-transfers] block-arg candidates="
                 << candidatesSeen << "\n";
  reportOwnershipWorkShape(insertReleases
                               ? "refcount-insertion.block-argument-releases"
                               : "post-cleanup-unwind-insertion"
                                 ".block-argument-groups",
                           moduleSymbolCount(module), calleeResolutions);
  return mlir::success();
}

mlir::LogicalResult
insertOwnedResultReleases(mlir::ModuleOp module, mlir::func::CallOp call,
                          FuncContractCache &contracts,
                          llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
                          own::AliasAnalysis &aliases,
                          const own::ReferenceMap &references,
                          mlir::SymbolTable *symbols) {
  if (call.getNumResults() == 0)
    return mlir::success();

  mlir::func::FuncOp enclosing = call->getParentOfType<mlir::func::FuncOp>();
  for (own::ResourceGroup group :
       own::collectOwnedCallResultGroups(module, call, deallocators, symbols)) {
    if (!group.deallocator)
      continue;

    // The lanes the call returned are the entity's BIRTH expansion. A payload
    // mutation since then re-rooted part of it, and both the release position
    // and the release operands must come from the current lanes: the old ones
    // name storage the mutation primitive has already reallocated away, and
    // their last use sits at the mutation itself -- in the middle of the
    // entity's live range.
    own::advanceGroupLanesThroughReRoots(contracts, enclosing, group, aliases);

    std::optional<ReleaseInsertion> release =
        findReleaseInsertion(contracts, call, group.values, deallocators,
                             aliases, references, /*mine=*/own::Reference{},
                             /*depth=*/0, group.views);
    if (release) {
      mlir::OpBuilder builder(call);
      if (release->before)
        builder.setInsertionPoint(release->before);
      else
        builder.setInsertionPointAfter(release->after);
      emitGroupRelease(builder, call.getLoc(), group, release->group,
                       /*ownsReference=*/true);
      tracePlacement("straight-line", call, group);
      continue;
    }

    // ⛔ Why NOT delete this one too, the way `conditional` went: it is not
    // dead, only rare. 142 placements over 297 programs (0.11%), and removing
    // it leaves 42 of the 61 affected programs BYTE-IDENTICAL -- the fallback
    // reaches the same instruction -- but changes the other 19. Thirteen of
    // those only slide the release later within its own block; six move
    // releases into different blocks, mostly `__ly_unwind_cleanup_*`.
    //
    // Nothing can tell the two placements apart: with this arm disabled the
    // suite is 541/541 and all 19 are LeakSanitizer- and AddressSanitizer-clean.
    // That is the reason it stays rather than a reason to drop it -- deleting it
    // would change codegen on 19 programs with no test able to see a regression,
    // which is the silent direction. `conditional` was deletable because it
    // placed NOTHING; this one places, and what it places is unverified.
    if (insertImmediateSuccessorReleases(contracts, call, group, aliases,
                                         /*ownsReference=*/true)) {
      tracePlacement("immediate-successor", call, group);
      continue;
    }

    if (insertOwnedValueReleasesByLiveness(contracts, call, group, aliases,
                                           references,
                                           /*ownsReference=*/true)) {
      tracePlacement("liveness", call, group);
      continue;
    }

    mlir::func::FuncOp function = call->getParentOfType<mlir::func::FuncOp>();
    if (!function) {
      tracePlacement("none.no-enclosing-function", call, group);
      continue;
    }

    // ⛔ Why NOT undo the transfer fold here, which reads like the place for
    // it: this arm never claims the groups it would apply to.
    // `releaseOwnedGroupByLiveness` takes them first, and that is where the
    // fold and its undoing both live. Handing the group down to this one
    // instead leaks -- 52-156 B in `leak.dict_literal_source_move_frequency`,
    // `leak.sequence_literal_source_move_frequency` and
    // `leak.loop_iterator_element_into_container_literal` -- because nothing
    // here claims it either: the trace reads `none.consumed-or-forwarded` and
    // no release is placed at all.
    //
    // What remains unhandled anywhere is the same shape at the YIELD
    // boundary. `yield a; yield a` is refused, because a value crossing a
    // yield is transferred to the resumer through the resume clone's
    // `owned_results` and never reaches either arm. `while i < 3: yield a`
    // fails too, which is a loop yielding a loop-invariant value.
    bool canReleaseAtExits = true;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.namesOf(result, equivalentValues);
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
    if (!canReleaseAtExits) {
      tracePlacement("none.consumed-or-forwarded", call, group);
      continue;
    }

    mlir::DominanceInfo dominance(function);
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp returnOp) {
      if (dominance.dominates(call.getOperation(), returnOp.getOperation()))
        returns.push_back(returnOp);
    });
    for (mlir::func::ReturnOp returnOp : returns) {
      mlir::OpBuilder builder(returnOp);
      emitGroupRelease(builder, returnOp.getLoc(), group, group.values,
                       /*ownsReference=*/true);
    }
    tracePlacement("dominated-returns", call, group);
  }
  return mlir::success();
}

bool ownedLocalTraceEnabled() {
  static bool enabled = [] {
    auto v = llvm::sys::Process::GetEnv("LYTHON_TRACE_OWNED_LOCAL");
    return v && !v->empty() && *v != "0";
  }();
  return enabled;
}

mlir::LogicalResult insertOwnedLocalObjectReleases(
    mlir::ModuleOp module, mlir::Operation *op, FuncContractCache &contracts,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases,
    const own::ReferenceMap &references) {
  mlir::func::FuncOp enclosing = op->getParentOfType<mlir::func::FuncOp>();
  // SEVERAL MINTS ON ONE HEAD NEED NO SPECIAL CASE, and the measurement that
  // said otherwise did not survive. Reading one element back twice (`c = [i, i]`
  // read at both indices, or the three `0`s of `counts = {"app": 0, "net": 0,
  // "val": 0}`) puts two or more retain-rooted markers on one object, and an
  // earlier tree refused `golden.cases.cross_except_star_views` for it with
  // `released or transferred more than once` -- so this pass declined the second
  // mint. Re-measured after the affine walk learned to prefer a release written
  // under the token's own name: 494/494 with the restriction and 494/494
  // without, and the four residual leaking shapes are byte-identical either way.
  // An inert restriction is not a safeguard, so it is gone.
  const bool retainRooted = own::ownedLocalMarkerIsRetainRooted(op, aliases);
  if (ownedLocalTraceEnabled()) {
    auto contract =
        op->getAttrOfType<mlir::StringAttr>(own::kOwnedLocalObjectContractAttr);
    llvm::errs() << "[owned-local] marker in @"
                 << (enclosing ? enclosing.getSymName() : "<none>")
                 << " contract=" << (contract ? contract.getValue() : "<none>")
                 << " retain-rooted=" << (retainRooted ? 1 : 0) << " groups="
                 << own::collectOwnedLocalObjectGroups(op, deallocators).size()
                 << "\n";
  }
  for (own::ResourceGroup group :
       own::collectOwnedLocalObjectGroups(op, deallocators)) {
    if (!group.deallocator) {
      if (ownedLocalTraceEnabled())
        llvm::errs() << "[owned-local]   no-deallocator\n";
      continue;
    }

    // Which increment this group is the obligation for, for `isNotOwnName`.
    const own::Reference mine =
        group.values.empty() ? own::Reference{}
                             : references.of(group.values.front());

    // Same reason as for call-result groups: an owned local whose payload was
    // mutated must be released through the post-mutation lanes. The marker
    // re-root covers the shapes the lowerer can see; this covers the ones only
    // the final IR shows (an in-place growth primitive threaded through the
    // instance's expansion).
    own::advanceGroupLanesThroughReRoots(contracts, enclosing, group, aliases);

    std::optional<ReleaseInsertion> release =
        findReleaseInsertion(contracts, op, group.values, deallocators,
                             aliases, references, mine, /*depth=*/0,
                             group.views);
    if (release) {
      mlir::OpBuilder builder(op);
      if (release->before)
        builder.setInsertionPoint(release->before);
      else
        builder.setInsertionPointAfter(release->after);
      emitGroupRelease(builder, op->getLoc(), group, release->group,
                       /*ownsReference=*/retainRooted);
      if (ownedLocalTraceEnabled())
        llvm::errs() << "[owned-local]   placed=straight-line\n";
      continue;
    }

    // Straight-line placement failed (e.g. the entity crosses blocks inside a
    // loop body): fall back to CFG liveness, mirroring the owned-call-result
    // path.
    unsigned before = 0;
    if (ownedLocalTraceEnabled() && enclosing)
      enclosing.walk([&](mlir::func::CallOp) { ++before; });
    if (releaseOwnedGroupByLiveness(contracts, op, op->getBlock(), op->getLoc(),
                                    group, aliases, references,
                                    /*ownsReference=*/retainRooted,
                                    /*consumeIsDeath=*/true,
                                    /*deallocators=*/{})) {
      if (ownedLocalTraceEnabled()) {
        unsigned after = 0;
        if (enclosing)
          enclosing.walk([&](mlir::func::CallOp) { ++after; });
        llvm::errs() << "[owned-local]   placed=liveness emitted="
                     << (after - before) << "\n";
      }
      continue;
    }

    mlir::func::FuncOp function = op->getParentOfType<mlir::func::FuncOp>();
    if (!function) {
      if (ownedLocalTraceEnabled())
        llvm::errs() << "[owned-local]   DROPPED no-enclosing-function\n";
      continue;
    }

    bool canReleaseAtExits = true;
    for (mlir::Value result : group.values) {
      llvm::SmallVector<mlir::Value, 8> equivalentValues;
      aliases.namesOf(result, equivalentValues);
      for (mlir::Value equivalent : equivalentValues) {
        for (mlir::OpOperand &use : equivalent.getUses()) {
          mlir::Operation *user = use.getOwner();
          if (user == op)
            continue;
          if (user->getParentOfType<mlir::func::FuncOp>() != function ||
              (!consumeIsAnotherReferencesDischarge(references, use,
                                                    group.values, equivalent,
                                                    mine) &&
               ownershipConsumingUseInvalidatesGroup(contracts, use,
                                                     group.values, aliases)) ||
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
    if (!canReleaseAtExits) {
      if (ownedLocalTraceEnabled())
        llvm::errs() << "[owned-local]   DROPPED consumed-or-forwarded\n";
      continue;
    }

    mlir::DominanceInfo dominance(function);
    llvm::SmallVector<mlir::func::ReturnOp, 4> returns;
    function.walk([&](mlir::func::ReturnOp returnOp) {
      if (dominance.dominates(op, returnOp.getOperation()))
        returns.push_back(returnOp);
    });
    for (mlir::func::ReturnOp returnOp : returns) {
      mlir::OpBuilder builder(returnOp);
      emitGroupRelease(builder, returnOp.getLoc(), group, group.values,
                       /*ownsReference=*/retainRooted);
    }
    if (ownedLocalTraceEnabled())
      llvm::errs() << "[owned-local]   placed=dominated-returns count="
                   << returns.size() << "\n";
  }
  return mlir::success();
}

// ---------------------------------------------------------------------------
// Unwind cleanup (rfc/stdlib-semantics.md R2: unwinding releases owned
// values; the verifier's "leak accepted" carve-out is gone).
//
// The setjmp-style EH model transfers control from each
// `LyEH_TryCallSiteMarker(id)`-guarded call to the handler entry of `id`
// (in-function try) and from an unguarded raise primitive out of the
// function. After the normal-path releases are placed, such an exceptional
// exit may still hold owned tokens the destination never releases: those
// tokens must be released ON the unwind edge itself, per call site, because
// the set of held tokens differs between call sites sharing one handler.
//
// A guarded call site gets a dedicated cleanup handler: the marker is
// re-pointed at a fresh id whose handler entry is a new block (DecRefs,
// then a branch to the original handler), wired with the same
// anchor/cond_br pattern the try lowering uses -- NOT as an unreachable
// block -- so exception-edge collection, the affine verifier, dominance,
// and the final LLVM invoke conversion all see it through machinery that
// already exists (an unreachable block would also be erased by the
// canonicalizer before the LLVM phase could wire it).
// An unguarded raise primitive leaves the function for good, so its
// releases go directly before the raise call.
// ---------------------------------------------------------------------------

// Whether the group's affine token is held when control reaches `point`,
// uniformly over every path. `Unknown` (mixed or loop-dependent states)
// inserts nothing; the affine verifier then reports the residual leak
// instead of this pass guessing (never silently mis-execute).
enum class TokenAtPoint { Held, NotHeld, Unknown };

struct UnwindTrackedGroup {
  llvm::SmallVector<mlir::Value, 4> values;
  llvm::SmallVector<mlir::Value, 4> views;
  const own::RuntimeDeallocator *deallocator = nullptr;
  // Retained / partially-consumed / otherwise ambiguous groups are left to
  // the verifier rather than released on a guess.
  bool skip = false;
  // Top-level (function-region) ancestors of consuming users: releasing
  // deallocator calls, transferring calls, and terminators that forward the
  // whole group into successor block arguments (the token moves to the
  // destination argument group there).
  llvm::SmallVector<mlir::Operation *, 4> consumeSites;
  // Top-level ancestors of every user (values, interior views, box-word
  // derived views): liveness pins for the handler-side check.
  llvm::SmallVector<mlir::Operation *, 8> useSites;
  // Which increment this group is the obligation for, when the analysis can
  // name one (`own::ReferenceMap`).
  //
  // AN ENTITY IS NOT A RESOURCE, and this is the only thing that tells the two
  // apart here. A retain-minted marker and the reference it was minted on share
  // every name -- `underlyingObjectValue` walks through the identity cast the
  // marker is spelled as, so even their entity roots are the same value -- yet
  // they are two increments with two releases. Both the dedup and the consume
  // scan below ask "is this the same obligation as that one", and without the
  // producer they could only answer "same entity, so yes".
  own::Reference reference;
};

// How the unwind-cleanup CFG questions below are answered.
//
//   LYTHON_UNWIND_REACH_CACHE unset/1  memoised sets (the shipped formulation)
//   LYTHON_UNWIND_REACH_CACHE=0        the direct walk, one per question
//   LYTHON_UNWIND_REACH_CACHE=verify   both, per question, and report every
//                                      disagreement with a running denominator
//
// Why the walk stays in the tree rather than being deleted with the measurement
// that needed it: this phase inserts ownership releases, so a divergence
// between the two formulations is a leak or a double free, and "they agree" is
// only checkable against a reference. Rebuilding one from an older
// commit checks that commit, not this tree -- and it makes the comparison
// cross-binary, which is how three of this session's conclusions went wrong.
//
// Why `verify` as well as `0`: comparing final IR only catches a divergence
// that changes the IR. A query can disagree and still leave the IR alone (the
// group was skipped for another reason), and that is the shape that ships one
// waiting for the next program. `verify` checks every query instead of one hash
// per program, and prints how many it checked -- a count of zero is then a
// finding rather than a pass.
enum class UnwindReachMode { Cached, Walk, Verify };

UnwindReachMode unwindReachMode() {
  static UnwindReachMode mode = [] {
    auto value = llvm::sys::Process::GetEnv("LYTHON_UNWIND_REACH_CACHE");
    if (!value || value->empty())
      return UnwindReachMode::Cached;
    if (*value == "0")
      return UnwindReachMode::Walk;
    if (llvm::StringRef(*value).equals_insensitive("verify"))
      return UnwindReachMode::Verify;
    return UnwindReachMode::Cached;
  }();
  return mode;
}

struct UnwindCleanupAnalysis {
  mlir::func::FuncOp function;
  mlir::DominanceInfo dominance;
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
      exceptionEdges;
  // Exception edges WITH the marker op that creates them: the block-level
  // edge map is too coarse when a consume sits in the same block as the
  // markers -- an unwind only carries the consumed state if the consume ran
  // BEFORE the marked call, i.e. precedes the marker in the block.
  llvm::DenseMap<mlir::Block *,
                 llvm::SmallVector<std::pair<mlir::Operation *, mlir::Block *>,
                                   2>>
      markerEdges;
  llvm::DenseMap<mlir::Block *, llvm::SmallPtrSet<mlir::Block *, 16>>
      reachableCache;

  // Numbering of the function's top-level blocks, so a reachability answer can
  // be a bit in a vector instead of a walk. Every from/to/avoid asked about
  // here is a top-level block (points nested in region ops are mapped to their
  // top-level ancestor before they arrive), and a query naming a block outside
  // the numbering falls back to the direct walk rather than guessing.
  llvm::SmallVector<mlir::Block *, 32> blocks;
  llvm::DenseMap<mlir::Block *, unsigned> blockIndex;
  // Reverse of `exceptionEdges`, for the backward walk that answers the
  // handler-path question for a whole group at once.
  llvm::DenseMap<mlir::Block *, llvm::SmallVector<mlir::Block *, 2>>
      exceptionPreds;

  // A reachability set, plus whether it is authoritative: a walk that left the
  // numbered blocks cannot be answered from bits, and the caller re-asks
  // directly instead of reading a set that is missing part of the CFG.
  struct ReachSet {
    llvm::BitVector members;
    bool exact = true;
  };
  llvm::DenseMap<std::pair<mlir::Block *, mlir::Block *>, ReachSet> forwardSets;
  llvm::DenseMap<std::pair<mlir::Operation *, mlir::Block *>, ReachSet>
      forwardAfterSets;
  // Keyed by group address: the group fixes both the avoided block and the set
  // of use blocks. Safe because every query happens after `groups` is fully
  // built (the cleanup records hold pointers into it for the same reason).
  llvm::DenseMap<const UnwindTrackedGroup *, ReachSet> useReachingSets;

  UnwindReachMode mode = UnwindReachMode::Cached;
  // CFG walks performed and blocks popped, printed per function under
  // LYTHON_PERF. These are the shape of the cost, not a restatement of the
  // wall time: the marker loop asked #markers x #groups x #sites walks, and
  // what says the product is gone is that `walks` stops moving when markers
  // are added.
  std::uint64_t walks = 0;
  std::uint64_t walkNodes = 0;
  // `verify` mode only: questions asked of both formulations, and the ones they
  // answered differently. Printed even when zero divergences, because a checker
  // that reports nothing is indistinguishable from a checker that checked
  // nothing (rfc/stdlib-semantics.md 13g).
  std::uint64_t verifiedQueries = 0;
  std::uint64_t divergences = 0;

  explicit UnwindCleanupAnalysis(mlir::func::FuncOp fn)
      : function(fn), dominance(fn),
        exceptionEdges(own::collectExceptionEdges(fn.getBody())),
        mode(unwindReachMode()) {
    for (mlir::Block &block : fn.getBody()) {
      blockIndex.insert({&block, static_cast<unsigned>(blocks.size())});
      blocks.push_back(&block);
    }
    for (auto &[block, successors] : exceptionEdges)
      for (mlir::Block *successor : successors)
        exceptionPreds[successor].push_back(block);

    llvm::DenseMap<std::int64_t, mlir::Block *> handlers =
        own::collectExceptionHandlerEntries(fn.getBody());
    if (handlers.empty())
      return;
    for (mlir::Block &block : fn.getBody()) {
      for (mlir::Operation &op : block) {
        auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
        if (!call || call.getCallee() != "LyEH_TryCallSiteMarker")
          continue;
        std::optional<std::int64_t> id = own::exceptionMarkerId(call);
        if (!id)
          continue;
        auto handler = handlers.find(*id);
        if (handler != handlers.end())
          markerEdges[&block].push_back({&op, handler->second});
      }
    }
  }

  ~UnwindCleanupAnalysis() {
    if (mode != UnwindReachMode::Verify || verifiedQueries == 0)
      return;
    llvm::errs() << "[unwind-reach-verify] @" << function.getName()
                 << " queries=" << verifiedQueries
                 << " divergences=" << divergences << "\n";
  }

  std::optional<unsigned> indexOf(mlir::Block *block) const {
    auto found = blockIndex.find(block);
    if (found == blockIndex.end())
      return std::nullopt;
    return found->second;
  }

  // In `verify` mode: does the memoised answer match the walk's? Returns the
  // memoised one either way -- the point is to REPORT a divergence on the IR
  // the shipped path produces, not to paper over it with the other answer.
  bool agree(bool cached, bool direct, llvm::StringRef question) {
    ++verifiedQueries;
    if (cached == direct)
      return cached;
    ++divergences;
    llvm::errs() << "[unwind-reach-verify] DIVERGENCE @" << function.getName()
                 << " question=" << question << " cached=" << cached
                 << " walk=" << direct << "\n";
    return cached;
  }

  const llvm::SmallPtrSet<mlir::Block *, 16> &reachableFrom(mlir::Block *from) {
    auto cached = reachableCache.find(from);
    if (cached != reachableCache.end())
      return cached->second;
    llvm::SmallPtrSet<mlir::Block *, 16> &reachable = reachableCache[from];
    llvm::SmallVector<mlir::Block *, 16> worklist;
    auto enqueue = [&](mlir::Block *block) {
      if (reachable.insert(block).second)
        worklist.push_back(block);
    };
    auto enqueueSuccessors = [&](mlir::Block *block) {
      for (mlir::Block *successor : block->getSuccessors())
        enqueue(successor);
      if (auto found = exceptionEdges.find(block);
          found != exceptionEdges.end())
        for (mlir::Block *successor : found->second)
          enqueue(successor);
    };
    enqueueSuccessors(from);
    while (!worklist.empty())
      enqueueSuccessors(worklist.pop_back_val());
    return reachable;
  }

  bool reaches(mlir::Block *from, mlir::Block *to) {
    return reachableFrom(from).count(to) != 0;
  }

  // Can control flow from `from` reach `to` without passing through
  // `avoid`? Successors plus exception edges, like reachableFrom. Paths
  // through `avoid` do not count: for a group defined in `avoid`, they
  // re-arm the token before `to` sees it. When `fromAfter` is set (an op in
  // `from`), the FIRST hop only takes exception edges of markers AFTER it:
  // an unwind at an earlier marked call happens before `fromAfter` ran, so
  // its edge cannot carry `fromAfter`'s effect.
  //
  // `to` is a membership test on a set fixed by (from, avoid, fromAfter), and
  // the marker loop asks the same (from, avoid, fromAfter) for every exit
  // point in the function. So the set is built once and cached; the walk below
  // is what the answer USED to cost per question.
  bool reachesAvoiding(mlir::Block *from, mlir::Block *to, mlir::Block *avoid,
                       mlir::Operation *fromAfter = nullptr) {
    if (mode == UnwindReachMode::Walk)
      return walkReachesAvoiding(from, to, avoid, fromAfter);
    std::optional<unsigned> target = indexOf(to);
    if (target && indexOf(from)) {
      const ReachSet &set = reachSetAvoiding(from, avoid, fromAfter);
      if (set.exact) {
        bool cached = set.members.test(*target);
        if (mode == UnwindReachMode::Verify)
          return agree(cached, walkReachesAvoiding(from, to, avoid, fromAfter),
                       "reaches-avoiding");
        return cached;
      }
    }
    return walkReachesAvoiding(from, to, avoid, fromAfter);
  }

  // The direct walk: one question, one traversal, early exit at `to`. Retained
  // as the reference formulation behind LYTHON_UNWIND_REACH_CACHE=0 and as the
  // fallback for a block outside the numbering.
  bool walkReachesAvoiding(mlir::Block *from, mlir::Block *to,
                           mlir::Block *avoid,
                           mlir::Operation *fromAfter = nullptr) {
    ++walks;
    llvm::SmallPtrSet<mlir::Block *, 16> visited;
    llvm::SmallVector<mlir::Block *, 16> worklist;
    auto enqueue = [&](mlir::Block *block) {
      if (block == avoid)
        return;
      if (visited.insert(block).second)
        worklist.push_back(block);
    };
    seedFirstHop(from, fromAfter, enqueue);
    auto enqueueSuccessors = [&](mlir::Block *block) {
      for (mlir::Block *successor : block->getSuccessors())
        enqueue(successor);
      if (auto found = exceptionEdges.find(block);
          found != exceptionEdges.end())
        for (mlir::Block *successor : found->second)
          enqueue(successor);
    };
    while (!worklist.empty()) {
      mlir::Block *block = worklist.pop_back_val();
      ++walkNodes;
      if (block == to)
        return true;
      enqueueSuccessors(block);
    }
    return false;
  }

  // The first hop, shared by the walk and the set build so the two cannot
  // drift: successors always, exception edges of the whole block when there is
  // no `fromAfter`, and only the edges of markers `fromAfter` reaches when
  // there is one. `avoid` is not a parameter because both callers' `enqueue`
  // filters it, and one filter cannot disagree with itself.
  template <typename EnqueueFn>
  void seedFirstHop(mlir::Block *from, mlir::Operation *fromAfter,
                    EnqueueFn enqueue) {
    for (mlir::Block *successor : from->getSuccessors())
      enqueue(successor);
    if (!fromAfter) {
      if (auto found = exceptionEdges.find(from); found != exceptionEdges.end())
        for (mlir::Block *successor : found->second)
          enqueue(successor);
      return;
    }
    if (auto found = markerEdges.find(from); found != markerEdges.end())
      for (auto &[marker, handler] : found->second)
        // The edge carries `fromAfter`'s effect when the op ran before the
        // unwind -- or IS the guarded call itself: a transfer/release
        // consumes its operand DURING the unwinding call.
        if (fromAfter->isBeforeInBlock(marker) ||
            own::guardedCallAfterMarker(marker) == fromAfter)
          enqueue(handler);
  }

  const ReachSet &reachSetAvoiding(mlir::Block *from, mlir::Block *avoid,
                                   mlir::Operation *fromAfter) {
    if (!fromAfter) {
      auto found = forwardSets.find({from, avoid});
      if (found != forwardSets.end())
        return found->second;
    } else {
      auto found = forwardAfterSets.find({fromAfter, avoid});
      if (found != forwardAfterSets.end())
        return found->second;
    }

    ++walks;
    ReachSet set;
    set.members.resize(blocks.size());
    llvm::SmallVector<mlir::Block *, 16> worklist;
    auto enqueue = [&](mlir::Block *block) {
      if (block == avoid)
        return;
      std::optional<unsigned> index = indexOf(block);
      if (!index) {
        set.exact = false;
        return;
      }
      if (!set.members.test(*index)) {
        set.members.set(*index);
        worklist.push_back(block);
      }
    };
    seedFirstHop(from, fromAfter, enqueue);
    while (!worklist.empty()) {
      mlir::Block *block = worklist.pop_back_val();
      ++walkNodes;
      for (mlir::Block *successor : block->getSuccessors())
        enqueue(successor);
      if (auto found = exceptionEdges.find(block);
          found != exceptionEdges.end())
        for (mlir::Block *successor : found->second)
          enqueue(successor);
    }
    if (!fromAfter)
      return forwardSets.insert({{from, avoid}, std::move(set)}).first->second;
    return forwardAfterSets.insert({{fromAfter, avoid}, std::move(set)})
        .first->second;
  }

  // The blocks from which SOME block in `targets` is reachable in one or more
  // edges without passing through `avoid` -- `reachesAvoiding(x, t, avoid)` for
  // every t at once, read backwards. The handler-path question asks exactly
  // that for a group's use blocks, with the handler as `x`, so one backward
  // walk per group replaces one forward walk per (exit point, use).
  const ReachSet &useReachingSetFor(const UnwindTrackedGroup &group,
                                    mlir::Block *avoid) {
    auto found = useReachingSets.find(&group);
    if (found != useReachingSets.end())
      return found->second;
    return useReachingSets
        .insert({&group, buildReachingSet(group.useSites, avoid)})
        .first->second;
  }

  ReachSet buildReachingSet(llvm::ArrayRef<mlir::Operation *> targets,
                            mlir::Block *avoid) {
    ++walks;
    ReachSet set;
    set.members.resize(blocks.size());
    llvm::BitVector onPath(blocks.size());
    llvm::SmallVector<mlir::Block *, 16> worklist;
    // A path's non-initial nodes must all avoid `avoid`, the target included;
    // the start node is not constrained, which is why `members` and `onPath`
    // are separate sets rather than one.
    for (mlir::Operation *target : targets) {
      mlir::Block *block = target->getBlock();
      if (block == avoid)
        continue;
      std::optional<unsigned> index = indexOf(block);
      if (!index) {
        set.exact = false;
        continue;
      }
      if (!onPath.test(*index)) {
        onPath.set(*index);
        worklist.push_back(block);
      }
    }
    auto visitPredecessor = [&](mlir::Block *pred) {
      std::optional<unsigned> index = indexOf(pred);
      if (!index) {
        set.exact = false;
        return;
      }
      set.members.set(*index);
      if (pred == avoid)
        return;
      if (!onPath.test(*index)) {
        onPath.set(*index);
        worklist.push_back(pred);
      }
    };
    while (!worklist.empty()) {
      mlir::Block *block = worklist.pop_back_val();
      ++walkNodes;
      for (mlir::Block *pred : block->getPredecessors())
        visitPredecessor(pred);
      if (auto found = exceptionPreds.find(block);
          found != exceptionPreds.end())
        for (mlir::Block *pred : found->second)
          visitPredecessor(pred);
    }
    return set;
  }
};

// Does `terminator` forward every value of `group` into successor
// #succIndex's block arguments (the edge on which the affine token moves to
// the destination argument group)?
bool terminatorForwardsGroupToSuccessor(mlir::Operation *terminator,
                                        unsigned succIndex,
                                        llvm::ArrayRef<mlir::Value> group,
                                        own::AliasAnalysis &aliases) {
  auto branch = mlir::dyn_cast<mlir::BranchOpInterface>(terminator);
  if (!branch)
    return false;
  mlir::SuccessorOperands operands = branch.getSuccessorOperands(succIndex);
  for (mlir::Value value : group) {
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
}

TokenAtPoint groupTokenAtPoint(UnwindCleanupAnalysis &analysis,
                               const UnwindTrackedGroup &group,
                               mlir::Operation *point,
                               own::AliasAnalysis &aliases) {
  if (group.skip || group.values.empty())
    return TokenAtPoint::Unknown;
  mlir::Value root = group.values.front();
  if (!analysis.dominance.properlyDominates(root, point))
    return TokenAtPoint::NotHeld;

  mlir::Block *pointBlock = point->getBlock();
  mlir::Operation *producer = root.getDefiningOp();
  // Every entry of the defining block re-arms the token (the producer runs
  // again / the block argument is re-bound), so a consume only matters at
  // `point` when a path from it reaches `point` while avoiding defBlock.
  mlir::Block *defBlock = producer
                              ? producer->getBlock()
                              : mlir::cast<mlir::BlockArgument>(root).getOwner();
  TokenAtPoint result = TokenAtPoint::Held;
  for (mlir::Operation *consume : group.consumeSites) {
    if (consume == point)
      continue;
    // A block-argument forward consumes the token only on the FORWARDING
    // edges: `cond_br %c, ^merge(%g...), ^other` still holds this
    // incarnation on the ^other edge, so treating the terminator like a
    // point consume would classify every post-branch point as token-free
    // and silently skip its cleanup (the eq/ne-merge string shape).
    if (consume->hasTrait<mlir::OpTrait::IsTerminator>() &&
        consume->getNumSuccessors() != 0) {
      if (!mlir::isa<mlir::BranchOpInterface>(consume))
        return TokenAtPoint::Unknown;
      for (unsigned index = 0, end = consume->getNumSuccessors(); index < end;
           ++index) {
        if (!terminatorForwardsGroupToSuccessor(consume, index, group.values,
                                                aliases))
          continue;
        mlir::Block *successor = consume->getSuccessor(index);
        if (successor == defBlock)
          continue; // the entry re-arms the token
        if (successor == pointBlock) {
          // The dead-token state arrives at the point's block on this edge,
          // but other predecessors may still carry the token.
          result = TokenAtPoint::Unknown;
          continue;
        }
        if (!analysis.reachesAvoiding(successor, pointBlock, defBlock))
          continue;
        if (analysis.dominance.dominates(successor, pointBlock))
          return TokenAtPoint::NotHeld;
        result = TokenAtPoint::Unknown;
      }
      continue;
    }
    if (consume->getBlock() == pointBlock &&
        consume->isBeforeInBlock(point)) {
      // Same pass through the block. A producer BETWEEN the consume and the
      // point re-arms the token textually (the consume released the
      // previous iteration's token).
      if (producer && producer->getBlock() == pointBlock &&
          consume->isBeforeInBlock(producer))
        continue;
      return TokenAtPoint::NotHeld;
    }
    // Later in the same block, or another block: only a path that reaches
    // the point without re-arming at defBlock carries the consumed state
    // (an unwind at a marker BEFORE the consume happens before it ran, so
    // those edges do not carry it).
    if (!analysis.reachesAvoiding(consume->getBlock(), pointBlock, defBlock,
                                  consume))
      continue;
    if (consume->getBlock() != pointBlock &&
        analysis.dominance.properlyDominates(consume, point))
      return TokenAtPoint::NotHeld;
    result = TokenAtPoint::Unknown;
  }
  return result;
}

// Is `op` syntactically after a no-return raise in its block? The normal-path
// release machinery parks releases before the TERMINATOR, which in a raising
// block is after the raise call: such a release never executes (the raise
// unwinds first), so counting it as a consume would poison the token state of
// every point its block can (exceptionally) reach with a false "maybe
// already released" -- exactly the mixed state that made loop bodies with a
// guarded raise skip their per-call-site cleanups.
// Is `op` unreachable because a raise-like call already ran earlier in its
// block? A raise primitive never returns, so everything after it in the block
// is dead code and its effects on a token are not effects at all.
//
// ONE FACT PER BLOCK, asked once. Written as a backward scan -- every preceding
// operation, with a contract lookup on each call -- it cost O(ops) per query
// and the query runs once per consuming use, which is O(ops) of them: measured
// at 5.9 s of a 45.7 s compile of a 400-statement module, all of it re-deriving
// where the block's first raise is. Whether ANY raise precedes `op` is the same
// question as whether the FIRST one does.
class DeadAfterRaiseCache {
public:
  bool deadAfter(FuncContractCache &contracts, mlir::Operation *op) {
    mlir::Block *block = op->getBlock();
    auto found = firstRaise.find(block);
    if (found == firstRaise.end())
      found = firstRaise.insert({block, findFirstRaise(contracts, block)}).first;
    return found->second && found->second->isBeforeInBlock(op);
  }

private:
  static mlir::Operation *findFirstRaise(FuncContractCache &contracts,
                                         mlir::Block *block) {
    for (mlir::Operation &op : *block) {
      auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
      if (!call)
        continue;
      auto cached = contracts.lookup(call.getCallee());
      if (mlir::succeeded(cached) && *cached &&
          own::isRaiseLikeFunction((*cached)->function))
        return &op;
    }
    return nullptr;
  }

  llvm::DenseMap<mlir::Block *, mlir::Operation *> firstRaise;
};



void collectUnwindGroupSites(FuncContractCache &contracts,
                             own::AliasAnalysis &aliases,
                             const own::ReferenceMap &references,
                             mlir::Region *region,
                             mlir::DominanceInfo &dominance,
                             DeadAfterRaiseCache &deadAfterRaise,
                             UnwindTrackedGroup &group) {
  // Operations before the producer belong to the token's production (e.g.
  // the boxing Ly_IncRef that mints an owned-local token): the token walk
  // starts after the producer, so pre-producer users are not group effects.
  mlir::Operation *producer =
      group.values.empty() ? nullptr : group.values.front().getDefiningOp();
  auto precedesProduction = [&](mlir::Operation *user) {
    return producer && user != producer &&
           dominance.properlyDominates(user, producer);
  };
  llvm::SmallVector<mlir::Value, 8> tracked(group.values.begin(),
                                            group.values.end());
  tracked.append(group.views.begin(), group.views.end());
  {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    for (mlir::Value value : group.values) {
      llvm::SmallVector<mlir::Value, 8> aliasValues;
      aliases.namesOf(value, aliasValues);
      equivalents.append(aliasValues.begin(), aliasValues.end());
    }
    own::collectBoxWordDerivedViews(equivalents, tracked);
  }

  llvm::SmallPtrSet<mlir::Operation *, 16> seenUses;
  llvm::SmallPtrSet<mlir::Operation *, 8> seenConsumes;
  for (mlir::Value value : tracked) {
    llvm::SmallVector<mlir::Value, 8> equivalents;
    aliases.namesOf(value, equivalents);
    for (mlir::Value equivalent : equivalents) {
      for (mlir::OpOperand &use : equivalent.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (precedesProduction(user))
          continue;
        mlir::Operation *top = ancestorInRegion(user, region);
        if (!top) {
          group.skip = true;
          return;
        }
        if (seenUses.insert(top).second)
          group.useSites.push_back(top);

        if (auto call = mlir::dyn_cast<mlir::func::CallOp>(user)) {
          // This analysis was the last one still reading a container's release
          // as a minted token's consume, so such a token looked already gone at
          // every unwind point and no cleanup was written for it: `try:
          // print(zs[0] / zs[1])` leaked both elements, 80 B, on the path the
          // exception actually takes.
          //
          // `group.reference`, not a locally recomputed one: the obligation this
          // analysis RECORDED is what LYTHON_ABLATE_UNWIND_MINTED_TOKENS zeroes,
          // so asking it keeps the switch covering the consume scan and the
          // dedup together instead of half of what it names.
          if (consumeIsAnotherReferencesDischarge(references, use, group.values,
                                                 equivalent, group.reference))
            continue;
          if (callPartiallyConsumesGroup(contracts, call, group.values,
                                         aliases) ||
              callConsumesTrackedHeader(contracts, call, group.values,
                                        aliases)) {
            group.skip = true;
            return;
          }
          if (callConsumesGroup(contracts, call, group.values, aliases)) {
            if (!deadAfterRaise.deadAfter(contracts, call) &&
                seenConsumes.insert(top).second)
              group.consumeSites.push_back(top);
            continue;
          }
          if (callRetainsGroup(contracts, call, group.values, aliases) &&
              !call->hasAttr(own::kAggregateRetainAttr)) {
            // A live extra token under no name this walk can charge it to: the
            // balance at an unwind point is no longer just held/consumed.
            group.skip = true;
            return;
          }
          // A LENT MERGE TOKEN IS SOMEBODY ELSE'S INCREMENT, not an unknown
          // balance. It pays for the destination argument group -- which this
          // analysis tracks separately and writes its own cleanup for -- and
          // leaves this group holding exactly what it held before. Bailing on it
          // dropped the source from the analysis entirely, so a raise out of a
          // `for` body inside `try` left the iterated container with no cleanup
          // at all: `try:\n for k in d: d["boom"] = 0` leaked the whole dict,
          // 17 KB, on the path the mutation guard actually takes.
          continue;
        }
        if (user->hasTrait<mlir::OpTrait::IsTerminator>() &&
            user->getBlock()->getParent() == region &&
            branchForwardsGroupToBlockArgument(user, group.values, aliases)) {
          // The token moves to the destination argument group on the
          // forwarding edge; treat the terminator as consuming so a point
          // past the merge never sees this group as held (the destination
          // group covers it there).
          //
          // Unless the edge LENDS (`blockLendsGroupToMergeArgument`): then the
          // destination was given an increment of its own and nothing moved, so
          // reading the forward as a consume would retire a token that is still
          // held.
          if (!blockLendsGroupToMergeArgument(user->getBlock(), group.values,
                                              aliases) &&
              !deadAfterRaise.deadAfter(contracts, user) &&
              seenConsumes.insert(user).second)
            group.consumeSites.push_back(user);
        }
      }
    }
  }
}

// Does the handler's own path still use this group -- i.e. do the handler-side
// releases already own the token?
//
// The answer is a disjunction over the group's use sites, so it is the same
// question as "is `handler` one of the blocks from which a use is reachable
// avoiding defBlock". Asked that way it costs one BACKWARD walk per group
// instead of one forward walk per (handler, use) pair, and the marker loop then
// answers it with a bit test. `avoid` and the targets both come from the group,
// which is what makes the walk shareable across every handler in the function.
bool groupUsedOnHandlerPath(UnwindCleanupAnalysis &analysis,
                            const UnwindTrackedGroup &group,
                            mlir::Block *handler) {
  if (group.values.empty())
    return true;
  mlir::Value root = group.values.front();
  mlir::Operation *producer = root.getDefiningOp();
  mlir::Block *defBlock = producer
                              ? producer->getBlock()
                              : mlir::cast<mlir::BlockArgument>(root).getOwner();
  // A use IN the handler counts whatever the CFG says (no edge required), so it
  // is not part of the reachability question either way.
  for (mlir::Operation *use : group.useSites)
    if (use->getBlock() == handler)
      return true;

  // A use only counts when the handler reaches it WITHOUT re-entering the
  // defining block: a path through defBlock re-arms the token, so the use
  // belongs to the next incarnation, not the one unwinding now (try inside a
  // loop).
  auto direct = [&] {
    for (mlir::Operation *use : group.useSites)
      if (analysis.walkReachesAvoiding(handler, use->getBlock(), defBlock))
        return true;
    return false;
  };
  if (analysis.mode == UnwindReachMode::Walk)
    return direct();

  std::optional<unsigned> index = analysis.indexOf(handler);
  if (!index)
    return direct();
  const UnwindCleanupAnalysis::ReachSet &set =
      analysis.useReachingSetFor(group, defBlock);
  if (!set.exact)
    return direct();
  bool cached = set.members.test(*index);
  if (analysis.mode == UnwindReachMode::Verify)
    return analysis.agree(cached, direct(), "handler-path");
  return cached;
}

// LYTHON_UNWIND_TRACE=1: one line per marker-cleanup decision, plus a decision
// COUNT at exit.
//
// Why the count: a trace that prints nothing looks the same whether the
// predicate answered "handler owns it" everywhere or the loop never reached a
// decision at all. Three of this session's wrong conclusions came from reading
// an empty instrument as a measurement (rfc/stdlib-semantics.md 13j-3/13j-9),
// so this one reports its own denominator.
struct UnwindTrace {
  bool enabled = false;
  std::uint64_t decisions = 0;

  UnwindTrace() {
    auto value = llvm::sys::Process::GetEnv("LYTHON_UNWIND_TRACE");
    enabled = value && (*value == "1");
  }

  ~UnwindTrace() {
    if (enabled)
      llvm::errs() << "[UNWIND_TRACE] marker_decisions=" << decisions << "\n";
  }

  void marker(UnwindCleanupAnalysis &analysis, const UnwindTrackedGroup &group,
              mlir::Block *handler, bool handlerOwns);
};

UnwindTrace &unwindTrace() {
  static UnwindTrace trace;
  return trace;
}

void UnwindTrace::marker(UnwindCleanupAnalysis &analysis,
                         const UnwindTrackedGroup &group, mlir::Block *handler,
                         bool handlerOwns) {
  if (!enabled)
    return;
  ++decisions;
  auto blockName = [&](mlir::Block *block) -> std::string {
    if (!block)
      return "<null>";
    if (std::optional<unsigned> index = analysis.indexOf(block))
      return ("bb" + llvm::Twine(*index)).str();
    return "<unnumbered>";
  };
  mlir::Value root = group.values.front();
  mlir::Operation *producer = root.getDefiningOp();
  mlir::Block *defBlock = producer
                              ? producer->getBlock()
                              : mlir::cast<mlir::BlockArgument>(root).getOwner();
  llvm::errs() << "[UNWIND_TRACE] fn=" << analysis.function.getName()
               << " dealloc="
               << mlir::func::FuncOp(group.deallocator->function).getName()
               << " root=" << (producer ? "op" : "blockarg")
               << " def=" << blockName(defBlock)
               << " handler=" << blockName(handler)
               << " handler_owns=" << (handlerOwns ? "yes" : "no")
               << " use_blocks=";
  for (mlir::Operation *use : group.useSites)
    llvm::errs() << blockName(use->getBlock()) << ",";
  llvm::errs() << " nuses=" << group.useSites.size() << "\n";
}

// Wall time of `insertUnwindCleanupReleases`' sub-steps for ONE function,
// reported as a single line under LYTHON_PERF when this function costs more than
// `kReportThresholdUs`.
//
// Why per function with a threshold rather than a PerfScope per step: PerfScope
// prints one line per construction and does not accumulate, so five scopes
// inside a per-function walk emit five lines for every function in the module --
// thousands, of which a handful matter. A threshold keeps the output at the
// functions that dominate while still being a measurement rather than a sample.
class UnwindStepTimer {
public:
  enum Step { Analysis, Groups, Markers, Raises, Calls, Nested, Mutate, Count };

  explicit UnwindStepTimer(mlir::func::FuncOp function)
      : name(function.getName()), last(Clock::now()) {
    static const bool on = [] {
      auto value = llvm::sys::Process::GetEnv("LYTHON_PERF");
      return value && (*value == "1" || llvm::StringRef(*value).equals_insensitive("true") ||
                       llvm::StringRef(*value).equals_insensitive("yes") ||
                       llvm::StringRef(*value).equals_insensitive("on"));
    }();
    enabled = on;
  }

  void mark(Step step) {
    if (!enabled)
      return;
    auto now = Clock::now();
    us[step] += std::chrono::duration_cast<std::chrono::microseconds>(now - last)
                    .count();
    last = now;
  }

  ~UnwindStepTimer() {
    if (!enabled)
      return;
    std::uint64_t total = 0;
    for (std::uint64_t value : us)
      total += value;
    if (total < kReportThresholdUs)
      return;
    static const char *kNames[Count] = {"analysis",    "groups", "markers",
                                        "raises",      "calls",  "nested",
                                        "mutate"};
    llvm::errs() << "[LYTHON_PERF] unwind-cleanup-releases @" << name
                 << " total_us=" << total;
    for (unsigned step = 0; step < Count; ++step)
      llvm::errs() << " " << kNames[step] << "_us=" << us[step];
    llvm::errs() << " blocks=" << blocks << " markers=" << markers
                 << " groups=" << groups << " walks=" << walks
                 << " walk_nodes=" << walkNodes << "\n";
  }

  // The SHAPE of the work, next to its duration. A time says a step is
  // expensive; `walks` says whether it is expensive per exit point or per
  // function, which is the difference between a constant and a product and the
  // only number that shows the product coming back.
  std::uint64_t blocks = 0;
  std::uint64_t markers = 0;
  std::uint64_t groups = 0;
  std::uint64_t walks = 0;
  std::uint64_t walkNodes = 0;

private:
  using Clock = std::chrono::steady_clock;
  static constexpr std::uint64_t kReportThresholdUs = 50000;

  std::string name;
  bool enabled = false;
  Clock::time_point last;
  std::uint64_t us[Count] = {};
};

// Is this block one THIS pass generated on an earlier run -- a cleanup handler
// whose body is `catch marker; outlined releaser; branch-or-rethrow`?
//
// The pass runs twice (`PostCleanupUnwindInsertionPass` re-runs it after
// canonicalization hoists calls out of folded region ops), and the second run
// re-scans the function from scratch. A cleanup handler that continues the
// unwind out of the frame ends in `LyEH_RethrowCurrent` with no call-site
// marker in front of it, which is byte-identical to what an unguarded raise
// looks like -- so the re-run classified its own handler as a new exceptional
// exit point and inserted an inline release there.
//
// Why that is a double release rather than a missing one: when the re-run finds
// a token newly held at a call site whose marker already has a handler, it
// CHAINS -- a fresh cleanup block releases the new token and branches to the
// existing handler. The inline release then runs on the same unwind, after the
// chained block already released it. `groupTokenAtPoint` cannot see this: the
// chained release is in one predecessor of the shared handler, so it dominates
// nothing, and the token reads as Held.
//
// Why NOT fix it by ordering the two shapes (skip a raise cleanup for a group
// some predecessor releases): a cleanup handler is SHARED by every call site
// that unwinds into it, so "the token is held here" is not a property of the
// block at all -- it is a property of each incoming unwind, which is exactly
// what the marker/chain shape expresses and an inline release cannot. The
// residual is on the incomplete side: a token held at a cleanup handler's
// rethrow and covered by no chain leaks rather than being freed twice.
bool isGeneratedUnwindCleanupBlock(mlir::Block *block) {
  for (mlir::Operation &op : *block)
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(&op))
      if (call.getCallee().starts_with("__ly_unwind_cleanup_"))
        return true;
  return false;
}

// One outlined releaser per cleanup requirement. The DecRefs could sit
// directly in the cleanup block, but structurally identical cleanup blocks
// would then be merged by aggressive region simplification (in canonicalizer
// runs and greedy conversion passes we do not control), turning the
// per-block marker id into a block argument the final EH phase cannot wire.
// A call's callee symbol is an attribute, so distinct outlined callees make
// the blocks non-equivalent and merge-proof at every dialect level.
mlir::func::FuncOp createOutlinedUnwindReleaser(
    mlir::ModuleOp module, mlir::Location loc,
    llvm::ArrayRef<const UnwindTrackedGroup *> groups, unsigned index) {
  mlir::OpBuilder builder(module.getContext());
  std::string name = (llvm::Twine("__ly_unwind_cleanup_") + llvm::Twine(index)).str();

  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<std::int64_t, 4> releaseOffsets;
  for (const UnwindTrackedGroup *group : groups) {
    releaseOffsets.push_back(static_cast<std::int64_t>(inputTypes.size()));
    for (mlir::Value value : group->values)
      inputTypes.push_back(value.getType());
  }

  builder.setInsertionPointToEnd(module.getBody());
  auto function = mlir::func::FuncOp::create(
      builder, loc, name, builder.getFunctionType(inputTypes, {}));
  function.setPrivate();
  function->setAttr(own::kReleaseArgsAttr,
                    builder.getDenseI64ArrayAttr(releaseOffsets));
  for (std::int64_t offset : releaseOffsets)
    function.setArgAttr(static_cast<unsigned>(offset), own::kObjectHeaderAttr,
                        builder.getUnitAttr());

  mlir::Block *body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);
  unsigned offset = 0;
  for (const UnwindTrackedGroup *group : groups) {
    llvm::SmallVector<mlir::Value, 4> operands;
    for (unsigned index2 = 0; index2 < group->values.size(); ++index2)
      operands.push_back(body->getArgument(offset + index2));
    mlir::func::CallOp::create(builder, loc, group->deallocator->function,
                               operands);
    offset += static_cast<unsigned>(group->values.size());
  }
  mlir::func::ReturnOp::create(builder, loc);
  return function;
}

std::int64_t nextUnusedExceptionHandlerId(mlir::ModuleOp module) {
  std::int64_t next = 1;
  module.walk([&](mlir::func::CallOp call) {
    llvm::StringRef callee = call.getCallee();
    if (callee != "LyEH_TryCallSiteMarker" && callee != "LyEH_TryCatchMarker" &&
        callee != "LyEH_TryCatchAnchor")
      return;
    if (std::optional<std::int64_t> id = own::exceptionMarkerId(call))
      next = std::max(next, *id + 1);
  });
  return next;
}

mlir::LogicalResult insertUnwindCleanupReleases(
    mlir::ModuleOp module, FuncContractCache &contracts,
    llvm::ArrayRef<own::RuntimeDeallocator> deallocators,
    own::AliasAnalysis &aliases,
    const own::ReferenceMap &references,
    mlir::SymbolTable *symbols,
    llvm::ArrayRef<own::ResourceGroup> blockArgGroups) {
  std::int64_t nextHandlerId = nextUnusedExceptionHandlerId(module);
  auto anchorFn = module.lookupSymbol<mlir::func::FuncOp>("LyEH_TryCatchAnchor");
  auto catchMarkerFn =
      module.lookupSymbol<mlir::func::FuncOp>("LyEH_TryCatchMarker");
  auto callSiteMarkerFn =
      module.lookupSymbol<mlir::func::FuncOp>("LyEH_TryCallSiteMarker");
  // A module without any try still needs the marker/anchor trio once a
  // may-raise call in a token-holding frame requires an unwind cleanup:
  // declare them on demand exactly like the try lowering would have.
  auto ensureEHMarkerFunctions = [&]() {
    mlir::OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto ensure = [&](mlir::func::FuncOp &slot, llvm::StringRef name,
                      mlir::FunctionType type) {
      if (slot)
        return;
      slot = mlir::func::FuncOp::create(builder, module.getLoc(), name, type);
      slot.setPrivate();
    };
    ensure(anchorFn, "LyEH_TryCatchAnchor",
           builder.getFunctionType({builder.getI64Type()},
                                   {builder.getI1Type()}));
    ensure(catchMarkerFn, "LyEH_TryCatchMarker",
           builder.getFunctionType({builder.getI64Type()}, {}));
    ensure(callSiteMarkerFn, "LyEH_TryCallSiteMarker",
           builder.getFunctionType({builder.getI64Type()}, {}));
  };
  unsigned nextReleaserIndex = 0;
  while (module.lookupSymbol((llvm::Twine("__ly_unwind_cleanup_") +
                              llvm::Twine(nextReleaserIndex))
                                 .str()))
    ++nextReleaserIndex;

  // Runtime-internal pre-lowered modules link after the final EH phase:
  // marker wiring inserted here would survive as unresolved symbols. Inline
  // releases before raise-like calls still apply; only the marker/anchor
  // cleanup shapes are skipped (RuntimePyLowering rejects leftover markers).
  bool deferMarkerWiring = module->hasAttr(own::kRuntimeInternalLoweringAttr);

  mlir::LogicalResult result = mlir::success();
  module.walk([&](mlir::func::FuncOp function) {
    if (mlir::failed(result) || function.isDeclaration() ||
        own::isRuntimeManifestFunction(function))
      return;
    mlir::Region *region = &function.getBody();
    // The final EH phase only invoke-converts functions carrying Python
    // debug info (attachPythonDebugInfo keys off the Python source loc):
    // wiring markers into any other function (manifest-module helpers such
    // as __ly_io_raise) would leave them un-erased -- unresolved LyEH_Try*
    // symbols at program link.
    bool ehPhaseProcessesFunction =
        !deferMarkerWiring &&
        findPythonSourceLoc(function.getLoc()).has_value();

    // Exceptional exit points: guarded call-site markers (unwind to an
    // in-function handler) and unguarded raise primitives (unwind out).
    llvm::DenseMap<std::int64_t, mlir::Block *> handlerEntries =
        own::collectExceptionHandlerEntries(*region);
    llvm::SmallVector<std::pair<mlir::func::CallOp, mlir::Block *>, 8> markers;
    llvm::SmallVector<mlir::func::CallOp, 4> unguardedRaises;
    // Unguarded may-raise calls that RETURN on the normal path: their unwind
    // leaves the function (no local handler), so held tokens need a
    // release-then-rethrow cleanup handler on the unwind edge -- releasing
    // before the call would free values the normal path still uses.
    llvm::SmallVector<mlir::func::CallOp, 8> unguardedMayRaiseCalls;
    for (mlir::Block &block : *region) {
      bool cleanupBlock = isGeneratedUnwindCleanupBlock(&block);
      for (mlir::Operation &op : block) {
        auto call = mlir::dyn_cast<mlir::func::CallOp>(&op);
        if (!call)
          continue;
        if (call.getCallee() == "LyEH_TryCallSiteMarker") {
          std::optional<std::int64_t> id = own::exceptionMarkerId(call);
          if (!id)
            continue;
          auto handler = handlerEntries.find(*id);
          if (handler != handlerEntries.end())
            markers.push_back({call, handler->second});
          continue;
        }
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (own::isRaiseLikeFunction(callee)) {
          if (!own::precedingTryCallSiteMarker(call) && !cleanupBlock)
            unguardedRaises.push_back(call);
          continue;
        }
        if (ehPhaseProcessesFunction && own::mayRaisePythonException(callee) &&
            !own::precedingTryCallSiteMarker(call))
          unguardedMayRaiseCalls.push_back(call);
      }
    }
    // Exit points NESTED in region ops (the int fast/slow scf.if is the
    // typical shape): single-block regions cannot host the anchor/cond_br
    // wiring, so the cleanup is anchored before the region op's TOP-LEVEL
    // ancestor and the in-region marker is re-pointed at it -- the final EH
    // phase pairs markers and calls after flattening, so the runtime edge
    // lands on the cleanup regardless of the nesting here.
    llvm::SmallVector<std::pair<mlir::func::CallOp, mlir::Block *>, 8>
        nestedMarkers;
    llvm::SmallVector<mlir::func::CallOp, 8> nestedUnguardedMayRaiseCalls;
    if (ehPhaseProcessesFunction) {
      function.walk([&](mlir::func::CallOp call) {
        if (call->getParentRegion() == region)
          return; // handled by the top-level scan above
        if (call.getCallee() == "LyEH_TryCallSiteMarker") {
          std::optional<std::int64_t> id = own::exceptionMarkerId(call);
          if (!id)
            return;
          auto handler = handlerEntries.find(*id);
          if (handler != handlerEntries.end())
            nestedMarkers.push_back({call, handler->second});
          return;
        }
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        if (own::isRaiseLikeFunction(callee)) {
          // Raise-like: inline releases before the raise stay valid inside
          // a region, so the top-level shape below handles them (a nested
          // unguarded raise-like call is collected like a top-level one).
          if (!own::precedingTryCallSiteMarker(call))
            unguardedRaises.push_back(call);
          return;
        }
        if (own::mayRaisePythonException(callee) &&
            !own::precedingTryCallSiteMarker(call))
          nestedUnguardedMayRaiseCalls.push_back(call);
      });
    }
    if (markers.empty() && unguardedRaises.empty() &&
        unguardedMayRaiseCalls.empty() && nestedMarkers.empty() &&
        nestedUnguardedMayRaiseCalls.empty())
      return;

    // Per-function stopwatch for the sub-steps below, printed as one line when
    // this function alone costs more than a threshold. The phase-level scope
    // says this step is ~90% of the phase; without this split the next question
    // ("of what?") needs a profiler.
    //
    // This comment used to say the marker loop's #markers x #groups CFG queries
    // were INTRINSIC to the design. They were not: the queries' answers do not
    // depend on the exit point (see `reachesAvoiding` and
    // `groupUsedOnHandlerPath`), so the traversals are now per group and per
    // consume site, and only the O(1) tests are still per cell. What survives
    // of the claim is narrower and still worth knowing -- the marker loop must
    // VISIT every cell, because which groups are held at which exit is the
    // output.
    UnwindStepTimer steps(function);

    // The plan the analysis produces, and the mutation consumes. Declared out
    // here so the analysis itself can live in a scope that ENDS before the
    // first block split: its reachability answers are memoised, and memoised
    // reachability outliving the CFG it describes would present as a MISSING
    // release -- the failure shape this suite has repeatedly been unable to
    // see. A convention would not survive an edit; a scope makes a query after
    // the mutation fail to compile.
    llvm::SmallVector<UnwindTrackedGroup, 16> groups;
    // ONE exceptional exit whose held tokens must be released on the unwind
    // edge. The four types this replaced -- guarded/unguarded x
    // top-level/nested -- differed in exactly two independent facts and in
    // nothing else, and each carried its own mutation loop that re-derived the
    // same five steps. The facts are now fields, and the steps are written once.
    struct UnwindCleanup {
      // Either the call-site marker that already guards the exit, or the
      // unguarded call that needs one minted before it.
      //
      // Why NOT a second field saying which: the op answers it (its callee is
      // `LyEH_TryCallSiteMarker` or it is not), and a field would be a copy of
      // that answer for the analysis and the mutation to disagree about.
      mlir::func::CallOp site;
      // The in-function handler to branch to once released, or null: the unwind
      // leaves the function and the cleanup rethrows instead.
      mlir::Block *handler;
      // Where the anchor's cond_br splits the block. The site itself, except
      // when the site sits inside a region op -- a single-block region cannot
      // host the wiring, so it goes before that op's top-level ancestor and the
      // in-region marker still points at the cleanup id (the final EH phase
      // pairs markers with calls after flattening, so the runtime edge lands on
      // the cleanup either way).
      mlir::Operation *anchorBefore;
      llvm::SmallVector<const UnwindTrackedGroup *, 4> groups;
    };
    llvm::SmallVector<UnwindCleanup, 8> cleanups;

    // NOT one of the above, which is why it is not in that list: a raise
    // primitive never returns, so its held tokens are released INLINE before
    // the call and no marker, anchor, or cleanup block is involved. It shares
    // the collection pass, not a single step of the mutation.
    struct InlineReleaseBeforeRaise {
      mlir::func::CallOp raiseCall;
      llvm::SmallVector<const UnwindTrackedGroup *, 4> groups;
    };
    llvm::SmallVector<InlineReleaseBeforeRaise, 4> inlineReleases;

    { // ---- analysis: the CFG is FROZEN from here to the closing brace ----
      UnwindCleanupAnalysis analysis(function);
      steps.mark(UnwindStepTimer::Analysis);

      // Owned groups whose token could be held at an exceptional exit.
      llvm::DenseSet<std::pair<own::Reference, mlir::Value>> seenGroupKeys;
      DeadAfterRaiseCache deadAfterRaise;
      auto addGroup = [&](const own::ResourceGroup &g) {
        // Only a MINTED reference discriminates here. A received one is the
        // obligation a region merge maps several producers onto, and telling
        // those apart would track one token as several and release it once per
        // cleanup handler -- which is what the root-only dedup was for.
        own::Reference groupReference;
        if (own::unwindTracksMintedTokensSeparately() && !g.values.empty()) {
          own::Reference head = references.of(g.values.front());
          if (references.isMinted(head))
            groupReference = head;
        }
        if (!g.deallocator || g.condition || g.values.empty())
          return;
        UnwindTrackedGroup tracked;
        tracked.values.assign(g.values.begin(), g.values.end());
        tracked.views.assign(g.views.begin(), g.views.end());
        tracked.deallocator = g.deallocator;
        // A group born inside a region op (the int fast/slow scf.if is the
        // typical shape) holds its token OUTSIDE the region under the parent
        // op's result names; tracking the in-region names would classify the
        // token as never-held at every top-level exit point (the in-region
        // value dominates nothing out there) and silently skip its cleanup.
        // Map through the region exits to the function level; a group whose
        // token never leaves its region is dropped -- top-level exit points
        // never hold it, and in-region exit points stay outside this model.
        mlir::Block *definingBlock = tracked.values.front().getParentBlock();
        while (definingBlock && definingBlock->getParentOp() &&
               !mlir::isa<mlir::func::FuncOp>(definingBlock->getParentOp())) {
          mlir::Operation *terminator = definingBlock->getTerminator();
          if (!terminator)
            return;
          auto mappedValues = mapRegionTerminatorGroupToParentResults(
              terminator, tracked.values, aliases);
          if (!mappedValues)
            return;
          llvm::SmallVector<mlir::Value, 4> mappedViews;
          for (mlir::Value view : tracked.views) {
            auto mappedView = mapRegionTerminatorGroupToParentResults(
                terminator, {view}, aliases);
            if (mappedView)
              mappedViews.push_back(mappedView->front());
            // An unmapped view has no uses outside the region: dropping it
            // only removes pins the outer points cannot observe anyway.
          }
          tracked.values = std::move(*mappedValues);
          tracked.views = std::move(mappedViews);
          definingBlock = definingBlock->getParentOp()->getBlock();
        }
        // Region merges can map several in-region producers onto ONE parent
        // result group (both arms yield into the same result lanes); tracking
        // it twice would release the same token twice in one cleanup handler.
        //
        // Same entity is NOT the same obligation, though, which is why the
        // reference has to match too (`UnwindTrackedGroup::reference`). A minted
        // marker shares an entity root with the reference it was minted on, so
        // the root test alone dropped it from this analysis entirely -- nothing
        // ever asked whether ITS token was held on an unwinding edge.
        // Indexed, not scanned. The scan this replaces asked every group
        // already collected whether it was this one, and `addGroup` runs once
        // per owned group in the function: 3,205 groups in a 400-statement
        // module is 5.1 MILLION `sameEntityRoot` calls, and this phase's group
        // collection measured 6.8 s of a 45.7 s compile with the CFG still one
        // block -- all of it here.
        //
        // The key is exactly the scan's condition: the same reference AND the
        // same entity root. A group with no resolvable root keeps out of the
        // index entirely, which is what `sameEntityRoot` already said about it
        // -- it matches nothing, including another rootless group.
        mlir::Value entityRoot = own::entityRootOf(tracked.values);
        if (entityRoot &&
            !seenGroupKeys.insert({groupReference, entityRoot}).second)
          return;
        // The parity instrument's value is that it examines every pair, which
        // is precisely what the index stops doing. It keeps its scan.
        if (own::ownershipRootParityEnabled())
          for (const UnwindTrackedGroup &existing : groups)
            if (existing.reference == groupReference)
              own::reportEntityRootParity("unwindGroupDedup", existing.values,
                                          tracked.values);
        tracked.reference = groupReference;
        collectUnwindGroupSites(contracts, aliases, references, region,
                                analysis.dominance, deadAfterRaise, tracked);
        groups.push_back(std::move(tracked));
      };
      function.walk([&](mlir::func::CallOp call) {
        for (const own::ResourceGroup &g : own::collectOwnedCallResultGroups(
                 module, call, deallocators, symbols))
          addGroup(g);
      });
      function.walk([&](mlir::Operation *op) {
        if (!op->hasAttr(own::kOwnedLocalObjectAttr) &&
            !op->hasAttr(own::kOwnedLocalObjectContractAttr))
          return;
        for (const own::ResourceGroup &g :
             own::collectOwnedLocalObjectGroups(op, deallocators))
          addGroup(g);
      });
      for (const own::ResourceGroup &g : blockArgGroups) {
        if (g.values.empty())
          continue;
        auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(g.values.front());
        if (!blockArg || blockArg.getOwner()->getParent() != region)
          continue;
        addGroup(g);
      }
      steps.mark(UnwindStepTimer::Groups);
      steps.blocks = analysis.blocks.size();
      steps.markers = markers.size() + nestedMarkers.size();
      steps.groups = groups.size();
      if (groups.empty())
        return;

      // Analysis first, mutation second: block splits invalidate dominance.
      // Exit points nested inside region ops are judged at their TOP-LEVEL
      // ancestor: the analysis' blocks/dominance are function-level, so an
      // in-region point has no position there. A consume whose top-level
      // ancestor IS that region op cannot be ordered against the nested exit
      // -- Unknown, never a guess.
      auto tokenAtExitPoint = [&](const UnwindTrackedGroup &group,
                                  mlir::Operation *exitOp) -> TokenAtPoint {
        mlir::Operation *point = exitOp->getParentRegion() == region
                                     ? exitOp
                                     : ancestorInRegion(exitOp, region);
        if (!point)
          return TokenAtPoint::Unknown;
        if (point != exitOp)
          for (mlir::Operation *consume : group.consumeSites)
            if (consume == point)
              return TokenAtPoint::Unknown;
        return groupTokenAtPoint(analysis, group, point, aliases);
      };
      // WHERE A GROUP'S TOKEN IS ALREADY GONE, one block bitset per group.
      //
      // The loop below asks #markers x #groups times "is this group held at
      // this exit point". On a 400-statement module that is 8,987,220 cells, of
      // which 9,604 can answer Held -- and the exact answer is not cheap: it
      // walks the group's consume sites and asks a reachability question per
      // site, which is 14 of the phase's 16 seconds.
      //
      // Two O(1) facts reject the rest. A point can hold the token only if the
      // definition DOMINATES it, and only if no consume's dead state has
      // reached its block: `groupTokenAtPoint` returns NotHeld or Unknown --
      // never Held -- for a point a consume reaches. The dead sets are the
      // reachability sets that test already builds and memoises, so this costs
      // one OR per consume site and no new traversal.
      //
      // Why NOT bound the candidates by reachability FROM the definition, which
      // reads like the natural companion to the dead set: dominance already
      // implies it, so it rejects nothing extra -- and asking for it built one
      // more reach set per group, 2,805 traversals and 10.1 M extra block
      // visits, which cost more than the cells it saved (17.7 s -> 13.1 s
      // instead of 17.7 s -> 2 s).
      //
      // A SUPERSET, deliberately. This decides nothing: every cell it keeps
      // still gets the exact test. Any set it cannot compute exactly -- a walk
      // that left the block numbering, a terminator consume this analysis does
      // not model -- switches that group back to visiting every marker.
      //
      // ⛔ WHAT IS LEFT IS NOT A CACHING PROBLEM. Building these sets is 9,999
      // traversals and 10,104,416 block visits on a 400-statement module, one
      // per (group, consume site) because the memo key carries the group's own
      // defining block as the avoided node. Two regroupings were implemented
      // and measured, and BOTH visited the same number of blocks:
      //
      //   per consume site  9,999 walks x 1,011 = 10,104,416   (shipped)
      //   union per group   2,805 walks x 3,602 = 10,104,416   (reverted)
      //   shared seed set   5,606 walks x 1,802 = 10,100,023   (reverted)
      //
      // Both are sound -- reachability is monotone in its seeds, and avoiding a
      // block you never reach is not avoiding anything (9,999 distinct
      // (seed, avoided) keys sit over only 5,606 distinct seeds). Both were
      // slower anyway, because a walk that starts from more seeds, or does not
      // stop at the avoided block, reaches proportionally further: fewer walks,
      // bigger walks, same work, plus the bookkeeping. 7.5 s -> 11.7 s and
      // 7.5 s -> 10.0 s.
      //
      // The block-visit count is invariant under every regrouping tried, so
      // going below it means changing WHAT is asked, not how it is cached or
      // shared. The question is per-group because the avoided node is the
      // group's own definition; a formulation that does not need it -- or an
      // under-approximation of "the consume reaches this point" that the exact
      // test can also use -- is the next thing to try, and neither is a memo.
      //
      // The dominance half is asked 4.5 M times, so it is asked of the
      // dominator tree's DFS numbering rather than through `DominanceInfo`: a
      // block dominates another exactly when its interval contains the other's
      // entry number, which is two integer comparisons instead of a ~1.5 us
      // query (6.7 s of the 8.4 s left after the dead set).
      //
      // A single-block region has no dominator tree to ask (`getDomTree`
      // requires more than one block) and nothing to prune: one block means one
      // marker block, so the prune is simply switched off there.
      llvm::DominatorTreeBase<mlir::Block, false> *domTree = nullptr;
      if (!region->hasOneBlock()) {
        domTree = &analysis.dominance.getDomTree(region);
        domTree->updateDFSNumbers();
      }
      auto dfsIntervalOf = [&](mlir::Block *block)
          -> std::optional<std::pair<unsigned, unsigned>> {
        if (!domTree)
          return std::nullopt;
        if (auto *node = domTree->getNode(block))
          return std::make_pair(node->getDFSNumIn(), node->getDFSNumOut());
        return std::nullopt;
      };
      struct HeldBlocks {
        llvm::BitVector dead;
        unsigned defIn = 0;
        unsigned defOut = 0;
        bool usable = false;
      };
      llvm::SmallVector<HeldBlocks, 8> heldBlocks(groups.size());
      for (auto [heldIndex, heldGroup] : llvm::enumerate(groups)) {
        if (heldGroup.skip || !heldGroup.deallocator || heldGroup.values.empty())
          continue;
        mlir::Value root = heldGroup.values.front();
        mlir::Operation *producer = root.getDefiningOp();
        mlir::Block *defBlock =
            producer ? producer->getBlock()
                     : mlir::cast<mlir::BlockArgument>(root).getOwner();
        llvm::BitVector dead(analysis.blocks.size(), false);
        auto mark = [&](mlir::Block *block, bool value) {
          if (std::optional<unsigned> at = analysis.indexOf(block))
            dead[*at] = value;
        };
        auto absorb = [&](const UnwindCleanupAnalysis::ReachSet &set) {
          llvm::BitVector members = set.members;
          members.resize(analysis.blocks.size());
          dead |= members;
        };
        bool exact = true;
        for (mlir::Operation *consume : heldGroup.consumeSites) {
          if (consume->hasTrait<mlir::OpTrait::IsTerminator>() &&
              consume->getNumSuccessors() != 0) {
            if (!mlir::isa<mlir::BranchOpInterface>(consume)) {
              exact = false; // the exact test answers Unknown for every point
              break;
            }
            for (unsigned edge = 0, end = consume->getNumSuccessors();
                 edge < end && exact; ++edge) {
              if (!terminatorForwardsGroupToSuccessor(consume, edge,
                                                      heldGroup.values, aliases))
                continue;
              mlir::Block *successor = consume->getSuccessor(edge);
              if (successor == defBlock)
                continue; // the entry re-arms the token
              mark(successor, true); // the point's own block: Unknown, not Held
              const auto &beyond = analysis.reachSetAvoiding(
                  successor, defBlock, /*fromAfter=*/nullptr);
              if (!beyond.exact) {
                exact = false;
                break;
              }
              absorb(beyond);
            }
            continue;
          }
          const auto &beyond =
              analysis.reachSetAvoiding(consume->getBlock(), defBlock, consume);
          if (!beyond.exact) {
            exact = false;
            break;
          }
          absorb(beyond);
        }
        if (!exact)
          continue;
        // Alive regardless: a marker in the defining block, or one BEFORE a
        // consume in the consume's own block, holds the token even when a loop
        // puts that block in the dead set.
        mark(defBlock, false);
        for (mlir::Operation *consume : heldGroup.consumeSites)
          mark(consume->getBlock(), false);
        std::optional<std::pair<unsigned, unsigned>> interval =
            dfsIntervalOf(defBlock);
        if (!interval)
          continue; // unreachable definition: leave the group unpruned
        heldBlocks[heldIndex].dead = std::move(dead);
        heldBlocks[heldIndex].defIn = interval->first;
        heldBlocks[heldIndex].defOut = interval->second;
        heldBlocks[heldIndex].usable = true;
      }

      // INVERTED, so a marker touches only its own candidates. Testing the two
      // facts per cell still walks #markers x #groups of them, and at that size
      // the walk itself is the cost -- streaming the group vector and a bitset
      // per group is gigabytes of traffic for an answer that is No 99.9% of the
      // time. Inverting costs one pass over blocks x groups of integer
      // comparisons and leaves ~8 candidate blocks per group.
      //
      // Group order within a marker is preserved: the lists are filled in
      // increasing group index, and the unprunable groups (which must be
      // visited at every marker) are merged back in by index below.
      llvm::SmallVector<llvm::SmallVector<unsigned, 4>, 8> candidatesByBlock(
          analysis.blocks.size());
      llvm::SmallVector<unsigned, 8> alwaysGroups;
      {
        llvm::SmallVector<unsigned, 32> dfsIn(analysis.blocks.size(), 0);
        llvm::BitVector numbered(analysis.blocks.size(), false);
        for (auto [at, block] : llvm::enumerate(analysis.blocks))
          if (std::optional<std::pair<unsigned, unsigned>> interval =
                  dfsIntervalOf(block)) {
            dfsIn[at] = interval->first;
            numbered.set(at);
          }
        for (unsigned index = 0, end = groups.size(); index < end; ++index) {
          const UnwindTrackedGroup &group = groups[index];
          if (group.skip || !group.deallocator)
            continue;
          const HeldBlocks &held = heldBlocks[index];
          if (!held.usable) {
            alwaysGroups.push_back(index);
            continue;
          }
          for (unsigned at = 0, blockEnd = analysis.blocks.size();
               at < blockEnd; ++at) {
            if (held.dead.test(at) || !numbered.test(at))
              continue;
            if (dfsIn[at] < held.defIn || dfsIn[at] > held.defOut)
              continue;
            candidatesByBlock[at].push_back(index);
          }
        }
      }

      for (auto &[marker, handler] : markers) {
        mlir::func::CallOp guarded =
            own::guardedCallAfterMarker(marker.getOperation());
        UnwindCleanup cleanup{marker, handler, marker.getOperation(), {}};
        std::optional<unsigned> markerBlock =
            analysis.indexOf(marker->getBlock());
        llvm::ArrayRef<unsigned> blockGroups =
            markerBlock ? llvm::ArrayRef<unsigned>(candidatesByBlock[*markerBlock])
                        : llvm::ArrayRef<unsigned>();
        std::size_t alwaysAt = 0, blockAt = 0;
        while (alwaysAt < alwaysGroups.size() || blockAt < blockGroups.size()) {
          unsigned groupIndex;
          if (blockAt == blockGroups.size() ||
              (alwaysAt < alwaysGroups.size() &&
               alwaysGroups[alwaysAt] < blockGroups[blockAt]))
            groupIndex = alwaysGroups[alwaysAt++];
          else
            groupIndex = blockGroups[blockAt++];
          const UnwindTrackedGroup &group = groups[groupIndex];
          if (group.skip || !group.deallocator)
            continue;
          if (guarded &&
              callConsumesGroup(contracts, guarded, group.values, aliases))
            continue; // ownership already moved into the unwinding callee
          // A guarded RAISE judges the token at the raise call, not at the
          // marker: the raise statement's dying locals release between the
          // two (normal-path releases parked before a never-returning call),
          // so at the moment the unwind edge materializes their tokens are
          // already gone — releasing them again in the cleanup double-frees.
          // Non-raise guarded calls keep the marker point: nothing releases
          // between their marker and the call.
          mlir::Operation *unwindPoint = marker.getOperation();
          if (guarded) {
            auto guardedContract = contracts.lookup(guarded.getCallee());
            if (mlir::succeeded(guardedContract) && *guardedContract &&
                own::isRaiseLikeFunction((*guardedContract)->function))
              unwindPoint = guarded.getOperation();
          }
          if (groupTokenAtPoint(analysis, group, unwindPoint, aliases) !=
              TokenAtPoint::Held)
            continue;
          bool handlerOwns = groupUsedOnHandlerPath(analysis, group, handler);
          unwindTrace().marker(analysis, group, handler, handlerOwns);
          if (handlerOwns)
            continue; // the handler-side releases own this token
          cleanup.groups.push_back(&group);
        }
        if (!cleanup.groups.empty())
          cleanups.push_back(std::move(cleanup));
      }

      steps.mark(UnwindStepTimer::Markers);

      for (mlir::func::CallOp raiseCall : unguardedRaises) {
        InlineReleaseBeforeRaise cleanup{raiseCall, {}};
        for (const UnwindTrackedGroup &group : groups) {
          if (group.skip || !group.deallocator)
            continue;
          if (callConsumesGroup(contracts, raiseCall, group.values, aliases))
            continue;
          if (tokenAtExitPoint(group, raiseCall.getOperation()) !=
              TokenAtPoint::Held)
            continue;
          // No live-after check: a raise primitive never returns, so every
          // use syntactically after it is dead code and the releases here are
          // the path's last live operations.
          cleanup.groups.push_back(&group);
        }
        if (!cleanup.groups.empty())
          inlineReleases.push_back(std::move(cleanup));
      }

      steps.mark(UnwindStepTimer::Raises);

      // Unguarded may-raise calls in a frame without a local handler: the
      // unwind edge exits the function, so every token held ACROSS the call
      // gets a cleanup handler that releases and rethrows. The releases cannot
      // go before the call like the raise-primitive ones -- the call usually
      // returns and the normal path still uses the values.
      for (mlir::func::CallOp call : unguardedMayRaiseCalls) {
        UnwindCleanup cleanup{call, /*handler=*/nullptr, call.getOperation(),
                              {}};
        for (const UnwindTrackedGroup &group : groups) {
          if (group.skip || !group.deallocator)
            continue;
          if (callConsumesGroup(contracts, call, group.values, aliases))
            continue; // ownership already moved into the unwinding callee
          if (groupTokenAtPoint(analysis, group, call.getOperation(),
                                aliases) != TokenAtPoint::Held)
            continue;
          cleanup.groups.push_back(&group);
        }
        if (!cleanup.groups.empty())
          cleanups.push_back(std::move(cleanup));
      }

      steps.mark(UnwindStepTimer::Calls);

      for (auto &[marker, handler] : nestedMarkers) {
        mlir::Operation *ancestor =
            ancestorInRegion(marker.getOperation(), region);
        if (!ancestor)
          continue;
        mlir::func::CallOp guarded =
            own::guardedCallAfterMarker(marker.getOperation());
        UnwindCleanup cleanup{marker, handler, ancestor, {}};
        for (const UnwindTrackedGroup &group : groups) {
          if (group.skip || !group.deallocator)
            continue;
          if (guarded &&
              callConsumesGroup(contracts, guarded, group.values, aliases))
            continue; // ownership already moved into the unwinding callee
          if (tokenAtExitPoint(group, marker.getOperation()) !=
              TokenAtPoint::Held)
            continue;
          if (groupUsedOnHandlerPath(analysis, group, handler))
            continue; // the handler-side releases own this token
          cleanup.groups.push_back(&group);
        }
        if (!cleanup.groups.empty())
          cleanups.push_back(std::move(cleanup));
      }

      for (mlir::func::CallOp call : nestedUnguardedMayRaiseCalls) {
        mlir::Operation *ancestor =
            ancestorInRegion(call.getOperation(), region);
        if (!ancestor)
          continue;
        UnwindCleanup cleanup{call, /*handler=*/nullptr, ancestor, {}};
        for (const UnwindTrackedGroup &group : groups) {
          if (group.skip || !group.deallocator)
            continue;
          if (callConsumesGroup(contracts, call, group.values, aliases))
            continue;
          if (tokenAtExitPoint(group, call.getOperation()) !=
              TokenAtPoint::Held)
            continue;
          cleanup.groups.push_back(&group);
        }
        if (!cleanup.groups.empty())
          cleanups.push_back(std::move(cleanup));
      }

      steps.mark(UnwindStepTimer::Nested);
      // Every CFG query is behind us: the mutation below rewires blocks, and a
      // walk counted after that would be counting a different CFG.
      steps.walks = analysis.walks;
      steps.walkNodes = analysis.walkNodes;
    } // ---- the analysis, and every memoised answer, dies here ----

    for (InlineReleaseBeforeRaise &release : inlineReleases) {
      mlir::OpBuilder builder(release.raiseCall);
      for (const UnwindTrackedGroup *group : llvm::reverse(release.groups))
        mlir::func::CallOp::create(builder, release.raiseCall.getLoc(),
                                   group->deallocator->function,
                                   group->values);
    }

    // One cleanup handler per distinct requirement (handler, group set):
    // markers sharing a requirement share the id, block, and releaser. A
    // null handler means the unwind exits the function: the cleanup block
    // releases and rethrows instead of branching to a local handler.
    struct CleanupHandler {
      mlir::Block *handler = nullptr;
      llvm::SmallVector<const UnwindTrackedGroup *, 4> groups;
      std::int64_t id = 0;
      mlir::Block *block = nullptr;
    };
    llvm::SmallVector<CleanupHandler, 8> cleanupHandlers;
    auto getOrCreateCleanupHandler =
        [&](mlir::Block *handler,
            llvm::ArrayRef<const UnwindTrackedGroup *> cleanupGroups,
            mlir::Location loc) -> CleanupHandler * {
      for (CleanupHandler &candidate : cleanupHandlers)
        if (candidate.handler == handler &&
            llvm::ArrayRef<const UnwindTrackedGroup *>(candidate.groups) ==
                cleanupGroups) {
          return &candidate;
        }
      CleanupHandler created;
      created.handler = handler;
      created.groups.assign(cleanupGroups.begin(), cleanupGroups.end());
      created.id = nextHandlerId++;

      mlir::func::FuncOp releaser = createOutlinedUnwindReleaser(
          module, loc, created.groups, nextReleaserIndex++);
      llvm::SmallVector<mlir::Value, 8> operands;
      for (const UnwindTrackedGroup *group : created.groups)
        operands.append(group->values.begin(), group->values.end());

      auto *cleanupBlock = new mlir::Block;
      region->getBlocks().insert(region->end(), cleanupBlock);
      mlir::OpBuilder builder(module.getContext());
      builder.setInsertionPointToStart(cleanupBlock);
      mlir::Value cleanupId =
          mlir::arith::ConstantIntOp::create(builder, loc, created.id, 64)
              .getResult();
      mlir::func::CallOp::create(builder, loc, catchMarkerFn,
                                 mlir::ValueRange{cleanupId});
      mlir::func::CallOp::create(builder, loc, releaser, operands);
      if (created.handler) {
        mlir::cf::BranchOp::create(builder, loc, created.handler);
      } else {
        // Continue the unwind out of the function (mirrors the try
        // lowering's finally-rethrow block: the rethrow never returns, the
        // self-loop only satisfies the terminator requirement).
        mlir::func::FuncOp rethrowFn =
            module.lookupSymbol<mlir::func::FuncOp>("LyEH_RethrowCurrent");
        if (!rethrowFn) {
          mlir::OpBuilder::InsertionGuard guard(builder);
          builder.setInsertionPointToEnd(module.getBody());
          rethrowFn = mlir::func::FuncOp::create(
              builder, module.getLoc(), "LyEH_RethrowCurrent",
              builder.getFunctionType({}, {}));
          rethrowFn.setPrivate();
        }
        mlir::func::CallOp::create(builder, loc, rethrowFn,
                                   mlir::ValueRange{});
        mlir::cf::BranchOp::create(builder, loc, cleanupBlock);
      }
      created.block = cleanupBlock;

      cleanupHandlers.push_back(created);
      return &cleanupHandlers.back();
    };
    // The anchor keeps the cleanup block a reachable cond_br successor rather
    // than a floating block later phases would drop or fail to verify. It is
    // wired at the END of `head`, so `tail` must lead with the op the unwind
    // edge belongs to -- the call-site marker, or the region op containing it.
    auto wireAnchorBeforeTail = [&](mlir::Block *head, mlir::Block *tail,
                                    mlir::Block *cleanupBlock,
                                    std::int64_t cleanupId,
                                    mlir::Location loc) {
      mlir::OpBuilder builder(module.getContext());
      builder.setInsertionPointToEnd(head);
      mlir::Value headId =
          mlir::arith::ConstantIntOp::create(builder, loc, cleanupId, 64)
              .getResult();
      auto anchor = mlir::func::CallOp::create(builder, loc, anchorFn,
                                               mlir::ValueRange{headId});
      mlir::cf::CondBranchOp::create(builder, loc, anchor.getResult(0),
                                     cleanupBlock, mlir::ValueRange{}, tail,
                                     mlir::ValueRange{});
    };
    for (UnwindCleanup &cleanup : cleanups) {
      if (deferMarkerWiring)
        break; // leaks stay visible: the affine verifier walks the handler
               // path and rejects them, instead of wiring markers the final
               // EH phase will never see. Breaking on the whole list is what
               // the old per-shape guard amounted to: the shapes that MINT a
               // marker are collected behind `ehPhaseProcessesFunction`, which
               // this same attribute already clears, so only re-pointed
               // existing markers can be here to skip.
      ensureEHMarkerFunctions();
      if (cleanup.handler && cleanup.handler->getNumArguments() != 0) {
        result = cleanup.site.emitError()
                 << "unwind cleanup cannot target a handler entry with block "
                    "arguments";
        return;
      }
      mlir::Location loc = cleanup.site.getLoc();
      CleanupHandler *shared =
          getOrCreateCleanupHandler(cleanup.handler, cleanup.groups, loc);

      mlir::Block *head = cleanup.anchorBefore->getBlock();
      mlir::Block *tail = head->splitBlock(cleanup.anchorBefore);
      mlir::OpBuilder builder(cleanup.site);
      mlir::Value id =
          mlir::arith::ConstantIntOp::create(builder, loc, shared->id, 64)
              .getResult();
      if (cleanup.site.getCallee() == "LyEH_TryCallSiteMarker")
        cleanup.site->setOperand(0, id); // re-point it at the cleanup handler
      else
        mlir::func::CallOp::create(builder, loc, callSiteMarkerFn,
                                   mlir::ValueRange{id}); // guard it with one
      wireAnchorBeforeTail(head, tail, shared->block, shared->id, loc);
    }
    steps.mark(UnwindStepTimer::Mutate);
  });
  return result;
}

// Post-cleanup unwind re-pass: the canonicalizer/CSE phases after refcount
// insertion fold statically-decided region ops (a constant-condition int
// fast/slow scf.if is the typical shape) and thereby HOIST calls to the
// function's top level that were nested -- and outside the unwind-cleanup
// model -- when the main insertion ran. Re-running only the unwind step
// wires those newly top-level unguarded may-raise calls.
//
// A call site already guarded does NOT simply keep its cleanup, which this
// comment used to claim. The re-run recomputes the held-token set at every
// marker, and a group the first run could not see (an owner group at a loop
// header, before `5595d16` made the destination groups exist) is newly Held
// there -- so the re-run CHAINS a fresh cleanup block in front of the existing
// handler. That is intended. What is not is treating the existing handler's own
// `LyEH_RethrowCurrent` as a new exceptional exit point: see
// `isGeneratedUnwindCleanupBlock`.
class PostCleanupUnwindInsertionPass
    : public mlir::PassWrapper<PostCleanupUnwindInsertionPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PostCleanupUnwindInsertionPass)

  llvm::StringRef getArgument() const final {
    return "lython-post-cleanup-unwind-insertion";
  }
  llvm::StringRef getDescription() const final {
    return "re-insert unwind cleanup for calls hoisted by post-lowering "
           "canonicalization";
  }

  // The four sub-scopes exist because this phase's total is the largest in the
  // pipeline (measured by another track: 451 s of 1376 s across the 270 golden
  // cases, and 92 s on the worst single case) and the total alone does not say
  // which of its two heavy analyses to attack. Both of them ALSO run inside
  // refcount insertion under their own scope names, so the pair of numbers is
  // directly comparable across the two runs -- which is the question anyone
  // optimising this phase asks first.
  //
  // Why NOT split further, per function, inside the two calls: PerfScope prints
  // one line per scope with no aggregation, so a per-function split emits
  // hundreds of lines and the phase total has to be re-summed by the reader.
  // Whoever needs that resolution should add accumulation to PerfScope rather
  // than more scopes here.
  void runOnOperation() final {
    mlir::ModuleOp module = getOperation();
    llvm::SmallVector<own::RuntimeDeallocator, 8> deallocators;
    {
      py::PerfScope perf("post-cleanup-unwind-insertion.collect-deallocators");
      deallocators = own::collectRuntimeDeallocators(module);
    }
    if (deallocators.empty())
      return;
    own::AliasAnalysis aliases;
    {
      py::PerfScope perf("post-cleanup-unwind-insertion.alias-analysis");
      aliases.build(module);
    }
    FuncContractCache contracts(module);
    // One table for the whole pass. Why not per call: the callee lookup is the
    // factor both analyses below pay per call op, and `ModuleOp::lookupSymbol`
    // answers it by scanning the module's symbol list -- which after phase 8
    // includes every imported stdlib symbol.
    std::optional<mlir::SymbolTable> symbols;
    if (!ownershipSymbolTableDisabled())
      symbols.emplace(module);
    mlir::SymbolTable *symbolTable = symbols ? &*symbols : nullptr;
    // Re-derive the owned block-argument merge groups analysis-only (their
    // normal-path releases and borrow-edge retains were placed by the main
    // pass): the held-token analysis needs them to cover calls the cleanup
    // canonicalization hoisted out of folded region ops.
    const own::ReferenceMap references(contracts, aliases);
    llvm::SmallVector<own::ResourceGroup, 8> blockArgGroups;
    {
      py::PerfScope perf(
          "post-cleanup-unwind-insertion.block-argument-groups");
      if (mlir::failed(insertOwnedBlockArgumentReleases(
              module, contracts, deallocators, aliases, references, symbolTable,
              &blockArgGroups, /*insertReleases=*/false))) {
        signalPassFailure();
        return;
      }
    }
    {
      py::PerfScope perf(
          "post-cleanup-unwind-insertion.unwind-cleanup-releases");
      if (mlir::failed(insertUnwindCleanupReleases(
              module, contracts, deallocators, aliases,
              references, symbolTable,
              blockArgGroups)))
        signalPassFailure();
    }
  }
};

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
    {
      // Expand `arith.select` over an object handle back into the branch the
      // canonicalizer folded it from.
      //
      // A select is the one place ownership cannot be spelled. Two owned
      // temporaries flow in, one flows out, and WHICH one is a runtime fact, so
      // there is no program point at which either can be released by name --
      // and the alias analysis, which sees through the select in both
      // directions, fuses all three values into one class. Two real objects
      // then share one release obligation: `min(2.5, 1.5)` leaked the operand
      // that lost, 40 B, and so did every float/str/heap-int min-max whose
      // operands are temporaries (`scalar_ops`,
      // `cross_float_range_contracts_fields`).
      //
      // The branch form has the answer already: each arm forwards one value and
      // the other DIES there, which is exactly what the per-edge machinery
      // below is built to release. So rather than teach three consumers (both
      // release-insertion steps and the affine verifier, which share the alias
      // model on purpose) to tell a may-alias from a must-alias, this puts the
      // IR back into the shape they already handle. Measured on the other road
      // first: making the insertion pass emit the second release turned five
      // goldens red, one of them aborting on a double free.
      //
      // Runs, not single ops: one contract-typed select becomes one select per
      // physical lane during runtime lowering (a str is header + payload, an
      // int is header + digits + primitive pair), and the lanes must land in ONE
      // diamond or the destination group is split across two merges.
      py::PerfScope perf("refcount-insertion.expand-object-selects");
      llvm::SmallVector<llvm::SmallVector<mlir::arith::SelectOp, 4>, 4> runs;
      module.walk([&](mlir::func::FuncOp function) {
        for (mlir::Block &block : function.getBody()) {
          // Branching out of a nested single-block region is not expressible,
          // so only a function's own blocks are candidates.
          if (!mlir::isa<mlir::func::FuncOp>(block.getParentOp()))
            continue;
          for (mlir::Operation &op : block) {
            auto select = mlir::dyn_cast<mlir::arith::SelectOp>(&op);
            if (!select)
              continue;
            if (!runs.empty() && !runs.back().empty() &&
                runs.back().back()->getNextNode() == select.getOperation() &&
                runs.back().back().getCondition() == select.getCondition()) {
              runs.back().push_back(select);
              continue;
            }
            runs.push_back({select});
          }
        }
      });
      // Does the frame own what this value names? Only then can a select
      // strand a reference.
      auto frameProduces = [&](mlir::Value value) {
        mlir::Value root = own::underlyingObjectValue(value);
        mlir::Operation *definition = root.getDefiningOp();
        if (!definition)
          return false;
        if (definition->hasAttr(own::kOwnedLocalObjectAttr) ||
            definition->hasAttr(own::kObjectHeaderAttr))
          return true;
        auto call = mlir::dyn_cast<mlir::func::CallOp>(definition);
        if (!call)
          return false;
        auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
        auto result = mlir::dyn_cast<mlir::OpResult>(root);
        return callee && result &&
               own::callResultGroupIsOwned(callee, result.getResultNumber());
      };
      for (auto &run : llvm::reverse(runs)) {
        bool ownsAnObject = false;
        for (mlir::arith::SelectOp select : run)
          if (own::isObjectHeaderLikeType(select.getType()))
            ownsAnObject = true;
        if (!ownsAnObject)
          continue;
        // ONLY when an operand is frame-owned, and the reason is a golden that
        // went red without this test. `def first(a: str, b: str): return a or b`
        // is a select over two BORROWED entry arguments, and the borrowed-return
        // rule recognises it there (`valueGroupDerivedFromEntryArguments` reads
        // through a select). As a merge it is a destination group no edge
        // transfers into, which the candidate scan drops entirely -- so the
        // retain an owned return needs was never placed and the affine verifier
        // refused the program. Nothing is stranded in that shape anyway: neither
        // arm holds a reference to lose.
        bool strandable = false;
        for (mlir::arith::SelectOp select : run)
          if (frameProduces(select.getTrueValue()) ||
              frameProduces(select.getFalseValue()))
            strandable = true;
        if (!strandable)
          continue;
        // A run that feeds itself cannot be lifted whole: the later select's
        // operand would be defined in a block the arms do not reach.
        llvm::SmallPtrSet<mlir::Operation *, 4> inRun;
        for (mlir::arith::SelectOp select : run)
          inRun.insert(select.getOperation());
        bool selfFeeding = false;
        for (mlir::arith::SelectOp select : run)
          for (mlir::Value operand : select->getOperands())
            if (operand.getDefiningOp() && inRun.contains(operand.getDefiningOp()))
              selfFeeding = true;
        if (selfFeeding)
          continue;

        mlir::arith::SelectOp first = run.front();
        mlir::Location loc = first.getLoc();
        mlir::Value condition = first.getCondition();
        mlir::Block *head = first->getBlock();
        mlir::Region *region = head->getParent();
        mlir::Block *join = head->splitBlock(first.getOperation());
        llvm::SmallVector<mlir::Value, 4> trueValues, falseValues;
        for (mlir::arith::SelectOp select : run) {
          trueValues.push_back(select.getTrueValue());
          falseValues.push_back(select.getFalseValue());
          mlir::Value merged = join->addArgument(select.getType(),
                                                 select.getLoc());
          select.getResult().replaceAllUsesWith(merged);
        }
        for (mlir::arith::SelectOp select : llvm::reverse(run))
          select.erase();
        auto makeArm = [&](mlir::ValueRange values) {
          auto *arm = new mlir::Block;
          region->getBlocks().insert(join->getIterator(), arm);
          mlir::OpBuilder armBuilder(arm, arm->begin());
          mlir::cf::BranchOp::create(armBuilder, loc, join, values);
          return arm;
        };
        mlir::Block *trueArm = makeArm(trueValues);
        mlir::Block *falseArm = makeArm(falseValues);
        mlir::OpBuilder builder(head, head->end());
        mlir::cf::CondBranchOp::create(builder, loc, condition, trueArm,
                                       mlir::ValueRange{}, falseArm,
                                       mlir::ValueRange{});
      }
    }
    own::AliasAnalysis aliases;
    {
      py::PerfScope perf("refcount-insertion.alias-analysis");
      aliases.build(module);
    }
    FuncContractCache contracts(module);
    // Same lifetime and the same staleness contract as `contracts` above, which
    // is why it is built here rather than inside each helper: both are
    // name->FuncOp maps snapshotted before the pass mutates, and the pass only
    // ever resolves callees that already existed (the `__ly_unwind_cleanup_*`
    // functions it creates are called only from the function being processed
    // when they are created, never re-resolved). Making the table shorter-lived
    // would not make that assumption weaker, it would only pay the scan again.
    std::optional<mlir::SymbolTable> symbols;
    if (!ownershipSymbolTableDisabled())
      symbols.emplace(module);
    mlir::SymbolTable *symbolTable = symbols ? &*symbols : nullptr;

    // ONE reference identity for the whole run, built beside the alias analysis
    // it is not: `aliases` answers "same entity", this answers "same reference",
    // and the walks below need both.
    const own::ReferenceMap references(contracts, aliases);

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
        if (mlir::failed(insertOwnedResultReleases(
                module, call, contracts, deallocators, aliases, references,
                symbolTable))) {
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
                module, op, contracts, deallocators, aliases,
                references))) {
          signalPassFailure();
          return;
        }
      }
    }

    llvm::SmallVector<own::ResourceGroup, 8> blockArgGroups;
    {
      py::PerfScope perf("refcount-insertion.block-argument-releases");
      if (mlir::failed(insertOwnedBlockArgumentReleases(
              module, contracts, deallocators, aliases, references,
              symbolTable, &blockArgGroups))) {
        signalPassFailure();
        return;
      }
    }

    {
      // Last: every normal-path release above is a consume site this step's
      // held-token analysis must see.
      py::PerfScope perf("refcount-insertion.unwind-cleanup-releases");
      if (mlir::failed(insertUnwindCleanupReleases(
              module, contracts, deallocators, aliases,
              references, symbolTable,
              blockArgGroups)))
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

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPostCleanupUnwindInsertionPass() {
  return std::make_unique<lowering::PostCleanupUnwindInsertionPass>();
}

} // namespace py
