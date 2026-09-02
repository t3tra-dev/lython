#include "Runtime/Core/Lowerer.h"

#include "Runtime/Core/OwnedLocalMarker.h"
#include "Runtime/Evidence/Callable.h"
#include "Runtime/ABI/BoxLayout.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace py::lowering {

namespace {

using callable_evidence::integerLiteralFromValue;

bool sameValueTypes(llvm::ArrayRef<mlir::Value> lhs,
                    llvm::ArrayRef<mlir::Value> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  for (auto [left, right] : llvm::zip(lhs, rhs))
    if (left.getType() != right.getType())
      return false;
  return true;
}

} // namespace

// ⭐ AN ELEMENT IS IN SLOT FORM, WHICH IS NOT ALWAYS ITS ABI FORM. A slot holds
// a header-fronted group, so `bool` is stored BOXED (its canonical i1 has no
// header) and the contents evidence records the box. The union lane for bool is
// the i1, and the injection that builds the lanes checks only how MANY values a
// member takes -- one either way -- so the header went into the lane and the
// shape check downstream reported it as the union's:
//
//     row = [True, "a"]
//     print(row[0])
//     # runtime bundle value 1 for '!py.union<bool, str>' has type
//     # 'memref<3xi64>', but ABI expects 'i1'
//
// bool is the only contract with a `box` primitive today, which is why this
// went unseen: every other element is stored in the form it is read in.
//
// ⛔ Why NOT check the types inside `appendUnionRuntimeValues` instead: its
// other callers hand it values that are already in ABI form, and a mismatch
// there is a real defect that should stay loud. Only a read out of a slot knows
// its operand came from storage.
mlir::LogicalResult
RuntimeBundleLowerer::canonicalizeSlotElementBundle(mlir::Operation *op,
                                                    RuntimeBundle &bundle) {
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> abiTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, bundle.objectValue.contract,
                                                 "container element union ABI");
  if (mlir::failed(abiTypes))
    return mlir::failure();
  if (bundle.physicalValues().size() != abiTypes->size() ||
      llvm::all_of(llvm::zip_equal(bundle.physicalValues(), *abiTypes),
                   [](const auto &pair) {
                     return std::get<0>(pair).getType() == std::get<1>(pair);
                   }))
    return mlir::success();
  builder.setInsertionPoint(op);
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> unboxed =
      RuntimeBundleLowerer::unboxSlotElementValues(
          op, bundle.objectValue.contract,
          llvm::SmallVector<mlir::Value, 4>(bundle.physicalValues().begin(),
                                            bundle.physicalValues().end()));
  if (mlir::failed(unboxed))
    return mlir::failure();
  RuntimeBundle canonical = RuntimeBundle::objectWithOwnership(
      bundle.objectValue.contract, *unboxed,
      ownership::logicalOwnershipKind(bundle.objectValue.contract,
                                      /*ownsObject=*/false));
  canonical.copyEvidenceFrom(bundle);
  bundle = std::move(canonical);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindEvidenceObjectResult(
    mlir::Operation *op, mlir::Value resultValue, llvm::StringRef label,
    const RuntimeValue &value) {
  // ⭐ EVERY SIGNATURE IS ONE RUNTIME CONTRACT. A stored function reads back
  // as `builtins.function`, which is not assignable to any particular
  // Callable -- so a dict of functions died here for a table the LIST
  // spelling has always compiled:
  //
  //     table = {"x": a, "y": b}
  //     table["x"]()   # dict __getitem__ evidence contract
  //                    # 'builtins.function' is not assignable to result
  //                    # '!py.callable<[], returns = ["builtins.str"]>'
  //
  // ⛔ Why NOT teach isAssignableTo that builtins.function accepts a Callable:
  // that would accept ANY function wherever a signature is declared, which is
  // the one thing the static contract is for. Here the container's element
  // type IS the promise about which signature, and it is what this result
  // already carries; the relabel below then puts it on the bundle.
  auto erasedFunctionEvidence = [&] {
    auto contract = mlir::dyn_cast_if_present<py::ContractType>(value.contract);
    return contract && contract.getArguments().empty() &&
           contract.getContractName() == "builtins.function" &&
           mlir::isa<py::CallableType>(resultValue.getType());
  };
  if (!py::isAssignableTo(value.contract, resultValue.getType(), op) &&
      !erasedFunctionEvidence())
    return op->emitError() << label << " evidence contract " << value.contract
                           << " is not assignable to result "
                           << resultValue.getType();

  mlir::Type bundleContract = value.contract;
  std::string resultContract = runtimeContractName(resultValue.getType());
  if (!resultContract.empty() &&
      objectShapeMatches(resultContract, value.values))
    bundleContract = resultValue.getType();

  RuntimeBundle bound = RuntimeBundle::objectWithOwnership(
      bundleContract, value.values,
      ownership::logicalOwnershipKind(bundleContract,
                                      /*ownsObject=*/false));
  // The union widening `bindSelectedEvidenceObjectResult` explains, on the
  // other binder. A heterogeneous DICT read arrives here rather than there --
  // `{"name": "ann", "age": 30}` is how a record literal is written, and its
  // `__getitem__` result type is `str | int`.
  if (auto resultUnion = mlir::dyn_cast<py::UnionType>(resultValue.getType());
      resultUnion && !mlir::isa<py::UnionType>(bundleContract)) {
    if (mlir::failed(
            RuntimeBundleLowerer::canonicalizeSlotElementBundle(op, bound)))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> widened =
        RuntimeBundleLowerer::materializeObjectBundleForStorage(
            op, bound, resultUnion, "container element union ABI");
    if (mlir::failed(widened))
      return mlir::failure();
    bound = std::move(*widened);
    bound.setObjectLogicalOwnership(/*ownsObject=*/false);
  }
  valueBundles[resultValue] = std::move(bound);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindSelectedEvidenceObjectResult(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle bundle) {
  // ⭐ A UNION-TYPED READ GETS THE UNION'S LANES, not the member's. Evidence
  // selection knows exactly which element it picked -- an int for `xs[0]` of
  // `[1, "a"]` -- and handing that bundle back where the result type is
  // `int | str` gives every consumer of a union a member where it reads the
  // TAG from lane 0. `arith.cmpi` then infers its result shape from the
  // operand it was given:
  //
  //     xs = [1, "a"]
  //     print(xs[0])
  //     # runtime bundle value 0 for 'builtins.bool' has type 'memref<2xi1>',
  //     # but ABI expects 'i1'
  //
  // and the same for a heterogeneous dict, which is how a record literal
  // (`{"name": "ann", "age": 30}`) is written. Same defect as the union FIELD
  // read; that one had the union's lanes already spliced into the instance and
  // only had to stop consulting the cache, and this one has to BUILD them --
  // `materializeObjectBundleForStorage` is the widening that already exists,
  // and it records the active member so the slot's reference stays nameable.
  if (auto resultUnion =
          mlir::dyn_cast<py::UnionType>(resultValue.getType());
      resultUnion && bundle.kind == RuntimeBundle::Kind::Object &&
      !mlir::isa<py::UnionType>(bundle.objectValue.contract)) {
    if (mlir::failed(
            RuntimeBundleLowerer::canonicalizeSlotElementBundle(op, bundle)))
      return mlir::failure();
    mlir::FailureOr<RuntimeBundle> widened =
        RuntimeBundleLowerer::materializeObjectBundleForStorage(
            op, bundle, resultUnion, "container element union ABI");
    if (mlir::failed(widened))
      return mlir::failure();
    bundle = std::move(*widened);
  }
  bundle.setObjectLogicalOwnership(/*ownsObject=*/false);
  valueBundles[resultValue] = std::move(bundle);
  erase.push_back(op);
  return mlir::success();
}

// An evidence-selected container element is a borrow whose provenance (the
// container) may be released before the element's last use — the container's
// liveness ends at its last direct IR use, which evidence selection does not
// create. Retaining the element through its contract's `own` primitive turns
// the borrow into a checked owned token that survives the container and is
// released by the ordinary owned-result machinery. The retain is inserted
// immediately after the element's defining ops, where the element is provably
// alive (its container has not been released yet). Contracts without an `own`
// primitive keep the borrowed binding (their uses stay tied to structurally
// live containers, e.g. instance fields).
namespace {

// The point where an evidence element's frame bookkeeping may be written: just
// after the last of its defining ops, where the element is provably alive (its
// container has not been released yet). Null when there is no single such point.
//
// Pure — it emits nothing — so the retain and the ownership marker can be placed
// at the same anchor without either recomputing it, and so an element that
// arrives already-owned can be marked at exactly the point one would have been
// retained at.
mlir::Operation *evidenceElementAnchor(const RuntimeValue &value) {
  // An inline-constructed local (its entity root is a raw alloc) is not yet
  // initialized at its defining op — the refcount word store lands later in
  // the construction sequence — so bookkeeping placed after the def would read
  // garbage. Such evidence elements need the at-operation form (with a
  // container pin, the caller's responsibility) or stay borrowed.
  // Slot-reconstructed and selection-merged elements (loads/casts/merges at
  // the access site) are long-initialized and are safe after their defs,
  // independent of any container pin.
  if (mlir::isa_and_nonnull<mlir::memref::AllocOp>(
          value.values.front().getDefiningOp()))
    return nullptr;
  mlir::Operation *latest = nullptr;
  for (mlir::Value physical : value.values) {
    mlir::Operation *definition = physical.getDefiningOp();
    if (!definition)
      return nullptr;
    if (!latest) {
      latest = definition;
    } else if (definition->getBlock() != latest->getBlock()) {
      return nullptr;
    } else if (latest->isBeforeInBlock(definition)) {
      latest = definition;
    }
  }
  return latest;
}

// ⭐ An owned-local token this frame ALREADY holds for exactly these values,
// reached from `op`. Null when there is none.
//
// Ownership downstream is tracked PER SSA VALUE: `refcount-insertion` emits one
// release for a value it finds marked, however many times it is marked. So a
// second token on the same values is not a second reference -- it is a retain
// with no release. The invariant "at most one owned token per SSA value" was
// assumed by the consumer and enforced nowhere, and ordinary code reaches the
// gap: reading the same container slot twice reconstructs the SAME handle, so
//
//     n = ((1, 2), (3, 4))
//     a = n[0]
//     b = n[0]
//
// minted two tokens on one value and leaked the whole inner entity -- its
// handle, its items array and every box it owns. Measured: 2 roots / 10368 B for
// a two-element inner tuple, 69 roots / 14656 B for a seventy-element one, and
// SATURATING (a third read cost nothing more), which is what a per-value map
// looks like from the outside.
//
// The frame already holds a reference to the entity, so a re-read is a BORROW,
// and a borrow costs nothing. That is the same answer the model gives:
// `proof/`'s `step-borrow` binds a second name and occupies no owner site, and
// `WFES.backed` says every owned name occupies its OWN site -- one token per
// name, which is exactly the invariant this restores at the producer.
//
// Why NOT teach `refcount-insertion` to count tokens instead: the count is not
// what the semantics needs. Two names for one entity is one reference, so a
// counting consumer would be maintaining a number that must always be one.
// `anchor` is the op the new token would be placed immediately after -- the last
// defining op of the values -- or null for the at-operation form, where `op` is
// the insertion point.
//
// ⛔ The reference point is the ANCHOR, not `op`, and the first version of this
// got it wrong. Tokens are placed after the values' defining op, which is
// frequently in a different block from the py operation being lowered, so
// comparing against `op` rejected the reuse and minted a second token anyway --
// measured: three goldens still carried duplicates
// (`sequence_literal_source_move_frequency`, `cross_exception_field_box_slot`,
// `cross_enum_generic_handler`), two of them leaking. The anchor is sound as the
// reference because it DEFINES the values, so it dominates every use of them
// including `op`; a token placed just after it dominates `op` for the same
// reason, which is what makes its results safe to use in place of a new one.
mlir::UnrealizedConversionCastOp
existingOwnedLocalToken(const RuntimeValue &value, mlir::Operation *op,
                        mlir::Operation *anchor, llvm::StringRef contract) {
  mlir::Operation *reference = anchor ? anchor : op;
  for (mlir::Operation *user : value.values.front().getUsers()) {
    auto cast = mlir::dyn_cast<mlir::UnrealizedConversionCastOp>(user);
    if (!cast || !cast->hasAttr(ownership::kOwnedLocalObjectAttr))
      continue;
    if (cast->getBlock() != reference->getBlock())
      continue;
    // Anchored: the token to reuse is the one already sitting after the values'
    // definition. At-operation: it has to precede this operation.
    if (anchor ? !anchor->isBeforeInBlock(cast) : !cast->isBeforeInBlock(op))
      continue;
    auto marked = cast->getAttrOfType<mlir::StringAttr>(
        ownership::kOwnedLocalObjectContractAttr);
    if (!marked || marked.getValue() != contract)
      continue;
    if (cast.getInputs().size() != value.values.size() ||
        cast.getOutputs().size() != value.values.size())
      continue;
    if (!llvm::all_of(llvm::zip_equal(cast.getInputs(), value.values),
                      [](auto pair) {
                        return std::get<0>(pair) == std::get<1>(pair);
                      }))
      continue;
    return cast;
  }
  return nullptr;
}

// IS THIS VALUE ITSELF A TOKEN THE FRAME ALREADY HOLDS?
//
// `existingOwnedLocalToken` above looks for a SIBLING marker over the same
// inputs. It cannot see the case where the value handed in is a marker's own
// RESULTS: those are different SSA values, so the input comparison misses and a
// second retain is minted on a name that already denotes the frame's reference.
//
// A module global read is exactly that shape. `lowerObjectGlobalGet` roots the
// reference it takes, so the value reaching a consumer is already a token, and
// the consumer asking for one again got `Ly_IncRef` + marker stacked on marker:
// two increments and two releases for one read of `Color.RED`. Correct, and one
// retain/release pair per read wasted.
//
// The whole result list has to match, not a prefix: a value that is only PART of
// a marker's results is a lane of that entity, not the reference itself, and
// answering yes for one would hand back a token for something the caller did not
// ask about.
bool valueIsOwnedLocalToken(const RuntimeValue &value,
                            llvm::StringRef contract) {
  if (value.values.empty())
    return false;
  auto marker = value.values.front().getDefiningOp<
      mlir::UnrealizedConversionCastOp>();
  if (!marker || !marker->hasAttr(ownership::kOwnedLocalObjectAttr))
    return false;
  auto marked = marker->getAttrOfType<mlir::StringAttr>(
      ownership::kOwnedLocalObjectContractAttr);
  if (!marked || marked.getValue() != contract)
    return false;
  return marker.getResults().size() == value.values.size() &&
         llvm::all_of(llvm::zip_equal(marker.getResults(), value.values),
                      [](auto pair) {
                        return std::get<0>(pair) == std::get<1>(pair);
                      });
}

// Mark an element as a frame-owned local, so the ordinary owned-result
// machinery releases it. Assumes the insertion point is already set.
//
// The point of splitting it out is that "take a reference" and "record that
// the frame holds one" are two operations, and a caller needs to be able to
// ask for the second alone. It used to claim to be the only place the marker
// is written; there were three. `Core/OwnedLocalMarker.h` is now that place,
// and this is one of its callers.
RuntimeValue rootAsOwnedLocal(mlir::OpBuilder &builder, mlir::Location loc,
                              const RuntimeValue &value,
                              llvm::StringRef contract) {
  mlir::UnrealizedConversionCastOp rooted =
      mintOwnedLocalMarker(builder, loc, value.values, contract);
  RuntimeValue out = value;
  out.values.assign(rooted.getResults().begin(), rooted.getResults().end());
  return out;
}

// Was this value PRODUCED owned by this frame? A call that declares the result
// owned, or a speculation `scf.if` whose every arm yields such a call at the
// same position -- the shape `j = pick(2)` actually has here, because the
// primitive-i64 speculation wraps the call in an if whose arms box the
// speculated word or make the real call.
bool valueIsFrameOwnedProduct(mlir::ModuleOp module, mlir::Value value,
                              unsigned depth) {
  if (depth > 4)
    return false;
  auto result = mlir::dyn_cast_or_null<mlir::OpResult>(
      ownership::underlyingObjectValue(value));
  if (!result)
    return false;
  if (auto call = mlir::dyn_cast<mlir::func::CallOp>(result.getOwner())) {
    auto callee = module.lookupSymbol<mlir::func::FuncOp>(call.getCallee());
    return callee &&
           ownership::callResultGroupIsOwned(callee, result.getResultNumber());
  }
  auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(result.getOwner());
  if (!ifOp)
    return false;
  unsigned index = result.getResultNumber();
  for (mlir::Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    if (region->empty())
      return false;
    mlir::Operation *terminator = region->front().getTerminator();
    if (!terminator || index >= terminator->getNumOperands())
      return false;
    if (!valueIsFrameOwnedProduct(module, terminator->getOperand(index),
                                  depth + 1))
      return false;
  }
  return true;
}

// ⭐ The frame already owns this entity under a NAME OF ITS OWN -- an owned call
// result it has not handed to any container -- so reading it back out is a
// borrow of that name and costs nothing.
//
// `existingOwnedLocalToken` above answers the same question for a value the
// frame owns through a TOKEN, and the reason it is not enough is that most
// bindings have no token: `j = pick()` owns its entity as a plain owned call
// result, released by liveness. Reading the same entity back out of a container
// therefore looked unowned and minted a second reference:
//
//     j = pick(2); t = (1, j); print(t[1], j)
//
// Downstream that is one entity with two owners and ONE release. The alias
// analysis sees through the token's identity cast, so `insertOwnedResultReleases`
// releasing the call result IS read as the token's death by
// `insertOwnedLocalObjectReleases`, and the token's reference is never given
// back. Measured 52 B per read -- `tuple_duplicate_element`, `class_protocol`,
// `generator_local_list`.
//
// Why NOT fix it downstream by teaching the two steps apart: they share the
// alias model with the affine-ownership verifier, which refuses two releases of
// one aliased resource on a path. Making the insertion pass emit the second
// release turned five goldens red -- `dict_methods_complete` aborting with
// `Ly_DecRef observed non-positive refcount`, four more refused as "released or
// transferred more than once". One entity with one owner is what the whole
// model is built on; the repair belongs where the second owner was invented.
//
// ⛔ "HAS NOT HANDED IT OVER" IS THE LOAD-BEARING HALF. A container literal whose
// element source is dead after the store MOVES the reference -- spelled as the
// slot retain plus an `aggregate_release`-marked release of the source -- and
// after that the frame owns nothing, so a borrow would be freed under the reader
// when the container dies. The literal lowers before the read, so that release
// is already in the IR: the question is answered by LOOKING, not by predicting.
// Any such release anywhere in the function declines the borrow, which is the
// conservative direction (a token is minted, as before).
// CPython names the type in an index error: "list index out of range",
// "tuple index out of range", "string index out of range". `bytes` is the odd
// one out -- bytearray_getitem and bytes_item say plain "index out of range" --
// so this follows the interpreter rather than regularising it.
llvm::StringRef indexOutOfRangeMessage(llvm::StringRef contractName) {
  if (contractName == "builtins.list")
    return "list index out of range";
  if (contractName == "builtins.tuple")
    return "tuple index out of range";
  if (contractName == "builtins.str")
    return "string index out of range";
  if (contractName == "builtins.bytes" || contractName == "builtins.bytearray")
    return "index out of range";
  return "sequence index out of range";
}

bool frameKeepsOwnedSourceOf(mlir::ModuleOp module, mlir::Operation *op,
                             const RuntimeValue &value) {
  if (value.values.empty())
    return false;
  mlir::Value head = ownership::underlyingObjectValue(value.values.front());
  mlir::Operation *producer = head.getDefiningOp();
  if (!producer || !valueIsFrameOwnedProduct(module, head, /*depth=*/0))
    return false;

  mlir::func::FuncOp function = op->getParentOfType<mlir::func::FuncOp>();
  if (!function || producer->getParentOfType<mlir::func::FuncOp>() != function)
    return false;
  mlir::DominanceInfo dominance(function);
  if (!dominance.properlyDominates(producer, op))
    return false;

  for (mlir::Operation *user : head.getUsers()) {
    auto userCall = mlir::dyn_cast<mlir::func::CallOp>(user);
    if (!userCall)
      continue;
    // Handed to a container, or consumed by a callee that takes ownership:
    // either way the frame's reference is no longer the frame's.
    if (userCall->hasAttr(ownership::kAggregateReleaseAttr))
      return false;
    for (unsigned index = 0, end = userCall.getNumOperands(); index < end;
         ++index) {
      if (ownership::underlyingObjectValue(userCall.getOperand(index)) != head)
        continue;
      if (ownership::functionConsumesOperandAt(
              module.lookupSymbol<mlir::func::FuncOp>(userCall.getCallee()),
              index))
        return false;
    }
  }
  return true;
}

} // namespace

std::optional<RuntimeValue>
RuntimeBundleLowerer::retainEvidenceElement(mlir::Operation *op,
                                            const RuntimeValue &value,
                                            bool atOperation) {
  std::string contract = runtimeContractName(value.contract);
  if (contract.empty() || value.values.empty())
    return std::nullopt;
  if (!ownership::isObjectHeaderLikeType(value.values.front().getType()))
    return std::nullopt;
  mlir::func::FuncOp retain = RuntimeBundleLowerer::findRetainFunction();
  if (!retain || retain.getFunctionType().getNumInputs() != 1)
    return std::nullopt;
  mlir::Operation *latest = nullptr;
  if (!atOperation) {
    latest = evidenceElementAnchor(value);
    if (!latest)
      return std::nullopt;
  }
  // ⭐ Already owned by this frame: borrow the existing token, take no second
  // reference. See `existingOwnedLocalToken` for why a second one leaks.
  //
  // ⛔ KNOWN DEFECT, and this is the predicate that decides it. The marker
  // says the frame owns the element; it does NOT say the frame still owns it.
  // A literal that MOVED its token into a container leaves the marker behind,
  // so a later read borrows a token the container now owns:
  //
  //     def f() -> list[int]:
  //         xs: list[list[int]] = [[1, 2]]
  //         return xs[0]      # "released owned resource ... is used by
  //                           #  function return"
  //
  // Returning an element of a container built in the SAME frame is the whole
  // shape; a parameter (`def f(xs): return xs[0]`) is fine, and so is a str
  // element, which has no token to move.
  //
  // ⛔ AN INT ELEMENT IS NO LONGER EXEMPT (measured 2026-09-02): `xs = [1, 2]`
  // then `return xs[0]` fails the same way, and so do the tuple and dict
  // spellings of it. An int element is box-fronted now and carries a token
  // like any other, so the sentence above -- written when it did not -- names
  // one type too many. The refusal reaches an ordinary two-line function. Returning the ITERATION
  // element lands here too, with the other half of the message --
  //
  //     def f(xs: list[int]) -> int:
  //         for x in xs:
  //             return x
  //         return -1
  //     # "owned resource from builtin.unrealized_conversion_cast reaches
  //     #  function exit without release, transfer, or owned return"
  //
  // -- and `break` in the same position is fine, so it is the return that the
  // marker's group cannot be matched against, not the loop. It is the read side of the
  // same accounting the `initializeSequencePayload` note describes from the
  // write side ("three releases for two references") -- the same *args and
  // **kwargs shapes (`return args[0]`, `return kwargs["a"]`) land here too.
  //
  // Five repairs measured, none right. The three in that note, plus: clearing
  // `kOwnedLocalObjectAttr` from the marker at the move, so this predicate
  // stops seeing a token the frame gave away. Measured -- the refusal did not
  // move AND two goldens broke (enum_desugar, cross_enum_generic_handler), so
  // the marker is load-bearing for something past the move as well.
  //
  // And the fifth (2026-09-02): asking at the READ whether the token was
  // already moved -- the move is visible as a release carrying
  // `ly.ownership.aggregate_release` ending in `.source` -- and declining the
  // borrow when it was. Measured by forcing `valueIsOwnedLocalToken` to answer
  // NO unconditionally: the refusal does not move at all, so the token this
  // return carries is not the one this predicate hands out. Whatever binds it
  // is upstream of here, which is where the sixth attempt has to look.
  //
  // Two more measured 2026-09-02, and together they say where the repair is
  // NOT. (6) Declining the borrow when the source was moved -- the move is
  // visible as an `aggregate_release`-marked release, so the question is
  // answerable by looking -- in BOTH borrow paths, this one and
  // `existingOwnedLocalToken`. The refusal did not move: forcing
  // `valueIsOwnedLocalToken` to answer NO unconditionally changes nothing
  // either, so the token the return carries is not one of these. (7) Minting
  // the read's token AT THE READ instead of at the value's defining op, so the
  // release stands before the mint rather than between the mint and the use.
  // The refusal moved from the marker to the retained CALL RESULT
  // ("released owned resource from @LyLong_FromI64 is used by function
  // return") -- the alias analysis identifies the two references to one entity
  // whichever order they are in.
  //
  // So the counts are right and the MODEL is what cannot follow them: one
  // entity has one owner in it, and this shape has two (the frame's read and
  // the container's slot). The next attempt has to be there, not here.
  //
  // The value may already BE the token rather than have one beside it, which is
  // the cheaper question and so the one asked first (`valueIsOwnedLocalToken`).
  if (valueIsOwnedLocalToken(value, contract))
    return value;
  if (mlir::UnrealizedConversionCastOp held =
          existingOwnedLocalToken(value, op, latest, contract)) {
    RuntimeValue borrowed = value;
    borrowed.values.assign(held.getResults().begin(), held.getResults().end());
    return borrowed;
  }
  // Same rule for the other way a frame owns an entity: under the name of an
  // owned call result, with no token at all. See `frameKeepsOwnedSourceOf`.
  if (frameKeepsOwnedSourceOf(module, op, value))
    return value;
  // Borrow → own: one retain on the entity root, then the owned-local marker.
  // The contract → retain relation is static; no per-contract runtime wrapper is
  // involved.
  mlir::OpBuilder::InsertionGuard guard(builder);
  if (atOperation)
    builder.setInsertionPoint(op);
  else
    builder.setInsertionPointAfter(latest);
  mlir::Location loc = atOperation ? op->getLoc() : latest->getLoc();
  mlir::Type retainInput = retain.getFunctionType().getInput(0);
  mlir::Value header = ownership::spellHeaderPrefix(
      builder, loc, value.values.front(), retainInput);
  if (!header)
    return std::nullopt;
  mlir::func::CallOp::create(builder, loc, retain, header);
  return rootAsOwnedLocal(builder, loc, value, contract);
}

std::optional<RuntimeValue>
RuntimeBundleLowerer::rootOwnedEvidenceElement(mlir::Operation *op,
                                               const RuntimeValue &value,
                                               bool atOperation) {
  std::string contract = runtimeContractName(value.contract);
  if (contract.empty() || value.values.empty())
    return std::nullopt;
  if (!ownership::isObjectHeaderLikeType(value.values.front().getType()))
    return std::nullopt;
  mlir::Operation *latest = nullptr;
  if (!atOperation) {
    latest = evidenceElementAnchor(value);
    if (!latest)
      return std::nullopt;
  }
  // Same one-token-per-value rule as the retain path. An already-owned element
  // normally comes from a fresh runtime call and so has no prior token, but the
  // rule is about the DOWNSTREAM map and does not care where the value came
  // from, so it is applied here too rather than assumed not to matter.
  if (valueIsOwnedLocalToken(value, contract))
    return value;
  if (mlir::UnrealizedConversionCastOp held =
          existingOwnedLocalToken(value, op, latest, contract)) {
    RuntimeValue borrowed = value;
    borrowed.values.assign(held.getResults().begin(), held.getResults().end());
    return borrowed;
  }
  mlir::OpBuilder::InsertionGuard guard(builder);
  if (atOperation)
    builder.setInsertionPoint(op);
  else
    builder.setInsertionPointAfter(latest);
  mlir::Location loc = atOperation ? op->getLoc() : latest->getLoc();
  return rootAsOwnedLocal(builder, loc, value, contract);
}

mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>>
RuntimeBundleLowerer::slotStorageShapesFor(mlir::Operation *op,
                                           mlir::Type contract,
                                           llvm::StringRef purpose) {
  std::string name = runtimeContractName(contract);
  if (std::optional<RuntimeSymbol> box = manifest.primitive(name, "box")) {
    mlir::FunctionType type = box->function.getFunctionType();
    return llvm::SmallVector<mlir::Type, 8>(type.getResults().begin(),
                                            type.getResults().end());
  }
  return RuntimeBundleLowerer::runtimeValueTypesFor(op, contract, purpose);
}

mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>>
RuntimeBundleLowerer::unboxSlotElementValues(mlir::Operation *op,
                                             mlir::Type contract,
                                             llvm::ArrayRef<mlir::Value> values) {
  std::string name = runtimeContractName(contract);
  std::optional<RuntimeSymbol> unbox = manifest.primitive(name, "unbox");
  if (!unbox)
    return llvm::SmallVector<mlir::Value, 4>(values.begin(), values.end());
  mlir::func::CallOp call =
      RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *unbox, values);
  return llvm::SmallVector<mlir::Value, 4>(call.getResults().begin(),
                                           call.getResults().end());
}

// The at-operation retain of an element does not itself use the container, so
// the container's release could still be placed before it; an explicit
// `__len__` use right after the retain pins the container past it (the same
// device the dynamic-index selection uses).
mlir::LogicalResult
RuntimeBundleLowerer::pinContainerLiveness(mlir::Operation *op,
                                           const RuntimeBundle &container,
                                           bool insertAfterOp) {
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod)
    return op->emitError()
           << "evidence element retain needs a runtime __len__ to pin "
           << container.contractName();
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  builder.setInsertionPoint(insertAfterOp ? op->getNextNode() : op);
  RuntimeBundleLowerer::createRuntimeCall(op->getLoc(), *lenMethod,
                                          lenOperands);
  return mlir::success();
}

mlir::FailureOr<std::optional<RuntimeValue>>
RuntimeBundleLowerer::retainEvidenceElementWithFallback(
    mlir::Operation *op, const RuntimeValue &value,
    const RuntimeBundle *container) {
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, value);
  if (retained || !container)
    return retained;
  if (!manifest.method(container->contractName(), "__len__"))
    return retained;
  retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, value,
                                                  /*atOperation=*/true);
  if (!retained)
    return retained;
  if (mlir::failed(pinContainerLiveness(op, *container)))
    return mlir::failure();
  return retained;
}

mlir::LogicalResult RuntimeBundleLowerer::bindRetainedEvidenceValue(
    mlir::Operation *op, mlir::Value resultValue, llvm::StringRef label,
    const RuntimeValue &value, const RuntimeBundle *container) {
  mlir::FailureOr<std::optional<RuntimeValue>> retained =
      retainEvidenceElementWithFallback(op, value, container);
  if (mlir::failed(retained))
    return mlir::failure();
  if (mlir::failed(bindEvidenceObjectResult(op, resultValue, label,
                                            *retained ? **retained : value)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindOwnedEvidenceValue(
    mlir::Operation *op, mlir::Value resultValue, llvm::StringRef label,
    const RuntimeValue &value) {
  std::optional<RuntimeValue> rooted =
      RuntimeBundleLowerer::rootOwnedEvidenceElement(op, value);
  if (!rooted)
    return op->emitError()
           << label
           << " element already carries a reference but has no point to mark it "
              "frame-owned, so nothing would release it";
  if (mlir::failed(bindEvidenceObjectResult(op, resultValue, label, *rooted)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

mlir::LogicalResult RuntimeBundleLowerer::bindRetainedEvidenceBundle(
    mlir::Operation *op, mlir::Value resultValue, RuntimeBundle bundle,
    const RuntimeBundle *container) {
  // Reading a mutable container OUT of another one hands out an ALIAS, and
  // the copy taken here and the parent's cached description are then two
  // records of one object that no longer track each other.
  //
  //     t: list[list[int]] = [[1, 2]]
  //     t[0].append(3)
  //     a = t[0]
  //     a.append(4)
  //     print(a)      # printed [1, 2, 4, None]; CPython prints [1, 2, 3, 4]
  //
  // The first append grew the object; the parent still described the element
  // as two long, so `a` appended at index 2 -- over the 3 -- and set the
  // length to 4, leaving index 3 never written. A tuple parent did the same
  // and then aborted in repr on the unwritten slot.
  //
  // So contents evidence does not cross the read; the payload does. Both
  // records then answer from the object, which is the only thing that can be
  // shared. Only for a container that HAS a payload -- for an evidence-only
  // one the evidence is the sole description of its contents.
  if (RuntimeBundleLowerer::containerHasRuntimePayload(bundle))
    RuntimeBundleLowerer::demoteMutableContainerEvidence(bundle);
  RuntimeValue element{bundle.objectValue.contract,
                       llvm::SmallVector<mlir::Value, 4>(
                           bundle.physicalValues().begin(),
                           bundle.physicalValues().end()),
                       bundle.objectValue.ownership};
  mlir::FailureOr<std::optional<RuntimeValue>> retained =
      retainEvidenceElementWithFallback(op, element, container);
  if (mlir::failed(retained))
    return mlir::failure();
  if (!*retained)
    return bindSelectedEvidenceObjectResult(op, resultValue, std::move(bundle));
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      bundle.objectValue.contract, (*retained)->values,
      ownership::logicalOwnershipKind(bundle.objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(bundle);
  return bindSelectedEvidenceObjectResult(op, resultValue, std::move(rebuilt));
}

mlir::FailureOr<RuntimeBundle>
RuntimeBundleLowerer::selectEvidenceObjectByMatch(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, mlir::ValueRange matches,
    llvm::StringRef label, llvm::StringRef missingContract,
    llvm::StringRef missingMessage, bool raiseOnMiss,
    const RuntimeBundle *missingKeyForRepr) {
  context->loadDialect<mlir::scf::SCFDialect>();
  if (candidates.empty() || candidates.size() != matches.size()) {
    op->emitError() << label << " evidence match/value count mismatch";
    return mlir::failure();
  }

  // ⭐ A UNION RESULT WIDENS EVERY CANDIDATE FIRST. The candidates are the
  // members the evidence recorded -- an int and a str have different lane
  // counts -- so the uniformity check below reported "candidate 1 has a
  // different physical ABI shape" for `for v in (1, "a")` typed
  // `tuple[int | str, ...]`, which is a shape mismatch only because nothing
  // had put them into the union's own form yet.
  llvm::SmallVector<RuntimeValue, 4> widened;
  if (auto resultUnion =
          mlir::dyn_cast<py::UnionType>(resultValue.getType())) {
    if (llvm::any_of(candidates, [&](const RuntimeValue &candidate) {
          return candidate.contract != resultValue.getType();
        })) {
      for (const RuntimeValue &candidate : candidates) {
        RuntimeBundle source = RuntimeBundle::objectWithOwnership(
            candidate.contract, candidate.values, candidate.ownership);
        llvm::SmallVector<mlir::Value, 8> values;
        if (mlir::failed(RuntimeBundleLowerer::appendUnionRuntimeValues(
                op, resultUnion, source, candidate.contract, values)))
          return mlir::failure();
        widened.push_back(RuntimeValue{
            resultValue.getType(),
            llvm::SmallVector<mlir::Value, 4>(values.begin(), values.end()),
            ownership::logicalOwnershipKind(resultValue.getType(),
                                            /*ownsObject=*/false)});
      }
      candidates = widened;
    }
  }

  const RuntimeValue &first = candidates.front();
  if (first.values.empty()) {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> expected =
        RuntimeBundleLowerer::runtimeValueTypesFor(op, first.contract,
                                                   "evidence candidate ABI");
    if (mlir::failed(expected))
      return mlir::failure();
    if (!expected->empty()) {
      op->emitError() << label << " evidence candidate has no physical values";
      return mlir::failure();
    }
  }

  llvm::ArrayRef<mlir::Value> firstValues = candidates.front().values;
  for (auto [position, candidate] : llvm::enumerate(candidates)) {
    if (!py::isAssignableTo(candidate.contract, resultValue.getType(), op)) {
      op->emitError() << label << " evidence candidate " << position
                      << " contract " << candidate.contract
                      << " is not assignable to result "
                      << resultValue.getType();
      return mlir::failure();
    }
    if (!sameValueTypes(firstValues, candidate.values)) {
      op->emitError() << label << " evidence candidate " << position
                      << " has a different physical ABI shape";
      return mlir::failure();
    }
    if (!matches[position].getType().isInteger(1)) {
      op->emitError() << label << " evidence match " << position
                      << " must be i1";
      return mlir::failure();
    }
  }

  mlir::Location loc = op->getLoc();
  llvm::SmallVector<mlir::Type, 4> resultTypes;
  for (mlir::Value value : firstValues)
    resultTypes.push_back(value.getType());

  // A zero-result scf.if (e.g. a None candidate with no physical values) gets
  // its empty terminators auto-inserted at build time; creating manual yields
  // there would leave two terminators per block.
  bool needsYields = !resultTypes.empty();
  auto emitChain = [&](auto &&self, unsigned position)
      -> mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> {
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, resultTypes, matches[position], /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    if (needsYields)
      mlir::scf::YieldOp::create(builder, loc, candidates[position].values);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    if (position + 1 < candidates.size()) {
      mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> nested =
          self(self, position + 1);
      if (mlir::failed(nested))
        return mlir::failure();
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, *nested);
    } else if (raiseOnMiss) {
      // The runtime key itself is the raise message object: KeyError
      // __init__ stores repr(message) (str(KeyError(x)) == repr(x)), so no
      // separate __repr__ call is needed on the miss branch.
      // The key cannot be consumed here directly (or via a retained alias):
      // this branch is CONDITIONAL, the exception __init__ consumes its
      // message, and the ownership pass cannot place a single static
      // release for both paths. A FRESH clone created and consumed inside
      // this branch keeps the key a plain borrow while the message still
      // carries the missing key, CPython-style.
      bool raisedWithKey = false;
      if (missingKeyForRepr &&
          missingKeyForRepr->contractName() == "builtins.str" &&
          missingKeyForRepr->physicalValues().size() == 2) {
        std::optional<RuntimeSymbol> clone =
            manifest.primitive("builtins.str", "clone");
        if (clone &&
            clone->function.getFunctionType().getNumInputs() == 2 &&
            clone->function.getFunctionType().getNumResults() == 2) {
          llvm::SmallVector<mlir::Value, 2> cloneOperands;
          bool typesMatch = true;
          for (auto [operand, want] :
               llvm::zip(missingKeyForRepr->physicalValues(),
                         clone->function.getFunctionType().getInputs())) {
            if (operand.getType() != want) {
              typesMatch = false;
              break;
            }
            cloneOperands.push_back(operand);
          }
          if (typesMatch) {
            mlir::func::CallOp cloneCall =
                RuntimeBundleLowerer::createRuntimeCall(loc, *clone,
                                                        cloneOperands);
            RuntimeBundle messageObject = RuntimeBundle::object(
                runtimeContractType(context, "builtins.str"),
                cloneCall.getResults());
            if (mlir::failed(
                    RuntimeBundleLowerer::emitRuntimeExceptionFromMessageObject(
                        op, missingContract, messageObject)))
              return mlir::failure();
            raisedWithKey = true;
          }
        }
      }
      if (!raisedWithKey &&
          mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
              op, missingContract, missingMessage)))
        return mlir::failure();
      llvm::SmallVector<mlir::Value, 4> deadValues;
      deadValues.reserve(resultTypes.size());
      for (mlir::Type resultType : resultTypes) {
        mlir::FailureOr<mlir::Value> dead =
            RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
        if (mlir::failed(dead))
          return mlir::failure();
        deadValues.push_back(*dead);
      }
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, deadValues);
    } else {
      // A raise-free miss (e.g. iterator exhaustion) executes on every normal
      // completion; use immortal static placeholders instead of heap
      // allocations so the miss path does not leak.
      mlir::FailureOr<RuntimeValue> dead =
          RuntimeBundleLowerer::materializeDeadObjectValueImpl(
              op, first.contract, label, DeadObjectStorage::StaticNonOwning);
      if (mlir::failed(dead))
        return mlir::failure();
      if (dead->values.size() != resultTypes.size())
        return op->emitError()
               << label << " static miss placeholder ABI mismatch";
      if (needsYields)
        mlir::scf::YieldOp::create(builder, loc, dead->values);
    }

    builder.setInsertionPointAfter(ifOp);
    return llvm::SmallVector<mlir::Value, 4>(ifOp.getResults().begin(),
                                             ifOp.getResults().end());
  };

  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> selected =
      emitChain(emitChain, 0);
  if (mlir::failed(selected))
    return mlir::failure();

  mlir::Type bundleContract = first.contract;
  std::string resultContract = runtimeContractName(resultValue.getType());
  if (!resultContract.empty() && objectShapeMatches(resultContract, *selected))
    bundleContract = resultValue.getType();
  return RuntimeBundle::objectWithOwnership(
      bundleContract, *selected,
      ownership::logicalOwnershipKind(bundleContract,
                                      /*ownsObject=*/false));
}

mlir::FailureOr<RuntimeBundle> RuntimeBundleLowerer::selectEvidenceObjectMiss(
    mlir::Operation *op, mlir::Value resultValue,
    llvm::ArrayRef<RuntimeValue> candidates, llvm::StringRef label,
    llvm::StringRef missingContract, llvm::StringRef missingMessage) {
  (void)candidates;
  builder.setInsertionPoint(op);
  if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
          op, missingContract, missingMessage)))
    return mlir::failure();

  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> resultTypes =
      RuntimeBundleLowerer::runtimeValueTypesFor(op, resultValue.getType(),
                                                 label);
  if (mlir::failed(resultTypes))
    return mlir::failure();

  llvm::SmallVector<mlir::Value, 4> deadValues;
  deadValues.reserve(resultTypes->size());
  for (mlir::Type resultType : *resultTypes) {
    mlir::FailureOr<mlir::Value> dead =
        RuntimeBundleLowerer::materializeDeadPhysicalValue(op, resultType);
    if (mlir::failed(dead))
      return mlir::failure();
    deadValues.push_back(*dead);
  }
  return RuntimeBundle::objectWithOwnership(
      resultValue.getType(), deadValues,
      ownership::logicalOwnershipKind(resultValue.getType(),
                                      /*ownsObject=*/false));
}

mlir::FailureOr<bool> RuntimeBundleLowerer::lowerSequenceEvidenceGetItem(
    py::GetItemOp op, const RuntimeBundle &container,
    const RuntimeBundle &index) {
  // This answers a read from evidence that describes the container as of its
  // DEFINITION. That is sound only where the walk has seen every store, which
  // for a mutable container is the block defining its storage and nowhere
  // else -- `demoteCrossBlockContainerEvidence` has already emptied the
  // element map by the time this runs anywhere else, so the check below is
  // what declines the read.
  //
  // ⛔ Do not restore the gate that was tried here instead. Returning false
  // from this function leaves the bundle still claiming to be evidence-backed,
  // and the read then has nowhere to go -- "runtime manifest has no
  // builtins.list.__getitem__ method". The invalidation has to happen to the
  // BUNDLE, which is why it lives with the container and not with the read.
  if (container.sequenceElements.empty())
    return false;

  std::optional<std::int64_t> rawIndex = integerLiteralFromValue(op.getIndex());
  if (rawIndex) {
    if (!container.sequenceIndices.empty()) {
      if (container.sequenceIndices.size() !=
          container.sequenceElements.size()) {
        op.emitError() << "sequence evidence index/value count mismatch";
        return mlir::failure();
      }
      for (auto [position, storedIndex] :
           llvm::enumerate(container.sequenceIndices)) {
        if (storedIndex != *rawIndex)
          continue;
        if (position < container.sequenceElementBundles.size() &&
            container.sequenceElementBundles[position]) {
          RuntimeBundle selected = *container.sequenceElementBundles[position];
          if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                      std::move(selected),
                                                      &container)))
            return mlir::failure();
          return true;
        }
        const RuntimeValue &element = container.sequenceElements[position];
        if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                                   "sequence __getitem__",
                                                   element, &container)))
          return mlir::failure();
        return true;
      }
      mlir::FailureOr<RuntimeBundle> selected =
          RuntimeBundleLowerer::selectEvidenceObjectMiss(
              op, op.getResult(), container.sequenceElements,
              "sequence __getitem__", "builtins.IndexError",
              indexOutOfRangeMessage(container.contractName()));
      if (mlir::failed(selected))
        return mlir::failure();
      if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                        std::move(*selected))))
        return mlir::failure();
      return true;
    }

    std::int64_t normalized = *rawIndex;
    std::int64_t size =
        static_cast<std::int64_t>(container.sequenceElements.size());
    if (normalized < 0)
      normalized += size;
    if (normalized < 0 || normalized >= size) {
      mlir::FailureOr<RuntimeBundle> selected =
          RuntimeBundleLowerer::selectEvidenceObjectMiss(
              op, op.getResult(), container.sequenceElements,
              "sequence __getitem__", "builtins.IndexError",
              indexOutOfRangeMessage(container.contractName()));
      if (mlir::failed(selected))
        return mlir::failure();
      if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                        std::move(*selected))))
        return mlir::failure();
      return true;
    }

    unsigned elementIndex = static_cast<unsigned>(normalized);
    if (elementIndex < container.sequenceElementBundles.size() &&
        container.sequenceElementBundles[elementIndex]) {
      RuntimeBundle selected = *container.sequenceElementBundles[elementIndex];
      if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                  std::move(selected),
                                                  &container)))
        return mlir::failure();
      return true;
    }

    const RuntimeValue &element = container.sequenceElements[elementIndex];
    if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                               "sequence __getitem__", element,
                                               &container)))
      return mlir::failure();
    return true;
  }

  if (!container.sequenceIndices.empty())
    return false;

  builder.setInsertionPoint(op);
  std::optional<RuntimeSymbol> lenMethod =
      manifest.method(container.contractName(), "__len__");
  if (!lenMethod) {
    op.emitError() << "sequence evidence dynamic index needs a runtime __len__";
    return mlir::failure();
  }
  llvm::SmallVector<const RuntimeBundle *, 1> lenSources{&container};
  llvm::SmallVector<mlir::Value, 4> lenOperands;
  if (mlir::failed(buildRuntimeCallOperands(op, *lenMethod, lenSources,
                                            lenOperands,
                                            /*allowUnusedSources=*/false)))
    return mlir::failure();
  mlir::func::CallOp lenCall = RuntimeBundleLowerer::createRuntimeCall(
      op.getLoc(), *lenMethod, lenOperands);
  if (lenCall.getNumResults() != 1 ||
      !lenCall.getResult(0).getType().isInteger(64)) {
    lenMethod->function.emitError()
        << "sequence __len__ evidence method must return one i64";
    return mlir::failure();
  }

  // ⭐ Through the shared reader, not a second call to the unbox primitive.
  // This site called it with the index bundle's physical values whatever they
  // were, and an index whose lanes do not match the primitive's arity crashed
  // the compiler -- `i: int = 0; print([1][i])` failed with "'func.call' op
  // incorrect number of operands for callee". `rawSequenceIndexValue` is the
  // same question three other sites already ask, and it answers a literal and
  // a primitive-i64 lane before reaching for the primitive at all.
  mlir::FailureOr<mlir::Value> dynamicIndex =
      RuntimeBundleLowerer::rawSequenceIndexValue(op.getOperation(),
                                                  op.getIndex(), index);
  if (mlir::failed(dynamicIndex))
    return mlir::failure();

  mlir::Location loc = op.getLoc();
  mlir::Value rawRuntimeIndex = *dynamicIndex;
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value runtimeSize = lenCall.getResult(0);
  mlir::Value isNegative = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, rawRuntimeIndex, zero);
  mlir::Value adjusted =
      mlir::arith::AddIOp::create(builder, loc, rawRuntimeIndex, runtimeSize);
  mlir::Value normalized = mlir::arith::SelectOp::create(
      builder, loc, isNegative, adjusted, rawRuntimeIndex);
  mlir::Value lowerOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, normalized, zero);
  mlir::Value upperOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, normalized, runtimeSize);
  mlir::Value inRange =
      mlir::arith::AndIOp::create(builder, loc, lowerOk, upperOk);

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(container.sequenceElements.size());
  for (unsigned position = 0,
                end = static_cast<unsigned>(container.sequenceElements.size());
       position < end; ++position) {
    mlir::Value expected = mlir::arith::ConstantIntOp::create(
        builder, loc, static_cast<std::int64_t>(position), 64);
    mlir::Value indexMatches = mlir::arith::CmpIOp::create(
        builder, loc, mlir::arith::CmpIPredicate::eq, normalized, expected);
    matches.push_back(
        mlir::arith::AndIOp::create(builder, loc, inRange, indexMatches));
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.sequenceElements, matches,
          "sequence __getitem__", "builtins.IndexError",
          indexOutOfRangeMessage(container.contractName()));
  if (mlir::failed(selected))
    return mlir::failure();

  // The selected element is a borrow of a container slot. Retain it right
  // after the selection (per selection, so a getitem inside a loop stays
  // balanced), then pin the container's liveness past the retain with an
  // explicit `__len__` use — otherwise the container's release is placed after
  // its previous last use (the bounds check above) and would free the elements
  // before the selection reads them.
  RuntimeValue chainElement{(*selected).objectValue.contract,
                            llvm::SmallVector<mlir::Value, 4>(
                                (*selected).physicalValues().begin(),
                                (*selected).physicalValues().end()),
                            (*selected).objectValue.ownership};
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, chainElement);
  if (!retained) {
    op.emitError() << "sequence __getitem__ cannot retain evidence element "
                   << chainElement.contract << " selected by a dynamic index";
    return mlir::failure();
  }
  if (mlir::failed(pinContainerLiveness(op, container)))
    return mlir::failure();
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      (*selected).objectValue.contract, retained->values,
      ownership::logicalOwnershipKind((*selected).objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(*selected);
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(rebuilt))))
    return mlir::failure();
  return true;
}

mlir::FailureOr<bool>
RuntimeBundleLowerer::lowerDictEvidenceGetItem(
    py::GetItemOp op, const RuntimeBundle &containerRef,
    const RuntimeBundle &indexRef) {
  // Copies: this function inserts into `valueBundles` and then keeps reading its
  // operand bundles, and the caller's arguments are references INTO that
  // DenseMap -- an insertion that rehashes moves the entry and every later read
  // is freed memory. Found as a live defect on `lowerBoundMethodCall`'s receiver
  // (see CallableOps.cpp); these are the rest of the same audit. Neither of
  // these keys is ever rewritten here, so the copy changes nothing else.
  RuntimeBundle container = containerRef;
  RuntimeBundle index = indexRef;
  if (container.contractName() != "builtins.dict" ||
      container.mappingKeys.empty())
    return false;

  if (container.mappingKeys.size() != container.mappingValues.size()) {
    op.emitError() << "dict evidence key/value count mismatch";
    return mlir::failure();
  }
  bool hasPresence = !container.mappingPresent.empty();
  if (hasPresence &&
      container.mappingPresent.size() != container.mappingKeys.size()) {
    op.emitError() << "dict evidence key/presence count mismatch";
    return mlir::failure();
  }

  std::optional<std::string> key =
      RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
  if (key) {
    for (auto [position, storedKey] : llvm::enumerate(container.mappingKeys)) {
      if (storedKey != *key)
        continue;
      if (!hasPresence && position < container.mappingValueBundles.size() &&
          container.mappingValueBundles[position]) {
        RuntimeBundle selected = *container.mappingValueBundles[position];
        if (mlir::failed(bindRetainedEvidenceBundle(op, op.getResult(),
                                                    std::move(selected),
                                                    &container)))
          return mlir::failure();
        return true;
      }
      const RuntimeValue &value = container.mappingValues[position];
      if (hasPresence) {
        builder.setInsertionPoint(op);
        // Retain the candidate at its definition so the selected value
        // survives the container's release; fall back to the borrowed view for
        // contracts without an `own` primitive (previous behavior). The
        // presence test is a guard that raises on a missing key — the value is
        // bound directly (not routed through scf.if results) so its ownership
        // stays visible to the affine verifier.
        RuntimeValue candidate = value;
        if (std::optional<RuntimeValue> retained =
                RuntimeBundleLowerer::retainEvidenceElement(op, value))
          candidate = std::move(*retained);
        context->loadDialect<mlir::scf::SCFDialect>();
        mlir::Location loc = op.getLoc();
        mlir::Value one =
            mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
        mlir::Value missing = mlir::arith::XOrIOp::create(
            builder, loc, container.mappingPresent[position], one);
        auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                             missing, /*withElseRegion=*/false);
        builder.setInsertionPointToStart(&guard.getThenRegion().front());
        // Raw key, not its repr: the KeyError __init__ stores repr(message)
        // (str(KeyError(x)) == repr(x)), so pre-repring would double-quote.
        if (mlir::failed(RuntimeBundleLowerer::emitRuntimeException(
                op, "builtins.KeyError", *key)))
          return mlir::failure();
        mlir::Block &thenBlock = guard.getThenRegion().front();
        if (thenBlock.empty() ||
            !thenBlock.back().hasTrait<mlir::OpTrait::IsTerminator>())
          mlir::scf::YieldOp::create(builder, loc);
        builder.setInsertionPointAfter(guard);
        if (mlir::failed(bindEvidenceObjectResult(
                op, op.getResult(), "dict __getitem__", candidate)))
          return mlir::failure();
        erase.push_back(op);
        return true;
      }
      if (mlir::failed(bindRetainedEvidenceValue(op, op.getResult(),
                                                 "dict __getitem__", value,
                                                 &container)))
        return mlir::failure();
      return true;
    }

    mlir::FailureOr<RuntimeBundle> selected =
        RuntimeBundleLowerer::selectEvidenceObjectMiss(
            op, op.getResult(), container.mappingValues, "dict __getitem__",
            "builtins.KeyError", *key);
    if (mlir::failed(selected))
      return mlir::failure();
    if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                      std::move(*selected))))
      return mlir::failure();
    return true;
  }

  if (index.contractName() != "builtins.str")
    return false;

  builder.setInsertionPoint(op);
  std::optional<RuntimeSymbol> eq =
      manifest.method(index.contractName(), "__eq__");
  if (!eq) {
    op.emitError() << "dict evidence dynamic key needs str.__eq__";
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 8> matches;
  matches.reserve(container.mappingKeys.size());
  llvm::SmallVector<RuntimeBundle, 8> materializedKeys;
  materializedKeys.reserve(container.mappingKeys.size());
  for (auto [position, storedKey] : llvm::enumerate(container.mappingKeys)) {
    RuntimeBundle &keyObject = materializedKeys.emplace_back();
    if (mlir::failed(RuntimeBundleLowerer::materializeStringObject(
            op, storedKey, keyObject)))
      return mlir::failure();
    llvm::SmallVector<const RuntimeBundle *, 2> eqSources{&index, &keyObject};
    llvm::SmallVector<mlir::Value, 4> eqOperands;
    if (mlir::failed(buildRuntimeCallOperands(op, *eq, eqSources, eqOperands,
                                              /*allowUnusedSources=*/false)))
      return mlir::failure();
    mlir::func::CallOp eqCall =
        RuntimeBundleLowerer::createRuntimeCall(op.getLoc(), *eq, eqOperands);
    if (eqCall.getNumResults() != 1 ||
        !eqCall.getResult(0).getType().isInteger(1)) {
      eq->function.emitError()
          << "str.__eq__ evidence method must return one i1";
      return mlir::failure();
    }
    mlir::Value match = eqCall.getResult(0);
    if (hasPresence)
      match = mlir::arith::AndIOp::create(builder, op.getLoc(), match,
                                          container.mappingPresent[position]);
    matches.push_back(match);
  }

  mlir::FailureOr<RuntimeBundle> selected =
      RuntimeBundleLowerer::selectEvidenceObjectByMatch(
          op, op.getResult(), container.mappingValues, matches,
          "dict __getitem__", "builtins.KeyError", "key not found",
          /*raiseOnMiss=*/true, &index);
  if (mlir::failed(selected))
    return mlir::failure();

  // Retain the selected value per selection and pin the container's liveness
  // past the retain (see the sequence dynamic-index path for the rationale).
  RuntimeValue chainValue{(*selected).objectValue.contract,
                          llvm::SmallVector<mlir::Value, 4>(
                              (*selected).physicalValues().begin(),
                              (*selected).physicalValues().end()),
                          (*selected).objectValue.ownership};
  std::optional<RuntimeValue> retained =
      RuntimeBundleLowerer::retainEvidenceElement(op, chainValue);
  if (!retained) {
    op.emitError() << "dict __getitem__ cannot retain evidence value "
                   << chainValue.contract << " selected by a dynamic key";
    return mlir::failure();
  }
  if (mlir::failed(pinContainerLiveness(op, container)))
    return mlir::failure();
  RuntimeBundle rebuilt = RuntimeBundle::objectWithOwnership(
      (*selected).objectValue.contract, retained->values,
      ownership::logicalOwnershipKind((*selected).objectValue.contract,
                                      /*ownsObject=*/false));
  rebuilt.copyEvidenceFrom(*selected);
  if (mlir::failed(bindSelectedEvidenceObjectResult(op, op.getResult(),
                                                    std::move(rebuilt))))
    return mlir::failure();
  return true;
}

// One element out of a runtime container's payload. Shared, because a
// subscript, an iteration step and a starred call argument all rebuild an
// element the same way and only differ in how they got the slot.
mlir::FailureOr<RuntimeValue> RuntimeBundleLowerer::payloadElementAt(
    mlir::Operation *op, const RuntimeBundle &container, mlir::Value slot,
    mlir::Value valid, mlir::Type elementContract, llvm::StringRef label,
    bool &arrivesOwned) {
  mlir::Location loc = op->getLoc();
  arrivesOwned = false;
  mlir::FailureOr<mlir::Value> itemsView =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Primary, label);
  if (mlir::failed(itemsView))
    return mlir::failure();
  mlir::Value wordsPerSlot =
      mlir::arith::ConstantIntOp::create(builder, loc, box_abi::kWordsPerBox, 64);
  mlir::Value base =
      mlir::arith::MulIOp::create(builder, loc, slot, wordsPerSlot).getResult();
  if (auto elementUnion = mlir::dyn_cast<py::UnionType>(elementContract)) {
    mlir::Value classWord =
        box_abi::loadContainerBoxWord(builder, loc, *itemsView, base, 1);
    mlir::Value entityWord = box_abi::loadContainerBoxWord(
        builder, loc, *itemsView, base, box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> unionValues =
        RuntimeBundleLowerer::unionValuesFromBoxWords(op, elementUnion,
                                                      classWord, entityWord);
    if (mlir::failed(unionValues))
      return mlir::failure();
    return RuntimeValue{
        elementContract,
        llvm::SmallVector<mlir::Value, 4>(unionValues->begin(),
                                          unionValues->end()),
        ownership::logicalOwnershipKind(elementContract,
                                        /*ownsObject=*/false)};
  }
  llvm::SmallVector<mlir::Value, 4> elementValues;
  if (runtimeContractName(elementContract) == "builtins.object") {
    // Erased read lane: see lowerRuntimeDictGetItem.
    std::optional<RuntimeSymbol> fromSlot =
        manifest.primitive("builtins.object", "from_slot");
    if (!fromSlot)
      return op->emitError()
             << "runtime manifest has no object from_slot primitive";
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *fromSlot, mlir::ValueRange{*itemsView, slot, valid});
    elementValues.push_back(boxed.getResult(0));
    arrivesOwned = true;
  } else {
    mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
        RuntimeBundleLowerer::slotStorageShapesFor(op, elementContract, label);
    if (mlir::failed(shapes))
      return mlir::failure();
    mlir::Value entityWord = box_abi::loadContainerBoxWord(
        builder, loc, *itemsView, base, box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> lanes =
        RuntimeBundleLowerer::lanesFromBoxEntity(
            builder, loc, entityWord, *shapes,
            runtimeContractName(elementContract), op);
    if (mlir::failed(lanes))
      return mlir::failure();
    elementValues.append(lanes->begin(), lanes->end());
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, elementContract,
                                                   elementValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  return RuntimeValue{elementContract, *canonical,
                      ownership::logicalOwnershipKind(elementContract,
                                                      /*ownsObject=*/false)};
}

// `xs[i]` on a runtime-mode list or tuple (identical physical layout: header,
// length, boxes): runtime bounds check (negative indices normalize),
// IndexError on miss, element rebuilt from its box words and retained
// (borrow → own).
mlir::FailureOr<bool> RuntimeBundleLowerer::lowerRuntimeSequenceGetItem(
    py::GetItemOp op, const RuntimeBundle &containerRef,
    const RuntimeBundle &indexRef) {
  // Copies: binding results below inserts into valueBundles (a DenseMap),
  // which invalidates references into it.
  RuntimeBundle container = containerRef;
  RuntimeBundle index = indexRef;
  // An evidence-BACKED container with no recorded elements (an annotated
  // empty literal grown by loop appends) reads through the payload too.
  //
  // ⭐ AND ONE WITH A SECOND HOLDER, whatever its evidence says:
  //
  //     seed = [3, 1, 2]
  //     b = Bag(seed)          # self.xs = xs -- one object, two names
  //     b.xs[0] = 9
  //     print(seed[0])         # printed 3; CPython prints 9
  //
  // The write landed (`print(seed)` was right) and only the ELEMENT read was
  // wrong: it answered from `seed`'s compile-time slot evidence, which no
  // mutation through the other name had any way to update. `b.xs.sort()` and
  // `holder[0][0] = 9` were wrong the same way, so this is the read side of
  // `sharedWithHolder` rather than one mutator's omission.
  //
  // ⛔ Why NOT demote the evidence at the absorption instead: measured three
  // ways and it takes 145-146 tests down, aborting at runtime -- the evidence
  // is where the slot's owned reference is BOOKED, not only what it describes
  // (`markAbsorbedContainerAsShared`). So the evidence stays and the read moves.
  bool secondHolderMayHaveMutated =
      container.sharedWithHolder &&
      container.contractName() == "builtins.list";
  if ((container.contractName() != "builtins.list" &&
       container.contractName() != "builtins.tuple") ||
      (!container.sequenceElements.empty() && !secondHolderMayHaveMutated) ||
      !RuntimeBundleLowerer::containerHasRuntimePayload(container))
    return false;
  mlir::Type elementContract = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, elementContract,
                                                 "runtime list element");
  if (mlir::failed(shapes))
    return mlir::failure();
  // ⭐ A UNION ELEMENT IS BUILT FROM THE BOX, not read as lanes. Its physical
  // form starts with a TAG, which is an i64 and not a memref, so the shape
  // check below declined it and the read fell through to a manifest
  // `__getitem__` that returns no union: "runtime manifest has no
  // builtins.list.__getitem__ method", for `list[int | str]` whenever the
  // container arrived as a parameter or was read in a loop -- the two places
  // the per-element evidence is gone.
  auto elementUnion = mlir::dyn_cast<py::UnionType>(elementContract);
  if (!elementUnion)
    for (mlir::Type shape : *shapes) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
      if (!memref || memref.getRank() != 1)
        return false;
    }
  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::Value raw;
  if (primitiveI64LaneKnownValid(index.primitiveI64)) {
    raw = index.primitiveI64->value;
  } else if (std::optional<std::int64_t> literal =
                 integerLiteralFromValue(op.getIndex())) {
    raw = mlir::arith::ConstantIntOp::create(builder, loc, *literal, 64);
  } else {
    std::optional<RuntimeSymbol> unbox =
        manifest.primitive(index.contractName(), "unbox.i64");
    if (unbox &&
        unbox->function.getNumArguments() == index.physicalValues().size()) {
      mlir::func::CallOp indexCall = RuntimeBundleLowerer::createRuntimeCall(
          loc, *unbox, index.physicalValues());
      raw = indexCall.getResult(0);
    } else if (index.primitiveI64 && index.primitiveI64->value) {
      // No boxed payload to fall back to (primitive-i64 clone lanes carry
      // only the (value, valid) pair): the lane is the sole carrier.
      raw = index.primitiveI64->value;
    } else {
      return false;
    }
  }
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::FailureOr<mlir::Value> lengthOr =
      RuntimeBundleLowerer::loadContainerLength(op, container,
                                                "runtime list getitem");
  if (mlir::failed(lengthOr))
    return mlir::failure();
  mlir::Value length = *lengthOr;
  mlir::Value isNegative = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, raw, zero);
  mlir::Value adjusted =
      mlir::arith::AddIOp::create(builder, loc, raw, length).getResult();
  mlir::Value normalized =
      mlir::arith::SelectOp::create(builder, loc, isNegative, adjusted, raw)
          .getResult();
  mlir::Value lowerOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::sge, normalized, zero);
  mlir::Value upperOk = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, normalized, length);
  mlir::Value inRange =
      mlir::arith::AndIOp::create(builder, loc, lowerOk, upperOk).getResult();
  mlir::Value outOfRange = mlir::arith::XOrIOp::create(
      builder, loc, inRange,
      mlir::arith::ConstantIntOp::create(builder, loc, 1, 1).getResult());
  auto guard = mlir::scf::IfOp::create(builder, loc, mlir::TypeRange{},
                                       outOfRange, /*withElseRegion=*/false);
  {
    mlir::OpBuilder::InsertionGuard insertionGuard(builder);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    if (mlir::failed(emitRuntimeException(
            op, "builtins.IndexError",
            indexOutOfRangeMessage(container.contractName()))))
      return mlir::failure();
  }
  builder.setInsertionPointAfter(guard);
  // Unreachable at runtime past the raise; clamp keeps the loads in bounds.
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, inRange, normalized, zero)
          .getResult();
  llvm::SmallVector<mlir::Value, 4> elementValues;
  // ⭐ Which branch ran decides whether the element ARRIVES with a reference.
  // The erased lane calls `from_slot`, which allocates a fresh box at refcount 1;
  // the multi-lane branch reads the container's own box words, which is a borrow.
  // Retaining both left the erased lane at 2 against one release -- one leaked
  // object per boxed slot read, unbounded.
  bool elementArrivesOwned = false;
  mlir::FailureOr<RuntimeValue> element = RuntimeBundleLowerer::payloadElementAt(
      op, container, safe, inRange, elementContract, "runtime list getitem",
      elementArrivesOwned);
  if (mlir::failed(element))
    return mlir::failure();
  if (elementUnion) {
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> owned =
        RuntimeBundleLowerer::retainUnionMemberValues(op, elementUnion,
                                                       element->values);
    if (mlir::failed(owned))
      return mlir::failure();
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundleWithOwnership(
            op, elementContract, *owned, result,
            ownership::logicalOwnershipKind(elementContract,
                                            /*ownsObject=*/true))))
      return mlir::failure();
    valueBundles[op.getResult()] = std::move(result);
    // ⛔ AND THE CONTAINER IS PINNED PAST THE RETAINS. The release planner puts
    // a temporary's death after its last USE, and the union's lanes are built
    // from the box by arithmetic on an ADDRESS -- nothing it reads as a use of
    // the container. It placed a returned tuple's release between the build
    // and the retain, so the retain ran on a string already at zero.
    if (mlir::failed(pinContainerLiveness(op, container,
                                          /*insertAfterOp=*/true)))
      return mlir::failure();
    erase.push_back(op);
    return true;
  }
  if (mlir::failed(elementArrivesOwned
                       ? bindOwnedEvidenceValue(op, op.getResult(),
                                                "runtime sequence __getitem__",
                                                *element)
                       : bindRetainedEvidenceValue(op, op.getResult(),
                                                   "runtime sequence __getitem__",
                                                   *element)))
    return mlir::failure();
  if (mlir::failed(pinContainerLiveness(op, container,
                                        /*insertAfterOp=*/true)))
    return mlir::failure();
  return true;
}

// `d[k]` on a runtime-mode dict: hash-based runtime probe over a transient
// key box (any hashable key class), KeyError carrying the key's repr on a
// miss, value rebuilt from its box words and retained (borrow → own).
mlir::FailureOr<bool> RuntimeBundleLowerer::lowerRuntimeDictGetItem(
    py::GetItemOp op, const RuntimeBundle &containerRef,
    const RuntimeBundle &indexRef) {
  // Copies: see lowerRuntimeSequenceGetItem.
  RuntimeBundle container = containerRef;
  RuntimeBundle index = indexRef;
  if (container.contractName() != "builtins.dict" ||
      !RuntimeBundleLowerer::containerHasRuntimePayload(container))
    return false;
  if (index.kind != RuntimeBundle::Kind::Object)
    return false;
  // Evidence-backed dicts qualify only for RUNTIME keys (an int variable, a
  // frozenset): a literal key stays on the evidence tier below, which owns
  // the literal-key semantics (and this runtime path runs first).
  //
  // ⭐ UNLESS A SECOND HOLDER CAN MUTATE IT, and then even a literal key comes
  // from the payload. This is the mapping half of `sharedWithHolder`'s read
  // side; the sequence half is in lowerRuntimeSequenceGetItem, with the
  // measurements. Same four shapes, same silence:
  //
  //     seed = {"a": 1}
  //     b = Bag(seed)          # self.d = d -- one object, two names
  //     b.d["a"] = 9
  //     print(seed["a"])       # printed 1; CPython prints 9
  //     b.d["z"] = 5
  //     print(seed["z"])       # KeyError; CPython prints 5
  //
  // The literal-key semantics this defers to are exactly what goes wrong: the
  // evidence tier answers a literal key from the keys it recorded, and a key
  // ADDED through the other name is not among them.
  if ((container.mappingEvidenceBacked || !container.mappingKeys.empty()) &&
      !container.sharedWithHolder) {
    std::optional<std::string> literalKey =
        RuntimeBundleLowerer::keywordNameFromValue(op.getIndex());
    if (!literalKey && index.literalText)
      literalKey = *index.literalText;
    if (literalKey)
      return false;
  }
  mlir::Type valueContract = op.getResult().getType();
  mlir::FailureOr<llvm::SmallVector<mlir::Type, 8>> shapes =
      RuntimeBundleLowerer::slotStorageShapesFor(op, valueContract,
                                                 "runtime dict value");
  if (mlir::failed(shapes))
    return mlir::failure();
  // A union value is rebuilt from the box below; its first physical value is
  // the tag, so the lane check that follows would decline it -- and declining
  // here is not a refusal but a fall-through to a `dict.__getitem__` method
  // the manifest does not have. `d[k]` for a runtime key k on a
  // `dict[str, int | str]` reported the missing method instead.
  auto valueUnion = mlir::dyn_cast<py::UnionType>(valueContract);
  if (!valueUnion)
    for (mlir::Type shape : *shapes) {
      auto memref = mlir::dyn_cast<mlir::MemRefType>(shape);
      if (!memref || memref.getRank() != 1)
        return false;
    }
  std::optional<RuntimeSymbol> lookupBox =
      manifest.primitive("builtins.dict", "lookup_box_checked");
  if (!lookupBox) {
    op.emitError()
        << "runtime manifest has no dict lookup_box_checked primitive";
    return mlir::failure();
  }

  builder.setInsertionPoint(op);
  mlir::Location loc = op.getLoc();
  mlir::FailureOr<RuntimeBundle> payloadKey =
      RuntimeBundleLowerer::materializePayloadObjectBundle(op, index);
  if (mlir::failed(payloadKey))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> keyBox =
      RuntimeBundleLowerer::transientPayloadBox(op, *payloadKey,
                                                /*ownsPayload=*/false);
  if (mlir::failed(keyBox))
    return mlir::failure();
  llvm::SmallVector<mlir::Value, 8> findOperands(
      container.physicalValues().begin(), container.physicalValues().end());
  findOperands.push_back(*keyBox);
  mlir::func::CallOp findCall =
      RuntimeBundleLowerer::createRuntimeCall(loc, *lookupBox, findOperands);
  mlir::Value found = findCall.getResult(0);
  // The probe consumed only raw pointer words: pin the key past the call
  // (owned keys are consumed by the release inside the pin).
  if (mlir::failed(RuntimeBundleLowerer::pinProbeOperandLiveness(
          op, *payloadKey, &index)))
    return mlir::failure();
  // The probe raised on a miss; `found` is a valid slot on every path that
  // continues here. `missing` survives only for the erased-lane read below.
  mlir::Value zero = mlir::arith::ConstantIntOp::create(builder, loc, 0, 64);
  mlir::Value missing = mlir::arith::CmpIOp::create(
      builder, loc, mlir::arith::CmpIPredicate::slt, found, zero);
  mlir::Value safe =
      mlir::arith::SelectOp::create(builder, loc, missing, zero, found)
          .getResult();
  llvm::SmallVector<mlir::Value, 4> resultValues;
  mlir::FailureOr<mlir::Value> valuesView =
      RuntimeBundleLowerer::containerInteriorView(
          op, container, ContainerInterior::Secondary, "runtime dict getitem");
  if (mlir::failed(valuesView))
    return mlir::failure();
  if (valueUnion) {
    mlir::Value wordsPerSlot = mlir::arith::ConstantIntOp::create(
        builder, loc, box_abi::kWordsPerBox, 64);
    mlir::Value base =
        mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
            .getResult();
    mlir::Value classWord =
        box_abi::loadContainerBoxWord(builder, loc, *valuesView, base, 1);
    mlir::Value entityWord = box_abi::loadContainerBoxWord(
        builder, loc, *valuesView, base, box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 8>> unionValues =
        RuntimeBundleLowerer::unionValuesFromBoxWords(op, valueUnion, classWord,
                                                      entityWord);
    if (mlir::failed(unionValues))
      return mlir::failure();
    RuntimeBundle result;
    if (mlir::failed(RuntimeBundleLowerer::makeObjectBundle(
            op, valueContract, *unionValues, result)))
      return mlir::failure();
    result.setObjectLogicalOwnership(/*ownsObject=*/false);
    valueBundles[op.getResult()] = std::move(result);
    if (mlir::failed(pinContainerLiveness(op, container,
                                          /*insertAfterOp=*/true)))
      return mlir::failure();
    erase.push_back(op);
    return true;
  }
  // See lowerRuntimeSequenceGetItem: "fresh owned object box" is literal, so
  // this branch's element needs marking and NOT retaining.
  bool valueArrivesOwned = false;
  if (runtimeContractName(valueContract) == "builtins.object") {
    // Erased read lane: box the slot's canonical payload handle into a
    // fresh owned object box (the slot's raw words are not an object box).
    std::optional<RuntimeSymbol> fromSlot =
        manifest.primitive("builtins.object", "from_slot");
    if (!fromSlot) {
      op.emitError() << "runtime manifest has no object from_slot primitive";
      return mlir::failure();
    }
    mlir::Value one1 = mlir::arith::ConstantIntOp::create(builder, loc, 1, 1);
    mlir::Value present =
        mlir::arith::XOrIOp::create(builder, loc, missing, one1).getResult();
    mlir::func::CallOp boxed = RuntimeBundleLowerer::createRuntimeCall(
        loc, *fromSlot, mlir::ValueRange{*valuesView, safe, present});
    resultValues.push_back(boxed.getResult(0));
    valueArrivesOwned = true;
  } else {
    mlir::Value wordsPerSlot = mlir::arith::ConstantIntOp::create(
        builder, loc, box_abi::kWordsPerBox, 64);
    mlir::Value base =
        mlir::arith::MulIOp::create(builder, loc, safe, wordsPerSlot)
            .getResult();
    mlir::Value entityWord = box_abi::loadContainerBoxWord(builder, loc, *valuesView, base,
                                                  box_abi::kEntityWord);
    mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> lanes =
        RuntimeBundleLowerer::lanesFromBoxEntity(
            builder, loc, entityWord, *shapes,
            runtimeContractName(valueContract), op);
    if (mlir::failed(lanes))
      return mlir::failure();
    resultValues.append(lanes->begin(), lanes->end());
  }
  mlir::FailureOr<llvm::SmallVector<mlir::Value, 4>> canonical =
      RuntimeBundleLowerer::unboxSlotElementValues(op, valueContract,
                                                   resultValues);
  if (mlir::failed(canonical))
    return mlir::failure();
  RuntimeValue value{valueContract, *canonical,
                     ownership::logicalOwnershipKind(valueContract,
                                                     /*ownsObject=*/false)};
  if (mlir::failed(valueArrivesOwned
                       ? bindOwnedEvidenceValue(op, op.getResult(),
                                                "runtime dict __getitem__", value)
                       : bindRetainedEvidenceValue(op, op.getResult(),
                                                   "runtime dict __getitem__",
                                                   value)))
    return mlir::failure();
  if (mlir::failed(pinContainerLiveness(op, container,
                                        /*insertAfterOp=*/true)))
    return mlir::failure();
  return true;
}

mlir::LogicalResult RuntimeBundleLowerer::lowerGetItem(py::GetItemOp op) {
  const RuntimeBundle *container =
      RuntimeBundleLowerer::bundleFor(op.getContainer());
  const RuntimeBundle *index = RuntimeBundleLowerer::bundleFor(op.getIndex());
  if (!container || !index)
    return op.emitError() << "getitem operands need runtime bundles";

  if (container->ctypes &&
      (container->ctypes->kind == RuntimeCtypesEvidence::Kind::Cell ||
       container->ctypes->kind == RuntimeCtypesEvidence::Kind::Pointer)) {
    mlir::LogicalResult ctypesHandled =
        RuntimeBundleLowerer::lowerStaticCtypesGetItem(op, *container, *index);
    if (mlir::succeeded(ctypesHandled))
      return mlir::success();
  }

  if (container->ctypes &&
      container->ctypes->kind == RuntimeCtypesEvidence::Kind::Library)
    return RuntimeBundleLowerer::lowerStaticCtypesLibraryGetItem(op,
                                                                 *container);

  mlir::FailureOr<bool> runtimeSequenceHandled =
      RuntimeBundleLowerer::lowerRuntimeSequenceGetItem(op, *container, *index);
  if (mlir::failed(runtimeSequenceHandled))
    return mlir::failure();
  if (*runtimeSequenceHandled)
    return mlir::success();

  mlir::FailureOr<bool> runtimeDictHandled =
      RuntimeBundleLowerer::lowerRuntimeDictGetItem(op, *container, *index);
  if (mlir::failed(runtimeDictHandled))
    return mlir::failure();
  if (*runtimeDictHandled)
    return mlir::success();

  mlir::FailureOr<bool> sequenceHandled =
      RuntimeBundleLowerer::lowerSequenceEvidenceGetItem(op, *container,
                                                         *index);
  if (mlir::failed(sequenceHandled))
    return mlir::failure();
  if (*sequenceHandled)
    return mlir::success();

  mlir::FailureOr<bool> dictHandled =
      RuntimeBundleLowerer::lowerDictEvidenceGetItem(op, *container, *index);
  if (mlir::failed(dictHandled))
    return mlir::failure();
  if (*dictHandled)
    return mlir::success();

  llvm::SmallVector<const RuntimeBundle *, 2> sources{container, index};
  mlir::FailureOr<llvm::StringRef> methodName =
      RuntimeBundleLowerer::requireMethodTarget(op, op.getTargetAttr(),
                                                "__getitem__");
  if (mlir::failed(methodName))
    return mlir::failure();
  if (mlir::failed(lowerManifestMethodResult(
          op, op.getResult(), *container, *methodName, sources,
          /*allowUnusedSources=*/false,
          /*preferManifestObjectResult=*/true)))
    return mlir::failure();
  erase.push_back(op);
  return mlir::success();
}

} // namespace py::lowering
