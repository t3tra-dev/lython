#include "runtime/Verification.h"

#include "Common/Instrumentation.h"

#include "Contracts.h"
#include "Ownership.h"
#include "runtime/Detail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include <cstddef>
#include <memory>
#include <optional>

namespace py::lowering {
namespace {

namespace own = py::ownership;
namespace contracts = py::contracts;

bool isRawObjectHeaderABI(mlir::Type type) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref || memref.getRank() != 1 || !memref.hasStaticShape() ||
      memref.getDimSize(0) != 2)
    return false;
  return isIntegerType(memref.getElementType(), 64);
}

mlir::LogicalResult verifyAggregateOwnershipCallee(
    mlir::Operation *op, const own::AggregateOwnershipMarker &marker,
    llvm::StringRef calleeName, mlir::Operation *callee) {
  if (!callee)
    return op->emitError() << "aggregate ownership call target @" << calleeName
                           << " is not a function";

  if (marker.action == own::AggregateOwnershipAction::Retain) {
    auto retained = own::parseIndexSetAttr(callee, own::kRetainArgsAttr);
    if (mlir::failed(retained))
      return mlir::failure();
    if (!retained->contains(0))
      return op->emitError() << own::kAggregateRetainAttr
                             << " call target must retain operand 0";
    return mlir::success();
  }

  auto released = own::parseIndexSetAttr(callee, own::kReleaseArgsAttr);
  if (mlir::failed(released))
    return mlir::failure();
  auto transferred = own::parseIndexSetAttr(callee, own::kTransferArgsAttr);
  if (mlir::failed(transferred))
    return mlir::failure();
  if (!released->contains(0) && !transferred->contains(0))
    return op->emitError() << own::kAggregateReleaseAttr
                           << " call target must release or transfer operand 0";
  return mlir::success();
}

mlir::LogicalResult verifyFunctionOwnershipShape(mlir::func::FuncOp function) {
  auto contract = own::readFunctionContract(function);
  if (mlir::failed(contract))
    return mlir::failure();

  // ⭐ A `builtins.bool` GROUP IS A BIT, and naming it owned is a statement
  // with nothing in it: no header to anchor, and `LyBool_DecRef` is a no-op
  // over an immortal singleton the group does not carry. Returning a closure
  // that captures a bool puts exactly that group in the result list --
  //
  //     def make() -> Callable[[int], int]:
  //         flag = True
  //         def pick(v: int) -> int: return v if flag else -v
  //         return pick
  //
  // was "ly.ownership.owned_results result 1 must start an object-header-like
  // result group", while the same closure over an int, a str or a float
  // returns fine.
  //
  // ⛔ Why the entry stays in `owned_results` rather than not being recorded:
  // measured. Dropping it clears the attribute entirely, and the caller's
  // group collection then falls back to a scan that grouped the function
  // handle with the lane and could place no cleanup for it -- "owned resource
  // from @make result 0 is still owned when a call to 'LyLong_FromI64' may
  // unwind". The declaration is what keeps result 0's own group separate; what
  // it must not do is claim a header the bit does not have.
  auto contractNameAt = [&](unsigned index) -> llvm::StringRef {
    if (index < contract->ownedResultContracts.size())
      return contract->ownedResultContracts[index];
    return llvm::StringRef();
  };
  for (auto [position, index] : llvm::enumerate(contract->ownedResults.values)) {
    if (contract->borrowedResults.contains(index))
      return function.emitError()
             << "result " << index
             << " cannot be both owned_results and borrowed_results";
    if (contractNameAt(static_cast<unsigned>(position)) == "builtins.bool")
      continue;
    if (!own::isObjectHeaderLikeType(
            function.getFunctionType().getResult(index)))
      return function.emitError()
             << own::kOwnedResultsAttr << " result " << index
             << " must start an object-header-like result group";
  }

  for (unsigned index : contract->borrowedResults.values) {
    if (!own::isObjectHeaderLikeType(
            function.getFunctionType().getResult(index)))
      return function.emitError()
             << own::kBorrowedResultsAttr << " result " << index
             << " must start an object-header-like result group";
  }

  auto verifyObjectArg = [&](unsigned index,
                             llvm::StringRef attrName) -> mlir::LogicalResult {
    if (!own::isObjectHeaderLikeType(
            function.getFunctionType().getInput(index)))
      return function.emitError() << attrName << " argument " << index
                                  << " must be an object-header-like memref";
    if (!function.getArgAttr(index, own::kObjectHeaderAttr))
      return function.emitError() << attrName << " argument " << index
                                  << " must carry " << own::kObjectHeaderAttr;
    return mlir::success();
  };

  auto verifyConcreteConsumerArg =
      [&](unsigned index, llvm::StringRef attrName) -> mlir::LogicalResult {
    auto manifestContract = function->getAttrOfType<mlir::StringAttr>(
        contracts::kManifestContractAttr);
    if (!manifestContract || manifestContract.getValue() != "builtins.object")
      return mlir::success();
    if (!isRawObjectHeaderABI(function.getFunctionType().getInput(index)))
      return mlir::success();
    return function.emitError()
           << attrName << " argument " << index
           << " consumes only a raw builtins.object header; consuming object "
              "ownership requires a concrete runtime value group or a boxed "
              "object handle";
  };

  for (unsigned index : contract->retainArgs.values)
    if (mlir::failed(verifyObjectArg(index, own::kRetainArgsAttr)))
      return mlir::failure();
  for (unsigned index : contract->releaseArgs.values) {
    if (mlir::failed(verifyObjectArg(index, own::kReleaseArgsAttr)))
      return mlir::failure();
    if (mlir::failed(verifyConcreteConsumerArg(index, own::kReleaseArgsAttr)))
      return mlir::failure();
  }
  for (unsigned index : contract->transferArgs.values) {
    if (mlir::failed(verifyObjectArg(index, own::kTransferArgsAttr)))
      return mlir::failure();
    if (mlir::failed(verifyConcreteConsumerArg(index, own::kTransferArgsAttr)))
      return mlir::failure();
  }

  if (function->hasAttr(contracts::kManifestDeallocatorAttr) &&
      contract->releaseArgs.empty())
    return function.emitError()
           << "runtime deallocator must declare release_args";

  if (contract->objectReleaseToZero) {
    if (function.getFunctionType().getNumResults() != 1 ||
        !isIntegerType(function.getFunctionType().getResult(0), 1))
      return function.emitError()
             << own::kObjectReleaseToZeroAttr
             << " function must return one i1 release-to-zero flag";
  }

  return mlir::success();
}

mlir::LogicalResult verifyOperationOwnershipShape(mlir::Operation *op) {
  if (op->hasAttr(own::kOwnedLocalObjectAttr)) {
    if (op->getNumResults() == 0 ||
        !own::isObjectHeaderLikeType(op->getResult(0).getType()))
      return op->emitError()
             << own::kOwnedLocalObjectAttr
             << " must mark an operation producing an object header";
    if (op->hasAttr(own::kOwnedLocalObjectContractAttr) &&
        !mlir::isa<mlir::StringAttr>(
            op->getAttr(own::kOwnedLocalObjectContractAttr)))
      return op->emitError() << own::kOwnedLocalObjectContractAttr
                             << " must be a string attribute";
  } else if (op->hasAttr(own::kOwnedLocalObjectContractAttr)) {
    return op->emitError() << own::kOwnedLocalObjectContractAttr << " requires "
                           << own::kOwnedLocalObjectAttr;
  }

  auto aggregate = own::readAggregateOwnershipMarker(op);
  if (mlir::failed(aggregate))
    return mlir::failure();
  if (*aggregate) {
    auto call = mlir::dyn_cast<mlir::func::CallOp>(op);
    mlir::ModuleOp module = op->getParentOfType<mlir::ModuleOp>();
    if (call) {
      if (call.getNumOperands() == 0)
        return op->emitError()
               << "aggregate ownership call must have at least one operand";
      mlir::Operation *callee =
          module ? module.lookupSymbol(call.getCallee()) : nullptr;
      return verifyAggregateOwnershipCallee(op, **aggregate, call.getCallee(),
                                            callee);
    }

    auto llvmCall = mlir::dyn_cast<mlir::LLVM::CallOp>(op);
    if (!llvmCall)
      return op->emitError()
             << "aggregate ownership marker must be attached to a call";
    if (llvmCall.getNumOperands() == 0)
      return op->emitError()
             << "aggregate ownership call must have at least one operand";
    std::optional<llvm::StringRef> calleeName = llvmCall.getCallee();
    if (!calleeName)
      return op->emitError()
             << "aggregate ownership marker requires a direct call target";
    mlir::Operation *callee =
        module ? module.lookupSymbol(*calleeName) : nullptr;
    return verifyAggregateOwnershipCallee(op, **aggregate, *calleeName, callee);
  }

  return mlir::success();
}

mlir::LogicalResult verifyOwnershipContractShapesImpl(mlir::ModuleOp module) {
  if (mlir::failed(
          walkVerify<mlir::func::FuncOp>(module, verifyFunctionOwnershipShape)))
    return mlir::failure();
  return walkVerifyOperations(module, verifyOperationOwnershipShape);
}

class OwnershipVerifierPass
    : public mlir::PassWrapper<OwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnershipVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-ownership-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify Lython ownership contracts and local affine sinks";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyOwnership(getOperation())))
      signalPassFailure();
  }
};

// ⭐ ONE OWNED TOKEN PER SSA VALUE.
//
// `ly.ownership.owned_local_object` marks a value the FRAME owns and must
// release. `refcount-insertion` emits one release per marked VALUE, however
// many times it is marked -- so two tokens on one value are two retains against
// one release, which is a leak.
//
// That was assumed by the consumer and enforced nowhere until an unbounded leak
// came out of it: reading one container slot twice reconstructs the same handle
// and the lowering minted a token per read, so
//
//     n = ((1, 2), (3, 4))
//     a = n[0]
//     b = n[0]
//
// never freed the inner entity -- 2 roots / 10368 B for a two-element inner
// tuple, 69 roots / 14656 B for a seventy-element one, and saturating, which is
// what a per-value map looks like from the outside.
//
// The invariant is `proof/`'s `WFES.backed` -- every owned name occupies its
// OWN site -- and the model had it before the leak was measured. A state
// invariant belongs in a phase gate, not in the memory of whoever next mints a
// token: this runs between the pass that mints tokens and the pass that
// consumes them.
//
// ⛔ DOMINANCE, not mere co-occurrence. Two markers on one value in mutually
// exclusive blocks are FINE: exactly one retain executes, and one release is
// emitted, so the counts balance. Rejecting those would refuse correct IR. The
// defect is two tokens on a path that runs both, which is exactly "one
// dominates the other" -- so that is the condition tested.
//
// This check found three duplicate-token goldens the producer fix had missed,
// two of which were leaking (`cross_exception_field_box_slot` 9 roots / 1120 B,
// `sequence_literal_source_move_frequency` 3 roots / 192 B) and one of which
// was benign because an enum member is an immortal singleton. All three came
// from one producer comparing against the wrong reference point; the check is
// what made them visible rather than a guess about where else to look. ⭐ A
// frame does not own what it did not acquire.
//
// `proof/`'s `WFES.backed` says every owned name occupies its OWN site. The
// uniqueness check beside this one reads that at the consumer -- two tokens for
// one site. This reads it at the producer: a token is a claim to a site, and a
// step that occupies one is `alloc`+`init`, `dup`, or `callIn`. `getField` is
// not among them -- it binds a BORROWED name and occupies nothing -- so a token
// on a value that only came out of a slot is a name owning something it never
// took, and the release it earns is an over-release rather than a leak.
//
// The three justifications are exactly those three steps, and the third is
// asked through `ownedLocalMarkerIsRetainRooted` rather than by inspection:
// that predicate is already shared by the release placer and the affine
// verifier so they cannot disagree about which token a release discharges, and
// a fourth reading of the same question is how they would.
//
// ⛔ Calibrated before it was a gate, and the calibration mattered. A
// hand-rolled walk over retain operands reported 2766 unbacked tokens across 78
// programs that the leak gate and ASan both say are clean -- an incomplete
// classifier refusing valid IR, which is worse than the gap it closes. With the
// shared predicate the count over all 297 golden cases is zero.
mlir::LogicalResult
verifyOwnedTokensAreAcquiredIn(mlir::func::FuncOp function) {
  own::AliasAnalysis aliases;
  aliases.build(function);
  mlir::LogicalResult result = mlir::success();
  function.walk([&](mlir::Operation *token) {
    if (!token->hasAttr(own::kOwnedLocalObjectAttr) ||
        token->getNumOperands() == 0)
      return;
    // Back through the identity casts to what defines the marked value. A
    // `memref.view` is one of them: a view of a fresh block IS that block's
    // only owner, which is how `__ly_unicode_alloc` shapes a str and how a
    // class instance's header sits in front of its body -- both are freed
    // through the view, so the frame that made the block holds the object.
    mlir::Value root = token->getOperand(0);
    for (bool moved = true; moved;) {
      moved = false;
      if (auto cast = root.getDefiningOp<mlir::UnrealizedConversionCastOp>()) {
        if (cast->getNumOperands() != 0) {
          root = cast->getOperand(0);
          moved = true;
          continue;
        }
      }
      if (auto view = root.getDefiningOp<mlir::memref::ViewOp>()) {
        root = view.getSource();
        moved = true;
      }
    }
    mlir::Operation *def = root.getDefiningOp();
    // `alloc` + `init`: the frame made the storage, so it owns the object.
    if (def && mlir::isa<mlir::memref::AllocOp>(def))
      return;
    // `callIn`: the callee's contract handed over a +1.
    if (auto call = def ? mlir::dyn_cast<mlir::func::CallOp>(def)
                        : mlir::func::CallOp()) {
      auto callee = function->getParentOfType<mlir::ModuleOp>()
                        .lookupSymbol<mlir::func::FuncOp>(call.getCallee());
      if (callee) {
        auto contract = own::readFunctionContract(callee);
        if (mlir::succeeded(contract) &&
            contract->ownedResults.contains(
                mlir::cast<mlir::OpResult>(root).getResultNumber()))
          return;
      }
    }
    // `dup`: a retain minted this token.
    if (own::ownedLocalMarkerIsRetainRooted(token, aliases))
      return;
    result =
        token->emitError()
        << own::kOwnedLocalObjectAttr
        << " marks a value this frame never acquired: it is not a fresh "
           "allocation, not a call result the contract declares owned, and "
           "no retain roots it. A value read out of a slot is BORROWED -- "
           "the slot still holds it -- so the release this token earns "
           "would discharge a reference the frame does not have";
  });
  return result;
}

mlir::LogicalResult verifyOwnedTokenUniquenessIn(mlir::func::FuncOp function) {
  llvm::SmallVector<mlir::Operation *, 8> tokens;
  function.walk([&](mlir::Operation *op) {
    if (op->hasAttr(own::kOwnedLocalObjectAttr))
      tokens.push_back(op);
  });
  if (tokens.size() < 2)
    return mlir::success();
  mlir::DominanceInfo dominance(function);
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    for (std::size_t j = i + 1; j < tokens.size(); ++j) {
      mlir::Operation *a = tokens[i];
      mlir::Operation *b = tokens[j];
      if (a->getNumOperands() == 0 || a->getOperands() != b->getOperands())
        continue;
      mlir::Operation *first = dominance.dominates(a, b) ? a : b;
      mlir::Operation *second = first == a ? b : a;
      if (!dominance.dominates(first, second))
        continue;
      return second->emitError()
             << own::kOwnedLocalObjectAttr
             << " marks a value this frame already owns; the earlier token at "
             << first->getLoc()
             << " dominates this one, so both retains run and only one release "
                "is emitted. A re-read of an entity the frame owns is a "
                "borrow: "
                "reuse the existing token instead of minting a second";
    }
  }
  return mlir::success();
}

// The PUBLISHING store, which is the earliest write to word 0 -- a later write
// is a re-initialisation, and picking one of those off the use list (whose
// order is not program order) reports every well-formed window as a violation.
// When no single write dominates the rest, the window has no identifiable
// close and this declines to judge it rather than guessing.
std::optional<mlir::Operation *>
refcountWordStoreFor(mlir::Value box, mlir::DominanceInfo &dominance) {
  llvm::SmallVector<mlir::Operation *, 4> writes;
  for (mlir::Operation *user : box.getUsers()) {
    auto store = mlir::dyn_cast<mlir::memref::StoreOp>(user);
    if (!store || store.getMemRef() != box || store.getIndices().size() != 1)
      continue;
    mlir::Operation *slot = store.getIndices().front().getDefiningOp();
    if (!slot)
      continue;
    auto index = slot->getAttrOfType<mlir::IntegerAttr>("value");
    if (index && index.getValue().getSExtValue() == 0)
      writes.push_back(store.getOperation());
  }
  for (mlir::Operation *candidate : writes) {
    bool first = true;
    for (mlir::Operation *other : writes)
      if (other != candidate && !dominance.properlyDominates(candidate, other))
        first = false;
    if (first)
      return candidate;
  }
  return std::nullopt;
}

// ⭐ Nothing counts a reference before the object exists.
//
// `proof/`'s `no-dup-in-the-initialisation-window`: while
// `lookupObj (objects m) o` is `nothing` -- between the allocation and the
// store that publishes the header -- `dup` cannot step. The compiler had a
// predicate for this, `entity_header::prefixIsInitializedAtDefinition`, with
// exactly one reader: `borrowEdgeRetainIsSpellable`. That asks whether a retain
// can be SPELLED on one edge kind. The theorem is about whether any counting
// operation may RUN, on every allocation.
//
// Only the dup half is checkable here, and this is the phase where it is
// checkable at all: the token exists from runtime-lowering, and releases are
// not emitted until phase 10. `no-drop-in-the-initialisation-window` therefore
// has no gate -- said plainly rather than implied by the name.
mlir::LogicalResult verifyInitialisationWindowIn(mlir::func::FuncOp function) {
  mlir::DominanceInfo dominance(function);
  mlir::LogicalResult result = mlir::success();
  function.walk([&](mlir::Operation *marker) {
    if (!marker->hasAttr(own::kOwnedLocalObjectAttr))
      return mlir::WalkResult::advance();
    // ⛔ Why NOT judge every op carrying the attribute: a marker on the op that
    // CREATES the handle -- `memref.alloc`, or the `memref.view` that first
    // names a header inside a raw block, as `@__ly_bytes_alloc` does -- is the
    // model's `alloc` handing back an owned binding. That is where the count
    // legitimately begins and the store of 1 into word 0 follows it by
    // construction; reading those as `dup` reports the runtime's own
    // allocators. Only the rooting cast says "this existing handle is now
    // frame-owned", which is the `dup` the window forbids.
    //
    // What that leaves unjudged, since a number is the only honest form of
    // this: over the first 110 golden programs the gate reaches 560 windows
    // out of 2605 rooting casts. 105 markers name a `func.call` or `scf.if`
    // result -- an entity someone else finished, whose window is not in this
    // function. The other 4099 operands root at an `llvm.insertvalue`: the
    // borrow->own path, where `memrefFromBoxPointer` assembles a view over a
    // payload the box already addresses, and the allocation belongs to the
    // boxed element rather than to this frame.
    //
    // ⛔ Why NOT read that number as descriptors left to remove, which is what
    // it was first written down as: `ABI/BoxLayout.cpp` says outright that this
    // one stays. A box is a `memref<16xi64>` and MLIR refuses a pointer element
    // type, so every reference a boxed object owns is an address in an integer
    // -- the dialect's constraint, not a shortcut. Its soundness comes from
    // `Proof.MemRef.Dialect.descFromAlignedPointer` with the premises
    // discharged by `Proof.RC.Address.site-address-recovers`, so what is
    // outside this gate is covered by a theorem rather than pending work.
    if (!mlir::isa<mlir::UnrealizedConversionCastOp>(marker))
      return mlir::WalkResult::advance();
    for (mlir::Value operand : marker->getOperands()) {
      auto alloc = operand.getDefiningOp<mlir::memref::AllocOp>();
      if (!alloc)
        continue;
      std::optional<mlir::Operation *> published =
          refcountWordStoreFor(alloc.getResult(), dominance);
      if (!published || dominance.dominates(*published, marker))
        continue;
      result =
          marker->emitError()
          << own::kOwnedLocalObjectAttr
          << " mints a frame reference before the header it counts is "
             "published; the refcount word is stored at "
          << (*published)->getLoc()
          << ", which does not dominate this token. Inside the initialisation "
             "window the object does not yet exist to be counted, so the "
             "matching release discharges a count that was never established";
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

mlir::LogicalResult verifyOwnedTokenUniquenessImpl(mlir::ModuleOp module) {
  if (mlir::failed(
          walkVerify<mlir::func::FuncOp>(module, verifyInitialisationWindowIn)))
    return mlir::failure();
  if (mlir::failed(walkVerify<mlir::func::FuncOp>(
          module, verifyOwnedTokensAreAcquiredIn)))
    return mlir::failure();
  return walkVerify<mlir::func::FuncOp>(module, verifyOwnedTokenUniquenessIn);
}

class OwnedTokenUniquenessVerifierPass
    : public mlir::PassWrapper<OwnedTokenUniquenessVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OwnedTokenUniquenessVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-owned-token-uniqueness-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify that no SSA value carries two frame-ownership tokens";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyOwnedTokenUniqueness(getOperation())))
      signalPassFailure();
  }
};

class LLVMCallOwnershipVerifierPass
    : public mlir::PassWrapper<LLVMCallOwnershipVerifierPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLVMCallOwnershipVerifierPass)

  llvm::StringRef getArgument() const final {
    return "lython-llvm-call-ownership-verifier";
  }
  llvm::StringRef getDescription() const final {
    return "verify lowered call ownership contracts";
  }

  void runOnOperation() final {
    if (mlir::failed(verifyLLVMCallOwnership(getOperation())))
      signalPassFailure();
  }
};

} // namespace

mlir::LogicalResult verifyOwnershipContractShapes(mlir::ModuleOp module) {
  return verifyOwnershipContractShapesImpl(module);
}

mlir::LogicalResult verifyOwnedTokenUniquenessShapes(mlir::ModuleOp module) {
  return verifyOwnedTokenUniquenessImpl(module);
}

} // namespace py::lowering

namespace py {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnershipVerifierPass() {
  return std::make_unique<lowering::OwnershipVerifierPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLLVMCallOwnershipVerifierPass() {
  return std::make_unique<lowering::LLVMCallOwnershipVerifierPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOwnedTokenUniquenessVerifierPass() {
  return std::make_unique<lowering::OwnedTokenUniquenessVerifierPass>();
}

mlir::LogicalResult verifyOwnedTokenUniqueness(mlir::ModuleOp module) {
  PerfScope perf("owned-token-uniqueness");
  return lowering::verifyOwnedTokenUniquenessShapes(module);
}

mlir::LogicalResult verifyOwnership(mlir::ModuleOp module) {
  if (mlir::failed(lowering::verifyOwnershipContractShapes(module)))
    return mlir::failure();
  return mlir::success();
}

mlir::LogicalResult verifyLLVMCallOwnership(mlir::ModuleOp module) {
  {
    PerfScope perf("llvm-call-ownership.func-contract-shapes");
    if (mlir::failed(lowering::verifyOwnershipContractShapes(module)))
      return mlir::failure();
  }
  {
    PerfScope perf("llvm-call-ownership.llvm-contract-shapes");
    if (mlir::failed(lowering::verifyLLVMOwnershipContractShapes(module)))
      return mlir::failure();
  }
  {
    PerfScope perf("llvm-call-ownership.llvm-call-contracts");
    if (mlir::failed(lowering::verifyLLVMCallOwnershipContracts(module)))
      return mlir::failure();
  }
  {
    PerfScope perf("llvm-call-ownership.func-call-contracts");
    return lowering::verifyFuncCallOwnershipContracts(module);
  }
}

} // namespace py
